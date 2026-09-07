#!/usr/bin/env Rscript
#
# Mixed (transformer-ish) tape: OLD vs NEW.
#
# WHAT THE TWO COLUMNS ARE
# ------------------------
#   old  every residency toggle OFF -- backward walked as per-node closures,
#        gradients downloaded to the host, forward computed one op at a time
#   new  full residency ON -- backward emitted as ONE graph, leaf gradients
#        left in their backend buffer, forward queued and computed as one
#        graph on first read
#
# Three switches move together to make that pair:
#   ag_backward_graph()     graph backward instead of closures   (default ON)
#   ag_backward_resident()  $grad stays a device handle          (default ON)
#   ag_defer_forward()      forward queued into one graph        (default OFF,
#                           behind GGMLR_AG_DEFER -- see
#                           project_ag_defer_gate_decision)
#
# So `new` here is deliberately NOT the shipped default: the default is the
# first two on and deferral off. `new` is full residency, which is what the
# gate would turn on.
#
# ⚠️ WHAT `old` IS NOT
# --------------------
# The resident FORWARD (component 4) is not behind a flag -- it is simply how
# .ag_run_op works now -- so it is present in BOTH columns. `old` is "the tape
# with every available toggle off", NOT a checkout of the pre-residency code.
# Do not quote it as "before residency".
#
# WHY CROSSINGS ARE COUNTED TOO
# -----------------------------
# Residency is a claim about host<->device crossings. Those are counted
# exactly, while milliseconds are noisy: a timing that moves for an unrelated
# reason (warm cache, quiet GPU) still shows the same crossing count.
# ⚠️ Expect the graph/gradient part to move crossings and DEFERRAL NOT TO --
# deferral collapses the number of COMPUTE calls, while the bytes on the bus
# are the same weights in and the same gradients out either way.
#
# ⚠️ MEASUREMENT TRAPS THIS SCRIPT IS BUILT AROUND (each cost a real session)
# ---------------------------------------------------------------------------
#  1. Weights are built ONCE PER SHAPE and reused. Building ag_param() inside
#     the timed function puts every rep's weights in the PERSISTENT pool, which
#     is correctly never freed between tapes. They accumulate; past ~700
#     buffers a d=256 step went 20 ms -> 435 ms, and every speedup in that run
#     was two saturated numbers divided by each other.
#  2. The persistent buffer count is printed at exit. Growth invalidates the
#     table above it.
#  3. min-max spread is printed next to every median. A speedup narrower than
#     its own run-to-run spread is not a result and is flagged `~noise`.
#  4. The loss is compared across the two columns. If they disagree these are
#     not two speeds of one computation, and the row says so instead of
#     printing a meaningless ratio.
#  5. f32 only: ggml_flash_attn_back asserts F32 on q/k/v/d, so an f16 run
#     would silently decline the resident path and measure something else.
#
# Run:  Rscript inst/scripts/measure_mixed_tape_residency.R
# Env:  GGMLR_MR_REPS   timed reps per configuration (default 20)
#       GGMLR_MR_WARM   discarded warm-up reps       (default 3)
#       GGMLR_MR_BIG    include d=512 s=256 (slow)   (default 0)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_MR_REPS", "20"))
warm <- as.integer(Sys.getenv("GGMLR_MR_WARM", "3"))
big  <- identical(Sys.getenv("GGMLR_MR_BIG", "0"), "1")

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns          <- asNamespace("ggmlR")
bwd_graph   <- get("ag_backward_graph",    envir = ns)
bwd_resid   <- get("ag_backward_resident", envir = ns)
bwd_path    <- get("ag_backward_path",     envir = ns)
defer_fwd   <- get("ag_defer_forward",     envir = ns)
xfer_count  <- get("ag_xfer_count",        envir = ns)
xfer_report <- get("ag_xfer_report",       envir = ns)
as_matrix   <- get(".ag_as_matrix",        envir = ns)
ag_data     <- get(".ag_data",             envir = ns)
dev_state   <- get(".ag_device_state",     envir = ns)

ag_device("gpu"); ag_dtype("f32")

# Restore the shipped defaults on the way out, whatever happens in between.
on.exit({ bwd_graph(TRUE); bwd_resid(TRUE); defer_fwd(FALSE)
          xfer_count(FALSE); ag_device("cpu") }, add = TRUE)

set_old <- function() { bwd_graph(FALSE); bwd_resid(FALSE); defer_fwd(FALSE) }
set_new <- function() { bwd_graph(TRUE);  bwd_resid(TRUE);  defer_fwd(TRUE)  }

# ---------------------------------------------------------------------------
# Workload: one transformer-ish block -- attention in the middle, ordinary ops
# on both sides. Parameters cached per shape (trap 1).
# ---------------------------------------------------------------------------
.params <- new.env(parent = emptyenv())

mixed_params <- function(d, s) {
  key <- sprintf("mixed-%d-%d", d, s)
  p <- get0(key, envir = .params)
  if (is.null(p)) {
    p <- list(wq = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              wk = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              wv = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              wo = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              x  = ag_tensor(matrix(rnorm(d * s), d, s)),
              y  = ag_tensor(matrix(0, d, s)))
    assign(key, p, envir = .params)
  }
  p
}

mixed_step <- function(d, s, heads) {
  p <- mixed_params(d, s)
  with_grad_tape({
    q    <- ag_matmul(p$wq, p$x)
    k    <- ag_matmul(p$wk, p$x)
    v    <- ag_matmul(p$wv, p$x)
    a    <- ag_flash_attention(q, k, v, n_heads = heads)
    out  <- ag_relu(ag_matmul(p$wo, a))
    loss <- ag_mse_loss(out, p$y)
  })
  backward(loss)
  # Force the value across: timing a deferred queue that never drains measures
  # nothing at all.
  list(loss = as.numeric(as_matrix(ag_data(loss))), path = bwd_path())
}

# MLP of comparable weight volume, no attention. The contrast says how much of
# any gain belongs to the tape AROUND attention rather than to attention.
mlp_params <- function(d, s) {
  key <- sprintf("mlp-%d-%d", d, s)
  p <- get0(key, envir = .params)
  if (is.null(p)) {
    p <- list(w1 = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              w2 = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              w3 = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              w4 = ag_param(matrix(rnorm(d * d) * 0.05, d, d)),
              x  = ag_tensor(matrix(rnorm(d * s), d, s)),
              y  = ag_tensor(matrix(0, d, s)))
    assign(key, p, envir = .params)
  }
  p
}

mlp_step <- function(d, s, heads) {
  p <- mlp_params(d, s)
  with_grad_tape({
    h1   <- ag_relu(ag_matmul(p$w1, p$x))
    h2   <- ag_relu(ag_matmul(p$w2, h1))
    h3   <- ag_relu(ag_matmul(p$w3, h2))
    out  <- ag_relu(ag_matmul(p$w4, h3))
    loss <- ag_mse_loss(out, p$y)
  })
  backward(loss)
  list(loss = as.numeric(as_matrix(ag_data(loss))), path = bwd_path())
}

# ---------------------------------------------------------------------------
# Timing. Median over `reps`, min/max kept: a spread wider than the claimed
# difference is the signature of a stall or a fallback, and a median hides it.
# ---------------------------------------------------------------------------
time_ms <- function(fn, ...) {
  for (i in seq_len(warm)) fn(...)
  t <- numeric(reps); last <- NULL
  for (i in seq_len(reps)) {
    t0 <- Sys.time()
    last <- fn(...)
    t[i] <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000
  }
  list(ms = stats::median(t), lo = min(t), hi = max(t),
       path = last$path, loss = last$loss)
}

# Crossings for ONE call, counted from zero.
xfers <- function(fn, ...) {
  fn(...)                       # warm: first-call allocation is not steady state
  xfer_count(TRUE)
  fn(...)
  tmp <- textConnection(NULL, "w"); sink(tmp)
  df <- tryCatch(xfer_report(0L), finally = { sink(); close(tmp) })
  xfer_count(FALSE)
  if (is.null(df)) return(list(n = 0L, mb = 0))
  list(n = sum(df$n), mb = sum(df$mb))
}

shapes <- list(
  list(d =  64L, s =  32L, h = 4L),
  list(d = 128L, s =  64L, h = 8L),
  list(d = 256L, s = 128L, h = 8L)
)
if (big) shapes <- c(shapes, list(list(d = 512L, s = 256L, h = 8L)))

workloads <- list(list(name = "mixed", fn = mixed_step),
                  list(name = "mlp",   fn = mlp_step))

p0 <- length(dev_state$p_buffers)

# ---------------------------------------------------------------------------
# Timings: old vs new.
# ---------------------------------------------------------------------------
cat("\n", strrep("=", 100), "\n", sep = "")
cat("Mixed tape, old vs full residency (median of ", reps,
    " reps, f32, Vulkan)\n", sep = "")
cat("  old = closures + host grads + per-op forward\n")
cat("  new = graph backward + resident grads + deferred forward\n")
cat(strrep("=", 100), "\n")
cat(sprintf("%-8s %-14s %10s %-15s %10s %-15s %9s   %s\n",
            "workload", "shape", "old ms", "old min-max",
            "new ms", "new min-max", "speedup", "path"))
cat(strrep("-", 100), "\n")

for (wl in workloads) {
  for (sh in shapes) {
    set_old(); o <- time_ms(wl$fn, sh$d, sh$s, sh$h)
    set_new(); n <- time_ms(wl$fn, sh$d, sh$s, sh$h)

    gain <- o$ms / n$ms
    # Trap 4: different losses are not two speeds of one computation.
    ok <- isTRUE(abs(n$loss - o$loss) <= 1e-3 * max(1, abs(o$loss)))
    # Trap 3: a speedup narrower than its own spread is not a result.
    spread <- (n$hi - n$lo) / max(n$ms, 1e-9)
    shaky  <- spread > abs(gain - 1)

    cat(sprintf("%-8s %-14s %10.2f %-15s %10.2f %-15s %8.2fx   %s%s%s\n",
                wl$name, sprintf("d=%d s=%d", sh$d, sh$s),
                o$ms, sprintf("%.1f-%.1f", o$lo, o$hi),
                n$ms, sprintf("%.1f-%.1f", n$lo, n$hi),
                gain, n$path,
                if (ok) "" else "  ⚠ LOSS MISMATCH",
                if (shaky) "  ~noise" else ""))
  }
}

# ---------------------------------------------------------------------------
# Crossings: old vs new.
# Read the totals against project_ag_residency_done.md, which records a plain
# MLP step at 4 crossings / 0.047 MB.
# ---------------------------------------------------------------------------
cat("\n", strrep("=", 100), "\n", sep = "")
cat("Host<->device crossings per step\n")
cat(strrep("=", 100), "\n")
cat(sprintf("%-8s %-14s %12s %10s %12s %10s\n",
            "workload", "shape", "old xings", "old MB", "new xings", "new MB"))
cat(strrep("-", 100), "\n")

for (wl in workloads) {
  for (sh in shapes) {
    set_old(); xo <- xfers(wl$fn, sh$d, sh$s, sh$h)
    set_new(); xn <- xfers(wl$fn, sh$d, sh$s, sh$h)
    cat(sprintf("%-8s %-14s %12d %10.3f %12d %10.3f\n",
                wl$name, sprintf("d=%d s=%d", sh$d, sh$s),
                xo$n, xo$mb, xn$n, xn$mb))
  }
}

# Per-site breakdown at full residency for the largest shape: the useful
# artefact when a count looks wrong.
sh <- shapes[[length(shapes)]]
cat(sprintf("\nPer-site breakdown, new (full residency), mixed d=%d s=%d:\n",
            sh$d, sh$s))
set_new()
invisible(mixed_step(sh$d, sh$s, sh$h))
xfer_count(TRUE)
invisible(mixed_step(sh$d, sh$s, sh$h))
invisible(xfer_report(40L))
xfer_count(FALSE)

# ---------------------------------------------------------------------------
# Guard (trap 2): did the persistent pool grow while we measured?
# Weights are built once per shape, so this must be flat. Growth means
# something allocated a persistent buffer per step -- the leak that once
# turned a 20 ms step into 435 ms and made every ratio meaningless.
# ---------------------------------------------------------------------------
#
# ⚠️ The expected count is 4 PARAMETERS per (shape x workload), not 2: both
# mixed_params and mlp_params build four weight matrices each. An earlier
# version of this guard used 2 and cried GROWTH at a perfectly flat pool of 24
# -- a false alarm that would have thrown away a valid table. The tensors that
# are NOT parameters (x, y) are ag_tensor, not ag_param, and do not land here.
p1 <- length(dev_state$p_buffers)
expected <- 4L * length(shapes) * length(workloads)
cat(sprintf("\npersistent buffers: %d at entry, %d at exit (expected <= %d)",
            p0, p1, p0 + expected))
if (p1 > p0 + expected) {
  cat("   ⚠ GROWTH -- the timings above are not trustworthy\n")
} else {
  cat("   flat: no leak\n")
}
cat(sprintf("pass buffers held at exit: %d\n", length(dev_state$buffers)))
cat("\nDone.\n")
