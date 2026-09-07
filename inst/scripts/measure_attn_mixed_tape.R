#!/usr/bin/env Rscript
#
# What did giving attention an `op` actually buy?
#
# THE THING BEING MEASURED
# ------------------------
# ag_flash_attention() used to call ag_record() without `op=`, and
# .ag_bwd_reject_reason() refuses a tape the moment ANY node lacks one. The
# rejection is all-or-nothing by design, so one attention block cost the WHOLE
# tape its graph backward and its stage-2 fusion -- the matmuls around it
# included. Nothing failed; it fell back to closures, which are correct and
# slow.
#
# That is why this script measures a MIXED tape rather than attention alone. A
# pure-attention benchmark understates the fix by construction: the loss was
# never confined to the attention node, it was everything the attention node
# dragged down with it. measure_ag_flash_attn.R already covers the attention
# core in isolation; this covers what the tape around it was paying.
#
# WHAT "OLD" MEANS HERE, AND WHY IT IS NOT A LIE
# ----------------------------------------------
# The resident forward cannot be switched off -- it is not behind a flag. So
# "old" is not a rebuild of the previous code; it is ag_backward_graph(FALSE),
# which puts the tape in exactly the state the missing `op` forced it into:
# closures for every node. That is the honest comparison, and the column is
# named for what it is (`closures`) rather than for a version.
#
# The one thing this therefore does NOT isolate is the resident forward's own
# contribution, which is present in both columns. Reported separately below via
# the transfer counts, where it IS visible.
#
# WHY TRANSFERS AND NOT ONLY TIME
# -------------------------------
# Residency is a claim about crossings, and crossings are counted exactly while
# milliseconds are noisy. A timing that improves for an unrelated reason (a
# warmer cache, a quieter GPU) still shows the same crossing count, so the two
# together say more than either alone. project_ag_residency_done.md records the
# step reaching 4 crossings; this asks what attention adds to that.
#
# ⚠️ RESET BETWEEN REPS. Every rep builds its params inside the timed function
# so the tape resets and nothing accumulates -- omitting that was worth 12 GB
# and a poisoned control row on an earlier measurement.
#
# Run:  Rscript inst/scripts/measure_attn_mixed_tape.R
# Env:  GGMLR_MT_REPS   timed reps per configuration (default 20)
#       GGMLR_MT_WARM   discarded warm-up reps        (default 3)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_MT_REPS", "20"))
warm <- as.integer(Sys.getenv("GGMLR_MT_WARM", "3"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns          <- asNamespace("ggmlR")
bwd_graph   <- get("ag_backward_graph",   envir = ns)
bwd_path    <- get("ag_backward_path",    envir = ns)
defer_fwd   <- get("ag_defer_forward",    envir = ns)
xfer_count  <- get("ag_xfer_count",       envir = ns)
xfer_report <- get("ag_xfer_report",      envir = ns)
as_matrix   <- get(".ag_as_matrix",       envir = ns)
ag_data     <- get(".ag_data",            envir = ns)

# f32 only: ggml_flash_attn_back asserts F32 on q/k/v/d, so an f16 run would
# decline the resident path and measure something else entirely.
ag_device("gpu"); ag_dtype("f32")
on.exit({ bwd_graph(TRUE); defer_fwd(FALSE); xfer_count(FALSE)
          ag_device("cpu") }, add = TRUE)

# ---------------------------------------------------------------------------
# The workload: one transformer-ish block. Attention in the middle, ordinary
# ops on both sides -- the shape the bug actually punished.
#
# Projections are real parameters, so the tape has matmuls that the graph
# backward can emit and that the closure fallback has to walk node by node.
#
# ⚠️ PARAMETERS ARE BUILT ONCE PER SHAPE AND REUSED, AND THAT IS NOT A DETAIL.
#
# The first version of this script created them inside the timed function. That
# put every rep's weights into the PERSISTENT residency pool, which -- correctly
# -- is not freed between tapes, because that is where weights live. They
# accumulated: 184 buffers, 368, 552, 736... and somewhere past ~700 the step
# time went from 27 ms to 433 ms. Measured, not guessed: with 880 buffers a
# d=256 step took 444.9 ms, and .ag_residency_reset(scope = "persistent") in the
# SAME process brought the identical step back to 22.9 ms.
#
# So the 435 ms wall in the first run was the benchmark's own leak, not the code
# under test, and every d=256 speedup in that run was two saturated numbers
# divided by each other. Building the parameters once is what a training loop
# does anyway, which is the behaviour worth measuring.
# ---------------------------------------------------------------------------

# Cache: shape key -> the parameters for that shape.
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

attn_params <- function(d, s) {
  key <- sprintf("attn-%d-%d", d, s)
  p <- get0(key, envir = .params)
  if (is.null(p)) {
    p <- list(q = ag_param(matrix(rnorm(d * s) * 0.1, d, s)),
              k = ag_param(matrix(rnorm(d * s) * 0.1, d, s)),
              v = ag_param(matrix(rnorm(d * s) * 0.1, d, s)),
              y = ag_tensor(matrix(0, d, s)))
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
  # Force the value across so a deferred graph is actually computed: timing a
  # queue that never drains measures nothing.
  list(loss = as.numeric(as_matrix(ag_data(loss))), path = bwd_path())
}

# Attention alone, for contrast: same op, no surrounding matmuls. The gap
# between this and mixed_step is the part of the fix that was never about
# attention itself.
attn_only <- function(d, s, heads) {
  p <- attn_params(d, s)
  with_grad_tape({
    a    <- ag_flash_attention(p$q, p$k, p$v, n_heads = heads)
    loss <- ag_mse_loss(a, p$y)
  })
  backward(loss)
  list(loss = as.numeric(as_matrix(ag_data(loss))), path = bwd_path())
}

# Median of `reps` timed calls, after `warm` discarded ones.
#
# min and max are kept because the first run produced two numbers that a median
# alone could not explain: attn-only d=128 at 10.30x (against 1.1-1.5x for its
# neighbours) and every d=256 shape flat at ~440 ms with a 1.0x speedup. Both
# are the signature of something other than the arithmetic dominating -- a
# stall, a fallback, a driver hiccup -- and a wide min-max spread says so
# directly, where a median hides it. ⚠️Do not read a speedup whose spread is
# wider than the difference it claims.
time_ms <- function(fn, ...) {
  for (i in seq_len(warm)) fn(...)
  t <- numeric(reps)
  last <- NULL
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
  fn(...)                       # warm: first-call allocation is not the steady state
  xfer_count(TRUE)
  fn(...)
  # ag_xfer_report prints unconditionally; only the data frame is wanted here,
  # so the printing goes to a sink. The per-site table IS printed at the end,
  # deliberately, where it is the point rather than noise.
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

for (mode in c("per-op forward", "deferred forward")) {
  defer_fwd(identical(mode, "deferred forward"))

  cat("\n", strrep("=", 92), "\n", sep = "")
  cat(mode, " (median of ", reps, " reps)\n", sep = "")
  cat(strrep("=", 92), "\n")
  cat(sprintf("%-22s %9s %-15s %10s %8s   %s\n",
              "workload", "graph ms", "graph min-max", "closure ms",
              "speedup", "path"))
  cat(strrep("-", 92), "\n")

  for (sh in shapes) {
    for (nm in c("mixed", "attn-only")) {
      fn <- if (nm == "mixed") mixed_step else attn_only

      bwd_graph(TRUE)
      g <- time_ms(fn, sh$d, sh$s, sh$h)
      bwd_graph(FALSE)
      c_ <- time_ms(fn, sh$d, sh$s, sh$h)
      bwd_graph(TRUE)

      # A loss that disagrees between paths means the comparison is void --
      # report it rather than quietly printing a speedup for two different
      # computations.
      ok <- isTRUE(abs(g$loss - c_$loss) <= 1e-3 * max(1, abs(c_$loss)))
      # A speedup narrower than the run-to-run spread is not a result. Flag it
      # rather than let a 1.02x sit in a column next to a 10x as if both were
      # findings.
      spread <- (g$hi - g$lo) / max(g$ms, 1e-9)
      shaky  <- spread > abs(c_$ms / g$ms - 1)

      cat(sprintf("%-22s %9.2f %-15s %10.2f %8.2fx   %s%s%s\n",
                  sprintf("%s d=%d s=%d", nm, sh$d, sh$s),
                  g$ms, sprintf("%.1f-%.1f", g$lo, g$hi),
                  c_$ms, c_$ms / g$ms, g$path,
                  if (ok) "" else "  ⚠ LOSS MISMATCH",
                  if (shaky) "  ~noise" else ""))
    }
  }
}

# ---------------------------------------------------------------------------
# Residency: how many times does one mixed step cross the bus?
#
# ⚠️ READ THIS BLOCK FOR WHAT IT MEASURES, WHICH IS NOT WHAT DEFERRAL CHANGES.
# The first run of this script showed per-op and deferred at an IDENTICAL 12
# crossings and identical MB, which looked like a broken toggle. It is not:
# deferral was verified to queue nodes (1,3,4,7 across a forward, attention
# contributing its own). Crossings are simply not the thing it moves --
# deferral collapses the number of COMPUTE calls, while the bytes crossing the
# bus are the same weights in and the same gradients out either way.
#
# So this block answers "what does a transformer step still pay", and the
# per-site table below is the useful half of it. It does NOT separate the two
# forward modes, and a difference here would be the surprise, not the
# expectation.
#
# Read the total against project_ag_residency_done.md, which records a plain
# MLP step at 4 crossings.
# ---------------------------------------------------------------------------
cat("\n", strrep("=", 78), "\n", sep = "")
cat("Crossings per mixed step, graph backward on\n")
cat(strrep("=", 78), "\n")

defer_fwd(FALSE); bwd_graph(TRUE)
for (sh in shapes) {
  x <- xfers(mixed_step, sh$d, sh$s, sh$h)
  cat(sprintf("  per-op   d=%-4d s=%-4d  %3d crossings  %7.3f MB\n",
              sh$d, sh$s, x$n, x$mb))
}
defer_fwd(TRUE)
for (sh in shapes) {
  x <- xfers(mixed_step, sh$d, sh$s, sh$h)
  cat(sprintf("  deferred d=%-4d s=%-4d  %3d crossings  %7.3f MB\n",
              sh$d, sh$s, x$n, x$mb))
}

# Per-site breakdown for the largest shape: the useful artefact when a count
# looks wrong. "flash_attn operands" is this change's own upload site.
cat("\nPer-site breakdown, deferred, mixed d=256 s=128:\n")
invisible(mixed_step(256L, 128L, 8L))
xfer_count(TRUE)
invisible(mixed_step(256L, 128L, 8L))
invisible(xfer_report(40L))
xfer_count(FALSE)


# ---------------------------------------------------------------------------
# Guard: did the persistent pool grow while we measured?
#
# It must not. Weights are built once per shape and the pool holds only them, so
# the buffer count should be flat from the first timed rep to the last. Growth
# means something allocated a persistent buffer per step -- exactly the leak
# that made the first run of this script report 435 ms for a 20 ms step, and
# exactly the kind of thing a timing alone cannot tell you about.
#
# Printed rather than asserted: this is a measurement script, and a reader who
# sees the count move should distrust the table above it.
# ---------------------------------------------------------------------------
dev_state <- get(".ag_device_state", envir = ns)
cat(sprintf("\npersistent buffers held at exit: %d  (flat = no leak; growth ",
            length(dev_state$p_buffers)))
cat("invalidates the timings above)\n")
cat(sprintf("pass buffers held at exit:       %d\n",
            length(dev_state$buffers)))
cat("\nDone.\n")
