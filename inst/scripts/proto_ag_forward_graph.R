#!/usr/bin/env Rscript
#
# PROTOTYPE. Does the unified forward graph survive contact with the real path?
#
# The claim under test. Section 4 of measure_ag_forward_profile.R put the whole
# forward as ONE ggml_cgraph at 1.91-3.50x against the per-op path, and TODO
# now says to enter the resident work through that graph rather than through
# per-op residency (which measured 1.13x/0.93x/1.13x on a step and is closed).
# Rebuilding with_grad_tape around a deferred graph is the largest piece of work
# in the section, so the number it rests on has to be a measurement of the real
# thing, not of an idealised one.
#
# Why this is not already answered. The residency ceiling looked like 8.20x on
# the forward and arrived at 1.35x on a step; the resident prototype looked
# positive and came in at 0.35x past d=2048. Both times the gap was work the
# idealised version did not do. For a unified graph that work is EMIT: building
# the nodes in R, one call per operation, which in the graph BACKWARD measured
# 12-24% and did not go away. Section 4 built its chain with a bare loop over
# ggml_mul_mat; the real path must also carry the R-side dispatch.
#
# Rows, one variable at a time:
#   per_op    today's path: .ag_run_op per operation, RESIDENT -- intermediates
#             stay on the device and only the final value is downloaded, so the
#             single variable against the graph row is the number of computes
#             (2*depth vs 1), not the number of crossings. (Before the
#             residency work this row downloaded every intermediate, which
#             flattered the graph by traffic that no longer exists.)
#   graph     the same chain as ONE graph: one ctx, one alloc, one compute, one
#             download -- built through the same primitives .ag_bwd_run_graph
#             uses, with per-stage timing so `emit` is visible rather than
#             folded into a total.
#   cpu       R's own %*% and pmax, threaded BLAS.
#
# Geometry, sizes and the Amdahl rule are the resident prototype's, so the two
# results are directly comparable: one weight per layer (not one shared weight),
# the three measured points carry a real step time and forward share, the four
# scaling probes print forward numbers only.
#
# ⚠️ Every timed pass resets the tape (see feedback_reset_tape_between_reps): a
# resident tensor otherwise survives the iteration and the growing context drove
# `ctx` from 0.3 to 8 ms in the first run of the resident prototype, poisoning
# the control row along with everything else.
#
# Section 5 (added after the deferred forward landed) measures the REAL engine
# rather than a prototype of it: the fused forward+backward path, under three
# reading patterns. The variable it isolates is how often the calling loop reads
# the loss, because the barrier fires at the first read of a value -- so a loop
# that prints its loss every step splits the fused graph back into two, and the
# same engine has two different answers depending on the caller's habit. Both
# engines are timed under all three patterns, so the comparison still varies one
# thing at a time.
#
# Run:  Rscript inst/scripts/proto_ag_forward_graph.R
# Env:  GGMLR_PROTO_REPS   timed passes per row (default 20)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_PROTO_REPS", "20"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns       <- asNamespace("ggmlR")
run_op   <- get(".ag_run_op",            envir = ns)
ctx_ens  <- get(".ag_ctx_ensure",        envir = ns)
ctx_flush<- get(".ag_ctx_flush",         envir = ns)
reset    <- get(".ag_residency_reset",   envir = ns)
gctx_b   <- get(".ag_graph_ctx_bytes",   envir = ns)
dt_ggml  <- get(".ag_dtype_to_ggml",     envir = ns)
cdtype   <- get(".ag_compute_dtype",     envir = ns)
state    <- get(".ag_device_state",      envir = ns)
as_mat   <- get(".ag_as_matrix",         envir = ns)
# .ag_data, not ag_data: the accessor is internal (the package exports no
# ag_data), and calling the public name is what made the first run of section 5
# report "не могу найти функцию" for all three step models.
ag_dat   <- get(".ag_data",              envir = ns)
# NULL on a tree without the deferred forward, so this script still runs there
# and simply prints no fused rows -- rather than failing and taking the whole
# comparison down with it.
defer_on <- tryCatch(get("ag_defer_forward", envir = ns),
                     error = function(e) NULL)

ag_device("gpu")
on.exit(ag_device("cpu"), add = TRUE)
get(".ag_tape_mem_limit", envir = ns)(12 * 1024^3)

`%||%` <- function(a, b) if (is.null(a)) b else a

RP <- reps

tm <- function(f, warm = 2L) {
  for (i in seq_len(warm)) { reset(); f() }
  reset()
  t0 <- Sys.time()
  for (i in seq_len(RP)) { f(); reset() }
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / RP
}

tm_plain <- function(f, warm = 2L) {
  for (i in seq_len(warm)) f()
  t0 <- Sys.time()
  for (i in seq_len(RP)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / RP
}

# --- row 1: today ------------------------------------------------------------
mm <- function() function(ctx, p) ggml_mul_mat(ctx, p[[1L]], p[[2L]])
rl <- function() function(ctx, p) ggml_relu(ctx, p[[1L]])

# ⚠️ resident = TRUE, and the final result is materialised explicitly.
#
# The first version of this row called run_op() with the default resident =
# FALSE, which downloads EVERY intermediate to the host -- two crossings per
# layer. That is not today's path: since the residency work landed, the forward
# keeps intermediates on the device as handles and only the last value comes
# back (.ag_gpu_matmul propagates residency through `resident =
# .ag_is_handle(b_data)`). Timing the graph against a control that pays 2*depth
# transfers it does not have to pay would inflate `graph x` by exactly the
# traffic residency already removed -- the "vary one variable" trap this
# script's own header warns about.
#
# So the control now differs from the graph row in ONE thing: how many computes
# it takes (2*depth vs 1), not how many crossings.
fwd_per_op <- function(Wt, X, d, b) {
  h <- X
  for (W in Wt) {
    h <- run_op(mm(), list(W, h), c(d, b), resident = TRUE)
    h <- run_op(rl(), list(h),    c(d, b), resident = TRUE)
  }
  # The graph row ends with one download; this one must too, or the comparison
  # charges the graph for a transfer the control skipped.
  as_mat(h)
}

# --- row 2: one graph --------------------------------------------------------
#
# Built the way .ag_bwd_run_graph builds the backward, because that is what the
# real implementation would reuse: one context sized from the chain, operand
# tensors created and collected for a single upload pass after allocation
# (nothing has memory before ggml_backend_alloc_ctx_tensors), the whole chain
# expanded into one cgraph, one compute, one download.
#
# The stage names match the per-op profiler's, plus `emit` -- the R-side node
# building that section 4 of the profile did not have to do and the graph
# backward measured at 12-24%.
#
# `stages` is an ENVIRONMENT, not a list, and that is deliberate: `<<-` assigns
# along the LEXICAL chain, so writing to a list argument updates fwd_graph's own
# copy and the caller never sees it -- which is exactly how the first run of this
# script printed a table of zeros. An environment is shared by reference, so
# mark() accumulates into the object the caller holds.
fwd_graph <- function(Wt, X, d, b, stages = NULL) {
  tk <- if (is.null(stages)) NULL else Sys.time()
  mark <- function(nm) {
    if (is.null(stages)) return(invisible(NULL))
    now <- Sys.time()
    prev <- if (exists(nm, envir = stages, inherits = FALSE))
              get(nm, envir = stages) else 0
    assign(nm, prev + as.numeric(difftime(now, tk, units = "secs")) * 1000,
           envir = stages)
    tk <<- now
    invisible(NULL)
  }

  backend <- state$backend
  gt      <- dt_ggml(cdtype())

  # One tensor per weight, one for the input, one node per op, plus slack --
  # overflowing a context aborts R inside ggml rather than returning.
  # Descriptors: one tensor per weight plus the input, two nodes per layer
  # (mul_mat and relu), and slack -- overflowing a context aborts R inside
  # ggml_new_tensor_impl rather than returning an error.
  ctx <- ctx_ens(3L * length(Wt) + 16L)
  mark("ctx")

  # The matrix is stored as-is and flattened at upload time, NOT here.
  #
  # .ag_bwd_run_graph does `val = as.numeric(m)` inside const(), and copying
  # that verbatim is what made the first working run of this script attribute
  # 79% of the graph's time to `create` (1193 ms at d=4096): as.numeric() copies
  # the whole matrix in R, so every weight was duplicated -- 2 GB of memcpy on
  # the largest point -- and charged to the wrong stage. Holding the matrix
  # itself costs nothing (R shares it), and .ag_run_op already flattens at the
  # point of the set_data call, which is where the cost belongs.
  #
  # ⚠️ The same pattern is live in R/ag_backward_graph.R:342, where it lands in
  # the `emit` stage -- so the 12-24% attributed to node building there may be
  # this copy instead. Worth checking before optimising anything named emit.
  uploads <- list()
  const <- function(m) {
    tt <- ggml_new_tensor_2d(ctx, gt, nrow(m), ncol(m))
    uploads[[length(uploads) + 1L]] <<- list(ptr = tt, val = m)
    tt
  }

  tX <- const(X)
  tW <- lapply(Wt, const)
  mark("create")

  node <- tX
  for (w in tW) node <- ggml_relu(ctx, ggml_mul_mat(ctx, w, node))
  mark("emit")

  ctx_flush(ctx)
  mark("flush")

  for (u in uploads) ggml_backend_tensor_set_data(u$ptr, as.numeric(u$val))
  mark("upload")

  ctx_g <- ggml_init(gctx_b(), no_alloc = TRUE)
  on.exit(ggml_free(ctx_g), add = TRUE)
  graph <- ggml_build_forward_expand(ctx_g, node)
  mark("graph")

  ggml_backend_graph_compute(backend, graph)
  mark("compute")

  out <- matrix(ggml_backend_tensor_get_data(node), d, b)
  mark("download")
  out
}

fwd_cpu <- function(Wt, X, d, b) {
  h <- X
  for (W in Wt) h <- pmax(W %*% h, 0)
  h
}

stages_of_graph <- function(Wt, X, d, b, warm = 2L) {
  for (i in seq_len(warm)) { reset(); fwd_graph(Wt, X, d, b) }
  st <- new.env(parent = emptyenv())
  for (i in seq_len(RP)) { reset(); fwd_graph(Wt, X, d, b, stages = st) }
  out <- as.list(st)
  lapply(out, function(v) v / RP)
}

# Per-op stages come from the package's own forward profiler.
fwd_prof  <- get("ag_forward_profile",       envir = ns)
fwd_reset <- get("ag_forward_profile_reset", envir = ns)
fwd_env   <- get(".ag_fwd",                  envir = ns)

stages_of_per_op <- function(Wt, X, d, b, warm = 2L) {
  for (i in seq_len(warm)) { reset(); fwd_per_op(Wt, X, d, b) }
  reset(); fwd_reset(); fwd_prof(TRUE)
  for (i in seq_len(RP)) { fwd_per_op(Wt, X, d, b); reset() }
  fwd_prof(FALSE)
  tot <- fwd_env$totals
  if (is.null(tot)) numeric(0) else tot / RP
}

# `step` marks the models whose FULL training step is measured here (section 4).
#
# ⚠️ It used to be two literals per model -- step_ms = 9.95, fwd_share = 0.401
# and so on -- carried over from a profile taken BEFORE the residency work
# landed. Those numbers described a step of 10 host<->device crossings with a
# host-side Adam and a backward that downloaded its gradients. The step is now
# 4 crossings with step = 0, so it is both shorter and differently divided:
# using the old constants would answer "what would a unified forward have been
# worth in June", which is not the question. Amdahl's p is now measured in the
# same run as the speedup it is applied to.
models <- list(
  list(tag = "d=256  b=32   depth=4",  d = 256L,  b = 32L,   depth = 4L,
       step = TRUE),
  list(tag = "d=512  b=64   depth=4",  d = 512L,  b = 64L,   depth = 4L,
       step = TRUE),
  list(tag = "d=1024 b=256  depth=4",  d = 1024L, b = 256L,  depth = 4L,
       step = TRUE),
  list(tag = "d=4096 b=256  depth=4",  d = 4096L, b = 256L,  depth = 4L),
  list(tag = "d=1024 b=1024 depth=4",  d = 1024L, b = 1024L, depth = 4L),
  list(tag = "d=1024 b=256  depth=16", d = 1024L, b = 256L,  depth = 16L),
  list(tag = "d=4096 b=1024 depth=16", d = 4096L, b = 1024L, depth = 16L)
)

# --- the full training step, measured rather than assumed ---------------------
#
# Amdahl's p is "the fraction of the step the forward occupies", and it decides
# whether a faster forward is worth building. It has to come from the same tree
# and the same run as the speedup, for two reasons:
#
#   1. the step changed underneath the old constants (4 crossings, step = 0),
#   2. p is exactly the quantity a stale number distorts most -- it multiplies
#      the whole conclusion.
#
# The model is the same chain the rows above time -- relu(W_i %*% h) -- so the
# forward being timed is the forward inside this step, not a different one. The
# loss is MSE against a fixed target; its cost lands in `rest` along with
# backward and the optimizer, which is where it belongs.
#
# ⚠️ The tape is reset by with_grad_tape() at the start of each step, so the
# per-rep reset() the other rows need is implicit here. Adding one would free
# the weights, which live in the persistent pool and must survive the step.
step_timing <- function(Wt, X, d, b, reps_step) {
  Wp <- lapply(Wt, ag_param)
  xt <- ag_tensor(X)
  yt <- ag_tensor(matrix(0, d, b))
  opt <- optimizer_adam(stats::setNames(Wp, paste0("W", seq_along(Wp))),
                        lr = 1e-3)

  fwd <- function() {
    h <- xt
    for (W in Wp) h <- ag_relu(ag_matmul(W, h))
    h
  }

  # `read_every` is the variable this section exists to isolate.
  #
  # Fusion computes the forward and the backward as one graph, and the barrier
  # fires at the first read of a VALUE. A loop that prints the loss every step
  # reads it before the optimizer does, which splits the graph again -- so the
  # question "does fusion pay" has a different answer depending on a habit of
  # the calling code, not on anything in the engine.
  #
  # read_every = 1   the ordinary loop: loss read every step (the common case)
  # read_every = 0   never read inside the timed region (the ceiling)
  # read_every = k   read every k-th step, the realistic middle
  #
  # The loss is accumulated into a variable rather than discarded, so R cannot
  # optimise the read away and the comparison stays honest.
  sink_val <- 0
  one_step <- function(i = 1L, read_every = 1L) {
    with_grad_tape({
      out  <- fwd()
      loss <- ag_mse_loss(out, yt)
    })
    backward(loss)
    opt$step()
    opt$zero_grad()
    if (read_every > 0L && (i %% read_every) == 0L)
      sink_val <<- sink_val + as.numeric(ag_dat(loss))
    invisible(NULL)
  }

  timed <- function(read_every) {
    # One untimed step: the first pass through any path pays for context
    # creation, buffer allocation and shader warm-up, none of which is the
    # steady state being measured.
    one_step(1L, read_every)
    t0 <- Sys.time()
    for (i in seq_len(reps_step)) one_step(i, read_every)
    as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps_step
  }

  step_ms <- timed(1L)

  # The forward's share, timed inside a tape so it carries the same recording
  # overhead it does during a step. Without the tape, ag_record is skipped and
  # the forward looks cheaper than it is -- understating p and, with it, the
  # value of the work under consideration.
  t0 <- Sys.time()
  for (i in seq_len(reps_step)) with_grad_tape({ fwd() })
  fwd_ms <- as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps_step

  # The same step under the deferred/fused engine, in three reading patterns.
  #
  # ⚠️ The gate is restored before returning. It is global state, and leaving it
  # on would silently move every later row of the table onto the fused path --
  # the control would stop being a control, which is the failure this script's
  # header warns about in its own terms.
  fused <- rep(NA_real_, 3L)
  if (!is.null(defer_on)) {
    old <- defer_on(TRUE)
    fused <- tryCatch({
      c(read1 = timed(1L),                    # loss read every step
        read8 = timed(8L),                    # read occasionally
        read0 = timed(0L))                    # never read: the ceiling
    }, error = function(e) {
      cat("\n     (fused timing failed: ", conditionMessage(e), ")\n     ",
          sep = "")
      rep(NA_real_, 3L)
    })
    defer_on(old)
  }

  # The unfused engine under the same three patterns, so the comparison varies
  # ONE thing. Reading the loss costs something on the per-op path too (it is a
  # download either way), and charging that difference to fusion would repeat
  # the mistake the per_op row was already fixed for.
  base <- c(read1 = step_ms, read8 = timed(8L), read0 = timed(0L))

  list(step_ms = step_ms, fwd_ms = fwd_ms,
       share = if (step_ms > 0) min(1, fwd_ms / step_ms) else NA_real_,
       base = base, fused = fused)
}

reps_for <- function(m) {
  w <- as.double(m$d) * m$d * m$depth + as.double(m$d) * m$b
  if (w > 2e8) 3L else if (w > 3e7) 6L else reps
}

cat(sprintf("reps = %d for small points, fewer for large ones;\n", reps))
cat("chain = relu(W_i %*% h), one weight per layer\n\n")
cat("  per_op  today's path, one .ag_run_op per operation\n")
cat("  graph   the same chain as ONE ggml_cgraph (emit timed separately)\n")
cat("  cpu     R %*% + pmax, threaded BLAS\n\n")

res <- list()

for (m in models) {
  RP <- reps_for(m)
  cat(sprintf("  measuring %-24s (%d passes) ... ", m$tag, RP))
  flush.console()
  ok <- tryCatch({
    set.seed(1L)
    d <- m$d; b <- m$b
    Wt <- lapply(seq_len(m$depth),
                 function(i) matrix(rnorm(d * d) * 0.05, d, d))
    X  <- matrix(rnorm(d * b), d, b)

    reset(); ref <- fwd_per_op(Wt, X, d, b)
    reset(); got <- fwd_graph(Wt, X, d, b)
    md  <- max(abs(ref - got))
    mdc <- max(abs(ref - fwd_cpu(Wt, X, d, b)))

    t_op    <- tm(function() fwd_per_op(Wt, X, d, b))
    t_gr    <- tm(function() fwd_graph(Wt, X, d, b))
    t_cpu   <- tm_plain(function() fwd_cpu(Wt, X, d, b))

    s_op <- stages_of_per_op(Wt, X, d, b)
    s_gr <- stages_of_graph(Wt, X, d, b)

    # Measured only for the three step models; the scaling probes stay
    # forward-only, as before.
    st <- if (isTRUE(m$step)) {
      reset()
      tryCatch(step_timing(Wt, X, d, b, max(3L, RP %/% 2L)),
               error = function(e) {
                 cat("\n     (step timing failed: ", conditionMessage(e),
                     ")\n     ", sep = "")
                 NULL
               })
    } else NULL
    reset()

    res[[m$tag]] <- list(m = m, reps = RP, t_op = t_op, t_gr = t_gr,
                         t_cpu = t_cpu, md = md, mdc = mdc,
                         s_op = s_op, s_gr = s_gr, st = st)
    TRUE
  }, error = function(e) {
    cat("FAILED: ", conditionMessage(e), "\n", sep = ""); FALSE })
  if (isTRUE(ok)) cat("ok\n")
  try(reset(), silent = TRUE)
  invisible(gc(FALSE))
}
cat("\n")

cat("1. Correctness (maxdiff against the per-op path)\n")
cat("   model                        graph      cpu\n")
for (r in res)
  cat(sprintf("   %-22s %12.3g %8.3g\n", r$m$tag, r$md, r$mdc))
cat("\n")

cat("2. Forward chain, milliseconds per pass\n")
cat("   model                   per_op    graph      cpu   graph x   vs cpu\n")
for (r in res)
  cat(sprintf("   %-22s %8.2f %8.2f %8.2f %8.2fx %8.2fx\n",
              r$m$tag, r$t_op, r$t_gr, r$t_cpu,
              r$t_op / r$t_gr, r$t_cpu / r$t_gr))
cat("\n   graph x = the unified graph against today's per-op path.\n")
cat("   vs cpu  = the unified graph against threaded BLAS.\n\n")

cat("3. Per-stage, milliseconds per pass\n")
cat("   `emit` exists only for the graph row: section 4 of the forward profile\n")
cat("   built its chain with a bare loop and never paid it. In the graph\n")
cat("   BACKWARD the same stage measured 12-24% and did not go away.\n")
keys <- c("ctx", "create", "emit", "flush", "upload", "graph", "compute", "download")
for (r in res) {
  cat(sprintf("\n   %s  (%d passes)\n", r$m$tag, r$reps))
  cat("     stage       per_op     graph\n")
  gop <- function(k) { v <- r$s_op[k]; if (is.na(v)) 0 else as.numeric(v) }
  ggr <- function(k) { v <- r$s_gr[[k]]; if (is.null(v)) 0 else v }
  for (k in keys)
    cat(sprintf("     %-10s %8.2f %9.2f\n", k, gop(k), ggr(k)))
  cat(sprintf("     %-10s %8.2f %9.2f\n", "TOTAL",
              sum(r$s_op), sum(unlist(r$s_gr))))
}
cat("\n")

cat("4. On a full training step (Amdahl; step and share MEASURED in this run)\n")
cat("   step ms  = with_grad_tape + backward + Adam step, this tree, now.\n")
cat("   fwd ms   = the same forward chain timed inside a tape.\n")
cat("   share    = fwd/step, the p a unified forward could act on.\n\n")
cat("   model                   step ms   fwd ms   share   graph step x   saves ms\n")
for (r in res) {
  if (is.null(r$st)) next
  p <- r$st$share
  if (is.na(p)) next
  s  <- r$t_op / r$t_gr
  sa <- 1 / ((1 - p) + p / s)
  cat(sprintf("   %-22s %8.2f %8.2f %6.1f%% %12.2fx %10.2f\n",
              r$m$tag, r$st$step_ms, r$st$fwd_ms, 100 * p, sa,
              r$st$step_ms - r$st$step_ms / sa))
}
cat("\n5. Fused forward+backward, by how often the loss is read\n")
cat("   The engine computes the whole step as ONE graph, and the barrier fires\n")
cat("   at the first read of a value. Reading the loss every step therefore\n")
cat("   splits the graph back apart -- so the payoff depends on the calling\n")
cat("   loop's habit, not only on the engine.\n\n")
cat("   read1 = loss read every step (the ordinary training loop)\n")
cat("   read8 = read every 8th step\n")
cat("   read0 = never read in the timed region (the ceiling)\n\n")
cat("   model                    base r1  fused r1     x |  base r0 fused r0     x\n")
for (r in res) {
  if (is.null(r$st) || is.null(r$st$fused)) next
  b <- r$st$base; f <- r$st$fused
  if (all(is.na(f))) next
  cat(sprintf("   %-22s %8.2f %9.2f %5.2fx | %8.2f %8.2f %5.2fx\n",
              r$m$tag, b[["read1"]], f[["read1"]], b[["read1"]] / f[["read1"]],
              b[["read0"]], f[["read0"]], b[["read0"]] / f[["read0"]]))
}
cat("\n   and the middle pattern (every 8th step):\n")
cat("   model                    base r8  fused r8     x\n")
for (r in res) {
  if (is.null(r$st) || is.null(r$st$fused)) next
  b <- r$st$base; f <- r$st$fused
  if (all(is.na(f))) next
  cat(sprintf("   %-22s %8.2f %9.2f %5.2fx\n",
              r$m$tag, b[["read8"]], f[["read8"]], b[["read8"]] / f[["read8"]]))
}
cat("\n  How to read section 5, decided before the numbers arrived:\n")
cat("    fused r1 >= 1.15x  -> fusion pays in the loop people actually write;\n")
cat("                          enable it by default.\n")
cat("    r1 ~ 1.0 but\n")
cat("    r0 >= 1.2x         -> the engine works and the READ is what costs.\n")
cat("                          Worth keeping behind the gate for loops that\n")
cat("                          batch their logging, not worth a default.\n")
cat("    r0 also ~ 1.0      -> fusion does not move a step on this hardware.\n")
cat("                          Close it by measurement, as the GPU optimizers\n")
cat("                          were closed, and keep the code gated off.\n")
cat("    fused < base       -> the barrier costs more than the merged compute\n")
cat("                          saves; report it rather than explaining it away.\n")

np <- Filter(function(r) is.null(r$st), res)
if (length(np))
  cat(sprintf("\n   (%d scaling probes above have forward numbers only)\n",
              length(np)))

cat("\nReading the result:\n")
cat("  graph x near section 4's 1.9-3.3x -> the number survives the real path;\n")
cat("                              rebuilding with_grad_tape is justified.\n")
cat("  graph x well below it     -> `emit` or `create` ate it, exactly as the\n")
cat("                              residency ceiling was eaten. Look at the\n")
cat("                              stage table before deciding anything.\n")
cat("  graph step x near 1.0     -> even a perfect forward cannot move the step;\n")
cat("                              the graph has to cover backward too, which is\n")
cat("                              level 2 and a larger piece of work.\n")
cat("\n  The decision rule for stage 2, set BEFORE the numbers are in:\n")
cat("    graph step x >= 1.3  -> a deferred forward pays for itself; build it.\n")
cat("    1.1 to 1.3           -> only worth it fused with backward as ONE graph,\n")
cat("                            since the forward alone does not carry it.\n")
cat("    below 1.1            -> close stage 2 by measurement, the way the GPU\n")
cat("                            optimizers were closed. The remaining 4\n")
cat("                            crossings are the batch and the loss scalar,\n")
cat("                            and those are not what a unified graph removes.\n")
cat("  d=4096 rows              -> compare against the resident prototype's\n")
cat("                              collapse (0.35x). If the graph holds up where\n")
cat("                              residency fell over, the threshold is about\n")
cat("                              per-op allocation, not about size as such.\n")
