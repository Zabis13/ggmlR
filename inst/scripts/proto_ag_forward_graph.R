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
#   per_op    today's path: .ag_run_op per operation, upload and download each.
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

fwd_per_op <- function(Wt, X, d, b) {
  h <- X
  for (W in Wt) {
    h <- run_op(mm(), list(W, h), c(d, b))
    h <- run_op(rl(), list(h),    c(d, b))
  }
  h
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

models <- list(
  list(tag = "d=256  b=32   depth=4",  d = 256L,  b = 32L,   depth = 4L,
       step_ms = 9.95,  fwd_share = 0.401),
  list(tag = "d=512  b=64   depth=4",  d = 512L,  b = 64L,   depth = 4L,
       step_ms = 21.89, fwd_share = 0.492),
  list(tag = "d=1024 b=256  depth=4",  d = 1024L, b = 256L,  depth = 4L,
       step_ms = 91.61, fwd_share = 0.429),
  list(tag = "d=4096 b=256  depth=4",  d = 4096L, b = 256L,  depth = 4L),
  list(tag = "d=1024 b=1024 depth=4",  d = 1024L, b = 1024L, depth = 4L),
  list(tag = "d=1024 b=256  depth=16", d = 1024L, b = 256L,  depth = 16L),
  list(tag = "d=4096 b=1024 depth=16", d = 4096L, b = 1024L, depth = 16L)
)

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

    res[[m$tag]] <- list(m = m, reps = RP, t_op = t_op, t_gr = t_gr,
                         t_cpu = t_cpu, md = md, mdc = mdc,
                         s_op = s_op, s_gr = s_gr)
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

cat("4. On a full training step (Amdahl, measured forward share only)\n")
cat("   model                  fwd share   step ms   graph step x   saves ms\n")
for (r in res) {
  p <- r$m$fwd_share
  if (is.null(p)) next
  s  <- r$t_op / r$t_gr
  sa <- 1 / ((1 - p) + p / s)
  cat(sprintf("   %-22s %8.1f%% %9.2f %12.2fx %10.2f\n",
              r$m$tag, 100 * p, r$m$step_ms, sa,
              r$m$step_ms - r$m$step_ms / sa))
}
np <- Filter(function(r) is.null(r$m$fwd_share), res)
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
cat("  d=4096 rows              -> compare against the resident prototype's\n")
cat("                              collapse (0.35x). If the graph holds up where\n")
cat("                              residency fell over, the threshold is about\n")
cat("                              per-op allocation, not about size as such.\n")
