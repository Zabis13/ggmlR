#!/usr/bin/env Rscript
#
# Is a GPU backward actually faster than the R BLAS one? -- PoC, not a feature.
#
# TODO theme 3 names this "the main item": grad_fn is an R closure that captures
# a_snap/b_snap and computes grad_out %*% t(b_snap) on the host, so backward
# never touches the GPU. The 2026-09-03 measurement put that at 58.8 ms against
# a 6.6 ms resident forward, which is why it is the main item.
#
# But this area has already produced three measurements that overturned the
# hypothesis they were meant to confirm (round-trip diagnosis, transpose-in-graph,
# GPU-SGD). "Move it to the GPU" is not self-evidently a win here: .ag_run_op
# uploads, computes and downloads PER OP, and that round trip is exactly what
# makes the current forward cost 42 ms instead of 6.6. A backward built the same
# way would inherit the same tax.
#
# So this measures the SAME quantity -- dL/dW for every W in a matmul chain
# under an MSE loss -- four ways, varying exactly one thing: how backward runs.
#
#   1. r_blas        the current backward(): R closures, %*% and t()
#   2. gpu_per_op    variant A: each grad_fn dispatched through .ag_run_op,
#                    i.e. one graph + one round trip per gradient
#   3. gpu_one_graph variant B: the whole backward as ONE ggml graph -- inputs
#                    already resident, one compute, one download per gradient
#   4. fwd_one_graph the forward chain as one graph, for scale: the hardware
#                    ceiling this machine can offer at all
#
# 2 exists to separate "GPU is faster" from "one graph is faster". If 3 beats 1
# but 2 does not, the win is batching, not the device -- which decides how the
# real implementation has to be written.
#
# Correctness is checked, not assumed: 2 and 3 are compared against 1 by
# max-abs difference. A fast wrong backward is not a result.
#
# Run:  Rscript inst/scripts/measure_ag_backward.R
# Env:  GGMLR_AG_BENCH_REPS   iterations per timing (default 20)
#       GGMLR_AG_BENCH_DEPTH  matmuls in the chain (default 8)
#       GGMLR_AG_BENCH_N      matrix side (default 512)

suppressMessages(library(ggmlR))

reps    <- as.integer(Sys.getenv("GGMLR_AG_BENCH_REPS",  "20"))
depth   <- as.integer(Sys.getenv("GGMLR_AG_BENCH_DEPTH", "8"))
n       <- as.integer(Sys.getenv("GGMLR_AG_BENCH_N",     "512"))
batches <- 5L

have_gpu <- ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

cat("ggmlR ag backward PoC\n")
cat("  chain depth : ", depth, " matmuls of ", n, "x", n, "\n", sep = "")
cat("  reps        : ", reps, " x ", batches, " batches\n", sep = "")
cat("  gpu         : ", if (have_gpu) "yes" else "NO -- GPU rows skipped", "\n\n", sep = "")

set.seed(7L)
X0 <- matrix(runif(n * n, -1, 1), n, n)
W  <- lapply(seq_len(depth), function(i) matrix(runif(n * n, -1, 1) / sqrt(n), n, n))
Y  <- matrix(runif(n * n, -1, 1), n, n)

timed <- function(f) {
  f()                                   # warm up
  b <- vapply(seq_len(batches), function(...) {
    t0 <- Sys.time()
    for (i in seq_len(reps)) f()
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
  }, numeric(1))
  list(ms = stats::median(b), lo = min(b), hi = max(b))
}

# --- the forward activations, shared by every backward variant ---------------
#
# H[[i]] is the input to matmul i, so H[[1]] = X0 and H[[i+1]] = H[[i]] %*% W[[i]].
# Every backward below consumes exactly these, so none of them is charged for
# the forward pass -- the comparison is backward against backward.
forward_acts <- function() {
  H <- vector("list", depth + 1L)
  H[[1L]] <- X0
  for (i in seq_len(depth)) H[[i + 1L]] <- H[[i]] %*% W[[i]]
  H
}
H <- forward_acts()
pred <- H[[depth + 1L]]

# dL/dpred for MSE loss mean((pred - Y)^2)
seed_grad <- 2 * (pred - Y) / length(pred)

# --- 1. the current backward: R closures over host matrices -----------------
#
# This is what grad_fn does today, unrolled: the matmul rule is
#   dL/dA = grad_out %*% t(B)    dL/dB = t(A) %*% grad_out
# walked from the loss end of the chain back to the start.
bwd_r_blas <- function() {
  g  <- seed_grad
  dW <- vector("list", depth)
  for (i in rev(seq_len(depth))) {
    dW[[i]] <- t(H[[i]]) %*% g
    g       <- g %*% t(W[[i]])
  }
  dW
}

ref_dW <- bwd_r_blas()

maxdiff <- function(a, b) {
  max(vapply(seq_along(a), function(i) max(abs(a[[i]] - b[[i]])), numeric(1)))
}

# --- 2. variant A: one .ag_run_op per gradient ------------------------------
#
# Same maths, but each %*% goes through the per-op path the ag_* ops use today:
# fresh graph, upload both operands, compute, download. Two matmuls per link of
# the chain, so 2*depth round trips.
#
# .ag_gpu_matmul is internal; ::: is deliberate -- this is a probe, not package
# code. It transposes its first argument in R, matching .ag_gpu_matmul's own
# contract (see the comment at R/ag_device.R:438).
bwd_gpu_per_op <- function() {
  mm <- ggmlR:::.ag_gpu_matmul
  g  <- seed_grad
  dW <- vector("list", depth)
  for (i in rev(seq_len(depth))) {
    dW[[i]] <- mm(t(H[[i]]), g)
    g       <- mm(g, t(W[[i]]))
  }
  dW
}

# --- 3. variant B: the whole backward as ONE ggml graph ---------------------
#
# The point of the variant. Inputs (activations, weights, seed gradient) are
# uploaded ONCE into a resident context; the backward chain is built as graph
# nodes; one compute runs all of it; only the depth gradient matrices come back.
#
# ggml_mul_mat(src0[K,M], src1[K,N]) -> [M,N], i.e. it contracts ne[0] of both
# operands, so an R matrix [m,k] (which lands as ne0=m, ne1=k) has to be
# uploaded already transposed wherever it plays the left-hand role. The uploads
# below therefore carry t(...) where the maths needs A %*% B, exactly as
# .ag_gpu_matmul does -- and that transpose happens once, outside the timed
# region, since these tensors are what "resident activations" would mean.
#
# Returns a closure so the setup (upload) is paid once and only the compute is
# timed -- a resident tape would not re-upload activations on every step.
make_bwd_one_graph <- function(resident = TRUE) {
  if (is.null(ggmlR:::.ag_device_state$backend)) ggmlR:::.ag_init_gpu_backend()
  backend <- ggmlR:::.ag_device_state$backend
  ggml_type <- ggmlR:::.ag_dtype_to_ggml(ggmlR:::.ag_compute_dtype())

  # One context for every tensor this probe needs. Sized generously: the graph
  # itself gets its own context below, as .ag_run_op does.
  #
  #   n_desc = depth activations + depth weights + seed + the nodes built below
  ctx <- ggml_init(512 * 1024 * 1024, no_alloc = TRUE)
  if (is.null(ctx)) stop("failed to create the probe context")

  # Inputs, uploaded transposed where the graph needs them that way.
  #  H_t[[i]]  = t(H[[i]])  used as the left operand of dW[[i]] = t(H[i]) %*% g
  #  W_t[[i]]  = t(W[[i]])  used as the right operand of g %*% t(W[i])
  H_t <- lapply(seq_len(depth), function(i) ggml_new_tensor_2d(ctx, ggml_type, n, n))
  W_p <- lapply(seq_len(depth), function(i) ggml_new_tensor_2d(ctx, ggml_type, n, n))
  G0  <- ggml_new_tensor_2d(ctx, ggml_type, n, n)

  # Build the backward chain as graph nodes.
  #
  #   dW[i] = t(H[i]) %*% g          -> mul_mat(H[i]      , g)     since
  #                                     mul_mat contracts ne0, and H[i] uploaded
  #                                     untransposed already has ne0 = rows(H[i])
  #   g     = g %*% t(W[i])          -> mul_mat(W[i]      , g)     likewise
  #
  # Both left operands are therefore the plain matrices, uploaded as-is; no
  # ggml_transpose node is needed anywhere in the chain. That is the structural
  # advantage of building backward in the graph: the transposes the R version
  # pays for (t(H[[i]]), t(W[[i]])) are free here, they are just how mul_mat
  # already reads its first argument.
  g_node  <- G0
  dW_node <- vector("list", depth)
  for (i in rev(seq_len(depth))) {
    dW_node[[i]] <- ggml_mul_mat(ctx, H_t[[i]], g_node)
    g_node       <- ggml_mul_mat(ctx, W_p[[i]], g_node)
  }

  # Allocate every tensor of this context in one backend buffer.
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  if (is.null(buf)) stop("failed to allocate the probe buffer")

  # Upload the inputs once. This is the "resident activations" assumption: a
  # real resident tape would already have these on the device from the forward
  # pass, so re-uploading them per step would be measuring the wrong thing.
  #
  # ... and that is exactly the assumption `resident = FALSE` removes. Path B has
  # NO resident forward today, so a backward graph built on top of it would have
  # to upload the activations itself. Timing both settings brackets the real win:
  # resident = TRUE is the ceiling (10.9 ms when first measured), FALSE is what
  # is available BEFORE residency exists. The gap between them is the price of
  # not being resident -- and it decides the order of work, since TODO currently
  # claims residency should come after the backward rewrite.
  upload <- function() {
    for (i in seq_len(depth)) {
      ggml_backend_tensor_set_data(H_t[[i]], as.numeric(H[[i]]))
      ggml_backend_tensor_set_data(W_p[[i]], as.numeric(W[[i]]))
    }
    ggml_backend_tensor_set_data(G0, as.numeric(seed_grad))
  }
  if (resident) upload()

  # The graph lives in its own context, freed with it -- same ownership rule as
  # .ag_run_op relies on (R/ag_device.R:406-421): the graph holds pointers to
  # tensors that outlive it, and ggml_free only releases the context's own
  # mem_buffer.
  run <- function() {
    if (!resident) upload()

    ctx_graph <- ggml_init(ggmlR:::.ag_graph_ctx_bytes(), no_alloc = TRUE)
    on.exit(ggml_free(ctx_graph), add = TRUE)

    # Every dW is a separate output: none is reachable from another, so each
    # needs its own expand or it would be pruned from the graph.
    graph <- ggml_build_forward_expand(ctx_graph, dW_node[[1L]])
    for (i in seq_len(depth)[-1L]) ggml_graph_expand(graph, dW_node[[i]])

    ggml_backend_graph_compute(backend, graph)

    lapply(dW_node, function(nd) matrix(ggml_backend_tensor_get_data(nd), n, n))
  }

  list(run = run, free = function() {
    tryCatch(ggml_backend_buffer_free(buf), error = function(e) NULL)
    tryCatch(ggml_free(ctx), error = function(e) NULL)
  })
}

# --- 4. forward as one graph, for scale -------------------------------------
make_fwd_one_graph <- function() {
  if (is.null(ggmlR:::.ag_device_state$backend)) ggmlR:::.ag_init_gpu_backend()
  backend <- ggmlR:::.ag_device_state$backend
  ggml_type <- ggmlR:::.ag_dtype_to_ggml(ggmlR:::.ag_compute_dtype())

  ctx <- ggml_init(512 * 1024 * 1024, no_alloc = TRUE)
  if (is.null(ctx)) stop("failed to create the probe context")

  # x %*% W[i] -> mul_mat needs the shared dim in ne0 of both, so the LEFT
  # operand is uploaded transposed, as .ag_gpu_matmul does.
  Xt <- ggml_new_tensor_2d(ctx, ggml_type, n, n)
  Wp <- lapply(seq_len(depth), function(i) ggml_new_tensor_2d(ctx, ggml_type, n, n))

  node <- Xt
  for (i in seq_len(depth)) node <- ggml_mul_mat(ctx, Wp[[i]], node)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  if (is.null(buf)) stop("failed to allocate the probe buffer")

  ggml_backend_tensor_set_data(Xt, as.numeric(X0))
  for (i in seq_len(depth)) ggml_backend_tensor_set_data(Wp[[i]], as.numeric(W[[i]]))

  run <- function() {
    ctx_graph <- ggml_init(ggmlR:::.ag_graph_ctx_bytes(), no_alloc = TRUE)
    on.exit(ggml_free(ctx_graph), add = TRUE)
    graph <- ggml_build_forward_expand(ctx_graph, node)
    ggml_backend_graph_compute(backend, graph)
    matrix(ggml_backend_tensor_get_data(node), n, n)
  }

  list(run = run, free = function() {
    tryCatch(ggml_backend_buffer_free(buf), error = function(e) NULL)
    tryCatch(ggml_free(ctx), error = function(e) NULL)
  })
}

# --- run ---------------------------------------------------------------------

report <- function(label, t, note = "") {
  cat(sprintf("  %-16s %8.1f ms  [%.1f-%.1f]  %s\n", label, t$ms, t$lo, t$hi, note))
}

cat("backward of a ", depth, "-matmul chain (", n, "x", n, "), same dW every row:\n", sep = "")

t1 <- timed(bwd_r_blas)
report("r_blas", t1, "baseline: current backward()")

if (have_gpu) {
  ag_device("gpu")

  d2 <- maxdiff(ref_dW, bwd_gpu_per_op())
  t2 <- timed(bwd_gpu_per_op)
  report("gpu_per_op", t2, sprintf("variant A | maxdiff %.2e", d2))

  b3 <- make_bwd_one_graph(resident = TRUE)
  on.exit(b3$free(), add = TRUE)
  d3 <- maxdiff(ref_dW, b3$run())
  t3 <- timed(b3$run)
  report("gpu_one_graph", t3, sprintf("variant B | maxdiff %.2e | acts RESIDENT", d3))

  # Same graph, same maths, one variable changed: the activations are uploaded
  # inside the timed region instead of once up front. This is variant B as it
  # would actually perform TODAY, on top of a forward that is not resident.
  b5 <- make_bwd_one_graph(resident = FALSE)
  on.exit(b5$free(), add = TRUE)
  d5 <- maxdiff(ref_dW, b5$run())
  t5 <- timed(b5$run)
  report("gpu_one_graph_up", t5, sprintf("variant B | maxdiff %.2e | acts UPLOADED", d5))

  f4 <- make_fwd_one_graph()
  on.exit(f4$free(), add = TRUE)
  t4 <- timed(f4$run)
  report("fwd_one_graph", t4, "forward only, for scale")

  cat("\n")
  cat(sprintf("  B resident vs current backward : %.2fx  (ceiling)\n",  t1$ms / t3$ms))
  cat(sprintf("  B uploaded vs current backward : %.2fx  (available now)\n", t1$ms / t5$ms))
  cat(sprintf("  B resident vs variant A        : %.2fx\n", t2$ms / t3$ms))
  cat(sprintf("  price of NOT being resident    : %.1f ms\n", t5$ms - t3$ms))
  cat("\n")
  cat("Read it this way:\n")
  cat("  * A slow, B fast -> the win is ONE GRAPH, not the device. The real\n")
  cat("    implementation must build a graph across the whole tape; a per-op\n")
  cat("    GPU grad_fn would reproduce A's numbers, not B's.\n")
  cat("  * B uploaded still beats r_blas -> the backward rewrite pays off ON ITS\n")
  cat("    OWN, before any resident forward exists. Do backward first, as TODO\n")
  cat("    has it, and residency becomes a later top-up.\n")
  cat("  * B uploaded does NOT beat r_blas -> the order in TODO is wrong: the\n")
  cat("    graph is only worth building once activations already live on the\n")
  cat("    device, so resident forward has to come FIRST.\n")
  cat("  * maxdiff should sit near f16/f32 accumulation noise (~1e-3 relative).\n")
  cat("    Anything larger means the graph computes something else, and the\n")
  cat("    timing above is meaningless.\n")
} else {
  cat("\n  No Vulkan device: only the r_blas baseline ran.\n")
}
