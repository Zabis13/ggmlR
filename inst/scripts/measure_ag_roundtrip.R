#!/usr/bin/env Rscript
#
# What does the per-op GPU round trip in .ag_run_op cost?
#
# TODO theme 2 opens with "the autograd tape is not GPU-resident". That claim
# deserved the same scrutiny that overturned two earlier measurements in this
# area: a "GPU problem" can easily turn out to be a badly built test stand
# rather than a property of the code.
#
# Here it is not. .ag_run_op (R/ag_device.R) does the following FOR EVERY
# OPERATION: ggml_init a fresh context -> allocate a backend buffer -> upload
# every input -> compute a one-node graph -> download the result -> free
# everything. Two ag_matmul calls in a row therefore push the same matrix across
# the bus twice, even though the first result never needed to come back.
#
# This measures three points, all computing the SAME thing -- a chain of matmuls:
#
#   ag_* on GPU     one context, one buffer, one upload+download per op
#   ag_* on CPU     the same R tape, no GPU involved
#   single graph    the whole chain as one ggml graph, one upload+download
#
# The third is what a resident tape could achieve, so the gap between it and the
# first is the price of the round trips -- not an estimate, a measurement.
#
# Run:  Rscript inst/scripts/measure_ag_roundtrip.R
# Env:  GGMLR_AG_BENCH_REPS   iterations per timing (default 20)
#       GGMLR_AG_BENCH_DEPTH  ops in the chain (default 8)
#       GGMLR_AG_BENCH_N      matrix side (default 512)

suppressMessages(library(ggmlR))

reps   <- as.integer(Sys.getenv("GGMLR_AG_BENCH_REPS", "20"))
depth  <- as.integer(Sys.getenv("GGMLR_AG_BENCH_DEPTH", "8"))
n      <- as.integer(Sys.getenv("GGMLR_AG_BENCH_N", "512"))
batches <- 5L

have_gpu <- ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

set.seed(7L)
A0 <- matrix(runif(n * n, -1, 1), n, n)
W  <- lapply(seq_len(depth), function(i) matrix(runif(n * n, -1, 1) / sqrt(n), n, n))

timed <- function(f) {
  f()                                   # warm up
  b <- vapply(seq_len(batches), function(...) {
    t0 <- Sys.time()
    for (i in seq_len(reps)) f()
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
  }, numeric(1))
  list(ms = stats::median(b), lo = min(b), hi = max(b))
}

# --- 1/2. the autograd tape, one .ag_run_op call per matmul ----------------
ag_chain <- function() {
  x <- ag_tensor(A0)
  for (i in seq_len(depth)) x <- ag_matmul(x, ag_tensor(W[[i]]))
  invisible(x$data)   # ag_tensor is an environment; $data is the materialised matrix
}

# --- 3. the same chain as a single ggml graph ------------------------------
# Everything stays in device memory between matmuls: one upload of the inputs,
# one download of the result, no per-op context or buffer churn.
graph_chain <- function(backend) {
  ctx <- ggml_init(1024 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  # ggml_mul_mat(src0[k,m], src1[k,n]) -> [m,n], so the left operand goes in
  # transposed, exactly as .ag_gpu_matmul does it.
  xt <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n)
  ws <- lapply(seq_len(depth), function(i) ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n))

  node <- xt
  for (i in seq_len(depth)) node <- ggml_mul_mat(ctx, node, ws[[i]])

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)

  ggml_backend_tensor_set_data(xt, as.numeric(t(A0)))
  for (i in seq_len(depth)) ggml_backend_tensor_set_data(ws[[i]], as.numeric(W[[i]]))

  ggml_backend_graph_compute(backend, ggml_build_forward_expand(ctx, node))
  invisible(ggml_backend_tensor_get_data(node))
}

cat(sprintf("Chain of %d matmuls, %dx%d, %d reps x %d batches\n\n",
            depth, n, n, reps, batches))
cat(sprintf("%-30s %10s %16s\n", "configuration", "ms/pass", "(min-max)"))
cat(strrep("-", 60), "\n")

show <- function(label, r)
  cat(sprintf("%-30s %10.2f %16s\n", label, r$ms, sprintf("%.2f-%.2f", r$lo, r$hi)))

res <- list()

if (have_gpu) {
  ag_device("gpu")
  res$ag_gpu <- timed(ag_chain)
  show("ag_* tape, GPU (per-op)", res$ag_gpu)
}

ag_device("cpu")
res$ag_cpu <- timed(ag_chain)
show("ag_* tape, CPU", res$ag_cpu)

if (have_gpu) {
  gpu <- ggml_vulkan_init(0L)
  res$graph_gpu <- timed(function() graph_chain(gpu))
  show("single graph, GPU (resident)", res$graph_gpu)
  ggml_vulkan_free(gpu)
}

cpu <- ggml_backend_cpu_init()
ggml_backend_cpu_set_n_threads(cpu, 2L)
res$graph_cpu <- timed(function() graph_chain(cpu))
show("single graph, CPU", res$graph_cpu)
ggml_backend_free(cpu)

cat("\n")
cat(strrep("=", 60), "\n")

if (have_gpu) {
  overhead <- res$ag_gpu$ms - res$graph_gpu$ms
  factor   <- res$ag_gpu$ms / res$graph_gpu$ms
  cat(sprintf("Per-op round trips cost %.1f ms per pass (%.1fx slower than one graph:\n",
              overhead, factor))
  cat(sprintf("  %.1f ms for the tape vs %.1f ms for the same maths resident on the GPU).\n",
              res$ag_gpu$ms, res$graph_gpu$ms))
  cat(sprintf("That is %.1f ms per operation, at depth %d.\n", overhead / depth, depth))

  if (res$ag_gpu$ms > res$ag_cpu$ms) {
    cat(sprintf("\nNote: the GPU tape (%.1f ms) is SLOWER than the CPU tape (%.1f ms) --\n",
                res$ag_gpu$ms, res$ag_cpu$ms))
    cat("at this size the copying costs more than the GPU maths saves.\n")
  }
} else {
  cat("No Vulkan device: only the CPU columns are meaningful.\n")
}
