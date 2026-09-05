#!/usr/bin/env Rscript
#
# What does one ag_* operation on the GPU actually cost, and how much of that
# is re-uploading operands that did not change?
#
# The fused-kernels item in TODO.md section 3 was closed by measurement: on the
# ag_* path the arithmetic is not what dominates. A trivial ag_scale on a 1024
# square costs more than the whole matmul does on the CPU, so fusing pairs of
# ops removes a fraction of the wrong cost. This script measures the cost that
# is actually there, and separates its two parts:
#
#   fixed   -- context, graph build, dispatch, download: paid per call
#   upload  -- moving operands to the device: paid per call, per operand
#
# The second one should not be paid at all for an operand that has not changed
# since the last call. .ag_run_op creates fresh tensors and calls
# ggml_backend_tensor_set_data on every input every time, so a weight matrix is
# re-sent on every operation that touches it. The download side already avoids
# this: .ag_data() caches the host copy per ctx_gen (R/ag_device.R:795) rather
# than reading the device repeatedly. Uploads have no such cache.
#
# Three things this script does that a quick interactive check does not:
#
#   * several warm-up calls, not one. The first call pays for buffer allocation
#     and pipeline setup; a single warm-up hides that but leaves other one-off
#     costs in the first measured iteration.
#   * non-square shapes. A d x d operand makes FLOP and bytes scale together,
#     which is exactly the case where the two cannot be told apart. Tall-thin
#     and short-wide shapes separate them.
#   * both weight regimes. During training optimizer$step() rewrites the
#     weights every iteration, so an upload cache can only help within a single
#     forward/backward -- where the same weight is read by several ops -- and
#     across steps only in eval. Those are measured separately because they
#     have different answers.
#
# Run:  Rscript inst/scripts/measure_ag_upload_cost.R
# Env:  GGMLR_UP_REPS   iterations per timing (default 20)
#       GGMLR_UP_WARMUP warm-up calls before each timing (default 4)

suppressMessages(library(ggmlR))

reps   <- as.integer(Sys.getenv("GGMLR_UP_REPS",   "20"))
warmup <- as.integer(Sys.getenv("GGMLR_UP_WARMUP", "4"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure here.\n")
  quit(status = 0L)
}

tm <- function(f) {
  for (i in seq_len(warmup)) f()
  t0 <- Sys.time()
  for (i in seq_len(reps)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps
}

mb <- function(nr, nc) nr * nc * 4 / 1024^2

cat(sprintf("reps = %d, warm-up = %d\n\n", reps, warmup))

# ---------------------------------------------------------------------------
# 1. Fixed cost vs operand size
#
# ag_scale does almost no arithmetic, so its time is the per-call overhead plus
# the upload of one operand. Growing the operand while the work stays trivial
# prices the upload directly.
# ---------------------------------------------------------------------------

cat("1. Trivial op (ag_scale): per-call cost against operand size\n")
cat("   shape           MB     ms\n")
ag_device("gpu")
for (shape in list(c(64L, 64L), c(256L, 256L), c(512L, 512L), c(1024L, 1024L),
                   c(4096L, 64L), c(64L, 4096L))) {
  nr <- shape[[1L]]; nc <- shape[[2L]]
  t  <- ag_tensor(matrix(rnorm(nr * nc), nr, nc))
  ms <- tm(function() ag_scale(t, 1.0))
  cat(sprintf("   %5dx%-5d  %6.2f  %6.2f\n", nr, nc, mb(nr, nc), ms))
}

# ---------------------------------------------------------------------------
# 2. Does the time follow the arithmetic or the bytes?
#
# Same left operand, right operand shrinking. FLOP falls with it; the bytes
# uploaded for the left operand do not. If time tracks FLOP the kernel
# dominates; if it flattens out, the transfer does.
# ---------------------------------------------------------------------------

cat("\n2. matmul with a shrinking right operand (left operand fixed at 1024x1024)\n")
cat("   rhs cols   GFLOP    ms     ms per GFLOP\n")
A <- ag_tensor(matrix(rnorm(1024L * 1024L), 1024L, 1024L))
for (nc in c(1024L, 256L, 64L, 8L)) {
  B     <- ag_tensor(matrix(rnorm(1024L * nc), 1024L, nc))
  ms    <- tm(function() ag_matmul(A, B))
  gflop <- 2 * 1024 * 1024 * nc / 1e9
  cat(sprintf("   %8d  %6.3f  %6.2f  %10.2f\n", nc, gflop, ms, ms / gflop))
}

# ---------------------------------------------------------------------------
# 3. The same weight used several times in one pass
#
# This is where an upload cache pays off during training: within a single
# forward the weights do not change, yet every op re-sends them.
# ---------------------------------------------------------------------------

cat("\n3. One weight, N ops per pass (uploads it N times today)\n")
cat("   n_ops     ms    ms per op\n")
W <- ag_tensor(matrix(rnorm(1024L * 1024L), 1024L, 1024L))
X <- ag_tensor(matrix(rnorm(1024L * 64L),   1024L, 64L))
for (n_ops in c(1L, 2L, 4L, 8L)) {
  ms <- tm(function() {
    out <- X
    for (i in seq_len(n_ops)) out <- ag_matmul(W, out)
    out
  })
  cat(sprintf("   %5d  %6.2f  %9.2f\n", n_ops, ms, ms / n_ops))
}

# ---------------------------------------------------------------------------
# 4. Training vs eval
#
# optimizer$step() rewrites the weights, so across steps a cache helps only
# when they are NOT being updated. Both regimes run the same forward; the only
# difference is whether a step happens between passes.
# ---------------------------------------------------------------------------

cat("\n4. Weights rewritten every pass (training) vs held fixed (eval)\n")
d     <- 512L
batch <- 64L
Wp    <- ag_param(matrix(rnorm(d * d), d, d))
Xt    <- ag_tensor(matrix(rnorm(d * batch), d, batch))
Y     <- matrix(0.0, d, batch)
opt   <- optimizer_sgd(list(W = Wp), lr = 1e-4)

train_pass <- function() {
  with_grad_tape({ loss <- ag_mse_loss(ag_matmul(Wp, Xt), Y) })
  grads <- backward(loss)
  # step() gained an optional `grads` in 0.8.x; pass it explicitly so this
  # script also runs against an older installed build.
  opt$step(grads)
  opt$zero_grad()
}
eval_pass <- function() ag_matmul(Wp, Xt)

cat(sprintf("   training pass (step rewrites W) : %6.2f ms\n", tm(train_pass)))
cat(sprintf("   eval pass     (W unchanged)     : %6.2f ms\n", tm(eval_pass)))

# ---------------------------------------------------------------------------
# 5. The CPU baseline the GPU path has to beat
# ---------------------------------------------------------------------------

cat("\n5. Same ops on the CPU path, for reference\n")
ag_device("cpu")
for (n in c(256L, 512L, 1024L)) {
  a  <- ag_tensor(matrix(rnorm(n * n), n, n))
  b  <- ag_tensor(matrix(rnorm(n * n), n, n))
  cat(sprintf("   matmul %4dx%-4d  CPU %7.2f ms\n", n, n,
              tm(function() ag_matmul(a, b))))
}
ag_device("cpu")

cat("\nReading the numbers:\n")
cat("  Section 1 flat across sizes  -> per-call overhead dominates, uploads are cheap.\n")
cat("  Section 1 rising with MB     -> uploads are the cost; an upload cache is worth it.\n")
cat("  Section 2 flat ms per GFLOP  -> the kernel dominates (fusion would help).\n")
cat("  Section 2 rising ms per GFLOP-> fixed cost dominates at small work.\n")
cat("  Section 3 ms per op falling  -> some cost is amortised across ops already.\n")
cat("  Section 4 eval << training   -> caching uploads pays off outside the step.\n")
