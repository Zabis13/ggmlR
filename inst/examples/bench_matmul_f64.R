#!/usr/bin/env Rscript
# ============================================================================
# FP64 matmul PoC — is honest double-precision matmul on the GPU worth it?
# ============================================================================
# ggml/Vulkan has no fp64 matmul (the GPU path is f32/f16); this measures a
# minimal hand-written double kernel (matmul_f64.comp) against CPU double
# (openBLAS, multithreaded) to decide whether a real fp64 path is worthwhile
# despite the hardware fp64:fp32 throughput penalty (RDNA4 ~1:16, GeForce ~1:64).
#
# It reports, per size n (square n x n):
#   * cpu_f64  : R's %*% (double, openBLAS) — the bar to beat
#   * gpu_f64  : the matmul_f64.comp kernel (honest double on the GPU)
#   * gpu_f32  : ggml_matmul() (our f32 GPU path) for context
#   * correctness of gpu_f64 vs the CPU reference (should be ~1e-13, true double)
#
# Usage:
#   Rscript bench_matmul_f64.R
#   Rscript bench_matmul_f64.R --sizes 512,1024,2048,3072
# ============================================================================

suppressMessages(library(ggmlR))

args   <- commandArgs(trailingOnly = TRUE)
arg_of <- function(flag, d) { i <- match(flag, args)
  if (is.na(i) || i == length(args)) d else args[i + 1L] }
SIZES <- as.integer(strsplit(arg_of("--sizes", "512,1024,2048"), ",")[[1]])

if (!(ggml_vulkan_available() && ggml_vulkan_device_count() > 0L))
  stop("no Vulkan GPU; this PoC needs a GPU with fp64 support.")

# GPU fp64 matmul via the public API (falls back to the CPU if the device has no
# shaderFloat64, but on a capable card this is the honest double kernel).
gpu_matmul_f64 <- function(A, B) ggml_matmul_f64(A, B, device = "gpu")

timed <- function(expr) { t0 <- proc.time()[["elapsed"]]
  v <- force(expr); list(v = v, s = proc.time()[["elapsed"]] - t0) }

cat(sprintf("BLAS: %s\n", extSoftVersion()["BLAS"]))
cat("FP64 matmul: GPU (matmul_f64.comp) vs CPU double (openBLAS) vs GPU f32\n\n")
cat(sprintf("%6s %11s %11s %11s %10s %12s\n",
            "n", "cpu_f64(s)", "gpu_f64(s)", "gpu_f32(s)", "f64 gain", "gpu_f64 err"))

for (n in SIZES) {
  set.seed(n)
  A <- matrix(rnorm(n * n), n, n)
  B <- matrix(rnorm(n * n), n, n)

  invisible(A %*% B)                       # warm BLAS
  tc <- timed(A %*% B)                     # cpu double

  invisible(gpu_matmul_f64(A, B))          # warm GPU fp64
  tg64 <- timed(gpu_matmul_f64(A, B))

  invisible(ggml_matmul(A, B, device = "gpu"))   # warm GPU f32
  tg32 <- timed(ggml_matmul(A, B, device = "gpu"))

  err <- max(abs(tg64$v - tc$v)) / max(abs(tc$v))
  cat(sprintf("%6d %11.3f %11.3f %11.3f %9.2fx %12.2e\n",
              n, tc$s, tg64$s, tg32$s, tc$s / tg64$s, err))
}

cat("\nRead-out:\n")
cat("  * gpu_f64 err ~1e-13 confirms honest double (vs ~1e-3 for the f32 path).\n")
cat("  * f64 gain > 1x means the GPU fp64 kernel beats multithreaded CPU double.\n")
cat("  * Compare gpu_f64 vs gpu_f32 to see the raw fp64:fp32 hardware penalty.\n")

ggml_vulkan_shutdown(hard = TRUE)
cat("\nDone.\n")
