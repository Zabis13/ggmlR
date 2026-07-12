#!/usr/bin/env Rscript
# ============================================================================
# GPU linear algebra — drop-in for R's %*% / crossprod / tcrossprod
# ============================================================================
# Accelerate an ordinary R matrix multiply on the Vulkan GPU without rewriting
# your code: plain matrices in, plain matrices out, with a transparent CPU
# fallback. This script
#   1. checks the GPU result against base R (a fast approximate multiply:
#      driver-dependent, typically ~1e-3 relative — not bit-for-bit),
#   2. shows the as_gpu_matrix() wrapper so `%*%` / crossprod / tcrossprod
#      dispatch to the GPU with no other code changes,
#   3. benchmarks GPU vs CPU across a few sizes so you can see where the GPU
#      starts to win (small multiplies stay on the CPU by design).
#
# Usage:
#   Rscript gpu_linalg.R
#   Rscript gpu_linalg.R --sizes 512,1024,2048,4096
#
# No extra packages required — everything is base R + ggmlR.
# ============================================================================

suppressMessages(library(ggmlR))

# ---- configuration ---------------------------------------------------------

args    <- commandArgs(trailingOnly = TRUE)
arg_of  <- function(flag, default) {
  i <- match(flag, args); if (is.na(i) || i == length(args)) default else args[i + 1L]
}
SIZES <- as.integer(strsplit(arg_of("--sizes", "256,512,1024,2048"), ",")[[1]])

have_gpu <- isTRUE(tryCatch(
  ggml_vulkan_available() && ggml_vulkan_device_count() > 0L,
  error = function(e) FALSE))
cat(sprintf("Vulkan GPU: %s\n\n", if (have_gpu) "yes" else "no (CPU fallback throughout)"))

# small timing helper
timed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  val <- force(expr)
  list(value = val, secs = proc.time()[["elapsed"]] - t0)
}

# ============================================================================
# 1. Correctness — the GPU result matches base R to f32 precision
# ============================================================================

cat("1. Correctness vs base R\n")
set.seed(1)
A <- matrix(rnorm(800 * 600), 800, 600)
B <- matrix(rnorm(600 * 400), 600, 400)

# force the GPU path (device = "gpu"); it still falls back to CPU if none.
C_gpu <- ggml_matmul(A, B, device = "gpu")            # f32 kernel by default
C_ref <- A %*% B                                      # base R (f64)
cat(sprintf("   matmul      max abs err  %.2e   (rel %.2e)\n",
            max(abs(C_gpu - C_ref)),
            max(abs(C_gpu - C_ref)) / max(abs(C_ref))))

G_gpu <- ggml_crossprod(A, device = "gpu")            # t(A) %*% A
cat(sprintf("   crossprod   max abs err  %.2e\n", max(abs(G_gpu - crossprod(A)))))

H_gpu <- ggml_tcrossprod(B, device = "gpu")           # B %*% t(B)
cat(sprintf("   tcrossprod  max abs err  %.2e\n", max(abs(H_gpu - tcrossprod(B)))))

# prec = "f16" trades accuracy for speed (larger error, ~2.7e-4 relative)
C_f16 <- ggml_matmul(A, B, device = "gpu", prec = "f16")
cat(sprintf("   matmul f16  max abs err  %.2e   (faster, lower precision)\n\n",
            max(abs(C_f16 - C_ref))))

# ============================================================================
# 2. Drop-in operators via as_gpu_matrix()
# ============================================================================
# Wrap one operand and the ordinary operators run on the GPU — nothing else in
# the surrounding code changes.

cat("2. Operator drop-in (as_gpu_matrix)\n")
Ag <- as_gpu_matrix(A)                # tag A as GPU-backed
print(Ag)                             # <ggml_matrix> 800 x 600  device=auto prec=f32

ok_mm  <- isTRUE(all.equal(Ag %*% B,      A %*% B,      check.attributes = FALSE))
ok_cp  <- isTRUE(all.equal(crossprod(Ag), crossprod(A), check.attributes = FALSE))
ok_tcp <- isTRUE(all.equal(tcrossprod(as_gpu_matrix(B)), tcrossprod(B),
                           check.attributes = FALSE, tolerance = 1e-3))
cat(sprintf("   Ag %%*%% B        == A %%*%% B        %s\n", if (ok_mm)  "TRUE" else "FALSE"))
cat(sprintf("   crossprod(Ag)  == crossprod(A)  %s\n", if (ok_cp)  "TRUE" else "FALSE"))
cat(sprintf("   tcrossprod(Bg) == tcrossprod(B) %s\n\n", if (ok_tcp) "TRUE" else "FALSE"))

# ============================================================================
# 3. Benchmark — GPU vs CPU across sizes
# ============================================================================
# device = "auto" keeps small multiplies on the CPU (the host<->VRAM transfer
# outweighs the compute); device = "gpu" / "cpu" force a path so both can be
# timed at every size. Timings are warm (a throwaway run first).

cat("3. Square matmul benchmark (n x n)\n")
cat(sprintf("   %6s %10s %10s %9s %11s\n",
            "n", "cpu (s)", "gpu (s)", "speedup", "max abs err"))

for (n in SIZES) {
  set.seed(n)
  X <- matrix(rnorm(n * n), n, n)
  Y <- matrix(rnorm(n * n), n, n)

  invisible(ggml_matmul(X, Y, device = "cpu"))          # warm
  tc <- timed(ggml_matmul(X, Y, device = "cpu"))

  if (have_gpu) {
    invisible(ggml_matmul(X, Y, device = "gpu"))        # warm
    tg  <- timed(ggml_matmul(X, Y, device = "gpu"))
    err <- max(abs(tg$value - tc$value))
    cat(sprintf("   %6d %10.3f %10.3f %8.2fx %11.2e\n",
                n, tc$secs, tg$secs, tc$secs / tg$secs, err))
  } else {
    cat(sprintf("   %6d %10.3f %10s %9s %11s\n",
                n, tc$secs, "-", "-", "-"))
  }
}

cat("\nNotes:\n")
cat("  * device = \"auto\" (the default) picks the GPU only when it is present and\n")
cat("    the multiply is large enough to amortise the transfer; otherwise CPU.\n")
cat("  * The GPU path is a fast approximate multiply. prec = \"f32\" (default)\n")
cat("    requests f32 accumulation; the actual error is driver-dependent (some,\n")
cat("    e.g. RADV/Mesa, accumulate in f16 regardless, ~1e-3). prec = \"f16\" only\n")
cat("    lowers precision further, for speed.\n")

if (have_gpu) ggml_vulkan_shutdown(hard = TRUE)         # avoid exit-time loader race
cat("\nDone.\n")
