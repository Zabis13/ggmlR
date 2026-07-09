#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Self-contained Flash-Attention check for P100 (no repo files needed).
# Runs ggml_flash_attn_ext on CPU and on Vulkan, compares the two.
#
#   Rscript scratch_fa_p100_check.R
#   GGML_SCHED_DEBUG=2 Rscript scratch_fa_p100_check.R 2>&1 | grep -iE "FLASH_ATTN|==="
#
# Verdict:
#   cor(GPU,CPU) > 0.999  -> scalar FA works on P100 (hardware is fine)
#   NaN / low cor / crash -> scalar FA path itself is broken on P100
# ---------------------------------------------------------------------------
suppressMessages(library(ggmlR))

cat("=== FA P100 check ===\n")
if (!ggml_vulkan_available()) {
  stop("Vulkan GPU not available — package cannot see the GPU on this server.")
}

# Realistic prefill shape: F16 K/V, head_dim=128, causal-style multi-head.
head_dim <- 128L
n_heads  <- 8L
seq_len  <- 64L
scale    <- 1.0 / sqrt(head_dim)

set.seed(42)
q_raw <- rnorm(head_dim * n_heads * seq_len)
k_raw <- rnorm(head_dim * n_heads * seq_len)
v_raw <- rnorm(head_dim * n_heads * seq_len)

run_fa <- function(use_gpu) {
  ctx <- ggml_init(64 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  # Q is F32, K/V are F16 (type 1) — matches real inference KV cache.
  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len, 1L)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_heads, seq_len, 1L)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_heads, seq_len, 1L)
  out <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0, 0.0)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(q, q_raw)   # F32
  ggml_backend_tensor_set_data(k, k_raw)   # auto-converted to F16
  ggml_backend_tensor_set_data(v, v_raw)   # auto-converted to F16

  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  res <- ggml_backend_tensor_get_data(out)
  ggml_backend_free(backend)
  ggml_free(ctx)
  res
}

cat("--- running CPU FA ---\n")
cpu_out <- run_fa(FALSE)
cat("--- running Vulkan FA ---\n")
gpu_out <- tryCatch(run_fa(TRUE),
                    error = function(e) { cat("GPU FA ERROR: ", conditionMessage(e), "\n"); NULL })

if (is.null(gpu_out)) {
  cat("=== VERDICT: Vulkan FA crashed/errored — scalar path broken on P100 ===\n")
  quit(status = 1)
}

n_na  <- sum(is.na(gpu_out))
n_inf <- sum(is.infinite(gpu_out))
cc    <- suppressWarnings(cor(cpu_out, gpu_out))
maxad <- max(abs(cpu_out - gpu_out))

cat(sprintf("GPU NaN=%d  Inf=%d  cor(GPU,CPU)=%.6f  max_abs_diff=%.5f\n",
            n_na, n_inf, cc, maxad))

if (n_na == 0 && n_inf == 0 && !is.na(cc) && cc > 0.999) {
  cat("=== VERDICT: FA WORKS on P100 (scalar path OK) — hardware is fine ===\n")
} else {
  cat("=== VERDICT: FA BROKEN on P100 — GPU output does not match CPU ===\n")
}
