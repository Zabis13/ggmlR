library(ggmlR)

# Regression test for the Flux-noise bug (2026-05-17): Q4_K matmul on the
# Vulkan backend produced output uncorrelated with the CPU reference
# (cor ~ 0), while Q6_K / Q4_0 / Q8_0 were correct. flux1-dev-Q4_K_S.gguf
# is a Q4_K diffusion model, so this bug made all Flux output pure noise
# on Vulkan while CPU rendered correctly.
#
# These tests pin Q4_K matmul Vulkan-vs-CPU agreement across the shape
# regimes Flux exercises, plus control quants that must stay correct.

# A: weights [K, M] quantized; B: activations [K, N] f32; out = A * B -> [M, N]
run_quant_mm <- function(use_gpu, qtype, qfn, M, K, N) {
  set.seed(42)
  a_raw <- matrix(rnorm(K * M, sd = 0.1), nrow = M, ncol = K)
  b_raw <- matrix(rnorm(K * N, sd = 0.1), nrow = K, ncol = N)
  a_q   <- qfn(as.vector(t(a_raw)), n_rows = M, n_per_row = K)

  ctx <- ggml_init(1024L * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  a   <- ggml_new_tensor_2d(ctx, qtype, K, M)
  b   <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)
  out <- ggml_mul_mat(ctx, a, b)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, a_q)
  ggml_backend_tensor_set_data(b, as.vector(b_raw))

  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

expect_backends_agree <- function(qname, qtype, qfn, M, K, N) {
  cpu <- run_quant_mm(FALSE, qtype, qfn, M, K, N)
  gpu <- run_quant_mm(TRUE,  qtype, qfn, M, K, N)
  cc  <- suppressWarnings(cor(cpu, gpu))
  md  <- max(abs(cpu - gpu))
  info <- sprintf("%s M=%d K=%d N=%d cor=%.5f max|diff|=%.4f",
                  qname, M, K, N, cc, md)
  expect_gt(cc, 0.999, label = info)
  expect_lt(md, 0.1,   label = info)
}

test_that("Q4_K matmul Vulkan == CPU (small N, mul_mat_vec path)", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")
  expect_backends_agree("Q4_K", GGML_TYPE_Q4_K, quantize_q4_K,
                         M = 512L, K = 512L, N = 64L)
})

test_that("Q4_K matmul Vulkan == CPU (large N, MMQ path, Flux-DiT shape)", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")
  # N >= 512 triggers the MMQ / subgroup-shuffle path used by Flux DiT.
  expect_backends_agree("Q4_K", GGML_TYPE_Q4_K, quantize_q4_K,
                         M = 1024L, K = 1024L, N = 2560L)
})

test_that("control quants Vulkan == CPU (Q6_K / Q4_0 / Q8_0 must stay correct)", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")
  expect_backends_agree("Q6_K", GGML_TYPE_Q6_K, quantize_q6_K,
                         M = 512L, K = 512L, N = 64L)
  expect_backends_agree("Q4_0", GGML_TYPE_Q4_0, quantize_q4_0,
                         M = 512L, K = 512L, N = 64L)
  expect_backends_agree("Q8_0", GGML_TYPE_Q8_0, quantize_q8_0,
                         M = 512L, K = 512L, N = 64L)
})
