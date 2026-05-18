library(ggmlR)

# Covers the Q4_0 and Q8_0 dequantize4 branches of the flash attention
# Vulkan shaders (flash_attn_base.glsl). Mirrors test-flash-attn-q4k.R test 2:
# GPU vs CPU numerical agreement. These branches are exercised by the cm1 and
# scalar FA paths after the G1 FLOAT_TYPEV4/FLOAT_TYPE type migration.

run_fa_quant <- function(use_gpu, ggml_type, quantize_fn, head_dim, n_heads,
                         seq_len, q_raw, k_raw, v_raw, scale) {
  k_q <- quantize_fn(k_raw, n_rows = n_heads * seq_len, n_per_row = head_dim)
  v_q <- quantize_fn(v_raw, n_rows = n_heads * seq_len, n_per_row = head_dim)

  ctx <- ggml_init(32 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len, 1L)
  k <- ggml_new_tensor_4d(ctx, ggml_type, head_dim, n_heads, seq_len, 1L)
  v <- ggml_new_tensor_4d(ctx, ggml_type, head_dim, n_heads, seq_len, 1L)
  out <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0, 0.0)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(q, as.vector(q_raw))
  ggml_backend_tensor_set_data(k, k_q)
  ggml_backend_tensor_set_data(v, v_q)

  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

for (case in list(
  list(name = "Q4_0",   type = GGML_TYPE_Q4_0, fn = quantize_q4_0,   seed = 11),
  list(name = "Q4_1",   type = GGML_TYPE_Q4_1, fn = quantize_q4_1,   seed = 13),
  list(name = "Q5_0",   type = 6L,             fn = quantize_q5_0,   seed = 17),
  list(name = "Q5_1",   type = 7L,             fn = quantize_q5_1,   seed = 19),
  list(name = "Q8_0",   type = GGML_TYPE_Q8_0, fn = quantize_q8_0,   seed = 23),
  list(name = "IQ4_NL", type = 20L,            fn = quantize_iq4_nl, seed = 29)
)) {
  local({
    cc <- case
    test_that(sprintf("%s flash attention GPU matches CPU (correlation > 0.999)", cc$name), {
      skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

      head_dim <- 256L
      n_heads  <- 4L
      seq_len  <- 32L
      scale    <- 1.0 / sqrt(head_dim)

      set.seed(cc$seed)
      q_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)
      k_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)
      v_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)

      cpu_out <- run_fa_quant(FALSE, cc$type, cc$fn, head_dim, n_heads,
                              seq_len, q_raw, k_raw, v_raw, scale)
      gpu_out <- run_fa_quant(TRUE, cc$type, cc$fn, head_dim, n_heads,
                              seq_len, q_raw, k_raw, v_raw, scale)

      expect_false(any(is.na(gpu_out)))
      expect_false(any(is.infinite(gpu_out)))
      expect_gt(cor(cpu_out, gpu_out), 0.999)
      expect_lt(max(abs(cpu_out - gpu_out)), 0.1)
    })
  })
}
