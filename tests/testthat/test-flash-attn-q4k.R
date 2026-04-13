library(ggmlR)

test_that("Q4_K flash attention produces finite, non-zero output on CPU", {
  head_dim <- 256L
  n_heads  <- 2L
  seq_len  <- 16L
  scale    <- 1.0 / sqrt(head_dim)

  set.seed(7)
  q_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)
  k_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)
  v_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)

  k_q <- quantize_q4_K(k_raw, n_rows = n_heads * seq_len, n_per_row = head_dim)
  v_q <- quantize_q4_K(v_raw, n_rows = n_heads * seq_len, n_per_row = head_dim)

  ctx <- ggml_init(32 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  q   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len, 1L)
  k   <- ggml_new_tensor_4d(ctx, 12L, head_dim, n_heads, seq_len, 1L)
  v   <- ggml_new_tensor_4d(ctx, 12L, head_dim, n_heads, seq_len, 1L)
  out <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0, 0.0)

  backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(q, as.vector(q_raw))
  ggml_backend_tensor_set_data(k, k_q)
  ggml_backend_tensor_set_data(v, v_q)

  gf  <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  res <- ggml_backend_tensor_get_data(out)

  expect_false(any(is.na(res)))
  expect_false(any(is.infinite(res)))
  expect_gt(var(res), 0)
})

test_that("Q4_K flash attention GPU matches CPU (correlation > 0.999)", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  head_dim <- 256L
  n_heads  <- 4L
  seq_len  <- 32L
  scale    <- 1.0 / sqrt(head_dim)

  set.seed(42)
  q_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)
  k_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)
  v_raw <- matrix(rnorm(head_dim * n_heads * seq_len), nrow = head_dim)

  k_q <- quantize_q4_K(k_raw, n_rows = n_heads * seq_len, n_per_row = head_dim)
  v_q <- quantize_q4_K(v_raw, n_rows = n_heads * seq_len, n_per_row = head_dim)

  run_fa <- function(use_gpu) {
    ctx <- ggml_init(32 * 1024 * 1024)
    ggml_set_no_alloc(ctx, TRUE)
    q   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len, 1L)
    k   <- ggml_new_tensor_4d(ctx, 12L, head_dim, n_heads, seq_len, 1L)
    v   <- ggml_new_tensor_4d(ctx, 12L, head_dim, n_heads, seq_len, 1L)
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

  cpu_out <- run_fa(FALSE)
  gpu_out <- run_fa(TRUE)

  expect_false(any(is.na(gpu_out)))
  expect_false(any(is.infinite(gpu_out)))
  expect_gt(cor(cpu_out, gpu_out), 0.999)
  expect_lt(max(abs(cpu_out - gpu_out)), 0.1)
})
