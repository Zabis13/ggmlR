# Tests for ggml_rope_multi (multi-dimensional / M-RoPE, new in 0.11.0).
# Exported with only constant checks; the operation itself had no functional
# test coverage. M-RoPE is used by multimodal models (Qwen2-VL etc.).

test_that("M-RoPE / VISION rope-type constants are defined", {
  expect_equal(GGML_ROPE_TYPE_MROPE, 8L)
  expect_equal(GGML_ROPE_TYPE_VISION, 24L)
})

test_that("ggml_rope_multi computes M-RoPE and changes the tensor", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  head_dim <- 16          # n_dims; M-RoPE splits across 4 sections
  n_head   <- 2
  seq_len  <- 4

  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
  ggml_set_f32(a, rep(1.0, head_dim * n_head * seq_len))

  # M-RoPE position tensor: 4 position components per token (t, h, w, extra)
  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len * 4)
  ggml_set_i32(pos, rep(0:(seq_len - 1), times = 4))

  # sections sum to n_dims/2 (= 8): split rotary dims across the 4 axes
  sec <- c(2L, 2L, 2L, 2L)

  r <- ggml_rope_multi(ctx, a, pos, NULL,
                       n_dims = head_dim,
                       sections = sec,
                       mode = GGML_ROPE_TYPE_MROPE,
                       n_ctx_orig = 2048,
                       freq_base = 10000.0,
                       freq_scale = 1.0,
                       ext_factor = 0.0,
                       attn_factor = 1.0,
                       beta_fast = 32.0,
                       beta_slow = 1.0)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  out <- ggml_get_f32(r)
  expect_equal(length(out), head_dim * n_head * seq_len)
  expect_true(all(is.finite(out)))
  # rotation must actually alter the constant input
  expect_true(sd(out) > 0)
})

test_that("ggml_rope_multi preserves tensor shape", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  head_dim <- 8
  seq_len  <- 3
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, 1, seq_len, 1)
  ggml_set_f32(a, as.numeric(seq_len_vals <- 1:(head_dim * seq_len)))

  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len * 4)
  ggml_set_i32(pos, rep(0:(seq_len - 1), times = 4))

  r <- ggml_rope_multi(ctx, a, pos, NULL,
                       n_dims = head_dim,
                       sections = c(1L, 1L, 1L, 1L),
                       mode = GGML_ROPE_TYPE_MROPE)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  shape <- ggml_tensor_shape(r)
  expect_equal(shape[1], head_dim)
  expect_equal(shape[3], seq_len)
})
