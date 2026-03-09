# Tests for Transformer Operations
# RoPE, Flash Attention, Causal Masking, Get Rows

test_that("ggml_set_i32 and ggml_get_i32 work correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create I32 tensor
  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5)

  # Set data
  ggml_set_i32(t, c(10L, 20L, 30L, 40L, 50L))

  # Get data back
  result <- ggml_get_i32(t)

  expect_equal(result, c(10L, 20L, 30L, 40L, 50L))
})

test_that("ggml_get_rows extracts embeddings by indices", {
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create embedding matrix: 4-dim embeddings, 10 tokens
  embeddings <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10)

  # Fill with sequential data for easy verification
  # Row i will have values [i*4, i*4+1, i*4+2, i*4+3]
  data <- as.numeric(0:39)
  ggml_set_f32(embeddings, data)

  # Create index tensor to select rows 0, 2, 5
  indices <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)
  ggml_set_i32(indices, c(0L, 2L, 5L))

  # Get rows
  result <- ggml_get_rows(ctx, embeddings, indices)

  # Build and compute
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  # Get result
  out <- ggml_get_f32(result)

  # Expected: rows 0, 2, 5 -> [0,1,2,3], [8,9,10,11], [20,21,22,23]
  expected <- c(0, 1, 2, 3, 8, 9, 10, 11, 20, 21, 22, 23)
  expect_equal(out, expected)

  # Check shape
  shape <- ggml_tensor_shape(result)
  expect_equal(shape[1], 4)  # embedding dim
  expect_equal(shape[2], 3)  # number of selected rows
})

test_that("ggml_diag_mask_inf creates causal mask", {
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create 4x4 matrix of ones
  m <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
  ggml_set_f32(m, rep(1.0, 16))

  # Apply causal mask (upper triangle -> -Inf)
  masked <- ggml_diag_mask_inf(ctx, m, 0)

  # Build and compute
  graph <- ggml_build_forward_expand(ctx, masked)
  ggml_graph_compute(ctx, graph)

  # Get result - GGML is row-major, R is column-major
  out <- ggml_get_f32(masked)
  # GGML stores [row0], [row1], ... so byrow=TRUE to match
  out_mat <- matrix(out, nrow = 4, ncol = 4, byrow = TRUE)

  # Lower triangle and diagonal should be 1
  # Upper triangle should be -Inf
  for (i in 1:4) {
    for (j in 1:4) {
      if (j <= i) {
        expect_equal(out_mat[i, j], 1.0)
      } else {
        expect_true(is.infinite(out_mat[i, j]) && out_mat[i, j] < 0)
      }
    }
  }
})

test_that("ggml_diag_mask_zero creates zero mask", {
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create 3x3 matrix of ones
  m <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3)
  ggml_set_f32(m, rep(1.0, 9))

  # Apply zero mask
  masked <- ggml_diag_mask_zero(ctx, m, 0)

  # Build and compute
  graph <- ggml_build_forward_expand(ctx, masked)
  ggml_graph_compute(ctx, graph)

  # Get result - GGML is row-major
  out <- ggml_get_f32(masked)
  out_mat <- matrix(out, nrow = 3, ncol = 3, byrow = TRUE)

  # Lower triangle and diagonal should be 1, upper should be 0
  for (i in 1:3) {
    for (j in 1:3) {
      if (j <= i) {
        expect_equal(out_mat[i, j], 1.0)
      } else {
        expect_equal(out_mat[i, j], 0.0)
      }
    }
  }
})

test_that("ggml_rope applies rotary position embedding", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create input tensor: [head_dim=8, n_head=2, seq_len=4, batch=1]
  head_dim <- 8
  n_head <- 2
  seq_len <- 4

  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)

  # Fill with ones
  n_elem <- head_dim * n_head * seq_len
  ggml_set_f32(q, rep(1.0, n_elem))

  # Create position tensor
  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len)
  ggml_set_i32(pos, 0:(seq_len - 1))

  # Apply RoPE
  q_rope <- ggml_rope(ctx, q, pos, head_dim, GGML_ROPE_TYPE_NORM)

  # Build and compute
  graph <- ggml_build_forward_expand(ctx, q_rope)
  ggml_graph_compute(ctx, graph)

  # Get result
  out <- ggml_get_f32(q_rope)

  # Basic sanity checks:
  # 1. Output should have same number of elements
  expect_equal(length(out), n_elem)

  # 2. Output should not be all the same (rotation applied)
  expect_true(sd(out) > 0)

  # 3. Shape should be preserved
  shape <- ggml_tensor_shape(q_rope)
  expect_equal(shape[1], head_dim)
  expect_equal(shape[2], n_head)
  expect_equal(shape[3], seq_len)
})

test_that("ggml_rope_ext works with frequency scaling", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  head_dim <- 8
  n_head <- 2
  seq_len <- 4

  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
  ggml_set_f32(q, rep(1.0, head_dim * n_head * seq_len))

  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len)
  ggml_set_i32(pos, 0:(seq_len - 1))

  # Apply extended RoPE with custom freq_base
  q_rope <- ggml_rope_ext(ctx, q, pos, NULL,
                          n_dims = head_dim,
                          mode = GGML_ROPE_TYPE_NORM,
                          n_ctx_orig = 2048,
                          freq_base = 10000.0,
                          freq_scale = 1.0,
                          ext_factor = 0.0,
                          attn_factor = 1.0,
                          beta_fast = 32.0,
                          beta_slow = 1.0)

  graph <- ggml_build_forward_expand(ctx, q_rope)
  ggml_graph_compute(ctx, graph)

  out <- ggml_get_f32(q_rope)

  expect_equal(length(out), head_dim * n_head * seq_len)
  expect_true(sd(out) > 0)
})

test_that("ggml_flash_attn_ext computes attention", {
  ctx <- ggml_init(128 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  head_dim <- 8
  n_head <- 2
  n_head_kv <- 2  # Same as n_head (no GQA)
  seq_len <- 4

  # Create Q, K, V tensors
  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)

  # Fill with random-ish data
  set.seed(42)
  ggml_set_f32(q, rnorm(head_dim * n_head * seq_len))
  ggml_set_f32(k, rnorm(head_dim * n_head_kv * seq_len))
  ggml_set_f32(v, rnorm(head_dim * n_head_kv * seq_len))

  # Scale = 1/sqrt(head_dim)
  scale <- 1.0 / sqrt(head_dim)

  # Compute attention
  out <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0, 0.0)

  graph <- ggml_build_forward_expand(ctx, out)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(out)

  # Output should have same number of elements as Q
  expect_equal(length(result), head_dim * n_head * seq_len)

  # Output should not be all zeros or NaN
  expect_true(all(is.finite(result)))
  expect_true(sd(result) > 0)

  # Check head_dim is preserved (first dimension)
  shape <- ggml_tensor_shape(out)
  expect_equal(shape[1], head_dim)
  # Total elements should match
  expect_equal(prod(shape), head_dim * n_head * seq_len)
})

test_that("RoPE type constants are defined", {
  expect_equal(GGML_ROPE_TYPE_NORM, 0L)
  expect_equal(GGML_ROPE_TYPE_NEOX, 2L)
  expect_equal(GGML_ROPE_TYPE_MROPE, 8L)
  expect_equal(GGML_ROPE_TYPE_VISION, 24L)
})
