# Tests for newly added functions

test_that("ggml_new_i32 creates scalar i32 tensor", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  scalar <- ggml_new_i32(ctx, 42L)
  expect_false(is.null(scalar))

  # Check shape is scalar (1 element)
  shape <- ggml_tensor_shape(scalar)
  expect_equal(prod(shape), 1)
})

test_that("ggml_new_f32 creates scalar f32 tensor", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  scalar <- ggml_new_f32(ctx, 3.14)
  expect_false(is.null(scalar))

  # Check shape is scalar (1 element)
  shape <- ggml_tensor_shape(scalar)
  expect_equal(prod(shape), 1)
})

test_that("ggml_view_1d creates 1D view with offset", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create source tensor
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  expect_false(is.null(a))

  # Create view of 10 elements starting at offset 40 bytes (10 floats * 4 bytes)
  v <- ggml_view_1d(ctx, a, 10, 40)
  expect_false(is.null(v))

  shape <- ggml_tensor_shape(v)
  expect_equal(shape[1], 10)
})

test_that("ggml_view_2d creates 2D view with offset", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create source tensor
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  expect_false(is.null(a))

  # Create 2D view: 5x4 starting at offset 0
  # nb1 = 5 * 4 = 20 bytes (stride for dimension 1)
  v <- ggml_view_2d(ctx, a, 5, 4, 20, 0)
  expect_false(is.null(v))

  shape <- ggml_tensor_shape(v)
  expect_equal(shape[1], 5)
  expect_equal(shape[2], 4)
})

test_that("ggml_view_3d creates 3D view with offset", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 120)
  expect_false(is.null(a))

  # Create 3D view: 4x5x6
  # nb1 = 4 * 4 = 16, nb2 = 4 * 5 * 4 = 80
  v <- ggml_view_3d(ctx, a, 4, 5, 6, 16, 80, 0)
  expect_false(is.null(v))

  shape <- ggml_tensor_shape(v)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 5)
  expect_equal(shape[3], 6)
})

test_that("ggml_view_4d creates 4D view with offset", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 240)
  expect_false(is.null(a))

  # Create 4D view: 2x3x4x5
  # nb1=8, nb2=24, nb3=96
  v <- ggml_view_4d(ctx, a, 2, 3, 4, 5, 8, 24, 96, 0)
  expect_false(is.null(v))

  shape <- ggml_tensor_shape(v)
  expect_equal(shape[1], 2)
  expect_equal(shape[2], 3)
  expect_equal(shape[3], 4)
  expect_equal(shape[4], 5)
})

test_that("ggml_cpy creates copy operation", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_false(is.null(a))
  expect_false(is.null(b))

  result <- ggml_cpy(ctx, a, b)
  expect_false(is.null(result))
})

test_that("ggml_set_1d creates set operation", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_false(is.null(a))
  expect_false(is.null(b))

  # Set b into a at offset 0
  result <- ggml_set_1d(ctx, a, b, 0)
  expect_false(is.null(result))
})

test_that("ggml_set_2d creates 2D set operation", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 10)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 5)
  expect_false(is.null(a))
  expect_false(is.null(b))

  # nb1 = 5 * 4 = 20
  result <- ggml_set_2d(ctx, a, b, 20, 0)
  expect_false(is.null(result))
})

test_that("ggml_out_prod creates outer product operation", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  expect_false(is.null(a))
  expect_false(is.null(b))

  result <- ggml_out_prod(ctx, a, b)
  expect_false(is.null(result))

  # Result should be 3x4 matrix
  shape <- ggml_tensor_shape(result)
  expect_equal(shape[1], 3)
  expect_equal(shape[2], 4)
})

test_that("ggml_diag creates diagonal matrix", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  expect_false(is.null(a))

  result <- ggml_diag(ctx, a)
  expect_false(is.null(result))
})

test_that("ggml_concat concatenates tensors", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
  expect_false(is.null(a))
  expect_false(is.null(b))

  # Concatenate along dimension 1
  result <- ggml_concat(ctx, a, b, 1)
  expect_false(is.null(result))

  # Result should be 4x5
  shape <- ggml_tensor_shape(result)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 5)
})

test_that("ggml_mul_mat_id creates MoE matmul operation", {
  ctx <- ggml_init(4 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create stacked expert weights: 8x16x4 (4 experts, each 8x16)
  experts <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4)
  # Input tensor
  input <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2)
  # Expert indices
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2)

  expect_false(is.null(experts))
  expect_false(is.null(input))
  expect_false(is.null(ids))

  result <- ggml_mul_mat_id(ctx, experts, input, ids)
  expect_false(is.null(result))
})

test_that("ggml_silu_back creates SiLU backward operation", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_false(is.null(a))
  expect_false(is.null(b))

  result <- ggml_silu_back(ctx, a, b)
  expect_false(is.null(result))
})

test_that("ggml_get_rows_back creates get_rows backward operation", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # Gradient tensor
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3)
  # Index tensor
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)
  # Reference tensor for output shape
  c <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 10)

  expect_false(is.null(a))
  expect_false(is.null(b))
  expect_false(is.null(c))

  result <- ggml_get_rows_back(ctx, a, b, c)
  expect_false(is.null(result))
})

test_that("ggml_soft_max_ext_back creates softmax backward operation", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 5)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 5)
  expect_false(is.null(a))
  expect_false(is.null(b))

  result <- ggml_soft_max_ext_back(ctx, a, b, scale = 1.0, max_bias = 0.0)
  expect_false(is.null(result))
})

test_that("ggml_rope_ext_back creates RoPE backward operation", {
  ctx <- ggml_init(2 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Gradient tensor: [head_dim, n_heads, seq_len, batch]
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 16, 1)
  # Position tensor
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16)

  expect_false(is.null(a))
  expect_false(is.null(b))

  result <- ggml_rope_ext_back(ctx, a, b, c = NULL,
                               n_dims = 64, mode = 0L, n_ctx_orig = 2048,
                               freq_base = 10000.0, freq_scale = 1.0,
                               ext_factor = 0.0, attn_factor = 1.0,
                               beta_fast = 32.0, beta_slow = 1.0)
  expect_false(is.null(result))
})

# NOTE: ggml_flash_attn_back is not implemented in current GGML version
# The function exists in header but contains GGML_ABORT("TODO: adapt to ggml_flash_attn_ext() changes")
# test_that("ggml_flash_attn_back creates flash attention backward operation", {
#   skip("ggml_flash_attn_back not implemented in current GGML version")
# })
