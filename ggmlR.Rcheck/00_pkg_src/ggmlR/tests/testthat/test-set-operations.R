# Tests for Set Operations (ggml_set, ggml_set_1d, ggml_set_2d)

# ============================================================================
# Set 1D
# ============================================================================

test_that("ggml_set_1d copies tensor at offset", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Destination tensor
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_f32(a, rep(0, 10))

  # Source tensor
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(b, c(1, 2, 3))

  # Set b into a at offset 2*4 = 8 bytes (position 2)
  result <- ggml_set_1d(ctx, a, b, offset = 2 * 4)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # Expected: [0, 0, 1, 2, 3, 0, 0, 0, 0, 0]
  expected <- c(0, 0, 1, 2, 3, 0, 0, 0, 0, 0)
  expect_equal(output, expected, tolerance = 1e-5)
})

test_that("ggml_set_1d at beginning", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, rep(0, 5))

  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(b, c(7, 8))

  # Set at offset 0
  result <- ggml_set_1d(ctx, a, b, offset = 0)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(7, 8, 0, 0, 0), tolerance = 1e-5)
})

# ============================================================================
# Set 2D
# ============================================================================

test_that("ggml_set_2d copies tensor with stride", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Destination: 4x3 matrix
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  ggml_set_f32(a, rep(0, 12))

  # Source: 2x2 matrix
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2)
  ggml_set_f32(b, c(1, 2, 3, 4))

  # Set b into a with proper stride
  # nb1 = stride for dimension 1 = 4 * sizeof(float) = 16 bytes
  result <- ggml_set_2d(ctx, a, b, nb1 = 4 * 4, offset = 0)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 12)
})

# ============================================================================
# Set (General)
# ============================================================================

test_that("ggml_set copies tensor with full control", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Destination: 4x4 matrix
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
  ggml_set_f32(a, rep(0, 16))

  # Source: 2x2 matrix
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2)
  ggml_set_f32(b, c(1, 2, 3, 4))

  # Set with explicit strides
  # nb1 = stride for dim 1 = 4 * 4 = 16
  # nb2 = stride for dim 2 (not used for 2D)
  # nb3 = stride for dim 3 (not used for 2D)
  result <- ggml_set(ctx, a, b, nb1 = 4 * 4, nb2 = 16 * 4, nb3 = 64 * 4, offset = 0)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 16)
  expect_false(any(is.na(output)))
})

# ============================================================================
# Set Zero
# ============================================================================

test_that("ggml_set_zero zeros all elements", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_f32(t, as.numeric(1:10))

  # Verify non-zero
  expect_true(any(ggml_get_f32(t) != 0))

  # Set to zero
  ggml_set_zero(t)

  output <- ggml_get_f32(t)
  expect_equal(output, rep(0, 10), tolerance = 1e-10)
})

test_that("ggml_set_zero on 2D tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  ggml_set_f32(t, rnorm(20))

  ggml_set_zero(t)

  output <- ggml_get_f32(t)
  expect_equal(output, rep(0, 20), tolerance = 1e-10)
})

# ============================================================================
# Set/Get F32 Data
# ============================================================================

test_that("ggml_set_f32 and ggml_get_f32 roundtrip", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  data <- c(1.1, 2.2, 3.3, 4.4, 5.5)

  ggml_set_f32(t, data)
  output <- ggml_get_f32(t)

  expect_equal(output, data, tolerance = 1e-6)
})

test_that("ggml_set_f32 handles negative values", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  data <- c(-10.5, -0.001, 0.001, 10.5)

  ggml_set_f32(t, data)
  output <- ggml_get_f32(t)

  expect_equal(output, data, tolerance = 1e-6)
})

# ============================================================================
# Set/Get I32 Data
# ============================================================================

test_that("ggml_set_i32 and ggml_get_i32 roundtrip", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5)
  data <- c(1L, 2L, 3L, 4L, 5L)

  ggml_set_i32(t, data)
  output <- ggml_get_i32(t)

  expect_equal(output, data)
})

test_that("ggml_set_i32 handles negative values", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4)
  data <- c(-100L, -1L, 0L, 100L)

  ggml_set_i32(t, data)
  output <- ggml_get_i32(t)

  expect_equal(output, data)
})

# ============================================================================
# Scalar Tensor Creation
# ============================================================================

test_that("ggml_new_i32 creates scalar integer tensor", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  scalar <- ggml_new_i32(ctx, 42L)

  expect_equal(ggml_nelements(scalar), 1)
  expect_equal(ggml_tensor_type(scalar), GGML_TYPE_I32)
  expect_equal(ggml_get_i32(scalar), 42L)
})

test_that("ggml_new_f32 creates scalar float tensor", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  scalar <- ggml_new_f32(ctx, 3.14159)

  expect_equal(ggml_nelements(scalar), 1)
  expect_equal(ggml_tensor_type(scalar), GGML_TYPE_F32)
  expect_equal(ggml_get_f32(scalar), 3.14159, tolerance = 1e-5)
})

test_that("ggml_new_i32 with negative value", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  scalar <- ggml_new_i32(ctx, -999L)

  expect_equal(ggml_get_i32(scalar), -999L)
})

test_that("ggml_new_f32 with zero", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  scalar <- ggml_new_f32(ctx, 0)

  expect_equal(ggml_get_f32(scalar), 0, tolerance = 1e-10)
})
