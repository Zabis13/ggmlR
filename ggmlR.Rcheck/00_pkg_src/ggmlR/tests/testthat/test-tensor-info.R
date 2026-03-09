# Tests for Tensor Information Functions

# ============================================================================
# Type Size
# ============================================================================

test_that("ggml_type_size returns correct sizes", {
  # F32 = 4 bytes
  expect_equal(ggml_type_size(GGML_TYPE_F32), 4)

  # F16 = 2 bytes
  expect_equal(ggml_type_size(GGML_TYPE_F16), 2)

  # I32 = 4 bytes
  expect_equal(ggml_type_size(GGML_TYPE_I32), 4)
})

# ============================================================================
# Element Size
# ============================================================================

test_that("ggml_element_size returns element size for F32 tensor", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_equal(ggml_element_size(t), 4)
})

test_that("ggml_element_size returns element size for I32 tensor", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10)
  expect_equal(ggml_element_size(t), 4)
})

# ============================================================================
# Number of Rows
# ============================================================================

test_that("ggml_nrows returns correct number of rows", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # 1D tensor - 1 row
  t1d <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_equal(ggml_nrows(t1d), 1)

  # 2D tensor - ne1 rows
  t2d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 3)
  expect_equal(ggml_nrows(t2d), 3)

  # 3D tensor - ne1 * ne2 rows
  t3d <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 2)
  expect_equal(ggml_nrows(t3d), 6)  # 3 * 2
})

# ============================================================================
# Same Shape Check
# ============================================================================

test_that("ggml_are_same_shape detects same shapes", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)

  expect_true(ggml_are_same_shape(a, b))
})

test_that("ggml_are_same_shape detects different shapes", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 4)

  expect_false(ggml_are_same_shape(a, b))
})

test_that("ggml_are_same_shape with different dimensions", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 20)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)

  expect_false(ggml_are_same_shape(a, b))
})

# ============================================================================
# Tensor Name
# ============================================================================

test_that("ggml_set_name and ggml_get_name work", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_name(t, "my_tensor")

  name <- ggml_get_name(t)
  expect_equal(name, "my_tensor")
})

test_that("ggml_set_name with empty string returns NULL or empty", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_name(t, "")

  name <- ggml_get_name(t)
  # Empty name may return NULL or ""
  expect_true(is.null(name) || name == "")
})

# ============================================================================
# Tensor Shape
# ============================================================================

test_that("ggml_tensor_shape returns correct shape for 1D", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  shape <- ggml_tensor_shape(t)

  expect_length(shape, 4)
  expect_equal(shape[1], 10)
  expect_equal(shape[2], 1)
  expect_equal(shape[3], 1)
  expect_equal(shape[4], 1)
})

test_that("ggml_tensor_shape returns correct shape for 2D", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 7)
  shape <- ggml_tensor_shape(t)

  expect_equal(shape[1], 5)
  expect_equal(shape[2], 7)
})

test_that("ggml_tensor_shape returns correct shape for 3D", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 5)
  shape <- ggml_tensor_shape(t)

  expect_equal(shape[1], 3)
  expect_equal(shape[2], 4)
  expect_equal(shape[3], 5)
})

test_that("ggml_tensor_shape returns correct shape for 4D", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5)
  shape <- ggml_tensor_shape(t)

  expect_equal(shape[1], 2)
  expect_equal(shape[2], 3)
  expect_equal(shape[3], 4)
  expect_equal(shape[4], 5)
})

# ============================================================================
# Tensor Type
# ============================================================================

test_that("ggml_tensor_type returns correct type", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t_f32 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_equal(ggml_tensor_type(t_f32), GGML_TYPE_F32)

  t_i32 <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10)
  expect_equal(ggml_tensor_type(t_i32), GGML_TYPE_I32)
})

# ============================================================================
# Number of Elements and Bytes
# ============================================================================

test_that("ggml_nelements returns correct count", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t1d <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_equal(ggml_nelements(t1d), 10)

  t2d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  expect_equal(ggml_nelements(t2d), 20)

  t3d <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 4)
  expect_equal(ggml_nelements(t3d), 24)

  t4d <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5)
  expect_equal(ggml_nelements(t4d), 120)
})

test_that("ggml_nbytes returns correct byte count", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_equal(ggml_nbytes(t), 40)  # 10 * 4 bytes
})

# ============================================================================
# Number of Dimensions
# ============================================================================

test_that("ggml_n_dims returns correct dimension count", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t1d <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  expect_equal(ggml_n_dims(t1d), 1)

  t2d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  expect_equal(ggml_n_dims(t2d), 2)

  t3d <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 4)
  expect_equal(ggml_n_dims(t3d), 3)

  t4d <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5)
  expect_equal(ggml_n_dims(t4d), 4)
})

# ============================================================================
# Contiguous Check
# ============================================================================

test_that("ggml_is_contiguous returns true for new tensors", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  expect_true(ggml_is_contiguous(t))
})

# ============================================================================
# Transposed Check
# ============================================================================

test_that("ggml_is_transposed returns false for new tensors", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  expect_false(ggml_is_transposed(t))
})

# ============================================================================
# Permuted Check
# ============================================================================

test_that("ggml_is_permuted returns false for new tensors", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  expect_false(ggml_is_permuted(t))
})

# ============================================================================
# Generic Tensor Creation
# ============================================================================

test_that("ggml_new_tensor creates tensor with arbitrary dims", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # 3D tensor
  t <- ggml_new_tensor(ctx, GGML_TYPE_F32, n_dims = 3, ne = c(4, 5, 6))
  shape <- ggml_tensor_shape(t)

  expect_equal(shape[1], 4)
  expect_equal(shape[2], 5)
  expect_equal(shape[3], 6)
  expect_equal(ggml_nelements(t), 120)
})

# ============================================================================
# Duplicate Tensor
# ============================================================================

test_that("ggml_dup_tensor creates tensor with same shape", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  original <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  duplicate <- ggml_dup_tensor(ctx, original)

  expect_equal(ggml_nelements(duplicate), ggml_nelements(original))
  expect_equal(ggml_tensor_shape(duplicate), ggml_tensor_shape(original))
  expect_equal(ggml_tensor_type(duplicate), ggml_tensor_type(original))
})
