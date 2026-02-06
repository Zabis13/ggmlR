# Tests for extended tensor functions (3D, 4D, dup, info)

test_that("3D tensor creation works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  tensor <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 20, 30)

  expect_type(tensor, "externalptr")
  expect_equal(ggml_nelements(tensor), 10 * 20 * 30)
  expect_equal(ggml_n_dims(tensor), 3)

  shape <- ggml_tensor_shape(tensor)
  expect_equal(shape[1], 10)
  expect_equal(shape[2], 20)
  expect_equal(shape[3], 30)
  expect_equal(shape[4], 1)

  ggml_free(ctx)
})

test_that("4D tensor creation works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  tensor <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5)

  expect_type(tensor, "externalptr")
  expect_equal(ggml_nelements(tensor), 2 * 3 * 4 * 5)
  expect_equal(ggml_n_dims(tensor), 4)

  shape <- ggml_tensor_shape(tensor)
  expect_equal(shape[1], 2)
  expect_equal(shape[2], 3)
  expect_equal(shape[3], 4)
  expect_equal(shape[4], 5)

  ggml_free(ctx)
})

test_that("dup_tensor creates copy with same shape", {
  ctx <- ggml_init(4 * 1024 * 1024)
  original <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
  duplicate <- ggml_dup_tensor(ctx, original)

  expect_type(duplicate, "externalptr")
  expect_equal(ggml_nelements(duplicate), ggml_nelements(original))
  expect_equal(ggml_tensor_shape(duplicate), ggml_tensor_shape(original))
  expect_equal(ggml_tensor_type(duplicate), ggml_tensor_type(original))

  ggml_free(ctx)
})

test_that("tensor info functions work correctly", {
  ctx <- ggml_init(4 * 1024 * 1024)
  tensor <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)

  # n_dims
  expect_equal(ggml_n_dims(tensor), 2)

  # is_contiguous (new tensor should be contiguous)
  expect_true(ggml_is_contiguous(tensor))

  # is_transposed (new tensor should not be transposed)
  expect_false(ggml_is_transposed(tensor))

  # is_permuted (new tensor should not be permuted)
  expect_false(ggml_is_permuted(tensor))

  # tensor_type
  expect_equal(ggml_tensor_type(tensor), GGML_TYPE_F32)

  ggml_free(ctx)
})

test_that("tensor_shape returns correct dimensions", {
  ctx <- ggml_init(4 * 1024 * 1024)

  # 1D
  t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  shape1 <- ggml_tensor_shape(t1)
  expect_equal(shape1[1], 100)
  expect_equal(shape1[2], 1)

  # 2D
  t2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
  shape2 <- ggml_tensor_shape(t2)
  expect_equal(shape2[1], 10)
  expect_equal(shape2[2], 20)

  ggml_free(ctx)
})
