# Tests for reshape and view operations

test_that("reshape_1d works", {
  ctx <- ggml_init(4 * 1024 * 1024)

  # Create 2D tensor and reshape to 1D
  t2d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 10)
  t1d <- ggml_reshape_1d(ctx, t2d, 100)

  expect_equal(ggml_nelements(t1d), 100)
  expect_equal(ggml_n_dims(t1d), 1)

  ggml_free(ctx)
})

test_that("reshape_2d works", {
  ctx <- ggml_init(4 * 1024 * 1024)

  # Create 1D tensor and reshape to 2D
  t1d <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  t2d <- ggml_reshape_2d(ctx, t1d, 10, 10)

  expect_equal(ggml_nelements(t2d), 100)
  shape <- ggml_tensor_shape(t2d)
  expect_equal(shape[1], 10)
  expect_equal(shape[2], 10)

  ggml_free(ctx)
})

test_that("reshape_3d works", {
  ctx <- ggml_init(4 * 1024 * 1024)

  t1d <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 120)
  t3d <- ggml_reshape_3d(ctx, t1d, 4, 5, 6)

  expect_equal(ggml_nelements(t3d), 120)
  shape <- ggml_tensor_shape(t3d)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 5)
  expect_equal(shape[3], 6)

  ggml_free(ctx)
})

test_that("reshape_4d works", {
  ctx <- ggml_init(4 * 1024 * 1024)

  t1d <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 120)
  t4d <- ggml_reshape_4d(ctx, t1d, 2, 3, 4, 5)

  expect_equal(ggml_nelements(t4d), 120)
  shape <- ggml_tensor_shape(t4d)
  expect_equal(shape[1], 2)
  expect_equal(shape[2], 3)
  expect_equal(shape[3], 4)
  expect_equal(shape[4], 5)

  ggml_free(ctx)
})

test_that("view_tensor creates view sharing data", {
  ctx <- ggml_init(4 * 1024 * 1024)

  original <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_f32(original, 1:10)

  view <- ggml_view_tensor(ctx, original)

  expect_type(view, "externalptr")
  expect_equal(ggml_nelements(view), 10)
  expect_equal(ggml_get_f32(view), 1:10)

  ggml_free(ctx)
})

test_that("permute changes tensor dimensions", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Create 2D tensor (3x4) and permute to (4x3)
  t2d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
  t_perm <- ggml_permute(ctx, t2d, 1, 0, 2, 3)

  shape <- ggml_tensor_shape(t_perm)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 3)

  ggml_free(ctx)
})

test_that("cont makes tensor contiguous", {
  ctx <- ggml_init(16 * 1024 * 1024)

  t2d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
  t_cont <- ggml_cont(ctx, t2d)

  expect_type(t_cont, "externalptr")
  expect_true(ggml_is_contiguous(t_cont))

  ggml_free(ctx)
})

test_that("ggml_view_1d creates view with offset", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_f32(a, 1:10)

  # View of 5 elements starting at offset 2*sizeof(float) = 8 bytes
  v <- ggml_view_1d(ctx, a, 5, 2 * 4)  # 2 elements * 4 bytes
  expect_equal(ggml_nelements(v), 5)
})

test_that("ggml_view_2d creates 2D view", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 12)
  ggml_set_f32(a, 1:12)

  # Create 3x4 view
  v <- ggml_view_2d(ctx, a, 3, 4, 3 * 4, 0)  # ne0=3, ne1=4, nb1=12, offset=0
  shape <- ggml_tensor_shape(v)
  expect_equal(shape[1], 3)
  expect_equal(shape[2], 4)
})

test_that("ggml_view_3d creates 3D view", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24)
  ggml_set_f32(a, 1:24)

  # Create 2x3x4 view
  v <- ggml_view_3d(ctx, a, 2, 3, 4, 2 * 4, 6 * 4, 0)
  shape <- ggml_tensor_shape(v)
  expect_equal(shape[1], 2)
  expect_equal(shape[2], 3)
  expect_equal(shape[3], 4)
})

test_that("ggml_view_4d creates 4D view", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 48)
  ggml_set_f32(a, 1:48)

  # Create 2x2x3x4 view
  v <- ggml_view_4d(ctx, a, 2, 2, 3, 4, 2 * 4, 4 * 4, 12 * 4, 0)
  shape <- ggml_tensor_shape(v)
  expect_equal(shape[1], 2)
  expect_equal(shape[2], 2)
  expect_equal(shape[3], 3)
  expect_equal(shape[4], 4)
})
