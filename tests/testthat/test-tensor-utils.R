# Tests for tensor utility functions

test_that("ggml_tensor_nb returns stride info", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  nb <- ggml_tensor_nb(t)
  expect_type(nb, "double")
  expect_length(nb, 5)  # GGML_MAX_DIMS=5 strides
  expect_equal(nb[1], 4)  # F32 = 4 bytes per element
  expect_equal(nb[2], 16) # 4 elements * 4 bytes
})

test_that("ggml_tensor_num counts tensors in context", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  expect_equal(ggml_tensor_num(ctx), 0L)

  ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  expect_equal(ggml_tensor_num(ctx), 1L)

  ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)
  expect_equal(ggml_tensor_num(ctx), 2L)
})

test_that("ggml_tensor_copy copies data between tensors", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  src <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  dst <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(src, c(1, 2, 3, 4))
  ggml_set_f32(dst, c(0, 0, 0, 0))

  ggml_tensor_copy(dst, src)
  result <- ggml_get_f32(dst)
  expect_equal(result, c(1, 2, 3, 4), tolerance = 1e-6)
})

test_that("ggml_tensor_set_f32_scalar sets all elements", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_tensor_set_f32_scalar(t, 42.0)
  result <- ggml_get_f32(t)
  expect_equal(result, rep(42.0, 4), tolerance = 1e-6)
})

test_that("ggml_get_f32_nd and ggml_set_f32_nd work for 2D", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
  ggml_set_f32(t, c(1, 2, 3, 4, 5, 6))

  # Read element [0,0] (0-based)
  expect_equal(ggml_get_f32_nd(t, 0, 0), 1.0, tolerance = 1e-6)
  # Read element [2,1] (col-major: index 5)
  expect_equal(ggml_get_f32_nd(t, 2, 1), 6.0, tolerance = 1e-6)

  # Write element [1,0]
  ggml_set_f32_nd(t, 1, 0, value = 99.0)
  expect_equal(ggml_get_f32_nd(t, 1, 0), 99.0, tolerance = 1e-6)
})

test_that("ggml_get_i32_nd and ggml_set_i32_nd work", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4)
  ggml_set_i32(t, c(10L, 20L, 30L, 40L))

  expect_equal(ggml_get_i32_nd(t, 0), 10L)
  expect_equal(ggml_get_i32_nd(t, 3), 40L)

  ggml_set_i32_nd(t, 1, value = 99L)
  expect_equal(ggml_get_i32_nd(t, 1), 99L)
})

test_that("ggml_get_first_tensor and ggml_get_next_tensor iterate", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_name(t1, "first")
  t2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)
  ggml_set_name(t2, "second")

  first <- ggml_get_first_tensor(ctx)
  expect_type(first, "externalptr")
  expect_equal(ggml_get_name(first), "first")

  second <- ggml_get_next_tensor(ctx, first)
  expect_type(second, "externalptr")
  expect_equal(ggml_get_name(second), "second")
})

test_that("ggml_backend_tensor_get_f32_first returns first element", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(t, c(3.14, 2.0, 1.0, 0.5))

  val <- ggml_backend_tensor_get_f32_first(t)
  expect_equal(val, 3.14, tolerance = 1e-5)
})
