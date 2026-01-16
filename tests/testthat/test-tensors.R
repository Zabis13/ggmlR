test_that("1D tensor creation works", {
  ctx <- ggml_init(1024 * 1024)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  
  expect_type(tensor, "externalptr")
  expect_equal(ggml_nelements(tensor), 10)
  expect_equal(ggml_nbytes(tensor), 10 * 4)  # 4 bytes per F32
  
  ggml_free(ctx)
})

test_that("2D tensor creation works", {
  ctx <- ggml_init(1024 * 1024)
  tensor <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  
  expect_type(tensor, "externalptr")
  expect_equal(ggml_nelements(tensor), 20)
  expect_equal(ggml_nbytes(tensor), 20 * 4)
  
  ggml_free(ctx)
})

test_that("tensor data access works", {
  ctx <- ggml_init(1024 * 1024)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  
  data_in <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  ggml_set_f32(tensor, data_in)
  data_out <- ggml_get_f32(tensor)
  
  expect_equal(data_out, data_in, tolerance = 1e-6)
  
  ggml_free(ctx)
})

test_that("tensor size mismatch gives error", {
  ctx <- ggml_init(1024 * 1024)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  
  expect_error(
    ggml_set_f32(tensor, c(1, 2, 3)),  # Wrong size
    "does not match"
  )
  
  ggml_free(ctx)
})
