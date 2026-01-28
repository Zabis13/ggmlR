test_that("GGML_TYPE_F32 constant is defined", {
  expect_equal(GGML_TYPE_F32, 0L)
})

test_that("type size calculation works", {
  # F32 должен быть 4 байта
  ctx <- ggml_init(1024 * 1024)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  
  expect_equal(ggml_nbytes(tensor), 400)  # 100 * 4
  
  ggml_free(ctx)
})
