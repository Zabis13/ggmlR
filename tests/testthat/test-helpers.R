test_that("ggml_init_auto creates appropriate context", {
  # Для одного тензора
  ctx <- ggml_init_auto(tensor1 = 1000)
  
  # Должен суметь создать тензор
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  expect_type(a, "externalptr")
  
  ggml_free(ctx)
})

test_that("ggml_init_auto handles multiple tensors", {
  ctx <- ggml_init_auto(
    vec = 1000,
    mat = c(100, 100)
  )
  
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 100)
  
  expect_type(a, "externalptr")
  expect_type(b, "externalptr")
  
  ggml_free(ctx)
})

test_that("ggml_with_temp_ctx executes and cleans up", {
  result <- ggml_with_temp_ctx(16 * 1024 * 1024, {
    a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
    ggml_set_f32(a, 1:100)
    ggml_get_f32(a)
  })
  
  expect_type(result, "double")
  expect_length(result, 100)
  expect_equal(result, 1:100)
})

test_that("ggml_with_temp_ctx allows operations", {
  result <- ggml_with_temp_ctx(16 * 1024 * 1024, {
    a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
    b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
    
    ggml_set_f32(a, c(1, 2, 3, 4, 5))
    ggml_set_f32(b, c(5, 4, 3, 2, 1))
    
    ggml_cpu_add(a, b)
  })
  
  expect_equal(result, c(6, 6, 6, 6, 6))
})
