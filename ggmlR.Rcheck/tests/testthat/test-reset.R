test_that("ggml_reset clears context memory", {
  ctx <- ggml_init(50 * 1024 * 1024)
  
  # Создать тензор
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000000)
  
  # Проверить что память используется
  used_before <- ggml_used_mem(ctx)
  expect_true(used_before > 0)
  
  # Reset
  ggml_reset(ctx)
  
  # Проверить что память освободилась
  used_after <- ggml_used_mem(ctx)
  expect_equal(used_after, 0)
  
  ggml_free(ctx)
})

test_that("ggml_reset allows reusing context", {
  ctx <- ggml_init(50 * 1024 * 1024)
  
  # Создать первый тензор
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5000000)
  used_1 <- ggml_used_mem(ctx)
  
  # Reset
  ggml_reset(ctx)
  
  # Создать второй тензор такого же размера
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5000000)
  used_2 <- ggml_used_mem(ctx)
  
  # Использованная память должна быть примерно одинаковая
  expect_equal(used_1, used_2, tolerance = 100)
  
  ggml_free(ctx)
})

test_that("ggml_reset works multiple times", {
  ctx <- ggml_init(50 * 1024 * 1024)
  
  for (i in 1:10) {
    # Создать тензор
    a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000000)
    
    # Reset
    ggml_reset(ctx)
    
    # Проверить что память освободилась
    expect_equal(ggml_used_mem(ctx), 0)
  }
  
  ggml_free(ctx)
})

test_that("ggml_reset enables memory reuse", {
  # Более простой тест - просто показываем что после reset 
  # можно создать тензор того же размера
  ctx <- ggml_init(30 * 1024 * 1024)
  
  # Заполнить почти всю память
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7000000)  # ~27 MB
  used_after_first <- ggml_used_mem(ctx)
  free_after_first <- ggml_get_mem_size(ctx) - used_after_first
  
  # Свободной памяти должно быть мало
  expect_true(free_after_first < 5 * 1024 * 1024)  # Меньше 5 MB
  
  # После reset память освобождается
  ggml_reset(ctx)
  expect_equal(ggml_used_mem(ctx), 0)
  
  # Можно снова создать большой тензор
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7000000)
  expect_type(b, "externalptr")
  
  ggml_free(ctx)
})

test_that("ggml_reset with data operations", {
  ctx <- ggml_init(50 * 1024 * 1024)
  
  # Первая операция
  a1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a1, 1:100)
  data1 <- ggml_get_f32(a1)
  expect_equal(data1, 1:100)
  
  # Reset
  ggml_reset(ctx)
  
  # Вторая операция (после reset можем переиспользовать память)
  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a2, 101:200)
  data2 <- ggml_get_f32(a2)
  expect_equal(data2, 101:200)
  
  ggml_free(ctx)
})

test_that("ggml_reset works with large matrices", {
  ctx <- ggml_init(100 * 1024 * 1024)
  
  # Создать большую матрицу
  m1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 1000)
  used1 <- ggml_used_mem(ctx)
  
  # Reset
  ggml_reset(ctx)
  expect_equal(ggml_used_mem(ctx), 0)
  
  # Создать еще большую матрицу
  m2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2000, 2000)
  used2 <- ggml_used_mem(ctx)
  
  # Вторая матрица должна занять больше памяти
  expect_true(used2 > used1)
  
  ggml_free(ctx)
})
