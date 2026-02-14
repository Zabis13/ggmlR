test_that("ggml_tensor_overhead returns positive value", {
  overhead <- ggml_tensor_overhead()
  expect_type(overhead, "double")
  expect_true(overhead > 0)
  expect_true(overhead < 10000)  # Разумный верхний предел
})

test_that("ggml_get_mem_size returns context size", {
  size <- 16 * 1024 * 1024
  ctx <- ggml_init(size)
  
  mem_size <- ggml_get_mem_size(ctx)
  expect_type(mem_size, "double")
  expect_equal(mem_size, size, tolerance = 1000)
  
  ggml_free(ctx)
})

test_that("ggml_used_mem tracks memory usage", {
  ctx <- ggml_init(16 * 1024 * 1024)
  
  # Начальное использование должно быть 0
  used_initial <- ggml_used_mem(ctx)
  expect_equal(used_initial, 0)
  
  # Создать тензор
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  used_after <- ggml_used_mem(ctx)
  
  # Память должна увеличиться
  expect_true(used_after > used_initial)
  expect_true(used_after >= 1000 * 4)  # Минимум размер данных
  
  ggml_free(ctx)
})

test_that("ggml_estimate_memory gives reasonable estimates", {
  # Для 1D тензора
  est_1d <- ggml_estimate_memory(GGML_TYPE_F32, 1000)
  expect_true(est_1d >= 1000 * 4)  # Минимум размер данных
  expect_true(est_1d < 1000 * 4 + 1000)  # Не слишком много overhead
  
  # Для 2D тензора
  est_2d <- ggml_estimate_memory(GGML_TYPE_F32, 100, 100)
  expect_true(est_2d >= 10000 * 4)
  
  # Для 3D тензора
  est_3d <- ggml_estimate_memory(GGML_TYPE_F32, 10, 10, 10)
  expect_true(est_3d >= 1000 * 4)
})

test_that("ggml_print_mem_status works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  
  # Должна вернуть список с информацией
  invisible(capture.output(info <- ggml_print_mem_status(ctx)))
  expect_type(info, "list")
  expect_named(info, c("total", "used", "free"))
  
  expect_equal(info$total, 16 * 1024 * 1024, tolerance = 1000)
  expect_equal(info$used + info$free, info$total, tolerance = 10)
  
  ggml_free(ctx)
})
