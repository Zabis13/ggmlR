test_that("context initialization works", {
  ctx <- ggml_init(1024 * 1024)
  expect_type(ctx, "externalptr")
  ggml_free(ctx)
})

test_that("context can be freed", {
  ctx <- ggml_init(1024 * 1024)
  expect_silent(ggml_free(ctx))
})

test_that("context can be reset", {
  ctx <- ggml_init(10 * 1024 * 1024)
  
  # Создать тензор
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10000)
  expect_true(ggml_used_mem(ctx) > 0)
  
  # Reset
  expect_silent(ggml_reset(ctx))
  expect_equal(ggml_used_mem(ctx), 0)
  
  ggml_free(ctx)
})

test_that("multiple contexts can coexist", {
  ctx1 <- ggml_init(1024 * 1024)
  ctx2 <- ggml_init(2 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 100)
  b <- ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 200)

  expect_equal(ggml_nelements(a), 100)
  expect_equal(ggml_nelements(b), 200)

  ggml_free(ctx1)
  ggml_free(ctx2)
})

test_that("ggml_set_n_threads and ggml_get_n_threads work", {
  original <- ggml_get_n_threads()
  expect_true(original >= 1)

  # Set to specific value - function should not error
  result <- ggml_set_n_threads(2)
  expect_true(result >= 1)  # returns the set value or 1 if no OpenMP

  # get_n_threads returns at least 1
  expect_true(ggml_get_n_threads() >= 1)

  # Restore original
  ggml_set_n_threads(original)
})
