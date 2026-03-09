test_that("ggml_version works", {
  expect_equal(ggml_version(), "0.9.5")
})

test_that("ggml_test works", {
  expect_true(ggml_test())
})

test_that("ggml context can be created and freed", {
  ctx <- ggml_init(1024 * 1024)  # 1MB
  expect_type(ctx, "externalptr")
  
  # Освобождение контекста
  expect_silent(ggml_free(ctx))
})

test_that("ggml_init handles invalid input", {
  skip("Platform-dependent behavior")
})
