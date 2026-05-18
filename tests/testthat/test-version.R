test_that("version function works", {
  version <- ggml_version()
  expect_type(version, "character")
  expect_equal(version, "0.11.0")
})

test_that("test function works", {
  result <- ggml_test()
  expect_true(result)
})
