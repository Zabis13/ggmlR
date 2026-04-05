# Tests for threading functions

test_that("ggml_set_omp_threads runs without error", {
  expect_no_error(ggml_set_omp_threads(2L))
})

test_that("ggml_set_omp_threads accepts 1 thread", {
  expect_no_error(ggml_set_omp_threads(1L))
  # Restore to 2 for CRAN compliance
  ggml_set_omp_threads(2L)
})
