# Tests for ONNX math ops

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

test_that("ONNX Sqrt works", {
  path <- .onnx_make_unary("Sqrt", c(4L))
  x <- c(1, 4, 9, 16)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), sqrt(x), tolerance = 1e-5)
})

test_that("ONNX Exp works", {
  path <- .onnx_make_unary("Exp", c(4L))
  x <- c(0, 1, 2, -1)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), exp(x), tolerance = 1e-4)
})

test_that("ONNX Log works", {
  path <- .onnx_make_unary("Log", c(4L))
  x <- c(1, 2, 10, 100)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), log(x), tolerance = 1e-4)
})

test_that("ONNX Abs works", {
  path <- .onnx_make_unary("Abs", c(4L))
  x <- c(-3, -1, 0, 5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), abs(x), tolerance = 1e-5)
})

test_that("ONNX Neg works", {
  path <- .onnx_make_unary("Neg", c(4L))
  x <- c(-3, -1, 0, 5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), -x, tolerance = 1e-5)
})

test_that("ONNX Floor works", {
  path <- .onnx_make_unary("Floor", c(4L))
  x <- c(-1.5, 0.3, 2.7, 3.0)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), floor(x), tolerance = 1e-5)
})

test_that("ONNX Ceil works", {
  path <- .onnx_make_unary("Ceil", c(4L))
  x <- c(-1.5, 0.3, 2.7, 3.0)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), ceiling(x), tolerance = 1e-5)
})
