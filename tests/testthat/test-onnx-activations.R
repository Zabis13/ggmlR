# Tests for ONNX activation and math ops

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Activations ──────────────────────────────────────────────────

test_that("ONNX Relu works", {
  path <- .onnx_make_unary("Relu", c(4L))
  x <- c(-2, -1, 0, 3)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("ONNX Sigmoid works", {
  path <- .onnx_make_unary("Sigmoid", c(4L))
  x <- c(-2, 0, 1, 5)
  result <- run_onnx(path, list(X = x))
  expected <- 1 / (1 + exp(-x))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX Tanh works", {
  path <- .onnx_make_unary("Tanh", c(4L))
  x <- c(-2, -0.5, 0, 1.5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), tanh(x), tolerance = 1e-5)
})

test_that("ONNX Silu works", {
  path <- .onnx_make_unary("Silu", c(4L))
  x <- c(-2, -1, 0, 2)
  result <- run_onnx(path, list(X = x))
  expected <- x / (1 + exp(-x))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX Elu works", {
  path <- .onnx_make_unary("Elu", c(4L))
  x <- c(-2, -1, 0, 1)
  result <- run_onnx(path, list(X = x))
  expected <- ifelse(x >= 0, x, exp(x) - 1)
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX Softmax works", {
  path <- .onnx_make_unary("Softmax", c(4L))
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expected <- exp(x) / sum(exp(x))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX LeakyRelu works", {
  attrs <- list(.onnx_attr_float("alpha", 0.1))
  path <- .onnx_make_unary("LeakyRelu", c(4L), attrs = attrs)
  x <- c(-2, -1, 0, 3)
  result <- run_onnx(path, list(X = x))
  expected <- ifelse(x >= 0, x, 0.1 * x)
  expect_equal(as.numeric(result), expected, tolerance = 1e-5)
})

# ── Identity / Dropout ───────────────────────────────────────────

test_that("ONNX Identity is pass-through", {
  path <- .onnx_make_unary("Identity", c(4L))
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

test_that("ONNX Dropout is pass-through (inference)", {
  path <- .onnx_make_unary("Dropout", c(4L))
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

# ── Chain ops ────────────────────────────────────────────────────

test_that("ONNX chain Relu -> Sigmoid works", {
  path <- .onnx_make_chain("Relu", "Sigmoid", c(4L))
  x <- c(-2, -1, 0, 3)
  result <- run_onnx(path, list(X = x))
  expected <- 1 / (1 + exp(-pmax(x, 0)))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})
