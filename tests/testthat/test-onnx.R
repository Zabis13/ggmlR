# Tests for ONNX model loading and inference
#
# Uses .onnx_make_* helpers from tests/testthat/helper-onnx.R to generate
# minimal protobuf .onnx files directly from R (no Python needed).

# ── Helper: load, run, return output vector ──────────────────────

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Binary ops ───────────────────────────────────────────────────

test_that("ONNX Add works", {
  path <- .onnx_make_binary("Add", c(4L))
  a <- c(1, 2, 3, 4)
  b <- c(10, 20, 30, 40)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a + b, tolerance = 1e-5)
})

test_that("ONNX Sub works", {
  path <- .onnx_make_binary("Sub", c(4L))
  a <- c(10, 20, 30, 40)
  b <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a - b, tolerance = 1e-5)
})

test_that("ONNX Mul works", {
  path <- .onnx_make_binary("Mul", c(4L))
  a <- c(2, 3, 4, 5)
  b <- c(10, 10, 10, 10)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a * b, tolerance = 1e-5)
})

test_that("ONNX Div works", {
  path <- .onnx_make_binary("Div", c(4L))
  a <- c(10, 20, 30, 40)
  b <- c(2, 4, 5, 8)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a / b, tolerance = 1e-5)
})

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

# ── Math ops ─────────────────────────────────────────────────────

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

# ── Reshape ──────────────────────────────────────────────────────

test_that("ONNX Reshape preserves elements", {
  path <- .onnx_make_reshape(c(2L, 3L), c(3L, 2L))
  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
})

# ── MatMul ───────────────────────────────────────────────────────

test_that("ONNX MatMul works for 2D", {
  # A[2,3] @ B[3,2] = Y[2,2]
  # A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
  # A@B = [[22,28],[49,64]]
  path <- .onnx_make_matmul(M = 2L, K = 3L, N = 2L)
  a <- c(1, 2, 3, 4, 5, 6)
  b <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(A = a, B = b))
  expected <- c(22, 28, 49, 64)
  expect_equal(as.numeric(result), expected, tolerance = 1e-3)
})

# ── LayerNormalization ───────────────────────────────────────────

test_that("ONNX LayerNormalization works (1D)", {
  path <- .onnx_make_layer_norm(c(4L), eps = 1e-5)
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  # With scale=1, bias=0: should be zero-mean, unit-var
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(abs(mean(r)) < 0.01)
  # ggml_norm uses population stddev (N), not sample stddev (N-1)
  pop_sd <- sqrt(mean((x - mean(x))^2))
  expected <- (x - mean(x)) / pop_sd
  expect_equal(r, expected, tolerance = 0.01)
})

# ── Model metadata ───────────────────────────────────────────────

test_that("onnx_load returns correct metadata", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  expect_s3_class(m, "onnx_model")
  expect_equal(m$n_nodes, 1L)
  expect_true("Relu" %in% m$ops)
})

test_that("onnx_summary works", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  s <- onnx_summary(m)
  expect_true(is.list(s))
  expect_equal(s$n_nodes, 1L)
})

test_that("onnx_inputs returns correct info", {
  path <- .onnx_make_binary("Add", c(4L))
  m <- onnx_load(path, device = "cpu")
  inp <- onnx_inputs(m)
  expect_true(is.list(inp))
  expect_true(length(inp) >= 2)
})

test_that("print.onnx_model works", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  expect_output(print(m), "ONNX Model")
})
