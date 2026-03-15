# Tests for ONNX Gemm, Pow, Erf, Where, Equal, Clip, Cast

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Gemm ─────────────────────────────────────────────────────────

test_that("ONNX Gemm basic A@B works", {
  # A[2,3] @ B[3,4] = Y[2,4]  (all-ones weights → row sums * 4)
  path <- .onnx_make_gemm(M = 2L, K = 3L, N = 4L)
  a <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(A = a))
  # A=[[1,2,3],[4,5,6]], B=ones(3,4) → [[6,6,6,6],[15,15,15,15]]
  expect_equal(as.numeric(result), c(6, 6, 6, 6, 15, 15, 15, 15), tolerance = 1e-3)
})

test_that("ONNX Gemm with bias works", {
  path <- .onnx_make_gemm(M = 2L, K = 3L, N = 4L,
                           bias_data = c(10, 20, 30, 40))
  a <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(A = a))
  # A@ones + bias: [[6+10, 6+20, 6+30, 6+40], [15+10, 15+20, 15+30, 15+40]]
  expect_equal(as.numeric(result),
               c(16, 26, 36, 46, 25, 35, 45, 55), tolerance = 1e-3)
})

# ── Pow ──────────────────────────────────────────────────────────

test_that("ONNX Pow scalar exponent works", {
  # X^2
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  exp_raw <- .float_bytes(2.0)
  exp_t  <- .onnx_tensor("E", c(1L), 1L, exp_raw)
  exp_vi <- .onnx_value_info("E", 1L, c(1L))
  node <- .onnx_node("Pow", c("X", "E"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, exp_vi), list(outp), list(exp_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x^2, tolerance = 1e-3)
})

test_that("ONNX Pow square root works", {
  # X^0.5
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  exp_raw <- .float_bytes(0.5)
  exp_t  <- .onnx_tensor("E", c(1L), 1L, exp_raw)
  exp_vi <- .onnx_value_info("E", 1L, c(1L))
  node <- .onnx_node("Pow", c("X", "E"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, exp_vi), list(outp), list(exp_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 4, 9, 16)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), sqrt(x), tolerance = 1e-3)
})

# ── Erf ──────────────────────────────────────────────────────────

test_that("ONNX Erf approximation works", {
  path <- .onnx_make_unary("Erf", c(4L))
  x <- c(-1, 0, 0.5, 1)
  result <- run_onnx(path, list(X = x))
  # Uses tanh approximation, so limited precision
  # erf(0) = 0, erf(x) has correct sign
  r <- as.numeric(result)
  expect_equal(r[2], 0, tolerance = 1e-5)      # erf(0) = 0
  expect_true(r[1] < 0)                         # erf(-1) < 0
  expect_true(r[3] > 0 && r[4] > 0)             # erf(0.5), erf(1) > 0
  expect_true(abs(r[4]) > abs(r[3]))             # |erf(1)| > |erf(0.5)|
})

# ── Clip ─────────────────────────────────────────────────────────

test_that("ONNX Clip with attribute min/max works", {
  attrs <- list(.onnx_attr_float("min", 0.0), .onnx_attr_float("max", 6.0))
  path <- .onnx_make_unary("Clip", c(4L), attrs = attrs)
  x <- c(-2, 3, 7, 0)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(0, 3, 6, 0), tolerance = 1e-5)
})

# ── Cast ─────────────────────────────────────────────────────────

test_that("ONNX Cast is pass-through for f32", {
  # Cast with to=1 (FLOAT) — should be identity
  attrs <- list(.onnx_attr_int("to", 1L))
  path <- .onnx_make_unary("Cast", c(4L), attrs = attrs)
  x <- c(1.5, 2.5, 3.5, 4.5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

# ── Equal ────────────────────────────────────────────────────────

test_that("ONNX Equal produces 0/1 mask", {
  inp_a <- .onnx_value_info("A", 1L, c(4L))
  inp_b <- .onnx_value_info("B", 1L, c(4L))
  outp  <- .onnx_value_info("Y", 1L, c(4L))
  node  <- .onnx_node("Equal", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 2, 3, 4)
  b <- c(1, 0, 3, 5)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(1, 0, 1, 0), tolerance = 1e-5)
})

# ── Where ────────────────────────────────────────────────────────

test_that("ONNX Where conditional select works", {
  # Where(cond, X, Y): cond ? X : Y
  inp_c <- .onnx_value_info("C", 1L, c(4L))
  inp_x <- .onnx_value_info("X", 1L, c(4L))
  inp_y <- .onnx_value_info("Y", 1L, c(4L))
  outp  <- .onnx_value_info("Z", 1L, c(4L))
  node  <- .onnx_node("Where", c("C", "X", "Y"), "Z")
  graph <- .onnx_graph("test", list(node),
                        list(inp_c, inp_x, inp_y), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  cond <- c(1, 0, 1, 0)  # boolean as float
  x <- c(10, 20, 30, 40)
  y <- c(100, 200, 300, 400)
  result <- run_onnx(path, list(C = cond, X = x, Y = y))
  expect_equal(as.numeric(result), c(10, 200, 30, 400), tolerance = 1e-3)
})
