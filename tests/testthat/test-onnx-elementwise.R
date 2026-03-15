# Tests for ONNX elementwise binary ops and broadcast

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Binary ops (same shape) ─────────────────────────────────────

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

# ── Broadcast binary ops ────────────────────────────────────────

test_that("ONNX Add broadcast [4] + [1]", {
  path <- .onnx_make_binary("Add", c(4L), c(1L))
  a <- c(1, 2, 3, 4)
  b <- c(10)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(11, 12, 13, 14), tolerance = 1e-5)
})

test_that("ONNX Mul broadcast [2,4] * [1,4]", {
  inp_a <- .onnx_value_info("A", 1L, c(2L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 4L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 4L))
  node  <- .onnx_node("Mul", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 2, 3, 4, 5, 6, 7, 8)
  b <- c(10, 20, 30, 40)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(10, 40, 90, 160, 50, 120, 210, 320), tolerance = 1e-3)
})

test_that("ONNX Add broadcast [2,3,4] + [1,1,4]", {
  inp_a <- .onnx_value_info("A", 1L, c(2L, 3L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 1L, 4L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 3L, 4L))
  node  <- .onnx_node("Add", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- seq_len(24)
  b <- c(100, 200, 300, 400)
  result <- run_onnx(path, list(A = a, B = b))
  expected <- a + rep(b, 6)
  expect_equal(as.numeric(result), expected, tolerance = 1e-3)
})

test_that("ONNX Add broadcast [2,3] + [3] (right-align)", {
  inp_a <- .onnx_value_info("A", 1L, c(2L, 3L))
  inp_b <- .onnx_value_info("B", 1L, c(3L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 3L))
  node  <- .onnx_node("Add", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 2, 3, 4, 5, 6)
  b <- c(10, 20, 30)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(11, 22, 33, 14, 25, 36), tolerance = 1e-5)
})

test_that("ONNX Add broadcast [1,4,1,1] + [1,4,3,3] (channel bias)", {
  inp_a <- .onnx_value_info("A", 1L, c(1L, 4L, 1L, 1L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 4L, 3L, 3L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 4L, 3L, 3L))
  node  <- .onnx_node("Add", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(10, 20, 30, 40)
  b <- rep(1, 36)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(length(result), 36)
  expect_equal(result[1], 11, tolerance = 1e-3)
  expect_equal(result[10], 21, tolerance = 1e-3)
  expect_equal(result[19], 31, tolerance = 1e-3)
  expect_equal(result[28], 41, tolerance = 1e-3)
})
