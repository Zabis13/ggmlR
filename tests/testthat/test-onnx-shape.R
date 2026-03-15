# Tests for ONNX shape manipulation ops:
# Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Concat, Split, Expand

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Reshape ──────────────────────────────────────────────────────

test_that("ONNX Reshape preserves elements", {
  path <- .onnx_make_reshape(c(2L, 3L), c(3L, 2L))
  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
})

test_that("ONNX Reshape with -1 infers dim", {
  path <- .onnx_make_reshape(c(2L, 3L), c(-1L))
  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

test_that("ONNX Reshape with -1 partial", {
  path <- .onnx_make_reshape(c(2L, 3L, 4L), c(2L, -1L))
  x <- seq_len(24)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
})

test_that("ONNX Reshape with 0 keeps dim", {
  inp <- .onnx_value_info("X", 1L, c(2L, 3L))
  shape_raw <- c(.int64_bytes(0L), .int64_bytes(-1L))
  shape_t  <- .onnx_tensor("shape", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(2L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))
  node <- .onnx_node("Reshape", c("X", "shape"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, shape_vi), list(outp),
                        list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

# ── Transpose ──────────────────────────────────────────────────

test_that("ONNX Transpose 2D default (reverse dims)", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 2L))
  node <- .onnx_node("Transpose", "X", "Y")
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(1, 4, 2, 5, 3, 6), tolerance = 1e-5)
})

test_that("ONNX Transpose 3D with perm", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L, 3L))
  node <- .onnx_node("Transpose", "X", "Y",
                      attrs = list(.onnx_attr_ints("perm", c(0L, 2L, 1L))))
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq_len(24)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
  expect_equal(result[1], 1, tolerance = 1e-5)
})

test_that("ONNX Transpose 4D NCHW to NHWC perm=[0,2,3,1]", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 3L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 3L, 4L, 2L))
  node <- .onnx_node("Transpose", "X", "Y",
                      attrs = list(.onnx_attr_ints("perm", c(0L, 2L, 3L, 1L))))
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq_len(24)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
  expect_equal(result[1], 1, tolerance = 1e-5)
  expect_equal(sum(result), sum(seq_len(24)), tolerance = 1e-3)
})

test_that("ONNX Transpose identity perm is no-op", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))
  node <- .onnx_node("Transpose", "X", "Y",
                      attrs = list(.onnx_attr_ints("perm", c(0L, 1L))))
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

# ── Flatten ────────────────────────────────────────────────────

test_that("ONNX Flatten works", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(6L))
  node <- .onnx_node("Flatten", "X", "Y")
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

# ── Unsqueeze / Squeeze ──────────────────────────────────────────

test_that("ONNX Unsqueeze works (attr axes)", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L))
  n1 <- .onnx_node("Unsqueeze", "X", "tmp",
                    attrs = list(.onnx_attr_ints("axes", c(0L))))
  n2 <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(n1, n2), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, -2, 3, -4, 5, -6)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("ONNX Squeeze works (attr axes)", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))
  n1 <- .onnx_node("Squeeze", "X", "tmp",
                    attrs = list(.onnx_attr_ints("axes", c(0L))))
  n2 <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(n1, n2), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, -2, 3, -4, 5, -6)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("ONNX Squeeze without axes squeezes all 1-dims", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 4L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  n1 <- .onnx_node("Squeeze", "X", "tmp")
  n2 <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(n1, n2), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-1, 2, -3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

# ── Concat ─────────────────────────────────────────────────────

test_that("ONNX Concat axis=0 works", {
  inp_a <- .onnx_value_info("A", 1L, c(3L))
  inp_b <- .onnx_value_info("B", 1L, c(2L))
  outp  <- .onnx_value_info("Y", 1L, c(5L))
  node  <- .onnx_node("Concat", c("A", "B"), "Y",
                        attrs = list(.onnx_attr_int("axis", 0L)))
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 2, 3)
  b <- c(4, 5)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(1, 2, 3, 4, 5), tolerance = 1e-5)
})

test_that("ONNX Concat axis=1 works (2D)", {
  inp_a <- .onnx_value_info("A", 1L, c(2L, 2L))
  inp_b <- .onnx_value_info("B", 1L, c(2L, 3L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 5L))
  node  <- .onnx_node("Concat", c("A", "B"), "Y",
                        attrs = list(.onnx_attr_int("axis", 1L)))
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 2, 3, 4)
  b <- c(5, 6, 7, 8, 9, 10)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(1, 2, 5, 6, 7, 3, 4, 8, 9, 10), tolerance = 1e-5)
})

# ── Split ────────────────────────────────────────────────────────

test_that("ONNX Split works with equal chunks", {
  inp  <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Z", 1L, c(3L))
  split_node <- .onnx_node("Split", "X", c("Y1", "Y2"),
                            attrs = list(.onnx_attr_int("axis", 0L)))
  add_node <- .onnx_node("Add", c("Y1", "Y2"), "Z")
  graph <- .onnx_graph("test", list(split_node, add_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 10, 20, 30)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(11, 22, 33), tolerance = 1e-5)
})

test_that("ONNX Split works with explicit sizes", {
  inp  <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Z", 1L, c(4L))
  split_raw <- c(.int64_bytes(2L), .int64_bytes(4L))
  split_t  <- .onnx_tensor("sp", c(2L), 7L, split_raw)
  split_vi <- .onnx_value_info("sp", 7L, c(2L))
  split_node <- .onnx_node("Split", c("X", "sp"), c("Y1", "Y2"),
                            attrs = list(.onnx_attr_int("axis", 0L)))
  relu_node <- .onnx_node("Relu", "Y2", "Z")
  graph <- .onnx_graph("test", list(split_node, relu_node),
                        list(inp, split_vi), list(outp),
                        list(split_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(100, 200, -1, 2, -3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(0, 2, 0, 4), tolerance = 1e-5)
})

# ── Expand ───────────────────────────────────────────────────────

test_that("ONNX Expand broadcasts 1D to 2D", {
  inp  <- .onnx_value_info("X", 1L, c(3L))
  shape_raw <- c(.int64_bytes(2L), .int64_bytes(3L))
  shape_t  <- .onnx_tensor("shape", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(2L))
  outp <- .onnx_value_info("Z", 1L, c(2L, 3L))
  expand_node <- .onnx_node("Expand", c("X", "shape"), "Y")
  relu_node   <- .onnx_node("Relu", "Y", "Z")
  graph <- .onnx_graph("test", list(expand_node, relu_node),
                        list(inp, shape_vi), list(outp),
                        list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-1, 2, 3)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(0, 2, 3, 0, 2, 3), tolerance = 1e-5)
})

test_that("ONNX Expand broadcasts scalar to 1D", {
  inp  <- .onnx_value_info("X", 1L, c(1L))
  shape_raw <- .int64_bytes(4L)
  shape_t  <- .onnx_tensor("shape", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(1L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  expand_node <- .onnx_node("Expand", c("X", "shape"), "Y")
  graph <- .onnx_graph("test", list(expand_node),
                        list(inp, shape_vi), list(outp),
                        list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(5, 5, 5, 5), tolerance = 1e-5)
})

# ── Slice ────────────────────────────────────────────────────────

test_that("ONNX Slice works on 1D", {
  inp  <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Y", 1L, c(3L))
  starts_raw <- .int64_bytes(1L)
  ends_raw   <- .int64_bytes(4L)
  starts_t <- .onnx_tensor("starts", c(1L), 7L, starts_raw)
  ends_t   <- .onnx_tensor("ends",   c(1L), 7L, ends_raw)
  starts_vi <- .onnx_value_info("starts", 7L, c(1L))
  ends_vi   <- .onnx_value_info("ends",   7L, c(1L))
  node <- .onnx_node("Slice", c("X", "starts", "ends"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, starts_vi, ends_vi), list(outp),
                        list(starts_t, ends_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(10, 20, 30, 40, 50, 60)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(20, 30, 40), tolerance = 1e-5)
})

test_that("ONNX Slice works on 2D with axes", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 2L))
  starts_raw <- .int64_bytes(1L)
  ends_raw   <- .int64_bytes(3L)
  axes_raw   <- .int64_bytes(1L)
  starts_t <- .onnx_tensor("starts", c(1L), 7L, starts_raw)
  ends_t   <- .onnx_tensor("ends",   c(1L), 7L, ends_raw)
  axes_t   <- .onnx_tensor("axes",   c(1L), 7L, axes_raw)
  starts_vi <- .onnx_value_info("starts", 7L, c(1L))
  ends_vi   <- .onnx_value_info("ends",   7L, c(1L))
  axes_vi   <- .onnx_value_info("axes",   7L, c(1L))
  node <- .onnx_node("Slice", c("X", "starts", "ends", "axes"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, starts_vi, ends_vi, axes_vi), list(outp),
                        list(starts_t, ends_t, axes_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(2, 3, 6, 7), tolerance = 1e-5)
})
