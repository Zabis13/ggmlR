# test-onnx-helpers.R
#
# Unit tests for internal C helper functions via ONNX graph execution:
#   - onnx_squeeze_ndims : trailing-1 collapsing for ggml compat
#   - onnx_reshape_nd    : dispatch to correct ggml_reshape_Nd branch
#   - ne_product         : element count verification in Reshape

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── onnx_squeeze_ndims ───────────────────────────────────────────
# Trailing 1s are collapsed so ggml sees compact tensors.
# Tested by reshaping to shapes with trailing 1s and verifying
# values pass through correctly.

test_that("onnx_squeeze_ndims: [2,3,1] trailing 1 is collapsed", {
  inp  <- .onnx_value_info("X", 1L, c(6L))
  shape_raw <- c(.int64_bytes(2L), .int64_bytes(3L), .int64_bytes(1L))
  shape_t  <- .onnx_tensor("shape", c(3L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L, 1L))
  node_r <- .onnx_node("Reshape", c("X", "shape"), "tmp")
  node_a <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(node_r, node_a),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-1, 2, -3, 4, -5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("onnx_squeeze_ndims: [1,1,4,1] multiple trailing 1s", {
  inp  <- .onnx_value_info("X", 1L, c(4L))
  shape_raw <- c(.int64_bytes(1L), .int64_bytes(1L),
                 .int64_bytes(4L), .int64_bytes(1L))
  shape_t  <- .onnx_tensor("shape", c(4L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 1L))
  node_r <- .onnx_node("Reshape", c("X", "shape"), "tmp")
  node_a <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(node_r, node_a),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-1, 2, -3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 4)
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("onnx_squeeze_ndims: [1,1,1,1] all-ones shape (scalar repeat)", {
  inp  <- .onnx_value_info("X", 1L, c(1L))
  shape_raw <- c(.int64_bytes(1L), .int64_bytes(1L),
                 .int64_bytes(1L), .int64_bytes(1L))
  shape_t  <- .onnx_tensor("shape", c(4L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 1L, 1L))
  node_r <- .onnx_node("Reshape", c("X", "shape"), "tmp")
  node_a <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(node_r, node_a),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = 5.0))
  expect_equal(as.numeric(result), 5.0, tolerance = 1e-5)
})

# ── onnx_reshape_nd dispatch ─────────────────────────────────────
# Each test exercises a different ggml_reshape_Nd branch (1D–4D).

test_that("onnx_reshape_nd 1D: [12] → [12] identity", {
  path <- .onnx_make_reshape(c(12L), c(12L))
  x <- seq_len(12) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

test_that("onnx_reshape_nd 2D: [3,4] → [4,3]", {
  path <- .onnx_make_reshape(c(3L, 4L), c(4L, 3L))
  x <- seq_len(12) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 12)
  expect_equal(sum(result), sum(x), tolerance = 1e-5)
})

test_that("onnx_reshape_nd 3D: [2,3,4] → [24]", {
  path <- .onnx_make_reshape(c(2L, 3L, 4L), c(24L))
  x <- seq_len(24) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

test_that("onnx_reshape_nd 4D: [6] → [1,1,2,3]", {
  path <- .onnx_make_reshape(c(6L), c(1L, 1L, 2L, 3L))
  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

test_that("onnx_reshape_nd 5D: [24] → [1,2,3,4,1] (5D branch)", {
  inp  <- .onnx_value_info("X", 1L, c(24L))
  shape_raw <- c(.int64_bytes(1L), .int64_bytes(2L), .int64_bytes(3L),
                 .int64_bytes(4L), .int64_bytes(1L))
  shape_t  <- .onnx_tensor("shape", c(5L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(5L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L, 4L, 1L))
  node_r <- .onnx_node("Reshape", c("X", "shape"), "tmp")
  node_a <- .onnx_node("Relu",    "tmp",            "Y")
  graph <- .onnx_graph("test", list(node_r, node_a),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq_len(24) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
  expect_equal(sum(result), sum(x), tolerance = 1e-3)
})

# ── ne_product: element count verification ───────────────────────
# Reshape with mismatched count must be caught, not silently corrupt.

test_that("ne_product: mismatch [6] → [2,4] node is skipped (output NULL)", {
  inp  <- .onnx_value_info("X", 1L, c(6L))
  shape_raw <- c(.int64_bytes(2L), .int64_bytes(4L))  # 8 != 6
  shape_t  <- .onnx_tensor("shape", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(2L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L))
  node <- .onnx_node("Reshape", c("X", "shape"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # Mismatch is logged as warning; node is skipped → output tensor absent → NULL
  m <- onnx_load(path, device = "cpu")
  res <- onnx_run(m, list(X = seq_len(6) * 1.0))
  expect_null(res[[1]])
})

test_that("ne_product: mismatch [12] → [3,5] output is NULL", {
  inp  <- .onnx_value_info("X", 1L, c(12L))
  shape_raw <- c(.int64_bytes(3L), .int64_bytes(5L))  # 15 != 12
  shape_t  <- .onnx_tensor("shape", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(2L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 5L))
  node <- .onnx_node("Reshape", c("X", "shape"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  m <- onnx_load(path, device = "cpu")
  res <- onnx_run(m, list(X = seq_len(12) * 1.0))
  expect_null(res[[1]])
})

test_that("ne_product: -1 inference correct: [2,3,4] → [-1,4] = [6,4]", {
  path <- .onnx_make_reshape(c(2L, 3L, 4L), c(-1L, 4L))
  x <- seq_len(24) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
  expect_equal(sum(result), sum(x), tolerance = 1e-5)
})

test_that("ne_product: -1 inference in middle: [2,3,4] → [2,-1,2]", {
  path <- .onnx_make_reshape(c(2L, 3L, 4L), c(2L, -1L, 2L))
  x <- seq_len(24) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
  expect_equal(sum(result), sum(x), tolerance = 1e-5)
})
