# Chain tests: Audio / voice model patterns (Whisper-style)
# Transpose → Reshape → MatMul → Softmax
#
# Tests ndims tracking through Transpose+Reshape, which is a common
# source of bugs when ggml ne[] order diverges from ONNX expectations.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}


# ── Minimal (2 ops): Transpose → MatMul ─────────────────────

test_that("chain audio: Transpose→MatMul (minimal)", {
  # Input: [2, 3] → Transpose → [3, 2] → MatMul with W[2, 4] → [3, 4]

  inp <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 4L))

  # W: [2, 4]
  w_data <- rep(1.0, 8)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 4L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 4L))

  trans_node <- .onnx_node("Transpose", "X", "t_out",
                            attrs = list(.onnx_attr_ints("perm", c(1L, 0L))))
  mm_node    <- .onnx_node("MatMul", c("t_out", "W"), "Y")

  graph <- .onnx_graph("test",
                        list(trans_node, mm_node),
                        list(inp, w_vi), list(outp),
                        list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # X = [[1,2,3],[4,5,6]] → Transpose = [[1,4],[2,5],[3,6]]
  # MatMul with ones → each row sums: [5,7,9] repeated 4 times
  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 12)
  # Row sums: 1+4=5, 2+5=7, 3+6=9, each repeated 4 cols
  expected <- rep(c(5, 7, 9), each = 4)
  expect_equal(r, expected, tolerance = 1e-4)
})


# ── Real (4 ops): Transpose → Reshape → MatMul → Softmax ────

test_that("chain audio: Transpose→Reshape→MatMul→Softmax (decoder)", {
  # Simulates audio decoder output:
  # Input: [1, 4, 3] (batch=1, time=4, features=3)
  # Transpose(0,2,1) → [1, 3, 4]  (swap time/features)
  # Reshape → [3, 4]  (remove batch)
  # MatMul W[4, 5] → [3, 5]  (project to vocab)
  # Softmax → [3, 5]  (token probabilities)

  inp <- .onnx_value_info("X", 1L, c(1L, 4L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 5L))

  # Reshape target: [3, 4]
  shape_raw <- c(writeBin(3L, raw(), size = 8, endian = "little"),
                 writeBin(4L, raw(), size = 8, endian = "little"))
  shape_t  <- .onnx_tensor("shape", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(2L))

  # W: [4, 5]
  set.seed(7)
  w_data <- rnorm(20, 0, 0.5)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(4L, 5L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(4L, 5L))

  trans_node <- .onnx_node("Transpose", "X", "t1",
                            attrs = list(.onnx_attr_ints("perm", c(0L, 2L, 1L))))
  resh_node  <- .onnx_node("Reshape", c("t1", "shape"), "r1")
  mm_node    <- .onnx_node("MatMul", c("r1", "W"), "mm")
  sm_node    <- .onnx_node("Softmax", "mm", "Y",
                            attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
                        list(trans_node, resh_node, mm_node, sm_node),
                        list(inp, shape_vi, w_vi),
                        list(outp),
                        list(shape_t, w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(12, -1, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 15)  # 3 x 5
  # All softmax outputs in [0,1], total sum = 3 (three rows)
  expect_true(all(r >= 0 & r <= 1))
  expect_equal(sum(r), 3.0, tolerance = 1e-3)
})


# ── Boundary: single time step ───────────────────────────────

test_that("chain audio: single time step (boundary)", {
  # Input: [1, 1, 4] → Transpose(0,2,1) → [1, 4, 1]
  # Reshape → [4] → MatMul W[4, 3] → ??? (1D x 2D)
  # Use Flatten instead to get [1, 4] then MatMul

  # Simpler: [2, 3] → Transpose → [3, 2] → Reshape → [6] → done
  # This tests ndims tracking at minimal scale

  inp <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(6L))

  shape_raw <- writeBin(6L, raw(), size = 8, endian = "little")
  shape_t  <- .onnx_tensor("shape", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(1L))

  trans_node <- .onnx_node("Transpose", "X", "t1",
                            attrs = list(.onnx_attr_ints("perm", c(1L, 0L))))
  resh_node  <- .onnx_node("Reshape", c("t1", "shape"), "Y")

  graph <- .onnx_graph("test",
                        list(trans_node, resh_node),
                        list(inp, shape_vi), list(outp),
                        list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # X = [[1,2,3],[4,5,6]] → Transpose = [[1,4],[2,5],[3,6]] → Flatten = [1,4,2,5,3,6]
  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 6)
  expect_equal(r, c(1, 4, 2, 5, 3, 6), tolerance = 1e-4)
})
