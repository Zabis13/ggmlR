# test-onnx-edge.R
#
# Boundary / edge case tests:
#   - 5D tensor Reshape and Transpose
#   - Scalar tensors (single element)
#   - Broadcast rank-1 vs rank-N
#   - Negative axis normalisation
#   - Large batch / many trailing 1s

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── 5D Reshape ───────────────────────────────────────────────────

test_that("5D Reshape: [2,3,4,3,2] → [144] flatten", {
  # 2*3*4*3*2 = 144; mixed-sign so Relu result is element-wise verifiable
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L, 4L, 3L, 2L))
  shape_raw <- .int64_bytes(144L)
  shape_t  <- .onnx_tensor("shape", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(1L))
  outp <- .onnx_value_info("Y", 1L, c(144L))
  node_r <- .onnx_node("Reshape", c("X", "shape"), "tmp")
  node_a <- .onnx_node("Relu",    "tmp",            "Y")
  graph <- .onnx_graph("test", list(node_r, node_a),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(-72, 71) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 144)
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("5D Reshape: [144] → [2,3,4,3,2] expand", {
  # 2*3*4*3*2 = 144; mixed-sign so Relu result is element-wise verifiable
  inp  <- .onnx_value_info("X", 1L, c(144L))
  shape_raw <- c(.int64_bytes(2L), .int64_bytes(3L), .int64_bytes(4L),
                 .int64_bytes(3L), .int64_bytes(2L))
  shape_t  <- .onnx_tensor("shape", c(5L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(5L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L, 4L, 3L, 2L))
  node_r <- .onnx_node("Reshape", c("X", "shape"), "tmp")
  node_a <- .onnx_node("Relu",    "tmp",            "Y")
  graph <- .onnx_graph("test", list(node_r, node_a),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(-72, 71) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 144)
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("5D Reshape round-trip: [24] → [1,2,3,4,1] preserves values", {
  # 1*2*3*4*1 = 24; use mixed-sign input so Relu result is deterministic
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

  x <- seq(-12, 11) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

# ── 5D Transpose ─────────────────────────────────────────────────
# Exercises the 5D path: squeeze-batch → 4D permute → restore 5D.

test_that("5D Transpose: perm=[0,1,2,4,3] swaps last two dims", {
  # Shape [1,2,1,2,3]: small enough to verify element positions by hand.
  # ONNX row-major layout, elements indexed as [n,a,b,c,d].
  # perm=[0,1,2,4,3]: output[n,a,b,d,c] = input[n,a,b,c,d]
  # x[1..12] row-major [1,2,1,2,3]:
  #   x[1]=input[0,0,0,0,0], x[2]=input[0,0,0,0,1], x[3]=input[0,0,0,0,2]
  #   x[4]=input[0,0,0,1,0], x[5]=input[0,0,0,1,1], x[6]=input[0,0,0,1,2]
  #   x[7]=input[0,1,0,0,0], ...
  # output shape [1,2,1,3,2]; output[n,a,b,d,c]=input[n,a,b,c,d]
  # output[0,0,0,0,0]=input[0,0,0,0,0]=1; output[0,0,0,0,1]=input[0,0,0,1,0]=4
  # output[0,0,0,1,0]=input[0,0,0,0,1]=2; output[0,0,0,1,1]=input[0,0,0,1,1]=5
  # output[0,0,0,2,0]=input[0,0,0,0,2]=3; output[0,0,0,2,1]=input[0,0,0,1,2]=6
  # → first 6 output elements (row-major): 1,4,2,5,3,6
  # second group (a=1): 7,10,8,11,9,12
  inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 1L, 2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 1L, 3L, 2L))
  node <- .onnx_node("Transpose", "X", "Y",
                      attrs = list(.onnx_attr_ints("perm", c(0L,1L,2L,4L,3L))))
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq_len(12) * 1.0
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(length(result), 12)
  expect_equal(result, c(1,4,2,5,3,6, 7,10,8,11,9,12), tolerance = 1e-5)
})

test_that("5D Transpose: perm=[0,2,1,3,4] swaps middle dims", {
  # Shape [1,2,3,1,1]: perm=[0,2,1,3,4] → output [1,3,2,1,1]
  # input[0,a,b,0,0] → output[0,b,a,0,0]
  # Row-major: x = [x[0,0,0], x[0,0,1], x[0,0,2], x[0,1,0], x[0,1,1], x[0,1,2]]
  #              = [1,2,3,4,5,6]
  # Output row-major [1,3,2,1,1]: output[0,b,a,0,0]
  #   b=0,a=0 → 1; b=0,a=1 → 4
  #   b=1,a=0 → 2; b=1,a=1 → 5
  #   b=2,a=0 → 3; b=2,a=1 → 6
  # → expected: 1,4,2,5,3,6
  inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 3L, 1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 3L, 2L, 1L, 1L))
  node <- .onnx_node("Transpose", "X", "Y",
                      attrs = list(.onnx_attr_ints("perm", c(0L,2L,1L,3L,4L))))
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq_len(6) * 1.0
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(length(result), 6)
  expect_equal(result, c(1, 4, 2, 5, 3, 6), tolerance = 1e-5)
})

# ── Scalar tensors ───────────────────────────────────────────────

test_that("scalar Add: [1] + [1]", {
  path <- .onnx_make_binary("Add", c(1L))
  result <- run_onnx(path, list(A = 3.0, B = 7.0))
  expect_equal(as.numeric(result), 10.0, tolerance = 1e-5)
})

test_that("scalar Mul: [1] * [1]", {
  path <- .onnx_make_binary("Mul", c(1L))
  result <- run_onnx(path, list(A = 6.0, B = 7.0))
  expect_equal(as.numeric(result), 42.0, tolerance = 1e-5)
})

test_that("scalar Relu: negative → 0", {
  path <- .onnx_make_unary("Relu", c(1L))
  result <- run_onnx(path, list(X = -5.0))
  expect_equal(as.numeric(result), 0.0, tolerance = 1e-5)
})

test_that("scalar Relu: positive → same", {
  path <- .onnx_make_unary("Relu", c(1L))
  result <- run_onnx(path, list(X = 3.14))
  expect_equal(as.numeric(result), 3.14, tolerance = 1e-4)
})

test_that("scalar Sigmoid: 0 → 0.5", {
  path <- .onnx_make_unary("Sigmoid", c(1L))
  result <- run_onnx(path, list(X = 0.0))
  expect_equal(as.numeric(result), 0.5, tolerance = 1e-5)
})

# ── Broadcast rank-1 vs rank-N ───────────────────────────────────

test_that("broadcast Add: [4] + [1] scalar", {
  path <- .onnx_make_binary("Add", c(4L), c(1L))
  result <- run_onnx(path, list(A = c(1, 2, 3, 4), B = c(10.0)))
  expect_equal(as.numeric(result), c(11, 12, 13, 14), tolerance = 1e-5)
})

test_that("broadcast Mul: [2,4] * [4] row broadcast", {
  inp_a <- .onnx_value_info("A", 1L, c(2L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(4L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 4L))
  node  <- .onnx_node("Mul", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 2, 3, 4, 5, 6, 7, 8)
  b <- c(2, 3, 4, 5)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(a[1:4] * b, a[5:8] * b), tolerance = 1e-5)
})

test_that("broadcast Add: [3,1,4] + [3,2,4] middle-dim broadcast", {
  inp_a <- .onnx_value_info("A", 1L, c(3L, 1L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(3L, 2L, 4L))
  outp  <- .onnx_value_info("Y", 1L, c(3L, 2L, 4L))
  node  <- .onnx_node("Add", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- rep(1.0, 12)
  b <- seq_len(24) * 1.0
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(length(result), 24)
  expect_equal(as.numeric(result), b + 1.0, tolerance = 1e-5)
})

test_that("broadcast Sub: [1,4] - [2,4] (smaller minus larger)", {
  # A=[1,4] broadcasts to [2,4]: both rows of output = a - b_row_i
  # Row-major layout: b[1:4] is row 0, b[5:8] is row 1
  inp_a <- .onnx_value_info("A", 1L, c(1L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(2L, 4L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 4L))
  node  <- .onnx_node("Sub", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(10, 20, 30, 40)
  b <- c(1, 2, 3, 4, 5, 6, 7, 8)
  result <- run_onnx(path, list(A = a, B = b))
  # Output row 0: a - b[1:4]; row 1: a - b[5:8]
  expected <- c(a - b[1:4], a - b[5:8])
  expect_equal(as.numeric(result), expected, tolerance = 1e-5)
})

# ── Negative axis normalisation ──────────────────────────────────

test_that("Softmax axis=-1 equals axis=1 for [2,4]", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L))

  mk_path <- function(axis) {
    node <- .onnx_node("Softmax", "X", "Y",
                        attrs = list(.onnx_attr_int("axis", axis)))
    graph <- .onnx_graph("test", list(node), list(inp), list(outp))
    path <- tempfile(fileext = ".onnx")
    writeBin(.onnx_model(graph), path)
    path
  }

  x <- c(1, 2, 3, 4, 5, 6, 7, 8)
  r_neg <- run_onnx(mk_path(-1L), list(X = x))
  r_pos <- run_onnx(mk_path(1L),  list(X = x))
  expect_equal(as.numeric(r_neg), as.numeric(r_pos), tolerance = 1e-5)
})

test_that("Flatten axis=-1 flattens all but last dim of [2,3,4]", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(6L, 4L))
  node <- .onnx_node("Flatten", "X", "Y",
                      attrs = list(.onnx_attr_int("axis", -1L)))
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq_len(24) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
  expect_equal(sum(result), sum(x), tolerance = 1e-3)
})

test_that("ReduceMean axis=-1 equals axis=last for [2,3,4]", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L, 4L))
  outp_neg <- .onnx_value_info("Y", 1L, c(2L, 3L))
  outp_pos <- .onnx_value_info("Y", 1L, c(2L, 3L))

  mk_path <- function(axis, outp) {
    node <- .onnx_node("ReduceMean", "X", "Y",
                        attrs = list(.onnx_attr_int("axis", axis),
                                     .onnx_attr_int("keepdims", 0L)))
    graph <- .onnx_graph("test", list(node), list(inp), list(outp))
    path <- tempfile(fileext = ".onnx")
    writeBin(.onnx_model(graph), path)
    path
  }

  x <- seq_len(24) * 1.0
  r_neg <- run_onnx(mk_path(-1L, outp_neg), list(X = x))
  r_pos <- run_onnx(mk_path(2L,  outp_pos), list(X = x))
  expect_equal(as.numeric(r_neg), as.numeric(r_pos), tolerance = 1e-5)
})

# ── Large batch / many trailing 1s ───────────────────────────────

test_that("large batch [1,1,1,8]: Add passes through correctly", {
  inp_a <- .onnx_value_info("A", 1L, c(1L, 1L, 1L, 8L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 1L, 1L, 8L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 1L, 1L, 8L))
  node  <- .onnx_node("Add", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- seq_len(8) * 1.0
  b <- rep(10, 8)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a + b, tolerance = 1e-5)
})

test_that("large batch [1,1,4,4]: Relu preserves non-neg values", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))
  node <- .onnx_node("Relu", "X", "Y")
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-4:-1, 1:12) * 1.0
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})
