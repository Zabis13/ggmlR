# Tests for ONNX QuantizeLinear, DequantizeLinear, QLinearConv,
# QLinearMatMul, QLinearAdd, QLinearConcat, QLinearSigmoid

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Helper: create a scalar float initializer
.make_scalar_f32 <- function(name, value) {
  raw <- .float_bytes(value)
  list(tensor = .onnx_tensor(name, c(1L), 1L, raw),
       vi     = .onnx_value_info(name, 1L, c(1L)))
}

# A zero_point stored with its real ONNX dtype (2 = UINT8).  The spec ties the
# quantised type to this input's dtype, and that is what decides the saturation
# range, so a zero_point written as FLOAT32 only reaches the right range by
# falling through quant_bounds()' default -- it tests the fallback, not the op.
.make_zp_uint8 <- function(name, value = 0L) {
  list(tensor = .onnx_tensor(name, c(1L), 2L, as.raw(as.integer(value))),
       vi     = .onnx_value_info(name, 2L, c(1L)))
}

# ── DequantizeLinear ───────────────────────────────────────────

test_that("ONNX DequantizeLinear per-tensor works", {
  # y = (x - zp) * scale
  # x=[0, 1, 2, 4], scale=2.0, zp=1 → [-2, 0, 2, 6]
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  sc <- .make_scalar_f32("scale", 2.0)
  zp <- .make_scalar_f32("zp", 1.0)

  node <- .onnx_node("DequantizeLinear", c("X", "scale", "zp"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, sc$vi, zp$vi), list(outp),
                        list(sc$tensor, zp$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(0, 1, 2, 4)))
  expect_equal(as.numeric(result), c(-2, 0, 2, 6), tolerance = 1e-3)
})

test_that("ONNX DequantizeLinear no zero point works", {
  # y = x * scale (no zp)
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  sc <- .make_scalar_f32("scale", 0.5)

  node <- .onnx_node("DequantizeLinear", c("X", "scale"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, sc$vi), list(outp),
                        list(sc$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(2, 4, 6, 8)))
  expect_equal(as.numeric(result), c(1, 2, 3, 4), tolerance = 1e-3)
})

# ── QuantizeLinear ─────────────────────────────────────────────

test_that("ONNX QuantizeLinear per-tensor works", {
  # y = round(x / scale) + zp  (we skip rounding in ggml, so test with exact values)
  # x=[0, 2, 4, 6], scale=2.0, zp=1 → [0/2+1, 2/2+1, 4/2+1, 6/2+1] = [1, 2, 3, 4]
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  sc <- .make_scalar_f32("scale", 2.0)
  zp <- .make_scalar_f32("zp", 1.0)

  node <- .onnx_node("QuantizeLinear", c("X", "scale", "zp"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, sc$vi, zp$vi), list(outp),
                        list(sc$tensor, zp$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(0, 2, 4, 6)))
  expect_equal(as.numeric(result), c(1, 2, 3, 4), tolerance = 1e-3)
})

# ── QLinearSigmoid ─────────────────────────────────────────────

test_that("ONNX QLinearSigmoid works", {
  # Dequant → Sigmoid → Requant.  The output of a QLinear* op lives in the
  # quantised domain, so the expectation is the uint8 code, not the float
  # sigmoid: y = clamp(round(sigmoid(dequant(x)) / y_scale) + y_zp, 0, 255).
  #
  # y_scale=1 would collapse the whole (0,1) range of sigmoid onto {0,1} and
  # the test would read as if the op had "returned sigmoid in F32" whenever
  # rounding happened to land right.  1/255 is what a real model uses and it
  # keeps every sample distinguishable across the low, middle and saturated
  # parts of the curve.
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  xs <- .make_scalar_f32("x_scale", 1.0)
  xz <- .make_scalar_f32("x_zp", 0.0)
  ys <- .make_scalar_f32("y_scale", 1.0 / 255.0)
  yz <- .make_zp_uint8("y_zp", 0L)

  node <- .onnx_node("QLinearSigmoid",
                      c("X", "x_scale", "x_zp", "y_scale", "y_zp"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, xs$vi, xz$vi, ys$vi, yz$vi), list(outp),
                        list(xs$tensor, xz$tensor, ys$tensor, yz$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0, 10, -10, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # sigmoid: 0.5, 0.99995, 4.54e-5, 0.7311 → ×255 → 127.5, 254.99, 0.0116, 186.4
  expect_equal(r, round(1 / (1 + exp(-x)) * 255), tolerance = 0.51)
  # The discretisation itself: distinct inputs stay distinct codes, and the
  # saturated ends land on the range boundaries.
  expect_equal(r[2], 255)
  expect_equal(r[3], 0)
  expect_true(r[1] > r[3] && r[4] > r[1] && r[2] > r[4])
  # Dequantising the code recovers sigmoid to within one quantisation step.
  expect_equal(r * (1 / 255), 1 / (1 + exp(-x)), tolerance = 1 / 255)
})

# ── QLinearAdd ─────────────────────────────────────────────────

test_that("ONNX QLinearAdd works", {
  # a=[1,2,3,4] with scale=1,zp=0 + b=[10,20,30,40] with scale=1,zp=0
  # y_scale=1, y_zp=0 → output = [11, 22, 33, 44]
  inp_a <- .onnx_value_info("A", 1L, c(4L))
  inp_b <- .onnx_value_info("B", 1L, c(4L))
  outp  <- .onnx_value_info("Y", 1L, c(4L))

  as_ <- .make_scalar_f32("a_scale", 1.0)
  az  <- .make_scalar_f32("a_zp", 0.0)
  bs  <- .make_scalar_f32("b_scale", 1.0)
  bz  <- .make_scalar_f32("b_zp", 0.0)
  ys  <- .make_scalar_f32("y_scale", 1.0)
  yz  <- .make_scalar_f32("y_zp", 0.0)

  node <- .onnx_node("QLinearAdd",
                      c("A", "a_scale", "a_zp", "B", "b_scale", "b_zp",
                        "y_scale", "y_zp"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp_a, as_$vi, az$vi, inp_b, bs$vi, bz$vi,
                             ys$vi, yz$vi),
                        list(outp),
                        list(as_$tensor, az$tensor, bs$tensor, bz$tensor,
                             ys$tensor, yz$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(A = c(1,2,3,4), B = c(10,20,30,40)))
  expect_equal(as.numeric(result), c(11, 22, 33, 44), tolerance = 1e-3)
})

# ── QLinearMatMul ──────────────────────────────────────────────

test_that("ONNX QLinearMatMul works", {
  # A[2,3] @ B[3,2], all scale=1, zp=0 → same as regular matmul
  inp_a <- .onnx_value_info("A", 1L, c(2L, 3L))
  inp_b <- .onnx_value_info("B", 1L, c(3L, 2L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 2L))

  as_ <- .make_scalar_f32("a_scale", 1.0)
  az  <- .make_scalar_f32("a_zp", 0.0)
  bs  <- .make_scalar_f32("b_scale", 1.0)
  bz  <- .make_scalar_f32("b_zp", 0.0)
  ys  <- .make_scalar_f32("y_scale", 1.0)
  yz  <- .make_scalar_f32("y_zp", 0.0)

  node <- .onnx_node("QLinearMatMul",
                      c("A", "a_scale", "a_zp", "B", "b_scale", "b_zp",
                        "y_scale", "y_zp"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp_a, as_$vi, az$vi, inp_b, bs$vi, bz$vi,
                             ys$vi, yz$vi),
                        list(outp),
                        list(as_$tensor, az$tensor, bs$tensor, bz$tensor,
                             ys$tensor, yz$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # A=[[1,2,3],[4,5,6]], B=[[1,0],[0,1],[1,1]]
  # A@B = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
  a <- c(1,2,3, 4,5,6)
  b <- c(1,0, 0,1, 1,1)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(4, 5, 10, 11), tolerance = 1e-3)
})

# ── QLinearConv ────────────────────────────────────────────────

test_that("ONNX QLinearConv 1D works", {
  # Simple 1D conv: x[1,1,4], w[1,1,2], scale=1, zp=0
  inp   <- .onnx_value_info("X", 1L, c(1L, 1L, 4L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 1L, 3L))

  xs <- .make_scalar_f32("x_scale", 1.0)
  xz <- .make_scalar_f32("x_zp", 0.0)
  ys <- .make_scalar_f32("y_scale", 1.0)
  yz <- .make_scalar_f32("y_zp", 0.0)
  ws <- .make_scalar_f32("w_scale", 1.0)
  wz <- .make_scalar_f32("w_zp", 0.0)

  # weight [1,1,2] = [1,1]
  w_raw <- unlist(lapply(c(1, 1), .float_bytes))
  w_t  <- .onnx_tensor("W", c(1L, 1L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L))

  node <- .onnx_node("QLinearConv",
                      c("X", "x_scale", "x_zp", "W", "w_scale", "w_zp",
                        "y_scale", "y_zp"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, xs$vi, xz$vi, w_vi, ws$vi, wz$vi,
                             ys$vi, yz$vi),
                        list(outp),
                        list(xs$tensor, xz$tensor, w_t, ws$tensor, wz$tensor,
                             ys$tensor, yz$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # x=[1,2,3,4], w=[1,1] → conv1d = [3,5,7]
  result <- run_onnx(path, list(X = c(1, 2, 3, 4)))
  expect_equal(as.numeric(result), c(3, 5, 7), tolerance = 1e-3)
})

# ── QLinearConcat ──────────────────────────────────────────────

test_that("ONNX QLinearConcat axis=0 works", {
  # Concat two [2] vectors → [4], all scale=1, zp=0
  inp_a <- .onnx_value_info("A", 1L, c(2L))
  inp_b <- .onnx_value_info("B", 1L, c(2L))
  outp  <- .onnx_value_info("Y", 1L, c(4L))

  ys  <- .make_scalar_f32("y_scale", 1.0)
  yz  <- .make_scalar_f32("y_zp", 0.0)
  as_ <- .make_scalar_f32("a_scale", 1.0)
  az  <- .make_scalar_f32("a_zp", 0.0)
  bs  <- .make_scalar_f32("b_scale", 1.0)
  bz  <- .make_scalar_f32("b_zp", 0.0)

  attrs <- list(.onnx_attr_int("axis", 0L))
  # QLinearConcat inputs: y_scale, y_zp, x1, x1_scale, x1_zp, x2, x2_scale, x2_zp
  node <- .onnx_node("QLinearConcat",
                      c("y_scale", "y_zp", "A", "a_scale", "a_zp",
                        "B", "b_scale", "b_zp"), "Y",
                      attrs = attrs)
  graph <- .onnx_graph("test", list(node),
                        list(ys$vi, yz$vi, inp_a, as_$vi, az$vi,
                             inp_b, bs$vi, bz$vi),
                        list(outp),
                        list(ys$tensor, yz$tensor, as_$tensor, az$tensor,
                             bs$tensor, bz$tensor))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(A = c(1, 2), B = c(3, 4)))
  expect_equal(as.numeric(result), c(1, 2, 3, 4), tolerance = 1e-3)
})
