# Tests for ONNX Gelu, EyeLike, ConstantOfShape, Shape, Upsample

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Gelu ───────────────────────────────────────────────────────

test_that("ONNX Gelu works", {
  path <- .onnx_make_unary("Gelu", c(4L))
  x <- c(-1, 0, 0.5, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # gelu(0) = 0
  expect_equal(r[2], 0, tolerance = 1e-5)
  # gelu(-1) < 0 but > -1
  expect_true(r[1] < 0 && r[1] > -1)
  # gelu(0.5) ~ 0.345, gelu(1) ~ 0.841
  expect_true(r[3] > 0.3 && r[3] < 0.4)
  expect_true(r[4] > 0.8 && r[4] < 0.9)
})

# ── EyeLike ────────────────────────────────────────────────────

test_that("ONNX EyeLike 3x3 identity works", {
  # EyeLike output is a leaf (set_input), must be consumed by a compute op
  # to get buffer allocated. Use Add(EyeLike(X), X) then subtract X.
  inp <- .onnx_value_info("X", 1L, c(3L, 3L))
  outp <- .onnx_value_info("Z", 1L, c(3L, 3L))
  eye_node <- .onnx_node("EyeLike", "X", "Y")
  add_node <- .onnx_node("Add", c("Y", "X"), "Z")
  graph <- .onnx_graph("test", list(eye_node, add_node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(0, 9)
  result <- run_onnx(path, list(X = x))
  # EyeLike + zeros = identity matrix
  expect_equal(as.numeric(result), c(1,0,0, 0,1,0, 0,0,1), tolerance = 1e-5)
})

test_that("ONNX EyeLike 2x4 rectangular works", {
  inp <- .onnx_value_info("X", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Z", 1L, c(2L, 4L))
  eye_node <- .onnx_node("EyeLike", "X", "Y")
  add_node <- .onnx_node("Add", c("Y", "X"), "Z")
  graph <- .onnx_graph("test", list(eye_node, add_node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(0, 8)
  result <- run_onnx(path, list(X = x))
  # 2x4 eye: [[1,0,0,0],[0,1,0,0]]
  expect_equal(as.numeric(result), c(1,0,0,0, 0,1,0,0), tolerance = 1e-5)
})

# ── ConstantOfShape ────────────────────────────────────────────

test_that("ONNX ConstantOfShape zeros works", {
  # shape=[4] → 4-element tensor filled with 0, then Add with X to force compute
  inp <- .onnx_value_info("X", 1L, c(4L))

  shape_raw <- .int64_bytes(4L)
  shape_t <- .onnx_tensor("shape", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(1L))

  outp <- .onnx_value_info("Z", 1L, c(4L))
  cos_node <- .onnx_node("ConstantOfShape", "shape", "C")
  add_node <- .onnx_node("Add", c("X", "C"), "Z")
  graph <- .onnx_graph("test", list(cos_node, add_node),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # ConstantOfShape fills with 0, so X + 0 = X
  result <- run_onnx(path, list(X = c(1, 2, 3, 4)))
  expect_equal(as.numeric(result), c(1, 2, 3, 4), tolerance = 1e-5)
})

test_that("ONNX ConstantOfShape with fill value works", {
  # shape=[4] → 4-element tensor filled with 7.0, Add with X to force compute
  inp <- .onnx_value_info("X", 1L, c(4L))

  shape_raw <- .int64_bytes(4L)
  shape_t <- .onnx_tensor("shape", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(1L))

  # value attribute: TensorProto with scalar 7.0
  val_attr <- .onnx_attr_tensor("value", c(1L), 1L, .float_bytes(7.0))

  outp <- .onnx_value_info("Z", 1L, c(4L))
  cos_node <- .onnx_node("ConstantOfShape", "shape", "C", attrs = list(val_attr))
  add_node <- .onnx_node("Add", c("X", "C"), "Z")
  graph <- .onnx_graph("test", list(cos_node, add_node),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # ConstantOfShape fills with 7, so X + 7 = [8, 9, 10, 11]
  result <- run_onnx(path, list(X = c(1, 2, 3, 4)))
  expect_equal(as.numeric(result), c(8, 9, 10, 11), tolerance = 1e-5)
})

# ── Shape ──────────────────────────────────────────────────────

test_that("ONNX Shape is tested indirectly via Reshape cval pipeline", {
  # Shape op output is I32 compile-time tensor (not float).
  # It's consumed internally by Reshape/Expand/etc. via cval propagation.
  # Direct float output is not supported — tested via Reshape in test-onnx-shape.R.
  # Here: X[12] → Reshape[-1,3] → Y[4,3] — triggers Shape-like dim inference
  path <- .onnx_make_reshape(c(12L), c(-1L, 3L))
  x <- 1:12
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 12)
  expect_equal(as.numeric(result), as.numeric(1:12), tolerance = 1e-5)
})

# ── Upsample ──────────────────────────────────────────────────

test_that("ONNX Upsample nearest 2x works", {
  # X[1,1,2,2] → Upsample(scales=[1,1,2,2]) → Y[1,1,4,4]
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))

  scales_raw <- unlist(lapply(c(1, 1, 2, 2), .float_bytes))
  scales_t <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))

  # Upsample: inputs = [X, scales] (no roi)
  node <- .onnx_node("Upsample", c("X", "scales"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, scales_vi), list(outp),
                        list(scales_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 16)
  # First element should be 1, last should be 4
  expect_equal(result[1], 1, tolerance = 1e-5)
  expect_equal(result[16], 4, tolerance = 1e-5)
})
