# Chain tests: MaskRCNN broadcast pattern
# Binary ops between tensors with mismatched spatial dims
# after DequantizeLinear in INT8 pipeline.
#
# Tests the pattern that causes MaskRCNN-12-int8 to fail:
# broadcast dim 0: a=14, b=7 — likely a Slice/Resize output
# feeding into Add/Mul with unexpected shape.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: Add with broadcast [1,C,H,W] + [1,C,1,1] ─────

test_that("chain maskrcnn-broadcast: Add 4D broadcast [1,2,3,3]+[1,2,1,1] (minimal)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L, 3L))

  # Bias: [1,2,1,1] — per-channel bias broadcast over spatial
  b_data <- c(10.0, 20.0)
  b_raw <- unlist(lapply(b_data, .float_bytes))
  b_t  <- .onnx_tensor("B", c(1L, 2L, 1L, 1L), 1L, b_raw)
  b_vi <- .onnx_value_info("B", 1L, c(1L, 2L, 1L, 1L))

  add_node <- .onnx_node("Add", c("X", "B"), "Y")

  graph <- .onnx_graph("test", list(add_node),
                        list(inp, b_vi), list(outp), list(b_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(1.0, 18)  # 1*2*3*3, all ones
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 18)
  expect_true(all(is.finite(r)))
  # Channel 0 should be 1+10=11, channel 1 should be 1+20=21
})

# ── Real: Conv→Slice→Add (FPN-like with spatial mismatch) ──

test_that("chain maskrcnn-broadcast: Conv→Slice→Add (spatial mismatch)", {
  # Simulates FPN: conv output [1,1,4,4], slice to [1,1,2,4], add with [1,1,2,4]
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  inp2 <- .onnx_value_info("X2", 1L, c(1L, 1L, 2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 4L))

  # 1x1 conv (identity)
  w_raw <- .float_bytes(1.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 1L, 1L))

  # Slice params: starts=[0], ends=[2], axes=[2] (ONNX H axis)
  starts_raw <- .int64_bytes(0L)
  starts_t  <- .onnx_tensor("st", c(1L), 7L, starts_raw)
  starts_vi <- .onnx_value_info("st", 7L, c(1L))

  ends_raw <- .int64_bytes(2L)
  ends_t  <- .onnx_tensor("en", c(1L), 7L, ends_raw)
  ends_vi <- .onnx_value_info("en", 7L, c(1L))

  axes_raw <- .int64_bytes(2L)
  axes_t  <- .onnx_tensor("ax", c(1L), 7L, axes_raw)
  axes_vi <- .onnx_value_info("ax", 7L, c(1L))

  conv_node  <- .onnx_node("Conv", c("X", "W"), "conv",
                             attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  slice_node <- .onnx_node("Slice", c("conv", "st", "en", "ax"), "sliced")
  add_node   <- .onnx_node("Add", c("sliced", "X2"), "Y")

  graph <- .onnx_graph("test",
    list(conv_node, slice_node, add_node),
    list(inp, inp2, w_vi, starts_vi, ends_vi, axes_vi),
    list(outp),
    list(w_t, starts_t, ends_t, axes_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(1, 16)
  x2 <- rep(100, 8)
  result <- run_onnx(path, list(X = x, X2 = x2))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  expect_true(all(is.finite(r)))
  expect_true(all(r > 90))  # sliced values + 100
})

# ── INT8 pattern: DequantizeLinear → Mul (broadcast) ────────

test_that("chain maskrcnn-broadcast: Dequant→Mul broadcast (INT8 pattern)", {
  # DequantizeLinear [1,2,3,3] → Mul with per-channel scale [1,2,1,1]
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L, 3L))

  dq_sc_raw <- .float_bytes(0.1)
  dq_sc_t  <- .onnx_tensor("dq_sc", c(1L), 1L, dq_sc_raw)
  dq_sc_vi <- .onnx_value_info("dq_sc", 1L, c(1L))

  dq_zp_raw <- .float_bytes(128.0)
  dq_zp_t  <- .onnx_tensor("dq_zp", c(1L), 1L, dq_zp_raw)
  dq_zp_vi <- .onnx_value_info("dq_zp", 1L, c(1L))

  # Per-channel multiplier [1,2,1,1]
  mul_data <- c(2.0, 3.0)
  mul_raw <- unlist(lapply(mul_data, .float_bytes))
  mul_t  <- .onnx_tensor("mul_sc", c(1L, 2L, 1L, 1L), 1L, mul_raw)
  mul_vi <- .onnx_value_info("mul_sc", 1L, c(1L, 2L, 1L, 1L))

  dq_node  <- .onnx_node("DequantizeLinear", c("X", "dq_sc", "dq_zp"), "dq")
  mul_node <- .onnx_node("Mul", c("dq", "mul_sc"), "Y")

  graph <- .onnx_graph("test", list(dq_node, mul_node),
    list(inp, dq_sc_vi, dq_zp_vi, mul_vi),
    list(outp),
    list(dq_sc_t, dq_zp_t, mul_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # All 128 → dequant = (128-128)*0.1 = 0 → mul = 0
  x <- rep(128, 18)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 18)
  expect_true(all(abs(r) < 1e-3))
})

# ── Boundary: Mul with [N,H,W] x [N,1,1] broadcast ────────

test_that("chain maskrcnn-broadcast: 3D spatial broadcast [2,3,3]x[2,1,1] (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(2L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L, 3L))

  sc_data <- c(2.0, 0.5)
  sc_raw <- unlist(lapply(sc_data, .float_bytes))
  sc_t  <- .onnx_tensor("S", c(2L, 1L, 1L), 1L, sc_raw)
  sc_vi <- .onnx_value_info("S", 1L, c(2L, 1L, 1L))

  mul_node <- .onnx_node("Mul", c("X", "S"), "Y")

  graph <- .onnx_graph("test", list(mul_node),
                        list(inp, sc_vi), list(outp), list(sc_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(1.0, 18)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 18)
  expect_true(all(is.finite(r)))
})
