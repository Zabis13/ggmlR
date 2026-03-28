# Chain tests: QLinearConv
# Quantized convolution: dequantize inputs → conv → requantize output.
# Covers: QLinearConv 2D, with bias, chain with activation, 3x3, boundary.
#
# Note: ggml needs C_in >= 2 for conv_2d to see 4D weight. All tests use
# input [1, 2, 4, 4] and weight [C_out, 2, kH, kW] to avoid scalar collapse.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Helper: make scalar F32 tensor initializer
make_scalar <- function(name, val) {
  raw <- .float_bytes(val)
  t  <- .onnx_tensor(name, c(1L), 1L, raw)
  vi <- .onnx_value_info(name, 1L, c(1L))
  list(t = t, vi = vi)
}

# ── Minimal: QLinearConv 1x1, scale=1 zp=0 (identity-like) ──

test_that("chain qlinearconv: QLinearConv 1x1 (minimal)", {
  # Input [1,2,4,4], weight [2,2,1,1] = identity across channels
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  x_sc <- make_scalar("x_sc", 1.0)
  x_zp <- make_scalar("x_zp", 0.0)

  # Weight: 2x2 identity kernel [C_out=2, C_in=2, 1, 1]
  w_data <- c(1, 0, 0, 1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 1L, 1L))
  w_sc <- make_scalar("w_sc", 1.0)
  w_zp <- make_scalar("w_zp", 0.0)

  y_sc <- make_scalar("y_sc", 1.0)
  y_zp <- make_scalar("y_zp", 0.0)

  qconv_node <- .onnx_node("QLinearConv",
    c("X", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp"), "Y",
    attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))

  graph <- .onnx_graph("test", list(qconv_node),
    list(inp, x_sc$vi, x_zp$vi, w_vi, w_sc$vi, w_zp$vi, y_sc$vi, y_zp$vi),
    list(outp),
    list(x_sc$t, x_zp$t, w_t, w_sc$t, w_zp$t, y_sc$t, y_zp$t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # 2 channels x 16 pixels = 32 values
  x <- as.numeric(1:32)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # Identity conv: output = input
  expect_equal(r, x, tolerance = 1e-3)
})

# ── QLinearConv with scale and zero-point ──────────────────

test_that("chain qlinearconv: QLinearConv with dequant/requant", {
  # x_scale=0.1, x_zp=128 → dequant, identity conv, requant back
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  x_sc <- make_scalar("x_sc", 0.1)
  x_zp <- make_scalar("x_zp", 128.0)

  w_data <- c(1, 0, 0, 1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 1L, 1L))
  w_sc <- make_scalar("w_sc", 1.0)
  w_zp <- make_scalar("w_zp", 0.0)

  y_sc <- make_scalar("y_sc", 0.1)
  y_zp <- make_scalar("y_zp", 128.0)

  qconv_node <- .onnx_node("QLinearConv",
    c("X", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp"), "Y",
    attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))

  graph <- .onnx_graph("test", list(qconv_node),
    list(inp, x_sc$vi, x_zp$vi, w_vi, w_sc$vi, w_zp$vi, y_sc$vi, y_zp$vi),
    list(outp),
    list(x_sc$t, x_zp$t, w_t, w_sc$t, w_zp$t, y_sc$t, y_zp$t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # All 138: dequant (138-128)*0.1=1.0, identity conv, requant 1.0/0.1+128=138
  x <- rep(138, 32)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(r, rep(138, 32), tolerance = 0.5)
})

# ── QLinearConv with bias ──────────────────────────────────

test_that("chain qlinearconv: QLinearConv with bias", {
  # Identity conv + bias=10 per channel, scale=1 zp=0
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  x_sc <- make_scalar("x_sc", 1.0)
  x_zp <- make_scalar("x_zp", 0.0)

  w_data <- c(1, 0, 0, 1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 1L, 1L))
  w_sc <- make_scalar("w_sc", 1.0)
  w_zp <- make_scalar("w_zp", 0.0)

  y_sc <- make_scalar("y_sc", 1.0)
  y_zp <- make_scalar("y_zp", 0.0)

  # Bias: [10.0, 10.0] for 2 output channels
  bias_data <- c(10, 10)
  bias_raw <- unlist(lapply(bias_data, .float_bytes))
  bias_t  <- .onnx_tensor("B", c(2L), 1L, bias_raw)
  bias_vi <- .onnx_value_info("B", 1L, c(2L))

  qconv_node <- .onnx_node("QLinearConv",
    c("X", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp", "B"), "Y",
    attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))

  graph <- .onnx_graph("test", list(qconv_node),
    list(inp, x_sc$vi, x_zp$vi, w_vi, w_sc$vi, w_zp$vi,
         y_sc$vi, y_zp$vi, bias_vi),
    list(outp),
    list(x_sc$t, x_zp$t, w_t, w_sc$t, w_zp$t, y_sc$t, y_zp$t, bias_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(3, 32)  # identity conv: 3 + bias 10 = 13
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(r, rep(13, 32), tolerance = 1e-2)
})

# ── QLinearConv → Relu chain ───────────────────────────────

test_that("chain qlinearconv: QLinearConv→Relu (activation chain)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  x_sc <- make_scalar("x_sc", 0.1)
  x_zp <- make_scalar("x_zp", 128.0)

  w_data <- c(1, 0, 0, 1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 1L, 1L))
  w_sc <- make_scalar("w_sc", 1.0)
  w_zp <- make_scalar("w_zp", 0.0)

  y_sc <- make_scalar("y_sc", 1.0)
  y_zp <- make_scalar("y_zp", 0.0)

  qconv_node <- .onnx_node("QLinearConv",
    c("X", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp"), "qc",
    attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  relu_node <- .onnx_node("Relu", "qc", "Y")

  graph <- .onnx_graph("test", list(qconv_node, relu_node),
    list(inp, x_sc$vi, x_zp$vi, w_vi, w_sc$vi, w_zp$vi, y_sc$vi, y_zp$vi),
    list(outp),
    list(x_sc$t, x_zp$t, w_t, w_sc$t, w_zp$t, y_sc$t, y_zp$t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # Alternate below/above zp: 118→-1.0, 138→1.0
  x <- rep(c(118, 138), 16)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # dequant→identity conv→relu: -1→0, 1→1
  expected <- rep(c(0, 1), 16)
  expect_equal(r, expected, tolerance = 1e-2)
})

# ── QLinearConv 3x3 with padding ───────────────────────────

test_that("chain qlinearconv: QLinearConv 3x3 same padding", {
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  x_sc <- make_scalar("x_sc", 1.0)
  x_zp <- make_scalar("x_zp", 0.0)

  # 3x3 identity-like: each output channel uses only its own input channel
  # W shape [2, 2, 3, 3] = 36 values
  # channel 0→0: all 1s, channel 1→0: all 0s, channel 0→1: all 0s, channel 1→1: all 1s
  w_data <- c(rep(1, 9), rep(0, 9),   # out_ch=0: in_ch=0 all 1, in_ch=1 all 0
              rep(0, 9), rep(1, 9))    # out_ch=1: in_ch=0 all 0, in_ch=1 all 1
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 3L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 3L, 3L))
  w_sc <- make_scalar("w_sc", 1.0)
  w_zp <- make_scalar("w_zp", 0.0)
  y_sc <- make_scalar("y_sc", 1.0)
  y_zp <- make_scalar("y_zp", 0.0)

  qconv_node <- .onnx_node("QLinearConv",
    c("X", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp"), "Y",
    attrs = list(
      .onnx_attr_ints("kernel_shape", c(3L, 3L)),
      .onnx_attr_string("auto_pad", "SAME_UPPER")))

  graph <- .onnx_graph("test", list(qconv_node),
    list(inp, x_sc$vi, x_zp$vi, w_vi, w_sc$vi, w_zp$vi, y_sc$vi, y_zp$vi),
    list(outp),
    list(x_sc$t, x_zp$t, w_t, w_sc$t, w_zp$t, y_sc$t, y_zp$t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(1, 32)  # all ones, box filter
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 32)
  expect_true(all(is.finite(r)))
  expect_true(all(r > 0))
})

# ── Boundary: QLinearConv all-zero input ───────────────────

test_that("chain qlinearconv: all-zero input (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  x_sc <- make_scalar("x_sc", 0.1)
  x_zp <- make_scalar("x_zp", 0.0)

  w_data <- c(1, 0, 0, 1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 1L, 1L))
  w_sc <- make_scalar("w_sc", 1.0)
  w_zp <- make_scalar("w_zp", 0.0)
  y_sc <- make_scalar("y_sc", 1.0)
  y_zp <- make_scalar("y_zp", 0.0)

  qconv_node <- .onnx_node("QLinearConv",
    c("X", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp"), "Y",
    attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))

  graph <- .onnx_graph("test", list(qconv_node),
    list(inp, x_sc$vi, x_zp$vi, w_vi, w_sc$vi, w_zp$vi, y_sc$vi, y_zp$vi),
    list(outp),
    list(x_sc$t, x_zp$t, w_t, w_sc$t, w_zp$t, y_sc$t, y_zp$t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(0, 32)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(r, rep(0, 32), tolerance = 1e-5)
})
