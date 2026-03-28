# Chain tests: Object detection patterns
# Conv → Sigmoid → Reshape → Squeeze
#
# Tests shape propagation through Conv, 4D→2D reshape, squeeze.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}


# ── Minimal (2 ops): Conv → Sigmoid ──────────────────────────

test_that("chain detect: Conv→Sigmoid (minimal)", {
  # Input: [1, 1, 3, 3], Conv 1→1, kernel 1x1 → [1, 1, 3, 3], Sigmoid
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 3L, 3L))

  w_raw <- .float_bytes(2.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 1L, 1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "conv_out",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  sig_node  <- .onnx_node("Sigmoid", "conv_out", "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, sig_node),
                        list(inp, w_vi), list(outp),
                        list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0, 0.5, 1, -1, 0, 1, -0.5, 0, 0.5)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 9)
  # Sigmoid output in (0, 1)
  expect_true(all(r > 0 & r < 1))
  # sigmoid(2*0) = 0.5
  expect_equal(r[1], 0.5, tolerance = 1e-4)
})


# ── Real (4 ops): Conv → Sigmoid → Reshape → Squeeze ────────

test_that("chain detect: Conv→Sigmoid→Reshape→Squeeze (detection head)", {
  # Input: [1, 1, 4, 4]
  # Conv 1→2, kernel 3x3 → [1, 2, 2, 2]
  # Sigmoid → [1, 2, 2, 2] (confidence scores)
  # Reshape → [1, 8]  (flatten spatial + channels)
  # Squeeze axis=0 → [8]  (remove batch)

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(8L))

  # Conv [2, 1, 3, 3]
  w_data <- rep(0.5, 18)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 1L, 3L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 1L, 3L, 3L))

  # Reshape target shape: [1, 8]
  shape_raw <- c(writeBin(1L, raw(), size = 8, endian = "little"),
                 writeBin(8L, raw(), size = 8, endian = "little"))
  shape_t  <- .onnx_tensor("shape", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(2L))

  # Squeeze axes
  axes_raw <- writeBin(0L, raw(), size = 8, endian = "little")
  axes_t  <- .onnx_tensor("axes", c(1L), 7L, axes_raw)
  axes_vi <- .onnx_value_info("axes", 7L, c(1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "conv_out",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(3L, 3L))))
  sig_node  <- .onnx_node("Sigmoid", "conv_out", "sig_out")
  resh_node <- .onnx_node("Reshape", c("sig_out", "shape"), "resh_out")
  sq_node   <- .onnx_node("Squeeze", c("resh_out", "axes"), "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, sig_node, resh_node, sq_node),
                        list(inp, w_vi, shape_vi, axes_vi),
                        list(outp),
                        list(w_t, shape_t, axes_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(1, 16, by = 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  # All values are sigmoid outputs → [0, 1]
  expect_true(all(r >= 0 & r <= 1))
})


# ── Boundary: single-pixel, multi-channel ────────────────────

test_that("chain detect: 1x1 spatial, 4 channels (boundary)", {
  # Input: [1, 2, 1, 1]
  # Conv 2→4, kernel 1x1 → [1, 4, 1, 1]
  # Sigmoid → [1, 4, 1, 1]
  # Reshape → [4]

  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  # Conv [4, 2, 1, 1]
  w_data <- c(1, 0, 0, 1, -1, 0, 0, -1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(4L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(4L, 2L, 1L, 1L))

  # Reshape shape: [4]
  shape_raw <- writeBin(4L, raw(), size = 8, endian = "little")
  shape_t  <- .onnx_tensor("shape", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "conv_out",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  sig_node  <- .onnx_node("Sigmoid", "conv_out", "sig_out")
  resh_node <- .onnx_node("Reshape", c("sig_out", "shape"), "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, sig_node, resh_node),
                        list(inp, w_vi, shape_vi), list(outp),
                        list(w_t, shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1.0, 2.0)  # 2 channels, 1x1 spatial
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # Conv: ch0=1*1+0*2=1, ch1=0*1+1*2=2, ch2=-1*1+0*2=-1, ch3=0*1+(-1)*2=-2
  # Sigmoid: σ(1), σ(2), σ(-1), σ(-2)
  expected <- 1 / (1 + exp(-c(1, 2, -1, -2)))
  expect_equal(r, expected, tolerance = 1e-3)
})
