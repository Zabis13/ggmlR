# Chain tests: ConvTranspose (decoder/upsampling) patterns
# ConvTranspose → BatchNorm → LeakyRelu → ConvTranspose
#
# Covers: ConvTranspose

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): ConvTranspose → Relu ────────────────────

test_that("chain convtranspose: ConvTranspose→Relu (minimal)", {
  # Input: [1, 2, 2, 2] → ConvTranspose 2→2, 2x2, stride 2 → [1, 2, 4, 4] → Relu
  # Need 2+ channels so ggml_n_dims(weight) > 2 (avoids 1D branch)
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  # ConvTranspose weight: [in_ch, out_ch, kH, kW] = [2, 2, 2, 2]
  w_data <- rep(0.5, 16)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 2L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 2L, 2L))

  ct_node   <- .onnx_node("ConvTranspose", c("X", "W"), "ct",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                        .onnx_attr_ints("strides", c(2L, 2L))))
  relu_node <- .onnx_node("Relu", "ct", "Y")

  graph <- .onnx_graph("test", list(ct_node, relu_node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, -1, 2, -2, 0.5, 1, -0.5, -1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 32)
  expect_true(all(r >= 0))  # Relu
})

# ── Real (4 ops): ConvTranspose → BN → LeakyRelu → ConvTranspose ──

test_that("chain convtranspose: CT→BN→LeakyRelu→CT (GAN decoder)", {
  # Input: [1, 2, 2, 2]
  # ConvTranspose 2→2, 2x2, stride 2 → [1, 2, 4, 4]
  # BatchNorm → [1, 2, 4, 4]
  # LeakyRelu → [1, 2, 4, 4]
  # ConvTranspose 2→1, 2x2, stride 2 → [1, 1, 8, 8]

  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 8L, 8L))

  # CT1: [2, 2, 2, 2]
  w1_data <- rep(0.5, 16)
  w1_raw <- unlist(lapply(w1_data, .float_bytes))
  w1_t  <- .onnx_tensor("W1", c(2L, 2L, 2L, 2L), 1L, w1_raw)
  w1_vi <- .onnx_value_info("W1", 1L, c(2L, 2L, 2L, 2L))

  # BN params for 2 channels
  ones2  <- unlist(lapply(rep(1.0, 2), .float_bytes))
  zeros2 <- unlist(lapply(rep(0.0, 2), .float_bytes))
  sc_t <- .onnx_tensor("sc", c(2L), 1L, ones2)
  bi_t <- .onnx_tensor("bi", c(2L), 1L, zeros2)
  mn_t <- .onnx_tensor("mn", c(2L), 1L, zeros2)
  vr_t <- .onnx_tensor("vr", c(2L), 1L, ones2)
  sc_vi <- .onnx_value_info("sc", 1L, c(2L))
  bi_vi <- .onnx_value_info("bi", 1L, c(2L))
  mn_vi <- .onnx_value_info("mn", 1L, c(2L))
  vr_vi <- .onnx_value_info("vr", 1L, c(2L))

  # CT2: [2, 1, 2, 2]
  w2_data <- rep(0.25, 8)
  w2_raw <- unlist(lapply(w2_data, .float_bytes))
  w2_t  <- .onnx_tensor("W2", c(2L, 1L, 2L, 2L), 1L, w2_raw)
  w2_vi <- .onnx_value_info("W2", 1L, c(2L, 1L, 2L, 2L))

  ct1_node <- .onnx_node("ConvTranspose", c("X", "W1"), "ct1",
                          attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                       .onnx_attr_ints("strides", c(2L, 2L))))
  bn_node  <- .onnx_node("BatchNormalization",
                          c("ct1", "sc", "bi", "mn", "vr"), "bn")
  lr_node  <- .onnx_node("LeakyRelu", "bn", "lr",
                          attrs = list(.onnx_attr_float("alpha", 0.2)))
  ct2_node <- .onnx_node("ConvTranspose", c("lr", "W2"), "Y",
                          attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                       .onnx_attr_ints("strides", c(2L, 2L))))

  graph <- .onnx_graph("test",
                        list(ct1_node, bn_node, lr_node, ct2_node),
                        list(inp, w1_vi, sc_vi, bi_vi, mn_vi, vr_vi, w2_vi),
                        list(outp),
                        list(w1_t, sc_t, bi_t, mn_t, vr_t, w2_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(8, -1, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 64)
  expect_true(all(is.finite(r)))
})

# ── Boundary: 1x1 input ─────────────────────────────────────

test_that("chain convtranspose: 1x1 → 2x2 (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))

  w_data <- c(1, 2, 3, 4)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(1L, 1L, 2L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L, 2L))

  ct_node <- .onnx_node("ConvTranspose", c("X", "W"), "Y",
                         attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                      .onnx_attr_ints("strides", c(2L, 2L))))

  graph <- .onnx_graph("test", list(ct_node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # X=2.0, weight=[1,2,3,4] → output = 2*[1,2,3,4] = [2,4,6,8]
  result <- run_onnx(path, list(X = c(2.0)))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_equal(sort(r), c(2, 4, 6, 8), tolerance = 1e-3)
})
