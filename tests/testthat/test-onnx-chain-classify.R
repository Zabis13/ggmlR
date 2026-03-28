# Chain tests: Image classification patterns
# Conv → BatchNorm → Relu → GlobalAveragePool → Flatten → MatMul → Softmax
#
# Tests shape propagation through Conv+BN, ndims tracking, 4D→2D transitions.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Helper: build Conv+BN block with initializers
# Returns list(nodes, inputs_vi, initializers, output_name)
.make_conv_bn_block <- function(prefix, in_channels, out_channels,
                                 kH = 3L, kW = 3L,
                                 input_name = "X") {
  conv_out <- paste0(prefix, "_conv_out")
  bn_out   <- paste0(prefix, "_bn_out")
  w_name   <- paste0(prefix, "_W")
  b_name   <- paste0(prefix, "_B")
  sc_name  <- paste0(prefix, "_scale")
  bi_name  <- paste0(prefix, "_bias")
  mn_name  <- paste0(prefix, "_mean")
  vr_name  <- paste0(prefix, "_var")

  # Conv weights: [out_ch, in_ch, kH, kW]
  n_weights <- out_channels * in_channels * kH * kW
  w_data <- rep(1.0 / n_weights, n_weights)  # small uniform
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_dims <- c(out_channels, in_channels, kH, kW)
  w_t  <- .onnx_tensor(w_name, w_dims, 1L, w_raw)
  w_vi <- .onnx_value_info(w_name, 1L, w_dims)

  # Conv bias
  b_data <- rep(0.0, out_channels)
  b_raw <- unlist(lapply(b_data, .float_bytes))
  b_t  <- .onnx_tensor(b_name, c(out_channels), 1L, b_raw)
  b_vi <- .onnx_value_info(b_name, 1L, c(out_channels))

  # BN params: scale=1, bias=0, mean=0, var=1
  ones  <- unlist(lapply(rep(1.0, out_channels), .float_bytes))
  zeros <- unlist(lapply(rep(0.0, out_channels), .float_bytes))
  sc_t  <- .onnx_tensor(sc_name, c(out_channels), 1L, ones)
  bi_t  <- .onnx_tensor(bi_name, c(out_channels), 1L, zeros)
  mn_t  <- .onnx_tensor(mn_name, c(out_channels), 1L, zeros)
  vr_t  <- .onnx_tensor(vr_name, c(out_channels), 1L, ones)
  sc_vi <- .onnx_value_info(sc_name, 1L, c(out_channels))
  bi_vi <- .onnx_value_info(bi_name, 1L, c(out_channels))
  mn_vi <- .onnx_value_info(mn_name, 1L, c(out_channels))
  vr_vi <- .onnx_value_info(vr_name, 1L, c(out_channels))

  conv_node <- .onnx_node("Conv", c(input_name, w_name, b_name), conv_out,
                           attrs = list(.onnx_attr_ints("kernel_shape", c(kH, kW)),
                                        .onnx_attr_ints("pads", c(0L, 0L, 0L, 0L))))
  bn_node <- .onnx_node("BatchNormalization",
                          c(conv_out, sc_name, bi_name, mn_name, vr_name),
                          bn_out)

  list(
    nodes = list(conv_node, bn_node),
    inputs_vi = list(w_vi, b_vi, sc_vi, bi_vi, mn_vi, vr_vi),
    initializers = list(w_t, b_t, sc_t, bi_t, mn_t, vr_t),
    output_name = bn_out
  )
}


# ── Minimal (3 ops): Conv → Relu → GlobalAveragePool ──────────

test_that("chain classify: Conv → Relu → GAP (minimal)", {
  # Input: [1, 1, 4, 4], Conv 1→2 with 3x3 → [1, 2, 2, 2], Relu, GAP → [1, 2, 1, 1]
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 1L, 1L))

  # Conv weights [2, 1, 3, 3]
  w_data <- rep(1.0 / 9, 18)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 1L, 3L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 1L, 3L, 3L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "conv_out",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(3L, 3L))))
  relu_node <- .onnx_node("Relu", "conv_out", "relu_out")
  gap_node  <- .onnx_node("GlobalAveragePool", "relu_out", "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, relu_node, gap_node),
                        list(inp, w_vi), list(outp),
                        list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(1, 16, by = 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 2)
  # Conv averages 3x3 patches, Relu keeps positive, GAP averages spatial
  expect_true(all(r > 0))
})


# ── Real (7 ops): Conv → BN → Relu → GAP → Flatten → MatMul → Softmax ──

test_that("chain classify: Conv→BN→Relu→GAP→Flatten→MatMul→Softmax (full)", {
  # Input: [1, 1, 5, 5]
  # Conv 1→4, kernel 3x3 → [1, 4, 3, 3]
  # BN → [1, 4, 3, 3]
  # Relu → [1, 4, 3, 3]
  # GAP → [1, 4, 1, 1]
  # Flatten → [1, 4]
  # MatMul with W[4,3] → [1, 3]
  # Softmax → [1, 3]  (3-class classification)

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 5L, 5L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 3L))

  block <- .make_conv_bn_block("c1", 1L, 4L, 3L, 3L, "X")

  relu_node <- .onnx_node("Relu", block$output_name, "relu_out")
  gap_node  <- .onnx_node("GlobalAveragePool", "relu_out", "gap_out")
  flat_node <- .onnx_node("Flatten", "gap_out", "flat_out",
                           attrs = list(.onnx_attr_int("axis", 1L)))

  # FC weights: [4, 3] (ONNX: K=4, N=3)
  fc_data <- rep(0.5, 12)
  fc_raw <- unlist(lapply(fc_data, .float_bytes))
  fc_t  <- .onnx_tensor("FC_W", c(4L, 3L), 1L, fc_raw)
  fc_vi <- .onnx_value_info("FC_W", 1L, c(4L, 3L))

  mm_node <- .onnx_node("MatMul", c("flat_out", "FC_W"), "mm_out")
  sm_node <- .onnx_node("Softmax", "mm_out", "Y",
                          attrs = list(.onnx_attr_int("axis", 1L)))

  all_nodes <- c(block$nodes, list(relu_node, gap_node, flat_node,
                                    mm_node, sm_node))
  all_inputs <- c(list(inp), block$inputs_vi, list(fc_vi))
  all_inits  <- c(block$initializers, list(fc_t))

  graph <- .onnx_graph("test", all_nodes, all_inputs, list(outp), all_inits)
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(25, 0, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)

  # Should output 3 probabilities summing to 1
  expect_equal(length(r), 3)
  expect_true(all(r >= 0 & r <= 1))
  expect_equal(sum(r), 1.0, tolerance = 1e-4)
})


# ── Boundary: batch=2, single spatial pixel ───────────────────

test_that("chain classify: batch=2, 1x1 spatial (boundary)", {
  # Input: [2, 1, 1, 1] — minimal spatial, batch of 2
  # Conv 1→2, kernel 1x1 → [2, 2, 1, 1]
  # Relu → [2, 2, 1, 1]
  # GAP → [2, 2, 1, 1]
  # Flatten → [2, 2]
  # MatMul with [2, 3] → [2, 3]
  # Softmax → [2, 3]

  inp <- .onnx_value_info("X", 1L, c(2L, 1L, 1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))

  # Conv 1x1: [2, 1, 1, 1]
  w_data <- c(1.0, -1.0)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 1L, 1L, 1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "conv_out",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  relu_node <- .onnx_node("Relu", "conv_out", "relu_out")
  gap_node  <- .onnx_node("GlobalAveragePool", "relu_out", "gap_out")
  flat_node <- .onnx_node("Flatten", "gap_out", "flat_out",
                           attrs = list(.onnx_attr_int("axis", 1L)))

  fc_data <- rep(1.0, 6)
  fc_raw <- unlist(lapply(fc_data, .float_bytes))
  fc_t  <- .onnx_tensor("FC", c(2L, 3L), 1L, fc_raw)
  fc_vi <- .onnx_value_info("FC", 1L, c(2L, 3L))

  mm_node <- .onnx_node("MatMul", c("flat_out", "FC"), "mm_out")
  sm_node <- .onnx_node("Softmax", "mm_out", "Y",
                          attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
                        list(conv_node, relu_node, gap_node, flat_node,
                             mm_node, sm_node),
                        list(inp, w_vi, fc_vi), list(outp),
                        list(w_t, fc_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(3.0, -1.0)  # batch of 2, one pixel each
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)

  expect_equal(length(r), 6)  # 2 batches x 3 classes
  # All softmax outputs in [0,1], total sum = 2 (two batches)
  expect_true(all(r >= 0 & r <= 1))
  expect_equal(sum(r), 2.0, tolerance = 1e-3)
})
