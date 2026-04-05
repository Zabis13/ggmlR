# Chain tests: Resize alignment + broadcast validation (MaskRCNN pattern)
#
# Tests the pattern that causes MaskRCNN-12-int8 to fail:
# Resize output spatial dims not divisible by subsequent Add/Mul input dims.
# ggml_repeat requires ne[d] % src_ne[d] == 0 for broadcast.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: Resize nearest 2x then Add with matching shape ──────

test_that("chain resize-broadcast: Resize nearest 2x + Add (aligned)", {
  # Input [1,1,7,7] → Resize nearest scales=[1,1,2,2] → [1,1,14,14] → Add bias[1,1,1,1]
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 7L, 7L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 14L, 14L))

  # Empty roi
  roi_raw <- raw(0)
  roi_t <- .onnx_tensor("roi", integer(0), 1L, roi_raw)
  roi_vi <- .onnx_value_info("roi", 1L, integer(0))

  # Scales [1,1,2,2]
  scales_raw <- unlist(lapply(c(1.0, 1.0, 2.0, 2.0), .float_bytes))
  scales_t <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))

  # Bias [1,1,1,1]
  bias_raw <- unlist(lapply(c(0.5), .float_bytes))
  bias_t <- .onnx_tensor("bias", c(1L, 1L, 1L, 1L), 1L, bias_raw)
  bias_vi <- .onnx_value_info("bias", 1L, c(1L, 1L, 1L, 1L))

  resize_node <- .onnx_node("Resize", c("X", "roi", "scales"), "resized",
                              attrs = list(.onnx_attr_int("mode", 1L)))  # nearest
  add_node <- .onnx_node("Add", c("resized", "bias"), "Y")

  graph <- .onnx_graph("test",
    list(resize_node, add_node),
    list(inp, roi_vi, scales_vi, bias_vi),
    list(outp),
    list(roi_t, scales_t, bias_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rnorm(49)  # 1*1*7*7
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 196)  # 14*14
  expect_true(all(is.finite(r)))
})

# ── Resize with non-integer scale (output_sizes) + Add ──────────

test_that("chain resize-broadcast: Resize sizes=[14,14] from [7,7] + Add", {
  # Input [1,1,7,7] → Resize sizes=[1,1,14,14] → Add with [1,1,14,14]
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 7L, 7L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 14L, 14L))

  roi_t <- .onnx_tensor("roi", integer(0), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, integer(0))

  # Empty scales (use sizes instead)
  scales_t <- .onnx_tensor("scales", integer(0), 1L, raw(0))
  scales_vi <- .onnx_value_info("scales", 1L, integer(0))

  # Target sizes [1,1,14,14] as INT64
  sizes_raw <- unlist(lapply(c(1L, 1L, 14L, 14L), .int64_bytes))
  sizes_t <- .onnx_tensor("sizes", c(4L), 7L, sizes_raw)
  sizes_vi <- .onnx_value_info("sizes", 7L, c(4L))

  # Second input for Add: [1,1,14,14]
  add_inp <- .onnx_value_info("B", 1L, c(1L, 1L, 14L, 14L))

  resize_node <- .onnx_node("Resize", c("X", "roi", "scales", "sizes"), "resized")
  add_node <- .onnx_node("Add", c("resized", "B"), "Y")

  graph <- .onnx_graph("test",
    list(resize_node, add_node),
    list(inp, roi_vi, scales_vi, sizes_vi, add_inp),
    list(outp),
    list(roi_t, scales_t, sizes_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rnorm(49)
  b <- rnorm(196)
  result <- run_onnx(path, list(X = x, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 196)
  expect_true(all(is.finite(r)))
})

# ── MaskRCNN pattern: Resize 7→14, then Add with 14×14 feature map ──

test_that("chain resize-broadcast: Resize 7→14 + Conv + Add (MaskRCNN head)", {
  # Simulates mask head: ROI features [1,C,7,7] → Resize [1,C,14,14] → Conv → Add skip
  # C=2 for speed
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 7L, 7L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 14L, 14L))

  roi_t <- .onnx_tensor("roi", integer(0), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, integer(0))

  scales_raw <- unlist(lapply(c(1.0, 1.0, 2.0, 2.0), .float_bytes))
  scales_t <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))

  # Conv 1x1: [2,2,1,1] weights
  w_data <- rep(0.5, 2*2*1*1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t <- .onnx_tensor("W", c(2L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 1L, 1L))

  # Skip connection input [1,2,14,14]
  skip <- .onnx_value_info("S", 1L, c(1L, 2L, 14L, 14L))

  resize_node <- .onnx_node("Resize", c("X", "roi", "scales"), "resized")
  conv_node <- .onnx_node("Conv", c("resized", "W"), "convout",
                            attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  add_node <- .onnx_node("Add", c("convout", "S"), "Y")

  graph <- .onnx_graph("test",
    list(resize_node, conv_node, add_node),
    list(inp, roi_vi, scales_vi, w_vi, skip),
    list(outp),
    list(roi_t, scales_t, w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rnorm(1*2*7*7)
  s <- rnorm(1*2*14*14)
  result <- run_onnx(path, list(X = x, S = s))
  r <- as.numeric(result)
  expect_equal(length(r), 1*2*14*14)
  expect_true(all(is.finite(r)))
})

# ── Boundary: non-divisible spatial broadcast (the actual MaskRCNN bug) ──

test_that("chain resize-broadcast: Add with non-divisible spatial dims 14x14 + 7x7 (expect error or pad)", {
  # This is the exact pattern that crashes MaskRCNN:
  # A[1,1,14,14] + B[1,1,7,7] → ggml_repeat fails because 14 % 7 == 0 ✓
  # BUT 7x7 + 14x14 where A is smaller fails because 14 % 7 == 0 but 7 % 14 != 0
  inp_a <- .onnx_value_info("A", 1L, c(1L, 1L, 14L, 14L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 1L, 7L, 7L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 1L, 14L, 14L))

  add_node <- .onnx_node("Add", c("A", "B"), "Y")

  graph <- .onnx_graph("test", list(add_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- rnorm(196)
  b <- rnorm(49)
  env <- environment()
  capture.output({
    env$result <- run_onnx(path, list(A = a, B = b))
  }, type = "message")
  r <- as.numeric(result)
  expect_equal(length(r), 196)
  expect_true(all(is.finite(r)))
})
