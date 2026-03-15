# Tests for ONNX GroupNormalization, RMSNormalization

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── GroupNormalization ─────────────────────────────────────────

test_that("ONNX GroupNormalization 1D works", {
  # X[1,4], num_groups=2, scale=[1,1,1,1], bias=[0,0,0,0]
  # With identity scale/bias, output should be group-normalized
  inp <- .onnx_value_info("X", 1L, c(1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  scale_raw <- unlist(lapply(rep(1.0, 4), .float_bytes))
  scale_t <- .onnx_tensor("scale", c(4L), 1L, scale_raw)
  scale_vi <- .onnx_value_info("scale", 1L, c(4L))

  bias_raw <- unlist(lapply(rep(0.0, 4), .float_bytes))
  bias_t <- .onnx_tensor("bias", c(4L), 1L, bias_raw)
  bias_vi <- .onnx_value_info("bias", 1L, c(4L))

  attrs <- list(.onnx_attr_int("num_groups", 2L))
  node <- .onnx_node("GroupNormalization", c("X", "scale", "bias"), "Y",
                      attrs = attrs)
  graph <- .onnx_graph("test", list(node),
                        list(inp, scale_vi, bias_vi), list(outp),
                        list(scale_t, bias_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 3, 2, 6)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # GroupNorm output should be normalized (mean ~ 0 within groups)
  # Exact values depend on ggml group_norm implementation
  expect_true(all(is.finite(r)))
  # Overall sum should be near 0 for identity scale/zero bias
  expect_equal(sum(r), 0, tolerance = 0.5)
})

test_that("ONNX GroupNormalization 4D works", {
  # X[1,2,2,2] (NCHW), num_groups=2, each channel is own group
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 2L, 2L))

  scale_raw <- unlist(lapply(rep(1.0, 2), .float_bytes))
  scale_t <- .onnx_tensor("scale", c(2L), 1L, scale_raw)
  scale_vi <- .onnx_value_info("scale", 1L, c(2L))

  bias_raw <- unlist(lapply(rep(0.0, 2), .float_bytes))
  bias_t <- .onnx_tensor("bias", c(2L), 1L, bias_raw)
  bias_vi <- .onnx_value_info("bias", 1L, c(2L))

  attrs <- list(.onnx_attr_int("num_groups", 2L))
  node <- .onnx_node("GroupNormalization", c("X", "scale", "bias"), "Y",
                      attrs = attrs)
  graph <- .onnx_graph("test", list(node),
                        list(inp, scale_vi, bias_vi), list(outp),
                        list(scale_t, bias_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # 8 elements: channel0=[1,2,3,4], channel1=[5,6,7,8]
  x <- c(1, 2, 3, 4, 5, 6, 7, 8)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  # Each channel normalized to mean=0: sum of each group ~ 0
  expect_equal(sum(r[1:4]), 0, tolerance = 0.1)
  expect_equal(sum(r[5:8]), 0, tolerance = 0.1)
})

# ── RMSNormalization ───────────────────────────────────────────

test_that("ONNX RMSNormalization works", {
  # RMSNorm: x / sqrt(mean(x^2) + eps)
  path <- .onnx_make_unary("RMSNormalization", c(4L))
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # RMS = sqrt(mean(1+4+9+16)) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
  rms <- sqrt(mean(x^2))
  expected <- x / rms
  expect_equal(r, expected, tolerance = 0.01)
})

test_that("ONNX RMSNormalization uniform input works", {
  path <- .onnx_make_unary("RMSNormalization", c(4L))
  x <- c(3, 3, 3, 3)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # RMS of uniform = |x|, so output ≈ sign(x) * 1
  expect_equal(r, rep(1, 4), tolerance = 0.01)
})
