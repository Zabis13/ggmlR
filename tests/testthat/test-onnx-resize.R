# Tests for ONNX Resize, Upsample, ConvTranspose

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Resize ───────────────────────────────────────────────────────

test_that("ONNX Resize nearest with scales works", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))
  roi_t  <- .onnx_tensor("roi", c(0L), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, c(0L))
  scales_raw <- unlist(lapply(c(1, 1, 2, 2), .float_bytes))
  scales_t  <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))
  node <- .onnx_node("Resize", c("X", "roi", "scales"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, roi_vi, scales_vi), list(outp),
                        list(roi_t, scales_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 16)
  expect_equal(result[1], 1, tolerance = 1e-5)
  expect_equal(result[16], 4, tolerance = 1e-5)
})

test_that("ONNX Resize nearest with sizes works", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 6L))
  roi_t  <- .onnx_tensor("roi", c(0L), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, c(0L))
  scales_t  <- .onnx_tensor("scales", c(0L), 1L, raw(0))
  scales_vi <- .onnx_value_info("scales", 1L, c(0L))
  sizes_raw <- c(.int64_bytes(1L), .int64_bytes(1L),
                 .int64_bytes(4L), .int64_bytes(6L))
  sizes_t  <- .onnx_tensor("sizes", c(4L), 7L, sizes_raw)
  sizes_vi <- .onnx_value_info("sizes", 7L, c(4L))
  node <- .onnx_node("Resize", c("X", "roi", "scales", "sizes"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, roi_vi, scales_vi, sizes_vi), list(outp),
                        list(roi_t, scales_t, sizes_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
})

# ── ConvTranspose ────────────────────────────────────────────────

test_that("ONNX ConvTranspose 1D works", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 6L))
  w_data <- rep(1.0, 2)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(1L, 1L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L))
  node <- .onnx_node("ConvTranspose", c("X", "W"), "Y",
                      attrs = list(.onnx_attr_ints("strides", c(2L))))
  graph <- .onnx_graph("test", list(node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
  expect_equal(as.numeric(result), c(1, 1, 2, 2, 3, 3), tolerance = 1e-5)
})
