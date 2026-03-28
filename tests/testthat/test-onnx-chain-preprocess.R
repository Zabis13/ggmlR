# Chain tests: Image preprocessing patterns
# Sub(mean) → Div(std) → Pad → Conv
#
# Covers: Sub, Div, Pad

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Sub → Div ──────────────────────────────

test_that("chain preprocess: Sub→Div (normalize)", {
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  mean_raw <- .float_bytes(2.0)
  mean_t  <- .onnx_tensor("mean", c(1L), 1L, mean_raw)
  mean_vi <- .onnx_value_info("mean", 1L, c(1L))

  std_raw <- .float_bytes(0.5)
  std_t  <- .onnx_tensor("std", c(1L), 1L, std_raw)
  std_vi <- .onnx_value_info("std", 1L, c(1L))

  sub_node <- .onnx_node("Sub", c("X", "mean"), "centered")
  div_node <- .onnx_node("Div", c("centered", "std"), "Y")

  graph <- .onnx_graph("test", list(sub_node, div_node),
                        list(inp, mean_vi, std_vi), list(outp),
                        list(mean_t, std_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expected <- (x - 2.0) / 0.5
  expect_equal(r, expected, tolerance = 1e-4)
})

# ── Real (4 ops): Sub → Div → Pad → Conv ────────────────────

test_that("chain preprocess: Sub→Div→Pad→Conv (image pipeline)", {
  # Input: [1, 1, 2, 2]
  # Sub mean → Div std → Pad(1,1,1,1) → [1, 1, 4, 4] → Conv 1→1, 3x3 → [1, 1, 2, 2]

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))

  mean_raw <- .float_bytes(0.5)
  mean_t  <- .onnx_tensor("mean", c(1L), 1L, mean_raw)
  mean_vi <- .onnx_value_info("mean", 1L, c(1L))

  std_raw <- .float_bytes(0.25)
  std_t  <- .onnx_tensor("std", c(1L), 1L, std_raw)
  std_vi <- .onnx_value_info("std", 1L, c(1L))

  # Pad: [0,0,1,1, 0,0,1,1] → pad H and W by 1 on each side
  # ONNX pads format: [dim0_begin, dim1_begin, dim2_begin, dim3_begin,
  #                     dim0_end, dim1_end, dim2_end, dim3_end]
  pads_data <- c(0L, 0L, 1L, 1L, 0L, 0L, 1L, 1L)
  pads_raw <- raw(0)
  for (p in pads_data) pads_raw <- c(pads_raw, writeBin(as.integer(p), raw(), size = 8, endian = "little"))
  pads_t  <- .onnx_tensor("pads", c(8L), 7L, pads_raw)
  pads_vi <- .onnx_value_info("pads", 7L, c(8L))

  # Conv: [1, 1, 3, 3]
  w_data <- rep(1.0 / 9, 9)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(1L, 1L, 3L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 3L, 3L))

  sub_node  <- .onnx_node("Sub", c("X", "mean"), "s")
  div_node  <- .onnx_node("Div", c("s", "std"), "d")
  pad_node  <- .onnx_node("Pad", c("d", "pads"), "p")
  conv_node <- .onnx_node("Conv", c("p", "W"), "Y",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(3L, 3L))))

  graph <- .onnx_graph("test",
                        list(sub_node, div_node, pad_node, conv_node),
                        list(inp, mean_vi, std_vi, pads_vi, w_vi),
                        list(outp),
                        list(mean_t, std_t, pads_t, w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0.1, 0.3, 0.5, 0.7)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(all(is.finite(r)))
})

# ── Boundary: zero std → large values ────────────────────────

test_that("chain preprocess: large scale (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(2L))
  outp <- .onnx_value_info("Y", 1L, c(2L))

  mean_raw <- .float_bytes(0.0)
  mean_t  <- .onnx_tensor("mean", c(1L), 1L, mean_raw)
  mean_vi <- .onnx_value_info("mean", 1L, c(1L))

  # Very small std → large output
  std_raw <- .float_bytes(0.001)
  std_t  <- .onnx_tensor("std", c(1L), 1L, std_raw)
  std_vi <- .onnx_value_info("std", 1L, c(1L))

  sub_node <- .onnx_node("Sub", c("X", "mean"), "s")
  div_node <- .onnx_node("Div", c("s", "std"), "Y")

  graph <- .onnx_graph("test", list(sub_node, div_node),
                        list(inp, mean_vi, std_vi), list(outp),
                        list(mean_t, std_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(1.0, -1.0)))
  r <- as.numeric(result)
  expect_equal(r, c(1000, -1000), tolerance = 1)
})
