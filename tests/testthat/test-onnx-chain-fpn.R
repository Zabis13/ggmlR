# Chain tests: Feature Pyramid Network (FPN) patterns
# Conv → Resize → Concat → Conv
#
# Covers: Resize/Upsample, Concat

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Conv → Resize ──────────────────────────

test_that("chain fpn: Conv→Resize (minimal)", {
  # Input: [1, 1, 2, 2], Conv 1→1, 1x1 → [1, 1, 2, 2], Resize 2x → [1, 1, 4, 4]
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))

  w_raw <- .float_bytes(1.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 1L, 1L))

  # Resize scales: [1.0, 1.0, 2.0, 2.0] (N, C, H, W)
  scales_data <- c(1.0, 1.0, 2.0, 2.0)
  scales_raw <- unlist(lapply(scales_data, .float_bytes))
  scales_t  <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))

  # Empty roi tensor (required by Resize but unused for "nearest")
  roi_t  <- .onnx_tensor("roi", c(0L), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, integer(0))

  conv_node   <- .onnx_node("Conv", c("X", "W"), "conv_out",
                             attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  resize_node <- .onnx_node("Resize", c("conv_out", "roi", "scales"), "Y",
                             attrs = list())

  graph <- .onnx_graph("test",
                        list(conv_node, resize_node),
                        list(inp, w_vi, roi_vi, scales_vi),
                        list(outp),
                        list(w_t, roi_t, scales_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 16)
  # Nearest upsample 2x: each pixel duplicated to 2x2 block
  expect_true(all(is.finite(r)))
})

# ── Real (4 ops): Conv → Resize → Concat → Conv ─────────────

test_that("chain fpn: Conv→Resize→Concat→Conv (FPN merge)", {
  # Two branches:
  # Branch A: X[1,1,2,2] → Conv 1→2, 1x1 → [1,2,2,2] → Resize 2x → [1,2,4,4]
  # Branch B: X[1,1,2,2] → Conv 1→2, 1x1 → [1,2,2,2] → Resize 2x → [1,2,4,4]
  # Concat(A, B, axis=1) → [1,4,4,4]
  # Conv 4→1, 1x1 → [1,1,4,4]

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))

  # Conv A: [2, 1, 1, 1]
  wa_data <- c(1.0, 0.5)
  wa_raw <- unlist(lapply(wa_data, .float_bytes))
  wa_t  <- .onnx_tensor("WA", c(2L, 1L, 1L, 1L), 1L, wa_raw)
  wa_vi <- .onnx_value_info("WA", 1L, c(2L, 1L, 1L, 1L))

  # Conv B: [2, 1, 1, 1]
  wb_data <- c(0.5, 1.0)
  wb_raw <- unlist(lapply(wb_data, .float_bytes))
  wb_t  <- .onnx_tensor("WB", c(2L, 1L, 1L, 1L), 1L, wb_raw)
  wb_vi <- .onnx_value_info("WB", 1L, c(2L, 1L, 1L, 1L))

  # Conv merge: [1, 4, 1, 1]
  wm_data <- rep(0.25, 4)
  wm_raw <- unlist(lapply(wm_data, .float_bytes))
  wm_t  <- .onnx_tensor("WM", c(1L, 4L, 1L, 1L), 1L, wm_raw)
  wm_vi <- .onnx_value_info("WM", 1L, c(1L, 4L, 1L, 1L))

  # Scales
  scales_data <- c(1.0, 1.0, 2.0, 2.0)
  scales_raw <- unlist(lapply(scales_data, .float_bytes))
  scales_t  <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))
  roi_t  <- .onnx_tensor("roi", c(0L), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, integer(0))

  conv_a  <- .onnx_node("Conv", c("X", "WA"), "ca",
                         attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  conv_b  <- .onnx_node("Conv", c("X", "WB"), "cb",
                         attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  res_a   <- .onnx_node("Resize", c("ca", "roi", "scales"), "ra")
  res_b   <- .onnx_node("Resize", c("cb", "roi", "scales"), "rb")
  cat_node <- .onnx_node("Concat", c("ra", "rb"), "cat",
                          attrs = list(.onnx_attr_int("axis", 1L)))
  conv_m  <- .onnx_node("Conv", c("cat", "WM"), "Y",
                         attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))

  graph <- .onnx_graph("test",
                        list(conv_a, conv_b, res_a, res_b, cat_node, conv_m),
                        list(inp, wa_vi, wb_vi, wm_vi, roi_vi, scales_vi),
                        list(outp),
                        list(wa_t, wb_t, wm_t, roi_t, scales_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 16)
  expect_true(all(is.finite(r)))
})

# ── Boundary: 1x1 spatial resize ─────────────────────────────

test_that("chain fpn: 1x1 spatial resize to 2x2 (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))

  scales_data <- c(1.0, 1.0, 2.0, 2.0)
  scales_raw <- unlist(lapply(scales_data, .float_bytes))
  scales_t  <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))
  roi_t  <- .onnx_tensor("roi", c(0L), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, integer(0))

  resize_node <- .onnx_node("Resize", c("X", "roi", "scales"), "Y")

  graph <- .onnx_graph("test", list(resize_node),
                        list(inp, roi_vi, scales_vi), list(outp),
                        list(roi_t, scales_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(5.0)))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # Nearest: all 4 pixels = 5.0
  expect_equal(r, rep(5.0, 4), tolerance = 1e-4)
})
