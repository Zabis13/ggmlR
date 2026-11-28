# Resize with an EMPTY "scales" input and "sizes" known only at build time
# (from Shape) -- the U^2-Net / RMBG pattern `F.interpolate(x, size=tar.shape[2:])`.
# An empty scales tensor used to count as "target found" and Resize became a no-op.

test_that("Resize uses build-time sizes when scales is empty", {
  x_vi <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  t_vi <- .onnx_value_info("T", 1L, c(1L, 1L, 4L, 4L))      # size donor
  y_vi <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))
  scales_t  <- .onnx_tensor("scales", 0L, 1L, raw(0))        # empty float
  scales_vi <- .onnx_value_info("scales", 1L, 0L)

  shape_node  <- .onnx_node("Shape", "T", "sizes")
  resize_node <- .onnx_node("Resize", c("X", "", "scales", "sizes"), "Y")
  graph <- .onnx_graph("resize", list(shape_node, resize_node),
                       list(x_vi, t_vi, scales_vi), list(y_vi), list(scales_t))
  path <- tempfile(fileext = ".onnx")
  on.exit(unlink(path))
  writeBin(.onnx_model(graph), path)

  m <- onnx_load(path, device = "cpu")
  x <- c(1, 2, 3, 4)                                         # row-major 2x2
  y <- as.numeric(onnx_run(m, list(X = x, T = rep(0, 16)))[[1]])
  expect_length(y, 16)                                       # was 4 (no-op)
  # nearest 2x: every input value appears in a 2x2 block
  expect_equal(sort(unique(y)), x)
  expect_equal(sum(y), 4 * sum(x))
})
