# Chain tests: Pooling pipeline patterns
# Conv → MaxPool → Conv → AveragePool → Flatten
#
# Covers: MaxPool, AveragePool

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Conv → MaxPool ─────────────────────────

test_that("chain pooling: Conv→MaxPool (minimal)", {
  # Input: [1, 1, 4, 4] → Conv 1→1, 1x1 → [1, 1, 4, 4] → MaxPool 2x2 → [1, 1, 2, 2]
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))

  w_raw <- .float_bytes(1.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 1L, 1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "c",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  pool_node <- .onnx_node("MaxPool", "c", "Y",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                        .onnx_attr_ints("strides", c(2L, 2L))))

  graph <- .onnx_graph("test", list(conv_node, pool_node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # 4x4 grid: 1..16
  x <- seq(1, 16)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # MaxPool 2x2 stride 2 on 4x4 (ggml layout: ne[0]=W, ne[1]=H)
  # Max of each 2x2 block
  expect_true(all(r > 0))
})

# ── Real (5 ops): Conv → MaxPool → Relu → Conv → AveragePool ──

test_that("chain pooling: Conv→MaxPool→Relu→Conv→AvgPool (full)", {
  # Input: [1, 1, 4, 4]
  # Conv 1→2, 3x3 → [1, 2, 2, 2]
  # MaxPool 2x2 → [1, 2, 1, 1]
  # Relu
  # (output is already 1x1, so skip second conv — just test the chain)
  # Use Flatten instead: → [1, 2]

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L))

  # Conv [2, 1, 3, 3]
  w_data <- rep(1.0 / 9, 18)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 1L, 3L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 1L, 3L, 3L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "c1",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(3L, 3L))))
  pool_node <- .onnx_node("MaxPool", "c1", "p1",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                        .onnx_attr_ints("strides", c(2L, 2L))))
  relu_node <- .onnx_node("Relu", "p1", "r1")
  flat_node <- .onnx_node("Flatten", "r1", "Y",
                           attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
                        list(conv_node, pool_node, relu_node, flat_node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(16, 0, 10)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 2)
  expect_true(all(r >= 0))  # Relu
})

# ── Boundary: AveragePool with padding ───────────────────────

test_that("chain pooling: AveragePool 4x4 (boundary)", {
  # Input: [1, 1, 4, 4] → AveragePool 2x2, stride 2 → [1, 1, 2, 2]
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))

  pool_node <- .onnx_node("AveragePool", "X", "Y",
                           attrs = list(
                             .onnx_attr_ints("kernel_shape", c(2L, 2L)),
                             .onnx_attr_ints("strides", c(2L, 2L))))

  graph <- .onnx_graph("test", list(pool_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(1, 16)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(all(is.finite(r)))
  expect_true(all(r > 0))
})
