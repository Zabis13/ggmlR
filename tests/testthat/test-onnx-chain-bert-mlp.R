# Chain tests: BERT MLP / FFN patterns
# MatMul → Add(bias) → Gelu → MatMul → Add(residual)
#
# Covers: Gelu, Add (bias + residual)

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): MatMul → Gelu ──────────────────────────

test_that("chain bert-mlp: MatMul→Gelu (minimal)", {
  inp <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L))

  w_data <- rep(0.5, 12)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(3L, 4L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(3L, 4L))

  mm_node   <- .onnx_node("MatMul", c("X", "W"), "mm")
  gelu_node <- .onnx_node("Gelu", "mm", "Y")

  graph <- .onnx_graph("test", list(mm_node, gelu_node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 0, -1, 0.5, 0.5, 0.5)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  expect_true(all(is.finite(r)))
  # Gelu is approximately x * sigmoid(1.702*x), so positive inputs stay positive
  expect_true(all(r >= 0))
})

# ── Real (5 ops): MatMul → Add(bias) → Gelu → MatMul → Add(residual) ──

test_that("chain bert-mlp: MatMul→Add→Gelu→MatMul→Add (full FFN)", {
  # Input: [2, 4], up-project to [2, 8], Gelu, down-project to [2, 4], residual

  inp <- .onnx_value_info("X", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L))

  # Up-projection [4, 8]
  set.seed(10)
  w1_data <- rnorm(32, 0, 0.3)
  w1_raw <- unlist(lapply(w1_data, .float_bytes))
  w1_t  <- .onnx_tensor("W1", c(4L, 8L), 1L, w1_raw)
  w1_vi <- .onnx_value_info("W1", 1L, c(4L, 8L))

  # Bias [8]
  b1_data <- rep(0.1, 8)
  b1_raw <- unlist(lapply(b1_data, .float_bytes))
  b1_t  <- .onnx_tensor("B1", c(8L), 1L, b1_raw)
  b1_vi <- .onnx_value_info("B1", 1L, c(8L))

  # Down-projection [8, 4]
  w2_data <- rnorm(32, 0, 0.3)
  w2_raw <- unlist(lapply(w2_data, .float_bytes))
  w2_t  <- .onnx_tensor("W2", c(8L, 4L), 1L, w2_raw)
  w2_vi <- .onnx_value_info("W2", 1L, c(8L, 4L))

  mm1_node  <- .onnx_node("MatMul", c("X", "W1"), "mm1")
  add1_node <- .onnx_node("Add", c("mm1", "B1"), "biased")
  gelu_node <- .onnx_node("Gelu", "biased", "act")
  mm2_node  <- .onnx_node("MatMul", c("act", "W2"), "mm2")
  add2_node <- .onnx_node("Add", c("X", "mm2"), "Y")  # residual

  graph <- .onnx_graph("test",
                        list(mm1_node, add1_node, gelu_node, mm2_node, add2_node),
                        list(inp, w1_vi, b1_vi, w2_vi),
                        list(outp),
                        list(w1_t, b1_t, w2_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(8, -1, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  expect_true(all(is.finite(r)))
})

# ── Boundary: single element ────────────────────────────────

test_that("chain bert-mlp: single element (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L))

  w_raw <- .float_bytes(2.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L))

  mm_node   <- .onnx_node("MatMul", c("X", "W"), "mm")
  gelu_node <- .onnx_node("Gelu", "mm", "Y")

  graph <- .onnx_graph("test", list(mm_node, gelu_node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # Gelu(2*1) = Gelu(2) ≈ 1.9545
  result <- run_onnx(path, list(X = c(1.0)))
  r <- as.numeric(result)
  expect_equal(length(r), 1)
  expect_true(r[1] > 1.9 && r[1] < 2.0)
})
