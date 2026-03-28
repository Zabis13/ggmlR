# Chain tests: Decoder / MLP patterns
# Gemm → Tanh → Gemm → Elu → Log
#
# Covers: Gemm, Tanh, Elu, Log

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Gemm → Tanh ────────────────────────────

test_that("chain decoder: Gemm→Tanh (minimal)", {
  # Gemm: X[2,3] @ W[3,2] + bias[2] → [2,2], then Tanh
  inp <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 2L))

  w_data <- rep(0.5, 6)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(3L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(3L, 2L))

  b_data <- c(0.1, -0.1)
  b_raw <- unlist(lapply(b_data, .float_bytes))
  b_t  <- .onnx_tensor("B", c(2L), 1L, b_raw)
  b_vi <- .onnx_value_info("B", 1L, c(2L))

  gemm_node <- .onnx_node("Gemm", c("X", "W", "B"), "g",
                           attrs = list(.onnx_attr_int("transA", 0L),
                                        .onnx_attr_int("transB", 0L)))
  tanh_node <- .onnx_node("Tanh", "g", "Y")

  graph <- .onnx_graph("test", list(gemm_node, tanh_node),
                        list(inp, w_vi, b_vi), list(outp),
                        list(w_t, b_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 0, -1, 0.5, 0.5, 0)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # Tanh output in (-1, 1)
  expect_true(all(r > -1 & r < 1))
})

# ── Real (5 ops): Gemm → Tanh → Gemm → Elu → Abs ───────────

test_that("chain decoder: Gemm→Tanh→Gemm→Elu→Abs (MLP decoder)", {
  # X[2,3] → Gemm(W1[3,4]+b1) → Tanh → Gemm(W2[4,2]+b2) → Elu → Abs

  inp <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 2L))

  set.seed(77)
  w1_data <- rnorm(12, 0, 0.5)
  w1_raw <- unlist(lapply(w1_data, .float_bytes))
  w1_t  <- .onnx_tensor("W1", c(3L, 4L), 1L, w1_raw)
  w1_vi <- .onnx_value_info("W1", 1L, c(3L, 4L))

  b1_data <- rep(0.0, 4)
  b1_raw <- unlist(lapply(b1_data, .float_bytes))
  b1_t  <- .onnx_tensor("B1", c(4L), 1L, b1_raw)
  b1_vi <- .onnx_value_info("B1", 1L, c(4L))

  w2_data <- rnorm(8, 0, 0.5)
  w2_raw <- unlist(lapply(w2_data, .float_bytes))
  w2_t  <- .onnx_tensor("W2", c(4L, 2L), 1L, w2_raw)
  w2_vi <- .onnx_value_info("W2", 1L, c(4L, 2L))

  b2_data <- c(0.1, 0.2)
  b2_raw <- unlist(lapply(b2_data, .float_bytes))
  b2_t  <- .onnx_tensor("B2", c(2L), 1L, b2_raw)
  b2_vi <- .onnx_value_info("B2", 1L, c(2L))

  gemm1_node <- .onnx_node("Gemm", c("X", "W1", "B1"), "g1",
                            attrs = list(.onnx_attr_int("transA", 0L),
                                         .onnx_attr_int("transB", 0L)))
  tanh_node  <- .onnx_node("Tanh", "g1", "t1")
  gemm2_node <- .onnx_node("Gemm", c("t1", "W2", "B2"), "g2",
                            attrs = list(.onnx_attr_int("transA", 0L),
                                         .onnx_attr_int("transB", 0L)))
  elu_node   <- .onnx_node("Elu", "g2", "e1")
  abs_node   <- .onnx_node("Abs", "e1", "Y")

  graph <- .onnx_graph("test",
                        list(gemm1_node, tanh_node, gemm2_node, elu_node, abs_node),
                        list(inp, w1_vi, b1_vi, w2_vi, b2_vi),
                        list(outp),
                        list(w1_t, b1_t, w2_t, b2_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(6, -1, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(all(r >= 0))  # Abs guarantees non-negative
  expect_true(all(is.finite(r)))
})

# ── Boundary: Gemm with transB ───────────────────────────────

test_that("chain decoder: Gemm transB (boundary)", {
  # X[1,3] @ W^T[3,2] where W stored as [2,3]
  inp <- .onnx_value_info("X", 1L, c(1L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L))

  # W stored as [2,3], transB=1 means X @ W^T = X[1,3] @ W[2,3]^T → [1,2]
  w_data <- c(1, 0, 0, 0, 1, 0)  # identity-like
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 3L))

  gemm_node <- .onnx_node("Gemm", c("X", "W"), "g",
                           attrs = list(.onnx_attr_int("transA", 0L),
                                        .onnx_attr_int("transB", 1L)))
  tanh_node <- .onnx_node("Tanh", "g", "Y")

  graph <- .onnx_graph("test", list(gemm_node, tanh_node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # X = [1, 2, 3], W^T first row = [1,0,0] → dot=1, second row = [0,1,0] → dot=2
  result <- run_onnx(path, list(X = c(1, 2, 3)))
  r <- as.numeric(result)
  expect_equal(length(r), 2)
  expect_equal(r, tanh(c(1, 2)), tolerance = 1e-3)
})
