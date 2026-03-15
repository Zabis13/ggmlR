# Tests for ONNX neural network ops:
# Conv, BatchNorm, LayerNorm, MatMul, Gemm, Gather

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── MatMul ───────────────────────────────────────────────────────

test_that("ONNX MatMul works for 2D", {
  path <- .onnx_make_matmul(M = 2L, K = 3L, N = 2L)
  a <- c(1, 2, 3, 4, 5, 6)
  b <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(A = a, B = b))
  expected <- c(22, 28, 49, 64)
  expect_equal(as.numeric(result), expected, tolerance = 1e-3)
})

# ── LayerNormalization ───────────────────────────────────────────

test_that("ONNX LayerNormalization works (1D)", {
  path <- .onnx_make_layer_norm(c(4L), eps = 1e-5)
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(abs(mean(r)) < 0.01)
  pop_sd <- sqrt(mean((x - mean(x))^2))
  expected <- (x - mean(x)) / pop_sd
  expect_equal(r, expected, tolerance = 0.01)
})

# ── Conv 2D ────────────────────────────────────────────────────

test_that("ONNX Conv 2D basic works", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))
  w_data <- rep(1.0, 4)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(1L, 1L, 2L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L, 2L))
  node <- .onnx_node("Conv", c("X", "W"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(12, 16, 24, 28), tolerance = 1e-3)
})

test_that("ONNX Conv 2D with bias works", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))
  w_data <- rep(1.0, 4)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(1L, 1L, 2L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L, 2L))
  b_raw <- .float_bytes(10.0)
  b_t  <- .onnx_tensor("B", c(1L), 1L, b_raw)
  b_vi <- .onnx_value_info("B", 1L, c(1L))
  node <- .onnx_node("Conv", c("X", "W", "B"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, w_vi, b_vi), list(outp),
                        list(w_t, b_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(12, 16, 24, 28) + 10, tolerance = 1e-3)
})

# ── BatchNormalization ─────────────────────────────────────────

test_that("ONNX BatchNormalization 2D works", {
  inp <- .onnx_value_info("X", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L))
  ones  <- unlist(lapply(rep(1.0, 4), .float_bytes))
  zeros <- unlist(lapply(rep(0.0, 4), .float_bytes))
  scale_t <- .onnx_tensor("scale", c(4L), 1L, ones)
  bias_t  <- .onnx_tensor("bias",  c(4L), 1L, zeros)
  mean_t  <- .onnx_tensor("mean",  c(4L), 1L, zeros)
  var_t   <- .onnx_tensor("var",   c(4L), 1L, ones)
  scale_vi <- .onnx_value_info("scale", 1L, c(4L))
  bias_vi  <- .onnx_value_info("bias",  1L, c(4L))
  mean_vi  <- .onnx_value_info("mean",  1L, c(4L))
  var_vi   <- .onnx_value_info("var",   1L, c(4L))
  node <- .onnx_node("BatchNormalization",
                      c("X", "scale", "bias", "mean", "var"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, scale_vi, bias_vi, mean_vi, var_vi),
                        list(outp),
                        list(scale_t, bias_t, mean_t, var_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-3)
})

test_that("ONNX BatchNormalization 4D works", {
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 2L, 2L))
  scale_raw <- unlist(lapply(c(2.0, 1.0), .float_bytes))
  bias_raw  <- unlist(lapply(c(1.0, 0.0), .float_bytes))
  zeros2    <- unlist(lapply(rep(0.0, 2), .float_bytes))
  ones2     <- unlist(lapply(rep(1.0, 2), .float_bytes))
  scale_t <- .onnx_tensor("scale", c(2L), 1L, scale_raw)
  bias_t  <- .onnx_tensor("bias",  c(2L), 1L, bias_raw)
  mean_t  <- .onnx_tensor("mean",  c(2L), 1L, zeros2)
  var_t   <- .onnx_tensor("var",   c(2L), 1L, ones2)
  scale_vi <- .onnx_value_info("scale", 1L, c(2L))
  bias_vi  <- .onnx_value_info("bias",  1L, c(2L))
  mean_vi  <- .onnx_value_info("mean",  1L, c(2L))
  var_vi   <- .onnx_value_info("var",   1L, c(2L))
  node <- .onnx_node("BatchNormalization",
                      c("X", "scale", "bias", "mean", "var"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, scale_vi, bias_vi, mean_vi, var_vi),
                        list(outp),
                        list(scale_t, bias_t, mean_t, var_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(3, 5, 7, 9, 5, 6, 7, 8), tolerance = 1e-3)
})

# ── Gather ─────────────────────────────────────────────────────

test_that("ONNX Gather works (embedding lookup)", {
  inp_idx <- .onnx_value_info("I", 7L, c(2L))
  w_data <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
              0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(4L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(4L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))
  node <- .onnx_node("Gather", c("W", "I"), "Y",
                      attrs = list(.onnx_attr_int("axis", 0L)))
  graph <- .onnx_graph("test", list(node),
                        list(inp_idx, w_vi), list(outp), list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  indices <- c(0, 2)
  result <- run_onnx(path, list(I = indices))
  expect_equal(as.numeric(result), c(0.1, 0.2, 0.3, 0.7, 0.8, 0.9), tolerance = 1e-3)
})
