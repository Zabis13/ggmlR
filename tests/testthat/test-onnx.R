# Tests for ONNX model loading and inference
#
# Uses .onnx_make_* helpers from tests/testthat/helper-onnx.R to generate
# minimal protobuf .onnx files directly from R (no Python needed).

# ── Helper: load, run, return output vector ──────────────────────

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Binary ops ───────────────────────────────────────────────────

test_that("ONNX Add works", {
  path <- .onnx_make_binary("Add", c(4L))
  a <- c(1, 2, 3, 4)
  b <- c(10, 20, 30, 40)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a + b, tolerance = 1e-5)
})

test_that("ONNX Sub works", {
  path <- .onnx_make_binary("Sub", c(4L))
  a <- c(10, 20, 30, 40)
  b <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a - b, tolerance = 1e-5)
})

test_that("ONNX Mul works", {
  path <- .onnx_make_binary("Mul", c(4L))
  a <- c(2, 3, 4, 5)
  b <- c(10, 10, 10, 10)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a * b, tolerance = 1e-5)
})

test_that("ONNX Div works", {
  path <- .onnx_make_binary("Div", c(4L))
  a <- c(10, 20, 30, 40)
  b <- c(2, 4, 5, 8)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), a / b, tolerance = 1e-5)
})

# ── Activations ──────────────────────────────────────────────────

test_that("ONNX Relu works", {
  path <- .onnx_make_unary("Relu", c(4L))
  x <- c(-2, -1, 0, 3)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("ONNX Sigmoid works", {
  path <- .onnx_make_unary("Sigmoid", c(4L))
  x <- c(-2, 0, 1, 5)
  result <- run_onnx(path, list(X = x))
  expected <- 1 / (1 + exp(-x))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX Tanh works", {
  path <- .onnx_make_unary("Tanh", c(4L))
  x <- c(-2, -0.5, 0, 1.5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), tanh(x), tolerance = 1e-5)
})

test_that("ONNX Silu works", {
  path <- .onnx_make_unary("Silu", c(4L))
  x <- c(-2, -1, 0, 2)
  result <- run_onnx(path, list(X = x))
  expected <- x / (1 + exp(-x))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX Elu works", {
  path <- .onnx_make_unary("Elu", c(4L))
  x <- c(-2, -1, 0, 1)
  result <- run_onnx(path, list(X = x))
  expected <- ifelse(x >= 0, x, exp(x) - 1)
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX Softmax works", {
  path <- .onnx_make_unary("Softmax", c(4L))
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expected <- exp(x) / sum(exp(x))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX LeakyRelu works", {
  attrs <- list(.onnx_attr_float("alpha", 0.1))
  path <- .onnx_make_unary("LeakyRelu", c(4L), attrs = attrs)
  x <- c(-2, -1, 0, 3)
  result <- run_onnx(path, list(X = x))
  expected <- ifelse(x >= 0, x, 0.1 * x)
  expect_equal(as.numeric(result), expected, tolerance = 1e-5)
})

# ── Math ops ─────────────────────────────────────────────────────

test_that("ONNX Sqrt works", {
  path <- .onnx_make_unary("Sqrt", c(4L))
  x <- c(1, 4, 9, 16)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), sqrt(x), tolerance = 1e-5)
})

test_that("ONNX Exp works", {
  path <- .onnx_make_unary("Exp", c(4L))
  x <- c(0, 1, 2, -1)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), exp(x), tolerance = 1e-4)
})

test_that("ONNX Log works", {
  path <- .onnx_make_unary("Log", c(4L))
  x <- c(1, 2, 10, 100)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), log(x), tolerance = 1e-4)
})

test_that("ONNX Abs works", {
  path <- .onnx_make_unary("Abs", c(4L))
  x <- c(-3, -1, 0, 5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), abs(x), tolerance = 1e-5)
})

test_that("ONNX Neg works", {
  path <- .onnx_make_unary("Neg", c(4L))
  x <- c(-3, -1, 0, 5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), -x, tolerance = 1e-5)
})

test_that("ONNX Floor works", {
  path <- .onnx_make_unary("Floor", c(4L))
  x <- c(-1.5, 0.3, 2.7, 3.0)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), floor(x), tolerance = 1e-5)
})

test_that("ONNX Ceil works", {
  path <- .onnx_make_unary("Ceil", c(4L))
  x <- c(-1.5, 0.3, 2.7, 3.0)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), ceiling(x), tolerance = 1e-5)
})

# ── Identity / Dropout ───────────────────────────────────────────

test_that("ONNX Identity is pass-through", {
  path <- .onnx_make_unary("Identity", c(4L))
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

test_that("ONNX Dropout is pass-through (inference)", {
  path <- .onnx_make_unary("Dropout", c(4L))
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), x, tolerance = 1e-5)
})

# ── Chain ops ────────────────────────────────────────────────────

test_that("ONNX chain Relu -> Sigmoid works", {
  path <- .onnx_make_chain("Relu", "Sigmoid", c(4L))
  x <- c(-2, -1, 0, 3)
  result <- run_onnx(path, list(X = x))
  expected <- 1 / (1 + exp(-pmax(x, 0)))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

# ── Reshape ──────────────────────────────────────────────────────

test_that("ONNX Reshape preserves elements", {
  path <- .onnx_make_reshape(c(2L, 3L), c(3L, 2L))
  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
})

# ── MatMul ───────────────────────────────────────────────────────

test_that("ONNX MatMul works for 2D", {
  # A[2,3] @ B[3,2] = Y[2,2]
  # A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
  # A@B = [[22,28],[49,64]]
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
  # With scale=1, bias=0: should be zero-mean, unit-var
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(abs(mean(r)) < 0.01)
  # ggml_norm uses population stddev (N), not sample stddev (N-1)
  pop_sd <- sqrt(mean((x - mean(x))^2))
  expected <- (x - mean(x)) / pop_sd
  expect_equal(r, expected, tolerance = 0.01)
})

# ── Model metadata ───────────────────────────────────────────────

test_that("onnx_load returns correct metadata", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  expect_s3_class(m, "onnx_model")
  expect_equal(m$n_nodes, 1L)
  expect_true("Relu" %in% m$ops)
})

test_that("onnx_summary works", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  s <- onnx_summary(m)
  expect_true(is.list(s))
  expect_equal(s$n_nodes, 1L)
})

test_that("onnx_inputs returns correct info", {
  path <- .onnx_make_binary("Add", c(4L))
  m <- onnx_load(path, device = "cpu")
  inp <- onnx_inputs(m)
  expect_true(is.list(inp))
  expect_true(length(inp) >= 2)
})

test_that("print.onnx_model works", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  expect_output(print(m), "ONNX Model")
})

# ── Constant op ──────────────────────────────────────────────────

test_that("ONNX Constant tensor + Add works", {
  # Model: Constant([1,2,3,4]) + X → Y
  const_data <- unlist(lapply(c(10, 20, 30, 40), .float_bytes))
  const_attr <- .onnx_attr_tensor("value", c(4L), 1L, const_data)
  const_node <- .onnx_node("Constant", character(0), "C", attrs = list(const_attr))

  inp  <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  add_node <- .onnx_node("Add", c("C", "X"), "Y")

  graph <- .onnx_graph("test", list(const_node, add_node),
                        list(inp), list(outp))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(11, 22, 33, 44), tolerance = 1e-5)
})

# ── Unsqueeze / Squeeze ──────────────────────────────────────────

test_that("ONNX Unsqueeze works (attr axes)", {
  # X[2,3] → Unsqueeze(axes=[0]) → Y[1,2,3] → Relu → Y
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L))
  n1 <- .onnx_node("Unsqueeze", "X", "tmp",
                    attrs = list(.onnx_attr_ints("axes", c(0L))))
  n2 <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(n1, n2), list(inp), list(outp))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(1, -2, 3, -4, 5, -6)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("ONNX Squeeze works (attr axes)", {
  # X[1,2,3] → Squeeze(axes=[0]) → Y[2,3] → Relu → Y
  inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))
  n1 <- .onnx_node("Squeeze", "X", "tmp",
                    attrs = list(.onnx_attr_ints("axes", c(0L))))
  n2 <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(n1, n2), list(inp), list(outp))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(1, -2, 3, -4, 5, -6)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("ONNX Squeeze without axes squeezes all 1-dims", {
  # X[1,4,1] → Squeeze() → Y[4] → Relu → Y
  inp  <- .onnx_value_info("X", 1L, c(1L, 4L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  n1 <- .onnx_node("Squeeze", "X", "tmp")
  n2 <- .onnx_node("Relu", "tmp", "Y")
  graph <- .onnx_graph("test", list(n1, n2), list(inp), list(outp))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(-1, 2, -3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

# ── Slice ────────────────────────────────────────────────────────

test_that("ONNX Slice works on 1D", {
  # X[6] → Slice(starts=[1], ends=[4]) → Y[3]
  inp  <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Y", 1L, c(3L))

  starts_raw <- .int64_bytes(1L)
  ends_raw   <- .int64_bytes(4L)
  starts_t <- .onnx_tensor("starts", c(1L), 7L, starts_raw)
  ends_t   <- .onnx_tensor("ends",   c(1L), 7L, ends_raw)
  starts_vi <- .onnx_value_info("starts", 7L, c(1L))
  ends_vi   <- .onnx_value_info("ends",   7L, c(1L))

  node <- .onnx_node("Slice", c("X", "starts", "ends"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, starts_vi, ends_vi), list(outp),
                        list(starts_t, ends_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(10, 20, 30, 40, 50, 60)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(20, 30, 40), tolerance = 1e-5)
})

test_that("ONNX Slice works on 2D with axes", {
  # X[2,4] → Slice(starts=[1], ends=[3], axes=[1]) → Y[2,2]
  inp  <- .onnx_value_info("X", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 2L))

  starts_raw <- .int64_bytes(1L)
  ends_raw   <- .int64_bytes(3L)
  axes_raw   <- .int64_bytes(1L)
  starts_t <- .onnx_tensor("starts", c(1L), 7L, starts_raw)
  ends_t   <- .onnx_tensor("ends",   c(1L), 7L, ends_raw)
  axes_t   <- .onnx_tensor("axes",   c(1L), 7L, axes_raw)
  starts_vi <- .onnx_value_info("starts", 7L, c(1L))
  ends_vi   <- .onnx_value_info("ends",   7L, c(1L))
  axes_vi   <- .onnx_value_info("axes",   7L, c(1L))

  node <- .onnx_node("Slice", c("X", "starts", "ends", "axes"), "Y")
  graph <- .onnx_graph("test", list(node),
                        list(inp, starts_vi, ends_vi, axes_vi), list(outp),
                        list(starts_t, ends_t, axes_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  # ONNX row-major: [[1,2,3,4],[5,6,7,8]], slice cols 1:3 → [[2,3],[6,7]]
  x <- c(1, 2, 3, 4, 5, 6, 7, 8)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(2, 3, 6, 7), tolerance = 1e-5)
})

# ── Split ────────────────────────────────────────────────────────

test_that("ONNX Split works with equal chunks", {
  # X[6] → Split(axis=0) → Y1[3], Y2[3] → Add(Y1, Y2) → Z[3]
  inp  <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Z", 1L, c(3L))

  split_node <- .onnx_node("Split", "X", c("Y1", "Y2"),
                            attrs = list(.onnx_attr_int("axis", 0L)))
  add_node <- .onnx_node("Add", c("Y1", "Y2"), "Z")

  graph <- .onnx_graph("test", list(split_node, add_node),
                        list(inp), list(outp))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(1, 2, 3, 10, 20, 30)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(11, 22, 33), tolerance = 1e-5)
})

test_that("ONNX Split works with explicit sizes", {
  # X[6] → Split(axis=0, split=[2,4]) → Y1[2], Y2[4]
  # → take Y2 through Relu → Z[4]
  inp  <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Z", 1L, c(4L))

  split_raw <- c(.int64_bytes(2L), .int64_bytes(4L))
  split_t  <- .onnx_tensor("sp", c(2L), 7L, split_raw)
  split_vi <- .onnx_value_info("sp", 7L, c(2L))

  split_node <- .onnx_node("Split", c("X", "sp"), c("Y1", "Y2"),
                            attrs = list(.onnx_attr_int("axis", 0L)))
  relu_node <- .onnx_node("Relu", "Y2", "Z")

  graph <- .onnx_graph("test", list(split_node, relu_node),
                        list(inp, split_vi), list(outp),
                        list(split_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(100, 200, -1, 2, -3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(0, 2, 0, 4), tolerance = 1e-5)
})

# ── Expand ───────────────────────────────────────────────────────

test_that("ONNX Expand broadcasts 1D to 2D", {
  # X[3] → Expand([2,3]) → Y[2,3] → Relu → Z
  inp  <- .onnx_value_info("X", 1L, c(3L))

  shape_raw <- c(.int64_bytes(2L), .int64_bytes(3L))
  shape_t  <- .onnx_tensor("shape", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(2L))

  outp <- .onnx_value_info("Z", 1L, c(2L, 3L))
  expand_node <- .onnx_node("Expand", c("X", "shape"), "Y")
  relu_node   <- .onnx_node("Relu", "Y", "Z")

  graph <- .onnx_graph("test", list(expand_node, relu_node),
                        list(inp, shape_vi), list(outp),
                        list(shape_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(-1, 2, 3)
  result <- run_onnx(path, list(X = x))
  # Expand [3] → [2,3], then relu: row-major [0,2,3, 0,2,3]
  expect_equal(as.numeric(result), c(0, 2, 3, 0, 2, 3), tolerance = 1e-5)
})

test_that("ONNX Expand broadcasts scalar to 1D", {
  # X[1] → Expand([4]) → Y[4]
  inp  <- .onnx_value_info("X", 1L, c(1L))

  shape_raw <- .int64_bytes(4L)
  shape_t  <- .onnx_tensor("shape", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, c(1L))

  outp <- .onnx_value_info("Y", 1L, c(4L))
  expand_node <- .onnx_node("Expand", c("X", "shape"), "Y")

  graph <- .onnx_graph("test", list(expand_node),
                        list(inp, shape_vi), list(outp),
                        list(shape_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(5)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(5, 5, 5, 5), tolerance = 1e-5)
})

# ── Resize / Upsample ───────────────────────────────────────────

test_that("ONNX Resize nearest with scales works", {
  # X[1,1,2,2] → Resize(scales=[1,1,2,2]) → Y[1,1,4,4] (nearest)
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))

  # roi = empty, scales = [1,1,2,2]
  roi_t  <- .onnx_tensor("roi", c(0L), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, c(0L))
  scales_raw <- unlist(lapply(c(1, 1, 2, 2), .float_bytes))
  scales_t  <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))

  node <- .onnx_node("Resize", c("X", "roi", "scales"), "Y",
                      attrs = list(.onnx_attr_int("mode", 0L)))
  # mode attr is string in real ONNX but we use default "nearest"
  node <- .onnx_node("Resize", c("X", "roi", "scales"), "Y")

  graph <- .onnx_graph("test", list(node),
                        list(inp, roi_vi, scales_vi), list(outp),
                        list(roi_t, scales_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  # 2x2 input: [[1,2],[3,4]] → nearest 2x → [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 16)
  # Check corners
  expect_equal(result[1], 1, tolerance = 1e-5)
  expect_equal(result[16], 4, tolerance = 1e-5)
})

test_that("ONNX Resize nearest with sizes works", {
  # X[1,1,2,3] → Resize(sizes=[1,1,4,6]) → Y[1,1,4,6]
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 6L))

  roi_t  <- .onnx_tensor("roi", c(0L), 1L, raw(0))
  roi_vi <- .onnx_value_info("roi", 1L, c(0L))
  # empty scales
  scales_t  <- .onnx_tensor("scales", c(0L), 1L, raw(0))
  scales_vi <- .onnx_value_info("scales", 1L, c(0L))
  # sizes
  sizes_raw <- c(.int64_bytes(1L), .int64_bytes(1L),
                 .int64_bytes(4L), .int64_bytes(6L))
  sizes_t  <- .onnx_tensor("sizes", c(4L), 7L, sizes_raw)
  sizes_vi <- .onnx_value_info("sizes", 7L, c(4L))

  node <- .onnx_node("Resize", c("X", "roi", "scales", "sizes"), "Y")

  graph <- .onnx_graph("test", list(node),
                        list(inp, roi_vi, scales_vi, sizes_vi), list(outp),
                        list(roi_t, scales_t, sizes_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 24)
})

# ── ConvTranspose ────────────────────────────────────────────────

test_that("ONNX ConvTranspose 1D works", {
  # X[1,1,3] (batch=1, channels=1, length=3) with kernel [1,1,2] stride=2
  # ConvTranspose1D: output length = (3-1)*2 + 2 = 6
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 6L))

  # Kernel: [C_in=1, C_out=1, K=2], all ones
  w_data <- rep(1.0, 2)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(1L, 1L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L))

  node <- .onnx_node("ConvTranspose", c("X", "W"), "Y",
                      attrs = list(.onnx_attr_ints("strides", c(2L))))

  graph <- .onnx_graph("test", list(node),
                        list(inp, w_vi), list(outp),
                        list(w_t))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(1, 2, 3)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(result), 6)
  # With kernel=[1,1] stride=2: inserts zeros then convolves
  # Expected: [1, 1, 2, 2, 3, 3]
  expect_equal(as.numeric(result), c(1, 1, 2, 2, 3, 3), tolerance = 1e-5)
})
