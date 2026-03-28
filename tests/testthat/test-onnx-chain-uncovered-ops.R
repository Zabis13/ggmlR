# Chain tests: Ops without chain coverage
# ScatterElements, Split, Log, RMSNormalization, Upsample, EyeLike

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Split → Concat round-trip ────────────────────────────────────

test_that("chain uncovered: Split axis=1 → process → Concat (attention QKV split)", {
  # Input [1,12] → Split axis=1 into 3x [1,4] → Relu each → Concat → [1,12]
  inp  <- .onnx_value_info("X", 1L, c(1L, 12L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 12L))

  # Split into 3 equal parts along axis 1
  split_node <- .onnx_node("Split", "X", c("a", "b", "c"),
                             attrs = list(.onnx_attr_int("axis", 1L)))
  relu_a <- .onnx_node("Relu", "a", "ra")
  relu_b <- .onnx_node("Relu", "b", "rb")
  relu_c <- .onnx_node("Relu", "c", "rc")
  cat_node <- .onnx_node("Concat", c("ra", "rb", "rc"), "Y",
                           attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
    list(split_node, relu_a, relu_b, relu_c, cat_node),
    list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 12)
  # Relu: negatives become 0
  expected <- pmax(x, 0)
  expect_equal(r, expected, tolerance = 1e-5)
})

# ── Log in a chain ───────────────────────────────────────────────

test_that("chain uncovered: Exp → Log round-trip (identity for positive)", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  exp_node <- .onnx_node("Exp", "X", "e")
  log_node <- .onnx_node("Log", "e", "Y")

  graph <- .onnx_graph("test", list(exp_node, log_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0.5, 1.0, 2.0, 3.0)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(r, x, tolerance = 1e-5)
})

# ── Log + Softmax (log-softmax pattern) ──────────────────────────

test_that("chain uncovered: Softmax → Log (log-softmax)", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  sm_node <- .onnx_node("Softmax", "X", "sm",
                          attrs = list(.onnx_attr_int("axis", 1L)))
  log_node <- .onnx_node("Log", "sm", "Y")

  graph <- .onnx_graph("test", list(sm_node, log_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(all(is.finite(r)))
  expect_true(all(r <= 0))  # log of probabilities should be <= 0
})

# ── RMSNormalization ─────────────────────────────────────────────

test_that("chain uncovered: RMSNormalization → MatMul (LLaMA pattern)", {
  # RMSNorm([1,4], scale=[4]) → MatMul([4,3]) → [1,3]
  inp  <- .onnx_value_info("X", 1L, c(1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 3L))

  # Scale = ones
  sc_raw <- unlist(lapply(rep(1.0, 4), .float_bytes))
  sc_t <- .onnx_tensor("scale", c(4L), 1L, sc_raw)
  sc_vi <- .onnx_value_info("scale", 1L, c(4L))

  # Weight [4,3]
  w_raw <- unlist(lapply(rep(0.5, 12), .float_bytes))
  w_t <- .onnx_tensor("W", c(4L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(4L, 3L))

  rms_node <- .onnx_node("RMSNormalization", c("X", "scale"), "normed",
                           attrs = list(.onnx_attr_float("epsilon", 1e-5)))
  mm_node <- .onnx_node("MatMul", c("normed", "W"), "Y")

  graph <- .onnx_graph("test",
    list(rms_node, mm_node),
    list(inp, sc_vi, w_vi), list(outp),
    list(sc_t, w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 3)
  expect_true(all(is.finite(r)))
})

# ── Upsample (alias for Resize) ─────────────────────────────────

test_that("chain uncovered: Upsample nearest 2x + Conv (decoder pattern)", {
  # [1,1,4,4] → Upsample 2x → [1,1,8,8] → Conv 3x3 → [1,1,6,6]
  inp  <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 6L, 6L))

  # Scales [1,1,2,2]
  scales_raw <- unlist(lapply(c(1.0, 1.0, 2.0, 2.0), .float_bytes))
  scales_t <- .onnx_tensor("scales", c(4L), 1L, scales_raw)
  scales_vi <- .onnx_value_info("scales", 1L, c(4L))

  # Conv 3x3 weight [1,1,3,3]
  w_raw <- unlist(lapply(rep(1.0/9, 9), .float_bytes))
  w_t <- .onnx_tensor("W", c(1L, 1L, 3L, 3L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 3L, 3L))

  up_node <- .onnx_node("Upsample", c("X", "scales"), "upsampled",
                          attrs = list(.onnx_attr_int("mode", 1L)))  # nearest
  conv_node <- .onnx_node("Conv", c("upsampled", "W"), "Y",
                            attrs = list(.onnx_attr_ints("kernel_shape", c(3L, 3L))))

  graph <- .onnx_graph("test",
    list(up_node, conv_node),
    list(inp, scales_vi, w_vi), list(outp),
    list(scales_t, w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rnorm(16)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 36)  # 6*6
  expect_true(all(is.finite(r)))
})

# ── ReduceMean with axes (new fix, was broken for cait) ──────────

test_that("chain uncovered: ReduceMean axes=[2] + Sub (LayerNorm manual pattern)", {
  # X[1,4,8] → ReduceMean(axes=2, i.e. last ONNX dim) → [1,4,1] → Sub(X, mean) → [1,4,8]
  # Note: axes=-1 would be equivalent but protobuf varint can't encode negative ints
  inp  <- .onnx_value_info("X", 1L, c(1L, 4L, 8L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L, 8L))

  mean_node <- .onnx_node("ReduceMean", "X", "mean",
                            attrs = list(.onnx_attr_ints("axes", c(2L)),
                                         .onnx_attr_int("keepdims", 1L)))
  sub_node <- .onnx_node("Sub", c("X", "mean"), "Y")

  graph <- .onnx_graph("test", list(mean_node, sub_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(42)
  x <- rnorm(32)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 32)
  expect_true(all(is.finite(r)))

  # Each group of 8 should sum to ~0 (centered)
  m <- matrix(r, nrow = 8)
  col_sums <- colSums(m)
  expect_true(all(abs(col_sums) < 1e-4))
})

# ── ReduceSum with axes ──────────────────────────────────────────

test_that("chain uncovered: ReduceSum axes=[1] on [2,3,4] (sum over middle dim)", {
  inp  <- .onnx_value_info("X", 1L, c(2L, 3L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 1L, 4L))

  sum_node <- .onnx_node("ReduceSum", "X", "Y",
                           attrs = list(.onnx_attr_ints("axes", c(1L)),
                                        .onnx_attr_int("keepdims", 1L)))

  graph <- .onnx_graph("test", list(sum_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(1.0, 24)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 8)  # 2*1*4
  expect_true(all(is.finite(r)))
  # Sum of 3 ones along axis 1 = 3.0
  expect_true(all(abs(r - 3.0) < 1e-5))
})
