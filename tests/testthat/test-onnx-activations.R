# Tests for ONNX activation and math ops

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

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

test_that("ONNX HardSigmoid uses the spec defaults", {
  path <- .onnx_make_unary("HardSigmoid", c(6L))
  x <- c(-4, -2.5, -1, 0, 1, 4)
  result <- run_onnx(path, list(X = x))
  # spec defaults: alpha = 0.2, beta = 0.5
  expected <- pmax(0, pmin(1, 0.2 * x + 0.5))
  expect_equal(as.numeric(result), expected, tolerance = 1e-5)
})

test_that("ONNX HardSigmoid honours non-default alpha/beta", {
  # The reason this op is built from scale_bias + clamp rather than
  # ggml_hardsigmoid: that primitive hardcodes alpha = 1/6, beta = 0.5, so a
  # model choosing its own would be silently computed as something else.
  attrs <- list(.onnx_attr_float("alpha", 0.5),
                .onnx_attr_float("beta", 0.1))
  path <- .onnx_make_unary("HardSigmoid", c(6L), attrs = attrs)
  x <- c(-4, -1, -0.2, 0, 1, 4)
  result <- run_onnx(path, list(X = x))
  expected <- pmax(0, pmin(1, 0.5 * x + 0.1))
  expect_equal(as.numeric(result), expected, tolerance = 1e-5)
  # and it must differ from the hardcoded-constant answer, or the test would
  # pass just as well against the primitive it replaced
  expect_false(isTRUE(all.equal(expected, pmax(0, pmin(1, x/6 + 0.5)))))
})

test_that("ONNX HardSwish works", {
  path <- .onnx_make_unary("HardSwish", c(6L))
  x <- c(-4, -2, -1, 0, 1, 4)
  result <- run_onnx(path, list(X = x))
  # x * max(0, min(1, x/6 + 0.5)) -- constants fixed by the spec
  expected <- x * pmax(0, pmin(1, x / 6 + 0.5))
  expect_equal(as.numeric(result), expected, tolerance = 1e-4)
})

test_that("ONNX PRelu applies a per-element slope", {
  # slope is a TENSOR here, which is what separates PRelu from LeakyRelu and
  # rules out ggml_leaky_relu (scalar alpha only).
  inp   <- .onnx_value_info("X", 1L, c(4L))
  outp  <- .onnx_value_info("Y", 1L, c(4L))
  slope <- c(0.1, 0.2, 0.3, 0.4)
  s_t   <- .onnx_tensor("slope", c(4L), 1L, .float_bytes(slope))
  s_vi  <- .onnx_value_info("slope", 1L, c(4L))
  node  <- .onnx_node("PRelu", c("X", "slope"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp, s_vi), list(outp), list(s_t))
  path  <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-2, -1, 1, 3)
  result <- run_onnx(path, list(X = x))
  expected <- ifelse(x >= 0, x, slope * x)
  expect_equal(as.numeric(result), expected, tolerance = 1e-5)
})

test_that("ONNX PRelu broadcasts a per-channel slope", {
  # ONNX [1,3,2,2] with a [3] slope: one value per channel, the shape real
  # models use.
  inp   <- .onnx_value_info("X", 1L, c(1L, 3L, 2L, 2L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 3L, 2L, 2L))
  slope <- c(0.1, 0.5, 0.9)
  s_t   <- .onnx_tensor("slope", c(3L), 1L, .float_bytes(slope))
  s_vi  <- .onnx_value_info("slope", 1L, c(3L))
  node  <- .onnx_node("PRelu", c("X", "slope"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp, s_vi), list(outp), list(s_t))
  path  <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(3)
  x <- round(runif(12, -2, 2), 3)
  result <- as.numeric(run_onnx(path, list(X = x)))
  # ONNX is row-major [c,h,w]; in column-major that is [w,h,c], so the channel
  # is the slowest axis and each block of 4 shares one slope.
  expected <- ifelse(x >= 0, x, rep(slope, each = 4) * x)
  expect_equal(result, expected, tolerance = 1e-5)
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

# ── Mod ──────────────────────────────────────────────────────────

.onnx_make_mod <- function(fmod, a_len = 8L) {
  inp  <- .onnx_value_info("X", 1L, c(a_len))
  outp <- .onnx_value_info("Y", 1L, c(a_len))
  b_vi <- .onnx_value_info("B", 1L, c(a_len))
  attrs <- if (is.null(fmod)) list() else list(.onnx_attr_int("fmod", fmod))
  node  <- .onnx_node("Mod", c("X", "B"), "Y", attrs = attrs)
  function(bvals) {
    b_t   <- .onnx_tensor("B", c(a_len), 1L, .float_bytes(bvals))
    graph <- .onnx_graph("test", list(node), list(inp, b_vi), list(outp), list(b_t))
    path  <- tempfile(fileext = ".onnx")
    writeBin(.onnx_model(graph), path)
    path
  }
}

test_that("ONNX Mod defaults to the sign of the divisor", {
  # fmod=0 is Python's %, which is what R's %% computes too.
  a <- c(-7, -5, -3, 3, 5, 7, -8, 8)
  b <- c( 3,  3, -3, 3, -3, 3,  3, -3)
  path <- .onnx_make_mod(NULL)(b)
  result <- run_onnx(path, list(X = a))
  expect_equal(as.numeric(result), a %% b, tolerance = 1e-5)
})

test_that("ONNX Mod with fmod=1 takes the sign of the dividend", {
  # C's fmod.  The two modes disagree exactly when the operands' signs differ,
  # so the divisors below are chosen to cover both matching and mixed signs --
  # a test using only positives would pass against either implementation.
  a <- c(-7, -5, -3, 3, 5, 7, -8, 8)
  b <- c( 3,  3, -3, 3, -3, 3,  3, -3)
  path <- .onnx_make_mod(1L)(b)
  result <- run_onnx(path, list(X = a))
  expected <- sign(a) * (abs(a) %% abs(b))
  expect_equal(as.numeric(result), expected, tolerance = 1e-5)
  # and it must actually differ from the fmod=0 answer somewhere
  expect_false(isTRUE(all.equal(expected, a %% b)))
})

test_that("ONNX Mod carries its value through to a shape computation", {
  # Mod is shape arithmetic more often than data arithmetic.  Swin sizes its
  # window padding as (window - size %% window) and feeds that to Pad; if the
  # value does not travel with the graph, Pad receives zeros, pads nothing,
  # and the 6-D window partition downstream gets a shape that contradicts its
  # own element count.  Here the same chain drives a Reshape, which fails
  # outright on a wrong size rather than quietly computing something else.
  #
  # Shape(X)[0] = 96; 96 %% 7 = 5; 7 - 5 = 2; so the reshape target is 2.
  inp  <- .onnx_value_info("X", 1L, c(96L))
  outp <- .onnx_value_info("Y", 1L, c(2L))

  seven <- .onnx_tensor("seven", c(1L), 7L, .int64_bytes(7))
  zero  <- .onnx_tensor("zero",  c(1L), 7L, .int64_bytes(0))
  src   <- .onnx_tensor("src",   c(2L), 1L, .float_bytes(c(11, 22)))
  vis <- list(inp,
              .onnx_value_info("seven", 7L, c(1L)),
              .onnx_value_info("zero",  7L, c(1L)),
              .onnx_value_info("src",   1L, c(2L)))

  nodes <- list(
    .onnx_node("Shape",   "X", "shp"),
    .onnx_node("Gather",  c("shp", "zero"), "dim0",
               attrs = list(.onnx_attr_int("axis", 0L))),
    .onnx_node("Mod",     c("dim0", "seven"), "rem"),
    .onnx_node("Sub",     c("seven", "rem"), "pad"),
    .onnx_node("Reshape", c("src", "pad"), "Y"))

  graph <- .onnx_graph("test", nodes, vis, list(outp),
                        list(seven, zero, src))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- as.numeric(run_onnx(path, list(X = rep(0, 96))))
  expect_equal(length(result), 2L)
  expect_equal(result, c(11, 22), tolerance = 1e-5)
})

test_that("ONNX Clip reads bounds stored in float_data, not just raw_data", {
  # A TensorProto may carry its payload in raw_data (field 9) or in the typed
  # arrays (float_data, field 4); both are legal and exporters use both.
  # Reading only raw_data leaves the bounds at their +-FLT_MAX defaults, so
  # Clip becomes a no-op -- silently, because the output is the input and
  # looks perfectly reasonable.
  #
  # MaskRCNN clips box-scale logits to log(1000/16) = 4.135 before exp().
  # Without the clamp exp() of a 1e8 logit is inf, and the detection branch
  # downstream turns to garbage: that is the failure this test stands for.
  inp  <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  lo <- .onnx_tensor_floatdata("lo", c(1L), -3.402823e+38)
  hi <- .onnx_tensor_floatdata("hi", c(1L), 4.135167)
  vis <- list(inp,
              .onnx_value_info("lo", 1L, c(1L)),
              .onnx_value_info("hi", 1L, c(1L)))

  node  <- .onnx_node("Clip", c("X", "lo", "hi"), "Y")
  graph <- .onnx_graph("test", list(node), vis, list(outp), list(lo, hi))
  path  <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-5, 0, 4, 1e8)
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(result, pmin(x, 4.135167), tolerance = 1e-4)
  # the last one is the point: unclamped it would go on to overflow exp()
  expect_true(result[4] < 5)
})
