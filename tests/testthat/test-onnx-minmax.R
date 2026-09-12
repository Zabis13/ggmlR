# Tests for ONNX Min/Max — elementwise, variadic.
#
# No model in the reference zoo uses these ops, so a hand-built graph is the
# only thing that exercises them at all: without these tests the handler is
# code that compiles and is never run.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Build a graph with N inputs feeding one variadic node.  .onnx_make_binary
# fixes the arity at two, and the whole point here is the arity.
.onnx_make_variadic <- function(op_type, n_inputs, dims = c(4L),
                                dims_each = NULL) {
  names_in <- paste0("X", seq_len(n_inputs))
  if (is.null(dims_each)) dims_each <- rep(list(dims), n_inputs)
  inputs <- Map(function(nm, d) .onnx_value_info(nm, 1L, d),
                names_in, dims_each)
  outp   <- .onnx_value_info("Y", 1L, dims)
  node   <- .onnx_node(op_type, names_in, "Y")
  graph  <- .onnx_graph("test", list(node), unname(inputs), list(outp))
  path   <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

# ── Two inputs: the ordinary case ───────────────────────────────

test_that("ONNX Min of two tensors works", {
  path <- .onnx_make_variadic("Min", 2L, c(5L))
  x <- c(-2, 0.5, 3, -7, 100)
  y <- c(1, -0.5, 3, 2, 99)
  result <- run_onnx(path, list(X1 = x, X2 = y))
  expect_equal(as.numeric(result), pmin(x, y), tolerance = 1e-5)
})

test_that("ONNX Max of two tensors works", {
  path <- .onnx_make_variadic("Max", 2L, c(5L))
  x <- c(-2, 0.5, 3, -7, 100)
  y <- c(1, -0.5, 3, 2, 99)
  result <- run_onnx(path, list(X1 = x, X2 = y))
  expect_equal(as.numeric(result), pmax(x, y), tolerance = 1e-5)
})

# Equal operands are where a formula built on |x - y| could go wrong: the
# absolute difference is zero and both branches have to agree.
test_that("ONNX Min/Max agree when the operands are equal", {
  x <- c(1, 2, 3, 4)
  p_min <- .onnx_make_variadic("Min", 2L, c(4L))
  p_max <- .onnx_make_variadic("Max", 2L, c(4L))
  expect_equal(as.numeric(run_onnx(p_min, list(X1 = x, X2 = x))), x,
               tolerance = 1e-5)
  expect_equal(as.numeric(run_onnx(p_max, list(X1 = x, X2 = x))), x,
               tolerance = 1e-5)
})

# ── Variadic: one input, and more than two ──────────────────────

test_that("ONNX Min with a single input is the identity", {
  path <- .onnx_make_variadic("Min", 1L, c(4L))
  x <- c(-1, 0, 2.5, 7)
  expect_equal(as.numeric(run_onnx(path, list(X1 = x))), x, tolerance = 1e-5)
})

test_that("ONNX Min folds three inputs", {
  path <- .onnx_make_variadic("Min", 3L, c(4L))
  x <- c(5, -1, 3, 0)
  y <- c(2,  4, 3, -8)
  z <- c(9,  0, 1, 6)
  result <- run_onnx(path, list(X1 = x, X2 = y, X3 = z))
  expect_equal(as.numeric(result), pmin(x, y, z), tolerance = 1e-5)
})

test_that("ONNX Max folds four inputs", {
  path <- .onnx_make_variadic("Max", 4L, c(3L))
  a <- c(1, -5, 2); b <- c(0, 4, 2)
  d <- c(-3, 1, 2); e <- c(2, 3, -9)
  result <- run_onnx(path, list(X1 = a, X2 = b, X3 = d, X4 = e))
  expect_equal(as.numeric(result), pmax(a, b, d, e), tolerance = 1e-5)
})

# ── Broadcasting ────────────────────────────────────────────────

test_that("ONNX Max broadcasts a scalar operand", {
  # The clamp idiom: Max(lo, x).  A scalar second input is how every real
  # model writes it, and it goes down the broadcast path rather than the
  # elementwise one.
  path <- .onnx_make_variadic("Max", 2L, dims = c(5L),
                              dims_each = list(c(5L), c(1L)))
  x  <- c(-3, -1, 0, 2, 5)
  lo <- 0
  result <- run_onnx(path, list(X1 = x, X2 = lo))
  expect_equal(as.numeric(result), pmax(x, lo), tolerance = 1e-5)
})

test_that("ONNX Min/Max express a clamp", {
  # Max(lo, Min(hi, x)) is the form MaskRCNN uses for FPN level assignment,
  # and the reason the handler folds compile-time values through both ops.
  inp_x  <- .onnx_value_info("X", 1L, c(6L))
  inp_hi <- .onnx_value_info("HI", 1L, c(1L))
  inp_lo <- .onnx_value_info("LO", 1L, c(1L))
  outp   <- .onnx_value_info("Y", 1L, c(6L))
  n1 <- .onnx_node("Min", c("X", "HI"), "T")
  n2 <- .onnx_node("Max", c("T", "LO"), "Y")
  graph <- .onnx_graph("clamp", list(n1, n2),
                       list(inp_x, inp_hi, inp_lo), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-5, -1, 0, 3, 8, 12)
  result <- run_onnx(path, list(X = x, HI = 7, LO = 0))
  expect_equal(as.numeric(result), pmin(pmax(x, 0), 7), tolerance = 1e-5)
})

# ── 2-D, to check the broadcast helper is fed a real shape ──────

test_that("ONNX Min works on a 2-D tensor", {
  path <- .onnx_make_variadic("Min", 2L, c(2L, 3L))
  x <- c(1, -2, 3, -4, 5, -6)
  y <- c(0,  0, 4, -9, 2,  1)
  result <- run_onnx(path, list(X1 = x, X2 = y))
  expect_equal(as.numeric(result), pmin(x, y), tolerance = 1e-5)
})
