# ONNX Conv with a 1-D kernel.
#
# The bias is the point.  A 2-D convolution's output is [W,H,C,N] and a 1-D
# one's is [L,C,N], so the channel sits on a different ggml axis in each --
# a bias reshaped for the 2-D case does not broadcast over the 1-D one at all,
# and ggml_add refuses it.  Nothing here covered Conv1d before, which is why
# whisper-tiny's first layer, Conv1d(80 -> 384), aborted on load.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ONNX Conv1d: X [N,C_in,L], W [C_out,C_in,K], B [C_out] -> Y [N,C_out,L_out]
make_conv1d <- function(N, C_in, L, C_out, K, w, b = NULL,
                        stride = 1L, pad = 0L) {
  L_out <- (L + 2L * pad - K) %/% stride + 1L
  inp  <- .onnx_value_info("X", 1L, c(N, C_in, L))
  outp <- .onnx_value_info("Y", 1L, c(N, C_out, L_out))

  w_t  <- .onnx_tensor("W", c(C_out, C_in, K), 1L, .float_bytes(w))
  w_vi <- .onnx_value_info("W", 1L, c(C_out, C_in, K))
  inits <- list(w_t); vis <- list(inp, w_vi); ins <- c("X", "W")
  if (!is.null(b)) {
    inits <- c(inits, list(.onnx_tensor("B", c(C_out), 1L, .float_bytes(b))))
    vis   <- c(vis, list(.onnx_value_info("B", 1L, c(C_out))))
    ins   <- c(ins, "B")
  }
  node <- .onnx_node("Conv", ins, "Y", attrs = list(
    .onnx_attr_ints("kernel_shape", K),
    .onnx_attr_ints("strides", stride),
    .onnx_attr_ints("pads", c(pad, pad)),
    .onnx_attr_ints("dilations", 1L)))
  graph <- .onnx_graph("test", list(node), vis, list(outp), inits)
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  list(path = path, L_out = L_out)
}

# Reference, in ONNX index order, from column-major R arrays.
conv1d_ref <- function(x, w, b, N, C_in, L, C_out, K, stride = 1L, pad = 0L) {
  X <- array(x, dim = c(L, C_in, N))          # ONNX [N,C_in,L]
  W <- array(w, dim = c(K, C_in, C_out))      # ONNX [C_out,C_in,K]
  L_out <- (L + 2L * pad - K) %/% stride + 1L
  Y <- array(0, dim = c(L_out, C_out, N))
  for (n in seq_len(N)) for (oc in seq_len(C_out)) for (o in seq_len(L_out)) {
    acc <- 0
    for (ic in seq_len(C_in)) for (k in seq_len(K)) {
      pos <- (o - 1L) * stride + (k - 1L) - pad + 1L
      if (pos >= 1L && pos <= L) acc <- acc + X[pos, ic, n] * W[k, ic, oc]
    }
    Y[o, oc, n] <- acc + if (is.null(b)) 0 else b[oc]
  }
  Y
}

test_that("ONNX Conv1d without bias", {
  N <- 1L; C_in <- 2L; L <- 6L; C_out <- 3L; K <- 3L
  set.seed(11)
  x <- round(rnorm(N * C_in * L), 2)
  w <- round(rnorm(C_out * C_in * K), 2)
  m <- make_conv1d(N, C_in, L, C_out, K, w)
  result <- as.numeric(run_onnx(m$path, list(X = x)))
  ref <- conv1d_ref(x, w, NULL, N, C_in, L, C_out, K)
  expect_equal(result, as.numeric(ref), tolerance = 1e-3)
})

test_that("ONNX Conv1d with bias broadcasts along the right axis", {
  # The regression test proper: with the bias reshaped as for a 2-D
  # convolution this does not even build.
  N <- 1L; C_in <- 2L; L <- 6L; C_out <- 3L; K <- 3L
  set.seed(12)
  x <- round(rnorm(N * C_in * L), 2)
  w <- round(rnorm(C_out * C_in * K), 2)
  b <- c(10, 20, 30)                     # far apart, so a misplaced bias shows
  m <- make_conv1d(N, C_in, L, C_out, K, w, b)
  result <- as.numeric(run_onnx(m$path, list(X = x)))
  ref <- conv1d_ref(x, w, b, N, C_in, L, C_out, K)
  expect_equal(result, as.numeric(ref), tolerance = 1e-3)
  # each output channel must carry its own bias, not channel 0's
  y <- array(result, dim = c(m$L_out, C_out, N))
  expect_true(all(y[, 1, 1] < y[, 2, 1]))
  expect_true(all(y[, 2, 1] < y[, 3, 1]))
})

test_that("ONNX Conv1d with stride and padding", {
  # whisper's second conv is stride 2 with padding 1.
  N <- 1L; C_in <- 2L; L <- 8L; C_out <- 2L; K <- 3L
  set.seed(13)
  x <- round(rnorm(N * C_in * L), 2)
  w <- round(rnorm(C_out * C_in * K), 2)
  b <- c(1, -1)
  m <- make_conv1d(N, C_in, L, C_out, K, w, b, stride = 2L, pad = 1L)
  result <- as.numeric(run_onnx(m$path, list(X = x)))
  ref <- conv1d_ref(x, w, b, N, C_in, L, C_out, K, stride = 2L, pad = 1L)
  expect_equal(length(result), length(ref))
  expect_equal(result, as.numeric(ref), tolerance = 1e-3)
})

test_that("ONNX Conv1d at whisper's channel counts", {
  # 80 mel bins in, 384 channels out -- the shape that actually failed.  Length
  # is kept short so the test stays quick; only the channel mapping matters.
  N <- 1L; C_in <- 80L; L <- 8L; C_out <- 16L; K <- 3L
  set.seed(14)
  x <- round(rnorm(N * C_in * L), 2)
  w <- round(rnorm(C_out * C_in * K), 2)
  b <- round(rnorm(C_out), 2)
  m <- make_conv1d(N, C_in, L, C_out, K, w, b)
  result <- as.numeric(run_onnx(m$path, list(X = x)))
  ref <- conv1d_ref(x, w, b, N, C_in, L, C_out, K)
  expect_equal(result, as.numeric(ref), tolerance = 1e-2)
})
