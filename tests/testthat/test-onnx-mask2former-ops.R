# Ops that Mask2Former needs and nothing else in the model zoo did:
# CumSum (positional embedding), InstanceNormalization (FPN decoder),
# Einsum (the mask head's final contraction).
#
# Each is checked against a reference computed elementwise in R, not against
# a length: every one of these is a rearrangement as much as an arithmetic,
# and the failure mode is right values in the wrong places.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── CumSum ───────────────────────────────────────────────────────

make_cumsum <- function(dims, axis) {
  n <- prod(dims)
  inp  <- .onnx_value_info("X", 1L, as.integer(dims))
  outp <- .onnx_value_info("Y", 1L, as.integer(dims))
  ax_t  <- .onnx_tensor("axis", integer(0), 6L,
                        writeBin(as.integer(axis), raw(), size = 4))
  ax_vi <- .onnx_value_info("axis", 6L, integer(0))
  node  <- .onnx_node("CumSum", c("X", "axis"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp, ax_vi), list(outp), list(ax_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

test_that("ONNX CumSum along the last axis", {
  # ONNX [2,4] axis=1 is ggml ne[0] -- the fast path, straight ggml_cumsum.
  dims <- c(2L, 4L)
  set.seed(2); x <- round(rnorm(prod(dims)), 2)
  result <- as.numeric(run_onnx(make_cumsum(dims, 1L), list(X = x)))
  # ONNX row-major [2,4] is column-major [4,2]: rows of ONNX are columns here.
  a <- matrix(x, nrow = 4)
  ref <- apply(a, 2, cumsum)
  expect_equal(result, as.numeric(ref), tolerance = 1e-5)
})

test_that("ONNX CumSum along an intermediate axis", {
  # ONNX [1,3,4] axis=1 maps to ggml dim 1, so the axis has to be rotated to
  # ne[0] and back -- the case that a last-axis-only implementation gets wrong.
  dims <- c(1L, 3L, 4L)
  set.seed(3); x <- round(rnorm(prod(dims)), 2)
  result <- as.numeric(run_onnx(make_cumsum(dims, 1L), list(X = x)))
  a <- array(x, dim = c(4, 3, 1))          # ggml [W,H,N]
  ref <- a
  for (w in 1:4) ref[w, , 1] <- cumsum(a[w, , 1])
  expect_equal(result, as.numeric(ref), tolerance = 1e-5)
})

test_that("ONNX CumSum accepts a negative axis", {
  dims <- c(2L, 4L)
  set.seed(2); x <- round(rnorm(prod(dims)), 2)
  a <- as.numeric(run_onnx(make_cumsum(dims, -1L), list(X = x)))
  b <- as.numeric(run_onnx(make_cumsum(dims,  1L), list(X = x)))
  expect_equal(a, b, tolerance = 1e-6)
})

# ── InstanceNormalization ────────────────────────────────────────

test_that("ONNX InstanceNormalization normalises each (batch, channel) plane", {
  N <- 2L; C <- 3L; H <- 2L; W <- 2L; eps <- 1e-5
  inp   <- .onnx_value_info("X", 1L, c(N, C, H, W))
  outp  <- .onnx_value_info("Y", 1L, c(N, C, H, W))
  scale <- c(2, 0.5, -1); bias <- c(0.1, -0.2, 0.3)
  s_t   <- .onnx_tensor("s", c(C), 1L, .float_bytes(scale))
  b_t   <- .onnx_tensor("b", c(C), 1L, .float_bytes(bias))
  s_vi  <- .onnx_value_info("s", 1L, c(C))
  b_vi  <- .onnx_value_info("b", 1L, c(C))
  node  <- .onnx_node("InstanceNormalization", c("X", "s", "b"), "Y",
                      attrs = list(.onnx_attr_float("epsilon", eps)))
  graph <- .onnx_graph("test", list(node), list(inp, s_vi, b_vi), list(outp),
                        list(s_t, b_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(4); x <- round(rnorm(N * C * H * W), 2)
  result <- as.numeric(run_onnx(path, list(X = x)))

  # ONNX [N,C,H,W] is column-major [W,H,C,N]; each (channel, batch) plane is
  # normalised over its own H*W values, with the population variance.
  a <- array(x, dim = c(W, H, C, N))
  ref <- a
  for (n in seq_len(N)) for (cc in seq_len(C)) {
    v <- a[, , cc, n]
    m <- mean(v); s <- sqrt(mean((v - m)^2) + eps)
    ref[, , cc, n] <- (v - m) / s * scale[cc] + bias[cc]
  }
  expect_equal(result, as.numeric(ref), tolerance = 1e-4)
})

# ── Einsum ───────────────────────────────────────────────────────

test_that("ONNX Einsum bqc,bchw->bqhw contracts the channel axis", {
  # Mask2Former's mask head: query embeddings against a per-pixel embedding.
  B <- 1L; Q <- 2L; C <- 3L; H <- 2L; W <- 2L
  inp_a <- .onnx_value_info("A", 1L, c(B, Q, C))
  inp_b <- .onnx_value_info("Bm", 1L, c(B, C, H, W))
  outp  <- .onnx_value_info("Y", 1L, c(B, Q, H, W))
  node  <- .onnx_node("Einsum", c("A", "Bm"), "Y",
                      attrs = list(.onnx_attr_string("equation", "bqc,bchw->bqhw")))
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(9)
  av <- round(rnorm(B * Q * C), 2)
  bv <- round(rnorm(B * C * H * W), 2)
  result <- as.numeric(run_onnx(path, list(A = av, Bm = bv)))

  # Column-major views: A is [C,Q,B], B is [W,H,C,B], output is [W,H,Q,B].
  # Checking the values elementwise matters here: the contraction itself is a
  # matmul that also comes out transposed, so a length check would pass on an
  # answer with every number in the wrong place.
  aA <- array(av, dim = c(C, Q, B))
  bB <- array(bv, dim = c(W, H, C, B))
  ref <- array(0, dim = c(W, H, Q, B))
  for (bb in seq_len(B)) for (q in seq_len(Q))
    for (h in seq_len(H)) for (w in seq_len(W))
      ref[w, h, q, bb] <- sum(aA[, q, bb] * bB[w, h, , bb])

  expect_equal(length(result), B * Q * H * W)
  expect_equal(result, as.numeric(ref), tolerance = 1e-4)
})

test_that("ONNX Einsum refuses an equation it does not implement", {
  # The contract is a named refusal, not an approximation: a general einsum
  # would need a parser, and quietly computing the nearest matmul is how a
  # model gets plausible wrong numbers.
  inp_a <- .onnx_value_info("A", 1L, c(2L, 3L))
  inp_b <- .onnx_value_info("Bm", 1L, c(3L, 2L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 2L))
  node  <- .onnx_node("Einsum", c("A", "Bm"), "Y",
                      attrs = list(.onnx_attr_string("equation", "ij,jk->ikj")))
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  # An unsupported op is not an R error: map_node returns -1, the ONNX
  # layer skips the node by design, and the output simply never gets
  # produced.  What must NOT happen is a plausible wrong answer.
  res <- suppressWarnings(try(
    onnx_run(onnx_load(path, device = "cpu"), list(A = rnorm(6), Bm = rnorm(6))),
    silent = TRUE))
  if (!inherits(res, "try-error")) {
    expect_equal(length(as.numeric(res[[1]])), 0L)
  } else {
    succeed()
  }
})
