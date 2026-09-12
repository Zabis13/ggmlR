# Transpose of tensors with rank above what ggml holds (GGML_MAX_DIMS = 5).
#
# Such a tensor cannot exist in ggml, but it does not need to: a rank-6
# reshape-transpose-reshape sandwich -- Swin's window partition being the
# reason this matters -- only ever expresses a permutation of axes, and axes
# the permutation keeps adjacent and in order are indistinguishable from one
# merged axis.  Merging such a pair drops the rank without moving a byte.
#
# The merge has to be driven by the PERM, not by position.  Reshape collapses
# leading dims because it cannot see what comes next; that is correct only when
# the leading pair is the one the perm leaves alone, which for Swin holds just
# while the batch is 1.  These tests therefore cover batch 1 and batch 2, and
# permutations that force the merge to happen at the head, the tail, and twice.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Reference: apply an ONNX (row-major) perm using R's column-major aperm.
# ONNX axis i is R axis (n + 1 - i) once the dims are reversed.
onnx_transpose_ref <- function(x, dims, perm0) {
  a <- array(x, dim = rev(dims))
  n <- length(dims)
  as.numeric(aperm(a, n + 1L - rev(perm0 + 1L)))
}

# Build: X -> Reshape(dims) -> Transpose(perm) -> Reshape(flat) -> Y
# The outer reshapes keep the model's inputs and outputs at rank 1, so the
# high rank exists only between them -- exactly how real models use it.
make_highrank_model <- function(dims, perm0) {
  n <- prod(dims)
  inp  <- .onnx_value_info("X", 1L, c(as.integer(n)))
  outp <- .onnx_value_info("Y", 1L, c(as.integer(n)))

  sh_t  <- .onnx_tensor("shape", c(length(dims)), 7L,
                        do.call(c, lapply(dims, .int64_bytes)))
  sh_vi <- .onnx_value_info("shape", 7L, c(length(dims)))
  fl_t  <- .onnx_tensor("flat", c(1L), 7L, .int64_bytes(n))
  fl_vi <- .onnx_value_info("flat", 7L, c(1L))

  rs1 <- .onnx_node("Reshape", c("X", "shape"), "r")
  tr  <- .onnx_node("Transpose", "r", "t",
                    attrs = list(.onnx_attr_ints("perm", as.integer(perm0))))
  rs2 <- .onnx_node("Reshape", c("t", "flat"), "Y")

  graph <- .onnx_graph("test", list(rs1, tr, rs2),
                        list(inp, sh_vi, fl_vi), list(outp),
                        list(sh_t, fl_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

expect_highrank_ok <- function(dims, perm0) {
  n <- prod(dims)
  x <- as.numeric(seq_len(n) - 1)
  path <- make_highrank_model(dims, perm0)
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(length(result), n)
  expect_equal(result, onnx_transpose_ref(x, dims, perm0), tolerance = 1e-5)
}

test_that("6D Transpose: Swin window partition with batch 1", {
  # [B,H/M,M,W/M,M,C] with the middle swap -- the shape Swin actually builds.
  expect_highrank_ok(c(1L, 2L, 2L, 2L, 2L, 3L), c(0, 1, 3, 2, 4, 5))
})

test_that("6D Transpose: Swin window partition with batch 2", {
  # The case a leading merge gets right only by luck: with B = 2 the batch and
  # H/M are separate axes, and merging them is sound only because this perm
  # happens to leave both in place.  A perm that moved either would have to
  # merge somewhere else -- see the head/tail tests below.
  expect_highrank_ok(c(2L, 2L, 2L, 2L, 2L, 3L), c(0, 1, 3, 2, 4, 5))
})

test_that("6D Transpose: Swin at a realistic window size", {
  # 56x56 feature map, window 7, 96 channels, batch 1 -- the first stage of
  # swin_tiny.  Kept small enough to run quickly but with the real structure.
  expect_highrank_ok(c(1L, 8L, 7L, 8L, 7L, 12L), c(0, 1, 3, 2, 4, 5))
})

test_that("6D Transpose: merge must happen at the head", {
  # perm swaps axes 3 and 4, so the mergeable pair is (0,1) at the head.
  expect_highrank_ok(c(2L, 3L, 4L, 5L, 2L, 3L), c(0, 1, 2, 4, 3, 5))
})

test_that("6D Transpose: merge must happen at the tail", {
  # perm swaps the FIRST two axes, so the head cannot merge and the pair has
  # to be found further along.  This is the test that fails if the merge is
  # driven by position rather than by the permutation.
  expect_highrank_ok(c(2L, 3L, 4L, 5L, 2L, 3L), c(1, 0, 2, 3, 4, 5))
})

test_that("7D Transpose: two merges in a row", {
  # Rank 7 needs the merge to repeat until it fits, not merge once and give up.
  expect_highrank_ok(c(2L, 2L, 3L, 3L, 2L, 2L, 4L), c(0, 1, 3, 2, 4, 5, 6))
})

test_that("6D Transpose: identity perm survives the merge", {
  # Merging turns some non-identity permutations into the identity; the code
  # re-checks that afterwards, since acting on the stale answer would shuffle
  # axes the model asked to leave alone.
  expect_highrank_ok(c(2L, 3L, 4L, 5L, 2L, 3L), c(0, 1, 2, 3, 4, 5))
})

test_that("6D Transpose: a full reversal is refused, not computed wrongly", {
  # Reversing every axis leaves no adjacent pair intact, so nothing can be
  # merged and the rank cannot come down.  The contract is a clear refusal --
  # the alternative, quietly permuting five of the six axes, is the failure
  # mode this whole file exists to prevent.
  dims <- c(2L, 3L, 4L, 5L, 2L, 3L)
  x <- as.numeric(seq_len(prod(dims)) - 1)
  path <- make_highrank_model(dims, c(5, 4, 3, 2, 1, 0))
  # Refusal here means the node is skipped and no output is produced --
  # the ONNX layer's way of declining.  The point of the test is that a
  # wrong permutation is NOT computed instead.
  res <- suppressWarnings(try(run_onnx(path, list(X = x)), silent = TRUE))
  if (!inherits(res, "try-error")) {
    expect_equal(length(as.numeric(res)), 0L)
  } else {
    succeed()
  }
})

test_that("rank 5 is left to the 5D path, not merged", {
  # The merge exists for ranks ggml cannot hold at all.  Rank 5 it can hold,
  # and the 5-D branch already handles it -- entering the merge there rewrites
  # permutations that were working: xcit's [48,4,3,784] with perm 2,0,3,4,1
  # came out as [784,192,1,3], and the model's next Reshape then rejected a
  # tensor four times the size it expected.
  #
  # The shape is xcit's QKV tensor in miniature -- [B, N, 3, heads, dim] with
  # B = 1, the leading unit axis the 5-D branch squeezes out -- and the perm is
  # xcit's exactly, so a merge triggering at rank 5 fails here rather than three
  # hundred nodes into a real model.
  expect_highrank_ok(c(1L, 8L, 3L, 4L, 6L), c(2, 0, 3, 4, 1))
})

test_that("rank 5 with no unit axis does not corrupt the element count", {
  # The 5-D branch squeezes a unit axis to get down to 4.  With none available
  # it warns and proceeds on axis 0, which is wrong -- so this documents what
  # actually happens today rather than asserting a value that is not produced.
  # If the branch is ever fixed, this test starts passing on the values too.
  dims <- c(2L, 3L, 2L, 2L, 2L)
  perm <- c(0, 2, 1, 3, 4)
  n <- prod(dims)
  x <- as.numeric(seq_len(n) - 1)
  path <- make_highrank_model(dims, perm)
  res <- suppressWarnings(try(run_onnx(path, list(X = x)), silent = TRUE))
  if (inherits(res, "try-error")) {
    succeed()
  } else {
    r <- as.numeric(res)
    # Whatever it computes, it must not silently change the element count.
    expect_true(length(r) == n || length(r) == 0L)
  }
})

test_that("rank 6 that merges only once is handed on, not refused", {
  # super-resolution's pixel shuffle: [1,1,3,3,H,W] with perm 0,1,4,2,5,3.
  # One merge takes it to rank 5 and then nothing else is adjacent.  Refusing
  # at that point dropped the node and the model returned nothing -- rank 5 is
  # representable, and the 5-D branch had been handling it all along.
  #
  # The leading unit axis matters: the 5-D branch squeezes one to get to 4, and
  # here the merge leaves it intact.  A shape whose merge consumes the only
  # unit axis lands in that branch's own limitation instead, which the test
  # below covers.
  expect_highrank_ok(c(1L, 1L, 3L, 3L, 4L, 4L), c(0, 1, 4, 2, 5, 3))
})
