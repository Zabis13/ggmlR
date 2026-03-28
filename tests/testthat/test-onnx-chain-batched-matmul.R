# Chain tests: 3D batched MatMul (CaiT pattern)
# MatMul with tensors [B, N, C] x [B, C, M] → [B, N, M]
#
# Tests the pattern that causes cait_xs24_384 to fail:
# ggml_get_rows assert ne[2] != ne[1] when Gather feeds 3D tensors,
# and batched MatMul where batch dims need broadcast.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: 3D MatMul [B,M,K] x [B,K,N] ──────────────────

test_that("chain batched-matmul: 3D MatMul [2,3,4]x[2,4,5] (minimal)", {
  # A[2,3,4] @ B[2,4,5] → Y[2,3,5]
  inp_a <- .onnx_value_info("A", 1L, c(2L, 3L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(2L, 4L, 5L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 3L, 5L))

  mm_node <- .onnx_node("MatMul", c("A", "B"), "Y")

  graph <- .onnx_graph("test", list(mm_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(42)
  a <- rnorm(24)  # 2*3*4
  b <- rnorm(40)  # 2*4*5
  result <- run_onnx(path, list(A = a, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 30)  # 2*3*5
  expect_true(all(is.finite(r)))
})

# ── 3D MatMul with batch broadcast: [1,M,K] x [B,K,N] ─────

test_that("chain batched-matmul: broadcast [1,3,4]x[2,4,5] (batch broadcast)", {
  # A[1,3,4] broadcast over B[2,4,5] → Y[2,3,5]
  inp_a <- .onnx_value_info("A", 1L, c(1L, 3L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(2L, 4L, 5L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 3L, 5L))

  mm_node <- .onnx_node("MatMul", c("A", "B"), "Y")

  graph <- .onnx_graph("test", list(mm_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(42)
  a <- rnorm(12)  # 1*3*4
  b <- rnorm(40)  # 2*4*5
  result <- run_onnx(path, list(A = a, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 30)  # 2*3*5
  expect_true(all(is.finite(r)))
})

# ── Real: Reshape→MatMul→Reshape (CaiT self-attention) ─────

test_that("chain batched-matmul: Reshape→3D MatMul→Reshape (CaiT pattern)", {
  # Simulate: flatten [B,H,N,C] → [B*H, N, C], batched matmul, reshape back
  # Input X: [1, 2, 4, 8] (B=1, H=2 heads, N=4 tokens, C=8 dim)
  # Reshape to [2, 4, 8]
  # MatMul with K^T: [2, 8, 4] → scores [2, 4, 4]
  # Softmax → MatMul with V [2, 4, 8] → [2, 4, 8]
  # Reshape to [1, 2, 4, 8]

  inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 8L))
  k_inp <- .onnx_value_info("K", 1L, c(2L, 4L, 8L))
  v_inp <- .onnx_value_info("V", 1L, c(2L, 4L, 8L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 8L))

  # Reshape target shapes
  shape3_raw <- unlist(lapply(c(2L, 4L, 8L), .int64_bytes))
  shape3_t  <- .onnx_tensor("s3", c(3L), 7L, shape3_raw)
  shape3_vi <- .onnx_value_info("s3", 7L, c(3L))

  shape4_raw <- unlist(lapply(c(1L, 2L, 4L, 8L), .int64_bytes))
  shape4_t  <- .onnx_tensor("s4", c(4L), 7L, shape4_raw)
  shape4_vi <- .onnx_value_info("s4", 7L, c(4L))

  # X[1,2,4,8] → reshape → [2,4,8]
  r1_node <- .onnx_node("Reshape", c("X", "s3"), "Q")
  # Transpose K: [2,4,8] → [2,8,4]
  tr_node <- .onnx_node("Transpose", "K", "KT",
                          attrs = list(.onnx_attr_ints("perm", c(0L, 2L, 1L))))
  # Q[2,4,8] @ KT[2,8,4] → scores[2,4,4]
  mm1_node <- .onnx_node("MatMul", c("Q", "KT"), "scores")
  # Softmax on last axis
  sm_node <- .onnx_node("Softmax", "scores", "probs",
                          attrs = list(.onnx_attr_int("axis", 2L)))
  # probs[2,4,4] @ V[2,4,8] → [2,4,8]
  mm2_node <- .onnx_node("MatMul", c("probs", "V"), "out3d")
  # Reshape [2,4,8] → [1,2,4,8]
  r2_node <- .onnx_node("Reshape", c("out3d", "s4"), "Y")

  graph <- .onnx_graph("test",
    list(r1_node, tr_node, mm1_node, sm_node, mm2_node, r2_node),
    list(inp, k_inp, v_inp, shape3_vi, shape4_vi),
    list(outp),
    list(shape3_t, shape4_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(42)
  x <- rnorm(64)   # 1*2*4*8
  k <- rnorm(64)   # 2*4*8
  v <- rnorm(64)   # 2*4*8
  result <- run_onnx(path, list(X = x, K = k, V = v))
  r <- as.numeric(result)
  expect_equal(length(r), 64)
  expect_true(all(is.finite(r)))
})

# ── Boundary: batch=1 3D MatMul (degenerate batch) ─────────

test_that("chain batched-matmul: batch=1 [1,3,4]x[1,4,2] (boundary)", {
  inp_a <- .onnx_value_info("A", 1L, c(1L, 3L, 4L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 4L, 2L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 3L, 2L))

  mm_node <- .onnx_node("MatMul", c("A", "B"), "mm")
  relu_node <- .onnx_node("Relu", "mm", "Y")

  graph <- .onnx_graph("test", list(mm_node, relu_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)  # ~identity-ish
  b <- c(1, -1, 2, -2, 3, -3, 4, -4)
  result <- run_onnx(path, list(A = a, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 6)  # 1*3*2
  expect_true(all(r >= 0))  # Relu
  expect_true(all(is.finite(r)))
})
