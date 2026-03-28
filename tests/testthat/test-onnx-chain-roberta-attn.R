# Chain tests: RoBERTa attention pattern (Softmax with -inf mask)
# MatMul → Scale → Add(-inf mask) → Softmax → MatMul
#
# Tests the pattern that causes NaN in roberta-9:
# attention_scores + mask(-inf for padding) → Softmax
# If -inf propagates incorrectly, softmax gets NaN.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: Add(-inf) → Softmax on 2D ─────────────────────

test_that("chain roberta-attn: Add(-inf mask)→Softmax 2D (minimal)", {
  # X[2,4] + mask[2,4] → Softmax(axis=1)
  # mask has -inf for "padding" positions
  inp  <- .onnx_value_info("X", 1L, c(2L, 4L))
  mask <- .onnx_value_info("M", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L))

  add_node <- .onnx_node("Add", c("X", "M"), "masked")
  sm_node  <- .onnx_node("Softmax", "masked", "Y",
                          attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test", list(add_node, sm_node),
                        list(inp, mask), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # Scores: row 0 = [1,2,3,4], row 1 = [0.5,0.5,0.5,0.5]
  x <- c(1, 2, 3, 4, 0.5, 0.5, 0.5, 0.5)
  # Mask: row 0 all valid (0), row 1 last 2 masked (-1e9 as proxy for -inf)
  m <- c(0, 0, 0, 0, 0, 0, -1e9, -1e9)

  result <- run_onnx(path, list(X = x, M = m))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  expect_true(all(is.finite(r)))
  # Row 0: normal softmax of [1,2,3,4]
  # Row 1: softmax of [0.5, 0.5, -1e9+0.5, -1e9+0.5] ≈ [0.5, 0.5, 0, 0]
  expect_true(all(r >= 0))
  # Each row should sum to ~1
  expect_equal(sum(r[1:4]), 1.0, tolerance = 1e-3)
  expect_equal(sum(r[5:8]), 1.0, tolerance = 1e-3)
})

# ── Real: MatMul(Q,K) → Scale → Add(mask) → Softmax → MatMul(attn,V) ──

test_that("chain roberta-attn: QK→Scale→Mask→Softmax→V (attention)", {
  # Simplified single-head attention: Q[1,4,8] K[1,4,8] V[1,4,8]
  # attn_scores = Q @ K^T → [1,4,4]
  # scaled = attn_scores * (1/sqrt(8))
  # masked = scaled + mask
  # attn_probs = softmax(masked, axis=-1)
  # output = attn_probs @ V → [1,4,8]

  # Flatten to 2D for simplicity: Q[4,8], K[4,8], V[4,8]
  q_inp  <- .onnx_value_info("Q", 1L, c(4L, 8L))
  k_inp  <- .onnx_value_info("K", 1L, c(4L, 8L))
  v_inp  <- .onnx_value_info("V", 1L, c(4L, 8L))
  mask_inp <- .onnx_value_info("M", 1L, c(4L, 4L))
  outp   <- .onnx_value_info("Y", 1L, c(4L, 8L))

  # Scale constant: 1/sqrt(8)
  scale_raw <- .float_bytes(1.0 / sqrt(8))
  scale_t  <- .onnx_tensor("sc", c(1L), 1L, scale_raw)
  scale_vi <- .onnx_value_info("sc", 1L, c(1L))

  # Q @ K^T → [4,4]
  # In ONNX: MatMul(Q[4,8], K^T[8,4]) but we need Transpose first
  trans_node <- .onnx_node("Transpose", "K", "KT",
                            attrs = list(.onnx_attr_ints("perm", c(1L, 0L))))
  mm1_node <- .onnx_node("MatMul", c("Q", "KT"), "scores")
  # Scale
  mul_node <- .onnx_node("Mul", c("scores", "sc"), "scaled")
  # Add mask
  add_node <- .onnx_node("Add", c("scaled", "M"), "masked")
  # Softmax on last axis
  sm_node  <- .onnx_node("Softmax", "masked", "probs",
                          attrs = list(.onnx_attr_int("axis", 1L)))
  # attn_probs @ V → [4,8]
  mm2_node <- .onnx_node("MatMul", c("probs", "V"), "Y")

  graph <- .onnx_graph("test",
    list(trans_node, mm1_node, mul_node, add_node, sm_node, mm2_node),
    list(q_inp, k_inp, v_inp, mask_inp, scale_vi),
    list(outp), list(scale_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(42)
  q <- rnorm(32, 0, 0.5)
  k <- rnorm(32, 0, 0.5)
  v <- rnorm(32, 0, 0.5)
  # Mask: all valid (0) except position 3 is masked
  m <- matrix(0, 4, 4)
  m[, 4] <- -1e9  # mask last token
  m_vec <- as.numeric(t(m))

  result <- run_onnx(path, list(Q = q, K = k, V = v, M = m_vec))
  r <- as.numeric(result)
  expect_equal(length(r), 32)
  expect_true(all(is.finite(r)))
})

# ── Boundary: all positions masked except one ───────────────

test_that("chain roberta-attn: single unmasked position (boundary)", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 4L))
  mask <- .onnx_value_info("M", 1L, c(1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  add_node <- .onnx_node("Add", c("X", "M"), "masked")
  sm_node  <- .onnx_node("Softmax", "masked", "Y",
                          attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test", list(add_node, sm_node),
                        list(inp, mask), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # Only position 0 is unmasked
  x <- c(1, 2, 3, 4)
  m <- c(0, -1e9, -1e9, -1e9)

  result <- run_onnx(path, list(X = x, M = m))
  r <- as.numeric(result)
  expect_true(all(is.finite(r)))
  # softmax should put ~1.0 on position 0
  expect_equal(r[1], 1.0, tolerance = 1e-3)
  expect_true(all(r[2:4] < 1e-3))
})
