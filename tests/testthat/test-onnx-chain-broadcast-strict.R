# Chain tests: Strict broadcast validation (XCiT pattern)
#
# Tests broadcast with non-divisible dimensions that cause ggml_repeat to assert.
# XCiT fails with dim mismatch a=28, b=32 in Add/Mul.
# The fix should either pad, resize, or handle non-divisible broadcast gracefully.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: broadcast Add where dims are divisible ───────────────

test_that("chain broadcast-strict: Add [1,4,8] + [1,1,8] (standard broadcast)", {
  inp_a <- .onnx_value_info("A", 1L, c(1L, 4L, 8L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 1L, 8L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 4L, 8L))

  add_node <- .onnx_node("Add", c("A", "B"), "Y")

  graph <- .onnx_graph("test", list(add_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- rnorm(32)
  b <- rnorm(8)
  result <- run_onnx(path, list(A = a, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 32)
  expect_true(all(is.finite(r)))
})

# ── Add with divisible spatial broadcast [1,C,H,W] + [1,C,1,1] ──

test_that("chain broadcast-strict: Add [1,3,28,28] + [1,3,1,1] (channel bias)", {
  inp_a <- .onnx_value_info("A", 1L, c(1L, 3L, 28L, 28L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 3L, 1L, 1L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 3L, 28L, 28L))

  add_node <- .onnx_node("Add", c("A", "B"), "Y")

  graph <- .onnx_graph("test", list(add_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- rnorm(1*3*28*28)
  b <- rnorm(3)
  result <- run_onnx(path, list(A = a, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 1*3*28*28)
  expect_true(all(is.finite(r)))
})

# ── Mul broadcast with different spatial dims (XCiT-like) ────────

test_that("chain broadcast-strict: Mul [1,1,28,28] * [1,1,28,1] (row scale)", {
  inp_a <- .onnx_value_info("A", 1L, c(1L, 1L, 28L, 28L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 1L, 28L, 1L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 1L, 28L, 28L))

  mul_node <- .onnx_node("Mul", c("A", "B"), "Y")

  graph <- .onnx_graph("test", list(mul_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- rnorm(784)
  b <- rnorm(28)
  result <- run_onnx(path, list(A = a, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 784)
  expect_true(all(is.finite(r)))
})

# ── XCiT attention pattern: MatMul with batch broadcast ──────────

test_that("chain broadcast-strict: MatMul [B,N,C] x [B,C,N] (XCiT self-attention)", {
  # B=2, N=28 tokens, C=32 dim (these are the dims that cause XCiT issues)
  inp_q <- .onnx_value_info("Q", 1L, c(2L, 28L, 32L))
  inp_k <- .onnx_value_info("K", 1L, c(2L, 32L, 28L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 28L, 28L))

  mm_node <- .onnx_node("MatMul", c("Q", "K"), "Y")

  graph <- .onnx_graph("test", list(mm_node),
                        list(inp_q, inp_k), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  q <- rnorm(2*28*32)
  k <- rnorm(2*32*28)
  result <- run_onnx(path, list(Q = q, K = k))
  r <- as.numeric(result)
  expect_equal(length(r), 2*28*28)
  expect_true(all(is.finite(r)))
})

# ── Non-divisible broadcast: Add [28] + [32] (the exact XCiT failure) ──

test_that("chain broadcast-strict: Add with non-divisible dims [1,28] + [1,32] (expect error or pad)", {
  # This is the pattern that crashes XCiT: 28 and 32 are not divisible
  # ggml_repeat asserts ne[d] % src_ne[d] == 0
  inp_a <- .onnx_value_info("A", 1L, c(1L, 28L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 32L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 32L))

  add_node <- .onnx_node("Add", c("A", "B"), "Y")

  graph <- .onnx_graph("test", list(add_node),
                        list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- rnorm(28)
  b <- rnorm(32)
  # This should either work (if we handle non-divisible broadcast)
  # or give a clear error (not a crash)
  result <- tryCatch(
    run_onnx(path, list(A = a, B = b)),
    error = function(e) e
  )
  # For now, we expect this to fail — document the failure mode
  if (inherits(result, "error")) {
    expect_true(grepl("abort|assert|repeat", tolower(result$message)))
  } else {
    r <- as.numeric(result)
    expect_equal(length(r), 32)
    expect_true(all(is.finite(r)))
  }
})

# ── Transpose → Add pattern (common source of dim mismatch) ─────

test_that("chain broadcast-strict: Transpose + Add with broadcast (attention score bias)", {
  # scores[2,8,28,28] + bias[1,1,28,28] → broadcast should work
  inp_s <- .onnx_value_info("S", 1L, c(2L, 8L, 28L, 28L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 1L, 28L, 28L))
  outp  <- .onnx_value_info("Y", 1L, c(2L, 8L, 28L, 28L))

  add_node <- .onnx_node("Add", c("S", "B"), "Y")

  graph <- .onnx_graph("test", list(add_node),
                        list(inp_s, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  s <- rnorm(2*8*28*28)
  b <- rnorm(28*28)
  result <- run_onnx(path, list(S = s, B = b))
  r <- as.numeric(result)
  expect_equal(length(r), 2*8*28*28)
  expect_true(all(is.finite(r)))
})
