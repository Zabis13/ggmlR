# Chain tests: XCiT dynamic shape pattern
# Shape → Slice → Reshape where Slice can produce dim=0
# Also: Gather on 3D tensors (class attention token)
#
# Tests the pattern that causes xcit_tiny to fail:
# broadcast dim 0: a=1, b=0 — a Reshape/Squeeze collapses
# a dimension to 0, killing downstream broadcast.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: Shape→Gather→Reshape (dynamic reshape) ────────

test_that("chain xcit-dynamic: Shape→Gather→Reshape (minimal)", {
  # X[2,3] → Shape → Gather(idx=0) → gets dim=2
  # Then use gathered dim in Reshape: X → Reshape([2,3]) ≡ identity
  inp <- .onnx_value_info("X", 1L, c(2L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 3L))

  # Target shape as initializer (static, so no dynamic Shape needed)
  shape_raw <- unlist(lapply(c(2L, 3L), .int64_bytes))
  shape_t  <- .onnx_tensor("sh", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("sh", 7L, c(2L))

  r_node <- .onnx_node("Reshape", c("X", "sh"), "r")
  relu_node <- .onnx_node("Relu", "r", "Y")

  graph <- .onnx_graph("test", list(r_node, relu_node),
                        list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-1, 2, -3, 4, -5, 6)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 6)
  expect_equal(r, pmax(x, 0), tolerance = 1e-5)
})

# ── Real: Reshape→Transpose→MatMul (XCiT class attention) ──

test_that("chain xcit-dynamic: Reshape→Transpose→MatMul (class attention)", {
  # Simulates XCiT class attention:
  # X[1,4,8] (B=1, N=4 tokens, C=8)
  # Reshape to [1,4,2,4] (split heads: H=2, D=4)
  # Transpose to [1,2,4,4] (B,H,N,D)
  # MatMul with itself transposed for attention

  inp  <- .onnx_value_info("X", 1L, c(1L, 4L, 8L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))

  # Reshape target: [1,4,2,4]
  shape_raw <- unlist(lapply(c(1L, 4L, 2L, 4L), .int64_bytes))
  shape_t  <- .onnx_tensor("sh", c(4L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("sh", 7L, c(4L))

  r_node  <- .onnx_node("Reshape", c("X", "sh"), "r4d")
  # [1,4,2,4] → [1,2,4,4] (B,H,N,D)
  tr_node <- .onnx_node("Transpose", "r4d", "Q",
                          attrs = list(.onnx_attr_ints("perm", c(0L, 2L, 1L, 3L))))
  # Transpose K: [1,2,4,4] → [1,2,4,4] (already square, just transpose last 2)
  trk_node <- .onnx_node("Transpose", "Q", "KT",
                           attrs = list(.onnx_attr_ints("perm", c(0L, 1L, 3L, 2L))))
  # Q @ KT → scores [1,2,4,4]
  mm_node <- .onnx_node("MatMul", c("Q", "KT"), "Y")

  graph <- .onnx_graph("test",
    list(r_node, tr_node, trk_node, mm_node),
    list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(42)
  x <- rnorm(32)  # 1*4*8
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 32)  # 1*2*4*4
  expect_true(all(is.finite(r)))
})

# ── Squeeze→Unsqueeze (dim manipulation) ────────────────────

test_that("chain xcit-dynamic: Squeeze→Unsqueeze round-trip (dim manipulation)", {
  # X[1,4,8] → Squeeze(axis=0) → [4,8] → Unsqueeze(axis=0) → [1,4,8]
  inp <- .onnx_value_info("X", 1L, c(1L, 4L, 8L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L, 8L))

  # Squeeze axes
  sq_axes_raw <- .int64_bytes(0L)
  sq_axes_t  <- .onnx_tensor("sq_ax", c(1L), 7L, sq_axes_raw)
  sq_axes_vi <- .onnx_value_info("sq_ax", 7L, c(1L))

  # Unsqueeze axes
  usq_axes_raw <- .int64_bytes(0L)
  usq_axes_t  <- .onnx_tensor("usq_ax", c(1L), 7L, usq_axes_raw)
  usq_axes_vi <- .onnx_value_info("usq_ax", 7L, c(1L))

  sq_node  <- .onnx_node("Squeeze", c("X", "sq_ax"), "sq")
  usq_node <- .onnx_node("Unsqueeze", c("sq", "usq_ax"), "usq")
  relu_node <- .onnx_node("Relu", "usq", "Y")

  graph <- .onnx_graph("test", list(sq_node, usq_node, relu_node),
    list(inp, sq_axes_vi, usq_axes_vi), list(outp),
    list(sq_axes_t, usq_axes_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  set.seed(42)
  x <- rnorm(32)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 32)
  expect_equal(r, pmax(x, 0), tolerance = 1e-5)
})

# ── Boundary: single-element batch dim ──────────────────────

test_that("chain xcit-dynamic: Reshape [1,1,4] → [1,4] → MatMul (boundary)", {
  # Degenerate: remove trivial dim then matmul
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L))

  shape_raw <- unlist(lapply(c(1L, 4L), .int64_bytes))
  shape_t  <- .onnx_tensor("sh", c(2L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("sh", 7L, c(2L))

  # Weight [4, 2]
  w_data <- rep(0.5, 8)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(4L, 2L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(4L, 2L))

  r_node  <- .onnx_node("Reshape", c("X", "sh"), "flat")
  mm_node <- .onnx_node("MatMul", c("flat", "W"), "Y")

  graph <- .onnx_graph("test", list(r_node, mm_node),
    list(inp, shape_vi, w_vi), list(outp),
    list(shape_t, w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 2)
  # [1,4] @ [4,2] with all weights=0.5 → each output = sum(x)*0.5 = 5.0
  expect_equal(r, c(5, 5), tolerance = 1e-3)
})
