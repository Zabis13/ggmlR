# Chain tests: Attention mask patterns
# Equal → Where → Mul → Exp → ReduceSum → Div
#
# Covers: Equal, Where, Mul, Div, Exp, ReduceSum

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Mul → Exp ──────────────────────────────

test_that("chain attn-mask: Mul→Exp (minimal)", {
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  scale_raw <- .float_bytes(0.5)
  scale_t  <- .onnx_tensor("S", c(1L), 1L, scale_raw)
  scale_vi <- .onnx_value_info("S", 1L, c(1L))

  mul_node <- .onnx_node("Mul", c("X", "S"), "scaled")
  exp_node <- .onnx_node("Exp", "scaled", "Y")

  graph <- .onnx_graph("test", list(mul_node, exp_node),
                        list(inp, scale_vi), list(outp), list(scale_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0, 1, 2, -1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expected <- exp(x * 0.5)
  expect_equal(r, expected, tolerance = 1e-4)
})

# ── Real (5 ops): Equal → Where → Exp → ReduceSum → Div ────

test_that("chain attn-mask: Equal→Where→Exp→ReduceSum→Div (masked softmax)", {
  # Simulates masked attention:
  # mask[4]: 0 or 1 values
  # Equal(mask, 0) → bool mask (1 where padding)
  # Where(eq, -1e9, scores) → masked scores
  # Exp(masked) → unnormalized
  # ReduceSum → denominator
  # Div(exp, sum) → normalized (manual softmax)

  # We use two inputs: mask (0/1) and scores
  inp_mask <- .onnx_value_info("M", 1L, c(4L))
  inp_scores <- .onnx_value_info("S", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(1L))  # ReduceSum collapses

  # Zero constant for Equal comparison
  zero_raw <- .float_bytes(0.0)
  zero_t  <- .onnx_tensor("zero", c(1L), 1L, zero_raw)
  zero_vi <- .onnx_value_info("zero", 1L, c(1L))

  # Large negative for masking
  neg_raw <- .float_bytes(-1e9)
  neg_t  <- .onnx_tensor("neginf", c(1L), 1L, neg_raw)
  neg_vi <- .onnx_value_info("neginf", 1L, c(1L))

  eq_node    <- .onnx_node("Equal", c("M", "zero"), "is_pad")
  where_node <- .onnx_node("Where", c("is_pad", "neginf", "S"), "masked")
  exp_node   <- .onnx_node("Exp", "masked", "e")
  sum_node   <- .onnx_node("ReduceSum", "e", "Y")

  graph <- .onnx_graph("test",
                        list(eq_node, where_node, exp_node, sum_node),
                        list(inp_mask, inp_scores, zero_vi, neg_vi),
                        list(outp),
                        list(zero_t, neg_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # mask = [1, 1, 0, 0] (last two are padding)
  # scores = [1, 2, 3, 4]
  # Equal(M, 0) → [0, 0, 1, 1]
  # Where → [1, 2, -1e9, -1e9]
  # Exp → [e^1, e^2, ~0, ~0]
  # Sum → e^1 + e^2 ≈ 10.107
  result <- run_onnx(path, list(M = c(1, 1, 0, 0), S = c(1, 2, 3, 4)))
  r <- as.numeric(result)
  expect_equal(length(r), 1)
  expected_sum <- exp(1) + exp(2) + exp(-1e9) + exp(-1e9)
  expect_equal(r[1], expected_sum, tolerance = 0.1)
})

# ── Boundary: all masked ─────────────────────────────────────

test_that("chain attn-mask: all zeros masked (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(3L))
  outp <- .onnx_value_info("Y", 1L, c(3L))

  # Mul by 0 → all zeros, then Exp → all ones
  zero_raw <- .float_bytes(0.0)
  zero_t  <- .onnx_tensor("Z", c(1L), 1L, zero_raw)
  zero_vi <- .onnx_value_info("Z", 1L, c(1L))

  mul_node <- .onnx_node("Mul", c("X", "Z"), "z")
  exp_node <- .onnx_node("Exp", "z", "Y")

  graph <- .onnx_graph("test", list(mul_node, exp_node),
                        list(inp, zero_vi), list(outp), list(zero_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(100, -100, 0)))
  r <- as.numeric(result)
  # exp(0) = 1 for all
  expect_equal(r, c(1, 1, 1), tolerance = 1e-4)
})
