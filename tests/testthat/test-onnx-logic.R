# Tests for ONNX Where, Equal, and attention mask patterns

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Where (basic) ──────────────────────────────────────────────

test_that("ONNX Where selects from X when cond=1, Y when cond=0", {
  inp_c <- .onnx_value_info("C", 1L, c(4L))
  inp_x <- .onnx_value_info("X", 1L, c(4L))
  inp_y <- .onnx_value_info("Y", 1L, c(4L))
  outp  <- .onnx_value_info("Z", 1L, c(4L))
  node  <- .onnx_node("Where", c("C", "X", "Y"), "Z")
  graph <- .onnx_graph("test", list(node),
                        list(inp_c, inp_x, inp_y), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  cond <- c(1, 0, 1, 0)
  x    <- c(10, 20, 30, 40)
  y    <- c(100, 200, 300, 400)
  result <- run_onnx(path, list(C = cond, X = x, Y = y))
  expect_equal(as.numeric(result), c(10, 200, 30, 400), tolerance = 1e-3)
})

# ── Where with -inf (the bug that triggered this test) ─────────

test_that("ONNX Where with -1e9 Y values does not produce NaN", {
  # Simulates causal mask: cond=1 → keep score, cond=0 → mask with -1e9
  inp_c <- .onnx_value_info("C", 1L, c(4L))
  inp_x <- .onnx_value_info("X", 1L, c(4L))
  inp_y <- .onnx_value_info("Y", 1L, c(4L))
  outp  <- .onnx_value_info("Z", 1L, c(4L))
  node  <- .onnx_node("Where", c("C", "X", "Y"), "Z")
  graph <- .onnx_graph("test", list(node),
                        list(inp_c, inp_x, inp_y), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  cond <- c(1, 1, 0, 0)
  x    <- c(0.5, 1.0, 2.0, 3.0)
  y    <- rep(-1e9, 4)
  result <- run_onnx(path, list(C = cond, X = x, Y = y))
  r <- as.numeric(result)
  expect_false(any(is.nan(r)))
  expect_equal(r[1], 0.5, tolerance = 1e-3)
  expect_equal(r[2], 1.0, tolerance = 1e-3)
  expect_true(r[3] < -1e8)  # large negative
  expect_true(r[4] < -1e8)
})

test_that("ONNX Where with large -inf Y survives softmax downstream", {
  # Full attention pattern: Where(mask, scores, -inf) → Softmax
  inp_c <- .onnx_value_info("C", 1L, c(4L))
  inp_x <- .onnx_value_info("X", 1L, c(4L))
  inp_y <- .onnx_value_info("Y", 1L, c(4L))
  outp  <- .onnx_value_info("Z", 1L, c(4L))
  where_node   <- .onnx_node("Where", c("C", "X", "Y"), "W")
  softmax_node <- .onnx_node("Softmax", "W", "Z")
  graph <- .onnx_graph("test", list(where_node, softmax_node),
                        list(inp_c, inp_x, inp_y), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  cond <- c(1, 1, 0, 0)
  x    <- c(1.0, 2.0, 3.0, 4.0)
  y    <- rep(-3.4e38, 4)  # -FLT_MAX (what models actually use)
  result <- run_onnx(path, list(C = cond, X = x, Y = y))
  r <- as.numeric(result)
  expect_false(any(is.nan(r)))
  expect_true(all(r >= 0))
  expect_equal(sum(r), 1.0, tolerance = 1e-3)
  # Masked positions should be ~0
  expect_true(r[3] < 1e-5)
  expect_true(r[4] < 1e-5)
  # Unmasked: softmax([1,2]) = [exp(1), exp(2)] / (exp(1)+exp(2))
  sm <- exp(c(1, 2)) / sum(exp(c(1, 2)))
  expect_equal(r[1], sm[1], tolerance = 1e-3)
  expect_equal(r[2], sm[2], tolerance = 1e-3)
})

# ── Equal ──────────────────────────────────────────────────────

test_that("ONNX Equal produces 0/1 mask", {
  path <- .onnx_make_binary("Equal", c(4L))
  a <- c(1, 2, 3, 4)
  b <- c(1, 0, 3, 0)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(as.numeric(result), c(1, 0, 1, 0), tolerance = 1e-3)
})

test_that("ONNX Equal with broadcast works", {
  inp_a <- .onnx_value_info("A", 1L, c(4L))
  inp_b <- .onnx_value_info("B", 1L, c(1L))
  outp  <- .onnx_value_info("Y", 1L, c(4L))
  node  <- .onnx_node("Equal", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(A = c(0, 1, 0, 2), B = c(0)))
  expect_equal(as.numeric(result), c(1, 0, 1, 0), tolerance = 1e-3)
})

# ── Equal → Where → Softmax (transformer attention mask) ──────

test_that("ONNX Equal+Where+Softmax attention mask pattern works", {
  # Pattern: mask = Equal(attention_mask, 0)
  # Then: Where(mask, -1e9, scores) — mask padding tokens
  # Then: Softmax
  inp_mask   <- .onnx_value_info("M", 1L, c(4L))
  inp_scores <- .onnx_value_info("S", 1L, c(4L))
  inp_fill   <- .onnx_value_info("F", 1L, c(1L))
  inp_zero   <- .onnx_value_info("Z", 1L, c(1L))
  outp       <- .onnx_value_info("Y", 1L, c(4L))

  eq_node      <- .onnx_node("Equal", c("M", "Z"), "eq_out")
  where_node   <- .onnx_node("Where", c("eq_out", "F", "S"), "masked")
  softmax_node <- .onnx_node("Softmax", "masked", "Y")
  graph <- .onnx_graph("test",
                        list(eq_node, where_node, softmax_node),
                        list(inp_mask, inp_scores, inp_fill, inp_zero),
                        list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # attention_mask=[1,1,0,0] → Equal(mask, 0) = [0,0,1,1]
  # Where(eq, -1e9, scores) = [scores[0], scores[1], -1e9, -1e9]
  # Softmax → probs concentrated on first 2 positions
  scores <- c(1.0, 2.0, 3.0, 4.0)
  result <- run_onnx(path, list(M = c(1,1,0,0), S = scores,
                                 F = c(-1e9), Z = c(0)))
  r <- as.numeric(result)
  expect_false(any(is.nan(r)))
  expect_equal(sum(r), 1.0, tolerance = 1e-3)
  expect_true(r[3] < 1e-5)
  expect_true(r[4] < 1e-5)
})

# ── Comparisons: Less / LessOrEqual / Greater / GreaterOrEqual ──
# Each produces a 0/1 float mask, matching how Equal and Where already
# represent booleans.

.cmp_graph <- function(op, n = 5L) {
  inp_a <- .onnx_value_info("A", 1L, c(n))
  inp_b <- .onnx_value_info("B", 1L, c(n))
  outp  <- .onnx_value_info("C", 1L, c(n))
  node  <- .onnx_node(op, c("A", "B"), "C")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path  <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

test_that("ONNX Less returns 1 where a < b", {
  a <- c(1, 5, 3, 9, 2)
  b <- c(4, 2, 3, 1, 8)
  result <- run_onnx(.cmp_graph("Less"), list(A = a, B = b))
  expect_equal(as.numeric(result), as.numeric(a < b), tolerance = 1e-3)
})

test_that("ONNX Greater returns 1 where a > b", {
  a <- c(1, 5, 3, 9, 2)
  b <- c(4, 2, 3, 1, 8)
  result <- run_onnx(.cmp_graph("Greater"), list(A = a, B = b))
  expect_equal(as.numeric(result), as.numeric(a > b), tolerance = 1e-3)
})

test_that("ONNX LessOrEqual and GreaterOrEqual include equal values", {
  # The middle pair is equal, which is what separates these from the
  # strict forms above.
  a <- c(1, 5, 3, 9, 2)
  b <- c(4, 2, 3, 1, 8)
  le <- run_onnx(.cmp_graph("LessOrEqual"), list(A = a, B = b))
  ge <- run_onnx(.cmp_graph("GreaterOrEqual"), list(A = a, B = b))
  expect_equal(as.numeric(le), as.numeric(a <= b), tolerance = 1e-3)
  expect_equal(as.numeric(ge), as.numeric(a >= b), tolerance = 1e-3)
})

# ── Not / And / Or / Xor on 0/1 masks ──────────────────────────

test_that("ONNX Not complements a 0/1 mask", {
  inp  <- .onnx_value_info("A", 1L, c(4L))
  outp <- .onnx_value_info("C", 1L, c(4L))
  node <- .onnx_node("Not", "A", "C")
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(A = c(1, 0, 1, 0)))
  expect_equal(as.numeric(result), c(0, 1, 0, 1), tolerance = 1e-3)
})

test_that("ONNX And, Or and Xor combine 0/1 masks", {
  a <- c(1, 1, 0, 0)
  b <- c(1, 0, 1, 0)
  and_r <- run_onnx(.cmp_graph("And", 4L), list(A = a, B = b))
  or_r  <- run_onnx(.cmp_graph("Or", 4L),  list(A = a, B = b))
  xor_r <- run_onnx(.cmp_graph("Xor", 4L), list(A = a, B = b))
  expect_equal(as.numeric(and_r), c(1, 0, 0, 0), tolerance = 1e-3)
  expect_equal(as.numeric(or_r),  c(1, 1, 1, 0), tolerance = 1e-3)
  expect_equal(as.numeric(xor_r), c(0, 1, 1, 0), tolerance = 1e-3)
})

test_that("ONNX Less feeding Where masks the selected branch", {
  # The pattern these ops actually appear in: a comparison builds a mask
  # that Where then uses to choose between two tensors.
  inp_a <- .onnx_value_info("A", 1L, c(4L))
  inp_b <- .onnx_value_info("B", 1L, c(4L))
  inp_y <- .onnx_value_info("Y", 1L, c(4L))
  outp  <- .onnx_value_info("Z", 1L, c(4L))
  n1 <- .onnx_node("Less", c("A", "B"), "M")
  n2 <- .onnx_node("Where", c("M", "A", "Y"), "Z")
  graph <- .onnx_graph("test", list(n1, n2),
                       list(inp_a, inp_b, inp_y), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- c(1, 5, 3, 9)
  b <- c(4, 2, 8, 1)
  y <- c(-1, -2, -3, -4)
  result <- run_onnx(path, list(A = a, B = b, Y = y))
  expect_equal(as.numeric(result), ifelse(a < b, a, y), tolerance = 1e-3)
})
