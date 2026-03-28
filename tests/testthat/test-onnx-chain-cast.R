# Chain tests: Cast and unary math patterns
# Cast(F32→I32) → Cast(I32→F32) → Floor → Ceil → Abs → Neg
#
# Covers: Cast, Floor, Ceil, Abs, Neg

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Abs → Neg ──────────────────────────────

test_that("chain cast: Abs→Neg (minimal)", {
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  abs_node <- .onnx_node("Abs", "X", "a")
  neg_node <- .onnx_node("Neg", "a", "Y")

  graph <- .onnx_graph("test", list(abs_node, neg_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-3, 2, -1, 0)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # -|x|
  expect_equal(r, c(-3, -2, -1, 0), tolerance = 1e-5)
})

# ── Real (4 ops): Floor → Ceil → Abs → Neg ──────────────────

test_that("chain cast: Floor→Ceil→Abs→Neg (rounding chain)", {
  # Floor(x) → Ceil(floor(x)) → Abs → Neg
  # Since floor(x) is integer, ceil(floor(x)) = floor(x)
  # So result = -|floor(x)|

  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  floor_node <- .onnx_node("Floor", "X", "f")
  ceil_node  <- .onnx_node("Ceil", "f", "c")
  abs_node   <- .onnx_node("Abs", "c", "a")
  neg_node   <- .onnx_node("Neg", "a", "Y")

  graph <- .onnx_graph("test", list(floor_node, ceil_node, abs_node, neg_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1.7, -2.3, 0.0, 3.9)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expected <- -abs(floor(x))
  expect_equal(r, expected, tolerance = 1e-5)
})

# ── Cast round-trip: F32 → I32 → F32 ────────────────────────

test_that("chain cast: Cast F32→I32→F32 (truncation)", {
  # Cast to I32 truncates, Cast back to F32
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  # Cast to I32 (ONNX type 6)
  cast_i32 <- .onnx_node("Cast", "X", "i32",
                          attrs = list(.onnx_attr_int("to", 6L)))
  # Cast to F32 (ONNX type 1)
  cast_f32 <- .onnx_node("Cast", "i32", "f32",
                          attrs = list(.onnx_attr_int("to", 1L)))
  # Then Floor (should be identity on integer values)
  floor_node <- .onnx_node("Floor", "f32", "Y")

  graph <- .onnx_graph("test", list(cast_i32, cast_f32, floor_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1.9, -2.7, 0.5, 3.1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # Cast to I32 truncates toward zero: 1, -2, 0, 3
  expected <- as.numeric(as.integer(x))
  expect_equal(r, expected, tolerance = 1e-5)
})

# ── Boundary: zero input ────────────────────────────────────

test_that("chain cast: all zeros (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(3L))
  outp <- .onnx_value_info("Y", 1L, c(3L))

  abs_node <- .onnx_node("Abs", "X", "a")
  neg_node <- .onnx_node("Neg", "a", "n")
  floor_node <- .onnx_node("Floor", "n", "Y")

  graph <- .onnx_graph("test", list(abs_node, neg_node, floor_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(0, 0, 0)))
  r <- as.numeric(result)
  expect_equal(r, c(0, 0, 0), tolerance = 1e-5)
})
