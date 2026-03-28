# Chain tests: EyeLike
# EyeLike generates an identity matrix of the same shape as input.
# Covers: EyeLike standalone, EyeLike→MatMul, EyeLike with k offset.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: EyeLike → Add ─────────────────────────────────

test_that("chain eyelike: EyeLike→Add (minimal)", {
  # EyeLike(X) produces 3x3 identity, then add X
  inp <- .onnx_value_info("X", 1L, c(3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 3L))

  eye_node <- .onnx_node("EyeLike", "X", "eye")
  add_node <- .onnx_node("Add", c("X", "eye"), "Y")

  graph <- .onnx_graph("test", list(eye_node, add_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
  result <- run_onnx(path, list(X = x))
  r <- matrix(as.numeric(result), 3, 3, byrow = TRUE)
  xm <- matrix(x, 3, 3, byrow = TRUE)
  eye <- diag(3)
  expect_equal(r, xm + eye, tolerance = 1e-5)
})

# ── EyeLike → MatMul (identity transform) ──────────────────

test_that("chain eyelike: EyeLike→MatMul (identity matmul)", {
  # EyeLike(shape_ref) produces 4x4 identity
  # MatMul(X, eye) = X
  inp <- .onnx_value_info("X", 1L, c(2L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L))

  # Shape reference for EyeLike: 4x4 input
  ref <- .onnx_value_info("ref", 1L, c(4L, 4L))
  # Use a constant zeros tensor as shape reference
  ref_raw <- .float_bytes(rep(0, 16))
  ref_t <- .onnx_tensor("ref", c(4L, 4L), 1L, ref_raw)

  eye_node <- .onnx_node("EyeLike", "ref", "eye")
  mm_node  <- .onnx_node("MatMul", c("X", "eye"), "Y")

  graph <- .onnx_graph("test", list(eye_node, mm_node),
                        list(inp, ref),
                        list(outp),
                        list(ref_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(r, x, tolerance = 1e-5)
})

# ── EyeLike with k=1 (super-diagonal) ─────────────────────

test_that("chain eyelike: EyeLike k=1 → Mul (super-diagonal mask)", {
  # EyeLike(X, k=1) → shifted identity, then element-wise Mul
  inp <- .onnx_value_info("X", 1L, c(3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 3L))

  eye_node <- .onnx_node("EyeLike", "X", "eye",
                          attrs = list(.onnx_attr_int("k", 1L)))
  mul_node <- .onnx_node("Mul", c("X", "eye"), "Y")

  graph <- .onnx_graph("test", list(eye_node, mul_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
  result <- run_onnx(path, list(X = x))
  r <- matrix(as.numeric(result), 3, 3, byrow = TRUE)
  # k=1 super-diagonal: [[0,1,0],[0,0,1],[0,0,0]]
  eye_k1 <- matrix(0, 3, 3)
  eye_k1[1, 2] <- 1; eye_k1[2, 3] <- 1
  xm <- matrix(x, 3, 3, byrow = TRUE)
  expect_equal(r, xm * eye_k1, tolerance = 1e-5)
})

# ── Non-square EyeLike ─────────────────────────────────────

test_that("chain eyelike: EyeLike non-square → Add (3x4)", {
  inp <- .onnx_value_info("X", 1L, c(3L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(3L, 4L))

  eye_node <- .onnx_node("EyeLike", "X", "eye")
  add_node <- .onnx_node("Add", c("eye", "eye"), "Y")

  graph <- .onnx_graph("test", list(eye_node, add_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(0, 12)
  result <- run_onnx(path, list(X = x))
  r <- matrix(as.numeric(result), 3, 4, byrow = TRUE)
  # 2 * eye for 3x4: diagonal 1s doubled
  expected <- matrix(0, 3, 4)
  for (i in 1:3) expected[i, i] <- 2
  expect_equal(r, expected, tolerance = 1e-5)
})

# ── Boundary: 1x1 EyeLike ─────────────────────────────────

test_that("chain eyelike: 1x1 EyeLike (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L))

  eye_node <- .onnx_node("EyeLike", "X", "Y")

  graph <- .onnx_graph("test", list(eye_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(42.0)))
  r <- as.numeric(result)
  expect_equal(r, 1.0, tolerance = 1e-5)
})
