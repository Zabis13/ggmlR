# Chain tests: Manual GELU via Erf pattern
# Mul(0.5) → Mul(sqrt(2)^-1) → Erf → Add(1) → Mul → (x * 0.5 * (1 + erf(x/sqrt(2))))
#
# Covers: Erf, Sin, Cos (trig ops in chains)

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Erf → Add ──────────────────────────────

test_that("chain erf-gelu: Erf→Add (minimal)", {
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  one_raw <- .float_bytes(1.0)
  one_t  <- .onnx_tensor("one", c(1L), 1L, one_raw)
  one_vi <- .onnx_value_info("one", 1L, c(1L))

  erf_node <- .onnx_node("Erf", "X", "e")
  add_node <- .onnx_node("Add", c("e", "one"), "Y")

  graph <- .onnx_graph("test", list(erf_node, add_node),
                        list(inp, one_vi), list(outp), list(one_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0, 1, -1, 2)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # ggml uses fast erf approx: erf(x) ≈ tanh(sqrt(2/pi) * (x + 0.044715*x^3))
  erf_approx <- function(v) tanh(sqrt(2/pi) * (v + 0.044715 * v^3))
  expected <- 1 + sapply(x, erf_approx)
  expect_equal(r, expected, tolerance = 1e-3)
})

# ── Real (5 ops): manual GELU ───────────────────────────────

test_that("chain erf-gelu: manual GELU via Erf (BERT pattern)", {
  # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
  # Graph: X → Div(sqrt(2)) → Erf → Add(1) → Mul(0.5) → Mul(X)

  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  # Constants
  sqrt2_raw <- .float_bytes(sqrt(2))
  sqrt2_t  <- .onnx_tensor("sqrt2", c(1L), 1L, sqrt2_raw)
  sqrt2_vi <- .onnx_value_info("sqrt2", 1L, c(1L))

  one_raw <- .float_bytes(1.0)
  one_t  <- .onnx_tensor("one", c(1L), 1L, one_raw)
  one_vi <- .onnx_value_info("one", 1L, c(1L))

  half_raw <- .float_bytes(0.5)
  half_t  <- .onnx_tensor("half", c(1L), 1L, half_raw)
  half_vi <- .onnx_value_info("half", 1L, c(1L))

  div_node  <- .onnx_node("Div", c("X", "sqrt2"), "scaled")
  erf_node  <- .onnx_node("Erf", "scaled", "e")
  add_node  <- .onnx_node("Add", c("e", "one"), "ep1")
  mul1_node <- .onnx_node("Mul", c("ep1", "half"), "half_ep1")
  mul2_node <- .onnx_node("Mul", c("X", "half_ep1"), "Y")

  graph <- .onnx_graph("test",
                        list(div_node, erf_node, add_node, mul1_node, mul2_node),
                        list(inp, sqrt2_vi, one_vi, half_vi),
                        list(outp),
                        list(sqrt2_t, one_t, half_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-2, -1, 0, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
  # ggml erf approx: tanh(sqrt(2/pi) * (x + 0.044715*x^3))
  erf_approx <- function(v) tanh(sqrt(2/pi) * (v + 0.044715 * v^3))
  expected <- x * 0.5 * (1 + sapply(x / sqrt(2), erf_approx))
  expect_equal(r, expected, tolerance = 0.02)
})

# ── Sin + Cos chain (positional encoding) ────────────────────

test_that("chain erf-gelu: Sin→Mul + Cos→Mul (trig chain)", {
  # Simulates sin/cos positional encoding with scaling
  # X → Sin → Mul(scale) → Y1
  # X → Cos → Mul(scale) → Y2
  # Concat(Y1, Y2) → Y

  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(8L))

  scale_raw <- .float_bytes(0.5)
  scale_t  <- .onnx_tensor("sc", c(1L), 1L, scale_raw)
  scale_vi <- .onnx_value_info("sc", 1L, c(1L))

  sin_node <- .onnx_node("Sin", "X", "s")
  cos_node <- .onnx_node("Cos", "X", "c")
  mul_s    <- .onnx_node("Mul", c("s", "sc"), "sm")
  mul_c    <- .onnx_node("Mul", c("c", "sc"), "cm")
  cat_node <- .onnx_node("Concat", c("sm", "cm"), "Y",
                          attrs = list(.onnx_attr_int("axis", 0L)))

  graph <- .onnx_graph("test",
                        list(sin_node, cos_node, mul_s, mul_c, cat_node),
                        list(inp, scale_vi), list(outp), list(scale_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0, pi/4, pi/2, pi)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  expect_equal(r[1:4], 0.5 * sin(x), tolerance = 1e-3)
  expect_equal(r[5:8], 0.5 * cos(x), tolerance = 1e-3)
})

# ── Boundary: Erf at zero ───────────────────────────────────

test_that("chain erf-gelu: erf(0) = 0 (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(1L))
  outp <- .onnx_value_info("Y", 1L, c(1L))

  erf_node <- .onnx_node("Erf", "X", "Y")

  graph <- .onnx_graph("test", list(erf_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(0)))
  r <- as.numeric(result)
  expect_equal(r, 0, tolerance = 1e-5)
})
