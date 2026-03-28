# Chain tests: Power normalization patterns (RMS-like)
# Pow → ReduceMean → Sqrt → Div
#
# Covers: Pow, ReduceMean, Sqrt

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Pow → Sqrt ─────────────────────────────

test_that("chain powernorm: Pow→Sqrt (minimal)", {
  # Pow(x, 2) then Sqrt → |x|
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  exp_raw <- .float_bytes(2.0)
  exp_t  <- .onnx_tensor("E", c(1L), 1L, exp_raw)
  exp_vi <- .onnx_value_info("E", 1L, c(1L))

  pow_node  <- .onnx_node("Pow", c("X", "E"), "sq")
  sqrt_node <- .onnx_node("Sqrt", "sq", "Y")

  graph <- .onnx_graph("test", list(pow_node, sqrt_node),
                        list(inp, exp_vi), list(outp), list(exp_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(4, 9, 1, 16)  # positive only (Pow uses log-exp)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(r, abs(x), tolerance = 1e-3)
})

# ── Real (4 ops): Pow → ReduceMean → Sqrt → Div (RMS norm) ──

test_that("chain powernorm: Pow→ReduceMean→Sqrt→Div (RMS normalization)", {
  # RMS norm: x / sqrt(mean(x^2))
  # Input [4] → Pow(2) → [4] → ReduceMean → [1] → Sqrt → [1] → Div(X, rms) → [4]

  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  exp_raw <- .float_bytes(2.0)
  exp_t  <- .onnx_tensor("E", c(1L), 1L, exp_raw)
  exp_vi <- .onnx_value_info("E", 1L, c(1L))

  pow_node  <- .onnx_node("Pow", c("X", "E"), "sq")
  mean_node <- .onnx_node("ReduceMean", "sq", "mean_sq")
  sqrt_node <- .onnx_node("Sqrt", "mean_sq", "rms")
  div_node  <- .onnx_node("Div", c("X", "rms"), "Y")

  graph <- .onnx_graph("test",
                        list(pow_node, mean_node, sqrt_node, div_node),
                        list(inp, exp_vi), list(outp), list(exp_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # RMS = sqrt(mean(x^2)) = sqrt(30/4) = sqrt(7.5)
  rms <- sqrt(mean(x^2))
  expected <- x / rms
  expect_equal(r, expected, tolerance = 1e-3)
})

# ── Boundary: all equal values ───────────────────────────────

test_that("chain powernorm: constant input (boundary)", {
  inp <- .onnx_value_info("X", 1L, c(3L))
  outp <- .onnx_value_info("Y", 1L, c(3L))

  exp_raw <- .float_bytes(2.0)
  exp_t  <- .onnx_tensor("E", c(1L), 1L, exp_raw)
  exp_vi <- .onnx_value_info("E", 1L, c(1L))

  pow_node  <- .onnx_node("Pow", c("X", "E"), "sq")
  mean_node <- .onnx_node("ReduceMean", "sq", "mean_sq")
  sqrt_node <- .onnx_node("Sqrt", "mean_sq", "rms")
  div_node  <- .onnx_node("Div", c("X", "rms"), "Y")

  graph <- .onnx_graph("test",
                        list(pow_node, mean_node, sqrt_node, div_node),
                        list(inp, exp_vi), list(outp), list(exp_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # All same value: RMS norm → all 1.0
  result <- run_onnx(path, list(X = c(5, 5, 5)))
  r <- as.numeric(result)
  expect_equal(r, c(1, 1, 1), tolerance = 1e-3)
})
