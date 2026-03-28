# Chain tests: UNet / Stable Diffusion patterns
# Conv → GroupNorm → Silu → MatMul → Add(skip connection)
#
# Tests GroupNorm shape propagation, 4D→2D transition for attention,
# skip connections with broadcasting.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}


# ── Minimal (2 ops): Conv → Silu ─────────────────────────────

test_that("chain unet: Conv→Silu (minimal)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 3L, 3L))

  w_raw <- .float_bytes(1.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 1L, 1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "conv_out",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  silu_node <- .onnx_node("Silu", "conv_out", "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, silu_node),
                        list(inp, w_vi), list(outp),
                        list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(0, 1, -1, 2, -2, 0.5, -0.5, 3, -3)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 9)
  # Silu(x) = x * sigmoid(x)
  expected <- x * (1 / (1 + exp(-x)))
  expect_equal(r, expected, tolerance = 1e-3)
})


# ── Real (5 ops): Conv → GroupNorm → Silu → Flatten → MatMul ──

test_that("chain unet: Conv→GroupNorm→Silu→Flatten→MatMul (UNet block)", {
  # Input: [1, 2, 3, 3]
  # Conv 2→4, kernel 1x1 → [1, 4, 3, 3]
  # GroupNorm(num_groups=2) → [1, 4, 3, 3]
  # Silu → [1, 4, 3, 3]
  # Flatten → [1, 36]
  # MatMul with W[36, 4] → [1, 4]  (attention projection)

  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  # Conv [4, 2, 1, 1]
  set.seed(123)
  w_data <- rnorm(8, 0, 0.5)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(4L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(4L, 2L, 1L, 1L))

  # GroupNorm: scale [4], bias [4]
  gn_sc_raw <- unlist(lapply(rep(1.0, 4), .float_bytes))
  gn_bi_raw <- unlist(lapply(rep(0.0, 4), .float_bytes))
  gn_sc_t <- .onnx_tensor("gn_sc", c(4L), 1L, gn_sc_raw)
  gn_bi_t <- .onnx_tensor("gn_bi", c(4L), 1L, gn_bi_raw)
  gn_sc_vi <- .onnx_value_info("gn_sc", 1L, c(4L))
  gn_bi_vi <- .onnx_value_info("gn_bi", 1L, c(4L))

  # FC: [36, 4]
  fc_data <- rep(1.0 / 36, 144)
  fc_raw <- unlist(lapply(fc_data, .float_bytes))
  fc_t  <- .onnx_tensor("FC", c(36L, 4L), 1L, fc_raw)
  fc_vi <- .onnx_value_info("FC", 1L, c(36L, 4L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "c1",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  gn_node   <- .onnx_node("GroupNormalization", c("c1", "gn_sc", "gn_bi"), "gn",
                           attrs = list(.onnx_attr_int("num_groups", 2L)))
  silu_node <- .onnx_node("Silu", "gn", "silu_out")
  flat_node <- .onnx_node("Flatten", "silu_out", "flat",
                           attrs = list(.onnx_attr_int("axis", 1L)))
  mm_node   <- .onnx_node("MatMul", c("flat", "FC"), "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, gn_node, silu_node, flat_node, mm_node),
                        list(inp, w_vi, gn_sc_vi, gn_bi_vi, fc_vi),
                        list(outp),
                        list(w_t, gn_sc_t, gn_bi_t, fc_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(18, -1, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # Just check it runs and produces finite output
  expect_true(all(is.finite(r)))
})


# ── Real with skip (6 ops): Conv → GN → Silu → Conv → Add(skip) → Silu ──

test_that("chain unet: residual block with skip connection", {
  # Input X: [1, 2, 3, 3]
  # Conv1 2→2, 1x1 → [1, 2, 3, 3]
  # GroupNorm → [1, 2, 3, 3]
  # Silu → [1, 2, 3, 3]
  # Conv2 2→2, 1x1 → [1, 2, 3, 3]
  # Add(X, conv2) → [1, 2, 3, 3]  (skip)
  # Silu → [1, 2, 3, 3]

  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L, 3L))

  # Conv1 [2, 2, 1, 1]
  w1_data <- c(0.5, 0, 0, 0.5)
  w1_raw <- unlist(lapply(w1_data, .float_bytes))
  w1_t  <- .onnx_tensor("W1", c(2L, 2L, 1L, 1L), 1L, w1_raw)
  w1_vi <- .onnx_value_info("W1", 1L, c(2L, 2L, 1L, 1L))

  # Conv2 [2, 2, 1, 1]
  w2_data <- c(0.1, 0, 0, 0.1)
  w2_raw <- unlist(lapply(w2_data, .float_bytes))
  w2_t  <- .onnx_tensor("W2", c(2L, 2L, 1L, 1L), 1L, w2_raw)
  w2_vi <- .onnx_value_info("W2", 1L, c(2L, 2L, 1L, 1L))

  # GN params: 2 channels, 1 group
  gn_sc_raw <- unlist(lapply(rep(1.0, 2), .float_bytes))
  gn_bi_raw <- unlist(lapply(rep(0.0, 2), .float_bytes))
  gn_sc_t <- .onnx_tensor("gn_sc", c(2L), 1L, gn_sc_raw)
  gn_bi_t <- .onnx_tensor("gn_bi", c(2L), 1L, gn_bi_raw)
  gn_sc_vi <- .onnx_value_info("gn_sc", 1L, c(2L))
  gn_bi_vi <- .onnx_value_info("gn_bi", 1L, c(2L))

  conv1_node <- .onnx_node("Conv", c("X", "W1"), "c1",
                            attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  gn_node    <- .onnx_node("GroupNormalization", c("c1", "gn_sc", "gn_bi"), "gn",
                            attrs = list(.onnx_attr_int("num_groups", 1L)))
  silu1_node <- .onnx_node("Silu", "gn", "s1")
  conv2_node <- .onnx_node("Conv", c("s1", "W2"), "c2",
                            attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  add_node   <- .onnx_node("Add", c("X", "c2"), "res")
  silu2_node <- .onnx_node("Silu", "res", "Y")

  graph <- .onnx_graph("test",
                        list(conv1_node, gn_node, silu1_node,
                             conv2_node, add_node, silu2_node),
                        list(inp, w1_vi, w2_vi, gn_sc_vi, gn_bi_vi),
                        list(outp),
                        list(w1_t, w2_t, gn_sc_t, gn_bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- runif(18, -1, 1)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 18)
  expect_true(all(is.finite(r)))
})


# ── Boundary: single spatial pixel ───────────────────────────

test_that("chain unet: 1x1 spatial (boundary)", {
  # Input: [1, 2, 1, 1] → Conv → GN → Silu
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 1L, 1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 1L, 1L))

  w_data <- c(1, 0, 0, 1)
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(2L, 2L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 1L, 1L))

  gn_sc_raw <- unlist(lapply(rep(1.0, 2), .float_bytes))
  gn_bi_raw <- unlist(lapply(rep(0.0, 2), .float_bytes))
  gn_sc_t <- .onnx_tensor("gn_sc", c(2L), 1L, gn_sc_raw)
  gn_bi_t <- .onnx_tensor("gn_bi", c(2L), 1L, gn_bi_raw)
  gn_sc_vi <- .onnx_value_info("gn_sc", 1L, c(2L))
  gn_bi_vi <- .onnx_value_info("gn_bi", 1L, c(2L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "c1",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  gn_node   <- .onnx_node("GroupNormalization", c("c1", "gn_sc", "gn_bi"), "gn",
                           attrs = list(.onnx_attr_int("num_groups", 1L)))
  silu_node <- .onnx_node("Silu", "gn", "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, gn_node, silu_node),
                        list(inp, w_vi, gn_sc_vi, gn_bi_vi),
                        list(outp),
                        list(w_t, gn_sc_t, gn_bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(2.0, -1.0)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 2)
  expect_true(all(is.finite(r)))
})
