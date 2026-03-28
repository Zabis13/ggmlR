# Chain tests: Super-resolution patterns (ESRGAN/EDSR-style)
# Conv → LeakyRelu → Add(residual) → Conv → Clip
#
# Tests residual connections, shape propagation through multiple Convs,
# value clamping with Clip.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}


# ── Minimal (2 ops): Conv → LeakyRelu ────────────────────────

test_that("chain superres: Conv→LeakyRelu (minimal)", {
  # Input: [1, 1, 3, 3], Conv 1→1, 1x1 → [1, 1, 3, 3], LeakyRelu

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 3L, 3L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 3L, 3L))

  w_raw <- .float_bytes(2.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 1L, 1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "conv_out",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  lr_node   <- .onnx_node("LeakyRelu", "conv_out", "Y",
                           attrs = list(.onnx_attr_float("alpha", 0.2)))

  graph <- .onnx_graph("test",
                        list(conv_node, lr_node),
                        list(inp, w_vi), list(outp),
                        list(w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, -1, 0, 0.5, -0.5, 2, -2, 3, -3)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 9)
  # Conv*2 then LeakyRelu(alpha=0.2)
  conv_out <- x * 2
  expected <- ifelse(conv_out > 0, conv_out, conv_out * 0.2)
  expect_equal(r, expected, tolerance = 1e-4)
})


# ── Real (5 ops): Conv → LeakyRelu → Add(residual) → Conv → Clip ──

test_that("chain superres: Conv→LeakyRelu→Add→Conv→Clip (residual block)", {
  # Residual block pattern:
  # Input X: [1, 1, 4, 4]
  # Conv1 1→1, 1x1 → [1, 1, 4, 4]
  # LeakyRelu → [1, 1, 4, 4]
  # Add(X, leaky_out) → [1, 1, 4, 4]  (skip connection)
  # Conv2 1→1, 1x1 → [1, 1, 4, 4]
  # Clip(0, 1) → [1, 1, 4, 4]  (output pixel range)

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))

  # Conv1: identity-like (weight=0.5)
  w1_raw <- .float_bytes(0.5)
  w1_t  <- .onnx_tensor("W1", c(1L, 1L, 1L, 1L), 1L, w1_raw)
  w1_vi <- .onnx_value_info("W1", 1L, c(1L, 1L, 1L, 1L))

  # Conv2: scale down (weight=0.1)
  w2_raw <- .float_bytes(0.1)
  w2_t  <- .onnx_tensor("W2", c(1L, 1L, 1L, 1L), 1L, w2_raw)
  w2_vi <- .onnx_value_info("W2", 1L, c(1L, 1L, 1L, 1L))

  # Clip min/max as initializers
  min_raw <- .float_bytes(0.0)
  max_raw <- .float_bytes(1.0)
  min_t  <- .onnx_tensor("clip_min", c(1L), 1L, min_raw)
  max_t  <- .onnx_tensor("clip_max", c(1L), 1L, max_raw)
  min_vi <- .onnx_value_info("clip_min", 1L, c(1L))
  max_vi <- .onnx_value_info("clip_max", 1L, c(1L))

  conv1_node <- .onnx_node("Conv", c("X", "W1"), "c1",
                            attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  lr_node    <- .onnx_node("LeakyRelu", "c1", "lr",
                            attrs = list(.onnx_attr_float("alpha", 0.1)))
  add_node   <- .onnx_node("Add", c("X", "lr"), "res")
  conv2_node <- .onnx_node("Conv", c("res", "W2"), "c2",
                            attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  clip_node  <- .onnx_node("Clip", c("c2", "clip_min", "clip_max"), "Y")

  graph <- .onnx_graph("test",
                        list(conv1_node, lr_node, add_node, conv2_node, clip_node),
                        list(inp, w1_vi, w2_vi, min_vi, max_vi),
                        list(outp),
                        list(w1_t, w2_t, min_t, max_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(0, 1, length.out = 16)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 16)
  # All values clipped to [0, 1]
  expect_true(all(r >= 0 & r <= 1))

  # Verify manually: conv1=x*0.5, leaky=max(conv1, 0.1*conv1), res=x+leaky,
  # conv2=res*0.1, clip(0,1)
  conv1 <- x * 0.5
  lr_out <- ifelse(conv1 > 0, conv1, conv1 * 0.1)
  res <- x + lr_out
  conv2 <- res * 0.1
  expected <- pmin(pmax(conv2, 0), 1)
  expect_equal(r, expected, tolerance = 1e-4)
})


# ── Boundary: all negative input (LeakyRelu slope matters) ───

test_that("chain superres: all negative input (boundary)", {
  # When all inputs are negative, LeakyRelu slope dominates
  # Conv → LeakyRelu → Clip

  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 2L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))

  w_raw <- .float_bytes(1.0)
  w_t  <- .onnx_tensor("W", c(1L, 1L, 1L, 1L), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 1L, 1L))

  min_raw <- .float_bytes(-0.5)
  max_raw <- .float_bytes(0.0)
  min_t  <- .onnx_tensor("cmin", c(1L), 1L, min_raw)
  max_t  <- .onnx_tensor("cmax", c(1L), 1L, max_raw)
  min_vi <- .onnx_value_info("cmin", 1L, c(1L))
  max_vi <- .onnx_value_info("cmax", 1L, c(1L))

  conv_node <- .onnx_node("Conv", c("X", "W"), "c1",
                           attrs = list(.onnx_attr_ints("kernel_shape", c(1L, 1L))))
  lr_node   <- .onnx_node("LeakyRelu", "c1", "lr",
                           attrs = list(.onnx_attr_float("alpha", 0.2)))
  clip_node <- .onnx_node("Clip", c("lr", "cmin", "cmax"), "Y")

  graph <- .onnx_graph("test",
                        list(conv_node, lr_node, clip_node),
                        list(inp, w_vi, min_vi, max_vi),
                        list(outp),
                        list(w_t, min_t, max_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-1, -2, -3, -4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # Conv is identity, LeakyRelu(x<0) = 0.2*x → [-0.2, -0.4, -0.6, -0.8]
  # Clip to [-0.5, 0] → [-0.2, -0.4, -0.5, -0.5]
  expected <- pmin(pmax(x * 0.2, -0.5), 0)
  expect_equal(r, expected, tolerance = 1e-4)
})
