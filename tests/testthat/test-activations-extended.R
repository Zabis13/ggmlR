# Extended tests for Activation Functions

# ============================================================================
# Sigmoid
# ============================================================================

test_that("ggml_sigmoid computes sigmoid correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  result <- ggml_sigmoid(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # sigmoid(x) = 1 / (1 + exp(-x))
  expected <- 1 / (1 + exp(-c(-2, -1, 0, 1, 2)))
  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_sigmoid(0) = 0.5", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(a, 0)

  result <- ggml_sigmoid(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 0.5, tolerance = 1e-5)
})

test_that("ggml_sigmoid output is in (0, 1)", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a, seq(-10, 10, length.out = 100))

  result <- ggml_sigmoid(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_true(all(output > 0))
  expect_true(all(output < 1))
})

# ============================================================================
# Tanh
# ============================================================================

test_that("ggml_tanh computes tanh correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  result <- ggml_tanh(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expected <- tanh(c(-2, -1, 0, 1, 2))
  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_tanh(0) = 0", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(a, 0)

  result <- ggml_tanh(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 0, tolerance = 1e-5)
})

test_that("ggml_tanh output is in [-1, 1]", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a, seq(-10, 10, length.out = 100))

  result <- ggml_tanh(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # tanh can reach exactly -1 or 1 at extreme values
  expect_true(all(output >= -1))
  expect_true(all(output <= 1))
})

# ============================================================================
# ELU
# ============================================================================

test_that("ggml_elu computes ELU correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  result <- ggml_elu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # ELU(x) = x if x > 0, else exp(x) - 1
  input <- c(-2, -1, 0, 1, 2)
  expected <- ifelse(input > 0, input, exp(input) - 1)
  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_elu(0) = 0", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(a, 0)

  result <- ggml_elu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 0, tolerance = 1e-5)
})

# ============================================================================
# Leaky ReLU
# ============================================================================

test_that("ggml_leaky_relu computes correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  result <- ggml_leaky_relu(ctx, a, negative_slope = 0.1)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # LeakyReLU(x) = x if x > 0, else 0.1 * x
  expected <- c(-0.2, -0.1, 0, 1, 2)
  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_leaky_relu with different slopes", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(-2, 0, 2))

  result <- ggml_leaky_relu(ctx, a, negative_slope = 0.2)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(-0.4, 0, 2), tolerance = 1e-4)
})

# ============================================================================
# Hard Swish
# ============================================================================

test_that("ggml_hardswish computes correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7)
  ggml_set_f32(a, c(-4, -3, -1, 0, 1, 3, 4))

  result <- ggml_hardswish(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # HardSwish(x) = x * clip(x+3, 0, 6) / 6
  input <- c(-4, -3, -1, 0, 1, 3, 4)
  expected <- input * pmin(pmax(input + 3, 0), 6) / 6
  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_hardswish(0) = 0", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(a, 0)

  result <- ggml_hardswish(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 0, tolerance = 1e-5)
})

# ============================================================================
# Hard Sigmoid
# ============================================================================

test_that("ggml_hardsigmoid computes correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7)
  ggml_set_f32(a, c(-4, -3, -1, 0, 1, 3, 4))

  result <- ggml_hardsigmoid(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # HardSigmoid(x) = clip(x+3, 0, 6) / 6
  input <- c(-4, -3, -1, 0, 1, 3, 4)
  expected <- pmin(pmax(input + 3, 0), 6) / 6
  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_hardsigmoid output is in [0, 1]", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a, seq(-10, 10, length.out = 100))

  result <- ggml_hardsigmoid(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_true(all(output >= 0))
  expect_true(all(output <= 1))
})

# ============================================================================
# Softplus
# ============================================================================

test_that("ggml_softplus computes correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  result <- ggml_softplus(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # Softplus(x) = log(1 + exp(x))
  expected <- log(1 + exp(c(-2, -1, 0, 1, 2)))
  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_softplus is always positive", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a, seq(-10, 10, length.out = 100))

  result <- ggml_softplus(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_true(all(output > 0))
})

test_that("ggml_softplus(0) = log(2)", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(a, 0)

  result <- ggml_softplus(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, log(2), tolerance = 1e-5)
})

# ============================================================================
# GELU Quick
# ============================================================================

test_that("ggml_gelu_quick computes fast GELU", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  result <- ggml_gelu_quick(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # GELU(0) = 0
  expect_equal(output[3], 0, tolerance = 1e-5)
  # GELU is smooth and bounded
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))
})

# ============================================================================
# Comparison: GELU vs GELU Quick
# ============================================================================

test_that("gelu and gelu_quick produce similar results", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a1, c(-1, -0.5, 0, 0.5, 1))
  ggml_set_f32(a2, c(-1, -0.5, 0, 0.5, 1))

  r1 <- ggml_gelu(ctx, a1)
  r2 <- ggml_gelu_quick(ctx, a2)

  graph1 <- ggml_build_forward_expand(ctx, r1)
  ggml_graph_compute(ctx, graph1)
  out1 <- ggml_get_f32(r1)

  graph2 <- ggml_build_forward_expand(ctx, r2)
  ggml_graph_compute(ctx, graph2)
  out2 <- ggml_get_f32(r2)

  # Should be within reasonable tolerance
  expect_equal(out1, out2, tolerance = 0.05)
})
