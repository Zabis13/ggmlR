# Tests for activation functions

test_that("relu produces correct output", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  r <- ggml_relu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(0, 0, 0, 1, 2), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("sigmoid produces values in (0,1)", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-10, -1, 0, 1, 10))

  s <- ggml_sigmoid(ctx, a)
  graph <- ggml_build_forward_expand(ctx, s)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(s)

  # All values should be between 0 and 1
  expect_true(all(result > 0))
  expect_true(all(result < 1))

  # sigmoid(0) should be 0.5
  expect_equal(result[3], 0.5, tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("tanh produces values in [-1,1]", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  t <- ggml_tanh(ctx, a)
  graph <- ggml_build_forward_expand(ctx, t)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(t)

  # All values should be between -1 and 1 (inclusive for large inputs)
  expect_true(all(result >= -1))
  expect_true(all(result <= 1))

  # tanh(0) should be 0
  expect_equal(result[3], 0, tolerance = 1e-5)

  # tanh is odd function: tanh(-x) = -tanh(x)
  expect_equal(result[1], -result[5], tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("gelu produces expected pattern", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  g <- ggml_gelu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, g)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(g)

  # GELU(0) should be 0
  expect_equal(result[3], 0, tolerance = 1e-5)

  # GELU should be monotonically increasing for positive values
  expect_lt(result[4], result[5])

  ggml_free(ctx)
})

test_that("gelu_quick produces similar results to gelu", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  g <- ggml_gelu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, g)
  ggml_graph_compute(ctx, graph)
  gelu_result <- ggml_get_f32(g)

  ggml_reset(ctx)

  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a2, c(-2, -1, 0, 1, 2))
  gq <- ggml_gelu_quick(ctx, a2)
  graph2 <- ggml_build_forward_expand(ctx, gq)
  ggml_graph_compute(ctx, graph2)
  quick_result <- ggml_get_f32(gq)

  # Results should be similar (within 10% for reasonable inputs)
  expect_equal(quick_result[3], gelu_result[3], tolerance = 0.1)

  ggml_free(ctx)
})

test_that("silu produces expected pattern", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  s <- ggml_silu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, s)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(s)

  # SiLU(0) = 0 * sigmoid(0) = 0
  expect_equal(result[3], 0, tolerance = 1e-5)

  # SiLU(x) = x * sigmoid(x), so SiLU(2) ~ 2 * 0.88 ~ 1.76
  expect_gt(result[5], 1.5)

  ggml_free(ctx)
})

test_that("softmax produces valid probability distribution", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  s <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, s)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(s)

  # All values should be positive
  expect_true(all(result > 0))

  # Sum should be approximately 1
  expect_equal(sum(result), 1, tolerance = 1e-5)

  # Values should be monotonically increasing (same as input)
  expect_true(all(diff(result) > 0))

  ggml_free(ctx)
})

test_that("ggml_elu produces correct output", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(-2, -1, 0, 1))

  result <- ggml_elu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # ELU: x if x > 0, alpha*(exp(x)-1) if x <= 0 (alpha=1)
  expect_equal(output[3], 0, tolerance = 1e-5)
  expect_equal(output[4], 1, tolerance = 1e-5)
  expect_true(output[1] < 0)  # negative for negative input
  expect_true(output[2] < 0)
})

test_that("ggml_leaky_relu produces correct output", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(-2, -1, 0, 1))

  # negative_slope = 0.01
  result <- ggml_leaky_relu(ctx, a, 0.01, FALSE)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output[1], -2 * 0.01, tolerance = 1e-5)
  expect_equal(output[4], 1, tolerance = 1e-5)
})

test_that("ggml_hardswish produces correct output", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-4, -3, 0, 3, 4))

  result <- ggml_hardswish(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # hardswish(x) = x * min(max(0, x+3), 6) / 6
  expect_equal(output[1], 0, tolerance = 1e-5)      # x <= -3
  expect_equal(output[2], 0, tolerance = 1e-5)      # x = -3
  expect_equal(output[3], 0, tolerance = 1e-5)      # x = 0
  expect_equal(output[4], 3, tolerance = 1e-5)      # x = 3
  expect_equal(output[5], 4, tolerance = 1e-5)      # x >= 3
})

test_that("ggml_hardsigmoid produces correct output", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-4, -3, 0, 3, 4))

  result <- ggml_hardsigmoid(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # hardsigmoid(x) = min(max(0, x+3), 6) / 6
  expect_equal(output[1], 0, tolerance = 1e-5)      # x <= -3
  expect_equal(output[3], 0.5, tolerance = 1e-5)    # x = 0
  expect_equal(output[5], 1, tolerance = 1e-5)      # x >= 3
})

test_that("ggml_softplus produces correct output", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(-1, 0, 1))

  result <- ggml_softplus(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # softplus(x) = log(1 + exp(x))
  expect_equal(output[2], log(2), tolerance = 1e-4)  # softplus(0) = ln(2)
  expect_true(all(output > 0))  # always positive
})

test_that("ggml_gelu_erf produces correct output", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(-1, 0, 1))

  result <- ggml_gelu_erf(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # GELU(0) = 0
  expect_equal(output[2], 0, tolerance = 1e-5)
  # GELU is asymmetric
  expect_true(output[1] < 0)
  expect_true(output[3] > 0)
})
