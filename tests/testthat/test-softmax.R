# Tests for Softmax Operations

# ============================================================================
# Basic Softmax
# ============================================================================

test_that("ggml_soft_max computes softmax correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # Softmax properties
  expect_true(all(output > 0))
  expect_true(all(output < 1))
  expect_equal(sum(output), 1, tolerance = 1e-5)
})

test_that("ggml_soft_max outputs sum to 1", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a, rnorm(100))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(sum(output), 1, tolerance = 1e-5)
})

test_that("ggml_soft_max preserves order", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # Larger input -> larger softmax output
  expect_true(all(diff(output) > 0))
})

test_that("ggml_soft_max with uniform input gives uniform output", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  n <- 5
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  ggml_set_f32(a, rep(1, n))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # All equal -> uniform 1/n
  expect_equal(output, rep(1/n, n), tolerance = 1e-5)
})

test_that("ggml_soft_max handles negative inputs", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-5, -3, -1, 0, 1))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  expect_true(all(output > 0))
  expect_equal(sum(output), 1, tolerance = 1e-5)
})

# ============================================================================
# Softmax In-place
# ============================================================================

test_that("ggml_soft_max_inplace computes in-place", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_soft_max_inplace(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  expect_true(all(output > 0))
  expect_equal(sum(output), 1, tolerance = 1e-5)
})

# ============================================================================
# Extended Softmax (with scale)
# ============================================================================

test_that("ggml_soft_max_ext with scale=1 matches basic softmax", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a1, c(1, 2, 3, 4, 5))
  ggml_set_f32(a2, c(1, 2, 3, 4, 5))

  r1 <- ggml_soft_max(ctx, a1)
  r2 <- ggml_soft_max_ext(ctx, a2, mask = NULL, scale = 1.0, max_bias = 0.0)

  g1 <- ggml_build_forward_expand(ctx, r1)
  ggml_graph_compute(ctx, g1)
  out1 <- ggml_get_f32(r1)

  g2 <- ggml_build_forward_expand(ctx, r2)
  ggml_graph_compute(ctx, g2)
  out2 <- ggml_get_f32(r2)

  expect_equal(out1, out2, tolerance = 1e-5)
})

test_that("ggml_soft_max_ext with scale < 1 produces flatter distribution", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a1, c(1, 2, 3, 4, 5))
  ggml_set_f32(a2, c(1, 2, 3, 4, 5))

  r1 <- ggml_soft_max_ext(ctx, a1, scale = 1.0)
  r2 <- ggml_soft_max_ext(ctx, a2, scale = 0.5)  # Lower temp = flatter

  g1 <- ggml_build_forward_expand(ctx, r1)
  ggml_graph_compute(ctx, g1)
  out1 <- ggml_get_f32(r1)

  g2 <- ggml_build_forward_expand(ctx, r2)
  ggml_graph_compute(ctx, g2)
  out2 <- ggml_get_f32(r2)

  # Lower scale -> flatter distribution (max is smaller)
  expect_lt(max(out2), max(out1))
})

test_that("ggml_soft_max_ext with scale > 1 produces sharper distribution", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a1, c(1, 2, 3, 4, 5))
  ggml_set_f32(a2, c(1, 2, 3, 4, 5))

  r1 <- ggml_soft_max_ext(ctx, a1, scale = 1.0)
  r2 <- ggml_soft_max_ext(ctx, a2, scale = 2.0)  # Higher temp = sharper

  g1 <- ggml_build_forward_expand(ctx, r1)
  ggml_graph_compute(ctx, g1)
  out1 <- ggml_get_f32(r1)

  g2 <- ggml_build_forward_expand(ctx, r2)
  ggml_graph_compute(ctx, g2)
  out2 <- ggml_get_f32(r2)

  # Higher scale -> sharper distribution (max is larger)
  expect_gt(max(out2), max(out1))
})

# ============================================================================
# 2D Softmax
# ============================================================================

test_that("ggml_soft_max on 2D tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # 4x3 tensor
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  ggml_set_f32(a, as.numeric(1:12))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # All values should be positive
  expect_true(all(output > 0))
  expect_true(all(output < 1))
})

# ============================================================================
# Numerical Stability
# ============================================================================

test_that("ggml_soft_max handles large values", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(100, 200, 300, 400, 500))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # Should not overflow
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))
  expect_equal(sum(output), 1, tolerance = 1e-4)
})

test_that("ggml_soft_max handles very negative values", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-500, -400, -300, -200, -100))

  result <- ggml_soft_max(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # Should not underflow to zero
  expect_false(any(is.na(output)))
  expect_equal(sum(output), 1, tolerance = 1e-4)
})
