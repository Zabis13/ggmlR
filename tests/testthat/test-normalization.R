# Tests for Normalization Functions

# ============================================================================
# Layer Normalization
# ============================================================================

test_that("ggml_norm computes layer normalization", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))

  result <- ggml_norm(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # Layer norm: (x - mean) / sqrt(var + eps)
  input <- c(1, 2, 3, 4)
  m <- mean(input)
  v <- var(input) * (length(input) - 1) / length(input)  # population variance
  expected <- (input - m) / sqrt(v + 1e-5)

  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_norm_inplace works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))

  result <- ggml_norm_inplace(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # Result should be normalized (mean ~ 0, std ~ 1)
  expect_equal(mean(output), 0, tolerance = 1e-4)
  expect_equal(sd(output) * sqrt(3/4), 1, tolerance = 1e-4)  # adjust for population std
})

test_that("ggml_norm on 2D tensor normalizes along first dimension", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # 4x2 tensor
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
  ggml_set_f32(a, c(1, 2, 3, 4, 5, 6, 7, 8))

  result <- ggml_norm(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 8)
  expect_false(any(is.na(output)))
})

# ============================================================================
# RMS Normalization
# ============================================================================

test_that("ggml_rms_norm computes RMS normalization", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))

  result <- ggml_rms_norm(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # RMS norm: x / sqrt(mean(x^2) + eps)
  input <- c(1, 2, 3, 4)
  rms <- sqrt(mean(input^2) + 1e-5)
  expected <- input / rms

  expect_equal(output, expected, tolerance = 1e-4)
})

test_that("ggml_rms_norm_inplace works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))

  result <- ggml_rms_norm_inplace(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # After RMS norm, sqrt(mean(output^2)) should be ~1
  rms_output <- sqrt(mean(output^2))
  expect_equal(rms_output, 1, tolerance = 1e-4)
})

test_that("ggml_rms_norm with different epsilon values", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(0.001, 0.002, 0.003, 0.004))  # Small values

  # With larger epsilon, normalization is more stable
  result <- ggml_rms_norm(ctx, a, eps = 1e-3)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))
})

# ============================================================================
# Group Normalization
# ============================================================================

test_that("ggml_group_norm computes group normalization", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # For group norm, tensor should be at least 2D
  # Shape: [4, 8] with 2 groups means each group has 4 channels
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
  ggml_set_f32(a, rnorm(32))

  result <- ggml_group_norm(ctx, a, n_groups = 2, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 32)
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))
})

test_that("ggml_group_norm_inplace works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
  ggml_set_f32(a, rnorm(32))

  result <- ggml_group_norm_inplace(ctx, a, n_groups = 2, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 32)
  expect_false(any(is.na(output)))
})

# ============================================================================
# L2 Normalization
# ============================================================================

test_that("ggml_l2_norm normalizes to unit length", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(3, 0, 4, 0))  # L2 norm = 5

  result <- ggml_l2_norm(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # L2 norm of output should be 1
  l2_norm <- sqrt(sum(output^2))
  expect_equal(l2_norm, 1, tolerance = 1e-4)

  # Check values
  expect_equal(output, c(0.6, 0, 0.8, 0), tolerance = 1e-4)
})

test_that("ggml_l2_norm_inplace works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 2, 4))  # L2 norm = 5

  result <- ggml_l2_norm_inplace(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)

  # L2 norm of output should be 1
  l2_norm <- sqrt(sum(output^2))
  expect_equal(l2_norm, 1, tolerance = 1e-4)
})

test_that("ggml_l2_norm handles near-zero vectors with epsilon", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(0.0001, 0.0001, 0.0001, 0.0001))  # Very small

  result <- ggml_l2_norm(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))
})

# ============================================================================
# RMS Norm Backward (for training)
# ============================================================================

test_that("ggml_rms_norm_back computes gradient", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Forward input
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))

  # Upstream gradient
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(b, c(1, 1, 1, 1))  # All ones for simplicity

  result <- ggml_rms_norm_back(ctx, a, b, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 4)
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))
})

# ============================================================================
# Edge Cases
# ============================================================================

test_that("normalization handles single element", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(a, 5)

  # RMS norm of single element
  result <- ggml_rms_norm(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_false(is.na(output))
})

test_that("normalization handles large tensors", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  n <- 4096  # Typical hidden size
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  ggml_set_f32(a, rnorm(n))

  result <- ggml_rms_norm(ctx, a, eps = 1e-5)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  rms_output <- sqrt(mean(output^2))
  expect_equal(rms_output, 1, tolerance = 1e-3)
})
