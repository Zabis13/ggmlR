# Tests for Reduction Operations (sum, mean, argmax, etc.)

# ============================================================================
# Sum
# ============================================================================

test_that("ggml_sum computes total sum", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_sum(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 15, tolerance = 1e-5)
})

test_that("ggml_sum on 2D tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
  ggml_set_f32(a, as.numeric(1:12))

  result <- ggml_sum(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, sum(1:12), tolerance = 1e-5)
})

test_that("ggml_sum of zeros", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_f32(a, rep(0, 10))

  result <- ggml_sum(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 0, tolerance = 1e-5)
})

test_that("ggml_sum handles negative numbers", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(-5, -3, 3, 5))

  result <- ggml_sum(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 0, tolerance = 1e-5)
})

# ============================================================================
# Sum Rows
# ============================================================================

test_that("ggml_sum_rows computes row sums", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # 3x2 matrix (column-major: each column is a "row" in ggml terms)
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
  # Column 1: [1, 2, 3], Column 2: [4, 5, 6]
  ggml_set_f32(a, c(1, 2, 3, 4, 5, 6))

  result <- ggml_sum_rows(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # Sum along first dimension: [1+2+3, 4+5+6] = [6, 15]
  expect_length(output, 2)
  expect_equal(output, c(6, 15), tolerance = 1e-5)
})

# ============================================================================
# Mean
# ============================================================================

test_that("ggml_mean computes average", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(2, 4, 6, 8, 10))

  result <- ggml_mean(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 6, tolerance = 1e-5)  # mean = 30/5 = 6
})

test_that("ggml_mean on 2D tensor computes row-wise mean", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # ggml_mean computes mean along first dimension (per column)
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  ggml_set_f32(a, as.numeric(1:12))  # [1,2,3,4; 5,6,7,8; 9,10,11,12] column-major

  result <- ggml_mean(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # Mean of each column: mean([1,2,3,4])=2.5, mean([5,6,7,8])=6.5, mean([9,10,11,12])=10.5
  expect_equal(output, c(2.5, 6.5, 10.5), tolerance = 1e-5)
})

test_that("ggml_mean of constant tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  ggml_set_f32(a, rep(7.5, 100))

  result <- ggml_mean(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 7.5, tolerance = 1e-5)
})

# ============================================================================
# Argmax
# ============================================================================

test_that("ggml_argmax finds index of maximum", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 5, 3, 2, 4))

  result <- ggml_argmax(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_i32(result)
  expect_equal(output, 1L)  # Index 1 has value 5 (0-indexed)
})

test_that("ggml_argmax with first element max", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(10, 5, 3, 1))

  result <- ggml_argmax(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_i32(result)
  expect_equal(output, 0L)  # Index 0 has maximum
})

test_that("ggml_argmax with last element max", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 3, 5, 10))

  result <- ggml_argmax(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_i32(result)
  expect_equal(output, 3L)  # Index 3 has maximum
})

test_that("ggml_argmax handles negative values", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(-10, -5, -1, -3))

  result <- ggml_argmax(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_i32(result)
  expect_equal(output, 2L)  # Index 2 has -1 (largest)
})

# ============================================================================
# Combined Reduction Tests
# ============================================================================

test_that("sum and mean are consistent", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  n <- 10
  a1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  data <- rnorm(n)
  ggml_set_f32(a1, data)
  ggml_set_f32(a2, data)

  sum_result <- ggml_sum(ctx, a1)
  mean_result <- ggml_mean(ctx, a2)

  graph1 <- ggml_build_forward_expand(ctx, sum_result)
  ggml_graph_compute(ctx, graph1)
  sum_val <- ggml_get_f32(sum_result)

  graph2 <- ggml_build_forward_expand(ctx, mean_result)
  ggml_graph_compute(ctx, graph2)
  mean_val <- ggml_get_f32(mean_result)

  expect_equal(sum_val / n, mean_val, tolerance = 1e-4)
})

# ============================================================================
# Repeat
# ============================================================================

test_that("ggml_repeat replicates tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Source tensor 1x2
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2)
  ggml_set_f32(a, c(1, 2))

  # Target shape 3x2 (repeat 3 times along first dim)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)

  result <- ggml_repeat(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 6)
  # Pattern should repeat: [1,1,1, 2,2,2]
  expect_equal(output, c(1, 1, 1, 2, 2, 2), tolerance = 1e-5)
})
