# Tests for Binary Operations (add, sub, mul, div)

# ============================================================================
# Subtraction
# ============================================================================

test_that("ggml_sub subtracts element-wise", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(10, 20, 30, 40, 50))
  ggml_set_f32(b, c(1, 2, 3, 4, 5))

  result <- ggml_sub(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(9, 18, 27, 36, 45), tolerance = 1e-5)
})

test_that("ggml_sub with negative result", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_f32(b, c(5, 5, 5, 5))

  result <- ggml_sub(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(-4, -3, -2, -1), tolerance = 1e-5)
})

test_that("ggml_sub a - a = 0", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_sub(ctx, a, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, rep(0, 5), tolerance = 1e-5)
})

test_that("ggml_sub on 2D tensors", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
  ggml_set_f32(a, c(10, 20, 30, 40, 50, 60))
  ggml_set_f32(b, c(1, 2, 3, 4, 5, 6))

  result <- ggml_sub(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(9, 18, 27, 36, 45, 54), tolerance = 1e-5)
})

# ============================================================================
# Division
# ============================================================================

test_that("ggml_div divides element-wise", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(10, 20, 30, 40, 50))
  ggml_set_f32(b, c(2, 4, 5, 8, 10))

  result <- ggml_div(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(5, 5, 6, 5, 5), tolerance = 1e-5)
})

test_that("ggml_div a / a = 1", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_div(ctx, a, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, rep(1, 5), tolerance = 1e-5)
})

test_that("ggml_div with fractional results", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 1, 1, 1))
  ggml_set_f32(b, c(2, 3, 4, 5))

  result <- ggml_div(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(0.5, 1/3, 0.25, 0.2), tolerance = 1e-5)
})

test_that("ggml_div on 2D tensors", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3)
  ggml_set_f32(a, c(2, 4, 6, 8, 10, 12))
  ggml_set_f32(b, c(1, 2, 2, 4, 5, 6))

  result <- ggml_div(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(2, 2, 3, 2, 2, 2), tolerance = 1e-5)
})

# ============================================================================
# Combined Operations
# ============================================================================

test_that("(a + b) - b = a", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))
  ggml_set_f32(b, c(10, 20, 30, 40, 50))

  sum_ab <- ggml_add(ctx, a, b)
  result <- ggml_sub(ctx, sum_ab, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(1, 2, 3, 4, 5), tolerance = 1e-5)
})

test_that("(a * b) / b = a", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))
  ggml_set_f32(b, c(2, 3, 4, 5, 6))

  prod_ab <- ggml_mul(ctx, a, b)
  result <- ggml_div(ctx, prod_ab, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(1, 2, 3, 4, 5), tolerance = 1e-4)
})

test_that("a / b * b = a", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(10, 20, 30, 40, 50))
  ggml_set_f32(b, c(2, 4, 5, 8, 10))

  div_ab <- ggml_div(ctx, a, b)
  result <- ggml_mul(ctx, div_ab, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(10, 20, 30, 40, 50), tolerance = 1e-4)
})

# ============================================================================
# Scalar Operations
# ============================================================================

test_that("ggml_add1 adds scalar to tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  scalar <- ggml_new_f32(ctx, 10)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_add1(ctx, a, scalar)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(11, 12, 13, 14, 15), tolerance = 1e-5)
})

test_that("ggml_scale multiplies by scalar", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_scale(ctx, a, 3)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(3, 6, 9, 12, 15), tolerance = 1e-5)
})
