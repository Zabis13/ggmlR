# Tests for mathematical operations

test_that("sqr squares elements correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  r <- ggml_sqr(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(1, 4, 9, 16, 25), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("sqr handles negative numbers", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(-3, 0, 3))

  r <- ggml_sqr(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(9, 0, 9), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("sqrt computes square root correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(0, 1, 4, 9, 16))

  r <- ggml_sqrt(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(0, 1, 2, 3, 4), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("log computes natural logarithm correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, exp(1), exp(2), exp(3)))

  r <- ggml_log(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(0, 1, 2, 3), tolerance = 1e-4)

  ggml_free(ctx)
})

test_that("exp computes exponential correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(0, 1, 2, 3))

  r <- ggml_exp(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(1, exp(1), exp(2), exp(3)), tolerance = 1e-4)

  ggml_free(ctx)
})

test_that("exp and log are inverse operations", {
  ctx <- ggml_init(16 * 1024 * 1024)

  original <- c(0.5, 1, 2, 3)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, original)

  # exp then log
  e <- ggml_exp(ctx, a)
  l <- ggml_log(ctx, e)
  graph <- ggml_build_forward_expand(ctx, l)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(l)
  expect_equal(result, original, tolerance = 1e-4)

  ggml_free(ctx)
})

test_that("abs computes absolute value correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-3, -1, 0, 1, 3))

  r <- ggml_abs(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(3, 1, 0, 1, 3), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("neg negates values correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-2, -1, 0, 1, 2))

  r <- ggml_neg(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(2, 1, 0, -1, -2), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("sin computes sine correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(0, pi/6, pi/4, pi/3, pi/2))

  r <- ggml_sin(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expected <- c(0, 0.5, sqrt(2)/2, sqrt(3)/2, 1)
  expect_equal(result, expected, tolerance = 1e-4)

  ggml_free(ctx)
})

test_that("cos computes cosine correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(0, pi/6, pi/4, pi/3, pi/2))

  r <- ggml_cos(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expected <- c(1, sqrt(3)/2, sqrt(2)/2, 0.5, 0)
  expect_equal(result, expected, tolerance = 1e-4)

  ggml_free(ctx)
})

test_that("sin^2 + cos^2 = 1 identity holds", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(0, 0.5, 1, 1.5, 2))

  s <- ggml_sin(ctx, a)
  c <- ggml_cos(ctx, a)
  s2 <- ggml_sqr(ctx, s)
  c2 <- ggml_sqr(ctx, c)
  sum_sc <- ggml_add(ctx, s2, c2)

  graph <- ggml_build_forward_expand(ctx, sum_sc)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(sum_sc)
  expect_equal(result, rep(1, 5), tolerance = 1e-4)

  ggml_free(ctx)
})

test_that("scale multiplies by scalar correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  r <- ggml_scale(ctx, a, 2.5)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(2.5, 5, 7.5, 10, 12.5), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("scale by zero produces zeros", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(1, 2, 3))

  r <- ggml_scale(ctx, a, 0)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(0, 0, 0), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("scale by negative inverts sign", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(1, 2, 3))

  r <- ggml_scale(ctx, a, -1)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(-1, -2, -3), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("clamp restricts values to range", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7)
  ggml_set_f32(a, c(-10, -2, 0, 2, 5, 10, 20))

  r <- ggml_clamp(ctx, a, -1, 6)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(-1, -1, 0, 2, 5, 6, 6), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("clamp with min=max produces constant", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-10, 0, 5, 10, 20))

  r <- ggml_clamp(ctx, a, 3, 3)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, rep(3, 5), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("floor rounds down correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6)
  ggml_set_f32(a, c(-1.9, -1.1, -0.5, 0.5, 1.1, 1.9))

  r <- ggml_floor(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(-2, -2, -1, 0, 1, 1), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("ceil rounds up correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6)
  ggml_set_f32(a, c(-1.9, -1.1, -0.5, 0.5, 1.1, 1.9))

  r <- ggml_ceil(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(-1, -1, 0, 1, 2, 2), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("round rounds to nearest integer", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6)
  ggml_set_f32(a, c(-1.6, -1.4, -0.5, 0.5, 1.4, 1.6))

  r <- ggml_round(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  # Note: rounding of 0.5 depends on implementation (banker's vs standard)
  expect_equal(result[1], -2, tolerance = 1e-5)
  expect_equal(result[2], -1, tolerance = 1e-5)
  expect_equal(result[5], 1, tolerance = 1e-5)
  expect_equal(result[6], 2, tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("floor <= ceil and round is between them", {
  ctx <- ggml_init(16 * 1024 * 1024)

  original <- c(-2.7, -1.3, 0.5, 1.8, 3.2)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, original)

  f <- ggml_floor(ctx, a)
  r <- ggml_round(ctx, a)
  c <- ggml_ceil(ctx, a)

  # Build and compute each operation separately
  graph_f <- ggml_build_forward_expand(ctx, f)
  ggml_graph_compute(ctx, graph_f)
  floor_result <- ggml_get_f32(f)

  graph_r <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph_r)
  round_result <- ggml_get_f32(r)

  graph_c <- ggml_build_forward_expand(ctx, c)
  ggml_graph_compute(ctx, graph_c)
  ceil_result <- ggml_get_f32(c)

  # floor <= ceil always
  expect_true(all(floor_result <= ceil_result))
  # round is always between floor and ceil (inclusive)
  expect_true(all(round_result >= floor_result & round_result <= ceil_result))

  ggml_free(ctx)
})

test_that("2D tensor math operations work correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
  ggml_set_f32(a, c(1, 2, 3, 4, 5, 6))

  r <- ggml_sqr(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(1, 4, 9, 16, 25, 36), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("chained math operations work correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Compute sqrt(sqr(x)) = |x|
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(-3, -1, 0, 1, 3))

  s <- ggml_sqr(ctx, a)
  r <- ggml_sqrt(ctx, s)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(result, c(3, 1, 0, 1, 3), tolerance = 1e-5)

  ggml_free(ctx)
})
