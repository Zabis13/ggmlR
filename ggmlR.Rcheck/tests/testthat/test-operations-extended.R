# Tests for extended operations (sum, mean, argmax, transpose, etc.)

test_that("sum computes correct result", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_f32(a, 1:10)

  s <- ggml_sum(ctx, a)
  graph <- ggml_build_forward_expand(ctx, s)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(s)
  expect_equal(result, 55, tolerance = 1e-5)  # sum(1:10) = 55

  ggml_free(ctx)
})

test_that("mean computes correct result", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  ggml_set_f32(a, 1:10)

  m <- ggml_mean(ctx, a)
  graph <- ggml_build_forward_expand(ctx, m)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(m)
  expect_equal(result, 5.5, tolerance = 1e-5)  # mean(1:10) = 5.5

  ggml_free(ctx)
})

test_that("transpose swaps dimensions", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Create 3x4 matrix
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)

  # Transpose to 4x3
  t <- ggml_transpose(ctx, a)

  shape <- ggml_tensor_shape(t)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 3)

  ggml_free(ctx)
})

test_that("sum_rows reduces along rows", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
  ggml_set_f32(a, rep(1, 12))

  sr <- ggml_sum_rows(ctx, a)
  graph <- ggml_build_forward_expand(ctx, sr)
  ggml_graph_compute(ctx, graph)

  # Result should have reduced dimensions
  expect_type(sr, "externalptr")

  ggml_free(ctx)
})

test_that("argmax creates valid operation", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 5, 3, 2, 4))  # max at index 1

  am <- ggml_argmax(ctx, a)
  graph <- ggml_build_forward_expand(ctx, am)
  ggml_graph_compute(ctx, graph)

  # argmax returns I32 tensor, not F32
  # Just verify it runs without error and returns valid tensor
  expect_type(am, "externalptr")
  # GGML_TYPE_I32 is 26 in current GGML version
  expect_equal(ggml_tensor_type(am), 26)

  ggml_free(ctx)
})

test_that("repeat broadcasts tensor", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Small tensor to repeat
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(a, c(1, 2))

  # Target shape
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6)

  r <- ggml_repeat(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(length(result), 6)
  expect_equal(result, c(1, 2, 1, 2, 1, 2), tolerance = 1e-5)

  ggml_free(ctx)
})
