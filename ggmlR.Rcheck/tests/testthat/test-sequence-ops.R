# Tests for Sequence/Token Operations
# pad, argsort, top_k

test_that("ggml_pad works correctly for 1D", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(1, 2, 3))
  b <- ggml_pad(ctx, a, 2)  # pad 2 on right
  expect_equal(ggml_tensor_shape(b)[1], 5)

  graph <- ggml_build_forward_expand(ctx, b)
  ggml_graph_compute(ctx, graph)
  result <- ggml_get_f32(b)
  expect_equal(result, c(1, 2, 3, 0, 0))
})

test_that("ggml_pad works for 2D tensors", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2)
  ggml_set_f32(a, c(1, 2, 3, 4))
  b <- ggml_pad(ctx, a, 1, 1)  # pad 1 on dim0, 1 on dim1

  shape <- ggml_tensor_shape(b)
  expect_equal(shape[1], 3)
  expect_equal(shape[2], 3)
})

test_that("ggml_argsort ascending works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(3, 1, 4, 1, 5))

  indices <- ggml_argsort(ctx, a, GGML_SORT_ORDER_ASC)
  graph <- ggml_build_forward_expand(ctx, indices)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_i32(indices)
  # Sorted values: 1, 1, 3, 4, 5 at positions 1, 3, 0, 2, 4
  expect_equal(result[1], 1)  # First smallest at index 1
  expect_equal(result[5], 4)  # Largest at index 4
})

test_that("ggml_argsort descending works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(10, 30, 20, 40))

  indices <- ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC)
  graph <- ggml_build_forward_expand(ctx, indices)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_i32(indices)
  expect_equal(result[1], 3)  # Largest (40) at index 3
  expect_equal(result[4], 0)  # Smallest (10) at index 0
})

test_that("ggml_top_k returns correct indices", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6)
  ggml_set_f32(a, c(1, 6, 2, 5, 3, 4))

  top3 <- ggml_top_k(ctx, a, 3)
  graph <- ggml_build_forward_expand(ctx, top3)
  ggml_graph_compute(ctx, graph)

  result <- sort(ggml_get_i32(top3))
  # Top 3 values: 6(idx1), 5(idx3), 4(idx5)
  expect_equal(result, c(1, 3, 5))
})

test_that("sort order constants are defined", {
  expect_equal(GGML_SORT_ORDER_ASC, 0L)
  expect_equal(GGML_SORT_ORDER_DESC, 1L)
})
