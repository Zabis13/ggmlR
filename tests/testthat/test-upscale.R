# Tests for ggml_upscale and scale-mode constants (C2: 0.11.0 new BICUBIC mode).
# Previously exported without any functional test coverage.

test_that("scale-mode constants are defined correctly", {
  expect_equal(GGML_SCALE_MODE_NEAREST, 0L)
  expect_equal(GGML_SCALE_MODE_BILINEAR, 1L)
  expect_equal(GGML_SCALE_MODE_BICUBIC, 2L)
})

test_that("ggml_upscale nearest doubles spatial dims", {
  ctx <- ggml_init(16 * 1024 * 1024)
  # 2x2 single-channel single-batch image
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 1, 1)
  ggml_set_f32(a, c(1, 2, 3, 4))

  up <- ggml_upscale(ctx, a, 2L, GGML_SCALE_MODE_NEAREST)
  graph <- ggml_build_forward_expand(ctx, up)
  ggml_graph_compute(ctx, graph)

  # nearest 2x upscale of 2x2 -> 4x4
  shape <- ggml_tensor_shape(up)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 4)
  res <- ggml_get_f32(up)
  expect_equal(length(res), 16)
  # nearest: each source pixel replicated into a 2x2 block; corner stays = 1
  expect_equal(res[1], 1, tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("ggml_upscale bilinear produces interpolated (non-replicated) values", {
  ctx <- ggml_init(16 * 1024 * 1024)
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 1, 1)
  ggml_set_f32(a, c(0, 4, 8, 12))

  up <- ggml_upscale(ctx, a, 2L, GGML_SCALE_MODE_BILINEAR)
  graph <- ggml_build_forward_expand(ctx, up)
  ggml_graph_compute(ctx, graph)

  res <- ggml_get_f32(up)
  expect_equal(length(res), 16)
  expect_true(all(is.finite(res)))
  # interpolation must produce at least one value not present in the source
  expect_false(all(res %in% c(0, 4, 8, 12)))

  ggml_free(ctx)
})

test_that("ggml_upscale bicubic (0.11.0 new mode) runs and is finite", {
  ctx <- ggml_init(16 * 1024 * 1024)
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 1, 1)
  ggml_set_f32(a, as.numeric(1:16))

  up <- ggml_upscale(ctx, a, 2L, GGML_SCALE_MODE_BICUBIC)
  graph <- ggml_build_forward_expand(ctx, up)
  ggml_graph_compute(ctx, graph)

  shape <- ggml_tensor_shape(up)
  expect_equal(shape[1], 8)
  expect_equal(shape[2], 8)
  res <- ggml_get_f32(up)
  expect_equal(length(res), 64)
  expect_true(all(is.finite(res)))

  ggml_free(ctx)
})

test_that("ggml_upscale default mode equals explicit nearest", {
  ctx <- ggml_init(16 * 1024 * 1024)
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 2, 1, 1)
  ggml_set_f32(a, c(5, 6, 7, 8))

  up_default <- ggml_upscale(ctx, a, 2L)
  g1 <- ggml_build_forward_expand(ctx, up_default)
  ggml_graph_compute(ctx, g1)
  res_default <- ggml_get_f32(up_default)

  up_nearest <- ggml_upscale(ctx, a, 2L, GGML_SCALE_MODE_NEAREST)
  g2 <- ggml_build_forward_expand(ctx, up_nearest)
  ggml_graph_compute(ctx, g2)
  res_nearest <- ggml_get_f32(up_nearest)

  expect_equal(res_default, res_nearest, tolerance = 1e-6)

  ggml_free(ctx)
})
