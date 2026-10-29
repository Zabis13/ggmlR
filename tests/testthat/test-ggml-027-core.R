# ggml 0.27 core changes in the graph builders (TODO 9.8 group 2)

test_that("ggml_clamp returns a new tensor and ggml_clamp_inplace a view", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))
  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(x, c(-2, -0.5, 0, 0.5, 2))
  y <- ggml_clamp(ctx, x, -1, 1)
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, y))
  expect_equal(ggml_get_f32(y), c(-1, -0.5, 0, 0.5, 1))
  expect_equal(ggml_get_f32(x), c(-2, -0.5, 0, 0.5, 2))   # input untouched

  z <- ggml_clamp_inplace(ctx, x, -1, 1)
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, z))
  expect_equal(ggml_get_f32(x), c(-1, -0.5, 0, 0.5, 1))   # written in place
})

test_that("ggml_set_input on a computed node is an R error, not an abort", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  expect_no_error(ggml_set_input(a))
  expect_error(ggml_set_input(ggml_add(ctx, a, a)), "leaf tensor")
})
