# Tests for graph utility functions

test_that("ggml_graph_get_tensor finds tensor by name", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_name(a, "my_tensor")
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_input(a)

  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_name(b, "other")
  ggml_set_f32(b, c(5, 6, 7, 8))
  ggml_set_input(b)

  r <- ggml_add(ctx, a, b)
  ggml_set_name(r, "result")
  ggml_set_output(r)

  graph <- ggml_build_forward_expand(ctx, r)

  found <- ggml_graph_get_tensor(graph, "my_tensor")
  expect_type(found, "externalptr")
  expect_equal(ggml_get_name(found), "my_tensor")
})

test_that("ggml_graph_print runs without error", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_input(a)

  r <- ggml_relu(ctx, a)
  ggml_set_output(r)

  graph <- ggml_build_forward_expand(ctx, r)
  expect_no_error(capture.output(ggml_graph_print(graph)))
})

test_that("ggml_graph_reset runs without error", {
  skip("ggml_graph_reset requires a backward graph with gradients allocated")
})
