# Tests for extended graph operations

test_that("graph_overhead returns positive value", {
  overhead <- ggml_graph_overhead()
  expect_type(overhead, "double")
  expect_gt(overhead, 0)
})

test_that("graph_n_nodes counts operations correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)

  # Single operation
  c <- ggml_add(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, c)
  expect_equal(ggml_graph_n_nodes(graph), 1)

  ggml_reset(ctx)

  # Chain of operations
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  c <- ggml_add(ctx, a, b)
  d <- ggml_mul(ctx, c, a)
  graph2 <- ggml_build_forward_expand(ctx, d)
  expect_gte(ggml_graph_n_nodes(graph2), 2)

  ggml_free(ctx)
})

test_that("graph_node retrieves correct node", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  c <- ggml_add(ctx, a, b)

  graph <- ggml_build_forward_expand(ctx, c)

  # Get last node (index -1 or n_nodes-1)
  node <- ggml_graph_node(graph, 0)
  expect_type(node, "externalptr")

  ggml_free(ctx)
})

# Note: ggml_graph_reset requires gradients to be enabled (for backprop)
# which we don't support yet, so we skip this test
# test_that("graph_reset works without error", { ... })
