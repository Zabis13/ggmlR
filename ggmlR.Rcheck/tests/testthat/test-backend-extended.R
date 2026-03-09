# Tests for extended backend and graph introspection functions

test_that("async graph compute works", {
  skip_on_cran()

  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  # Create simple computation
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  c <- ggml_add(ctx, a, b)

  # Build graph
  graph <- ggml_build_forward_expand(ctx, c)

  # Get CPU backend
  backend <- ggml_backend_cpu_init()
  skip_if(is.null(backend), "CPU backend not available")
  on.exit(ggml_backend_free(backend), add = TRUE)

  # Allocate tensors (requires no_alloc = TRUE context)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  # Set data
  ggml_backend_tensor_set_data(a, c(1, 2, 3, 4))
  ggml_backend_tensor_set_data(b, c(5, 6, 7, 8))

  # Async compute
  status <- ggml_backend_graph_compute_async(backend, graph)
  expect_type(status, "integer")

  # Synchronize
  ggml_backend_synchronize(backend)

  # Check result
  result <- ggml_backend_tensor_get_data(c)
  expect_equal(result, c(6, 8, 10, 12), tolerance = 1e-5)
})

test_that("multi-buffer functions work", {
  skip_on_cran()

  backend <- ggml_backend_cpu_init()
  skip_if(is.null(backend), "CPU backend not available")
  on.exit(ggml_backend_free(backend), add = TRUE)

  # This test checks if the functions exist and work at basic level
  # Creating actual multi-buffers requires lower-level buffer allocation

  # Test is_multi_buffer on regular buffer
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  # Allocate tensors (requires no_alloc = TRUE context)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  # Regular buffer should not be multi-buffer
  is_multi <- ggml_backend_buffer_is_multi_buffer(buffer)
  expect_type(is_multi, "logical")
  expect_false(is_multi)
})

test_that("graph_view works", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create a chain of operations
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_add(ctx, a, a)
  c <- ggml_mul(ctx, b, b)
  d <- ggml_sqrt(ctx, c)

  # Build graph
  graph <- ggml_build_forward_expand(ctx, d)

  # Check graph has nodes
  n_nodes <- ggml_graph_n_nodes(graph)
  expect_gte(n_nodes, 3)

  # Create view of first 2 nodes
  if (n_nodes >= 2) {
    view <- ggml_graph_view(graph, 0, 2)
    expect_false(is.null(view))
  }
})

test_that("op_can_inplace works", {
  # Test some known operations
  # ADD operation (op code 1) can typically be inplace
  # We test that the function returns logical values

  result <- ggml_op_can_inplace(1L)  # GGML_OP_ADD
  expect_type(result, "logical")

  result <- ggml_op_can_inplace(0L)  # GGML_OP_NONE
  expect_type(result, "logical")
})

test_that("are_same_layout works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # Same shape and type -> same layout
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
  expect_true(ggml_are_same_layout(a, b))

  # Different shape -> different layout
  c <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
  expect_false(ggml_are_same_layout(a, c))

  # Different type -> different layout
  d <- ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 4)
  expect_false(ggml_are_same_layout(a, d))

  # 1D tensors with same size
  e <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16)
  f <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16)
  expect_true(ggml_are_same_layout(e, f))
})

test_that("backend_register and device_register exist", {
  # These are advanced functions that require registry/device pointers

  # We just check that they exist and are callable

  expect_true(is.function(ggml_backend_register))
  expect_true(is.function(ggml_backend_device_register))
})
