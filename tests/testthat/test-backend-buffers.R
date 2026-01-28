# Tests for Backend Buffer Operations

# ============================================================================
# Backend Buffer Management
# ============================================================================

test_that("ggml_backend_alloc_ctx_tensors allocates buffer", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  expect_type(buffer, "externalptr")
  expect_false(is.null(buffer))

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("ggml_backend_buffer_name returns buffer name", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  name <- ggml_backend_buffer_name(buffer)
  expect_type(name, "character")
  expect_gt(nchar(name), 0)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("ggml_backend_buffer_get_size returns positive size", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  size <- ggml_backend_buffer_get_size(buffer)
  expect_gt(size, 0)
  # Should be at least 1000 * 4 bytes
  expect_gte(size, 4000)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("ggml_backend_buffer_free works without error", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  expect_no_error(ggml_backend_buffer_free(buffer))

  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# Backend Tensor Data Operations
# ============================================================================

test_that("ggml_backend_tensor_set_data and get_data roundtrip", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  data_in <- c(1.1, 2.2, 3.3, 4.4, 5.5)
  ggml_backend_tensor_set_data(a, data_in)
  data_out <- ggml_backend_tensor_get_data(a)

  expect_equal(data_out, data_in, tolerance = 1e-6)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("ggml_backend_tensor_set_data handles large tensors", {
  ctx <- ggml_init(64 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  n <- 10000
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  data_in <- rnorm(n)
  ggml_backend_tensor_set_data(a, data_in)
  data_out <- ggml_backend_tensor_get_data(a)

  expect_equal(data_out, data_in, tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("ggml_backend_tensor_set_data handles 2D tensors", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  data_in <- as.numeric(1:20)
  ggml_backend_tensor_set_data(a, data_in)
  data_out <- ggml_backend_tensor_get_data(a)

  expect_equal(data_out, data_in, tolerance = 1e-6)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# Graph Allocator
# ============================================================================

test_that("ggml_gallocr_new creates allocator", {
  galloc <- ggml_gallocr_new()
  expect_type(galloc, "externalptr")
  expect_false(is.null(galloc))

  ggml_gallocr_free(galloc)
})

test_that("ggml_gallocr_reserve and alloc_graph work", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  c <- ggml_add(ctx, a, b)

  graph <- ggml_build_forward_expand(ctx, c)

  galloc <- ggml_gallocr_new()

  # Reserve based on graph
  success <- ggml_gallocr_reserve(galloc, graph)
  expect_true(success)

  # Allocate graph
  result <- ggml_gallocr_alloc_graph(galloc, graph)
  expect_true(result)

  ggml_gallocr_free(galloc)
})

test_that("ggml_gallocr_get_buffer_size returns size", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  c <- ggml_add(ctx, a, b)

  graph <- ggml_build_forward_expand(ctx, c)

  galloc <- ggml_gallocr_new()
  ggml_gallocr_reserve(galloc, graph)

  size <- ggml_gallocr_get_buffer_size(galloc, buffer_id = 0)
  expect_type(size, "double")
  expect_gte(size, 0)

  ggml_gallocr_free(galloc)
})

test_that("ggml_gallocr_free is safe to call", {
  galloc <- ggml_gallocr_new()
  expect_no_error(ggml_gallocr_free(galloc))
})

# ============================================================================
# Multiple Buffers
# ============================================================================

test_that("multiple buffers can be allocated", {
  ctx1 <- ggml_init(16 * 1024 * 1024)
  ctx2 <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx1, TRUE)
  ggml_set_no_alloc(ctx2, TRUE)

  a1 <- ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 100)
  a2 <- ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 200)

  backend <- ggml_backend_cpu_init()

  buffer1 <- ggml_backend_alloc_ctx_tensors(ctx1, backend)
  buffer2 <- ggml_backend_alloc_ctx_tensors(ctx2, backend)

  expect_type(buffer1, "externalptr")
  expect_type(buffer2, "externalptr")

  ggml_backend_buffer_free(buffer1)
  ggml_backend_buffer_free(buffer2)
  ggml_backend_free(backend)
  ggml_free(ctx1)
  ggml_free(ctx2)
})
