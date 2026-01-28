# Tests for CPU Backend

# ============================================================================
# Basic CPU Backend Tests
# ============================================================================

test_that("ggml_backend_cpu_init creates CPU backend", {
  backend <- ggml_backend_cpu_init()

  expect_type(backend, "externalptr")
  expect_false(is.null(backend))

  ggml_backend_free(backend)
})

test_that("ggml_backend_name returns 'CPU' for CPU backend", {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend))

  name <- ggml_backend_name(backend)
  expect_type(name, "character")
  expect_equal(name, "CPU")
})

test_that("ggml_backend_cpu_set_n_threads works", {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend))

  # Should not error with valid thread counts
  expect_no_error(ggml_backend_cpu_set_n_threads(backend, 1L))
  expect_no_error(ggml_backend_cpu_set_n_threads(backend, 4L))
  expect_no_error(ggml_backend_cpu_set_n_threads(backend, 8L))
})

test_that("ggml_backend_free works without error", {
  backend <- ggml_backend_cpu_init()
  expect_no_error(ggml_backend_free(backend))
})

# ============================================================================
# CPU Backend Graph Computation Tests
# ============================================================================

test_that("CPU backend computes simple addition", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_add(ctx, a, b)

  # Setup CPU backend
  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  # Set data
  ggml_backend_tensor_set_data(a, c(1, 2, 3, 4, 5))
  ggml_backend_tensor_set_data(b, c(10, 20, 30, 40, 50))

  # Compute
  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  # Get result
  output <- ggml_backend_tensor_get_data(result)
  expect_equal(output, c(11, 22, 33, 44, 55), tolerance = 1e-5)

  # Cleanup
  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes subtraction", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_sub(ctx, a, b)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, c(10, 20, 30, 40, 50))
  ggml_backend_tensor_set_data(b, c(1, 2, 3, 4, 5))

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  expect_equal(output, c(9, 18, 27, 36, 45), tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes multiplication", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_mul(ctx, a, b)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, c(1, 2, 3, 4, 5))
  ggml_backend_tensor_set_data(b, c(2, 3, 4, 5, 6))

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  expect_equal(output, c(2, 6, 12, 20, 30), tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes division", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_div(ctx, a, b)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, c(10, 20, 30, 40, 50))
  ggml_backend_tensor_set_data(b, c(2, 4, 5, 8, 10))

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  expect_equal(output, c(5, 5, 6, 5, 5), tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# CPU Backend with Multiple Threads
# ============================================================================

test_that("CPU backend computation with multiple threads", {
  ctx <- ggml_init(64 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  # Larger tensor for multi-threaded computation
  n <- 10000
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  result <- ggml_add(ctx, a, b)

  backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(backend, 4L)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  a_data <- as.numeric(seq(1, n))
  b_data <- as.numeric(seq(n, 1))
  ggml_backend_tensor_set_data(a, a_data)
  ggml_backend_tensor_set_data(b, b_data)

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  # All sums should be n + 1
  expect_equal(output, rep(n + 1, n), tolerance = 1e-4)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# CPU Backend with Complex Operations
# ============================================================================

test_that("CPU backend computes chained operations", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)

  # Chain: sqrt(sqr(a)) = |a|
  sqr_a <- ggml_sqr(ctx, a)
  result <- ggml_sqrt(ctx, sqr_a)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, c(-3, -1, 0, 1, 3))

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  expect_equal(output, c(3, 1, 0, 1, 3), tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes matrix multiplication", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  # In GGML mul_mat: a[k,n] @ b[k,m] -> result[n,m]
  # First dimension (k) must match
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)  # k=4, n=3
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)  # k=4, m=2
  result <- ggml_mul_mat(ctx, a, b)  # result: [3, 2]

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  # Set simple data
  ggml_backend_tensor_set_data(a, rep(1, 12))  # 4*3 = 12
  ggml_backend_tensor_set_data(b, rep(1, 8))   # 4*2 = 8

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  # Result should be 3x2 = 6 elements
  expect_length(output, 6)
  expect_false(any(is.na(output)))
  # With all ones: each element = dot product of 4 ones = 4
  expect_equal(output, rep(4, 6), tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes RMS normalization", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  result <- ggml_rms_norm(ctx, a, 1e-5)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  input <- c(1, 2, 3, 4)
  ggml_backend_tensor_set_data(a, input)

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)

  # RMS = sqrt(mean(x^2))
  rms <- sqrt(mean(input^2))
  expected <- input / rms
  expect_equal(output, expected, tolerance = 1e-4)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes softmax", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_soft_max(ctx, a)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  input <- c(1, 2, 3, 4, 5)
  ggml_backend_tensor_set_data(a, input)

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)

  # Softmax properties
  expect_true(all(output >= 0))
  expect_true(all(output <= 1))
  expect_equal(sum(output), 1, tolerance = 1e-5)

  # Larger input -> larger softmax output
  expect_true(all(diff(output) > 0))

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# CPU Backend with 2D/3D Tensors
# ============================================================================

test_that("CPU backend computes on 2D tensors", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  result <- ggml_add(ctx, a, b)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, as.numeric(1:12))
  ggml_backend_tensor_set_data(b, rep(10.0, 12))

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  expect_equal(output, 11:22, tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes transpose", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  # 3x4 matrix
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
  result <- ggml_transpose(ctx, a)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, as.numeric(1:12))

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  # Check shape changed
  shape <- ggml_tensor_shape(result)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 3)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# CPU Backend Activation Functions
# ============================================================================

test_that("CPU backend computes GELU", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_gelu(ctx, a)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  input <- c(-2, -1, 0, 1, 2)
  ggml_backend_tensor_set_data(a, input)

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)

  # GELU properties
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))
  # GELU(0) = 0
  expect_equal(output[3], 0, tolerance = 1e-5)
  # GELU is approximately monotonic for x > -2 (positive values increase)
  expect_true(output[5] > output[4])  # GELU(2) > GELU(1)
  expect_true(output[4] > output[3])  # GELU(1) > GELU(0)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes SILU (SwiSH)", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_silu(ctx, a)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  input <- c(-2, -1, 0, 1, 2)
  ggml_backend_tensor_set_data(a, input)

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)

  # SILU(x) = x * sigmoid(x)
  # SILU(0) = 0
  expect_equal(output[3], 0, tolerance = 1e-5)
  # SILU is not strictly monotonic but should be smooth
  expect_false(any(is.na(output)))

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

test_that("CPU backend computes ReLU", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  result <- ggml_relu(ctx, a)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  input <- c(-2, -1, 0, 1, 2)
  ggml_backend_tensor_set_data(a, input)

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)
  expect_equal(output, c(0, 0, 0, 1, 2), tolerance = 1e-5)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# CPU Backend vs Standard Computation Comparison
# ============================================================================

test_that("CPU backend produces same results as ggml_graph_compute", {
  # Test with ggml_graph_compute (context-based)
  ctx1 <- ggml_init(16 * 1024 * 1024)
  a1 <- ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 5)
  ggml_set_f32(a1, c(1, 2, 3, 4, 5))
  r1 <- ggml_sqr(ctx1, a1)
  graph1 <- ggml_build_forward_expand(ctx1, r1)
  ggml_graph_compute(ctx1, graph1)
  result1 <- ggml_get_f32(r1)
  ggml_free(ctx1)

  # Test with CPU backend
  ctx2 <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx2, TRUE)
  a2 <- ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 5)
  r2 <- ggml_sqr(ctx2, a2)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx2, backend)
  ggml_backend_tensor_set_data(a2, c(1, 2, 3, 4, 5))

  graph2 <- ggml_build_forward_expand(ctx2, r2)
  ggml_backend_graph_compute(backend, graph2)
  result2 <- ggml_backend_tensor_get_data(r2)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx2)

  # Results should match
  expect_equal(result1, result2, tolerance = 1e-6)
  expect_equal(result1, c(1, 4, 9, 16, 25))
})

# ============================================================================
# CPU Backend GLU Operations
# ============================================================================

test_that("CPU backend computes SwiGLU", {
  ctx <- ggml_init(16 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  # SwiGLU expects input of size 2*hidden_dim
  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256)
  result <- ggml_swiglu(ctx, x)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  x_data <- seq(-2, 2, length.out = 256)
  ggml_backend_tensor_set_data(x, x_data)

  graph <- ggml_build_forward_expand(ctx, result)
  status <- ggml_backend_graph_compute(backend, graph)

  expect_equal(status, 0L)

  output <- ggml_backend_tensor_get_data(result)

  expect_length(output, 128)  # Output is half input size
  expect_false(any(is.na(output)))
  expect_false(any(is.infinite(output)))

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})

# ============================================================================
# Error Handling
# ============================================================================

test_that("CPU backend handles empty graph gracefully", {
  ctx <- ggml_init(1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  # Create a tensor but don't add operations
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)

  backend <- ggml_backend_cpu_init()
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  # Build graph with just the input tensor
  graph <- ggml_build_forward_expand(ctx, a)

  # This should still work (no-op computation)
  status <- ggml_backend_graph_compute(backend, graph)
  expect_equal(status, 0L)

  ggml_backend_buffer_free(buffer)
  ggml_backend_free(backend)
  ggml_free(ctx)
})
