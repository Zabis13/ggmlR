# Tests for miscellaneous operations not covered elsewhere

# ============================================================================
# ggml_timestep_embedding
# ============================================================================

test_that("ggml_timestep_embedding works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  timesteps <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(timesteps, c(1.0, 10.0))
  ggml_set_input(timesteps)

  r <- ggml_timestep_embedding(ctx, timesteps, dim = 8L)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_length(result, 16)  # 8 dim * 2 timesteps
  expect_true(all(is.finite(result)))
})

# ============================================================================
# ggml_repeat_back
# ============================================================================

test_that("ggml_repeat_back works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(a, c(0, 0))
  ggml_set_input(a)

  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(b, c(1, 2, 3, 4))
  ggml_set_input(b)

  r <- ggml_repeat_back(ctx, b, a)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_length(result, 2)
  expect_equal(result, c(4, 6), tolerance = 1e-5)
})

# ============================================================================
# ggml_flash_attn_back
# ============================================================================

test_that("ggml_flash_attn_back returns externalptr", {
  ctx <- ggml_init(4 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  d <- 8L
  n <- 4L

  q <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n)
  k <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n)
  v <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n)

  ggml_set_f32(q, rnorm(d * n))
  ggml_set_f32(k, rnorm(d * n))
  ggml_set_f32(v, rnorm(d * n))
  ggml_set_input(q)
  ggml_set_input(k)
  ggml_set_input(v)

  skip("ggml_flash_attn_back not implemented (TODO in ggml.c)")
})

# ============================================================================
# ggml_rms_norm_back
# ============================================================================

test_that("ggml_rms_norm_back computes gradient", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_f32(b, c(1, 1, 1, 1))
  ggml_set_input(a)
  ggml_set_input(b)

  r <- ggml_rms_norm_back(ctx, a, b, 1e-5)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_length(result, 4)
  expect_true(all(is.finite(result)))
})

# ============================================================================
# ggml_silu_back
# ============================================================================

test_that("ggml_silu_back computes gradient", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(0, 1, -1, 2))
  ggml_set_f32(b, c(1, 1, 1, 1))
  ggml_set_input(a)
  ggml_set_input(b)

  # ggml_silu_back(ctx, grad, x): first arg is upstream gradient, second is input
  r <- ggml_silu_back(ctx, b, a)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_length(result, 4)
  expect_true(all(is.finite(result)))
  expect_equal(result[1], 0.5, tolerance = 1e-4)
})

# ============================================================================
# ggml_group_norm_inplace
# ============================================================================

test_that("ggml_group_norm_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
  ggml_set_f32(a, c(1, 2, 3, 4, 5, 6, 7, 8))
  ggml_set_input(a)

  r <- ggml_group_norm_inplace(ctx, a, 2L)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_length(result, 8)
  expect_true(all(is.finite(result)))
})
