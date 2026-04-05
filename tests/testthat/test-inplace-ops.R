# Tests for inplace tensor operations

test_that("ggml_abs_inplace computes absolute value", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(-1, 2, -3, 4))
  ggml_set_input(a)

  r <- ggml_abs_inplace(ctx, a)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(result, c(1, 2, 3, 4), tolerance = 1e-5)
})

# Helper to test unary inplace ops
test_inplace_op <- function(op_name, op_fn, input, expected, tol = 1e-4) {
  test_that(paste0(op_name, " works"), {
    ctx <- ggml_init(1024 * 1024)
    on.exit(ggml_free(ctx))

    a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(input))
    ggml_set_f32(a, input)
    ggml_set_input(a)

    r <- op_fn(ctx, a)
    ggml_set_output(r)

    backend <- ggml_backend_cpu_init()
    on.exit(ggml_backend_free(backend), add = TRUE)
    ggml_backend_cpu_set_n_threads(backend, 2L)

    graph <- ggml_build_forward_expand(ctx, r)
    ggml_backend_graph_compute(backend, graph)
    result <- ggml_get_f32(r)
    expect_equal(result, expected, tolerance = tol)
  })
}

test_inplace_op("ggml_neg_inplace", ggml_neg_inplace,
                c(1, -2, 3, -4), c(-1, 2, -3, 4))

test_inplace_op("ggml_sqr_inplace", ggml_sqr_inplace,
                c(1, 2, 3, 4), c(1, 4, 9, 16))

test_inplace_op("ggml_sqrt_inplace", ggml_sqrt_inplace,
                c(1, 4, 9, 16), c(1, 2, 3, 4))

test_inplace_op("ggml_relu_inplace", ggml_relu_inplace,
                c(-1, 0, 1, 2), c(0, 0, 1, 2))

test_inplace_op("ggml_sigmoid_inplace", ggml_sigmoid_inplace,
                c(0, 0, 0, 0), c(0.5, 0.5, 0.5, 0.5))

test_inplace_op("ggml_tanh_inplace", ggml_tanh_inplace,
                c(0, 0, 0, 0), c(0, 0, 0, 0))

test_inplace_op("ggml_exp_inplace", ggml_exp_inplace,
                c(0, 1, 0, 0), c(1, exp(1), 1, 1))

test_inplace_op("ggml_silu_inplace", ggml_silu_inplace,
                c(0, 0, 0, 0), c(0, 0, 0, 0))

test_inplace_op("ggml_gelu_inplace", ggml_gelu_inplace,
                c(0, 0, 0, 0), c(0, 0, 0, 0))

test_inplace_op("ggml_ceil_inplace", ggml_ceil_inplace,
                c(1.1, 2.5, -0.1, 3.0), c(2, 3, 0, 3))

test_inplace_op("ggml_floor_inplace", ggml_floor_inplace,
                c(1.1, 2.5, -0.1, 3.0), c(1, 2, -1, 3))

test_inplace_op("ggml_round_inplace", ggml_round_inplace,
                c(1.1, 2.5, 2.6, 3.0), c(1, 3, 3, 3))

test_inplace_op("ggml_log_inplace", ggml_log_inplace,
                c(1, exp(1), exp(2), exp(3)), c(0, 1, 2, 3))

# Binary inplace ops need two input tensors
test_that("ggml_mul_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_f32(b, c(2, 3, 4, 5))
  ggml_set_input(a)
  ggml_set_input(b)

  r <- ggml_mul_inplace(ctx, a, b)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(result, c(2, 6, 12, 20), tolerance = 1e-5)
})

test_that("ggml_div_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(10, 20, 30, 40))
  ggml_set_f32(b, c(2, 4, 5, 8))
  ggml_set_input(a)
  ggml_set_input(b)

  r <- ggml_div_inplace(ctx, a, b)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(result, c(5, 5, 6, 5), tolerance = 1e-5)
})

test_that("ggml_sub_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(10, 20, 30, 40))
  ggml_set_f32(b, c(1, 2, 3, 4))
  ggml_set_input(a)
  ggml_set_input(b)

  r <- ggml_sub_inplace(ctx, a, b)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(result, c(9, 18, 27, 36), tolerance = 1e-5)
})

test_that("ggml_dup_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_input(a)

  r <- ggml_dup_inplace(ctx, a)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(result, c(1, 2, 3, 4), tolerance = 1e-5)
})

test_that("ggml_scale_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_input(a)

  r <- ggml_scale_inplace(ctx, a, 3.0)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(result, c(3, 6, 9, 12), tolerance = 1e-5)
})

test_that("ggml_soft_max_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_input(a)

  r <- ggml_soft_max_inplace(ctx, a)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(sum(result), 1.0, tolerance = 1e-5)
  expect_true(all(diff(result) > 0))
})

test_that("ggml_elu_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(-1, 0, 1, 2))
  ggml_set_input(a)

  r <- ggml_elu_inplace(ctx, a)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(result[2], 0.0, tolerance = 1e-5)
  expect_equal(result[3], 1.0, tolerance = 1e-5)
  expect_true(result[1] < 0)  # exp(-1)-1
})

test_that("ggml_softplus_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(0, 1, 2, 3))
  ggml_set_input(a)

  r <- ggml_softplus_inplace(ctx, a)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expected <- log(1 + exp(c(0, 1, 2, 3)))
  expect_equal(result, expected, tolerance = 1e-4)
})

test_that("ggml_diag_mask_inf_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 3)
  ggml_set_f32(a, rep(1.0, 9))
  ggml_set_input(a)

  r <- ggml_diag_mask_inf_inplace(ctx, a, 0L)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  # n_past=0: mask where i > j, so (i=1,j=0) at index 2 is -Inf
  expect_equal(result[1], 1.0, tolerance = 1e-5)
  expect_true(is.infinite(result[2]) && result[2] < 0)
})

test_that("ggml_norm_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_input(a)

  r <- ggml_norm_inplace(ctx, a, 1e-5)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(mean(result), 0.0, tolerance = 1e-4)
})

test_that("ggml_rms_norm_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_input(a)

  r <- ggml_rms_norm_inplace(ctx, a, 1e-5)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(sqrt(mean(result^2)), 1.0, tolerance = 1e-3)
})

test_that("ggml_l2_norm_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(3, 0, 4, 0))
  ggml_set_input(a)

  r <- ggml_l2_norm_inplace(ctx, a, 1e-5)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_equal(sqrt(sum(result^2)), 1.0, tolerance = 1e-4)
})

test_that("ggml_rope_inplace works", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # rope requires 3D+ tensor: ne[2] == b->ne[0] (n_positions)
  a <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 1)
  ggml_set_f32(a, rep(1.0, 8))
  ggml_set_input(a)

  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1)
  ggml_set_i32(pos, 0L)
  ggml_set_input(pos)

  r <- ggml_rope_inplace(ctx, a, pos, n_dims = 4L, mode = 0L)
  ggml_set_output(r)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_get_f32(r)
  expect_length(result, 8)
  # pos=0: no rotation applied
  expect_equal(result[1:4], rep(1.0, 4), tolerance = 1e-4)
})
