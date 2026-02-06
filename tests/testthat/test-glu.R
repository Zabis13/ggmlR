# Tests for GLU (Gated Linear Unit) operations

test_that("GLU constants are defined correctly", {
  expect_equal(GGML_GLU_OP_REGLU, 0L)
  expect_equal(GGML_GLU_OP_GEGLU, 1L)
  expect_equal(GGML_GLU_OP_SWIGLU, 2L)
  expect_equal(GGML_GLU_OP_SWIGLU_OAI, 3L)
  expect_equal(GGML_GLU_OP_GEGLU_ERF, 4L)
  expect_equal(GGML_GLU_OP_GEGLU_QUICK, 5L)
})

test_that("reglu computes correctly for 1D input", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Create tensor with 4 elements (splits into 2 + 2)
  # First half: x values (activation applied here), Second half: gate values
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  # x = [-2, 3], gate = [1, 2]
  ggml_set_f32(a, c(-2, 3, 1, 2))

  r <- ggml_reglu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)

  # ReGLU: ReLU(x) * gate
  # [ReLU(-2) * 1, ReLU(3) * 2] = [0*1, 3*2] = [0, 6]
  expect_equal(result, c(0, 6), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("geglu computes correctly for 1D input", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Create tensor with 4 elements
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  # x = [1, 1], gate = [0, 2]
  ggml_set_f32(a, c(1, 1, 0, 2))

  r <- ggml_geglu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)

  # GELU(0) = 0, so first result should be 0
  expect_equal(result[1], 0, tolerance = 1e-4)
  # GELU(2) > 0, so second result should be positive
  expect_gt(result[2], 0)

  ggml_free(ctx)
})

test_that("swiglu computes correctly for 1D input", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Create tensor with 4 elements
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  # x = [1, 1], gate = [0, 2]
  ggml_set_f32(a, c(1, 1, 0, 2))

  r <- ggml_swiglu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)

  # SiLU(0) = 0 * sigmoid(0) = 0, so first result should be 0
  expect_equal(result[1], 0, tolerance = 1e-5)
  # SiLU(2) > 0, so second result should be positive
  expect_gt(result[2], 0)

  ggml_free(ctx)
})

test_that("glu generic function works with different ops", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Test with SWIGLU op
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 1, 0, 2))

  r <- ggml_glu(ctx, a, GGML_GLU_OP_SWIGLU, FALSE)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  expect_equal(length(result), 2)
  expect_equal(result[1], 0, tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("reglu_split computes correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Create separate tensors for input (x) and gate (b)
  # For split variant: a is the value tensor, b is the gate tensor
  # Formula: ReLU(a) * b
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(-1, 2, 3))    # Input values (activation applied here)
  ggml_set_f32(b, c(1, 2, 2))     # Gate values (multiplier)

  r <- ggml_reglu_split(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)

  # ReGLU split: ReLU(a) * b
  # [ReLU(-1) * 1, ReLU(2) * 2, ReLU(3) * 2] = [0, 4, 6]
  expect_equal(result, c(0, 4, 6), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("geglu_split computes correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(a, c(1, 1))
  ggml_set_f32(b, c(0, 2))

  r <- ggml_geglu_split(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)

  # a * GELU(b)
  expect_equal(result[1], 0, tolerance = 1e-4)  # 1 * GELU(0) = 0
  expect_gt(result[2], 0)                        # 1 * GELU(2) > 0

  ggml_free(ctx)
})

test_that("swiglu_split computes correctly", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(a, c(1, 1))
  ggml_set_f32(b, c(0, 2))

  r <- ggml_swiglu_split(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)

  # a * SiLU(b)
  expect_equal(result[1], 0, tolerance = 1e-5)  # 1 * SiLU(0) = 0
  expect_gt(result[2], 0)                        # 1 * SiLU(2) > 0

  ggml_free(ctx)
})

test_that("glu_split generic function works", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Formula: ReLU(a) * b
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(-1, 2, 3))    # Activation applied here
  ggml_set_f32(b, c(1, 2, 2))     # Gate/multiplier

  r <- ggml_glu_split(ctx, a, b, GGML_GLU_OP_REGLU)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)
  # ReLU([-1, 2, 3]) * [1, 2, 2] = [0, 4, 6]
  expect_equal(result, c(0, 4, 6), tolerance = 1e-5)

  ggml_free(ctx)
})

test_that("GLU works with 2D tensors", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Create 2D tensor: 4 columns (will split to 2), 3 rows
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  # Each row: [x1, x2, gate1, gate2]
  ggml_set_f32(a, c(
    1, 2, 1, 1,   # Row 0: x=[1,2], gate=[1,1]
    3, 4, 2, 0,   # Row 1: x=[3,4], gate=[2,0]
    5, 6, -1, 1   # Row 2: x=[5,6], gate=[-1,1]
  ))

  r <- ggml_reglu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, r)
  ggml_graph_compute(ctx, graph)

  result <- ggml_get_f32(r)

  # Output should be 2x3 = 6 elements
  expect_equal(length(result), 6)

  ggml_free(ctx)
})

test_that("geglu_quick produces similar results to geglu", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 1, 2))

  g <- ggml_geglu(ctx, a)
  graph <- ggml_build_forward_expand(ctx, g)
  ggml_graph_compute(ctx, graph)
  geglu_result <- ggml_get_f32(g)

  ggml_reset(ctx)

  a2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a2, c(1, 2, 1, 2))

  gq <- ggml_geglu_quick(ctx, a2)
  graph2 <- ggml_build_forward_expand(ctx, gq)
  ggml_graph_compute(ctx, graph2)
  quick_result <- ggml_get_f32(gq)

  # Results should be similar (within 10% tolerance)
  expect_equal(quick_result, geglu_result, tolerance = 0.1)

  ggml_free(ctx)
})

test_that("GLU output shape is correct", {
  ctx <- ggml_init(16 * 1024 * 1024)

  # Input: 8 elements -> output: 4 elements
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)
  ggml_set_f32(a, 1:8)

  r <- ggml_swiglu(ctx, a)
  shape <- ggml_tensor_shape(r)

  expect_equal(shape[1], 4)  # First dim halved

  ggml_free(ctx)
})

test_that("GLU split output shape matches input", {
  ctx <- ggml_init(16 * 1024 * 1024)

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 3)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 3)
  ggml_set_f32(a, rnorm(15))
  ggml_set_f32(b, rnorm(15))

  r <- ggml_swiglu_split(ctx, a, b)
  shape <- ggml_tensor_shape(r)

  expect_equal(shape[1], 5)  # Same as input
  expect_equal(shape[2], 3)  # Same as input

  ggml_free(ctx)
})
