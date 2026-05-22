# Numeric transposed-convolution tests, ported from upstream
# ggml tests/test-conv-transpose.c
#
# Input data is 0,1,2,... and the F16 kernel is 0,1,2,... as well; upstream
# checks exact output values for strides 1, 2 and 3. Existing ggmlR coverage
# only asserts that conv_transpose_1d builds (externalptr), and there was no
# binding at all for conv_transpose_2d_p0 before.

library(ggmlR)

cpu_backend <- function() {
  b <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(b, 2L)
  b
}

# ---- conv_transpose_1d ---------------------------------------------------
# t: [L=3, Cin=2] F32 = 0..5 ;  k: [K=2, Cout=3, Cin=2] F16 = 0..11
run_ct1d <- function(stride) {
  ctx <- ggml_init(8 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
  k <- ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 2, 3, 2)
  r <- ggml_conv_transpose_1d(ctx, k, t, as.integer(stride), 0L, 1L)

  be <- cpu_backend()
  on.exit(ggml_backend_free(be), add = TRUE)
  ggml_backend_alloc_ctx_tensors(ctx, be)
  ggml_backend_tensor_set_data(t, as.numeric(0:5))
  ggml_backend_tensor_set_data(k, as.numeric(0:11))

  gf <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(be, gf)
  ggml_backend_tensor_get_data(r)
}

test_that("conv_transpose_1d matches upstream values (stride 1)", {
  expect_equal(run_ct1d(1),
    c(18, 45, 59, 37,
      24, 61, 83, 51,
      30, 77, 107, 65), tolerance = 1e-3)
})

test_that("conv_transpose_1d matches upstream values (stride 2)", {
  expect_equal(run_ct1d(2),
    c(18, 21, 24, 29, 30, 37,
      24, 27, 34, 39, 44, 51,
      30, 33, 44, 49, 58, 65), tolerance = 1e-3)
})

test_that("conv_transpose_1d matches upstream values (stride 3)", {
  expect_equal(run_ct1d(3),
    c(18, 21, 0, 24, 29, 0, 30, 37,
      24, 27, 0, 34, 39, 0, 44, 51,
      30, 33, 0, 44, 49, 0, 58, 65), tolerance = 1e-3)
})

# ---- conv_transpose_2d_p0 ------------------------------------------------
# t: [w=3, h=2, cin=2, N=1] F32 = 0..11 ; k: [w=2, h=2, cin=2, cout=3] F16 = 0..23
run_ct2d <- function(stride) {
  ctx <- ggml_init(8 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 2, 1)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 2, 3, 2)
  r <- ggml_conv_transpose_2d_p0(ctx, k, t, as.integer(stride))

  be <- cpu_backend()
  on.exit(ggml_backend_free(be), add = TRUE)
  ggml_backend_alloc_ctx_tensors(ctx, be)
  ggml_backend_tensor_set_data(t, as.numeric(0:11))
  ggml_backend_tensor_set_data(k, as.numeric(0:23))

  gf <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(be, gf)
  ggml_backend_tensor_get_data(r)
}

test_that("conv_transpose_2d_p0 matches upstream values (stride 1)", {
  # 3 output channels, each a 4x3 map (ne[0]=4 fastest)
  expected <- c(
    # channel 0
    72, 162, 188, 106,  192, 430, 490, 274,  132, 292, 326, 180,
    # channel 1
    96, 218, 260, 146,  264, 590, 682, 378,  180, 396, 446, 244,
    # channel 2
    120, 274, 332, 186, 336, 750, 874, 482,  228, 500, 566, 308)
  expect_equal(run_ct2d(1), expected, tolerance = 1e-3)
})

test_that("conv_transpose_2d_p0 output shapes are correct (strides 2, 3)", {
  # upstream: stride 2 -> 6x4x3 = 72 ; stride 3 -> 8x5x3 = 120
  expect_length(run_ct2d(2), 72)
  expect_length(run_ct2d(3), 120)
})
