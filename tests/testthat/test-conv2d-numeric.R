# Numeric conv2d / im2col tests, ported from upstream ggml tests/test-conv2d.cpp
#
# Upstream sets a constant F16 kernel (2.5) and constant F32 input (1.5) with
# padding 1, and checks the exact conv2d output values. Existing ggmlR conv
# tests only assert that the op builds (externalptr); this verifies the actual
# computed result on the CPU backend, and on Vulkan when available.

library(ggmlR)

# Upstream geometry: KW=KH=3, IC=OC=10, IW=8, IH=6, N=1, stride/pad/dil = 1
KW <- 3L; KH <- 3L; IC <- 10L; OC <- 10L
IW <- 8L; IH <- 6L; N <- 1L

# With kernel=2.5, input=1.5, pad=1: each output is 2.5*1.5*IC * (#kernel taps
# overlapping the input). Corner=4 taps, edge=6, interior=9 ->
#   corner   = 2.5*1.5*10*4 = 150
#   edge     = 2.5*1.5*10*6 = 225
#   interior = 2.5*1.5*10*9 = 337.5
# One output channel is an 8x6 map laid out row-major in ne[0]=8 (width):
expected_row_edge     <- c(150, 225, 225, 225, 225, 225, 225, 150)   # top/bottom rows
expected_row_interior <- c(225, 337.5, 337.5, 337.5, 337.5, 337.5, 337.5, 225)

# full 8x6 map for one channel = edge, interior x4, edge
expected_map <- c(expected_row_edge,
                  rep(expected_row_interior, 4),
                  expected_row_edge)
# 10 identical output channels
expected_conv2d <- rep(expected_map, OC)

run_conv2d <- function(backend) {
  ctx <- ggml_init(64 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KW, KH, IC, OC)
  b <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, IW, IH, IC, N)
  im <- ggml_im2col(ctx, a, b, 1L, 1L, 1L, 1L, 1L, 1L, is_2D = TRUE,
                    dst_type = GGML_TYPE_F16)
  r  <- ggml_conv_2d(ctx, a, b, 1L, 1L, 1L, 1L, 1L, 1L)

  ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, rep(2.5, KW * KH * IC * OC))
  ggml_backend_tensor_set_data(b, rep(1.5, IW * IH * IC * N))

  gf <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(backend, gf)

  list(conv = ggml_backend_tensor_get_data(r),
       im_len = length(ggml_backend_tensor_get_data(im)))
}

test_that("conv2d produces upstream reference values (CPU)", {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  res <- run_conv2d(backend)

  expect_length(res$conv, OC * IW * IH)          # 480
  expect_equal(res$im_len, 4320)                 # upstream n_im2col_test
  expect_equal(res$conv, expected_conv2d, tolerance = 1e-2)
})

test_that("conv2d matches between CPU and Vulkan", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  cpu <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(cpu, 2L)
  cpu_res <- run_conv2d(cpu)
  ggml_backend_free(cpu)

  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(gpu), add = TRUE)
  gpu_res <- run_conv2d(gpu)

  expect_equal(gpu_res$conv, cpu_res$conv, tolerance = 1e-2)
})
