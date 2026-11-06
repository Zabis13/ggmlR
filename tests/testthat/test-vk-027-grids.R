# Dispatch grids in ggml_vk_op_f32 / ggml_vk_load_shaders brought back to
# ggml 0.27 after the line-by-line check of the big Vulkan C++ functions
# (TODO 9.10):
#
# * LEAKY_RELU runs through unary.comp, which indexes z*262144 + y*512 + x.
#   It used to fall to the default {ne, 1, 1} grid, so past 512*65535
#   elements the x workgroup count went over NVIDIA's 65535 limit. On AMD
#   the x limit is far larger and the old grid also worked -- the large case
#   only proves the fix on NVIDIA.
# * IM2COL: grid {min(CHW, max(cap, OW*KH*KW)), OW, OH*batch} as in 0.27,
#   matching im2col.comp (y = ow, z = oh + OH*batch, X loop over CHW). The
#   cases below reach the capped X loop (CHW > cap) and several batches.
# * CONV_2D: CONV_SHAPE_64x32 now runs with WG_SIZE 128 (0.27). Small
#   convolutions (few tiles) pick that shape; F32 kernels keep it off our
#   cm1 path.
# * CROSS_ENTROPY_LOSS_BACK: taken from 0.27 whole (see the last test).

skip_no_vk <- function() skip_if_not(ggml_vulkan_available(), "no Vulkan device")

# inputs: list of list(type, shape, data)
run_graph <- function(use_gpu, inputs, build) {
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  tensors <- lapply(inputs, function(x) {
    ne <- c(x$shape, rep(1, 4 - length(x$shape)))
    ggml_new_tensor_4d(ctx, x$type, ne[1], ne[2], ne[3], ne[4])
  })
  out <- do.call(build, c(list(ctx), tensors))
  gf <- ggml_build_forward_expand(ctx, out)
  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  for (i in seq_along(inputs)) ggml_backend_tensor_set_data(tensors[[i]], inputs[[i]]$data)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

f32 <- function(data, ...) list(type = GGML_TYPE_F32, shape = c(...), data = data)

# ---- LEAKY_RELU grid ---------------------------------------------------------

test_that("Vulkan leaky_relu matches R on small and odd sizes", {
  skip_no_vk()
  for (n in c(1L, 511L, 513L, 262145L)) {
    x <- ((seq_len(n) %% 17L) - 8L) * 0.5
    gpu <- run_graph(TRUE, list(f32(x, n)),
                     function(ctx, a) ggml_leaky_relu(ctx, a, negative_slope = 0.25))
    expect_equal(gpu, ifelse(x > 0, x, 0.25 * x), tolerance = 0)
  }
})

test_that("Vulkan leaky_relu covers tensors past 512*65535 elements", {
  skip_no_vk()
  skip_on_cran()  # ~134 MB per tensor on the device
  n <- 512L * 65536L + 7L
  x <- ((seq_len(n) %% 17L) - 8L) * 0.5
  gpu <- run_graph(TRUE, list(f32(x, n)),
                   function(ctx, a) ggml_leaky_relu(ctx, a, negative_slope = 0.25))
  expect_length(gpu, n)
  # exact: halves times 0.25 are representable in f32
  expect_equal(gpu, ifelse(x > 0, x, 0.25 * x), tolerance = 0)
})

# ---- IM2COL grid -------------------------------------------------------------

test_that("Vulkan im2col (F32 out) matches the CPU, including the capped X loop", {
  skip_no_vk()
  set.seed(1)
  # KW, KH, IW, IH, IC, N, s, p, d
  cfgs <- list(
    c(3, 3,  8,  8, 1024, 2, 1, 1, 1),  # CHW = 9216 > cap on every vendor
    c(3, 3, 37,  5,   16, 3, 1, 1, 1),  # OW > OH, several batches
    c(5, 5, 12, 40,   64, 1, 2, 2, 1),  # OH > OW, stride 2
    c(3, 3, 20, 20,    8, 4, 1, 2, 2)   # dilation
  )
  for (cf in cfgs) {
    KW <- cf[1]; KH <- cf[2]; IW <- cf[3]; IH <- cf[4]; IC <- cf[5]; N <- cf[6]
    s <- cf[7]; p <- cf[8]; d <- cf[9]
    inputs <- list(f32(rnorm(KW * KH * IC), KW, KH, IC, 1),
                   f32(rnorm(IW * IH * IC * N), IW, IH, IC, N))
    build <- function(ctx, a, b) ggml_im2col(ctx, a, b, s, s, p, p, d, d,
                                             is_2D = TRUE, dst_type = GGML_TYPE_F32)
    cpu <- run_graph(FALSE, inputs, build)
    gpu <- run_graph(TRUE,  inputs, build)
    expect_equal(gpu, cpu, tolerance = 0, info = paste(cf, collapse = ","))
  }
})

test_that("Vulkan im2col 1D (F32 out) matches the CPU", {
  skip_no_vk()
  set.seed(2)
  K <- 7; IL <- 300; IC <- 600; N <- 2
  inputs <- list(f32(rnorm(K * IC), K, IC, 1),
                 f32(rnorm(IL * IC * N), IL, IC, N))
  build <- function(ctx, a, b) ggml_im2col(ctx, a, b, 1L, 0L, 3L, 0L, 1L, 0L,
                                           is_2D = FALSE, dst_type = GGML_TYPE_F32)
  cpu <- run_graph(FALSE, inputs, build)
  gpu <- run_graph(TRUE,  inputs, build)
  expect_equal(gpu, cpu, tolerance = 0)
})

# ---- CONV_2D, CONV_SHAPE_64x32 with WG_SIZE 128 --------------------------------

test_that("Vulkan conv_2d_direct on small shapes (64x32 tile) matches the CPU", {
  skip_no_vk()
  set.seed(3)
  # KW, KH, W, H, Cin, Cout, N
  cfgs <- list(
    c(3, 3,  6,  6,  4,  8, 1),   # one K tile, NPQ = 36
    c(3, 3,  9,  7,  5, 70, 2),   # Cout > 64: two K tiles
    c(1, 1,  5,  5, 33, 17, 1),   # CRS = 33, not a multiple of the CRS block
    c(5, 5, 11, 11,  3, 64, 1)    # Cout exactly 64
  )
  for (cf in cfgs) {
    KW <- cf[1]; KH <- cf[2]; W <- cf[3]; H <- cf[4]
    Cin <- cf[5]; Cout <- cf[6]; N <- cf[7]
    inputs <- list(f32(rnorm(KW * KH * Cin * Cout), KW, KH, Cin, Cout),
                   f32(rnorm(W * H * Cin * N), W, H, Cin, N))
    build <- function(ctx, k, x) ggml_conv_2d_direct(ctx, k, x, 1L, 1L, 1L, 1L, 1L, 1L)
    cpu <- run_graph(FALSE, inputs, build)
    gpu <- run_graph(TRUE,  inputs, build)
    # same bound as test-conv2d-cpu-gpu.R
    expect_lt(max(abs(gpu - cpu)), 1e-3, label = paste(cf, collapse = ","))
  }
})

# ---- CROSS_ENTROPY_LOSS_BACK = 0.27 --------------------------------------------
# Shader, pipelines and dispatch taken from 0.27: subgroup-size workgroups,
# a wg512 pipeline for rows longer than 1024, and rows spread over x/y/z by
# ggml_vk_nrows_elements (the old {nrows, 1, 1} grid went past NVIDIA's 65535
# workgroup limit). Cases straddle the 1024 switch and pass 65535 rows.

test_that("Vulkan cross_entropy_loss_back matches the CPU", {
  skip_no_vk()
  set.seed(4)
  # C (classes = row length), N (rows)
  for (cf in list(c(10, 1024), c(1000, 256), c(1024, 16), c(1025, 16),
                  c(32000, 64), c(16, 70000))) {
    C <- cf[1]; N <- cf[2]
    lab <- matrix(0, C, N)
    lab[cbind(sample.int(C, N, replace = TRUE), seq_len(N))] <- 1
    inputs <- list(list(type = GGML_TYPE_F32, shape = 1, data = 0.7),
                   f32(rnorm(C * N, sd = 3), C, N),
                   list(type = GGML_TYPE_F32, shape = c(C, N), data = as.numeric(lab)))
    cpu <- run_graph(FALSE, inputs, ggml_cross_entropy_loss_back)
    gpu <- run_graph(TRUE,  inputs, ggml_cross_entropy_loss_back)
    expect_equal(gpu, cpu, tolerance = 1e-5, info = paste(cf, collapse = "x"))
  }
})
