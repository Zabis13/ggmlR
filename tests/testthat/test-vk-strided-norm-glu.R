# NORM, L2_NORM and GLU on non-contiguous inputs, Vulkan against the CPU.
#
# ggml 0.27 moved norm.comp and l2_norm.comp onto generic_unary_head (each row
# is addressed through src0_idx/dst_idx instead of row * ne00), and rewrote
# glu_main.glsl to take element strides for all three tensors plus the
# misalignment of each. Before that, NORM needed a fully contiguous tensor,
# split GLU read its second input with the first one's index, and a GLU input
# that did not start on a storage-buffer alignment boundary tripped an assert.
#
# Each case feeds the op a view -- permuted, row-padded, or starting one
# element into its buffer -- and compares against the CPU on the same view.
# The graph runs on the backend directly, so supports_op does not get a say:
# these exercise the shaders, not the scheduler's choice of backend.

run_view_op <- function(x_shape, x_data, build, use_gpu) {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  x <- if (length(x_shape) == 2) {
    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_shape[1], x_shape[2])
  } else {
    ggml_new_tensor_3d(ctx, GGML_TYPE_F32, x_shape[1], x_shape[2], x_shape[3])
  }
  out <- build(ctx, x)
  gf <- ggml_build_forward_expand(ctx, out)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(x, x_data)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

expect_gpu_matches_cpu <- function(x_shape, build, tol = 1e-5) {
  set.seed(prod(x_shape))
  x_data <- rnorm(prod(x_shape))
  cpu <- run_view_op(x_shape, x_data, build, use_gpu = FALSE)
  gpu <- run_view_op(x_shape, x_data, build, use_gpu = TRUE)
  expect_equal(gpu, cpu, tolerance = tol)
}

F32 <- 4  # bytes per float

# [16, 5, 3] with dims 1 and 2 swapped: rows stay contiguous, the row strides
# do not follow the shape any more.
permuted <- function(ctx, x) ggml_permute(ctx, x, 0, 2, 1, 3)

# 6 rows of 8 out of a [12, 6] buffer: each row is followed by 4 unused floats.
row_padded <- function(ctx, x) ggml_view_2d(ctx, x, 8, 6, 12 * F32, 0)

# The same padding over two planes of a [12, 6, 2] buffer; the planes stay
# packed against each other (nb2 = nb1 * ne1), which GLU requires.
row_padded_3d <- function(ctx, x) ggml_view_3d(ctx, x, 8, 6, 2, 12 * F32, 12 * 6 * F32, 0)

# 5 rows of 8, starting one float into the buffer, so the view's first element
# is not on a storage-buffer offset alignment boundary.
misaligned <- function(ctx, x) ggml_view_2d(ctx, x, 8, 5, 12 * F32, 1 * F32)

test_that("Vulkan NORM matches the CPU on non-contiguous rows", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  expect_gpu_matches_cpu(c(16, 5, 3), function(ctx, x) ggml_norm(ctx, permuted(ctx, x)))
  expect_gpu_matches_cpu(c(12, 6),    function(ctx, x) ggml_norm(ctx, row_padded(ctx, x)))
  expect_gpu_matches_cpu(c(12, 6),    function(ctx, x) ggml_norm(ctx, misaligned(ctx, x)))
})

test_that("Vulkan L2_NORM matches the CPU on non-contiguous rows", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  expect_gpu_matches_cpu(c(16, 5, 3), function(ctx, x) ggml_l2_norm(ctx, permuted(ctx, x)))
  expect_gpu_matches_cpu(c(12, 6),    function(ctx, x) ggml_l2_norm(ctx, row_padded(ctx, x)))
  expect_gpu_matches_cpu(c(12, 6),    function(ctx, x) ggml_l2_norm(ctx, misaligned(ctx, x)))
})

test_that("Vulkan SwiGLU matches the CPU on non-contiguous input", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  # GLU requires ggml_is_contiguous_1 (rows at any stride, dims above 1
  # packed), so the permuted view used for NORM is not a legal GLU input;
  # row padding is the non-contiguous layout it accepts.
  expect_gpu_matches_cpu(c(12, 6, 2), function(ctx, x) ggml_swiglu(ctx, row_padded_3d(ctx, x)))
  expect_gpu_matches_cpu(c(12, 6),    function(ctx, x) ggml_swiglu(ctx, row_padded(ctx, x)))
  expect_gpu_matches_cpu(c(12, 6),    function(ctx, x) ggml_swiglu(ctx, misaligned(ctx, x)))
})

test_that("Vulkan split SwiGLU reads each input with its own strides", {
  # a and b are the two halves of the same rows, so they share a shape but
  # not an offset; b is additionally misaligned. The 0.11 shader indexed b
  # with a's position, which happens to be right only when both are
  # contiguous and laid out identically.
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  expect_gpu_matches_cpu(c(16, 6), function(ctx, x) {
    a <- ggml_view_2d(ctx, x, 7, 6, 16 * F32, 0)
    b <- ggml_view_2d(ctx, x, 7, 6, 16 * F32, 9 * F32)
    ggml_swiglu_split(ctx, a, b)
  })
  # Same shape, different strides: a is packed, b is row-padded.
  expect_gpu_matches_cpu(c(12, 6, 2), function(ctx, x) {
    b <- row_padded_3d(ctx, x)
    a <- ggml_cont(ctx, b)
    ggml_swiglu_split(ctx, a, b)
  })
})
