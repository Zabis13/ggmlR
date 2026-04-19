
run_conv2d <- function(use_gpu, input_data, kernel_data,
                       W, H, Cin, N, KW, KH, Cout,
                       s0 = 1L, s1 = 1L, p0 = 0L, p1 = 0L, d0 = 1L, d1 = 1L) {
  mem <- 64 * 1024 * 1024
  ctx <- ggml_init(mem)
  ggml_set_no_alloc(ctx, TRUE)

  # input  [W, H, Cin, N]
  inp <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, N)
  # kernel [KW, KH, Cin, Cout]  — ggml_conv_2d_direct layout
  ker <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, Cin, Cout)

  # Use conv_2d_direct — same GGML_OP_CONV_2D op used by ONNX path
  out <- ggml_conv_2d_direct(ctx, ker, inp, s0, s1, p0, p1, d0, d1)

  backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(inp, input_data)
  ggml_backend_tensor_set_data(ker, kernel_data)

  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

# ---- helper: random conv and compare CPU vs GPU ----
check_conv2d <- function(W, H, Cin, N, KW, KH, Cout,
                         s0 = 1L, s1 = 1L, p0 = 0L, p1 = 0L, d0 = 1L, d1 = 1L,
                         tol = 1e-3, label = "") {
  set.seed(42)
  inp_data <- rnorm(W * H * Cin * N)
  ker_data <- rnorm(KW * KH * Cin * Cout)

  cpu <- run_conv2d(FALSE, inp_data, ker_data, W, H, Cin, N, KW, KH, Cout, s0, s1, p0, p1, d0, d1)
  gpu <- run_conv2d(TRUE,  inp_data, ker_data, W, H, Cin, N, KW, KH, Cout, s0, s1, p0, p1, d0, d1)

  diff_max <- max(abs(cpu - gpu))
  expect_lt(diff_max, tol)
}

skip_no_gpu <- function() {
  skip_if(
    tryCatch({ ggml_vulkan_init(0L); FALSE }, error = function(e) TRUE),
    "No Vulkan GPU available"
  )
}

# ============================================================
# 1×1 conv — most common in attention blocks (matches node_155 suspect)
# ============================================================
test_that("conv2d 1x1 CPU==GPU, small", {
  skip_no_gpu()
  check_conv2d(W=8L, H=8L, Cin=16L, N=1L, KW=1L, KH=1L, Cout=16L,
               label="1x1 8x8 Cin=Cout=16")
})

test_that("conv2d 1x1 CPU==GPU, bat_resnext node_155 shape [64,64,16,1]->Cout=16", {
  skip_no_gpu()
  # node_155 output [64,64,16,1] — guessing 1x1 conv, Cin=16, Cout=16
  check_conv2d(W=64L, H=64L, Cin=16L, N=1L, KW=1L, KH=1L, Cout=16L,
               label="1x1 64x64 Cin=Cout=16")
})

# ============================================================
# 3×3 conv
# ============================================================
test_that("conv2d 3x3 CPU==GPU, stride=1 pad=1", {
  skip_no_gpu()
  check_conv2d(W=16L, H=16L, Cin=8L, N=1L, KW=3L, KH=3L, Cout=8L,
               s0=1L, s1=1L, p0=1L, p1=1L,
               label="3x3 pad=1")
})

test_that("conv2d 3x3 CPU==GPU, stride=2 pad=1", {
  skip_no_gpu()
  check_conv2d(W=32L, H=32L, Cin=8L, N=1L, KW=3L, KH=3L, Cout=16L,
               s0=2L, s1=2L, p0=1L, p1=1L,
               label="3x3 stride=2 pad=1")
})

# ============================================================
# 7×7 conv (common stem)
# ============================================================
test_that("conv2d 7x7 CPU==GPU, stride=2 pad=3", {
  skip_no_gpu()
  check_conv2d(W=64L, H=64L, Cin=3L, N=1L, KW=7L, KH=7L, Cout=16L,
               s0=2L, s1=2L, p0=3L, p1=3L,
               label="7x7 stride=2 pad=3")
})

# ============================================================
# dilation
# ============================================================
test_that("conv2d 3x3 CPU==GPU, dilation=2", {
  skip_no_gpu()
  check_conv2d(W=32L, H=32L, Cin=8L, N=1L, KW=3L, KH=3L, Cout=8L,
               s0=1L, s1=1L, p0=2L, p1=2L, d0=2L, d1=2L,
               label="3x3 dilation=2")
})

# ============================================================
# batch > 1
# ============================================================
test_that("conv2d 1x1 CPU==GPU, batch=4", {
  skip_no_gpu()
  check_conv2d(W=16L, H=16L, Cin=8L, N=4L, KW=1L, KH=1L, Cout=8L,
               label="1x1 batch=4")
})
