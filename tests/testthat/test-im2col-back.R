# Backward of convolution: GGML_OP_IM2COL_BACK, on the CPU and on Vulkan.
#
# The Vulkan shader is a ggmlR addition -- upstream has none, so training any
# convolution ran its backward on the CPU, which measured SLOWER end to end than
# training on the CPU outright (0.64x, 8 splits, 7 nodes off the device).
#
# The risk in this op is not the arithmetic but the index tests: with stride > 1
# the forward skips input columns, and those pixels must receive no gradient at
# all. An off-by-one there produces gradients that look plausible and train
# slightly wrong, so these compare against finite differences across a matrix of
# stride/padding/dilation settings rather than one default case.

skip_no_vulkan <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

# d(sum(conv2d(kernel, image)))/d(image), computed through the graph.
conv_grad <- function(img_data, ker_data, dims, s0, s1, p0, p1, d0, d1,
                      backend = c("cpu", "gpu")) {
  backend <- match.arg(backend)
  W <- dims$W; H <- dims$H; Cin <- dims$Cin; Cout <- dims$Cout
  KW <- dims$KW; KH <- dims$KH; N <- dims$N

  ctx <- ggml_init(256 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  ker <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, Cin, Cout)
  img <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, N)
  ggml_set_input(ker); ggml_set_input(img)
  ggml_set_param(img)

  be <- if (backend == "gpu") ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  if (backend == "cpu") ggml_backend_cpu_set_n_threads(be, 2L)
  sched <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit({
    ggml_backend_sched_free(sched)
    if (backend == "gpu") ggml_vulkan_free(be) else ggml_backend_free(be)
  }, add = TRUE)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)

  ggml_backend_tensor_set_data(ker, ker_data)
  ggml_backend_tensor_set_data(img, img_data)

  loss <- ggml_sum(ctx, ggml_conv_2d(ctx, ker, img, s0, s1, p0, p1, d0, d1))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  # The gradient tensors are allocated here; ggml_graph_reset() needs their
  # storage to exist, and asserts grad_acc->data otherwise.
  ggml_backend_sched_alloc_graph(sched, graph)
  ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  ggml_backend_tensor_get_data(ggml_graph_get_grad(graph, img))
}

# The same loss, evaluated forward-only, for finite differences.
conv_loss <- function(img_data, ker_data, dims, s0, s1, p0, p1, d0, d1) {
  W <- dims$W; H <- dims$H; Cin <- dims$Cin; Cout <- dims$Cout
  KW <- dims$KW; KH <- dims$KH; N <- dims$N

  ctx <- ggml_init(256 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  ker <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, Cin, Cout)
  img <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, N)
  ggml_set_f32(ker, ker_data)
  ggml_set_f32(img, img_data)

  out <- ggml_sum(ctx, ggml_conv_2d(ctx, ker, img, s0, s1, p0, p1, d0, d1))
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, out))
  ggml_get_f32(out)
}

# A small case, so finite differences stay cheap and exact enough.
dims <- list(W = 6L, H = 5L, Cin = 2L, Cout = 3L, KW = 3L, KH = 3L, N = 1L)

# stride, padding and dilation combinations. The stride > 1 rows are the ones
# that exercise the divisibility test; asymmetric values catch axis mix-ups.
configs <- list(
  list(name = "stride 1, no padding",     s0 = 1L, s1 = 1L, p0 = 0L, p1 = 0L, d0 = 1L, d1 = 1L),
  list(name = "stride 1, padding 1",      s0 = 1L, s1 = 1L, p0 = 1L, p1 = 1L, d0 = 1L, d1 = 1L),
  list(name = "stride 2",                 s0 = 2L, s1 = 2L, p0 = 1L, p1 = 1L, d0 = 1L, d1 = 1L),
  list(name = "asymmetric stride 2x1",    s0 = 2L, s1 = 1L, p0 = 1L, p1 = 0L, d0 = 1L, d1 = 1L),
  list(name = "asymmetric padding 2x0",   s0 = 1L, s1 = 1L, p0 = 2L, p1 = 0L, d0 = 1L, d1 = 1L),
  list(name = "dilation 2",               s0 = 1L, s1 = 1L, p0 = 2L, p1 = 2L, d0 = 2L, d1 = 2L)
)

test_that("conv gradients match finite differences across stride and padding", {
  set.seed(41L)
  n_img <- dims$W * dims$H * dims$Cin * dims$N
  n_ker <- dims$KW * dims$KH * dims$Cin * dims$Cout
  img_data <- runif(n_img, -1, 1)
  ker_data <- runif(n_ker, -1, 1)

  for (cf in configs) {
    analytic <- conv_grad(img_data, ker_data, dims,
                          cf$s0, cf$s1, cf$p0, cf$p1, cf$d0, cf$d1, backend = "cpu")

    eps <- 1e-3
    numeric <- vapply(seq_along(img_data), function(i) {
      hi <- img_data; hi[i] <- hi[i] + eps
      lo <- img_data; lo[i] <- lo[i] - eps
      (conv_loss(hi, ker_data, dims, cf$s0, cf$s1, cf$p0, cf$p1, cf$d0, cf$d1) -
       conv_loss(lo, ker_data, dims, cf$s0, cf$s1, cf$p0, cf$p1, cf$d0, cf$d1)) / (2 * eps)
    }, numeric(1))

    expect_equal(analytic, numeric, tolerance = 1e-2, info = cf$name)
  }
})

test_that("the Vulkan shader agrees with the CPU kernel across stride and padding", {
  # The shader's whole job is to produce what the CPU kernel already produces.
  # A mismatch confined to one configuration points straight at the index tests.
  skip_no_vulkan()
  set.seed(42L)
  n_img <- dims$W * dims$H * dims$Cin * dims$N
  n_ker <- dims$KW * dims$KH * dims$Cin * dims$Cout
  img_data <- runif(n_img, -1, 1)
  ker_data <- runif(n_ker, -1, 1)

  for (cf in configs) {
    g_cpu <- conv_grad(img_data, ker_data, dims,
                       cf$s0, cf$s1, cf$p0, cf$p1, cf$d0, cf$d1, backend = "cpu")
    g_gpu <- conv_grad(img_data, ker_data, dims,
                       cf$s0, cf$s1, cf$p0, cf$p1, cf$d0, cf$d1, backend = "gpu")
    expect_equal(g_gpu, g_cpu, tolerance = 1e-4, info = cf$name)
  }
})

test_that("a strided convolution leaves skipped input columns without gradient", {
  # With stride 2 and no padding, some input columns are never read by the
  # forward pass, so their gradient must be exactly zero -- the case the
  # divisibility test in the kernel exists for.
  set.seed(43L)
  d <- list(W = 7L, H = 1L, Cin = 1L, Cout = 1L, KW = 1L, KH = 1L, N = 1L)
  img_data <- runif(d$W, -1, 1)
  ker_data <- 1

  g <- conv_grad(img_data, ker_data, d, 2L, 1L, 0L, 0L, 1L, 1L, backend = "cpu")

  # A 1x1 kernel with stride 2 reads columns 0, 2, 4, 6 and skips 1, 3, 5.
  expect_true(all(g[c(2, 4, 6)] == 0))
  expect_true(all(g[c(1, 3, 5, 7)] != 0))
})
