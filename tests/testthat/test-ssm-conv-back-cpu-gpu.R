# Backward of ggml_ssm_conv on the CPU and on Vulkan.
#
# ggmlR extension: ggml has no Vulkan shader for SSM_CONV_BACK, so before this
# every training step through a Mamba block's convolution branch fell back to
# the CPU. These tests check the shader against the CPU kernel AND both of them
# against the closed form written out in R -- two backends that are wrong in the
# same way would agree with each other and pass a CPU-vs-GPU comparison alone.

# Closed form of the backward, straight from the definition:
#   y[i1, t, s]         = sum_{i0} sx[t + i0, i1, s] * c[i0, i1]
#   d_sx[t + i0, i1, s] = sum_t g[i1, t, s] * c[i0, i1]
#   d_c[i0, i1]         = sum_{t, s} g[i1, t, s] * sx[t + i0, i1, s]
ref_conv_back <- function(sx, cw, g, nc, ncs, nr, n_t, n_s) {
  d_sx <- array(0, dim = c(ncs, nr, n_s))
  d_c  <- array(0, dim = c(nc, nr))
  for (s in seq_len(n_s)) {
    for (i1 in seq_len(nr)) {
      for (t in seq_len(n_t)) {
        gv <- g[i1, t, s]
        for (i0 in seq_len(nc)) {
          d_sx[t + i0 - 1, i1, s] <- d_sx[t + i0 - 1, i1, s] + gv * cw[i0, i1]
          d_c[i0, i1] <- d_c[i0, i1] + gv * sx[t + i0 - 1, i1, s]
        }
      }
    }
  }
  list(d_sx = as.vector(d_sx), d_c = as.vector(d_c))
}

# Run one SSM_CONV_BACK on the requested backend and return the packed result.
run_conv_back <- function(sx, cw, g, nc, ncs, nr, n_t, n_s, gpu) {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  t_sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncs, nr, n_s)
  t_c  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr)
  t_g  <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nr, n_t, n_s)

  out <- ggml_ssm_conv_back(ctx, t_sx, t_c, t_g)
  ggml_set_output(out)
  graph <- ggml_build_forward_expand(ctx, out)

  backend <- if (gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)

  ggml_backend_tensor_set_data(t_sx, as.vector(sx))
  ggml_backend_tensor_set_data(t_c,  as.vector(cw))
  ggml_backend_tensor_set_data(t_g,  as.vector(g))

  ggml_backend_sched_graph_compute(sched, graph)

  list(data  = ggml_backend_tensor_get_data(out),
       where = ggml_vulkan_backend_name(
                 ggml_backend_sched_get_tensor_backend(sched, out)))
}

make_case <- function(nc, nr, n_t, n_s, seed = 42) {
  set.seed(seed)
  ncs <- nc - 1L + n_t
  list(nc = nc, ncs = ncs, nr = nr, n_t = n_t, n_s = n_s,
       sx = array(runif(ncs * nr * n_s, -1, 1), dim = c(ncs, nr, n_s)),
       cw = array(runif(nc * nr, -1, 1),        dim = c(nc, nr)),
       g  = array(runif(nr * n_t * n_s, -1, 1), dim = c(nr, n_t, n_s)))
}

split_packed <- function(v, ncs, nr, n_s) {
  n_sx <- ncs * nr * n_s
  list(d_sx = v[seq_len(n_sx)], d_c = v[-seq_len(n_sx)])
}

test_that("ssm_conv_back on the CPU matches the closed form", {
  cs <- make_case(nc = 4L, nr = 8L, n_t = 6L, n_s = 1L)
  got <- run_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t, cs$n_s,
                       gpu = FALSE)
  ref <- ref_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t, cs$n_s)
  parts <- split_packed(got$data, cs$ncs, cs$nr, cs$n_s)

  expect_equal(parts$d_sx, ref$d_sx, tolerance = 1e-5)
  expect_equal(parts$d_c,  ref$d_c,  tolerance = 1e-5)
})

test_that("ssm_conv_back on Vulkan matches the closed form", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0, "No Vulkan devices found")

  cs <- make_case(nc = 4L, nr = 8L, n_t = 6L, n_s = 1L)
  got <- run_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t, cs$n_s,
                       gpu = TRUE)

  # The point of the whole exercise: if the scheduler put this on the CPU the
  # comparison below would pass while testing nothing.
  expect_equal(got$where, "Vulkan0")

  ref <- ref_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t, cs$n_s)
  parts <- split_packed(got$data, cs$ncs, cs$nr, cs$n_s)

  expect_equal(parts$d_sx, ref$d_sx, tolerance = 1e-5)
  expect_equal(parts$d_c,  ref$d_c,  tolerance = 1e-5)
})

test_that("ssm_conv_back agrees between CPU and Vulkan across shapes", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0, "No Vulkan devices found")

  shapes <- list(
    list(nc = 4L, nr = 1L,    n_t = 1L,  n_s = 1L),   # degenerate
    list(nc = 2L, nr = 3L,    n_t = 5L,  n_s = 1L),   # narrow kernel
    list(nc = 4L, nr = 1024L, n_t = 64L, n_s = 1L),   # Mamba-sized d_inner
    list(nc = 4L, nr = 300L,  n_t = 7L,  n_s = 3L),   # several sequences
    list(nc = 8L, nr = 16L,   n_t = 9L,  n_s = 2L)    # widest supported kernel
  )

  for (sh in shapes) {
    cs <- make_case(sh$nc, sh$nr, sh$n_t, sh$n_s, seed = sh$nr + sh$n_t)
    cpu <- run_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t,
                         cs$n_s, gpu = FALSE)
    gpu <- run_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t,
                         cs$n_s, gpu = TRUE)

    expect_equal(gpu$where, "Vulkan0",
                 info = sprintf("nc=%d nr=%d n_t=%d n_s=%d ran on %s",
                                sh$nc, sh$nr, sh$n_t, sh$n_s, gpu$where))
    expect_equal(gpu$data, cpu$data, tolerance = 1e-4,
                 info = sprintf("nc=%d nr=%d n_t=%d n_s=%d",
                                sh$nc, sh$nr, sh$n_t, sh$n_s))
  }
})

test_that("a kernel wider than the shader supports falls back to the CPU", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0, "No Vulkan devices found")

  # d_conv = 12 exceeds MAX_D_CONV in ssm_conv_back.comp, so supports_op must
  # refuse it -- and the result must still be correct, on the CPU.
  cs <- make_case(nc = 12L, nr = 4L, n_t = 5L, n_s = 1L)
  got <- run_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t, cs$n_s,
                       gpu = TRUE)
  expect_equal(got$where, "CPU")

  ref <- ref_conv_back(cs$sx, cs$cw, cs$g, cs$nc, cs$ncs, cs$nr, cs$n_t, cs$n_s)
  parts <- split_packed(got$data, cs$ncs, cs$nr, cs$n_s)
  expect_equal(parts$d_sx, ref$d_sx, tolerance = 1e-5)
  expect_equal(parts$d_c,  ref$d_c,  tolerance = 1e-5)
})

test_that("the gradient trains: conv weights converge on both backends", {
  # End to end rather than op-level: fit the convolution kernel of a known
  # target by gradient descent and require the loss to fall on each backend.
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0, "No Vulkan devices found")

  fit <- function(gpu) {
    nc <- 4L; nr <- 16L; n_t <- 12L; n_s <- 1L
    ncs <- nc - 1L + n_t
    set.seed(11)
    sx     <- array(runif(ncs * nr * n_s, -1, 1), dim = c(ncs, nr, n_s))
    target <- array(runif(nr * n_t * n_s, -1, 1), dim = c(nr, n_t, n_s))
    w <- runif(nc * nr, -0.2, 0.2)

    losses <- numeric(0)
    for (step in seq_len(30)) {
      ctx <- ggml_init(64 * 1024 * 1024)
      ggml_set_no_alloc(ctx, TRUE)
      t_sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncs, nr, n_s)
      t_c  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr)
      ggml_set_param(t_c)
      t_y  <- ggml_ssm_conv(ctx, t_sx, t_c)
      t_t  <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nr, n_t, n_s)
      loss <- ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, t_y, t_t)))
      ggml_set_output(loss); ggml_set_loss(loss)
      graph <- ggml_build_forward_expand_grads(ctx, loss, graph_size = 2048L)
      ggml_build_backward_expand(ctx, graph)

      backend <- if (gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
      sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
      ggml_backend_sched_reset(sched)
      ggml_backend_sched_alloc_graph(sched, graph)
      ggml_backend_tensor_set_data(t_sx, as.vector(sx))
      ggml_backend_tensor_set_data(t_c,  w)
      ggml_backend_tensor_set_data(t_t,  as.vector(target))
      ggml_graph_reset(graph)
      ggml_backend_sched_graph_compute(sched, graph)

      losses <- c(losses, ggml_backend_tensor_get_data(loss)[1])
      gr <- ggml_graph_get_grad(graph, t_c)
      w <- w - 0.002 * ggml_backend_tensor_get_data(gr)

      ggml_backend_sched_free(sched)
      ggml_backend_free(backend)
      ggml_free(ctx)
    }
    losses
  }

  l_cpu <- fit(FALSE)
  l_gpu <- fit(TRUE)

  expect_lt(l_cpu[length(l_cpu)], l_cpu[1])
  expect_lt(l_gpu[length(l_gpu)], l_gpu[1])
  # Same arithmetic, so the two curves should differ only by float noise.
  expect_equal(l_gpu, l_cpu, tolerance = 1e-3)
})
