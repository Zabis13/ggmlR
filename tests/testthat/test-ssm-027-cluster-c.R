# ggml 0.27 "cluster C" (TODO 9.8): ssm_scan K state snapshots and the Vulkan
# SSM_CONV(+ADD)+SILU fusion.

leaf <- function(data, ...) list(shape = c(...), data = data)

# Run `build` once. Inputs live in their own buffer; the graph either goes
# through the scheduler (sched = TRUE) or is allocated whole by
# ggml_backend_alloc_ctx_tensors and computed directly.
run_c <- function(gpu, inputs, build, sched = TRUE) {
  be <- if (gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be))
  if (!gpu) ggml_backend_cpu_set_n_threads(be, 2L)
  wctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(wctx), add = TRUE, after = FALSE)
  ts <- lapply(inputs, function(x) {
    ne <- c(x$shape, rep(1, 4 - length(x$shape)))
    type <- if (is.integer(x$data)) GGML_TYPE_I32 else GGML_TYPE_F32
    ggml_new_tensor_4d(wctx, type, ne[1], ne[2], ne[3], ne[4])
  })
  if (!sched) {
    out <- do.call(build, c(list(wctx), ts))
    gf <- ggml_build_forward_expand(wctx, out)
  }
  buf <- ggml_backend_alloc_ctx_tensors(wctx, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  for (i in seq_along(inputs)) ggml_backend_tensor_set_data(ts[[i]], inputs[[i]]$data)
  if (!sched) {
    ggml_backend_graph_compute(be, gf)
    return(ggml_backend_tensor_get_data(out))
  }
  g <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(g), add = TRUE, after = FALSE)
  out <- do.call(build, c(list(g), ts))
  gf <- ggml_build_forward_expand(g, out)
  s <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit(ggml_backend_sched_free(s), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(s, gf)
  ggml_backend_sched_graph_compute(s, gf)
  ggml_backend_tensor_get_data(out)
}

# ---- ssm_scan K snapshots ------------------------------------------------------

# Mamba-2 sizes the Vulkan shader takes (d_state 128, head_dim % 16 == 0)
scan_case <- function(n_tok, d_state = 128L, head_dim = 16L, n_head = 2L, seed = 3L) {
  set.seed(seed)
  list(d_state = d_state, head_dim = head_dim, n_head = n_head, n_tok = n_tok,
       s0 = runif(d_state * head_dim * n_head, -0.5, 0.5),
       x  = runif(head_dim * n_head * n_tok, -1, 1),
       dt = runif(n_head * n_tok, -0.5, 0.5),
       A  = -runif(n_head, 0.2, 1.0),
       B  = runif(d_state * n_tok, -0.5, 0.5),
       C  = runif(d_state * n_tok, -0.5, 0.5))
}

# the first n tokens of a case
scan_prefix <- function(cs, n) {
  cs$x  <- cs$x[seq_len(cs$head_dim * cs$n_head * n)]
  cs$dt <- cs$dt[seq_len(cs$n_head * n)]
  cs$B  <- cs$B[seq_len(cs$d_state * n)]
  cs$C  <- cs$C[seq_len(cs$d_state * n)]
  cs$n_tok <- n
  cs
}

run_scan <- function(gpu, cs, K) {
  inputs <- list(leaf(cs$s0, cs$d_state, cs$head_dim, cs$n_head, 1),
                 leaf(cs$x, cs$head_dim, cs$n_head, cs$n_tok, 1),
                 leaf(cs$dt, cs$n_head, cs$n_tok, 1),
                 leaf(cs$A, 1, cs$n_head),
                 leaf(cs$B, cs$d_state, 1, cs$n_tok, 1),
                 leaf(cs$C, cs$d_state, 1, cs$n_tok, 1),
                 leaf(0L, 1))
  run_c(gpu, inputs, function(ctx, s, x, dt, A, B, C, ids) {
    ggml_ssm_scan(ctx, ggml_reshape_4d(ctx, s, cs$d_state, cs$head_dim, cs$n_head, 1),
                  x, ggml_reshape_3d(ctx, dt, cs$n_head, cs$n_tok, 1),
                  ggml_reshape_2d(ctx, A, 1, cs$n_head), B, C,
                  ggml_reshape_1d(ctx, ids, 1), K = K)
  })
}

test_that("ssm_scan K snapshots hold the states 0, 1, 2 tokens before the end (CPU)", {
  cs <- scan_case(n_tok = 5L)
  n_y <- cs$head_dim * cs$n_head * cs$n_tok
  n_s <- length(cs$s0)
  K <- 3L
  r <- run_scan(FALSE, cs, K)
  expect_length(r, n_y + K * n_s)
  r1 <- run_scan(FALSE, cs, 1L)
  expect_equal(r[seq_len(n_y)], r1[seq_len(n_y)])                 # outputs unchanged
  for (k in 0:(K - 1)) {
    ref <- run_scan(FALSE, scan_prefix(cs, cs$n_tok - k), 1L)
    ref_state <- ref[length(ref) - n_s + seq_len(n_s)]
    expect_equal(r[n_y + k * n_s + seq_len(n_s)], ref_state, tolerance = 1e-5,
                 label = sprintf("slot %d", k))
  }
})

test_that("Vulkan ssm_scan K snapshots match the CPU", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan")
  cs <- scan_case(n_tok = 6L, seed = 4L)
  n_y <- cs$head_dim * cs$n_head * cs$n_tok
  n_s <- length(cs$s0)
  K <- 4L
  cpu <- run_scan(FALSE, cs, K)
  gpu <- run_scan(TRUE, cs, K)
  expect_equal(gpu[seq_len(n_y)], cpu[seq_len(n_y)], tolerance = 1e-4)
  # slots 0 .. K-1 are all written (n_tok >= K)
  expect_equal(gpu[n_y + seq_len(K * n_s)], cpu[n_y + seq_len(K * n_s)], tolerance = 1e-4)
})

test_that("ssm_scan rejects a bad K as an R error", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  s  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 4, 2, 1)
  x  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 2, 3, 1)
  dt <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 3, 1)
  A1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2)
  A16 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 2)
  B  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 1, 3, 1)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1)
  expect_error(ggml_ssm_scan(ctx, s, x, dt, A1, B, B, ids, K = 0), "positive")
  expect_error(ggml_ssm_scan(ctx, s, x, dt, A16, B, B, ids, K = 2), "Mamba-2")
  expect_equal(ggml_nelements(ggml_ssm_scan(ctx, s, x, dt, A1, B, B, ids, K = 3)),
               4 * 2 * 3 + 3 * 16 * 4 * 2)
})

# ---- Vulkan SSM_CONV(+ADD)+SILU fusion --------------------------------------------

conv_fusions <- function() .Call("R_ggml_vk_ssm_conv_fusion_count", PACKAGE = "ggmlR")

conv_case <- function(d_conv = 4L, d_inner = 48L, n_t = 7L, n_s = 2L, seed = 9L) {
  set.seed(seed)
  list(d_conv = d_conv, d_inner = d_inner, n_t = n_t, n_s = n_s,
       sx = rnorm((d_conv - 1 + n_t) * d_inner * n_s),
       c  = rnorm(d_conv * d_inner, sd = 0.5),
       b  = rnorm(d_inner))
}

# with_bias: ssm_conv -> add(bias) -> silu; otherwise ssm_conv -> silu.
# intermediate: the conv input is computed in the graph (sx + sx), so the
# scheduler could hand its memory to the fused output without the alloc deps.
conv_build <- function(cc, with_bias, intermediate) {
  function(ctx, sx, c, b) {
    sx3 <- ggml_reshape_3d(ctx, sx, cc$d_conv - 1 + cc$n_t, cc$d_inner, cc$n_s)
    if (intermediate) sx3 <- ggml_scale(ctx, ggml_add(ctx, sx3, sx3), 0.5)
    y <- ggml_ssm_conv(ctx, sx3, ggml_reshape_2d(ctx, c, cc$d_conv, cc$d_inner))
    # b is used as is: a reshape would be a graph node between the conv and the
    # add, and without graph_optimize (sched = FALSE) the chain would not be
    # consecutive any more
    if (with_bias) y <- ggml_add(ctx, y, b)
    ggml_silu(ctx, y)
  }
}

conv_inputs <- function(cc) {
  list(leaf(cc$sx, (cc$d_conv - 1 + cc$n_t) * cc$d_inner * cc$n_s),
       leaf(cc$c, cc$d_conv * cc$d_inner), leaf(cc$b, cc$d_inner))
}

test_that("Vulkan SSM_CONV(+bias)+SILU fuses and matches the CPU", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan")
  cc <- conv_case()
  for (with_bias in c(TRUE, FALSE)) for (sched in c(TRUE, FALSE)) {
    lbl <- sprintf("bias=%s sched=%s", with_bias, sched)
    build <- conv_build(cc, with_bias, intermediate = sched)
    cpu <- run_c(FALSE, conv_inputs(cc), build, sched = sched)
    c0 <- conv_fusions()
    gpu <- run_c(TRUE, conv_inputs(cc), build, sched = sched)
    expect_equal(gpu, cpu, tolerance = 1e-4, label = lbl)
    expect_equal(conv_fusions() - c0, 1, label = paste("fusions,", lbl))
  }
})
