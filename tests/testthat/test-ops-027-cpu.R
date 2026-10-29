# Ops added with the ggml 0.27 sync, on the CPU against references written in
# R: SwiGLU with clamped inputs, col2im_1d, and RoPE restricted to a window of
# dims (ggml_rope_set_offset). The Vulkan versions are checked against the
# same references once they land.

run_cpu <- function(inputs, build) {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  tensors <- lapply(inputs, function(x) {
    t <- switch(length(x$shape),
      ggml_new_tensor_1d(ctx, x$type, x$shape[1]),
      ggml_new_tensor_2d(ctx, x$type, x$shape[1], x$shape[2]),
      ggml_new_tensor_3d(ctx, x$type, x$shape[1], x$shape[2], x$shape[3]))
    t
  })
  out <- do.call(build, c(list(ctx), tensors))
  gf <- ggml_build_forward_expand(ctx, out)
  backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  for (i in seq_along(inputs)) ggml_backend_tensor_set_data(tensors[[i]], inputs[[i]]$data)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

f32 <- function(data, ...) list(type = GGML_TYPE_F32, shape = c(...), data = data)

# ---- SwiGLU clamp -----------------------------------------------------------

swiglu_clamp_ref <- function(a, b, limit) {
  g <- pmin(a, limit)
  g / (1 + exp(-g)) * pmin(pmax(b, -limit), limit)
}

test_that("swiglu_clamp matches the reference with separate gate and value", {
  set.seed(1)
  nc <- 16; nr <- 5; limit <- 1.5
  a <- rnorm(nc * nr, sd = 2); b <- rnorm(nc * nr, sd = 2)
  got <- run_cpu(list(f32(a, nc, nr), f32(b, nc, nr)),
                 function(ctx, ta, tb) ggml_swiglu_clamp(ctx, ta, tb, limit = limit))
  expect_equal(got, swiglu_clamp_ref(a, b, limit), tolerance = 1e-5)
})

test_that("swiglu_clamp splits one input into gate and value halves", {
  set.seed(2)
  nc <- 8; nr <- 4; limit <- 0.7
  x <- matrix(rnorm(2 * nc * nr, sd = 2), nrow = 2 * nc)
  got <- run_cpu(list(f32(as.vector(x), 2 * nc, nr)),
                 function(ctx, tx) ggml_swiglu_clamp(ctx, tx, NULL, limit = limit))
  ref <- swiglu_clamp_ref(x[seq_len(nc), ], x[nc + seq_len(nc), ], limit)
  expect_equal(got, as.vector(ref), tolerance = 1e-5)
})

# ---- col2im_1d --------------------------------------------------------------

# col: [K*OC, T_in] (column-major, element (oc*K + k, t_in)); out: [T_out, OC]
col2im_1d_ref <- function(col, K, OC, T_in, s0, p0) {
  T_out <- (T_in - 1) * s0 + K - 2 * p0
  out <- matrix(0, T_out, OC)
  for (t_in in 0:(T_in - 1)) for (oc in 0:(OC - 1)) for (k in 0:(K - 1)) {
    t <- t_in * s0 + k - p0
    if (t >= 0 && t < T_out) {
      out[t + 1, oc + 1] <- out[t + 1, oc + 1] + col[oc * K + k + 1, t_in + 1]
    }
  }
  out
}

for (cfg in list(c(K = 4, OC = 3, T_in = 6, s0 = 2, p0 = 1),   # overlapping windows, cropped
                 c(K = 3, OC = 1, T_in = 5, s0 = 3, p0 = 0),   # no overlap, mono
                 c(K = 5, OC = 2, T_in = 4, s0 = 1, p0 = 2))) { # stride 1, heavy overlap
  test_that(sprintf("col2im_1d matches the reference (K=%d OC=%d T_in=%d s0=%d p0=%d)",
                    cfg[["K"]], cfg[["OC"]], cfg[["T_in"]], cfg[["s0"]], cfg[["p0"]]), {
    K <- cfg[["K"]]; OC <- cfg[["OC"]]; T_in <- cfg[["T_in"]]; s0 <- cfg[["s0"]]; p0 <- cfg[["p0"]]
    set.seed(K * 100 + T_in)
    col <- matrix(rnorm(K * OC * T_in), nrow = K * OC)
    got <- run_cpu(list(f32(as.vector(col), K * OC, T_in)),
                   function(ctx, tc) ggml_col2im_1d(ctx, tc, s0 = s0, oc = OC, p0 = p0))
    expect_equal(got, as.vector(col2im_1d_ref(col, K, OC, T_in, s0, p0)), tolerance = 1e-5)
  })
}

# ---- RoPE with an offset window ---------------------------------------------

# x: [ne0, n_head, n_tok]; rotates dims [n_offs, n_offs + n_dims) of every row,
# theta for pair j = pos * base^(-2j / n_dims). Normal mode pairs neighbours
# (2j, 2j+1); NeoX pairs (j, j + n_dims/2).
rope_ref <- function(x, ne0, n_head, n_tok, pos, n_dims, n_offs, neox, base = 10000) {
  x <- array(x, c(ne0, n_head, n_tok))
  out <- x
  for (t in seq_len(n_tok)) for (h in seq_len(n_head)) for (j in 0:(n_dims / 2 - 1)) {
    theta <- pos[t] * base^(-2 * j / n_dims)
    i0 <- if (neox) j else 2 * j
    i1 <- if (neox) j + n_dims / 2 else 2 * j + 1
    i0 <- n_offs + i0 + 1; i1 <- n_offs + i1 + 1
    a <- x[i0, h, t]; b <- x[i1, h, t]
    out[i0, h, t] <- a * cos(theta) - b * sin(theta)
    out[i1, h, t] <- a * sin(theta) + b * cos(theta)
  }
  as.vector(out)
}

run_rope <- function(x, ne0, n_head, n_tok, pos, n_dims, n_offs, mode) {
  run_cpu(list(f32(x, ne0, n_head, n_tok),
               list(type = GGML_TYPE_I32, shape = n_tok, data = as.integer(pos))),
          function(ctx, tx, tp) {
            r <- ggml_rope_ext(ctx, tx, tp, NULL, n_dims = n_dims, mode = mode,
                               n_ctx_orig = 0L, freq_base = 10000, freq_scale = 1,
                               ext_factor = 0, attn_factor = 1, beta_fast = 32, beta_slow = 1)
            if (!is.null(n_offs)) r <- ggml_rope_set_offset(r, n_offs)
            r
          })
}

for (mode in c(0L, 2L)) {   # 0 = normal, 2 = NeoX
  test_that(sprintf("RoPE rotates only the offset window (mode %d)", mode), {
    set.seed(10 + mode)
    ne0 <- 12; n_head <- 2; n_tok <- 3; n_dims <- 4; n_offs <- 4
    pos <- c(0, 3, 7)
    x <- rnorm(ne0 * n_head * n_tok)
    got <- run_rope(x, ne0, n_head, n_tok, pos, n_dims, n_offs, mode)
    expect_equal(got, rope_ref(x, ne0, n_head, n_tok, pos, n_dims, n_offs, neox = mode == 2L),
                 tolerance = 1e-5)
  })

  test_that(sprintf("RoPE with offset 0 is plain RoPE (mode %d)", mode), {
    set.seed(20 + mode)
    ne0 <- 8; n_head <- 2; n_tok <- 2; n_dims <- 6
    pos <- c(1, 5)
    x <- rnorm(ne0 * n_head * n_tok)
    expect_equal(run_rope(x, ne0, n_head, n_tok, pos, n_dims, 0L, mode),
                 run_rope(x, ne0, n_head, n_tok, pos, n_dims, NULL, mode))
  })
}

test_that("ggml_rope_set_offset rejects offsets the kernel cannot honour", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  x <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 2)
  p <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2)
  r <- ggml_rope_ext(ctx, x, p, NULL, n_dims = 4, mode = 0L)
  expect_error(ggml_rope_set_offset(r, 3), "even")
  expect_error(ggml_rope_set_offset(r, 6), "exceeds")
  expect_error(ggml_rope_set_offset(x, 2), "not a RoPE node")
})
