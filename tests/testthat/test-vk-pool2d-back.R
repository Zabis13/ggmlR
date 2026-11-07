# Vulkan kernel pool2d_back.comp for POOL_2D_BACK (TODO 9.5-gpu-gaps).
# The shader gathers what the CPU kernel scatters, in the same window order,
# so Vulkan must agree with the CPU bit for bit. Both are checked against a
# reference written in R, and the Vulkan device must actually offer the op:
# without the supports_op check a missing kernel would fall back to the CPU
# and pass.

skip_if_no_vk <- function() {
  skip_on_cran()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
}

# Returns list(out, supported).
run_pool_back <- function(g, f, gshape, fshape, prm, backend = c("cpu", "vulkan")) {
  backend <- match.arg(backend)
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  b <- if (backend == "vulkan") ggml_vulkan_init(0L) else {
    cb <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cb, 2L); cb
  }
  on.exit(ggml_backend_free(b), add = TRUE, after = FALSE)

  tg <- ggml_new_tensor(ctx, GGML_TYPE_F32, length(gshape), gshape)
  tf <- ggml_new_tensor(ctx, GGML_TYPE_F32, length(fshape), fshape)
  ggml_set_input(tg); ggml_set_input(tf)
  out <- ggml_pool_2d_back(ctx, tg, tf, prm$op, prm$k0, prm$k1, prm$s0, prm$s1, prm$p0, prm$p1)
  ggml_set_output(out)
  supported <- ggml_backend_dev_supports_op(ggml_backend_get_device(b), out)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(tg, g)
  ggml_backend_tensor_set_data(tf, f)
  ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
  list(out = ggml_backend_tensor_get_data(out), supported = supported)
}

# Same loops as ggml_compute_forward_pool_2d_back (0-based offsets).
pool_back_ref <- function(g, f, IW, IH, OW, OH, P, prm) {
  G <- array(g, c(OW, OH, P)); Fa <- array(f, c(IW, IH, P))
  D <- array(0, c(IW, IH, P))
  for (pl in seq_len(P)) for (oy in 0:(OH - 1)) for (ox in 0:(OW - 1)) {
    ix <- ox * prm$s0 - prm$p0; iy <- oy * prm$s1 - prm$p1
    gv <- G[ox + 1, oy + 1, pl]
    if (prm$op == GGML_OP_POOL_MAX) {
      best <- -.Machine$double.xmax; bx <- -1; by <- -1
      for (ky in 0:(prm$k1 - 1)) {
        yy <- iy + ky; if (yy < 0 || yy >= IH) next
        for (kx in 0:(prm$k0 - 1)) {
          xx <- ix + kx; if (xx < 0 || xx >= IW) next
          v <- Fa[xx + 1, yy + 1, pl]
          if (v > best) { best <- v; bx <- xx; by <- yy }
        }
      }
      if (bx >= 0) D[bx + 1, by + 1, pl] <- D[bx + 1, by + 1, pl] + gv
    } else {
      for (ky in 0:(prm$k1 - 1)) {
        yy <- iy + ky; if (yy < 0 || yy >= IH) next
        for (kx in 0:(prm$k0 - 1)) {
          xx <- ix + kx; if (xx < 0 || xx >= IW) next
          D[xx + 1, yy + 1, pl] <- D[xx + 1, yy + 1, pl] + gv / (prm$k0 * prm$k1)
        }
      }
    }
  }
  as.vector(D)
}

# fshape: c(IW, IH, ...planes); output size as ggml_pool_2d computes it.
pool_case <- function(fshape, op, k, s, p, ties = FALSE) {
  prm <- list(op = op, k0 = k[1], k1 = k[2], s0 = s[1], s1 = s[2], p0 = p[1], p1 = p[2])
  IW <- fshape[1]; IH <- fshape[2]; P <- prod(fshape[-(1:2)])
  OW <- (IW + 2 * prm$p0 - prm$k0) %/% prm$s0 + 1
  OH <- (IH + 2 * prm$p1 - prm$k1) %/% prm$s1 + 1
  nf <- prod(fshape)
  f <- if (ties) as.numeric(sample(0:3, nf, replace = TRUE)) else runif(nf, -1, 1)
  list(prm = prm, IW = IW, IH = IH, OW = OW, OH = OH, P = P, fshape = fshape,
       gshape = c(OW, OH, fshape[-(1:2)]), f = f, g = runif(OW * OH * P, -1, 1))
}

both_pool_back <- function(cc) {
  cpu <- run_pool_back(cc$g, cc$f, cc$gshape, cc$fshape, cc$prm, "cpu")
  vk  <- run_pool_back(cc$g, cc$f, cc$gshape, cc$fshape, cc$prm, "vulkan")
  expect_true(vk$supported)
  expect_identical(vk$out, cpu$out)
  cpu$out
}

test_that("Vulkan POOL_2D_BACK matches the CPU bit for bit and the definition", {
  skip_if_no_vk()
  set.seed(5)
  cases <- list(
    list(fshape = c(8, 6),        k = c(2, 2), s = c(2, 2), p = c(0, 0)),   # plain, no overlap
    list(fshape = c(9, 7, 3),     k = c(3, 3), s = c(2, 2), p = c(1, 1)),   # overlap + padding
    list(fshape = c(13, 11, 2, 2), k = c(3, 2), s = c(1, 1), p = c(1, 0)),  # dense overlap, ne3
    list(fshape = c(10, 9, 2),    k = c(2, 2), s = c(3, 3), p = c(0, 0)))   # stride > kernel: gaps
  for (cs in cases) for (op in c(GGML_OP_POOL_MAX, GGML_OP_POOL_AVG)) {
    cc <- pool_case(cs$fshape, op, cs$k, cs$s, cs$p)
    got <- both_pool_back(cc)
    expect_equal(got, pool_back_ref(cc$g, cc$f, cc$IW, cc$IH, cc$OW, cc$OH, cc$P, cc$prm),
                 tolerance = 1e-6)   # f32 vs double
  }
})

test_that("Vulkan POOL_2D_BACK max: ties go to the first position, as on the CPU", {
  skip_if_no_vk()
  set.seed(6)
  cc <- pool_case(c(12, 10, 2), GGML_OP_POOL_MAX, c(3, 3), c(1, 1), c(1, 1), ties = TRUE)
  got <- both_pool_back(cc)
  expect_equal(got, pool_back_ref(cc$g, cc$f, cc$IW, cc$IH, cc$OW, cc$OH, cc$P, cc$prm),
               tolerance = 1e-6)
})

test_that("Vulkan POOL_2D_BACK above 512*65535 elements (NVIDIA x-limit)", {
  skip_if_no_vk()
  set.seed(7)
  cc <- pool_case(c(1024, 1024, 32), GGML_OP_POOL_MAX, c(2, 2), c(2, 2), c(0, 0))  # 33 554 432
  both_pool_back(cc)
})
