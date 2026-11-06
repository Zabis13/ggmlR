# ROPE / ROPE_BACK over ne[4] > 1, against a reference written in R.
#
# The CPU kernel used to walk only i1/i2/i3 while ggml_nrows() (and so the
# thread split) counted ne[4]: every dim-4 slice past the first was left
# unwritten. Vulkan refused ne[4] > 1 for ROPE and had no guard at all for
# ROPE_BACK. rope_funcs.glsl addresses a row as i3*nb03 with i3 running over
# ne03*ne04, which is exact when dims 3-4 lie back to back -- true for every
# tensor below, so on Vulkan these run on the GPU (supports_op is checked).
# A dim-4 layout that does not collapse (dims 3-4 swapped by ggml_view_5d)
# must be refused by Vulkan supports_op and land on the CPU kernel, which
# reads it through nb[3]/nb[4] -- checked through a GPU+CPU scheduler.
#
# n_dims < ne0 so the pass-through tail (channels past n_dims) is covered too.
# Both backends are checked against the R reference, not against each other.

D <- 8L; H <- 2L; S <- 3L; B <- 2L; E <- 3L
N_DIMS <- 4L
BASE   <- 10000

set.seed(11)
X_DAT <- runif(D * H * S * B * E, -1, 1)
POS   <- c(0L, 3L, 7L)  # one position per ne[2] (sequence) index

rope_ref <- function(mode, back) {
  x   <- array(X_DAT, c(D, H, S, B, E))
  out <- x
  sgn <- if (back) -1 else 1
  half <- N_DIMS %/% 2L
  for (i2 in seq_len(S)) {
    theta <- POS[i2] * BASE^(-2 * (0:(half - 1L)) / N_DIMS)
    cs <- cos(theta); sn <- sgn * sin(theta)
    for (k in seq_len(half)) {
      if (mode == GGML_ROPE_TYPE_NEOX) {
        a <- k; b <- k + half
      } else {
        a <- 2L * k - 1L; b <- 2L * k
      }
      x0 <- x[a, , i2, , , drop = FALSE]
      x1 <- x[b, , i2, , , drop = FALSE]
      out[a, , i2, , ] <- x0 * cs[k] - x1 * sn[k]
      out[b, , i2, , ] <- x0 * sn[k] + x1 * cs[k]
    }
  }
  as.numeric(out)
}

run_rope_5d <- function(backend, mode, back) {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  x <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, c(D, H, S, B, E))
  p <- ggml_new_tensor(ctx, GGML_TYPE_I32, 1L, S)
  f <- if (back) ggml_rope_ext_back else ggml_rope_ext
  out <- f(ctx, x, p, NULL, n_dims = N_DIMS, mode = mode, n_ctx_orig = 0L,
           freq_base = BASE, freq_scale = 1.0, ext_factor = 0.0,
           attn_factor = 1.0, beta_fast = 32.0, beta_slow = 1.0)
  gf  <- ggml_build_forward_expand(ctx, out)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(x, X_DAT)
  ggml_backend_tensor_set_data(p, POS)
  supported <- ggml_backend_dev_supports_op(ggml_backend_get_device(backend), out)
  ggml_backend_graph_compute(backend, gf)
  list(val = ggml_backend_tensor_get_data(out), supported = supported)
}

for (md in c("NORMAL", "NEOX")) for (bk in c(FALSE, TRUE)) {
  local({
    mode_name <- md
    back      <- bk
    mode <- if (mode_name == "NEOX") GGML_ROPE_TYPE_NEOX else 0L
    label <- paste0(if (back) "rope_back " else "rope ", mode_name)

    test_that(paste("CPU", label, "covers ne[4]"), {
      backend <- ggml_backend_cpu_init()
      on.exit(ggml_backend_free(backend))
      ggml_backend_cpu_set_n_threads(backend, 2L)
      r <- run_rope_5d(backend, mode, back)
      expect_equal(r$val, rope_ref(mode, back), tolerance = 1e-5)
    })

    test_that(paste("Vulkan", label, "covers ne[4] on the GPU"), {
      skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      backend <- ggml_vulkan_init(0)
      on.exit(ggml_backend_free(backend))
      r <- run_rope_5d(backend, mode, back)
      expect_true(r$supported)
      expect_equal(r$val, rope_ref(mode, back), tolerance = 1e-5)
    })
  })
}

# The same logical input [D, H, S, B, E], stored as [D, H, S, E, B] and read
# back through a view that swaps dims 3 and 4: nb[4] != ne[3]*nb[3].
run_rope_5d_swapped <- function(backends, mode, back) {
  cs <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(cs), add = TRUE, after = FALSE)
  xs <- ggml_new_tensor(cs, GGML_TYPE_F32, 5L, c(D, H, S, E, B))
  p  <- ggml_new_tensor(cs, GGML_TYPE_I32, 1L, S)
  buf <- ggml_backend_alloc_ctx_tensors(cs, backends[[1]])
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(
    xs, as.numeric(aperm(array(X_DAT, c(D, H, S, B, E)), c(1, 2, 3, 5, 4))))
  ggml_backend_tensor_set_data(p, POS)

  g <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(g), add = TRUE, after = FALSE)
  nb <- ggml_tensor_nb(xs)
  x  <- ggml_view_5d(g, xs, D, H, S, B, E, nb[2], nb[3], nb[5], nb[4])
  f  <- if (back) ggml_rope_ext_back else ggml_rope_ext
  out <- f(g, x, p, NULL, n_dims = N_DIMS, mode = mode, n_ctx_orig = 0L,
           freq_base = BASE, freq_scale = 1.0, ext_factor = 0.0,
           attn_factor = 1.0, beta_fast = 32.0, beta_slow = 1.0)
  ggml_set_output(out)
  gr <- ggml_build_forward_expand(g, out)

  s <- ggml_backend_sched_new(backends, parallel = FALSE)
  on.exit(ggml_backend_sched_free(s), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(s, gr)
  ggml_backend_sched_graph_compute(s, gr)
  list(val       = ggml_backend_tensor_get_data(out),
       supported = ggml_backend_dev_supports_op(
                     ggml_backend_get_device(backends[[1]]), out),
       ran_on    = ggml_backend_name(ggml_backend_sched_get_tensor_backend(s, out)))
}

test_that("view_5d swaps dims 3 and 4 without a copy", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))
  xs <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, c(D, H, S, E, B))
  nb <- ggml_tensor_nb(xs)
  v  <- ggml_view_5d(ctx, xs, D, H, S, B, E, nb[2], nb[3], nb[5], nb[4])
  expect_equal(ggml_nelements(v), D * H * S * B * E)
  expect_equal(ggml_tensor_nb(v), nb[c(1, 2, 3, 5, 4)])
  expect_error(ggml_view_5d(ctx, xs, D, H, S, B, E + 1L,
                            nb[2], nb[3], nb[5], nb[4]), "spans")
})

for (md in c("NORMAL", "NEOX")) for (bk in c(FALSE, TRUE)) {
  local({
    mode_name <- md
    back      <- bk
    mode <- if (mode_name == "NEOX") GGML_ROPE_TYPE_NEOX else 0L
    label <- paste0(if (back) "rope_back " else "rope ", mode_name)

    test_that(paste("CPU", label, "reads dims 3-4 through strides"), {
      cpu <- ggml_backend_cpu_init()
      on.exit(ggml_backend_free(cpu))
      ggml_backend_cpu_set_n_threads(cpu, 2L)
      r <- run_rope_5d_swapped(list(cpu), mode, back)
      expect_equal(r$val, rope_ref(mode, back), tolerance = 1e-5)
    })

    test_that(paste("Vulkan", label, "with swapped dims 3-4 falls back to CPU"), {
      skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      gpu <- ggml_vulkan_init(0)
      on.exit(ggml_backend_free(gpu))
      cpu <- ggml_backend_cpu_init()
      on.exit(ggml_backend_free(cpu), add = TRUE, after = FALSE)
      ggml_backend_cpu_set_n_threads(cpu, 2L)
      r <- run_rope_5d_swapped(list(gpu, cpu), mode, back)
      expect_false(r$supported)
      expect_equal(r$ran_on, ggml_backend_name(cpu))
      expect_equal(r$val, rope_ref(mode, back), tolerance = 1e-5)
    })
  })
}
