# Vulkan kernels for ops that had none anywhere (TODO 9.5-gpu-gaps):
#   DIAG_MASK_ZERO  -- diag_mask_inf.comp built with MASK_ZERO
#   WIN_PART/UNPART -- win_part.comp (SAM window partition)
# Each runs on the CPU and on Vulkan, is compared with a reference written in
# R, and must actually be offered by the Vulkan device: without the
# supports_op check a missing kernel would fall back to the CPU and pass.

skip_if_no_vk <- function() {
  skip_on_cran()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
}

# build(ctx, x) -> output tensor; x is created from `shape` and filled with
# `data`. Returns list(out, supported) where supported is what the backend's
# device says about the output node.
run_op <- function(build, shape, data, backend = c("cpu", "vulkan")) {
  backend <- match.arg(backend)
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  b <- if (backend == "vulkan") ggml_vulkan_init(0L) else {
    cb <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cb, 2L); cb
  }
  on.exit(ggml_backend_free(b), add = TRUE, after = FALSE)

  x <- ggml_new_tensor(ctx, GGML_TYPE_F32, length(shape), shape)
  ggml_set_input(x)
  out <- build(ctx, x)
  ggml_set_output(out)
  supported <- ggml_backend_dev_supports_op(ggml_backend_get_device(b), out)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(x, data)
  ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
  list(out = ggml_backend_tensor_get_data(out), supported = supported)
}

# Runs on both backends, checks Vulkan really ran it and agrees exactly
# (these ops only move or zero values), returns the CPU result.
both <- function(build, shape, data) {
  cpu <- run_op(build, shape, data, "cpu")
  vk  <- run_op(build, shape, data, "vulkan")
  expect_true(vk$supported)
  expect_identical(vk$out, cpu$out)
  cpu$out
}

# ---- DIAG_MASK_ZERO ---------------------------------------------------------

diag_mask_zero_ref <- function(x, shape, n_past) {
  a <- array(x, shape)
  nc <- shape[1]; nr <- shape[2]
  # 0-based column i, row j: masked when i > n_past + j
  keep <- outer(seq_len(nc) - 1L, seq_len(nr) - 1L, function(i, j) i <= n_past + j)
  as.vector(a * as.vector(keep))   # keep recycles over the trailing dims
}

test_that("Vulkan DIAG_MASK_ZERO matches the CPU and the definition", {
  skip_if_no_vk()
  set.seed(1)
  for (case in list(list(shape = c(7, 5),        n_past = 0L),
                    list(shape = c(7, 5, 3),     n_past = 2L),
                    list(shape = c(64, 64, 4),   n_past = 0L),
                    list(shape = c(33, 17, 3, 2), n_past = 5L))) {
    n <- prod(case$shape)
    x <- runif(n, 0.5, 1.5)   # no zeros in the input: a zero in the output means masked
    got <- both(function(ctx, t) ggml_diag_mask_zero(ctx, t, case$n_past), case$shape, x)
    expect_equal(got, diag_mask_zero_ref(x, case$shape, case$n_past), tolerance = 1e-6)  # f32 vs double
  }
})

# ---- WIN_PART / WIN_UNPART --------------------------------------------------

# x: [C, W, H] -> [C, w, w, npx*npy], zero padded; window index = py*npx + px
win_part_ref <- function(x, C, W, H, w) {
  npx <- ceiling(W / w); npy <- ceiling(H / w)
  a <- array(x, c(C, W, H))
  pad <- array(0, c(C, npx * w, npy * w))
  pad[, seq_len(W), seq_len(H)] <- a
  out <- array(0, c(C, w, w, npx * npy))
  for (py in seq_len(npy) - 1L) for (px in seq_len(npx) - 1L) {
    out[, , , py * npx + px + 1L] <- pad[, px * w + seq_len(w), py * w + seq_len(w)]
  }
  as.vector(out)
}

WIN_CASES <- list(
  list(C = 3L,  W = 10L, H = 7L,  w = 4L),    # padding on both axes
  list(C = 16L, W = 14L, H = 14L, w = 7L),    # exact tiling
  list(C = 1L,  W = 5L,  H = 9L,  w = 3L),    # C = 1, padding on W
  list(C = 64L, W = 64L, H = 64L, w = 14L))   # > 262144 dst elements: z > 1 grid

test_that("Vulkan WIN_PART matches the CPU and the definition", {
  skip_if_no_vk()
  set.seed(2)
  for (cs in WIN_CASES) {
    x <- runif(cs$C * cs$W * cs$H)
    got <- both(function(ctx, t) ggml_win_part(ctx, t, cs$w), c(cs$C, cs$W, cs$H), x)
    expect_equal(got, win_part_ref(x, cs$C, cs$W, cs$H, cs$w), tolerance = 1e-6)
  }
})

test_that("Vulkan WIN_UNPART matches the CPU and inverts WIN_PART", {
  skip_if_no_vk()
  set.seed(3)
  for (cs in WIN_CASES) {
    x <- runif(cs$C * cs$W * cs$H)
    np <- ceiling(cs$W / cs$w) * ceiling(cs$H / cs$w)
    parts <- win_part_ref(x, cs$C, cs$W, cs$H, cs$w)
    got <- both(function(ctx, t) ggml_win_unpart(ctx, t, cs$W, cs$H, cs$w),
                c(cs$C, cs$w, cs$w, np), parts)
    expect_equal(got, x, tolerance = 1e-6)
  }
})
