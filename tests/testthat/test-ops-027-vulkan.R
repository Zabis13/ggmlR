# Vulkan paths that arrived with the ggml 0.27 sync, against the CPU (and,
# where the CPU cannot run the case, against a reference written in R).
#
# New ops: COL2IM_1D, SWIGLU_CLAMP, RoPE with an offset window, POOL_1D,
# PAD_REFLECT_1D, CROSS_ENTROPY_LOSS (direct forward), GATED_LINEAR_ATTN,
# CONV_3D. New types / kernels: Q2_0, and the mul_mat_vec shaders for TQ1_0,
# TQ2_0 and IQ4_XS. Paths internal to existing ops, reached through the
# shapes chosen here: top-k radix select (k past the tournament pipelines),
# cpy_transpose_02 (cont of a 0<->2 permute), the snake fusion
# (x + sin(a*x)^2 * inv_b), the FWHT replacement of a Hadamard-hinted
# matmul, and the sparse K/V gather of flash attention (n_kv_max).
#
# Device-specific paths are reached only on the matching hardware: mul_mmq_cm1
# on RDNA3/4 (large-N quantized matmul below), the Intel Xe flash-attention
# decode phases (single-query decode below). Elsewhere those cases still check
# whichever kernel the device picks.
#
# Graphs run on the backend directly, so supports_op does not get a say: these
# exercise the shaders, not the scheduler's choice of backend.

# inputs: list of list(type, shape (1-4 dims), data); data is numeric, or the
# raw output of a quantize_*() for quantized types.
run_graph <- function(use_gpu, inputs, build) {
  ctx <- ggml_init(256 * 1024 * 1024, no_alloc = TRUE)
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

expect_gpu_matches_cpu <- function(inputs, build, tol = 1e-5) {
  cpu <- run_graph(FALSE, inputs, build)
  gpu <- run_graph(TRUE,  inputs, build)
  expect_equal(gpu, cpu, tolerance = tol)
  invisible(cpu)
}

f32 <- function(data, ...) list(type = GGML_TYPE_F32, shape = c(...), data = data)
f16 <- function(data, ...) list(type = GGML_TYPE_F16, shape = c(...), data = data)
i32 <- function(data, ...) list(type = GGML_TYPE_I32, shape = c(...), data = as.integer(data))

skip_no_vk <- function() skip_if_not(ggml_vulkan_available(), "no Vulkan device")

# ---- COL2IM_1D, SWIGLU_CLAMP, RoPE offset -------------------------------------
# (R references for these live in test-ops-027-cpu.R; the CPU is the yardstick.)

test_that("Vulkan col2im_1d matches the CPU", {
  skip_no_vk()
  for (cfg in list(c(K = 4, OC = 3, T_in = 6, s0 = 2, p0 = 1),
                   c(K = 3, OC = 1, T_in = 5, s0 = 3, p0 = 0),
                   c(K = 5, OC = 8, T_in = 300, s0 = 1, p0 = 2))) {
    K <- cfg[["K"]]; OC <- cfg[["OC"]]; T_in <- cfg[["T_in"]]
    set.seed(K * 100 + T_in)
    expect_gpu_matches_cpu(
      list(f32(rnorm(K * OC * T_in), K * OC, T_in)),
      function(ctx, tc) ggml_col2im_1d(ctx, tc, s0 = cfg[["s0"]], oc = OC, p0 = cfg[["p0"]]))
  }
})

test_that("Vulkan swiglu_clamp matches the CPU (split and fused input)", {
  skip_no_vk()
  set.seed(1)
  nc <- 64; nr <- 7
  expect_gpu_matches_cpu(
    list(f32(rnorm(nc * nr, sd = 2), nc, nr), f32(rnorm(nc * nr, sd = 2), nc, nr)),
    function(ctx, a, b) ggml_swiglu_clamp(ctx, a, b, limit = 1.5))
  expect_gpu_matches_cpu(
    list(f32(rnorm(2 * nc * nr, sd = 2), 2 * nc, nr)),
    function(ctx, x) ggml_swiglu_clamp(ctx, x, NULL, limit = 0.7))
})

test_that("Vulkan RoPE with an offset window matches the CPU", {
  skip_no_vk()
  for (mode in c(0L, 2L)) {
    set.seed(30 + mode)
    ne0 <- 64; n_head <- 4; n_tok <- 5
    expect_gpu_matches_cpu(
      list(f32(rnorm(ne0 * n_head * n_tok), ne0, n_head, n_tok), i32(c(0, 3, 7, 11, 40), n_tok)),
      function(ctx, x, p) {
        r <- ggml_rope_ext(ctx, x, p, NULL, n_dims = 32, mode = mode)
        ggml_rope_set_offset(r, 16)
      }, tol = 1e-4)
  }
})

# ---- POOL_1D, PAD_REFLECT_1D ------------------------------------------------

# x: [L, C]; out: [L %/% k0, C], non-overlapping windows of k0.
pool_1d_ref <- function(x, L, C, op, k0) {
  x <- matrix(x, L, C)
  OL <- L %/% k0
  out <- matrix(0, OL, C)
  for (c in seq_len(C)) for (o in 0:(OL - 1)) {
    v <- x[o * k0 + 1:k0, c]
    out[o + 1, c] <- if (op == GGML_OP_POOL_MAX) max(v) else mean(v)
  }
  as.vector(out)
}

test_that("Vulkan pool_1d matches the CPU (k0 == s0, no padding)", {
  skip_no_vk()
  set.seed(5)
  L <- 120; C <- 6
  x <- rnorm(L * C)
  for (op in c(GGML_OP_POOL_MAX, GGML_OP_POOL_AVG)) {
    cpu <- expect_gpu_matches_cpu(list(f32(x, L, C)),
                                  function(ctx, t) ggml_pool_1d(ctx, t, op, k0 = 4))
    expect_equal(cpu, pool_1d_ref(x, L, C, op, 4), tolerance = 1e-6)
  }
})

test_that("ggml_pool_1d rejects what the CPU kernel cannot run", {
  # The CPU kernel asserts k0 == s0 and p0 == 0; the graph does not know its
  # backend yet, so the wrapper refuses these for every backend.
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 2)
  expect_error(ggml_pool_1d(ctx, x, GGML_OP_POOL_MAX, k0 = 4, s0 = 2), "s0 == k0")
  expect_error(ggml_pool_1d(ctx, x, GGML_OP_POOL_AVG, k0 = 4, p0 = 1), "padding")
  expect_error(ggml_pool_1d(ctx, x, GGML_OP_POOL_AVG, k0 = 0), "positive")
})

test_that("ggml_pool_1d/2d reject an op that is neither max nor avg", {
  # "sum" used to coerce to NA and silently run as avg pooling
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 2)
  x4 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 2, 1)
  for (op in list("sum", NA, 2L, c(0L, 1L), NULL)) {
    expect_error(ggml_pool_1d(ctx, x, op, k0 = 2), "GGML_OP_POOL_MAX")
    expect_error(ggml_pool_2d(ctx, x4, op, k0 = 2, k1 = 2), "GGML_OP_POOL_MAX")
  }
  expect_no_error(ggml_pool_1d(ctx, x, GGML_OP_POOL_MAX, k0 = 2))
  expect_no_error(ggml_pool_2d(ctx, x4, 1, k0 = 2, k1 = 2))
})

test_that("Vulkan pad_reflect_1d matches the CPU", {
  skip_no_vk()
  set.seed(7)
  L <- 50; C <- 4; B <- 3
  expect_gpu_matches_cpu(list(f32(rnorm(L * C * B), L, C, B)),
                         function(ctx, t) ggml_pad_reflect_1d(ctx, t, 3, 7), tol = 0)
})

# ---- CROSS_ENTROPY_LOSS -----------------------------------------------------

test_that("Vulkan cross_entropy_loss (direct forward) matches the CPU", {
  skip_no_vk()
  for (dims in list(c(10, 32), c(1000, 7), c(3, 1))) {
    nc <- dims[1]; nr <- dims[2]
    set.seed(nc + nr)
    logits <- rnorm(nc * nr, sd = 3)
    lab <- matrix(runif(nc * nr), nc); lab <- sweep(lab, 2, colSums(lab), "/")
    expect_gpu_matches_cpu(list(f32(logits, nc, nr), f32(as.vector(lab), nc, nr)),
                           function(ctx, a, b) ggml_cross_entropy_loss(ctx, a, b), tol = 1e-5)
  }
})

# ---- GATED_LINEAR_ATTN ------------------------------------------------------

test_that("Vulkan gated_linear_attn matches the CPU (head size 64)", {
  skip_no_vk()
  S <- 64; H <- 2; n_tok <- 6; n_seqs <- 2
  set.seed(8)
  mk <- function(sd = 0.3) f32(rnorm(S * H * n_tok, sd = sd), S, H, n_tok)
  g <- f32(log(runif(S * H * n_tok, 0.8, 0.99)), S, H, n_tok)  # log-decay gates
  expect_gpu_matches_cpu(
    list(mk(), mk(), mk(), g, f32(rnorm(S * S * H * n_seqs, sd = 0.1), S * S * H, n_seqs)),
    function(ctx, k, v, q, g, st) ggml_gated_linear_attn(ctx, k, v, q, g, st, scale = 1 / sqrt(S)),
    tol = 1e-4)
})

# ---- CONV_3D ----------------------------------------------------------------

test_that("Vulkan conv_3d_direct matches the CPU", {
  skip_no_vk()
  # With coopmat (cm1/cm2) conv3d_mm.comp stages both operands in float16_t
  # shared memory whatever the kernel type, so an F32 kernel is also f16-exact
  # only (~1e-3 relative), as for conv2d_mm.
  IC <- 2; OC <- 3; N <- 2
  for (cfg in list(list(s = 1, p = 1, d = 1), list(s = c(2, 1, 2), p = 0, d = 1),
                   list(s = 1, p = 2, d = 2))) {
    for (ktype in c(GGML_TYPE_F32, GGML_TYPE_F16)) {
      set.seed(9)
      kern <- list(type = ktype, shape = c(3, 3, 3, IC * OC), data = rnorm(27 * IC * OC, sd = 0.3))
      expect_gpu_matches_cpu(
        list(kern, f32(rnorm(7 * 6 * 5 * IC * N), 7, 6, 5, IC * N)),
        function(ctx, a, b) ggml_conv_3d_direct(ctx, a, b, s = cfg$s, p = cfg$p, d = cfg$d,
                                                n_channels = IC, n_batch = N, n_channels_out = OC),
        tol = 2e-3)
    }
  }
})

# ---- quantized matmul: Q2_0, TQ1_0, TQ2_0, IQ4_XS ---------------------------

# ggml type ids (no exported R constants for these)
TYPE_IQ4_XS <- 23L
TYPE_TQ1_0  <- 34L
TYPE_TQ2_0  <- 35L
TYPE_Q2_0   <- 42L

# A [K, M] quantized; B [K, N] f32. N <= 8 takes mul_mat_vec, larger N the
# dequant + matmul (or MMQ) path.
expect_quant_mm_agrees <- function(qtype, qfn, M, K, N) {
  set.seed(M + K + N)
  a <- qfn(rnorm(K * M, sd = 0.1), n_rows = M, n_per_row = K)
  b <- rnorm(K * N, sd = 0.1)
  inputs <- list(list(type = qtype, shape = c(K, M), data = a), f32(b, K, N))
  build <- function(ctx, ta, tb) ggml_mul_mat(ctx, ta, tb)
  cpu <- run_graph(FALSE, inputs, build)
  gpu <- run_graph(TRUE,  inputs, build)
  info <- sprintf("type %d M=%d K=%d N=%d", qtype, M, K, N)
  expect_gt(cor(cpu, gpu), 0.999, label = info)
  expect_lt(max(abs(cpu - gpu)) / max(abs(cpu)), 0.02, label = info)
}

for (case in list(
  list(name = "Q2_0",   type = TYPE_Q2_0,   fn = quantize_q2_0),
  list(name = "TQ1_0",  type = TYPE_TQ1_0,  fn = quantize_tq1_0),
  list(name = "TQ2_0",  type = TYPE_TQ2_0,  fn = quantize_tq2_0),
  list(name = "IQ4_XS", type = TYPE_IQ4_XS, fn = quantize_iq4_xs)
)) {
  local({
    cc <- case
    test_that(sprintf("Vulkan %s matmul matches the CPU (vec and matrix paths)", cc$name), {
      skip_no_vk()
      for (N in c(1L, 4L, 64L)) expect_quant_mm_agrees(cc$type, cc$fn, M = 96L, K = 512L, N = N)
    })
  })
}

test_that("Vulkan large-N quantized matmul matches the CPU (MMQ / mul_mmq_cm1)", {
  skip_no_vk()
  expect_quant_mm_agrees(GGML_TYPE_Q4_0, quantize_q4_0, M = 256L, K = 512L, N = 512L)
  expect_quant_mm_agrees(GGML_TYPE_Q8_0, quantize_q8_0, M = 256L, K = 512L, N = 512L)
})

# ---- TOP_K radix select -----------------------------------------------------

test_that("Vulkan top_k with k past the tournament pipelines (radix) matches the CPU", {
  # num_topk_pipelines = 11: k >= 1024 has no tournament pipeline.
  skip_no_vk()
  ncols <- 8192; nrows <- 3
  set.seed(11)
  x <- rnorm(ncols * nrows)
  for (k in c(1024L, 3000L)) {
    build <- function(ctx, t) ggml_top_k(ctx, t, k)
    cpu <- matrix(run_graph(FALSE, list(f32(x, ncols, nrows)), build), k)
    gpu <- matrix(run_graph(TRUE,  list(f32(x, ncols, nrows)), build), k)
    # top_k returns its indices in no particular order: compare as sets
    for (r in seq_len(nrows)) {
      expect_equal(sort(gpu[, r]), sort(cpu[, r]), label = sprintf("k=%d row %d", k, r))
      expect_equal(sort(cpu[, r]), sort(order(-x[(r - 1) * ncols + 1:ncols])[1:k] - 1))
    }
  }
})

# ---- cpy_transpose_02 -------------------------------------------------------

test_that("Vulkan cont of a 0<->2 permute (cpy_transpose_02) is exact", {
  skip_no_vk()
  A <- 33; B <- 5; C <- 70
  set.seed(12)
  x <- round(rnorm(A * B * C), 2)
  ref <- as.vector(aperm(array(x, c(A, B, C)), c(3, 2, 1)))
  for (mk in list(f32, f16)) {
    gpu <- run_graph(TRUE, list(mk(x, A, B, C)),
                     function(ctx, t) ggml_cont(ctx, ggml_permute(ctx, t, 2, 1, 0, 3)))
    cpu <- run_graph(FALSE, list(mk(x, A, B, C)),
                     function(ctx, t) ggml_cont(ctx, ggml_permute(ctx, t, 2, 1, 0, 3)))
    expect_identical(gpu, cpu)
    expect_equal(cpu, ref, tolerance = 1e-3)
  }
})

# ---- snake fusion -----------------------------------------------------------

test_that("Vulkan snake chain (fused) matches the CPU", {
  # x + sin(a * x)^2 * inv_b, x [T, C], a / inv_b [1, C]: the five nodes
  # MUL, SIN, SQR, MUL, ADD run as one fused snake dispatch on Vulkan.
  skip_no_vk()
  T_ <- 200; C <- 16
  set.seed(13)
  a <- runif(C, 0.5, 2)
  inv_b <- 1 / (a + 1e-9)
  x <- rnorm(T_ * C)
  snake <- function(ctx, tx, ta, tb) {
    s <- ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, tx, ta)))
    ggml_add(ctx, tx, ggml_mul(ctx, s, tb))
  }
  cpu <- expect_gpu_matches_cpu(list(f32(x, T_, C), f32(a, 1, C), f32(inv_b, 1, C)),
                                snake, tol = 1e-5)
  xm <- matrix(x, T_, C)
  ref <- xm + sweep(sin(sweep(xm, 2, a, "*"))^2, 2, inv_b, "*")
  expect_equal(cpu, as.vector(ref), tolerance = 1e-5)
})

# ---- Hadamard-hinted matmul (FWHT) ------------------------------------------

hadamard <- function(n) {
  h <- matrix(1, 1, 1)
  while (nrow(h) < n) h <- rbind(cbind(h, h), cbind(h, -h))
  h / sqrt(n)
}

run_hadamard <- function(use_gpu, n, nr, x, hint) {
  run_graph(use_gpu, list(f32(as.vector(hadamard(n)), n, n), f32(x, n, nr)),
            function(ctx, h, b) {
              r <- ggml_mul_mat(ctx, h, b)
              if (hint) ggml_mul_mat_set_hint(r, GGML_HINT_SRC0_IS_HADAMARD) else r
            })
}

test_that("Hadamard-hinted matmul equals H %*% x on both backends", {
  skip_no_vk()
  nr <- 9
  for (n in c(64L, 128L, 256L, 512L, 1024L)) {   # 1024: no FWHT pipeline, plain matmul
    set.seed(n)
    x <- rnorm(n * nr)
    ref <- as.vector(hadamard(n) %*% matrix(x, n, nr))
    cpu <- run_hadamard(FALSE, n, nr, x, hint = TRUE)
    gpu <- run_hadamard(TRUE,  n, nr, x, hint = TRUE)
    expect_equal(cpu, ref, tolerance = 1e-5, label = sprintf("CPU n=%d", n))
    # the no-FWHT fallback is an ordinary f16-accumulated Vulkan matmul
    expect_equal(gpu, ref, tolerance = if (n == 1024L) 1e-2 else 1e-5,
                 label = sprintf("Vulkan n=%d", n))
  }
})

test_that("ggml_mul_mat_set_hint rejects nodes the FWHT kernel cannot run", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  h <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 48, 48)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 48, 2)
  expect_error(ggml_mul_mat_set_hint(ggml_mul_mat(ctx, h, b), GGML_HINT_SRC0_IS_HADAMARD),
               "power-of-two")
  expect_error(ggml_mul_mat_set_hint(b, GGML_HINT_SRC0_IS_HADAMARD), "not a MUL_MAT")
  expect_error(ggml_mul_mat_set_hint(ggml_mul_mat(ctx, h, b), 7L), "unknown hint")
  r <- ggml_mul_mat(ctx, h, b)
  expect_identical(ggml_mul_mat_set_hint(r, GGML_HINT_NONE), r)
})

# ---- flash attention: sparse K/V (n_kv_max), single-query decode ------------

# q [D, N, Hq], k/v [D, KV, Hkv] F16, mask [KV, N] F16. Hq / Hkv > 1 (GQA) is
# what the sparse gather requires outside the scalar single-query path.
run_fa <- function(use_gpu, D, N, Hq, Hkv, KV, q, k, v, mask, n_kv_max = 0L) {
  inputs <- list(f32(q, D, N, Hq), f16(k, D, KV, Hkv), f16(v, D, KV, Hkv))
  if (!is.null(mask)) inputs[[4]] <- f16(mask, KV, N)
  run_graph(use_gpu, inputs, function(ctx, tq, tk, tv, tm = NULL) {
    r <- ggml_flash_attn_ext(ctx, tq, tk, tv, tm, scale = 1 / sqrt(D))
    if (n_kv_max > 0) ggml_flash_attn_ext_set_n_kv_max(r, n_kv_max) else r
  })
}

test_that("Vulkan flash attention with a sparse K/V bound matches the CPU", {
  # KV >= max(4096, 2 * n_kv_max) with F16 K/V switches on the sparse gather.
  skip_no_vk()
  D <- 64; Hq <- 4; Hkv <- 1; KV <- 4096; n_kv_max <- 64L
  for (N in c(1L, 3L)) {
    set.seed(100 + N)
    q <- rnorm(D * N * Hq); k <- rnorm(D * KV * Hkv); v <- rnorm(D * KV * Hkv)
    mask <- matrix(-Inf, KV, N)
    for (j in seq_len(N)) mask[sample.int(KV, n_kv_max - j), j] <- 0
    mask <- as.vector(mask)
    cpu <- run_fa(FALSE, D, N, Hq, Hkv, KV, q, k, v, mask)
    sparse <- run_fa(TRUE, D, N, Hq, Hkv, KV, q, k, v, mask, n_kv_max = n_kv_max)
    dense  <- run_fa(TRUE, D, N, Hq, Hkv, KV, q, k, v, mask)
    expect_equal(sparse, cpu, tolerance = 5e-3, label = sprintf("sparse N=%d", N))
    expect_equal(sparse, dense, tolerance = 5e-3, label = sprintf("sparse vs dense N=%d", N))
  }
})

test_that("Vulkan single-query flash attention over a long KV matches the CPU", {
  # The decode shape: Intel Xe takes the dual-phase decode shaders here,
  # other devices the split-k path.
  skip_no_vk()
  D <- 128; Hq <- 8; Hkv <- 2; KV <- 2048; N <- 1L
  set.seed(200)
  q <- rnorm(D * N * Hq); k <- rnorm(D * KV * Hkv); v <- rnorm(D * KV * Hkv)
  expect_equal(run_fa(TRUE,  D, N, Hq, Hkv, KV, q, k, v, NULL),
               run_fa(FALSE, D, N, Hq, Hkv, KV, q, k, v, NULL), tolerance = 5e-3)
})

test_that("ggml_flash_attn_ext_set_n_kv_max rejects bad input", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  q <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, 2)
  k <- ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 16, 2)
  r <- ggml_flash_attn_ext(ctx, q, k, k, NULL, scale = 0.125)
  expect_error(ggml_flash_attn_ext_set_n_kv_max(r, -1), "non-negative")
  expect_error(ggml_flash_attn_ext_set_n_kv_max(q, 8), "not a FLASH_ATTN_EXT")
  expect_identical(ggml_flash_attn_ext_set_n_kv_max(r, 0), r)
})

# ---- POOL_2D (0.27 pool2d.comp: BLOCK_SIZE spec constant, subgroup variant) ----

test_that("Vulkan pool_2d matches the CPU (narrow window and wide subgroup window)", {
  # A window at least the subgroup size wide takes pool2d_subgroup_f32 (one
  # subgroup per output, lanes along the row); a 3x3 window the plain variant.
  # AVG now scales after the sum, so it is close to the CPU rather than exact.
  skip_no_vk()
  W <- 200; H <- 6; C <- 3; N <- 2
  set.seed(14)
  x <- rnorm(W * H * C * N)
  for (op in c(GGML_OP_POOL_MAX, GGML_OP_POOL_AVG)) {
    for (cfg in list(c(k0 = 3, k1 = 3, s0 = 2, s1 = 2, p0 = 1, p1 = 1),
                     c(k0 = 64, k1 = 2, s0 = 16, s1 = 1, p0 = 0, p1 = 0))) {
      expect_gpu_matches_cpu(
        list(f32(x, W, H, C, N)),
        function(ctx, t) ggml_pool_2d(ctx, t, op, k0 = cfg[["k0"]], k1 = cfg[["k1"]],
                                      s0 = cfg[["s0"]], s1 = cfg[["s1"]],
                                      p0 = cfg[["p0"]], p1 = cfg[["p1"]]),
        tol = 1e-5)
    }
  }
})

test_that("Vulkan pool_2d over more than 65535 workgroups (2D grid) matches the CPU", {
  # The subgroup variant packs 128 / subgroup_size outputs per workgroup (2 on
  # 64-wide AMD, 4 on NVIDIA), so 320000 outputs need 80000-160000 workgroups:
  # past the x cap of 65535, the grid wraps into y. RADV reports
  # maxComputeWorkGroupCount[0] = UINT32_MAX, which used to make groups_y 0 and
  # kill R with SIGFPE; GGML_VK_DEBUG_MAX_WG_COUNT_X=4294967295 reproduces that
  # limit on any GPU.
  skip_no_vk()
  W <- 127; H <- 50; C <- 100           # k0 = 64, s0 = 1 -> OW = 64; 64 * 50 * 100 outputs
  set.seed(15)
  x <- rnorm(W * H * C)
  for (op in c(GGML_OP_POOL_MAX, GGML_OP_POOL_AVG)) {
    expect_gpu_matches_cpu(
      list(f32(x, W, H, C, 1)),
      function(ctx, t) ggml_pool_2d(ctx, t, op, k0 = 64, k1 = 1, s0 = 1, s1 = 1, p0 = 0, p1 = 0),
      tol = 1e-5)
  }
})

test_that("Vulkan cumsum matches the CPU (small, single-pass and multipass rows)", {
  # <= 512 columns: cumsum_small; above: cumsum_f32 (256 threads, ELEM_PER_THREAD
  # columns each, 0.27); a few rows over 4096 columns: the multipass shaders.
  # Column counts are not multiples of the block so the bounds checks run.
  skip_no_vk()
  for (cfg in list(c(nc = 300, nr = 7), c(nc = 1999, nr = 6), c(nc = 9001, nr = 2))) {
    nc <- cfg[["nc"]]; nr <- cfg[["nr"]]
    set.seed(nc)
    x <- rnorm(nc * nr)
    cpu <- expect_gpu_matches_cpu(list(f32(x, nc, nr)), function(ctx, t) ggml_cumsum(ctx, t),
                                  tol = 1e-4)
    expect_equal(cpu, as.vector(apply(matrix(x, nc, nr), 2, cumsum)), tolerance = 1e-4)
  }
})
