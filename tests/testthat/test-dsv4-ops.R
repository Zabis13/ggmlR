# ggml 0.27 DeepSeek V4 hyper-connection ops and the DSA lightning indexer:
# the CPU against references written in R, Vulkan against the CPU where the
# Vulkan backend takes the node (hc = 4; the indexer needs n_embd = 128 and
# fp16 support).

run_graph <- function(use_gpu, inputs, build) {
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  tensors <- lapply(inputs, function(x) {
    ne <- c(x$shape, rep(1, 4 - length(x$shape)))
    ggml_new_tensor_4d(ctx, x$type, ne[1], ne[2], ne[3], ne[4])
  })
  out <- do.call(build, c(list(ctx), tensors))
  gf <- ggml_build_forward_expand(ctx, out)
  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  for (i in seq_along(inputs)) ggml_backend_tensor_set_data(tensors[[i]], inputs[[i]]$data)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

# TRUE when the Vulkan device accepts the node `build` makes from `inputs`
vk_supports <- function(inputs, build) {
  if (!ggml_vulkan_available()) return(FALSE)
  gpu <- ggml_vulkan_init(0)
  on.exit(ggml_backend_free(gpu))
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  tensors <- lapply(inputs, function(x) {
    ne <- c(x$shape, rep(1, 4 - length(x$shape)))
    ggml_new_tensor_4d(ctx, x$type, ne[1], ne[2], ne[3], ne[4])
  })
  ggml_backend_dev_supports_op(ggml_backend_get_device(gpu), do.call(build, c(list(ctx), tensors)))
}

expect_gpu_matches_cpu <- function(inputs, build, tol = 1e-4) {
  skip_if_not(vk_supports(inputs, build), "Vulkan does not take this node")
  expect_equal(run_graph(TRUE, inputs, build), run_graph(FALSE, inputs, build), tolerance = tol)
}

f32 <- function(data, ...) list(type = GGML_TYPE_F32, shape = c(...), data = data)
f16 <- function(data, ...) list(type = GGML_TYPE_F16, shape = c(...), data = data)
sigmoid <- function(x) 1 / (1 + exp(-x))
HC <- 4

# ---- hc_comb ------------------------------------------------------------------

hc_comb_ref <- function(mixes, scale, base, eps, n_iter, n_tokens) {
  mixes <- matrix(mixes, 24, n_tokens)
  out <- numeric(0)
  for (t in seq_len(n_tokens)) {
    v <- mixes[9:24, t] * scale[3] + base[9:24]   # comb block, 0-based offset 8
    C <- matrix(v, HC, HC)                         # [dst, src]
    C <- apply(C, 2, function(col) { e <- exp(col - max(col)); e / sum(e) }) + eps
    norm_dst <- function(C) C / (eps + rowSums(C))                    # over src, per dst
    norm_src <- function(C) sweep(C, 2, eps + colSums(C), "/")        # over dst, per src
    C <- norm_dst(C)
    for (i in seq_len(n_iter - 1)) C <- norm_dst(norm_src(C))
    out <- c(out, as.vector(C))
  }
  out
}

hc_comb_case <- function(n_tokens, eps, n_iter) {
  set.seed(n_tokens + n_iter)
  mixes <- rnorm(24 * n_tokens); scale <- c(0.7, 1.3, 0.9); base <- rnorm(24, sd = 0.5)
  list(inputs = list(f32(mixes, 24, n_tokens), f32(scale, 3), f32(base, 24)),
       build  = function(ctx, m, s, b) ggml_dsv4_hc_comb(ctx, m, s, b, eps = eps, n_iter = n_iter),
       ref    = hc_comb_ref(mixes, scale, base, eps, n_iter, n_tokens))
}

test_that("dsv4_hc_comb matches the R reference on the CPU", {
  for (cfg in list(c(n = 1, it = 1), c(n = 7, it = 3), c(n = 20, it = 20))) {
    cc <- hc_comb_case(cfg[["n"]], eps = 1e-6, n_iter = cfg[["it"]])
    expect_equal(run_graph(FALSE, cc$inputs, cc$build), cc$ref, tolerance = 1e-5)
  }
})

test_that("Vulkan dsv4_hc_comb matches the CPU", {
  for (n in c(1, 7, 33)) {
    cc <- hc_comb_case(n, eps = 1e-6, n_iter = 5)
    expect_gpu_matches_cpu(cc$inputs, cc$build)
  }
})

# ---- hc_pre -------------------------------------------------------------------

hc_pre_case <- function(n_embd, n_tokens, gated, scale = 0.5) {
  set.seed(n_embd + n_tokens + gated)
  x <- rnorm(n_embd * HC * n_tokens)
  if (gated) {
    g <- rnorm(n_embd * HC * n_tokens)
    xa <- array(x, c(n_embd, HC, n_tokens)); ga <- array(g, c(n_embd, HC, n_tokens))
    ref <- as.vector(scale * apply(xa * sigmoid(ga), c(1, 3), sum))
    list(inputs = list(f32(x, n_embd, HC, n_tokens), f32(g, n_embd, HC, n_tokens)),
         build  = function(ctx, tx, tg) ggml_dsv4_hc_pre(ctx, tx, gate = tg, scale = scale), ref = ref)
  } else {
    w <- rnorm(HC * n_tokens)
    xa <- array(x, c(n_embd, HC, n_tokens)); wm <- matrix(w, HC, n_tokens)
    ref <- as.vector(sapply(seq_len(n_tokens), function(t) xa[, , t] %*% wm[, t]))
    list(inputs = list(f32(x, n_embd, HC, n_tokens), f32(w, HC, n_tokens)),
         build  = function(ctx, tx, tw) ggml_dsv4_hc_pre(ctx, tx, weights = tw), ref = ref)
  }
}

test_that("dsv4_hc_pre (weights and gated) matches the R reference on the CPU", {
  for (gated in c(FALSE, TRUE)) {
    cc <- hc_pre_case(48, 6, gated)
    expect_equal(run_graph(FALSE, cc$inputs, cc$build), cc$ref, tolerance = 1e-5)
  }
})

test_that("Vulkan dsv4_hc_pre (weights and gated) matches the CPU", {
  for (gated in c(FALSE, TRUE)) for (n_embd in c(64, 300)) {
    cc <- hc_pre_case(n_embd, 5, gated)
    expect_gpu_matches_cpu(cc$inputs, cc$build)
  }
})

# ---- hc_post ------------------------------------------------------------------

hc_post_case <- function(n_embd, n_tokens, with_comb) {
  set.seed(n_embd * 3 + n_tokens + with_comb)
  x <- rnorm(n_embd * n_tokens); r <- rnorm(n_embd * HC * n_tokens); p <- rnorm(HC * n_tokens)
  xm <- matrix(x, n_embd, n_tokens); ra <- array(r, c(n_embd, HC, n_tokens)); pm <- matrix(p, HC, n_tokens)
  out <- array(0, c(n_embd, HC, n_tokens))
  if (with_comb) {
    cmb <- rnorm(HC * HC * n_tokens); ca <- array(cmb, c(HC, HC, n_tokens))
    for (t in seq_len(n_tokens)) out[, , t] <- outer(xm[, t], pm[, t]) + ra[, , t] %*% t(ca[, , t])
    inputs <- list(f32(x, n_embd, n_tokens), f32(r, n_embd, HC, n_tokens), f32(p, HC, n_tokens),
                   f32(cmb, HC, HC, n_tokens))
    build <- function(ctx, tx, tr, tp, tc) ggml_dsv4_hc_post(ctx, tx, tr, tp, tc)
  } else {
    for (t in seq_len(n_tokens)) out[, , t] <- outer(xm[, t], pm[, t]) + ra[, , t]
    inputs <- list(f32(x, n_embd, n_tokens), f32(r, n_embd, HC, n_tokens), f32(p, HC, n_tokens))
    build <- function(ctx, tx, tr, tp) ggml_dsv4_hc_post(ctx, tx, tr, tp)
  }
  list(inputs = inputs, build = build, ref = as.vector(out))
}

test_that("dsv4_hc_post (with and without comb) matches the R reference on the CPU", {
  for (with_comb in c(TRUE, FALSE)) {
    cc <- hc_post_case(40, 6, with_comb)
    expect_equal(run_graph(FALSE, cc$inputs, cc$build), cc$ref, tolerance = 1e-5)
  }
})

test_that("Vulkan dsv4_hc_post (with and without comb) matches the CPU", {
  for (with_comb in c(TRUE, FALSE)) for (n_embd in c(64, 300)) {
    cc <- hc_post_case(n_embd, 5, with_comb)
    expect_gpu_matches_cpu(cc$inputs, cc$build)
  }
})

# ---- HC_POST_GATE fusion (scale -> sigmoid -> scale -> hc_post) ----------------
# The fused shader writes hc_post's output while it still reads the first scale's
# input, so Vulkan fuses only when, in the memory actually allocated, the output
# overlaps no external input. The scheduler guarantees that through the allocator
# deps graph_optimize registers; a gallocr graph fuses when its layout allows.

hc_gate_count <- function() .Call("R_ggml_vk_hc_post_gate_count", PACKAGE = "ggmlR")
hc_gate_overlap <- function(gf, mode) {
  .Call("R_ggml_vk_test_fused_overlap", gf, 0L, 4L, as.integer(mode), PACKAGE = "ggmlR")
}

# post = s_out * sigmoid(s_in * p_src)
hc_gate_build <- function(ctx, tx, tr, p_src, s_in = 0.7, s_out = 1.3) {
  post <- ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, p_src, s_in)), s_out)
  ggml_dsv4_hc_post(ctx, tx, tr, post)
}

hc_gate_ref <- function(x, r, p, n_embd, n_tokens, s_in = 0.7, s_out = 1.3) {
  pm <- matrix(s_out / (1 + exp(-s_in * p)), HC, n_tokens)
  xm <- matrix(x, n_embd, n_tokens); ra <- array(r, c(n_embd, HC, n_tokens))
  out <- array(0, c(n_embd, HC, n_tokens))
  for (t in seq_len(n_tokens)) out[, , t] <- outer(xm[, t], pm[, t]) + ra[, , t]
  as.vector(out)
}

test_that("the fused-output overlap check catches every input the output lands on", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan")
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))
  tx <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 5)
  tr <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, HC, 5)
  tp <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HC, 5)
  gf <- ggml_build_forward_expand(ctx, hc_gate_build(ctx, tx, tr, tp))
  expect_equal(ggml_graph_n_nodes(gf), 4)
  expect_false(hc_gate_overlap(gf, 0))   # separate allocations
  expect_true(hc_gate_overlap(gf, 1))    # output = first scale's input
  expect_true(hc_gate_overlap(gf, 2))    # partial overlap
  expect_true(hc_gate_overlap(gf, 3))    # unallocated output: not provably safe
  expect_false(hc_gate_overlap(gf, 4))   # same address, another buffer
  expect_true(hc_gate_overlap(gf, 5))    # output = hc_post's residual
})

test_that("Vulkan HC_POST_GATE fuses on a gallocr graph and matches the CPU", {
  n_embd <- 64; n_tokens <- 5
  set.seed(31)
  x <- rnorm(n_embd * n_tokens); r <- rnorm(n_embd * HC * n_tokens); p <- rnorm(HC * n_tokens)
  inputs <- list(f32(x, n_embd, n_tokens), f32(r, n_embd, HC, n_tokens), f32(p, HC, n_tokens))
  ref <- hc_gate_ref(x, r, p, n_embd, n_tokens)
  expect_equal(run_graph(FALSE, inputs, hc_gate_build), ref, tolerance = 1e-5)
  skip_if_not(vk_supports(inputs, function(ctx, tx, tr, tp) ggml_dsv4_hc_post(ctx, tx, tr, tp)),
              "Vulkan does not take hc_post")
  c0 <- hc_gate_count()
  expect_equal(run_graph(TRUE, inputs, hc_gate_build), ref, tolerance = 1e-4)
  expect_equal(hc_gate_count() - c0, 1)
})

test_that("Vulkan HC_POST_GATE fuses under the scheduler, again on a repeated compute", {
  # p_src is an intermediate (p + p), so without the allocator deps its memory is
  # free once the first scale has read it and the fused output could land there.
  skip_if_not(ggml_vulkan_available(), "no Vulkan")
  n_embd <- 64; n_tokens <- 5
  set.seed(32)
  x <- rnorm(n_embd * n_tokens); r <- rnorm(n_embd * HC * n_tokens); p <- rnorm(HC * n_tokens)
  be <- ggml_vulkan_init(0)
  on.exit(ggml_backend_free(be))
  wctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(wctx), add = TRUE, after = FALSE)
  tx <- ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, n_tokens)
  tr <- ggml_new_tensor_3d(wctx, GGML_TYPE_F32, n_embd, HC, n_tokens)
  tp <- ggml_new_tensor_2d(wctx, GGML_TYPE_F32, HC, n_tokens)
  buf <- ggml_backend_alloc_ctx_tensors(wctx, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(tx, x)
  ggml_backend_tensor_set_data(tr, r)
  ggml_backend_tensor_set_data(tp, p)
  skip_if_not(ggml_backend_dev_supports_op(ggml_backend_get_device(be),
                                           ggml_dsv4_hc_post(wctx, tx, tr, tp)),
              "Vulkan does not take hc_post")

  g <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(g), add = TRUE, after = FALSE)
  out <- hc_gate_build(g, tx, tr, ggml_add(g, tp, tp))
  gr <- ggml_build_forward_expand(g, out)
  s <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit(ggml_backend_sched_free(s), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(s, gr)

  ref <- hc_gate_ref(x, r, 2 * p, n_embd, n_tokens)
  for (rep in 1:2) {
    c0 <- hc_gate_count()
    ggml_backend_sched_graph_compute(s, gr)
    expect_equal(ggml_backend_tensor_get_data(out), ref, tolerance = 1e-4, label = sprintf("run %d", rep))
    expect_equal(hc_gate_count() - c0, 1, label = sprintf("fusions on run %d", rep))
  }
})

# ---- lightning indexer --------------------------------------------------------

li_case <- function(n_kv, n_tokens, ne3 = 1, ne33 = 1, n_embd = 128, n_head = 4, k_type = "f32") {
  set.seed(n_kv + n_tokens + ne3)
  q <- rnorm(n_embd * n_head * n_tokens * ne3, sd = 0.3)
  k <- rnorm(n_embd * n_kv * ne3, sd = 0.3)
  w <- runif(n_head * n_tokens * ne3)
  m <- sample(c(0, -1, -Inf), n_kv * n_tokens * ne33, replace = TRUE, prob = c(0.6, 0.3, 0.1))

  qa <- array(q, c(n_embd, n_head, n_tokens, ne3)); ka <- array(k, c(n_embd, n_kv, ne3))
  wa <- array(w, c(n_head, n_tokens, ne3));        ma <- array(m, c(n_kv, n_tokens, ne33))
  ref <- array(0, c(n_kv, n_tokens, ne3))
  for (s in seq_len(ne3)) for (t in seq_len(n_tokens)) {
    qk <- crossprod(ka[, , s], qa[, , t, s])          # [n_kv, n_head]
    ref[, t, s] <- pmax(qk, 0) %*% wa[, t, s] + ma[, t, (s - 1) %% ne33 + 1]
  }

  k_in <- switch(k_type,
    f32 = f32(k, n_embd, 1, n_kv, ne3),
    f16 = list(type = GGML_TYPE_F16, shape = c(n_embd, 1, n_kv, ne3), data = k),
    q8_0 = list(type = GGML_TYPE_Q8_0, shape = c(n_embd, 1, n_kv, ne3),
                data = quantize_q8_0(k, n_rows = n_kv * ne3, n_per_row = n_embd)))
  list(inputs = list(f32(q, n_embd, n_head, n_tokens, ne3), k_in,
                     f32(w, n_head, n_tokens, 1, ne3), f16(m, n_kv, n_tokens, 1, ne33)),
       build  = function(ctx, tq, tk, tw, tm) ggml_lightning_indexer(ctx, tq, tk, tw, tm),
       ref    = as.vector(ref))
}

test_that("lightning_indexer matches the R reference on the CPU", {
  for (cfg in list(c(kv = 37, t = 5, s = 1, m = 1), c(kv = 64, t = 3, s = 2, m = 1),
                   c(kv = 20, t = 4, s = 2, m = 2))) {
    cc <- li_case(cfg[["kv"]], cfg[["t"]], cfg[["s"]], cfg[["m"]], n_embd = 32)
    expect_equal(run_graph(FALSE, cc$inputs, cc$build), cc$ref, tolerance = 1e-5)
  }
})

test_that("Vulkan lightning_indexer matches the CPU (K in F32, F16 and Q8_0)", {
  for (k_type in c("f32", "f16", "q8_0")) {
    for (cfg in list(c(kv = 37, t = 5, s = 1, m = 1), c(kv = 130, t = 9, s = 2, m = 1))) {
      cc <- li_case(cfg[["kv"]], cfg[["t"]], cfg[["s"]], cfg[["m"]], k_type = k_type)
      # the Vulkan kernel computes q.k in f16
      expect_gpu_matches_cpu(cc$inputs, cc$build, tol = 5e-3)
    }
  }
})

# ---- wrapper checks -----------------------------------------------------------

test_that("DSV4 wrappers raise R errors instead of aborting on bad shapes", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  t3 <- function(a, b, c) ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a, b, c)
  t2 <- function(a, b) ggml_new_tensor_2d(ctx, GGML_TYPE_F32, a, b)
  t1 <- function(a) ggml_new_tensor_1d(ctx, GGML_TYPE_F32, a)

  expect_error(ggml_dsv4_hc_comb(ctx, t2(20, 3), t1(3), t1(20), 1e-6, 2), "24")
  expect_error(ggml_dsv4_hc_comb(ctx, t2(24, 3), t1(3), t1(24), 1e-6, 0), "n_iter")
  expect_error(ggml_dsv4_hc_pre(ctx, t3(8, 4, 2)), "exactly one")
  expect_error(ggml_dsv4_hc_pre(ctx, t3(8, 4, 2), weights = t2(3, 2)), "weights must be")
  expect_error(ggml_dsv4_hc_pre(ctx, t3(8, 4, 2), gate = t3(8, 4, 3)), "gate must")
  expect_error(ggml_dsv4_hc_post(ctx, t2(8, 2), t3(8, 4, 2), t2(3, 2)), "post must be")
  expect_error(ggml_dsv4_hc_post(ctx, t2(8, 2), t3(8, 4, 2), t2(4, 2), t3(4, 4, 3)), "comb must be")

  q <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 2, 3)
  k <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 1, 5)
  w <- t2(2, 3)
  expect_error(ggml_lightning_indexer(ctx, q, k, w, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 3)), "F16")
  expect_error(ggml_lightning_indexer(ctx, q, k, w, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 3)), "n_kv")
})
