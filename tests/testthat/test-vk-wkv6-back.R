# Vulkan kernels wkv6_back.comp + seq_partial_sum.comp for RWKV_WKV6_BACK
# (TODO 9.5-gpu-gaps). The CPU kernel is checked against a numeric gradient in
# test-ssm-rwkv.R; here Vulkan is compared with the CPU. As for GLA the GPU
# replays from checkpoints every K = 32 tokens (GGML_VK_RECUR_BACK_K), so the
# token counts straddle segment boundaries. d_tf is shared by all sequences and
# reduced in a second pass over per-sequence partials in a fixed order: two
# Vulkan runs must agree bit for bit, with one sequence and with several. The
# Vulkan device must actually offer the op, or the CPU fallback would pass.

skip_if_no_vk <- function() {
  skip_on_cran()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
}

wkv6_back_case <- function(H, n_tok, ns) {
  S <- 64L
  n <- S * H * n_tok
  list(S = S, H = H, n_tok = n_tok, ns = ns,
       k = runif(n, -0.4, 0.4), v = runif(n, -0.4, 0.4), r = runif(n, -0.4, 0.4),
       tf = runif(S * H, -0.5, 0.5),
       td = exp(runif(n, -1, -0.1)),
       s = runif(S * S * H * ns, -0.4, 0.4),
       grad = runif(S * H * (n_tok + S * ns), -1, 1))
}

run_wkv6_back <- function(cc, backend = c("cpu", "vulkan")) {
  backend <- match.arg(backend)
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  b <- if (backend == "vulkan") ggml_vulkan_init(0L) else {
    cb <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cb, 2L); cb
  }
  on.exit(ggml_backend_free(b), add = TRUE, after = FALSE)

  t3 <- function() ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cc$S, cc$H, cc$n_tok)
  tk <- t3(); tv <- t3(); tr <- t3(); ttd <- t3()
  ttf <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cc$S, cc$H)
  ts  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cc$S * cc$S * cc$H, cc$ns)
  tg  <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(cc$grad))
  ins <- list(tk, tv, tr, ttf, ttd, ts, tg)
  for (t in ins) ggml_set_input(t)

  out <- ggml_rwkv_wkv6_back(ctx, tk, tv, tr, ttf, ttd, ts, tg)
  ggml_set_output(out)
  supported <- ggml_backend_dev_supports_op(ggml_backend_get_device(b), out)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  vals <- list(cc$k, cc$v, cc$r, cc$tf, cc$td, cc$s, cc$grad)
  for (i in seq_along(ins)) ggml_backend_tensor_set_data(ins[[i]], vals[[i]])
  ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
  list(out = ggml_backend_tensor_get_data(out), supported = supported)
}

# d_tf sits after d_k, d_v, d_r in the packed output
d_tf_of <- function(out, cc) {
  C <- cc$S * cc$H
  out[3 * C * cc$n_tok + seq_len(C)]
}

test_that("Vulkan WKV6_BACK matches the CPU across checkpoint boundaries", {
  skip_if_no_vk()
  set.seed(71)
  # (H, n_tok, n_seqs); tokens per sequence L = n_tok / n_seqs, K = 32
  cases <- list(c(1, 5, 1),     # L < K, one sequence
                c(2, 32, 1),    # L == K
                c(2, 66, 2),    # L == K + 1, two sequences
                c(3, 70, 1),    # 3 segments, last one partial
                c(1, 128, 4))   # L == K, four sequences
  for (cs in cases) {
    cc <- wkv6_back_case(cs[1], cs[2], cs[3])
    cpu <- run_wkv6_back(cc, "cpu")
    vk  <- run_wkv6_back(cc, "vulkan")
    info <- sprintf("H=%d n_tok=%d ns=%d", cs[1], cs[2], cs[3])
    expect_true(vk$supported, info = info)
    expect_true(all(is.finite(vk$out)), info = info)
    expect_equal(vk$out, cpu$out, tolerance = 1e-4, info = info)
    expect_equal(d_tf_of(vk$out, cc), d_tf_of(cpu$out, cc), tolerance = 1e-4, info = info)
  }
})

test_that("Vulkan WKV6_BACK is reproducible run to run (d_tf included)", {
  skip_if_no_vk()
  set.seed(72)
  for (ns in c(1L, 3L)) {     # single sequence and the cross-sequence reduction
    cc <- wkv6_back_case(2, 33L * ns, ns)
    a <- run_wkv6_back(cc, "vulkan")
    b <- run_wkv6_back(cc, "vulkan")
    expect_identical(a$out, b$out, info = sprintf("ns=%d", ns))
    expect_identical(d_tf_of(a$out, cc), d_tf_of(b$out, cc), info = sprintf("ns=%d", ns))
  }
})

test_that("Vulkan WKV6_BACK long sequence (many segments)", {
  skip_if_no_vk()
  set.seed(73)
  cc <- wkv6_back_case(4, 1000, 1)        # 32 segments, last one 8 tokens
  cpu <- run_wkv6_back(cc, "cpu")
  vk  <- run_wkv6_back(cc, "vulkan")
  expect_true(vk$supported)
  expect_equal(vk$out, cpu$out, tolerance = 1e-4)
})
