# Vulkan kernel wkv7_back.comp for RWKV_WKV7_BACK (TODO 9.5-gpu-gaps).
# The CPU kernel is checked against a numeric gradient in test-ssm-rwkv.R; here
# Vulkan is compared with the CPU. As for GLA the GPU replays from checkpoints
# every K = 32 tokens (GGML_VK_RECUR_BACK_K), so the token counts straddle
# segment boundaries. Five gradients are row sums reduced through one shared
# tile in a fixed order: two Vulkan runs must agree bit for bit. The Vulkan
# device must actually offer the op, or the CPU fallback would pass.
#
# a and b are kept small: with S = 64 the sa*b term can otherwise grow the
# state without bound over a long sequence, and CPU and GPU would then differ
# by overflow rather than by a defect.

skip_if_no_vk <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "Vulkan not available")
}

wkv7_back_case <- function(H, n_tok, ns) {
  S <- 64L
  n <- S * H * n_tok
  list(S = S, H = H, n_tok = n_tok, ns = ns,
       r = runif(n, -0.4, 0.4), w = exp(runif(n, -1, -0.1)),
       k = runif(n, -0.4, 0.4), v = runif(n, -0.4, 0.4),
       a = runif(n, -0.05, 0.05), b = runif(n, -0.05, 0.05),
       s = runif(S * S * H * ns, -0.4, 0.4),
       grad = runif(S * H * (n_tok + S * ns), -1, 1))
}

run_wkv7_back <- function(cc, backend = c("cpu", "vulkan")) {
  backend <- match.arg(backend)
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  b <- if (backend == "vulkan") ggml_vulkan_init(0L) else {
    cb <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cb, 2L); cb
  }
  on.exit(ggml_backend_free(b), add = TRUE, after = FALSE)

  t3 <- function() ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cc$S, cc$H, cc$n_tok)
  tr <- t3(); tw <- t3(); tk <- t3(); tv <- t3(); ta <- t3(); tb <- t3()
  ts <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cc$S * cc$S * cc$H, cc$ns)
  tg <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(cc$grad))
  ins <- list(tr, tw, tk, tv, ta, tb, ts, tg)
  for (t in ins) ggml_set_input(t)

  out <- ggml_rwkv_wkv7_back(ctx, tr, tw, tk, tv, ta, tb, ts, tg)
  ggml_set_output(out)
  supported <- ggml_backend_dev_supports_op(ggml_backend_get_device(b), out)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  vals <- list(cc$r, cc$w, cc$k, cc$v, cc$a, cc$b, cc$s, cc$grad)
  for (i in seq_along(ins)) ggml_backend_tensor_set_data(ins[[i]], vals[[i]])
  ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
  list(out = ggml_backend_tensor_get_data(out), supported = supported)
}

test_that("Vulkan WKV7_BACK matches the CPU across checkpoint boundaries", {
  skip_if_no_vk()
  set.seed(81)
  # (H, n_tok, n_seqs); tokens per sequence L = n_tok / n_seqs, K = 32
  cases <- list(c(1, 5, 1),     # L < K, one sequence
                c(2, 32, 1),    # L == K
                c(2, 66, 2),    # L == K + 1, two sequences
                c(3, 70, 1),    # 3 segments, last one partial
                c(1, 128, 4))   # L == K, four sequences
  for (cs in cases) {
    cc <- wkv7_back_case(cs[1], cs[2], cs[3])
    cpu <- run_wkv7_back(cc, "cpu")
    vk  <- run_wkv7_back(cc, "vulkan")
    info <- sprintf("H=%d n_tok=%d ns=%d", cs[1], cs[2], cs[3])
    expect_true(vk$supported, info = info)
    expect_true(all(is.finite(vk$out)), info = info)
    expect_equal(vk$out, cpu$out, tolerance = 1e-4, info = info)
  }
})

test_that("Vulkan WKV7_BACK is reproducible run to run", {
  skip_if_no_vk()
  set.seed(82)
  cc <- wkv7_back_case(2, 100, 2)
  a <- run_wkv7_back(cc, "vulkan")
  b <- run_wkv7_back(cc, "vulkan")
  expect_identical(a$out, b$out)
})

test_that("Vulkan WKV7_BACK long sequence (many segments)", {
  skip_if_no_vk()
  set.seed(83)
  cc <- wkv7_back_case(4, 1000, 1)        # 32 segments, last one 8 tokens
  cpu <- run_wkv7_back(cc, "cpu")
  vk  <- run_wkv7_back(cc, "vulkan")
  expect_true(vk$supported)
  expect_equal(vk$out, cpu$out, tolerance = 1e-4)
})
