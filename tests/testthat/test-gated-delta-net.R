# ggml_gated_delta_net: R reference vs CPU, and Vulkan vs CPU for every head
# size the Vulkan backend claims (S_v = 16, 32, 64, 128).
#
# S_v = 16 used to pass supports_op while load_shaders built pipelines only for
# 32/64/128, so the op reached ggml_vk_op_f32 with a null pipeline. The Vulkan
# run below computes the graph directly on the Vulkan backend (no scheduler) so
# a CPU fallback cannot hide that.

gdn_case <- function(S_v, H_v = 2L, H_k = H_v, n_tok = 3L, n_seqs = 1L, kda = FALSE, seed = 1L) {
  set.seed(seed)
  list(S_v = S_v, H_v = H_v, H_k = H_k, n_tok = n_tok, n_seqs = n_seqs, kda = kda,
       q = runif(S_v * H_k * n_tok * n_seqs, -1, 1),
       k = runif(S_v * H_k * n_tok * n_seqs, -1, 1) / sqrt(S_v),
       v = runif(S_v * H_v * n_tok * n_seqs, -1, 1),
       g = -runif((if (kda) S_v else 1L) * H_v * n_tok * n_seqs, 0.05, 0.5),
       beta = runif(H_v * n_tok * n_seqs, 0.2, 0.9),
       s0 = runif(S_v * S_v * H_v * n_seqs, -0.5, 0.5))
}

# Mirrors the CPU kernel: state stored transposed, s[j*S_v + i] = S[i][j], which
# as a column-major R matrix is S itself. Heads of q/k broadcast tiled (h %% H_k).
gdn_ref <- function(cs, K = 1L) {
  S <- cs$S_v; H <- cs$H_v; T <- cs$n_tok; B <- cs$n_seqs
  gw <- if (cs$kda) S else 1L
  out <- numeric(S * H * T * B)
  snaps <- matrix(0, S * S * H * B, K)
  for (b in seq_len(B)) for (h in seq_len(H)) {
    hk <- (h - 1L) %% cs$H_k
    Sm <- matrix(cs$s0[((b - 1L) * H + h - 1L) * S * S + seq_len(S * S)], S, S)
    for (t in seq_len(T)) {
      tok <- (b - 1L) * T + t - 1L
      q <- cs$q[(tok * cs$H_k + hk) * S + seq_len(S)]
      k <- cs$k[(tok * cs$H_k + hk) * S + seq_len(S)]
      v <- cs$v[(tok * H + h - 1L) * S + seq_len(S)]
      g <- cs$g[(tok * H + h - 1L) * gw + seq_len(gw)]
      beta <- cs$beta[tok * H + h]
      Sm <- Sm * exp(g)
      delta <- (v - drop(crossprod(Sm, k))) * beta
      Sm <- Sm + outer(k, delta)
      out[(tok * H + h - 1L) * S + seq_len(S)] <- drop(crossprod(Sm, q)) / sqrt(S)
      slot <- T - t
      if (slot < K) snaps[((b - 1L) * H + h - 1L) * S * S + seq_len(S * S), slot + 1L] <- Sm
    }
  }
  c(out, snaps)
}

run_gdn <- function(gpu, cs, K = 1L) {
  be <- if (gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be))
  if (!gpu) ggml_backend_cpu_set_n_threads(be, 2L)
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE, after = FALSE)
  S <- cs$S_v; H <- cs$H_v; T <- cs$n_tok; B <- cs$n_seqs
  f32 <- function(...) ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ...)
  q <- f32(S, cs$H_k, T, B); k <- f32(S, cs$H_k, T, B); v <- f32(S, H, T, B)
  g <- f32(if (cs$kda) S else 1, H, T, B); beta <- f32(1, H, T, B); s0 <- f32(S, S, H, B)
  out <- ggml_gated_delta_net(ctx, q, k, v, g, beta, s0, K = K)
  if (gpu) {
    expect_true(ggml_backend_dev_supports_op(ggml_backend_get_device(be), out),
                label = sprintf("Vulkan supports_op S_v=%d", S))
  }
  gf <- ggml_build_forward_expand(ctx, out)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(q, cs$q); ggml_backend_tensor_set_data(k, cs$k)
  ggml_backend_tensor_set_data(v, cs$v); ggml_backend_tensor_set_data(g, cs$g)
  ggml_backend_tensor_set_data(beta, cs$beta); ggml_backend_tensor_set_data(s0, cs$s0)
  ggml_backend_graph_compute(be, gf)
  ggml_backend_tensor_get_data(out)
}

test_that("gated_delta_net CPU matches an R reference (scalar gate and KDA)", {
  for (kda in c(FALSE, TRUE)) {
    cs <- gdn_case(16L, H_v = 2L, n_tok = 4L, n_seqs = 2L, kda = kda)
    K <- 2L
    got <- run_gdn(FALSE, cs, K)
    want <- gdn_ref(cs, K)
    expect_length(got, length(want))
    expect_equal(got, want, tolerance = 1e-5, label = sprintf("CPU kda=%s", kda))
  }
})

test_that("gated_delta_net broadcasts q/k heads (H_v = 2 * H_k)", {
  cs <- gdn_case(16L, H_v = 4L, H_k = 2L, n_tok = 3L, seed = 2L)
  expect_equal(run_gdn(FALSE, cs), gdn_ref(cs), tolerance = 1e-5)
})

test_that("Vulkan gated_delta_net matches the CPU for S_v = 16, 32, 64, 128", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan")
  for (S_v in c(16L, 32L, 64L, 128L)) for (kda in c(FALSE, TRUE)) {
    cs <- gdn_case(S_v, H_v = 2L, n_tok = 3L, n_seqs = 2L, kda = kda, seed = S_v)
    K <- 2L
    cpu <- run_gdn(FALSE, cs, K)
    gpu <- run_gdn(TRUE, cs, K)
    expect_equal(gpu, cpu, tolerance = 1e-4, label = sprintf("Vulkan S_v=%d kda=%s", S_v, kda))
  }
})

test_that("gated_delta_net rejects bad arguments as R errors", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  f32 <- function(...) ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ...)
  q <- f32(16, 2, 3, 1); v <- f32(16, 2, 3, 1); g <- f32(1, 2, 3, 1)
  beta <- f32(1, 2, 3, 1); s0 <- f32(16, 16, 2, 1)
  expect_error(ggml_gated_delta_net(ctx, q, q, v, g, beta, s0, K = 0), "positive")
  expect_error(ggml_gated_delta_net(ctx, f32(8, 2, 3, 1), f32(8, 2, 3, 1), v, g, beta, s0), "S_k == S_v")
  expect_error(ggml_gated_delta_net(ctx, q, q, v, f32(4, 2, 3, 1), beta, s0), "g must be")
  expect_error(ggml_gated_delta_net(ctx, q, q, v, g, f32(2, 2, 3, 1), s0), "beta must be")
  expect_error(ggml_gated_delta_net(ctx, q, q, v, g, beta, f32(16, 16, 1, 1)), "state must hold")
  expect_error(ggml_gated_delta_net(ctx, q, q, v, g, beta,
                                    ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 16, 2, 1)), "F32")
  r <- ggml_gated_delta_net(ctx, q, q, v, g, beta, s0, K = 3)
  expect_equal(ggml_nelements(r), 16 * 2 * 3 + 3 * 16 * 16 * 2)
})
