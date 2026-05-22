#!/usr/bin/env Rscript
# Micro-benchmark: conv2d code paths on the Vulkan backend.
#
# Context: after the ggml 0.9.5 -> 0.11.0 migration, F32 convolutions on RDNA4
# (RX 9070) regressed badly (Inception 7->32 ms, SqueezeNet 2->7 ms) while the
# LLM mul_mat path stayed fast (~100 t/s). Suspected cause: the new direct
# GGML_OP_CONV_2D kernel runs a scalar FMA path for F32 kernels (cm1/coopmat is
# only enabled for F16 kernels: see ggml-vulkan-attn.cpp use_cm1), whereas the
# old IM2COL + MUL_MAT path uses the coopmat-accelerated matmul.
#
# This bench times the two paths head-to-head on the same convolution so we can
# confirm the hypothesis numerically before changing any dispatch logic:
#   1. ggml_conv_2d_direct  -> GGML_OP_CONV_2D (direct kernel)
#   2. ggml_conv_2d         -> IM2COL + MUL_MAT (old path, coopmat matmul)
# Run with both F32 and F16 kernel weights.

suppressMessages(library(ggmlR))

if (!ggml_vulkan_available()) {
  stop("Vulkan GPU not available — this bench is GPU-only")
}

# kernel a = [KW, KH, IC, OC], input b = [W, H, C=IC, N]
time_path <- function(path_fn, ktype, KW, KH, IC, OC, W, H, N,
                      s = 1L, p = 1L, reps = 50L) {
  ctx <- ggml_init(1024L * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_4d(ctx, ktype,          KW, KH, IC, OC)
  b <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  W,  H,  IC, N)
  out <- path_fn(ctx, a, b, s0 = s, s1 = s, p0 = p, p1 = p)

  backend <- ggml_vulkan_init(0)
  ggml_backend_alloc_ctx_tensors(ctx, backend)

  set.seed(1)
  # kernel weights: set as F32 vector; the setter converts to ktype.
  ggml_backend_tensor_set_data(a, rnorm(KW * KH * IC * OC, sd = 0.05))
  ggml_backend_tensor_set_data(b, rnorm(W * H * IC * N,   sd = 1.0))

  gf <- ggml_build_forward_expand(ctx, out)

  # warm up (pipeline compile + first dispatch)
  ggml_backend_graph_compute(backend, gf)
  invisible(ggml_backend_tensor_get_data(out))

  t0 <- Sys.time()
  for (i in seq_len(reps)) ggml_backend_graph_compute(backend, gf)
  invisible(ggml_backend_tensor_get_data(out))  # force sync
  elapsed_ms <- as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps

  ggml_free(ctx)
  elapsed_ms
}

bench_shape <- function(label, KW, KH, IC, OC, W, H, N) {
  cat(sprintf("\n## %s  kernel[%d,%d,%d,%d] in[%d,%d,%d,%d]\n",
              label, KW, KH, IC, OC, W, H, IC, N))
  for (kt in list(c("F32", GGML_TYPE_F32), c("F16", GGML_TYPE_F16))) {
    ktype <- as.integer(kt[[2]])
    direct <- tryCatch(time_path(ggml_conv_2d_direct, ktype, KW, KH, IC, OC, W, H, N),
                       error = function(e) NA_real_)
    im2col <- tryCatch(time_path(ggml_conv_2d,        ktype, KW, KH, IC, OC, W, H, N),
                       error = function(e) NA_real_)
    cat(sprintf("  %-3s kernel | direct(CONV_2D)=%7.3f ms | im2col+mul_mat=%7.3f ms | direct/im2col=%.2fx\n",
                kt[[1]], direct, im2col, direct / im2col))
  }
}

cat("conv2d path micro-benchmark on Vulkan (lower ms = faster)\n")

# Representative shapes from the regressed models (3x3 convs, mid feature maps).
bench_shape("mid 3x3 (Inception-ish)", 3L, 3L, 192L, 384L, 35L, 35L, 1L)
bench_shape("wide 1x1 (SqueezeNet)",   1L, 1L, 256L, 256L, 27L, 27L, 1L)
bench_shape("early 3x3 (large HW)",    3L, 3L, 64L,  128L, 112L, 112L, 1L)
