# Vulkan TOPK_MOE fusion (TODO 9.5.5-check): the MoE router subgraphs, built in
# the node order llama.cpp's build_moe_ffn produces, so the backend's pattern
# matcher fuses them into one topk_moe shader. Checked against the CPU, which
# runs the ops one by one: expert ids must be identical, weights within f32
# rounding. The per-mode fusion counter proves the fused path ran -- without
# it a silently unfused graph would pass the comparison too.
#
# Patterns (ggml-vulkan-device.cpp):
#   early_softmax       SOFT_MAX RESHAPE ARGSORT VIEW GET_ROWS
#   early_softmax_norm  ... + RESHAPE SUM_ROWS CLAMP DIV RESHAPE
#   late_softmax        ARGSORT VIEW GET_ROWS RESHAPE SOFT_MAX RESHAPE
#   sigmoid_norm_bias   UNARY(sigmoid) RESHAPE ADD(bias) ARGSORT VIEW GET_ROWS
#                       RESHAPE SUM_ROWS CLAMP DIV RESHAPE
# Expert ids are compared through the full ARGSORT buffer: the top-k VIEW is
# strided (k of n_expert per row), so only its first k entries per row count.

topk_moe_count <- function(mode) .Call("R_ggml_vk_topk_moe_fusion_count", as.integer(mode), PACKAGE = "ggmlR")
TOPK_MOE_MODE <- c(early_softmax = 0L, early_softmax_norm = 1L, late_softmax = 2L, sigmoid_norm_bias = 3L)

# ggml_argsort_top_k(): ARGSORT (desc) + VIEW of the first k ids per row
argsort_top_k <- function(ctx, a, k, n_expert, n_tokens) {
  srt <- ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC)
  nb1 <- 4 * n_expert
  sel <- ggml_view_4d(ctx, srt, k, n_tokens, 1, 1, nb1, nb1 * n_tokens, nb1 * n_tokens, 0)
  list(srt = srt, sel = sel)
}

# weights [1, k, T] -> sum-normalised, as llama.cpp's norm_w branch
norm_weights <- function(ctx, w, k, n_tokens) {
  w2 <- ggml_reshape_2d(ctx, w, k, n_tokens)
  s  <- ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5, Inf)
  ggml_reshape_3d(ctx, ggml_div(ctx, w2, s), 1, k, n_tokens)
}

topk_moe_builders <- list(
  early_softmax = function(ctx, logits, bias, k, E, T) {
    probs <- ggml_soft_max(ctx, logits)
    p3 <- ggml_reshape_3d(ctx, probs, 1, E, T)
    tk <- argsort_top_k(ctx, probs, k, E, T)
    list(srt = tk$srt, w = ggml_get_rows(ctx, p3, tk$sel))
  },
  early_softmax_norm = function(ctx, logits, bias, k, E, T) {
    probs <- ggml_soft_max(ctx, logits)
    p3 <- ggml_reshape_3d(ctx, probs, 1, E, T)
    tk <- argsort_top_k(ctx, probs, k, E, T)
    list(srt = tk$srt, w = norm_weights(ctx, ggml_get_rows(ctx, p3, tk$sel), k, T))
  },
  late_softmax = function(ctx, logits, bias, k, E, T) {
    l3 <- ggml_reshape_3d(ctx, logits, 1, E, T)
    tk <- argsort_top_k(ctx, logits, k, E, T)
    w  <- ggml_reshape_2d(ctx, ggml_get_rows(ctx, l3, tk$sel), k, T)
    list(srt = tk$srt, w = ggml_reshape_3d(ctx, ggml_soft_max(ctx, w), 1, k, T))
  },
  sigmoid_norm_bias = function(ctx, logits, bias, k, E, T) {
    probs <- ggml_sigmoid(ctx, logits)
    p3 <- ggml_reshape_3d(ctx, probs, 1, E, T)
    tk <- argsort_top_k(ctx, ggml_add(ctx, probs, bias), k, E, T)
    list(srt = tk$srt, w = norm_weights(ctx, ggml_get_rows(ctx, p3, tk$sel), k, T))
  }
)

run_topk_moe <- function(backend, build, logits_data, bias_data, k, E, T) {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  logits <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T)
  bias   <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, E)
  out <- build(ctx, logits, bias, k, E, T)
  gf <- ggml_build_forward_expand(ctx, out$w)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(logits, logits_data)
  ggml_backend_tensor_set_data(bias, bias_data)
  ggml_backend_graph_compute(backend, gf)
  ids <- matrix(ggml_backend_tensor_get_data(out$srt), nrow = E)[seq_len(k), , drop = FALSE]
  list(ids = ids, w = ggml_backend_tensor_get_data(out$w))
}

# (n_expert, k, n_tokens): small/odd token counts, k = 1, and the 128/256
# expert widths of current MoE models
topk_moe_shapes <- list(c(8, 2, 7), c(64, 8, 33), c(128, 8, 5), c(256, 8, 3), c(32, 1, 4))

test_that("Vulkan TOPK_MOE fuses and matches the CPU", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(gpu))
  cpu <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(cpu), add = TRUE)
  ggml_backend_cpu_set_n_threads(cpu, 2L)
  set.seed(97)
  for (mode in names(topk_moe_builders)) {
    for (sh in topk_moe_shapes) {
      E <- sh[1]; k <- sh[2]; T <- sh[3]
      label <- sprintf("%s E=%d k=%d T=%d", mode, E, k, T)
      # continuous random logits: no ties, so argsort tie-breaking is not tested here
      logits <- rnorm(E * T)
      bias   <- rnorm(E, sd = 0.5)
      before <- topk_moe_count(TOPK_MOE_MODE[[mode]])
      got  <- run_topk_moe(gpu, topk_moe_builders[[mode]], logits, bias, k, E, T)
      fused <- topk_moe_count(TOPK_MOE_MODE[[mode]]) - before
      want <- run_topk_moe(cpu, topk_moe_builders[[mode]], logits, bias, k, E, T)
      expect_gt(fused, 0, label = paste("fusion count,", label))
      expect_identical(as.integer(got$ids), as.integer(want$ids), label = paste("expert ids,", label))
      expect_equal(got$w, want$w, tolerance = 1e-5, label = paste("weights,", label))
    }
  }
})
