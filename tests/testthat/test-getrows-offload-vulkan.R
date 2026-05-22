library(ggmlR)

# Regression test for the GPU x8 decode regression (2026-05-20).
#
# Root cause: ggml_backend_vk_device_offload_op hard-coded `return true`, so the
# scheduler offloaded GET_ROWS (the token-embedding lookup) onto Vulkan even for
# a single-token decode step. That forced the entire token_embd.weight matrix
# (~408MB for a 3B model) to be marshalled host->device on EVERY decode step,
# collapsing throughput from ~116 t/s to ~14 t/s.
#
# Fix (ported from ggml-0.11.0): ggml_vk_get_op_batch_size() reports 0 for
# GET_ROWS, and offload_op only offloads ops whose effective batch size is
# >= op_offload_min_batch_size (default 32). So a batch=1 embedding lookup stays
# off the "force to GPU" path.
#
# This test pins the observable contract rather than the C internals:
#   1. GET_ROWS on a large embedding table at batch=1 is numerically correct
#      on Vulkan and matches the CPU reference.
#   2. The same holds at batch=8 (still below the offload threshold) and at
#      batch=64 (above it) -- correctness must be invariant to which side of
#      the offload threshold the op lands on.

# Embedding lookup: table [n_embd, n_vocab] f32, indices [n_tok] i32 (0-based),
# out = table[:, indices] -> [n_embd, n_tok]. Returns the gathered rows.
run_get_rows <- function(use_gpu, n_embd, n_vocab, idx0) {
  n_tok <- length(idx0)
  set.seed(7)
  tbl <- matrix(rnorm(n_embd * n_vocab, sd = 0.1), nrow = n_embd, ncol = n_vocab)

  ctx <- ggml_init(512L * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  emb <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab)
  idx <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok)
  out <- ggml_get_rows(ctx, emb, idx)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(emb, as.vector(tbl))
  # I32 indices must go through the backend setter (ggml_set_i32 writes the
  # CPU data pointer directly, which is NULL for a Vulkan-allocated tensor).
  ggml_backend_tensor_set_data(idx, as.integer(idx0))

  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  res <- ggml_backend_tensor_get_data(out)
  ggml_free(ctx)
  list(res = res, ref = as.vector(tbl[, idx0 + 1L, drop = FALSE]))
}

expect_get_rows_correct <- function(n_embd, n_vocab, idx0) {
  out <- run_get_rows(TRUE, n_embd, n_vocab, idx0)
  md  <- max(abs(out$res - out$ref))
  info <- sprintf("get_rows n_embd=%d n_vocab=%d batch=%d max|diff|=%.6g",
                  n_embd, n_vocab, length(idx0), md)
  expect_lt(md, 1e-4, label = info)
}

test_that("get_rows Vulkan == CPU across offload-threshold batch sizes", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")
  # n_vocab large enough that the embedding table is the dominant tensor,
  # mimicking the token_embd.weight that triggered the per-step copy.
  n_embd  <- 2048L
  n_vocab <- 32000L
  set.seed(11)
  # batch=1 (decode), batch=8 (small prompt), batch=64 (above min_batch_size).
  expect_get_rows_correct(n_embd, n_vocab, sample.int(n_vocab, 1L)  - 1L)
  expect_get_rows_correct(n_embd, n_vocab, sample.int(n_vocab, 8L)  - 1L)
  expect_get_rows_correct(n_embd, n_vocab, sample.int(n_vocab, 64L) - 1L)
})
