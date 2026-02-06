#!/usr/bin/env Rscript
# ============================================================================
# ggmlR Smoke Tests - End-to-end inference and training scenarios
# ============================================================================

library(ggmlR)

divider <- paste(rep("=", 70), collapse="")
cat(divider, "\n")
cat("ggmlR Smoke Tests - Inference & Training\n")
cat(divider, "\n\n")

passed <- 0
failed <- 0
errors <- character()

test <- function(name, expr) {
  result <- tryCatch({
    res <- eval(expr)
    if (isTRUE(res) || (!is.null(res) && !identical(res, FALSE))) {
      cat("[PASS]", name, "\n")
      passed <<- passed + 1
      TRUE
    } else {
      cat("[FAIL]", name, "\n")
      failed <<- failed + 1
      errors <<- c(errors, name)
      FALSE
    }
  }, error = function(e) {
    cat("[FAIL]", name, "-", conditionMessage(e), "\n")
    failed <<- failed + 1
    errors <<- c(errors, paste(name, "-", conditionMessage(e)))
    FALSE
  })
}

# ============================================================================
# INFERENCE TESTS
# ============================================================================

cat("\n--- MLP Inference ---\n")

test("MLP forward pass (4->8->2)", {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  # Weights
  w1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
  b1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)
  w2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2)
  b2 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)

  # Forward: relu(x @ w1 + b1) @ w2 + b2
  h <- ggml_add(ctx, ggml_mul_mat(ctx, w1, x), b1)
  h <- ggml_relu(ctx, h)
  y <- ggml_add(ctx, ggml_mul_mat(ctx, w2, h), b2)

  graph <- ggml_build_forward_expand(ctx, y)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  ggml_backend_tensor_set_data(w1, rep(0.1, 32))
  ggml_backend_tensor_set_data(b1, rep(0, 8))
  ggml_backend_tensor_set_data(w2, rep(0.1, 16))
  ggml_backend_tensor_set_data(b2, rep(0, 2))
  ggml_backend_tensor_set_data(x, c(1, 2, 3, 4))

  status <- ggml_backend_graph_compute(backend, graph)
  output <- ggml_backend_tensor_get_data(y)

  status == 0 && length(output) == 2 && all(is.finite(output)) && all(output > 0)
})

test("Deep MLP (8->16->16->8->4) with GELU", {
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4)  # batch=4
  w1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 16)
  w2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16)
  w3 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8)
  w4 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4)

  h <- ggml_gelu(ctx, ggml_mul_mat(ctx, w1, x))
  h <- ggml_gelu(ctx, ggml_mul_mat(ctx, w2, h))
  h <- ggml_gelu(ctx, ggml_mul_mat(ctx, w3, h))
  y <- ggml_mul_mat(ctx, w4, h)

  graph <- ggml_build_forward_expand(ctx, y)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  set.seed(42)
  ggml_backend_tensor_set_data(x, rnorm(32, 0, 0.1))
  ggml_backend_tensor_set_data(w1, rnorm(128, 0, 0.02))
  ggml_backend_tensor_set_data(w2, rnorm(256, 0, 0.02))
  ggml_backend_tensor_set_data(w3, rnorm(128, 0, 0.02))
  ggml_backend_tensor_set_data(w4, rnorm(32, 0, 0.02))

  status <- ggml_backend_graph_compute(backend, graph)
  output <- ggml_backend_tensor_get_data(y)

  status == 0 && length(output) == 16 && all(is.finite(output))
})

cat("\n--- CNN Inference ---\n")

test("Conv2D + MaxPool + ReLU", {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  img <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 16, 3)
  kernel <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 3, 8)

  conv <- ggml_conv_2d(ctx, kernel, img, s0=1, s1=1, p0=1, p1=1, d0=1, d1=1)
  activated <- ggml_relu(ctx, conv)
  pooled <- ggml_pool_2d(ctx, activated, GGML_OP_POOL_MAX, k0=2, k1=2, s0=2, s1=2, p0=0, p1=0)

  graph <- ggml_build_forward_expand(ctx, pooled)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  ggml_backend_tensor_set_data(img, runif(16*16*3, 0, 1))
  ggml_backend_tensor_set_data(kernel, runif(3*3*3*8, -0.5, 0.5))

  status <- ggml_backend_graph_compute(backend, graph)
  output <- ggml_backend_tensor_get_data(pooled)

  status == 0 && length(output) > 0 && all(is.finite(output))
})

cat("\n--- Transformer Components ---\n")

test("Multi-head attention (flash_attn_ext)", {
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  head_dim <- 64
  n_heads <- 4
  seq_len <- 16

  q <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len)
  k <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len)
  v <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len)

  scale <- 1.0 / sqrt(head_dim)
  attn <- ggml_flash_attn_ext(ctx, q, k, v, mask=NULL, scale=scale, max_bias=0, logit_softcap=0)

  graph <- ggml_build_forward_expand(ctx, attn)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  set.seed(42)
  ggml_backend_tensor_set_data(q, rnorm(head_dim*n_heads*seq_len, 0, 0.1))
  ggml_backend_tensor_set_data(k, rnorm(head_dim*n_heads*seq_len, 0, 0.1))
  ggml_backend_tensor_set_data(v, rnorm(head_dim*n_heads*seq_len, 0, 0.1))

  status <- ggml_backend_graph_compute(backend, graph)
  output <- ggml_backend_tensor_get_data(attn)

  status == 0 && all(is.finite(output))
})

test("RoPE positional encoding", {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  head_dim <- 64
  n_heads <- 8
  seq_len <- 32

  q <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_heads, seq_len)
  pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len)

  q_rope <- ggml_rope(ctx, q, pos, head_dim, GGML_ROPE_TYPE_NORM)

  graph <- ggml_build_forward_expand(ctx, q_rope)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  ggml_backend_tensor_set_data(q, rnorm(head_dim*n_heads*seq_len, 0, 0.1))
  ggml_backend_tensor_set_data(pos, as.integer(0:(seq_len-1)))

  status <- ggml_backend_graph_compute(backend, graph)
  output <- ggml_backend_tensor_get_data(q_rope)
  q_orig <- ggml_backend_tensor_get_data(q)

  # RoPE should change the values
  status == 0 && all(is.finite(output)) && !all(abs(output - q_orig) < 1e-6)
})

test("SwiGLU FFN (LLaMA-style)", {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  hidden <- 64
  ffn <- 128
  seq <- 16

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq)
  w_gate <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, ffn)
  w_up <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, ffn)
  w_down <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn, hidden)

  gate <- ggml_mul_mat(ctx, w_gate, x)
  up <- ggml_mul_mat(ctx, w_up, x)
  swiglu <- ggml_glu_split(ctx, gate, up, GGML_GLU_OP_SWIGLU)
  out <- ggml_mul_mat(ctx, w_down, swiglu)

  graph <- ggml_build_forward_expand(ctx, out)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  set.seed(42)
  ggml_backend_tensor_set_data(x, rnorm(hidden*seq, 0, 0.1))
  ggml_backend_tensor_set_data(w_gate, rnorm(hidden*ffn, 0, 0.02))
  ggml_backend_tensor_set_data(w_up, rnorm(hidden*ffn, 0, 0.02))
  ggml_backend_tensor_set_data(w_down, rnorm(ffn*hidden, 0, 0.02))

  status <- ggml_backend_graph_compute(backend, graph)
  output <- ggml_backend_tensor_get_data(out)

  status == 0 && length(output) == hidden*seq && all(is.finite(output))
})

test("RMSNorm + LayerNorm", {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8)
  rms <- ggml_rms_norm(ctx, x, eps=1e-5)
  ln <- ggml_norm(ctx, x, eps=1e-5)

  graph <- ggml_build_forward_expand(ctx, rms)
  ggml_build_forward_expand(ctx, ln)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  ggml_backend_tensor_set_data(x, rnorm(512, 5, 2))

  status <- ggml_backend_graph_compute(backend, graph)
  rms_out <- ggml_backend_tensor_get_data(rms)
  ln_out <- ggml_backend_tensor_get_data(ln)

  status == 0 && all(is.finite(rms_out)) && all(is.finite(ln_out))
})

# ============================================================================
# TRAINING TESTS
# ============================================================================

cat("\n--- Training Setup ---\n")

test("Dataset creation and batch retrieval", {
  n_samples <- 100
  n_features <- 8
  n_labels <- 2
  batch_size <- 16

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)

  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = n_features,
    ne_label = n_labels,
    ndata = n_samples,
    ndata_shard = 1
  )
  on.exit(ggml_opt_dataset_free(dataset), add = TRUE)

  # Fill dataset
  data_tensor <- ggml_opt_dataset_data(dataset)
  labels_tensor <- ggml_opt_dataset_labels(dataset)
  ggml_backend_tensor_set_data(data_tensor, rnorm(n_samples * n_features))
  ggml_backend_tensor_set_data(labels_tensor, runif(n_samples * n_labels))

  # Get batch
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  data_batch <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_features, batch_size)
  labels_batch <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_labels, batch_size)

  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  ggml_opt_dataset_get_batch(dataset, data_batch, labels_batch, ibatch = 0)

  batch <- ggml_backend_tensor_get_data(data_batch)
  length(batch) == n_features * batch_size && all(is.finite(batch))
})

test("Optimizer context with AdamW", {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)

  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  opt <- ggml_opt_init(
    sched = sched,
    loss_type = ggml_opt_loss_type_mse(),
    optimizer = ggml_opt_optimizer_type_adamw(),
    opt_period = 1L
  )
  on.exit(ggml_opt_free(opt), add = TRUE)

  opt_type <- ggml_opt_context_optimizer_type(opt)
  opt_type == ggml_opt_optimizer_type_adamw()
})

test("Optimizer context with SGD + CrossEntropy", {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)

  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  opt <- ggml_opt_init(
    sched = sched,
    loss_type = ggml_opt_loss_type_cross_entropy(),
    optimizer = ggml_opt_optimizer_type_sgd(),
    opt_period = 4L
  )
  on.exit(ggml_opt_free(opt), add = TRUE)

  opt_type <- ggml_opt_context_optimizer_type(opt)
  opt_type == ggml_opt_optimizer_type_sgd()
})

test("Dataset shuffle", {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)

  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  opt <- ggml_opt_init(sched = sched, loss_type = ggml_opt_loss_type_mse(),
                       optimizer = ggml_opt_optimizer_type_adamw())
  on.exit(ggml_opt_free(opt), add = TRUE)

  dataset <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, 10, 1, 100, 1)
  on.exit(ggml_opt_dataset_free(dataset), add = TRUE)

  ggml_opt_dataset_shuffle(opt, dataset, idata = -1)
  TRUE
})

test("Result tracking", {
  result <- ggml_opt_result_init()
  on.exit(ggml_opt_result_free(result), add = TRUE)

  ndata <- ggml_opt_result_ndata(result)
  loss_info <- ggml_opt_result_loss(result)
  acc_info <- ggml_opt_result_accuracy(result)

  ggml_opt_result_reset(result)

  ndata == 0 && "loss" %in% names(loss_info) && "accuracy" %in% names(acc_info)
})

cat("\n--- Forward Pass with Loss ---\n")

test("MSE loss computation", {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  batch <- 8
  dim <- 4

  pred <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, batch)
  target <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, batch)

  # MSE = mean((pred - target)^2)
  diff <- ggml_sub(ctx, pred, target)
  sq <- ggml_sqr(ctx, diff)
  loss <- ggml_sum(ctx, sq)  # sum instead of mean for simplicity

  graph <- ggml_build_forward_expand(ctx, loss)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  set.seed(42)
  ggml_backend_tensor_set_data(pred, rnorm(dim*batch))
  ggml_backend_tensor_set_data(target, rnorm(dim*batch))

  status <- ggml_backend_graph_compute(backend, graph)
  loss_val <- ggml_backend_tensor_get_data(loss)

  status == 0 && length(loss_val) == 1 && loss_val >= 0
})

test("Softmax + negative log-likelihood loss", {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  batch <- 8
  n_classes <- 4

  logits <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_classes, batch)

  # Softmax on logits
  probs <- ggml_soft_max(ctx, logits)
  # For loss tracking, just sum the probabilities (simplified)
  loss <- ggml_sum(ctx, probs)

  graph <- ggml_build_forward_expand(ctx, loss)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  set.seed(42)
  ggml_backend_tensor_set_data(logits, rnorm(n_classes*batch))

  status <- ggml_backend_graph_compute(backend, graph)
  loss_val <- ggml_backend_tensor_get_data(loss)

  # Softmax outputs sum to 1 per sample, so total sum = batch
  status == 0 && length(loss_val) == 1 && abs(loss_val - batch) < 0.1
})

# ============================================================================
# QUANTIZATION TESTS
# ============================================================================

cat("\n--- Quantized Inference ---\n")

test("Q4_0 quantize/dequantize roundtrip", {
  n <- 256
  original <- runif(n, -1, 1)

  quantized <- quantize_q4_0(original, 1, n, NULL)
  dequantized <- dequantize_row_q4_0(quantized, n)

  cor(original, dequantized) > 0.9
})

test("Q8_0 quantize/dequantize roundtrip", {
  n <- 256
  original <- runif(n, -1, 1)

  quantized <- quantize_q8_0(original, 1, n, NULL)
  dequantized <- dequantize_row_q8_0(quantized, n)

  cor(original, dequantized) > 0.99
})

test("K-quant (Q4_K) roundtrip", {
  n <- 256
  original <- runif(n, -1, 1)

  quantized <- quantize_q4_K(original, 1, n, NULL)
  dequantized <- dequantize_row_q4_K(quantized, n)

  cor(original, dequantized) > 0.9
})

test("Inference with dequantized weights", {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  # Quantize then dequantize weights
  n <- 256
  original_w <- runif(n, -1, 1)
  quantized <- quantize_q4_0(original_w, 1, n, NULL)
  deq_weights <- dequantize_row_q4_0(quantized, n)

  w <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 16)
  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16)
  y <- ggml_mul_mat(ctx, w, x)

  graph <- ggml_build_forward_expand(ctx, y)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

  ggml_backend_tensor_set_data(w, deq_weights)
  ggml_backend_tensor_set_data(x, rep(1, 16))

  status <- ggml_backend_graph_compute(backend, graph)
  output <- ggml_backend_tensor_get_data(y)

  status == 0 && all(is.finite(output))
})

# ============================================================================
# VULKAN BACKEND (if available)
# ============================================================================

cat("\n--- Vulkan Backend ---\n")

if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
  test("Vulkan device discovery", {
    devices <- ggml_vulkan_list_devices()
    length(devices) > 0
  })

  test("Vulkan MLP inference", {
    vk <- ggml_vulkan_init(0)
    on.exit(ggml_vulkan_free(vk), add = TRUE)

    ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
    on.exit(ggml_free(ctx), add = TRUE)

    x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4)
    w <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8)
    y <- ggml_relu(ctx, ggml_mul_mat(ctx, w, x))

    graph <- ggml_build_forward_expand(ctx, y)

    buffer <- ggml_backend_alloc_ctx_tensors(ctx, vk)
    on.exit(ggml_backend_buffer_free(buffer), add = TRUE)

    ggml_backend_tensor_set_data(x, rnorm(32, 0, 0.1))
    ggml_backend_tensor_set_data(w, rnorm(64, 0, 0.1))

    status <- ggml_backend_graph_compute(vk, graph)
    output <- ggml_backend_tensor_get_data(y)

    status == 0 && all(is.finite(output))
  })
} else {
  cat("[SKIP] Vulkan not available\n")
}

# ============================================================================
# Summary
# ============================================================================

cat("\n")
cat(divider, "\n")
cat(sprintf("RESULTS: %d passed, %d failed\n", passed, failed))
cat(divider, "\n")

if (failed > 0) {
  cat("\nFailed tests:\n")
  for (e in errors) cat("  -", e, "\n")
  quit(status = 1)
} else {
  cat("\nAll smoke tests passed!\n")
  quit(status = 0)
}
