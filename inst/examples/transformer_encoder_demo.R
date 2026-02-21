# Transformer Encoder — полный пример
#
# Демонстрирует:
#   - ag_multihead_attention (self-attention + causal mask)
#   - LayerNorm через примитивы autograd (pre-norm схема)
#   - FeedForward блок (Linear → ReLU → Linear)
#   - Positional encoding (синусоидальный, фиксированный)
#   - Mixed precision: f16 параметры на CPU/GPU
#   - optimizer_adam + backward()
#   - Задача: предсказать следующий токен (tiny language model)
#   - 10-шаговый цикл обучения с выводом loss
#
# Тензорный макет: [d_model, seq_len] — столбцы = позиции.
# Словарь: 16 токенов, d_model = 32, 2 головы, 1 энкодер-блок.

library(ggmlR)

cat("ggmlR version:", ggml_version(), "\n")

# =============================================================================
# 0.  Гиперпараметры и настройка устройства
# =============================================================================

set.seed(42L)

VOCAB_SIZE <- 16L
D_MODEL    <- 32L
N_HEADS    <- 4L
D_FF       <- 64L   # FeedForward hidden dim
SEQ_LEN    <- 8L
N_SEQS     <- 8L    # фиксированный датасет (memorization)
N_ITER     <- 50L
LR         <- 5e-4
CLIP_NORM  <- 1.0   # gradient clipping

# Выбор устройства: GPU если доступен, иначе CPU
device <- tryCatch({
  ag_device("gpu")
  cat("Device: GPU\n")
  "gpu"
}, error = function(e) {
  cat("Device: CPU\n")
  "cpu"
})

# Mixed precision: f16 на GPU, f32 на CPU (bf16 опционально для Vulkan)
if (device == "gpu") {
  ag_dtype("f16")
  cat("Dtype:  f16 (mixed precision)\n")
} else {
  ag_dtype("f32")
  cat("Dtype:  f32\n")
}
cat("\n")

# =============================================================================
# 1.  Синтетический датасет: фиксированные последовательности (memorization)
#     Цель: предсказать каждый следующий токен (causal LM).
#     Фиксируем данные — модель должна выучить их наизусть.
# =============================================================================

train_seqs <- lapply(seq_len(N_SEQS), function(i)
  sample.int(VOCAB_SIZE, SEQ_LEN, replace = TRUE) - 1L   # 0-based
)

# =============================================================================
# 2.  Embedding + Positional Encoding
# =============================================================================

emb <- ag_embedding(VOCAB_SIZE, D_MODEL)

# Синусоидальный PE — фиксированный (не обучается)
make_pe <- function(d_model, seq_len) {
  pe <- matrix(0.0, d_model, seq_len)
  for (pos in seq_len(seq_len)) {
    for (i in seq(1, d_model, by = 2)) {
      denom      <- 10000^((i - 1) / d_model)
      pe[i, pos] <- sin((pos - 1) / denom)
      if (i + 1 <= d_model)
        pe[i + 1, pos] <- cos((pos - 1) / denom)
    }
  }
  pe
}

PE <- make_pe(D_MODEL, SEQ_LEN)   # [D_MODEL, SEQ_LEN] — зафиксировано

# =============================================================================
# 3.  LayerNorm через autograd-примитивы
#     Normalise over d_model (axis = 1, column-wise).
#     gamma / beta — trainable vectors [d_model, 1], broadcast по seq_len.
# =============================================================================

ag_layer_norm_custom <- function(d_model, eps = 1e-5) {
  env <- new.env(parent = emptyenv())
  env$gamma   <- ag_param(matrix(1.0, d_model, 1))
  env$beta    <- ag_param(matrix(0.0, d_model, 1))
  env$d_model <- as.integer(d_model)
  env$eps     <- eps

  env$forward <- function(x) {
    # x: [d_model, seq_len]
    # Normalize over d_model (first dim) for each position:
    # colMeans = mean over rows = dim=2 in ggmlR convention → [1, seq_len]
    mu      <- ag_mean(x, dim = 2L, keepdim = TRUE)    # [1, seq_len]
    xc      <- ag_sub(x, mu)                           # [d_model, seq_len]
    var     <- ag_mean(ag_pow(xc, 2.0), dim = 2L, keepdim = TRUE)  # [1, seq_len]
    inv_std <- ag_pow(ag_add(var, ag_tensor(matrix(env$eps, 1L, 1L))), -0.5)
    xn      <- ag_mul(xc, inv_std)                     # [d_model, seq_len]
    # affine: gamma[d,1] * xn[d,s] + beta[d,1]
    # ggml_mul requires larger tensor first
    ag_add(ag_mul(xn, env$gamma), env$beta)
  }

  env$parameters <- function() list(gamma = env$gamma, beta = env$beta)
  class(env) <- c("ag_layer_norm_custom", "ag_layer")
  env
}

ag_train.ag_layer_norm_custom <- function(model) { model$training <- TRUE; invisible(model) }
ag_eval.ag_layer_norm_custom  <- function(model) { model$training <- FALSE; invisible(model) }

# =============================================================================
# 4.  FeedForward блок: Linear(D_MODEL→D_FF) → ReLU → Linear(D_FF→D_MODEL)
#     ag_linear возвращает list с $forward и $params.
# =============================================================================

ff_block <- function(d_model, d_ff) {
  env <- new.env(parent = emptyenv())
  lim1 <- sqrt(2.0 / d_model)
  lim2 <- sqrt(2.0 / d_ff)
  env$W1 <- ag_param(matrix(runif(d_ff * d_model, -lim1, lim1), d_ff, d_model))
  env$b1 <- ag_param(matrix(0.0, d_ff, 1L))
  env$W2 <- ag_param(matrix(runif(d_model * d_ff, -lim2, lim2), d_model, d_ff))
  env$b2 <- ag_param(matrix(0.0, d_model, 1L))

  env$forward <- function(x) {
    # x: [d_model, seq_len]
    h <- ag_add(ag_matmul(env$W1, x), env$b1)   # [d_ff, seq_len]
    h <- ag_relu(h)
    ag_add(ag_matmul(env$W2, h), env$b2)         # [d_model, seq_len]
  }

  env$parameters <- function()
    list(W1 = env$W1, b1 = env$b1, W2 = env$W2, b2 = env$b2)

  class(env) <- c("ff_block", "ag_layer")
  env
}

# =============================================================================
# 5.  Transformer Encoder Block
#     Pre-Norm схема: LN → MHA → residual → LN → FF → residual
# =============================================================================

transformer_encoder_block <- function(d_model, n_heads, d_ff, dropout = 0.0) {
  env <- new.env(parent = emptyenv())
  env$ln1  <- ag_layer_norm_custom(d_model)
  env$mha  <- ag_multihead_attention(d_model, n_heads, dropout = dropout)
  env$ln2  <- ag_layer_norm_custom(d_model)
  env$ff   <- ff_block(d_model, d_ff)
  env$drop <- ag_dropout(dropout)

  env$forward <- function(x, causal_mask = FALSE) {
    # Self-attention sub-block (pre-norm)
    xn   <- env$ln1$forward(x)
    attn <- env$mha$forward(xn, causal_mask = causal_mask)
    x    <- ag_add(x, attn)                     # residual

    # FeedForward sub-block (pre-norm)
    xn   <- env$ln2$forward(x)
    ff   <- env$drop$forward(env$ff$forward(xn))
    ag_add(x, ff)                               # residual
  }

  env$parameters <- function() {
    c(env$ln1$parameters(),
      env$mha$parameters(),
      env$ln2$parameters(),
      env$ff$parameters())
  }

  class(env) <- c("transformer_encoder_block", "ag_layer")
  env
}

ag_train.transformer_encoder_block <- function(model) {
  ag_train(model$mha); ag_train(model$drop)
  model$training <- TRUE; invisible(model)
}
ag_eval.transformer_encoder_block <- function(model) {
  ag_eval(model$mha); ag_eval(model$drop)
  model$training <- FALSE; invisible(model)
}

# =============================================================================
# 6.  Tiny LM: Embedding → PE → Encoder → Linear projection → Softmax
#     Параметры проекции head: [vocab_size, d_model]
# =============================================================================

tinyLM <- new.env(parent = emptyenv())
tinyLM$emb     <- emb
tinyLM$encoder <- transformer_encoder_block(D_MODEL, N_HEADS, D_FF, dropout = 0.0)
lim_head        <- sqrt(2.0 / D_MODEL)
tinyLM$W_head  <- ag_param(
  matrix(runif(VOCAB_SIZE * D_MODEL, -lim_head, lim_head), VOCAB_SIZE, D_MODEL)
)

tinyLM$forward <- function(tokens, causal_mask = TRUE) {
  # tokens: integer vector length SEQ_LEN (0-based)
  x <- tinyLM$emb$forward(tokens)                   # [D_MODEL, SEQ_LEN]
  x <- ag_add(x, ag_tensor(PE))                      # + positional encoding
  x <- tinyLM$encoder$forward(x, causal_mask = causal_mask)
  ag_matmul(tinyLM$W_head, x)                        # [VOCAB_SIZE, SEQ_LEN]
}

tinyLM$parameters <- function() {
  c(list(W_head = tinyLM$W_head),
    tinyLM$emb$parameters(),
    tinyLM$encoder$parameters())
}

# =============================================================================
# 7.  Подсчёт параметров
# =============================================================================

all_params  <- tinyLM$parameters()
n_params    <- sum(sapply(all_params, function(p) length(p$data)))
cat(sprintf("Model parameters: %d  (%d tensors)\n", n_params, length(all_params)))

# =============================================================================
# 8.  Оптимизатор
# =============================================================================

opt <- optimizer_adam(all_params, lr = LR)

# =============================================================================
# 9.  Цикл обучения
#     Одна последовательность за шаг; фиксированный датасет → memorization.
#     Цель: tokens[2..SEQ_LEN] при входе tokens[1..SEQ_LEN] с causal mask.
# =============================================================================

cat("\nTraining loop (causal LM, cross-entropy):\n")
cat(sprintf("  %-6s  %-10s\n", "iter", "loss"))
cat(strrep("-", 22), "\n")

ag_train(tinyLM$encoder)

loss_history <- numeric(N_ITER)

for (iter in seq_len(N_ITER)) {

  epoch_loss <- 0.0

  for (tokens in train_seqs) {
    # Вход: tokens[1..SEQ_LEN]; targets: tokens[2..SEQ_LEN] + 0 на позиции SEQ_LEN
    # Используем все SEQ_LEN колонок logits, targets тоже длиной SEQ_LEN
    targets <- c(tokens[-1L], 0L)   # длина SEQ_LEN

    with_grad_tape({
      logits <- tinyLM$forward(tokens, causal_mask = TRUE)
      # logits: [VOCAB_SIZE, SEQ_LEN], targets: length SEQ_LEN
      loss <- ag_softmax_cross_entropy_loss(logits, targets)
    })
    grads <- backward(loss)

    # Gradient clipping: scale all grads if global norm exceeds CLIP_NORM
    param_ids <- sapply(all_params, function(p) as.character(p$id))
    gs <- lapply(param_ids, function(id) get0(id, envir = grads))
    global_norm <- sqrt(sum(sapply(gs, function(g) if (is.null(g)) 0 else sum(g^2))))
    if (!is.nan(global_norm) && global_norm > CLIP_NORM) {
      scale_factor <- CLIP_NORM / global_norm
      for (id in param_ids) {
        g <- get0(id, envir = grads)
        if (!is.null(g)) assign(id, g * scale_factor, envir = grads)
      }
    }

    opt$step(grads)
    opt$zero_grad()
    lv <- as.numeric(ggmlR:::.ag_data(loss))
    if (!is.nan(lv)) epoch_loss <- epoch_loss + lv
  }

  loss_history[iter] <- epoch_loss / N_SEQS

  if (iter %% 10 == 0 || iter == 1L)
    cat(sprintf("  %-6d  %.6f\n", iter, loss_history[iter]))
}

# =============================================================================
# 11.  Итоги
# =============================================================================

cat(strrep("-", 22), "\n")
cat(sprintf("Loss: %.4f → %.4f  (delta = %.4f)\n",
            loss_history[1L],
            loss_history[N_ITER],
            loss_history[1L] - loss_history[N_ITER]))

if (!is.nan(loss_history[N_ITER]) && loss_history[N_ITER] < loss_history[1L]) {
  cat("Training converged.\n")
} else {
  cat("Warning: loss did not decrease. Check LR or increase N_ITER.\n")
}

# =============================================================================
# 12.  Inference demo (eval mode, no dropout, no causal mask for full context)
# =============================================================================

ag_eval(tinyLM$encoder)

cat("\nInference demo (greedy next-token prediction):\n")
prompt_tokens <- sample.int(VOCAB_SIZE, 4L, replace = TRUE) - 1L
cat("  Prompt:    [", paste(prompt_tokens, collapse = " "), "]\n")

logits_inf <- tinyLM$forward(c(prompt_tokens, rep(0L, SEQ_LEN - 4L)),
                              causal_mask = FALSE)
logits_mat_inf <- ggmlR:::.ag_data(logits_inf)  # [VOCAB_SIZE, SEQ_LEN]
predicted_tokens <- apply(logits_mat_inf, 2, which.max) - 1L
cat("  Predicted: [", paste(predicted_tokens, collapse = " "), "]\n")

cat("\nDone.\n")
