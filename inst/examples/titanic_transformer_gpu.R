# =============================================================================
# Titanic survival prediction — a tabular transformer on the GPU (ggmlR 0.8.4)
# =============================================================================
#
# Kaggle: https://www.kaggle.com/c/titanic
#
# Demonstrates the features added in 0.8.4:
#
#   * ggml_layer_attention()                — multi-head attention; all heads
#                                             run in one batched pass
#   * ggml_layer_dense(time_distributed=)   — one kernel per position, keeping
#                                             the sequence axis
#   * loss = "binary_crossentropy"          — the natural loss for a binary
#                                             target (model ends in a sigmoid)
#   * multi-output + loss_weights           — an auxiliary head as a
#                                             regularizer, with per-head history
#   * causal = TRUE                         — compared against full attention
#   * ggml_ssm_conv() / ggml_ssm_scan()     — a Mamba block (variant D), built
#                                             from raw tensors and trained by
#                                             its own loop; see the caveats there
#
# GPU-only: every model trains with backend = "vulkan". If no Vulkan device is
# present the script stops rather than silently falling back to the CPU —
# otherwise the timing comparison would be meaningless.
#
# Model idea (TabTransformer / FT-Transformer):
# a table row is not a sequence, but projecting EACH feature into a shared
# d_model space turns the row into a "sequence of features" of length
# n_features. Attention then learns interactions between features (Sex x
# Pclass, say) instead of leaving them for a dense layer to discover.
#
#   x: c(n_features, 1)
#     |> dense(d_model, time_distributed = TRUE)   -> c(n_features, d_model)
#     |> attention(d_model, n_heads)               -> c(n_features, d_model)
#     |> dense(d_model, time_distributed = TRUE)   -> c(n_features, d_model)
#     |> flatten() |> dense(1, "sigmoid")          -> c(1)
#
# Shapes with the 10 features below: c(10,1) -> c(10,32) -> c(320) -> c(1).
#
# =============================================================================

library(ggmlR)

set.seed(42)

DATA_DIR <- "/mnt/Data2/DS_Data/titanic"

# ---- GPU is mandatory -------------------------------------------------------

if (!ggml_vulkan_available() || ggml_vulkan_device_count() == 0L) {
  stop("This example is GPU-only: no Vulkan device found.\n",
       "Check ggml_vulkan_status().")
}
cat("GPU:", ggml_vulkan_device_description(0L), "\n\n")

# =============================================================================
# 1. Data and feature engineering
# =============================================================================

train_data <- read.csv(file.path(DATA_DIR, "train.csv"), stringsAsFactors = FALSE)
test_data  <- read.csv(file.path(DATA_DIR, "test.csv"),  stringsAsFactors = FALSE)

prep_features <- function(df, ref = df) {
  df$Age[is.na(df$Age)]   <- median(ref$Age,  na.rm = TRUE)
  df$Fare[is.na(df$Fare)] <- median(ref$Fare, na.rm = TRUE)
  df$Embarked[df$Embarked == "" | is.na(df$Embarked)] <- "S"

  # The title carries sex, age band and social status at once — a strong
  # feature that none of the raw columns provides on its own.
  title <- gsub(".*,\\s*(\\w+)\\..*", "\\1", df$Name)
  df$Title <- ifelse(title == "Mr",                  "Mr",
              ifelse(title %in% c("Mrs","Mme","Ms"), "Mrs",
              ifelse(title %in% c("Miss","Mlle"),    "Miss",
              ifelse(title == "Master",              "Master", "Rare"))))

  df$FamilySize <- df$SibSp + df$Parch + 1L
  df$IsAlone    <- as.integer(df$FamilySize == 1L)
  df$Sex        <- as.integer(df$Sex == "male")
  df$Embarked   <- as.integer(factor(df$Embarked, levels = c("S","C","Q"))) - 1L
  df$TitleIdx   <- as.integer(factor(df$Title,
                     levels = c("Mr","Mrs","Miss","Master","Rare"))) - 1L

  df[, c("Pclass","Sex","Age","SibSp","Parch","Fare",
         "Embarked","FamilySize","IsAlone","TitleIdx")]
}

x_raw      <- prep_features(train_data)
x_test_raw <- prep_features(test_data, ref = train_data)

# Scale with the TRAINING statistics only — using the test set's own mean and
# sd would leak information from it into the model.
x_scaled <- scale(x_raw)
ctr <- attr(x_scaled, "scaled:center")
scl <- attr(x_scaled, "scaled:scale")

x_all  <- matrix(as.numeric(x_scaled), nrow = nrow(x_raw))
x_test <- scale(as.matrix(x_test_raw), center = ctr, scale = scl)
x_test <- matrix(as.numeric(x_test), nrow = nrow(x_test_raw))

y_surv <- matrix(as.numeric(train_data$Survived), ncol = 1L)

# Second target for the multi-output variant: ticket class, one-hot over 3.
onehot <- function(idx0, n) {
  m <- matrix(0, nrow = length(idx0), ncol = n)
  m[cbind(seq_along(idx0), idx0 + 1L)] <- 1
  m
}
y_pclass <- onehot(x_raw$Pclass - 1L, 3L)

N_FEATURES <- ncol(x_all)
cat(sprintf("Training rows: %d, features: %d\n", nrow(x_all), N_FEATURES))

# Hold-out split so the variants are compared on data none of them trained on.
idx     <- sample(nrow(x_all))
n_val   <- as.integer(0.2 * nrow(x_all))
val_i   <- idx[seq_len(n_val)]
train_i <- idx[-seq_len(n_val)]

x_tr <- x_all[train_i, , drop = FALSE]; y_tr <- y_surv[train_i, , drop = FALSE]
x_va <- x_all[val_i,  , drop = FALSE]; y_va <- y_surv[val_i,  , drop = FALSE]
p_tr <- y_pclass[train_i, , drop = FALSE]
p_va <- y_pclass[val_i,  , drop = FALSE]

# Each feature becomes its own sequence position: c(n_features, 1).
as_seq <- function(m) array(as.numeric(m), dim = c(nrow(m), ncol(m), 1L))

x_tr_seq   <- as_seq(x_tr)
x_va_seq   <- as_seq(x_va)
x_test_seq <- as_seq(x_test)

# =============================================================================
# 2. Shared hyperparameters
# =============================================================================

D_MODEL <- 32L
N_HEADS <- 4L
EPOCHS  <- 600L
BATCH   <- 32L
DROPOUT <- 0.3

# 891 rows is a small dataset, so the models overfit long before EPOCHS is
# reached: training loss keeps falling while validation loss turns back up.
# Early stopping ends each run once val_loss has not improved for PATIENCE
# epochs, which is what makes a large EPOCHS budget safe rather than harmful.
# Note it stops at the plateau but does NOT roll the weights back to the best
# epoch, so a little overfitting past the optimum still remains.
PATIENCE <- 20L

early_stop <- function() {
  list(ggml_callback_early_stopping(monitor = "val_loss", patience = PATIENCE))
}

accuracy <- function(prob, truth) mean((prob > 0.5) == (truth > 0.5))

# Epochs actually run, for the summary table -- early stopping makes this
# differ between variants, and a run that stopped early is the interesting one.
epochs_run <- function(model) length(model$history$train_loss)

# One transformer block: attention + residual, then an FFN + residual. Both
# branches are time_distributed, so the feature axis survives for the next block.
transformer_block <- function(h, causal = FALSE) {
  attn <- h |> ggml_layer_attention(d_model = D_MODEL, n_heads = N_HEADS,
                                    causal = causal)
  h1   <- ggml_layer_add(list(h, attn))

  ff   <- h1 |> ggml_layer_dense(D_MODEL, activation = "relu",
                                 time_distributed = TRUE)
  ggml_layer_add(list(h1, ff))
}

results <- list()

# =============================================================================
# 3. Variant A — transformer over features, binary_crossentropy
# =============================================================================

cat("\n=== A: attention + time_distributed dense, binary_crossentropy ===\n")

inp_a <- ggml_input(shape = c(N_FEATURES, 1L), name = "features")

# Project each feature's scalar into d_model — a "feature embedding".
h_a   <- inp_a |> ggml_layer_dense(D_MODEL, time_distributed = TRUE,
                                   name = "feature_embed")
h_a   <- transformer_block(h_a)
out_a <- h_a |>
  ggml_layer_flatten() |>
  ggml_layer_dense(32L, activation = "relu") |>
  ggml_layer_dropout(rate = DROPOUT) |>
  ggml_layer_dense(1L, activation = "sigmoid", name = "survived")

model_a <- ggml_model(inputs = inp_a, outputs = out_a)

model_a <- ggml_compile(model_a,
                        optimizer = "adam",
                        loss      = "binary_crossentropy",
                        backend   = "vulkan")

t_a <- system.time(
  model_a <- ggml_fit(model_a, x_tr_seq, y_tr,
                      epochs     = EPOCHS,
                      batch_size = BATCH,
                      validation_data = list(x_va_seq, y_va),
                      callbacks  = early_stop(),
                      verbose    = 1L)
)

prob_a <- ggml_predict(model_a, x_va_seq, batch_size = BATCH)
results$A <- list(acc = accuracy(prob_a[, 1], y_va[, 1]),
                  sec = as.numeric(t_a["elapsed"]),
                  ep  = epochs_run(model_a),
                  model = model_a)
cat(sprintf("A: val accuracy %.4f  (%.1f s, %d epochs)\n",
            results$A$acc, results$A$sec, results$A$ep))

# =============================================================================
# 4. Variant B — causal attention, for comparison
# =============================================================================
#
# NOTE: a causal mask stops each position from attending to later ones. That is
# required for text (a token must not see its own future), but the COLUMN order
# of a table is arbitrary — "Pclass comes before Sex" means nothing. Here the
# mask only removes information, so B is expected to be no better than A. It is
# in this example to show the flag and its real effect, not to recommend it for
# tabular data.

cat("\n=== B: same model with causal = TRUE (expected: no better) ===\n")

inp_b <- ggml_input(shape = c(N_FEATURES, 1L), name = "features")
h_b   <- inp_b |> ggml_layer_dense(D_MODEL, time_distributed = TRUE)
h_b   <- transformer_block(h_b, causal = TRUE)
out_b <- h_b |>
  ggml_layer_flatten() |>
  ggml_layer_dense(32L, activation = "relu") |>
  ggml_layer_dropout(rate = DROPOUT) |>
  ggml_layer_dense(1L, activation = "sigmoid", name = "survived")

model_b <- ggml_compile(ggml_model(inputs = inp_b, outputs = out_b),
                        optimizer = "adam",
                        loss      = "binary_crossentropy",
                        backend   = "vulkan")

t_b <- system.time(
  model_b <- ggml_fit(model_b, x_tr_seq, y_tr,
                      epochs = EPOCHS, batch_size = BATCH,
                      validation_data = list(x_va_seq, y_va),
                      callbacks = early_stop(),
                      verbose = 0L)
)

prob_b <- ggml_predict(model_b, x_va_seq, batch_size = BATCH)
results$B <- list(acc = accuracy(prob_b[, 1], y_va[, 1]),
                  sec = as.numeric(t_b["elapsed"]),
                  ep  = epochs_run(model_b),
                  model = model_b)
cat(sprintf("B: val accuracy %.4f  (%.1f s, %d epochs)\n",
            results$B$acc, results$B$sec, results$B$ep))

# =============================================================================
# 5. Variant C — multi-output: survived + Pclass as an auxiliary head
# =============================================================================
#
# The second head predicts the ticket class. It is not useful in itself (Pclass
# is already an input), but it forces the shared trunk to keep class-related
# information — an auxiliary task acting as a regularizer. loss_weights keeps
# its influence small so the survival head stays the primary objective.

cat("\n=== C: multi-output (survived + pclass), loss_weights ===\n")

inp_c  <- ggml_input(shape = c(N_FEATURES, 1L), name = "features")
h_c    <- inp_c |> ggml_layer_dense(D_MODEL, time_distributed = TRUE)
h_c    <- transformer_block(h_c)
trunk  <- h_c |>
  ggml_layer_flatten() |>
  ggml_layer_dense(32L, activation = "relu")

out_surv <- trunk |>
  ggml_layer_dropout(rate = DROPOUT) |>
  ggml_layer_dense(1L, activation = "sigmoid", name = "survived")

out_pcls <- trunk |>
  ggml_layer_dense(3L, activation = "softmax", name = "pclass")

model_c <- ggml_model(inputs = inp_c, outputs = list(out_surv, out_pcls))

# Names in loss / loss_weights match the output layer names (as in keras).
model_c <- ggml_compile(model_c,
                        optimizer    = "adam",
                        loss         = list(survived = "binary_crossentropy",
                                            pclass   = "categorical_crossentropy"),
                        loss_weights = c(survived = 1.0, pclass = 0.3),
                        backend      = "vulkan")

t_c <- system.time(
  model_c <- ggml_fit(model_c, x_tr_seq, list(y_tr, p_tr),
                      epochs = EPOCHS, batch_size = BATCH,
                      validation_data = list(x_va_seq, list(y_va, p_va)),
                      callbacks = early_stop(),
                      verbose = 1L)
)

pred_c <- ggml_predict(model_c, x_va_seq, batch_size = BATCH)
prob_c <- if (is.list(pred_c)) pred_c[[1]] else pred_c
results$C <- list(acc = accuracy(prob_c[, 1], y_va[, 1]),
                  sec = as.numeric(t_c["elapsed"]),
                  ep  = epochs_run(model_c),
                  model = model_c)
cat(sprintf("C: val accuracy %.4f  (%.1f s, %d epochs)\n",
            results$C$acc, results$C$sec, results$C$ep))

# Per-head history — shows whether each head is actually learning, instead of
# hiding a stalled head inside the aggregate loss.
hist_c <- model_c$history
head_keys <- grep("^(train|val)_.*_loss$", names(hist_c), value = TRUE)
if (length(head_keys)) {
  cat("Per-head loss (last epoch):\n")
  for (k in head_keys) {
    v <- hist_c[[k]]
    cat(sprintf("  %-28s %.4f\n", k, v[length(v)]))
  }
}

# =============================================================================
# Variant D — a Mamba (state-space) block over the same feature sequence
# =============================================================================
#
# Unlike A/B/C this variant is NOT built from functional-API layers. ggml's
# state-space ops are low-level tensor ops with no ggml_layer_* wrapper, so
# they cannot be placed inside a ggml_model() graph; the block is assembled
# from raw tensors and trained by its own loop. It is included to show the
# 0.8.4 SSM bindings running GPU-resident on real data, next to the attention
# variants, not because a recurrence suits a table.
#
# Two honest caveats:
#
#   * A recurrence assumes the order of its steps means something. For text or
#     a time series it does. Here the "sequence" is the feature columns, whose
#     order is arbitrary -- the same objection that makes causal attention
#     pointless in variant B, and it applies to the scan just as much.
#   * The Vulkan ssm_scan shader accepts d_state 128 or 256 ONLY. Anything else
#     is refused and the scheduler silently moves the scan to the CPU. So
#     d_state = 128 is forced here by the shader, not chosen for 10 features --
#     the state is far wider than the problem needs, which is what keeps this
#     variant GPU-resident.
#
# The block: ssm_conv over the feature sequence, then a selective scan, then
# the scan output pooled and fed to a small sigmoid head.

cat("\n=== D: Mamba / SSM block over the feature sequence ===\n")

# One thread: the SSM backward kernels are single-threaded by design.
ggml_set_n_threads(1L)

D_STATE  <- 128L    # forced by the Vulkan shader (128 or 256 only)
D_CONV   <- 4L      # Mamba uses 4
HEAD_DIM <- 8L
N_HEAD   <- 4L
D_INNER  <- HEAD_DIM * N_HEAD
N_TOK    <- N_FEATURES     # 10 features = 10 sequence steps
D_EPOCHS <- 60L
D_LR     <- 0.05
D_SEQS   <- 32L            # sequences (rows) per batch

# Build one forward+backward pass over a batch of D_SEQS rows and return the
# loss with the gradient of every parameter.
ssm_step <- function(p, xb, yb, n_seqs, use_gpu = TRUE) {
  ctx <- ggml_init(256 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  # The conv branch needs d_conv-1 positions of left context.
  conv_len <- D_CONV - 1L + N_TOK
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, conv_len, D_INNER, n_seqs)
  cw <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_CONV, D_INNER)
  ggml_set_param(cw)

  conv_out <- ggml_ssm_conv(ctx, sx, cw)
  x_scan   <- ggml_reshape_4d(ctx, conv_out, HEAD_DIM, N_HEAD, N_TOK, n_seqs)

  s0 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_STATE, HEAD_DIM, N_HEAD, n_seqs)
  dt <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_HEAD, N_TOK, n_seqs)
  ggml_set_param(dt)
  A  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, N_HEAD)
  ggml_set_param(A)
  B  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_STATE, 1L, N_TOK, n_seqs)
  ggml_set_param(B)
  C  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_STATE, 1L, N_TOK, n_seqs)
  ggml_set_param(C)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs)

  scan  <- ggml_ssm_scan(ctx, s0, x_scan, dt, A, B, C, ids)
  y_scan <- ggml_ssm_scan_output(ctx, scan, x_scan)

  # Pool the scan output down to one logit per sequence. ggml_sum reduces to a
  # scalar over everything, so the pooling is done as a mul_mat against a
  # learned weight over the flattened [D_INNER * N_TOK] activation instead.
  flat <- ggml_reshape_2d(ctx, y_scan, D_INNER * N_TOK, n_seqs)
  w    <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_INNER * N_TOK, 1L)
  ggml_set_param(w)
  b    <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1L)
  ggml_set_param(b)

  logit <- ggml_add(ctx, ggml_mul_mat(ctx, w, flat), b)   # [1, n_seqs]
  prob  <- ggml_sigmoid(ctx, logit)

  target <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_seqs)
  diff   <- ggml_sub(ctx, prob, target)
  # Mean squared error on the probability. Plain MSE rather than BCE: the loss
  # has to be a scalar built from ops that all have a backward pass, and this
  # keeps the block's gradient path short and stable.
  loss <- ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, diff)), 1 / n_seqs)
  ggml_set_output(loss)
  ggml_set_output(prob)
  ggml_set_loss(loss)

  graph <- ggml_build_forward_expand_grads(ctx, loss, graph_size = 8192L)
  ggml_build_backward_expand(ctx, graph)

  backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)

  # Each row's features are broadcast across the d_inner channels: the block
  # sees the same 10-step sequence in every channel, which is what a real Mamba
  # gets from an input projection.
  # sx layout is [conv_len, D_INNER, n_seqs]: left pad, then the sequence,
  # repeated for every channel.
  sxdat <- numeric(conv_len * D_INNER * n_seqs)
  for (s in seq_len(n_seqs)) {
    for (ch in seq_len(D_INNER)) {
      off <- (s - 1L) * conv_len * D_INNER + (ch - 1L) * conv_len
      sxdat[(off + 1L):(off + D_CONV - 1L)] <- 0
      sxdat[(off + D_CONV):(off + conv_len)] <- xb[s, ]
    }
  }
  ggml_backend_tensor_set_data(sx, sxdat)
  ggml_backend_tensor_set_data(cw, p$conv)
  ggml_backend_tensor_set_data(s0, rep(0, D_STATE * HEAD_DIM * N_HEAD * n_seqs))
  ggml_backend_tensor_set_data(dt, p$dt)
  ggml_backend_tensor_set_data(A,  p$A)
  ggml_backend_tensor_set_data(B,  p$B)
  ggml_backend_tensor_set_data(C,  p$C)
  ggml_backend_tensor_set_data(w,  p$w)
  ggml_backend_tensor_set_data(b,  p$b)
  ggml_backend_tensor_set_data(ids, as.integer(seq_len(n_seqs) - 1L))
  ggml_backend_tensor_set_data(target, yb)

  # Without ggml_graph_reset() the gradients come back silently zero.
  ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  grab <- function(t) {
    g <- ggml_graph_get_grad(graph, t)
    if (is.null(g)) NULL else ggml_backend_tensor_get_data(g)
  }
  list(loss = ggml_backend_tensor_get_data(loss)[1],
       prob = ggml_backend_tensor_get_data(prob),
       grads = list(conv = grab(cw), dt = grab(dt), A = grab(A),
                    B = grab(B), C = grab(C), w = grab(w), b = grab(b)))
}

# ---- parameters -------------------------------------------------------------

set.seed(7L)
n_tr_d <- nrow(x_tr)
d_params <- list(
  conv = runif(D_CONV * D_INNER, -0.3, 0.3),
  dt   = rep(0.05, N_HEAD * N_TOK * D_SEQS),
  A    = rep(-0.5, N_HEAD),
  B    = runif(D_STATE * N_TOK * D_SEQS, -0.3, 0.3),
  C    = runif(D_STATE * N_TOK * D_SEQS, -0.3, 0.3),
  w    = runif(D_INNER * N_TOK, -0.05, 0.05),
  b    = 0
)

# Fixed-size batches: dt/B/C are shaped per batch, so the last short batch is
# dropped rather than reshaping the parameters.
n_batch_d <- n_tr_d %/% D_SEQS

t_d <- system.time({
  d_hist <- numeric(D_EPOCHS)
  for (epoch in seq_len(D_EPOCHS)) {
    ord <- sample(n_tr_d)
    ep_loss <- 0
    for (bi in seq_len(n_batch_d)) {
      rows <- ord[((bi - 1L) * D_SEQS + 1L):(bi * D_SEQS)]
      r <- ssm_step(d_params, x_tr[rows, , drop = FALSE],
                    y_tr[rows, 1], D_SEQS, use_gpu = TRUE)
      ep_loss <- ep_loss + r$loss
      for (nm in names(r$grads)) {
        g <- r$grads[[nm]]
        if (is.null(g)) next
        # Clip to unit norm: A's gradient spans orders of magnitude.
        nrm <- sqrt(sum(g^2))
        if (is.finite(nrm) && nrm > 1) g <- g / nrm
        d_params[[nm]] <- d_params[[nm]] - D_LR * g
      }
      # A must stay negative and dt positive for exp(dt*A) to contract.
      d_params$A  <- pmin(d_params$A, -0.05)
      d_params$dt <- pmax(d_params$dt, 1e-3)
    }
    d_hist[epoch] <- ep_loss / n_batch_d
    if (epoch %% 10L == 0L || epoch == 1L) {
      cat(sprintf("  epoch %2d   loss %.5f\n", epoch, d_hist[epoch]))
    }
  }
})

# ---- validation -------------------------------------------------------------
# Predict in fixed D_SEQS batches, for the same reason training does.

d_pred <- numeric(0)
n_va_full <- (nrow(x_va) %/% D_SEQS) * D_SEQS
for (bi in seq_len(n_va_full %/% D_SEQS)) {
  rows <- ((bi - 1L) * D_SEQS + 1L):(bi * D_SEQS)
  r <- ssm_step(d_params, x_va[rows, , drop = FALSE],
                y_va[rows, 1], D_SEQS, use_gpu = TRUE)
  d_pred <- c(d_pred, r$prob)
}

acc_d <- accuracy(d_pred, y_va[seq_len(n_va_full), 1])
cat(sprintf("D: val accuracy %.4f  (%.1f s, %d epochs, %d/%d val rows)\n",
            acc_d, as.numeric(t_d["elapsed"]), D_EPOCHS,
            n_va_full, nrow(x_va)))
cat(sprintf("   loss %.5f -> %.5f\n", d_hist[1], d_hist[D_EPOCHS]))

# D is scored on a truncated validation set and has no ggml_model() behind it,
# so it is reported but deliberately kept out of the best-variant selection
# and the submission -- those stay a like-for-like comparison of A/B/C.
# =============================================================================
# 6. Summary and submission
# =============================================================================

cat("\n================ Results (20% hold-out) ================\n")
for (nm in names(results)) {
  cat(sprintf("  %-3s  accuracy %.4f   %6.1f s   %4d/%d epochs%s\n",
              nm, results[[nm]]$acc, results[[nm]]$sec,
              results[[nm]]$ep, EPOCHS,
              if (results[[nm]]$ep < EPOCHS) "  (early stop)" else ""))
}

# D is listed apart: it is scored on a truncated validation set and trained by
# its own loop, so it is not a like-for-like row in the table above.
cat(sprintf("  %-3s  accuracy %.4f   %6.1f s   %4d/%d epochs   (SSM, %d/%d val rows)\n",
            "D", acc_d, as.numeric(t_d["elapsed"]), D_EPOCHS, D_EPOCHS,
            n_va_full, nrow(x_va)))

best_nm <- names(results)[which.max(vapply(results, function(r) r$acc, numeric(1)))]
best    <- results[[best_nm]]
cat(sprintf("\nBest variant: %s (accuracy %.4f)  [A/B/C only]\n", best_nm, best$acc))

pred_test <- ggml_predict(best$model, x_test_seq, batch_size = BATCH)
prob_test <- if (is.list(pred_test)) pred_test[[1]] else pred_test
survived  <- as.integer(prob_test[, 1] > 0.5)

submission <- file.path(tempdir(), "submission.csv")
write.csv(data.frame(PassengerId = test_data$PassengerId, Survived = survived),
          submission, row.names = FALSE)

cat(sprintf("Submission: %d rows -> %s  (survival rate %.1f%%)\n",
            length(survived), submission, 100 * mean(survived)))
