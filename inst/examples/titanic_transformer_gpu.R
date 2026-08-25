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

# Every variant is scored on the SAME validation rows. Variant D's parameters
# are shaped per batch, so it can only predict in whole D_SEQS-sized batches
# and its score would otherwise come from fewer rows than A/B/C's -- a
# difference of a couple of rows is the same size as the gap between the
# variants, which would make the table misleading. Truncating all four to the
# common prefix costs a handful of rows and buys a like-for-like comparison.
D_SEQS  <- 32L
N_SCORE <- (as.integer(0.2 * nrow(x_all)) %/% D_SEQS) * D_SEQS
score_i <- seq_len(N_SCORE)

# The bar any binary classifier has to clear: always predicting the majority
# class. On Titanic that is about 0.62.
base_d <- max(mean(y_va[score_i, 1]), 1 - mean(y_va[score_i, 1]))

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
results$A <- list(acc = accuracy(prob_a[score_i, 1], y_va[score_i, 1]),
                  sec = as.numeric(t_a["elapsed"]),
                  ep  = epochs_run(model_a),
                  dev = model_a$compilation$device,
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
results$B <- list(acc = accuracy(prob_b[score_i, 1], y_va[score_i, 1]),
                  sec = as.numeric(t_b["elapsed"]),
                  ep  = epochs_run(model_b),
                  dev = model_b$compilation$device,
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
results$C <- list(acc = accuracy(prob_c[score_i, 1], y_va[score_i, 1]),
                  sec = as.numeric(t_c["elapsed"]),
                  ep  = epochs_run(model_c),
                  dev = model_c$compilation$device,
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
# from raw tensors and trained by its own loop.
#
# This is a real selective SSM: dt, B and C are PROJECTIONS OF THE INPUT
# (x -> mul_mat -> per-step values), which is what makes the scan selective.
# Storing them as free parameters instead would tie them to a row's position
# in the batch rather than to its content, and with shuffled batches the block
# then cannot learn anything beyond the mean.
#
# ---- what the Vulkan ssm_scan shader requires (ggml-vulkan-graph.cpp) -------
# The scan silently falls back to the CPU unless ALL of these hold:
#   * d_state is exactly 128 or 256
#   * head_dim %% 16 == 0          <- easy to miss; head_dim 8 runs on the CPU
#   * A is [1, n_head] (nb[1] == 4 bytes), i.e. the Mamba-2 layout
#   * every source is F32, and the ids tensor is I32
# The placement is printed below, because "a GPU was created" and "the GPU ran
# the scan" are different claims and only the second one matters here.
#
# Caveat that no amount of tuning fixes: a recurrence assumes the order of its
# steps carries meaning. For text or a time series it does; here the sequence
# is the feature columns, whose order is arbitrary. D is included to show the
# 0.8.4 SSM bindings training GPU-resident on real data, not because a scan
# suits a table.

cat("\n=== D: Mamba / SSM block over the feature sequence ===\n")

# One thread: the SSM backward kernels are single-threaded by design.
invisible(ggml_set_n_threads(1L))

D_STATE  <- 128L   # shader: 128 or 256 only
HEAD_DIM <- 16L    # shader: must be a multiple of 16
N_HEAD   <- 4L
D_INNER  <- HEAD_DIM * N_HEAD
D_CONV   <- 4L
N_TOK    <- N_FEATURES    # 10 features = 10 sequence steps
D_EPOCHS <- 60L
D_LR     <- 0.5
# D_SEQS is defined with the shared hyperparameters -- the scoring subset
# depends on it, and A/B/C need that before variant D runs.

# Forward + backward over one batch. Returns the loss, the predicted
# probabilities and the gradient of every parameter.
ssm_step <- function(p, xb, yb, n_seqs, want_placement = FALSE) {
  ctx <- ggml_init(512 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  # One scalar per sequence step: [1, n_tok, n_seqs].
  xseq <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1L, N_TOK, n_seqs)

  # Input projection, then transpose into the [conv_len, d_inner, n_seqs]
  # layout ssm_conv expects, with d_conv-1 steps of zero left context.
  w_in <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, D_INNER)
  ggml_set_param(w_in)
  proj <- ggml_cont(ctx, ggml_transpose(ctx, ggml_mul_mat(ctx, w_in, xseq)))
  zpad <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_CONV - 1L, D_INNER, n_seqs)
  padded <- ggml_concat(ctx, zpad, proj, 0L)

  cw <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_CONV, D_INNER)
  ggml_set_param(cw)
  conv <- ggml_silu(ctx, ggml_ssm_conv(ctx, padded, cw))
  x_scan <- ggml_reshape_4d(ctx, conv, HEAD_DIM, N_HEAD, N_TOK, n_seqs)

  # Selective parameters, all projected from the input.
  # softplus keeps dt positive, which is what makes exp(dt*A) contract --
  # the principled version of clamping dt after each step.
  w_dt <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, N_HEAD)
  ggml_set_param(w_dt)
  dt <- ggml_softplus(ctx, ggml_mul_mat(ctx, w_dt, xseq))

  w_B <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, D_STATE)
  ggml_set_param(w_B)
  B <- ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w_B, xseq), D_STATE, 1L, N_TOK, n_seqs)

  w_C <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, D_STATE)
  ggml_set_param(w_C)
  C <- ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w_C, xseq), D_STATE, 1L, N_TOK, n_seqs)

  # A stays a free parameter: it is the state decay, not a function of x.
  # [1, n_head] is the Mamba-2 layout the shader tests for.
  A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, N_HEAD)
  ggml_set_param(A)

  s0  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D_STATE, HEAD_DIM, N_HEAD, n_seqs)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs)

  scan   <- ggml_ssm_scan(ctx, s0, x_scan, dt, A, B, C, ids)
  y_scan <- ggml_ssm_scan_output(ctx, scan, x_scan)

  # Pool the whole scan output into one logit per row.
  flat <- ggml_reshape_2d(ctx, y_scan, D_INNER * N_TOK, n_seqs)
  w    <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_INNER * N_TOK, 1L)
  ggml_set_param(w)
  b    <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1L)
  ggml_set_param(b)
  prob <- ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w, flat), b))

  target <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_seqs)
  # MSE on the probability: the loss must be a scalar built from ops that all
  # have a backward pass, and this keeps the gradient path short and stable.
  loss <- ggml_scale(ctx,
            ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, prob, target))), 1 / n_seqs)
  ggml_set_output(loss)
  ggml_set_output(prob)
  ggml_set_loss(loss)

  graph <- ggml_build_forward_expand_grads(ctx, loss, graph_size = 8192L)
  ggml_build_backward_expand(ctx, graph)

  backend <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(backend), add = TRUE)
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)
  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)

  placement <- NULL
  if (want_placement) {
    where <- function(t) {
      tryCatch(ggml_vulkan_backend_name(
                 ggml_backend_sched_get_tensor_backend(sched, t)),
               error = function(e) "unassigned")
    }
    placement <- list(conv = where(conv), scan = where(scan), loss = where(loss),
                      splits = ggml_backend_sched_get_n_splits(sched))
  }

  ggml_backend_tensor_set_data(xseq, as.numeric(t(xb)))
  ggml_backend_tensor_set_data(zpad, rep(0, (D_CONV - 1L) * D_INNER * n_seqs))
  ggml_backend_tensor_set_data(w_in, p$w_in)
  ggml_backend_tensor_set_data(cw,   p$conv)
  ggml_backend_tensor_set_data(w_dt, p$w_dt)
  ggml_backend_tensor_set_data(w_B,  p$w_B)
  ggml_backend_tensor_set_data(w_C,  p$w_C)
  ggml_backend_tensor_set_data(A,    p$A)
  ggml_backend_tensor_set_data(w,    p$w)
  ggml_backend_tensor_set_data(b,    p$b)
  ggml_backend_tensor_set_data(s0,  rep(0, D_STATE * HEAD_DIM * N_HEAD * n_seqs))
  ggml_backend_tensor_set_data(ids, as.integer(seq_len(n_seqs) - 1L))
  ggml_backend_tensor_set_data(target, as.numeric(yb))

  # Without ggml_graph_reset() every gradient comes back silently zero.
  ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  grab <- function(t) {
    g <- ggml_graph_get_grad(graph, t)
    if (is.null(g)) NULL else ggml_backend_tensor_get_data(g)
  }
  list(loss = ggml_backend_tensor_get_data(loss)[1],
       prob = ggml_backend_tensor_get_data(prob),
       placement = placement,
       grads = list(w_in = grab(w_in), conv = grab(cw), w_dt = grab(w_dt),
                    w_B = grab(w_B), w_C = grab(w_C), A = grab(A),
                    w = grab(w), b = grab(b)))
}

set.seed(7L)
d_params <- list(
  w_in = runif(D_INNER, -0.5, 0.5),
  conv = runif(D_CONV * D_INNER, -0.3, 0.3),
  w_dt = rep(0.1, N_HEAD),
  w_B  = runif(D_STATE, -0.3, 0.3),
  w_C  = runif(D_STATE, -0.3, 0.3),
  A    = rep(-0.5, N_HEAD),
  w    = runif(D_INNER * N_TOK, -0.05, 0.05),
  b    = 0
)

n_tr_d    <- nrow(x_tr)
n_batch_d <- n_tr_d %/% D_SEQS   # fixed-size batches; the short tail is dropped

# Filled in from the scheduler on the first batch (see want_placement below).
d_scan_dev <- "?"

t_d <- system.time({
  d_hist <- numeric(D_EPOCHS)
  for (epoch in seq_len(D_EPOCHS)) {
    ord <- sample(n_tr_d)
    ep_loss <- 0
    for (bi in seq_len(n_batch_d)) {
      rows <- ord[((bi - 1L) * D_SEQS + 1L):(bi * D_SEQS)]
      r <- ssm_step(d_params, x_tr[rows, , drop = FALSE], y_tr[rows, 1], D_SEQS,
                    want_placement = (epoch == 1L && bi == 1L))
      if (!is.null(r$placement)) {
        pl <- r$placement
        cat(sprintf("  placement: ssm_conv -> %s | ssm_scan -> %s | loss -> %s | splits %d\n",
                    pl$conv, pl$scan, pl$loss, pl$splits))
        # The scan is the op with the shape constraints, so its placement is
        # what the summary table reports for D.
        d_scan_dev <<- if (grepl("^Vulkan", pl$scan)) "GPU" else "CPU"
        if (d_scan_dev != "GPU") {
          cat("  NOTE: the scan fell back to the CPU -- check d_state / head_dim.\n")
        }
      }
      ep_loss <- ep_loss + r$loss
      for (nm in names(r$grads)) {
        g <- r$grads[[nm]]
        if (is.null(g)) next
        nrm <- sqrt(sum(g^2))
        if (is.finite(nrm) && nrm > 1) g <- g / nrm   # clip to unit norm
        d_params[[nm]] <- d_params[[nm]] - D_LR * g
      }
      # A is the decay: it must stay negative for exp(dt*A) to contract.
      # dt needs no clamp -- softplus already keeps it positive.
      d_params$A <- pmin(d_params$A, -0.05)
    }
    d_hist[epoch] <- ep_loss / n_batch_d
    if (epoch %% 10L == 0L || epoch == 1L) {
      cat(sprintf("  epoch %2d   loss %.5f\n", epoch, d_hist[epoch]))
    }
  }
})

# Validation over exactly the rows A/B/C are scored on (score_i), in the
# fixed-size batches this block is limited to.
d_pred <- numeric(0)
for (bi in seq_len(N_SCORE %/% D_SEQS)) {
  rows <- ((bi - 1L) * D_SEQS + 1L):(bi * D_SEQS)
  r <- ssm_step(d_params, x_va[rows, , drop = FALSE], y_va[rows, 1], D_SEQS)
  d_pred <- c(d_pred, r$prob)
}

acc_d <- accuracy(d_pred, y_va[score_i, 1])
cat(sprintf("D: val accuracy %.4f  (%.1f s, %d epochs, %d/%d val rows)\n",
            acc_d, as.numeric(t_d["elapsed"]), D_EPOCHS, N_SCORE, nrow(x_va)))
cat(sprintf("   loss %.5f -> %.5f   majority-class baseline %.4f\n",
            d_hist[1], d_hist[D_EPOCHS], base_d))

# D competes on equal terms: every variant is scored on the same score_i rows,
# so the only thing that still sets D apart is how it predicts -- it is not a
# ggml_model(), so `kind` tells the submission step which path to take.
results$D <- list(acc = acc_d, sec = as.numeric(t_d["elapsed"]),
                  ep = D_EPOCHS, dev = d_scan_dev, kind = "ssm",
                  params = d_params)

# D competes with A/B/C on equal terms: all four are scored on the same rows.
# The only difference left is the prediction path -- D has no ggml_model(), so
# the submission step runs its block directly (see predict_ssm below).
# =============================================================================
# 6. Summary and submission
# =============================================================================

short_dev <- function(d) {
  if (is.null(d) || is.na(d)) return("?")
  # A/B/C store compilation$device, the full Vulkan device description; D
  # stores the scan's placement, already reduced to "GPU"/"CPU".
  if (d %in% c("GPU", "CPU")) return(d)
  if (grepl("cpu", d, ignore.case = TRUE)) "CPU" else "GPU"
}

desc <- c(A = "attention",
          B = "attention, causal",
          C = "attention, 2 heads",
          D = "SSM / Mamba scan")

cat("\n=========================== Results (20% hold-out) ===========================\n")
cat(sprintf("  %-3s %-20s %8s %8s %14s %6s\n",
            "", "model", "accuracy", "time", "epochs", "device"))
cat("  ", strrep("-", 74), "\n", sep = "")

for (nm in names(results)) {
  r <- results[[nm]]
  # D runs a fixed epoch budget rather than early stopping, so its own count
  # is the ceiling; A/B/C share the EPOCHS budget.
  budget <- if (identical(r$kind, "ssm")) r$ep else EPOCHS
  cat(sprintf("  %-3s %-20s %8.4f %7.1fs %6d/%-7d %6s%s\n",
              nm, desc[[nm]], r$acc, r$sec, r$ep, budget, short_dev(r$dev),
              if (r$ep < budget) "  (early stop)" else ""))
}
cat(sprintf("  %-3s %-20s %8.4f %7s %14s %6s\n",
            "", "majority-class", base_d, "-", "-", "-"))
cat(sprintf("\n  all variants scored on the same %d of %d validation rows\n",
            N_SCORE, nrow(x_va)))

best_nm <- names(results)[which.max(vapply(results, function(r) r$acc, numeric(1)))]
best    <- results[[best_nm]]
cat(sprintf("\nBest variant: %s (accuracy %.4f)\n", best_nm, best$acc))

# Predicting with D means running its own block over the test rows in whole
# D_SEQS batches. 418 test rows are not a multiple of 32, so the final short
# batch is padded by repeating its last row and only the real predictions are
# kept -- dropping the tail instead would leave rows without a prediction, and
# a submission has to cover every PassengerId.
predict_ssm <- function(p, xm) {
  n <- nrow(xm)
  out <- numeric(0)
  for (start in seq(1L, n, by = D_SEQS)) {
    rows <- start:min(start + D_SEQS - 1L, n)
    n_real <- length(rows)
    xb <- xm[rows, , drop = FALSE]
    if (n_real < D_SEQS) {
      xb <- rbind(xb, xb[rep(n_real, D_SEQS - n_real), , drop = FALSE])
    }
    r <- ssm_step(p, xb, rep(0, D_SEQS), D_SEQS)
    out <- c(out, r$prob[seq_len(n_real)])
  }
  out
}

if (identical(best$kind, "ssm")) {
  prob_test <- matrix(predict_ssm(best$params, x_test), ncol = 1L)
} else {
  pred_test <- ggml_predict(best$model, x_test_seq, batch_size = BATCH)
  prob_test <- if (is.list(pred_test)) pred_test[[1]] else pred_test
}
survived  <- as.integer(prob_test[, 1] > 0.5)

write.csv(
  data.frame(PassengerId = test_data$PassengerId, Survived = survived),
  "submission.csv", row.names = FALSE
)
cat(sprintf("Submission (%s): %d rows -> submission.csv  (survival rate %.1f%%)\n",
            best_nm, length(survived), 100 * mean(survived)))
