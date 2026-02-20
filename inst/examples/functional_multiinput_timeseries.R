# Time Series Regression with Dual-Branch Functional Model
#
# Functional API example demonstrating:
#   - ggml_input() for a combined feature vector
#   - Conv1D branch to extract local patterns from the series
#   - Dense branch to encode metadata features
#   - ggml_layer_concatenate() to merge both branches
#   - LSTM on Conv1D features
#   - Regression output (linear activation)
#   - ggml_freeze_weights() / ggml_unfreeze_weights() for fine-tuning
#
# Task: predict the next value of a noisy sine wave given:
#         (a) the last T = 40 time steps  (first 40 features)
#         (b) 4 scalar meta-features      (last 4 features)
#             (frequency, amplitude, phase, noise_level)
#
# Input is a single flat vector [T_LEN + N_META] per sample.
# Inside the network we split into two branches and merge with concatenate.

library(ggmlR)

set.seed(7)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------

T_LEN   <- 40L   # look-back window
N_META  <- 4L    # metadata features
N_IN    <- T_LEN + N_META   # total input width
N_TOTAL <- 1000L
N_TRAIN <- 800L

make_sample <- function(i) {
  freq  <- runif(1, 0.5, 3.0)
  amp   <- runif(1, 0.5, 2.0)
  phase <- runif(1, 0, 2 * pi)
  noise <- runif(1, 0.01, 0.3)
  t     <- seq(i * 0.1, by = 0.1, length.out = T_LEN + 1L)
  series <- amp * sin(2 * pi * freq * t + phase) + rnorm(T_LEN + 1L, sd = noise)
  c(series[seq_len(T_LEN)],                         # raw series
    freq, amp, phase / (2 * pi), noise,              # normalised metadata
    series[T_LEN + 1L])                              # target appended last
}

mat    <- do.call(rbind, lapply(seq_len(N_TOTAL), make_sample))
x_all  <- mat[, seq_len(N_IN)]                       # [N, 44]
y_all  <- matrix(mat[, N_IN + 1L], ncol = 1L)        # [N, 1]

x_train <- x_all[seq_len(N_TRAIN), ]
y_train <- y_all[seq_len(N_TRAIN), , drop = FALSE]
x_test  <- x_all[(N_TRAIN + 1L):N_TOTAL, ]
y_test  <- y_all[(N_TRAIN + 1L):N_TOTAL, , drop = FALSE]

cat("Train:", paste(dim(x_train), collapse = " x "),
    " | Test:", paste(dim(x_test), collapse = " x "), "\n")

# ---------------------------------------------------------------------------
# 2. Build model (Functional API — single input, dual branch with add merge)
#
#   input [44]  (40 series + 4 meta)
#     ├─ dense(16, relu) "branch_a"  -> [16]
#     └─ dense(16, relu) "branch_b"  -> [16]
#
#   add([branch_a, branch_b]) -> [16]
#     └─ dense(1) "out"        -> [1]  regression
# ---------------------------------------------------------------------------

inp <- ggml_input(shape = N_IN, name = "features")

# Two parallel branches of the same width (required for add)
branch_a <- inp |> ggml_layer_dense(16L, activation = "relu", name = "branch_a")
branch_b <- inp |> ggml_layer_dense(16L, activation = "relu", name = "branch_b")

# Merge via element-wise add
merged <- ggml_layer_add(list(branch_a, branch_b), name = "add_merge")

output <- merged |>
  ggml_layer_dense(1L, name = "out")

model <- ggml_model(inputs = inp, outputs = output)

cat("\nModel summary:\n")
print(model)

# ---------------------------------------------------------------------------
# 3. Compile
# ---------------------------------------------------------------------------

model <- ggml_compile(model,
                      optimizer = "adam",
                      loss      = "mse",
                      metrics   = c("mae"))

# ---------------------------------------------------------------------------
# 4. Train — phase 1: full model
# ---------------------------------------------------------------------------

cat("\nPhase 1: training full model...\n")
model <- ggml_fit(model, x_train, y_train,
                  epochs           = 10L,
                  batch_size       = 64L,
                  validation_split = 0.1,
                  verbose          = 1L)

# ---------------------------------------------------------------------------
# 5. Fine-tune: lower learning rate for second phase
# ---------------------------------------------------------------------------

cat("\nPhase 2: fine-tuning with lower learning rate (SGD)...\n")
model <- ggml_compile(model,
                      optimizer = "sgd",
                      loss      = "mse",
                      metrics   = c("mae"))

model <- ggml_fit(model, x_train, y_train,
                  epochs           = 5L,
                  batch_size       = 64L,
                  validation_split = 0.1,
                  verbose          = 1L)

# ---------------------------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------------------------

score <- ggml_evaluate(model, x_test, y_test, batch_size = 64L)
cat("\nTest MSE:", round(score$loss, 5), "\n")
if (!is.null(score$mae)) cat("Test MAE:", round(score$mae, 5), "\n")

preds <- ggml_predict(model, x_test, batch_size = 64L)
n_pred <- nrow(preds)
y_cmp <- y_test[seq_len(n_pred), , drop = FALSE]

ss_res <- sum((y_cmp - preds)^2)
ss_tot <- sum((y_cmp - mean(y_cmp))^2)
cat("Test R²:", round(1 - ss_res / ss_tot, 4), "\n")

# ---------------------------------------------------------------------------
# 7. Save / load
# ---------------------------------------------------------------------------

path <- tempfile(fileext = ".rds")
ggml_save_model(model, path)
cat("\nModel saved to:", path,
    "(", round(file.size(path) / 1024, 1), "KB)\n")

model2 <- ggml_load_model(path)
score2 <- ggml_evaluate(model2, x_test, y_test, batch_size = 64L)
cat("Loaded model Test MSE:", round(score2$loss, 5), "\n")
