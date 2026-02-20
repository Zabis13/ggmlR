# True Multi-Input Model: Time Series + Metadata -> Classification
#
# Functional API example with genuinely separate inputs:
#   inp1: time series window  [T_LEN]  float32
#   inp2: scalar metadata     [N_META] float32
#
# Architecture:
#   inp1 -> Dense(16, relu) -> Dense(8, relu)   \
#                                                concatenate -> Dense(8, relu) -> Dense(2, softmax)
#   inp2 -> Dense(8,  relu)                     /
#
# Task: binary classification — is the next value above or below the mean?

library(ggmlR)

set.seed(99)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------

T_LEN  <- 20L   # time steps in the look-back window
N_META <- 3L    # metadata: (mean, std, trend_slope)
N      <- 640L  # divisible by batch_size=32

make_sample <- function() {
  freq  <- runif(1, 0.5, 3.0)
  amp   <- runif(1, 0.5, 2.0)
  phase <- runif(1, 0, 2 * pi)
  noise <- runif(1, 0.02, 0.2)
  t     <- seq(0, by = 0.1, length.out = T_LEN + 1L)
  y_raw <- amp * sin(2 * pi * freq * t + phase) + rnorm(T_LEN + 1L, sd = noise)

  series <- y_raw[seq_len(T_LEN)]
  target <- y_raw[T_LEN + 1L]

  # metadata: normalised statistics of the window
  meta <- c(
    mean(series) / amp,           # mean (normalised)
    sd(series)   / amp,           # std  (normalised)
    (series[T_LEN] - series[1L]) / (T_LEN * amp)  # linear slope
  )

  label <- if (target > mean(series)) c(1, 0) else c(0, 1)  # one-hot

  list(ts = series, meta = meta, y = label)
}

samples <- lapply(seq_len(N), function(i) make_sample())

x1_all <- matrix(unlist(lapply(samples, `[[`, "ts")),   nrow = N, byrow = TRUE)  # [N, T_LEN]
x2_all <- matrix(unlist(lapply(samples, `[[`, "meta")), nrow = N, byrow = TRUE)  # [N, N_META]
y_all  <- matrix(unlist(lapply(samples, `[[`, "y")),    nrow = N, byrow = TRUE)  # [N, 2]

n_train <- 512L
x1_train <- x1_all[seq_len(n_train), ];  x1_test <- x1_all[(n_train + 1L):N, ]
x2_train <- x2_all[seq_len(n_train), ];  x2_test <- x2_all[(n_train + 1L):N, ]
y_train  <- y_all[seq_len(n_train),  ];  y_test  <- y_all[(n_train + 1L):N,  ]

cat(sprintf("Train: x1=%dx%d  x2=%dx%d  y=%dx%d\n",
            nrow(x1_train), ncol(x1_train),
            nrow(x2_train), ncol(x2_train),
            nrow(y_train),  ncol(y_train)))
cat(sprintf("Test:  x1=%dx%d  x2=%dx%d  y=%dx%d\n",
            nrow(x1_test), ncol(x1_test),
            nrow(x2_test), ncol(x2_test),
            nrow(y_test),  ncol(y_test)))

# ---------------------------------------------------------------------------
# 2. Build model — two separate inputs
# ---------------------------------------------------------------------------

inp1 <- ggml_input(shape = T_LEN,  name = "timeseries")
inp2 <- ggml_input(shape = N_META, name = "metadata")

branch1 <- inp1 |>
  ggml_layer_dense(16L, activation = "relu", name = "ts_fc1") |>
  ggml_layer_dense(8L,  activation = "relu", name = "ts_fc2")

branch2 <- inp2 |>
  ggml_layer_dense(8L, activation = "relu", name = "meta_fc")

merged <- ggml_layer_concatenate(list(branch1, branch2), axis = 0L, name = "concat")

output <- merged |>
  ggml_layer_dense(8L, activation = "relu",    name = "head_fc") |>
  ggml_layer_dense(2L, activation = "softmax", name = "predictions")

model <- ggml_model(inputs = list(inp1, inp2), outputs = output)

cat("\nModel summary:\n")
print(model)

# ---------------------------------------------------------------------------
# 3. Compile
# ---------------------------------------------------------------------------

model <- ggml_compile(model,
                      optimizer = "adam",
                      loss      = "categorical_crossentropy",
                      metrics   = c("accuracy"))

# ---------------------------------------------------------------------------
# 4. Train  (x passed as list of two matrices)
# ---------------------------------------------------------------------------

cat("\nTraining...\n")
model <- ggml_fit(model,
                  x          = list(x1_train, x2_train),
                  y          = y_train,
                  epochs     = 15L,
                  batch_size = 32L,
                  validation_split = 0.1,
                  verbose    = 1L)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------

score <- ggml_evaluate(model,
                       x = list(x1_test, x2_test),
                       y = y_test,
                       batch_size = 32L)
cat(sprintf("\nTest loss    : %.4f\n", score$loss))
cat(sprintf("Test accuracy: %.4f\n",  score$accuracy))

# ---------------------------------------------------------------------------
# 6. Predict
# ---------------------------------------------------------------------------

probs   <- ggml_predict(model, x = list(x1_test, x2_test), batch_size = 32L)
classes <- apply(probs, 1, which.max) - 1L
true    <- apply(y_test, 1, which.max) - 1L

n_show <- min(length(true), length(classes))
cat("\nConfusion matrix (rows = true, cols = predicted):\n")
print(table(true = true[seq_len(n_show)], predicted = classes[seq_len(n_show)]))

# ---------------------------------------------------------------------------
# 7. Save / load round-trip
# ---------------------------------------------------------------------------

path <- tempfile(fileext = ".rds")
ggml_save_model(model, path)
cat(sprintf("\nModel saved to: %s (%.1f KB)\n", path, file.size(path) / 1024))

model2 <- ggml_load_model(path)
score2 <- ggml_evaluate(model2,
                        x = list(x1_test, x2_test),
                        y = y_test,
                        batch_size = 32L)
cat(sprintf("Loaded model test accuracy: %.4f\n", score2$accuracy))
