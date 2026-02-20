# Sentiment Classification with Embedding + GRU + Skip Connection
#
# Functional API example demonstrating:
#   - ggml_input() with dtype = "int32" for token indices
#   - ggml_embedding() for learned word representations
#   - Stacked GRU layers with return_sequences
#   - Skip (residual) connection via ggml_layer_add()
#   - ggml_layer_dropout() for regularisation
#   - ggml_model() / ggml_compile() / ggml_fit()
#
# Task: binary sentiment classification (positive / negative)
#       on synthetically generated token sequences.

library(ggmlR)

set.seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------

VOCAB_SIZE  <- 500L   # tokens 0..499
SEQ_LEN     <- 30L    # words per review
N_TRAIN     <- 800L
N_TEST      <- 200L
N_TOTAL     <- N_TRAIN + N_TEST

# "Positive" reviews use tokens from the upper half of the vocabulary;
# "Negative" reviews use tokens from the lower half — a learnable signal.
make_seq <- function(n, positive) {
  low  <- if (positive) as.integer(VOCAB_SIZE / 2) else 0L
  high <- if (positive) VOCAB_SIZE - 1L            else as.integer(VOCAB_SIZE / 2) - 1L
  matrix(sample(low:high, n * SEQ_LEN, replace = TRUE),
         nrow = n, ncol = SEQ_LEN)
}

n_pos <- as.integer(N_TOTAL / 2)
n_neg <- N_TOTAL - n_pos

x_pos <- make_seq(n_pos, positive = TRUE)
x_neg <- make_seq(n_neg, positive = FALSE)

x_all <- rbind(x_pos, x_neg)                      # [N, SEQ_LEN]  int
y_all <- matrix(c(rep(c(1, 0), n_pos),
                  rep(c(0, 1), n_neg)),
                nrow = N_TOTAL, ncol = 2L, byrow = TRUE)  # one-hot

shuffle <- sample(N_TOTAL)
x_all   <- x_all[shuffle, ]
y_all   <- y_all[shuffle, ]

x_train <- x_all[seq_len(N_TRAIN), ]
y_train <- y_all[seq_len(N_TRAIN), ]
x_test  <- x_all[N_TRAIN + seq_len(N_TEST), ]
y_test  <- y_all[N_TRAIN + seq_len(N_TEST), ]

cat("Train:", nrow(x_train), "x", ncol(x_train),
    " | Test:", nrow(x_test), "x", ncol(x_test), "\n")

# ---------------------------------------------------------------------------
# 2. Build model (Functional API)
#
#   input (SEQ_LEN int32)
#     └─ embedding -> [SEQ_LEN, 32]
#         ├─ gru(32, return_sequences=FALSE) -> [32]   (branch A)
#         │    └─ dense(32) -> [32]
#         └─ dense(32) -> [32]                          (branch B, direct projection)
#              └─ add(A, B) -> [32]  (residual merge)
#                   └─ dense(16, relu) -> [16]
#                        └─ dense(2, softmax) -> [2]
# ---------------------------------------------------------------------------

EMBED_DIM <- 32L
GRU_UNITS <- 32L

inp <- ggml_input(shape = SEQ_LEN, dtype = "int32", name = "tokens")

emb <- inp |> ggml_layer_embedding(VOCAB_SIZE, EMBED_DIM, name = "embed")

# Branch A: GRU path — output [GRU_UNITS]
proj_a <- emb |>
  ggml_layer_gru(GRU_UNITS, return_sequences = FALSE, name = "gru_a") |>
  ggml_layer_dense(32L, name = "proj_a")

# Branch B: flatten embedding [SEQ_LEN * EMBED_DIM] -> dense projection [32]
proj_b <- emb |>
  ggml_layer_flatten(name = "emb_flat") |>
  ggml_layer_dense(32L, activation = "relu", name = "proj_b1") |>
  ggml_layer_dense(32L, name = "proj_b2")

# Merge via element-wise add (skip / residual connection)
merged <- ggml_layer_add(list(proj_a, proj_b), name = "residual_add")

hidden <- merged |>
  ggml_layer_dense(16L, activation = "relu",    name = "hidden") |>
  ggml_layer_dropout(rate = 0.3,                name = "drop") |>
  ggml_layer_dense(2L,  activation = "softmax", name = "output")

model <- ggml_model(inputs = inp, outputs = hidden)

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
# 4. Train
# ---------------------------------------------------------------------------

cat("\nTraining...\n")
model <- ggml_fit(model, x_train, y_train,
                  epochs           = 10L,
                  batch_size       = 64L,
                  validation_split = 0.15,
                  verbose          = 1L)

# ---------------------------------------------------------------------------
# 5. Evaluate & predict
# ---------------------------------------------------------------------------

score <- ggml_evaluate(model, x_test, y_test, batch_size = 64L)
cat("\nTest loss    :", round(score$loss,     4), "\n")
cat("Test accuracy:", round(score$accuracy, 4), "\n")

probs   <- ggml_predict(model, x_test, batch_size = 64L)
classes <- apply(probs, 1, which.max) - 1L   # 0 = positive, 1 = negative
true    <- apply(y_test, 1, which.max) - 1L

n_show <- min(length(true), length(classes))
cat("\nConfusion matrix (rows = true, cols = predicted):\n")
print(table(true = true[seq_len(n_show)], predicted = classes[seq_len(n_show)]))

# ---------------------------------------------------------------------------
# 6. Save / load
# ---------------------------------------------------------------------------

path <- tempfile(fileext = ".rds")
ggml_save_model(model, path)
cat("\nModel saved to:", path, "\n")

model2 <- ggml_load_model(path)
score2 <- ggml_evaluate(model2, x_test, y_test, batch_size = 64L)
cat("Loaded model test accuracy:", round(score2$accuracy, 4), "\n")
