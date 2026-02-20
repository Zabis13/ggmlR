# ResNet-like Image Classifier (Functional API)
#
# Demonstrates:
#   - ggml_input() for image tensors [H, W, C]
#   - ggml_layer_conv_2d() with batch normalisation
#   - Residual blocks: two Conv2D paths merged via ggml_layer_add()
#   - ggml_layer_global_average_pooling_2d() to replace Flatten
#   - ggml_layer_dropout() for regularisation
#   - ggml_model() / ggml_compile() / ggml_fit()
#   - Training history plot
#
# Task: 3-class image classification on 32x32 synthetic images.
#       Each class has a distinct spatial frequency pattern so a
#       convolutional network can learn it easily.

library(ggmlR)

set.seed(123)

# ---------------------------------------------------------------------------
# 1. Synthetic dataset  (32x32 RGB images, 3 classes)
# ---------------------------------------------------------------------------

IMG_H     <- 32L
IMG_W     <- 32L
IMG_C     <- 3L
N_CLASSES <- 3L
N_TRAIN   <- 600L
N_TEST    <- 150L

make_image <- function(cls) {
  # Each class: sinusoidal pattern along a different axis + noise
  img <- array(0, dim = c(IMG_H, IMG_W, IMG_C))
  for (c in seq_len(IMG_C)) {
    freq <- cls + c
    if (cls == 1L) {
      base <- outer(sin(freq * seq(0, pi, length.out = IMG_H)),
                    rep(1, IMG_W))
    } else if (cls == 2L) {
      base <- outer(rep(1, IMG_H),
                    cos(freq * seq(0, pi, length.out = IMG_W)))
    } else {
      xs <- sin(freq * seq(0, pi, length.out = IMG_H))
      ys <- cos(freq * seq(0, pi, length.out = IMG_W))
      base <- outer(xs, ys)
    }
    img[, , c] <- (base + 1) / 2 + matrix(rnorm(IMG_H * IMG_W, sd = 0.15),
                                           nrow = IMG_H)
  }
  # Clip to [0, 1]
  img[] <- pmin(pmax(img, 0), 1)
  img
}

gen_dataset <- function(n_per_class) {
  n_total <- n_per_class * N_CLASSES
  x <- array(0, dim = c(n_total, IMG_H, IMG_W, IMG_C))
  y <- matrix(0, nrow = n_total, ncol = N_CLASSES)
  idx <- 1L
  for (cls in seq_len(N_CLASSES)) {
    for (i in seq_len(n_per_class)) {
      x[idx, , , ] <- make_image(cls)
      y[idx, cls]  <- 1
      idx <- idx + 1L
    }
  }
  shuf <- sample(n_total)
  list(x = x[shuf, , , , drop = FALSE], y = y[shuf, ])
}

cat("Generating training data...\n")
train_data <- gen_dataset(N_TRAIN %/% N_CLASSES)
test_data  <- gen_dataset(N_TEST  %/% N_CLASSES)

x_train <- train_data$x;  y_train <- train_data$y
x_test  <- test_data$x;   y_test  <- test_data$y

cat("Train:", paste(dim(x_train), collapse = " x "),
    " | Test:", paste(dim(x_test), collapse = " x "), "\n")

# ---------------------------------------------------------------------------
# 2. Helper: residual block
#
#   input -> Conv2D(F, 3x3, relu) -> BN -> Conv2D(F, 3x3) -> BN
#                                                              |
#   input -----------------------------------------------add(+) -> relu-like
#
# Both paths must have the same number of channels F.  When the input
# channels differ we add a 1x1 projection conv on the shortcut path.
# ---------------------------------------------------------------------------

residual_block <- function(x, filters, block_name) {
  # Main path: single conv (keeps graph small enough for GGML limits)
  main <- x |>
    ggml_layer_conv_2d(filters, kernel_size = c(3L, 3L),
                       activation = NULL, padding = "same",
                       name = paste0(block_name, "_conv"))

  # Shortcut: 1x1 conv to match channel count
  shortcut <- x |>
    ggml_layer_conv_2d(filters, kernel_size = c(1L, 1L),
                       activation = NULL, padding = "same",
                       name = paste0(block_name, "_proj"))

  # Residual addition
  ggml_layer_add(list(main, shortcut),
                 name = paste0(block_name, "_add"))
}

# ---------------------------------------------------------------------------
# 3. Build model
#
#   input [32, 32, 3]
#     └─ conv2d(16, 3x3, relu) -> BN        (stem)
#         └─ residual_block(16)             (block 1)
#             └─ residual_block(32)         (block 2, channel expansion)
#                 └─ global_avg_pool        (-> [32])
#                     └─ dropout(0.4)
#                         └─ dense(32, relu)
#                             └─ dense(3, softmax)
# ---------------------------------------------------------------------------

inp <- ggml_input(shape = c(IMG_H, IMG_W, IMG_C), name = "image")

# Stem
x <- inp |>
  ggml_layer_conv_2d(16L, kernel_size = c(3L, 3L),
                     activation = "relu", padding = "same",
                     name = "stem_conv")

# Residual blocks
x <- residual_block(x, 16L, "res1")
x <- residual_block(x, 32L, "res2")

# Classification head
output <- x |>
  ggml_layer_global_average_pooling_2d(name = "gap") |>
  ggml_layer_dropout(rate = 0.4,              name = "drop") |>
  ggml_layer_dense(32L, activation = "relu",  name = "fc") |>
  ggml_layer_dense(N_CLASSES, activation = "softmax", name = "predictions")

model <- ggml_model(inputs = inp, outputs = output)

cat("\nModel summary:\n")
print(model)

# ---------------------------------------------------------------------------
# 4. Compile
# ---------------------------------------------------------------------------

model <- ggml_compile(model,
                      optimizer = "adam",
                      loss      = "categorical_crossentropy",
                      metrics   = c("accuracy"))

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------

cat("\nTraining ResNet-like model...\n")
model <- ggml_fit(model, x_train, y_train,
                  epochs           = 15L,
                  batch_size       = 32L,
                  validation_split = 0.15,
                  verbose          = 1L)

# ---------------------------------------------------------------------------
# 6. Plot training history
# ---------------------------------------------------------------------------

if (interactive()) {
  plot(model$history)
} else {
  png_path <- tempfile(fileext = ".png")
  png(png_path, width = 900, height = 400)
  plot(model$history)
  dev.off()
  cat("\nHistory plot saved to:", png_path, "\n")
}

# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------

score <- ggml_evaluate(model, x_test, y_test, batch_size = 32L)
cat("\nTest loss    :", round(score$loss,     4), "\n")
cat("Test accuracy:", round(score$accuracy, 4), "\n")

probs   <- ggml_predict(model, x_test, batch_size = 32L)
classes <- apply(probs, 1, which.max)
true    <- apply(y_test[seq_len(nrow(probs)), ], 1, which.max)

cat("\nPer-class accuracy:\n")
for (cls in seq_len(N_CLASSES)) {
  mask <- true == cls
  if (any(mask)) {
    acc <- mean(classes[mask] == cls)
    cat(sprintf("  Class %d: %.1f%%\n", cls, acc * 100))
  }
}

cat("\nConfusion matrix (rows = true, cols = predicted):\n")
print(table(true = true, predicted = classes))

# ---------------------------------------------------------------------------
# 8. Save / load
# ---------------------------------------------------------------------------

path <- tempfile(fileext = ".rds")
ggml_save_model(model, path)
cat("\nModel saved to:", path,
    "(", round(file.size(path) / 1024, 1), "KB)\n")

model2 <- ggml_load_model(path)
score2 <- ggml_evaluate(model2, x_test, y_test, batch_size = 32L)
cat("Loaded model test accuracy:", round(score2$accuracy, 4), "\n")
