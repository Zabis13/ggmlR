# Demo of new ggmlR 0.5.3 features:
#   - ggml_predict_classes()
#   - summary()
#   - Training history with plot()
#   - ggml_layer_conv_1d()
#   - ggml_layer_batch_norm()

library(ggmlR)

cat("=== ggmlR 0.5.3 New Features Demo ===\n\n")

# ============================================================================
# 1. Dense model with batch_norm, predict_classes, summary, history
# ============================================================================

cat("--- 1. Dense + batch_norm model ---\n\n")

set.seed(42)
n <- 256
x <- matrix(runif(n * 4), nrow = n, ncol = 4)
y <- matrix(0, nrow = n, ncol = 3)
for (i in seq_len(n)) {
  s <- sum(x[i, ])
  cls <- if (s < 1.5) 1L else if (s < 2.5) 2L else 3L
  y[i, cls] <- 1
}

model <- ggml_model_sequential() |>
  ggml_layer_dense(32, activation = "relu") |>
  ggml_layer_batch_norm() |>
  ggml_layer_dense(16, activation = "relu") |>
  ggml_layer_dense(3, activation = "softmax")
model$input_shape <- 4L

# summary() — detailed view
cat("Model summary:\n")
summary(model)
cat("\n")

# Compile and train
model <- ggml_compile(model, optimizer = "adam",
                      loss = "categorical_crossentropy")

model <- ggml_fit(model, x, y, epochs = 10, batch_size = 32,
                  validation_split = 0.2, verbose = 1)

# History
cat("\nTraining history:\n")
print(model$history)

# Plot history (saves to file if non-interactive)
if (interactive()) {
  plot(model$history)
} else {
  png(tempfile(fileext = ".png"), width = 800, height = 400)
  plot(model$history)
  dev.off()
  cat("(History plot saved to temp file)\n")
}

# predict_classes()
classes <- ggml_predict_classes(model, x, batch_size = 32)
true_classes <- apply(y, 1, which.max)
acc <- mean(classes == true_classes)
cat("\npredict_classes accuracy:", sprintf("%.1f%%", acc * 100), "\n")
cat("First 10 predicted:", classes[1:10], "\n")
cat("First 10 true:     ", true_classes[1:10], "\n")

# Cleanup
ggml_backend_sched_free(model$compilation$sched)
ggml_backend_free(model$compilation$backend)

# ============================================================================
# 2. Conv1D model (shape inference demo only — training requires
#    ggml_backend_sched fix for im2col F16 intermediate tensors)
# ============================================================================

cat("\n--- 2. Conv1D model (shape inference) ---\n\n")

model_1d <- ggml_model_sequential() |>
  ggml_layer_conv_1d(16, kernel_size = 5, activation = "relu",
                     input_shape = c(50, 2)) |>
  ggml_layer_conv_1d(32, kernel_size = 3, activation = "relu") |>
  ggml_layer_flatten() |>
  ggml_layer_dense(16, activation = "relu") |>
  ggml_layer_dense(2, activation = "softmax")

cat("Conv1D model summary:\n")
summary(model_1d)

cat("\n=== Demo complete ===\n")
