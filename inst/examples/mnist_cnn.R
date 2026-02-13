# MNIST CNN Example using ggmlR Sequential API
#
# This example trains a simple CNN on the MNIST handwritten digit dataset.
# It demonstrates the Keras-like high-level API built on top of ggml.
#
# Requirements:
#   - ggmlR package (installed)
#   - keras3 package (for loading MNIST data and to_categorical)
#     or you can load MNIST data any other way

library(ggmlR)

# Load MNIST data without keras3
if (!require("dslabs", quietly = TRUE)) install.packages("dslabs")
library(dslabs)

mnist <- read_mnist()

# Convert to keras3-like format
x_train <- array(mnist$train$images, dim = c(nrow(mnist$train$images), 28, 28)) / 255.0
y_train <- mnist$train$labels

x_test <- array(mnist$test$images, dim = c(nrow(mnist$test$images), 28, 28)) / 255.0
y_test <- mnist$test$labels

# One-hot encode labels (replace keras3::to_categorical)
to_categorical <- function(y, num_classes = 10) {
  n <- length(y)
  y_cat <- matrix(0, nrow = n, ncol = num_classes)
  for (i in seq_len(n)) {
    y_cat[i, y[i] + 1] <- 1
  }
  y_cat
}

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Reshape for CNN: [N, H, W] -> [N, H, W, C]
x_train <- array(x_train, dim = c(dim(x_train), 1))
x_test <- array(x_test, dim = c(dim(x_test), 1))

cat("Training data shape:", paste(dim(x_train), collapse = " x "), "\n")
cat("Training labels shape:", paste(dim(y_train), collapse = " x "), "\n")
cat("Test data shape:", paste(dim(x_test), collapse = " x "), "\n")

# --- Build Model ---
model <- ggml_model_sequential() |>
  ggml_layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                     activation = "relu", input_shape = c(28, 28, 1)) |>
  ggml_layer_max_pooling_2d(pool_size = c(2, 2)) |>
  ggml_layer_flatten() |>
  ggml_layer_dense(units = 128, activation = "relu") |>
  ggml_layer_dense(units = 10, activation = "softmax")

print(model)

# --- Compile ---
model <- model |> ggml_compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

cat("\nModel compiled. Starting training...\n\n")

# --- Train ---
model <- model |> ggml_fit(
  x_train, y_train,
  epochs = 5,
  batch_size = 128,
  validation_split = 0.2
)

# --- Evaluate ---
score <- model |> ggml_evaluate(x_test, y_test, batch_size = 128)

cat("\nTest loss:", score$loss, "\n")
cat("Test accuracy:", score$accuracy, "\n")

# --- Predict ---
preds <- model |> ggml_predict(x_test, batch_size = 128)

cat("\nPrediction matrix shape:", paste(dim(preds), collapse = " x "), "\n")
cat("First 5 predicted classes:", apply(preds[1:5, ], 1, which.max) - 1, "\n")
cat("First 5 true labels:     ", apply(y_test[1:5, ], 1, which.max) - 1, "\n")

# --- Save/Load Weights ---
weights_path <- tempfile(fileext = ".rds")
ggml_save_weights(model, weights_path)
cat("\nWeights saved to:", weights_path, "\n")
cat("File size:", file.size(weights_path) / 1024, "KB\n")
