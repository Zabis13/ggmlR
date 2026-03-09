# Tests for Sequential Neural Network API (v0.5.3)

# Helper: free all backends associated with a compiled model
cleanup_model <- function(model) {
  ggml_backend_sched_free(model$compilation$sched)
  ggml_backend_free(model$compilation$backend)
  if (!is.null(model$compilation$cpu_backend)) {
    ggml_backend_free(model$compilation$cpu_backend)
  }
}

# ============================================================================
# Model creation
# ============================================================================

test_that("ggml_model_sequential creates empty model", {
  model <- ggml_model_sequential()

  expect_s3_class(model, "ggml_sequential_model")
  expect_equal(length(model$layers), 0)
  expect_null(model$input_shape)
  expect_false(model$compiled)
})

# ============================================================================
# Layer addition
# ============================================================================

test_that("ggml_layer_conv_2d adds layer correctly", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), activation = "relu",
                       input_shape = c(28, 28, 1))

  expect_equal(length(model$layers), 1)
  expect_equal(model$layers[[1]]$type, "conv_2d")
  expect_equal(model$layers[[1]]$config$filters, 32L)
  expect_equal(model$layers[[1]]$config$kernel_size, c(3L, 3L))
  expect_equal(model$layers[[1]]$config$activation, "relu")
  expect_equal(model$input_shape, c(28L, 28L, 1L))
})

test_that("ggml_layer_conv_2d handles scalar kernel_size", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(16, 5, input_shape = c(28, 28, 1))

  expect_equal(model$layers[[1]]$config$kernel_size, c(5L, 5L))
})

test_that("ggml_layer_conv_2d default padding and strides", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), input_shape = c(28, 28, 1))

  expect_equal(model$layers[[1]]$config$strides, c(1L, 1L))
  expect_equal(model$layers[[1]]$config$padding, "valid")
  expect_null(model$layers[[1]]$config$activation)
})

test_that("ggml_layer_max_pooling_2d adds layer correctly", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), input_shape = c(28, 28, 1)) |>
    ggml_layer_max_pooling_2d(c(2, 2))

  expect_equal(length(model$layers), 2)
  expect_equal(model$layers[[2]]$type, "max_pooling_2d")
  expect_equal(model$layers[[2]]$config$pool_size, c(2L, 2L))
  # strides default to pool_size
  expect_equal(model$layers[[2]]$config$strides, c(2L, 2L))
})

test_that("ggml_layer_flatten adds layer correctly", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), input_shape = c(28, 28, 1)) |>
    ggml_layer_flatten()

  expect_equal(length(model$layers), 2)
  expect_equal(model$layers[[2]]$type, "flatten")
})

test_that("ggml_layer_dense adds layer correctly", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), input_shape = c(28, 28, 1)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(128, activation = "relu")

  expect_equal(length(model$layers), 3)
  expect_equal(model$layers[[3]]$type, "dense")
  expect_equal(model$layers[[3]]$config$units, 128L)
  expect_equal(model$layers[[3]]$config$activation, "relu")
})

# ============================================================================
# Shape inference
# ============================================================================

test_that("nn_infer_shapes computes correct output shapes for CNN", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), activation = "relu",
                       input_shape = c(28, 28, 1)) |>
    ggml_layer_max_pooling_2d(c(2, 2)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(128, activation = "relu") |>
    ggml_layer_dense(10, activation = "softmax")

  model <- ggmlR:::nn_infer_shapes(model)

  # conv_2d: (28-3)/1 + 1 = 26, output (26, 26, 32)
  expect_equal(model$layers[[1]]$output_shape, c(26L, 26L, 32L))

  # max_pool: (26-2)/2 + 1 = 13, output (13, 13, 32)
  expect_equal(model$layers[[2]]$output_shape, c(13L, 13L, 32L))

  # flatten: 13*13*32 = 5408
  expect_equal(model$layers[[3]]$output_shape, 5408L)

  # dense 128
  expect_equal(model$layers[[4]]$output_shape, 128L)

  # dense 10
  expect_equal(model$layers[[5]]$output_shape, 10L)
})

test_that("nn_infer_shapes works with same padding", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(16, c(3, 3), padding = "same",
                       input_shape = c(28, 28, 1))

  model <- ggmlR:::nn_infer_shapes(model)

  # same padding: output H,W = input H,W (with stride 1)
  expect_equal(model$layers[[1]]$output_shape, c(28L, 28L, 16L))
})

test_that("nn_infer_shapes fails without input_shape", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(10)

  expect_error(ggmlR:::nn_infer_shapes(model), "input_shape")
})

# ============================================================================
# Compile
# ============================================================================

test_that("ggml_compile works and sets compiled flag", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), activation = "relu",
                       input_shape = c(28, 28, 1)) |>
    ggml_layer_max_pooling_2d(c(2, 2)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(10, activation = "softmax")

  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy",
                        metrics = c("accuracy"))

  expect_true(model$compiled)
  expect_equal(model$compilation$optimizer, "adam")
  expect_equal(model$compilation$loss, "categorical_crossentropy")
  expect_equal(model$compilation$metrics, "accuracy")
  expect_true(!is.null(model$compilation$backend))
  expect_true(!is.null(model$compilation$sched))

  # Shapes should be inferred
  expect_equal(model$layers[[1]]$output_shape, c(26L, 26L, 32L))

  # Cleanup
  cleanup_model(model)
})

test_that("ggml_compile fails on empty model", {
  model <- ggml_model_sequential()
  expect_error(ggml_compile(model), "no layers")
})

test_that("ggml_compile with sgd optimizer", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(8, c(3, 3), input_shape = c(10, 10, 1)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(5)

  model <- ggml_compile(model, optimizer = "sgd", loss = "mse")

  expect_true(model$compiled)
  expect_equal(model$compilation$optimizer, "sgd")
  expect_equal(model$compilation$loss, "mse")

  cleanup_model(model)
})

# ============================================================================
# Print
# ============================================================================

test_that("print.ggml_sequential_model works", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), activation = "relu",
                       input_shape = c(28, 28, 1)) |>
    ggml_layer_max_pooling_2d(c(2, 2)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(128, activation = "relu") |>
    ggml_layer_dense(10, activation = "softmax")

  output <- capture.output(print(model))

  expect_true(any(grepl("Sequential Model", output)))
  expect_true(any(grepl("conv_2d", output)))
  expect_true(any(grepl("max_pooling_2d", output)))
  expect_true(any(grepl("flatten", output)))
  expect_true(any(grepl("dense", output)))
  expect_true(any(grepl("Total parameters", output)))
})

test_that("print.ggml_sequential_model works on empty model", {
  model <- ggml_model_sequential()
  output <- capture.output(print(model))
  expect_true(any(grepl("no layers", output)))
})

# ============================================================================
# Pipe chaining
# ============================================================================

test_that("pipe chaining builds full model", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), activation = "relu",
                       input_shape = c(28, 28, 1)) |>
    ggml_layer_conv_2d(64, c(3, 3), activation = "relu") |>
    ggml_layer_max_pooling_2d(c(2, 2)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(128, activation = "relu") |>
    ggml_layer_dense(10, activation = "softmax")

  expect_equal(length(model$layers), 6)
  expect_equal(model$layers[[1]]$type, "conv_2d")
  expect_equal(model$layers[[2]]$type, "conv_2d")
  expect_equal(model$layers[[3]]$type, "max_pooling_2d")
  expect_equal(model$layers[[4]]$type, "flatten")
  expect_equal(model$layers[[5]]$type, "dense")
  expect_equal(model$layers[[6]]$type, "dense")
})

# ============================================================================
# ggml_set_input / ggml_set_output
# ============================================================================

test_that("ggml_set_input and ggml_set_output work on tensors", {
  ctx <- ggml_init(1024 * 1024)
  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10L)

  # Should not error
  expect_no_error(ggml_set_input(t))
  expect_no_error(ggml_set_output(t))

  ggml_free(ctx)
})

# ============================================================================
# Training (small synthetic data)
# ============================================================================

test_that("ggml_fit trains a small dense model", {
  # Simple XOR-like problem: 4 samples, 2 features, 2 classes
  n <- 128  # must be divisible by batch_size
  x <- matrix(runif(n * 4), nrow = n, ncol = 4)
  y <- matrix(0, nrow = n, ncol = 2)
  for (i in seq_len(n)) {
    cls <- if (sum(x[i, ]) > 2) 1L else 2L
    y[i, cls] <- 1
  }

  model <- ggml_model_sequential() |>
    ggml_layer_dense(8, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")

  # Set input_shape manually (dense-only model)
  model$input_shape <- 4L

  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy")

  # Train 1 epoch
  expect_no_error({
    model <- ggml_fit(model, x, y, epochs = 1,
                      batch_size = 32, verbose = 0)
  })

  # Cleanup
  cleanup_model(model)
})

test_that("ggml_evaluate returns accuracy consistent with training", {
  set.seed(42)
  n <- 256
  x <- matrix(runif(n * 4), nrow = n, ncol = 4)
  y <- matrix(0, nrow = n, ncol = 2)
  for (i in seq_len(n)) {
    cls <- if (sum(x[i, ]) > 2) 1L else 2L
    y[i, cls] <- 1
  }

  model <- ggml_model_sequential() |>
    ggml_layer_dense(16, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model$input_shape <- 4L

  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy")

  model <- ggml_fit(model, x, y, epochs = 20,
                    batch_size = 32, verbose = 0)

  # Evaluate on same data — should retain trained weights
  result <- ggml_evaluate(model, x, y, batch_size = 32)

  expect_true(!is.null(result$loss))
  expect_true(!is.null(result$accuracy))
  # Trained model should do better than random (50%)
  expect_gt(result$accuracy, 0.55)
  # Loss should be below random baseline (-ln(0.5) ≈ 0.693)
  expect_lt(result$loss, 0.69)

  # Cleanup
  cleanup_model(model)
})

# ============================================================================
# Predict
# ============================================================================

test_that("ggml_predict returns correct shape and consistent with evaluate", {
  set.seed(42)
  n <- 256
  x <- matrix(runif(n * 4), nrow = n, ncol = 4)
  y <- matrix(0, nrow = n, ncol = 2)
  for (i in seq_len(n)) {
    cls <- if (sum(x[i, ]) > 2) 1L else 2L
    y[i, cls] <- 1
  }

  model <- ggml_model_sequential() |>
    ggml_layer_dense(16, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model$input_shape <- 4L

  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy")

  model <- ggml_fit(model, x, y, epochs = 20,
                    batch_size = 32, verbose = 0)

  # Predict
  preds <- ggml_predict(model, x, batch_size = 32)

  # Check shape: [N, 2]
  expect_equal(nrow(preds), n)
  expect_equal(ncol(preds), 2)

  # Predictions should be probabilities (softmax output)
  expect_true(all(preds >= 0))
  expect_true(all(preds <= 1))
  # Each row should sum to ~1 (softmax)
  row_sums <- rowSums(preds)
  expect_true(all(abs(row_sums - 1) < 0.01))

  # Argmax predictions should be consistent with evaluate accuracy
  pred_classes <- apply(preds, 1, which.max)
  true_classes <- apply(y, 1, which.max)
  predict_accuracy <- mean(pred_classes == true_classes)

  eval_result <- ggml_evaluate(model, x, y, batch_size = 32)

  # Predict and evaluate accuracy should be close
  expect_equal(predict_accuracy, eval_result$accuracy, tolerance = 0.05)

  # Cleanup
  cleanup_model(model)
})

# ============================================================================
# Save / Load Weights
# ============================================================================

test_that("ggml_save_weights and ggml_load_weights roundtrip", {
  set.seed(42)
  n <- 256
  x <- matrix(runif(n * 4), nrow = n, ncol = 4)
  y <- matrix(0, nrow = n, ncol = 2)
  for (i in seq_len(n)) {
    cls <- if (sum(x[i, ]) > 2) 1L else 2L
    y[i, cls] <- 1
  }

  model <- ggml_model_sequential() |>
    ggml_layer_dense(16, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model$input_shape <- 4L

  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy")

  model <- ggml_fit(model, x, y, epochs = 20,
                    batch_size = 32, verbose = 0)

  # Evaluate original model
  result_orig <- ggml_evaluate(model, x, y, batch_size = 32)

  # Save weights
  tmp_path <- tempfile(fileext = ".rds")
  ggml_save_weights(model, tmp_path)
  expect_true(file.exists(tmp_path))

  # Cleanup original model backend resources
  cleanup_model(model)

  # Create a fresh model with same architecture and load weights
  model2 <- ggml_model_sequential() |>
    ggml_layer_dense(16, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model2$input_shape <- 4L

  model2 <- ggml_compile(model2, optimizer = "adam",
                         loss = "categorical_crossentropy")

  model2 <- ggml_load_weights(model2, tmp_path)

  # Evaluate loaded model
  result_loaded <- ggml_evaluate(model2, x, y, batch_size = 32)

  # Accuracy should match
  expect_equal(result_loaded$accuracy, result_orig$accuracy, tolerance = 0.01)
  expect_equal(result_loaded$loss, result_orig$loss, tolerance = 0.01)

  # Cleanup
  cleanup_model(model2)
  unlink(tmp_path)
})

test_that("ggml_load_weights rejects mismatched architecture", {
  model1 <- ggml_model_sequential() |>
    ggml_layer_dense(16, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model1$input_shape <- 4L
  model1 <- ggml_compile(model1, optimizer = "adam",
                         loss = "categorical_crossentropy")

  # Train minimally
  x <- matrix(runif(64 * 4), nrow = 64, ncol = 4)
  y <- matrix(0, nrow = 64, ncol = 2)
  for (i in seq_len(64)) { y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1 }
  model1 <- ggml_fit(model1, x, y, epochs = 1, batch_size = 32, verbose = 0)

  tmp_path <- tempfile(fileext = ".rds")
  ggml_save_weights(model1, tmp_path)

  # Different architecture
  model2 <- ggml_model_sequential() |>
    ggml_layer_dense(32, activation = "relu") |>
    ggml_layer_dense(8, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model2$input_shape <- 4L
  model2 <- ggml_compile(model2, optimizer = "adam",
                         loss = "categorical_crossentropy")

  expect_error(ggml_load_weights(model2, tmp_path), "mismatch")

  # Cleanup
  cleanup_model(model1)
  cleanup_model(model2)
  unlink(tmp_path)
})

# ============================================================================
# Error handling
# ============================================================================

# ============================================================================
# conv_1d layer
# ============================================================================

test_that("ggml_layer_conv_1d adds layer correctly", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_1d(32, 3, activation = "relu",
                       input_shape = c(100, 1))

  expect_equal(length(model$layers), 1)
  expect_equal(model$layers[[1]]$type, "conv_1d")
  expect_equal(model$layers[[1]]$config$filters, 32L)
  expect_equal(model$layers[[1]]$config$kernel_size, 3L)
  expect_equal(model$input_shape, c(100L, 1L))
})

test_that("nn_infer_shapes works for conv_1d", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_1d(16, 5, input_shape = c(100, 1)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(10)

  model <- ggmlR:::nn_infer_shapes(model)

  # conv_1d: (100 - 5) / 1 + 1 = 96, output (96, 16)
  expect_equal(model$layers[[1]]$output_shape, c(96L, 16L))
  # flatten: 96 * 16 = 1536
  expect_equal(model$layers[[2]]$output_shape, 1536L)
  # dense: 10
  expect_equal(model$layers[[3]]$output_shape, 10L)
})

test_that("nn_infer_shapes works for conv_1d with same padding", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_1d(16, 5, padding = "same", input_shape = c(100, 1))

  model <- ggmlR:::nn_infer_shapes(model)

  # same padding: L_out = L (with stride 1)
  expect_equal(model$layers[[1]]$output_shape, c(100L, 16L))
})

# ============================================================================
# batch_norm layer
# ============================================================================

test_that("ggml_layer_batch_norm adds layer correctly", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(64, activation = "relu") |>
    ggml_layer_batch_norm()

  model$input_shape <- 10L

  expect_equal(length(model$layers), 2)
  expect_equal(model$layers[[2]]$type, "batch_norm")
})

test_that("nn_infer_shapes works for batch_norm", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(64) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(10)

  model$input_shape <- 10L
  model <- ggmlR:::nn_infer_shapes(model)

  # batch_norm preserves shape
  expect_equal(model$layers[[2]]$output_shape, 64L)
  expect_equal(model$layers[[3]]$output_shape, 10L)
})

# ============================================================================
# predict_classes
# ============================================================================

test_that("ggml_predict_classes returns correct 1-based indices", {
  set.seed(42)
  n <- 128
  x <- matrix(runif(n * 4), nrow = n, ncol = 4)
  y <- matrix(0, nrow = n, ncol = 2)
  for (i in seq_len(n)) {
    cls <- if (sum(x[i, ]) > 2) 1L else 2L
    y[i, cls] <- 1
  }

  model <- ggml_model_sequential() |>
    ggml_layer_dense(16, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model$input_shape <- 4L

  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy")
  model <- ggml_fit(model, x, y, epochs = 10,
                    batch_size = 32, verbose = 0)

  classes <- ggml_predict_classes(model, x, batch_size = 32)

  expect_true(is.integer(classes))
  expect_equal(length(classes), n)
  expect_true(all(classes %in% c(1L, 2L)))

  # Cleanup
  cleanup_model(model)
})

# ============================================================================
# summary
# ============================================================================

test_that("summary.ggml_sequential_model works", {
  model <- ggml_model_sequential() |>
    ggml_layer_conv_2d(32, c(3, 3), activation = "relu",
                       input_shape = c(28, 28, 1)) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(10, activation = "softmax")

  output <- capture.output(summary(model))

  expect_true(any(grepl("Summary", output)))
  expect_true(any(grepl("Input shape", output)))
  expect_true(any(grepl("Trainable", output)))
  expect_true(any(grepl("memory", output, ignore.case = TRUE)))
})

# ============================================================================
# history
# ============================================================================

test_that("ggml_fit returns model with history", {
  set.seed(42)
  n <- 128
  x <- matrix(runif(n * 4), nrow = n, ncol = 4)
  y <- matrix(0, nrow = n, ncol = 2)
  for (i in seq_len(n)) {
    cls <- if (sum(x[i, ]) > 2) 1L else 2L
    y[i, cls] <- 1
  }

  model <- ggml_model_sequential() |>
    ggml_layer_dense(8, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  model$input_shape <- 4L

  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy")
  model <- ggml_fit(model, x, y, epochs = 3,
                    batch_size = 32, verbose = 0)

  expect_true(!is.null(model$history))
  expect_s3_class(model$history, "ggml_history")
  expect_equal(length(model$history$train_loss), 3)
  expect_equal(length(model$history$train_accuracy), 3)
  expect_true(all(is.numeric(model$history$train_loss)))

  # Print should work
  output <- capture.output(print(model$history))
  expect_true(any(grepl("Training History", output)))

  # Cleanup
  cleanup_model(model)
})

# ============================================================================
# Error handling
# ============================================================================

test_that("ggml_fit fails on uncompiled model", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(10)
  model$input_shape <- 5L

  x <- matrix(0, nrow = 32, ncol = 5)
  y <- matrix(0, nrow = 32, ncol = 10)

  expect_error(ggml_fit(model, x, y), "compiled")
})
