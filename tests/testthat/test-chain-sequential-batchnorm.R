# Chain test: Sequential API with BatchNorm training
# Pattern from titanic_classification.R variant 3:
#   dense→BatchNorm→dense→BatchNorm→dense(softmax) + adam + fit→predict
#
# Uses synthetic linearly separable data (no external files).

# ── Sequential + BatchNorm: compile→fit→predict ─────────────

test_that("chain sequential-batchnorm: fit reduces loss", {
  set.seed(123)
  n <- 100L
  # 2-class linearly separable: class 0 centered at (-1,-1), class 1 at (1,1)
  x <- rbind(matrix(rnorm(n, -1, 0.5), n/2, 2),
             matrix(rnorm(n,  1, 0.5), n/2, 2))
  y <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
             matrix(c(0,1), n/2, 2, byrow = TRUE))

  m <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  h <- ggml_fit(m, x, y, epochs = 100L, batch_size = 10L, verbose = 0L)

  # Loss should decrease
  expect_true(h$history$train_loss[length(h$history$train_loss)] < h$history$train_loss[1])
})

test_that("chain sequential-batchnorm: predict gives valid probabilities", {
  set.seed(123)
  n <- 100L
  x <- rbind(matrix(rnorm(n, -1, 0.5), n/2, 2),
             matrix(rnorm(n,  1, 0.5), n/2, 2))
  y <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
             matrix(c(0,1), n/2, 2, byrow = TRUE))

  m <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  m <- ggml_fit(m, x, y, epochs = 30L, batch_size = 16L, verbose = 0L)
  p <- ggml_predict(m, x, batch_size = 32L)

  expect_equal(nrow(p), nrow(x))
  expect_equal(ncol(p), 2L)
  # Probabilities sum to ~1
  row_sums <- rowSums(p)
  expect_true(all(abs(row_sums - 1.0) < 0.01))
  # All probabilities in [0,1]
  expect_true(all(p >= 0 & p <= 1))
})

test_that("chain sequential-batchnorm: accuracy > 70% on separable data", {
  set.seed(42)
  n <- 200L
  x <- rbind(matrix(rnorm(n, -3, 0.5), n/2, 2),
             matrix(rnorm(n,  3, 0.5), n/2, 2))
  y <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
             matrix(c(0,1), n/2, 2, byrow = TRUE))

  m <- ggml_model_sequential() |>
    ggml_layer_dense(32L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  m <- ggml_fit(m, x, y, epochs = 120L, batch_size = 32L, verbose = 0L)
  p <- ggml_predict(m, x, batch_size = 32L)
  pred_class <- max.col(p)
  true_class <- max.col(y)
  acc <- mean(pred_class == true_class)
  expect_true(acc > 0.7)
})
