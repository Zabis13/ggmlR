# Tests for model operations: save/load, freeze/unfreeze, predict_classes

# ============================================================================
# Sequential model: predict_classes
# ============================================================================

test_that("ggml_predict_classes returns integer class indices", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 3L, activation = "softmax")

  data(iris)
  x <- as.matrix(iris[1:10, 1:4])
  x <- scale(x)
  y <- model.matrix(~ Species - 1, iris[1:10, ])

  m <- ggml_compile(m, optimizer = "adam", loss = "cross_entropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  classes <- ggml_predict_classes(m, x)
  expect_type(classes, "integer")
  expect_length(classes, 10)
  expect_true(all(classes >= 1L & classes <= 3L))
})

# ============================================================================
# freeze / unfreeze weights (sequential)
# ============================================================================

test_that("ggml_freeze_weights and ggml_unfreeze_weights work for sequential", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")

  m_frozen <- ggml_freeze_weights(m, from = 1, to = 1)
  expect_false(m_frozen$layers[[1]]$trainable)
  expect_true(m_frozen$layers[[2]]$trainable)

  m_unfrozen <- ggml_unfreeze_weights(m_frozen, from = 1, to = 1)
  expect_true(m_unfrozen$layers[[1]]$trainable)
})

# ============================================================================
# freeze / unfreeze weights (functional)
# ============================================================================

test_that("ggml_freeze_weights and ggml_unfreeze_weights work for functional", {
  x   <- ggml_input(shape = 4L)
  h   <- x |> ggml_layer_dense(8L, name = "d1")
  out <- h |> ggml_layer_dense(2L, name = "d2")
  m   <- ggml_model(inputs = x, outputs = out)

  m_frozen <- ggml_freeze_weights(m, layers = "d1")
  # Check that freeze modifies the model
  expect_s3_class(m_frozen, "ggml_functional_model")

  m_unfrozen <- ggml_unfreeze_weights(m_frozen, layers = "d1")
  expect_s3_class(m_unfrozen, "ggml_functional_model")
})

# ============================================================================
# save / load weights (sequential)
# ============================================================================

test_that("ggml_save_weights and ggml_load_weights roundtrip", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1

  m <- ggml_compile(m, optimizer = "adam", loss = "cross_entropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  pred_before <- ggml_predict(m, x)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_weights(m, tmp)

  # Load into a fresh model with same architecture
  m2 <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")
  m2 <- ggml_compile(m2, optimizer = "adam", loss = "cross_entropy")
  m2 <- ggml_load_weights(m2, tmp)

  pred_after <- ggml_predict(m2, x)
  expect_equal(pred_before, pred_after, tolerance = 1e-5)
})

# ============================================================================
# save / load model (sequential)
# ============================================================================

test_that("ggml_save_model and ggml_load_model roundtrip for sequential", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1

  m <- ggml_compile(m, optimizer = "adam", loss = "cross_entropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  pred_before <- ggml_predict(m, x)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_model(m, tmp)

  m2 <- ggml_load_model(tmp)
  pred_after <- ggml_predict(m2, x)
  expect_equal(pred_before, pred_after, tolerance = 1e-5)
})
