# Model diagnostics: ggml_training_history() and ggml_model_backend().
# Run on the CPU backend so they pass without a GPU.

fit_seq_cpu <- function(epochs = 3L, validation_split = 0.0) {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(3L, activation = "softmax")
  model <- ggml_compile(model, optimizer = "adam",
                        loss = "categorical_crossentropy", backend = "cpu")

  x <- as.matrix(iris[, 1:4]); storage.mode(x) <- "double"
  y <- matrix(0, nrow(x), 3L)
  y[cbind(seq_len(nrow(x)), as.integer(iris$Species))] <- 1
  ggml_fit(model, x = x, y = y, epochs = epochs, batch_size = 16L,
           validation_split = validation_split, verbose = 0L)
}

# ── ggml_training_history() ─────────────────────────────────────────────────

test_that("training_history wide format has one row per epoch", {
  model <- fit_seq_cpu(epochs = 3L)

  h <- ggml_training_history(model)
  expect_equal(nrow(h), 3L)
  expect_true(all(c("epoch", "train_loss", "train_accuracy") %in% names(h)))
  expect_equal(h$epoch, 1:3)
})

test_that("training_history long format reshapes to epoch/metric/split/value", {
  model <- fit_seq_cpu(epochs = 3L)

  hl <- ggml_training_history(model, format = "long")
  expect_true(all(c("epoch", "metric", "split", "value") %in% names(hl)))
  expect_setequal(unique(hl$metric), c("loss", "accuracy"))
  expect_setequal(unique(hl$split), "train")          # no val split
  # 3 epochs x 2 metrics x 1 split = 6 rows
  expect_equal(nrow(hl), 6L)
})

test_that("training_history includes val columns when validation_split > 0", {
  model <- fit_seq_cpu(epochs = 2L, validation_split = 0.3)

  h <- ggml_training_history(model)
  expect_true(all(c("val_loss", "val_accuracy") %in% names(h)))
})

test_that("training_history warns + NULL for unfitted model", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(4L, activation = "relu", input_shape = 4L)
  expect_warning(res <- ggml_training_history(model), "no training history")
  expect_null(res)
})

# ── ggml_model_backend() ────────────────────────────────────────────────────

test_that("model_backend reports cpu and full detail list", {
  model <- fit_seq_cpu(epochs = 1L)

  expect_equal(ggml_model_backend(model), "cpu")

  info <- ggml_model_backend(model, verbose = TRUE)
  expect_equal(info$requested, "cpu")
  expect_equal(info$used, "cpu")
  expect_equal(info$device, "cpu")
  expect_false(info$fallback)   # cpu requested -> cpu used, not a fallback
})

test_that("model_backend errors on uncompiled model", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(4L, activation = "relu", input_shape = 4L)
  expect_error(ggml_model_backend(model), "not compiled")
})

# ── parsnip engine unwrapping ───────────────────────────────────────────────

test_that("diagnostics accessors unwrap a fitted parsnip engine", {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("tibble")

  spec <- parsnip::mlp(hidden_units = 8L, epochs = 3L) |>
    parsnip::set_engine("ggml", batch_size = 16L, backend = "cpu", seed = 1L) |>
    parsnip::set_mode("classification")
  eng <- parsnip::extract_fit_engine(parsnip::fit(spec, Species ~ ., data = iris))

  expect_equal(nrow(ggml_training_history(eng)), 3L)
  expect_equal(ggml_model_backend(eng), "cpu")
})
