# Tests for tidymodels / parsnip integration

if (!requireNamespace("parsnip", quietly = TRUE)) {
  testthat::skip("parsnip not available")
}

# Force attach so the mlr3misc::register_namespace_callback installed in
# ggmlR's .onLoad fires and registers the ggml engine.
library(parsnip)

skip_if_no_parsnip <- function() {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("tibble")
  # Ensure engine is registered (may not happen if ggmlR was loaded before parsnip)
  if (!("ggml" %in% parsnip::get_from_env("mlp_fit")$engine)) {
    ggmlR:::make_mlp_ggml()
  }
}

# ---------------------------------------------------------------------------
# Engine registration
# ---------------------------------------------------------------------------

test_that("ggml engine is registered for mlp in parsnip", {
  skip_if_no_parsnip()
  engines <- parsnip::get_from_env("mlp_fit")
  expect_true("ggml" %in% engines$engine)
})

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

test_that("parsnip mlp(engine='ggml') classifies iris", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(
    hidden_units = c(32L, 16L),
    epochs       = 10L,
    dropout      = 0.1
  ) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  fit_obj <- parsnip::fit(spec, Species ~ ., data = iris)
  expect_s3_class(fit_obj$fit, "ggmlr_parsnip_model")
  expect_equal(fit_obj$fit$mode, "classification")
  expect_equal(fit_obj$fit$class_names, levels(iris$Species))
})

test_that("parsnip ggml predict type 'class' returns factor tibble", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  fit_obj <- parsnip::fit(spec, Species ~ ., data = iris)
  preds <- predict(fit_obj, new_data = iris)

  expect_s3_class(preds, "tbl_df")
  expect_true(".pred_class" %in% names(preds))
  expect_true(is.factor(preds$.pred_class))
  expect_equal(levels(preds$.pred_class), levels(iris$Species))
  expect_equal(nrow(preds), nrow(iris))
})

test_that("parsnip ggml predict type 'prob' returns probability tibble", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  fit_obj <- parsnip::fit(spec, Species ~ ., data = iris)
  probs <- predict(fit_obj, new_data = iris, type = "prob")

  expect_s3_class(probs, "tbl_df")
  prob_cols <- paste0(".pred_", levels(iris$Species))
  expect_true(all(prob_cols %in% names(probs)))
  expect_equal(nrow(probs), nrow(iris))

  # Probabilities sum to ~1
  row_sums <- rowSums(as.matrix(probs[, prob_cols]))
  expect_true(all(abs(row_sums - 1) < 1e-4))
})

# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

test_that("parsnip mlp(engine='ggml') does regression on mtcars", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(
    hidden_units = c(32L, 16L),
    epochs       = 10L
  ) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)
  expect_s3_class(fit_obj$fit, "ggmlr_parsnip_model")
  expect_equal(fit_obj$fit$mode, "regression")
})

test_that("parsnip ggml regression predict returns numeric tibble", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)
  preds <- predict(fit_obj, new_data = mtcars)

  expect_s3_class(preds, "tbl_df")
  expect_true(".pred" %in% names(preds))
  expect_equal(nrow(preds), nrow(mtcars))
  expect_true(all(is.finite(preds$.pred)))
})

# ---------------------------------------------------------------------------
# Argument mapping
# ---------------------------------------------------------------------------

test_that("parsnip passes hidden_units as hidden_layers to ggmlR", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(8L, 4L), epochs = 3L) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  fit_obj <- parsnip::fit(spec, Species ~ ., data = iris)
  expect_s3_class(fit_obj$fit, "ggmlr_parsnip_model")
})

test_that("parsnip ggml learn_rate is applied without error", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(
    hidden_units = 16L,
    epochs       = 5L,
    learn_rate   = 0.005
  ) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  fit_obj <- parsnip::fit(spec, Species ~ ., data = iris)
  expect_s3_class(fit_obj$fit, "ggmlr_parsnip_model")

  preds <- predict(fit_obj, new_data = iris)
  expect_equal(nrow(preds), nrow(iris))
})

test_that("parsnip ggml backend='gpu' is accepted (converted to vulkan)", {
  skip_if_no_parsnip()
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = 16L, epochs = 3L) |>
    parsnip::set_engine("ggml", batch_size = 10L, backend = "gpu") |>
    parsnip::set_mode("classification")

  expect_no_error(
    fit_obj <- parsnip::fit(spec, Species ~ ., data = iris)
  )
  expect_s3_class(fit_obj$fit, "ggmlr_parsnip_model")
})
