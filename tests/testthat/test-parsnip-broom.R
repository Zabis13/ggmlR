# broom (tidy / glance / augment) methods for fitted parsnip "ggml" models.
# Run on the CPU backend; structure/shape checks only, no GPU dependence.

skip_if_no_parsnip <- function() {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("tibble")
}

fit_classif <- function() {
  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 3L, dropout = 0.1) |>
    parsnip::set_engine("ggml", batch_size = 10L, backend = "cpu", seed = 1L) |>
    parsnip::set_mode("classification")
  parsnip::extract_fit_engine(parsnip::fit(spec, Species ~ ., data = iris))
}

fit_regr <- function() {
  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 3L) |>
    parsnip::set_engine("ggml", batch_size = 8L, backend = "cpu", seed = 1L) |>
    parsnip::set_mode("regression")
  parsnip::extract_fit_engine(parsnip::fit(spec, mpg ~ ., data = mtcars))
}

# ── tidy() ───────────────────────────────────────────────────────────────────

test_that("tidy() returns one row per layer with expected columns", {
  skip_if_no_parsnip()
  eng <- fit_regr()

  td <- generics::tidy(eng)
  expect_s3_class(td, "tbl_df")
  expect_setequal(
    names(td),
    c("layer", "type", "units", "activation", "output_shape",
      "params", "trainable")
  )
  expect_equal(nrow(td), length(eng$model$layers))
  expect_true(all(td$params >= 0))
  expect_true(is.logical(td$trainable))
})

# ── glance() ─────────────────────────────────────────────────────────────────

test_that("glance() returns a one-row summary with config + fit_time", {
  skip_if_no_parsnip()
  eng <- fit_classif()

  gl <- generics::glance(eng)
  expect_s3_class(gl, "tbl_df")
  expect_equal(nrow(gl), 1L)
  expect_setequal(
    names(gl),
    c("mode", "n_features", "n_layers", "total_params", "optimizer",
      "loss", "backend", "epochs", "fit_time", "final_loss")
  )
  expect_equal(gl$mode, "classification")
  expect_equal(gl$n_features, 4L)          # iris predictors
  expect_equal(gl$backend, "cpu")
  expect_equal(gl$epochs, 3L)
  expect_true(is.finite(gl$fit_time))
  expect_gt(gl$total_params, 0L)
})

# ── augment() ────────────────────────────────────────────────────────────────

test_that("augment() appends .pred_class + per-class probs for classification", {
  skip_if_no_parsnip()
  eng <- fit_classif()

  aug <- generics::augment(eng, iris)
  expect_s3_class(aug, "tbl_df")
  expect_equal(nrow(aug), nrow(iris))
  expect_true(".pred_class" %in% names(aug))
  expect_true(all(paste0(".pred_", levels(iris$Species)) %in% names(aug)))
  # original columns preserved
  expect_true(all(names(iris) %in% names(aug)))
})

test_that("augment() appends .pred for regression", {
  skip_if_no_parsnip()
  eng <- fit_regr()

  aug <- generics::augment(eng, mtcars)
  expect_s3_class(aug, "tbl_df")
  expect_equal(nrow(aug), nrow(mtcars))
  expect_true(".pred" %in% names(aug))
  expect_true(is.numeric(aug$.pred))
})
