# Tests for the wider tidymodels stack: workflows, resampling (rsample),
# tuning (tune), and the tunable-parameter surface (dials/hardhat).
#
# These pull in optional packages that are NOT in ggmlR's Suggests beyond
# parsnip itself, so every test skips gracefully when a package is missing.
# To run them locally, install:
#   install.packages(c("workflows", "rsample", "tune", "yardstick", "dials"))

if (!requireNamespace("parsnip", quietly = TRUE)) {
  testthat::skip("parsnip not available")
}

library(parsnip)

skip_if_no_parsnip <- function() {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("tibble")
  if (!("ggml" %in% parsnip::get_from_env("mlp_fit")$engine)) {
    ggmlR:::make_mlp_ggml()
  }
}

# ---------------------------------------------------------------------------
# Tunable parameter surface (dials / hardhat)
# ---------------------------------------------------------------------------

test_that("extract_parameter_set_dials() exposes tunable mlp params", {
  skip_if_no_parsnip()
  skip_if_not_installed("hardhat")
  skip_if_not_installed("dials")

  spec <- parsnip::mlp(
    hidden_units = tune(),
    epochs       = tune(),
    dropout      = tune()
  ) |>
    parsnip::set_engine("ggml") |>
    parsnip::set_mode("classification")

  ps <- hardhat::extract_parameter_set_dials(spec)
  ids <- ps$id
  expect_true(all(c("hidden_units", "epochs", "dropout") %in% ids))
  # parameters are finalized (have ranges) -> grids can be built without data
  expect_true(nrow(dials::grid_regular(ps, levels = 2L)) > 0L)
})

# ---------------------------------------------------------------------------
# workflow() + fit() + predict()
# ---------------------------------------------------------------------------

test_that("workflow fit + predict works (classification)", {
  skip_if_no_parsnip()
  skip_if_not_installed("workflows")
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  wf <- workflows::workflow() |>
    workflows::add_model(spec) |>
    workflows::add_formula(Species ~ .)

  fit_obj <- parsnip::fit(wf, data = iris)
  expect_s3_class(fit_obj, "workflow")

  # engine object is reachable through the workflow
  if (requireNamespace("hardhat", quietly = TRUE)) {
    eng <- hardhat::extract_fit_engine(fit_obj)
    expect_s3_class(eng, "ggmlr_parsnip_model")
  }

  preds <- predict(fit_obj, new_data = iris)
  expect_s3_class(preds, "tbl_df")
  expect_true(".pred_class" %in% names(preds))
  expect_equal(nrow(preds), nrow(iris))
})

test_that("workflow fit + predict works (regression)", {
  skip_if_no_parsnip()
  skip_if_not_installed("workflows")
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  wf <- workflows::workflow() |>
    workflows::add_model(spec) |>
    workflows::add_formula(mpg ~ .)

  fit_obj <- parsnip::fit(wf, data = mtcars)
  preds <- predict(fit_obj, new_data = mtcars)
  expect_true(".pred" %in% names(preds))
  expect_equal(nrow(preds), nrow(mtcars))
  expect_true(all(is.finite(preds$.pred)))
})

# ---------------------------------------------------------------------------
# fit_resamples()
# ---------------------------------------------------------------------------

test_that("fit_resamples() runs over cv folds (classification)", {
  skip_if_no_parsnip()
  skip_if_not_installed("workflows")
  skip_if_not_installed("rsample")
  skip_if_not_installed("tune")
  skip_if_not_installed("yardstick")
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  wf <- workflows::workflow() |>
    workflows::add_model(spec) |>
    workflows::add_formula(Species ~ .)

  folds <- rsample::vfold_cv(iris, v = 3L)
  res <- tune::fit_resamples(wf, resamples = folds)

  metrics <- tune::collect_metrics(res)
  expect_s3_class(metrics, "tbl_df")
  expect_true("accuracy" %in% metrics$.metric)
})

test_that("fit_resamples() runs over cv folds (regression)", {
  skip_if_no_parsnip()
  skip_if_not_installed("workflows")
  skip_if_not_installed("rsample")
  skip_if_not_installed("tune")
  skip_if_not_installed("yardstick")
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  wf <- workflows::workflow() |>
    workflows::add_model(spec) |>
    workflows::add_formula(mpg ~ .)

  folds <- rsample::vfold_cv(mtcars, v = 3L)
  res <- tune::fit_resamples(wf, resamples = folds)

  metrics <- tune::collect_metrics(res)
  expect_s3_class(metrics, "tbl_df")
  expect_true("rmse" %in% metrics$.metric)
})

# ---------------------------------------------------------------------------
# tune_grid()
# ---------------------------------------------------------------------------

test_that("tune_grid() tunes hidden_units + epochs (classification)", {
  skip_if_no_parsnip()
  skip_if_not_installed("workflows")
  skip_if_not_installed("rsample")
  skip_if_not_installed("tune")
  skip_if_not_installed("yardstick")
  skip_if_not_installed("dials")
  set.seed(42)

  spec <- parsnip::mlp(
    hidden_units = tune(),
    epochs       = tune()
  ) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  wf <- workflows::workflow() |>
    workflows::add_model(spec) |>
    workflows::add_formula(Species ~ .)

  grid <- dials::grid_regular(
    dials::hidden_units(range = c(8L, 16L)),
    dials::epochs(range = c(3L, 5L)),
    levels = 2L
  )

  folds <- rsample::vfold_cv(iris, v = 3L)
  res <- tune::tune_grid(wf, resamples = folds, grid = grid)

  metrics <- tune::collect_metrics(res)
  expect_s3_class(metrics, "tbl_df")
  expect_true("accuracy" %in% metrics$.metric)
  # one row per (grid point x metric) combination is collected
  expect_gt(nrow(metrics), 0L)

  best <- tune::show_best(res, metric = "accuracy", n = 1L)
  expect_equal(nrow(best), 1L)
})

test_that("tune_grid() tunes epochs (regression)", {
  skip_if_no_parsnip()
  skip_if_not_installed("workflows")
  skip_if_not_installed("rsample")
  skip_if_not_installed("tune")
  skip_if_not_installed("yardstick")
  skip_if_not_installed("dials")
  set.seed(42)

  spec <- parsnip::mlp(epochs = tune()) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  wf <- workflows::workflow() |>
    workflows::add_model(spec) |>
    workflows::add_formula(mpg ~ .)

  grid <- dials::grid_regular(dials::epochs(range = c(3L, 5L)), levels = 2L)
  folds <- rsample::vfold_cv(mtcars, v = 3L)
  res <- tune::tune_grid(wf, resamples = folds, grid = grid)

  metrics <- tune::collect_metrics(res)
  expect_true("rmse" %in% metrics$.metric)
})
