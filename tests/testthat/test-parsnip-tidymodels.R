# Tests for the wider tidymodels surface of the ggml mlp engine: the fit_xy()
# entry point (what workflows uses), resampling, tuning-parameter substitution,
# and the tunable-parameter set (dials/hardhat).
#
# Written against parsnip + dials + hardhat only. workflows, rsample and tune
# are deliberately not used: each test reproduces what those packages do with
# our engine (fit_xy(), v-fold splits, update(spec, parameters = ...)) so the
# engine contract stays covered without pulling the tune/workflows stack.

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


# v-fold split of row indices, like rsample::vfold_cv(): each element is one
# fold's assessment rows; the analysis rows are the rest.
vfold_idx <- function(n, v = 2L) {
  idx <- sample.int(n)
  split(idx, rep(seq_len(v), length.out = n))
}

# ---------------------------------------------------------------------------
# fit_xy() + predict() — the path workflows takes for add_formula()
# ---------------------------------------------------------------------------

test_that("fit_xy + predict works (classification)", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  fit_obj <- parsnip::fit_xy(spec, x = iris[, 1:4], y = iris$Species)
  expect_s3_class(fit_obj, "model_fit")

  # engine object is reachable through the fit
  if (requireNamespace("hardhat", quietly = TRUE)) {
    eng <- hardhat::extract_fit_engine(fit_obj)
    expect_s3_class(eng, "ggmlr_parsnip_model")
  }

  preds <- predict(fit_obj, new_data = iris[, 1:4])
  expect_s3_class(preds, "tbl_df")
  expect_true(".pred_class" %in% names(preds))
  expect_equal(nrow(preds), nrow(iris))
})

test_that("fit_xy + predict works (regression)", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  x <- mtcars[, setdiff(names(mtcars), "mpg")]
  fit_obj <- parsnip::fit_xy(spec, x = x, y = mtcars$mpg)
  preds <- predict(fit_obj, new_data = x)
  expect_true(".pred" %in% names(preds))
  expect_equal(nrow(preds), nrow(mtcars))
  expect_true(all(is.finite(preds$.pred)))
})

# ---------------------------------------------------------------------------
# Resampling: repeated fits on v-fold splits (what fit_resamples() does)
# ---------------------------------------------------------------------------

test_that("resampled fits run over cv folds (classification)", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  folds <- vfold_idx(nrow(iris), v = 2L)
  acc <- vapply(folds, function(te) {
    fit_obj <- parsnip::fit_xy(spec, x = iris[-te, 1:4], y = iris$Species[-te])
    pred <- predict(fit_obj, new_data = iris[te, 1:4])$.pred_class
    mean(pred == iris$Species[te])
  }, numeric(1))

  expect_length(acc, 2L)
  expect_true(all(is.finite(acc) & acc >= 0 & acc <= 1))
})

test_that("resampled fits run over cv folds (regression)", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  x <- mtcars[, setdiff(names(mtcars), "mpg")]
  folds <- vfold_idx(nrow(mtcars), v = 2L)
  rmse <- vapply(folds, function(te) {
    fit_obj <- parsnip::fit_xy(spec, x = x[-te, ], y = mtcars$mpg[-te])
    pred <- predict(fit_obj, new_data = x[te, ])$.pred
    sqrt(mean((pred - mtcars$mpg[te])^2))
  }, numeric(1))

  expect_length(rmse, 2L)
  expect_true(all(is.finite(rmse)))
})

# ---------------------------------------------------------------------------
# Tuning: tune() placeholders substituted per grid point, as tune_grid() does
# via update(spec, parameters = <grid row>), then fit on every fold
# ---------------------------------------------------------------------------

# Fit every grid point on every fold; returns one row per (grid point, fold)
# with the metric. `metric_fn(pred_tbl, truth)` computes the fold metric.
grid_fit <- function(spec, grid, x, y, folds, metric_fn) {
  rows <- list()
  for (i in seq_len(nrow(grid))) {
    final <- update(spec, parameters = grid[i, , drop = FALSE])
    for (te in folds) {
      fit_obj <- parsnip::fit_xy(final, x = x[-te, , drop = FALSE], y = y[-te])
      pred <- predict(fit_obj, new_data = x[te, , drop = FALSE])
      rows[[length(rows) + 1L]] <- cbind(grid[i, , drop = FALSE],
                                         metric = metric_fn(pred, y[te]))
    }
  }
  do.call(rbind, rows)
}

test_that("grid tuning over hidden_units + epochs (classification)", {
  skip_if_no_parsnip()
  set.seed(42)

  spec <- parsnip::mlp(
    hidden_units = tune(),
    epochs       = tune()
  ) |>
    parsnip::set_engine("ggml", batch_size = 10L) |>
    parsnip::set_mode("classification")

  # Two grid points (not the full 2x2 regular grid) keep the fit count low
  # while still giving the best-candidate pick a choice.
  grid <- data.frame(hidden_units = c(8L, 16L), epochs = c(3L, 5L))

  folds <- vfold_idx(nrow(iris), v = 2L)
  res <- grid_fit(spec, grid, iris[, 1:4], iris$Species, folds,
                  function(p, y) mean(p$.pred_class == y))

  # one row per (grid point x fold) combination is collected
  expect_equal(nrow(res), nrow(grid) * length(folds))
  expect_true(all(is.finite(res$metric)))

  # mean over folds per grid point, then pick the best (show_best(n = 1))
  agg <- aggregate(metric ~ hidden_units + epochs, data = res, FUN = mean)
  best <- agg[which.max(agg$metric), ]
  expect_equal(nrow(best), 1L)
})

test_that("grid tuning over epochs (regression)", {
  skip_if_no_parsnip()
  skip_if_not_installed("dials")
  set.seed(42)

  spec <- parsnip::mlp(epochs = tune()) |>
    parsnip::set_engine("ggml", batch_size = 8L) |>
    parsnip::set_mode("regression")

  grid <- as.data.frame(
    dials::grid_regular(dials::epochs(range = c(3L, 5L)), levels = 2L))
  x <- mtcars[, setdiff(names(mtcars), "mpg")]
  folds <- vfold_idx(nrow(mtcars), v = 2L)
  res <- grid_fit(spec, grid, x, mtcars$mpg, folds,
                  function(p, y) sqrt(mean((p$.pred - y)^2)))

  expect_equal(nrow(res), nrow(grid) * length(folds))
  expect_true(all(is.finite(res$metric)))
})

test_that("grid tuning over the whole dials activation grid fits every value", {
  skip_if_no_parsnip()
  skip_if_not_installed("dials")
  skip_if_not_installed("hardhat")
  set.seed(42)

  spec <- parsnip::mlp(activation = tune(), hidden_units = 4L, epochs = 2L) |>
    parsnip::set_engine("ggml", batch_size = 16L, backend = "cpu") |>
    parsnip::set_mode("classification")

  # The engine registers only activations that nn_apply_activation() knows and
  # that train; plain dials::activation() would offer 23.
  ps <- hardhat::extract_parameter_set_dials(spec)
  act_values <- ps$object[[which(ps$id == "activation")]]$values
  expect_setequal(act_values, c("relu", "sigmoid", "tanh", "silu", "gelu"))

  # the grid tune itself would build from the parameter set: every value.
  # Any engine error (e.g. "Unknown activation") fails the test here.
  grid  <- as.data.frame(dials::grid_regular(ps, levels = length(act_values)))
  folds <- vfold_idx(nrow(iris), v = 2L)
  res   <- grid_fit(spec, grid, iris[, 1:4], iris$Species, folds,
                    function(p, y) mean(p$.pred_class == y))

  expect_setequal(res$activation, act_values)
  # every activation was fitted on both folds
  expect_true(all(table(res$activation) == 2L))
  expect_true(all(is.finite(res$metric)))
})
