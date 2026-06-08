# Tests for the seed / determinism API: ggml_set_seed() and its wiring into
# the parsnip "ggml" engine and the mlr3 learners.
#
# All randomness in ggmlR is base-R RNG (weight init, dropout, shuffle), so a
# fixed seed must reproduce starting weights / masks / batch order. Tests run
# on the CPU backend to avoid GPU run-to-run float jitter, and compare
# predictions rather than raw weights.

skip_if_no_parsnip <- function() {
  skip_if_not_installed("parsnip")
  skip_if_not_installed("tibble")
}

skip_if_no_mlr3 <- function() {
  skip_if_not_installed("mlr3")
  skip_if_not_installed("paradox")
  skip_if_not_installed("R6")
}

# parameter ids of a learner
learner_seed_params <- function(id) {
  mlr3::lrn(id)$param_set$ids()
}

# ── Base facade ──────────────────────────────────────────────────────────────

test_that("ggml_set_seed reproduces base-R RNG draws", {
  ggml_set_seed(123)
  a <- runif(5)
  ggml_set_seed(123)
  b <- runif(5)
  expect_identical(a, b)

  ggml_set_seed(124)
  c <- runif(5)
  expect_false(isTRUE(all.equal(a, c)))
})

test_that("ggml_set_seed(NULL) is a no-op and bad input errors", {
  expect_null(ggml_set_seed(NULL))
  expect_invisible(ggml_set_seed(1L))
  expect_error(ggml_set_seed(c(1L, 2L)), "single")
  expect_error(ggml_set_seed(NA_integer_), "single")
})

# ── parsnip engine: seed argument ────────────────────────────────────────────

test_that("parsnip ggml engine seed makes classification reproducible", {
  skip_if_no_parsnip()

  fit_with <- function(s) {
    spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L, dropout = 0.1) |>
      parsnip::set_engine("ggml", batch_size = 10L, backend = "cpu", seed = s) |>
      parsnip::set_mode("classification")
    fit_obj <- parsnip::fit(spec, Species ~ ., data = iris)
    predict(fit_obj, new_data = iris, type = "prob")
  }

  p1 <- fit_with(7L)
  p2 <- fit_with(7L)
  p3 <- fit_with(8L)

  expect_equal(p1, p2)                       # same seed -> identical
  expect_false(isTRUE(all.equal(p1, p3)))    # different seed -> different
})

test_that("parsnip ggml engine seed makes regression reproducible", {
  skip_if_no_parsnip()

  fit_with <- function(s) {
    spec <- parsnip::mlp(hidden_units = c(16L), epochs = 5L) |>
      parsnip::set_engine("ggml", batch_size = 8L, backend = "cpu", seed = s) |>
      parsnip::set_mode("regression")
    fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)
    predict(fit_obj, new_data = mtcars)
  }

  p1 <- fit_with(11L)
  p2 <- fit_with(11L)
  expect_equal(p1, p2)
})

# ── mlr3 learners: seed hyperparameter ───────────────────────────────────────

test_that("mlr3 classif.ggml exposes seed and trains reproducibly", {
  skip_if_no_mlr3()

  task <- mlr3::tsk("iris")

  fit_pred <- function(s) {
    learner <- mlr3::lrn("classif.ggml",
                         epochs = 5L, batch_size = 10L,
                         backend = "cpu", seed = s,
                         predict_type = "prob")
    learner$train(task)
    learner$predict(task)$prob
  }

  expect_true("seed" %in% learner_seed_params("classif.ggml"))

  p1 <- fit_pred(5L)
  p2 <- fit_pred(5L)
  p3 <- fit_pred(6L)
  expect_equal(p1, p2)
  expect_false(isTRUE(all.equal(p1, p3)))
})

test_that("mlr3 regr.ggml exposes seed and trains reproducibly", {
  skip_if_no_mlr3()

  task <- mlr3::tsk("mtcars")

  fit_pred <- function(s) {
    learner <- mlr3::lrn("regr.ggml",
                         epochs = 5L, batch_size = 8L,
                         backend = "cpu", seed = s)
    learner$train(task)
    learner$predict(task)$response
  }

  expect_true("seed" %in% learner_seed_params("regr.ggml"))

  p1 <- fit_pred(9L)
  p2 <- fit_pred(9L)
  expect_equal(p1, p2)
})
