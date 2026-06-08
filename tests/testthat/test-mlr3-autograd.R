# Tests for the mlr3 autograd tradepath (v2 item 1):
# when model_fn returns an ag_sequential module, the learner trains it with an
# R-level gradient-tape loop (Adam/SGD) instead of ggml_compile()/ggml_fit().

if (!requireNamespace("mlr3",    quietly = TRUE) ||
    !requireNamespace("paradox", quietly = TRUE) ||
    !requireNamespace("R6",      quietly = TRUE)) {
  testthat::skip("mlr3 / paradox / R6 not available")
}

library(mlr3)
ggmlR:::.register_mlr3()

skip_if_no_mlr3 <- function() {
  skip_if_not_installed("mlr3")
  skip_if_not_installed("paradox")
  skip_if_not_installed("R6")
}

# An ag_sequential builder produces a plain wrapper around ag_linear layers.
# ag_linear exposes $forward and $parameters, which ag_sequential collects.
# For classif the final layer must emit raw logits (no softmax) because the
# learner applies fused softmax cross-entropy.
ag_classif_builder <- function(task, n_features, n_out, pars) {
  ag_sequential(
    ag_linear(n_features, 16L, activation = "relu"),
    ag_linear(16L, n_out, activation = NULL)
  )
}

ag_regr_builder <- function(task, n_features, n_out, pars) {
  ag_sequential(
    ag_linear(n_features, 16L, activation = "relu"),
    ag_linear(16L, 1L, activation = NULL)
  )
}

# ---------------------------------------------------------------------------
# Classification autograd tradepath
# ---------------------------------------------------------------------------

test_that("LearnerClassifGGML trains an ag_sequential via autograd tradepath", {
  skip_if_no_mlr3()
  set.seed(1)
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs        <- 20L
  learner$param_set$values$batch_size    <- 16L
  learner$param_set$values$learning_rate <- 0.05
  learner$model_fn     <- ag_classif_builder
  learner$predict_type <- "prob"

  learner$train(task)
  expect_s3_class(learner$model, "classif_ggml_model")
  expect_s3_class(learner$model$model, "ag_sequential")
  expect_equal(learner$model$class_names, task$class_names)

  pred <- learner$predict(task)
  expect_s3_class(pred, "PredictionClassif")
  expect_true(is.factor(pred$response))
  expect_equal(levels(pred$response), task$class_names)
  # softmax over logits -> rows sum to 1
  expect_true(all(abs(rowSums(pred$prob) - 1) < 1e-4))
  # iris is easy: a tiny MLP should beat chance comfortably
  expect_gt(mean(pred$response == task$truth()), 0.7)
})

# ---------------------------------------------------------------------------
# Regression autograd tradepath
# ---------------------------------------------------------------------------

test_that("LearnerRegrGGML trains an ag_sequential via autograd tradepath", {
  skip_if_no_mlr3()
  set.seed(1)
  task <- mlr3::tsk("mtcars")

  learner <- mlr3::lrn("regr.ggml")
  learner$param_set$values$epochs        <- 30L
  learner$param_set$values$batch_size    <- 8L
  learner$param_set$values$learning_rate <- 0.01
  learner$model_fn <- ag_regr_builder

  learner$train(task)
  expect_s3_class(learner$model$model, "ag_sequential")

  pred <- learner$predict(task)
  expect_equal(length(pred$response), task$nrow)
  expect_true(all(is.finite(pred$response)))
})

# ---------------------------------------------------------------------------
# learning_rate is exposed for tuning (autograd-only parameter)
# ---------------------------------------------------------------------------

test_that("learning_rate and max_grad_norm are tunable params", {
  skip_if_no_mlr3()
  learner <- mlr3::lrn("classif.ggml")
  ids <- learner$param_set$ids()
  expect_true("learning_rate" %in% ids)
  expect_true("max_grad_norm" %in% ids)

  learner <- mlr3::lrn("regr.ggml")
  ids <- learner$param_set$ids()
  expect_true("learning_rate" %in% ids)
  expect_true("max_grad_norm" %in% ids)
})

# ---------------------------------------------------------------------------
# max_grad_norm activates gradient clipping (smoke: training still converges)
# ---------------------------------------------------------------------------

test_that("max_grad_norm clipping does not break training", {
  skip_if_no_mlr3()
  set.seed(2)
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs        <- 20L
  learner$param_set$values$batch_size    <- 16L
  learner$param_set$values$learning_rate <- 0.05
  learner$param_set$values$max_grad_norm <- 1.0
  learner$model_fn <- ag_classif_builder

  learner$train(task)
  pred <- learner$predict(task)
  expect_gt(mean(pred$response == task$truth()), 0.6)
})

# ---------------------------------------------------------------------------
# User-supplied training_fn overrides the default loop
# ---------------------------------------------------------------------------

test_that("training_fn override is used for ag_sequential models", {
  skip_if_no_mlr3()
  set.seed(3)
  task <- mlr3::tsk("iris")

  called <- FALSE
  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 5L
  learner$param_set$values$batch_size <- 16L
  learner$model_fn <- ag_classif_builder
  learner$training_fn <- function(model, x, y, pars) {
    called <<- TRUE
    expect_s3_class(model, "ag_sequential")
    expect_equal(nrow(x), task$nrow)
    expect_equal(ncol(y), length(task$class_names))
    # do nothing else; return the untrained model
    model
  }

  learner$train(task)
  expect_true(called)
  # predict still works on the (untrained) model
  pred <- learner$predict(task)
  expect_true(is.factor(pred$response))
})

# ---------------------------------------------------------------------------
# sgd optimizer path
# ---------------------------------------------------------------------------

test_that("autograd tradepath honours optimizer = 'sgd'", {
  skip_if_no_mlr3()
  set.seed(4)
  task <- mlr3::tsk("mtcars")

  learner <- mlr3::lrn("regr.ggml")
  learner$param_set$values$epochs        <- 20L
  learner$param_set$values$batch_size    <- 8L
  learner$param_set$values$optimizer     <- "sgd"
  learner$param_set$values$learning_rate <- 0.001
  learner$model_fn <- ag_regr_builder

  learner$train(task)
  pred <- learner$predict(task)
  expect_true(all(is.finite(pred$response)))
})

# ---------------------------------------------------------------------------
# Marshal / unmarshal of autograd learner models (M2 state dict)
# ---------------------------------------------------------------------------

test_that("autograd classif learner marshal/unmarshal preserves predictions", {
  skip_if_no_mlr3()
  set.seed(5)
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs        <- 15L
  learner$param_set$values$batch_size    <- 16L
  learner$param_set$values$learning_rate <- 0.05
  learner$model_fn     <- ag_classif_builder
  learner$predict_type <- "prob"
  learner$train(task)

  expect_s3_class(learner$model$model, "ag_sequential")
  p_before <- learner$predict(task)$prob

  expect_false(learner$marshaled)
  learner$marshal()
  expect_true(learner$marshaled)
  expect_s3_class(learner$model, "classif_ggml_model_marshaled")

  learner$unmarshal()
  expect_false(learner$marshaled)
  expect_s3_class(learner$model$model, "ag_sequential")

  p_after <- learner$predict(task)$prob
  expect_equal(dim(p_before), dim(p_after))
  expect_lt(max(abs(p_before - p_after)), 1e-5)
})

test_that("autograd regr learner marshal/unmarshal preserves predictions", {
  skip_if_no_mlr3()
  set.seed(6)
  task <- mlr3::tsk("mtcars")

  learner <- mlr3::lrn("regr.ggml")
  learner$param_set$values$epochs        <- 20L
  learner$param_set$values$batch_size    <- 8L
  learner$param_set$values$learning_rate <- 0.01
  learner$model_fn <- ag_regr_builder
  learner$train(task)

  r_before <- learner$predict(task)$response
  learner$marshal()
  expect_true(learner$marshaled)
  learner$unmarshal()
  r_after <- learner$predict(task)$response

  expect_lt(max(abs(r_before - r_after)), 1e-5)
})

# ---------------------------------------------------------------------------
# Observation weights are NOT silently ignored by the autograd tradepath
# ---------------------------------------------------------------------------

test_that("autograd tradepath warns when task carries observation weights", {
  skip_if_no_mlr3()
  set.seed(1)

  d <- data.frame(
    x1 = rnorm(30), x2 = rnorm(30),
    y  = factor(rep(c("a", "b"), each = 15)),
    w  = runif(30, 0.5, 2.0)
  )
  task <- mlr3::as_task_classif(d, target = "y")
  task$set_col_roles("w", roles = "weights_learner")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs        <- 2L
  learner$param_set$values$batch_size    <- 10L
  learner$param_set$values$learning_rate <- 0.05
  learner$model_fn <- ag_classif_builder   # forces the autograd tradepath

  expect_warning(learner$train(task), "weights are ignored")
})

test_that("autograd tradepath does NOT warn when task has no weights", {
  skip_if_no_mlr3()
  set.seed(1)
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs        <- 2L
  learner$param_set$values$batch_size    <- 16L
  learner$param_set$values$learning_rate <- 0.05
  learner$model_fn <- ag_classif_builder

  expect_no_warning(learner$train(task))
})
