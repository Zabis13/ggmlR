# Tests for mlr3 integration: LearnerClassifGGML / LearnerRegrGGML,
# ggml_default_mlp(), marshal roundtrip, resampling.

if (!requireNamespace("mlr3",    quietly = TRUE) ||
    !requireNamespace("paradox", quietly = TRUE) ||
    !requireNamespace("R6",      quietly = TRUE)) {
  testthat::skip("mlr3 / paradox / R6 not available")
}

# In the R CMD check test process mlr3 may load before or after ggmlR,
# so the setHook callbacks in .onLoad may not have fired yet.
# Force registration explicitly — .register_mlr3() is idempotent.
library(mlr3)
ggmlR:::.register_mlr3()

skip_if_no_mlr3 <- function() {
  skip_if_not_installed("mlr3")
  skip_if_not_installed("paradox")
  skip_if_not_installed("R6")
}

suppress_mlr3_output <- function(expr) {
  if (requireNamespace("lgr", quietly = TRUE)) {
    old <- lgr::get_logger("mlr3")$threshold
    lgr::get_logger("mlr3")$set_threshold("warn")
    on.exit(lgr::get_logger("mlr3")$set_threshold(old), add = TRUE)
  }
  suppressMessages(expr)
}

# ---------------------------------------------------------------------------
# ggml_default_mlp
# ---------------------------------------------------------------------------

test_that("ggml_default_mlp builds a classif MLP with softmax head", {
  m <- ggml_default_mlp(
    n_features   = 4L,
    n_out        = 3L,
    task_type    = "classif",
    hidden_layers = c(16L, 8L),
    dropout      = 0.1
  )
  expect_s3_class(m, "ggml_sequential_model")
  expect_false(m$compiled)
  # dense + dropout + dense + dropout + dense(softmax) = 5 layers
  expect_equal(length(m$layers), 5L)
  last <- m$layers[[length(m$layers)]]
  expect_equal(last$type, "dense")
  expect_equal(last$config$activation, "softmax")
  expect_equal(last$config$units, 3L)
})

test_that("ggml_default_mlp builds a regr MLP with linear head", {
  m <- ggml_default_mlp(
    n_features   = 10L,
    n_out        = 1L,
    task_type    = "regr",
    hidden_layers = c(8L),
    dropout      = 0
  )
  expect_s3_class(m, "ggml_sequential_model")
  # dense + dense(identity) = 2 layers (dropout=0 → no dropout layer)
  expect_equal(length(m$layers), 2L)
  last <- m$layers[[2L]]
  expect_null(last$config$activation)
  expect_equal(last$config$units, 1L)
})

test_that("ggml_default_mlp with empty hidden_layers yields a linear model", {
  m <- ggml_default_mlp(
    n_features    = 5L,
    n_out         = 2L,
    task_type     = "classif",
    hidden_layers = integer(0)
  )
  expect_equal(length(m$layers), 1L)
  expect_equal(m$layers[[1L]]$config$activation, "softmax")
})

test_that("ggml_default_mlp validates its arguments", {
  expect_error(ggml_default_mlp(n_features = 0L, n_out = 2L),
               "n_features")
  expect_error(ggml_default_mlp(n_features = 4L, n_out = 0L),
               "n_out")
  expect_error(ggml_default_mlp(n_features = 4L, n_out = 2L, dropout = 1),
               "dropout")
})

# ---------------------------------------------------------------------------
# LearnerClassifGGML
# ---------------------------------------------------------------------------

test_that("LearnerClassifGGML constructs and has the expected contract", {
  skip_if_no_mlr3()
  learner <- mlr3::lrn("classif.ggml")

  expect_s3_class(learner, "LearnerClassifGGML")
  expect_s3_class(learner, "LearnerClassif")
  expect_equal(learner$id, "classif.ggml")
  expect_setequal(learner$predict_types, c("response", "prob"))
  expect_equal(learner$feature_types, "numeric")
  expect_true(all(c("multiclass", "twoclass", "marshal", "weights")
                  %in% learner$properties))
  expect_true("ggmlR" %in% learner$packages)
  expect_null(learner$model_fn)
})

test_that("LearnerClassifGGML trains and predicts on iris (response + prob)", {
  skip_if_no_mlr3()
  set.seed(42)
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 5L
  learner$param_set$values$batch_size <- 10L
  learner$param_set$values$backend    <- "cpu"
  learner$predict_type <- "prob"

  learner$train(task)
  expect_s3_class(learner$model, "classif_ggml_model")
  expect_equal(learner$model$class_names, task$class_names)
  expect_equal(learner$model$n_features, length(task$feature_names))

  pred <- learner$predict(task)
  expect_s3_class(pred, "PredictionClassif")
  expect_equal(length(pred$response), task$nrow)
  expect_true(is.factor(pred$response))
  expect_equal(levels(pred$response), task$class_names)

  # prob matrix sums to ~1 per row
  expect_true(all(abs(rowSums(pred$prob) - 1) < 1e-4))
})

test_that("LearnerClassifGGML response-only predict_type works", {
  skip_if_no_mlr3()
  set.seed(42)
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 3L
  learner$param_set$values$batch_size <- 10L
  learner$param_set$values$backend    <- "cpu"
  # predict_type defaults to "response"

  learner$train(task)
  pred <- learner$predict(task)
  expect_true(is.factor(pred$response))
})

test_that("LearnerClassifGGML accepts a user-supplied model_fn", {
  skip_if_no_mlr3()
  set.seed(42)
  task <- mlr3::tsk("iris")

  called <- FALSE
  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 3L
  learner$param_set$values$batch_size <- 10L
  learner$param_set$values$backend    <- "cpu"
  learner$model_fn <- function(task, n_features, n_out, pars) {
    called <<- TRUE
    expect_equal(n_features, 4L)
    expect_equal(n_out, 3L)
    ggml_model_sequential() |>
      ggml_layer_dense(8L, activation = "relu", input_shape = n_features) |>
      ggml_layer_dense(n_out, activation = "softmax")
  }

  learner$train(task)
  expect_true(called)
  expect_s3_class(learner$model, "classif_ggml_model")
})

test_that("LearnerClassifGGML rejects an unsupported model_fn return", {
  skip_if_no_mlr3()
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 1L
  learner$param_set$values$batch_size <- 10L
  learner$param_set$values$backend    <- "cpu"
  learner$model_fn <- function(task, n_features, n_out, pars) {
    list(not = "a model")
  }

  expect_error(learner$train(task), "sequential or")
})

test_that("LearnerClassifGGML honours observation weights", {
  skip_if_no_mlr3()
  set.seed(42)

  # Build a small numeric task with weights column
  d <- data.frame(
    x1 = rnorm(30),
    x2 = rnorm(30),
    y  = factor(rep(c("a", "b"), each = 15)),
    w  = runif(30, 0.5, 2.0)
  )
  task <- mlr3::as_task_classif(d, target = "y")
  task$set_col_roles("w", roles = "weights_learner")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 2L
  learner$param_set$values$batch_size <- 10L
  learner$param_set$values$backend    <- "cpu"

  expect_no_error(learner$train(task))
  expect_s3_class(learner$model, "classif_ggml_model")
})

# ---------------------------------------------------------------------------
# LearnerRegrGGML
# ---------------------------------------------------------------------------

test_that("LearnerRegrGGML constructs and has the expected contract", {
  skip_if_no_mlr3()
  learner <- mlr3::lrn("regr.ggml")

  expect_s3_class(learner, "LearnerRegrGGML")
  expect_s3_class(learner, "LearnerRegr")
  expect_equal(learner$id, "regr.ggml")
  expect_equal(learner$predict_types, "response")
  expect_equal(learner$feature_types, "numeric")
  expect_true("marshal" %in% learner$properties)
})

test_that("LearnerRegrGGML trains and predicts on mtcars", {
  skip_if_no_mlr3()
  set.seed(42)
  task <- mlr3::tsk("mtcars")

  learner <- mlr3::lrn("regr.ggml")
  learner$param_set$values$epochs     <- 5L
  learner$param_set$values$batch_size <- 8L
  learner$param_set$values$backend    <- "cpu"

  learner$train(task)
  expect_s3_class(learner$model, "regr_ggml_model")

  pred <- learner$predict(task)
  expect_s3_class(pred, "PredictionRegr")
  expect_equal(length(pred$response), task$nrow)
  expect_true(all(is.finite(pred$response)))
})

# ---------------------------------------------------------------------------
# Dictionary registration
# ---------------------------------------------------------------------------

test_that("learners are registered in mlr_learners", {
  skip_if_no_mlr3()
  keys <- mlr3::mlr_learners$keys()
  expect_true("classif.ggml" %in% keys)
  expect_true("regr.ggml"    %in% keys)

  learner <- mlr3::lrn("classif.ggml")
  expect_s3_class(learner, "LearnerClassifGGML")
})

# ---------------------------------------------------------------------------
# Marshal roundtrip
# ---------------------------------------------------------------------------

test_that("ggml_marshal_model / ggml_unmarshal_model roundtrip on sequential", {
  set.seed(42)
  m <- ggml_default_mlp(n_features = 4L, n_out = 3L, task_type = "classif",
                        hidden_layers = c(8L))
  m <- ggml_compile(m, optimizer = "adam",
                    loss = "categorical_crossentropy", backend = "cpu")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 3); y[cbind(1:10, sample(1:3, 10, replace = TRUE))] <- 1
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = 0L)

  blob <- ggml_marshal_model(m)
  expect_s3_class(blob, "ggmlR_marshaled")
  expect_equal(blob$format, "ggmlR.marshal")
  expect_equal(blob$version, 1L)
  expect_equal(blob$api, "sequential")
  expect_true(is.raw(blob$payload))
  expect_gt(length(blob$payload), 0L)

  m2 <- ggml_unmarshal_model(blob, backend = "cpu")
  expect_s3_class(m2, "ggml_sequential_model")
  expect_true(m2$compiled)

  # predictions should match bitwise-close (same weights, same backend)
  p1 <- ggml_predict(m,  x)
  p2 <- ggml_predict(m2, x)
  expect_equal(dim(p1), dim(p2))
  expect_lt(max(abs(p1 - p2)), 1e-5)
})

test_that("ggml_unmarshal_model detects a corrupted payload via sha256", {
  skip_if_not_installed("digest")

  m <- ggml_default_mlp(n_features = 3L, n_out = 2L, task_type = "classif",
                        hidden_layers = c(4L))
  m <- ggml_compile(m, optimizer = "adam",
                    loss = "categorical_crossentropy", backend = "cpu")
  blob <- ggml_marshal_model(m)

  # Flip one byte in the payload
  blob$payload[1L] <- as.raw(bitwXor(as.integer(blob$payload[1L]), 0xFFL))
  expect_error(ggml_unmarshal_model(blob, backend = "cpu"),
               "checksum mismatch")
})

test_that("learner $marshal() / $unmarshal() roundtrip on classif", {
  skip_if_no_mlr3()
  set.seed(42)
  task <- mlr3::tsk("iris")

  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 3L
  learner$param_set$values$batch_size <- 10L
  learner$param_set$values$backend    <- "cpu"
  learner$predict_type <- "prob"
  learner$train(task)

  p_before <- learner$predict(task)$prob

  expect_false(learner$marshaled)
  learner$marshal()
  expect_true(learner$marshaled)
  expect_s3_class(learner$model, "classif_ggml_model_marshaled")

  learner$unmarshal()
  expect_false(learner$marshaled)
  expect_s3_class(learner$model, "classif_ggml_model")

  p_after <- learner$predict(task)$prob
  expect_equal(dim(p_before), dim(p_after))
  expect_lt(max(abs(p_before - p_after)), 1e-5)
})

test_that("learner $marshal() / $unmarshal() roundtrip on regr", {
  skip_if_no_mlr3()
  set.seed(42)
  task <- mlr3::tsk("mtcars")

  learner <- mlr3::lrn("regr.ggml")
  learner$param_set$values$epochs     <- 3L
  learner$param_set$values$batch_size <- 8L
  learner$param_set$values$backend    <- "cpu"
  learner$train(task)

  r_before <- learner$predict(task)$response
  learner$marshal()
  expect_true(learner$marshaled)
  learner$unmarshal()
  r_after <- learner$predict(task)$response

  expect_lt(max(abs(r_before - r_after)), 1e-5)
})

# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

test_that("classif.ggml works with 3-fold CV resampling", {
  skip_if_no_mlr3()
  set.seed(42)

  task <- mlr3::tsk("iris")
  learner <- mlr3::lrn("classif.ggml")
  learner$param_set$values$epochs     <- 3L
  learner$param_set$values$batch_size <- 10L
  learner$param_set$values$backend    <- "cpu"

  rr <- suppress_mlr3_output(
    mlr3::resample(task, learner, mlr3::rsmp("cv", folds = 3L))
  )
  acc <- rr$aggregate(mlr3::msr("classif.acc"))
  expect_true(is.numeric(acc))
  expect_true(acc >= 0 && acc <= 1)
})

test_that("regr.ggml works with 3-fold CV resampling", {
  skip_if_no_mlr3()
  set.seed(42)

  task <- mlr3::tsk("mtcars")
  learner <- mlr3::lrn("regr.ggml")
  learner$param_set$values$epochs     <- 3L
  learner$param_set$values$batch_size <- 8L
  learner$param_set$values$backend    <- "cpu"

  rr <- suppress_mlr3_output(
    mlr3::resample(task, learner, mlr3::rsmp("cv", folds = 3L))
  )
  rmse <- rr$aggregate(mlr3::msr("regr.rmse"))
  expect_true(is.numeric(rmse))
  expect_true(is.finite(rmse))
})
