#' @name LearnerRegrGGML
#' @title mlr3 Regression Learner for ggmlR
#'
#' @description
#' An \code{\link[mlr3:LearnerRegr]{mlr3::LearnerRegr}} that trains and
#' predicts with a ggmlR sequential neural network. The network architecture is
#' provided by the user through the \code{model_fn} field; if left \code{NULL},
#' a default MLP built by \code{\link{ggml_default_mlp}} is used.
#'
#' @section Feature types:
#' Only \code{numeric} features are supported. Convert factors with an mlr3
#' pipeline step such as \code{mlr3pipelines::po("encode")} before passing the
#' task to this learner.
#'
#' @section The \code{model_fn} field:
#' A function with signature
#' \preformatted{function(task, n_features, n_out, pars) -> ggml_sequential_model}
#' where \code{task} is the training \code{\link[mlr3:TaskRegr]{TaskRegr}},
#' \code{n_features} is the number of numeric features, \code{n_out} is always
#' \code{1L} for this learner, and \code{pars} is a named list of the learner's
#' current parameter values. The function must return an \strong{uncompiled}
#' sequential model; the learner will call \code{\link{ggml_compile}} with the
#' correct loss (\code{"mse"}).
#'
#' The \code{model_fn} is stored outside the ParamSet (R6/paradox do not
#' serialize functions), so set it via \code{learner$model_fn <- ...} after
#' construction.
#'
#' @section Parameters:
#' \describe{
#'   \item{\code{epochs}}{Integer, \eqn{\ge 1}. Number of training epochs. Default 10.}
#'   \item{\code{batch_size}}{Integer, \eqn{\ge 1}. Minibatch size. Default 32.}
#'   \item{\code{optimizer}}{Character, one of \code{"adam"}, \code{"sgd"}. Default \code{"adam"}.}
#'   \item{\code{validation_split}}{Numeric in \eqn{[0, 1)}. Default 0.}
#'   \item{\code{verbose}}{Integer in \eqn{\{0, 1, 2\}}. Default 0 (silent).}
#'   \item{\code{backend}}{Character, one of \code{"auto"}, \code{"cpu"}, \code{"gpu"}. Default \code{"auto"}.}
#'   \item{\code{hidden_layers}}{Integer vector. Widths of hidden layers used by
#'     the default \code{model_fn}. Default \code{c(128, 64)}. Ignored if
#'     \code{model_fn} is user-supplied and does not read it.}
#'   \item{\code{activation}}{Character. Hidden-layer activation for the default
#'     \code{model_fn}. Default \code{"relu"}. Any activation string accepted by
#'     \code{\link{ggml_layer_dense}} is allowed.}
#'   \item{\code{dropout}}{Numeric in \eqn{[0, 1)}. Dropout rate for the default
#'     \code{model_fn}. Default \code{0.2}.}
#'   \item{\code{callbacks}}{List of ggmlR callbacks (e.g.
#'     \code{ggml_callback_early_stopping()}) passed through to
#'     \code{\link{ggml_fit}}. Default empty list. Only honoured for sequential
#'     models; functional-API fit silently ignores callbacks.}
#' }
#'
#' @section Predict types:
#' Supports \code{"response"} only.
#'
#' @examples
#' \dontrun{
#' if (requireNamespace("mlr3", quietly = TRUE)) {
#'   library(mlr3)
#'   task <- tsk("mtcars")
#'
#'   learner <- LearnerRegrGGML$new()
#'   learner$param_set$values$epochs <- 50
#'   learner$param_set$values$batch_size <- 8
#'
#'   learner$train(task)
#'   pred <- learner$predict(task)
#'   pred$score(msr("regr.rmse"))
#' }
#' }
#'
#' @keywords internal
LearnerRegrGGML <- R6::R6Class(
  "LearnerRegrGGML",
  inherit = mlr3::LearnerRegr,
  public = list(

    #' @field model_fn Optional user-supplied model builder. See the
    #'   \dQuote{The \code{model_fn} field} section above. If \code{NULL}
    #'   (default), \code{\link{ggml_default_mlp}} is used.
    model_fn = NULL,

    #' @description Marshal \code{self$model} in place for transport to
    #'   parallel workers.
    #' @param ... Additional arguments forwarded to
    #'   \code{\link[mlr3]{marshal_model}}.
    marshal = function(...) {
      self$model <- mlr3::marshal_model(self$model, inplace = TRUE, ...)
      invisible(self)
    },

    #' @description Reverse a prior call to \code{$marshal()}.
    #' @param ... Additional arguments forwarded to
    #'   \code{\link[mlr3]{unmarshal_model}}.
    unmarshal = function(...) {
      self$model <- mlr3::unmarshal_model(self$model, inplace = TRUE, ...)
      invisible(self)
    },

    #' @description Create a new \code{LearnerRegrGGML}.
    initialize = function() {
      ps <- paradox::ps(
        epochs           = paradox::p_int(lower = 1L, default = 10L, tags = "train"),
        batch_size       = paradox::p_int(lower = 1L, default = 32L, tags = "train"),
        optimizer        = paradox::p_fct(levels = c("adam", "sgd"),
                                          default = "adam", tags = "train"),
        validation_split = paradox::p_dbl(lower = 0, upper = 1 - 1e-8,
                                          default = 0, tags = "train"),
        verbose          = paradox::p_int(lower = 0L, upper = 2L,
                                          default = 0L, tags = "train"),
        backend          = paradox::p_fct(levels = c("auto", "cpu", "gpu"),
                                          default = "auto", tags = "train"),
        hidden_layers    = paradox::p_uty(default = c(128L, 64L), tags = "train"),
        activation       = paradox::p_uty(default = "relu", tags = "train"),
        dropout          = paradox::p_dbl(lower = 0, upper = 1 - 1e-8,
                                          default = 0.2, tags = "train"),
        callbacks        = paradox::p_uty(default = list(), tags = "train")
      )

      ps$values <- list(
        epochs           = 10L,
        batch_size       = 32L,
        optimizer        = "adam",
        validation_split = 0,
        verbose          = 0L,
        backend          = "auto",
        hidden_layers    = c(128L, 64L),
        activation       = "relu",
        dropout          = 0.2,
        callbacks        = list()
      )

      super$initialize(
        id             = "regr.ggml",
        param_set      = ps,
        predict_types  = "response",
        feature_types  = "numeric",
        properties     = "marshal",
        packages       = c("mlr3", "ggmlR"),
        label          = "ggmlR Neural Network",
        man            = "ggmlR::LearnerRegrGGML"
      )
    }
  ),

  private = list(
    .train = function(task) {
      pars <- self$param_set$get_values(tags = "train")
      if (identical(pars$backend, "gpu")) pars$backend <- "vulkan"

      x <- as.matrix(task$data(cols = task$feature_names))
      storage.mode(x) <- "double"

      y <- matrix(as.double(task$truth()), ncol = 1L)

      builder <- self$model_fn %||% function(task, n_features, n_out, pars) {
        ggml_default_mlp(
          n_features    = n_features,
          n_out         = n_out,
          task_type     = "regr",
          hidden_layers = pars$hidden_layers,
          activation    = pars$activation,
          dropout       = pars$dropout
        )
      }

      model <- builder(task, ncol(x), 1L, pars)
      if (!inherits(model, c("ggml_sequential_model", "ggml_functional_model"))) {
        stop("`model_fn` must return an uncompiled ggml sequential or ",
             "functional model (got class: ",
             paste(class(model), collapse = "/"), ").")
      }

      model <- ggml_compile(
        model,
        optimizer = pars$optimizer,
        loss      = "mse",
        backend   = pars$backend
      )

      model <- ggml_fit(
        model,
        x                = x,
        y                = y,
        epochs           = pars$epochs,
        batch_size       = pars$batch_size,
        validation_split = pars$validation_split,
        verbose          = pars$verbose,
        callbacks        = pars$callbacks %||% list()
      )

      out <- list(
        model         = model,
        n_features    = ncol(x),
        feature_names = task$feature_names
      )
      class(out) <- c("regr_ggml_model", "list")
      out
    },

    .predict = function(task) {
      x <- as.matrix(task$data(cols = task$feature_names))
      storage.mode(x) <- "double"

      pred <- ggml_predict(self$model$model, x)
      response <- if (is.matrix(pred)) pred[, 1L] else as.numeric(pred)

      list(response = response)
    }
  ),

  active = list(
    #' @field marshaled Logical, read-only. \code{TRUE} if \code{self$model}
    #'   has been marshaled and not yet unmarshaled.
    marshaled = function(rhs) {
      if (!missing(rhs)) stop("`marshaled` is read-only.", call. = FALSE)
      mlr3::is_marshaled_model(self$model)
    }
  )
)

