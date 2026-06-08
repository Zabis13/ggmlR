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
#' @noRd
.make_LearnerRegrGGML <- function() R6::R6Class(
  "LearnerRegrGGML",
  inherit = mlr3::LearnerRegr,
  public = list(

    #' @field training_fn Optional user-supplied training loop for the
    #'   autograd tradepath. Only consulted when \code{model_fn} returns an
    #'   \code{ag_sequential} module. Signature
    #'   \code{function(model, x, y, pars) -> trained model}, where \code{x} is
    #'   the feature matrix \code{[rows, features]}, \code{y} is the target
    #'   matrix \code{[rows, 1]}, and \code{pars} is the learner's current
    #'   parameter list. If \code{NULL} (default), a built-in Adam/SGD loop is
    #'   used. Sequential and functional models ignore this field.
    training_fn = NULL,

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
        callbacks        = paradox::p_uty(default = list(), tags = "train"),
        # Autograd-only parameters (ignored by the sequential/functional
        # tradepath, where ggml_compile() owns the learning rate).
        learning_rate    = paradox::p_dbl(lower = 0, default = 1e-3,
                                          tags = "train"),
        max_grad_norm    = paradox::p_dbl(lower = 0, default = Inf,
                                          tags = "train"),
        # Reproducibility: fixes weight init, dropout masks and shuffling.
        # Unset = non-deterministic. See ggml_set_seed().
        seed             = paradox::p_int(tags = "train")
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
        callbacks        = list(),
        learning_rate    = 1e-3,
        max_grad_norm    = Inf
      )

      super$initialize(
        id             = "regr.ggml",
        param_set      = ps,
        predict_types  = "response",
        feature_types  = "numeric",
        properties     = c("marshal", "weights"),
        packages       = c("mlr3", "ggmlR"),
        label          = "ggmlR Neural Network",
        man            = "ggmlR::LearnerRegrGGML"
      )
    }
  ),

  private = list(
    # Read observation weights from a task, handling both current
    # (`weights_learner`) and legacy (`weights`) mlr3 API. Returns a numeric
    # vector aligned to the row order produced by `task$data()`, or NULL.
    .extract_weights = function(task) {
      wdt <- tryCatch(task$weights_learner, error = function(e) NULL)
      if (is.null(wdt) || nrow(wdt) == 0L) return(NULL)
      w <- wdt$weight[match(task$row_ids, wdt$row_id)]
      if (anyNA(w)) {
        stop("LearnerRegrGGML: task weights are missing for ",
             sum(is.na(w)), " of ", length(w), " training rows.",
             call. = FALSE)
      }
      as.double(w)
    },

    # Default autograd training loop (regression / MSE). Activated when
    # `model_fn` returns an ag_sequential module. `x` is [rows, features] and
    # `y` is [rows, 1]; autograd modules expect [features, batch], so both are
    # transposed. Returns the trained (in-place updated) ag_sequential module.
    .train_autograd = function(model, x, y, pars) {
      xt <- t(x)                       # [features, rows]
      yt <- t(y)                       # [1, rows]
      n  <- ncol(xt)
      bs <- min(pars$batch_size, n)

      params <- model$parameters()
      opt <- switch(pars$optimizer,
        sgd  = optimizer_sgd(params, lr = pars$learning_rate),
        adam = optimizer_adam(params, lr = pars$learning_rate),
        optimizer_adam(params, lr = pars$learning_rate)
      )

      ag_train(model)
      for (epoch in seq_len(pars$epochs)) {
        perm <- sample.int(n)
        for (start in seq(1L, n, by = bs)) {
          idx <- perm[start:min(start + bs - 1L, n)]
          xb  <- ag_tensor(xt[, idx, drop = FALSE])
          yb  <- yt[, idx, drop = FALSE]

          with_grad_tape({
            out  <- model$forward(xb)
            loss <- ag_mse_loss(out, yb)
          })
          grads <- backward(loss)
          if (is.finite(pars$max_grad_norm)) {
            clip_grad_norm(params, grads, pars$max_grad_norm)
          }
          opt$step(grads)
          opt$zero_grad()
        }
        if (pars$verbose > 0L) {
          cat(sprintf("[regr.ggml autograd] epoch %d/%d  loss = %.6f\n",
                      epoch, pars$epochs, as.numeric(.ag_data(loss))))
        }
      }
      ag_eval(model)
      model
    },

    .train = function(task) {
      pars <- self$param_set$get_values(tags = "train")
      # Reproducibility: fix RNG before any weight init / shuffling / dropout.
      ggml_set_seed(pars$seed)
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
      if (!inherits(model, c("ggml_sequential_model", "ggml_functional_model",
                             "ag_sequential"))) {
        stop("`model_fn` must return an uncompiled ggml sequential, ",
             "functional, or ag_sequential model (got class: ",
             paste(class(model), collapse = "/"), ").")
      }

      # Autograd tradepath: ag_sequential modules use an R-level training loop
      # (gradient tape + Adam/SGD) rather than ggml_compile()/ggml_fit().
      if (inherits(model, "ag_sequential")) {
        # Observation weights are only honoured by the sequential/functional
        # tradepath (via ggml_fit). The autograd loop does not apply them, so
        # warn loudly rather than silently ignoring task weights.
        if (!is.null(private$.extract_weights(task))) {
          warning("LearnerRegrGGML: task observation weights are ignored ",
                  "by the autograd tradepath (ag_sequential model_fn). They ",
                  "are only applied by the sequential/functional tradepath.",
                  call. = FALSE)
        }
        train_loop <- self$training_fn %||% private$.train_autograd
        model <- train_loop(model, x, y, pars)

        # Zero-arg rebuild closure for marshal (M2): captures only dims + pars,
        # NOT the task (so it serializes cheaply). User `model_fn`s that read
        # `task` are therefore not marshalable in the autograd tradepath.
        local_builder <- builder
        nf <- ncol(x); no <- 1L; pars_snap <- pars
        rebuild_fn <- function() local_builder(NULL, nf, no, pars_snap)

        out <- list(
          model         = model,
          n_features    = ncol(x),
          feature_names = task$feature_names,
          ag_rebuild_fn = rebuild_fn
        )
        class(out) <- c("regr_ggml_model", "list")
        return(out)
      }

      model <- ggml_compile(
        model,
        optimizer = pars$optimizer,
        loss      = "mse",
        backend   = pars$backend
      )

      sample_weight <- private$.extract_weights(task)

      fit_args <- list(
        model,
        x                = x,
        y                = y,
        epochs           = pars$epochs,
        batch_size       = pars$batch_size,
        validation_split = pars$validation_split,
        verbose          = pars$verbose,
        callbacks        = pars$callbacks %||% list()
      )
      if (!is.null(sample_weight)) {
        fit_args$sample_weight <- sample_weight
      }
      model <- do.call(ggml_fit, fit_args)

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

      model <- self$model$model

      if (inherits(model, "ag_sequential")) {
        ag_eval(model)
        out <- model$forward(ag_tensor(t(x)))   # [1, rows]
        response <- as.numeric(.ag_data(out))
        return(list(response = response))
      }

      pred <- ggml_predict(model, x)
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
