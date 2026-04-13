# ============================================================================
# tidymodels / parsnip integration
#
# Registers "ggml" as an engine for parsnip::mlp() in both classification
# and regression modes. All registration happens lazily from .onLoad()
# via make_mlp_ggml(), so parsnip remains in Suggests only.
#
# Interface: "matrix" — parsnip passes numeric x matrix + y factor/numeric.
# Encoding:  predictor_indicators = "one_hot" — parsnip encodes factors for us.
# ============================================================================

# ── Registration function (called from .onLoad) ─────────────────────────────

make_mlp_ggml <- function() {

  # --- engine + dependency ------------------------------------------------
  parsnip::set_model_engine("mlp", mode = "classification", eng = "ggml")
  parsnip::set_model_engine("mlp", mode = "regression",     eng = "ggml")
  parsnip::set_dependency("mlp",   eng = "ggml", pkg = "ggmlR")

  # --- argument mapping ---------------------------------------------------
  # parsnip name    → ggmlR name
  parsnip::set_model_arg("mlp", "ggml", "hidden_units", "hidden_layers",
                         has_submodel = FALSE,
                         func = list(pkg = "dials", fun = "hidden_units"))
  parsnip::set_model_arg("mlp", "ggml", "epochs", "epochs",
                         has_submodel = FALSE,
                         func = list(pkg = "dials", fun = "epochs"))
  parsnip::set_model_arg("mlp", "ggml", "dropout", "dropout",
                         has_submodel = FALSE,
                         func = list(pkg = "dials", fun = "dropout"))
  parsnip::set_model_arg("mlp", "ggml", "activation", "activation",
                         has_submodel = FALSE,
                         func = list(pkg = "dials", fun = "activation"))
  parsnip::set_model_arg("mlp", "ggml", "learn_rate", "learn_rate",
                         has_submodel = FALSE,
                         func = list(pkg = "dials", fun = "learn_rate"))

  # --- fit ----------------------------------------------------------------
  parsnip::set_fit(
    model = "mlp",
    eng   = "ggml",
    mode  = "classification",
    value = list(
      interface = "matrix",
      protect   = c("x", "y"),
      func      = c(pkg = "ggmlR", fun = "ggmlr_parsnip_fit_classif"),
      defaults  = list()
    )
  )

  parsnip::set_fit(
    model = "mlp",
    eng   = "ggml",
    mode  = "regression",
    value = list(
      interface = "matrix",
      protect   = c("x", "y"),
      func      = c(pkg = "ggmlR", fun = "ggmlr_parsnip_fit_regr"),
      defaults  = list()
    )
  )

  # --- encoding -----------------------------------------------------------
  parsnip::set_encoding(
    model                = "mlp",
    eng                  = "ggml",
    mode                 = "classification",
    options              = list(
      predictor_indicators = "one_hot",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  parsnip::set_encoding(
    model                = "mlp",
    eng                  = "ggml",
    mode                 = "regression",
    options              = list(
      predictor_indicators = "one_hot",
      compute_intercept    = FALSE,
      remove_intercept     = FALSE,
      allow_sparse_x       = FALSE
    )
  )

  # --- predict: classification --------------------------------------------
  parsnip::set_pred(
    model = "mlp",
    eng   = "ggml",
    mode  = "classification",
    type  = "class",
    value = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object   = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type     = "class"
      )
    )
  )

  parsnip::set_pred(
    model = "mlp",
    eng   = "ggml",
    mode  = "classification",
    type  = "prob",
    value = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object   = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type     = "prob"
      )
    )
  )

  # --- predict: regression ------------------------------------------------
  parsnip::set_pred(
    model = "mlp",
    eng   = "ggml",
    mode  = "regression",
    type  = "numeric",
    value = list(
      pre  = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object   = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type     = "numeric"
      )
    )
  )
}

# ── Internal helpers ────────────────────────────────────────────────────────

# One-shot callback that sets the optimizer learning rate at the start of the
# first epoch. Works for both adam and sgd optimizers — we update the slot
# matching `optimizer`. Internal; not exported.
.ggmlr_parsnip_lr_callback <- function(lr, optimizer = "adam") {
  list(
    on_epoch_begin = function(epoch, logs, state) {
      if (epoch == 1L) {
        adamw_lr <- if (identical(optimizer, "adam")) as.numeric(lr) else as.numeric(NA)
        sgd_lr   <- if (identical(optimizer, "sgd"))  as.numeric(lr) else as.numeric(NA)
        .Call("R_ggml_opt_set_lr", state$lr_ud, adamw_lr, sgd_lr)
      }
      invisible(NULL)
    }
  )
}

# ── Fit wrappers (called by parsnip) ────────────────────────────────────────

#' parsnip ggml engine: classification fit
#'
#' Internal fit wrapper called by parsnip when `mlp()` is used with
#' `engine = "ggml"` in classification mode. Not intended for direct use.
#'
#' @param x Numeric matrix of predictors.
#' @param y Factor of class labels.
#' @param hidden_layers Integer vector of hidden layer widths.
#' @param epochs Number of training epochs.
#' @param dropout Dropout rate in `[0, 1)`.
#' @param activation Hidden activation, e.g. `"relu"`.
#' @param learn_rate Optional learning rate (applied via callback).
#' @param batch_size Minibatch size.
#' @param verbose Verbosity level (0/1/2).
#' @param validation_split Fraction in `[0, 1)` for validation.
#' @param callbacks List of ggmlR callbacks.
#' @param optimizer One of `"adam"`, `"sgd"`.
#' @param backend One of `"auto"`, `"cpu"`, `"vulkan"`.
#' @param ... Unused.
#' @return A fitted `ggmlr_parsnip_model` object.
#' @keywords internal
#' @export
ggmlr_parsnip_fit_classif <- function(x, y,
                                      hidden_layers = c(128L, 64L),
                                      epochs = 10L,
                                      dropout = 0.2,
                                      activation = "relu",
                                      learn_rate = NULL,
                                      batch_size = 32L,
                                      verbose = 0L,
                                      validation_split = 0.0,
                                      callbacks = list(),
                                      optimizer = "adam",
                                      backend = "auto",
                                      ...) {
  if (!is.null(learn_rate)) {
    callbacks <- c(callbacks,
                   list(.ggmlr_parsnip_lr_callback(learn_rate, optimizer)))
  }

  n_features <- ncol(x)
  class_names <- levels(y)
  n_out <- length(class_names)

  # One-hot encode y
  y_int <- as.integer(y)
  y_mat <- matrix(0, nrow = nrow(x), ncol = n_out)
  y_mat[cbind(seq_len(nrow(x)), y_int)] <- 1

  model <- ggml_default_mlp(
    n_features    = n_features,
    n_out         = n_out,
    task_type     = "classif",
    hidden_layers = as.integer(hidden_layers),
    activation    = activation,
    dropout       = dropout
  )

  if (identical(backend, "gpu")) backend <- "vulkan"
  model <- ggml_compile(model,
                        optimizer = optimizer,
                        loss      = "categorical_crossentropy",
                        backend   = backend)

  model <- ggml_fit(
    model,
    x                = x,
    y                = y_mat,
    epochs           = as.integer(epochs),
    batch_size       = as.integer(batch_size),
    validation_split = validation_split,
    verbose          = as.integer(verbose),
    callbacks        = callbacks
  )

  out <- list(
    model       = model,
    class_names = class_names,
    n_features  = n_features,
    mode        = "classification"
  )
  class(out) <- "ggmlr_parsnip_model"
  out
}

#' parsnip ggml engine: regression fit
#'
#' Internal fit wrapper called by parsnip when `mlp()` is used with
#' `engine = "ggml"` in regression mode. Not intended for direct use.
#'
#' @inheritParams ggmlr_parsnip_fit_classif
#' @param y Numeric response vector.
#' @return A fitted `ggmlr_parsnip_model` object.
#' @keywords internal
#' @export
ggmlr_parsnip_fit_regr <- function(x, y,
                                   hidden_layers = c(128L, 64L),
                                   epochs = 10L,
                                   dropout = 0.2,
                                   activation = "relu",
                                   learn_rate = NULL,
                                   batch_size = 32L,
                                   verbose = 0L,
                                   validation_split = 0.0,
                                   callbacks = list(),
                                   optimizer = "adam",
                                   backend = "auto",
                                   ...) {
  if (!is.null(learn_rate)) {
    callbacks <- c(callbacks,
                   list(.ggmlr_parsnip_lr_callback(learn_rate, optimizer)))
  }

  n_features <- ncol(x)

  model <- ggml_default_mlp(
    n_features    = n_features,
    n_out         = 1L,
    task_type     = "regr",
    hidden_layers = as.integer(hidden_layers),
    activation    = activation,
    dropout       = dropout
  )

  if (identical(backend, "gpu")) backend <- "vulkan"
  model <- ggml_compile(model,
                        optimizer = optimizer,
                        loss      = "mse",
                        backend   = backend)

  y_mat <- matrix(as.double(y), ncol = 1L)

  model <- ggml_fit(
    model,
    x                = x,
    y                = y_mat,
    epochs           = as.integer(epochs),
    batch_size       = as.integer(batch_size),
    validation_split = validation_split,
    verbose          = as.integer(verbose),
    callbacks        = callbacks
  )

  out <- list(
    model      = model,
    n_features = n_features,
    mode       = "regression"
  )
  class(out) <- "ggmlr_parsnip_model"
  out
}

# ── Predict method ──────────────────────────────────────────────────────────

#' @keywords internal
#' @export
predict.ggmlr_parsnip_model <- function(object, new_data, type = "class", ...) {
  x <- as.matrix(new_data)
  storage.mode(x) <- "double"

  raw <- ggml_predict(object$model, x)
  if (!is.matrix(raw)) {
    raw <- matrix(raw, nrow = nrow(x))
  }

  if (object$mode == "classification") {
    if (type == "prob") {
      colnames(raw) <- object$class_names
      out <- as.data.frame(raw)
      names(out) <- paste0(".pred_", object$class_names)
      tibble::as_tibble(out)
    } else {
      idx <- max.col(raw, ties.method = "first")
      tibble::tibble(.pred_class = factor(object$class_names[idx],
                                          levels = object$class_names))
    }
  } else {
    tibble::tibble(.pred = as.double(raw[, 1L]))
  }
}
