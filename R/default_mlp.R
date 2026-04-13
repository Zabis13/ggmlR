#' Default MLP builder for classification and regression
#'
#' Constructs an uncompiled sequential multi-layer perceptron suitable as a
#' starting point for tabular classification or regression. This is the default
#' \code{model_fn} used by \code{LearnerClassifGGML} and \code{LearnerRegrGGML}
#' when the user does not supply a custom builder, and it is also exported for
#' direct use or as a template for user-defined builders.
#'
#' The returned model is \strong{not compiled}: the caller is responsible for
#' calling \code{\link{ggml_compile}} with the appropriate loss
#' (\code{"categorical_crossentropy"} for classification, \code{"mse"} for
#' regression) before training.
#'
#' The final layer is chosen based on \code{task_type}:
#' \itemize{
#'   \item \code{"classif"} — dense with \code{units = n_out} and softmax activation.
#'   \item \code{"regr"}    — dense with \code{units = n_out} and no activation
#'     (identity / linear output).
#' }
#'
#' @param n_features Integer. Number of input features. Required.
#' @param n_out Integer. Number of output units. For classification this is the
#'   number of classes; for regression this is typically 1.
#' @param task_type Character. One of \code{"classif"} or \code{"regr"}. Controls
#'   the final layer's activation.
#' @param hidden_layers Integer vector. Widths of the hidden dense layers.
#'   Default \code{c(128L, 64L)}. Pass \code{integer(0)} for a linear model.
#' @param activation Character. Activation applied to each hidden layer.
#'   Default \code{"relu"}. Passed through to \code{\link{ggml_layer_dense}}.
#' @param dropout Numeric in \code{[0, 1)}. Dropout rate applied after each
#'   hidden layer. Set to \code{0} to disable dropout. Default \code{0.2}.
#'
#' @return An uncompiled \code{ggml_sequential_model} object. Call
#'   \code{\link{ggml_compile}} before \code{\link{ggml_fit}}.
#'
#' @seealso \code{\link{ggml_model_sequential}}, \code{\link{ggml_layer_dense}},
#'   \code{\link{ggml_layer_dropout}}, \code{\link{ggml_compile}}
#'
#' @examples
#' \dontrun{
#' # 3-class classifier on 20 features
#' model <- ggml_default_mlp(
#'   n_features   = 20L,
#'   n_out        = 3L,
#'   task_type    = "classif",
#'   hidden_layers = c(64L, 32L),
#'   dropout      = 0.1
#' )
#' model <- ggml_compile(model, optimizer = "adam",
#'                       loss = "categorical_crossentropy")
#'
#' # Single-output regressor
#' reg <- ggml_default_mlp(
#'   n_features = 10L,
#'   n_out      = 1L,
#'   task_type  = "regr"
#' )
#' reg <- ggml_compile(reg, optimizer = "adam", loss = "mse")
#' }
#' @export
ggml_default_mlp <- function(n_features,
                             n_out,
                             task_type = c("classif", "regr"),
                             hidden_layers = c(128L, 64L),
                             activation = "relu",
                             dropout = 0.2) {
  task_type <- match.arg(task_type)

  n_features <- as.integer(n_features)
  n_out      <- as.integer(n_out)
  if (length(n_features) != 1L || is.na(n_features) || n_features < 1L) {
    stop("`n_features` must be a single positive integer.")
  }
  if (length(n_out) != 1L || is.na(n_out) || n_out < 1L) {
    stop("`n_out` must be a single positive integer.")
  }
  if (length(hidden_layers) > 0L) {
    hidden_layers <- as.integer(hidden_layers)
    if (any(is.na(hidden_layers)) || any(hidden_layers < 1L)) {
      stop("`hidden_layers` must contain positive integers.")
    }
  }
  dropout <- as.double(dropout)
  if (length(dropout) != 1L || is.na(dropout) || dropout < 0 || dropout >= 1) {
    stop("`dropout` must be a single numeric in [0, 1).")
  }

  # For regression, pass NULL (identity) — ggml's nn_apply_activation treats
  # NULL as no-op. The string "linear" is NOT recognised.
  final_activation <- if (task_type == "classif") "softmax" else NULL

  model <- ggml_model_sequential()

  if (length(hidden_layers) == 0L) {
    model <- ggml_layer_dense(
      model,
      units       = n_out,
      activation  = final_activation,
      input_shape = n_features
    )
    return(model)
  }

  model <- ggml_layer_dense(
    model,
    units       = hidden_layers[1L],
    activation  = activation,
    input_shape = n_features
  )
  if (dropout > 0) {
    model <- ggml_layer_dropout(model, rate = dropout)
  }

  for (units in hidden_layers[-1L]) {
    model <- ggml_layer_dense(model, units = units, activation = activation)
    if (dropout > 0) {
      model <- ggml_layer_dropout(model, rate = dropout)
    }
  }

  model <- ggml_layer_dense(model, units = n_out, activation = final_activation)
  model
}
