# Keras-compatible generic API for ggmlR
#
# Re-exports compile(), fit(), evaluate() generics from the `generics` package
# and registers S3 methods for ggml model classes.  predict() methods use the
# stats::predict generic.  All methods delegate to the existing
# ggml_compile(), ggml_fit(), ggml_evaluate(), ggml_predict() implementations.

#' @importFrom generics compile
#' @export
generics::compile

#' @importFrom generics fit
#' @export
generics::fit

#' @importFrom generics evaluate
#' @export
generics::evaluate

# ============================================================================
# compile()
# ============================================================================

#' Compile a Model
#'
#' Configures the model for training by setting the optimizer, loss function,
#' and metrics.  This is the keras-compatible interface; it delegates to
#' \code{\link{ggml_compile}}.
#'
#' @param object A model object (e.g. \code{ggml_sequential_model} or
#'   \code{ggml_functional_model}).
#' @param optimizer Character: \code{"adam"}, \code{"adamw"}, or \code{"sgd"}.
#' @param loss Character: \code{"categorical_crossentropy"} or \code{"mse"}.
#' @param metrics Character vector of metrics (default \code{"accuracy"}).
#' @param ... Additional arguments passed to \code{\link{ggml_compile}}.
#' @return The compiled model (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(10, activation = "softmax", input_shape = 4)
#' model <- compile(model, optimizer = "adam",
#'                  loss = "categorical_crossentropy")
#' }

#' @rdname compile
#' @export
compile.ggml_sequential_model <- function(object, optimizer = "adam",
                                           loss = "categorical_crossentropy",
                                           metrics = c("accuracy"), ...) {
  ggml_compile(object, optimizer = optimizer, loss = loss, metrics = metrics, ...)
}

#' @rdname compile
#' @export
compile.ggml_functional_model <- function(object, optimizer = "adam",
                                           loss = "categorical_crossentropy",
                                           metrics = c("accuracy"), ...) {
  ggml_compile(object, optimizer = optimizer, loss = loss, metrics = metrics, ...)
}

# ============================================================================
# fit()
# ============================================================================

#' Train a Model
#'
#' Trains the model on data for a fixed number of epochs.  This is the
#' keras-compatible interface; it delegates to \code{\link{ggml_fit}}.
#'
#' @param object A compiled model object.
#' @param x Training data.  Matrix, array, or list of matrices (multi-input).
#' @param y Training labels (matrix, one-hot encoded for classification).
#' @param epochs Number of training epochs (default 1).
#' @param batch_size Batch size (default 32).
#' @param validation_split Fraction of data for validation (default 0).
#' @param validation_data Optional \code{list(x_val, y_val)}.
#' @param verbose 0 = silent, 1 = progress (default 1).
#' @param callbacks List of callback objects (default \code{list()}).
#' @param ... Additional arguments passed to \code{\link{ggml_fit}}.
#' @return The trained model (invisibly), with \code{model$history}.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(10, activation = "softmax", input_shape = 4)
#' model <- compile(model, optimizer = "adam",
#'                  loss = "categorical_crossentropy")
#' # model <- fit(model, x_train, y_train, epochs = 5, batch_size = 32)
#' }
#' @rdname fit
#' @export
fit.ggml_sequential_model <- function(object, x, y, epochs = 1L,
                                       batch_size = 32L,
                                       validation_split = 0.0,
                                       validation_data = NULL,
                                       verbose = 1L,
                                       callbacks = list(), ...) {
  ggml_fit(object, x = x, y = y, epochs = epochs, batch_size = batch_size,
           validation_split = validation_split, validation_data = validation_data,
           verbose = verbose, callbacks = callbacks, ...)
}

#' @rdname fit
#' @export
fit.ggml_functional_model <- function(object, x, y, epochs = 1L,
                                       batch_size = 32L,
                                       validation_split = 0.0,
                                       validation_data = NULL,
                                       verbose = 1L,
                                       callbacks = list(), ...) {
  ggml_fit(object, x = x, y = y, epochs = epochs, batch_size = batch_size,
           validation_split = validation_split, validation_data = validation_data,
           verbose = verbose, ...)
}

# ============================================================================
# evaluate()
# ============================================================================

#' Evaluate a Model
#'
#' Computes loss and metrics on test data.  This is the keras-compatible
#' interface; it delegates to \code{\link{ggml_evaluate}}.
#'
#' @param x A trained model object.
#' @param test_x Test data.
#' @param test_y Test labels.
#' @param batch_size Batch size (default 32).
#' @param ... Additional arguments passed to \code{\link{ggml_evaluate}}.
#' @return A named list with \code{loss} and metric values.
#' @rdname evaluate
#' @export
evaluate.ggml_sequential_model <- function(x, test_x, test_y,
                                            batch_size = 32L, ...) {
  ggml_evaluate(x, x = test_x, y = test_y, batch_size = batch_size, ...)
}

#' @rdname evaluate
#' @export
evaluate.ggml_functional_model <- function(x, test_x, test_y,
                                            batch_size = 32L, ...) {
  ggml_evaluate(x, x = test_x, y = test_y, batch_size = batch_size, ...)
}

# ============================================================================
# predict() -- S3 methods for stats::predict generic
# ============================================================================

#' Predict with a Trained Model
#'
#' Generates predictions from a trained model.  Uses the standard R
#' \code{\link[stats]{predict}} generic for compatibility with keras3 and
#' the broader R ecosystem.
#'
#' @param object A trained model object.
#' @param x Input data (matrix, array, or list for multi-input models).
#' @param batch_size Batch size for inference (default 32).
#' @param ... Additional arguments (ignored).
#' @return Matrix of predictions.
#' @export
predict.ggml_sequential_model <- function(object, x, batch_size = 32L, ...) {
  ggml_predict(object, x = x, batch_size = batch_size, ...)
}

#' @rdname predict.ggml_sequential_model
#' @export
predict.ggml_functional_model <- function(object, x, batch_size = 32L, ...) {
  ggml_predict(object, x = x, batch_size = batch_size, ...)
}
