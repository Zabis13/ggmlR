# Model diagnostics: training-history and backend accessors --------------------
#
# These give a *standard* way to pull the loss/metric curve and the
# actually-used backend out of a fitted model, whether the user holds a raw
# sequential/functional model or a parsnip fit engine (ggmlr_parsnip_model,
# which wraps the compiled model in $model).

# internal: unwrap to the underlying compiled ggml model
.ggmlr_unwrap_model <- function(object) {
  if (inherits(object, "ggmlr_parsnip_model")) return(object$model)
  object
}

#' Training history of a fitted ggml model
#'
#' Returns the per-epoch loss / accuracy curve recorded during
#' \code{\link{ggml_fit}}, in a tidy data frame. This is the standard accessor
#' for the loss curve; it works on a raw sequential/functional model or on a
#' fitted parsnip engine object (e.g. from \code{extract_fit_engine()}).
#'
#' @param object A fitted \code{ggml_sequential_model},
#'   \code{ggml_functional_model} or \code{ggmlr_parsnip_model}.
#' @param format \code{"wide"} (default) returns one row per epoch with one
#'   column per metric. \code{"long"} returns one row per (epoch, metric) with
#'   columns \code{epoch}, \code{metric} (e.g. \code{"loss"}/\code{"accuracy"}),
#'   \code{split} (\code{"train"}/\code{"val"}) and \code{value} — convenient for
#'   \pkg{ggplot2} faceting.
#' @param ... Unused; for extensibility.
#'
#' @return A data frame (tibble if \pkg{tibble} is installed). Wide columns:
#'   \code{epoch}, \code{train_loss}, \code{train_accuracy}, and
#'   \code{val_loss} / \code{val_accuracy} when a validation split was used.
#'   Returns \code{NULL} with a warning if the model has no recorded history
#'   (e.g. not yet fitted).
#'
#' @seealso \code{\link{ggml_model_backend}}
#' @export
ggml_training_history <- function(object, format = c("wide", "long"), ...) {
  format <- match.arg(format)
  model  <- .ggmlr_unwrap_model(object)
  hist   <- model$history

  if (is.null(hist) || is.null(hist$epochs) || length(hist$epochs) == 0) {
    warning("Model has no training history (not fitted, or fit recorded none).")
    return(NULL)
  }

  df <- data.frame(epoch = as.integer(hist$epochs))
  df$train_loss <- as.double(hist$train_loss)
  df$train_accuracy <- as.double(hist$train_accuracy)

  has_val <- !is.null(hist$val_loss) && any(!is.na(hist$val_loss))
  if (has_val) {
    df$val_loss <- as.double(hist$val_loss)
    df$val_accuracy <- as.double(hist$val_accuracy)
  }

  out <- if (identical(format, "long")) {
    .history_to_long(df, has_val)
  } else {
    df
  }

  if (requireNamespace("tibble", quietly = TRUE)) tibble::as_tibble(out) else out
}

# internal: reshape a wide history data.frame to long (epoch/metric/split/value)
.history_to_long <- function(df, has_val) {
  specs <- list(
    list(col = "train_loss",     metric = "loss",     split = "train"),
    list(col = "train_accuracy", metric = "accuracy", split = "train")
  )
  if (has_val) {
    specs <- c(specs, list(
      list(col = "val_loss",     metric = "loss",     split = "val"),
      list(col = "val_accuracy", metric = "accuracy", split = "val")
    ))
  }

  parts <- lapply(specs, function(s) {
    data.frame(
      epoch  = df$epoch,
      metric = s$metric,
      split  = s$split,
      value  = df[[s$col]],
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, parts)
}

#' Backend a fitted ggml model actually ran on
#'
#' Reports the backend the model was \emph{actually} compiled onto, making a
#' silent \code{backend = "auto"} fallback to CPU (when no GPU is available)
#' inspectable. Works on a raw sequential/functional model or a fitted parsnip
#' engine object.
#'
#' @param object A compiled/fitted \code{ggml_sequential_model},
#'   \code{ggml_functional_model} or \code{ggmlr_parsnip_model}.
#' @param verbose If \code{FALSE} (default) returns a single string
#'   (\code{"vulkan"} or \code{"cpu"}). If \code{TRUE} returns a list with
#'   details (see Value).
#'
#' @return If \code{verbose = FALSE}, a length-1 character: the backend in use
#'   (\code{"vulkan"} or \code{"cpu"}). If \code{verbose = TRUE}, a list with:
#'   \code{requested} (what was asked: \code{"auto"}/\code{"cpu"}/\code{"vulkan"}),
#'   \code{used} (\code{"vulkan"}/\code{"cpu"}), \code{device} (GPU device
#'   description, or \code{"cpu"}) and \code{fallback} (logical: \code{TRUE} when
#'   a non-CPU backend was requested but CPU was used instead).
#'
#' @seealso \code{\link{ggml_training_history}}
#' @export
ggml_model_backend <- function(object, verbose = FALSE) {
  model <- .ggmlr_unwrap_model(object)
  comp  <- model$compilation

  if (is.null(comp) || isFALSE(model$compiled)) {
    stop("Model is not compiled; backend is unknown. Call ggml_compile() first.",
         call. = FALSE)
  }

  used <- comp$backend_used
  if (is.null(used)) {
    # Fallback for models compiled before backend bookkeeping existed:
    # compile() only sets $cpu_backend when a GPU backend is in use.
    used <- if (!is.null(comp$cpu_backend)) "vulkan" else "cpu"
  }

  if (!verbose) return(used)

  requested <- comp$backend_requested %||% NA_character_
  device    <- comp$device %||% (if (identical(used, "cpu")) "cpu" else NA_character_)
  fallback  <- !is.na(requested) &&
    !identical(requested, "cpu") && identical(used, "cpu")

  list(
    requested = requested,
    used      = used,
    device    = device,
    fallback  = fallback
  )
}
