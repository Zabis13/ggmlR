# broom (tidy / glance / augment) methods for fitted parsnip "ggml" models ----
#
# These operate on the `ggmlr_parsnip_model` object returned by the fit wrappers
# in parsnip_mlp.R (it wraps a compiled ggml_sequential_model in $model). They
# follow broom conventions: tidy() = one row per component (layer), glance() =
# one-row model summary, augment() = new_data + .pred* columns. Generics come
# from the `generics` package (same source broom re-exports), already a ggmlR
# dependency.

# internal: infer a printable backend name from the compiled model
.ggmlr_backend_name <- function(model) {
  comp <- model$compilation
  if (is.null(comp) || is.null(comp$backend)) return(NA_character_)
  # Prefer the recorded backend; fall back to the cpu_backend heuristic for
  # models compiled before that bookkeeping existed (set only when a GPU
  # backend is in use).
  comp$backend_used %||% (if (!is.null(comp$cpu_backend)) "vulkan" else "cpu")
}

#' Tidy a fitted ggml parsnip model into a per-layer table
#'
#' Returns one row per layer of the underlying sequential network, in
#' broom style. Useful for comparing architectures across experiments in a
#' R Markdown / Quarto report.
#'
#' @param x A fitted `ggmlr_parsnip_model` (the engine object inside a parsnip
#'   fit; e.g. from `extract_fit_engine()`).
#' @param ... Unused; for generic compatibility.
#'
#' @return A [tibble][tibble::tibble] with columns: `layer` (name), `type`,
#'   `units` (output units, `NA` if not applicable), `activation`,
#'   `output_shape` (character), `params` (trainable parameter count) and
#'   `trainable` (logical).
#'
#' @examplesIf rlang::is_installed(c("parsnip", "tibble"))
#' spec <- parsnip::mlp(hidden_units = 8L, epochs = 3L) |>
#'   parsnip::set_engine("ggml", backend = "cpu") |>
#'   parsnip::set_mode("regression")
#' fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)
#' generics::tidy(parsnip::extract_fit_engine(fit_obj))
#'
#' @importFrom generics tidy
#' @method tidy ggmlr_parsnip_model
#' @export
tidy.ggmlr_parsnip_model <- function(x, ...) {
  model <- x$model
  layers <- model$layers

  # Make sure output shapes are populated.
  if (length(layers) > 0 &&
      is.null(layers[[1]]$output_shape) &&
      !is.null(model$input_shape)) {
    model <- nn_infer_shapes(model)
    layers <- model$layers
  }

  if (length(layers) == 0) {
    return(tibble::tibble(
      layer = character(), type = character(), units = integer(),
      activation = character(), output_shape = character(),
      params = integer(), trainable = logical()
    ))
  }

  # NB: the loop variable must NOT be named `layer` — tibble() evaluates the
  # column expressions in a data mask where the `layer = ...` column name would
  # shadow it, so `layer$type` would resolve to the (atomic) column being built
  # and fail with "$ operator is invalid for atomic vectors".
  rows <- lapply(layers, function(ly) {
    cfg <- ly$config %||% list()
    units <- cfg$units %||% cfg$filters %||% NA_integer_
    act   <- cfg$activation %||% NA_character_
    shape <- if (!is.null(ly$output_shape)) {
      paste0("(", paste(ly$output_shape, collapse = ", "), ")")
    } else NA_character_

    tibble::tibble(
      layer        = ly$name %||% ly$type %||% NA_character_,
      type         = ly$type %||% NA_character_,
      units        = as.integer(units),
      activation   = as.character(act),
      output_shape = shape,
      params       = as.integer(nn_count_layer_params(ly)),
      trainable    = isTRUE(ly$trainable %||% TRUE)
    )
  })

  do.call(rbind, rows)
}

#' One-row summary of a fitted ggml parsnip model
#'
#' Returns a single-row [tibble][tibble::tibble] summarising the fitted model,
#' in broom `glance()` style.
#'
#' @inheritParams tidy.ggmlr_parsnip_model
#'
#' @return A one-row tibble with columns: `mode`, `n_features`, `n_layers`,
#'   `total_params`, `optimizer`, `loss`, `backend`, `epochs`, `fit_time` (wall
#'   seconds) and `final_loss` (last training loss, `NA` if no history).
#'
#' @examplesIf rlang::is_installed(c("parsnip", "tibble"))
#' spec <- parsnip::mlp(hidden_units = 8L, epochs = 3L) |>
#'   parsnip::set_engine("ggml", backend = "cpu") |>
#'   parsnip::set_mode("regression")
#' fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)
#' generics::glance(parsnip::extract_fit_engine(fit_obj))
#'
#' @importFrom generics glance
#' @method glance ggmlr_parsnip_model
#' @export
glance.ggmlr_parsnip_model <- function(x, ...) {
  model <- x$model
  comp  <- model$compilation %||% list()
  hist  <- model$history

  total_params <- sum(vapply(model$layers, nn_count_layer_params, numeric(1)))

  epochs <- if (!is.null(hist$epochs)) length(hist$epochs) else NA_integer_
  final_loss <- if (!is.null(hist$train_loss) && length(hist$train_loss) > 0) {
    hist$train_loss[[length(hist$train_loss)]]
  } else NA_real_

  tibble::tibble(
    mode         = x$mode %||% NA_character_,
    n_features   = as.integer(x$n_features %||% NA_integer_),
    n_layers     = length(model$layers),
    total_params = as.integer(total_params),
    optimizer    = as.character(comp$optimizer %||% NA_character_),
    loss         = as.character(comp$loss %||% NA_character_),
    backend      = .ggmlr_backend_name(model),
    epochs       = as.integer(epochs),
    fit_time     = as.double(x$fit_time %||% NA_real_),
    final_loss   = as.double(final_loss)
  )
}

#' Augment new data with predictions from a fitted ggml parsnip model
#'
#' Adds prediction columns to `new_data`, broom style. For classification this
#' appends `.pred_class` plus one `.pred_<level>` probability column per class;
#' for regression it appends `.pred`. Predictions are produced by the existing
#' `predict()` method for ggml parsnip models (no duplicate inference logic).
#'
#' @inheritParams tidy.ggmlr_parsnip_model
#' @param new_data A data frame of predictors (same columns used for fitting).
#'
#' @return `new_data` as a tibble with prediction columns appended.
#'
#' @examplesIf rlang::is_installed(c("parsnip", "tibble"))
#' spec <- parsnip::mlp(hidden_units = 8L, epochs = 3L) |>
#'   parsnip::set_engine("ggml", backend = "cpu") |>
#'   parsnip::set_mode("regression")
#' fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)
#' generics::augment(parsnip::extract_fit_engine(fit_obj), mtcars)
#'
#' @importFrom generics augment
#' @importFrom stats predict
#' @method augment ggmlr_parsnip_model
#' @export
augment.ggmlr_parsnip_model <- function(x, new_data, ...) {
  out <- tibble::as_tibble(new_data)

  preds <- if (identical(x$mode, "classification")) {
    c(predict(x, new_data = new_data, type = "class"),
      predict(x, new_data = new_data, type = "prob"))
  } else {
    predict(x, new_data = new_data)
  }

  tibble::as_tibble(c(out, preds))
}
