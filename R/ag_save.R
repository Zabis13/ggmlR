# ============================================================================
# Save / load for autograd (ag_*) modules
#
# Design (marshal variant "M2"): we do NOT serialize the live module (a tree of
# environments + closures, which is fragile across package versions). Instead we
# save a *state dict* of plain numeric matrices:
#
#   - parameters : trainable weights, from model$parameters()  (ag_param $data)
#   - buffers    : non-trainable persistent state not returned by parameters()
#                  (currently ag_batch_norm running_mean / running_var)
#
# Reconstruction needs the architecture back. Two ways:
#   1. supply `model_fn` (a 0-arg closure rebuilding the module) to
#      ag_load_model(); OR
#   2. let ag_save_model() store `model_fn` inside the container, so
#      ag_load_model() can rebuild without arguments.
#
# After rebuilding, parameter and buffer values are copied back by name.
# ============================================================================

AG_SAVE_FORMAT  <- "ggmlR.ag_state"
AG_SAVE_VERSION <- 1L

# Collect non-parameter persistent buffers from an ag_sequential, keyed with the
# same "layer{i}_" prefix scheme used by ag_sequential$parameters().
.ag_collect_buffers <- function(model) {
  buffers <- list()
  layers <- if (!is.null(model$layers)) model$layers else list(model)
  for (i in seq_along(layers)) {
    lyr <- layers[[i]]
    if (is.environment(lyr) && inherits(lyr, "ag_batch_norm")) {
      buffers[[paste0("layer", i, "_running_mean")]] <- lyr$running_mean
      buffers[[paste0("layer", i, "_running_var")]]  <- lyr$running_var
    }
  }
  buffers
}

# Write collected buffers back into a freshly-rebuilt module (by name).
.ag_restore_buffers <- function(model, buffers) {
  if (length(buffers) == 0L) return(invisible(model))
  layers <- if (!is.null(model$layers)) model$layers else list(model)
  for (i in seq_along(layers)) {
    lyr <- layers[[i]]
    if (is.environment(lyr) && inherits(lyr, "ag_batch_norm")) {
      rm_key <- paste0("layer", i, "_running_mean")
      rv_key <- paste0("layer", i, "_running_var")
      if (!is.null(buffers[[rm_key]])) lyr$running_mean <- buffers[[rm_key]]
      if (!is.null(buffers[[rv_key]])) lyr$running_var  <- buffers[[rv_key]]
    }
  }
  invisible(model)
}

#' Save an autograd module's state to disk
#'
#' Serializes the trainable parameters and persistent buffers of an
#' \code{ag_sequential} (or single \code{ag_*} layer) module as a portable
#' state dictionary of plain numeric matrices. This avoids serializing the live
#' module (environments + closures), which is brittle across ggmlR versions and
#' carries non-portable GPU pointers.
#'
#' Reconstruction requires the architecture. Either pass \code{model_fn} here so
#' it is stored in the file, or pass it later to \code{\link{ag_load_model}}.
#'
#' @param model An \code{ag_sequential} module or a single \code{ag_*} layer
#'   exposing \code{parameters()}.
#' @param path File path to write (an RDS container).
#' @param model_fn Optional zero-argument function that rebuilds the module
#'   architecture (fresh, untrained). If supplied, it is stored in the container
#'   so \code{\link{ag_load_model}} can rebuild without arguments. Must not
#'   capture GPU tensors in its enclosing environment.
#' @return \code{path}, invisibly.
#' @seealso \code{\link{ag_load_model}}
#' @export
#' @examples
#' \donttest{
#' build <- function() ag_sequential(ag_linear(4L, 8L), ag_linear(8L, 3L))
#' model <- build()
#' ag_save_model(model, tempfile(fileext = ".rds"), model_fn = build)
#' }
ag_save_model <- function(model, path, model_fn = NULL) {
  if (!inherits(model, c("ag_sequential", "ag_layer"))) {
    stop("ag_save_model(): `model` must be an ag_sequential or ag_* layer.")
  }
  if (!is.null(model_fn) && !is.function(model_fn)) {
    stop("ag_save_model(): `model_fn` must be a function or NULL.")
  }

  params_raw <- model$parameters()
  # Pull each parameter's value to a plain CPU matrix (works on GPU too).
  parameters <- lapply(params_raw, .ag_data)
  # Record dtypes alongside (forward-compat metadata; not used on load yet).
  dtypes <- vapply(params_raw, function(p) {
    if (is_ag_tensor(p) && !is.null(p$dtype)) p$dtype else "f32"
  }, character(1L))

  container <- list(
    format        = AG_SAVE_FORMAT,
    version       = AG_SAVE_VERSION,
    parameters    = parameters,
    param_dtypes  = dtypes,
    buffers       = .ag_collect_buffers(model),
    model_fn      = model_fn,
    ggmlR_version = utils::packageVersion("ggmlR"),
    R_version     = getRversion(),
    created       = Sys.time()
  )
  class(container) <- "ggmlR_ag_state"

  saveRDS(container, path)
  invisible(path)
}

#' Load an autograd module from a saved state
#'
#' Reconstructs an \code{ag_*} module saved with \code{\link{ag_save_model}}.
#' The architecture is rebuilt by calling \code{model_fn} (either the one passed
#' here, or the one stored inside the container at save time), and the saved
#' parameter and buffer values are copied back by name.
#'
#' @param path File path written by \code{\link{ag_save_model}}.
#' @param model_fn Optional zero-argument rebuild function. Required if no
#'   \code{model_fn} was stored at save time. If both are present, this argument
#'   takes precedence.
#' @param device Optional device for the rebuilt module (\code{"cpu"} or
#'   \code{"gpu"}). If \code{NULL} (default), the current \code{ag_device()} is
#'   used by the rebuild.
#' @return The reconstructed module with restored weights, in eval mode.
#' @seealso \code{\link{ag_save_model}}
#' @export
#' @examples
#' \donttest{
#' build <- function() ag_sequential(ag_linear(4L, 8L), ag_linear(8L, 3L))
#' f <- tempfile(fileext = ".rds")
#' ag_save_model(build(), f, model_fn = build)
#' model <- ag_load_model(f)
#' }
ag_load_model <- function(path, model_fn = NULL, device = NULL) {
  container <- readRDS(path)

  if (!is.list(container) || !identical(container$format, AG_SAVE_FORMAT)) {
    stop("ag_load_model(): file is not a ggmlR ag_state container.")
  }
  if (!identical(container$version, AG_SAVE_VERSION)) {
    stop("ag_load_model(): unsupported container version ", container$version,
         " (this ggmlR supports version ", AG_SAVE_VERSION, ").")
  }

  builder <- model_fn %||% container$model_fn
  if (is.null(builder) || !is.function(builder)) {
    stop("ag_load_model(): no `model_fn` available. Pass one via the `model_fn` ",
         "argument, or re-save the model with `model_fn` so it is stored in ",
         "the file.")
  }

  # Rebuild architecture (optionally on a specific device).
  if (!is.null(device)) {
    old_dev <- ag_default_device()
    on.exit(ag_device(old_dev), add = TRUE)
    ag_device(device)
  }
  model <- builder()

  if (!inherits(model, c("ag_sequential", "ag_layer"))) {
    stop("ag_load_model(): `model_fn` must return an ag_sequential or ",
         "ag_* layer (got class: ", paste(class(model), collapse = "/"), ").")
  }

  # Copy parameters back by name.
  params <- model$parameters()
  saved  <- container$parameters
  missing_names <- setdiff(names(params), names(saved))
  extra_names   <- setdiff(names(saved), names(params))
  if (length(missing_names) || length(extra_names)) {
    stop("ag_load_model(): parameter mismatch between saved state and rebuilt ",
         "model.\n  missing in file: ",
         paste(missing_names, collapse = ", "),
         "\n  not in model: ",
         paste(extra_names, collapse = ", "),
         "\nThe `model_fn` architecture must match the saved one.")
  }
  for (nm in names(params)) {
    p <- params[[nm]]
    new_data <- saved[[nm]]
    if (!all(dim(.ag_data(p)) == dim(new_data))) {
      stop("ag_load_model(): shape mismatch for parameter '", nm, "': model ",
           paste(dim(.ag_data(p)), collapse = "x"), " vs file ",
           paste(dim(new_data), collapse = "x"), ".")
    }
    p$data <- new_data
    # Drop any stale GPU pointer so the next forward re-uploads $data.
    if (!is.null(p$ptr)) p$ptr <- NULL
  }

  # Restore buffers (BN running stats).
  .ag_restore_buffers(model, container$buffers)

  # Loaded models are for inference unless the user re-enters training.
  ag_eval(model)
  model
}

#' @export
print.ggmlR_ag_state <- function(x, ...) {
  cat("<ggmlR ag_state>\n")
  cat("  format:        ", x$format, " v", x$version, "\n", sep = "")
  cat("  parameters:    ", length(x$parameters), "\n", sep = "")
  cat("  buffers:       ", length(x$buffers), "\n", sep = "")
  cat("  model_fn saved:", !is.null(x$model_fn), "\n")
  cat("  ggmlR version: ", format(x$ggmlR_version), "\n", sep = "")
  cat("  created:       ", format(x$created), "\n", sep = "")
  invisible(x)
}
