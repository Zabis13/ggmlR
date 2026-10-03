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

# Is `x` a layer-like object (environment module, or a plain-list layer such as
# ag_linear exposing $params()/$parameters())?  Used to tell a genuine list of
# sublayers apart from the fields of a single layer: ag_sequential() with one
# list-based layer stores that layer's fields directly in $layers, so a naive
# descent would recurse into its closures.
.ag_is_layer <- function(x) {
  if (is.environment(x)) return(inherits(x, "ag_layer") || is.function(x$forward))
  is.list(x) && (is.function(x[["params"]]) || is.function(x[["parameters"]]))
}

# Sublayers of a module, or an empty list when it has none.
.ag_sublayers <- function(model) {
  layers <- model$layers
  if (is.null(layers) || !is.list(layers)) return(list())
  if (!all(vapply(layers, .ag_is_layer, logical(1L)))) return(list())
  layers
}

# Collect non-parameter persistent buffers, keyed with the same nested
# "layer{i}_" prefix scheme used by ag_sequential$parameters(): the recursion
# descends into nested ag_sequential modules and accumulates the prefix, so a
# batch_norm at layers[[2]]$layers[[1]] is keyed "layer2_layer1_running_mean".
# Keys for flat modules are unchanged, so files written by earlier versions
# still load.
.ag_collect_buffers <- function(model, prefix = "") {
  buffers <- list()
  if (is.environment(model) && inherits(model, "ag_batch_norm")) {
    buffers[[paste0(prefix, "running_mean")]] <- model$running_mean
    buffers[[paste0(prefix, "running_var")]]  <- model$running_var
    return(buffers)
  }
  layers <- .ag_sublayers(model)
  for (i in seq_along(layers)) {
    nested <- .ag_collect_buffers(layers[[i]], paste0(prefix, "layer", i, "_"))
    for (nm in names(nested)) buffers[[nm]] <- nested[[nm]]
  }
  buffers
}

# Write collected buffers back into a freshly-rebuilt module (by name), using
# the same nested key scheme as .ag_collect_buffers().
.ag_restore_buffers <- function(model, buffers, prefix = "") {
  if (length(buffers) == 0L) return(invisible(model))
  if (is.environment(model) && inherits(model, "ag_batch_norm")) {
    rm_key <- paste0(prefix, "running_mean")
    rv_key <- paste0(prefix, "running_var")
    if (!is.null(buffers[[rm_key]])) model$running_mean <- buffers[[rm_key]]
    if (!is.null(buffers[[rv_key]])) model$running_var  <- buffers[[rv_key]]
    return(invisible(model))
  }
  layers <- .ag_sublayers(model)
  for (i in seq_along(layers)) {
    .ag_restore_buffers(layers[[i]], buffers, paste0(prefix, "layer", i, "_"))
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

  # Same content as the in-memory state dict (plain CPU matrices, works on GPU
  # too); the file format itself is unchanged.
  sd <- ag_state_dict(model)

  container <- list(
    format        = AG_SAVE_FORMAT,
    version       = AG_SAVE_VERSION,
    parameters    = sd$parameters,
    param_dtypes  = sd$param_dtypes,
    buffers       = sd$buffers,
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
#' @param dtype Optional GPU upload precision for the restored parameters
#'   (\code{"f32"}, \code{"f16"}, or \code{"bf16"}). If \code{NULL} (default),
#'   the dtype recorded in the file is restored, so a module saved under
#'   \code{ag_dtype("f16")} keeps computing in f16 regardless of the loading
#'   session's default. Pass an explicit value to override the file. Files
#'   written before dtypes were recorded fall back to whatever
#'   \code{model_fn} builds.
#'
#'   Note that this is an \emph{upload/compute} precision, not a storage
#'   precision: parameter values always live in \code{$data} as full-precision
#'   R numeric matrices (that is what backward needs), and \code{dtype} only
#'   controls the precision they are uploaded to the GPU with. Saving therefore
#'   never rounds the weights, whatever dtype is in effect.
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
ag_load_model <- function(path, model_fn = NULL, device = NULL, dtype = NULL) {
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
  if (!is.null(dtype)) dtype <- match.arg(dtype, c("f32", "f16", "bf16"))

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
    # .ag_data_set() installs the value and drops any device residency, so the
    # next forward re-uploads instead of computing on the pre-load weights
    # (inst/docs/ag_data_contract.md). It subsumes the explicit $ptr/$ctx_gen
    # clearing that used to follow this line.
    .ag_data_set(p, new_data)
    # dtype resolution: an explicit `dtype` argument wins; otherwise restore
    # what the file recorded, so a module trained under ag_dtype("f16") keeps
    # computing in f16 no matter what the loading session's default is. Files
    # written before param_dtypes existed leave the rebuilt dtype untouched.
    saved_dtype <- unname(container$param_dtypes[nm])
    if (length(saved_dtype) != 1L || is.na(saved_dtype)) saved_dtype <- NULL
    resolved <- dtype %||% saved_dtype
    if (!is.null(resolved)) p$dtype <- resolved
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

# ============================================================================
# In-memory state dicts: the same content ag_save_model() writes, without a file.
# ============================================================================

AG_STATE_DICT_FORMAT  <- "ggmlR.ag_state_dict"
AG_STATE_DICT_VERSION <- 1L

.ag_model_params <- function(model, caller) {
  if (is.function(model$parameters)) return(model$parameters())
  if (is.function(model$params))     return(model$params())
  stop(caller, ": `model` has no parameters() / params() method", call. = FALSE)
}

#' Model weights as plain R data
#'
#' A copy of every parameter and persistent buffer (\code{ag_batch_norm}
#' running statistics) as host matrices, keyed like \code{model$parameters()}.
#' Values are R doubles whatever the device or \code{ag_dtype()}: an f32 value
#' is represented exactly, so a state taken on the GPU in f32 restores on the
#' CPU (or back) to the same numbers. f16/bf16 weights carry only their own
#' precision; continuing exactly across f16 and f32 is not possible.
#'
#' Use it for checkpoints, for copying weights between models (target networks:
#' \code{ag_load_state_dict(target, ag_state_dict(online))}), or with
#' \code{saveRDS}. As with any \code{.rds}, only read files you trust.
#'
#' @param model An \code{ag_sequential}, an \code{ag_*} layer, or anything with a
#'   \code{parameters()} or \code{params()} method.
#' @return A list with \code{parameters}, \code{buffers} and
#'   \code{param_dtypes}, plus \code{format}/\code{version}.
#' @seealso \code{\link{ag_load_state_dict}}, \code{\link{ag_save_model}}
#' @export
ag_state_dict <- function(model) {
  params <- .ag_model_params(model, "ag_state_dict()")
  list(
    format       = AG_STATE_DICT_FORMAT,
    version      = AG_STATE_DICT_VERSION,
    parameters   = lapply(params, function(p) .ag_as_matrix(.ag_data(p))),
    param_dtypes = vapply(params, function(p)
      if (is_ag_tensor(p) && !is.null(p$dtype)) p$dtype else "f32", character(1L)),
    buffers      = .ag_collect_buffers(model)
  )
}

#' Load weights into an existing model
#'
#' Writes a state from \code{\link{ag_state_dict}} (or the container of an
#' \code{\link{ag_save_model}} file read with \code{readRDS}) into \code{model}
#' in place. Device-resident weights are overwritten in their own buffers, so
#' an optimizer built on the model stays valid and keeps its residency. Values
#' are converted to the model's dtype on upload.
#'
#' Everything is checked before anything is written: an error leaves the model
#' untouched.
#'
#' @param model Target model (see \code{\link{ag_state_dict}}).
#' @param state A state dict.
#' @param strict \code{TRUE} (default): the parameter and buffer names must match
#'   exactly. \code{FALSE}: names present on only one side are skipped (partial
#'   loading, transfer learning); names present on both sides are still
#'   shape-checked, always.
#' @return \code{model}, invisibly.
#' @export
ag_load_state_dict <- function(model, state, strict = TRUE) {
  caller <- "ag_load_state_dict()"
  if (!is.list(state) || !is.list(state$parameters))
    stop(caller, ": `state` is not a state dict (no $parameters)", call. = FALSE)
  params <- .ag_model_params(model, caller)
  bufs   <- .ag_collect_buffers(model)
  sbufs  <- state$buffers %||% list()

  match_names <- function(have, saved, what) {
    missing <- setdiff(names(have), names(saved))
    extra   <- setdiff(names(saved), names(have))
    if (strict && (length(missing) || length(extra)))
      stop(caller, ": ", what, " names differ (strict = TRUE)",
           "\n  missing in state: ", paste(missing, collapse = ", "),
           "\n  not in model: ",     paste(extra,   collapse = ", "), call. = FALSE)
    intersect(names(have), names(saved))
  }
  p_names <- match_names(params, state$parameters, "parameter")
  b_names <- match_names(bufs, sbufs, "buffer")

  check <- function(cur_dim, val, nm) {
    if (!is.numeric(val) || !identical(as.integer(dim(val)), as.integer(cur_dim)))
      stop(caller, ": shape mismatch for '", nm, "': model ",
           paste(cur_dim, collapse = "x"), ", state ",
           paste(dim(val) %||% length(val), collapse = "x"), call. = FALSE)
    if (!all(is.finite(val)))
      stop(caller, ": '", nm, "' contains NA, NaN or Inf", call. = FALSE)
  }
  for (nm in p_names) check(.ag_sel_dim(params[[nm]]), state$parameters[[nm]], nm)
  for (nm in b_names) check(dim(bufs[[nm]]), sbufs[[nm]], nm)

  # All checks passed -- now write.
  for (nm in p_names) {
    v <- state$parameters[[nm]]
    .ag_opt_store_weight(params[[nm]], matrix(as.numeric(v), nrow(v), ncol(v)))
  }
  if (length(b_names)) .ag_restore_buffers(model, sbufs[b_names])
  invisible(model)
}
