# High-level sequential model API for ggmlR
# Provides Keras-like model building, compilation, training and evaluation

# ============================================================================
# Model Constructor
# ============================================================================

#' Create a Sequential Neural Network Model
#'
#' Creates an empty sequential model that layers can be added to using
#' pipe (\code{|>}) operators.
#'
#' @return A ggml_sequential_model object
#' @export
#' @examples
#' \dontrun{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_max_pooling_2d(c(2,2)) |>
#'   ggml_layer_flatten() |>
#'   ggml_layer_dense(128, activation = "relu") |>
#'   ggml_layer_dense(10, activation = "softmax")
#' }
ggml_model_sequential <- function() {
  model <- list(
    layers = list(),
    input_shape = NULL,
    compiled = FALSE,
    compilation = list(
      ctx_weights = NULL,
      backend = NULL,
      sched = NULL,
      buffer = NULL,
      optimizer = NULL,
      loss = NULL,
      metrics = NULL
    )
  )
  class(model) <- c("ggml_sequential_model", "list")
  model
}

# ============================================================================
# Layer access and manipulation
# ============================================================================

#' Get a Layer from a Sequential Model
#'
#' Retrieves a layer by name or by integer index (1-based).
#'
#' @param model A ggml_sequential_model object
#' @param index Integer index of the layer (1-based), or NULL
#' @param name Character name of the layer, or NULL
#' @return The layer list object
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(64, activation = "relu", name = "hidden") |>
#'   ggml_layer_dense(10, activation = "softmax", name = "output")
#'
#' ggml_get_layer(model, index = 1)
#' ggml_get_layer(model, name = "output")
#' }
ggml_get_layer <- function(model, index = NULL, name = NULL) {
  if (is.null(index) && is.null(name)) {
    stop("Provide either index or name.")
  }
  if (!is.null(index) && !is.null(name)) {
    stop("Provide either index or name, not both.")
  }
  if (!is.null(index)) {
    n <- length(model$layers)
    if (index < 1L || index > n) {
      stop("Layer index ", index, " out of range (model has ", n, " layers).")
    }
    return(model$layers[[index]])
  }
  # by name
  for (layer in model$layers) {
    if (!is.null(layer$name) && layer$name == name) return(layer)
  }
  stop("No layer with name '", name, "'.")
}

#' Remove the Last Layer from a Sequential Model
#'
#' Removes the last layer from the model. The model must not be compiled.
#'
#' @param model A ggml_sequential_model object
#' @return The model with the last layer removed.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(64, activation = "relu") |>
#'   ggml_layer_dense(10, activation = "softmax")
#'
#' model <- ggml_pop_layer(model)
#' length(model$layers)  # 1
#' }
ggml_pop_layer <- function(model) {
  if (model$compiled) {
    stop("Cannot pop layers from a compiled model.")
  }
  n <- length(model$layers)
  if (n == 0L) {
    stop("Model has no layers to remove.")
  }
  model$layers <- model$layers[-n]
  if (n == 1L) model$input_shape <- NULL
  model
}

# ============================================================================
# Freeze / unfreeze weights
# ============================================================================

# Which layers a freeze call means. layer_names and prefix select by name (a
# layer matching either is selected, and each must select something); without
# them from/to select by index. Shared by every freeze/unfreeze method, so the
# rules are the same everywhere.
nn_layer_name_of <- function(l) if (!is.null(l$name)) l$name else ""

nn_match_names <- function(nms, layer_names, prefix) {
  hit <- rep(FALSE, length(nms))
  if (!is.null(layer_names)) {
    missing <- setdiff(layer_names, nms)
    if (length(missing))
      stop("No layers found with names: ", paste(missing, collapse = ", "),
           call. = FALSE)
    hit <- hit | nms %in% layer_names
  }
  for (p in prefix) {
    by_p <- startsWith(nms, p)
    if (!any(by_p))
      stop("No layer name starts with '", p, "'", call. = FALSE)
    hit <- hit | by_p
  }
  hit
}

nn_select_seq_layers <- function(model, from, to, layer_names, prefix) {
  n <- length(model$layers)
  if (n == 0L) stop("Model has no layers.", call. = FALSE)
  if (!is.null(layer_names) || !is.null(prefix)) {
    nms <- vapply(model$layers, nn_layer_name_of, character(1))
    return(which(nn_match_names(nms, layer_names, prefix)))
  }
  from <- as.integer(from); to <- as.integer(to)
  if (from < 1L || to > n || from > to)
    stop("Invalid range: from=", from, " to=", to, " (model has ", n, " layers).",
         call. = FALSE)
  from:to
}

# Functional: the non-input nodes selected by name or prefix; all of them when
# neither is given.
nn_select_fun_nodes <- function(model, layer_names, prefix) {
  nodes <- Filter(function(nd) !is.null(nd$node_type) && nd$node_type != "input",
                  nn_topo_sort(model$outputs))
  if (is.null(layer_names) && is.null(prefix)) return(nodes)
  nms <- vapply(nodes, function(nd)
    if (!is.null(nd$config$name)) nd$config$name else "", character(1))
  nodes[nn_match_names(nms, layer_names, prefix)]
}

#' Freeze Layer Weights
#'
#' Sets \code{trainable = FALSE} on layers, preventing their weights from being
#' updated during training. Layers are chosen by \code{layer_names} and/or
#' \code{prefix}; without either, by the index range \code{from} / \code{to}
#' (sequential models), or all of them.
#'
#' @param model A model object (ggml_sequential_model or ggml_functional_model)
#' @param from Integer index of the first layer to freeze (default: 1)
#' @param to Integer index of the last layer to freeze (default: last layer)
#' @param layer_names Character vector of layer names to freeze (overrides from/to)
#' @param prefix Character vector: freeze every layer whose name starts with one
#'   of these -- a whole transformer block, say: \code{prefix = "block1_"}
#'   reaches \code{block1_norm1}, \code{block1_attn}, \code{block1_norm2},
#'   \code{block1_ff1} and \code{block1_ff2} -- not \code{block10_*}. Given with
#'   \code{layer_names}, the selection is the union of the two; every name and
#'   every prefix must match at least one layer, or the call stops.
#' @param ... Additional arguments passed to methods
#' @return The model with selected layers frozen.
#' @seealso \code{\link{ggml_freeze_bn_stats}} to fix batch-norm statistics,
#'   which freezing the weights does not do.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(64, activation = "relu") |>
#'   ggml_layer_dense(10, activation = "softmax")
#'
#' # Freeze all layers
#' model <- ggml_freeze_weights(model)
#'
#' # Freeze only the first layer
#' model <- ggml_freeze_weights(model, from = 1, to = 1)
#' }
ggml_freeze_weights <- function(model, from = 1L, to = length(model$layers),
                                 layer_names = NULL, prefix = NULL, ...) {
  UseMethod("ggml_freeze_weights")
}

#' @export
ggml_freeze_weights.ggml_sequential_model <- function(model,
                                                        from = 1L,
                                                        to = length(model$layers),
                                                        layer_names = NULL,
                                                        prefix = NULL, ...) {
  for (i in nn_select_seq_layers(model, from, to, layer_names, prefix))
    model$layers[[i]]$trainable <- FALSE
  model
}

#' @export
ggml_freeze_weights.ggml_functional_model <- function(model,
                                                        from = NULL, to = NULL,
                                                        layer_names = NULL,
                                                        prefix = NULL, ...) {
  frozen <- if (is.null(model$frozen_nodes)) list() else model$frozen_nodes
  for (node in nn_select_fun_nodes(model, layer_names, prefix))
    frozen[[node$id]] <- FALSE
  model$frozen_nodes <- frozen
  model
}

#' Unfreeze Layer Weights
#'
#' Sets \code{trainable = TRUE} on layers, chosen as in
#' \code{\link{ggml_freeze_weights}}: by \code{layer_names} and/or
#' \code{prefix}, else by \code{from} / \code{to}, else all of them.
#'
#' @param model A model object (ggml_sequential_model or ggml_functional_model)
#' @param from Integer index of the first layer to unfreeze (default: 1)
#' @param to Integer index of the last layer to unfreeze (default: last layer)
#' @param layer_names Character vector of layer names to unfreeze (overrides from/to)
#' @param prefix Character vector: unfreeze every layer whose name starts with
#'   one of these. With \code{layer_names}, the union of the two.
#' @param ... Additional arguments passed to methods
#' @return The model with selected layers unfrozen.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(64, activation = "relu") |>
#'   ggml_layer_dense(10, activation = "softmax")
#'
#' model <- ggml_freeze_weights(model)
#' model <- ggml_unfreeze_weights(model, from = 2)  # unfreeze last layer only
#' }
ggml_unfreeze_weights <- function(model, from = 1L, to = length(model$layers),
                                   layer_names = NULL, prefix = NULL, ...) {
  UseMethod("ggml_unfreeze_weights")
}

#' @export
ggml_unfreeze_weights.ggml_sequential_model <- function(model,
                                                          from = 1L,
                                                          to = length(model$layers),
                                                          layer_names = NULL,
                                                          prefix = NULL, ...) {
  for (i in nn_select_seq_layers(model, from, to, layer_names, prefix))
    model$layers[[i]]$trainable <- TRUE
  model
}

#' @export
ggml_unfreeze_weights.ggml_functional_model <- function(model,
                                                          from = NULL, to = NULL,
                                                          layer_names = NULL,
                                                          prefix = NULL, ...) {
  if (is.null(layer_names) && is.null(prefix)) {
    model$frozen_nodes <- list()
    return(model)
  }
  frozen <- if (is.null(model$frozen_nodes)) list() else model$frozen_nodes
  for (node in nn_select_fun_nodes(model, layer_names, prefix))
    frozen[[node$id]] <- NULL
  model$frozen_nodes <- frozen
  model
}

#' Fix or Release Batch-Norm Statistics
#'
#' \code{ggml_freeze_bn_stats()} fixes the running mean and variance of
#' batch-norm layers: they then normalize with those stored statistics during
#' training as well as at prediction, and \code{\link{ggml_fit}} no longer
#' recalibrates them. \code{gamma} and \code{beta} still train unless the layer
#' is also frozen with \code{\link{ggml_freeze_weights}} -- the usual fine-tuning
#' setting when batches are small. \code{ggml_unfreeze_bn_stats()} returns the
#' layers to batch statistics in training and recalibration after a fit, which
#' is the default.
#'
#' Freezing the weights alone does not fix the statistics: a frozen batch-norm
#' still normalizes each training batch by its own mean and variance, and its
#' running statistics are still recalibrated after \code{ggml_fit()}.
#'
#' The fixed state is saved by \code{\link{ggml_save_model}}. Only sequential
#' models carry running statistics; the functional \code{batch_norm} is a
#' per-sample RMS normalization (see \code{\link{ggml_layer_batch_norm}}),
#' already independent of the batch, and is refused with a message.
#'
#' @inheritParams ggml_freeze_weights
#' @return The model with the selected batch-norm layers changed. Selected
#'   layers of other types are skipped; a selection with no batch-norm layer
#'   at all is an error.
#' @export
ggml_freeze_bn_stats <- function(model, from = 1L, to = length(model$layers),
                                 layer_names = NULL, prefix = NULL, ...) {
  UseMethod("ggml_freeze_bn_stats")
}

#' @rdname ggml_freeze_bn_stats
#' @export
ggml_unfreeze_bn_stats <- function(model, from = 1L, to = length(model$layers),
                                   layer_names = NULL, prefix = NULL, ...) {
  UseMethod("ggml_unfreeze_bn_stats")
}

nn_set_bn_stats_fixed <- function(model, from, to, layer_names, prefix, fixed) {
  idx <- nn_select_seq_layers(model, from, to, layer_names, prefix)
  bn  <- idx[vapply(model$layers[idx], function(l) identical(l$type, "batch_norm"),
                    logical(1))]
  if (!length(bn))
    stop("no batch_norm layer among the selected layers", call. = FALSE)
  for (i in bn) model$layers[[i]]$config$fixed_stats <- fixed
  model
}

#' @export
ggml_freeze_bn_stats.ggml_sequential_model <- function(model, from = 1L,
                                                         to = length(model$layers),
                                                         layer_names = NULL,
                                                         prefix = NULL, ...)
  nn_set_bn_stats_fixed(model, from, to, layer_names, prefix, TRUE)

#' @export
ggml_unfreeze_bn_stats.ggml_sequential_model <- function(model, from = 1L,
                                                           to = length(model$layers),
                                                           layer_names = NULL,
                                                           prefix = NULL, ...)
  nn_set_bn_stats_fixed(model, from, to, layer_names, prefix, FALSE)

nn_bn_stats_functional <- function(fn)
  stop(fn, "(): a functional model has no batch-norm statistics to fix -- its ",
       "batch_norm is a per-sample RMS normalization, already independent of ",
       "the batch. Freeze its gamma and beta with ggml_freeze_weights() instead.",
       call. = FALSE)

#' @export
ggml_freeze_bn_stats.ggml_functional_model <- function(model, from = NULL, to = NULL,
                                                         layer_names = NULL,
                                                         prefix = NULL, ...)
  nn_bn_stats_functional("ggml_freeze_bn_stats")

#' @export
ggml_unfreeze_bn_stats.ggml_functional_model <- function(model, from = NULL, to = NULL,
                                                           layer_names = NULL,
                                                           prefix = NULL, ...)
  nn_bn_stats_functional("ggml_unfreeze_bn_stats")

# Anything else -- ag_* modules above all -- has no layers these marks reach.
# Without a default method R would stop with "no applicable method", which
# names neither what is supported nor what to do instead.
nn_freeze_unsupported <- function(fn, model) {
  cls <- paste(class(model), collapse = "/")
  hint <- if (inherits(model, "ag_layer"))
    paste0(" For ag_* modules, leave the parameters to keep out of the list ",
           "given to the optimizer (e.g. optimizer_adam(params)).")
  else ""
  stop(fn, "() works on models from ggml_model_sequential() and ggml_model(); ",
       "got an object of class '", cls, "'.", hint, call. = FALSE)
}

#' @export
ggml_freeze_weights.default <- function(model, from = 1L, to = length(model$layers),
                                 layer_names = NULL, prefix = NULL, ...)
  nn_freeze_unsupported("ggml_freeze_weights", model)

#' @export
ggml_unfreeze_weights.default <- function(model, from = 1L, to = length(model$layers),
                                 layer_names = NULL, prefix = NULL, ...)
  nn_freeze_unsupported("ggml_unfreeze_weights", model)

#' @export
ggml_freeze_bn_stats.default <- function(model, from = 1L, to = length(model$layers),
                                 layer_names = NULL, prefix = NULL, ...)
  nn_freeze_unsupported("ggml_freeze_bn_stats", model)

#' @export
ggml_unfreeze_bn_stats.default <- function(model, from = 1L, to = length(model$layers),
                                 layer_names = NULL, prefix = NULL, ...)
  nn_freeze_unsupported("ggml_unfreeze_bn_stats", model)

# ============================================================================
# Compile
# ============================================================================

#' Compile a Sequential Model
#'
#' Configures the model for training: infers shapes, creates backend.
#' Weight tensors are created at training time when batch_size is known.
#'
#' @param model A ggml_sequential_model object
#' @param optimizer Optimizer name: "adam" or "sgd"
#' @param loss Loss function name: \code{"categorical_crossentropy"} (expects a
#'   softmax output), \code{"mse"}, \code{"mae"}, \code{"huber"} (smooth L1,
#'   delta 1) or \code{"binary_crossentropy"} (expects a sigmoid output, one
#'   independent label per unit).  \code{"mean_squared_error"},
#'   \code{"mean_absolute_error"}, \code{"huber_loss"} and
#'   \code{"binary_cross_entropy"} are accepted as aliases.
#'   For a multi-output functional model, a list or character vector with one
#'   loss per output -- named after the output layers, or positional:
#'   \code{loss = list(policy = "categorical_crossentropy", value = "mse")}.
#' @param metrics Character vector of metrics to report from
#'   \code{\link{ggml_evaluate}}.  \code{"accuracy"} is reported for
#'   multi-class outputs whether or not it is requested; \code{"mae"},
#'   \code{"mse"} and \code{"rmse"} (and the long forms
#'   \code{"mean_absolute_error"}, \code{"mean_squared_error"}) are computed
#'   only when asked for.  Any other name is dropped with a warning.
#' @param backend Backend to use: "auto" (GPU if available, else CPU), "cpu", or "vulkan"
#' @param loss_weights Optional per-output weights for a multi-output model;
#'   the optimized total is \code{sum(loss_weights[i] * loss_i)}. Named after
#'   the output layers or positional. Default: all outputs weighted 1.
#' @param weight_decay Decoupled weight decay for the optimizer, in \code{[0, 1]};
#'   0 (the default) turns it off. It is applied to trainable weights only:
#'   layers frozen with \code{\link{ggml_freeze_weights}} are not decayed.
#' @param predict_precision Matrix-multiply precision for prediction:
#'   \code{"default"} (as ggml builds it; f16 tiles on an fp16 Vulkan device)
#'   or \code{"f32"}. Used by \code{\link{ggml_predict}}, a trainer's
#'   \code{$predict()} and its snapshots unless a call gives its own
#'   \code{matmul_precision}; training is unaffected. See the precision section
#'   of \code{\link{ggml_predict}}.
#' @return The compiled model (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_max_pooling_2d(c(2, 2)) |>
#'   ggml_layer_flatten() |>
#'   ggml_layer_dense(10, activation = "softmax")
#' model <- ggml_compile(model, optimizer = "adam",
#'                       loss = "categorical_crossentropy")
#' }
ggml_compile <- function(model, optimizer = "adam",
                          loss = "categorical_crossentropy",
                          metrics = c("accuracy"),
                          backend = "auto",
                          loss_weights = NULL,
                          weight_decay = 0,
                          predict_precision = "default") {
  UseMethod("ggml_compile")
}

# Matrix-multiply precision: "default" leaves each multiply as ggml builds it
# (f16 tiles on an fp16 Vulkan device), "f32" marks it GGML_PREC_F32.
NN_MATMUL_PRECISIONS <- c("default", "f32")

nn_check_precision <- function(x, arg) {
  if (!is.character(x) || length(x) != 1L || !x %in% NN_MATMUL_PRECISIONS)
    stop("'", arg, "' must be \"default\" or \"f32\"", call. = FALSE)
  x
}

# The precision a prediction runs at: the call's argument, then the model's
# predict_precision, then "default".
nn_predict_precision <- function(model, override = NULL) {
  if (!is.null(override)) return(nn_check_precision(override, "matmul_precision"))
  model$compilation$predict_precision %||% "default"
}

# Marks every multiply in a built graph with the precision; nothing to do for
# "default". GGML_PREC_F32 is 10 (ggml.h).
nn_apply_matmul_precision <- function(graph, precision) {
  if (identical(precision, "f32"))
    .Call("R_ggml_graph_set_matmul_prec", graph, 10L, PACKAGE = "ggmlR")
  invisible(NULL)
}

# Supported names, shared by both compile() methods. The training code maps
# these with switch() statements whose fallback is cross-entropy/adamw, so an
# unrecognised name would otherwise be silently substituted -- and
# ggml_evaluate(), which has no such fallback, would then report loss = NA for
# the very same model.
NN_LOSSES <- c("categorical_crossentropy", "crossentropy", "cross_entropy",
               "mse", "mean_squared_error",
               "mae", "mean_absolute_error",
               "huber", "huber_loss",
               "binary_crossentropy", "binary_cross_entropy")
NN_OPTIMIZERS <- c("adam", "adamw", "sgd")
# Metrics ggml_evaluate() knows how to compute. "accuracy" is reported for
# multi-class outputs whether or not it is asked for; the regression metrics are
# computed only when requested.
NN_METRICS <- c("accuracy", "acc",
                "mae", "mean_absolute_error",
                "mse", "mean_squared_error",
                "rmse")

nn_validate_compilation <- function(optimizer, loss, metrics) {
  # `loss` may be a single name (all heads alike) or, for multi-output models,
  # a list/vector with one name per output head -- optionally named after the
  # output layers. Every element still has to be a supported loss name.
  loss_names <- if (is.list(loss)) unlist(loss, use.names = FALSE) else loss
  if (!is.character(loss_names) || length(loss_names) < 1L ||
      !all(loss_names %in% NN_LOSSES)) {
    bad <- if (is.character(loss_names)) setdiff(loss_names, NN_LOSSES) else loss
    stop("Unsupported loss: ", paste(format(bad), collapse = ", "),
         ". Supported: ", paste(NN_LOSSES, collapse = ", "),
         ". (binary_crossentropy is not implemented; for two classes use ",
         "categorical_crossentropy with one-hot labels.)", call. = FALSE)
  }
  if (!is.character(optimizer) || length(optimizer) != 1L ||
      !(optimizer %in% NN_OPTIMIZERS)) {
    stop("Unsupported optimizer: ", paste(format(optimizer), collapse = ", "),
         ". Supported: ", paste(NN_OPTIMIZERS, collapse = ", "), call. = FALSE)
  }
  # Accuracy is reported for multi-class outputs whether or not it is asked
  # for; the regression metrics in NN_METRICS are computed by ggml_evaluate()
  # when requested. Anything outside that list would be silently dropped, so
  # say so rather than letting the caller assume it is being tracked.
  if (!is.null(metrics)) {
    unknown <- setdiff(as.character(metrics), NN_METRICS)
    if (length(unknown) > 0L) {
      warning("Ignoring unsupported metric(s): ", paste(unknown, collapse = ", "),
              ". Supported: ", paste(NN_METRICS, collapse = ", "), ".",
              call. = FALSE)
    }
  }
  invisible(TRUE)
}

# Map a loss name to its ggml loss-type constant. Kept next to NN_LOSSES so the
# two cannot drift apart: every name accepted by nn_validate_compilation() must
# be handled here.
nn_loss_type_of <- function(loss_name) {
  switch(loss_name,
    "categorical_crossentropy" = ,
    "crossentropy" = ,
    "cross_entropy" = ggml_opt_loss_type_cross_entropy(),
    "mse" = ,
    "mean_squared_error" = ggml_opt_loss_type_mse(),
    "mae" = ,
    "mean_absolute_error" = ggml_opt_loss_type_mae(),
    "huber" = ,
    "huber_loss" = ggml_opt_loss_type_huber(),
    "binary_crossentropy" = ,
    "binary_cross_entropy" = ggml_opt_loss_type_binary_cross_entropy(),
    stop("Unsupported loss: ", loss_name, call. = FALSE)
  )
}

# Is this a cross-entropy loss? CE training needs logits, since
# ggml_cross_entropy_loss() applies its own softmax -- so a head with this loss
# has its final softmax stripped when the training graph is built.
#
# "binary_crossentropy" is deliberately NOT in this list: it treats every output
# as an independent Bernoulli and consumes probabilities, so its sigmoid must
# survive into the training graph. Stripping it would feed logits to log().
nn_loss_is_ce <- function(loss_name) {
  loss_name %in% c("categorical_crossentropy", "crossentropy", "cross_entropy")
}

# Resolve the compiled `loss`/`loss_weights` against a model's output heads.
#
# `loss` is either one name for every head, or one name per head -- as a list or
# character vector, named after the output layers or positional. `output_names`
# comes from the model's outputs and is used for name-based matching (keras
# semantics) and for labelling the per-head history.
#
# Returns a list with one entry per head: name, loss_type, weight, is_ce.
nn_resolve_losses <- function(loss, loss_weights, output_names) {
  n_head <- length(output_names)

  spec_names <- if (is.list(loss)) unlist(loss, use.names = FALSE) else loss
  keys       <- names(loss)

  if (length(spec_names) == 1L && is.null(keys)) {
    spec_names <- rep(spec_names, n_head) # one loss for every head
  } else if (length(spec_names) != n_head) {
    stop("'loss' must have one entry per output (", length(spec_names),
         " given, ", n_head, " expected).", call. = FALSE)
  } else if (!is.null(keys) && !any(keys == "")) {
    # Named: match outputs by name, as keras does, rather than by position.
    unknown <- setdiff(keys, output_names)
    if (length(unknown) > 0L) {
      stop("'loss' names do not match model outputs: ",
           paste(unknown, collapse = ", "), ". Model outputs: ",
           paste(output_names, collapse = ", "), call. = FALSE)
    }
    spec_names <- spec_names[match(output_names, keys)]
  }

  w <- loss_weights
  if (is.null(w)) {
    w <- rep(1, n_head)
  } else {
    if (is.list(w)) w <- unlist(w, use.names = FALSE)
    wkeys <- names(loss_weights)
    if (length(w) == 1L && is.null(wkeys)) {
      w <- rep(w, n_head)
    } else if (length(w) != n_head) {
      stop("'loss_weights' must have one entry per output (", length(w),
           " given, ", n_head, " expected).", call. = FALSE)
    } else if (!is.null(wkeys) && !any(wkeys == "")) {
      unknown <- setdiff(wkeys, output_names)
      if (length(unknown) > 0L) {
        stop("'loss_weights' names do not match model outputs: ",
             paste(unknown, collapse = ", "), call. = FALSE)
      }
      w <- w[match(output_names, wkeys)]
    }
  }
  if (!is.numeric(w) || anyNA(w)) {
    stop("'loss_weights' must be numeric and free of NA.", call. = FALSE)
  }

  lapply(seq_len(n_head), function(i) {
    list(
      name      = output_names[[i]],
      loss      = spec_names[[i]],
      loss_type = nn_loss_type_of(spec_names[[i]]),
      weight    = as.numeric(w[[i]]),
      is_ce     = nn_loss_is_ce(spec_names[[i]])
    )
  })
}

#' @rdname ggml_compile
#' @export
ggml_compile.ggml_sequential_model <- function(model, optimizer = "adam",
                                                loss = "categorical_crossentropy",
                                                metrics = c("accuracy"),
                                                backend = "auto",
                                                loss_weights = NULL,
                                                weight_decay = 0,
                                                predict_precision = "default") {
  if (length(model$layers) == 0) {
    stop("Model has no layers. Add layers before compiling.")
  }
  # Sequential models have exactly one output, so per-head losses/weights have
  # nothing to attach to. Rejecting is better than accepting and ignoring them.
  if (!is.null(loss_weights)) {
    stop("'loss_weights' applies to multi-output models; a sequential model ",
         "has a single output.", call. = FALSE)
  }
  if (length(if (is.list(loss)) unlist(loss, use.names = FALSE) else loss) != 1L) {
    stop("A sequential model has a single output, so 'loss' must be one loss name.",
         call. = FALSE)
  }
  nn_validate_compilation(optimizer, loss, metrics)
  weight_decay <- nn_check_weight_decay(weight_decay)
  predict_precision <- nn_check_precision(predict_precision, "predict_precision")

  # 1. Shape inference
  model <- nn_infer_shapes(model)

  # 2. Create backend and scheduler
  use_vulkan <- FALSE
  if (backend == "auto") {
    if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) {
      use_vulkan <- TRUE
    }
  } else if (backend == "vulkan") {
    if (!ggml_vulkan_available() || ggml_vulkan_device_count() == 0) {
      stop("Vulkan backend requested but not available. ",
           "Install libvulkan-dev and glslc, then reinstall ggmlR")
    }
    use_vulkan <- TRUE
  } else if (backend != "cpu") {
    stop("Unknown backend: '", backend, "'. Use 'auto', 'cpu', or 'vulkan'.")
  }

  if (use_vulkan) {
    gpu_backend <- ggml_vulkan_init(0L)
    # sched_new auto-adds CPU as fallback for unsupported ops
    sched <- ggml_backend_sched_new(list(gpu_backend), parallel = FALSE)
    # Use separate CPU backend for weight allocation
    cpu_backend <- ggml_backend_cpu_init()
    if (!isTRUE(.ggmlr_state$backend_msg_shown)) {
      message("Using Vulkan GPU backend: ", ggml_vulkan_device_description(0L))
      .ggmlr_state$backend_msg_shown <- TRUE
    }
  } else {
    cpu_backend <- ggml_backend_cpu_init()
    sched <- ggml_backend_sched_new(list(cpu_backend), parallel = FALSE)
    if (!isTRUE(.ggmlr_state$backend_msg_shown)) {
      message("Using CPU backend")
      .ggmlr_state$backend_msg_shown <- TRUE
    }
  }

  # 3. Store compilation config (weights created later in fit/evaluate)
  # Weights go on GPU for performance (avoids per-iteration copies)
  if (use_vulkan) {
    model$compilation$backend <- gpu_backend
    model$compilation$cpu_backend <- cpu_backend
  } else {
    model$compilation$backend <- cpu_backend
  }
  model$compilation$sched <- sched
  model$compilation$optimizer <- optimizer
  model$compilation$loss <- loss
  model$compilation$metrics <- metrics
  model$compilation$weight_decay <- weight_decay
  model$compilation$predict_precision <- predict_precision
  # Record requested vs actually-used backend so a silent "auto" -> CPU
  # fallback is inspectable later (see ggml_model_backend()).
  model$compilation$backend_requested <- backend
  model$compilation$backend_used      <- if (use_vulkan) "vulkan" else "cpu"
  model$compilation$device <- if (use_vulkan) {
    ggml_vulkan_device_description(0L)
  } else "cpu"
  model$compiled <- TRUE

  invisible(model)
}

# ============================================================================
# Internal: Build graph with weights for a given batch_size
# ============================================================================

# A sequential model whose first layer is an embedding is fed token indices.
nn_seq_takes_indices <- function(model)
  length(model$layers) > 0L && identical(model$layers[[1L]]$type, "embedding")

# The vocabulary its token indices must fit, or NULL when the input is not
# indices. For a functional model: the smallest table among the embeddings fed
# straight from an int32 input.
nn_index_vocab <- function(model) {
  if (inherits(model, "ggml_sequential_model"))
    return(if (nn_seq_takes_indices(model)) model$layers[[1L]]$config$vocab_size)
  inp <- model$inputs[[1L]]
  if (!identical(inp$config$dtype, "int32")) return(NULL)
  v <- integer(0)
  for (node in nn_topo_sort(model$outputs))
    if (identical(node$node_type, "embedding") &&
        identical(node$parents[[1L]]$id, inp$id))
      v <- c(v, node$config$vocab_size)
  if (length(v)) min(v)
}

#' Build computation graph with allocated weights and inputs
#' @param model Compiled model
#' @param batch_size Batch size
#' @return List with ctx_weights, ctx_compute, inputs, outputs, buffer
#' @keywords internal
# logits_output: drop a final softmax from the graph, leaving the last layer's
# raw logits as the output. ggml_cross_entropy_loss() applies log_softmax to its
# own input, so feeding it softmax probabilities would apply softmax twice --
# that flattens the distribution, understates the reported loss and weakens the
# gradient. Training with a cross-entropy loss therefore builds the graph with
# logits_output = TRUE; inference leaves it FALSE and keeps the probabilities.
nn_build_graph <- function(model, batch_size, training = TRUE,
                           logits_output = FALSE) {
  if (length(model$layers) > 1L)
    for (l in model$layers[-1L]) if (identical(l$type, "embedding"))
      stop("an embedding layer must be the first layer of a sequential model ",
           "(it reads token indices); use the functional API to place it ",
           "elsewhere", call. = FALSE)
  input_shape <- model$input_shape
  ne_datapoint <- prod(input_shape)
  backend <- model$compilation$backend

  # Count total parameters + input tensor size for memory estimation
  total_elements <- 0
  for (layer in model$layers) {
    total_elements <- total_elements + nn_count_layer_params(layer)
  }
  # Add input tensor
  total_elements <- total_elements + ne_datapoint * batch_size

  mem_size <- max((total_elements + 1000) * 4 + length(model$layers) * 2048,
                  2 * 1024 * 1024)
  ctx_weights <- ggml_init(mem_size, no_alloc = TRUE)

  # An embedding first takes token indices, so the input is I32, not F32.
  in_type <- if (nn_seq_takes_indices(model)) GGML_TYPE_I32 else GGML_TYPE_F32

  # Create input tensor in ctx_weights (will be allocated with backend)
  if (length(input_shape) == 3) {
    # Image: R [H, W, C] -> ggml [W, H, C, N]
    inputs <- ggml_new_tensor_4d(ctx_weights, in_type,
                                  input_shape[2], input_shape[1],
                                  input_shape[3], batch_size)
  } else if (length(input_shape) == 2) {
    # Sequence: R [seq_len, input_size] -> ggml [input_size, seq_len, N]
    # (input_size is dim0 so each time step's features are contiguous)
    inputs <- ggml_new_tensor_3d(ctx_weights, in_type,
                                  input_shape[2], input_shape[1], batch_size)
  } else {
    # Flat vector: R [features] -> ggml [features, N]
    inputs <- ggml_new_tensor_2d(ctx_weights, in_type,
                                  ne_datapoint, batch_size)
  }
  ggml_set_name(inputs, "inputs")
  ggml_set_input(inputs)

  # Create weight tensors in ctx_weights
  layers_built <- model$layers
  for (i in seq_along(layers_built)) {
    layer <- layers_built[[i]]

    if (layer$type == "conv_1d") {
      k <- layer$config$kernel_size
      ic <- layer$input_shape[2]
      oc <- layer$config$filters

      # ggml conv_1d kernel: [K, IC, OC]
      layer$weights$kernel <- ggml_new_tensor_3d(ctx_weights, GGML_TYPE_F32,
                                                  k, ic, oc)
      layer$weights$bias <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, oc)
      ggml_set_name(layer$weights$kernel, paste0("conv1d_", i, "_kernel"))
      ggml_set_name(layer$weights$bias, paste0("conv1d_", i, "_bias"))

    } else if (layer$type == "conv_2d") {
      kh <- layer$config$kernel_size[1]
      kw <- layer$config$kernel_size[2]
      ic <- layer$input_shape[3]
      oc <- layer$config$filters

      layer$weights$kernel <- ggml_new_tensor_4d(ctx_weights, GGML_TYPE_F32,
                                                  kw, kh, ic, oc)
      layer$weights$bias <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, oc)
      ggml_set_name(layer$weights$kernel, paste0("conv_", i, "_kernel"))
      ggml_set_name(layer$weights$bias, paste0("conv_", i, "_bias"))

    } else if (layer$type == "dense") {
      fan_in <- if (length(layer$input_shape) == 1) layer$input_shape else prod(layer$input_shape)
      units <- layer$config$units

      layer$weights$weight <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                  fan_in, units)
      layer$weights$bias <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
      ggml_set_name(layer$weights$weight, paste0("dense_", i, "_weight"))
      ggml_set_name(layer$weights$bias, paste0("dense_", i, "_bias"))

    } else if (nn_is_norm_type(layer$type)) {
      # Determine number of features for gamma/beta
      n_features <- if (length(layer$input_shape) == 1) layer$input_shape
                    else if (length(layer$input_shape) == 2) layer$input_shape[2]
                    else layer$input_shape[3]

      layer$weights$gamma <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
      layer$weights$beta <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
      pfx <- switch(layer$type, rms_norm = "rmsn_", layer_norm = "ln_", "bn_")
      ggml_set_name(layer$weights$gamma, paste0(pfx, i, "_gamma"))
      ggml_set_name(layer$weights$beta, paste0(pfx, i, "_beta"))

      # Running estimates used at inference time. Not parameters: they are
      # updated by an EMA during training, never by the optimizer. RMS norm has
      # no batch statistics, so it needs none of this.
      if (layer$type == "batch_norm") {
        layer$weights$running_mean <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
        layer$weights$running_var  <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
        ggml_set_name(layer$weights$running_mean, paste0("bn_", i, "_running_mean"))
        ggml_set_name(layer$weights$running_var, paste0("bn_", i, "_running_var"))
      }

    } else if (layer$type == "embedding") {
      # Table [dim, vocab_size], as in the functional embedding.
      layer$weights$weight <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                   layer$config$dim,
                                                   layer$config$vocab_size)
      ggml_set_name(layer$weights$weight, paste0(layer$name, "_weight"))

    } else if (layer$type == "lstm") {
      # input_shape: c(seq_len, input_size)
      input_sz <- layer$input_shape[2]
      units    <- layer$config$units
      nm       <- layer$name

      layer$weights$W_gates <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                    input_sz, 4L * units)
      layer$weights$U_gates <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                    units, 4L * units)
      layer$weights$b_gates <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32,
                                                    4L * units)
      layer$weights$h0      <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32,
                                                    units)
      layer$weights$c0      <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32,
                                                    units)
      ggml_set_name(layer$weights$W_gates, paste0(nm, "_W_gates"))
      ggml_set_name(layer$weights$U_gates, paste0(nm, "_U_gates"))
      ggml_set_name(layer$weights$b_gates, paste0(nm, "_b_gates"))
      ggml_set_name(layer$weights$h0,      paste0(nm, "_h0"))
      ggml_set_name(layer$weights$c0,      paste0(nm, "_c0"))

    } else if (layer$type == "gru") {
      input_sz <- layer$input_shape[2]
      units    <- layer$config$units
      nm       <- layer$name

      layer$weights$W_zh <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                 input_sz, 2L * units)
      layer$weights$U_zh <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                 units, 2L * units)
      layer$weights$b_zh <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32,
                                                 2L * units)
      layer$weights$W_n  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                 input_sz, units)
      layer$weights$U_n  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
                                                 units, units)
      layer$weights$b_n  <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
      layer$weights$h0   <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
      ggml_set_name(layer$weights$W_zh, paste0(nm, "_W_zh"))
      ggml_set_name(layer$weights$U_zh, paste0(nm, "_U_zh"))
      ggml_set_name(layer$weights$b_zh, paste0(nm, "_b_zh"))
      ggml_set_name(layer$weights$W_n,  paste0(nm, "_W_n"))
      ggml_set_name(layer$weights$U_n,  paste0(nm, "_U_n"))
      ggml_set_name(layer$weights$b_n,  paste0(nm, "_b_n"))
      ggml_set_name(layer$weights$h0,   paste0(nm, "_h0"))
    }

    layers_built[[i]] <- layer
  }

  # Allocate all tensors in ctx_weights (inputs + weights) via backend
  buffer <- ggml_backend_alloc_ctx_tensors(ctx_weights, backend)

  # Initialize weights: prefer weights_data (R vectors from load), then trained
  # tensor weights, then random init
  for (i in seq_along(layers_built)) {
    layer <- layers_built[[i]]
    old_layer <- model$layers[[i]]
    has_weights_data <- !is.null(old_layer$weights_data)
    if (has_weights_data)
      nn_check_saved_weights(layer$weights, old_layer$weights_data, layer$type,
                             sprintf("layer %d (%s)", i, layer$type))
    has_trained_weights <- !is.null(old_layer$weights) &&
      (!is.null(old_layer$weights$kernel) || !is.null(old_layer$weights$weight) ||
       !is.null(old_layer$weights$gamma))

    if (layer$type == "conv_1d") {
      if (has_weights_data && !is.null(old_layer$weights_data$kernel)) {
        ggml_backend_tensor_set_data(layer$weights$kernel, old_layer$weights_data$kernel)
        ggml_backend_tensor_set_data(layer$weights$bias, old_layer$weights_data$bias)
      } else if (has_trained_weights && !is.null(old_layer$weights$kernel)) {
        kernel_data <- ggml_backend_tensor_get_data(old_layer$weights$kernel)
        ggml_backend_tensor_set_data(layer$weights$kernel, kernel_data)
        bias_data <- ggml_backend_tensor_get_data(old_layer$weights$bias)
        ggml_backend_tensor_set_data(layer$weights$bias, bias_data)
      } else {
        k <- layer$config$kernel_size
        fan_in <- k * layer$input_shape[2]
        nn_init_he_uniform(layer$weights$kernel, fan_in)
        nn_init_zeros(layer$weights$bias)
      }
      if (isTRUE(layer$trainable)) {
        ggml_set_param(layer$weights$kernel)
        ggml_set_param(layer$weights$bias)
      }

    } else if (layer$type == "conv_2d") {
      if (has_weights_data && !is.null(old_layer$weights_data$kernel)) {
        # Load from R vectors (save/load path)
        ggml_backend_tensor_set_data(layer$weights$kernel, old_layer$weights_data$kernel)
        ggml_backend_tensor_set_data(layer$weights$bias, old_layer$weights_data$bias)
      } else if (has_trained_weights && !is.null(old_layer$weights$kernel)) {
        # Copy trained weights from tensors
        kernel_data <- ggml_backend_tensor_get_data(old_layer$weights$kernel)
        ggml_backend_tensor_set_data(layer$weights$kernel, kernel_data)
        bias_data <- ggml_backend_tensor_get_data(old_layer$weights$bias)
        ggml_backend_tensor_set_data(layer$weights$bias, bias_data)
      } else {
        # Random initialization
        kh <- layer$config$kernel_size[1]
        kw <- layer$config$kernel_size[2]
        fan_in <- kw * kh * layer$input_shape[3]
        nn_init_he_uniform(layer$weights$kernel, fan_in)
        nn_init_zeros(layer$weights$bias)
      }
      if (isTRUE(layer$trainable)) {
        ggml_set_param(layer$weights$kernel)
        ggml_set_param(layer$weights$bias)
      }

    } else if (layer$type == "dense") {
      if (has_weights_data && !is.null(old_layer$weights_data$weight)) {
        # Load from R vectors (save/load path)
        ggml_backend_tensor_set_data(layer$weights$weight, old_layer$weights_data$weight)
        ggml_backend_tensor_set_data(layer$weights$bias, old_layer$weights_data$bias)
      } else if (has_trained_weights && !is.null(old_layer$weights$weight)) {
        # Copy trained weights from tensors
        weight_data <- ggml_backend_tensor_get_data(old_layer$weights$weight)
        ggml_backend_tensor_set_data(layer$weights$weight, weight_data)
        bias_data <- ggml_backend_tensor_get_data(old_layer$weights$bias)
        ggml_backend_tensor_set_data(layer$weights$bias, bias_data)
      } else {
        # Random initialization
        fan_in <- if (length(layer$input_shape) == 1) layer$input_shape else prod(layer$input_shape)
        fan_out <- layer$config$units
        nn_init_glorot_uniform(layer$weights$weight, fan_in, fan_out)
        nn_init_zeros(layer$weights$bias)
      }
      if (isTRUE(layer$trainable)) {
        ggml_set_param(layer$weights$weight)
        ggml_set_param(layer$weights$bias)
      }

    } else if (nn_is_norm_type(layer$type)) {
      if (has_weights_data && !is.null(old_layer$weights_data$gamma)) {
        ggml_backend_tensor_set_data(layer$weights$gamma, old_layer$weights_data$gamma)
        ggml_backend_tensor_set_data(layer$weights$beta, old_layer$weights_data$beta)
      } else if (has_trained_weights && !is.null(old_layer$weights$gamma)) {
        gamma_data <- ggml_backend_tensor_get_data(old_layer$weights$gamma)
        ggml_backend_tensor_set_data(layer$weights$gamma, gamma_data)
        beta_data <- ggml_backend_tensor_get_data(old_layer$weights$beta)
        ggml_backend_tensor_set_data(layer$weights$beta, beta_data)
      } else {
        # gamma=1, beta=0
        n <- ggml_nelements(layer$weights$gamma)
        ggml_backend_tensor_set_data(layer$weights$gamma, rep(1.0, n))
        nn_init_zeros(layer$weights$beta)
      }

      # Running estimates: restore when available, else start from the identity
      # transform (mean 0, variance 1), matching ag_batch_norm(). RMS norm keeps
      # no batch statistics, so it has no such tensors to restore.
      if (!is.null(layer$weights$running_mean)) {
        nbn <- ggml_nelements(layer$weights$running_mean)
        if (has_weights_data && !is.null(old_layer$weights_data$running_mean)) {
          ggml_backend_tensor_set_data(layer$weights$running_mean, old_layer$weights_data$running_mean)
          ggml_backend_tensor_set_data(layer$weights$running_var, old_layer$weights_data$running_var)
        } else if (has_trained_weights && !is.null(old_layer$weights$running_mean)) {
          ggml_backend_tensor_set_data(layer$weights$running_mean,
            ggml_backend_tensor_get_data(old_layer$weights$running_mean))
          ggml_backend_tensor_set_data(layer$weights$running_var,
            ggml_backend_tensor_get_data(old_layer$weights$running_var))
        } else {
          nn_init_zeros(layer$weights$running_mean)
          ggml_backend_tensor_set_data(layer$weights$running_var, rep(1.0, nbn))
        }
      }

      if (isTRUE(layer$trainable)) {
        ggml_set_param(layer$weights$gamma)
        ggml_set_param(layer$weights$beta)
      }

    } else if (layer$type == "embedding") {
      if (has_weights_data && !is.null(old_layer$weights_data$weight)) {
        ggml_backend_tensor_set_data(layer$weights$weight, old_layer$weights_data$weight)
      } else if (!is.null(old_layer$weights$weight)) {
        ggml_backend_tensor_set_data(layer$weights$weight,
          ggml_backend_tensor_get_data(old_layer$weights$weight))
      } else {
        # Same init as the functional embedding.
        ggml_backend_tensor_set_data(layer$weights$weight,
          runif(ggml_nelements(layer$weights$weight), -0.05, 0.05))
      }
      if (isTRUE(layer$trainable)) ggml_set_param(layer$weights$weight)

    } else if (layer$type == "lstm") {
      units    <- layer$config$units
      input_sz <- layer$input_shape[2]
      # Restore or init W_gates
      if (!is.null(old_layer$weights$W_gates)) {
        ggml_backend_tensor_set_data(layer$weights$W_gates,
          ggml_backend_tensor_get_data(old_layer$weights$W_gates))
        ggml_backend_tensor_set_data(layer$weights$U_gates,
          ggml_backend_tensor_get_data(old_layer$weights$U_gates))
        ggml_backend_tensor_set_data(layer$weights$b_gates,
          ggml_backend_tensor_get_data(old_layer$weights$b_gates))
      } else if (!is.null(old_layer$weights_data$W_gates)) {
        ggml_backend_tensor_set_data(layer$weights$W_gates, old_layer$weights_data$W_gates)
        ggml_backend_tensor_set_data(layer$weights$U_gates, old_layer$weights_data$U_gates)
        ggml_backend_tensor_set_data(layer$weights$b_gates, old_layer$weights_data$b_gates)
      } else {
        nn_init_recurrent_uniform(layer$weights$W_gates)
        nn_init_recurrent_uniform(layer$weights$U_gates)
        nn_init_zeros(layer$weights$b_gates)
      }
      nn_init_zeros(layer$weights$h0)
      nn_init_zeros(layer$weights$c0)
      if (isTRUE(layer$trainable)) {
        ggml_set_param(layer$weights$W_gates)
        ggml_set_param(layer$weights$U_gates)
        ggml_set_param(layer$weights$b_gates)
      }

    } else if (layer$type == "gru") {
      units    <- layer$config$units
      input_sz <- layer$input_shape[2]
      if (!is.null(old_layer$weights$W_zh)) {
        ggml_backend_tensor_set_data(layer$weights$W_zh,
          ggml_backend_tensor_get_data(old_layer$weights$W_zh))
        ggml_backend_tensor_set_data(layer$weights$U_zh,
          ggml_backend_tensor_get_data(old_layer$weights$U_zh))
        ggml_backend_tensor_set_data(layer$weights$b_zh,
          ggml_backend_tensor_get_data(old_layer$weights$b_zh))
        ggml_backend_tensor_set_data(layer$weights$W_n,
          ggml_backend_tensor_get_data(old_layer$weights$W_n))
        ggml_backend_tensor_set_data(layer$weights$U_n,
          ggml_backend_tensor_get_data(old_layer$weights$U_n))
        ggml_backend_tensor_set_data(layer$weights$b_n,
          ggml_backend_tensor_get_data(old_layer$weights$b_n))
      } else if (!is.null(old_layer$weights_data$W_zh)) {
        ggml_backend_tensor_set_data(layer$weights$W_zh, old_layer$weights_data$W_zh)
        ggml_backend_tensor_set_data(layer$weights$U_zh, old_layer$weights_data$U_zh)
        ggml_backend_tensor_set_data(layer$weights$b_zh, old_layer$weights_data$b_zh)
        ggml_backend_tensor_set_data(layer$weights$W_n,  old_layer$weights_data$W_n)
        ggml_backend_tensor_set_data(layer$weights$U_n,  old_layer$weights_data$U_n)
        ggml_backend_tensor_set_data(layer$weights$b_n,  old_layer$weights_data$b_n)
      } else {
        nn_init_recurrent_uniform(layer$weights$W_zh)
        nn_init_recurrent_uniform(layer$weights$U_zh)
        nn_init_zeros(layer$weights$b_zh)
        nn_init_recurrent_uniform(layer$weights$W_n)
        nn_init_recurrent_uniform(layer$weights$U_n)
        nn_init_zeros(layer$weights$b_n)
      }
      nn_init_zeros(layer$weights$h0)
      if (isTRUE(layer$trainable)) {
        ggml_set_param(layer$weights$W_zh)
        ggml_set_param(layer$weights$U_zh)
        ggml_set_param(layer$weights$b_zh)
        ggml_set_param(layer$weights$W_n)
        ggml_set_param(layer$weights$U_n)
        ggml_set_param(layer$weights$b_n)
      }
    }
  }

  # Create compute context for intermediate tensors (no_alloc, ggml_opt manages)
  compute_mem <- max(64 * 1024 * 1024,
                     ne_datapoint * batch_size * 4 * 20)
  ctx_compute <- ggml_init(compute_mem, no_alloc = TRUE)

  # Build forward graph.  When the caller wants logits, strip the softmax from
  # the last layer for the duration of the build; layers_built is a local copy,
  # so the model's own definition is untouched and inference still sees it.
  n_layers <- length(layers_built)
  softmax_stripped <- FALSE
  if (logits_output && n_layers > 0L) {
    last_act <- layers_built[[n_layers]]$config$activation
    if (identical(last_act, "softmax")) {
      layers_built[[n_layers]]$config$activation <- NULL
      softmax_stripped <- TRUE
    }
  }

  current <- inputs
  for (i in seq_along(layers_built)) {
    current <- nn_build_layer(ctx_compute, current, layers_built[[i]],
                              training = training)
  }
  outputs <- current

  # Restore the activation so the returned layers still describe the model.
  if (softmax_stripped) {
    layers_built[[n_layers]]$config$activation <- "softmax"
  }
  ggml_set_output(outputs)

  list(
    ctx_weights = ctx_weights,
    ctx_compute = ctx_compute,
    inputs = inputs,
    outputs = outputs,
    buffer = buffer,
    layers_built = layers_built
  )
}

# Calibrate batch_norm running statistics after training.
#
# For each batch_norm layer, runs the trained network up to the layer that feeds
# it and records the per-feature mean and variance of those activations over the
# whole training set. Inference then normalizes with these fixed estimates, so a
# prediction depends only on the sample itself rather than on which other
# samples share its batch.
#
# Conv-shaped inputs are calibrated too: their statistics pool over the batch
# and every spatial position, matching nn_bn_normalize_conv(). See
# nn_bn_channel_stats() for how the flattened activations are folded back.
#
#' @keywords internal
# Per-channel mean and population variance of the activations feeding a
# batch_norm layer.
#
# `acts` is what ggml_predict() returns: one row per sample, each row the
# layer's input flattened in ggml order (ne[0] varies fastest). For a flat
# [features] input that is already one column per channel, so a plain colMeans
# does. For conv-shaped inputs the row packs the spatial axes together with the
# channel, and the statistics must pool over batch AND space -- matching what
# nn_bn_normalize_conv() computes during training.
#
# Returns NULL when the activation width does not match the layer, so the caller
# can skip rather than write wrong statistics.
nn_bn_channel_stats <- function(acts, input_shape, n_features) {
  if (!is.matrix(acts) || ncol(acts) != prod(input_shape)) return(NULL)

  pooled <- if (length(input_shape) == 1L) {
    # [features, N]: columns are already channels.
    acts
  } else if (length(input_shape) == 2L) {
    # ggml [C, L, N] with C = input_shape[2]: within a row the channel index
    # varies fastest, so folding to [C, L*samples] puts each channel on a row.
    matrix(as.vector(t(acts)), nrow = as.integer(input_shape[2]))
  } else {
    # ggml [W, H, C, N] with C = input_shape[3]: W and H vary fastest, so a row
    # is [W*H, C]. Stack all samples and average each channel's block.
    wh <- as.integer(input_shape[1]) * as.integer(input_shape[2])
    matrix(as.vector(t(acts)), nrow = wh)
  }

  if (length(input_shape) == 3L) {
    # pooled is [W*H, C*samples]: collapse the spatial rows, then average the
    # per-sample copies of each channel.
    per_col <- colMeans(pooled)
    sq_col  <- colMeans(pooled^2)
    n_ch <- as.integer(input_shape[3])
    mu <- rowMeans(matrix(per_col, nrow = n_ch))
    m2 <- rowMeans(matrix(sq_col,  nrow = n_ch))
  } else if (length(input_shape) == 2L) {
    mu <- rowMeans(pooled)
    m2 <- rowMeans(pooled^2)
  } else {
    mu <- colMeans(pooled)
    m2 <- colMeans(pooled^2)
  }

  if (length(mu) != n_features) return(NULL)
  # Population variance, matching the batch statistics used during training.
  list(mean = as.numeric(mu), var = as.numeric(pmax(m2 - mu^2, 0)))
}

nn_bn_calibrate <- function(model, x) {
  bn_idx <- which(vapply(model$layers, function(l) identical(l$type, "batch_norm") &&
                                                !isTRUE(l$config$fixed_stats),
                         logical(1)))
  if (length(bn_idx) == 0L) return(model)

  # Cap the calibration set: the statistics converge long before the full data
  # is needed, and this keeps fit() cheap for large inputs.
  n_samples <- if (is.matrix(x)) nrow(x) else dim(x)[1]
  n_use <- min(n_samples, 1024L)
  x_use <- slice_first_dim(x, seq_len(n_use))

  for (i in bn_idx) {
    layer <- model$layers[[i]]
    if (is.null(layer$weights$running_mean)) next

    # A model consisting of everything before this batch_norm layer, sharing the
    # trained weight tensors.
    head_model <- model
    head_model$layers <- model$layers[seq_len(i - 1L)]
    if (length(head_model$layers) == 0L) next
    head_model$history <- NULL

    acts <- tryCatch(
      ggml_predict(head_model, x_use, batch_size = min(n_use, 32L)),
      error = function(e) NULL
    )
    if (is.null(acts) || !is.matrix(acts)) next

    nf <- ggml_nelements(layer$weights$running_mean)
    stats <- nn_bn_channel_stats(acts, layer$input_shape, nf)
    if (is.null(stats)) next

    ggml_backend_tensor_set_data(layer$weights$running_mean, stats$mean)
    ggml_backend_tensor_set_data(layer$weights$running_var, stats$var)
  }

  model
}

# ============================================================================
# Fit (Training)
# ============================================================================

#' Train a Model (dispatcher)
#'
#' Dispatcher: if the first argument is a \code{ggml_sequential_model}, delegates
#' to the Keras-style high-level API (\code{ggml_fit_sequential}); otherwise
#' delegates to the low-level optimizer loop (\code{ggml_fit_opt}).
#'
#' \strong{Keras-style (Sequential model):}
#' \describe{
#'   \item{model}{A compiled \code{ggml_sequential_model}}
#'   \item{x}{Training data (matrix or array)}
#'   \item{y}{Training labels (matrix, one-hot encoded for classification)}
#'   \item{epochs}{Number of training epochs (default: 1)}
#'   \item{batch_size}{Batch size (default: 32)}
#'   \item{validation_split}{Fraction of data for validation (default: 0)}
#'   \item{validation_data}{Optional list(x_val, y_val) for validation. Overrides validation_split.}
#'   \item{class_weight}{Named vector of weights per class, e.g. c("0"=1, "1"=10). Cannot be used with sample_weight.}
#'   \item{sample_weight}{Per-sample weights: a numeric vector of length
#'     \code{nrow(x)}, or -- for \code{mean_squared_error} only -- a
#'     \code{nrow(x)} by \code{ncol(y)} matrix acting as a per-output loss mask.
#'     A mask trains only the outputs whose weight is non-zero, which is what a
#'     Q-model needs: 1 on the action that has a TD target, 0 elsewhere. The
#'     denominator is unchanged (\code{sum(w*(pred-y)^2)/nelements}), so masking
#'     does not renormalise over the active outputs. Cannot be used with
#'     class_weight.}
#'   \item{verbose}{0 = silent, 1 = progress (default: 1)}
#'   \item{shuffle}{Shuffle the data (default: TRUE). Shuffled once before the
#'     train/validation split, then the training portion each epoch; the
#'     validation portion stays fixed. FALSE for time series or exactly
#'     reproducible runs.}
#'   \item{callbacks}{List of callback objects (early stopping, LR schedules)}
#'   \item{generator}{Function of no arguments yielding \code{list(x, y)} for
#'     one batch, or \code{NULL} when exhausted -- an alternative to \code{x}
#'     and \code{y} that keeps only the current batch in memory. See
#'     \code{\link{ggml_fit_generator}}, to which the call is forwarded along
#'     with \code{steps_per_epoch}, \code{initial_epoch},
#'     \code{validation_generator} and \code{validation_steps}. The arguments
#'     that are defined in terms of a whole dataset --
#'     \code{validation_split}, \code{validation_data}, \code{class_weight}
#'     and \code{sample_weight} -- are rejected rather than ignored, and
#'     \code{shuffle} is inert because the generator owns its own order.}
#' }
#'
#' \strong{Low-level (optimizer loop):}
#' \describe{
#'   \item{sched}{Backend scheduler}
#'   \item{ctx_compute}{Compute context}
#'   \item{inputs}{Input tensor}
#'   \item{outputs}{Output tensor}
#'   \item{dataset}{Dataset from \code{ggml_opt_dataset_init()}}
#'   \item{loss_type}{Loss type (default: MSE)}
#'   \item{optimizer}{Optimizer type (default: AdamW)}
#'   \item{nepoch}{Number of epochs (default: 10)}
#'   \item{nbatch_logical}{Logical batch size (default: 32)}
#'   \item{val_split}{Validation fraction (default: 0)}
#'   \item{shuffle}{Shuffle the data (default: TRUE)}
#'   \item{callbacks}{List of callback objects}
#'   \item{silent}{Suppress output (default: FALSE)}
#' }
#'
#' @param ... Arguments passed to the appropriate implementation.
#' @return For Sequential models: the trained model (invisibly).
#'   For the low-level API: a data frame with columns
#'   \code{epoch}, \code{train_loss}, \code{train_accuracy},
#'   \code{val_loss}, \code{val_accuracy}.
#' @seealso \code{\link{ggml_fit_opt}}, \code{\link{ggml_compile}}
#' @examples
#' \donttest{
#' ggml_set_n_threads(1L)  # deterministic, single OpenMP pool
#' n <- 128
#' x <- matrix(runif(n * 4), nrow = n, ncol = 4)
#' y <- matrix(0, nrow = n, ncol = 2)
#' for (i in seq_len(n)) { y[i, if (sum(x[i,]) > 2) 1L else 2L] <- 1 }
#'
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(8, activation = "relu") |>
#'   ggml_layer_dense(2, activation = "softmax")
#' model$input_shape <- 4L
#' model <- ggml_compile(model, optimizer = "adam",
#'                       loss = "categorical_crossentropy")
#'
#' # Basic training
#' model <- ggml_fit(model, x, y, epochs = 5, batch_size = 32, verbose = 0)
#'
#' # With validation_data
#' x_val <- matrix(runif(32 * 4), nrow = 32, ncol = 4)
#' y_val <- matrix(0, nrow = 32, ncol = 2)
#' for (i in seq_len(32)) { y_val[i, if (sum(x_val[i,]) > 2) 1L else 2L] <- 1 }
#' model <- ggml_fit(model, x, y, epochs = 3, batch_size = 32,
#'                   validation_data = list(x_val, y_val), verbose = 0)
#'
#' # With class_weight (useful for imbalanced classes)
#' model <- ggml_fit(model, x, y, epochs = 3, batch_size = 32,
#'                   class_weight = c("0" = 1, "1" = 2), verbose = 0)
#'
#' # With sample_weight
#' sw <- runif(n, 0.5, 1.5)
#' model <- ggml_fit(model, x, y, epochs = 3, batch_size = 32,
#'                   sample_weight = sw, verbose = 0)
#' }
#' @export
ggml_fit <- function(model, ...) {
  # Detect low-level call: all keyword args, no positional 'model'
  # (sched = ..., ctx_compute = ..., etc.)
  if (missing(model)) {
    return(do.call(ggml_fit_opt, list(...)))
  }
  UseMethod("ggml_fit")
}

#' @rdname ggml_fit
#' @export
ggml_fit.ggml_sequential_model <- function(model, ...) {
  ggml_fit_sequential(model, ...)
}

#' @rdname ggml_fit
#' @export
ggml_fit.default <- function(model, ...) {
  # Positional first arg — forward to low-level optimizer loop
  ggml_fit_opt(model, ...)
}

ggml_fit_sequential <- function(model, x, y, epochs = 1, batch_size = 32,
                                validation_split = 0.0, validation_data = NULL,
                                class_weight = NULL, sample_weight = NULL,
                                verbose = 1, shuffle = TRUE,
                                callbacks = list(), generator = NULL, ...) {
  # Streaming path: no dataset is materialised, so the arguments that are
  # defined in terms of one are rejected rather than quietly ignored. `shuffle`
  # is simply inert here -- the generator owns the order it yields batches in.
  if (!is.null(generator)) {
    if (!missing(x) && !is.null(x)) {
      stop("Give either 'x'/'y' or 'generator', not both.", call. = FALSE)
    }
    if (!identical(validation_split, 0.0) && validation_split > 0) {
      stop("'validation_split' needs the total number of samples in order to ",
           "cut a tail off it, which a generator does not have. Use ",
           "'validation_generator' with 'validation_steps' instead.",
           call. = FALSE)
    }
    if (!is.null(validation_data)) {
      stop("'validation_data' is not used with a generator; pass ",
           "'validation_generator' with 'validation_steps' instead.",
           call. = FALSE)
    }
    if (!is.null(class_weight) || !is.null(sample_weight)) {
      stop("'class_weight'/'sample_weight' are applied to a whole dataset ",
           "before training and are not supported with a generator; scale the ",
           "labels inside the generator instead.", call. = FALSE)
    }
    return(ggml_fit_generator(model, generator = generator, epochs = epochs,
                              batch_size = batch_size, verbose = verbose,
                              callbacks = callbacks, ...))
  }
  if (!model$compiled) {
    stop("Model must be compiled before training. Call ggml_compile() first.")
  }

  # Apply class_weight / sample_weight by scaling labels (one-hot)
  # For cross-entropy: CE(p, w*y_onehot) = w * CE(p, y_onehot)
  if (!is.null(class_weight) && !is.null(sample_weight)) {
    stop("Specify either class_weight or sample_weight, not both.")
  }
  if (!is.null(class_weight)) {
    # class_weight: named vector or list, e.g. c("0"=1, "1"=10) or list("0"=1, "1"=10)
    # y must be one-hot; true class = argmax per row
    true_classes <- max.col(y) - 1L  # 0-based
    keys <- as.character(true_classes)
    w <- unlist(class_weight)[keys]
    if (anyNA(w)) stop("class_weight must cover all classes present in y.")
    y <- y * w  # scale each row
  }
  # When TRUE, sample weights are applied through the weighted-MSE loss node
  # (sum(w*(pred-y)^2)) instead of by scaling the labels, which would be wrong
  # for squared error. CE is unaffected: CE(p, w*y) == w*CE(p, y).
  # A matrix sample_weight is a per-output mask rather than one weight per
  # sample; loss_mask_ne0 carries its width down to the loss node. NULL keeps
  # the [1, ndata] layout, so the vector case is untouched.
  use_weighted_mse <- FALSE
  loss_mask_ne0 <- NULL
  if (!is.null(sample_weight)) {
    is_mse <- model$compilation$loss %in% c("mse", "mean_squared_error")

    if (is.matrix(sample_weight)) {
      if (!is_mse) {
        stop("A matrix 'sample_weight' is a per-output loss mask and only ",
             "applies to mean_squared_error; got loss '",
             model$compilation$loss, "'.", call. = FALSE)
      }
      if (nrow(sample_weight) != nrow(y)) {
        stop("'sample_weight' must have one row per training sample (",
             nrow(sample_weight), " rows for ", nrow(y), " samples).",
             call. = FALSE)
      }
      if (ncol(sample_weight) != ncol(y)) {
        stop("A matrix 'sample_weight' must have one column per output (",
             ncol(sample_weight), " columns for ", ncol(y), " outputs).",
             call. = FALSE)
      }
      use_weighted_mse <- TRUE
      loss_mask_ne0    <- ncol(sample_weight)
    } else {
      if (length(sample_weight) != nrow(y)) {
        stop("sample_weight length must match number of training samples.")
      }
      if (is_mse) {
        use_weighted_mse <- TRUE   # weights go into the loss node below, not y
      } else {
        y <- y * sample_weight     # cross-entropy: scaling the label is correct
      }
    }
  }

  # Shuffling before the split is only safe when the split is a fraction we
  # choose; an explicit validation_data set is positional and must stay put.
  shuffle_all <- shuffle

  if (!is.null(validation_data)) {
    if (!is.list(validation_data) || length(validation_data) < 2) {
      stop("validation_data must be a list: list(x_val, y_val)")
    }
    x_val <- validation_data[[1]]
    y_val <- validation_data[[2]]
    n_val <- if (is.matrix(x_val)) nrow(x_val) else dim(x_val)[1]
    n_train <- if (is.matrix(x)) nrow(x) else dim(x)[1]
    # Combine train + val; val_split = fraction of combined that is val
    if (is.matrix(x)) {
      x <- rbind(x, x_val)
    } else {
      x <- abind_first(x, x_val)
    }
    y <- rbind(y, y_val)
    # Validation rows carry no user weight; default to 1.0 so weighted MSE
    # reduces to plain MSE on the validation split.
    if (use_weighted_mse) {
      sample_weight <- if (is.matrix(sample_weight)) {
        rbind(sample_weight,
              matrix(1.0, nrow = n_val, ncol = ncol(sample_weight)))
      } else {
        c(sample_weight, rep(1.0, n_val))
      }
    }
    validation_split <- n_val / (n_train + n_val)
    # The split is positional now: these rows ARE the user's validation set, so
    # a pre-split shuffle would mix them back into training. Per-epoch shuffling
    # of the training portion is unaffected.
    shuffle_all <- FALSE
  }

  input_shape <- model$input_shape
  n_samples <- if (is.matrix(x)) nrow(x) else dim(x)[1]
  ne_datapoint <- prod(input_shape)
  ne_label <- ncol(y)

  # Ensure batch_size divides data evenly
  usable_samples <- (n_samples %/% batch_size) * batch_size
  if (usable_samples < n_samples) {
    dropped <- n_samples - usable_samples
    if (verbose > 0) {
      message("Note: dropping last ", dropped, " sample(s) (", n_samples,
              " -> ", usable_samples, ") because batch_size=", batch_size,
              " must divide evenly. Training metrics are computed on ",
              usable_samples, " samples only.")
    }
    x <- slice_first_dim(x, seq_len(usable_samples))
    y <- y[seq_len(usable_samples), , drop = FALSE]
    if (use_weighted_mse) {
      sample_weight <- if (is.matrix(sample_weight)) {
        sample_weight[seq_len(usable_samples), , drop = FALSE]
      } else {
        sample_weight[seq_len(usable_samples)]
      }
    }
    n_samples <- usable_samples
  }

  # Prepare dataset
  dataset <- ggml_opt_dataset_init(
    type_data = if (nn_seq_takes_indices(model)) GGML_TYPE_I32 else GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = ne_datapoint,
    ne_label = ne_label,
    ndata = n_samples,
    ndata_shard = 1
  )

  # Convert data to ggml format -- the layout the generator and the trainer use,
  # with token indices checked against the embedding's vocabulary.
  x_ggml <- .gen_to_ggml_layout(x, input_shape, nn_index_vocab(model))

  # Labels: R [N, classes] -> ggml [classes, N]
  y_ggml <- as.vector(t(y))

  # Fill dataset
  data_tensor <- ggml_opt_dataset_data(dataset)
  labels_tensor <- ggml_opt_dataset_labels(dataset)
  ggml_backend_tensor_set_data(data_tensor, x_ggml)
  ggml_backend_tensor_set_data(labels_tensor, y_ggml)

  # Loss weights for weighted MSE (loss = sum(w*(pred-y)^2)/nelem).
  # A vector gives [1, ndata] (one weight per sample); a matrix gives
  # [n_out, ndata] -- a per-output mask, transposed on the way in exactly like
  # the labels above, since ggml wants the output axis on ne[0].
  if (use_weighted_mse) {
    w_ne0 <- if (is.null(loss_mask_ne0)) 1L else as.integer(loss_mask_ne0)
    weights_tensor <- ggml_opt_dataset_weights(dataset, w_ne0)
    w_ggml <- if (is.matrix(sample_weight)) {
      as.numeric(t(sample_weight))
    } else {
      as.numeric(sample_weight)
    }
    ggml_backend_tensor_set_data(weights_tensor, w_ggml)
  }

  # Build graph (creates contexts, weights, inputs, outputs).
  # Cross-entropy training needs logits, not probabilities -- see the note on
  # nn_build_graph(). Weighted MSE goes through a different loss node and wants
  # the model's actual output.
  # Must agree with the loss_type switch below; both reject anything else.
  use_ce_loss <- !use_weighted_mse &&
    model$compilation$loss %in% c("categorical_crossentropy", "crossentropy", "cross_entropy")
  graph_info <- nn_build_graph(model, batch_size, logits_output = use_ce_loss)

  # Map optimizer and loss
  optimizer_type <- switch(model$compilation$optimizer,
    "adam" = , "adamw" = ggml_opt_optimizer_type_adamw(),
    "sgd" = ggml_opt_optimizer_type_sgd(),
    stop("Unsupported optimizer: ", model$compilation$optimizer, call. = FALSE)
  )

  # nn_loss_type_of() rather than a second switch: a duplicate mapping here
  # would silently reject any loss added to the shared one.
  loss_type <- if (use_weighted_mse) {
    ggml_opt_loss_type_weighted_mse()
  } else {
    nn_loss_type_of(model$compilation$loss)
  }

  # Train.  Uses the R-side epoch loop (ggml_fit_opt) rather than the single
  # C call (ggml_opt_fit) so that callbacks get an on_epoch_begin/on_epoch_end
  # hook between epochs; early stopping can also cut the run short, so the
  # history may be shorter than `epochs`.
  #
  # Behaviour matches the old C path: shuffling is identical (whole dataset
  # once, then the training split each epoch), and the LR is unchanged --
  # ggml_opt_get_default_optimizer_params() ignores its userdata and returns
  # hard-coded constants (adamw.alpha = 0.001), which is exactly what
  # ggml_opt_init_for_fit() seeds its LR userdata with.  The difference is only
  # that the LR is now reachable: a ggml_schedule_* callback can change it
  # between epochs via ggml_opt_set_lr().
  history_raw <- ggml_fit_opt(
    sched = model$compilation$sched,
    ctx_compute = graph_info$ctx_compute,
    inputs = graph_info$inputs,
    outputs = graph_info$outputs,
    dataset = dataset,
    loss_type = loss_type,
    optimizer = optimizer_type,
    nepoch = epochs,
    nbatch_logical = batch_size,
    val_split = validation_split,
    shuffle = shuffle,
    shuffle_all = shuffle_all,
    callbacks = callbacks,
    silent = (verbose == 0),
    loss_mask_ne0 = loss_mask_ne0,
    weight_decay = model$compilation$weight_decay %||% 0
  )

  # Store built layers (with trained weights)
  model$layers <- graph_info$layers_built
  model$compilation$ctx_weights <- graph_info$ctx_weights
  model$compilation$buffer <- graph_info$buffer

  # Calibrate batch_norm running statistics for inference.
  #
  # These have to come from a forward pass over the training data with the final
  # weights. The epoch loop above is R-side, but a single epoch still runs as one
  # C call and the intermediate mean/variance tensors it allocates are not
  # addressable from here. So instead of folding an EMA in per batch, the
  # statistics are computed once, after training, from the activations feeding
  # each batch_norm layer. The result is the exact mean and variance over the
  # training set rather than an exponentially-weighted approximation of it,
  # which is what inference actually wants.
  #
  # Must run after ctx_weights/buffer are attached: it calls ggml_predict(),
  # which rebuilds the graph from model$compilation.
  model <- nn_bn_calibrate(model, x)

  # Build history object.  ggml_fit_opt() returns one row per epoch actually
  # run, which is fewer than `epochs` when a callback stopped training early.
  model$history <- structure(
    list(
      train_loss     = history_raw$train_loss,
      train_accuracy = history_raw$train_accuracy,
      val_loss       = history_raw$val_loss,
      val_accuracy   = history_raw$val_accuracy,
      epochs         = seq_len(nrow(history_raw))
    ),
    class = "ggml_history"
  )

  # Cleanup
  ggml_free(graph_info$ctx_compute)
  ggml_opt_dataset_free(dataset)

  invisible(model)
}

# ============================================================================
# Evaluate
# ============================================================================

#' Evaluate a Trained Model
#'
#' @param model A trained ggml_sequential_model
#' @param x Test data
#' @param y Test labels (one-hot encoded)
#' @param batch_size Batch size for evaluation
#' @param sample_weight Numeric vector of per-sample weights (length = nrow(x)).
#' @param class_weight Named vector of weights per class, e.g. c("0"=1, "1"=10). Cannot be used with sample_weight.
#' @return Named list with \code{loss} and \code{accuracy}. A multi-output
#'   functional model additionally reports each head as \code{<output>_loss}.
#'   Note the deliberate difference from the training history, which names the
#'   same quantities \code{train_<output>_loss} and \code{val_<output>_loss}:
#'   there the prefix distinguishes the two phases (and keeps an output named
#'   \code{val_x} from colliding with another output's validation column),
#'   whereas evaluation has no phases to tell apart, so the bare name is used.
#' @examples
#' \donttest{
#' ggml_set_n_threads(1L)  # deterministic, single OpenMP pool
#' n <- 128
#' x <- matrix(runif(n * 4), nrow = n, ncol = 4)
#' y <- matrix(0, nrow = n, ncol = 2)
#' for (i in seq_len(n)) { y[i, if (sum(x[i,]) > 2) 1L else 2L] <- 1 }
#'
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(8, activation = "relu") |>
#'   ggml_layer_dense(2, activation = "softmax")
#' model$input_shape <- 4L
#' model <- ggml_compile(model, optimizer = "adam",
#'                       loss = "categorical_crossentropy")
#' model <- ggml_fit(model, x, y, epochs = 5, batch_size = 32, verbose = 0)
#'
#' # Basic evaluation
#' result <- ggml_evaluate(model, x, y, batch_size = 32)
#'
#' # With sample_weight
#' sw <- runif(n, 0.5, 1.5)
#' result <- ggml_evaluate(model, x, y, batch_size = 32, sample_weight = sw)
#'
#' # With class_weight
#' result <- ggml_evaluate(model, x, y, batch_size = 32,
#'                         class_weight = c("0" = 1, "1" = 2))
#' }
#' @export
ggml_evaluate <- function(model, ...) {
  UseMethod("ggml_evaluate")
}

#' @rdname ggml_evaluate
#' @export
ggml_evaluate.ggml_sequential_model <- function(model, x, y, batch_size = 32,
                                                  sample_weight = NULL,
                                                  class_weight = NULL, ...) {
  if (!model$compiled) {
    stop("Model must be compiled before evaluation.")
  }

  if (!is.null(class_weight) && !is.null(sample_weight)) {
    stop("Specify either class_weight or sample_weight, not both.")
  }
  if (!is.null(class_weight)) {
    true_classes <- max.col(y) - 1L
    keys <- as.character(true_classes)
    w <- unlist(class_weight)[keys]
    if (anyNA(w)) stop("class_weight must cover all classes present in y.")
    y <- y * w
  }
  if (!is.null(sample_weight)) {
    if (length(sample_weight) != nrow(y)) {
      stop("sample_weight length must match number of samples.")
    }
    y <- y * sample_weight
  }

  n_samples <- nrow(y)
  ne_label <- ncol(y)

  # Get predictions for ALL samples (no truncation)
  preds <- ggml_predict(model, x, batch_size = batch_size)

  # Compute loss. nn_head_loss() is the shared implementation the functional
  # path uses, so both report the same number for the same loss name -- and a
  # loss added there is not silently NA here.
  loss_val <- nn_head_loss(model$compilation$loss, preds, y)

  # Compute accuracy (classification: argmax match)
  if (ne_label > 1L) {
    pred_classes <- max.col(preds)
    true_classes <- max.col(y)
    acc_val <- mean(pred_classes == true_classes)
  } else {
    acc_val <- NA_real_
  }

  out <- list(loss = loss_val, accuracy = acc_val, n_samples = n_samples)
  # Same helper the functional path uses, so a metric means the same thing in
  # both APIs -- previously the sequential evaluate() computed none at all.
  nn_add_extra_metrics(out, model$compilation$metrics, preds, y)
}

# ============================================================================
# Predict
# ============================================================================

#' Get Predictions from a Trained Model
#'
#' Runs forward pass on input data and returns prediction probabilities
#' (or raw output values for regression). Unlike \code{ggml_evaluate()}, this
#' does not require labels.
#'
#' @param model A trained ggml_sequential_model
#' @param x Input data (matrix or array)
#' @param batch_size Batch size for inference
#' @param matmul_precision \code{"default"} or \code{"f32"} for this call;
#'   \code{NULL} (the default) takes the model's \code{predict_precision}
#'   from \code{\link{ggml_compile}}. See the section on precision.
#' @section Precision:
#'   On a Vulkan device with fp16 support the matrix multiplies keep their
#'   tiles in f16 by default. Results then carry a relative error near 1e-3
#'   that depends on the batch size (a single row takes an exact f32 path,
#'   larger batches the tiled one), so the same input can predict slightly
#'   differently in a batch of 1 and a batch of 64. \code{"f32"} marks every
#'   multiply \code{GGML_PREC_F32} and selects f32 tiles: the result then
#'   matches the CPU to about 1e-6, at a cost in matmul throughput that depends
#'   on the device and driver. The CPU backend is exact either way.
#'
#'   Which setting wins: the \code{matmul_precision} argument of the call, then
#'   the model's \code{predict_precision}, then \code{"default"}.
#' @return Matrix of predictions with shape \code{[N, output_units]}
#' @export
ggml_predict <- function(model, ...) {
  # A trainer snapshot whose memory has been released would read freed tensors.
  if (is.environment(model$snapshot_state) && !isTRUE(model$snapshot_state$alive)) {
    stop("this snapshot has been released with ggml_snapshot_free(); take a ",
         "new one with $snapshot().", call. = FALSE)
  }
  UseMethod("ggml_predict")
}

#' @rdname ggml_predict
#' @export
ggml_predict.ggml_sequential_model <- function(model, x, batch_size = 32L,
                                               matmul_precision = NULL, ...) {
  precision <- nn_predict_precision(model, matmul_precision)
  if (!model$compiled) {
    stop("Model must be compiled before prediction.")
  }

  input_shape <- model$input_shape
  n_samples <- if (is.matrix(x)) nrow(x) else dim(x)[1]
  ne_datapoint <- prod(input_shape)

  # Get output size from last layer
  last_layer <- model$layers[[length(model$layers)]]
  ne_output <- if (length(last_layer$output_shape) == 1) {
    last_layer$output_shape
  } else {
    prod(last_layer$output_shape)
  }

  if (n_samples == 0L) {
    stop("No samples provided.")
  }

  # Convert all data to ggml format upfront (before any slicing)
  vocab <- nn_index_vocab(model)
  nn_convert_to_ggml <- function(x, input_shape)
    .gen_to_ggml_layout(x, input_shape, vocab)

  # Helper: run forward pass for a batch of given size, return prediction matrix
  nn_predict_batch_run <- function(model, x_ggml, bs, ne_datapoint, ne_output,
                                   n_batches, input_shape) {
    graph_info <- nn_build_graph(model, bs, training = FALSE)
    graph <- ggml_build_forward_expand(graph_info$ctx_compute, graph_info$outputs)
    nn_apply_matmul_precision(graph, precision)
    sched <- model$compilation$sched
    preds <- matrix(0, nrow = n_batches * bs, ncol = ne_output)

    # Allocate once, outside the loop. Re-running reset + alloc_graph per batch
    # gives the scheduler a fresh chance to lay out the intermediate buffers, and
    # on Vulkan the second and later passes then read stale data -- every batch
    # after the first came out wrong (~0.4 absolute error on softmax outputs)
    # while the first was exact. The weights live in their own buffer, so nothing
    # needs re-uploading between batches: set the inputs and compute. This
    # mirrors the ONNX path, which allocates once behind an `is_allocated` flag.
    ggml_backend_sched_reset(sched)
    ggml_backend_sched_alloc_graph(sched, graph)

    for (ib in seq_len(n_batches)) {
      data_start <- (ib - 1L) * bs * ne_datapoint + 1L
      data_end <- ib * bs * ne_datapoint
      batch_data <- x_ggml[data_start:data_end]
      ggml_backend_tensor_set_data(graph_info$inputs, batch_data)

      ggml_backend_sched_graph_compute(sched, graph)

      batch_output <- ggml_backend_tensor_get_data(graph_info$outputs)
      batch_matrix <- matrix(batch_output, nrow = ne_output, ncol = bs)
      row_start <- (ib - 1L) * bs + 1L
      row_end <- ib * bs
      preds[row_start:row_end, ] <- t(batch_matrix)
    }

    ggml_free(graph_info$ctx_compute)
    ggml_backend_buffer_free(graph_info$buffer)
    ggml_free(graph_info$ctx_weights)

    preds
  }

  n_full <- (n_samples %/% batch_size) * batch_size
  remainder <- n_samples - n_full

  all_preds <- matrix(0, nrow = n_samples, ncol = ne_output)

  # Main batches
  if (n_full > 0L) {
    x_main <- slice_first_dim(x, seq_len(n_full))
    x_ggml_main <- nn_convert_to_ggml(x_main, input_shape)
    n_batches <- n_full %/% batch_size
    all_preds[seq_len(n_full), ] <- nn_predict_batch_run(
      model, x_ggml_main, batch_size, ne_datapoint, ne_output,
      n_batches, input_shape
    )
  }

  # Remainder batch with rebuilt graph
  if (remainder > 0L) {
    idx_rem <- (n_full + 1L):n_samples
    x_rem <- slice_first_dim(x, idx_rem)
    x_ggml_rem <- nn_convert_to_ggml(x_rem, input_shape)
    all_preds[idx_rem, ] <- nn_predict_batch_run(
      model, x_ggml_rem, remainder, ne_datapoint, ne_output,
      1L, input_shape
    )
  }

  all_preds
}

#' Predict Classes from a Trained Model
#'
#' Returns predicted class indices (1-based) by applying argmax
#' to the output of \code{ggml_predict()}.
#'
#' @param model A trained ggml_sequential_model
#' @param x Input data (matrix or array)
#' @param batch_size Batch size for inference
#' @return Integer vector of predicted class indices (1-based)
#' @export
ggml_predict_classes <- function(model, x, batch_size = 32L) {
  probs <- ggml_predict(model, x, batch_size)
  apply(probs, 1, which.max)
}

# ============================================================================
# Save / Load Weights
# ============================================================================

# File format written by ggml_save_model() and the newest ggml_load_model()
# reads. 1 is ggml_save_weights(); 2 added whole models; 3 adds a functional
# model's ggml_freeze_weights() marks (frozen_nodes), the compiled
# weight_decay and predict_precision, and batch-norm fixed_stats. Raise it
# whenever a field appears that an older reader would drop without noticing:
# the reader refuses anything newer than this, rather than load it with the
# field ignored.
NN_MODEL_FORMAT <- 3L

# The weights ggml_save_model() writes for each layer type, by the names the
# graph builders read them back under. A layer with saved data must have every
# one of them: a missing name would otherwise fall through to random init.
NN_SAVED_WEIGHTS <- list(
  dense                = c("weight", "bias"),
  conv_1d              = c("kernel", "bias"),
  conv_2d              = c("kernel", "bias"),
  batch_norm           = c("gamma", "beta"),
  layer_norm           = c("gamma", "beta"),
  rms_norm             = c("gamma", "beta"),
  embedding            = "weight",
  positional_embedding = "pos",
  lstm                 = c("W_gates", "U_gates", "b_gates"),
  gru                  = c("W_zh", "U_zh", "b_zh", "W_n", "U_n", "b_n"),
  attention            = c("W_q", "W_k", "W_v", "W_o")
)
# Saved only by some layers of a type; checked when present.
NN_SAVED_OPTIONAL <- c("running_mean", "running_var", "b_o")

# Checks saved weight vectors against the tensors they are about to be written
# into, before any is written. A file from another architecture, or a damaged
# one, then stops with the layer and tensor named instead of being copied in
# short (ggml reads past the vector) or long (silently cut).
nn_check_saved_weights <- function(tensors, data, type, where) {
  if (length(data) == 0L) return(invisible(NULL))
  required <- NN_SAVED_WEIGHTS[[type]]
  missing  <- setdiff(required, names(data))
  if (length(missing))
    stop(where, ": the saved weights lack ", paste(missing, collapse = ", "),
         " -- the file does not match this model", call. = FALSE)
  for (nm in intersect(c(required, NN_SAVED_OPTIONAL), names(data))) {
    v <- data[[nm]]
    if (is.null(v)) next
    t <- tensors[[nm]]
    if (is.null(t))
      stop(where, ": the file has saved '", nm, "', which this layer does not ",
           "have", call. = FALSE)
    if (!is.numeric(v) || anyNA(v))
      stop(where, ": saved '", nm, "' is not a complete numeric vector -- ",
           "the file is damaged", call. = FALSE)
    n <- ggml_nelements(t)
    if (length(v) != n)
      stop(sprintf("%s: saved '%s' has %d values, the layer expects %d (shape %s)",
                   where, nm, length(v), n,
                   paste(ggml_tensor_shape(t), collapse = " x ")), call. = FALSE)
  }
  invisible(NULL)
}

#' Save Model Weights to File
#'
#' Saves the trained weights of a sequential model to an RDS file.
#' The file includes both weights and architecture metadata for validation
#' when loading.
#'
#' @param model A trained ggml_sequential_model
#' @param path File path to save weights (typically with .rds extension)
#' @return The model (invisibly).
#' @export
ggml_save_weights <- function(model, path) {
  if (!model$compiled) {
    stop("Model must be compiled before saving weights.")
  }

  # Extract weights as R vectors from each layer
  weights_list <- list()
  for (i in seq_along(model$layers)) {
    layer <- model$layers[[i]]
    layer_weights <- list()

    if (layer$type %in% c("conv_1d", "conv_2d")) {
      if (!is.null(layer$weights$kernel)) {
        layer_weights$kernel <- ggml_backend_tensor_get_data(layer$weights$kernel)
        layer_weights$bias <- ggml_backend_tensor_get_data(layer$weights$bias)
      }
    } else if (layer$type == "dense") {
      if (!is.null(layer$weights$weight)) {
        layer_weights$weight <- ggml_backend_tensor_get_data(layer$weights$weight)
        layer_weights$bias <- ggml_backend_tensor_get_data(layer$weights$bias)
      }
    } else if (nn_is_norm_type(layer$type)) {
      if (!is.null(layer$weights$gamma)) {
        layer_weights$gamma <- ggml_backend_tensor_get_data(layer$weights$gamma)
        layer_weights$beta <- ggml_backend_tensor_get_data(layer$weights$beta)
      }
    }

    weights_list[[i]] <- layer_weights
  }

  # Build architecture description for validation on load
  architecture <- list(
    input_shape = model$input_shape,
    n_layers = length(model$layers),
    layer_configs = lapply(model$layers, function(l) {
      list(type = l$type, config = l$config,
           input_shape = l$input_shape, output_shape = l$output_shape)
    })
  )

  data <- list(
    weights = weights_list,
    architecture = architecture,
    version = 1L
  )

  saveRDS(data, path)
  invisible(model)
}

#' Load Model Weights from File
#'
#' Loads previously saved weights into a compiled model. The model architecture
#' must match the saved weights (same layer types, sizes, and shapes).
#'
#' @param model A compiled ggml_sequential_model (same architecture as saved)
#' @param path File path to load weights from
#' @return The model with loaded weights.
#' @export
ggml_load_weights <- function(model, path) {
  if (!model$compiled) {
    stop("Model must be compiled before loading weights.")
  }

  data <- readRDS(path)

  if (is.null(data$weights) || is.null(data$architecture)) {
    stop("Invalid weights file format.")
  }

  # Validate architecture match
  arch <- data$architecture
  if (length(model$layers) != arch$n_layers) {
    stop("Architecture mismatch: model has ", length(model$layers),
         " layers, saved weights have ", arch$n_layers)
  }

  for (i in seq_along(model$layers)) {
    if (model$layers[[i]]$type != arch$layer_configs[[i]]$type) {
      stop("Architecture mismatch at layer ", i, ": model has '",
           model$layers[[i]]$type, "', saved weights have '",
           arch$layer_configs[[i]]$type, "'")
    }
  }

  # Store R weight vectors in layers for nn_build_graph to pick up
  for (i in seq_along(model$layers)) {
    if (length(data$weights[[i]]) > 0) {
      model$layers[[i]]$weights_data <- data$weights[[i]]
    }
  }

  invisible(model)
}

# ============================================================================
# Print / Summary
# ============================================================================

#' Print method for ggml_sequential_model
#'
#' Prints a summary of the model architecture including layer types,
#' output shapes, and parameter counts.
#'
#' @param x A ggml_sequential_model object
#' @param ... Additional arguments (ignored)
#' @return The model object (invisibly).
#' @export
print.ggml_sequential_model <- function(x, ...) {
  model <- x
  cat("ggmlR Sequential Model\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")

  if (length(model$layers) == 0) {
    cat("  (no layers)\n")
    return(invisible(model))
  }

  # Infer shapes if not done yet
  if (is.null(model$layers[[1]]$output_shape) && !is.null(model$input_shape)) {
    model <- nn_infer_shapes(model)
  }

  total_params <- 0
  cat(sprintf("%-20s %-20s %-10s\n", "Layer", "Output Shape", "Params"))
  cat(paste(rep("-", 60), collapse = ""), "\n")

  for (i in seq_along(model$layers)) {
    layer <- model$layers[[i]]

    n_params <- nn_count_layer_params(layer)
    total_params <- total_params + n_params

    shape_str <- if (!is.null(layer$output_shape)) {
      paste0("(", paste(layer$output_shape, collapse = ", "), ")")
    } else {
      "?"
    }

    cat(sprintf("%-20s %-20s %-10d\n", layer$type, shape_str, n_params))
  }

  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat(sprintf("Total parameters: %d\n", total_params))
  cat(sprintf("Compiled: %s\n", if (model$compiled) "yes" else "no"))

  invisible(x)
}

#' Summary method for ggml_sequential_model
#'
#' Prints a detailed summary including input shape, layer details,
#' trainable/non-trainable parameter counts, and memory estimate.
#'
#' @param object A ggml_sequential_model object
#' @param ... Additional arguments (ignored)
#' @return The model object (invisibly).
#' @export
summary.ggml_sequential_model <- function(object, ...) {
  model <- object

  # Infer shapes if needed
  if (length(model$layers) > 0 &&
      is.null(model$layers[[1]]$output_shape) &&
      !is.null(model$input_shape)) {
    model <- nn_infer_shapes(model)
  }

  cat("ggmlR Sequential Model Summary\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")

  if (!is.null(model$input_shape)) {
    cat(sprintf("Input shape: (%s)\n", paste(model$input_shape, collapse = ", ")))
  }
  cat("\n")

  if (length(model$layers) == 0) {
    cat("  (no layers)\n")
    return(invisible(object))
  }

  trainable <- 0
  non_trainable <- 0

  cat(sprintf("%-4s %-20s %-20s %-12s %-12s\n",
              "#", "Layer", "Output Shape", "Trainable", "Non-train."))
  cat(paste(rep("-", 70), collapse = ""), "\n")

  for (i in seq_along(model$layers)) {
    layer <- model$layers[[i]]
    n_params <- nn_count_layer_params(layer)
    trainable <- trainable + n_params

    shape_str <- if (!is.null(layer$output_shape)) {
      paste0("(", paste(layer$output_shape, collapse = ", "), ")")
    } else {
      "?"
    }

    cat(sprintf("%-4d %-20s %-20s %-12d %-12d\n",
                i, layer$type, shape_str, n_params, 0L))
  }

  cat(paste(rep("=", 70), collapse = ""), "\n")
  total <- trainable + non_trainable
  cat(sprintf("Total parameters:         %s\n", format(total, big.mark = ",")))
  cat(sprintf("  Trainable:              %s\n", format(trainable, big.mark = ",")))
  cat(sprintf("  Non-trainable:          %s\n", format(non_trainable, big.mark = ",")))

  # Memory estimate (F32 = 4 bytes per param)
  mem_bytes <- total * 4
  if (mem_bytes >= 1024 * 1024) {
    cat(sprintf("Estimated weight memory:  %.1f MB\n", mem_bytes / (1024 * 1024)))
  } else {
    cat(sprintf("Estimated weight memory:  %.1f KB\n", mem_bytes / 1024))
  }

  cat(sprintf("Compiled:                 %s\n", if (model$compiled) "yes" else "no"))

  invisible(object)
}

#' Count parameters for a single layer
#' @param layer A layer list
#' @return Number of parameters
#' @keywords internal
nn_count_layer_params <- function(layer) {
  if (layer$type == "embedding") {
    return(as.numeric(layer$config$vocab_size) * layer$config$dim)
  }
  if (layer$type == "conv_2d") {
    if (!is.null(layer$input_shape)) {
      ksize <- layer$config$kernel_size
      ksize[1] * ksize[2] * layer$input_shape[3] * layer$config$filters +
        layer$config$filters
    } else 0
  } else if (layer$type == "conv_1d") {
    if (!is.null(layer$input_shape)) {
      layer$config$kernel_size * layer$input_shape[2] * layer$config$filters +
        layer$config$filters
    } else 0
  } else if (layer$type == "dense") {
    if (!is.null(layer$input_shape)) {
      fan_in <- if (length(layer$input_shape) == 1) layer$input_shape else prod(layer$input_shape)
      fan_in * layer$config$units + layer$config$units
    } else 0
  } else if (nn_is_norm_type(layer$type)) {
    if (!is.null(layer$input_shape)) {
      n <- if (length(layer$input_shape) == 1) layer$input_shape
           else if (length(layer$input_shape) == 2) layer$input_shape[2]
           else layer$input_shape[3]
      n * 2L  # gamma + beta
    } else 0
  } else if (layer$type == "lstm") {
    if (!is.null(layer$input_shape)) {
      input_sz <- layer$input_shape[2]
      units    <- layer$config$units
      # W_gates + U_gates + b_gates
      input_sz * 4L * units + units * 4L * units + 4L * units
    } else 0
  } else if (layer$type == "gru") {
    if (!is.null(layer$input_shape)) {
      input_sz <- layer$input_shape[2]
      units    <- layer$config$units
      # W_zh + U_zh + b_zh + W_n + U_n + b_n
      input_sz * 2L * units + units * 2L * units + 2L * units +
        input_sz * units + units * units + units
    } else 0
  } else {
    0
  }
}

# ============================================================================
# History Class
# ============================================================================

#' Print method for ggml_history
#'
#' @param x A ggml_history object
#' @param ... Additional arguments (ignored)
#' @return The history object (invisibly).
#' @export
print.ggml_history <- function(x, ...) {
  n <- length(x$epochs)
  cat("Training History (", n, " epoch", if (n != 1) "s", ")\n", sep = "")
  cat(sprintf("  Final train loss:     %.4f\n", x$train_loss[n]))
  cat(sprintf("  Final train accuracy: %.4f\n", x$train_accuracy[n]))
  if (!is.na(x$val_loss[n])) {
    cat(sprintf("  Final val loss:       %.4f\n", x$val_loss[n]))
    cat(sprintf("  Final val accuracy:   %.4f\n", x$val_accuracy[n]))
  }
  invisible(x)
}

#' Plot training history
#'
#' Plots loss and accuracy curves over epochs.
#'
#' @param x A ggml_history object
#' @param ... Additional arguments (ignored)
#' @return The history object (invisibly).
#' @importFrom graphics plot lines legend par
#' @export
plot.ggml_history <- function(x, ...) {
  has_val <- !is.na(x$val_loss[1])
  old_par <- par(mfrow = c(1, 2))
  on.exit(par(old_par))

  # Loss plot
  ylim_loss <- range(c(x$train_loss, if (has_val) x$val_loss), na.rm = TRUE)
  plot(x$epochs, x$train_loss, type = "l", col = "blue",
       xlab = "Epoch", ylab = "Loss", main = "Loss", ylim = ylim_loss)
  if (has_val) {
    lines(x$epochs, x$val_loss, col = "red")
    legend("topright", legend = c("Train", "Val"),
           col = c("blue", "red"), lty = 1, cex = 0.8)
  }

  # Accuracy plot
  ylim_acc <- range(c(x$train_accuracy, if (has_val) x$val_accuracy), na.rm = TRUE)
  plot(x$epochs, x$train_accuracy, type = "l", col = "blue",
       xlab = "Epoch", ylab = "Accuracy", main = "Accuracy", ylim = ylim_acc)
  if (has_val) {
    lines(x$epochs, x$val_accuracy, col = "red")
    legend("bottomright", legend = c("Train", "Val"),
           col = c("blue", "red"), lty = 1, cex = 0.8)
  }

  invisible(x)
}

# ============================================================================
# Save / load full model (architecture + weights)
# ============================================================================

#' Save a Full Model (Architecture + Weights)
#'
#' Saves both the architecture and trained weights of a model to an RDS file.
#' Unlike \code{ggml_save_weights()}, which requires the model to be manually
#' reconstructed before loading, \code{ggml_save_model()} saves everything
#' needed to restore the model with a single call to \code{ggml_load_model()}.
#'
#' @section Supported model types:
#' \itemize{
#'   \item \code{ggml_sequential_model} — input shape, layer configs, trained
#'     weights, and compilation settings are all saved.
#'   \item \code{ggml_functional_model} — input/output node graphs (pure R
#'     lists, no ggml pointers) and trained \code{node_weights} are saved.
#' }
#'
#' @param model A trained \code{ggml_sequential_model} or
#'   \code{ggml_functional_model}.
#' @param path File path (typically \code{.rds}).
#' @return The model (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(16L, activation = "relu", input_shape = 4L) |>
#'   ggml_layer_dense(2L,  activation = "softmax")
#' model <- ggml_compile(model, optimizer = "adam",
#'                        loss = "categorical_crossentropy")
#' x <- matrix(runif(64 * 4), 64, 4)
#' y <- matrix(c(rep(c(1,0), 32), rep(c(0,1), 32)), 64, 2)
#' model <- ggml_fit(model, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)
#' tmp <- tempfile(fileext = ".rds")
#' ggml_save_model(model, tmp)
#' model2 <- ggml_load_model(tmp)
#' }
ggml_save_model <- function(model, path) {
  UseMethod("ggml_save_model")
}

#' @export
ggml_save_model.ggml_sequential_model <- function(model, path) {
  if (!model$compiled) stop("Model must be compiled before saving.")

  # Strip live ggml tensors; keep only serialisable config + weight values
  layers_clean <- lapply(model$layers, function(l) {
    wdata <- list()
    if (l$type %in% c("conv_1d", "conv_2d") && !is.null(l$weights$kernel)) {
      wdata$kernel <- ggml_backend_tensor_get_data(l$weights$kernel)
      wdata$bias   <- ggml_backend_tensor_get_data(l$weights$bias)
    } else if (l$type == "dense" && !is.null(l$weights$weight)) {
      wdata$weight <- ggml_backend_tensor_get_data(l$weights$weight)
      wdata$bias   <- ggml_backend_tensor_get_data(l$weights$bias)
    } else if (nn_is_norm_type(l$type) &&
               !is.null(l$weights$gamma)) {
      wdata$gamma <- ggml_backend_tensor_get_data(l$weights$gamma)
      wdata$beta  <- ggml_backend_tensor_get_data(l$weights$beta)
      # Running estimates are part of the model: without them a reloaded model
      # would normalize inference batches with the (0, 1) identity transform.
      if (!is.null(l$weights$running_mean)) {
        wdata$running_mean <- ggml_backend_tensor_get_data(l$weights$running_mean)
        wdata$running_var  <- ggml_backend_tensor_get_data(l$weights$running_var)
      }
    } else if (l$type == "lstm" && !is.null(l$weights$W_gates)) {
      wdata$W_gates <- ggml_backend_tensor_get_data(l$weights$W_gates)
      wdata$U_gates <- ggml_backend_tensor_get_data(l$weights$U_gates)
      wdata$b_gates <- ggml_backend_tensor_get_data(l$weights$b_gates)
    } else if (l$type == "gru" && !is.null(l$weights$W_zh)) {
      wdata$W_zh <- ggml_backend_tensor_get_data(l$weights$W_zh)
      wdata$U_zh <- ggml_backend_tensor_get_data(l$weights$U_zh)
      wdata$b_zh <- ggml_backend_tensor_get_data(l$weights$b_zh)
      wdata$W_n  <- ggml_backend_tensor_get_data(l$weights$W_n)
      wdata$U_n  <- ggml_backend_tensor_get_data(l$weights$U_n)
      wdata$b_n  <- ggml_backend_tensor_get_data(l$weights$b_n)
    } else if (l$type == "embedding" && !is.null(l$weights$weight)) {
      wdata$weight <- ggml_backend_tensor_get_data(l$weights$weight)
    }
    list(type        = l$type,
         name        = l$name,
         trainable   = l$trainable,
         config      = l$config,
         input_shape = l$input_shape,
         output_shape= l$output_shape,
         weights_data= wdata)
  })

  saveRDS(list(
    model_class  = "ggml_sequential_model",
    input_shape  = model$input_shape,
    layers       = layers_clean,
    compilation  = list(
      optimizer = model$compilation$optimizer,
      loss      = model$compilation$loss,
      metrics   = model$compilation$metrics,
      weight_decay = model$compilation$weight_decay,
      predict_precision = model$compilation$predict_precision
    ),
    version = NN_MODEL_FORMAT
  ), path)
  invisible(model)
}

#' @export
ggml_save_model.ggml_functional_model <- function(model, path) {
  if (!model$compiled) stop("Model must be compiled before saving.")

  # node_weights: list of node_id -> list of ggml tensors
  # Extract numeric data from each tensor
  nw_data <- NULL
  if (!is.null(model$node_weights)) {
    nw_data <- lapply(model$node_weights, function(wlist) {
      lapply(wlist, function(t) {
        if (is.null(t)) NULL else ggml_backend_tensor_get_data(t)
      })
    })
  }

  saveRDS(list(
    model_class  = "ggml_functional_model",
    inputs       = model$inputs,     # pure R ggml_tensor_node lists
    outputs      = model$outputs,
    compilation  = list(
      optimizer    = model$compilation$optimizer,
      loss         = model$compilation$loss,
      metrics      = model$compilation$metrics,
      # Per-output weights must survive the round trip: without them a reloaded
      # multi-output model would silently retrain with every head weighted 1.
      loss_weights = model$compilation$loss_weights,
      weight_decay = model$compilation$weight_decay,
      predict_precision = model$compilation$predict_precision
    ),
    node_weights_data = nw_data,
    # ggml_freeze_weights() keeps its marks here, keyed by node id, not on the
    # nodes themselves; without them a reloaded model would train every layer.
    frozen_nodes      = model$frozen_nodes,
    version = NN_MODEL_FORMAT
  ), path)
  invisible(model)
}

#' Load a Full Model (Architecture + Weights)
#'
#' Restores a model previously saved with \code{ggml_save_model()}.  The
#' returned model is compiled and ready for \code{ggml_predict()} /
#' \code{ggml_evaluate()}.  Call \code{ggml_fit()} again to continue training.
#'
#' @param path File path to an RDS file written by \code{ggml_save_model()}.
#' @param backend Backend selection: \code{"auto"}, \code{"cpu"}, or
#'   \code{"vulkan"}.
#' @return A compiled model object.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(16L, activation = "relu", input_shape = 4L) |>
#'   ggml_layer_dense(2L,  activation = "softmax")
#' model <- ggml_compile(model, optimizer = "adam",
#'                        loss = "categorical_crossentropy")
#' x <- matrix(runif(64 * 4), 64, 4)
#' y <- matrix(c(rep(c(1,0), 32), rep(c(0,1), 32)), 64, 2)
#' model <- ggml_fit(model, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)
#' tmp <- tempfile(fileext = ".rds")
#' ggml_save_model(model, tmp)
#' model2 <- ggml_load_model(tmp)
#' }
ggml_load_model <- function(path, backend = "auto") {
  data <- tryCatch(readRDS(path), error = function(e)
    stop("ggml_load_model(): cannot read '", path, "' -- the file is missing, ",
         "truncated or not an RDS file (", conditionMessage(e), ")", call. = FALSE))
  if (!is.list(data) || is.null(data$model_class)) {
    if (is.list(data) && !is.null(data$weights) && !is.null(data$architecture))
      stop("This file was saved with ggml_save_weights(). ",
           "Use ggml_load_weights() instead, or re-save with ggml_save_model().",
           call. = FALSE)
    stop("ggml_load_model(): '", path, "' is not a file written by ",
         "ggml_save_model()", call. = FALSE)
  }
  if (is.null(data$version) || data$version < 2L) {
    stop("This file was saved with ggml_save_weights(). ",
         "Use ggml_load_weights() instead, or re-save with ggml_save_model().",
         call. = FALSE)
  }
  if (data$version > NN_MODEL_FORMAT) {
    stop(sprintf(paste0("ggml_load_model(): '%s' is format %d, written by a newer ",
                        "ggmlR; this version reads up to format %d. Update ggmlR."),
                 path, as.integer(data$version), NN_MODEL_FORMAT), call. = FALSE)
  }

  if (data$model_class == "ggml_sequential_model") {
    # Reconstruct sequential model
    model <- ggml_model_sequential()
    model$input_shape <- data$input_shape

    for (ldata in data$layers) {
      layer <- list(
        type         = ldata$type,
        name         = ldata$name,
        trainable    = ldata$trainable,
        config       = ldata$config,
        input_shape  = ldata$input_shape,
        output_shape = ldata$output_shape,
        weights      = list(),
        weights_data = ldata$weights_data
      )
      model$layers <- c(model$layers, list(layer))
    }

    model <- ggml_compile(model,
      optimizer = data$compilation$optimizer,
      loss      = data$compilation$loss,
      metrics   = data$compilation$metrics,
      backend   = backend,
      # NULL in files saved before weight_decay existed: off, as it was then.
      weight_decay = data$compilation$weight_decay %||% 0,
      predict_precision = data$compilation$predict_precision %||% "default"
    )
    return(invisible(model))

  } else if (data$model_class == "ggml_functional_model") {
    model <- ggml_model(inputs = data$inputs, outputs = data$outputs)
    # NULL for files saved before frozen marks were kept: nothing frozen.
    model$frozen_nodes <- data$frozen_nodes
    model <- ggml_compile(model,
      optimizer    = data$compilation$optimizer,
      loss         = data$compilation$loss,
      metrics      = data$compilation$metrics,
      backend      = backend,
      # NULL for models saved before multi-output support, which is exactly the
      # single-head default.
      loss_weights = data$compilation$loss_weights,
      weight_decay = data$compilation$weight_decay %||% 0,
      predict_precision = data$compilation$predict_precision %||% "default"
    )

    # Ensure no stale ggml tensor pointers from the freshly-created model.
    model$node_weights <- NULL

    # Restore node_weights as R-vector lists so nn_build_functional_graph
    # picks them up via the swd (saved_weights_data) path.
    if (!is.null(data$node_weights_data)) {
      model$node_weights_data <- data$node_weights_data
    }
    return(invisible(model))
  } else {
    stop("Unknown model class in saved file: ", data$model_class)
  }
}
