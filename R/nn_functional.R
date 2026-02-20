# Functional API for ggmlR
# Allows building arbitrary DAG computation graphs (skip connections, residual
# blocks, multi-input / multi-output models) using a Keras-functional style.
#
# Key design: ggml_tensor_node objects store the **configuration** of the graph
# (like Sequential model$layers), not live ggml tensors.  Real ggml tensors are
# created lazily in nn_build_functional_graph() when compile/fit is called and
# batch_size is known.

# ============================================================================
# Counter for auto-generated node IDs and layer IDs
# ============================================================================

.fn_node_counter <- new.env(parent = emptyenv())
.fn_node_counter$n <- 0L
# Per-type counters for auto-generated layer names (input_1, add_2, ...)
.fn_type_counters <- new.env(parent = emptyenv())
# Counter for ggml_layer object IDs (shared-layer identity)
.fn_layer_counter <- new.env(parent = emptyenv())
.fn_layer_counter$n <- 0L

nn_next_node_id <- function() {
  .fn_node_counter$n <- .fn_node_counter$n + 1L
  paste0("node_", .fn_node_counter$n)
}

nn_next_layer_id <- function() {
  .fn_layer_counter$n <- .fn_layer_counter$n + 1L
  paste0("layer_", .fn_layer_counter$n)
}

# Auto-generate a layer name like "input_1", "add_2"
nn_auto_name <- function(type) {
  cur <- if (is.null(.fn_type_counters[[type]])) 0L else .fn_type_counters[[type]]
  cur <- cur + 1L
  .fn_type_counters[[type]] <- cur
  paste0(type, "_", cur)
}

# ============================================================================
# Layer object constructors (for shared-layer / ggml_apply() workflow)
# ============================================================================

#' Create a Dense Layer Object
#'
#' Returns a reusable layer object for use with \code{ggml_apply()}.
#' Applying the same object to multiple tensor nodes shares weights.
#'
#' @param units Number of output units.
#' @param activation Activation function name or NULL.
#' @param name Optional character name.
#' @param trainable Logical; whether weights are updated during training.
#' @return A \code{ggml_layer} object.
#' @export
#' @examples
#' \donttest{
#' encoder <- ggml_dense(64L, activation = "relu")
#' x1 <- ggml_input(shape = 32L)
#' x2 <- ggml_input(shape = 32L)
#' out1 <- x1 |> ggml_apply(encoder)
#' out2 <- x2 |> ggml_apply(encoder)  # shared weights
#' }
ggml_dense <- function(units, activation = NULL, name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("dense")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "dense",
      name      = name,
      config    = list(units = as.integer(units), activation = activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a Conv2D Layer Object
#'
#' @param filters Number of output filters.
#' @param kernel_size Integer or length-2 integer vector.
#' @param activation Activation function name or NULL.
#' @param strides Integer or length-2 integer vector (default 1).
#' @param padding \code{"valid"} or \code{"same"}.
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_layer_conv_2d <- function(filters, kernel_size, activation = NULL,
                          strides = c(1L, 1L), padding = "valid",
                          name = NULL, trainable = TRUE) {
  if (length(kernel_size) == 1L) kernel_size <- rep(as.integer(kernel_size), 2L)
  if (length(strides)     == 1L) strides     <- rep(as.integer(strides),     2L)
  if (is.null(name)) name <- nn_auto_name("conv_2d")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "conv_2d",
      name      = name,
      config    = list(filters     = as.integer(filters),
                       kernel_size = as.integer(kernel_size),
                       strides     = as.integer(strides),
                       padding     = padding,
                       activation  = activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a Conv1D Layer Object
#'
#' @param filters Number of output filters.
#' @param kernel_size Integer kernel size.
#' @param activation Activation function name or NULL.
#' @param strides Integer stride (default 1).
#' @param padding \code{"valid"} or \code{"same"}.
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_layer_conv_1d <- function(filters, kernel_size, activation = NULL,
                          strides = 1L, padding = "valid",
                          name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("conv_1d")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "conv_1d",
      name      = name,
      config    = list(filters     = as.integer(filters),
                       kernel_size = as.integer(kernel_size),
                       strides     = as.integer(strides),
                       padding     = padding,
                       activation  = activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a Batch Normalization Layer Object
#'
#' @param eps Small constant for numerical stability (default 1e-5).
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_batch_norm <- function(eps = 1e-5, name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("batch_norm")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "batch_norm",
      name      = name,
      config    = list(eps = eps),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create an Embedding Layer Object
#'
#' @param vocab_size Number of distinct tokens.
#' @param dim Embedding dimension.
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_embedding <- function(vocab_size, dim, name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("embedding")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "embedding",
      name      = name,
      config    = list(vocab_size = as.integer(vocab_size),
                       dim        = as.integer(dim)),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create an LSTM Layer Object
#'
#' @param units Integer, number of hidden units.
#' @param return_sequences Logical.
#' @param activation Cell gate activation (default \code{"tanh"}).
#' @param recurrent_activation Recurrent gate activation (default \code{"sigmoid"}).
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_lstm <- function(units, return_sequences = FALSE,
                       activation = "tanh", recurrent_activation = "sigmoid",
                       name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("lstm")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "lstm",
      name      = name,
      config    = list(units                = as.integer(units),
                       return_sequences     = return_sequences,
                       activation           = activation,
                       recurrent_activation = recurrent_activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

#' Create a GRU Layer Object
#'
#' @param units Integer, number of hidden units.
#' @param return_sequences Logical.
#' @param activation Candidate activation (default \code{"tanh"}).
#' @param recurrent_activation Gate activation (default \code{"sigmoid"}).
#' @param name Optional character name.
#' @param trainable Logical.
#' @return A \code{ggml_layer} object.
#' @export
ggml_gru <- function(units, return_sequences = FALSE,
                      activation = "tanh", recurrent_activation = "sigmoid",
                      name = NULL, trainable = TRUE) {
  if (is.null(name)) name <- nn_auto_name("gru")
  structure(
    list(
      layer_id  = nn_next_layer_id(),
      node_type = "gru",
      name      = name,
      config    = list(units                = as.integer(units),
                       return_sequences     = return_sequences,
                       activation           = activation,
                       recurrent_activation = recurrent_activation),
      trainable = trainable
    ),
    class = "ggml_layer"
  )
}

# ============================================================================
# ggml_apply() -- apply a ggml_layer object to a tensor node
# ============================================================================

#' Apply a Layer Object to a Tensor Node
#'
#' Applies a \code{ggml_layer} object (created with \code{ggml_dense()},
#' \code{ggml_lstm()}, etc.) to a \code{ggml_tensor_node}.  Applying the
#' \emph{same} layer object to multiple tensor nodes produces shared weights --
#' the identity of the layer object (\code{layer$layer_id}) is used as the
#' sharing key, not its name.
#'
#' @param tensor A \code{ggml_tensor_node} (e.g. from \code{ggml_input()}).
#' @param layer A \code{ggml_layer} object.
#' @return A new \code{ggml_tensor_node}.
#' @export
#' @examples
#' \donttest{
#' encoder <- ggml_dense(64L, activation = "relu")
#' x1 <- ggml_input(shape = 32L)
#' x2 <- ggml_input(shape = 32L)
#' out1 <- x1 |> ggml_apply(encoder)
#' out2 <- x2 |> ggml_apply(encoder)  # shared weights
#' model <- ggml_model(inputs = list(x1, x2),
#'                     outputs = list(out1, out2))
#' }
ggml_apply <- function(tensor, layer) {
  if (!inherits(tensor, "ggml_tensor_node")) {
    stop("'tensor' must be a ggml_tensor_node (from ggml_input() or a layer call).")
  }
  if (!inherits(layer, "ggml_layer")) {
    stop("'layer' must be a ggml_layer object (from ggml_dense(), ggml_lstm(), etc.).")
  }
  structure(
    list(
      id        = nn_next_node_id(),
      node_type = layer$node_type,
      layer_id  = layer$layer_id,   # sharing key -- identity of the layer object
      trainable = layer$trainable,
      config    = c(layer$config, list(name = layer$name)),
      parents   = list(tensor)
    ),
    class = "ggml_tensor_node"
  )
}

# ============================================================================
# ggml_input() -- declare an input tensor
# ============================================================================

#' Declare a Functional API Input Tensor
#'
#' Creates a symbolic input node for the Functional API.  The node records
#' only the \emph{shape} of one sample (without batch dimension); actual
#' memory is allocated when \code{ggml_compile()} is called.
#'
#' @param shape Integer vector describing the shape of a single sample.
#'   For flat feature vectors use a scalar, e.g. \code{shape = 64L}.
#'   For 2-D inputs (sequences) use \code{c(length, channels)}.
#'   For 3-D inputs (images) use \code{c(H, W, C)}.
#' @param name Optional character name for the input tensor.
#' @param dtype Data type of the input: \code{"float32"} (default) or
#'   \code{"int32"} (for embedding/token-index inputs).
#' @return A \code{ggml_tensor_node} object.
#' @export
#' @examples
#' \donttest{
#' x <- ggml_input(shape = 64L)
#' x <- ggml_input(shape = c(28L, 28L, 1L), name = "image")
#' x <- ggml_input(shape = 10L, dtype = "int32")  # token indices
#' }
ggml_input <- function(shape, name = NULL, dtype = "float32") {
  shape <- as.integer(shape)
  if (is.null(name)) name <- nn_auto_name("input")
  if (!dtype %in% c("float32", "int32")) {
    stop("dtype must be 'float32' or 'int32', got: ", dtype)
  }

  structure(
    list(
      id        = nn_next_node_id(),
      node_type = "input",
      config    = list(shape = shape, name = name, dtype = dtype),
      parents   = list()
    ),
    class = "ggml_tensor_node"
  )
}

# ============================================================================
# ggml_model() -- assemble a functional model from input/output nodes
# ============================================================================

#' Create a Functional Model
#'
#' Assembles a \code{ggml_functional_model} from symbolic input and output
#' nodes produced by \code{ggml_input()} and \code{ggml_layer_*()} calls.
#'
#' @param inputs A \code{ggml_tensor_node} or a list of them (model inputs).
#' @param outputs A \code{ggml_tensor_node} or a list of them (model outputs).
#' @return A \code{ggml_functional_model} object.
#' @export
#' @examples
#' \donttest{
#' x   <- ggml_input(shape = 64L)
#' out <- x |> ggml_layer_dense(10, activation = "softmax")
#' model <- ggml_model(inputs = x, outputs = out)
#' }
ggml_model <- function(inputs, outputs) {
  if (inherits(inputs, "ggml_tensor_node")) inputs <- list(inputs)
  if (inherits(outputs, "ggml_tensor_node")) outputs <- list(outputs)

  if (!is.list(inputs) || !all(vapply(inputs, inherits, logical(1), "ggml_tensor_node"))) {
    stop("'inputs' must be a ggml_tensor_node or a list of ggml_tensor_node objects.")
  }
  if (!is.list(outputs) || !all(vapply(outputs, inherits, logical(1), "ggml_tensor_node"))) {
    stop("'outputs' must be a ggml_tensor_node or a list of ggml_tensor_node objects.")
  }
  # All inputs must be declared with ggml_input() (node_type == "input")
  bad <- which(vapply(inputs, function(n) n$node_type, character(1)) != "input")
  if (length(bad) > 0L) {
    stop("'inputs[[", bad[1], "]]' has node_type '",
         inputs[[bad[1]]]$node_type,
         "' -- only nodes created with ggml_input() are valid model inputs.")
  }

  structure(
    list(
      inputs      = inputs,
      outputs     = outputs,
      compiled    = FALSE,
      compilation = list(
        sched      = NULL,
        backend    = NULL,
        optimizer  = NULL,
        loss       = NULL,
        metrics    = NULL
      )
    ),
    class = c("ggml_functional_model", "list")
  )
}

# ============================================================================
# ggml_layer_add() / ggml_layer_concatenate()
# ============================================================================

#' Element-wise Addition of Two Tensor Nodes
#'
#' Adds two (or more) tensor nodes element-wise.  All tensors must have the
#' same shape.  This is the functional equivalent of a residual / skip
#' connection.
#'
#' @param tensors A list of \code{ggml_tensor_node} objects (length >= 2).
#' @param name Optional character name for the layer.
#' @return A new \code{ggml_tensor_node} representing the sum.
#' @export
#' @examples
#' \donttest{
#' x    <- ggml_input(shape = 64L)
#' a    <- x |> ggml_layer_dense(64, activation = "relu")
#' b    <- x |> ggml_layer_dense(64)
#' out  <- ggml_layer_add(list(a, b))
#' }
ggml_layer_add <- function(tensors, name = NULL) {
  if (!is.list(tensors) || length(tensors) < 2L) {
    stop("'tensors' must be a list of at least 2 ggml_tensor_node objects.")
  }
  if (!all(vapply(tensors, inherits, logical(1), "ggml_tensor_node"))) {
    stop("All elements of 'tensors' must be ggml_tensor_node objects.")
  }
  if (is.null(name)) name <- nn_auto_name("add")

  structure(
    list(
      id        = nn_next_node_id(),
      node_type = "add",
      config    = list(name = name),
      parents   = tensors
    ),
    class = "ggml_tensor_node"
  )
}

#' Concatenate Tensor Nodes Along an Axis
#'
#' Concatenates two or more tensor nodes along the specified axis.
#'
#' @param tensors A list of \code{ggml_tensor_node} objects (length >= 2).
#' @param axis Integer axis along which to concatenate (0-based, ggml convention).
#'   Default \code{0L} concatenates along the first dimension (features for
#'   flat tensors).
#' @param name Optional character name for the layer.
#' @return A new \code{ggml_tensor_node} representing the concatenated tensor.
#' @export
#' @examples
#' \donttest{
#' x   <- ggml_input(shape = 32L)
#' y   <- ggml_input(shape = 32L)
#' out <- ggml_layer_concatenate(list(x, y), axis = 0L)
#' }
ggml_layer_concatenate <- function(tensors, axis = 0L, name = NULL) {
  if (!is.list(tensors) || length(tensors) < 2L) {
    stop("'tensors' must be a list of at least 2 ggml_tensor_node objects.")
  }
  if (!all(vapply(tensors, inherits, logical(1), "ggml_tensor_node"))) {
    stop("All elements of 'tensors' must be ggml_tensor_node objects.")
  }
  if (is.null(name)) name <- nn_auto_name("concatenate")
  # axis stored as-is (may be negative); resolved at shape inference time

  structure(
    list(
      id        = nn_next_node_id(),
      node_type = "concatenate",
      config    = list(axis = as.integer(axis), name = name),
      parents   = tensors
    ),
    class = "ggml_tensor_node"
  )
}

# ============================================================================
# Topological sort
# ============================================================================

#' Topologically sort nodes reachable from output nodes
#' @param outputs List of output ggml_tensor_node objects
#' @return Named list: nodes in topological order (inputs first, outputs last)
#' @export
nn_topo_sort <- function(outputs) {
  visited <- list()
  ordered <- list()

  visit <- function(node) {
    if (isTRUE(visited[[node$id]])) return()
    visited[[node$id]] <<- TRUE
    for (parent in node$parents) {
      visit(parent)
    }
    ordered[[length(ordered) + 1L]] <<- node
  }

  for (out in outputs) visit(out)
  ordered
}

# ============================================================================
# Build functional graph (analogous to nn_build_graph for Sequential)
# ============================================================================

#' Infer output shape of a functional node given its parent shapes
#' @keywords internal
nn_functional_output_shape <- function(node, parent_shapes) {
  switch(node$node_type,
    "input" = node$config$shape,
    "dense" = as.integer(node$config$units),
    "flatten" = {
      psh <- parent_shapes[[1]]
      as.integer(prod(psh))
    },
    "batch_norm" = parent_shapes[[1]],
    "add" = parent_shapes[[1]],
    "concatenate" = {
      ndim <- length(parent_shapes[[1]])
      axis <- node$config$axis  # 0-based, may be negative
      # Resolve negative axis (e.g. -1 -> last dimension)
      if (axis < 0L) axis <- ndim + axis
      if (axis < 0L || axis >= ndim) {
        stop("ggml_layer_concatenate: axis ", node$config$axis,
             " is out of range for tensors with ", ndim, " dimensions ",
             "(valid range: [", -ndim, ", ", ndim - 1L, "]).")
      }
      total <- 0L
      for (psh in parent_shapes) {
        total <- total + psh[axis + 1L]
      }
      out <- parent_shapes[[1]]
      out[axis + 1L] <- total
      out
    },
    "conv_2d" = {
      psh <- parent_shapes[[1]]  # c(H, W, C) R-order
      H <- psh[1]; W <- psh[2]
      kh <- node$config$kernel_size[1]; kw <- node$config$kernel_size[2]
      sh <- node$config$strides[1]; sw <- node$config$strides[2]
      if (node$config$padding == "same") {
        H_out <- ceiling(H / sh); W_out <- ceiling(W / sw)
      } else {
        H_out <- floor((H - kh) / sh) + 1L
        W_out <- floor((W - kw) / sw) + 1L
      }
      as.integer(c(H_out, W_out, node$config$filters))
    },
    "max_pooling_2d" = {
      psh <- parent_shapes[[1]]
      H <- psh[1]; W <- psh[2]; C <- psh[3]
      ph <- node$config$pool_size[1]; pw <- node$config$pool_size[2]
      sh <- node$config$strides[1]; sw <- node$config$strides[2]
      H_out <- floor((H - ph) / sh) + 1L
      W_out <- floor((W - pw) / sw) + 1L
      as.integer(c(H_out, W_out, C))
    },
    "conv_1d" = {
      psh <- parent_shapes[[1]]  # c(L, C)
      L <- psh[1]
      k <- node$config$kernel_size; s <- node$config$strides
      if (node$config$padding == "same") {
        L_out <- ceiling(L / s)
      } else {
        L_out <- floor((L - k) / s) + 1L
      }
      as.integer(c(L_out, node$config$filters))
    },
    "global_max_pooling_2d" = ,
    "global_average_pooling_2d" = {
      # [H, W, C] -> [C]
      psh <- parent_shapes[[1]]
      as.integer(psh[3])
    },
    "lstm" = {
      # input shape: c(seq_len, input_size)
      psh   <- parent_shapes[[1]]
      units <- node$config$units
      if (isTRUE(node$config$return_sequences)) {
        as.integer(c(psh[1], units))
      } else {
        as.integer(units)
      }
    },
    "gru" = {
      psh   <- parent_shapes[[1]]
      units <- node$config$units
      if (isTRUE(node$config$return_sequences)) {
        as.integer(c(psh[1], units))
      } else {
        as.integer(units)
      }
    },
    "dropout" = parent_shapes[[1]],  # shape unchanged
    "embedding" = {
      # input shape: c(seq_len) -> output: c(dim, seq_len)
      psh <- parent_shapes[[1]]
      seq_len <- if (length(psh) == 1L) psh else prod(psh)
      as.integer(c(node$config$dim, seq_len))
    },
    stop("Unknown node_type in shape inference: ", node$node_type)
  )
}

#' Build a single ggml tensor for one functional node
#' @param reuse_weights Named list of pre-allocated weight tensors to reuse
#'   (for shared layers -- second+ application of a named layer).  When not
#'   NULL the function uses these tensors instead of allocating new ones.
#' @keywords internal
nn_build_functional_node <- function(node, built_tensors, built_shapes,
                                      ctx_weights, ctx_compute, batch_size,
                                      training = FALSE,
                                      reuse_weights = NULL) {
  switch(node$node_type,

    "input" = {
      shape <- node$config$shape
      dtype <- if (!is.null(node$config$dtype)) node$config$dtype else "float32"
      ggml_type <- if (dtype == "int32") GGML_TYPE_I32 else GGML_TYPE_F32
      # Create tensor with proper dimensionality so spatial ops (conv, pool)
      # see the correct ne[0..3] fields.
      t <- if (length(shape) == 3L) {
        # Image: R [H, W, C] -> ggml [W, H, C, N]
        ggml_new_tensor_4d(ctx_weights, ggml_type,
                           shape[2L], shape[1L], shape[3L], batch_size)
      } else if (length(shape) == 2L) {
        # Sequence: R [seq_len, input_size] -> ggml [input_size, seq_len, N]
        ggml_new_tensor_3d(ctx_weights, ggml_type,
                           shape[2L], shape[1L], batch_size)
      } else {
        # Flat: R [n] -> ggml [n, N]
        ggml_new_tensor_2d(ctx_weights, ggml_type, prod(shape), batch_size)
      }
      ggml_set_name(t, node$config$name)
      ggml_set_input(t)
      list(tensor = t, weights = list())
    },

    "dense" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]
      fan_in    <- if (length(psh) == 1L) psh else prod(psh)
      units     <- node$config$units

      if (!is.null(reuse_weights)) {
        W <- reuse_weights$weight
        b <- reuse_weights$bias
      } else {
        W <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, fan_in, units)
        b <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(W, paste0(nm, "_weight"))
        ggml_set_name(b, paste0(nm, "_bias"))
      }

      out <- ggml_mul_mat(ctx_compute, W, input_t)
      out <- ggml_add(ctx_compute, out, b)
      out <- nn_apply_activation(ctx_compute, out, node$config$activation)

      list(tensor = out, weights = list(weight = W, bias = b))
    },

    "batch_norm" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]
      n_features <- if (length(psh) == 1L) psh
                    else if (length(psh) == 2L) psh[2]
                    else psh[3]

      if (!is.null(reuse_weights)) {
        gamma <- reuse_weights$gamma
        beta  <- reuse_weights$beta
      } else {
        gamma <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
        beta  <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(gamma, paste0(nm, "_gamma"))
        ggml_set_name(beta,  paste0(nm, "_beta"))
      }

      eps    <- node$config$eps
      normed <- ggml_rms_norm(ctx_compute, input_t, eps = eps)

      if (length(psh) == 3L) {
        gamma_r <- ggml_reshape_4d(ctx_compute, gamma, 1L, 1L, as.integer(psh[3]), 1L)
        beta_r  <- ggml_reshape_4d(ctx_compute, beta,  1L, 1L, as.integer(psh[3]), 1L)
      } else if (length(psh) == 2L) {
        gamma_r <- ggml_reshape_3d(ctx_compute, gamma, 1L, as.integer(psh[2]), 1L)
        beta_r  <- ggml_reshape_3d(ctx_compute, beta,  1L, as.integer(psh[2]), 1L)
      } else {
        gamma_r <- gamma
        beta_r  <- beta
      }

      out <- ggml_mul(ctx_compute, normed, gamma_r)
      out <- ggml_add(ctx_compute, out, beta_r)

      list(tensor = out, weights = list(gamma = gamma, beta = beta))
    },

    "flatten" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      psh        <- built_shapes[[parent_id]]
      n_features <- prod(psh)
      ndims      <- ggml_n_dims(input_t)
      shape      <- ggml_tensor_shape(input_t)
      bs         <- shape[ndims]
      out <- ggml_reshape_2d(ctx_compute, input_t, n_features, bs)
      list(tensor = out, weights = list())
    },

    "add" = {
      tensors <- lapply(node$parents, function(p) built_tensors[[p$id]])
      # Validate shapes match
      ref_shape <- built_shapes[[node$parents[[1]]$id]]
      for (i in seq(2L, length(node$parents))) {
        sh <- built_shapes[[node$parents[[i]]$id]]
        if (!identical(as.integer(ref_shape), as.integer(sh))) {
          stop("ggml_layer_add: shape mismatch -- input 1 has shape [",
               paste(ref_shape, collapse = ", "), "] but input ", i,
               " has shape [", paste(sh, collapse = ", "), "].")
        }
      }
      out <- tensors[[1]]
      for (i in seq(2L, length(tensors))) {
        out <- ggml_add(ctx_compute, out, tensors[[i]])
      }
      list(tensor = out, weights = list())
    },

    "concatenate" = {
      parent_tensors <- lapply(node$parents, function(p) built_tensors[[p$id]])
      # Resolve axis (negative allowed)
      ndim <- length(built_shapes[[node$parents[[1]]$id]])
      axis <- node$config$axis
      if (axis < 0L) axis <- ndim + axis
      out <- parent_tensors[[1]]
      for (i in seq(2L, length(parent_tensors))) {
        out <- ggml_concat(ctx_compute, out, parent_tensors[[i]], dim = axis)
      }
      list(tensor = out, weights = list())
    },

    "conv_2d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]  # c(H, W, C) R-order
      kh <- node$config$kernel_size[1]
      kw <- node$config$kernel_size[2]
      ic <- psh[3]
      oc <- node$config$filters

      if (!is.null(reuse_weights)) {
        kernel <- reuse_weights$kernel
        bias   <- reuse_weights$bias
      } else {
        kernel <- ggml_new_tensor_4d(ctx_weights, GGML_TYPE_F32, kw, kh, ic, oc)
        bias   <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, oc)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(kernel, paste0(nm, "_kernel"))
        ggml_set_name(bias,   paste0(nm, "_bias"))
      }

      s0 <- node$config$strides[2]; s1 <- node$config$strides[1]
      if (node$config$padding == "same") {
        p0 <- as.integer(floor(kw / 2)); p1 <- as.integer(floor(kh / 2))
      } else {
        p0 <- 0L; p1 <- 0L
      }
      out <- ggml_conv_2d(ctx_compute, kernel, input_t,
                          s0 = s0, s1 = s1, p0 = p0, p1 = p1, d0 = 1L, d1 = 1L)
      bias_4d <- ggml_reshape_4d(ctx_compute, bias, 1L, 1L, oc, 1L)
      out <- ggml_add(ctx_compute, out, bias_4d)
      out <- nn_apply_activation(ctx_compute, out, node$config$activation)

      list(tensor = out, weights = list(kernel = kernel, bias = bias))
    },

    "max_pooling_2d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      k0 <- node$config$pool_size[2]; k1 <- node$config$pool_size[1]
      s0 <- node$config$strides[2];   s1 <- node$config$strides[1]
      out <- ggml_pool_2d(ctx_compute, input_t, GGML_OP_POOL_MAX,
                          k0 = k0, k1 = k1, s0 = s0, s1 = s1, p0 = 0L, p1 = 0L)
      list(tensor = out, weights = list())
    },

    "global_max_pooling_2d" = ,
    "global_average_pooling_2d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      sh <- ggml_tensor_shape(input_t)   # [W, H, C, N] (ggml order)
      W  <- sh[1]; H <- sh[2]; C <- sh[3]; N <- sh[4]
      pool_op <- if (node$node_type == "global_max_pooling_2d") {
        GGML_OP_POOL_MAX
      } else {
        GGML_OP_POOL_AVG
      }
      pooled <- ggml_pool_2d(ctx_compute, input_t, pool_op,
                              k0 = W, k1 = H, s0 = W, s1 = H,
                              p0 = 0L, p1 = 0L)
      out <- ggml_reshape_2d(ctx_compute, pooled, C, N)
      list(tensor = out, weights = list())
    },

    "conv_1d" = {
      parent_id <- node$parents[[1]]$id
      input_t   <- built_tensors[[parent_id]]
      psh       <- built_shapes[[parent_id]]  # c(L, C)
      k  <- node$config$kernel_size
      ic <- psh[2]
      oc <- node$config$filters

      if (!is.null(reuse_weights)) {
        kernel <- reuse_weights$kernel
        bias   <- reuse_weights$bias
      } else {
        kernel <- ggml_new_tensor_3d(ctx_weights, GGML_TYPE_F32, k, ic, oc)
        bias   <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, oc)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(kernel, paste0(nm, "_kernel"))
        ggml_set_name(bias,   paste0(nm, "_bias"))
      }

      s0 <- node$config$strides
      p0 <- if (node$config$padding == "same") as.integer(floor(k / 2)) else 0L
      out <- ggml_conv_1d(ctx_compute, kernel, input_t, s0 = s0, p0 = p0, d0 = 1L)
      bias_3d <- ggml_reshape_3d(ctx_compute, bias, 1L, oc, 1L)
      out <- ggml_add(ctx_compute, out, bias_3d)
      out <- nn_apply_activation(ctx_compute, out, node$config$activation)

      list(tensor = out, weights = list(kernel = kernel, bias = bias))
    },

    "dropout" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      stochastic <- isTRUE(node$config$stochastic)
      if (!training) {
        out <- input_t  # identity at inference
        list(tensor = out, weights = list())
      } else if (stochastic) {
        # Inverted dropout: input * mask * (1 / (1 - rate))
        # mask is a F32 tensor of 0/1 values, same shape as input_t
        psh   <- built_shapes[[parent_id]]
        ne    <- prod(psh)
        mask  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, ne, batch_size)
        nm    <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(mask, paste0(nm, "_mask"))
        out <- ggml_mul(ctx_compute, input_t, mask)
        out <- ggml_scale(ctx_compute, out, 1.0 / (1.0 - node$config$rate))
        list(tensor = out, weights = list(mask = mask))
      } else {
        # Deterministic expected-value scaling
        out <- ggml_scale(ctx_compute, input_t, 1.0 - node$config$rate)
        list(tensor = out, weights = list())
      }
    },

    "embedding" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]  # I32 [seq_len, N]
      vocab_size <- node$config$vocab_size
      dim        <- node$config$dim

      # Embedding table: [dim, vocab_size]
      if (!is.null(reuse_weights)) {
        E <- reuse_weights$weight
      } else {
        E  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, dim, vocab_size)
        nm <- if (!is.null(node$config$name)) node$config$name else node$id
        ggml_set_name(E, paste0(nm, "_weight"))
      }

      # ggml_get_rows requires 1D index tensor
      # Flatten [seq_len, N] -> [seq_len * N], lookup -> [dim, seq_len*N]
      psh_in <- built_shapes[[parent_id]]
      seq_len <- if (length(psh_in) == 1L) psh_in else prod(psh_in)
      total   <- as.integer(seq_len * batch_size)
      idx_1d  <- ggml_reshape_1d(ctx_compute, input_t, total)
      flat    <- ggml_get_rows(ctx_compute, E, idx_1d)
      # Reshape to [dim, seq_len, N]
      out <- ggml_reshape_3d(ctx_compute, flat, dim, seq_len, batch_size)
      list(tensor = out, weights = list(weight = E))
    },

    "lstm" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      psh        <- built_shapes[[parent_id]]  # c(seq_len, input_size)
      seq_len    <- psh[1]; input_sz <- psh[2]
      units      <- node$config$units
      nm         <- if (!is.null(node$config$name)) node$config$name else node$id

      if (!is.null(reuse_weights)) {
        W_gates <- reuse_weights$W_gates
        U_gates <- reuse_weights$U_gates
        b_gates <- reuse_weights$b_gates
        h0      <- reuse_weights$h0
        c0      <- reuse_weights$c0
      } else {
        W_gates <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, input_sz, 4L * units)
        U_gates <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, units,    4L * units)
        b_gates <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, 4L * units)
        h0      <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        c0      <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        ggml_set_name(W_gates, paste0(nm, "_W_gates"))
        ggml_set_name(U_gates, paste0(nm, "_U_gates"))
        ggml_set_name(b_gates, paste0(nm, "_b_gates"))
        ggml_set_name(h0,      paste0(nm, "_h0"))
        ggml_set_name(c0,      paste0(nm, "_c0"))
      }

      # Build input tensor [input_sz, seq_len, N] from parent [seq_len*input_sz, N]
      # Parent shape is c(seq_len, input_size) in R order -> ggml [input_sz, seq_len, N]
      input_3d <- if (length(ggml_tensor_shape(input_t)) == 2L) {
        ggml_reshape_3d(ctx_compute, input_t, input_sz, seq_len, batch_size)
      } else {
        input_t
      }

      act_cell <- node$config$activation
      act_rec  <- node$config$recurrent_activation
      # Use properly allocated zero tensors from ctx_weights to avoid uninitialized
      # memory in the compute context (NaN * 0 = NaN under IEEE 754).
      h_shape <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, units, batch_size)
      c_shape <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, units, batch_size)
      h_t <- ggml_repeat(ctx_compute, h0, h_shape)
      c_t <- ggml_repeat(ctx_compute, c0, c_shape)
      h_steps <- vector("list", seq_len)

      for (t in seq_len(seq_len)) {
        offset_t <- as.integer((t - 1L) * input_sz * 4L)
        x_t <- ggml_view_2d(ctx_compute, input_3d, input_sz, batch_size,
                              nb1 = as.integer(input_sz * seq_len * 4L),
                              offset = offset_t)
        step <- nn_lstm_step(ctx_compute, x_t, h_t, c_t,
                              W_gates, U_gates, b_gates,
                              units, act_cell, act_rec)
        h_t <- step$h
        c_t <- step$c
        h_steps[[t]] <- h_t
      }

      if (isTRUE(node$config$return_sequences)) {
        out <- h_steps[[1]]
        for (t in seq(2L, seq_len)) out <- ggml_concat(ctx_compute, out, h_steps[[t]], dim = 1L)
      } else {
        out <- h_t
      }

      list(tensor = out,
           weights = list(W_gates = W_gates, U_gates = U_gates, b_gates = b_gates,
                          h0 = h0, c0 = c0))
    },

    "gru" = {
      parent_id  <- node$parents[[1]]$id
      input_t    <- built_tensors[[parent_id]]
      psh        <- built_shapes[[parent_id]]
      seq_len    <- psh[1]; input_sz <- psh[2]
      units      <- node$config$units
      nm         <- if (!is.null(node$config$name)) node$config$name else node$id

      if (!is.null(reuse_weights)) {
        W_zh <- reuse_weights$W_zh; U_zh <- reuse_weights$U_zh; b_zh <- reuse_weights$b_zh
        W_n  <- reuse_weights$W_n;  U_n  <- reuse_weights$U_n;  b_n  <- reuse_weights$b_n
        h0   <- reuse_weights$h0
      } else {
        W_zh <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, input_sz, 2L * units)
        U_zh <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, units,    2L * units)
        b_zh <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, 2L * units)
        W_n  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, input_sz, units)
        U_n  <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32, units,    units)
        b_n  <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        h0   <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, units)
        ggml_set_name(W_zh, paste0(nm, "_W_zh")); ggml_set_name(U_zh, paste0(nm, "_U_zh"))
        ggml_set_name(b_zh, paste0(nm, "_b_zh")); ggml_set_name(W_n,  paste0(nm, "_W_n"))
        ggml_set_name(U_n,  paste0(nm, "_U_n"));  ggml_set_name(b_n,  paste0(nm, "_b_n"))
        ggml_set_name(h0,   paste0(nm, "_h0"))
      }

      input_3d <- if (length(ggml_tensor_shape(input_t)) == 2L) {
        ggml_reshape_3d(ctx_compute, input_t, input_sz, seq_len, batch_size)
      } else {
        input_t
      }

      act_cell <- node$config$activation
      act_rec  <- node$config$recurrent_activation
      h_shape <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, units, batch_size)
      h_t <- ggml_repeat(ctx_compute, h0, h_shape)
      h_steps <- vector("list", seq_len)

      for (t in seq_len(seq_len)) {
        offset_t <- as.integer((t - 1L) * input_sz * 4L)
        x_t <- ggml_view_2d(ctx_compute, input_3d, input_sz, batch_size,
                              nb1 = as.integer(input_sz * seq_len * 4L),
                              offset = offset_t)
        step <- nn_gru_step(ctx_compute, x_t, h_t, W_zh, U_zh, b_zh,
                             W_n, U_n, b_n, units, act_cell, act_rec)
        h_t <- step$h
        h_steps[[t]] <- h_t
      }

      if (isTRUE(node$config$return_sequences)) {
        out <- h_steps[[1]]
        for (t in seq(2L, seq_len)) out <- ggml_concat(ctx_compute, out, h_steps[[t]], dim = 1L)
      } else {
        out <- h_t
      }

      list(tensor = out,
           weights = list(W_zh = W_zh, U_zh = U_zh, b_zh = b_zh,
                          W_n = W_n, U_n = U_n, b_n = b_n, h0 = h0))
    },

    stop("Unknown node_type in graph build: ", node$node_type)
  )
}

#' Build ggml computation graph for a functional model
#' @param model A ggml_functional_model
#' @param batch_size Integer batch size
#' @param training Logical; TRUE during fit (activates dropout scaling), FALSE
#'   during evaluate/predict (dropout becomes identity).
#' @return Named list with inputs, outputs, ctx_weights, ctx_compute, buffer, node_weights
#' @keywords internal
nn_build_functional_graph <- function(model, batch_size, training = FALSE) {
  backend      <- model$compilation$backend
  saved_weights <- model$node_weights  # NULL before first fit, list after
  # R-vector weights from ggml_load_model (node_id -> named list of numeric)
  saved_weights_data <- model$node_weights_data

  # Topological sort -- inputs first, outputs last
  nodes_sorted <- nn_topo_sort(model$outputs)

  # ---- Memory estimation ----
  total_elements <- 0L
  shapes <- list()  # node_id -> R-order shape

  # First pass: compute shapes
  for (node in nodes_sorted) {
    parent_shapes <- lapply(node$parents, function(p) shapes[[p$id]])
    out_shape <- nn_functional_output_shape(node, parent_shapes)
    shapes[[node$id]] <- out_shape
    total_elements <- total_elements + prod(out_shape) * batch_size
  }

  mem_size <- max((total_elements + 1000L) * 4L + length(nodes_sorted) * 2048L,
                  2L * 1024L * 1024L)

  ctx_weights <- ggml_init(mem_size, no_alloc = TRUE)

  compute_mem <- max(64L * 1024L * 1024L,
                     total_elements * 4L * 20L)
  ctx_compute <- ggml_init(compute_mem, no_alloc = TRUE)

  # ---- Second pass: build tensors ----
  built_tensors      <- list()   # node_id -> ggml tensor (external pointer)
  node_weights       <- list()   # node_id -> list of weight tensors
  # Shared-layer cache keyed by layer_id (object identity from ggml_apply()).
  # Nodes without layer_id (created via ggml_layer_*() pipe style) are never
  # shared -- they each allocate their own weights.
  shared_weight_cache <- list()  # layer_id -> weight list

  for (node in nodes_sorted) {
    layer_id     <- node$layer_id  # NULL for non-shared nodes
    is_shareable <- !is.null(layer_id) &&
                    node$node_type %in% c("dense", "batch_norm",
                                          "conv_2d", "conv_1d", "embedding",
                                          "lstm", "gru")

    reuse_w <- if (is_shareable && !is.null(shared_weight_cache[[layer_id]])) {
      shared_weight_cache[[layer_id]]
    } else {
      NULL
    }

    result <- nn_build_functional_node(
      node, built_tensors, shapes, ctx_weights, ctx_compute, batch_size,
      training = training,
      reuse_weights = reuse_w
    )
    built_tensors[[node$id]] <- result$tensor
    node_weights[[node$id]]  <- result$weights

    # Cache weights for first occurrence of a shared layer
    if (is_shareable && is.null(shared_weight_cache[[layer_id]]) &&
        length(result$weights) > 0L) {
      shared_weight_cache[[layer_id]] <- result$weights
    }
  }

  # Allocate weights on backend
  buffer <- ggml_backend_alloc_ctx_tensors(ctx_weights, backend)

  # ---- Initialize weights ----
  # Track which layer_ids have already been initialized so that secondary
  # applications of a shared layer skip init and ggml_set_param.
  initialized_layer_ids <- character(0L)

  frozen_nodes <- if (!is.null(model$frozen_nodes)) model$frozen_nodes else list()

  for (node in nodes_sorted) {
    w <- node_weights[[node$id]]
    # frozen_nodes override takes priority over node$trainable
    trainable <- if (!is.null(frozen_nodes[[node$id]])) {
      isTRUE(frozen_nodes[[node$id]])
    } else if (is.null(node$trainable)) {
      TRUE
    } else {
      isTRUE(node$trainable)
    }

    # R-vector weights from ggml_load_model -- always checked first to avoid
    # any risk of accessing freed ggml tensor pointers on a loaded model.
    swd <- if (!is.null(saved_weights_data)) saved_weights_data[[node$id]] else NULL

    # Saved ggml-tensor weights from a previous fit (keyed by node$id).
    sw <- if (is.null(swd) && !is.null(saved_weights)) saved_weights[[node$id]] else NULL

    # Skip init for secondary applications of a shared layer (by layer_id).
    layer_id     <- node$layer_id
    is_shareable <- !is.null(layer_id) &&
                    node$node_type %in% c("dense", "batch_norm",
                                          "conv_2d", "conv_1d", "embedding",
                                          "lstm", "gru")
    if (is_shareable && layer_id %in% initialized_layer_ids) {
      next  # weights already initialized and params set by primary occurrence
    }

    if (node$node_type == "dense") {
      psh    <- shapes[[node$parents[[1]]$id]]
      fan_in <- if (length(psh) == 1L) psh else prod(psh)
      fan_out <- node$config$units

      if (!is.null(sw$weight)) {
        ggml_backend_tensor_set_data(w$weight, ggml_backend_tensor_get_data(sw$weight))
        ggml_backend_tensor_set_data(w$bias,   ggml_backend_tensor_get_data(sw$bias))
      } else if (!is.null(swd$weight)) {
        ggml_backend_tensor_set_data(w$weight, swd$weight)
        ggml_backend_tensor_set_data(w$bias,   swd$bias)
      } else {
        nn_init_glorot_uniform(w$weight, fan_in, fan_out)
        nn_init_zeros(w$bias)
      }
      if (trainable) {
        ggml_set_param(w$weight)
        ggml_set_param(w$bias)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "batch_norm") {
      if (!is.null(sw$gamma)) {
        ggml_backend_tensor_set_data(w$gamma, ggml_backend_tensor_get_data(sw$gamma))
        ggml_backend_tensor_set_data(w$beta,  ggml_backend_tensor_get_data(sw$beta))
      } else if (!is.null(swd$gamma)) {
        ggml_backend_tensor_set_data(w$gamma, swd$gamma)
        ggml_backend_tensor_set_data(w$beta,  swd$beta)
      } else {
        n <- ggml_nelements(w$gamma)
        ggml_backend_tensor_set_data(w$gamma, rep(1.0, n))
        nn_init_zeros(w$beta)
      }
      if (trainable) {
        ggml_set_param(w$gamma)
        ggml_set_param(w$beta)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "conv_2d") {
      psh    <- shapes[[node$parents[[1]]$id]]
      kh <- node$config$kernel_size[1]; kw <- node$config$kernel_size[2]
      fan_in <- kh * kw * psh[3]

      if (!is.null(sw$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, ggml_backend_tensor_get_data(sw$kernel))
        ggml_backend_tensor_set_data(w$bias,   ggml_backend_tensor_get_data(sw$bias))
      } else if (!is.null(swd$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, swd$kernel)
        ggml_backend_tensor_set_data(w$bias,   swd$bias)
      } else {
        nn_init_he_uniform(w$kernel, fan_in)
        nn_init_zeros(w$bias)
      }
      if (trainable) {
        ggml_set_param(w$kernel)
        ggml_set_param(w$bias)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "conv_1d") {
      psh    <- shapes[[node$parents[[1]]$id]]
      fan_in <- node$config$kernel_size * psh[2]

      if (!is.null(sw$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, ggml_backend_tensor_get_data(sw$kernel))
        ggml_backend_tensor_set_data(w$bias,   ggml_backend_tensor_get_data(sw$bias))
      } else if (!is.null(swd$kernel)) {
        ggml_backend_tensor_set_data(w$kernel, swd$kernel)
        ggml_backend_tensor_set_data(w$bias,   swd$bias)
      } else {
        nn_init_he_uniform(w$kernel, fan_in)
        nn_init_zeros(w$bias)
      }
      if (trainable) {
        ggml_set_param(w$kernel)
        ggml_set_param(w$bias)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "embedding") {
      if (!is.null(sw$weight)) {
        ggml_backend_tensor_set_data(w$weight, ggml_backend_tensor_get_data(sw$weight))
      } else if (!is.null(swd$weight)) {
        ggml_backend_tensor_set_data(w$weight, swd$weight)
      } else {
        n <- ggml_nelements(w$weight)
        ggml_backend_tensor_set_data(w$weight, runif(n, -0.05, 0.05))
      }
      if (trainable) {
        ggml_set_param(w$weight)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "lstm") {
      psh      <- shapes[[node$parents[[1]]$id]]
      input_sz <- psh[2]; units <- node$config$units
      if (!is.null(sw$W_gates)) {
        ggml_backend_tensor_set_data(w$W_gates, ggml_backend_tensor_get_data(sw$W_gates))
        ggml_backend_tensor_set_data(w$U_gates, ggml_backend_tensor_get_data(sw$U_gates))
        ggml_backend_tensor_set_data(w$b_gates, ggml_backend_tensor_get_data(sw$b_gates))
      } else if (!is.null(swd$W_gates)) {
        ggml_backend_tensor_set_data(w$W_gates, swd$W_gates)
        ggml_backend_tensor_set_data(w$U_gates, swd$U_gates)
        ggml_backend_tensor_set_data(w$b_gates, swd$b_gates)
      } else {
        nn_init_recurrent_uniform(w$W_gates)
        nn_init_recurrent_uniform(w$U_gates)
        nn_init_zeros(w$b_gates)
      }
      nn_init_zeros(w$h0)
      nn_init_zeros(w$c0)
      if (trainable) {
        ggml_set_param(w$W_gates); ggml_set_param(w$U_gates); ggml_set_param(w$b_gates)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "gru") {
      psh      <- shapes[[node$parents[[1]]$id]]
      input_sz <- psh[2]; units <- node$config$units
      if (!is.null(sw$W_zh)) {
        ggml_backend_tensor_set_data(w$W_zh, ggml_backend_tensor_get_data(sw$W_zh))
        ggml_backend_tensor_set_data(w$U_zh, ggml_backend_tensor_get_data(sw$U_zh))
        ggml_backend_tensor_set_data(w$b_zh, ggml_backend_tensor_get_data(sw$b_zh))
        ggml_backend_tensor_set_data(w$W_n,  ggml_backend_tensor_get_data(sw$W_n))
        ggml_backend_tensor_set_data(w$U_n,  ggml_backend_tensor_get_data(sw$U_n))
        ggml_backend_tensor_set_data(w$b_n,  ggml_backend_tensor_get_data(sw$b_n))
      } else if (!is.null(swd$W_zh)) {
        ggml_backend_tensor_set_data(w$W_zh, swd$W_zh); ggml_backend_tensor_set_data(w$U_zh, swd$U_zh)
        ggml_backend_tensor_set_data(w$b_zh, swd$b_zh); ggml_backend_tensor_set_data(w$W_n,  swd$W_n)
        ggml_backend_tensor_set_data(w$U_n,  swd$U_n);  ggml_backend_tensor_set_data(w$b_n,  swd$b_n)
      } else {
        nn_init_recurrent_uniform(w$W_zh)
        nn_init_recurrent_uniform(w$U_zh)
        nn_init_zeros(w$b_zh)
        nn_init_recurrent_uniform(w$W_n)
        nn_init_recurrent_uniform(w$U_n)
        nn_init_zeros(w$b_n)
      }
      nn_init_zeros(w$h0)
      if (trainable) {
        ggml_set_param(w$W_zh); ggml_set_param(w$U_zh); ggml_set_param(w$b_zh)
        ggml_set_param(w$W_n);  ggml_set_param(w$U_n);  ggml_set_param(w$b_n)
      }
      if (is_shareable) initialized_layer_ids <- c(initialized_layer_ids, layer_id)

    } else if (node$node_type == "dropout" && !is.null(w$mask)) {
      # Stochastic dropout: initialize mask to all-ones (identity until first epoch update)
      n <- ggml_nelements(w$mask)
      ggml_backend_tensor_set_data(w$mask, rep(1.0, n))
      # mask is NOT a param -- not trained, updated externally each epoch
    }
    # input / flatten / add / concatenate / max_pooling_2d / det.dropout have no weights
  }

  # Collect input/output ggml tensors (always lists)
  input_tensors  <- lapply(model$inputs,  function(n) built_tensors[[n$id]])
  output_tensors <- lapply(model$outputs, function(n) built_tensors[[n$id]])

  # Mark outputs
  for (t in output_tensors) ggml_set_output(t)

  # Collect stochastic dropout masks (node_id -> mask tensor)
  dropout_masks <- list()
  for (node in nodes_sorted) {
    if (node$node_type == "dropout" && isTRUE(node$config$stochastic)) {
      w <- node_weights[[node$id]]
      if (!is.null(w$mask)) {
        dropout_masks[[node$id]] <- list(
          mask = w$mask,
          rate = node$config$rate,
          ne   = ggml_nelements(w$mask)
        )
      }
    }
  }

  list(
    ctx_weights   = ctx_weights,
    ctx_compute   = ctx_compute,
    inputs        = input_tensors,
    outputs       = output_tensors,
    buffer        = buffer,
    node_weights  = node_weights,
    built_tensors = built_tensors,
    shapes        = shapes,
    dropout_masks = dropout_masks
  )
}

# ============================================================================
# Compile -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_compile
#' @export
ggml_compile.ggml_functional_model <- function(model,
                                                optimizer = "adam",
                                                loss = "categorical_crossentropy",
                                                metrics = c("accuracy"),
                                                backend = "auto") {
  # Backend selection (same logic as Sequential)
  use_vulkan <- FALSE
  if (backend == "auto") {
    if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0) use_vulkan <- TRUE
  } else if (backend == "vulkan") {
    if (!ggml_vulkan_available() || ggml_vulkan_device_count() == 0) {
      stop("Vulkan backend requested but not available.")
    }
    use_vulkan <- TRUE
  } else if (backend != "cpu") {
    stop("Unknown backend: '", backend, "'. Use 'auto', 'cpu', or 'vulkan'.")
  }

  if (use_vulkan) {
    gpu_backend <- ggml_vulkan_init(0L)
    sched       <- ggml_backend_sched_new(list(gpu_backend), parallel = FALSE)
    cpu_backend <- ggml_backend_cpu_init()
    if (!isTRUE(.ggmlr_state$backend_msg_shown)) {
      message("Using Vulkan GPU backend: ", ggml_vulkan_device_description(0L))
      .ggmlr_state$backend_msg_shown <- TRUE
    }
  } else {
    cpu_backend <- ggml_backend_cpu_init()
    sched       <- ggml_backend_sched_new(list(cpu_backend), parallel = FALSE)
    if (!isTRUE(.ggmlr_state$backend_msg_shown)) {
      message("Using CPU backend")
      .ggmlr_state$backend_msg_shown <- TRUE
    }
  }

  if (use_vulkan) {
    model$compilation$backend     <- gpu_backend
    model$compilation$cpu_backend <- cpu_backend
  } else {
    model$compilation$backend <- cpu_backend
  }

  model$compilation$sched     <- sched
  model$compilation$optimizer <- optimizer
  model$compilation$loss      <- loss
  model$compilation$metrics   <- metrics
  model$compiled              <- TRUE

  invisible(model)
}

# ============================================================================
# Multi-input helpers
# ============================================================================

# Normalise x for multi-input models.
# Returns a list: list(x_ggml, ne_per_input, is_multi)
#   x_ggml       : numeric vector, column-major, [ne_total * N] for dataset
#   ne_per_input : integer vector, one element per input node
#   is_multi     : TRUE when model has >1 inputs
#
# For single-input models x may be matrix [N, ne] or array [N, ...].
# For multi-input models x must be list(x1, x2, ...) where each xi is
# a matrix [N, ne_i].  All xi must have the same nrow.
nn_prepare_x <- function(model, x) {
  n_inputs <- length(model$inputs)
  if (n_inputs == 1L) {
    shape <- model$inputs[[1L]]$config$shape
    ne    <- prod(shape)
    dtype <- if (!is.null(model$inputs[[1L]]$config$dtype)) model$inputs[[1L]]$config$dtype else "float32"
    if (dtype == "int32") {
      x_ggml <- as.integer(t(x))
    } else {
      x_ggml <- if (length(shape) == 3L) as.vector(aperm(x, c(3L, 2L, 4L, 1L)))
                else if (length(shape) == 2L) as.vector(aperm(x, c(3L, 2L, 1L)))
                else as.vector(t(x))
    }
    return(list(x_ggml = x_ggml, ne_per_input = as.integer(ne), is_multi = FALSE))
  }

  # Multi-input
  if (!is.list(x) || is.data.frame(x))
    stop("For multi-input models x must be a list: list(x1, x2, ...)")
  if (length(x) != n_inputs)
    stop("x has ", length(x), " elements but model has ", n_inputs, " inputs.")

  ne_per_input <- vapply(model$inputs, function(inp) as.integer(prod(inp$config$shape)), integer(1))
  # Each xi must be a matrix [N, ne_i].  t(xi) is [ne_i, N] (column-major),
  # which matches the layout ggml_backend_tensor_set_data expects for a
  # [ne_i, batch] tensor.
  N <- nrow(as.matrix(x[[1]]))
  # cbind of transposed mats -> [ne_total, N], then as.vector = column-major
  x_ggml <- as.numeric(do.call(rbind, lapply(seq_len(n_inputs), function(i) {
    xi  <- x[[i]]
    ne_i <- ne_per_input[i]
    xi_mat <- matrix(as.numeric(xi), nrow = N, ncol = ne_i)
    t(xi_mat)   # [ne_i, N]
  })))   # result: [ne_total, N] column-major = ne_total * N values
  list(x_ggml = x_ggml, ne_per_input = ne_per_input, is_multi = TRUE)
}

# Fill each ggml input tensor for one batch from the full interleaved vector.
# x_ggml: full flat vector [ne_total * N] in interleaved sample layout (see nn_prepare_x)
# ne_per_input: elements per sample for each input
# input_tensors: list of ggml tensor pointers (one per input)
# batch_size: number of samples in this batch
# samp_start: 0-based index of first sample in this batch
nn_fill_inputs <- function(x_ggml, ne_per_input, input_tensors, batch_size, samp_start) {
  ne_total <- sum(ne_per_input)
  for (i in seq_along(input_tensors)) {
    ne_i <- ne_per_input[i]
    # offsets of this input's block within each sample's interleaved row
    inp_offset <- sum(ne_per_input[seq_len(i - 1L)])
    # collect ne_i values for each of the batch_size samples
    chunk <- unlist(lapply(seq_len(batch_size) - 1L, function(s) {
      base <- (samp_start + s) * ne_total + inp_offset
      x_ggml[(base + 1L):(base + ne_i)]
    }), use.names = FALSE)
    ggml_backend_tensor_set_data(input_tensors[[i]], chunk)
  }
}

# ============================================================================
# Fit -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_fit
#' @param model A compiled model object.
#' @param x Training data (matrix or array).
#' @param y Training labels (matrix, one-hot encoded).
#' @param epochs Number of training epochs (default: 1).
#' @param batch_size Batch size (default: 32).
#' @param validation_split Fraction of data for validation (default: 0).
#' @param validation_data Optional list(x_val, y_val). Overrides validation_split.
#' @param verbose 0 = silent, 1 = progress (default: 1).
#' @param ... Additional arguments (ignored).
#' @export
ggml_fit.ggml_functional_model <- function(model, x, y,
                                            epochs = 1L,
                                            batch_size = 32L,
                                            validation_split = 0.0,
                                            validation_data = NULL,
                                            verbose = 1L,
                                            ...) {
  if (!model$compiled) {
    stop("Model must be compiled before training. Call ggml_compile() first.")
  }

  # Prepare input data (handles both single and multi-input)
  xp <- nn_prepare_x(model, x)
  is_multi      <- xp$is_multi
  x_ggml        <- xp$x_ggml
  ne_per_input  <- xp$ne_per_input
  ne_datapoint  <- sum(ne_per_input)   # total elements per sample across all inputs

  # Handle validation_data
  if (!is.null(validation_data)) {
    if (!is.list(validation_data) || length(validation_data) < 2L) {
      stop("validation_data must be a list: list(x_val, y_val)")
    }
    x_val <- validation_data[[1]]
    y_val <- validation_data[[2]]
    xp_val <- nn_prepare_x(model, x_val)
    n_val   <- length(xp_val$x_ggml) %/% ne_datapoint
    n_train <- length(x_ggml)         %/% ne_datapoint
    x_ggml  <- c(x_ggml, xp_val$x_ggml)
    y       <- rbind(y, y_val)
    validation_split <- n_val / (n_train + n_val)
  }

  n_samples <- length(x_ggml) %/% ne_datapoint
  ne_label  <- ncol(y)

  # Truncate to batch boundary
  usable <- (n_samples %/% batch_size) * batch_size
  if (usable < n_samples) {
    message("Truncating data from ", n_samples, " to ", usable,
            " samples (batch_size=", batch_size, " must divide evenly)")
    keep_idx <- seq_len(usable * ne_datapoint)
    x_ggml   <- x_ggml[keep_idx]
    y        <- y[seq_len(usable), , drop = FALSE]
    n_samples <- usable
  }

  y_ggml <- as.vector(t(y))

  # Determine input dtype (first input only; multi-input always float32 for now)
  input_dtype <- if (!is.null(model$inputs[[1L]]$config$dtype)) {
    model$inputs[[1L]]$config$dtype
  } else {
    "float32"
  }
  if (is_multi) input_dtype <- "float32"

  optimizer_type <- switch(model$compilation$optimizer,
    "adam" = , "adamw" = ggml_opt_optimizer_type_adamw(),
    "sgd"  = ggml_opt_optimizer_type_sgd(),
    ggml_opt_optimizer_type_adamw()
  )
  loss_type <- switch(model$compilation$loss,
    "categorical_crossentropy" = , "crossentropy" = ggml_opt_loss_type_cross_entropy(),
    "mse" = , "mean_squared_error" = ggml_opt_loss_type_mse(),
    ggml_opt_loss_type_cross_entropy()
  )

  train_loss_vec <- numeric(epochs)
  train_acc_vec  <- numeric(epochs)
  val_loss_vec   <- numeric(epochs)
  val_acc_vec    <- numeric(epochs)

  if (!is_multi) {
    # -----------------------------------------------------------------------
    # Single-input path — use dataset + ggml_opt_fit / ggml_opt_epoch
    # -----------------------------------------------------------------------
    data_type <- if (input_dtype == "int32") GGML_TYPE_I32 else GGML_TYPE_F32
    dataset <- ggml_opt_dataset_init(
      type_data    = data_type,
      type_label   = GGML_TYPE_F32,
      ne_datapoint = ne_datapoint,
      ne_label     = ne_label,
      ndata        = n_samples,
      ndata_shard  = 1L
    )
    ggml_backend_tensor_set_data(ggml_opt_dataset_data(dataset),   x_ggml)
    ggml_backend_tensor_set_data(ggml_opt_dataset_labels(dataset), y_ggml)

    graph_info <- nn_build_functional_graph(model, batch_size, training = TRUE)
    fit_input  <- graph_info$inputs[[1L]]
    fit_output <- graph_info$outputs[[length(graph_info$outputs)]]

    has_stochastic_dropout <- length(graph_info$dropout_masks) > 0L

    if (!has_stochastic_dropout) {
      history_raw <- ggml_opt_fit(
        sched          = model$compilation$sched,
        ctx_compute    = graph_info$ctx_compute,
        inputs         = fit_input,
        outputs        = fit_output,
        dataset        = dataset,
        loss_type      = loss_type,
        optimizer      = optimizer_type,
        nepoch         = epochs,
        nbatch_logical = batch_size,
        val_split      = validation_split,
        silent         = (verbose == 0L)
      )
      train_loss_vec <- history_raw$train_loss
      train_acc_vec  <- history_raw$train_accuracy
      val_loss_vec   <- history_raw$val_loss
      val_acc_vec    <- history_raw$val_accuracy

    } else {
      n_batches_log <- n_samples %/% batch_size
      idata_split   <- as.integer((1.0 - validation_split) * n_batches_log) * batch_size

      init_info <- ggml_opt_init_for_fit(
        sched       = model$compilation$sched,
        loss_type   = loss_type,
        optimizer   = optimizer_type,
        opt_period  = 1L,
        ctx_compute = graph_info$ctx_compute,
        inputs      = fit_input,
        outputs     = fit_output
      )
      opt_ctx <- init_info$opt_ctx

      result_train <- ggml_opt_result_init()
      result_val   <- ggml_opt_result_init()

      for (ep in seq_len(epochs)) {
        for (dm in graph_info$dropout_masks) {
          keep_prob <- 1.0 - dm$rate
          mask_vals <- as.numeric(runif(dm$ne) < keep_prob)
          ggml_backend_tensor_set_data(dm$mask, mask_vals)
        }
        if (verbose > 0L) cat(sprintf("Epoch %d/%d:\n", ep, epochs))
        ggml_opt_result_reset(result_train)
        ggml_opt_result_reset(result_val)
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val,
                       idata_split = idata_split,
                       callback_train = (verbose > 0L),
                       callback_eval  = (verbose > 0L))
        tl <- ggml_opt_result_loss(result_train)
        ta <- ggml_opt_result_accuracy(result_train)
        vl <- ggml_opt_result_loss(result_val)
        va <- ggml_opt_result_accuracy(result_val)
        train_loss_vec[ep] <- tl[["loss"]]
        train_acc_vec[ep]  <- ta[["accuracy"]]
        val_loss_vec[ep]   <- if (validation_split > 0) vl[["loss"]] else NA_real_
        val_acc_vec[ep]    <- if (validation_split > 0) va[["accuracy"]] else NA_real_
      }

      ggml_opt_result_free(result_train)
      ggml_opt_result_free(result_val)
      ggml_opt_free(opt_ctx)
    }

    model$node_weights            <- graph_info$node_weights
    model$compilation$ctx_weights <- graph_info$ctx_weights
    model$compilation$buffer      <- graph_info$buffer
    ggml_free(graph_info$ctx_compute)
    ggml_opt_dataset_free(dataset)

  } else {
    # -----------------------------------------------------------------------
    # Multi-input path — manual batch loop filling each input tensor
    # -----------------------------------------------------------------------
    # Split into train / val portions (no shuffle for simplicity)
    n_train_samples <- as.integer(floor((1.0 - validation_split) * n_samples %/% batch_size) * batch_size)
    if (n_train_samples == 0L) n_train_samples <- n_samples

    graph_info <- nn_build_functional_graph(model, batch_size, training = TRUE)
    fit_output <- graph_info$outputs[[length(graph_info$outputs)]]

    init_info <- ggml_opt_init_for_fit(
      sched       = model$compilation$sched,
      loss_type   = loss_type,
      optimizer   = optimizer_type,
      opt_period  = 1L,
      ctx_compute = graph_info$ctx_compute,
      inputs      = graph_info$inputs[[1L]],
      outputs     = fit_output
    )
    opt_ctx      <- init_info$opt_ctx
    labels_tensor <- ggml_opt_labels(opt_ctx)

    result_train <- ggml_opt_result_init()
    result_val   <- ggml_opt_result_init()

    n_batches_train <- n_train_samples %/% batch_size
    n_batches_val   <- (n_samples - n_train_samples) %/% batch_size

    for (ep in seq_len(epochs)) {
      # Regenerate dropout masks
      for (dm in graph_info$dropout_masks) {
        keep_prob <- 1.0 - dm$rate
        mask_vals <- as.numeric(runif(dm$ne) < keep_prob)
        ggml_backend_tensor_set_data(dm$mask, mask_vals)
      }

      if (verbose > 0L) cat(sprintf("Epoch %d/%d:\n", ep, epochs))

      ggml_opt_result_reset(result_train)
      ggml_opt_result_reset(result_val)

      # Training batches
      for (ib in seq_len(n_batches_train)) {
        samp_start <- (ib - 1L) * batch_size
        nn_fill_inputs(x_ggml, ne_per_input, graph_info$inputs, batch_size, samp_start)

        # Fill labels for this batch
        lab_start <- samp_start * ne_label + 1L
        lab_end   <- lab_start + batch_size * ne_label - 1L
        ggml_backend_tensor_set_data(labels_tensor, y_ggml[lab_start:lab_end])

        ggml_opt_alloc(opt_ctx, backward = TRUE)
        ggml_opt_eval(opt_ctx, result_train)
      }

      # Validation batches (forward only)
      if (n_batches_val > 0L) {
        for (ib in seq_len(n_batches_val)) {
          samp_start <- n_train_samples + (ib - 1L) * batch_size
          nn_fill_inputs(x_ggml, ne_per_input, graph_info$inputs, batch_size, samp_start)

          lab_start <- samp_start * ne_label + 1L
          lab_end   <- lab_start + batch_size * ne_label - 1L
          ggml_backend_tensor_set_data(labels_tensor, y_ggml[lab_start:lab_end])

          ggml_opt_alloc(opt_ctx, backward = FALSE)
          ggml_opt_eval(opt_ctx, result_val)
        }
      }

      tl <- ggml_opt_result_loss(result_train)
      ta <- ggml_opt_result_accuracy(result_train)
      vl <- ggml_opt_result_loss(result_val)
      va <- ggml_opt_result_accuracy(result_val)

      train_loss_vec[ep] <- tl[["loss"]]
      train_acc_vec[ep]  <- ta[["accuracy"]]
      val_loss_vec[ep]   <- if (validation_split > 0 && n_batches_val > 0L) vl[["loss"]]     else NA_real_
      val_acc_vec[ep]    <- if (validation_split > 0 && n_batches_val > 0L) va[["accuracy"]] else NA_real_

      if (verbose > 0L) {
        cat(sprintf("  train_loss=%.4f  train_acc=%.4f",
                    train_loss_vec[ep], train_acc_vec[ep]))
        if (!is.na(val_loss_vec[ep]))
          cat(sprintf("  val_loss=%.4f  val_acc=%.4f",
                      val_loss_vec[ep], val_acc_vec[ep]))
        cat("\n")
      }
    }

    ggml_opt_result_free(result_train)
    ggml_opt_result_free(result_val)
    ggml_opt_free(opt_ctx)

    model$node_weights            <- graph_info$node_weights
    model$compilation$ctx_weights <- graph_info$ctx_weights
    model$compilation$buffer      <- graph_info$buffer
    ggml_free(graph_info$ctx_compute)
  }

  model$history <- structure(
    list(
      train_loss     = train_loss_vec,
      train_accuracy = train_acc_vec,
      val_loss       = val_loss_vec,
      val_accuracy   = val_acc_vec,
      epochs         = seq_len(epochs)
    ),
    class = "ggml_history"
  )

  invisible(model)
}

# ============================================================================
# Evaluate -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_evaluate
#' @param ... Additional arguments (ignored).
#' @export
ggml_evaluate.ggml_functional_model <- function(model, x, y,
                                                  batch_size = 32L, ...) {
  if (!model$compiled) stop("Model must be compiled before evaluation.")

  xp           <- nn_prepare_x(model, x)
  is_multi     <- xp$is_multi
  x_ggml       <- xp$x_ggml
  ne_per_input <- xp$ne_per_input
  ne_datapoint <- sum(ne_per_input)
  ne_label     <- ncol(y)

  n_samples <- length(x_ggml) %/% ne_datapoint

  usable <- (n_samples %/% batch_size) * batch_size
  if (usable < n_samples) {
    x_ggml    <- x_ggml[seq_len(usable * ne_datapoint)]
    y         <- y[seq_len(usable), , drop = FALSE]
    n_samples <- usable
  }

  y_ggml <- as.vector(t(y))

  loss_type <- switch(model$compilation$loss,
    "categorical_crossentropy" = , "crossentropy" = ggml_opt_loss_type_cross_entropy(),
    "mse" = , "mean_squared_error" = ggml_opt_loss_type_mse(),
    ggml_opt_loss_type_cross_entropy()
  )

  graph_info  <- nn_build_functional_graph(model, batch_size, training = FALSE)
  eval_output <- graph_info$outputs[[length(graph_info$outputs)]]

  result_eval <- ggml_opt_result_init()

  if (!is_multi) {
    # ------------------------------------------------------------------
    # Single-input: use dataset + ggml_opt_epoch
    # ------------------------------------------------------------------
    input_dtype <- if (!is.null(model$inputs[[1L]]$config$dtype))
      model$inputs[[1L]]$config$dtype else "float32"
    data_type <- if (input_dtype == "int32") GGML_TYPE_I32 else GGML_TYPE_F32

    dataset <- ggml_opt_dataset_init(
      type_data    = data_type,
      type_label   = GGML_TYPE_F32,
      ne_datapoint = ne_datapoint,
      ne_label     = ne_label,
      ndata        = n_samples,
      ndata_shard  = 1L
    )
    ggml_backend_tensor_set_data(ggml_opt_dataset_data(dataset),   x_ggml)
    ggml_backend_tensor_set_data(ggml_opt_dataset_labels(dataset), y_ggml)

    opt_ctx <- ggml_opt_init(
      sched       = model$compilation$sched,
      loss_type   = loss_type,
      optimizer   = ggml_opt_optimizer_type_adamw(),
      ctx_compute = graph_info$ctx_compute,
      inputs      = graph_info$inputs[[1L]],
      outputs     = eval_output
    )

    ggml_opt_epoch(opt_ctx, dataset, NULL, result_eval,
                   idata_split = 0L, callback_train = FALSE, callback_eval = FALSE)

    ggml_opt_free(opt_ctx)
    ggml_opt_dataset_free(dataset)

  } else {
    # ------------------------------------------------------------------
    # Multi-input: manual batch loop
    # ------------------------------------------------------------------
    init_info <- ggml_opt_init_for_fit(
      sched       = model$compilation$sched,
      loss_type   = loss_type,
      optimizer   = ggml_opt_optimizer_type_adamw(),
      opt_period  = 1L,
      ctx_compute = graph_info$ctx_compute,
      inputs      = graph_info$inputs[[1L]],
      outputs     = eval_output
    )
    opt_ctx       <- init_info$opt_ctx
    labels_tensor <- ggml_opt_labels(opt_ctx)
    n_batches     <- n_samples %/% batch_size

    for (ib in seq_len(n_batches)) {
      samp_start <- (ib - 1L) * batch_size
      nn_fill_inputs(x_ggml, ne_per_input, graph_info$inputs, batch_size, samp_start)

      lab_start <- samp_start * ne_label + 1L
      lab_end   <- lab_start + batch_size * ne_label - 1L
      ggml_backend_tensor_set_data(labels_tensor, y_ggml[lab_start:lab_end])

      ggml_opt_alloc(opt_ctx, backward = FALSE)
      ggml_opt_eval(opt_ctx, result_eval)
    }

    ggml_opt_free(opt_ctx)
  }

  loss_val <- ggml_opt_result_loss(result_eval)
  acc_val  <- ggml_opt_result_accuracy(result_eval)
  ggml_opt_result_free(result_eval)
  ggml_free(graph_info$ctx_compute)
  ggml_backend_buffer_free(graph_info$buffer)
  ggml_free(graph_info$ctx_weights)

  out <- list(loss = loss_val[["loss"]], accuracy = acc_val[["accuracy"]])

  # Дополнительные метрики через predict
  extra_metrics <- setdiff(model$compilation$metrics, c("accuracy", "acc"))
  if (length(extra_metrics) > 0L) {
    preds <- ggml_predict(model, x, batch_size = batch_size)
    preds_mat <- if (is.matrix(preds)) preds else preds[[1L]]
    n_cmp <- min(nrow(preds_mat), nrow(y))
    y_cmp <- y[seq_len(n_cmp), , drop = FALSE]
    p_cmp <- preds_mat[seq_len(n_cmp), , drop = FALSE]
    for (m in extra_metrics) {
      out[[m]] <- switch(m,
        "mae"  = , "mean_absolute_error" = mean(abs(y_cmp - p_cmp)),
        "mse"  = , "mean_squared_error"  = mean((y_cmp - p_cmp)^2),
        "rmse" = sqrt(mean((y_cmp - p_cmp)^2)),
        NULL
      )
    }
  }
  out
}

# ============================================================================
# Predict -- S3 method for ggml_functional_model
# ============================================================================

#' @rdname ggml_predict
#' @param ... Additional arguments (ignored).
#' @export
ggml_predict.ggml_functional_model <- function(model, x, batch_size = 32L, ...) {
  if (!model$compiled) stop("Model must be compiled before prediction.")

  xp           <- nn_prepare_x(model, x)
  is_multi     <- xp$is_multi
  ne_per_input <- xp$ne_per_input
  ne_datapoint <- sum(ne_per_input)

  # Determine n_samples_orig before possible padding
  n_samples_orig <- if (is_multi) {
    nrow(as.matrix(x[[1L]]))
  } else if (is.matrix(x)) {
    nrow(x)
  } else {
    dim(x)[1L]
  }

  if (n_samples_orig < batch_size) {
    stop("Not enough samples (", n_samples_orig, ") for batch_size=", batch_size)
  }

  # Pad to batch boundary
  remainder <- n_samples_orig %% batch_size
  if (remainder != 0L) {
    n_pad <- batch_size - remainder
    if (is_multi) {
      x <- lapply(x, function(xi) {
        xi_mat <- matrix(as.numeric(xi), nrow = n_samples_orig)
        rbind(xi_mat, matrix(0.0, nrow = n_pad, ncol = ncol(xi_mat)))
      })
    } else if (is.matrix(x)) {
      x <- rbind(x, matrix(0.0, nrow = n_pad, ncol = ncol(x)))
    } else {
      pad_dims <- dim(x); pad_dims[1L] <- n_pad
      x <- abind_first(x, array(0.0, dim = pad_dims))
    }
    xp <- nn_prepare_x(model, x)
  }

  x_ggml    <- xp$x_ggml
  n_samples <- length(x_ggml) %/% ne_datapoint
  n_batches <- n_samples %/% batch_size

  graph_info <- nn_build_functional_graph(model, batch_size, training = FALSE)
  n_outputs  <- length(graph_info$outputs)
  sched      <- model$compilation$sched

  # Build forward graph covering all outputs
  graph <- ggml_build_forward_expand(graph_info$ctx_compute,
                                     graph_info$outputs[[n_outputs]])

  out_shapes     <- lapply(model$outputs, function(o) graph_info$shapes[[o$id]])
  ne_outputs_vec <- vapply(out_shapes, prod, numeric(1))
  all_preds_list <- lapply(ne_outputs_vec, function(ne) {
    matrix(0.0, nrow = n_samples, ncol = ne)
  })

  for (ib in seq_len(n_batches)) {
    samp_start <- (ib - 1L) * batch_size

    if (is_multi) {
      nn_fill_inputs(x_ggml, ne_per_input, graph_info$inputs, batch_size, samp_start)
    } else {
      data_start <- samp_start * ne_datapoint + 1L
      data_end   <- data_start + batch_size * ne_datapoint - 1L
      ggml_backend_tensor_set_data(graph_info$inputs[[1L]], x_ggml[data_start:data_end])
    }

    ggml_backend_sched_reset(sched)
    ggml_backend_sched_alloc_graph(sched, graph)
    ggml_backend_sched_graph_compute(sched, graph)

    row_start <- samp_start + 1L
    row_end   <- samp_start + batch_size
    for (io in seq_len(n_outputs)) {
      ne_out    <- ne_outputs_vec[io]
      batch_out <- ggml_backend_tensor_get_data(graph_info$outputs[[io]])
      mat       <- matrix(batch_out, nrow = ne_out, ncol = batch_size)
      all_preds_list[[io]][row_start:row_end, ] <- t(mat)
    }
  }

  ggml_free(graph_info$ctx_compute)
  ggml_backend_buffer_free(graph_info$buffer)
  ggml_free(graph_info$ctx_weights)

  # Trim padding and return
  if (n_outputs == 1L) {
    return(all_preds_list[[1L]][seq_len(n_samples_orig), , drop = FALSE])
  } else {
    return(lapply(all_preds_list, function(m) m[seq_len(n_samples_orig), , drop = FALSE]))
  }
}

# ============================================================================
# Print method
# ============================================================================

#' Print method for ggml_functional_model
#' @param x A ggml_functional_model object
#' @param ... Additional arguments (ignored)
#' @return The model object (invisibly).
#' @export
print.ggml_functional_model <- function(x, ...) {
  model <- x
  cat("ggmlR Functional Model\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")
  cat(sprintf("Inputs:   %d\n", length(model$inputs)))
  cat(sprintf("Outputs:  %d\n", length(model$outputs)))
  cat(sprintf("Compiled: %s\n", if (model$compiled) "yes" else "no"))

  nodes <- nn_topo_sort(model$outputs)
  total_params <- 0L
  cat(sprintf("\n%-20s %-15s\n", "Layer (type)", "Node type"))
  cat(paste(rep("-", 40), collapse = ""), "\n")
  for (node in nodes) {
    n_params <- switch(node$node_type,
      "dense" = {
        # approximate: fan_in * units + units
        0L  # shape not available here without building
      },
      0L
    )
    nm <- if (!is.null(node$config$name)) node$config$name else node$id
    cat(sprintf("%-20s %-15s\n", nm, node$node_type))
  }
  cat(paste(rep("=", 60), collapse = ""), "\n")
  invisible(x)
}
