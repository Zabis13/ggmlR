# High-level neural network layer definitions for ggmlR
# Provides Keras-like layer API with pipe (%>%) support

# ============================================================================
# Layer name auto-generation
# ============================================================================

# Returns the next auto-generated name for a given layer type, e.g. "dense_1"
nn_layer_name <- function(model, type) {
  count <- sum(vapply(model$layers, function(l) l$type == type, logical(1)))
  paste0(type, "_", count + 1L)
}

# ============================================================================
# Layer Constructors (pipe-friendly, add layer config to model)
# ============================================================================

#' Add 2D Convolution Layer
#'
#' @param model A ggml_sequential_model object
#' @param filters Number of output filters
#' @param kernel_size Integer or vector of 2 integers for kernel height and width
#' @param activation Activation function name: "relu", "sigmoid", "tanh", "softmax", or NULL
#' @param input_shape Input shape c(H, W, C) - required for first layer only
#' @param strides Integer or vector of 2 integers for stride
#' @param padding "valid" (no padding) or "same" (preserve spatial dims)
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the layer weights are updated during training.
#' @return The model object with the conv_2d layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1))
#' }
ggml_layer_conv_2d <- function(model, filters, kernel_size, activation = NULL,
                                input_shape = NULL, strides = c(1L, 1L),
                                padding = "valid", name = NULL, trainable = TRUE) {
  if (length(kernel_size) == 1) kernel_size <- rep(as.integer(kernel_size), 2)
  if (length(strides) == 1) strides <- rep(as.integer(strides), 2)

  # Functional API: model is actually a tensor node
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("conv_2d_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "conv_2d",
      trainable = trainable,
      config    = list(
        filters     = as.integer(filters),
        kernel_size = as.integer(kernel_size),
        strides     = as.integer(strides),
        padding     = padding,
        activation  = activation,
        name        = name
      ),
      parents = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "conv_2d")

  layer <- list(
    type = "conv_2d",
    name = name,
    trainable = trainable,
    config = list(
      filters = as.integer(filters),
      kernel_size = as.integer(kernel_size),
      strides = as.integer(strides),
      padding = padding,
      activation = activation
    ),
    input_shape = input_shape,
    output_shape = NULL,
    weights = list(kernel = NULL, bias = NULL)
  )

  if (!is.null(input_shape) && is.null(model$input_shape)) {
    model$input_shape <- as.integer(input_shape)
  }

  model$layers <- c(model$layers, list(layer))
  model
}

# ============================================================================
# Global pooling layers
# ============================================================================

#' Global Max Pooling for 2D Feature Maps
#'
#' Reduces a \code{[H, W, C]} feature map to \code{[C]} by taking the maximum
#' value per channel across all spatial positions.  Equivalent to Keras
#' \code{GlobalMaxPooling2D()}.
#'
#' @param model A \code{ggml_sequential_model} or \code{ggml_tensor_node}.
#' @param name Optional character name for the layer.
#' @param trainable Logical; reserved for API consistency (no weights).
#' @return Updated model or a new \code{ggml_tensor_node}.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_global_max_pooling_2d() |>
#'   ggml_layer_dense(10, activation = "softmax")
#' }
ggml_layer_global_max_pooling_2d <- function(model, name = NULL, trainable = TRUE) {
  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- nn_auto_name("global_max_pooling_2d")
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "global_max_pooling_2d",
      trainable = trainable,
      config    = list(name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "global_max_pooling_2d")

  layer <- list(
    type = "global_max_pooling_2d",
    name = name,
    trainable = trainable,
    config = list(),
    input_shape = NULL,
    output_shape = NULL,
    weights = list()
  )

  model$layers <- c(model$layers, list(layer))
  model
}

#' Global Average Pooling for 2D Feature Maps
#'
#' Reduces a \code{[H, W, C]} feature map to \code{[C]} by averaging all
#' spatial positions per channel.  Equivalent to Keras
#' \code{GlobalAveragePooling2D()}.
#'
#' @param model A \code{ggml_sequential_model} or \code{ggml_tensor_node}.
#' @param name Optional character name for the layer.
#' @param trainable Logical; reserved for API consistency (no weights).
#' @return Updated model or a new \code{ggml_tensor_node}.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_global_average_pooling_2d() |>
#'   ggml_layer_dense(10, activation = "softmax")
#' }
ggml_layer_global_average_pooling_2d <- function(model, name = NULL, trainable = TRUE) {
  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- nn_auto_name("global_average_pooling_2d")
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "global_average_pooling_2d",
      trainable = trainable,
      config    = list(name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "global_average_pooling_2d")

  layer <- list(
    type = "global_average_pooling_2d",
    name = name,
    trainable = trainable,
    config = list(),
    input_shape = NULL,
    output_shape = NULL,
    weights = list()
  )

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add 2D Max Pooling Layer
#'
#' @param model A ggml_sequential_model object
#' @param pool_size Integer or vector of 2 integers for pool height and width
#' @param strides Integer or vector of 2 integers (defaults to pool_size)
#' @param name Optional character name for the layer.
#' @param trainable Logical; reserved for API consistency (no weights).
#' @return The model object with the max pooling layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_max_pooling_2d(c(2, 2))
#' }
ggml_layer_max_pooling_2d <- function(model, pool_size = c(2L, 2L), strides = NULL,
                                       name = NULL, trainable = TRUE) {
  if (length(pool_size) == 1) pool_size <- rep(as.integer(pool_size), 2)
  if (is.null(strides)) strides <- pool_size
  if (length(strides) == 1) strides <- rep(as.integer(strides), 2)

  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("max_pooling_2d_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "max_pooling_2d",
      trainable = trainable,
      config    = list(
        pool_size = as.integer(pool_size),
        strides   = as.integer(strides),
        name      = name
      ),
      parents = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "max_pooling_2d")

  layer <- list(
    type = "max_pooling_2d",
    name = name,
    trainable = trainable,
    config = list(
      pool_size = as.integer(pool_size),
      strides = as.integer(strides)
    ),
    input_shape = NULL,
    output_shape = NULL,
    weights = list()
  )

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add Flatten Layer
#'
#' Flattens the spatial dimensions into a single vector per sample.
#'
#' @param model A ggml_sequential_model object
#' @param name Optional character name for the layer.
#' @param trainable Logical; reserved for API consistency (no weights).
#' @return The model object with the flatten layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_flatten()
#' }
ggml_layer_flatten <- function(model, name = NULL, trainable = TRUE) {
  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("flatten_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "flatten",
      trainable = trainable,
      config    = list(name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "flatten")

  layer <- list(
    type = "flatten",
    name = name,
    trainable = trainable,
    config = list(),
    input_shape = NULL,
    output_shape = NULL,
    weights = list()
  )

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add 1D Convolution Layer
#'
#' @param model A ggml_sequential_model object
#' @param filters Number of output filters
#' @param kernel_size Integer kernel size
#' @param activation Activation function name: "relu", "sigmoid", "tanh", "softmax", or NULL
#' @param input_shape Input shape c(L, C) - required for first layer only (length, channels)
#' @param strides Integer stride (default 1)
#' @param padding "valid" (no padding) or "same" (preserve length)
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the layer weights are updated during training.
#' @return The model object with the conv_1d layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_1d(32, 3, activation = "relu",
#'                      input_shape = c(100, 1))
#' }
ggml_layer_conv_1d <- function(model, filters, kernel_size, activation = NULL,
                                input_shape = NULL, strides = 1L,
                                padding = "valid", name = NULL, trainable = TRUE) {
  kernel_size <- as.integer(kernel_size)
  strides <- as.integer(strides)

  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("conv_1d_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "conv_1d",
      trainable = trainable,
      config    = list(
        filters     = as.integer(filters),
        kernel_size = kernel_size,
        strides     = strides,
        padding     = padding,
        activation  = activation,
        name        = name
      ),
      parents = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "conv_1d")

  layer <- list(
    type = "conv_1d",
    name = name,
    trainable = trainable,
    config = list(
      filters = as.integer(filters),
      kernel_size = kernel_size,
      strides = strides,
      padding = padding,
      activation = activation
    ),
    input_shape = input_shape,
    output_shape = NULL,
    weights = list(kernel = NULL, bias = NULL)
  )

  if (!is.null(input_shape) && is.null(model$input_shape)) {
    model$input_shape <- as.integer(input_shape)
  }

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add Batch Normalization Layer
#'
#' Applies normalization: RMS-normalizes the input, then scales by gamma
#' and shifts by beta (both learnable). Uses \code{ggml_rms_norm} which
#' supports backward pass for training.
#'
#' @param model A ggml_sequential_model object
#' @param eps Small constant for numerical stability (default 1e-5)
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the layer weights are updated during training.
#' @return The model object with the batch_norm layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(128, input_shape = 784) |>
#'   ggml_layer_batch_norm() |>
#'   ggml_layer_dense(10, activation = "softmax")
#' }
ggml_layer_batch_norm <- function(model, eps = 1e-5, name = NULL, trainable = TRUE) {
  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("batch_norm_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "batch_norm",
      trainable = trainable,
      config    = list(eps = eps, name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "batch_norm")

  layer <- list(
    type = "batch_norm",
    name = name,
    trainable = trainable,
    config = list(eps = eps),
    input_shape = NULL,
    output_shape = NULL,
    weights = list(gamma = NULL, beta = NULL)
  )

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add Dense (Fully Connected) Layer
#'
#' @param model A ggml_sequential_model object
#' @param units Number of output units
#' @param activation Activation function name: "relu", "sigmoid", "tanh", "softmax", or NULL
#' @param input_shape Integer or integer vector specifying the input shape (only needed for the first layer)
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the layer weights are updated during training.
#' @return The model object with the dense layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_flatten() |>
#'   ggml_layer_dense(128, activation = "relu")
#' }
ggml_layer_dense <- function(model, units, activation = NULL, input_shape = NULL,
                              name = NULL, trainable = TRUE) {
  # Functional API: model is a tensor node
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("dense_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "dense",
      trainable = trainable,
      config    = list(
        units      = as.integer(units),
        activation = activation,
        name       = name
      ),
      parents = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "dense")

  layer <- list(
    type = "dense",
    name = name,
    trainable = trainable,
    config = list(
      units = as.integer(units),
      activation = activation
    ),
    input_shape = input_shape,
    output_shape = NULL,
    weights = list(weight = NULL, bias = NULL)
  )

  if (!is.null(input_shape) && is.null(model$input_shape)) {
    model$input_shape <- as.integer(input_shape)
  }

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add Dropout Layer
#'
#' Applies dropout regularization. During training, multiplies all activations
#' by \code{(1 - rate)} (deterministic expected-value scaling).
#' During inference (\code{training = FALSE}), the layer is an identity (no change).
#'
#' @section Difference from Keras / inverted dropout:
#' Keras implements \emph{inverted dropout}: during training it applies a random
#' Bernoulli mask and scales surviving activations \emph{up} by
#' \code{1 / (1 - rate)}, so the expected value of each unit is preserved and
#' no scaling is needed at inference.
#'
#' This implementation uses \emph{deterministic scaling} (multiply by
#' \code{(1 - rate)} at training, identity at inference) — equivalent in
#' expected value but without stochastic noise.  Consequences:
#' \itemize{
#'   \item No random mask → the regularization signal is weaker (no co-adaptation
#'     breaking).
#'   \item Activations at training are scaled \emph{down}, not up — the magnitude
#'     seen by subsequent layers differs from Keras behaviour.
#'   \item Results are fully deterministic and reproducible without setting a seed.
#' }
#'
#' @note With \code{stochastic = TRUE} the Bernoulli mask is regenerated once
#'   per epoch (not per batch), because \code{ggml_opt_fit} processes all
#'   batches inside a single C call.  This is weaker than per-batch dropout
#'   but stronger than the deterministic variant.
#'
#' @param model A \code{ggml_sequential_model} or \code{ggml_tensor_node}.
#' @param rate Dropout rate in \code{[0, 1)}.  Fraction of units to "drop".
#' @param stochastic Logical.  If \code{TRUE}, use inverted dropout with a
#'   random Bernoulli mask regenerated each epoch (proper regularization).
#'   If \code{FALSE} (default), use deterministic scaling by
#'   \code{(1 - rate)} — cheaper but weaker regularization.
#' @param name Optional layer name.
#' @param trainable Ignored for dropout (no weights); kept for API consistency.
#' @return The model with the dropout layer appended, or a new tensor node.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(128, activation = "relu", input_shape = 784L) |>
#'   ggml_layer_dropout(0.5, stochastic = TRUE) |>
#'   ggml_layer_dense(10, activation = "softmax")
#' }
ggml_layer_dropout <- function(model, rate, stochastic = FALSE, name = NULL,
                                trainable = FALSE) {
  rate <- as.double(rate)
  stopifnot(rate >= 0, rate < 1)

  if (inherits(model, "ggml_tensor_node")) {
    node_name <- if (is.null(name)) paste0("dropout_", nn_next_node_id_peek()) else name
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "dropout",
      trainable = FALSE,
      config    = list(rate = rate, stochastic = stochastic, name = node_name),
      parents   = list(model)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "dropout")
  layer <- list(
    type = "dropout", name = name, trainable = FALSE,
    config = list(rate = rate, stochastic = stochastic),
    input_shape = NULL, output_shape = NULL, weights = list()
  )
  model$layers <- c(model$layers, list(layer))
  model
}

#' Add Embedding Layer
#'
#' Looks up dense vectors for integer token indices.  The input must be an
#' integer matrix of 0-based indices in \code{[0, vocab_size - 1]} (use
#' \code{ggml_input(shape, dtype = "int32")} in Functional mode).
#'
#' @section Axis order (ggml vs Keras):
#' ggml stores tensors in column-major order, so the output shape is
#' \code{[dim, seq_len]} per sample (ggml convention) rather than
#' \code{[seq_len, dim]} as in Keras.  When you call \code{ggml_layer_flatten()}
#' after embedding the result is the same flattened vector regardless of order,
#' but if you access raw output tensors be aware of this transposition.
#'
#' @section Index validation:
#' Indices must be in \code{[0, vocab_size - 1]}.  Out-of-range values cause
#' undefined behaviour inside the ggml kernel (no bounds check is performed at
#' the R level).
#'
#' @param model A \code{ggml_sequential_model} or \code{ggml_tensor_node}.
#' @param vocab_size Number of distinct tokens (vocabulary size).
#' @param dim Embedding dimension (vector length per token).
#' @param name Optional layer name.
#' @param trainable Logical; whether embedding weights are updated during training.
#' @return The model with the embedding layer appended, or a new tensor node.
#' @export
#' @examples
#' \donttest{
#' inp <- ggml_input(shape = 10L, dtype = "int32")
#' out <- inp |>
#'   ggml_layer_embedding(vocab_size = 1000L, dim = 32L) |>
#'   ggml_layer_flatten() |>
#'   ggml_layer_dense(10L, activation = "softmax")
#' model <- ggml_model(inputs = inp, outputs = out)
#' }
ggml_layer_embedding <- function(model, vocab_size, dim, name = NULL, trainable = TRUE) {
  vocab_size <- as.integer(vocab_size)
  dim        <- as.integer(dim)

  if (inherits(model, "ggml_tensor_node")) {
    node_id   <- nn_next_node_id()
    node_name <- if (is.null(name)) paste0("embedding_", node_id) else name
    return(structure(list(
      id        = node_id,
      node_type = "embedding",
      trainable = trainable,
      config    = list(vocab_size = vocab_size, dim = dim, name = node_name),
      parents   = list(model)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "embedding")
  layer <- list(
    type = "embedding", name = name, trainable = trainable,
    config = list(vocab_size = vocab_size, dim = dim),
    input_shape = NULL, output_shape = NULL, weights = list(weight = NULL)
  )
  model$layers <- c(model$layers, list(layer))
  model
}

# Peek at next node id without incrementing
nn_next_node_id_peek <- function() {
  paste0("node_", .fn_node_counter$n + 1L)
}

# ============================================================================
# Shape Inference
# ============================================================================

#' Infer shapes for all layers in model
#' @param model Sequential model
#' @return Model with shapes filled in
#' @keywords internal
nn_infer_shapes <- function(model) {
  if (is.null(model$input_shape)) {
    stop("First layer must have input_shape specified")
  }

  current_shape <- model$input_shape

  for (i in seq_along(model$layers)) {
    layer <- model$layers[[i]]
    layer$input_shape <- current_shape

    current_shape <- switch(layer$type,
      "conv_1d" = {
        L <- current_shape[1]
        C_in <- current_shape[2]
        k <- layer$config$kernel_size
        s <- layer$config$strides

        if (layer$config$padding == "same") {
          L_out <- ceiling(L / s)
        } else {
          L_out <- floor((L - k) / s) + 1L
        }
        as.integer(c(L_out, layer$config$filters))
      },
      "conv_2d" = {
        H <- current_shape[1]
        W <- current_shape[2]
        C_in <- current_shape[3]
        kh <- layer$config$kernel_size[1]
        kw <- layer$config$kernel_size[2]
        sh <- layer$config$strides[1]
        sw <- layer$config$strides[2]

        if (layer$config$padding == "same") {
          H_out <- ceiling(H / sh)
          W_out <- ceiling(W / sw)
        } else {
          H_out <- floor((H - kh) / sh) + 1L
          W_out <- floor((W - kw) / sw) + 1L
        }
        as.integer(c(H_out, W_out, layer$config$filters))
      },
      "max_pooling_2d" = {
        H <- current_shape[1]
        W <- current_shape[2]
        C <- current_shape[3]
        ph <- layer$config$pool_size[1]
        pw <- layer$config$pool_size[2]
        sh <- layer$config$strides[1]
        sw <- layer$config$strides[2]
        H_out <- floor((H - ph) / sh) + 1L
        W_out <- floor((W - pw) / sw) + 1L
        as.integer(c(H_out, W_out, C))
      },
      "global_max_pooling_2d" = ,
      "global_average_pooling_2d" = {
        # [H, W, C] -> [C]
        as.integer(current_shape[3])
      },
      "flatten" = {
        as.integer(prod(current_shape))
      },
      "dense" = {
        as.integer(layer$config$units)
      },
      "batch_norm" = {
        current_shape  # batch_norm doesn't change shape
      },
      "dropout" = {
        current_shape  # dropout doesn't change shape
      },
      "embedding" = {
        # input shape: c(seq_len) -> output: c(dim, seq_len)
        seq_len <- if (length(current_shape) == 1L) current_shape else prod(current_shape)
        as.integer(c(layer$config$dim, seq_len))
      },
      "lstm" = {
        # input: c(seq_len, input_size)
        # output: c(units) or c(seq_len, units)
        seq_len <- current_shape[1]
        units   <- layer$config$units
        if (isTRUE(layer$config$return_sequences)) {
          as.integer(c(seq_len, units))
        } else {
          as.integer(units)
        }
      },
      "gru" = {
        seq_len <- current_shape[1]
        units   <- layer$config$units
        if (isTRUE(layer$config$return_sequences)) {
          as.integer(c(seq_len, units))
        } else {
          as.integer(units)
        }
      },
      stop("Unknown layer type: ", layer$type)
    )

    layer$output_shape <- current_shape
    model$layers[[i]] <- layer
  }

  model
}

# ============================================================================
# Build Functions (create ggml computation graph for each layer)
# ============================================================================

#' Apply activation function
#' @keywords internal
nn_apply_activation <- function(ctx, tensor, activation) {
  if (is.null(activation)) return(tensor)

  switch(activation,
    "relu"         = ggml_relu(ctx, tensor),
    "sigmoid"      = ggml_sigmoid(ctx, tensor),
    "tanh"         = ggml_tanh(ctx, tensor),
    "silu"         = ggml_silu(ctx, tensor),
    "gelu"         = ggml_gelu(ctx, tensor),
    "hardsigmoid"  = ggml_hardsigmoid(ctx, tensor),
    "hardswish"    = ggml_hardswish(ctx, tensor),
    "softmax"      = ggml_soft_max(ctx, tensor),
    stop("Unknown activation: ", activation)
  )
}

#' Build conv_1d forward pass
#' @keywords internal
nn_build_conv_1d <- function(ctx, input_tensor, layer) {
  kernel <- layer$weights$kernel
  bias <- layer$weights$bias

  s0 <- layer$config$strides

  if (layer$config$padding == "same") {
    k <- layer$config$kernel_size
    p0 <- as.integer(floor(k / 2))
  } else {
    p0 <- 0L
  }

  out <- ggml_conv_1d(ctx, kernel, input_tensor, s0 = s0, p0 = p0, d0 = 1L)

  # Reshape bias [OC] -> [1, OC, 1] for broadcasting with [OL, OC, N]
  bias_3d <- ggml_reshape_3d(ctx, bias, 1L, layer$config$filters, 1L)
  out <- ggml_add(ctx, out, bias_3d)
  nn_apply_activation(ctx, out, layer$config$activation)
}

#' Build conv_2d forward pass
#' @keywords internal
nn_build_conv_2d <- function(ctx, input_tensor, layer) {
  kernel <- layer$weights$kernel
  bias <- layer$weights$bias

  s0 <- layer$config$strides[2]  # width stride (ne0 = W)
  s1 <- layer$config$strides[1]  # height stride (ne1 = H)

  if (layer$config$padding == "same") {
    kw <- layer$config$kernel_size[2]
    kh <- layer$config$kernel_size[1]
    p0 <- as.integer(floor(kw / 2))
    p1 <- as.integer(floor(kh / 2))
  } else {
    p0 <- 0L
    p1 <- 0L
  }

  out <- ggml_conv_2d(ctx, kernel, input_tensor,
                       s0 = s0, s1 = s1, p0 = p0, p1 = p1, d0 = 1L, d1 = 1L)

  # Reshape bias [OC] -> [1, 1, OC, 1] for broadcasting with [W, H, OC, N]
  bias_4d <- ggml_reshape_4d(ctx, bias, 1L, 1L, layer$config$filters, 1L)
  out <- ggml_add(ctx, out, bias_4d)
  nn_apply_activation(ctx, out, layer$config$activation)
}

#' Build max_pooling_2d forward pass
#' @keywords internal
nn_build_max_pooling_2d <- function(ctx, input_tensor, layer) {
  k0 <- layer$config$pool_size[2]  # width (ne0)
  k1 <- layer$config$pool_size[1]  # height (ne1)
  s0 <- layer$config$strides[2]
  s1 <- layer$config$strides[1]

  ggml_pool_2d(ctx, input_tensor, GGML_OP_POOL_MAX,
               k0 = k0, k1 = k1, s0 = s0, s1 = s1, p0 = 0L, p1 = 0L)
}

#' Build global_max_pooling_2d forward pass
#'
#' Tensor layout in ggml (column-major): [W, H, C, N].
#' Pool the entire spatial extent (k0=W, k1=H), then reshape [1,1,C,N]->[C,N].
#' @keywords internal
nn_build_global_max_pooling_2d <- function(ctx, input_tensor, layer) {
  sh <- ggml_tensor_shape(input_tensor)   # [W, H, C, N] (ggml order)
  W  <- sh[1]; H <- sh[2]; C <- sh[3]; N <- sh[4]
  pooled <- ggml_pool_2d(ctx, input_tensor, GGML_OP_POOL_MAX,
                          k0 = W, k1 = H, s0 = W, s1 = H,
                          p0 = 0L, p1 = 0L)
  # pooled: [1, 1, C, N] -> reshape to [C, N]
  ggml_reshape_2d(ctx, pooled, C, N)
}

#' Build global_average_pooling_2d forward pass
#' @keywords internal
nn_build_global_average_pooling_2d <- function(ctx, input_tensor, layer) {
  sh <- ggml_tensor_shape(input_tensor)
  W  <- sh[1]; H <- sh[2]; C <- sh[3]; N <- sh[4]
  pooled <- ggml_pool_2d(ctx, input_tensor, GGML_OP_POOL_AVG,
                          k0 = W, k1 = H, s0 = W, s1 = H,
                          p0 = 0L, p1 = 0L)
  ggml_reshape_2d(ctx, pooled, C, N)
}

#' Build flatten forward pass
#' @keywords internal
nn_build_flatten <- function(ctx, input_tensor, layer) {
  n_features <- prod(layer$input_shape)
  # Batch dim is the last non-1 dimension, determined by n_dims
  ndims <- ggml_n_dims(input_tensor)
  shape <- ggml_tensor_shape(input_tensor)
  batch_size <- shape[ndims]

  ggml_reshape_2d(ctx, input_tensor, n_features, batch_size)
}

#' Build dense forward pass
#' @keywords internal
nn_build_dense <- function(ctx, input_tensor, layer) {
  W <- layer$weights$weight
  b <- layer$weights$bias

  out <- ggml_mul_mat(ctx, W, input_tensor)
  out <- ggml_add(ctx, out, b)
  nn_apply_activation(ctx, out, layer$config$activation)
}

#' Build batch_norm forward pass
#' @keywords internal
nn_build_batch_norm <- function(ctx, input_tensor, layer) {
  gamma <- layer$weights$gamma
  beta <- layer$weights$beta
  eps <- layer$config$eps

  # Use rms_norm (has backward pass, unlike ggml_norm)
  normed <- ggml_rms_norm(ctx, input_tensor, eps = eps)

  # Scale and shift: gamma * normed + beta
  # gamma and beta are 1D [n_features], need reshape for broadcasting
  input_shape <- layer$input_shape
  if (length(input_shape) == 3) {
    # [W, H, C, N] -> reshape gamma to [1, 1, C, 1]
    gamma_r <- ggml_reshape_4d(ctx, gamma, 1L, 1L, as.integer(input_shape[3]), 1L)
    beta_r <- ggml_reshape_4d(ctx, beta, 1L, 1L, as.integer(input_shape[3]), 1L)
  } else if (length(input_shape) == 2) {
    # [L, C, N] -> reshape gamma to [1, C, 1]
    gamma_r <- ggml_reshape_3d(ctx, gamma, 1L, as.integer(input_shape[2]), 1L)
    beta_r <- ggml_reshape_3d(ctx, beta, 1L, as.integer(input_shape[2]), 1L)
  } else {
    # [features, N] -> gamma is already [features], broadcast over N
    gamma_r <- gamma
    beta_r <- beta
  }

  out <- ggml_mul(ctx, normed, gamma_r)
  ggml_add(ctx, out, beta_r)
}

#' Build dropout forward pass
#' @keywords internal
nn_build_dropout <- function(ctx, input_tensor, layer, training = TRUE) {
  if (!training) return(input_tensor)
  stochastic <- isTRUE(layer$config$stochastic)
  if (stochastic && !is.null(layer$weights$mask)) {
    # Inverted dropout: input * mask * (1 / (1 - rate))
    out <- ggml_mul(ctx, input_tensor, layer$weights$mask)
    ggml_scale(ctx, out, 1.0 / (1.0 - layer$config$rate))
  } else {
    # Deterministic expected-value scaling
    ggml_scale(ctx, input_tensor, 1.0 - layer$config$rate)
  }
}

#' Build embedding forward pass
#' @keywords internal
nn_build_embedding <- function(ctx_weights, ctx_compute, input_tensor, layer) {
  vocab_size <- layer$config$vocab_size
  dim        <- layer$config$dim
  E <- layer$weights$weight
  ggml_get_rows(ctx_compute, E, input_tensor)
}

#' Build a layer's forward pass
#' @keywords internal
nn_build_layer <- function(ctx, input_tensor, layer, training = TRUE,
                            ctx_weights = NULL) {
  switch(layer$type,
    "conv_1d" = nn_build_conv_1d(ctx, input_tensor, layer),
    "conv_2d" = nn_build_conv_2d(ctx, input_tensor, layer),
    "max_pooling_2d" = nn_build_max_pooling_2d(ctx, input_tensor, layer),
    "global_max_pooling_2d"     = nn_build_global_max_pooling_2d(ctx, input_tensor, layer),
    "global_average_pooling_2d" = nn_build_global_average_pooling_2d(ctx, input_tensor, layer),
    "flatten" = nn_build_flatten(ctx, input_tensor, layer),
    "dense" = nn_build_dense(ctx, input_tensor, layer),
    "batch_norm" = nn_build_batch_norm(ctx, input_tensor, layer),
    "dropout" = nn_build_dropout(ctx, input_tensor, layer, training),
    "embedding" = nn_build_embedding(ctx_weights, ctx, input_tensor, layer),
    "lstm" = nn_build_lstm(ctx, input_tensor, layer, batch_size = NULL),
    "gru"  = nn_build_gru(ctx, input_tensor, layer, batch_size = NULL),
    stop("Unknown layer type: ", layer$type)
  )
}

# ============================================================================
# Weight Initialization
# ============================================================================

#' Initialize weight tensor with He uniform distribution
#' @importFrom stats runif
#' @keywords internal
nn_init_he_uniform <- function(tensor, fan_in) {
  n <- ggml_nelements(tensor)
  limit <- sqrt(6.0 / fan_in)
  ggml_backend_tensor_set_data(tensor, runif(n, -limit, limit))
}

#' Initialize weight tensor with Glorot uniform distribution
#' @keywords internal
nn_init_glorot_uniform <- function(tensor, fan_in, fan_out) {
  n <- ggml_nelements(tensor)
  limit <- sqrt(6.0 / (fan_in + fan_out))
  ggml_backend_tensor_set_data(tensor, runif(n, -limit, limit))
}

#' Initialize bias tensor to zeros
#' @keywords internal
nn_init_zeros <- function(tensor) {
  n <- ggml_nelements(tensor)
  ggml_backend_tensor_set_data(tensor, rep(0.0, n))
}

#' Initialize recurrent weight tensor with small deterministic values
#'
#' Uses a fixed zigzag pattern in [-0.01, 0.01] — no RNG, fully reproducible
#' across all platforms and independent of the R random seed state.
#' @keywords internal
nn_init_recurrent_uniform <- function(tensor) {
  n <- ggml_nelements(tensor)
  vals <- ((seq_len(n) - 1L) %% 20L - 10L) / 1000.0
  ggml_backend_tensor_set_data(tensor, vals)
}

# ============================================================================
# Recurrent layers — LSTM and GRU
# ============================================================================

#' Add an LSTM Layer
#'
#' Long Short-Term Memory recurrent layer.  Implemented as an unrolled
#' computation graph (BPTT) so that ggml's automatic differentiation works
#' without any C extensions.
#'
#' @section Weight layout:
#' \itemize{
#'   \item \code{W_gates} \code{[input_size, 4*units]} — input kernel for all
#'     four gates (i, f, g, o) concatenated.
#'   \item \code{U_gates} \code{[units, 4*units]} — recurrent kernel.
#'   \item \code{b_gates} \code{[4*units]} — bias.
#' }
#'
#' @section Input / output shapes:
#' Input: \code{[seq_len, input_size]} per sample (R row-major), or a 3-D
#' array \code{[N, seq_len, input_size]}.  In the Functional API the input
#' node shape should be \code{c(seq_len, input_size)}.
#'
#' Output (Sequential): \code{[units]} per sample when
#' \code{return_sequences = FALSE} (default), or \code{c(seq_len, units)}
#' when \code{return_sequences = TRUE}.
#'
#' @param model A \code{ggml_sequential_model} or \code{ggml_tensor_node}.
#' @param units Integer, number of hidden units.
#' @param return_sequences Logical; if \code{TRUE} return all hidden states,
#'   otherwise return only the last hidden state.
#' @param activation Activation for the cell gate (default \code{"tanh"}).
#' @param recurrent_activation Activation for the recurrent step (default
#'   \code{"sigmoid"}).
#' @param input_shape Input shape \code{c(seq_len, input_size)} -- required for the first layer only.
#' @param name Optional layer name.
#' @param trainable Logical.
#' @return Updated model or a new \code{ggml_tensor_node}.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_lstm(64L, input_shape = c(10L, 32L)) |>
#'   ggml_layer_dense(10L, activation = "softmax")
#' }
ggml_layer_lstm <- function(model, units, return_sequences = FALSE,
                              activation = "tanh",
                              recurrent_activation = "sigmoid",
                              input_shape = NULL,
                              name = NULL, trainable = TRUE) {
  units <- as.integer(units)

  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- nn_auto_name("lstm")
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "lstm",
      trainable = trainable,
      config    = list(
        units                = units,
        return_sequences     = return_sequences,
        activation           = activation,
        recurrent_activation = recurrent_activation,
        name                 = name
      ),
      parents = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "lstm")

  layer <- list(
    type = "lstm",
    name = name,
    trainable = trainable,
    config = list(
      units                = units,
      return_sequences     = return_sequences,
      activation           = activation,
      recurrent_activation = recurrent_activation
    ),
    input_shape  = NULL,
    output_shape = NULL,
    weights = list()
  )

  if (!is.null(input_shape) && is.null(model$input_shape)) {
    model$input_shape <- as.integer(input_shape)
  }

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add a GRU Layer
#'
#' Gated Recurrent Unit recurrent layer.  Implemented as an unrolled
#' computation graph (BPTT).
#'
#' @section Weight layout:
#' \itemize{
#'   \item \code{W_zh} \code{[input_size, 2*units]} — input kernel for z and r
#'     gates.
#'   \item \code{U_zh} \code{[units, 2*units]} — recurrent kernel for z and r.
#'   \item \code{b_zh} \code{[2*units]} — bias for z and r.
#'   \item \code{W_n}  \code{[input_size, units]} — input kernel for candidate.
#'   \item \code{U_n}  \code{[units, units]} — recurrent kernel for candidate.
#'   \item \code{b_n}  \code{[units]} — bias for candidate.
#' }
#'
#' @param model A \code{ggml_sequential_model} or \code{ggml_tensor_node}.
#' @param units Integer, number of hidden units.
#' @param return_sequences Logical; return all hidden states or only the last.
#' @param activation Activation for the candidate hidden state (\code{"tanh"}).
#' @param recurrent_activation Activation for z/r gates (\code{"sigmoid"}).
#' @param input_shape Input shape \code{c(seq_len, input_size)} -- required for the first layer only.
#' @param name Optional layer name.
#' @param trainable Logical.
#' @return Updated model or a new \code{ggml_tensor_node}.
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_gru(64L, input_shape = c(10L, 32L)) |>
#'   ggml_layer_dense(10L, activation = "softmax")
#' }
ggml_layer_gru <- function(model, units, return_sequences = FALSE,
                             activation = "tanh",
                             recurrent_activation = "sigmoid",
                             input_shape = NULL,
                             name = NULL, trainable = TRUE) {
  units <- as.integer(units)

  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- nn_auto_name("gru")
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "gru",
      trainable = trainable,
      config    = list(
        units                = units,
        return_sequences     = return_sequences,
        activation           = activation,
        recurrent_activation = recurrent_activation,
        name                 = name
      ),
      parents = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "gru")

  layer <- list(
    type = "gru",
    name = name,
    trainable = trainable,
    config = list(
      units                = units,
      return_sequences     = return_sequences,
      activation           = activation,
      recurrent_activation = recurrent_activation
    ),
    input_shape  = NULL,
    output_shape = NULL,
    weights = list()
  )

  if (!is.null(input_shape) && is.null(model$input_shape)) {
    model$input_shape <- as.integer(input_shape)
  }

  model$layers <- c(model$layers, list(layer))
  model
}

#' Build one LSTM step
#'
#' @param ctx ggml compute context
#' @param x_t  Input at this step: tensor [input_size, N]
#' @param h_t  Previous hidden state: tensor [units, N]
#' @param c_t  Previous cell state: tensor [units, N]
#' @param W_gates [input_size, 4*units]
#' @param U_gates [units, 4*units]
#' @param b_gates [4*units]
#' @param units Integer
#' @param act_cell Activation name for cell gate (tanh by default)
#' @param act_rec  Activation name for i/f/o gates (sigmoid by default)
#' @return list(h = new_h, c = new_c)
#' @keywords internal
nn_lstm_step <- function(ctx, x_t, h_t, c_t, W_gates, U_gates, b_gates,
                          units, act_cell, act_rec) {
  # gates_raw: [4*units, N]
  gates_raw <- ggml_add(ctx,
    ggml_add(ctx,
      ggml_mul_mat(ctx, W_gates, x_t),
      ggml_mul_mat(ctx, U_gates, h_t)
    ),
    b_gates
  )

  # Split into 4 gates via view: each [units, N]
  N      <- ggml_tensor_shape(x_t)[2]  # batch dimension
  stride <- as.integer(units * 4L)     # ne0 of gates_raw (column-major)

  i_raw <- ggml_view_2d(ctx, gates_raw, units, N,
                          nb1 = as.integer(stride * 4L), offset = 0L)
  f_raw <- ggml_view_2d(ctx, gates_raw, units, N,
                          nb1 = as.integer(stride * 4L),
                          offset = as.integer(units * 4L))
  g_raw <- ggml_view_2d(ctx, gates_raw, units, N,
                          nb1 = as.integer(stride * 4L),
                          offset = as.integer(units * 4L * 2L))
  o_raw <- ggml_view_2d(ctx, gates_raw, units, N,
                          nb1 = as.integer(stride * 4L),
                          offset = as.integer(units * 4L * 3L))

  i_gate <- nn_apply_activation(ctx, i_raw, act_rec)
  f_gate <- nn_apply_activation(ctx, f_raw, act_rec)
  g_gate <- nn_apply_activation(ctx, g_raw, act_cell)
  o_gate <- nn_apply_activation(ctx, o_raw, act_rec)

  new_c <- ggml_add(ctx,
    ggml_mul(ctx, f_gate, c_t),
    ggml_mul(ctx, i_gate, g_gate)
  )
  new_h <- ggml_mul(ctx, o_gate, nn_apply_activation(ctx, new_c, act_cell))

  list(h = new_h, c = new_c)
}

#' Build one GRU step
#'
#' @param ctx     ggml compute context
#' @param x_t     Input at this step: [input_size, N]
#' @param h_t     Previous hidden state: [units, N]
#' @param W_zh    [input_size, 2*units]
#' @param U_zh    [units, 2*units]
#' @param b_zh    [2*units]
#' @param W_n     [input_size, units]
#' @param U_n     [units, units]
#' @param b_n     [units]
#' @param units   Integer
#' @param act_cell Activation for candidate (tanh)
#' @param act_rec  Activation for z/r gates (sigmoid)
#' @return list(h = new_h)
#' @keywords internal
nn_gru_step <- function(ctx, x_t, h_t, W_zh, U_zh, b_zh,
                         W_n, U_n, b_n, units, act_cell, act_rec) {
  N <- ggml_tensor_shape(x_t)[2]

  # z/r gates combined: [2*units, N]
  zr_raw <- ggml_add(ctx,
    ggml_add(ctx,
      ggml_mul_mat(ctx, W_zh, x_t),
      ggml_mul_mat(ctx, U_zh, h_t)
    ),
    b_zh
  )

  stride <- as.integer(units * 2L * 4L)   # bytes per row (F32)
  z_raw <- ggml_view_2d(ctx, zr_raw, units, N,
                          nb1 = as.integer(units * 2L * 4L), offset = 0L)
  r_raw <- ggml_view_2d(ctx, zr_raw, units, N,
                          nb1 = as.integer(units * 2L * 4L),
                          offset = as.integer(units * 4L))

  z_gate <- nn_apply_activation(ctx, z_raw, act_rec)
  r_gate <- nn_apply_activation(ctx, r_raw, act_rec)

  # Candidate: n = tanh(W_n*x + U_n*(r*h) + b_n)
  r_h <- ggml_mul(ctx, r_gate, h_t)
  n_raw <- ggml_add(ctx,
    ggml_add(ctx,
      ggml_mul_mat(ctx, W_n, x_t),
      ggml_mul_mat(ctx, U_n, r_h)
    ),
    b_n
  )
  n_gate <- nn_apply_activation(ctx, n_raw, act_cell)

  # h' = (1-z)*h + z*n  equivalent to:  h + z*(n - h)
  new_h <- ggml_add(ctx,
    h_t,
    ggml_mul(ctx, z_gate, ggml_sub(ctx, n_gate, h_t))
  )

  list(h = new_h)
}

#' Build LSTM forward pass for Sequential model
#' @keywords internal
nn_build_lstm <- function(ctx, input_tensor, layer, batch_size) {
  units        <- layer$config$units
  ret_seq      <- isTRUE(layer$config$return_sequences)
  act_cell     <- layer$config$activation
  act_rec      <- layer$config$recurrent_activation
  W_gates      <- layer$weights$W_gates
  U_gates      <- layer$weights$U_gates
  b_gates_w    <- layer$weights$b_gates

  # input_tensor layout: ggml [input_size, seq_len, N]
  sh       <- ggml_tensor_shape(input_tensor)
  input_sz <- sh[1]; seq_len <- sh[2]; N <- sh[3]

  # Initial states: zeros [units, N]
  # Use ctx_weights tensors (properly allocated + zero-initialised) to avoid
  # uninitialized memory in the compute context (NaN * 0 = NaN under IEEE 754).
  h_shape <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, units, N)
  c_shape <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, units, N)
  if (!is.null(layer$weights$h0)) {
    h_t <- ggml_repeat(ctx, layer$weights$h0, h_shape)
    c_t <- ggml_repeat(ctx, layer$weights$c0, c_shape)
  } else {
    # Fallback: derive zeros from b_gates (always zero-initialised) to avoid
    # uninitialized memory (NaN * 0 = NaN under IEEE 754).
    b_h <- ggml_view_1d(ctx, layer$weights$b_gates, units, 0L)
    b_c <- ggml_view_1d(ctx, layer$weights$b_gates, units, 0L)
    h_t <- ggml_repeat(ctx, b_h, h_shape)
    c_t <- ggml_repeat(ctx, b_c, c_shape)
  }

  h_steps <- vector("list", seq_len)

  for (t in seq_len(seq_len)) {
    # Slice time step t: view [input_sz, N] out of [input_sz, seq_len, N]
    offset_t <- as.integer((t - 1L) * input_sz * 4L)   # F32 bytes
    x_t <- ggml_view_2d(ctx, input_tensor, input_sz, N,
                          nb1 = as.integer(input_sz * seq_len * 4L),
                          offset = offset_t)
    step <- nn_lstm_step(ctx, x_t, h_t, c_t, W_gates, U_gates, b_gates_w,
                          units, act_cell, act_rec)
    h_t <- step$h
    c_t <- step$c
    h_steps[[t]] <- h_t
  }

  if (ret_seq) {
    # Stack all h_steps: [units, seq_len, N]
    out <- h_steps[[1]]
    for (t in seq(2L, seq_len)) {
      out <- ggml_concat(ctx, out, h_steps[[t]], dim = 1L)
    }
    out
  } else {
    h_t  # last hidden state [units, N]
  }
}

#' Build GRU forward pass for Sequential model
#' @keywords internal
nn_build_gru <- function(ctx, input_tensor, layer, batch_size) {
  units     <- layer$config$units
  ret_seq   <- isTRUE(layer$config$return_sequences)
  act_cell  <- layer$config$activation
  act_rec   <- layer$config$recurrent_activation
  W_zh      <- layer$weights$W_zh
  U_zh      <- layer$weights$U_zh
  b_zh_w    <- layer$weights$b_zh
  W_n       <- layer$weights$W_n
  U_n       <- layer$weights$U_n
  b_n_w     <- layer$weights$b_n

  sh       <- ggml_tensor_shape(input_tensor)
  input_sz <- sh[1]; seq_len <- sh[2]; N <- sh[3]

  h_shape <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, units, N)
  if (!is.null(layer$weights$h0)) {
    h_t <- ggml_repeat(ctx, layer$weights$h0, h_shape)
  } else {
    # Fallback: derive zeros from b_zh (always zero-initialised).
    b_h <- ggml_view_1d(ctx, layer$weights$b_zh, units, 0L)
    h_t <- ggml_repeat(ctx, b_h, h_shape)
  }

  h_steps <- vector("list", seq_len)

  for (t in seq_len(seq_len)) {
    offset_t <- as.integer((t - 1L) * input_sz * 4L)
    x_t <- ggml_view_2d(ctx, input_tensor, input_sz, N,
                          nb1 = as.integer(input_sz * seq_len * 4L),
                          offset = offset_t)
    step <- nn_gru_step(ctx, x_t, h_t, W_zh, U_zh, b_zh_w,
                         W_n, U_n, b_n_w, units, act_cell, act_rec)
    h_t <- step$h
    h_steps[[t]] <- h_t
  }

  if (ret_seq) {
    out <- h_steps[[1]]
    for (t in seq(2L, seq_len)) {
      out <- ggml_concat(ctx, out, h_steps[[t]], dim = 1L)
    }
    out
  } else {
    h_t
  }
}
