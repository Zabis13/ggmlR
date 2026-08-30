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

#' Add Permute Layer
#'
#' Reorder a sample's axes. The batch axis is not part of \code{dims} and is
#' never moved -- \code{dims} indexes the per-sample shape, exactly as
#' \code{input_shape} does.
#'
#' @section Argument order:
#' \code{dims} follows base R's \code{\link[base]{aperm}}: \code{dims[i]} names
#' the \emph{source} axis that ends up at position \code{i}. So
#' \code{dims = c(2, 1)} means "the new first axis is the old second one".
#'
#' This is deliberately the inverse of the underlying
#' \code{\link{ggml_permute}}, whose arguments are \emph{destination}
#' positions; the layer converts between the two so that callers can think in
#' the R idiom. Read \code{ggml_permute} itself as "source axis \code{i} goes
#' to position \code{p_i}" -- mixing the two conventions up silently inverts
#' every non-trivial permutation.
#'
#' @section Typical use:
#' Reordering a sample's axes to match what the next layer reads: a
#' \code{c(features, seq_len)} tensor into the \code{c(seq_len, features)} that
#' attention and the recurrent layers expect, or the reverse.
#'
#' Note that \code{\link{ggml_layer_embedding}} does \strong{not} need this --
#' it already reports \code{c(seq_len, dim)} and feeds attention, GRU and LSTM
#' directly. Adding a permute there transposes the underlying tensor and breaks
#' the graph.
#'
#' @param model A ggml_sequential_model object, or a \code{ggml_tensor_node}
#'   for the functional API.
#' @param dims Integer vector, a permutation of \code{seq_along(input_shape)}.
#'   Must not include the batch axis.
#' @param input_shape Input shape, required for the first layer only.
#' @param name Optional character name for the layer.
#' @param trainable Logical; reserved for API consistency (no weights).
#' @return The model object with the permute layer appended (invisibly), or a
#'   \code{ggml_tensor_node} in the functional API.
#' @seealso \code{\link{ggml_permute}} for the low-level op and its
#'   destination-position convention.
#' @export
#' @examples
#' \donttest{
#' # Swap a c(features, seq_len) sample into c(seq_len, features).
#' model <- ggml_model_sequential() |>
#'   ggml_layer_permute(c(2, 1), input_shape = c(32, 16))
#' }
ggml_layer_permute <- function(model, dims, input_shape = NULL, name = NULL,
                                trainable = TRUE) {
  dims <- nn_check_permute_dims(dims)

  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("permute_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "permute",
      trainable = trainable,
      config    = list(dims = dims, name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "permute")

  layer <- list(
    type = "permute",
    name = name,
    trainable = trainable,
    config = list(dims = dims),
    input_shape = input_shape,
    output_shape = NULL,
    weights = list()
  )

  model$layers <- c(model$layers, list(layer))

  if (!is.null(input_shape) && is.null(model$input_shape)) {
    model$input_shape <- as.integer(input_shape)
  }

  model
}

#' Validate a permute layer's `dims`
#'
#' Checked at layer-construction time rather than at build time so a typo
#' surfaces where it was written, not several layers later.
#'
#' @param dims Candidate permutation.
#' @return `dims` as an integer vector.
#' @keywords internal
nn_check_permute_dims <- function(dims) {
  if (!is.numeric(dims) || anyNA(dims)) {
    stop("'dims' must be a numeric vector free of NA.", call. = FALSE)
  }
  dims <- as.integer(dims)
  if (!setequal(dims, seq_along(dims))) {
    stop("'dims' must be a permutation of 1:", length(dims),
         "; got c(", paste(dims, collapse = ", "), ").", call. = FALSE)
  }
  dims
}

#' Apply a permute layer's `dims` to a per-sample shape
#'
#' Shared by the sequential and functional shape passes so the two cannot
#' disagree.
#'
#' @param dims Permutation in `aperm` (source) order.
#' @param shape Per-sample input shape.
#' @return The permuted per-sample shape.
#' @keywords internal
nn_permute_output_shape <- function(dims, shape) {
  if (length(dims) != length(shape)) {
    stop("'dims' has ", length(dims), " entry/entries but the layer input is ",
         length(shape), "-dimensional; they must agree.", call. = FALSE)
  }
  as.integer(shape[dims])
}

#' Add Reshape Layer
#'
#' Reinterpret a sample's shape without moving any data. The batch axis is not
#' part of \code{shape} and is never reshaped -- \code{shape} describes one
#' sample, exactly as \code{input_shape} does.
#'
#' @section Inferred axis:
#' One entry of \code{shape} may be \code{-1}, in which case it is computed
#' from the element count: \code{ggml_layer_reshape(c(-1, 32))} on a
#' \code{c(4, 8, 16)} input gives \code{c(16, 32)}. At most one axis may be
#' \code{-1}, and the remaining sizes must divide the element count exactly.
#'
#' @section Reshape versus permute:
#' Reshape only relabels the axes; the elements keep their memory order. To
#' actually reorder axes -- swapping \code{c(a, b)} into \code{c(b, a)}, say --
#' use \code{\link{ggml_layer_permute}}. Using reshape for that silently
#' interleaves the data instead.
#'
#' @section Non-contiguous input:
#' \code{ggml_reshape_*()} requires contiguous data, and a permuted tensor is
#' not contiguous. The layer inserts \code{\link{ggml_cont}} when needed, so it
#' works after a permute; the cost is one copy.
#'
#' @param model A ggml_sequential_model object, or a \code{ggml_tensor_node}
#'   for the functional API.
#' @param shape Integer vector, the new per-sample shape. At most one entry may
#'   be \code{-1} to have it inferred. Must not include the batch axis.
#' @param input_shape Input shape, required for the first layer only.
#' @param name Optional character name for the layer.
#' @param trainable Logical; reserved for API consistency (no weights).
#' @return The model object with the reshape layer appended (invisibly), or a
#'   \code{ggml_tensor_node} in the functional API.
#' @seealso \code{\link{ggml_layer_permute}} to reorder axes,
#'   \code{\link{ggml_layer_flatten}} to collapse them all.
#' @export
#' @examples
#' \donttest{
#' # Split a flat feature vector into a sequence of 32-wide steps.
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(512, input_shape = 128) |>
#'   ggml_layer_reshape(c(-1, 32))
#' }
ggml_layer_reshape <- function(model, shape, input_shape = NULL, name = NULL,
                                trainable = TRUE) {
  shape <- nn_check_reshape_shape(shape)

  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("reshape_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "reshape",
      trainable = trainable,
      config    = list(shape = shape, name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "reshape")

  layer <- list(
    type = "reshape",
    name = name,
    trainable = trainable,
    config = list(shape = shape),
    input_shape = input_shape,
    output_shape = NULL,
    weights = list()
  )

  model$layers <- c(model$layers, list(layer))

  if (!is.null(input_shape) && is.null(model$input_shape)) {
    model$input_shape <- as.integer(input_shape)
  }

  model
}

#' Validate a reshape layer's target `shape`
#'
#' Checked at layer-construction time so a typo surfaces where it was written.
#' The element count cannot be checked yet -- the input shape is unknown until
#' the shape pass -- so that half lives in [nn_reshape_output_shape()].
#'
#' @param shape Candidate per-sample shape.
#' @return `shape` as an integer vector.
#' @keywords internal
nn_check_reshape_shape <- function(shape) {
  if (!is.numeric(shape) || anyNA(shape) || length(shape) == 0L) {
    stop("'shape' must be a non-empty numeric vector free of NA.", call. = FALSE)
  }
  shape <- as.integer(shape)
  if (sum(shape == -1L) > 1L) {
    stop("'shape' may contain at most one -1; got ", sum(shape == -1L), ".",
         call. = FALSE)
  }
  if (any(shape < 1L & shape != -1L)) {
    stop("'shape' entries must be positive, or -1 to be inferred; got c(",
         paste(shape, collapse = ", "), ").", call. = FALSE)
  }
  shape
}

#' Resolve a reshape layer's target shape against its input
#'
#' Fills in a `-1` axis and checks that the element count is preserved. Shared
#' by the sequential and functional shape passes so the two cannot disagree.
#'
#' This check is not cosmetic: `ggml_reshape_*()` enforces the element count
#' with a `GGML_ASSERT`, which aborts the process rather than raising an R
#' error, so a bad shape would take the whole session down.
#'
#' @param shape Target per-sample shape, possibly containing one `-1`.
#' @param input_shape Per-sample input shape.
#' @return The resolved per-sample shape, with no `-1` left.
#' @keywords internal
nn_reshape_output_shape <- function(shape, input_shape) {
  n_in <- prod(as.numeric(input_shape))

  if (any(shape == -1L)) {
    known <- prod(as.numeric(shape[shape != -1L]))
    if (known <= 0 || n_in %% known != 0) {
      stop("cannot infer the -1 axis: an input of ", format(n_in),
           " element(s) is not divisible by ", format(known), ".",
           call. = FALSE)
    }
    shape[shape == -1L] <- as.integer(n_in / known)
  }

  n_out <- prod(as.numeric(shape))
  if (n_out != n_in) {
    stop("'shape' c(", paste(shape, collapse = ", "), ") has ", format(n_out),
         " element(s) but the layer input has ", format(n_in),
         "; a reshape must preserve the element count.", call. = FALSE)
  }

  as.integer(shape)
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

#' Add RMS Normalization Layer
#'
#' Normalizes each sample by the root mean square of its features, then scales
#' by gamma and shifts by beta (both learnable). Unlike
#' \code{\link{ggml_layer_batch_norm}} it does not subtract the mean and keeps
#' no running statistics, so it behaves identically in training and inference
#' and does not couple the samples in a batch. This is the normalization used
#' by transformer blocks.
#'
#' @param model A \code{ggml_sequential_model} or a \code{ggml_tensor_node}
#'   (functional API).
#' @param eps Small constant for numerical stability (default 1e-5)
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the layer weights are updated during training.
#' @return The model with the layer appended, or a new \code{ggml_tensor_node}
#'   in the functional API.
#' @seealso \code{\link{ggml_layer_batch_norm}}
#' @export
#' @examples
#' \donttest{
#' # Functional API: the normalization of a transformer block.
#' x <- ggml_input(shape = c(10L, 32L))
#' h <- x |> ggml_layer_rms_norm()
#' h <- h |> ggml_layer_attention(d_model = 32L, n_heads = 4L)
#' }
ggml_layer_rms_norm <- function(model, eps = 1e-5, name = NULL,
                                trainable = TRUE) {
  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("rms_norm_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "rms_norm",
      trainable = trainable,
      config    = list(eps = eps, name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "rms_norm")

  layer <- list(
    type = "rms_norm",
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

#' Add Layer Normalization Layer
#'
#' Normalizes each sample over its own features -- subtract the mean, divide by
#' the standard deviation -- then scales by gamma and shifts by beta (both
#' learnable). Like \code{\link{ggml_layer_rms_norm}} it keeps no running
#' statistics and does not couple the samples in a batch, but unlike it, it does
#' centre the input. This is the normalization of the original transformer.
#'
#' @param model A \code{ggml_sequential_model} or a \code{ggml_tensor_node}
#'   (functional API).
#' @param eps Small constant for numerical stability (default 1e-5)
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the layer weights are updated during training.
#' @return The model with the layer appended, or a new \code{ggml_tensor_node}
#'   in the functional API.
#' @seealso \code{\link{ggml_layer_rms_norm}}, \code{\link{ggml_layer_batch_norm}}
#' @export
#' @examples
#' \donttest{
#' x <- ggml_input(shape = c(10L, 32L))
#' h <- x |> ggml_layer_layer_norm()
#' h <- h |> ggml_layer_attention(d_model = 32L, n_heads = 4L)
#' }
ggml_layer_layer_norm <- function(model, eps = 1e-5, name = NULL,
                                  trainable = TRUE) {
  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("layer_norm_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "layer_norm",
      trainable = trainable,
      config    = list(eps = eps, name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "layer_norm")

  layer <- list(
    type = "layer_norm",
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

#' Add a Learned Positional Embedding
#'
#' Adds a learned vector to each position of a sequence, so that attention --
#' which is order-blind on its own -- can tell one position from another. The
#' table is \code{[d_model, seq_len]}, one row per position, added to the input
#' and broadcast over the batch.
#'
#' Unlike \code{\link{ggml_layer_embedding}} there are no indices to look up:
#' the position IS the place in the sequence, so this layer takes the sequence
#' itself and returns it with the positional term added. The shape is
#' unchanged.
#'
#' @param model A \code{ggml_sequential_model} or a \code{ggml_tensor_node}
#'   carrying a sequence of shape \code{c(seq_len, d_model)}.
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the table is updated during training.
#' @return The model with the layer appended, or a new \code{ggml_tensor_node}
#'   of the same shape as the input.
#' @seealso \code{\link{ggml_layer_attention}}, \code{\link{ggml_layer_embedding}}
#' @export
#' @examples
#' \donttest{
#' # Without this the encoder cannot distinguish "ab" from "ba".
#' x <- ggml_input(shape = c(10L, 32L))
#' h <- x |> ggml_layer_positional_embedding()
#' h <- h |> ggml_layer_attention(d_model = 32L, n_heads = 4L)
#' }
ggml_layer_positional_embedding <- function(model, name = NULL,
                                            trainable = TRUE) {
  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("pos_embed_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "positional_embedding",
      trainable = trainable,
      config    = list(name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "positional_embedding")

  layer <- list(
    type = "positional_embedding",
    name = name,
    trainable = trainable,
    config = list(),
    input_shape = NULL,
    output_shape = NULL,
    weights = list(pos = NULL)
  )

  model$layers <- c(model$layers, list(layer))
  model
}

#' Pool a Sequence Down to One Vector
#'
#' Collapses the sequence axis, turning \code{c(seq_len, d_model)} into
#' \code{d_model}: the step that lets a classification or regression head sit
#' on top of an encoder. \code{ggml_layer_flatten()} also produces a flat
#' vector, but keeps every position separately (\code{seq_len * d_model}
#' features), so its width -- and the head above it -- depends on the sequence
#' length. Pooling does not.
#'
#' \code{mode = "mean"} averages over the positions; every position
#' contributes equally, which is the usual choice for an encoder without a
#' dedicated summary token. \code{mode = "first"} takes position 1 and ignores
#' the rest -- the CLS-token convention, where attention is expected to have
#' gathered what matters into that position.
#'
#' @param model A \code{ggml_sequential_model} or a \code{ggml_tensor_node}
#'   carrying a sequence of shape \code{c(seq_len, d_model)}.
#' @param mode \code{"mean"} (default) or \code{"first"}.
#' @param name Optional character name for the layer.
#' @return The model with the layer appended, or a new \code{ggml_tensor_node}
#'   of shape \code{d_model}.
#' @seealso \code{\link{ggml_layer_flatten}}, \code{\link{ggml_layer_attention}}
#' @export
#' @examples
#' \donttest{
#' # An encoder with a regression head: the head's width is d_model, whatever
#' # the sequence length turns out to be.
#' x <- ggml_input(shape = c(10L, 32L))
#' h <- x |> ggml_layer_attention(d_model = 32L, n_heads = 4L)
#' h <- h |> ggml_layer_sequence_pooling()
#' y <- h |> ggml_layer_dense(1L)
#' }
ggml_layer_sequence_pooling <- function(model, mode = c("mean", "first"),
                                        name = NULL) {
  mode <- match.arg(mode)

  # Functional API
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("seq_pool_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "sequence_pooling",
      trainable = FALSE,
      config    = list(mode = mode, name = name),
      parents   = list(node)
    ), class = "ggml_tensor_node"))
  }

  if (is.null(name)) name <- nn_layer_name(model, "sequence_pooling")

  layer <- list(
    type = "sequence_pooling",
    name = name,
    trainable = FALSE,
    config = list(mode = mode),
    input_shape = NULL,
    output_shape = NULL,
    weights = list()
  )

  model$layers <- c(model$layers, list(layer))
  model
}

#' Add Dense (Fully Connected) Layer
#'
#' @section Time-distributed application:
#' By default a dense layer flattens its input, so a sequence node of shape
#' \code{c(seq_len, features)} becomes a single vector of \code{units} and the
#' sequence axis is gone.  With \code{time_distributed = TRUE} the same kernel
#' is applied independently at every position instead, sharing one set of
#' weights across the sequence, and the output keeps its length:
#' \code{c(seq_len, units)}.  This is the semantics of Keras'
#' \code{TimeDistributed}, and it is what the position-wise feed-forward
#' sublayer of a transformer block needs -- without it, a dense layer after
#' \code{\link{ggml_layer_attention}} would collapse every position into one
#' vector.
#'
#' It is computed as a single batched \code{ggml_mul_mat()} over the combined
#' position/batch axes, not as a loop over positions.
#'
#' Functional API only: a sequential model carries a flat running shape, so
#' there is no sequence axis to distribute over.
#'
#' @param model A ggml_sequential_model object
#' @param units Number of output units
#' @param activation Activation function name: "relu", "sigmoid", "tanh", "softmax", or NULL
#' @param input_shape Integer or integer vector specifying the input shape (only needed for the first layer)
#' @param name Optional character name for the layer.
#' @param trainable Logical; whether the layer weights are updated during training.
#' @param time_distributed Logical; apply the kernel per position of a sequence
#'   input rather than flattening it (default \code{FALSE}). See
#'   \emph{Time-distributed application}.
#' @return The model object with the dense layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_flatten() |>
#'   ggml_layer_dense(128, activation = "relu")
#'
#' # Position-wise feed-forward, as in a transformer block.
#' x  <- ggml_input(shape = c(10L, 32L))
#' ff <- x |> ggml_layer_dense(64L, activation = "relu", time_distributed = TRUE)
#' }
ggml_layer_dense <- function(model, units, activation = NULL, input_shape = NULL,
                              name = NULL, trainable = TRUE,
                              time_distributed = FALSE) {
  # Functional API: model is a tensor node
  if (inherits(model, "ggml_tensor_node")) {
    node <- model
    if (is.null(name)) name <- paste0("dense_", node$id)
    return(structure(list(
      id        = nn_next_node_id(),
      node_type = "dense",
      trainable = trainable,
      config    = list(
        units            = as.integer(units),
        activation       = activation,
        time_distributed = isTRUE(time_distributed),
        name             = name
      ),
      parents = list(node)
    ), class = "ggml_tensor_node"))
  }
  # Sequential models carry a running flat shape, so there is no sequence axis
  # for a time-distributed kernel to be applied along.
  if (isTRUE(time_distributed)) {
    stop("'time_distributed' applies to the functional API, where a layer is ",
         "applied to a sequence node; a sequential model has no sequence axis ",
         "to distribute over.", call. = FALSE)
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
#' @section Axis order:
#' The layer's output shape is \code{c(seq_len, dim)} -- one row per position,
#' as in Keras -- which is the same convention a declared sequence input
#' (\code{ggml_input(shape = c(seq_len, features))}) uses. So the output feeds
#' \code{\link{ggml_layer_attention}}, \code{\link{ggml_layer_gru}} and
#' \code{\link{ggml_layer_lstm}} directly, with no axis juggling in between.
#'
#' The underlying tensor is \code{[dim, seq_len, N]}: ggml is column-major, so
#' an R shape \code{c(a, b)} is the tensor \code{[b, a, N]} throughout the
#' package. That matters only when reading raw output tensors --
#' \code{ggml_layer_flatten()} gives the same vector either way.
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
      "permute" = {
        nn_permute_output_shape(layer$config$dims, current_shape)
      },
      "reshape" = {
        nn_reshape_output_shape(layer$config$shape, current_shape)
      },
      "dense" = {
        as.integer(layer$config$units)
      },
      "batch_norm" = {
        current_shape  # batch_norm doesn't change shape
      },
      "rms_norm" = {
        current_shape  # rms_norm doesn't change shape
      },
      "layer_norm" = {
        current_shape  # layer_norm doesn't change shape
      },
      "dropout" = {
        current_shape  # dropout doesn't change shape
      },
      "embedding" = {
        # input shape: c(seq_len) -> output: c(seq_len, dim).
        # Matches the functional branch and the package's R-to-ggml rule
        # (R c(a, b) is the tensor [b, a, N]); the build emits [dim, seq_len, N].
        seq_len <- if (length(current_shape) == 1L) current_shape else prod(current_shape)
        as.integer(c(seq_len, layer$config$dim))
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
#' @return A \code{ggml_tensor} with the activation applied, or \code{tensor} unchanged when \code{activation} is \code{NULL}.
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
#'
#' The sequential API lays a 3-D input out as \code{[size, seq, N]} -- the
#' channel on \code{ne[0]} -- which is what LSTM/GRU and batch_norm expect.
#' The convolution wants the opposite: data as \code{[L, IC, N]} against a
#' \code{[K, IC, OC]} kernel. So the input is transposed on the way in and the
#' result transposed back, keeping the layer's contract with its neighbours
#' unchanged.
#'
#' The convolution is assembled here rather than via \code{ggml_conv_1d()},
#' which hard-codes an F16 im2col and therefore needs an F16 kernel. Casting the
#' kernel makes the forward pass work but breaks training: the gradient w.r.t.
#' the kernel is an \code{OUT_PROD} whose first operand is then F16, and the CPU
#' backend only implements \code{OUT_PROD} for F32 and quantized inputs, so the
#' scheduler finds no backend for that node. Building the same im2col +
#' \code{mul_mat} in F32 -- which is what \code{ggml_conv_2d()} does, passing the
#' kernel's own type -- keeps the whole path differentiable.
#'
#' @return A \code{ggml_tensor} \code{[OC, OL, N]} holding the 1-D convolution
#'   output (with bias and activation applied).
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

  # [IC, L, N] -> [L, IC, N]
  data_t <- ggml_cont(ctx, ggml_permute(ctx, input_tensor, 1L, 0L, 2L, 3L))

  k_shape <- ggml_tensor_shape(kernel)          # [K, IC, OC]
  kk <- as.integer(k_shape[1]); kic <- as.integer(k_shape[2])
  koc <- as.integer(k_shape[3])

  im <- ggml_im2col(ctx, kernel, data_t, s0 = s0, s1 = 0L, p0 = p0, p1 = 0L,
                    d0 = 1L, d1 = 0L, is_2D = FALSE, dst_type = GGML_TYPE_F32)
  im_shape <- ggml_tensor_shape(im)             # [IC*K, OL, N]

  ol <- as.integer(im_shape[2])
  nb <- as.integer(im_shape[3])

  out <- ggml_mul_mat(
    ctx,
    ggml_reshape_2d(ctx, im, as.integer(im_shape[1]), nb * ol),
    ggml_reshape_2d(ctx, kernel, kk * kic, koc)
  )
  # mul_mat returns [OL*N, OC] -- the whole batch for one filter, then the next.
  # Splitting that straight into [OL, OC, N] (what ggml_conv_1d() does) only
  # holds for N == 1; past that it interleaves filters and samples. Split along
  # the axes the data actually has, then move OC into place.
  out <- ggml_reshape_3d(ctx, out, ol, nb, koc)            # [OL, N, OC]
  out <- ggml_cont(ctx, ggml_permute(ctx, out, 0L, 2L, 1L, 3L))  # [OL, OC, N]

  # [OL, OC, N] -> [OC, OL, N], back to the layout the next layer expects.
  #
  # The bias is added after this transpose, not before. Adding it first makes the
  # bias gradient a ggml_repeat_back() of the permute's output, and that kernel
  # requires a contiguous first operand (nb00 == sizeof(float)) which a permuted
  # view never is -- training then aborts. It only survived with an activation
  # in between, because the activation's backward materializes the gradient.
  # Applied here the bias sees a contiguous tensor either way.
  out <- ggml_cont(ctx, ggml_permute(ctx, out, 1L, 0L, 2L, 3L))

  # Reshape bias [OC] -> [OC, 1, 1] for broadcasting with [OC, OL, N]
  bias_3d <- ggml_reshape_3d(ctx, bias, layer$config$filters, 1L, 1L)
  out <- ggml_add(ctx, out, bias_3d)
  nn_apply_activation(ctx, out, layer$config$activation)
}

#' Build conv_2d forward pass
#' @return A \code{ggml_tensor} holding the 2-D convolution output (with bias and activation applied).
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
#' @return A \code{ggml_tensor} holding the 2-D max-pooled output.
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
#' @return A \code{ggml_tensor} of shape \code{[C, N]} with channel-wise maxima.
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
#' @return A \code{ggml_tensor} of shape \code{[C, N]} with channel-wise means.
#' @keywords internal
nn_build_global_average_pooling_2d <- function(ctx, input_tensor, layer) {
  sh <- ggml_tensor_shape(input_tensor)
  W  <- sh[1]; H <- sh[2]; C <- sh[3]; N <- sh[4]
  pooled <- ggml_pool_2d(ctx, input_tensor, GGML_OP_POOL_AVG,
                          k0 = W, k1 = H, s0 = W, s1 = H,
                          p0 = 0L, p1 = 0L)
  ggml_reshape_2d(ctx, pooled, C, N)
}

#' Build permute forward pass
#'
#' Shared by the sequential and functional builds.
#'
#' The R per-sample shape maps onto ggml axes in order -- R axis \code{i} is
#' \code{ne[i - 1]} -- with the batch on the axis just past the sample's, so a
#' permutation of the sample's axes leaves the batch where it is.
#'
#' Two conversions happen here, and both are easy to get backwards:
#' \enumerate{
#'   \item 1-based R axes become 0-based ggml axes.
#'   \item \code{dims} arrives in \code{aperm} (source) order, while
#'     \code{ggml_permute()} takes destination positions: the argument at
#'     position \code{i} says where source axis \code{i} lands. Inverting the
#'     permutation is what bridges the two. A self-inverse permutation (a plain
#'     two-axis swap, the common case) is unchanged by this step, which is
#'     precisely why an error here can go unnoticed -- see the test on a 3-D
#'     non-symmetric permutation.
#' }
#'
#' The result is passed through \code{ggml_cont()}: \code{ggml_permute()}
#' returns a non-contiguous view, and downstream ops (matmul, reshape) require
#' contiguous data.
#'
#' @param ctx Compute context.
#' @param input_tensor Input \code{ggml_tensor}.
#' @param dims Permutation in `aperm` (source) order, over the sample's axes.
#' @param input_shape Per-sample input shape, used only to count the axes.
#' @return A contiguous \code{ggml_tensor} with the axes reordered.
#' @keywords internal
nn_build_permute_op <- function(ctx, input_tensor, dims, input_shape) {
  n <- length(input_shape)
  if (length(dims) != n) {
    stop("'dims' has ", length(dims), " entry/entries but the layer input is ",
         n, "-dimensional; they must agree.", call. = FALSE)
  }

  # aperm (source) order -> ggml destination positions.
  dest <- integer(n)
  dest[dims] <- seq_len(n)

  # 0-based, and pad the untouched axes (batch and beyond) with identity:
  # ggml_permute() always takes four.
  axes <- as.integer(c(dest - 1L, seq.int(from = n, to = 3L)))

  out <- ggml_permute(ctx, input_tensor, axes[1], axes[2], axes[3], axes[4])
  ggml_cont(ctx, out)
}

#' Build permute forward pass (sequential API)
#' @return A contiguous \code{ggml_tensor} with the sample's axes reordered.
#' @keywords internal
nn_build_permute <- function(ctx, input_tensor, layer) {
  nn_build_permute_op(ctx, input_tensor, layer$config$dims, layer$input_shape)
}

#' Build reshape forward pass
#'
#' Shared by the sequential and functional builds.
#'
#' The batch is appended as the trailing ggml axis, and is derived from the
#' element count rather than from \code{ggml_n_dims()}: ggml reports a trailing
#' unit dimension as absent, so a batch of 1 would otherwise be read off a real
#' axis (the same reasoning as in [nn_build_flatten()]).
#'
#' \code{ggml_reshape_*()} asserts that its input is contiguous, and an assert
#' aborts the process instead of raising an R error, so a non-contiguous input
#' (a permute that was not followed by \code{ggml_cont()}, say) is made
#' contiguous first.
#'
#' @param ctx Compute context.
#' @param input_tensor Input \code{ggml_tensor}.
#' @param shape Resolved per-sample output shape (no `-1` left).
#' @param input_shape Per-sample input shape.
#' @return A \code{ggml_tensor} viewing the input under the new shape.
#' @keywords internal
nn_build_reshape_op <- function(ctx, input_tensor, shape, input_shape) {
  shape <- nn_reshape_output_shape(shape, input_shape)

  batch_size <- as.integer(ggml_nelements(input_tensor) / prod(as.numeric(shape)))

  if (!ggml_is_contiguous(input_tensor)) {
    input_tensor <- ggml_cont(ctx, input_tensor)
  }

  ne <- c(as.integer(shape), batch_size)
  if (length(ne) > 4L) {
    stop("a reshape layer supports at most 3 per-sample axes (plus the batch); ",
         "got ", length(shape), ".", call. = FALSE)
  }

  switch(length(ne),
    stop("unreachable: reshape needs at least one axis plus the batch"),
    ggml_reshape_2d(ctx, input_tensor, ne[1], ne[2]),
    ggml_reshape_3d(ctx, input_tensor, ne[1], ne[2], ne[3]),
    ggml_reshape_4d(ctx, input_tensor, ne[1], ne[2], ne[3], ne[4])
  )
}

#' Build reshape forward pass (sequential API)
#' @return A \code{ggml_tensor} viewing the input under the new shape.
#' @keywords internal
nn_build_reshape <- function(ctx, input_tensor, layer) {
  nn_build_reshape_op(ctx, input_tensor, layer$config$shape, layer$input_shape)
}

#' Build flatten forward pass
#' @return A 2-D \code{ggml_tensor} of shape \code{[features, batch]}.
#' @keywords internal
nn_build_flatten <- function(ctx, input_tensor, layer) {
  n_features <- prod(layer$input_shape)
  # Derive the batch size from the element count rather than from
  # ggml_n_dims(): ggml reports a trailing unit dimension as absent, so a batch
  # of 1 makes a [W, H, C, 1] input look 3-D and the batch size would be read
  # off the channel axis instead.
  batch_size <- as.integer(ggml_nelements(input_tensor) / n_features)

  ggml_reshape_2d(ctx, input_tensor, n_features, batch_size)
}

#' Build dense forward pass
#' @return A \code{ggml_tensor} with the dense (matmul + bias + activation) output.
#' @keywords internal
nn_build_dense <- function(ctx, input_tensor, layer) {
  W <- layer$weights$weight
  b <- layer$weights$bias

  out <- ggml_mul_mat(ctx, W, input_tensor)
  out <- ggml_add(ctx, out, b)
  nn_apply_activation(ctx, out, layer$config$activation)
}

#' Normalize a conv-shaped input over the batch and all spatial positions
#'
#' Batch normalization pools its statistics per channel over the batch AND every
#' spatial position. \code{ggml_mean()} only reduces \code{ne[0]}, so the input is
#' first rearranged to put all reduced axes into \code{ne[0]} and the channel into
#' \code{ne[1]}. Both layouts fold adjacent axes, which is a pure view; only the
#' one \code{permute} in the 4-D route costs a copy.
#'
#' \describe{
#'   \item{3-D \code{[C, L, N]}}{channel is \code{ne[0]}. \code{L} and \code{N} are
#'     adjacent, so \code{reshape} to \code{[C, L*N]} then transpose to
#'     \code{[L*N, C]}.}
#'   \item{4-D \code{[W, H, C, N]}}{channel is \code{ne[2]}. \code{W} and \code{H}
#'     are adjacent (\code{reshape} to \code{[W*H, C, N]}), then \code{permute}
#'     brings \code{N} beside \code{W*H} (\code{[W*H, N, C]}) so a second
#'     \code{reshape} folds both into \code{[W*H*N, C]}.}
#' }
#'
#' At inference time the stored running estimates are used instead, broadcast
#' along the channel axis, so a prediction does not depend on batch composition.
#'
#' @param training Logical; use batch statistics (TRUE) or running estimates (FALSE).
#' @return A \code{ggml_tensor} with the same shape as \code{input_tensor}.
#' @keywords internal
nn_bn_normalize_conv <- function(ctx, input_tensor, layer, training, eps) {
  input_shape <- layer$input_shape
  shape <- ggml_tensor_shape(input_tensor)

  if (!isTRUE(training)) {
    # Inference: normalize with the stored running estimates. Reshaping them onto
    # the channel axis lets ggml_repeat broadcast over the spatial and batch axes.
    rm <- layer$weights$running_mean
    rv <- layer$weights$running_var
    if (is.null(rm) || is.null(rv)) {
      # No running estimates (e.g. an untrained or legacy model): fall back to
      # the current batch's statistics rather than a wrong constant.
      return(nn_bn_normalize_conv(ctx, input_tensor, layer, TRUE, eps))
    }
    n_ch <- as.integer(if (length(input_shape) == 3) input_shape[3] else input_shape[2])
    if (length(input_shape) == 3) {
      rm_b <- ggml_reshape_4d(ctx, rm, 1L, 1L, n_ch, 1L)
      rv_b <- ggml_reshape_4d(ctx, rv, 1L, 1L, n_ch, 1L)
    } else {
      rm_b <- ggml_reshape_3d(ctx, rm, n_ch, 1L, 1L)
      rv_b <- ggml_reshape_3d(ctx, rv, n_ch, 1L, 1L)
    }
    centred <- ggml_sub(ctx, input_tensor,
                        ggml_cont(ctx, ggml_repeat(ctx, rm_b, input_tensor)))
    denom <- ggml_sqrt(ctx, ggml_scale_bias(ctx, rv_b, 1.0, eps))
    return(ggml_div(ctx, centred,
                    ggml_cont(ctx, ggml_repeat(ctx, denom, input_tensor))))
  }

  # --- Training: statistics from the current batch -------------------------
  if (length(input_shape) == 2) {
    # [C, L, N] -> [C, L*N] -> [L*N, C]: the channel lands on ne[1].
    n_ch <- as.integer(shape[1])
    flat <- ggml_reshape_2d(ctx, input_tensor, n_ch, as.integer(shape[2] * shape[3]))
    work <- ggml_cont(ctx, ggml_transpose(ctx, flat))
  } else {
    # [W, H, C, N] -> [W*H, C, N] -> [W*H, N, C] -> [W*H*N, C].
    wh <- as.integer(shape[1] * shape[2])
    n_ch <- as.integer(shape[3])
    nb <- as.integer(shape[4])
    folded <- ggml_reshape_3d(ctx, input_tensor, wh, n_ch, nb)
    swapped <- ggml_cont(ctx, ggml_permute(ctx, folded, 0L, 2L, 1L, 3L))
    work <- ggml_reshape_2d(ctx, swapped, wh * nb, n_ch)
  }

  mu <- ggml_mean(ctx, work)                                   # [1, C]
  centred <- ggml_cont(ctx, ggml_sub(ctx, work,
                                     ggml_cont(ctx, ggml_repeat(ctx, mu, work))))
  var <- ggml_mean(ctx, ggml_sqr(ctx, centred))                # [1, C]
  denom <- ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0, eps))
  xhat <- ggml_div(ctx, centred,
                   ggml_cont(ctx, ggml_repeat(ctx, denom, centred)))

  # Undo the rearrangement so the result has the layer's own shape again.
  #
  # The reshape must restore the input's rank, not just its element count. With
  # a batch of 1 the trailing dimension is a unit, and ggml_reshape_3d() on a
  # 4-D layout would leave a tensor whose ggml_n_dims() is 3 -- nn_build_flatten()
  # then reads the batch size off the wrong axis and aborts. Rebuilding at the
  # original rank keeps ne[] identical to what came in.
  if (length(input_shape) == 2) {
    back <- ggml_cont(ctx, ggml_transpose(ctx, xhat))           # [C, L*N]
    ggml_reshape_3d(ctx, back, n_ch, as.integer(shape[2]), as.integer(shape[3]))
  } else {
    unfolded <- ggml_reshape_3d(ctx, ggml_cont(ctx, xhat), wh, nb, n_ch)
    restored <- ggml_cont(ctx, ggml_permute(ctx, unfolded, 0L, 2L, 1L, 3L))
    ggml_reshape_4d(ctx, restored, as.integer(shape[1]), as.integer(shape[2]),
                    n_ch, nb)
  }
}

#' Build batch_norm forward pass
#'
#' Normalizes per channel over the BATCH axis -- and, for conv-shaped inputs, over
#' every spatial position as well -- the defining property of batch normalization.
#' During training the statistics come from the current batch; at inference time
#' the stored running estimates are used instead, so a prediction depends only on
#' the sample itself and not on which other samples share its batch. Mirrors
#' \code{ag_batch_norm()} in the autograd API.
#'
#' For a flat \code{[features, batch]} input the batch axis is \code{ne[1]}.
#' Reductions in ggml run along \code{ne[0]}, hence the transposes: they move the
#' batch axis into reduction position and back. Conv-shaped inputs need a
#' different rearrangement and are handled by \code{nn_bn_normalize_conv()}.
#'
#' @param training Logical; use batch statistics (TRUE) or running estimates (FALSE).
#' @return A \code{ggml_tensor} with batch-normalized, scaled and shifted values.
#' @keywords internal
nn_build_batch_norm <- function(ctx, input_tensor, layer, training = TRUE) {
  gamma <- layer$weights$gamma
  beta <- layer$weights$beta
  eps <- layer$config$eps
  input_shape <- layer$input_shape

  # gamma and beta are 1D [n_features]; reshape so they broadcast along the
  # channel axis when the input carries spatial dimensions. The channel axis
  # differs per input rank -- see nn_bn_channel_axis() for the layout.
  if (length(input_shape) == 3) {
    # Conv2D: ggml [W, H, C, N], channel is ne[2] -> gamma as [1, 1, C, 1]
    gamma_r <- ggml_reshape_4d(ctx, gamma, 1L, 1L, as.integer(input_shape[3]), 1L)
    beta_r <- ggml_reshape_4d(ctx, beta, 1L, 1L, as.integer(input_shape[3]), 1L)
  } else if (length(input_shape) == 2) {
    # Conv1D/LSTM: R [seq, size] -> ggml [size, seq, N]. The channel is `size`,
    # i.e. ne[0], so gamma broadcasts as [C, 1, 1]. (This used to reshape to
    # [1, C, 1], which put the per-channel scale on the sequence axis.)
    gamma_r <- ggml_reshape_3d(ctx, gamma, as.integer(input_shape[2]), 1L, 1L)
    beta_r <- ggml_reshape_3d(ctx, beta, as.integer(input_shape[2]), 1L, 1L)
  } else {
    # [features, N] -> gamma is already [features], broadcast over N
    gamma_r <- gamma
    beta_r <- beta
  }

  if (length(input_shape) > 1) {
    normed <- nn_bn_normalize_conv(ctx, input_tensor, layer, training, eps)
    out <- ggml_mul(ctx, normed, gamma_r)
    return(ggml_add(ctx, out, beta_r))
  }

  if (isTRUE(training)) {
    # [features, batch] -> [batch, features] so ne[0] is the batch axis
    xt <- ggml_cont(ctx, ggml_transpose(ctx, input_tensor))
    mu <- ggml_mean(ctx, xt)                                  # [1, features]
    centred <- ggml_cont(ctx, ggml_sub(ctx, xt, ggml_cont(ctx, ggml_repeat(ctx, mu, xt))))
    var <- ggml_mean(ctx, ggml_sqr(ctx, centred))             # [1, features]
    denom <- ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0, eps))
    xhat <- ggml_div(ctx, centred, ggml_cont(ctx, ggml_repeat(ctx, denom, centred)))
    normed <- ggml_cont(ctx, ggml_transpose(ctx, xhat))       # back to [features, batch]

  } else {
    # Inference: normalize with the stored running estimates.
    rm <- layer$weights$running_mean
    rv <- layer$weights$running_var
    centred <- ggml_sub(ctx, input_tensor, ggml_cont(ctx, ggml_repeat(ctx, rm, input_tensor)))
    denom <- ggml_sqrt(ctx, ggml_scale_bias(ctx, rv, 1.0, eps))
    normed <- ggml_div(ctx, centred, ggml_cont(ctx, ggml_repeat(ctx, denom, input_tensor)))
  }

  out <- ggml_mul(ctx, normed, gamma_r)
  ggml_add(ctx, out, beta_r)
}

#' Is this a normalization layer carrying learnable gamma/beta?
#'
#' The three of them are allocated, restored, counted and saved the same way;
#' only batch_norm additionally keeps running statistics.
#'
#' @param type A layer type string.
#' @return \code{TRUE} for the gamma/beta normalization layers.
#' @keywords internal
nn_is_norm_type <- function(type) {
  type %in% c("batch_norm", "rms_norm", "layer_norm")
}

#' Layer-normalize over the feature axis
#'
#' \code{ggml_norm()} normalizes over ne[0] -- the feature axis for every input
#' rank the layers use -- keeping the sequence and batch axes, so one call
#' covers flat and sequence inputs alike. Upstream ggml has no backward rule for
#' it; ggmlR adds \code{GGML_OP_NORM_BACK} (CPU and Vulkan kernels), so a graph
#' using this is trainable.
#'
#' @param ctx A \code{ggml_context}.
#' @param x A \code{ggml_tensor} whose features lie along ne[0].
#' @param eps Small constant added to the variance.
#' @return A \code{ggml_tensor} of the same shape as \code{x}.
#' @keywords internal
nn_layer_norm_core <- function(ctx, x, eps) {
  ggml_norm(ctx, x, eps = eps)
}

#' Reshape 1-D gamma/beta so they broadcast along the feature axis
#'
#' The feature axis moves with the input rank: an image is ggml
#' \code{[W, H, C, N]} (channels at ne[2]), a sequence is \code{[size, seq, N]}
#' (features at ne[0]), and a flat input needs no reshape at all.
#'
#' @param ctx A \code{ggml_context}.
#' @param w A 1-D \code{ggml_tensor} of per-feature weights.
#' @param input_shape The layer's input shape, in R order.
#' @return \code{w}, reshaped to broadcast.
#' @keywords internal
nn_norm_broadcast <- function(ctx, w, input_shape) {
  if (length(input_shape) == 3) {
    ggml_reshape_4d(ctx, w, 1L, 1L, as.integer(input_shape[3]), 1L)
  } else if (length(input_shape) == 2) {
    ggml_reshape_3d(ctx, w, as.integer(input_shape[2]), 1L, 1L)
  } else {
    w
  }
}

#' Build layer normalization forward pass
#' @return A \code{ggml_tensor}.
#' @keywords internal
nn_build_layer_norm <- function(ctx, input_tensor, layer) {
  normed  <- nn_layer_norm_core(ctx, input_tensor, layer$config$eps)
  gamma_r <- nn_norm_broadcast(ctx, layer$weights$gamma, layer$input_shape)
  beta_r  <- nn_norm_broadcast(ctx, layer$weights$beta,  layer$input_shape)
  ggml_add(ctx, ggml_mul(ctx, normed, gamma_r), beta_r)
}

#' Build RMS normalization forward pass
#'
#' RMS-normalize, then scale by gamma and shift by beta. No batch statistics
#' are involved, so training and inference build the same graph.
#' @return A \code{ggml_tensor}.
#' @keywords internal
nn_build_rms_norm <- function(ctx, input_tensor, layer) {
  normed  <- ggml_rms_norm(ctx, input_tensor, eps = layer$config$eps)
  gamma_r <- nn_norm_broadcast(ctx, layer$weights$gamma, layer$input_shape)
  beta_r  <- nn_norm_broadcast(ctx, layer$weights$beta,  layer$input_shape)
  ggml_add(ctx, ggml_mul(ctx, normed, gamma_r), beta_r)
}

#' Build dropout forward pass
#' @return A \code{ggml_tensor}: scaled input during training, the input unchanged when \code{training = FALSE}.
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
#' @return A \code{ggml_tensor} with the embedded rows for each input index.
#' @keywords internal
nn_build_embedding <- function(ctx_weights, ctx_compute, input_tensor, layer) {
  vocab_size <- layer$config$vocab_size
  dim        <- layer$config$dim
  E <- layer$weights$weight
  ggml_get_rows(ctx_compute, E, input_tensor)
}

#' Build a layer's forward pass
#' @return A \code{ggml_tensor} produced by the appropriate per-type build helper.
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
    "permute" = nn_build_permute(ctx, input_tensor, layer),
    "reshape" = nn_build_reshape(ctx, input_tensor, layer),
    "dense" = nn_build_dense(ctx, input_tensor, layer),
    "batch_norm" = nn_build_batch_norm(ctx, input_tensor, layer, training),
    "rms_norm" = nn_build_rms_norm(ctx, input_tensor, layer),
    "layer_norm" = nn_build_layer_norm(ctx, input_tensor, layer),
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
#' @return Called for its side effect (writes initial weights into \code{tensor}); invisibly returns \code{NULL}.
#' @importFrom stats runif
#' @keywords internal
nn_init_he_uniform <- function(tensor, fan_in) {
  n <- ggml_nelements(tensor)
  limit <- sqrt(6.0 / fan_in)
  ggml_backend_tensor_set_data(tensor, runif(n, -limit, limit))
}

#' Initialize weight tensor with Glorot uniform distribution
#' @return Called for its side effect (writes initial weights into \code{tensor}); invisibly returns \code{NULL}.
#' @keywords internal
nn_init_glorot_uniform <- function(tensor, fan_in, fan_out) {
  n <- ggml_nelements(tensor)
  limit <- sqrt(6.0 / (fan_in + fan_out))
  ggml_backend_tensor_set_data(tensor, runif(n, -limit, limit))
}

#' Initialize bias tensor to zeros
#' @return Called for its side effect (zero-fills \code{tensor}); invisibly returns \code{NULL}.
#' @keywords internal
nn_init_zeros <- function(tensor) {
  n <- ggml_nelements(tensor)
  ggml_backend_tensor_set_data(tensor, rep(0.0, n))
}

#' Initialize recurrent weight tensor with small deterministic values
#'
#' Uses a fixed zigzag pattern in [-0.01, 0.01] — no RNG, fully reproducible
#' across all platforms and independent of the R random seed state.
#' @return Called for its side effect (fills \code{tensor} with deterministic small values); invisibly returns \code{NULL}.
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

#' Stack a recurrent layer's per-step outputs into a sequence tensor
#'
#' Shared by the LSTM and GRU builds for \code{return_sequences = TRUE}.
#'
#' Each step is \code{[units, N]}, and the result has to be
#' \code{[units, seq_len, N]} -- the layout every sequence consumer expects.
#' Concatenating the steps directly along dim 1 does \strong{not} give that: it
#' produces \code{[units, seq_len * N]}, folding the batch into the sequence
#' axis. Nothing complains at build time, and a consumer that only reads
#' \code{ne[0]} (attention) still works, so the mistake surfaces far away --
#' \code{sequence_pooling} reduces the wrong axis and its
#' \code{ggml_reshape_2d()} aborts the process on the element count.
#'
#' Giving each step an explicit unit sequence axis first (\code{[units, 1, N]})
#' makes the concatenation grow that axis instead, which is the intended
#' result.
#'
#' @param ctx ggml compute context.
#' @param h_steps List of per-step tensors, each \code{[units, N]}.
#' @param units Integer, the hidden width.
#' @param batch_size Integer, N.
#' @return A tensor \code{[units, seq_len, N]}.
#' @keywords internal
nn_stack_time_steps <- function(ctx, h_steps, units, batch_size) {
  units      <- as.integer(units)
  batch_size <- as.integer(batch_size)

  stepped <- lapply(h_steps, function(h) {
    ggml_reshape_3d(ctx, h, units, 1L, batch_size)
  })

  out <- stepped[[1]]
  for (t in seq_along(stepped)[-1L]) {
    out <- ggml_concat(ctx, out, stepped[[t]], dim = 1L)
  }
  out
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
#' @return A \code{ggml_tensor}: last hidden state \code{[units, N]}, or all hidden states \code{[units, seq_len, N]} if \code{return_sequences = TRUE}.
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
    # Stack all h_steps into [units, seq_len, N] -- see nn_stack_time_steps()
    # for why the steps cannot simply be concatenated along dim 1.
    nn_stack_time_steps(ctx, h_steps, units, N)
  } else {
    h_t  # last hidden state [units, N]
  }
}

#' Build GRU forward pass for Sequential model
#' @return A \code{ggml_tensor}: last hidden state \code{[units, N]}, or all hidden states \code{[units, seq_len, N]} if \code{return_sequences = TRUE}.
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
    # [units, seq_len, N] -- see nn_stack_time_steps() for why a plain concat
    # along dim 1 folds the batch into the sequence axis instead.
    nn_stack_time_steps(ctx, h_steps, units, N)
  } else {
    h_t
  }
}
