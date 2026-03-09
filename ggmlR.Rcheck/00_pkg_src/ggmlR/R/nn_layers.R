# High-level neural network layer definitions for ggmlR
# Provides Keras-like layer API with pipe (%>%) support

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
                                padding = "valid") {
  if (length(kernel_size) == 1) kernel_size <- rep(as.integer(kernel_size), 2)
  if (length(strides) == 1) strides <- rep(as.integer(strides), 2)

  layer <- list(
    type = "conv_2d",
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

#' Add 2D Max Pooling Layer
#'
#' @param model A ggml_sequential_model object
#' @param pool_size Integer or vector of 2 integers for pool height and width
#' @param strides Integer or vector of 2 integers (defaults to pool_size)
#' @return The model object with the max pooling layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_max_pooling_2d(c(2, 2))
#' }
ggml_layer_max_pooling_2d <- function(model, pool_size = c(2L, 2L), strides = NULL) {
  if (length(pool_size) == 1) pool_size <- rep(as.integer(pool_size), 2)
  if (is.null(strides)) strides <- pool_size
  if (length(strides) == 1) strides <- rep(as.integer(strides), 2)

  layer <- list(
    type = "max_pooling_2d",
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
#' @return The model object with the flatten layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_conv_2d(32, c(3,3), activation = "relu",
#'                      input_shape = c(28, 28, 1)) |>
#'   ggml_layer_flatten()
#' }
ggml_layer_flatten <- function(model) {
  layer <- list(
    type = "flatten",
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
                                padding = "valid") {
  kernel_size <- as.integer(kernel_size)
  strides <- as.integer(strides)

  layer <- list(
    type = "conv_1d",
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
#' @return The model object with the batch_norm layer appended (invisibly).
#' @export
#' @examples
#' \donttest{
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(128, input_shape = 784) |>
#'   ggml_layer_batch_norm() |>
#'   ggml_layer_dense(10, activation = "softmax")
#' }
ggml_layer_batch_norm <- function(model, eps = 1e-5) {
  layer <- list(
    type = "batch_norm",
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
ggml_layer_dense <- function(model, units, activation = NULL, input_shape = NULL) {
  layer <- list(
    type = "dense",
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
      "flatten" = {
        as.integer(prod(current_shape))
      },
      "dense" = {
        as.integer(layer$config$units)
      },
      "batch_norm" = {
        current_shape  # batch_norm doesn't change shape
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
    "relu" = ggml_relu(ctx, tensor),
    "sigmoid" = ggml_sigmoid(ctx, tensor),
    "tanh" = ggml_tanh(ctx, tensor),
    "softmax" = ggml_soft_max(ctx, tensor),
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

#' Build a layer's forward pass
#' @keywords internal
nn_build_layer <- function(ctx, input_tensor, layer) {
  switch(layer$type,
    "conv_1d" = nn_build_conv_1d(ctx, input_tensor, layer),
    "conv_2d" = nn_build_conv_2d(ctx, input_tensor, layer),
    "max_pooling_2d" = nn_build_max_pooling_2d(ctx, input_tensor, layer),
    "flatten" = nn_build_flatten(ctx, input_tensor, layer),
    "dense" = nn_build_dense(ctx, input_tensor, layer),
    "batch_norm" = nn_build_batch_norm(ctx, input_tensor, layer),
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
