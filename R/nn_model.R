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
# Compile
# ============================================================================

#' Compile a Sequential Model
#'
#' Configures the model for training: infers shapes, creates backend.
#' Weight tensors are created at training time when batch_size is known.
#'
#' @param model A ggml_sequential_model object
#' @param optimizer Optimizer name: "adam" or "sgd"
#' @param loss Loss function name: "categorical_crossentropy" or "mse"
#' @param metrics Character vector of metrics (currently "accuracy")
#' @param backend Backend to use: "auto" (GPU if available, else CPU), "cpu", or "vulkan"
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
                          backend = "auto") {
  if (length(model$layers) == 0) {
    stop("Model has no layers. Add layers before compiling.")
  }

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
  model$compiled <- TRUE

  invisible(model)
}

# ============================================================================
# Internal: Build graph with weights for a given batch_size
# ============================================================================

#' Build computation graph with allocated weights and inputs
#' @param model Compiled model
#' @param batch_size Batch size
#' @return List with ctx_weights, ctx_compute, inputs, outputs, buffer
#' @keywords internal
nn_build_graph <- function(model, batch_size) {
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

  # Create input tensor in ctx_weights (will be allocated with backend)
  if (length(input_shape) == 3) {
    # Image: R [H, W, C] -> ggml [W, H, C, N]
    inputs <- ggml_new_tensor_4d(ctx_weights, GGML_TYPE_F32,
                                  input_shape[2], input_shape[1],
                                  input_shape[3], batch_size)
  } else if (length(input_shape) == 2) {
    # 1D sequence: R [L, C] -> ggml [L, C, N]
    inputs <- ggml_new_tensor_3d(ctx_weights, GGML_TYPE_F32,
                                  input_shape[1], input_shape[2], batch_size)
  } else {
    # Flat vector: R [features] -> ggml [features, N]
    inputs <- ggml_new_tensor_2d(ctx_weights, GGML_TYPE_F32,
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

    } else if (layer$type == "batch_norm") {
      # Determine number of features for gamma/beta
      n_features <- if (length(layer$input_shape) == 1) layer$input_shape
                    else if (length(layer$input_shape) == 2) layer$input_shape[2]
                    else layer$input_shape[3]

      layer$weights$gamma <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
      layer$weights$beta <- ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_features)
      ggml_set_name(layer$weights$gamma, paste0("bn_", i, "_gamma"))
      ggml_set_name(layer$weights$beta, paste0("bn_", i, "_beta"))
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
      ggml_set_param(layer$weights$kernel)
      ggml_set_param(layer$weights$bias)

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
      ggml_set_param(layer$weights$kernel)
      ggml_set_param(layer$weights$bias)

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
      ggml_set_param(layer$weights$weight)
      ggml_set_param(layer$weights$bias)

    } else if (layer$type == "batch_norm") {
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
      ggml_set_param(layer$weights$gamma)
      ggml_set_param(layer$weights$beta)
    }
  }

  # Create compute context for intermediate tensors (no_alloc, ggml_opt manages)
  compute_mem <- max(64 * 1024 * 1024,
                     ne_datapoint * batch_size * 4 * 20)
  ctx_compute <- ggml_init(compute_mem, no_alloc = TRUE)

  # Build forward graph
  current <- inputs
  for (i in seq_along(layers_built)) {
    current <- nn_build_layer(ctx_compute, current, layers_built[[i]])
  }
  outputs <- current
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

# ============================================================================
# Fit (Training)
# ============================================================================

#' Train a Sequential Model
#'
#' @param model A compiled ggml_sequential_model
#' @param x Training data (array or matrix)
#' @param y Training labels (matrix, one-hot encoded for classification)
#' @param epochs Number of training epochs
#' @param batch_size Batch size
#' @param validation_split Fraction of data for validation (0 to 1)
#' @param verbose 0 = silent, 1 = progress
#' @return The trained model (invisibly).
#' @export
ggml_fit <- function(model, x, y, epochs = 1, batch_size = 32,
                      validation_split = 0.0, verbose = 1) {
  if (!model$compiled) {
    stop("Model must be compiled before training. Call ggml_compile() first.")
  }

  input_shape <- model$input_shape
  n_samples <- if (is.matrix(x)) nrow(x) else dim(x)[1]
  ne_datapoint <- prod(input_shape)
  ne_label <- ncol(y)

  # Ensure batch_size divides data evenly
  usable_samples <- (n_samples %/% batch_size) * batch_size
  if (usable_samples < n_samples) {
    message("Truncating data from ", n_samples, " to ", usable_samples,
            " samples (batch_size=", batch_size, " must divide evenly)")
    if (is.matrix(x)) {
      x <- x[seq_len(usable_samples), , drop = FALSE]
    } else {
      x <- x[seq_len(usable_samples), , , , drop = FALSE]
    }
    y <- y[seq_len(usable_samples), , drop = FALSE]
    n_samples <- usable_samples
  }

  # Prepare dataset
  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = ne_datapoint,
    ne_label = ne_label,
    ndata = n_samples,
    ndata_shard = 1
  )

  # Convert data to ggml format
  if (length(input_shape) == 3) {
    # Image data: R [N, H, W, C] -> ggml [W, H, C, N]
    x_ggml <- as.vector(aperm(x, c(3, 2, 4, 1)))
  } else if (length(input_shape) == 2) {
    # 1D sequence: R [N, L, C] -> ggml [L, C, N]
    x_ggml <- as.vector(aperm(x, c(2, 3, 1)))
  } else if (length(input_shape) == 1) {
    # Vector data: R [N, features] -> ggml [features, N]
    x_ggml <- as.vector(t(x))
  } else {
    stop("Unsupported input_shape length: ", length(input_shape))
  }

  # Labels: R [N, classes] -> ggml [classes, N]
  y_ggml <- as.vector(t(y))

  # Fill dataset
  data_tensor <- ggml_opt_dataset_data(dataset)
  labels_tensor <- ggml_opt_dataset_labels(dataset)
  ggml_backend_tensor_set_data(data_tensor, x_ggml)
  ggml_backend_tensor_set_data(labels_tensor, y_ggml)

  # Build graph (creates contexts, weights, inputs, outputs)
  graph_info <- nn_build_graph(model, batch_size)

  # Map optimizer and loss
  optimizer_type <- switch(model$compilation$optimizer,
    "adam" = , "adamw" = ggml_opt_optimizer_type_adamw(),
    "sgd" = ggml_opt_optimizer_type_sgd(),
    ggml_opt_optimizer_type_adamw()
  )

  loss_type <- switch(model$compilation$loss,
    "categorical_crossentropy" = , "crossentropy" = ggml_opt_loss_type_cross_entropy(),
    "mse" = , "mean_squared_error" = ggml_opt_loss_type_mse(),
    ggml_opt_loss_type_cross_entropy()
  )

  # Train (returns history list from C)
  history_raw <- ggml_opt_fit(
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
    silent = (verbose == 0)
  )

  # Store built layers (with trained weights)
  model$layers <- graph_info$layers_built
  model$compilation$ctx_weights <- graph_info$ctx_weights
  model$compilation$buffer <- graph_info$buffer

  # Build history object
  model$history <- structure(
    list(
      train_loss     = history_raw$train_loss,
      train_accuracy = history_raw$train_accuracy,
      val_loss       = history_raw$val_loss,
      val_accuracy   = history_raw$val_accuracy,
      epochs         = seq_len(epochs)
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
#' @return Named list with loss and accuracy
#' @export
ggml_evaluate <- function(model, x, y, batch_size = 32) {
  if (!model$compiled) {
    stop("Model must be compiled before evaluation.")
  }

  input_shape <- model$input_shape
  n_samples <- if (is.matrix(x)) nrow(x) else dim(x)[1]
  ne_datapoint <- prod(input_shape)
  ne_label <- ncol(y)

  # Truncate to fit batch_size
  usable_samples <- (n_samples %/% batch_size) * batch_size
  if (usable_samples < n_samples) {
    if (is.matrix(x)) {
      x <- x[seq_len(usable_samples), , drop = FALSE]
    } else {
      x <- x[seq_len(usable_samples), , , , drop = FALSE]
    }
    y <- y[seq_len(usable_samples), , drop = FALSE]
    n_samples <- usable_samples
  }

  # Prepare dataset
  dataset <- ggml_opt_dataset_init(
    type_data = GGML_TYPE_F32,
    type_label = GGML_TYPE_F32,
    ne_datapoint = ne_datapoint,
    ne_label = ne_label,
    ndata = n_samples,
    ndata_shard = 1
  )

  # Convert data
  if (length(input_shape) == 3) {
    x_ggml <- as.vector(aperm(x, c(3, 2, 4, 1)))
  } else if (length(input_shape) == 2) {
    x_ggml <- as.vector(aperm(x, c(2, 3, 1)))
  } else {
    x_ggml <- as.vector(t(x))
  }
  y_ggml <- as.vector(t(y))

  data_tensor <- ggml_opt_dataset_data(dataset)
  labels_tensor <- ggml_opt_dataset_labels(dataset)
  ggml_backend_tensor_set_data(data_tensor, x_ggml)
  ggml_backend_tensor_set_data(labels_tensor, y_ggml)

  # Build eval graph (reuses trained weights via rebuild)
  graph_info <- nn_build_graph(model, batch_size)

  # Map loss
  loss_type <- switch(model$compilation$loss,
    "categorical_crossentropy" = , "crossentropy" = ggml_opt_loss_type_cross_entropy(),
    "mse" = , "mean_squared_error" = ggml_opt_loss_type_mse(),
    ggml_opt_loss_type_cross_entropy()
  )

  # Use ggml_opt_init with static graph mode for eval
  opt_ctx <- ggml_opt_init(
    sched = model$compilation$sched,
    loss_type = loss_type,
    optimizer = ggml_opt_optimizer_type_adamw(),
    ctx_compute = graph_info$ctx_compute,
    inputs = graph_info$inputs,
    outputs = graph_info$outputs
  )

  result_eval <- ggml_opt_result_init()
  ggml_opt_epoch(opt_ctx, dataset, NULL, result_eval,
                  idata_split = 0, callback_train = FALSE, callback_eval = FALSE)

  loss_val <- ggml_opt_result_loss(result_eval)
  acc_val <- ggml_opt_result_accuracy(result_eval)

  # Cleanup
  ggml_opt_result_free(result_eval)
  ggml_opt_free(opt_ctx)
  ggml_free(graph_info$ctx_compute)
  ggml_backend_buffer_free(graph_info$buffer)
  ggml_free(graph_info$ctx_weights)
  ggml_opt_dataset_free(dataset)

  list(loss = loss_val[["loss"]], accuracy = acc_val[["accuracy"]])
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
#' @return Matrix of predictions with shape \code{[N, output_units]}
#' @export
ggml_predict <- function(model, x, batch_size = 32L) {
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

  # Truncate to fit batch_size
  usable_samples <- (n_samples %/% batch_size) * batch_size
  if (usable_samples == 0) {
    stop("Not enough samples (", n_samples, ") for batch_size=", batch_size)
  }
  if (usable_samples < n_samples) {
    message("Truncating data from ", n_samples, " to ", usable_samples,
            " samples (batch_size=", batch_size, " must divide evenly)")
    if (is.matrix(x)) {
      x <- x[seq_len(usable_samples), , drop = FALSE]
    } else {
      x <- x[seq_len(usable_samples), , , , drop = FALSE]
    }
    n_samples <- usable_samples
  }

  # Convert all data to ggml format upfront
  if (length(input_shape) == 3) {
    x_ggml <- as.vector(aperm(x, c(3, 2, 4, 1)))
  } else if (length(input_shape) == 2) {
    x_ggml <- as.vector(aperm(x, c(2, 3, 1)))
  } else {
    x_ggml <- as.vector(t(x))
  }

  # Build graph with trained weights
  graph_info <- nn_build_graph(model, batch_size)

  # Build forward computation graph
  graph <- ggml_build_forward_expand(graph_info$ctx_compute, graph_info$outputs)

  # Batch-by-batch forward pass using scheduler directly (no ggml_opt needed)
  sched <- model$compilation$sched
  n_batches <- n_samples %/% batch_size
  all_preds <- matrix(0, nrow = n_samples, ncol = ne_output)

  for (ib in seq_len(n_batches)) {
    # Extract batch data and write to input tensor
    data_start <- (ib - 1L) * batch_size * ne_datapoint + 1L
    data_end <- ib * batch_size * ne_datapoint
    batch_data <- x_ggml[data_start:data_end]
    ggml_backend_tensor_set_data(graph_info$inputs, batch_data)

    # Reset scheduler, allocate graph, compute forward pass
    ggml_backend_sched_reset(sched)
    ggml_backend_sched_alloc_graph(sched, graph)
    ggml_backend_sched_graph_compute(sched, graph)

    # Read output tensor [ne_output, batch_size]
    batch_output <- ggml_backend_tensor_get_data(graph_info$outputs)
    batch_matrix <- matrix(batch_output, nrow = ne_output, ncol = batch_size)
    row_start <- (ib - 1L) * batch_size + 1L
    row_end <- ib * batch_size
    all_preds[row_start:row_end, ] <- t(batch_matrix)
  }

  # Cleanup
  ggml_free(graph_info$ctx_compute)
  ggml_backend_buffer_free(graph_info$buffer)
  ggml_free(graph_info$ctx_weights)

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
    } else if (layer$type == "batch_norm") {
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
  } else if (layer$type == "batch_norm") {
    if (!is.null(layer$input_shape)) {
      n <- if (length(layer$input_shape) == 1) layer$input_shape
           else if (length(layer$input_shape) == 2) layer$input_shape[2]
           else layer$input_shape[3]
      n * 2L  # gamma + beta
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
