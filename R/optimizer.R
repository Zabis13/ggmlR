# Optimization functions for training and fine-tuning
# Wraps ggml-opt API

# ============================================================================
# Loss Type Constants
# ============================================================================

#' Loss type: Mean
#'
#' Returns the constant for mean loss type.
#' Custom loss - reduces outputs to mean value.
#'
#' @return Integer constant for mean loss
#' @export
#' @family optimization
ggml_opt_loss_type_mean <- function() {
  .Call("R_ggml_opt_loss_type_mean")
}

#' Loss type: Sum
#'
#' Returns the constant for sum loss type.
#' Custom loss - reduces outputs to sum value.
#'
#' @return Integer constant for sum loss
#' @export
#' @family optimization
ggml_opt_loss_type_sum <- function() {
  .Call("R_ggml_opt_loss_type_sum")
}

#' Loss type: Cross Entropy
#'
#' Returns the constant for cross entropy loss type.
#' Use for classification tasks.
#'
#' @return Integer constant for cross entropy loss
#' @export
#' @family optimization
ggml_opt_loss_type_cross_entropy <- function() {
  .Call("R_ggml_opt_loss_type_cross_entropy")
}

#' Loss type: Mean Squared Error
#'
#' Returns the constant for MSE loss type.
#' Use for regression tasks.
#'
#' @return Integer constant for MSE loss
#' @export
#' @family optimization
ggml_opt_loss_type_mse <- function() {
  .Call("R_ggml_opt_loss_type_mse")
}

# ============================================================================
# Optimizer Type Constants
# ============================================================================

#' Optimizer type: AdamW
#'
#' Returns the constant for AdamW optimizer.
#' Adam with weight decay - recommended for most tasks.
#'
#' @return Integer constant for AdamW optimizer
#' @export
#' @family optimization
ggml_opt_optimizer_type_adamw <- function() {
  .Call("R_ggml_opt_optimizer_type_adamw")
}

#' Optimizer type: SGD
#'
#' Returns the constant for SGD optimizer.
#' Stochastic gradient descent - simpler but may require tuning.
#'
#' @return Integer constant for SGD optimizer
#' @export
#' @family optimization
ggml_opt_optimizer_type_sgd <- function() {
  .Call("R_ggml_opt_optimizer_type_sgd")
}

# ============================================================================
# Dataset Functions
# ============================================================================

#' Create a new optimization dataset
#'
#' Creates a dataset for training with specified data and label types.
#'
#' @param type_data GGML type for data tensor (e.g., GGML_TYPE_F32)
#' @param type_label GGML type for label tensor (e.g., GGML_TYPE_F32)
#' @param ne_datapoint Number of elements per datapoint
#' @param ne_label Number of elements per label (0 if no labels)
#' @param ndata Total number of datapoints
#' @param ndata_shard Shard size for shuffling (1 is fine for most cases)
#' @return External pointer to dataset
#' @export
#' @family optimization
ggml_opt_dataset_init <- function(type_data, type_label, ne_datapoint, ne_label, ndata, ndata_shard = 1) {
  .Call("R_ggml_opt_dataset_init",
        as.integer(type_data),
        as.integer(type_label),
        as.numeric(ne_datapoint),
        as.numeric(ne_label),
        as.numeric(ndata),
        as.numeric(ndata_shard))
}

#' Free optimization dataset
#'
#' Releases memory associated with a dataset.
#'
#' @param dataset External pointer to dataset
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_dataset_free <- function(dataset) {
  invisible(.Call("R_ggml_opt_dataset_free", dataset))
}

#' Get number of datapoints in dataset
#'
#' @param dataset External pointer to dataset
#' @return Number of datapoints
#' @export
#' @family optimization
ggml_opt_dataset_ndata <- function(dataset) {
  .Call("R_ggml_opt_dataset_ndata", dataset)
}

#' Get data tensor from dataset
#'
#' Returns the underlying data tensor with shape [ne_datapoint, ndata].
#'
#' @param dataset External pointer to dataset
#' @return External pointer to data tensor
#' @export
#' @family optimization
ggml_opt_dataset_data <- function(dataset) {
  .Call("R_ggml_opt_dataset_data", dataset)
}

#' Get labels tensor from dataset
#'
#' Returns the underlying labels tensor with shape [ne_label, ndata].
#'
#' @param dataset External pointer to dataset
#' @return External pointer to labels tensor, or NULL if no labels
#' @export
#' @family optimization
ggml_opt_dataset_labels <- function(dataset) {
  .Call("R_ggml_opt_dataset_labels", dataset)
}

#' Shuffle dataset
#'
#' Shuffles the dataset using the RNG from the optimizer context.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param dataset External pointer to dataset
#' @param idata Number of datapoints to shuffle (-1 for all)
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_dataset_shuffle <- function(opt_ctx, dataset, idata = -1) {
  invisible(.Call("R_ggml_opt_dataset_shuffle", opt_ctx, dataset, as.numeric(idata)))
}

#' Get batch from dataset
#'
#' Copies a batch of data and labels to the provided tensors.
#'
#' @param dataset External pointer to dataset
#' @param data_batch Tensor to receive data batch
#' @param labels_batch Tensor to receive labels batch (can be NULL)
#' @param ibatch Batch index
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_dataset_get_batch <- function(dataset, data_batch, labels_batch = NULL, ibatch) {
  invisible(.Call("R_ggml_opt_dataset_get_batch", dataset, data_batch, labels_batch, as.numeric(ibatch)))
}

# ============================================================================
# Optimizer Context Functions
# ============================================================================

#' Get default optimizer parameters
#'
#' Returns a list with default optimization parameters.
#'
#' @param sched Backend scheduler
#' @param loss_type Loss type constant
#' @return List with loss_type, build_type, opt_period, optimizer
#' @export
#' @family optimization
ggml_opt_default_params <- function(sched, loss_type) {
  .Call("R_ggml_opt_default_params", sched, as.integer(loss_type))
}

#' Initialize optimizer context
#'
#' Creates a new optimizer context for training.
#'
#' @param sched Backend scheduler
#' @param loss_type Loss type (use ggml_opt_loss_type_* functions)
#' @param optimizer Optimizer type (use ggml_opt_optimizer_type_* functions)
#' @param opt_period Gradient accumulation steps before optimizer step
#' @param ctx_compute Compute context for static graph mode (or NULL)
#' @param inputs Input tensor for static graph mode (or NULL)
#' @param outputs Output tensor for static graph mode (or NULL)
#' @return External pointer to optimizer context
#' @export
#' @family optimization
ggml_opt_init <- function(sched, loss_type, optimizer = ggml_opt_optimizer_type_adamw(), opt_period = 1L,
                          ctx_compute = NULL, inputs = NULL, outputs = NULL) {
  .Call("R_ggml_opt_init", sched, as.integer(loss_type), as.integer(optimizer), as.integer(opt_period),
        ctx_compute, inputs, outputs)
}

#' Free optimizer context
#'
#' Releases memory associated with an optimizer context.
#'
#' @param opt_ctx External pointer to optimizer context
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_free <- function(opt_ctx) {
  invisible(.Call("R_ggml_opt_free", opt_ctx))
}

#' Reset optimizer context
#'
#' Resets gradients to zero, initializes loss, and optionally resets optimizer state.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param optimizer Whether to also reset optimizer state (momentum, etc.)
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_reset <- function(opt_ctx, optimizer = FALSE) {
  invisible(.Call("R_ggml_opt_reset", opt_ctx, as.logical(optimizer)))
}

#' Check if using static graphs
#'
#' @param opt_ctx External pointer to optimizer context
#' @return Logical indicating if graphs are statically allocated
#' @export
#' @family optimization
ggml_opt_static_graphs <- function(opt_ctx) {
  .Call("R_ggml_opt_static_graphs", opt_ctx)
}

#' Get inputs tensor from optimizer context
#'
#' @param opt_ctx External pointer to optimizer context
#' @return External pointer to inputs tensor
#' @export
#' @family optimization
ggml_opt_inputs <- function(opt_ctx) {
  .Call("R_ggml_opt_inputs", opt_ctx)
}

#' Get outputs tensor from optimizer context
#'
#' @param opt_ctx External pointer to optimizer context
#' @return External pointer to outputs tensor
#' @export
#' @family optimization
ggml_opt_outputs <- function(opt_ctx) {
  .Call("R_ggml_opt_outputs", opt_ctx)
}

#' Get labels tensor from optimizer context
#'
#' @param opt_ctx External pointer to optimizer context
#' @return External pointer to labels tensor
#' @export
#' @family optimization
ggml_opt_labels <- function(opt_ctx) {
  .Call("R_ggml_opt_labels", opt_ctx)
}

#' Get loss tensor from optimizer context
#'
#' @param opt_ctx External pointer to optimizer context
#' @return External pointer to loss tensor
#' @export
#' @family optimization
ggml_opt_loss <- function(opt_ctx) {
  .Call("R_ggml_opt_loss", opt_ctx)
}

#' Get predictions tensor from optimizer context
#'
#' @param opt_ctx External pointer to optimizer context
#' @return External pointer to predictions tensor
#' @export
#' @family optimization
ggml_opt_pred <- function(opt_ctx) {
  .Call("R_ggml_opt_pred", opt_ctx)
}

#' Get number of correct predictions tensor
#'
#' @param opt_ctx External pointer to optimizer context
#' @return External pointer to ncorrect tensor
#' @export
#' @family optimization
ggml_opt_ncorrect <- function(opt_ctx) {
  .Call("R_ggml_opt_ncorrect", opt_ctx)
}

#' Get optimizer type from context
#'
#' @param opt_ctx External pointer to optimizer context
#' @return Integer optimizer type constant
#' @export
#' @family optimization
ggml_opt_context_optimizer_type <- function(opt_ctx) {
  .Call("R_ggml_opt_context_optimizer_type", opt_ctx)
}

#' Get optimizer name
#'
#' @param optimizer_type Integer optimizer type constant
#' @return Character string with optimizer name
#' @export
#' @family optimization
ggml_opt_optimizer_name <- function(optimizer_type) {
  .Call("R_ggml_opt_optimizer_name", as.integer(optimizer_type))
}

# ============================================================================
# Result Functions
# ============================================================================

#' Initialize optimization result
#'
#' Creates a new result object to accumulate training statistics.
#'
#' @return External pointer to result object
#' @export
#' @family optimization
ggml_opt_result_init <- function() {
  .Call("R_ggml_opt_result_init")
}

#' Free optimization result
#'
#' @param result External pointer to result object
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_result_free <- function(result) {
  invisible(.Call("R_ggml_opt_result_free", result))
}

#' Reset optimization result
#'
#' @param result External pointer to result object
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_result_reset <- function(result) {
  invisible(.Call("R_ggml_opt_result_reset", result))
}

#' Get number of datapoints from result
#'
#' @param result External pointer to result object
#' @return Number of datapoints processed
#' @export
#' @family optimization
ggml_opt_result_ndata <- function(result) {
  .Call("R_ggml_opt_result_ndata", result)
}

#' Get loss from result
#'
#' @param result External pointer to result object
#' @return Named numeric vector with 'loss' and 'uncertainty'
#' @export
#' @family optimization
ggml_opt_result_loss <- function(result) {
  .Call("R_ggml_opt_result_loss", result)
}

#' Get accuracy from result
#'
#' @param result External pointer to result object
#' @return Named numeric vector with 'accuracy' and 'uncertainty'
#' @export
#' @family optimization
ggml_opt_result_accuracy <- function(result) {
  .Call("R_ggml_opt_result_accuracy", result)
}

# ============================================================================
# Computation Functions
# ============================================================================

#' Allocate graph for evaluation
#'
#' Must be called before ggml_opt_eval. Allocates forward or forward+backward graph.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param backward Whether to allocate backward graph (for training)
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_alloc <- function(opt_ctx, backward = TRUE) {
  invisible(.Call("R_ggml_opt_alloc", opt_ctx, as.logical(backward)))
}

#' Evaluate model
#'
#' Performs forward pass, optionally increments result, and does backward pass if allocated.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param result External pointer to result object (optional)
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_eval <- function(opt_ctx, result = NULL) {
  invisible(.Call("R_ggml_opt_eval", opt_ctx, result))
}

# ============================================================================
# High-Level Training Function
# ============================================================================

#' Fit model to dataset
#'
#' High-level function to train a model on a dataset.
#' This is the recommended way to train models.
#'
#' @param sched Backend scheduler
#' @param ctx_compute Compute context (for temporary tensors)
#' @param inputs Input tensor with shape [ne_datapoint, batch_size]
#' @param outputs Output tensor with shape [ne_label, batch_size]
#' @param dataset Dataset created with ggml_opt_dataset_init
#' @param loss_type Loss type (default: MSE)
#' @param optimizer Optimizer type (default: AdamW)
#' @param nepoch Number of epochs
#' @param nbatch_logical Logical batch size (for gradient accumulation)
#' @param val_split Fraction of data for validation (0.0 to 1.0)
#' @param silent Whether to suppress progress output
#' @return NULL invisibly
#' @export
#' @family optimization
#' @examples
#' # Full training requires building a computation graph
#' # See package vignettes for complete examples
#' if (FALSE) {
#' cpu <- ggml_backend_cpu_init()
#' sched <- ggml_backend_sched_new(list(cpu))
#' dataset <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, 10, 1, 1000)
#' # ... build model graph with ctx_compute, inputs, outputs ...
#' ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset,
#'              nepoch = 10, val_split = 0.1)
#' ggml_opt_dataset_free(dataset)
#' ggml_backend_sched_free(sched)
#' ggml_backend_free(cpu)
#' }
ggml_opt_fit <- function(sched, ctx_compute, inputs, outputs, dataset,
                         loss_type = ggml_opt_loss_type_mse(),
                         optimizer = ggml_opt_optimizer_type_adamw(),
                         nepoch = 1, nbatch_logical = 32,
                         val_split = 0.0, silent = FALSE) {
  invisible(.Call("R_ggml_opt_fit",
                  sched, ctx_compute, inputs, outputs, dataset,
                  as.integer(loss_type), as.integer(optimizer),
                  as.numeric(nepoch), as.numeric(nbatch_logical),
                  as.numeric(val_split), as.logical(silent)))
}

# ============================================================================
# Additional Functions
# ============================================================================

#' Get gradient accumulator for a tensor
#'
#' Returns the gradient accumulator tensor for a node from the forward graph.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param node External pointer to tensor node
#' @return External pointer to gradient accumulator tensor, or NULL if not found
#' @export
#' @family optimization
ggml_opt_grad_acc <- function(opt_ctx, node) {
  .Call("R_ggml_opt_grad_acc", opt_ctx, node)
}

#' Get predictions from result
#'
#' Returns the predictions as an integer vector.
#' The length equals the number of datapoints processed.
#'
#' @param result External pointer to result object
#' @return Integer vector of predictions
#' @export
#' @family optimization
ggml_opt_result_pred <- function(result) {
  .Call("R_ggml_opt_result_pred", result)
}

#' Prepare allocation for non-static graphs
#'
#' Must be called before ggml_opt_alloc when not using static graphs.
#' Sets up the optimizer context with the computation graph and input/output tensors.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param ctx_compute Compute context for temporary tensors
#' @param graph Computation graph (from ggml_build_forward_expand)
#' @param inputs Input tensor
#' @param outputs Output tensor
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_prepare_alloc <- function(opt_ctx, ctx_compute, graph, inputs, outputs) {
  invisible(.Call("R_ggml_opt_prepare_alloc", opt_ctx, ctx_compute, graph, inputs, outputs))
}

#' Run one training epoch
#'
#' Performs training on the front portion of the dataset and evaluation
#' on the back portion. This gives more control than ggml_opt_fit.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param dataset External pointer to dataset
#' @param result_train Result object to accumulate training stats (or NULL)
#' @param result_eval Result object to accumulate evaluation stats (or NULL)
#' @param idata_split Data index at which to split training and evaluation
#' @param callback_train Callback for training: TRUE for progress bar, FALSE for none,
#'   or a function(train, ibatch, ibatch_max, t_start_us, result)
#' @param callback_eval Callback for evaluation: TRUE for progress bar, FALSE for none,
#'   or a function(train, ibatch, ibatch_max, t_start_us, result)
#' @return NULL invisibly
#' @export
#' @family optimization
#' @examples
#' # Requires full optimizer setup - see ggml_opt_fit() for simpler API
#' if (FALSE) {
#' result_train <- ggml_opt_result_init()
#' result_eval <- ggml_opt_result_init()
#' ggml_opt_epoch(opt_ctx, dataset, result_train, result_eval,
#'                idata_split = 900, callback_train = TRUE)
#' ggml_opt_result_free(result_train)
#' ggml_opt_result_free(result_eval)
#' }
ggml_opt_epoch <- function(opt_ctx, dataset, result_train = NULL, result_eval = NULL,
                           idata_split, callback_train = TRUE, callback_eval = TRUE) {
  invisible(.Call("R_ggml_opt_epoch", opt_ctx, dataset, result_train, result_eval,
                  as.numeric(idata_split), callback_train, callback_eval))
}
