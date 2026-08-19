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

#' Loss type: Weighted Mean Squared Error
#'
#' Returns the constant for per-datapoint weighted MSE loss type. Computes
#' \code{sum(w * (pred - y)^2) / nelements}, where \code{w} is a per-sample
#' weight supplied via \code{\link{ggml_opt_dataset_weights}}.
#'
#' @return Integer constant for weighted MSE loss
#' @export
#' @family optimization
ggml_opt_loss_type_weighted_mse <- function() {
  .Call("R_ggml_opt_loss_type_weighted_mse")
}

#' Loss type: Mean Absolute Error
#'
#' Returns the constant for MAE (L1) loss, \code{mean(|pred - y|)}.  Its
#' gradient is \code{sgn(pred - y)}, so a far-off datapoint pulls no harder than
#' a near one -- which is what makes it more robust to outliers than MSE, at the
#' cost of a gradient that does not vanish as the fit improves.
#'
#' @return Integer constant for MAE loss
#' @export
#' @family optimization
ggml_opt_loss_type_mae <- function() {
  .Call("R_ggml_opt_loss_type_mae")
}

#' Loss type: Huber (smooth L1)
#'
#' Returns the constant for the Huber loss with \code{delta = 1}:
#' \code{0.5 * e^2} where \code{|e| <= 1} and \code{|e| - 0.5} beyond it.
#' Quadratic near zero, so unlike MAE the gradient vanishes at the optimum, and
#' linear far out, so unlike MSE a single outlier cannot dominate the batch.
#'
#' \code{delta} is fixed at 1, the usual default.
#'
#' @return Integer constant for Huber loss
#' @export
#' @family optimization
ggml_opt_loss_type_huber <- function() {
  .Call("R_ggml_opt_loss_type_huber")
}

#' Loss type: Binary Cross-Entropy
#'
#' Returns the constant for element-wise binary cross-entropy,
#' \code{mean(-[y*log(p) + (1-y)*log(1-p)])}.
#'
#' Unlike \code{\link{ggml_opt_loss_type_cross_entropy}}, which softmaxes its
#' own input across a class axis and therefore expects logits, this one treats
#' every output as an independent Bernoulli and expects \emph{probabilities} --
#' so the model's last layer should end in a sigmoid.  It is the loss for
#' multi-label targets and for a single-unit binary output.  Inputs are clamped
#' away from 0 and 1 internally, since \code{log(0)} would otherwise poison the
#' whole batch.
#'
#' @return Integer constant for binary cross-entropy loss
#' @export
#' @family optimization
ggml_opt_loss_type_binary_cross_entropy <- function() {
  .Call("R_ggml_opt_loss_type_binary_cross_entropy")
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

#' Get dataset per-datapoint weights tensor
#'
#' Returns the (lazily allocated) per-datapoint weights tensor with shape
#' [1, ndata]. The first call allocates it; fill it via
#' \code{ggml_backend_tensor_set_data()}. Used together with
#' \code{\link{ggml_opt_loss_type_weighted_mse}}.
#'
#' @param dataset External pointer to dataset
#' @return External pointer to weights tensor
#' @export
#' @family optimization
ggml_opt_dataset_weights <- function(dataset) {
  .Call("R_ggml_opt_dataset_weights", dataset)
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

#' Number of output heads in an optimizer context
#'
#' @param opt_ctx External pointer to optimizer context
#' @return Integer number of output heads (1 for single-output models)
#' @export
#' @family optimization
ggml_opt_n_loss <- function(opt_ctx) {
  .Call("R_ggml_opt_n_loss", opt_ctx)
}

#' Get one output head's tensors from an optimizer context
#'
#' Per-head counterparts of \code{ggml_opt_outputs()} and friends, for
#' multi-output models. \code{ihead} is 1-based; head 1 is what the
#' single-head accessors return.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param ihead Output head index, 1-based
#' @return External pointer to the tensor, or NULL if this head has none
#' @export
#' @family optimization
ggml_opt_outputs_i <- function(opt_ctx, ihead) {
  .Call("R_ggml_opt_outputs_i", opt_ctx, as.integer(ihead))
}

#' @rdname ggml_opt_outputs_i
#' @export
ggml_opt_labels_i <- function(opt_ctx, ihead) {
  .Call("R_ggml_opt_labels_i", opt_ctx, as.integer(ihead))
}

#' @rdname ggml_opt_outputs_i
#' @export
ggml_opt_loss_weights_i <- function(opt_ctx, ihead) {
  .Call("R_ggml_opt_loss_weights_i", opt_ctx, as.integer(ihead))
}

#' @rdname ggml_opt_outputs_i
#' @export
ggml_opt_pred_i <- function(opt_ctx, ihead) {
  .Call("R_ggml_opt_pred_i", opt_ctx, as.integer(ihead))
}

#' @rdname ggml_opt_outputs_i
#' @export
ggml_opt_ncorrect_i <- function(opt_ctx, ihead) {
  .Call("R_ggml_opt_ncorrect_i", opt_ctx, as.integer(ihead))
}

#' Get one head's loss scalar, before weighting
#'
#' Returns the head's own loss, before it is multiplied by its
#' \code{loss_weights} entry and summed into the total. \code{ggml_opt_loss()}
#' returns the weighted total that is actually optimized.
#'
#' @param opt_ctx External pointer to optimizer context
#' @param ihead Output head index, 1-based
#' @return External pointer to the head's loss tensor
#' @export
#' @family optimization
ggml_opt_loss_i <- function(opt_ctx, ihead) {
  .Call("R_ggml_opt_loss_i", opt_ctx, as.integer(ihead))
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

#' Number of output heads accumulated in a result
#'
#' Zero before the first epoch has run, so callers must treat a freshly reset
#' result as holding no per-head data yet.
#'
#' @param result External pointer to result object
#' @return Integer number of heads
#' @export
#' @family optimization
ggml_opt_result_n_loss <- function(result) {
  .Call("R_ggml_opt_result_n_loss", result)
}

#' Get one output head's loss or accuracy from a result
#'
#' Per-head counterparts of \code{ggml_opt_result_loss()} and
#' \code{ggml_opt_result_accuracy()}, which report the weighted total the
#' optimizer minimizes. These report a single head on its own, before
#' weighting, for the training history. \code{ihead} is 1-based.
#'
#' Heads whose loss is not cross-entropy have no accuracy; those return
#' \code{NA}.
#'
#' @param result External pointer to result object
#' @param ihead Output head index, 1-based
#' @return Named numeric vector with the value and its 'uncertainty'
#' @export
#' @family optimization
ggml_opt_result_loss_i <- function(result, ihead) {
  .Call("R_ggml_opt_result_loss_i", result, as.integer(ihead))
}

#' @rdname ggml_opt_result_loss_i
#' @export
ggml_opt_result_accuracy_i <- function(result, ihead) {
  .Call("R_ggml_opt_result_accuracy_i", result, as.integer(ihead))
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

#' Fit a multi-output model
#'
#' Trains a model with several output heads. Each head has its own labels,
#' loss type and weight; the optimized total is
#' \code{sum(loss_weights[i] * loss_i)}.
#'
#' All heads share the dataset's datapoints. The dataset's labels hold every
#' head's labels concatenated along the first dimension: head \code{i} occupies
#' \code{labels_offs[i] + seq_len(width_i)} of each label row.
#'
#' @param sched Backend scheduler
#' @param ctx_compute Compute context holding the model graph
#' @param inputs Input tensor
#' @param outputs List of output tensors, one per head
#' @param dataset Dataset with data and concatenated labels
#' @param loss_types Integer vector of loss type constants, one per head
#' @param loss_weights Numeric vector of head weights (default: all 1)
#' @param labels_offs Numeric vector of label offsets, 0-based, one per head
#' @param optimizer Optimizer type constant
#' @param nepoch Number of epochs
#' @param nbatch_logical Logical batch size
#' @param val_split Fraction of the data used for validation
#' @param silent Suppress progress output
#' @return Named list with train_loss, train_accuracy, val_loss, val_accuracy,
#'   head_loss (a nepoch x nhead matrix of unweighted per-head losses) and
#'   head_accuracy (same shape; NA for heads whose loss is not cross-entropy)
#' @seealso \code{\link{ggml_fit_opt_multi}}, which this wraps and which adds
#'   callbacks and shuffling control
#' @export
#' @family optimization
ggml_opt_fit_multi <- function(sched, ctx_compute, inputs, outputs, dataset,
                               loss_types,
                               loss_weights = rep(1, length(outputs)),
                               labels_offs = NULL,
                               optimizer = ggml_opt_optimizer_type_adamw(),
                               nepoch = 1, nbatch_logical = 32,
                               val_split = 0.0, silent = FALSE) {
  # Thin wrapper: ggml_fit_opt_multi() is the single implementation. Argument
  # checking and the labels_offs default live there, so the two paths cannot
  # drift apart again. The only behavioural difference kept here is the legacy
  # shuffling contract -- the whole dataset once, never the training portion
  # per epoch -- which is what the C loop this used to call did.
  hist <- ggml_fit_opt_multi(
    sched = sched, ctx_compute = ctx_compute, inputs = inputs,
    outputs = outputs, dataset = dataset,
    loss_types = loss_types, loss_weights = loss_weights,
    labels_offs = labels_offs, optimizer = optimizer,
    nepoch = nepoch, nbatch_logical = nbatch_logical,
    val_split = val_split,
    shuffle = FALSE, shuffle_all = TRUE,
    callbacks = list(), silent = silent
  )

  list(train_loss     = hist$train_loss,
       train_accuracy = hist$train_accuracy,
       val_loss       = hist$val_loss,
       val_accuracy   = hist$val_accuracy,
       head_loss      = attr(hist, "head_loss"),
       head_accuracy  = attr(hist, "head_accuracy"))
}

#' Get one output head's slice of a batch's labels
#'
#' Multi-output counterpart of \code{ggml_opt_dataset_get_batch()}. The
#' dataset's labels hold every head's labels concatenated along the first
#' dimension; this copies just the slice belonging to one head.
#'
#' @param dataset Dataset pointer
#' @param data_batch Input batch tensor, or NULL to copy only the labels
#'   (the caller copies the inputs once, with the first head)
#' @param labels_batch This head's labels batch tensor
#' @param labels_off Offset of this head within a label row, 0-based, in elements
#' @param ibatch Batch index, 0-based
#' @return NULL, invisibly
#' @export
#' @family optimization
ggml_opt_dataset_get_batch_head <- function(dataset, data_batch, labels_batch,
                                            labels_off, ibatch) {
  invisible(.Call("R_ggml_opt_dataset_get_batch_head",
                  dataset, data_batch, labels_batch,
                  as.numeric(labels_off), as.numeric(ibatch)))
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

# ============================================================================
# Low-level: init optimizer context for R-side epoch loop
# ============================================================================

#' Initialize optimizer context for R-side epoch loop
#'
#' Returns a list with `opt_ctx` and `lr_ud` (learning rate userdata pointer).
#' Use `ggml_opt_set_lr()` to update LR between epochs.
#' The optimizer state (momentum) is preserved across epochs.
#'
#' @param sched Backend scheduler
#' @param loss_type Loss type constant
#' @param optimizer Optimizer type constant
#' @param opt_period Gradient accumulation period
#' @param ctx_compute Compute context (for static graphs)
#' @param inputs Input tensor (for static graphs)
#' @param outputs Output tensor (for static graphs)
#' @return List with elements `opt_ctx` and `lr_ud`
#' @export
#' @family optimization
ggml_opt_init_for_fit <- function(sched, loss_type, optimizer = ggml_opt_optimizer_type_adamw(),
                                   opt_period = 1L, ctx_compute = NULL,
                                   inputs = NULL, outputs = NULL) {
  .Call("R_ggml_opt_init_for_fit",
        sched, as.integer(loss_type), as.integer(optimizer), as.integer(opt_period),
        ctx_compute, inputs, outputs)
}

#' Initialize a multi-output optimizer context for an R-side epoch loop
#'
#' Multi-output counterpart of \code{ggml_opt_init_for_fit()}. Returns the same
#' \code{opt_ctx}/\code{lr_ud} pair, so \code{ggml_opt_set_lr()} and hence LR
#' schedulers work identically; in addition it records each head's loss type,
#' weight and label offset, which \code{ggml_opt_fit_multi()} would otherwise
#' set internally.
#'
#' @param sched Backend scheduler
#' @param loss_types Integer vector of loss type constants, one per head
#' @param loss_weights Numeric vector of head weights, one per head
#' @param labels_offs Numeric vector of 0-based label offsets, one per head
#' @param optimizer Optimizer type constant
#' @param opt_period Gradient accumulation period
#' @param ctx_compute Compute context holding the model graph
#' @param inputs Input tensor
#' @param outputs List of output tensors, one per head
#' @return List with elements `opt_ctx` and `lr_ud`
#' @export
#' @family optimization
ggml_opt_init_for_fit_multi <- function(sched, loss_types, loss_weights, labels_offs,
                                        optimizer = ggml_opt_optimizer_type_adamw(),
                                        opt_period = 1L, ctx_compute = NULL,
                                        inputs = NULL, outputs = NULL) {
  if (!is.list(outputs)) outputs <- list(outputs)
  .Call("R_ggml_opt_init_for_fit_multi",
        sched, as.integer(loss_types), as.numeric(loss_weights), as.numeric(labels_offs),
        as.integer(optimizer), as.integer(opt_period),
        ctx_compute, inputs, outputs)
}

#' Set learning rate in optimizer context
#'
#' Updates the LR used for subsequent backward passes.
#' Can be called between epochs to implement LR scheduling.
#'
#' @param lr_ud LR userdata pointer (from `ggml_opt_init_for_fit()$lr_ud`)
#' @param adamw_lr New AdamW learning rate (NA to keep current)
#' @param sgd_lr New SGD learning rate (NA to keep current)
#' @return NULL invisibly
#' @export
#' @family optimization
ggml_opt_set_lr <- function(lr_ud, adamw_lr = NA, sgd_lr = NA) {
  invisible(.Call("R_ggml_opt_set_lr", lr_ud, as.numeric(adamw_lr), as.numeric(sgd_lr)))
}

#' Get current learning rate from optimizer context
#'
#' @param lr_ud LR userdata pointer (from `ggml_opt_init_for_fit()$lr_ud`)
#' @return Named numeric vector with 'adamw' and 'sgd' LR values
#' @export
#' @family optimization
ggml_opt_get_lr <- function(lr_ud) {
  .Call("R_ggml_opt_get_lr", lr_ud)
}

# ============================================================================
# High-level: ggml_fit_opt() with R epoch loop and callbacks
# ============================================================================

#' Fit model with R-side epoch loop and callbacks
#'
#' Trains a model epoch by epoch in R, allowing callbacks for early stopping
#' and learning rate scheduling. Optimizer state (momentum) is preserved
#' across all epochs.
#'
#' @param sched Backend scheduler
#' @param ctx_compute Compute context (for temporary tensors)
#' @param inputs Input tensor with shape [ne_datapoint, batch_size]
#' @param outputs Output tensor with shape [ne_label, batch_size]
#' @param dataset Dataset created with `ggml_opt_dataset_init()`
#' @param loss_type Loss type (default: MSE)
#' @param optimizer Optimizer type (default: AdamW)
#' @param nepoch Number of epochs
#' @param nbatch_logical Logical batch size (for gradient accumulation)
#' @param val_split Fraction of data for validation (0.0 to 1.0)
#' @param shuffle Shuffle the training portion each epoch (default
#'   \code{TRUE}).  Set to \code{FALSE} for time series or to make a run
#'   exactly reproducible.
#' @param shuffle_all Shuffle the whole dataset once before the
#'   train/validation split is taken, so the validation portion is a random
#'   sample rather than the tail of the input.  Defaults to \code{shuffle}.
#'   Callers that append an explicit validation set to the data pass
#'   \code{FALSE}, since there the split is positional and must be preserved.
#' @param callbacks List of callback lists. Each element may have
#'   `on_epoch_begin(epoch, logs, state)` and/or `on_epoch_end(epoch, logs, state)`.
#'   Built-in factories: `ggml_callback_early_stopping()`,
#'   `ggml_schedule_step_decay()`, `ggml_schedule_cosine_decay()`,
#'   `ggml_schedule_reduce_on_plateau()`.
#'   `state` is a mutable environment with fields:
#'   `stop` (set TRUE to stop training), `lr_ud`, `nepoch`.
#' @param silent Whether to suppress per-epoch progress output
#' @return Data frame with columns epoch, train_loss, train_accuracy, val_loss, val_accuracy
#' @export
#' @family optimization
#' @examples
#' if (FALSE) {
#' history <- ggml_fit_opt(sched, ctx_compute, inputs, outputs, dataset,
#'   nepoch = 50, val_split = 0.2,
#'   callbacks = list(
#'     ggml_callback_early_stopping(monitor = "val_loss", patience = 5),
#'     ggml_schedule_cosine_decay()
#'   ))
#' }
ggml_fit_opt <- function(sched, ctx_compute, inputs, outputs, dataset,
                     loss_type   = ggml_opt_loss_type_mse(),
                     optimizer   = ggml_opt_optimizer_type_adamw(),
                     nepoch      = 10L,
                     nbatch_logical = 32L,
                     val_split   = 0.0,
                     shuffle     = TRUE,
                     shuffle_all = shuffle,
                     callbacks   = list(),
                     silent      = FALSE) {

  # --- compute parameters (same as R_ggml_opt_fit) ---
  ndata <- as.integer(ggml_opt_dataset_ndata(dataset))
  nbatch_physical <- .ggml_input_batch_size(inputs, dataset)
  # A logical batch smaller than the graph's physical batch cannot be honoured:
  # the forward pass always consumes nbatch_physical samples, so treat that as
  # the floor rather than deriving an opt_period of 1 from a ratio below one.
  nbatch_logical  <- as.integer(max(nbatch_logical, nbatch_physical))
  opt_period      <- as.integer(max(1L, nbatch_logical %/% nbatch_physical))
  nbatches_logical  <- ndata %/% nbatch_logical
  ibatch_split    <- as.integer(floor((1.0 - val_split) * nbatches_logical) * opt_period)
  # Never split past the end of the dataset: ggml_opt_dataset_shuffle() asserts
  # idata <= ndata, and rounding in the product above can overshoot when ndata
  # is not a whole number of logical batches.
  idata_split     <- min(as.integer(ibatch_split * nbatch_physical), ndata)

  # --- init optimizer context (preserves momentum across epochs) ---
  ctx_list <- ggml_opt_init_for_fit(
    sched, loss_type, optimizer, opt_period,
    ctx_compute, inputs, outputs
  )
  opt_ctx <- ctx_list$opt_ctx
  lr_ud   <- ctx_list$lr_ud
  on.exit({
    ggml_opt_free(opt_ctx)
  }, add = TRUE)

  # --- shuffle all data once at start ---
  # Done before the train/val split is applied, so the validation portion is a
  # random sample rather than whatever happened to sit at the end of the input.
  if (shuffle_all && nbatch_logical < ndata) {
    ggml_opt_dataset_shuffle(opt_ctx, dataset, -1)
  }

  result_train <- ggml_opt_result_init()
  result_eval  <- ggml_opt_result_init()
  on.exit({
    ggml_opt_result_free(result_train)
    ggml_opt_result_free(result_eval)
  }, add = TRUE)

  # --- mutable state shared with callbacks ---
  state <- new.env(parent = emptyenv())
  state$stop   <- FALSE
  state$lr_ud  <- lr_ud
  state$nepoch <- as.integer(nepoch)

  # --- history ---
  hist <- vector("list", nepoch)

  for (epoch in seq_len(nepoch)) {
    # Pick up a ggml_set_n_threads() issued after the scheduler was built --
    # the single C entry points sync once before their loop, an R loop has no
    # equivalent moment.
    .ggml_sched_sync_threads(sched)

    # shuffle training portion (leaves the validation tail untouched)
    if (shuffle && nbatch_logical < idata_split) {
      ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split)
    }

    ggml_opt_result_reset(result_train)
    ggml_opt_result_reset(result_eval)

    logs <- list()

    # on_epoch_begin callbacks
    for (cb in callbacks) {
      if (is.function(cb$on_epoch_begin))
        cb$on_epoch_begin(epoch, logs, state)
      if (isTRUE(state$stop)) break
    }
    if (isTRUE(state$stop)) break

    if (!silent) message(sprintf("Epoch %d/%d", epoch, nepoch))

    cb_progress <- if (silent) FALSE else TRUE
    ggml_opt_epoch(opt_ctx, dataset, result_train, result_eval,
                   idata_split,
                   callback_train = cb_progress,
                   callback_eval  = cb_progress)

    # collect metrics
    train_loss_res <- ggml_opt_result_loss(result_train)
    train_acc_res  <- ggml_opt_result_accuracy(result_train)

    logs$train_loss     <- train_loss_res[["loss"]]
    logs$train_accuracy <- train_acc_res[["accuracy"]]

    if (val_split > 0) {
      val_loss_res  <- ggml_opt_result_loss(result_eval)
      val_acc_res   <- ggml_opt_result_accuracy(result_eval)
      logs$val_loss     <- val_loss_res[["loss"]]
      logs$val_accuracy <- val_acc_res[["accuracy"]]
    } else {
      logs$val_loss     <- NA_real_
      logs$val_accuracy <- NA_real_
    }

    hist[[epoch]] <- c(epoch = epoch, logs)

    if (!silent) {
      message(sprintf("  train_loss=%.4f  train_acc=%.4f  val_loss=%s  val_acc=%s",
                      logs$train_loss, logs$train_accuracy,
                      if (is.na(logs$val_loss)) "NA" else sprintf("%.4f", logs$val_loss),
                      if (is.na(logs$val_accuracy)) "NA" else sprintf("%.4f", logs$val_accuracy)))
    }

    # on_epoch_end callbacks
    for (cb in callbacks) {
      if (is.function(cb$on_epoch_end))
        cb$on_epoch_end(epoch, logs, state)
      if (isTRUE(state$stop)) break
    }
    if (isTRUE(state$stop)) break
  }

  # --- build history data frame ---
  filled <- Filter(Negate(is.null), hist)
  if (length(filled) == 0) {
    return(data.frame(epoch = integer(0), train_loss = numeric(0),
                      train_accuracy = numeric(0), val_loss = numeric(0),
                      val_accuracy = numeric(0)))
  }
  do.call(rbind.data.frame, lapply(filled, function(x) as.data.frame(as.list(x))))
}

# ============================================================================
# High-level: ggml_fit_opt_multi() -- multi-output R epoch loop with callbacks
# ============================================================================

#' Fit a multi-output model with an R-side epoch loop and callbacks
#'
#' Multi-output counterpart of \code{ggml_fit_opt()}. Each output head has its
#' own labels, loss type and weight; the optimized total is
#' \code{sum(loss_weights[i] * loss_i)}. Because the epoch loop runs in R, this
#' path supports callbacks (early stopping, LR schedulers) and the
#' \code{shuffle}/\code{shuffle_all} controls, which the single C call
#' \code{ggml_opt_fit_multi()} cannot offer.
#'
#' All heads share the dataset's datapoints. The dataset's labels hold every
#' head's labels concatenated along the first dimension: head \code{i} occupies
#' \code{labels_offs[i] + seq_len(width_i)} of each label row.
#'
#' @param sched Backend scheduler
#' @param ctx_compute Compute context holding the model graph
#' @param inputs Input tensor with shape [ne_datapoint, batch_size]
#' @param outputs List of output tensors, one per head
#' @param dataset Dataset with data and concatenated labels
#' @param loss_types Integer vector of loss type constants, one per head
#' @param loss_weights Numeric vector of head weights (default: all 1)
#' @param labels_offs Numeric vector of 0-based label offsets, one per head.
#'   Defaults to laying the heads out back to back in head order, using each
#'   head's \code{ne[0]} as its width.
#' @param head_names Character vector naming the heads, one per head, used for
#'   the per-head keys in \code{logs} and as the column names of the returned
#'   matrices. Defaults to \code{head_1}, \code{head_2}, ... Callers that have
#'   real output names should pass them, so a callback's \code{monitor=} matches
#'   what the training history shows.
#' @param optimizer Optimizer type (default: AdamW)
#' @param nepoch Number of epochs
#' @param nbatch_logical Logical batch size (for gradient accumulation)
#' @param val_split Fraction of data for validation (0.0 to 1.0)
#' @param shuffle Shuffle the training portion each epoch (default
#'   \code{TRUE}).  Set to \code{FALSE} for time series or to make a run
#'   exactly reproducible.
#' @param shuffle_all Shuffle the whole dataset once before the
#'   train/validation split is taken, so the validation portion is a random
#'   sample rather than the tail of the input.  Defaults to \code{shuffle}.
#' @param callbacks List of callback lists, as for \code{ggml_fit_opt()}. Each
#'   element may have `on_epoch_begin(epoch, logs, state)` and/or
#'   `on_epoch_end(epoch, logs, state)`. `state` is a mutable environment with
#'   `stop`, `lr_ud` and `nepoch`. `logs` carries the aggregate metrics
#'   (`train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`) and, for each
#'   head, `train_<head>_loss`, `train_<head>_accuracy`, `val_<head>_loss` and
#'   `val_<head>_accuracy` -- so a scheduler or early stopping can monitor one
#'   head rather than the total. Both phases are prefixed, so a head named
#'   `val_x` cannot collide with another head's validation key. Per-head
#'   validation keys are `NA` when `val_split` is 0, and accuracy keys are `NA`
#'   for heads whose loss is not cross-entropy.
#' @param silent Whether to suppress per-epoch progress output
#' @return Data frame with columns epoch, train_loss, train_accuracy, val_loss,
#'   val_accuracy -- one row per epoch actually run, so a callback that stops
#'   early shortens it. The per-head metrics come as attributes rather than
#'   columns, since they are matrices, each [nepoch_run x n_head] with columns
#'   named after the heads: \code{attr(h, "head_loss")},
#'   \code{attr(h, "head_accuracy")}, \code{attr(h, "val_head_loss")} and
#'   \code{attr(h, "val_head_accuracy")}, plus \code{attr(h, "head_names")}.
#' @export
#' @family optimization
#' @examples
#' if (FALSE) {
#' history <- ggml_fit_opt_multi(sched, ctx_compute, inputs, list(out_a, out_b),
#'   dataset, loss_types = c(ggml_opt_loss_type_mse(), ggml_opt_loss_type_mse()),
#'   nepoch = 50, val_split = 0.2,
#'   callbacks = list(ggml_callback_early_stopping(monitor = "val_loss", patience = 5)))
#' attr(history, "head_loss")
#' }
ggml_fit_opt_multi <- function(sched, ctx_compute, inputs, outputs, dataset,
                               loss_types,
                               loss_weights = rep(1, length(outputs)),
                               labels_offs = NULL,
                               head_names  = NULL,
                               optimizer   = ggml_opt_optimizer_type_adamw(),
                               nepoch      = 10L,
                               nbatch_logical = 32L,
                               val_split   = 0.0,
                               shuffle     = TRUE,
                               shuffle_all = shuffle,
                               callbacks   = list(),
                               silent      = FALSE) {

  if (!is.list(outputs)) outputs <- list(outputs)
  n_head <- length(outputs)
  if (n_head < 1L) {
    stop("'outputs' must contain at least one output head.")
  }
  if (length(loss_types) != n_head) {
    stop(sprintf("'loss_types' must have one entry per output head (%d given, %d expected).",
                 length(loss_types), n_head))
  }
  if (length(loss_weights) != n_head) {
    stop(sprintf("'loss_weights' must have one entry per output head (%d given, %d expected).",
                 length(loss_weights), n_head))
  }

  # Default layout: heads laid out back to back in the label rows, in order.
  # Widths come from each head's ne[0], matching how the caller must have built
  # the concatenated labels.
  if (is.null(labels_offs)) {
    widths      <- vapply(outputs, function(o) ggml_tensor_shape(o)[1L], numeric(1))
    labels_offs <- c(0, cumsum(widths))[seq_len(n_head)]
  }
  if (length(labels_offs) != n_head) {
    stop(sprintf("'labels_offs' must have one entry per output head (%d given, %d expected).",
                 length(labels_offs), n_head))
  }

  # Names the per-head metrics are reported under in `logs`. The caller passes
  # the model's output names so a callback's monitor= matches what the training
  # history shows; standing alone, the heads are just numbered.
  if (is.null(head_names)) {
    head_names <- paste0("head_", seq_len(n_head))
  }
  head_names <- as.character(head_names)
  if (length(head_names) != n_head) {
    stop(sprintf("'head_names' must have one entry per output head (%d given, %d expected).",
                 length(head_names), n_head))
  }
  if (anyNA(head_names) || any(!nzchar(head_names))) {
    stop("'head_names' must not contain NA or empty strings.")
  }
  if (anyDuplicated(head_names)) {
    stop("'head_names' must be unique: ",
         paste(unique(head_names[duplicated(head_names)]), collapse = ", "))
  }

  nepoch <- as.integer(nepoch)

  # --- batching parameters (same derivation as ggml_fit_opt) ---
  ndata <- as.integer(ggml_opt_dataset_ndata(dataset))
  nbatch_physical <- .ggml_input_batch_size(inputs, dataset)
  nbatch_logical  <- as.integer(max(nbatch_logical, nbatch_physical))
  opt_period      <- as.integer(max(1L, nbatch_logical %/% nbatch_physical))
  nbatches_logical <- ndata %/% nbatch_logical
  ibatch_split    <- as.integer(floor((1.0 - val_split) * nbatches_logical) * opt_period)
  idata_split     <- min(as.integer(ibatch_split * nbatch_physical), ndata)

  # --- init optimizer context (preserves momentum across epochs) ---
  ctx_list <- ggml_opt_init_for_fit_multi(
    sched, loss_types, loss_weights, labels_offs,
    optimizer, opt_period, ctx_compute, inputs, outputs
  )
  opt_ctx <- ctx_list$opt_ctx
  lr_ud   <- ctx_list$lr_ud
  on.exit({
    ggml_opt_free(opt_ctx)
  }, add = TRUE)

  # --- shuffle all data once at start, before the train/val split ---
  if (shuffle_all && nbatch_logical < ndata) {
    ggml_opt_dataset_shuffle(opt_ctx, dataset, -1)
  }

  result_train <- ggml_opt_result_init()
  result_eval  <- ggml_opt_result_init()
  on.exit({
    ggml_opt_result_free(result_train)
    ggml_opt_result_free(result_eval)
  }, add = TRUE)

  # --- mutable state shared with callbacks ---
  state <- new.env(parent = emptyenv())
  state$stop   <- FALSE
  state$lr_ud  <- lr_ud
  state$nepoch <- nepoch

  hist <- vector("list", nepoch)
  # [epoch x head], columns named after the heads. Validation matrices stay
  # all-NA when val_split is 0.
  new_head_mat <- function() {
    matrix(NA_real_, nrow = nepoch, ncol = n_head,
           dimnames = list(NULL, head_names))
  }
  head_loss_mat     <- new_head_mat()
  head_acc_mat      <- new_head_mat()
  val_head_loss_mat <- new_head_mat()
  val_head_acc_mat  <- new_head_mat()
  epochs_run        <- 0L

  for (epoch in seq_len(nepoch)) {
    # See ggml_fit_opt(): an R-side loop re-syncs the thread count itself.
    .ggml_sched_sync_threads(sched)

    # shuffle training portion (leaves the validation tail untouched)
    if (shuffle && nbatch_logical < idata_split) {
      ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split)
    }

    ggml_opt_result_reset(result_train)
    ggml_opt_result_reset(result_eval)

    logs <- list()

    for (cb in callbacks) {
      if (is.function(cb$on_epoch_begin))
        cb$on_epoch_begin(epoch, logs, state)
      if (isTRUE(state$stop)) break
    }
    if (isTRUE(state$stop)) break

    if (!silent) message(sprintf("Epoch %d/%d", epoch, nepoch))

    cb_progress <- if (silent) FALSE else TRUE
    ggml_opt_epoch(opt_ctx, dataset, result_train, result_eval,
                   idata_split,
                   callback_train = cb_progress,
                   callback_eval  = cb_progress)

    train_loss_res <- ggml_opt_result_loss(result_train)
    train_acc_res  <- ggml_opt_result_accuracy(result_train)

    logs$train_loss     <- train_loss_res[["loss"]]
    logs$train_accuracy <- train_acc_res[["accuracy"]]

    if (val_split > 0) {
      val_loss_res <- ggml_opt_result_loss(result_eval)
      val_acc_res  <- ggml_opt_result_accuracy(result_eval)
      logs$val_loss     <- val_loss_res[["loss"]]
      logs$val_accuracy <- val_acc_res[["accuracy"]]
    } else {
      logs$val_loss     <- NA_real_
      logs$val_accuracy <- NA_real_
    }

    # Per-head metrics, for the history matrices and for `logs`, so a callback
    # can monitor an individual head. A result holds heads only once an epoch
    # has accumulated into it, so guard on its own head count rather than
    # n_head. Both phases carry an explicit prefix -- "train_<head>_loss" and
    # "val_<head>_loss" -- so a head named "val_x" cannot collide with the
    # validation key of a head named "x".
    n_res_train <- ggml_opt_result_n_loss(result_train)
    n_res_val   <- if (val_split > 0) ggml_opt_result_n_loss(result_eval) else 0L
    for (i in seq_len(n_head)) {
      nm <- head_names[i]
      if (i <= n_res_train) {
        head_loss_mat[epoch, i] <- ggml_opt_result_loss_i(result_train, i)[["loss"]]
        head_acc_mat[epoch, i]  <- ggml_opt_result_accuracy_i(result_train, i)[["accuracy"]]
      }
      # unname(): the matrices carry the head names as column names, which a
      # scalar subset would drag along into `logs` -- callbacks compare these
      # against plain numbers, so they must stay bare.
      logs[[paste0("train_", nm, "_loss")]]     <- unname(head_loss_mat[epoch, i])
      logs[[paste0("train_", nm, "_accuracy")]] <- unname(head_acc_mat[epoch, i])

      if (i <= n_res_val) {
        val_head_loss_mat[epoch, i] <- ggml_opt_result_loss_i(result_eval, i)[["loss"]]
        val_head_acc_mat[epoch, i]  <- ggml_opt_result_accuracy_i(result_eval, i)[["accuracy"]]
      }
      logs[[paste0("val_", nm, "_loss")]]     <- unname(val_head_loss_mat[epoch, i])
      logs[[paste0("val_", nm, "_accuracy")]] <- unname(val_head_acc_mat[epoch, i])
    }

    # The data frame keeps only the aggregate columns -- the per-head metrics
    # are matrices and ride as attributes -- so `hist` is built from the
    # aggregates rather than from all of `logs`.
    hist[[epoch]] <- c(epoch = epoch, logs[c("train_loss", "train_accuracy",
                                             "val_loss", "val_accuracy")])
    epochs_run    <- epoch

    if (!silent) {
      message(sprintf("  train_loss=%.4f  train_acc=%.4f  val_loss=%s  val_acc=%s",
                      logs$train_loss, logs$train_accuracy,
                      if (is.na(logs$val_loss)) "NA" else sprintf("%.4f", logs$val_loss),
                      if (is.na(logs$val_accuracy)) "NA" else sprintf("%.4f", logs$val_accuracy)))
    }

    for (cb in callbacks) {
      if (is.function(cb$on_epoch_end))
        cb$on_epoch_end(epoch, logs, state)
      if (isTRUE(state$stop)) break
    }
    if (isTRUE(state$stop)) break
  }

  filled <- Filter(Negate(is.null), hist)
  if (length(filled) == 0) {
    empty <- data.frame(epoch = integer(0), train_loss = numeric(0),
                        train_accuracy = numeric(0), val_loss = numeric(0),
                        val_accuracy = numeric(0))
    none <- matrix(NA_real_, nrow = 0, ncol = n_head,
                   dimnames = list(NULL, head_names))
    attr(empty, "head_loss")         <- none
    attr(empty, "head_accuracy")     <- none
    attr(empty, "val_head_loss")     <- none
    attr(empty, "val_head_accuracy") <- none
    attr(empty, "head_names")        <- head_names
    return(empty)
  }

  df <- do.call(rbind.data.frame, lapply(filled, function(x) as.data.frame(as.list(x))))
  # The per-head metrics are matrices, so they ride as attributes rather than
  # columns; a callback that stopped early trims them to the epochs actually run.
  keep <- seq_len(epochs_run)
  attr(df, "head_loss")         <- head_loss_mat[keep, , drop = FALSE]
  attr(df, "head_accuracy")     <- head_acc_mat[keep, , drop = FALSE]
  attr(df, "val_head_loss")     <- val_head_loss_mat[keep, , drop = FALSE]
  attr(df, "val_head_accuracy") <- val_head_acc_mat[keep, , drop = FALSE]
  attr(df, "head_names")        <- head_names
  df
}

# Internal helper: re-apply the current ggml_set_n_threads() setting to the
# scheduler's CPU backends.
#
# A backend gets the thread count when it is created, and the single C entry
# points (ggml_opt_fit, ggml_opt_fit_multi) re-sync once before their loop. An
# R-side epoch loop has no such moment, so it calls this before each epoch:
# without it, a ggml_set_n_threads() issued after the scheduler was built --
# or between building the model and training it -- would be silently ignored
# for the whole run. Cheap: it walks the scheduler's backends, a handful at
# most, once per epoch.
.ggml_sched_sync_threads <- function(sched) {
  if (is.null(sched)) return(invisible(NULL))
  invisible(.Call("R_ggml_sched_sync_threads", sched))
}

# Internal helper: how many datapoints the input tensor holds per forward pass.
#
# The batch axis cannot be read off the tensor alone. ggml_tensor_shape() always
# returns four ne[] entries, so its last element is ne[3] whatever the rank --
# right for a 4-D [W, H, C, N] input, but a dense [features, N] keeps its batch
# in ne[1] and a sequence [size, seq, N] in ne[2], both reporting ne[3] == 1.
# Falling back to ggml_n_dims() does not help either: ggml treats a trailing unit
# dimension as absent, so a batch of 1 makes [W, H, C, 1] look 3-D and the count
# would come off the channel axis.
#
# The dataset carries the one piece of information that disambiguates this --
# ne_datapoint, the size of a single sample -- so divide by it.
.ggml_input_batch_size <- function(tensor_ptr, dataset) {
  ne_datapoint <- ggml_tensor_shape(ggml_opt_dataset_data(dataset))[1]
  if (is.na(ne_datapoint) || ne_datapoint <= 0) return(1L)
  as.integer(max(1, ggml_nelements(tensor_ptr) %/% ne_datapoint))
}
