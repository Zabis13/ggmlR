# Streaming training: an R-side batch loop driven by a generator.
#
# ggml_fit_opt() hands a whole epoch to one C call (ggml_opt_epoch), which walks
# a ggml_opt_dataset holding every sample at once. That is the wrong shape for
# three cases: data larger than RAM, augmentation (which must produce a
# different batch each epoch), and generated/streaming sources such as RL
# rollouts, where there is no dataset to walk in the first place.
#
# This file drives the same optimizer context one micro-batch at a time using
# ggml_opt_alloc()/ggml_opt_eval(), the per-batch primitive ggml_opt_epoch() is
# itself built on. No dataset is created, so nothing but the current batch is
# ever resident.
#
# The FFI cost of returning to R per batch rather than per epoch is noise
# against a forward+backward pass, so the loop lives in R, where a generator is
# an ordinary closure and an error in it propagates normally.

# Internal: number of datapoints one forward pass consumes.
#
# The C side uses ne[ggml_n_dims(t) - 1] -- the last axis AFTER trailing unit
# dims are collapsed. ggml_tensor_shape() never collapses, so the trailing 1s
# have to be dropped here or a 2D [features, batch] input would report a batch
# of 1 (its ne[3]) and quietly halve or zero the gradient-accumulation period.
#
# Limited to ne[0..3]: ggml_tensor_shape() returns four entries while
# GGML_MAX_DIMS is 5, so a 5D input would need its batch passed explicitly via
# nbatch_logical. Opt graphs built by nn_build_graph() are at most 4D.
# Split in two so the rule itself is testable without a live tensor.
.gen_batch_size_of_ne <- function(ne) {
  ne <- as.numeric(ne)
  ne <- ne[!is.na(ne) & ne > 0]
  if (length(ne) == 0L) return(1L)
  ndims <- max(which(ne != 1), 1L)   # ggml_n_dims(): last axis with ne > 1
  as.integer(max(1, ne[ndims]))
}

.gen_batch_size <- function(tensor) {
  .gen_batch_size_of_ne(ggml_tensor_shape(tensor))
}

# Internal: fetch one batch and validate its shape.
#
# The generator contract is deliberately the same for finite and infinite
# sources: a call returns a batch, or NULL when exhausted. `function(i)` was
# rejected because indexed access does not exist for a queue or a replay
# buffer.
#
# A batch is list(inputs = list(...), labels = list(...)); the single-input
# case may also be given as the bare pair list(x, y), which is what the
# sequential API produces.
.gen_next_batch <- function(generator, what = "generator") {
  batch <- generator()
  if (is.null(batch)) return(NULL)
  if (!is.list(batch)) {
    stop("'", what, "' must return a list, or NULL when exhausted; got ",
         class(batch)[1], ".", call. = FALSE)
  }

  # Named form takes precedence; the bare pair is sugar for one input/one label.
  if (!is.null(batch$inputs) || !is.null(batch$labels)) {
    inputs <- batch$inputs
    labels <- batch$labels
  } else {
    if (length(batch) != 2L) {
      stop("'", what, "' must return list(inputs = , labels = ), or a plain ",
           "list(x, y) for the single-input case; got a list of length ",
           length(batch), ".", call. = FALSE)
    }
    inputs <- batch[[1L]]
    labels <- batch[[2L]]
  }

  if (!is.list(inputs)) inputs <- list(inputs)
  if (!is.list(labels)) labels <- list(labels)
  if (length(inputs) == 0L) {
    stop("'", what, "' returned a batch with no inputs.", call. = FALSE)
  }

  # The list form is the contract so that multi-head fits it later without a
  # break, but only one head is wired up so far: ggml_fit_opt_multi() gives all
  # heads one labels tensor addressed by offset, not a tensor each, and an
  # offset that is wrong converges just as smoothly as one that is right. That
  # cannot be told apart by a "loss goes down" test, so the branch waits for the
  # functional API rather than being written blind.
  if (length(inputs) > 1L || length(labels) > 1L) {
    stop("multi-head/multi-output through a generator is not supported until ",
         "the functional API step; this is a current limit, not a bug. Use ",
         "list(inputs = list(x), labels = list(y)) with a single head.",
         call. = FALSE)
  }
  list(inputs = inputs, labels = labels)
}

# Internal: run one micro-batch through the optimizer context.
#
# Order matters and is the one ggml_opt_epoch() itself uses: alloc first, then
# fill, then eval. ggml_opt_alloc() resets the gradient accumulator and swaps in
# the graph for this step, so data written before it would be reset or land in
# the wrong buffer.
#
# Gradient accumulation is NOT tracked here. opt_ctx carries opt_i internally,
# ggml_opt_eval() advances it and ggml_opt_alloc() picks GRAD or OPT from it, so
# the optimizer steps on the period boundary by itself. A second counter on the
# R side would silently step at the wrong rate rather than error.
.gen_run_step <- function(opt_ctx, batch, result, backward) {
  ggml_opt_alloc(opt_ctx, backward = backward)

  in_tensor <- ggml_opt_inputs(opt_ctx)
  ggml_backend_tensor_set_data(in_tensor, as.numeric(batch$inputs[[1L]]))

  if (length(batch$labels) >= 1L) {
    lab_tensor <- ggml_opt_labels(opt_ctx)
    if (!is.null(lab_tensor)) {
      ggml_backend_tensor_set_data(lab_tensor, as.numeric(batch$labels[[1L]]))
    }
  }

  ggml_opt_eval(opt_ctx, result)
  invisible(NULL)
}

# Internal: drive one epoch's worth of batches from a generator.
#
# Returns the number of steps actually run.
#
# NULL means "no more batches", and what that implies depends on whether the
# caller declared a length. Without `steps` the source itself decides where the
# epoch ends, so a short epoch is simply how a finite generator works. With
# `steps` the caller promised that many batches, so running out early means the
# source broke its promise -- a rollout that died, fewer shards than expected --
# and training on a truncated epoch would hide it.
.gen_run_epoch <- function(generator, opt_ctx, result, backward, steps, what,
                           epoch = NA_integer_) {
  done <- 0L
  repeat {
    if (!is.null(steps) && done >= steps) break
    batch <- .gen_next_batch(generator, what)
    if (is.null(batch)) {
      if (!is.null(steps) && done > 0L) {
        warning(sprintf(
          "epoch %s: '%s' returned NULL after %d of %d requested batches. The epoch is short; if the source can end on its own, drop 'steps_per_epoch' and let NULL close the epoch.",
          if (is.na(epoch)) "?" else as.character(epoch), what, done, steps),
          call. = FALSE)
      }
      break
    }
    .gen_run_step(opt_ctx, batch, result, backward)
    done <- done + 1L
  }
  done
}

#' Train from a generator, one batch at a time
#'
#' Streaming counterpart of \code{\link{ggml_fit_opt}}. Instead of a
#' \code{ggml_opt_dataset} holding every sample, batches are pulled from a
#' generator function, so only the current batch is ever in memory. This is what
#' makes three things possible that the dataset path cannot express: data larger
#' than RAM, augmentation that must differ per epoch, and generated sources
#' (RL rollouts, simulators) that have no fixed dataset at all.
#'
#' @section The generator contract:
#' \code{generator} is a function of no arguments. Each call returns one batch;
#' returning \code{NULL} ends the epoch. The same contract covers both finite
#' and infinite sources -- an infinite generator simply never returns
#' \code{NULL}, and \code{steps_per_epoch} then becomes required, because
#' nothing else can say where an epoch ends.
#'
#' \strong{A finite generator must restart itself.} Training runs several
#' epochs over the same source, so a generator that ends must reset its own
#' state before returning \code{NULL}, leaving the next call to yield the first
#' batch again. Nothing else can do it: a bare closure has no handle to reset
#' from. An epoch that receives no batch at all stops training with a warning
#' rather than looping over nothing.
#'
#' \preformatted{
#' batch_gen <- function(x, y, batch_size) {
#'   i <- 0L
#'   nb <- nrow(x) \%/\% batch_size
#'   function() {
#'     if (i >= nb) {
#'       i <<- 0L          # reset, so the next epoch starts over
#'       return(NULL)
#'     }
#'     idx <- (i * batch_size + 1L):((i + 1L) * batch_size)
#'     i <<- i + 1L
#'     list(x[idx, , drop = FALSE], y[idx, , drop = FALSE])
#'   }
#' }
#' }
#'
#' A batch is \code{list(inputs = list(...), labels = list(...))}. For the
#' common single-input case a plain \code{list(x, y)} is accepted as well. Each
#' element is a numeric vector or array already in ggml layout, holding exactly
#' one physical batch.
#'
#' The generator is called from R, so it may do anything R can: read a shard
#' from disk, augment the previous batch, or sample a replay buffer that a
#' simulator is filling.
#'
#' @section Gradient accumulation:
#' When \code{nbatch_logical} exceeds the graph's physical batch size, the
#' optimizer accumulates gradients over several generator batches and steps once
#' per logical batch. The period is tracked inside \code{opt_ctx}; the generator
#' is unaware of it and simply keeps producing physical batches.
#'
#' @param sched Backend scheduler
#' @param ctx_compute Compute context holding the model graph
#' @param inputs Input tensor of the built graph
#' @param outputs Output tensor of the built graph
#' @param generator Function of no arguments returning one batch, or
#'   \code{NULL} when exhausted. See the contract above.
#' @param steps_per_epoch Number of batches that make up one epoch. Required
#'   when \code{generator} is infinite; when omitted, an epoch runs until the
#'   generator returns \code{NULL}.
#' @param epochs Number of epochs to run.
#' @param initial_epoch Epoch number to start counting from, as in Keras. This
#'   shifts the numbering seen by callbacks and learning-rate schedules only;
#'   it does not restore optimizer state. A fresh \code{opt_ctx} starts with
#'   zeroed Adam moments regardless, so this is a warm restart with the LR
#'   schedule resumed, not a full resume.
#' @param validation_generator Optional generator used for validation after each
#'   epoch, following the same contract. Evaluated without a backward pass.
#' @param validation_steps Number of validation batches per epoch. Required when
#'   \code{validation_generator} is infinite.
#' @param loss_type Loss type constant
#' @param optimizer Optimizer type constant
#' @param nbatch_logical Logical batch size, for gradient accumulation
#' @param callbacks List of callback lists, as for \code{\link{ggml_fit_opt}}
#' @param silent Suppress per-epoch messages
#' @param loss_mask_ne0 Width of the per-output loss mask, or NULL
#' @return A history data frame, one row per epoch, as
#'   \code{\link{ggml_fit_opt}} returns.
#' @export
#' @family optimization
#' @seealso \code{\link{ggml_fit_opt}} for the in-memory dataset path.
ggml_fit_opt_gen <- function(sched, ctx_compute, inputs, outputs, generator,
                             steps_per_epoch = NULL,
                             epochs = 1L,
                             initial_epoch = 0L,
                             validation_generator = NULL,
                             validation_steps = NULL,
                             loss_type   = ggml_opt_loss_type_mse(),
                             optimizer   = ggml_opt_optimizer_type_adamw(),
                             nbatch_logical = NULL,
                             callbacks   = list(),
                             silent      = FALSE,
                             loss_mask_ne0 = NULL) {

  if (!is.function(generator)) {
    stop("'generator' must be a function of no arguments returning one batch, ",
         "or NULL when exhausted.", call. = FALSE)
  }
  if (!is.null(steps_per_epoch)) {
    steps_per_epoch <- as.integer(steps_per_epoch)
    if (is.na(steps_per_epoch) || steps_per_epoch < 1L) {
      stop("'steps_per_epoch' must be a positive integer.", call. = FALSE)
    }
  }
  if (!is.null(validation_generator) && !is.function(validation_generator)) {
    stop("'validation_generator' must be a function.", call. = FALSE)
  }
  if (!is.null(validation_steps)) {
    validation_steps <- as.integer(validation_steps)
    if (is.na(validation_steps) || validation_steps < 1L) {
      stop("'validation_steps' must be a positive integer.", call. = FALSE)
    }
  }
  epochs        <- as.integer(epochs)
  initial_epoch <- as.integer(initial_epoch)
  if (is.na(initial_epoch) || initial_epoch < 0L) {
    stop("'initial_epoch' must be a non-negative integer.", call. = FALSE)
  }

  # The physical batch is a property of the graph, not of a dataset -- there is
  # no dataset here to derive it from, so it comes from the input tensor. This
  # mirrors ggml_opt_batch_size(): the batch is the LAST NON-TRIVIAL axis, not
  # ne[3]. ggml_tensor_shape() always returns four entries without collapsing,
  # so a [features, batch] input reads as c(features, batch, 1, 1) and taking
  # the last element would yield 1 and silently mis-derive opt_period.
  nbatch_physical <- .gen_batch_size(inputs)

  nbatch_logical <- if (is.null(nbatch_logical)) {
    nbatch_physical
  } else {
    as.integer(max(nbatch_logical, nbatch_physical))
  }
  opt_period <- as.integer(max(1L, nbatch_logical %/% nbatch_physical))

  ctx_list <- ggml_opt_init_for_fit(
    sched, loss_type, optimizer, opt_period,
    ctx_compute, inputs, outputs, loss_mask_ne0
  )
  opt_ctx <- ctx_list$opt_ctx
  lr_ud   <- ctx_list$lr_ud
  on.exit(ggml_opt_free(opt_ctx), add = TRUE)

  result_train <- ggml_opt_result_init()
  result_eval  <- ggml_opt_result_init()
  on.exit({
    ggml_opt_result_free(result_train)
    ggml_opt_result_free(result_eval)
  }, add = TRUE)

  state <- new.env(parent = emptyenv())
  state$stop   <- FALSE
  state$lr_ud  <- lr_ud
  state$nepoch <- initial_epoch + epochs

  hist <- vector("list", epochs)

  for (k in seq_len(epochs)) {
    # Callbacks and LR schedules see the shifted number; the history rows carry
    # it too, so resuming a run continues the curve rather than restarting it.
    epoch <- initial_epoch + k

    .ggml_sched_sync_threads(sched)

    ggml_opt_result_reset(result_train)
    ggml_opt_result_reset(result_eval)

    logs <- list()

    for (cb in callbacks) {
      if (is.function(cb$on_epoch_begin)) cb$on_epoch_begin(epoch, logs, state)
      if (isTRUE(state$stop)) break
    }
    if (isTRUE(state$stop)) break

    if (!silent) message(sprintf("Epoch %d/%d", epoch, state$nepoch))

    nsteps <- .gen_run_epoch(generator, opt_ctx, result_train,
                             backward = TRUE, steps = steps_per_epoch,
                             what = "generator", epoch = epoch)
    if (nsteps == 0L) {
      # A whole epoch without a single batch means the generator did not restart
      # itself. There is nothing to train on and no way to make the next epoch
      # differ, so stop -- but say which of the two causes it was, since they
      # look identical from here.
      warning(sprintf(
        "epoch %d: 'generator' returned NULL on the first step; trained %d of %d epochs. If the source is finite and now exhausted, lower 'epochs'. If it was meant to restart, reset its state before returning NULL.",
        epoch, k - 1L, epochs), call. = FALSE)
      break
    }

    train_loss_res <- ggml_opt_result_loss(result_train)
    train_acc_res  <- ggml_opt_result_accuracy(result_train)
    logs$train_loss     <- train_loss_res[["loss"]]
    logs$train_accuracy <- train_acc_res[["accuracy"]]

    if (!is.null(validation_generator)) {
      nval <- .gen_run_epoch(validation_generator, opt_ctx, result_eval,
                             backward = FALSE, steps = validation_steps,
                             what = "validation_generator")
      if (nval > 0L) {
        val_loss_res <- ggml_opt_result_loss(result_eval)
        val_acc_res  <- ggml_opt_result_accuracy(result_eval)
        logs$val_loss     <- val_loss_res[["loss"]]
        logs$val_accuracy <- val_acc_res[["accuracy"]]
      } else {
        logs$val_loss     <- NA_real_
        logs$val_accuracy <- NA_real_
      }
    } else {
      logs$val_loss     <- NA_real_
      logs$val_accuracy <- NA_real_
    }

    logs$steps <- nsteps
    hist[[k]] <- c(epoch = epoch, logs)

    if (!silent) {
      message(sprintf("  steps=%d  train_loss=%.4f  train_acc=%.4f  val_loss=%s  val_acc=%s",
                      nsteps, logs$train_loss, logs$train_accuracy,
                      if (is.na(logs$val_loss)) "NA" else sprintf("%.4f", logs$val_loss),
                      if (is.na(logs$val_accuracy)) "NA" else sprintf("%.4f", logs$val_accuracy)))
    }

    for (cb in callbacks) {
      if (is.function(cb$on_epoch_end)) cb$on_epoch_end(epoch, logs, state)
      if (isTRUE(state$stop)) break
    }
    if (isTRUE(state$stop)) break
  }

  filled <- Filter(Negate(is.null), hist)
  if (length(filled) == 0) {
    return(data.frame(epoch = integer(0), train_loss = numeric(0),
                      train_accuracy = numeric(0), val_loss = numeric(0),
                      val_accuracy = numeric(0), steps = integer(0)))
  }
  do.call(rbind.data.frame, lapply(filled, function(x) as.data.frame(as.list(x))))
}

# ============================================================================
# Sequential API wrapper
# ============================================================================

# Internal: R-layout batch -> ggml layout, the same transposition
# ggml_fit_sequential() applies to a whole dataset.
#
# The user's generator yields data in R layout ([N, ...], sample-major) because
# that is what every other entry point in the package takes; the column-major
# reordering is an implementation detail of ggml and stays here. This is also
# what lets the batch be handed to nn_bn_calibrate() unchanged.
.gen_to_ggml_layout <- function(x, input_shape) {
  if (length(input_shape) == 3L) {
    as.vector(aperm(x, c(3, 2, 4, 1)))   # [N,H,W,C] -> [W,H,C,N]
  } else if (length(input_shape) == 2L) {
    as.vector(aperm(x, c(3, 2, 1)))      # [N,seq,feat] -> [feat,seq,N]
  } else if (length(input_shape) == 1L) {
    as.vector(t(x))                      # [N,feat] -> [feat,N]
  } else {
    stop("Unsupported input_shape length: ", length(input_shape), call. = FALSE)
  }
}

# Internal: wrap a user generator so it yields ggml-layout batches, and
# optionally record the R-layout inputs for batch-norm calibration.
#
# `recorder` is an environment; when non-NULL each batch's R-layout x is pushed
# onto it, capped at the number of samples nn_bn_calibrate() actually uses.
.gen_adapt_sequential <- function(generator, input_shape, batch_size,
                                  what = "generator", recorder = NULL) {
  function() {
    batch <- generator()
    if (is.null(batch)) return(NULL)
    if (!is.list(batch) || length(batch) < 2L) {
      stop("'", what, "' must return list(x, y), or NULL when exhausted.",
           call. = FALSE)
    }
    x <- batch[[1L]]
    y <- batch[[2L]]

    n <- if (is.matrix(x) || is.array(x)) dim(x)[1] else length(x)
    if (!identical(as.integer(n), as.integer(batch_size))) {
      stop("'", what, "' returned a batch of ", n, " sample(s) but the graph ",
           "was built for exactly ", batch_size, ". A generator must yield ",
           "full batches; drop or pad the remainder yourself.", call. = FALSE)
    }
    if (!is.matrix(y) && !is.array(y)) y <- matrix(y, nrow = n)

    if (!is.null(recorder) && recorder$n < recorder$cap) {
      recorder$parts[[length(recorder$parts) + 1L]] <- x
      recorder$n <- recorder$n + n
    }

    list(inputs = list(.gen_to_ggml_layout(x, input_shape)),
         labels = list(as.vector(t(y))))
  }
}

#' Train a Sequential Model from a Generator
#'
#' Streaming counterpart of \code{\link{ggml_fit}} for sequential models:
#' batches are pulled from a generator instead of being sliced out of a matrix
#' held in memory. Reachable either directly or as
#' \code{ggml_fit(model, generator = ...)}.
#'
#' Batches are given in the same R layout that \code{x} and \code{y} use; the
#' column-major conversion ggml wants happens internally, which is also what
#' lets a recorded batch feed batch-norm calibration unchanged.
#'
#' @param model A compiled \code{ggml_sequential_model}
#' @param epochs Number of epochs to run
#' @param batch_size Samples per batch. The graph is built for a fixed batch, so
#'   the generator must yield exactly this many; a short batch is an error
#'   rather than being padded.
#' @param verbose 0 to silence per-epoch messages
#' @param callbacks List of callback lists, as for \code{\link{ggml_fit}}
#' @return The trained model, invisibly.
#' @seealso \code{\link{ggml_fit_opt_gen}} for the low-level loop this wraps.
#' @param generator Function of no arguments yielding \code{list(x, y)} for one
#'   batch, or \code{NULL} to end the epoch. Mutually exclusive with \code{x}/
#'   \code{y}. Batches are in the same R layout those arguments take, and must
#'   contain exactly \code{batch_size} samples. A finite generator has to reset
#'   its own state before returning \code{NULL}, so that the next epoch starts
#'   over; see \code{\link{ggml_fit_opt_gen}} for the full contract and an
#'   example.
#' @param steps_per_epoch Batches per epoch. Required for a generator that never
#'   returns \code{NULL}; otherwise the epoch ends when it does.
#' @param initial_epoch Epoch number to start counting from. Shifts the
#'   numbering seen by callbacks and learning-rate schedules; it does not
#'   restore optimizer state (see \code{\link{ggml_fit_opt_gen}}).
#' @param validation_generator Optional generator for validation, same contract.
#' @param validation_steps Validation batches per epoch.
#' @param calibration_generator Generator used to recompute batch-norm running
#'   statistics after training. Defaults to \code{generator}; note that a finite
#'   generator already exhausted by training yields nothing, so pass a fresh one.
#' @param calibration_steps Batches to draw for that calibration.
#' @export
ggml_fit_generator <- function(model, generator, steps_per_epoch = NULL,
                               epochs = 1L, batch_size = 32L,
                               initial_epoch = 0L,
                               validation_generator = NULL,
                               validation_steps = NULL,
                               calibration_generator = NULL,
                               calibration_steps = NULL,
                               verbose = 1, callbacks = list()) {
  if (!inherits(model, "ggml_sequential_model")) {
    stop("ggml_fit_generator() currently supports sequential models; the ",
         "functional API is a later step.", call. = FALSE)
  }
  if (!model$compiled) {
    stop("Model must be compiled before training. Call ggml_compile() first.",
         call. = FALSE)
  }

  batch_size  <- as.integer(batch_size)
  input_shape <- model$input_shape

  use_ce_loss <- model$compilation$loss %in%
    c("categorical_crossentropy", "crossentropy", "cross_entropy")
  graph_info <- nn_build_graph(model, batch_size, logits_output = use_ce_loss)

  optimizer_type <- switch(model$compilation$optimizer,
    "adam" = , "adamw" = ggml_opt_optimizer_type_adamw(),
    "sgd" = ggml_opt_optimizer_type_sgd(),
    stop("Unsupported optimizer: ", model$compilation$optimizer, call. = FALSE)
  )
  loss_type <- nn_loss_type_of(model$compilation$loss)

  # Batch-norm statistics have to come from a forward pass over real data with
  # the final weights, and there is no dataset here to replay -- so the batches
  # are recorded as they stream past, up to what nn_bn_calibrate() would use
  # anyway. Only worth doing when the model actually has a batch_norm layer.
  has_bn <- any(vapply(model$layers,
                       function(l) identical(l$type, "batch_norm"), logical(1)))
  recorder <- NULL
  if (has_bn && is.null(calibration_generator) && is.null(calibration_steps)) {
    recorder <- new.env(parent = emptyenv())
    recorder$parts <- list()
    recorder$n     <- 0L
    recorder$cap   <- 1024L
  }

  gen_wrapped <- .gen_adapt_sequential(generator, input_shape, batch_size,
                                       "generator", recorder)
  val_wrapped <- if (is.null(validation_generator)) NULL else {
    .gen_adapt_sequential(validation_generator, input_shape, batch_size,
                          "validation_generator")
  }

  history_raw <- ggml_fit_opt_gen(
    sched = model$compilation$sched,
    ctx_compute = graph_info$ctx_compute,
    inputs  = graph_info$inputs,
    outputs = graph_info$outputs,
    generator = gen_wrapped,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    initial_epoch = initial_epoch,
    validation_generator = val_wrapped,
    validation_steps = validation_steps,
    loss_type = loss_type,
    optimizer = optimizer_type,
    nbatch_logical = batch_size,
    callbacks = callbacks,
    silent = (verbose == 0)
  )

  model$layers <- graph_info$layers_built
  model$compilation$ctx_weights <- graph_info$ctx_weights
  model$compilation$buffer <- graph_info$buffer

  if (has_bn) {
    x_cal <- .gen_calibration_data(calibration_generator, calibration_steps,
                                   generator, recorder, batch_size)
    if (!is.null(x_cal)) model <- nn_bn_calibrate(model, x_cal)
  }

  model$history <- structure(
    list(
      train_loss     = history_raw$train_loss,
      train_accuracy = history_raw$train_accuracy,
      val_loss       = history_raw$val_loss,
      val_accuracy   = history_raw$val_accuracy,
      epochs         = history_raw$epoch
    ),
    class = "ggml_history"
  )

  ggml_free(graph_info$ctx_compute)
  invisible(model)
}

# Internal: assemble the R-layout inputs used to recalibrate batch norm.
#
# An explicit calibration_generator wins; otherwise the batches recorded during
# training are used. A finite training generator is typically exhausted by now,
# which is why the recorded path exists at all -- calling it again would yield
# NULL immediately.
.gen_calibration_data <- function(calibration_generator, calibration_steps,
                                  generator, recorder, batch_size) {
  gen <- calibration_generator
  if (is.null(gen) && !is.null(calibration_steps)) gen <- generator

  if (!is.null(gen)) {
    steps <- if (is.null(calibration_steps)) {
      max(1L, ceiling(1024L / batch_size))
    } else {
      as.integer(calibration_steps)
    }
    parts <- list()
    for (i in seq_len(steps)) {
      b <- gen()
      if (is.null(b)) break
      parts[[length(parts) + 1L]] <- b[[1L]]
    }
    if (length(parts) == 0L) return(NULL)
    return(.gen_bind_parts(parts))
  }

  if (!is.null(recorder) && length(recorder$parts) > 0L) {
    return(.gen_bind_parts(recorder$parts))
  }
  NULL
}

# Internal: concatenate batches along the sample axis, matrix or array alike.
.gen_bind_parts <- function(parts) {
  if (length(parts) == 1L) return(parts[[1L]])
  if (is.matrix(parts[[1L]])) {
    do.call(rbind, parts)
  } else {
    Reduce(abind_first, parts)
  }
}
