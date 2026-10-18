# A long-lived training context you drive yourself.
#
# ggml_fit() and ggml_fit_generator() both own their optimizer context: it is
# created inside the call and freed on the way out. That is the right shape for
# "train this model on this data", but it makes a whole class of loops
# unexpressible -- reinforcement learning above all, where there is no epoch,
# the data comes from a replay buffer the agent is still filling, and rollout
# and optimisation run at different rates.
#
# Calling ggml_fit() once per batch does not get around it: every call builds a
# fresh opt_ctx, so the Adam moments and the bias-correction counter start from
# zero each time, and every step is Adam's first -- roughly lr * sign(g).
#
# The per-step primitive for that already exists (ggml_opt_alloc/ggml_opt_eval);
# what was missing is a context that outlives a single call, plus the graph
# building that has to happen before it. ggml_trainer() supplies both, so a
# custom loop needs nothing from ggmlR:::.

# Internal: R-layout batch -> the flat ggml-layout vectors the input, label and
# loss-weight tensors want. The input layout is passed in as a function, so the
# sequential path shares .gen_to_ggml_layout() with the generator and the
# functional path shares nn_prepare_x() with ggml_fit().
.trainer_prepare <- function(x, y, w, to_ggml, batch_size, w_ne0) {
  n <- if (is.matrix(x) || is.array(x)) dim(x)[1] else length(x)
  if (!identical(as.integer(n), as.integer(batch_size))) {
    stop("this trainer was built for batches of exactly ", batch_size,
         " sample(s); got ", n, ". Rebuild with a different batch_size, or ",
         "pad the batch yourself.", call. = FALSE)
  }
  if (!is.null(y) && !is.matrix(y) && !is.array(y)) y <- matrix(y, nrow = n)

  w_ggml <- NULL
  if (is.null(w_ne0)) {
    if (!is.null(w)) {
      stop("'sample_weight' given, but this trainer was built without ",
           "'loss_mask'; rebuild with loss_mask = 1 (one weight per sample) ",
           "or loss_mask = <outputs> (one per output).", call. = FALSE)
    }
  } else if (is.null(w)) {
    # A weighted loss always reads its weights tensor; ones make it plain MSE.
    w_ggml <- rep(1, w_ne0 * n)
  } else {
    if (w_ne0 == 1L && is.null(dim(w))) w <- matrix(w, ncol = 1L)
    if (!is.matrix(w) || nrow(w) != n || ncol(w) != w_ne0) {
      stop("'sample_weight' must be a ", n, " x ", w_ne0, " matrix (",
           "samples x ", if (w_ne0 == 1L) "1" else "outputs", ").",
           call. = FALSE)
    }
    w_ggml <- as.numeric(t(w))
  }

  list(x = to_ggml(x),
       y = if (is.null(y)) NULL else as.vector(t(y)),
       w = w_ggml)
}

#' A Training Context You Step Yourself
#'
#' Builds the graph and optimizer context for a compiled model and hands them
#' back, so a training loop can be written in R rather than handed to
#' \code{\link{ggml_fit}}. The context lives until \code{$free()}, keeping
#' weights, Adam moments and the compute graph across as many steps as the loop
#' takes.
#'
#' This is the level below \code{\link{ggml_fit_opt_gen}}. Reach for it when the
#' loop is not a pass over data: reinforcement learning, where experience comes
#' from a replay buffer the agent is still filling and a target network is
#' refreshed every so many steps; curriculum schemes that change the data
#' distribution as they go; data generated fresh for every step; or anything
#' where "epoch" is not the unit of progress.
#'
#' Do not emulate this by calling \code{\link{ggml_fit}} once per batch: each
#' such call builds a new optimizer context, so the Adam moments are reset and
#' every step is the optimizer's first.
#'
#' @section Using it:
#' \preformatted{
#' tr <- ggml_trainer(model, batch_size = 32L)
#' on.exit(tr$free())
#'
#' for (i in seq_len(10000L)) {
#'   b <- sample_batch(replay, 32L)     # your own bookkeeping
#'   loss <- tr$step(b$x, b$y)
#'   if (i \%\% 500L == 0L) target <- tr$model()   # refresh a target network
#' }
#'
#' model <- tr$model()                  # weights back into an R model object
#' tr$free()
#' }
#'
#' @section What a step does:
#' \code{$step(x, y, sample_weight)} runs one forward and backward pass and
#' returns that step's loss. Gradient accumulation, if \code{nbatch_logical}
#' asked for it, is tracked inside the context: the optimizer updates the
#' weights only on the period boundary, and the intervening steps only
#' accumulate. \code{$eval(x, y, sample_weight)} is the same without the
#' backward pass, for measuring on held-out data.
#'
#' @section Loss weights and masks:
#' With \code{loss_mask} set, the loss is weighted mean squared error,
#' \code{sum(w * (pred - y)^2) / (batch * outputs)}, and \code{$step()} and
#' \code{$eval()} take a \code{sample_weight} matrix:
#' \itemize{
#'   \item \code{loss_mask = 1}: \code{sample_weight} is \code{[batch, 1]} (or
#'     a vector of length \code{batch}) -- one weight per sample, as in Keras.
#'   \item \code{loss_mask = <outputs>}: \code{sample_weight} is
#'     \code{[batch, outputs]} -- one weight per output element. This is
#'     \emph{not} Keras' \code{sample_weight}: a 0/1 matrix is a mask that
#'     trains only the outputs whose label is known (e.g. one action of a
#'     Q-network), and a 0 removes that element's gradient exactly. The loss is
#'     not renormalised by the number of active elements.
#' }
#' Omitting \code{sample_weight} on a weighted trainer means all ones. Only
#' \code{loss = "mse"} supports weights.
#'
#' @section Models:
#' Sequential models and functional models (\code{\link{ggml_model}}) with
#' \strong{one input and one output} are supported. Functional models with
#' several inputs or outputs are refused. For a functional model with dropout
#' the dropout masks are redrawn before every \code{$step()}.
#'
#' @section Predicting during training:
#' \code{$predict(x, batch_size)} returns predictions from the current weights,
#' as \code{\link{ggml_predict}} would, in eval mode (no dropout). It runs on a
#' scheduler of its own, created on first use: the weights are copied into a
#' separate buffer for the call and released after it, so neither the training
#' graph's buffers nor the optimizer state are touched. \code{batch_size}
#' defaults to the trainer's; \code{x} may have any number of rows.
#'
#' @section Target networks:
#' \code{$snapshot()} returns an independent copy of the model: its own weight
#' tensors in its own buffer and its own scheduler, sharing nothing mutable with
#' the trainer. Later steps do not move it, \code{\link{ggml_predict}} on it
#' never touches the training path, and it outlives \code{$free()}. It is built
#' in eval mode, so dropout is off in it. Release it with
#' \code{\link{ggml_snapshot_free}} (a finaliser is only a backstop); after that
#' \code{ggml_predict} and \code{$update_target} on it are refused with an error.
#'
#' \code{$update_target(target, tau = 1)} moves a snapshot towards the current
#' weights, \eqn{\theta' \leftarrow \tau\theta + (1 - \tau)\theta'}:
#' \code{tau = 1} is the hard copy of DQN, a small \code{tau} the Polyak
#' average of DDPG/SAC. Non-trainable buffers (a sequential batch-norm's
#' running mean and variance) are copied whole, never averaged. The target
#' must have the same weights, by name and shape, or it is refused. The update
#' goes through host memory, one read and one write per tensor.
#'
#' @section Lifetime:
#' The context holds GPU buffers and must be released with \code{$free()};
#' calling it twice is harmless. A trainer that goes out of scope without it is
#' cleaned up when garbage collected, but that moment is not predictable, so
#' free it yourself. After \code{$free()} every other method errors rather than
#' touching released memory.
#'
#' The model object passed in is unchanged -- R semantics being what they are,
#' the trainer works on its own copy of the weights. \code{$model()} returns a
#' model carrying the current ones.
#'
#' The returned model shares the trainer's backend scheduler, and predicting
#' with it re-lays the scheduler's buffers for another graph. The trainer
#' therefore re-allocates its own graph on the first step after any
#' \code{$model()} call; weights and Adam moments are kept. \code{$predict()}
#' needs none of this. Predicting with the
#' \emph{original} model object while the trainer is alive is not covered by
#' this and can crash the session on the next step -- take the model from
#' \code{$model()} instead.
#'
#' @param model A compiled \code{ggml_sequential_model}, or a compiled
#'   functional model with one input and one output
#' @param batch_size Samples per step. The graph is built for exactly this many;
#'   every batch handed to \code{$step()} must match.
#' @param nbatch_logical Logical batch size for gradient accumulation. Defaults
#'   to \code{batch_size}, meaning every step updates the weights.
#' @param loss_mask \code{NULL} (default) for the compiled loss unweighted;
#'   \code{1} for one weight per sample; the number of outputs for one weight
#'   per output element. See the section on loss weights.
#' @param weight_decay Decoupled weight decay in \code{[0, 1]}. \code{NULL}
#'   (default) takes the value given to \code{\link{ggml_compile}}. Frozen
#'   weights are not decayed.
#' @return An object of class \code{ggml_trainer}: a list of functions
#'   \code{$step(x, y, sample_weight = NULL)},
#'   \code{$eval(x, y, sample_weight = NULL)},
#'   \code{$predict(x, batch_size = NULL, matmul_precision = NULL)}, \code{$snapshot()},
#'   \code{$update_target(target, tau = 1)}, \code{$model()},
#'   \code{$calibrate_bn(x)}, \code{$set_lr(lr)}, \code{$n_steps()}, \code{$free()}.
#' @section Batch norm:
#'   A step normalizes each batch by its own statistics and does not update the
#'   running ones a sequential \code{batch_norm} predicts with; \code{ggml_fit()}
#'   recalibrates them after training, a trainer does not. Call
#'   \code{$calibrate_bn(x)} with representative inputs (up to 1024 rows are
#'   used) before predicting, snapshotting or saving -- or fix the statistics
#'   with \code{\link{ggml_freeze_bn_stats}}. Creating a trainer for a model
#'   with unfixed batch-norm layers warns once.
#' @export
#' @family optimization
#' @seealso \code{\link{ggml_fit_opt_gen}} for a generator-driven epoch loop,
#'   \code{\link{ggml_fit}} for the ordinary in-memory path.
#' @examples
#' \donttest{
#' ggml_set_n_threads(1L)  # deterministic, single OpenMP pool
#' n <- 64
#' x <- matrix(runif(n * 4), nrow = n, ncol = 4)
#' y <- matrix(0, nrow = n, ncol = 2)
#' for (i in seq_len(n)) { y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1 }
#'
#' model <- ggml_model_sequential() |>
#'   ggml_layer_dense(8, activation = "relu") |>
#'   ggml_layer_dense(2, activation = "softmax")
#' model$input_shape <- 4L
#' model <- ggml_compile(model, optimizer = "adam",
#'                       loss = "categorical_crossentropy")
#'
#' tr <- ggml_trainer(model, batch_size = 16L)
#' for (i in 1:4) {
#'   idx <- ((i - 1L) * 16L + 1L):(i * 16L)
#'   tr$step(x[idx, , drop = FALSE], y[idx, , drop = FALSE])
#' }
#' model <- tr$model()
#' tr$free()
#' }
ggml_trainer <- function(model, batch_size = 32L, nbatch_logical = NULL,
                         loss_mask = NULL, weight_decay = NULL) {
  if (!is.null(weight_decay)) nn_check_weight_decay(weight_decay)
  is_seq <- inherits(model, "ggml_sequential_model")
  is_fun <- inherits(model, "ggml_functional_model")
  if (!is_seq && !is_fun) {
    stop("ggml_trainer() supports sequential models and functional models ",
         "with one input and one output.", call. = FALSE)
  }
  if (!isTRUE(model$compiled)) {
    stop("Model must be compiled before training. Call ggml_compile() first.",
         call. = FALSE)
  }
  if (is_fun && (length(model$inputs) != 1L || length(model$outputs) != 1L)) {
    stop("ggml_trainer() supports functional models with one input and one ",
         "output; this one has ", length(model$inputs), " input(s) and ",
         length(model$outputs), " output(s).", call. = FALSE)
  }

  batch_size <- as.integer(batch_size)
  batch_size_tr <- batch_size

  loss <- model$compilation$loss
  if (is.list(loss)) loss <- loss[[1L]]
  is_mse <- loss %in% c("mse", "mean_squared_error")

  w_ne0 <- NULL
  if (!is.null(loss_mask)) {
    w_ne0 <- as.integer(loss_mask)
    if (length(w_ne0) != 1L || is.na(w_ne0) || w_ne0 < 1L) {
      stop("'loss_mask' must be a positive integer: 1, or the number of ",
           "outputs.", call. = FALSE)
    }
    if (!is_mse) {
      stop("'loss_mask' needs loss = \"mse\"; this model was compiled with '",
           loss, "'.", call. = FALSE)
    }
  }

  use_ce_loss <- is.null(w_ne0) &&
    loss %in% c("categorical_crossentropy", "crossentropy", "cross_entropy")

  if (is_seq && any(vapply(model$layers, function(l)
        identical(l$type, "batch_norm") && !isTRUE(l$config$fixed_stats),
        logical(1))))
    warning("this model has batch_norm layers whose running statistics a trainer ",
            "does not update; call $calibrate_bn(x) before predicting or saving, ",
            "or fix them with ggml_freeze_bn_stats()", call. = FALSE)

  if (is_seq) {
    graph_info  <- nn_build_graph(model, batch_size, logits_output = use_ce_loss)
    in_tensor   <- graph_info$inputs
    out_tensor  <- graph_info$outputs
    input_shape <- model$input_shape
    vocab       <- nn_index_vocab(model)
    to_ggml     <- function(x) .gen_to_ggml_layout(x, input_shape, vocab)
  } else {
    graph_info <- nn_build_functional_graph(model, batch_size, training = TRUE,
                                            logits_output = use_ce_loss)
    # A model from ggml_load_model() carries its weights as R vectors that win
    # over any tensors at every graph build. The build above has just copied
    # them into the trainer's tensors; left in place they would shadow those
    # tensors from then on, and $predict(), $snapshot() and $model() would all
    # rebuild from the weights as loaded, not as trained. One source of truth.
    model$node_weights_data <- NULL
    in_tensor  <- graph_info$inputs[[1L]]
    out_tensor <- graph_info$outputs[[1L]]
    to_ggml    <- function(x) nn_prepare_x(model, x)$x_ggml
  }

  # ggml asserts on a mask width that matches neither form, which would abort
  # the R session; check it here instead.
  if (!is.null(w_ne0)) {
    n_out <- as.integer(ggml_tensor_shape(out_tensor)[1L])
    if (w_ne0 != 1L && w_ne0 != n_out) {
      ggml_free(graph_info$ctx_compute)
      stop("'loss_mask' must be 1 or the number of outputs (", n_out,
           "); got ", w_ne0, ".", call. = FALSE)
    }
  }

  optimizer_type <- switch(model$compilation$optimizer,
    "adam" = , "adamw" = ggml_opt_optimizer_type_adamw(),
    "sgd" = ggml_opt_optimizer_type_sgd(),
    stop("Unsupported optimizer: ", model$compilation$optimizer, call. = FALSE)
  )
  loss_type <- if (is.null(w_ne0)) {
    nn_loss_type_of(loss)
  } else {
    ggml_opt_loss_type_weighted_mse()
  }

  nbatch_physical <- .gen_batch_size(in_tensor)
  nbatch_logical  <- if (is.null(nbatch_logical)) {
    nbatch_physical
  } else {
    as.integer(max(nbatch_logical, nbatch_physical))
  }
  opt_period <- as.integer(max(1L, nbatch_logical %/% nbatch_physical))

  ctx_list <- ggml_opt_init_for_fit(
    model$compilation$sched, loss_type, optimizer_type, opt_period,
    graph_info$ctx_compute, in_tensor, out_tensor, w_ne0,
    weight_decay = weight_decay %||% model$compilation$weight_decay %||% 0
  )

  # One result object, reset per step: ggml_opt_result_* accumulate over
  # whatever is fed into them, so a step's own loss needs a clean slate each
  # time rather than a running epoch total.
  result <- ggml_opt_result_init()

  # Everything mutable lives here so the closures below share one state and
  # $free() can be made idempotent.
  st <- new.env(parent = emptyenv())
  st$opt_ctx    <- ctx_list$opt_ctx
  st$lr_ud      <- ctx_list$lr_ud
  st$result     <- result
  st$graph      <- graph_info
  st$model      <- model
  st$alive      <- TRUE
  st$nstep      <- 0L
  st$threads_ok <- FALSE
  st$handed_out <- FALSE   # $model() called since the last step
  st$pred_sched <- NULL    # $predict()'s own scheduler, made on first use

  check_alive <- function() {
    if (!isTRUE(st$alive)) {
      stop("this trainer has been freed; build a new one with ggml_trainer().",
           call. = FALSE)
    }
  }

  run <- function(x, y, w, backward) {
    check_alive()
    b <- .trainer_prepare(x, y, w, to_ggml, batch_size, w_ne0)

    # A ggml_set_n_threads() issued after the scheduler was built is picked up
    # once, on the first step -- the single C entry points sync before their own
    # loop, and a hand-written loop has no equivalent moment.
    if (!isTRUE(st$threads_ok)) {
      .ggml_sched_sync_threads(model$compilation$sched)
      st$threads_ok <- TRUE
    }

    # Fresh dropout masks per training step, as ggml_fit() draws them per epoch.
    if (backward) {
      for (dm in st$graph$dropout_masks) {
        ggml_backend_tensor_set_data(
          dm$mask, as.numeric(runif(dm$ne) < 1.0 - dm$rate))
      }
    }

    # A model handed out by $model() may have run a predict on the shared
    # scheduler, leaving it laid out for that graph while opt_ctx still believes
    # its own is allocated. Make the next alloc start over.
    if (isTRUE(st$handed_out)) {
      .Call("R_ggml_opt_invalidate_alloc", st$opt_ctx)
      st$handed_out <- FALSE
    }

    ggml_opt_result_reset(st$result)
    .gen_run_step(st$opt_ctx,
                  list(inputs = list(b$x), labels = list(b$y), weights = b$w),
                  st$result, backward)
    if (backward) st$nstep <- st$nstep + 1L

    ggml_opt_result_loss(st$result)[["loss"]]
  }

  free_all <- function() {
    if (!isTRUE(st$alive)) return(invisible(NULL))
    st$alive <- FALSE
    ggml_opt_result_free(st$result)
    ggml_opt_free(st$opt_ctx)
    if (!is.null(st$pred_sched)) ggml_backend_sched_free(st$pred_sched)
    # ctx_weights and buffer are deliberately left alone: they hold the trained
    # weights, which $model() hands to a model object that outlives the trainer.
    # Only the compute graph is ours to release, exactly as ggml_fit() does.
    ggml_free(st$graph$ctx_compute)
    invisible(NULL)
  }

  # A scheduler on the model's backend, sized as ggml_compile() sizes it.
  new_sched <- function() {
    if (is_seq) {
      ggml_backend_sched_new(list(model$compilation$backend), parallel = FALSE)
    } else {
      ggml_backend_sched_new(list(model$compilation$backend), parallel = FALSE,
                             graph_size = nn_functional_graph_size(model))
    }
  }

  # The model as it stands, carrying the live weight tensors.
  current_model <- function() {
    m <- st$model
    if (is_seq) {
      m$layers <- st$graph$layers_built
    } else {
      m$node_weights <- st$graph$node_weights
    }
    m$compilation$ctx_weights <- st$graph$ctx_weights
    m$compilation$buffer <- st$graph$buffer
    m
  }

  # A forgotten $free() should not strand GPU buffers for the life of the
  # session. The finaliser runs on the state environment, so it fires once the
  # trainer is unreachable; it is a backstop, not a substitute for $free(),
  # since gc() timing is not predictable.
  reg.finalizer(st, function(e) free_all(), onexit = TRUE)

  structure(list(
    step = function(x, y, sample_weight = NULL)
      run(x, y, sample_weight, backward = TRUE),

    eval = function(x, y, sample_weight = NULL)
      run(x, y, sample_weight, backward = FALSE),

    model = function() {
      check_alive()
      st$handed_out <- TRUE
      current_model()
    },

    # Running statistics of a sequential batch_norm from x, into the live
    # weights -- the step never updates them. Runs on the predict scheduler so
    # the training graph is left as it was.
    calibrate_bn = function(x) {
      check_alive()
      if (!is_seq)
        stop("$calibrate_bn(): a functional model has no batch-norm statistics ",
             "(its batch_norm is per-sample RMS normalization)", call. = FALSE)
      if (is.null(st$pred_sched)) st$pred_sched <- new_sched()
      m <- current_model()
      m$compilation$sched <- st$pred_sched
      nn_bn_calibrate(m, x)
      invisible(NULL)
    },

    predict = function(x, batch_size = NULL, matmul_precision = NULL) {
      check_alive()
      if (is.null(st$pred_sched)) st$pred_sched <- new_sched()
      # A local copy: the scheduler is swapped on it alone, never on a model
      # that $model() has handed out.
      m <- current_model()
      m$compilation$sched <- st$pred_sched
      bs <- if (is.null(batch_size)) batch_size_tr else as.integer(batch_size)
      # Fewer rows than a batch: the functional predict refuses that outright
      # and pads only a short LAST batch, so the batch shrinks to the rows.
      n  <- if (is.null(dim(x))) 1L else dim(x)[1L]   # a bare vector is one row
      ggml_predict(m, x, batch_size = max(1L, min(bs, as.integer(n))),
                   matmul_precision = matmul_precision)
    },

    set_lr = function(lr) {
      check_alive()
      # ggml_opt_set_lr() takes one rate per optimizer kind and ignores the
      # other; setting the one this trainer was built with keeps the caller from
      # having to know which that is.
      if (identical(optimizer_type, ggml_opt_optimizer_type_sgd())) {
        ggml_opt_set_lr(st$lr_ud, sgd_lr = lr)
      } else {
        ggml_opt_set_lr(st$lr_ud, adamw_lr = lr)
      }
      invisible(NULL)
    },

    snapshot = function() {
      check_alive()
      # Building an inference graph from the live model copies every weight
      # into fresh tensors in a buffer of their own; the compute graph itself
      # is not needed and goes at once.
      m  <- current_model()
      gi <- if (is_seq) {
        nn_build_graph(m, batch_size_tr, training = FALSE)
      } else {
        nn_build_functional_graph(m, batch_size_tr, training = FALSE)
      }
      ggml_free(gi$ctx_compute)
      if (is_seq) m$layers <- gi$layers_built else m$node_weights <- gi$node_weights
      m$compilation$ctx_weights <- gi$ctx_weights
      m$compilation$buffer      <- gi$buffer
      m$compilation$sched       <- new_sched()
      .snapshot_attach(m)
    },

    update_target = function(target, tau = 1) {
      check_alive()
      tau <- as.numeric(tau)
      if (length(tau) != 1L || is.na(tau) || tau < 0 || tau > 1) {
        stop("'tau' must be a single number in [0, 1].", call. = FALSE)
      }
      if (is.null(target$snapshot_state) || !isTRUE(target$snapshot_state$alive)) {
        stop("'target' must be a live snapshot made by $snapshot().",
             call. = FALSE)
      }
      src <- .weight_tensors(current_model())
      dst <- .weight_tensors(target)
      if (!identical(names(src), names(dst)) ||
          !identical(lapply(src, ggml_tensor_shape),
                     lapply(dst, ggml_tensor_shape))) {
        stop("'target' does not have the same weights (by name and shape) ",
             "as this trainer's model.", call. = FALSE)
      }
      if (tau == 0) return(invisible(target))
      is_buffer <- grepl("/running_(mean|var)$", names(src))
      for (k in seq_along(src)) {
        s <- ggml_backend_tensor_get_data(src[[k]])
        if (tau < 1 && !is_buffer[k]) {
          s <- tau * s + (1 - tau) * ggml_backend_tensor_get_data(dst[[k]])
        }
        ggml_backend_tensor_set_data(dst[[k]], s)
      }
      invisible(target)
    },

    n_steps = function() st$nstep,

    free = free_all
  ), class = "ggml_trainer")
}

#' @export
print.ggml_trainer <- function(x, ...) {
  cat("<ggml_trainer>", x$n_steps(), "step(s) taken\n")
  invisible(x)
}

# Internal: every weight tensor of a built model, flattened to a named list in a
# fixed order ("<node or layer>/<field>"). A tensor shared between layers is
# listed once, under its first name -- averaging it twice would be wrong.
.weight_tensors <- function(m) {
  roots <- if (inherits(m, "ggml_sequential_model")) {
    stats::setNames(lapply(m$layers, function(l) l$weights),
                    sprintf("%04d", seq_along(m$layers)))
  } else {
    m$node_weights
  }
  out  <- list()
  seen <- list()
  walk <- function(v, path) {
    if (typeof(v) == "externalptr") {
      for (p in seen) if (identical(p, v)) return(invisible(NULL))
      seen[[length(seen) + 1L]] <<- v
      out[[path]] <<- v
    } else if (is.list(v)) {
      nms <- if (is.null(names(v))) as.character(seq_along(v)) else names(v)
      for (i in seq_along(v)) walk(v[[i]], paste(path, nms[i], sep = "/"))
    }
  }
  for (nm in sort(names(roots))) walk(roots[[nm]], nm)
  out
}

# Internal: give a snapshot the state that ggml_snapshot_free() and the
# finaliser share, so that copies of the R object release the memory once.
.snapshot_attach <- function(m) {
  s <- new.env(parent = emptyenv())
  s$alive       <- TRUE
  s$sched       <- m$compilation$sched
  s$buffer      <- m$compilation$buffer
  s$ctx_weights <- m$compilation$ctx_weights
  reg.finalizer(s, function(e) .snapshot_release(e), onexit = TRUE)
  m$snapshot_state <- s
  m
}

.snapshot_release <- function(s) {
  if (!isTRUE(s$alive)) return(invisible(NULL))
  s$alive <- FALSE
  ggml_backend_sched_free(s$sched)
  ggml_backend_buffer_free(s$buffer)
  ggml_free(s$ctx_weights)
  invisible(NULL)
}

#' Release a Snapshot Made by a Trainer
#'
#' Frees the weights buffer and scheduler of a model returned by a
#' \code{\link{ggml_trainer}}'s \code{$snapshot()}. Calling it twice is
#' harmless. The snapshot must not be used afterwards.
#'
#' @param model A snapshot from \code{$snapshot()}
#' @return \code{NULL}, invisibly.
#' @export
#' @family optimization
#' @seealso \code{\link{ggml_trainer}}
ggml_snapshot_free <- function(model) {
  if (is.null(model$snapshot_state)) {
    stop("'model' is not a snapshot made by a trainer's $snapshot().",
         call. = FALSE)
  }
  .snapshot_release(model$snapshot_state)
}
