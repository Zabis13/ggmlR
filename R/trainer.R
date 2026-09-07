# A long-lived training context you drive yourself.
#
# ggml_fit() and ggml_fit_generator() both own their optimizer context: it is
# created inside the call and freed on the way out. That is the right shape for
# "train this model on this data", but it makes a whole class of loops
# unexpressible -- reinforcement learning above all, where there is no epoch,
# the data comes from a replay buffer the agent is still filling, and rollout
# and optimisation run at different rates.
#
# The per-step primitive for that already exists (ggml_opt_alloc/ggml_opt_eval);
# what was missing is a context that outlives a single call, plus the graph
# building that has to happen before it. ggml_trainer() supplies both, so a
# custom loop needs nothing from ggmlR:::.

# Internal: R-layout batch -> the flat ggml-layout vector the input tensor wants.
# Shares .gen_to_ggml_layout() with the generator path so the two cannot drift.
.trainer_prepare <- function(x, y, input_shape, batch_size) {
  n <- if (is.matrix(x) || is.array(x)) dim(x)[1] else length(x)
  if (!identical(as.integer(n), as.integer(batch_size))) {
    stop("this trainer was built for batches of exactly ", batch_size,
         " sample(s); got ", n, ". Rebuild with a different batch_size, or ",
         "pad the batch yourself.", call. = FALSE)
  }
  if (!is.null(y) && !is.matrix(y) && !is.array(y)) y <- matrix(y, nrow = n)
  list(x = .gen_to_ggml_layout(x, input_shape),
       y = if (is.null(y)) NULL else as.vector(t(y)))
}

#' A Training Context You Step Yourself
#'
#' Builds the graph and optimizer context for a compiled sequential model and
#' hands them back, so a training loop can be written in R rather than handed to
#' \code{\link{ggml_fit}}. The context lives until \code{$free()}, keeping
#' weights, Adam moments and the compute graph across as many steps as the loop
#' takes.
#'
#' This is the level below \code{\link{ggml_fit_opt_gen}}. Reach for it when the
#' loop is not a pass over data: reinforcement learning, where experience comes
#' from a replay buffer the agent is still filling and a target network is
#' refreshed every so many steps; curriculum schemes that change the data
#' distribution as they go; or anything where "epoch" is not the unit of
#' progress.
#'
#' @section Using it:
#' \preformatted{
#' tr <- ggml_trainer(model, batch_size = 32L)
#' on.exit(tr$free())
#'
#' for (i in seq_len(10000L)) {
#'   b <- sample_batch(replay, 32L)     # your own bookkeeping
#'   loss <- tr$step(b$x, b$y)
#'   if (i %% 500L == 0L) target <- tr$model()   # refresh a target network
#' }
#'
#' model <- tr$model()                  # weights back into an R model object
#' tr$free()
#' }
#'
#' @section What a step does:
#' \code{$step(x, y)} runs one forward and backward pass and returns that step's
#' loss. Gradient accumulation, if \code{nbatch_logical} asked for it, is
#' tracked inside the context: the optimizer updates the weights only on the
#' period boundary, and the intervening steps only accumulate. \code{$eval(x, y)}
#' is the same without the backward pass, for measuring on held-out data.
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
#' @param model A compiled \code{ggml_sequential_model}
#' @param batch_size Samples per step. The graph is built for exactly this many;
#'   every batch handed to \code{$step()} must match.
#' @param nbatch_logical Logical batch size for gradient accumulation. Defaults
#'   to \code{batch_size}, meaning every step updates the weights.
#' @return An object of class \code{ggml_trainer}: a list of functions
#'   \code{$step(x, y)}, \code{$eval(x, y)}, \code{$model()},
#'   \code{$set_lr(lr)}, \code{$free()}.
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
ggml_trainer <- function(model, batch_size = 32L, nbatch_logical = NULL) {
  if (!inherits(model, "ggml_sequential_model")) {
    stop("ggml_trainer() currently supports sequential models.", call. = FALSE)
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

  nbatch_physical <- .gen_batch_size(graph_info$inputs)
  nbatch_logical  <- if (is.null(nbatch_logical)) {
    nbatch_physical
  } else {
    as.integer(max(nbatch_logical, nbatch_physical))
  }
  opt_period <- as.integer(max(1L, nbatch_logical %/% nbatch_physical))

  ctx_list <- ggml_opt_init_for_fit(
    model$compilation$sched, loss_type, optimizer_type, opt_period,
    graph_info$ctx_compute, graph_info$inputs, graph_info$outputs, NULL
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

  check_alive <- function() {
    if (!isTRUE(st$alive)) {
      stop("this trainer has been freed; build a new one with ggml_trainer().",
           call. = FALSE)
    }
  }

  run <- function(x, y, backward) {
    check_alive()
    b <- .trainer_prepare(x, y, input_shape, batch_size)

    # A ggml_set_n_threads() issued after the scheduler was built is picked up
    # once, on the first step -- the single C entry points sync before their own
    # loop, and a hand-written loop has no equivalent moment.
    if (!isTRUE(st$threads_ok)) {
      .ggml_sched_sync_threads(model$compilation$sched)
      st$threads_ok <- TRUE
    }

    ggml_opt_result_reset(st$result)
    .gen_run_step(st$opt_ctx,
                  list(inputs = list(b$x), labels = list(b$y)),
                  st$result, backward)
    if (backward) st$nstep <- st$nstep + 1L

    ggml_opt_result_loss(st$result)[["loss"]]
  }

  free_all <- function() {
    if (!isTRUE(st$alive)) return(invisible(NULL))
    st$alive <- FALSE
    ggml_opt_result_free(st$result)
    ggml_opt_free(st$opt_ctx)
    # ctx_weights and buffer are deliberately left alone: they hold the trained
    # weights, which $model() hands to a model object that outlives the trainer.
    # Only the compute graph is ours to release, exactly as ggml_fit() does.
    ggml_free(st$graph$ctx_compute)
    invisible(NULL)
  }

  # A forgotten $free() should not strand GPU buffers for the life of the
  # session. The finaliser runs on the state environment, so it fires once the
  # trainer is unreachable; it is a backstop, not a substitute for $free(),
  # since gc() timing is not predictable.
  reg.finalizer(st, function(e) free_all(), onexit = TRUE)

  structure(list(
    step = function(x, y) run(x, y, backward = TRUE),

    eval = function(x, y) run(x, y, backward = FALSE),

    model = function() {
      check_alive()
      m <- st$model
      m$layers <- st$graph$layers_built
      m$compilation$ctx_weights <- st$graph$ctx_weights
      m$compilation$buffer <- st$graph$buffer
      m
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

    n_steps = function() st$nstep,

    free = free_all
  ), class = "ggml_trainer")
}

#' @export
print.ggml_trainer <- function(x, ...) {
  cat("<ggml_trainer>", x$n_steps(), "step(s) taken\n")
  invisible(x)
}
