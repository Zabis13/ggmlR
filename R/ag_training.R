# Training utilities for the dynamic autograd engine (ag_*).
#
# Contents:
#   ag_dataloader()       — mini-batch iterator with shuffle
#   lr_scheduler_step()   — step-decay learning rate scheduler
#   lr_scheduler_cosine() — cosine-annealing scheduler
#   clip_grad_norm()      — gradient clipping by global L2 norm

# ============================================================================
# DataLoader
# ============================================================================

#' Create a mini-batch data loader
#'
#' Returns an iterator environment.  Each call to \code{$next_batch()} returns
#' a named list \code{list(x, y)} with ag_tensor objects of shape
#' \code{[features, batch_size]} / \code{[labels, batch_size]}.
#' After the last batch, \code{$has_next()} returns \code{FALSE}; call
#' \code{$reset()} (or start a new epoch via \code{$epoch()}) to reshuffle
#' and restart.
#'
#' @param x Feature matrix \code{[features, n_samples]} or
#'   \code{[n_samples, features]} — see \code{col_major}.
#' @param y Label matrix with the same convention.
#' @param batch_size Integer batch size.
#' @param shuffle Logical; if \code{TRUE} (default) shuffle at each \code{reset()}.
#' @param col_major Logical; if \code{TRUE} (default) \code{x} and \code{y} are
#'   already \code{[features, n]} (ggml/ag convention).  Set \code{FALSE} for
#'   row-major \code{[n, features]} (R/Keras convention) — they will be
#'   transposed automatically.
#' @return An \code{ag_dataloader} environment
#' @export
#' @examples
#' \donttest{
#' n  <- 128L
#' x  <- matrix(runif(4 * n), 4, n)   # [4, 128] col-major
#' y  <- matrix(runif(2 * n), 2, n)
#' dl <- ag_dataloader(x, y, batch_size = 32L)
#' dl$reset()
#' while (dl$has_next()) {
#'   batch <- dl$next_batch()
#'   # batch$x: [4, 32],  batch$y: [2, 32]
#' }
#' }
ag_dataloader <- function(x, y = NULL, batch_size = 32L,
                           shuffle = TRUE, col_major = TRUE) {
  batch_size <- as.integer(batch_size)

  # normalise to col-major [features, n]
  if (!col_major) {
    x <- t(x)
    if (!is.null(y)) y <- t(y)
  }

  n <- ncol(x)
  if (!is.null(y) && ncol(y) != n) {
    stop("x and y must have the same number of samples (columns in col-major)")
  }

  env <- new.env(parent = emptyenv())
  env$x          <- x
  env$y          <- y
  env$n          <- n
  env$batch_size <- batch_size
  env$shuffle    <- shuffle
  env$order      <- seq_len(n)
  env$pos        <- 1L

  env$reset <- function() {
    if (env$shuffle) env$order <- sample(env$n)
    env$pos <- 1L
  }

  env$has_next <- function() {
    env$pos + env$batch_size - 1L <= env$n
  }

  env$next_batch <- function() {
    if (!env$has_next()) stop("No more batches. Call $reset() to start a new epoch.")
    idx    <- env$order[env$pos:(env$pos + env$batch_size - 1L)]
    env$pos <- env$pos + env$batch_size

    bx <- ag_tensor(env$x[, idx, drop = FALSE])
    if (!is.null(env$y)) {
      by <- ag_tensor(env$y[, idx, drop = FALSE])
      list(x = bx, y = by)
    } else {
      list(x = bx)
    }
  }

  # Convenience: iterate over all complete batches, return list
  env$epoch <- function() {
    env$reset()
    batches <- list()
    while (env$has_next()) batches <- c(batches, list(env$next_batch()))
    batches
  }

  # Number of complete batches per epoch
  env$n_batches <- function() env$n %/% env$batch_size

  class(env) <- "ag_dataloader"
  env$reset()
  env
}

#' @export
print.ag_dataloader <- function(x, ...) {
  cat(sprintf("ag_dataloader | n=%d | batch_size=%d | n_batches=%d | shuffle=%s\n",
              x$n, x$batch_size, x$n_batches(),
              if (x$shuffle) "TRUE" else "FALSE"))
  invisible(x)
}

# ============================================================================
# Learning rate schedulers
# ============================================================================

#' Step-decay learning rate scheduler
#'
#' Multiplies the optimizer learning rate by \code{gamma} every
#' \code{step_size} calls to \code{$step()}.
#'
#' @param optimizer An \code{ag_optimizer_adam} or \code{ag_optimizer_sgd}
#'   environment.
#' @param step_size Decay every this many steps (epochs).
#' @param gamma Multiplicative decay factor (default 0.1).
#' @return An \code{lr_scheduler_step} environment
#' @export
#' @examples
#' \donttest{
#' w   <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_adam(list(w = w), lr = 0.1)
#' sch <- lr_scheduler_step(opt, step_size = 10L, gamma = 0.5)
#' for (epoch in 1:30) sch$step()
#' opt$lr  # 0.1 * 0.5^3 = 0.0125
#' }
lr_scheduler_step <- function(optimizer, step_size, gamma = 0.1) {
  step_size <- as.integer(step_size)
  env <- new.env(parent = emptyenv())
  env$optimizer  <- optimizer
  env$step_size  <- step_size
  env$gamma      <- gamma
  env$last_epoch <- 0L

  env$step <- function() {
    env$last_epoch <- env$last_epoch + 1L
    if (env$last_epoch %% env$step_size == 0L) {
      env$optimizer$lr <- env$optimizer$lr * env$gamma
    }
    invisible(env$optimizer$lr)
  }

  env$get_lr <- function() env$optimizer$lr

  class(env) <- "lr_scheduler_step"
  env
}

#' @export
print.lr_scheduler_step <- function(x, ...) {
  cat(sprintf("lr_scheduler_step | step_size=%d | gamma=%.4f | epoch=%d | lr=%.6f\n",
              x$step_size, x$gamma, x$last_epoch, x$get_lr()))
  invisible(x)
}

#' Cosine-annealing learning rate scheduler
#'
#' Varies the learning rate following a cosine curve from \code{lr_max} down to
#' \code{lr_min} over \code{T_max} steps.  Restarts (SGDR-style) if
#' \code{restart = TRUE}.
#'
#' @param optimizer Optimizer environment.
#' @param T_max Number of steps for one cosine cycle.
#' @param lr_min Minimum learning rate (default 0).
#' @param restart Logical; if \code{TRUE} restart after \code{T_max} steps.
#' @return An \code{lr_scheduler_cosine} environment
#' @export
#' @examples
#' \donttest{
#' w   <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_adam(list(w = w), lr = 0.1)
#' sch <- lr_scheduler_cosine(opt, T_max = 50L)
#' for (epoch in 1:50) sch$step()
#' }
lr_scheduler_cosine <- function(optimizer, T_max, lr_min = 0, restart = FALSE) {
  T_max <- as.integer(T_max)
  env <- new.env(parent = emptyenv())
  env$optimizer  <- optimizer
  env$T_max      <- T_max
  env$lr_min     <- lr_min
  env$lr_max     <- optimizer$lr   # initial lr is the max
  env$restart    <- restart
  env$last_epoch <- 0L

  env$step <- function() {
    env$last_epoch <- env$last_epoch + 1L
    t <- if (env$restart) {
      (env$last_epoch - 1L) %% env$T_max
    } else {
      min(env$last_epoch - 1L, env$T_max - 1L)
    }
    new_lr <- env$lr_min + 0.5 * (env$lr_max - env$lr_min) *
                (1 + cos(pi * t / env$T_max))
    env$optimizer$lr <- new_lr
    invisible(new_lr)
  }

  env$get_lr <- function() env$optimizer$lr

  class(env) <- "lr_scheduler_cosine"
  env
}

#' @export
print.lr_scheduler_cosine <- function(x, ...) {
  cat(sprintf("lr_scheduler_cosine | T_max=%d | lr_max=%.6f | lr_min=%.6f | epoch=%d | lr=%.6f\n",
              x$T_max, x$lr_max, x$lr_min, x$last_epoch, x$get_lr()))
  invisible(x)
}

# ============================================================================
# Gradient clipping
# ============================================================================

#' Clip gradients by global L2 norm
#'
#' Rescales all gradients in \code{grads} so that their global L2 norm does
#' not exceed \code{max_norm}.  Modifies the \code{grads} environment
#' in-place and returns the pre-clip norm.
#'
#' Call this \strong{after} \code{backward()} and \strong{before}
#' \code{optimizer$step()}.
#'
#' @param params Named list of ag_param tensors (same as passed to optimizer).
#' @param grads Gradient environment returned by \code{backward()}.
#' @param max_norm Maximum allowed global L2 norm.
#' @return Numeric: the global L2 norm before clipping (invisibly).
#' @export
#' @examples
#' \donttest{
#' w  <- ag_param(matrix(runif(4), 2, 2))
#' x  <- ag_tensor(matrix(c(1, 1), 2, 1))
#' with_grad_tape({
#'   out  <- ag_matmul(w, x)
#'   loss <- ag_mse_loss(out, matrix(0, 2, 1))
#' })
#' grads <- backward(loss)
#' clip_grad_norm(list(w = w), grads, max_norm = 1.0)
#' }
clip_grad_norm <- function(params, grads, max_norm) {
  # compute global L2 norm of all gradients
  total_sq <- 0
  grad_list <- list()

  for (nm in names(params)) {
    p   <- params[[nm]]
    key <- as.character(p$id)
    g   <- get0(key, envir = grads)
    if (!is.null(g)) {
      total_sq       <- total_sq + sum(g^2)
      grad_list[[key]] <- g
    }
  }

  global_norm <- sqrt(total_sq)

  if (global_norm > max_norm) {
    scale <- max_norm / (global_norm + 1e-6)
    for (key in names(grad_list)) {
      assign(key, grad_list[[key]] * scale, envir = grads)
    }
  }

  invisible(global_norm)
}

# ============================================================================
# Data Parallel Training
# ============================================================================

#' Data-parallel training across multiple GPUs
#'
#' Runs synchronous data-parallel training:
#' \enumerate{
#'   \item \code{make_model()} is called \code{n_gpu} times to create one
#'     independent model replica per GPU (each with its own parameters).
#'   \item Each iteration: the current data item is forwarded through every
#'     replica in parallel; gradients are computed via \code{backward()}.
#'   \item Gradients are averaged across all replicas (element-wise mean).
#'   \item One optimizer step is taken on replica 0; updated weights are then
#'     broadcast to replicas 1 … N-1 so all replicas stay in sync.
#' }
#'
#' Because all replicas live in the same R process and \code{ag_param} uses
#' environment (reference) semantics, no IPC or NCCL is required — weight
#' synchronisation is a simple in-place copy.
#'
#' @param make_model A zero-argument function that returns a model object with
#'   at least \code{$forward(x)} and \code{$parameters()} methods.  Called
#'   \code{n_gpu} times; each call must produce independent parameters.
#' @param data A list of training samples.  Each element is passed directly to
#'   \code{forward_fn} (or to \code{model$forward()} if \code{forward_fn} is
#'   \code{NULL}).
#' @param loss_fn A function \code{(logits, target) -> scalar ag_tensor}.
#'   If \code{NULL}, \code{forward_fn} must return the loss directly.
#' @param forward_fn Optional function \code{(model, sample) -> logits}.
#'   If \code{NULL}, the sample is passed directly as
#'   \code{model$forward(sample)}.
#' @param target_fn Optional function \code{(sample) -> target}.  Used when
#'   \code{loss_fn} is not \code{NULL} to extract the target from a sample.
#'   If \code{NULL}, \code{sample} itself is used as the target.
#' @param n_gpu Number of GPU replicas (default: all available Vulkan devices,
#'   minimum 1).
#' @param n_iter Number of training iterations (passes over \code{data}).
#' @param lr Learning rate for Adam optimizer (default 1e-3).
#' @param max_norm Gradient clipping threshold (default \code{Inf} = no clip).
#' @param verbose Print loss every \code{verbose} iterations, or \code{FALSE}
#'   to suppress output.
#' @return A list with:
#'   \describe{
#'     \item{\code{params}}{Named list of final parameters (from replica 0).}
#'     \item{\code{loss_history}}{Numeric vector of per-iteration mean loss.}
#'     \item{\code{model}}{Replica 0 model object.}
#'   }
#' @export
#' @examples
#' \donttest{
#' make_model <- function() {
#'   W <- ag_param(matrix(rnorm(4), 2, 2))
#'   list(
#'     forward    = function(x) ag_matmul(W, x),
#'     parameters = function() list(W = W)
#'   )
#' }
#' data <- lapply(1:8, function(i) matrix(rnorm(2), 2, 1))
#' result <- dp_train(
#'   make_model = make_model,
#'   data       = data,
#'   loss_fn    = function(out, tgt) ag_mse_loss(out, tgt),
#'   target_fn  = function(s) s,
#'   n_gpu      = 1L,
#'   n_iter     = 10L,
#'   lr         = 1e-3,
#'   verbose    = FALSE
#' )
#' }
dp_train <- function(make_model,
                     data,
                     loss_fn    = NULL,
                     forward_fn = NULL,
                     target_fn  = NULL,
                     n_gpu      = NULL,
                     n_iter     = 10L,
                     lr         = 1e-3,
                     max_norm   = Inf,
                     verbose    = 10L) {

  # ---- determine n_gpu ----
  if (is.null(n_gpu)) {
    n_avail <- tryCatch(ggml_vulkan_device_count(), error = function(e) 0L)
    n_gpu   <- max(1L, n_avail)
  }
  n_gpu <- as.integer(n_gpu)

  # save device so we can restore it on exit
  orig_device <- .ag_device_state$device
  on.exit(tryCatch(ag_device(orig_device), error = function(e) NULL), add = TRUE)

  # ---- create replicas ----
  replicas <- vector("list", n_gpu)
  for (i in seq_len(n_gpu)) {
    dev <- if (n_gpu > 1L) {
      tryCatch({ ag_device("gpu"); "gpu" }, error = function(e) "cpu")
    } else {
      .ag_device_state$device
    }
    replicas[[i]] <- make_model()
  }

  # ---- parameter name order (from replica 0) ----
  param_names <- names(replicas[[1L]]$parameters())

  # ---- broadcast initial weights from replica 0 to all others ----
  # make_model() initialises each replica with independent random weights;
  # we must sync before the first step so all replicas start identically.
  if (n_gpu > 1L) {
    p0 <- replicas[[1L]]$parameters()
    for (i in seq(2L, n_gpu)) {
      pi <- replicas[[i]]$parameters()
      for (nm in param_names) pi[[nm]]$data <- p0[[nm]]$data
    }
  }

  # ---- optimizer on replica 0 ----
  opt <- optimizer_adam(replicas[[1L]]$parameters(), lr = lr)

  # ---- helper: copy weights from replica 0 to replica i ----
  .sync_weights <- function(i) {
    p0 <- replicas[[1L]]$parameters()
    pi <- replicas[[i]]$parameters()
    for (nm in param_names) {
      pi[[nm]]$data <- p0[[nm]]$data
    }
  }

  # ---- helper: wrap plain matrix/vector in ag_tensor if needed ----
  .as_ag <- function(x) {
    if (is_ag_tensor(x)) x else ag_tensor(if (is.matrix(x)) x else matrix(x, ncol = 1L))
  }

  # ---- helper: run one forward+backward on replica i, sample s ----
  .replica_step <- function(i, s) {
    model <- replicas[[i]]

    with_grad_tape({
      logits <- if (is.null(forward_fn)) {
        model$forward(.as_ag(s))
      } else {
        forward_fn(model, s)
      }

      loss <- if (is.null(loss_fn)) {
        logits   # forward_fn already returns loss
      } else {
        tgt <- if (is.null(target_fn)) s else target_fn(s)
        loss_fn(logits, tgt)
      }
    })

    grads <- backward(loss)
    list(loss = as.numeric(.ag_data(loss)), grads = grads)
  }

  # ---- helper: average gradients from all replicas into grads0 ----
  .average_grads <- function(grad_list) {
    # grad_list: list of grads environments, one per replica
    # Result written into grad_list[[1]] in-place
    if (length(grad_list) == 1L) return(grad_list[[1L]])

    p0 <- replicas[[1L]]$parameters()
    for (nm in param_names) {
      key <- as.character(p0[[nm]]$id)
      g0  <- get0(key, envir = grad_list[[1L]])
      if (is.null(g0)) next

      # sum contributions from replicas 2..N
      g_sum <- g0
      for (j in seq(2L, length(grad_list))) {
        pj  <- replicas[[j]]$parameters()
        # replica j has its own param id for the same "slot"
        key_j <- as.character(pj[[nm]]$id)
        gj    <- get0(key_j, envir = grad_list[[j]])
        if (!is.null(gj)) g_sum <- g_sum + gj
      }
      assign(key, g_sum / length(grad_list), envir = grad_list[[1L]])
    }
    grad_list[[1L]]
  }

  # ---- training loop ----
  loss_history <- numeric(n_iter)
  n_data       <- length(data)

  for (iter in seq_len(n_iter)) {

    # partition data round-robin across replicas for this iteration
    sample_idx  <- ((iter - 1L) %% n_data) + 1L
    sample      <- data[[sample_idx]]

    # run forward+backward on each replica
    results <- lapply(seq_len(n_gpu), function(i) .replica_step(i, sample))

    # average loss
    iter_loss <- mean(vapply(results, `[[`, numeric(1L), "loss"), na.rm = TRUE)
    loss_history[[iter]] <- iter_loss

    # average gradients (written into results[[1]]$grads)
    avg_grads <- .average_grads(lapply(results, `[[`, "grads"))

    # gradient clipping on averaged grads
    if (is.finite(max_norm)) {
      clip_grad_norm(replicas[[1L]]$parameters(), avg_grads, max_norm)
    }

    # optimizer step on replica 0
    opt$step(avg_grads)
    opt$zero_grad()

    # broadcast updated weights to all other replicas
    if (n_gpu > 1L) {
      for (i in seq(2L, n_gpu)) .sync_weights(i)
    }

    if (!isFALSE(verbose) && (iter %% as.integer(verbose) == 0L || iter == 1L)) {
      cat(sprintf("[dp_train] iter %4d / %d  loss = %.6f\n", iter, n_iter, iter_loss))
    }
  }

  list(
    params       = replicas[[1L]]$parameters(),
    loss_history = loss_history,
    model        = replicas[[1L]]
  )
}
