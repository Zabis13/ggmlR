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
