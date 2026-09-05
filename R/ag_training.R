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
#' With \code{restart = TRUE} and \code{T_mult > 1} this is the full
#' \emph{CosineAnnealingWarmRestarts} (SGDR) schedule: the first cycle lasts
#' \code{T_max} steps and every subsequent cycle is \code{T_mult} times longer
#' than the previous one.  \code{T_mult = 1} (the default) keeps every cycle the
#' same length.
#'
#' @param optimizer Optimizer environment.
#' @param T_max Number of steps for the first cosine cycle.
#' @param lr_min Minimum learning rate (default 0).
#' @param restart Logical; if \code{TRUE} restart after each cycle.
#' @param T_mult Cycle-length multiplier applied after every restart (default
#'   1, i.e. constant period).  Only meaningful when \code{restart = TRUE}.
#' @return An \code{lr_scheduler_cosine} environment
#' @references Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic Gradient
#'   Descent with Warm Restarts. ICLR.
#' @export
#' @examples
#' \donttest{
#' w   <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_adam(list(w = w), lr = 0.1)
#' sch <- lr_scheduler_cosine(opt, T_max = 50L)
#' for (epoch in 1:50) sch$step()
#'
#' # SGDR with doubling cycle length: 10, 20, 40, ...
#' opt2 <- optimizer_adam(list(w = w), lr = 0.1)
#' sch2 <- lr_scheduler_cosine(opt2, T_max = 10L, restart = TRUE, T_mult = 2)
#' for (epoch in 1:70) sch2$step()
#' }
lr_scheduler_cosine <- function(optimizer, T_max, lr_min = 0, restart = FALSE,
                                T_mult = 1) {
  T_max <- as.integer(T_max)
  if (T_max < 1L) stop("T_max must be >= 1")
  if (!is.numeric(T_mult) || length(T_mult) != 1L || T_mult < 1) {
    stop("T_mult must be a single number >= 1")
  }
  env <- new.env(parent = emptyenv())
  env$optimizer  <- optimizer
  env$T_max      <- T_max
  env$lr_min     <- lr_min
  env$lr_max     <- optimizer$lr   # initial lr is the max
  env$restart    <- restart
  env$T_mult     <- T_mult
  env$last_epoch <- 0L
  # SGDR state: length of the current cycle and step index inside it
  env$T_cur      <- T_max
  env$cycle_pos  <- 0L

  env$step <- function() {
    env$last_epoch <- env$last_epoch + 1L

    if (env$restart) {
      # position and period of the cycle we are currently in
      t      <- env$cycle_pos
      period <- env$T_cur
    } else {
      t      <- min(env$last_epoch - 1L, env$T_max - 1L)
      period <- env$T_max
    }

    new_lr <- env$lr_min + 0.5 * (env$lr_max - env$lr_min) *
                (1 + cos(pi * t / period))

    if (env$restart) {
      # advance only after the LR for this step has been computed, so the
      # closing step of a cycle still uses that cycle's period
      env$cycle_pos <- env$cycle_pos + 1L
      if (env$cycle_pos >= env$T_cur) {
        env$cycle_pos <- 0L
        env$T_cur     <- max(1L, as.integer(round(env$T_cur * env$T_mult)))
      }
    }
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
      # Resident gradients (component 3) arrive as device handles; the norm is
      # host arithmetic, so materialise once here and reuse below rather than
      # letting `g^2` meet a handle.
      g              <- .ag_as_matrix(g)
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
      # Copy through the contract: a replica's parameter may be resident, and a
      # direct $data assignment would leave its device buffer on the old
      # weights (inst/docs/ag_data_contract.md).
      for (nm in param_names) .ag_opt_store_weight(pi[[nm]], .ag_data(p0[[nm]]))
    }
  }

  # ---- optimizer on replica 0 ----
  opt <- optimizer_adam(replicas[[1L]]$parameters(), lr = lr)

  # ---- helper: copy weights from replica 0 to replica i ----
  .sync_weights <- function(i) {
    p0 <- replicas[[1L]]$parameters()
    pi <- replicas[[i]]$parameters()
    for (nm in param_names) {
      .ag_opt_store_weight(pi[[nm]], .ag_data(p0[[nm]]))
    }
  }

  # ---- helper: wrap plain matrix/vector in ag_tensor if needed ----
  .as_ag <- function(x) {
    if (is_ag_tensor(x)) x else ag_tensor(if (is.matrix(x)) x else matrix(x, ncol = 1L))
  }

  # ---- helper: run one forward+backward on replica i, sample s ----
  .replica_step <- function(i, s) {
    model <- replicas[[i]]

    # Clear this replica's gradients before its pass.
    #
    # backward() ADDS to $grad when one is already there, which is right inside
    # one step and wrong across iterations: a replica's gradient from the
    # previous iteration is a handle whose pass-pool buffer another replica's
    # tape has already freed, and the accumulation would try to read it. The
    # averaging below takes each replica's gradients fresh every iteration, so
    # there is nothing here worth carrying over.
    for (p_ in model$parameters()) p_$grad <- NULL

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

    # Materialise before returning, while this replica's tape is still alive.
    #
    # With resident gradients backward() hands back device handles into the pass
    # pool, and the caller runs every replica before averaging any of them --
    # so replica 1's handles are freed by replica 2's with_grad_tape() long
    # before .average_grads() reads them ("buffer freed by a tape reset,
    # generation 428 < 429"). The allreduce is host arithmetic across replicas
    # that may sit on different devices, so these values have to come back
    # regardless; the only question is whether they come back in time.
    if (.ag_bwd_resident_grads()) {
      for (k in ls(grads, all.names = TRUE))
        assign(k, .ag_as_matrix(get(k, envir = grads)), envir = grads)
    }
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
      #
      # Resident gradients (component 3) are device handles, and the allreduce
      # is host arithmetic across replicas that may sit on different devices --
      # so materialise as they are summed.
      g_sum <- .ag_as_matrix(g0)
      for (j in seq(2L, length(grad_list))) {
        pj  <- replicas[[j]]$parameters()
        # replica j has its own param id for the same "slot"
        key_j <- as.character(pj[[nm]]$id)
        gj    <- get0(key_j, envir = grad_list[[j]])
        if (!is.null(gj)) g_sum <- g_sum + .ag_as_matrix(gj)
      }
      assign(key, g_sum / length(grad_list), envir = grad_list[[1L]])
    }
    grad_list[[1L]]
  }

  # ---- training loop ----
  loss_history <- numeric(n_iter)
  n_data       <- length(data)

  for (iter in seq_len(n_iter)) {

    # each replica gets a different sample (true data-parallel):
    # replica i processes data[[ base + i ]], round-robin over dataset
    base <- ((iter - 1L) * n_gpu) %% n_data

    # run forward+backward on each replica with its own sample
    results <- lapply(seq_len(n_gpu), function(i) {
      idx <- (base + i - 1L) %% n_data + 1L
      .replica_step(i, data[[idx]])
    })

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

  tryCatch(ag_device(orig_device), error = function(e) NULL)

  list(
    params       = replicas[[1L]]$parameters(),
    loss_history = loss_history,
    model        = replicas[[1L]]
  )
}

# ============================================================================
# Gradient clipping by value
# ============================================================================

#' Clip gradients element-wise by value
#'
#' Clamps every gradient element into
#' \code{[-clip_value, clip_value]}.  Unlike \code{\link{clip_grad_norm}} this
#' does not preserve the gradient direction — each element is treated
#' independently.  Modifies the \code{grads} environment in-place.
#'
#' Call this \strong{after} \code{backward()} and \strong{before}
#' \code{optimizer$step()}.
#'
#' @param params Named list of ag_param tensors (same as passed to optimizer).
#' @param grads Gradient environment returned by \code{backward()}.
#' @param clip_value Positive numeric; maximum absolute value per element.
#' @return Numeric: the largest absolute gradient value before clipping
#'   (invisibly).  Returns 0 when no gradients were found.
#' @seealso \code{\link{clip_grad_norm}}
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
#' clip_grad_value(list(w = w), grads, clip_value = 0.5)
#' }
clip_grad_value <- function(params, grads, clip_value) {
  if (!is.numeric(clip_value) || length(clip_value) != 1L ||
      !is.finite(clip_value) || clip_value <= 0) {
    stop("clip_value must be a single positive finite number")
  }

  max_abs <- 0

  for (nm in names(params)) {
    p   <- params[[nm]]
    key <- as.character(p$id)
    g   <- .ag_as_matrix(get0(key, envir = grads))
    if (is.null(g)) next

    finite_g <- g[is.finite(g)]
    if (length(finite_g)) max_abs <- max(max_abs, max(abs(finite_g)))

    clipped <- pmin(pmax(g, -clip_value), clip_value)
    # pmin/pmax drop the dim attribute on some inputs — restore it
    dim(clipped) <- dim(g)
    assign(key, clipped, envir = grads)
  }

  invisible(max_abs)
}

# ============================================================================
# Gradient anomaly detection
# ============================================================================

#' Detect NaN / Inf gradients
#'
#' Scans the gradient environment produced by \code{backward()} and reports
#' every parameter whose gradient contains \code{NaN} or \code{Inf} values.
#' Useful as a debugging hook right after \code{backward()} — it pinpoints the
#' offending layer instead of leaving you with a model that silently turns to
#' \code{NaN} several epochs later.
#'
#' @param params Named list of ag_param tensors (same as passed to optimizer).
#' @param grads Gradient environment returned by \code{backward()}.
#' @param action One of \code{"warn"} (default), \code{"stop"} or
#'   \code{"silent"} — what to do when an anomaly is found.
#' @param max_abs Optional numeric threshold; gradients whose maximum absolute
#'   finite value exceeds it are also reported (as \code{"large"}).
#'   \code{NULL} (default) disables this check.
#' @return A data frame (invisibly) with one row per parameter that has a
#'   gradient, containing columns \code{param}, \code{n_nan}, \code{n_inf},
#'   \code{max_abs} and \code{status}.  \code{status} is one of \code{"ok"},
#'   \code{"nan"}, \code{"inf"}, \code{"nan+inf"} or \code{"large"}.
#'   Parameters without a gradient are reported with \code{status = "missing"}.
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
#' check_grad_anomaly(list(w = w), grads)
#' }
check_grad_anomaly <- function(params, grads, action = c("warn", "stop", "silent"),
                               max_abs = NULL) {
  action <- match.arg(action)

  rows <- list()

  for (nm in names(params)) {
    p   <- params[[nm]]
    key <- as.character(p$id)
    g   <- .ag_as_matrix(get0(key, envir = grads))

    if (is.null(g)) {
      rows[[length(rows) + 1L]] <- data.frame(
        param = nm, n_nan = NA_integer_, n_inf = NA_integer_,
        max_abs = NA_real_, status = "missing",
        stringsAsFactors = FALSE
      )
      next
    }

    n_nan <- sum(is.nan(g))
    n_inf <- sum(is.infinite(g))
    finite_g <- g[is.finite(g)]
    g_max <- if (length(finite_g)) max(abs(finite_g)) else NA_real_

    status <- if (n_nan > 0 && n_inf > 0) {
      "nan+inf"
    } else if (n_nan > 0) {
      "nan"
    } else if (n_inf > 0) {
      "inf"
    } else if (!is.null(max_abs) && !is.na(g_max) && g_max > max_abs) {
      "large"
    } else {
      "ok"
    }

    rows[[length(rows) + 1L]] <- data.frame(
      param = nm, n_nan = as.integer(n_nan), n_inf = as.integer(n_inf),
      max_abs = g_max, status = status,
      stringsAsFactors = FALSE
    )
  }

  report <- if (length(rows)) {
    do.call(rbind, rows)
  } else {
    data.frame(param = character(0), n_nan = integer(0), n_inf = integer(0),
               max_abs = numeric(0), status = character(0),
               stringsAsFactors = FALSE)
  }

  bad <- report[!report$status %in% c("ok", "missing"), , drop = FALSE]

  if (nrow(bad) && action != "silent") {
    msg <- paste0(
      "gradient anomaly detected in ", nrow(bad), " parameter(s):\n",
      paste(sprintf("  %-20s status=%-8s n_nan=%d n_inf=%d max_abs=%s",
                    bad$param, bad$status, bad$n_nan, bad$n_inf,
                    format(bad$max_abs, digits = 4)),
            collapse = "\n")
    )
    if (action == "stop") stop(msg, call. = FALSE) else warning(msg, call. = FALSE)
  }

  invisible(report)
}

# ============================================================================
# Additional learning rate schedulers
# ============================================================================

# Internal: read/write the momentum-like field of an optimizer.
# SGD cycles $momentum, Adam cycles $beta1.  Returns NULL when the optimizer
# has no cyclable field.
.ag_momentum_field <- function(optimizer) {
  if (inherits(optimizer, "ag_optimizer_sgd"))  return("momentum")
  if (inherits(optimizer, "ag_optimizer_adam")) return("beta1")
  NULL
}

#' Cyclical learning rate scheduler
#'
#' Cycles the learning rate linearly between \code{base_lr} and \code{max_lr}
#' over a triangular cycle of \code{2 * step_size_up} steps (Smith, 2017).
#' Each call to \code{$step()} advances one step — call it \strong{per batch},
#' not per epoch.
#'
#' @param optimizer Optimizer environment.
#' @param base_lr Lower learning rate bound.
#' @param max_lr Upper learning rate bound.
#' @param step_size_up Steps in the increasing half of a cycle (default 2000).
#' @param step_size_down Steps in the decreasing half.  \code{NULL} (default)
#'   mirrors \code{step_size_up}.
#' @param mode One of \code{"triangular"} (default), \code{"triangular2"}
#'   (amplitude halved each cycle) or \code{"exp_range"} (amplitude scaled by
#'   \code{gamma^step}).
#' @param gamma Decay factor for \code{mode = "exp_range"} (default 1.0).
#' @return An \code{lr_scheduler_cyclic} environment with \code{$step()} and
#'   \code{$get_lr()}.
#' @references Smith, L. N. (2017). Cyclical Learning Rates for Training Neural
#'   Networks. WACV.
#' @export
#' @examples
#' \donttest{
#' w   <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_sgd(list(w = w), lr = 0.01)
#' sch <- lr_scheduler_cyclic(opt, base_lr = 0.001, max_lr = 0.01,
#'                            step_size_up = 10L)
#' for (i in 1:40) sch$step()
#' }
lr_scheduler_cyclic <- function(optimizer, base_lr, max_lr, step_size_up = 2000L,
                                step_size_down = NULL,
                                mode = c("triangular", "triangular2", "exp_range"),
                                gamma = 1.0) {
  mode <- match.arg(mode)
  step_size_up <- as.integer(step_size_up)
  if (step_size_up < 1L) stop("step_size_up must be >= 1")
  step_size_down <- if (is.null(step_size_down)) step_size_up else as.integer(step_size_down)
  if (step_size_down < 1L) stop("step_size_down must be >= 1")
  if (max_lr < base_lr) stop("max_lr must be >= base_lr")

  env <- new.env(parent = emptyenv())
  env$optimizer      <- optimizer
  env$base_lr        <- base_lr
  env$max_lr         <- max_lr
  env$step_size_up   <- step_size_up
  env$step_size_down <- step_size_down
  env$mode           <- mode
  env$gamma          <- gamma
  env$last_step      <- 0L

  # start at base_lr
  optimizer$lr <- base_lr

  env$step <- function() {
    env$last_step <- env$last_step + 1L
    total  <- env$step_size_up + env$step_size_down
    cycle  <- floor(env$last_step / total)
    pos    <- env$last_step - cycle * total

    frac <- if (pos <= env$step_size_up) {
      pos / env$step_size_up
    } else {
      1 - (pos - env$step_size_up) / env$step_size_down
    }

    amp <- env$max_lr - env$base_lr
    amp <- switch(env$mode,
      triangular  = amp,
      triangular2 = amp / (2^cycle),
      exp_range   = amp * env$gamma^env$last_step
    )

    new_lr <- env$base_lr + amp * frac
    env$optimizer$lr <- new_lr
    invisible(new_lr)
  }

  env$get_lr <- function() env$optimizer$lr

  class(env) <- "lr_scheduler_cyclic"
  env
}

#' @export
print.lr_scheduler_cyclic <- function(x, ...) {
  cat(sprintf("lr_scheduler_cyclic | mode=%s | base_lr=%.6f | max_lr=%.6f | step=%d | lr=%.6f\n",
              x$mode, x$base_lr, x$max_lr, x$last_step, x$get_lr()))
  invisible(x)
}

#' One-cycle learning rate scheduler
#'
#' Implements the 1cycle policy (Smith, 2018): the learning rate rises from
#' \code{max_lr / div_factor} to \code{max_lr} during the first
#' \code{pct_start} fraction of training, then anneals down to
#' \code{max_lr / final_div_factor}.  Call \code{$step()} \strong{per batch};
#' \code{total_steps} is the total number of batches over the whole run
#' (\code{epochs * batches_per_epoch}).
#'
#' When \code{cycle_momentum = TRUE} the momentum-like hyper-parameter moves
#' inversely to the learning rate: \code{$momentum} for SGD, \code{$beta1} for
#' Adam.  For other optimizers momentum cycling is skipped.
#'
#' @param optimizer Optimizer environment.
#' @param max_lr Peak learning rate.
#' @param total_steps Total number of \code{$step()} calls for the whole run.
#' @param pct_start Fraction of \code{total_steps} spent increasing the LR
#'   (default 0.3).
#' @param div_factor Initial LR is \code{max_lr / div_factor} (default 25).
#' @param final_div_factor Final LR is \code{max_lr / final_div_factor}
#'   (default 1e4).
#' @param anneal_strategy \code{"cos"} (default) or \code{"linear"}.
#' @param cycle_momentum Logical; cycle momentum inversely to LR (default
#'   \code{TRUE}).
#' @param base_momentum,max_momentum Momentum bounds used when
#'   \code{cycle_momentum = TRUE} (defaults 0.85 / 0.95).
#' @return An \code{lr_scheduler_onecycle} environment with \code{$step()} and
#'   \code{$get_lr()}.
#' @references Smith, L. N. (2018). A disciplined approach to neural network
#'   hyper-parameters. arXiv:1803.09820.
#' @export
#' @examples
#' \donttest{
#' w   <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_sgd(list(w = w), lr = 0.01, momentum = 0.9)
#' sch <- lr_scheduler_onecycle(opt, max_lr = 0.1, total_steps = 100L)
#' for (i in 1:100) sch$step()
#' }
lr_scheduler_onecycle <- function(optimizer, max_lr, total_steps,
                                  pct_start = 0.3, div_factor = 25,
                                  final_div_factor = 1e4,
                                  anneal_strategy = c("cos", "linear"),
                                  cycle_momentum = TRUE,
                                  base_momentum = 0.85, max_momentum = 0.95) {
  anneal_strategy <- match.arg(anneal_strategy)
  total_steps <- as.integer(total_steps)
  if (total_steps < 2L) stop("total_steps must be >= 2")
  if (pct_start <= 0 || pct_start >= 1) stop("pct_start must be in (0, 1)")

  env <- new.env(parent = emptyenv())
  env$optimizer       <- optimizer
  env$max_lr          <- max_lr
  env$total_steps     <- total_steps
  env$initial_lr      <- max_lr / div_factor
  env$final_lr        <- max_lr / final_div_factor
  env$anneal_strategy <- anneal_strategy
  env$step_up         <- max(1L, as.integer(round(pct_start * total_steps)))
  env$last_step       <- 0L

  env$mom_field <- if (isTRUE(cycle_momentum)) .ag_momentum_field(optimizer) else NULL
  env$base_momentum <- base_momentum
  env$max_momentum  <- max_momentum

  # anneal from a to b, progress p in [0, 1]
  env$anneal <- function(a, b, p) {
    if (env$anneal_strategy == "cos") {
      b + (a - b) * (1 + cos(pi * p)) / 2
    } else {
      a + (b - a) * p
    }
  }

  optimizer$lr <- env$initial_lr
  if (!is.null(env$mom_field)) {
    assign(env$mom_field, max_momentum, envir = optimizer)
  }

  env$step <- function() {
    env$last_step <- min(env$last_step + 1L, env$total_steps)
    s <- env$last_step

    if (s <= env$step_up) {
      p      <- s / env$step_up
      new_lr <- env$anneal(env$initial_lr, env$max_lr, p)
      new_mom <- env$anneal(env$max_momentum, env$base_momentum, p)
    } else {
      down_len <- max(1L, env$total_steps - env$step_up)
      p        <- min(1, (s - env$step_up) / down_len)
      new_lr   <- env$anneal(env$max_lr, env$final_lr, p)
      new_mom  <- env$anneal(env$base_momentum, env$max_momentum, p)
    }

    env$optimizer$lr <- new_lr
    if (!is.null(env$mom_field)) {
      assign(env$mom_field, new_mom, envir = env$optimizer)
    }
    invisible(new_lr)
  }

  env$get_lr <- function() env$optimizer$lr

  class(env) <- "lr_scheduler_onecycle"
  env
}

#' @export
print.lr_scheduler_onecycle <- function(x, ...) {
  cat(sprintf("lr_scheduler_onecycle | max_lr=%.6f | total_steps=%d | step=%d | lr=%.6f\n",
              x$max_lr, x$total_steps, x$last_step, x$get_lr()))
  invisible(x)
}

#' Linear warmup followed by cosine annealing
#'
#' Raises the learning rate linearly from \code{warmup_start_lr} to the
#' optimizer's initial \code{lr} over \code{warmup_steps} steps, then anneals it
#' to \code{lr_min} following a cosine curve over the remaining
#' \code{total_steps - warmup_steps} steps.  This is the standard transformer
#' training schedule.  Call \code{$step()} \strong{per batch}.
#'
#' @param optimizer Optimizer environment.  Its current \code{lr} is taken as
#'   the peak learning rate.
#' @param warmup_steps Number of linear warmup steps.
#' @param total_steps Total number of \code{$step()} calls (warmup included).
#' @param lr_min Final learning rate after annealing (default 0).
#' @param warmup_start_lr Learning rate at step 0 (default 0).
#' @return An \code{lr_scheduler_warmup_cosine} environment with \code{$step()}
#'   and \code{$get_lr()}.
#' @export
#' @examples
#' \donttest{
#' w   <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_adam(list(w = w), lr = 1e-3)
#' sch <- lr_scheduler_warmup_cosine(opt, warmup_steps = 10L, total_steps = 100L)
#' for (i in 1:100) sch$step()
#' }
lr_scheduler_warmup_cosine <- function(optimizer, warmup_steps, total_steps,
                                       lr_min = 0, warmup_start_lr = 0) {
  warmup_steps <- as.integer(warmup_steps)
  total_steps  <- as.integer(total_steps)
  if (warmup_steps < 0L) stop("warmup_steps must be >= 0")
  if (total_steps <= warmup_steps) stop("total_steps must be > warmup_steps")

  env <- new.env(parent = emptyenv())
  env$optimizer       <- optimizer
  env$warmup_steps    <- warmup_steps
  env$total_steps     <- total_steps
  env$lr_max          <- optimizer$lr   # current lr is the peak
  env$lr_min          <- lr_min
  env$warmup_start_lr <- warmup_start_lr
  env$last_step       <- 0L

  optimizer$lr <- if (warmup_steps > 0L) warmup_start_lr else env$lr_max

  env$step <- function() {
    env$last_step <- min(env$last_step + 1L, env$total_steps)
    s <- env$last_step

    new_lr <- if (s <= env$warmup_steps) {
      env$warmup_start_lr +
        (env$lr_max - env$warmup_start_lr) * s / env$warmup_steps
    } else {
      denom <- env$total_steps - env$warmup_steps
      p     <- (s - env$warmup_steps) / denom
      env$lr_min + 0.5 * (env$lr_max - env$lr_min) * (1 + cos(pi * p))
    }

    env$optimizer$lr <- new_lr
    invisible(new_lr)
  }

  env$get_lr <- function() env$optimizer$lr

  class(env) <- "lr_scheduler_warmup_cosine"
  env
}

#' @export
print.lr_scheduler_warmup_cosine <- function(x, ...) {
  cat(sprintf("lr_scheduler_warmup_cosine | warmup=%d | total=%d | lr_max=%.6f | step=%d | lr=%.6f\n",
              x$warmup_steps, x$total_steps, x$lr_max, x$last_step, x$get_lr()))
  invisible(x)
}
