# RMSprop for the autograd path, torch.optim.RMSprop semantics.
#
# Built like optimizer_adam: the state lives where the weights live (persistent
# pool on the device), the device step is per-op with explicit copies into the
# persistent tensors (.ag_adam_step_device_impl explains why), and the host
# step is the reference.

#' Create an RMSprop optimizer
#'
#' Same algorithm and defaults as \code{torch.optim.RMSprop}. Per step, with
#' gradient \code{g}:
#' \preformatted{
#' g   <- g + weight_decay * w                      # L2, added to the gradient
#' v   <- alpha * v + (1 - alpha) * g^2
#' ga  <- alpha * ga + (1 - alpha) * g              # centered only
#' den <- sqrt(v - ga^2) + eps                      # sqrt(v) + eps if not centered
#' buf <- momentum * buf + g / den;  w <- w - lr * buf   # momentum > 0
#' w   <- w - lr * g / den                               # momentum = 0
#' }
#' \code{eps} is added OUTSIDE the square root, as in PyTorch (Keras puts it
#' elsewhere). \code{weight_decay} is L2 regularisation, as in PyTorch's
#' RMSprop -- not the decoupled decay of \code{\link{optimizer_adamw}}.
#'
#' State slots are created only when used: \code{square_avg} always,
#' \code{momentum_buffer} when \code{momentum > 0}, \code{grad_avg} when
#' \code{centered}. On the GPU the step runs in F32 on resident state, with an
#' F32 master for weights held in f16, as for \code{\link{optimizer_adam}}.
#' \code{\link{ag_capture_step}} does not support RMSprop.
#'
#' @inheritParams optimizer_sgd
#' @param lr Learning rate (default 0.01)
#' @param alpha Smoothing constant of the squared-gradient average (default 0.99)
#' @param eps Stability constant, outside the square root (default 1e-8)
#' @param weight_decay L2 penalty added to the gradient (default 0)
#' @param momentum Momentum factor (default 0)
#' @param centered Normalise by the variance estimate instead of the raw second
#'   moment (default \code{FALSE})
#' @return An optimizer environment with \code{step(grads)}, \code{zero_grad()},
#'   \code{state()} and \code{load_state(state)}; see \code{\link{optimizer_adam}}.
#' @export
#' @examples
#' \donttest{
#' w <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_rmsprop(list(w = w), lr = 0.01, momentum = 0.9)
#' }
optimizer_rmsprop <- function(params, lr = 0.01, alpha = 0.99, eps = 1e-8,
                              weight_decay = 0, momentum = 0, centered = FALSE,
                              accumulate_steps = 1L, average = TRUE) {
  stopifnot(is.list(params))
  num_ok <- function(x) is.numeric(x) && length(x) == 1L && !is.na(x)
  if (!num_ok(alpha) || alpha < 0 || alpha > 1)
    stop("optimizer_rmsprop: alpha must be in [0, 1]", call. = FALSE)
  if (!num_ok(eps) || eps < 0)
    stop("optimizer_rmsprop: eps must be non-negative", call. = FALSE)
  if (!num_ok(weight_decay) || weight_decay < 0)
    stop("optimizer_rmsprop: weight_decay must be non-negative", call. = FALSE)
  if (!num_ok(momentum) || momentum < 0)
    stop("optimizer_rmsprop: momentum must be non-negative", call. = FALSE)
  accumulate_steps <- as.integer(accumulate_steps)
  if (is.na(accumulate_steps) || accumulate_steps < 1L)
    stop("accumulate_steps must be a positive integer")

  env <- new.env(parent = emptyenv())
  params <- .ag_opt_unique_params(params)
  env$params       <- params
  env$lr           <- lr
  env$alpha        <- alpha
  env$eps          <- eps
  env$weight_decay <- weight_decay
  env$momentum     <- momentum
  env$centered     <- isTRUE(centered)
  env$accumulate_steps <- accumulate_steps
  env$average          <- isTRUE(average)
  env$accum_count      <- 0L
  # Residency decided once, as in optimizer_adam.
  env$resident <- all(vapply(params, function(p) !is.null(.ag_handle_of(p)),
                             logical(1))) && length(params) > 0L
  zeros <- function() lapply(params, function(p) .ag_opt_zero_like(p, env$resident))
  env$square_avg <- zeros()
  if (momentum > 0) env$momentum_buffer <- zeros()
  if (env$centered) env$grad_avg <- zeros()

  env$step <- function(grads = NULL) {
    n <- .ag_opt_accum_tick(env)
    if (is.null(n)) return(invisible(FALSE))
    # momentum switched on after construction (by hand or a scheduler)
    if (env$momentum > 0 && is.null(env$momentum_buffer))
      env$momentum_buffer <- lapply(env$params, function(p)
        .ag_opt_zero_like(p, env$resident))
    for (i in seq_along(env$params)) {
      p <- env$params[[i]]
      g <- .ag_opt_grad_for(p, grads, n, as_matrix = !env$resident)
      if (is.null(g)) next
      if (env$resident && .ag_handle_live(env$square_avg[[i]]) &&
          !is.null(.ag_handle_of(p))) {
        gh <- if (.ag_is_handle(g)) g else
                .ag_handle(.ag_r_to_gpu(g, dtype = "f32", scope = "pass"), dim(g),
                           scope = "pass")
        .ag_defer_suspend(.ag_rmsprop_step_device(env, i, p, gh))
        next
      }
      .ag_rmsprop_step_host(env, i, p, .ag_as_matrix(g))
    }
    invisible(TRUE)
  }

  env$zero_grad <- function() {
    for (i in seq_along(env$params)) env$params[[i]]$grad <- NULL
    .ag_forget_pending_grads()
    .ag_tape$nodes <- list()
  }

  env$state <- function() .ag_opt_state(env, "rmsprop", .ag_rmsprop_slots(env),
    .ag_rmsprop_hyper(env), master = TRUE)
  env$load_state <- function(state) .ag_opt_load_state(env, state, "rmsprop",
    .ag_rmsprop_slots(env), .ag_rmsprop_hyper(env))

  class(env) <- "ag_optimizer_rmsprop"
  env
}

.ag_rmsprop_slots <- function(env) {
  c("square_avg",
    if (!is.null(env$momentum_buffer)) "momentum_buffer",
    if (isTRUE(env$centered)) "grad_avg")
}

.ag_rmsprop_hyper <- function(env) {
  list(lr = env$lr, alpha = env$alpha, eps = env$eps,
       weight_decay = env$weight_decay, momentum = env$momentum,
       centered = env$centered)
}

# Host step: the reference. Slots that are handles (residency lapsed) are
# materialised once; a dead handle restarts from zeros, as in Adam.
.ag_rmsprop_step_host <- function(env, i, p, g) {
  slot <- function(s) {
    x <- env[[s]][[i]]
    if (.ag_is_handle(x)) x <- tryCatch(.ag_as_matrix(x), error = function(e) g * 0)
    x
  }
  w <- .ag_data_mut(p)
  if (env$weight_decay != 0) g <- g + env$weight_decay * w
  a <- env$alpha
  v <- a * slot("square_avg") + (1 - a) * g^2
  env$square_avg[[i]] <- v
  if (env$centered) {
    ga <- a * slot("grad_avg") + (1 - a) * g
    env$grad_avg[[i]] <- ga
    den <- sqrt(v - ga^2) + env$eps
  } else {
    den <- sqrt(v) + env$eps
  }
  if (env$momentum > 0) {
    buf <- env$momentum * slot("momentum_buffer") + g / den
    env$momentum_buffer[[i]] <- buf
    .ag_opt_store_weight(p, w - env$lr * buf)
  } else {
    .ag_opt_store_weight(p, w - env$lr * g / den)
  }
}

# Device step. New values are computed into pass-pool tensors, then copied
# into the persistent ones, each copy its own graph: a tensor must not be both
# an operand and the destination of one graph (see .ag_adam_step_device_impl).
.ag_rmsprop_step_device <- function(env, i, p, g) {
  wh <- .ag_handle_of(p)
  mw <- .ag_master_of(p)                  # F32: the weight itself, or its master
  sh <- .ag_dim(env$square_avg[[i]])
  a  <- env$alpha; eps <- env$eps; lr <- env$lr; mom <- env$momentum; wd <- env$weight_decay
  op <- function(f, inputs)
    .ag_run_op(f, inputs = inputs, out_shape = sh, dtype = "f32",
               scope = "pass", resident = TRUE)

  # L2: g + wd * w, in F32
  g2 <- if (wd != 0)
          op(function(ctx, ptrs) ggml_add(ctx, .ag_f32_operand(ctx, ptrs[[1L]]),
                                          ggml_scale(ctx, ptrs[[2L]], wd)),
             list(g, mw))
        else g

  new_v <- op(function(ctx, ptrs)
                ggml_add(ctx, ggml_scale(ctx, ptrs[[1L]], a),
                         ggml_scale(ctx, ggml_sqr(ctx, .ag_f32_operand(ctx, ptrs[[2L]])),
                                    1 - a)),
              list(env$square_avg[[i]], g2))
  new_ga <- if (env$centered)
              op(function(ctx, ptrs)
                   ggml_add(ctx, ggml_scale(ctx, ptrs[[1L]], a),
                            ggml_scale(ctx, .ag_f32_operand(ctx, ptrs[[2L]]), 1 - a)),
                 list(env$grad_avg[[i]], g2))

  # g / (sqrt(v [- ga^2]) + eps)
  upd <- op(function(ctx, ptrs) {
              v <- ptrs[[2L]]
              if (length(ptrs) == 3L) v <- ggml_sub(ctx, v, ggml_sqr(ctx, ptrs[[3L]]))
              den <- ggml_scale_bias(ctx, ggml_sqrt(ctx, v), 1, eps)
              ggml_div(ctx, .ag_f32_operand(ctx, ptrs[[1L]]), den)
            },
            c(list(g2, new_v), if (env$centered) list(new_ga)))

  new_buf <- if (mom > 0)
               op(function(ctx, ptrs)
                    ggml_add(ctx, ggml_scale(ctx, ptrs[[1L]], mom), ptrs[[2L]]),
                  list(env$momentum_buffer[[i]], upd))
  step <- if (mom > 0) new_buf else upd
  new_w <- op(function(ctx, ptrs) ggml_sub(ctx, ptrs[[1L]], ggml_scale(ctx, ptrs[[2L]], lr)),
              list(mw, step))

  cpy <- function(src, dst)
    .ag_run_op(function(ctx, ptrs) ggml_dup(ctx, ptrs[[1L]]),
               inputs = list(src), out_shape = sh, dtype = "f32",
               scope = "pass", out = dst)
  cpy(new_w, mw)
  if (!identical(mw$ptr, wh$ptr)) cpy(new_w, wh)
  cpy(new_v, env$square_avg[[i]])
  if (env$centered) cpy(new_ga, env$grad_avg[[i]])
  if (mom > 0) cpy(new_buf, env$momentum_buffer[[i]])

  p$data     <- NULL
  p$data_gen <- NULL
  invisible(NULL)
}

#' @export
print.ag_optimizer_rmsprop <- function(x, ...) {
  cat("RMSprop optimizer | lr =", x$lr, "| alpha =", x$alpha,
      "| momentum =", x$momentum, "| centered =", x$centered,
      "| weight_decay =", x$weight_decay,
      "| params:", length(x$params), "\n")
  invisible(x)
}
