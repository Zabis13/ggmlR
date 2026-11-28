# High-level layers for the dynamic autograd engine (ag_*).
#
# Design:
#   - Every layer is an R environment (reference semantics).
#   - Required fields: $forward(x), $parameters() -> named list of ag_param.
#   - Stateful layers (BatchNorm, Dropout) carry a $training flag.
#   - ag_sequential() wraps a list of layers and manages train/eval state.
#
# Note on ag_linear vs layer objects:
#   ag_linear() (in autograd.R) returns a plain list with key "params".
#   All layer objects created here use environments and expose "parameters".
#   ag_sequential$parameters() handles both via .layer_params() helper.

# ============================================================================
# Internal helper: extract parameters from any layer object
# ============================================================================

.layer_params <- function(lyr) {
  # environment-based layer: $parameters()
  if (is.environment(lyr) && is.function(lyr$parameters)) {
    return(lyr$parameters())
  }
  # ag_linear (plain list) uses $params()
  if (is.list(lyr) && is.function(lyr[["params"]])) {
    return(lyr[["params"]]())
  }
  # layer with $parameters as a function in a list
  if (is.list(lyr) && is.function(lyr[["parameters"]])) {
    return(lyr[["parameters"]]())
  }
  list()
}

# ============================================================================
# Train / eval mode helpers
# ============================================================================

#' Switch a layer or sequential model to training mode
#' @param model An ag_sequential, ag_batch_norm, or ag_dropout layer
#' @return The model/layer (invisibly)
#' @export
ag_train <- function(model) {
  UseMethod("ag_train")
}

#' Switch a layer or sequential model to eval mode
#' @param model An ag_sequential, ag_batch_norm, or ag_dropout layer
#' @return The model/layer (invisibly)
#' @export
ag_eval <- function(model) {
  UseMethod("ag_eval")
}

#' @export
ag_train.default <- function(model) {
  if (is.environment(model)) model$training <- TRUE
  invisible(model)
}
#' @export
ag_eval.default <- function(model) {
  if (is.environment(model)) model$training <- FALSE
  invisible(model)
}

# ============================================================================
# ag_sequential
# ============================================================================

#' Create a sequential container of layers
#'
#' Chains layers so that \code{forward(x)} passes \code{x} through each layer
#' in order.  \code{parameters()} collects all trainable params from all layers.
#' \code{ag_train()} / \code{ag_eval()} propagate mode to stateful sub-layers.
#'
#' @param ... Layer objects (ag_linear, ag_dropout, ag_batch_norm, ag_embedding)
#'   or a single list of layers.
#' @return An \code{ag_sequential} environment
#' @export
#' @examples
#' \donttest{
#' model <- ag_sequential(
#'   ag_linear(4L, 16L, activation = "relu"),
#'   ag_dropout(0.5),
#'   ag_linear(16L, 2L, activation = "softmax")
#' )
#' x   <- ag_tensor(matrix(runif(4 * 8), 4, 8))
#' out <- model$forward(x)
#' }
ag_sequential <- function(...) {
  layers <- list(...)
  # unwrap if a single list was passed
  if (length(layers) == 1L && is.list(layers[[1L]]) &&
      !is.environment(layers[[1L]])) {
    layers <- layers[[1L]]
  }

  env <- new.env(parent = emptyenv())
  env$layers   <- layers
  env$training <- TRUE

  env$forward <- function(x) {
    for (lyr in env$layers) {
      x <- lyr$forward(x)
    }
    x
  }

  env$parameters <- function() {
    params <- list()
    for (i in seq_along(env$layers)) {
      lp <- .layer_params(env$layers[[i]])
      for (nm in names(lp)) {
        params[[paste0("layer", i, "_", nm)]] <- lp[[nm]]
      }
    }
    params
  }

  class(env) <- c("ag_sequential", "ag_layer")
  env
}

#' @export
ag_train.ag_sequential <- function(model) {
  model$training <- TRUE
  for (lyr in model$layers) {
    if (is.environment(lyr) && !is.null(lyr$training)) lyr$training <- TRUE
  }
  invisible(model)
}

#' @export
ag_eval.ag_sequential <- function(model) {
  model$training <- FALSE
  for (lyr in model$layers) {
    if (is.environment(lyr) && !is.null(lyr$training)) lyr$training <- FALSE
  }
  invisible(model)
}

#' @export
print.ag_sequential <- function(x, ...) {
  cat("ag_sequential (", length(x$layers), " layers, ",
      if (x$training) "train" else "eval", " mode)\n", sep = "")
  for (i in seq_along(x$layers)) {
    lyr <- x$layers[[i]]
    nm  <- if (!is.null(lyr$name)) lyr$name else class(lyr)[1L]
    cat(sprintf("  [%d] %s\n", i, nm))
  }
  invisible(x)
}

# ============================================================================
# ag_dropout
# ============================================================================

#' Create a Dropout layer
#'
#' In training mode applies inverted dropout (random Bernoulli mask, scale by
#' \code{1/(1-rate)} to preserve expected values).  In eval mode is identity.
#'
#' @param rate Drop probability in [0, 1)
#' @return An \code{ag_dropout} environment
#' @export
#' @examples
#' \donttest{
#' drop <- ag_dropout(0.5)
#' x    <- ag_tensor(matrix(runif(8), 4, 2))
#' out  <- drop$forward(x)  # training mode by default
#' ag_eval(drop)
#' out2 <- drop$forward(x)  # identity
#' }
ag_dropout <- function(rate) {
  rate <- as.double(rate)
  stopifnot(rate >= 0, rate < 1)

  env <- new.env(parent = emptyenv())
  env$rate     <- rate
  env$training <- TRUE
  env$name     <- paste0("dropout(", rate, ")")

  env$forward <- function(x) {
    if (!env$training || env$rate == 0) return(x)
    x_data <- .ag_data(x)
    mask_vals <- matrix(
      (stats::runif(length(x_data)) > env$rate) / (1 - env$rate),
      nrow(x_data), ncol(x_data)
    )
    ag_mul(x, ag_tensor(mask_vals))
  }

  env$parameters <- function() list()

  class(env) <- c("ag_dropout", "ag_layer")
  env
}

#' @export
ag_train.ag_dropout <- function(model) { model$training <- TRUE;  invisible(model) }
#' @export
ag_eval.ag_dropout  <- function(model) { model$training <- FALSE; invisible(model) }

# ============================================================================
# ag_batch_norm
# ============================================================================

#' Create a Batch Normalisation layer
#'
#' Normalises each feature (row) over the batch dimension.
#' Learnable scale \code{gamma} [F,1] and shift \code{beta} [F,1].
#'
#' \strong{Training mode}: use batch statistics; update running mean/var.
#' \strong{Eval mode}: use stored running statistics.
#'
#' The running statistics live in one \code{[F, 2]} \code{\link{ag_buffer}}
#' (\code{$stats}: column 0 the mean, column 1 the variance), so on the GPU
#' they stay on the device and are updated in place, at the same address,
#' every training step; \code{$running_mean} and \code{$running_var} read and
#' write that buffer. The variance is the biased batch variance.
#'
#' On the GPU the normalisation is one tape node (\code{ggml_norm} over the
#' batch axis; backward \code{ggml_norm_back}, or \code{g / std} in eval mode)
#' and \code{gamma}, \code{beta} an ordinary \code{\link{ag_mul}} and
#' \code{\link{ag_add}}, so the layer runs in the graph backward.
#'
#' @param num_features Number of features (rows of input)
#' @param eps Numerical stability constant (default 1e-5)
#' @param momentum Running-stats momentum (default 0.1)
#' @return An \code{ag_batch_norm} environment
#' @export
#' @examples
#' \donttest{
#' bn <- ag_batch_norm(16L)
#' x  <- ag_tensor(matrix(rnorm(16 * 32), 16, 32))
#' out <- bn$forward(x)
#' }
ag_batch_norm <- function(num_features, eps = 1e-5, momentum = 0.1) {
  num_features <- as.integer(num_features)

  env <- new.env(parent = emptyenv())
  env$gamma        <- ag_param(matrix(1.0, num_features, 1L))
  env$beta         <- ag_param(matrix(0.0, num_features, 1L))
  env$stats        <- ag_buffer(num_features, 2L)
  ag_buffer_write(env$stats, cbind(rep(0, num_features), rep(1, num_features)), 0)
  env$num_features <- num_features
  env$eps          <- eps
  env$momentum     <- momentum
  env$training     <- TRUE
  env$name         <- paste0("batch_norm(", num_features, ")")

  # The old matrix fields, kept as views of the buffer: ag_save()/ag_load()
  # and user code read and assign them as before.
  stat_binding <- function(j) function(value) {
    if (missing(value)) return(ag_buffer_read(env$stats, j, 1L))
    ag_buffer_write(env$stats, matrix(as.numeric(value), num_features, 1L), j)
  }
  makeActiveBinding("running_mean", stat_binding(0L), env)
  makeActiveBinding("running_var",  stat_binding(1L), env)

  env$forward <- function(x) {
    gpu <- is_ag_tensor(x) && identical(x$device, "gpu") && .ag_buffer_gpu()
    x_hat <- if (gpu) .ag_bn_normalize_gpu(env, x) else .ag_bn_normalize_cpu(env, x)
    ag_add(ag_mul(x_hat, env$gamma), env$beta)
  }

  env$parameters <- function() list(gamma = env$gamma, beta = env$beta)

  class(env) <- c("ag_batch_norm", "ag_layer")
  env
}

# Host path: statistics in R, one closure node (no op record).
.ag_bn_normalize_cpu <- function(env, x) {
  x_data <- .ag_data(x)
  n      <- ncol(x_data)
  m      <- env$momentum

  if (env$training) {
    mu  <- rowMeans(x_data)
    var <- rowMeans((x_data - mu)^2)
    # update running stats (no gradient through this write)
    st <- ag_buffer_read(env$stats)
    ag_buffer_write(env$stats, cbind((1 - m) * st[, 1L] + m * mu,
                                     (1 - m) * st[, 2L] + m * var), 0)
  } else {
    st  <- ag_buffer_read(env$stats)
    mu  <- st[, 1L]
    var <- st[, 2L]
  }

  std     <- sqrt(var + env$eps)
  std_m   <- matrix(std, env$num_features, n)
  xh_data <- (x_data - mu) / std_m
  x_hat   <- ag_tensor(xh_data)
  x_hat$requires_grad <- is_ag_tensor(x) && x$requires_grad

  if (x_hat$requires_grad) {
    if (env$training) {
      # mu and var come from this very batch, so their dependence on x
      # contributes:  dx = (g - mean_N(g) - x_hat * mean_N(g * x_hat)) / std
      grad_fn <- function(grad_out) {
        list(x = (grad_out - rowMeans(grad_out) -
                    xh_data * rowMeans(grad_out * xh_data)) / std_m)
      }
    } else {
      # running statistics are constants w.r.t. x
      grad_fn <- function(grad_out) list(x = grad_out / std_m)
    }
    x_hat$grad_fn <- grad_fn
    ag_record(x_hat, grad_fn, list(x = x))
  }
  x_hat
}

# Device path. Training: x_hat = ggml_norm over the batch axis (ggml_norm
# normalises ne0, so x is transposed around it), and the running statistics
# are updated in place in the buffer -- new values computed into the pass
# pool, then copied onto the buffer through `out =` (the Adam-step pattern), so
# the address never changes. Eval: (x - mean) / sqrt(var + eps) read straight
# from the buffer. One tape node, op "batch_norm".
.ag_bn_normalize_gpu <- function(env, x) {
  F   <- env$num_features
  eps <- env$eps
  m   <- env$momentum
  core <- .ag_buffer_core(env$stats)
  .ag_buffer_on_device(core)
  h <- .ag_handle(core$ptr, c(F, 2L), scope = "persistent")
  x_op <- .ag_operand(x)
  if (.ag_nrow(x_op) != F)
    stop(sprintf("ag_batch_norm: expected %d features, got %d", F, .ag_nrow(x_op)),
         call. = FALSE)
  n <- .ag_ncol(x_op)
  # column j (0-based) of the [F, 2] statistics as an [F, 1] view
  col <- function(ctx, hp, j) {
    nb1 <- ggml_tensor_nb(hp)[2L]
    ggml_view_2d(ctx, hp, F, 1L, nb1, j * nb1)
  }
  training <- isTRUE(env$training)
  std_h <- NULL
  # This layer's previous statistics write may still be queued (ag_graph_mode).
  # Both branches below read the buffer, and a read and that write in one graph
  # have no data dependency -- the Vulkan backend may reorder them -- so the
  # queue runs first. Only then: a training step's barrier usually ran it.
  if (!is.null(env$stats_write) && !.ag_handle_computed(env$stats_write) &&
      !isTRUE(.ag_capture_reg$recording))
    .ag_defer_drain()
  env$stats_write <- NULL

  if (training) {
    res <- .ag_run_op(
      op_fn = function(ctx, ptrs) {
        xt <- ggml_cont(ctx, ggml_transpose(ctx, ptrs[[1L]]))
        ggml_cont(ctx, ggml_transpose(ctx, ggml_norm(ctx, xt, eps)))
      },
      inputs = list(x_op), out_shape = c(F, n), resident = TRUE)
    out <- .ag_wrap_result(res, "gpu")

    # new [F, 2] statistics from x (ptrs[[1]]) and the current ones (ptrs[[2]])
    new_stats <- function(ctx, ptrs) {
      xt <- ggml_cont(ctx, ggml_transpose(ctx, .ag_f32_operand(ctx, ptrs[[1L]])))
      mu <- ggml_mean(ctx, xt)                                   # [1, F]
      va <- ggml_mean(ctx, ggml_sqr(ctx, ggml_sub(ctx, xt, mu))) # biased
      rm <- ggml_add(ctx, ggml_scale(ctx, col(ctx, ptrs[[2L]], 0L), 1 - m),
                     ggml_scale(ctx, ggml_reshape_2d(ctx, mu, F, 1L), m))
      rv <- ggml_add(ctx, ggml_scale(ctx, col(ctx, ptrs[[2L]], 1L), 1 - m),
                     ggml_scale(ctx, ggml_reshape_2d(ctx, va, F, 1L), m))
      ggml_concat(ctx, rm, rv, 1L)
    }
    # The write is ONE node: ggml_cpy of the new statistics onto the buffer
    # tensor itself (h$ptr, not its operand alias ptrs[[2]], which is a leaf
    # with no buffer of its own as a destination). Read through the alias,
    # written to the real tensor: the copy depends on the reads, so it runs
    # after them. Without deferral it runs at once; under ag_graph_mode it is
    # queued with the forward and runs at the next barrier (a read of the
    # statistics, backward, the next tape's reset -- all drain first).
    #
    # Not "compute into the pass pool, then copy through out =" (the Adam
    # pattern): under ag_graph_mode that pair lost the write -- measured, the
    # statistics stayed at their initial values.
    #
    # Inside ag_capture / ag_capture_step the same node is kept as an effect
    # (as ag_set_rows() does), so every replay updates the statistics in place,
    # and the buffer is noted so that moving or freeing it re-records.
    recording <- isTRUE(.ag_capture_reg$recording)
    if (recording) .ag_capture_note_buffer(env$stats)
    dst <- h$ptr
    eff <- .ag_run_op(
      op_fn = function(ctx, ptrs) ggml_cpy(ctx, new_stats(ctx, ptrs), dst),
      inputs = list(x_op, h), out_shape = c(F, 2L), dtype = "f32",
      resident = TRUE)
    if (recording)
      .ag_capture_reg$effects[[length(.ag_capture_reg$effects) + 1L]] <- eff
    else
      env$stats_write <- eff
  } else {
    # eval reads the buffer: a recording depends on it too
    .ag_capture_note_buffer(env$stats)
    std_h <- .ag_run_op(
      op_fn = function(ctx, ptrs)
        ggml_sqrt(ctx, ggml_scale_bias(ctx, col(ctx, ptrs[[1L]], 1L), 1, eps)),
      inputs = list(h), out_shape = c(F, 1L), dtype = "f32", resident = TRUE)
    res <- .ag_run_op(
      op_fn = function(ctx, ptrs)
        ggml_div(ctx, ggml_sub(ctx, .ag_f32_operand(ctx, ptrs[[1L]]),
                               col(ctx, ptrs[[2L]], 0L)), ptrs[[3L]]),
      inputs = list(x_op, h, std_h), out_shape = c(F, n), dtype = "f32",
      resident = TRUE)
    out <- .ag_wrap_result(res, "gpu", dtype = "f32")
  }

  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    # Closure (fallback path): exact, in double; downloads x (or std) only
    # when it actually runs.
    grad_fn <- function(grad_out) {
      if (training) {
        xm  <- .ag_as_matrix(x_op)
        mu  <- rowMeans(xm)
        std <- sqrt(rowMeans((xm - mu)^2) + eps)
        xh  <- (xm - mu) / std
        list(x = (grad_out - rowMeans(grad_out) -
                    xh * rowMeans(grad_out * xh)) / std)
      } else {
        list(x = grad_out / as.numeric(.ag_as_matrix(std_h)))
      }
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x), op = "batch_norm",
              mode = if (training) "train" else "eval",
              x_snap = x_op, std_snap = std_h, eps = eps)
  }
  out
}

#' @export
ag_train.ag_batch_norm <- function(model) { model$training <- TRUE;  invisible(model) }
#' @export
ag_eval.ag_batch_norm  <- function(model) { model$training <- FALSE; invisible(model) }

# ============================================================================
# ag_layer_norm
# ============================================================================

#' Create a Layer Normalization layer
#'
#' Normalizes each example independently over its \code{normalized_shape}
#' features, then applies a learnable per-feature scale (\code{gamma}) and shift
#' (\code{beta}):  \code{y = gamma * (x - mu) / sqrt(var + eps) + beta}.
#'
#' Unlike \code{\link{ag_batch_norm}}, which normalizes each feature across the
#' batch, LayerNorm normalizes each column (example) across its features.  It
#' therefore has no running statistics and behaves identically in training and
#' evaluation mode, which makes it the usual choice for transformers and for
#' small or varying batch sizes.
#'
#' The backward pass is exact: because \code{mu} and \code{sigma} are computed
#' from the same example being normalized, their contribution to the gradient is
#' included rather than treated as constant. The normalization is a single tape
#' node (\code{ggml_norm} / \code{ggml_norm_back}); \code{gamma} and
#' \code{beta} are an ordinary \code{\link{ag_mul}} and \code{\link{ag_add}}.
#' On the GPU all three stay on the device and run in the graph backward, so
#' the layer can be part of an \code{\link{ag_capture_step}} recording.
#'
#' @param normalized_shape Number of features to normalize over (the number of
#'   rows of the input).
#' @param eps Small constant added to the variance for numerical stability
#'   (default 1e-5).
#' @param elementwise_affine Logical; when \code{TRUE} (default) the layer
#'   learns \code{gamma} and \code{beta}.  When \code{FALSE} it normalizes only
#'   and has no parameters.
#' @return An \code{ag_layer_norm} environment with \code{$forward()} and
#'   \code{$parameters()}.
#' @seealso \code{\link{ag_batch_norm}}
#' @export
#' @examples
#' \donttest{
#' ln <- ag_layer_norm(4L)
#' x  <- ag_param(matrix(rnorm(8), 4, 2))   # [features, batch]
#' with_grad_tape({
#'   y    <- ln$forward(x)
#'   loss <- ag_mse_loss(y, matrix(0, 4, 2))
#' })
#' grads <- backward(loss)
#' }
ag_layer_norm <- function(normalized_shape, eps = 1e-5,
                          elementwise_affine = TRUE) {
  normalized_shape <- as.integer(normalized_shape)
  if (length(normalized_shape) != 1L || is.na(normalized_shape) ||
      normalized_shape < 1L) {
    stop("normalized_shape must be a single positive integer")
  }

  env <- new.env(parent = emptyenv())
  env$normalized_shape   <- normalized_shape
  env$eps                <- eps
  env$elementwise_affine <- isTRUE(elementwise_affine)
  env$training           <- TRUE
  env$name               <- paste0("layer_norm(", normalized_shape, ")")

  if (env$elementwise_affine) {
    env$gamma <- ag_param(matrix(1.0, normalized_shape, 1L))
    env$beta  <- ag_param(matrix(0.0, normalized_shape, 1L))
  } else {
    env$gamma <- NULL
    env$beta  <- NULL
  }

  env$forward <- function(x) {
    # Shape only: a resident x is read through its handle, never downloaded
    # (as in ag_rms_norm).
    xs <- if (is_ag_tensor(x)) {
      h <- .ag_handle_of(x); if (!is.null(h)) h else .ag_data(x)
    } else x
    d <- if (.ag_is_handle(xs) || !is.null(dim(xs))) .ag_nrow(xs) else length(xs)
    if (d != env$normalized_shape) {
      stop(sprintf("ag_layer_norm: expected %d features, got %d",
                   env$normalized_shape, d))
    }
    # One tape node for the normalisation (ggml_norm / ggml_norm_back), gamma
    # and beta an ordinary ag_mul / ag_add: all run in the graph backward.
    y <- .ag_layer_normalize(x, env$eps)
    if (env$elementwise_affine) y <- ag_add(ag_mul(y, env$gamma), env$beta)
    y
  }

  env$parameters <- function() {
    if (env$elementwise_affine) list(gamma = env$gamma, beta = env$beta) else list()
  }

  class(env) <- c("ag_layer_norm", "ag_layer")
  env
}

#' @export
print.ag_layer_norm <- function(x, ...) {
  cat(sprintf("ag_layer_norm(%d) | eps=%g | affine=%s\n",
              x$normalized_shape, x$eps, x$elementwise_affine))
  invisible(x)
}

#' @export
ag_train.ag_layer_norm <- function(model) { model$training <- TRUE;  invisible(model) }
#' @export
ag_eval.ag_layer_norm  <- function(model) { model$training <- FALSE; invisible(model) }

# ============================================================================
# ag_rms_norm
# ============================================================================

# RMS normalisation of each column, no affine part: x / sqrt(mean(x^2) + eps).
# One tape node (forward ggml_rms_norm, backward ggml_rms_norm_back), so unlike
# ag_layer_norm -- which computes in R and records no `op` -- it keeps a tape
# eligible for the graph backward and for ag_capture_step.
.ag_rms_normalize <- function(x, eps) {
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    x_data <- .ag_operand(x)
    out    <- .ag_wrap_result(.ag_gpu_rms_norm(x_data, eps), device)
  } else {
    x_data <- .ag_data(x)
    if (!is.matrix(x_data)) x_data <- matrix(x_data, ncol = 1L)
    d <- nrow(x_data)
    r <- sqrt(colMeans(x_data^2) + eps)
    out <- ag_tensor(x_data / rep(r, each = d), device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    x_snap <- x_data
    grad_fn <- function(grad_out) {
      # y = x / r, r = sqrt(mean(x^2) + eps), per column:
      # dx = (g - y * mean(g * y)) / r
      xm <- .ag_as_matrix(x_snap)
      d  <- nrow(xm)
      r  <- sqrt(colMeans(xm^2) + eps)
      y  <- xm / rep(r, each = d)
      list(x = (grad_out - y * rep(colMeans(grad_out * y), each = d)) /
               rep(r, each = d))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x), op = "rms_norm", x_snap = x_snap,
              eps = eps)
  }
  out
}

# (x - mean) / sqrt(var + eps) per column, var the biased (population)
# variance, as torch.nn.LayerNorm. One tape node "layer_norm": forward
# ggml_norm on the GPU, graph backward one ggml_norm_back node.
.ag_layer_normalize <- function(x, eps) {
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    x_data <- .ag_operand(x)
    out    <- .ag_wrap_result(.ag_gpu_layer_norm(x_data, eps), device)
  } else {
    x_data <- .ag_data(x)
    if (!is.matrix(x_data)) x_data <- matrix(x_data, ncol = 1L)
    d  <- nrow(x_data)
    xc <- x_data - rep(colMeans(x_data), each = d)
    out <- ag_tensor(xc / rep(sqrt(colMeans(xc^2) + eps), each = d), device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    x_snap <- x_data
    grad_fn <- function(grad_out) {
      # y = (x - mu) / s, s = sqrt(var + eps), per column:
      # dx = (g - mean(g) - y * mean(g * y)) / s
      xm <- .ag_as_matrix(x_snap)
      d  <- nrow(xm)
      xc <- xm - rep(colMeans(xm), each = d)
      s  <- sqrt(colMeans(xc^2) + eps)
      y  <- xc / rep(s, each = d)
      list(x = (grad_out - rep(colMeans(grad_out), each = d) -
                  y * rep(colMeans(grad_out * y), each = d)) / rep(s, each = d))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x), op = "layer_norm", x_snap = x_snap,
              eps = eps)
  }
  out
}

#' Create an RMS Normalization layer
#'
#' Normalizes each example (column) by its root mean square over the
#' \code{normalized_shape} features, then applies a learnable per-feature scale
#' (\code{gamma}) and shift (\code{beta}):
#' \code{y = gamma * x / sqrt(mean(x^2) + eps) + beta}.
#'
#' This is the normalization of \code{\link{ggml_layer_rms_norm}} and of
#' \code{\link{ggml_layer_transformer_block}} (\code{norm = "rms"}), with the
#' same parameters and the same default \code{eps}, so weights carry over
#' between the two APIs. The normalization itself is a single tape node
#' (\code{ggml_rms_norm} / \code{ggml_rms_norm_back}); \code{gamma} and
#' \code{beta} are an ordinary \code{\link{ag_mul}} and \code{\link{ag_add}}.
#' All three run in the graph backward, so the layer can be part of an
#' \code{\link{ag_capture_step}} recording, as can \code{\link{ag_layer_norm}}.
#'
#' @param normalized_shape Number of features to normalize over (the number of
#'   rows of the input).
#' @param eps Small constant added to the mean square (default 1e-5).
#' @param elementwise_affine Logical; when \code{TRUE} (default) the layer
#'   learns \code{gamma} and \code{beta}. When \code{FALSE} it normalizes only
#'   and has no parameters.
#' @return An \code{ag_rms_norm} environment with \code{$forward()} and
#'   \code{$parameters()}.
#' @seealso \code{\link{ag_layer_norm}}
#' @export
#' @examples
#' \donttest{
#' rn <- ag_rms_norm(4L)
#' x  <- ag_param(matrix(rnorm(8), 4, 2))   # [features, batch]
#' with_grad_tape({
#'   y    <- rn$forward(x)
#'   loss <- ag_mse_loss(y, matrix(0, 4, 2))
#' })
#' grads <- backward(loss)
#' }
ag_rms_norm <- function(normalized_shape, eps = 1e-5,
                        elementwise_affine = TRUE) {
  normalized_shape <- as.integer(normalized_shape)
  if (length(normalized_shape) != 1L || is.na(normalized_shape) ||
      normalized_shape < 1L) {
    stop("normalized_shape must be a single positive integer")
  }

  env <- new.env(parent = emptyenv())
  env$normalized_shape   <- normalized_shape
  env$eps                <- eps
  env$elementwise_affine <- isTRUE(elementwise_affine)
  env$training           <- TRUE
  env$name               <- paste0("rms_norm(", normalized_shape, ")")

  if (env$elementwise_affine) {
    env$gamma <- ag_param(matrix(1.0, normalized_shape, 1L))
    env$beta  <- ag_param(matrix(0.0, normalized_shape, 1L))
  } else {
    env$gamma <- NULL
    env$beta  <- NULL
  }

  env$forward <- function(x) {
    # Shape only: a resident x is read through its handle, never downloaded,
    # and a host x is not uploaded just to be measured.
    xs <- if (is_ag_tensor(x)) {
      h <- .ag_handle_of(x); if (!is.null(h)) h else .ag_data(x)
    } else x
    d <- if (.ag_is_handle(xs) || !is.null(dim(xs))) .ag_nrow(xs) else length(xs)
    if (d != env$normalized_shape) {
      stop(sprintf("ag_rms_norm: expected %d features, got %d",
                   env$normalized_shape, d))
    }
    y <- .ag_rms_normalize(x, env$eps)
    if (env$elementwise_affine) y <- ag_add(ag_mul(y, env$gamma), env$beta)
    y
  }

  env$parameters <- function() {
    if (env$elementwise_affine) list(gamma = env$gamma, beta = env$beta) else list()
  }

  class(env) <- c("ag_rms_norm", "ag_layer")
  env
}

#' @export
print.ag_rms_norm <- function(x, ...) {
  cat(sprintf("ag_rms_norm(%d) | eps=%g | affine=%s\n",
              x$normalized_shape, x$eps, x$elementwise_affine))
  invisible(x)
}

#' @export
ag_train.ag_rms_norm <- function(model) { model$training <- TRUE;  invisible(model) }
#' @export
ag_eval.ag_rms_norm  <- function(model) { model$training <- FALSE; invisible(model) }

# ============================================================================
# ag_embedding
# ============================================================================

#' Create an Embedding layer
#'
#' Maps 0-based integer indices to dense vectors via table lookup.
#' Input: integer matrix or vector of 0-based indices.
#' Output: float tensor \code{[dim, length(idx)]}.
#'
#' Backward: scatter-add — only the looked-up rows accumulate gradient.
#'
#' @param vocab_size Vocabulary size
#' @param dim Embedding dimension
#' @return An \code{ag_embedding} environment
#' @export
#' @examples
#' \donttest{
#' emb <- ag_embedding(100L, 16L)
#' idx <- c(0L, 3L, 7L, 2L)
#' out <- emb$forward(idx)   # [16, 4]
#' }
ag_embedding <- function(vocab_size, dim) {
  vocab_size <- as.integer(vocab_size)
  dim        <- as.integer(dim)
  limit      <- sqrt(1.0 / vocab_size)

  env <- new.env(parent = emptyenv())
  env$weight     <- ag_param(
    matrix(stats::runif(vocab_size * dim, -limit, limit), dim, vocab_size)
  )
  env$vocab_size <- vocab_size
  env$dim        <- dim
  env$training   <- TRUE
  env$name       <- paste0("embedding(", vocab_size, ", ", dim, ")")

  env$forward <- function(idx) {
    idx_num <- as.numeric(.ag_data(idx))
    if (anyNA(idx_num) || any(idx_num != round(idx_num)) ||
        any(idx_num < 0) || any(idx_num > vocab_size - 1))
      stop(sprintf("ag_embedding: idx must be whole numbers in 0..%d",
                   vocab_size - 1L), call. = FALSE)
    idx_int <- as.integer(idx_num)
    n       <- length(idx_int)
    if (n < 1L) stop("ag_embedding: idx is empty", call. = FALSE)

    if (identical(env$weight$device, "gpu")) {
      # The table stays on the device: one ggml_get_rows over the resident
      # weight, only the indices are sent. They go up as f32 (exact below
      # 2^24) and are cast to I32 in the graph, as in ag_get_rows(); the clamp
      # is a guard against reading outside the table, the host check above
      # already refused anything out of range.
      if (vocab_size >= 2^24)
        stop("ag_embedding: vocab_size from 2^24 on is not exact in f32 indices",
             call. = FALSE)
      W_op <- .ag_operand(env$weight)
      res <- .ag_run_op(
        op_fn = function(ctx, ptrs) {
          safe <- ggml_clamp(ctx, ptrs[[2L]], 0, vocab_size - 1)
          i32  <- ggml_reshape_1d(ctx, ggml_cast(ctx, safe, GGML_TYPE_I32), n)
          ggml_get_rows(ctx, ptrs[[1L]], i32)
        },
        inputs    = list(W_op, matrix(idx_num, ncol = 1L)),
        out_shape = c(dim, n),
        dtype     = "f32",
        resident  = .ag_is_handle(W_op)
      )
      out <- .ag_wrap_result(res, "gpu", dtype = "f32")
    } else {
      # Read the current weight table through .ag_data(), which sees a value
      # written by gradcheck's .ag_data_set() as well as one living on the
      # device (inst/docs/ag_data_contract.md).
      W_data   <- .ag_data(env$weight)                    # [dim, vocab_size]
      out <- ag_tensor(W_data[, idx_int + 1L, drop = FALSE])   # [dim, n]
    }
    out$requires_grad <- env$weight$requires_grad

    if (out$requires_grad) {
      idx_snap  <- idx_int
      vocab_sz  <- vocab_size
      d         <- dim
      W_ref     <- env$weight   # reference to the env, not a copy

      grad_fn <- function(grad_out) {
        dW <- matrix(0.0, d, vocab_sz)
        for (k in seq_along(idx_snap)) {
          col       <- idx_snap[k] + 1L
          dW[, col] <- dW[, col] + grad_out[, k]
        }
        list(weight = dW)
      }
      out$grad_fn <- grad_fn
      # The graph backward emits dW as one ggml_get_rows_back (scatter-add of
      # the columns of g into a zeroed [dim, vocab] table).
      ag_record(out, grad_fn, list(weight = env$weight), op = "embedding",
                idx = idx_num, vocab = vocab_size)
    }
    out
  }

  env$parameters <- function() list(weight = env$weight)

  class(env) <- c("ag_embedding", "ag_layer")
  env
}

#' @export
ag_train.ag_embedding <- function(model) { model$training <- TRUE;  invisible(model) }
#' @export
ag_eval.ag_embedding  <- function(model) { model$training <- FALSE; invisible(model) }

# ============================================================================
# ag_multihead_attention
# ============================================================================

#' Create a Multi-Head Attention layer
#'
#' Implements scaled dot-product multi-head attention as in "Attention Is All
#' You Need" (Vaswani et al., 2017).
#'
#' Calling convention (mirrors PyTorch \code{nn.MultiheadAttention}):
#' \itemize{
#'   \item \code{layer$forward(q)} — self-attention (\code{k = v = q})
#'   \item \code{layer$forward(q, k, v)} — cross-attention
#' }
#'
#' Tensor layout: \strong{\code{[d_model, seq_len]}} — columns are tokens,
#' consistent with the rest of the ag_* API.
#'
#' Forward pass:
#' \preformatted{
#'   Q = W_q \%*\% q            [d_k * n_heads, seq_len]
#'   K = W_k \%*\% k            [d_k * n_heads, seq_len]
#'   V = W_v \%*\% v            [d_v * n_heads, seq_len]
#'
#'   for each head h:
#'     q_h = Q[h*d_k+1 : (h+1)*d_k, ]    [d_k, seq_len]
#'     k_h = K[h*d_k+1 : (h+1)*d_k, ]    [d_k, seq_len]
#'     v_h = V[h*d_v+1 : (h+1)*d_v, ]    [d_v, seq_len]
#'     A_h = softmax(t(q_h) \%*\% k_h / sqrt(d_k))   [seq_len, seq_len]
#'     if causal_mask: A_h[i,j] = 0 for j > i
#'     head_h = v_h \%*\% A_h             [d_v, seq_len]
#'
#'   concat = rbind(head_1, ..., head_H)  [d_v*n_heads, seq_len]
#'   out    = W_o \%*\% concat + b_o      [d_model, seq_len]
#' }
#'
#' @section Choosing between this and \code{ag_flash_attention()}:
#' This layer loops over heads in R. Since \code{ag_*} has no slice operation,
#' extracting one head is a matmul against a selector matrix, so a 4-head block
#' records around 52 tape nodes for the attention core -- against one for
#' \code{\link{ag_flash_attention}}, which fuses all heads into a single
#' \code{ggml_flash_attn_ext()} call and its gradient into one
#' \code{ggml_flash_attn_back()}. Measured on the attention core alone, the
#' fused op is 15-37x faster depending on shape
#' (\code{inst/scripts/measure_ag_flash_attn.R}).
#'
#' Prefer \code{ag_flash_attention()} for plain and causal multi-head
#' attention. This layer remains the fallback for what the fused op cannot do:
#'
#' \itemize{
#'   \item attention dropout (\code{dropout > 0});
#'   \item packaged projections -- it owns \code{W_q}, \code{W_k}, \code{W_v},
#'     \code{W_o} and \code{b_o}, whereas \code{ag_flash_attention()} takes
#'     already-projected Q/K/V and leaves the output projection to the caller;
#'   \item shapes or options \code{ggml_flash_attn_back()} rejects -- ALiBi,
#'     logit softcap and attention sinks all assert.
#' }
#'
#' @section Which API this belongs to:
#' This is an \code{ag_*} layer: it runs on the dynamic autograd path, where a
#' tape records each operation as it executes and \code{\link{backward}}
#' differentiates through R closures. It is not a layer for
#' \code{\link{ggml_model_sequential}} or \code{\link{ggml_model}} and cannot be
#' passed to them; the two APIs do not share layer objects. For attention inside
#' a compiled model, build the block from \code{nn_*} layers instead (see the
#' transformer encoder section of the README).
#'
#' The practical consequences of being on this path: gradients and optimizer
#' state are R doubles regardless of \code{\link{ag_dtype}}, so budget 8 bytes
#' per scalar (\code{\link{ag_estimate_training_memory}}); the tape holds every
#' activation until \code{zero_grad()} (\code{\link{ag_tape_memory}}); and
#' \code{ag_device("gpu")} is usually slower than the CPU for individual ops,
#' because each one pays a separate upload and download.
#'
#' @seealso \code{\link{ag_flash_attention}} for the fused implementation.
#'
#' @param d_model Model (embedding) dimension
#' @param n_heads Number of attention heads. \code{d_model} must be divisible
#'   by \code{n_heads}.
#' @param dropout Attention dropout probability (default 0, applied in
#'   training mode only)
#' @param bias Logical: add bias to output projection (default TRUE)
#' @return An \code{ag_multihead_attention} environment with
#'   \code{$forward(q, k, v, causal_mask)} and \code{$parameters()}
#' @export
#' @examples
#' \donttest{
#' # Self-attention
#' mha <- ag_multihead_attention(64L, 8L)
#' x   <- ag_tensor(matrix(rnorm(64 * 10), 64, 10))  # [d_model=64, seq_len=10]
#' out <- mha$forward(x)                              # [64, 10]
#'
#' # Cross-attention
#' q   <- ag_tensor(matrix(rnorm(64 * 10), 64, 10))
#' kv  <- ag_tensor(matrix(rnorm(64 * 15), 64, 15))
#' out <- mha$forward(q, kv, kv)
#'
#' # Causal (GPT-style)
#' out <- mha$forward(x, causal_mask = TRUE)
#' }
ag_multihead_attention <- function(d_model, n_heads, dropout = 0.0, bias = TRUE) {
  d_model  <- as.integer(d_model)
  n_heads  <- as.integer(n_heads)
  stopifnot(d_model %% n_heads == 0L)

  d_k <- d_model %/% n_heads   # key/query head dim
  d_v <- d_model %/% n_heads   # value head dim (equal to d_k here)
  scale <- 1.0 / sqrt(d_k)

  # Glorot uniform initialisation
  .glorot <- function(fan_in, fan_out) {
    lim <- sqrt(6.0 / (fan_in + fan_out))
    matrix(stats::runif(fan_out * fan_in, -lim, lim), fan_out, fan_in)
  }

  env <- new.env(parent = emptyenv())
  env$W_q   <- ag_param(.glorot(d_model, d_model))   # [d_model, d_model]
  env$W_k   <- ag_param(.glorot(d_model, d_model))
  env$W_v   <- ag_param(.glorot(d_model, d_model))
  env$W_o   <- ag_param(.glorot(d_model, d_model))   # [d_model, d_v*n_heads]
  env$b_o   <- if (bias) ag_param(matrix(0.0, d_model, 1L)) else NULL
  env$d_model  <- d_model
  env$n_heads  <- n_heads
  env$d_k      <- d_k
  env$d_v      <- d_v
  env$scale    <- scale
  env$dropout  <- dropout
  env$training <- TRUE
  env$name     <- sprintf("multihead_attention(d=%d, h=%d)", d_model, n_heads)

  env$forward <- function(q, k = q, v = k, causal_mask = FALSE) {
    # Project: [d_model, seq_len] -> [d_model, seq_len]
    Q <- ag_matmul(env$W_q, q)
    K <- ag_matmul(env$W_k, k)
    V <- ag_matmul(env$W_v, v)

    seq_q  <- ncol(.ag_data(Q))
    seq_kv <- ncol(.ag_data(K))

    # Collect head outputs: list of [d_v, seq_q]
    heads <- vector("list", env$n_heads)

    for (h in seq_len(env$n_heads)) {
      row_start <- (h - 1L) * env$d_k + 1L
      row_end_k <- h * env$d_k
      row_end_v <- h * env$d_v

      # Slice head: use ag_reshape trick — extract rows via index
      # ag_* has no slice op yet, so we use a thin wrapper below
      q_h <- .ag_row_slice(Q, row_start, row_end_k)   # [d_k, seq_q]
      k_h <- .ag_row_slice(K, row_start, row_end_k)   # [d_k, seq_kv]
      v_h <- .ag_row_slice(V, row_start, row_end_v)   # [d_v, seq_kv]

      # Scaled attention scores: [seq_q, seq_kv]
      scores <- ag_scale(ag_matmul(ag_transpose(q_h), k_h), env$scale)

      # Causal mask: set upper triangle (j > i) to -Inf before softmax
      if (isTRUE(causal_mask)) {
        scores <- .ag_causal_mask(scores, seq_q, seq_kv)
      }

      # Softmax over key dimension (each query attends over all keys)
      # scores is [seq_q, seq_kv]; softmax should be column-wise
      # ag_softmax applies over rows (ne0) — need softmax over rows of scores
      # i.e. for each query row, softmax over seq_kv keys
      # scores[i,j] = query i attends to key j
      # softmax over j for each i = row-wise softmax = transpose trick
      attn <- ag_transpose(ag_softmax(ag_transpose(scores)))  # [seq_q, seq_kv]

      # Optional attention dropout
      if (env$training && env$dropout > 0) {
        attn_data <- .ag_data(attn)
        mask_vals <- matrix(
          (stats::runif(length(attn_data)) > env$dropout) / (1 - env$dropout),
          nrow(attn_data), ncol(attn_data)
        )
        attn <- ag_mul(attn, ag_tensor(mask_vals))
      }

      # Weighted sum: [d_v, seq_kv] %*% [seq_kv, seq_q]^T
      # = v_h %*% t(attn) but attn is [seq_q, seq_kv]
      # head = V_h %*% attn^T  [d_v, seq_q]
      heads[[h]] <- ag_matmul(v_h, ag_transpose(attn))
    }

    # Concatenate heads row-wise: [d_v*n_heads, seq_q]
    concat <- .ag_row_concat(heads)

    # Output projection
    out <- ag_matmul(env$W_o, concat)
    if (!is.null(env$b_o)) out <- ag_add(out, env$b_o)
    out
  }

  env$parameters <- function() {
    p <- list(W_q = env$W_q, W_k = env$W_k, W_v = env$W_v, W_o = env$W_o)
    if (!is.null(env$b_o)) p[["b_o"]] <- env$b_o
    p
  }

  class(env) <- c("ag_multihead_attention", "ag_layer")
  env
}

#' @export
ag_train.ag_multihead_attention <- function(model) {
  model$training <- TRUE; invisible(model)
}
#' @export
ag_eval.ag_multihead_attention <- function(model) {
  model$training <- FALSE; invisible(model)
}

# ============================================================================
# Internal helpers for ag_multihead_attention
# ============================================================================

# Extract rows [from, to] from an ag_tensor [d, seq] -> [to-from+1, seq]
# Implemented via linear projection with a fixed identity-slice matrix.
# The slice matrix is not tracked for gradients (it's a constant).
.ag_row_slice <- function(x, from, to) {
  x_data <- .ag_data(x)
  d      <- nrow(x_data)
  n_rows <- to - from + 1L
  # Selector matrix S: [n_rows, d] — row i selects row (from+i-1) of x
  S      <- matrix(0.0, n_rows, d)
  for (i in seq_len(n_rows)) S[i, from + i - 1L] <- 1.0

  S_t <- ag_tensor(S)   # not a param, no gradient

  # Result = S %*% x   [n_rows, seq]
  # But we want gradient to flow back through x only.
  # ag_matmul(S_t, x) works since S_t$requires_grad = FALSE
  ag_matmul(S_t, x)
}

# Concatenate a list of ag_tensors row-wise: list of [d, seq] -> [d*n, seq]
.ag_row_concat <- function(tensors) {
  if (length(tensors) == 1L) return(tensors[[1L]])

  # Get data from all tensors for shape checks
  data_list <- lapply(tensors, function(t) .ag_data(t))
  total_rows <- sum(vapply(data_list, nrow, integer(1L)))
  n_cols     <- ncol(data_list[[1L]])

  # Build result by stacking: use ag_add with a zero base + place each block
  # Simplest correct approach: sum of ag_tensors projected into position
  # using fixed position matrices (no gradient on position matrices)

  d_list <- vapply(data_list, nrow, integer(1L))
  row_offsets <- c(0L, cumsum(d_list[-length(d_list)]))

  result <- NULL
  for (i in seq_along(tensors)) {
    # Expand matrix E_i: [total_rows, d_i] — places d_i rows at offset
    d_i    <- d_list[[i]]
    offset <- row_offsets[[i]]
    E_i    <- matrix(0.0, total_rows, d_i)
    for (r in seq_len(d_i)) E_i[offset + r, r] <- 1.0

    # E_i %*% tensor_i  ->  [total_rows, n_cols]
    block <- ag_matmul(ag_tensor(E_i), tensors[[i]])
    result <- if (is.null(result)) block else ag_add(result, block)
  }
  result
}

# Apply causal mask: set scores[i,j] = -Inf for j > i (future positions)
# scores: ag_tensor [seq_q, seq_kv]
#
# INVARIANT: row i keeps j <= i open, so every row has at least j = 1 open and
# no row is fully -Inf -- for any seq_q/seq_kv, cross-attention included. That
# matters downstream: ag_softmax subtracts the column max, and a fully masked
# row would make it compute exp(-Inf - -Inf) = NaN, which the backward pass
# would then spread over the whole graph. Partial masking is safe (p is exactly
# 0 there, so the softmax backward contributes 0 * (...) = 0), and the tests in
# test-ag-mha.R pin this down. A future change to the mask shape must not drop
# the open diagonal. Note that a user-supplied mask in ag_flash_attention() has
# no such guarantee -- that is a different path and a different kernel.
.ag_causal_mask <- function(scores, seq_q, seq_kv) {
  scores_data <- .ag_data(scores)
  mask <- matrix(0.0, seq_q, seq_kv)
  for (i in seq_len(seq_q)) {
    for (j in seq_len(seq_kv)) {
      if (j > i) mask[i, j] <- -Inf
    }
  }
  # Add mask as constant (no gradient)
  ag_add(scores, ag_tensor(mask))
}
