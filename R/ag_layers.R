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
    x_data <- if (is_ag_tensor(x)) x$data else x
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
# Internal: broadcast-mul [F,1] x [F,N] -> [F,N]
# Used by batch_norm for gamma/beta scaling.
# ============================================================================

.ag_mul_broadcast_col <- function(scalar_col, mat) {
  # scalar_col: ag_param [F, 1]; mat: ag_tensor [F, N]
  # Expands scalar_col to [F, N] then calls ag_mul
  s_data <- if (is_ag_tensor(scalar_col)) scalar_col$data else scalar_col
  m_data <- if (is_ag_tensor(mat))        mat$data        else mat
  n      <- ncol(m_data)
  # broadcast: replicate column n times
  s_exp  <- matrix(as.numeric(s_data), nrow(s_data), n)
  s_t    <- ag_tensor(s_exp)
  s_t$requires_grad <- is_ag_tensor(scalar_col) && scalar_col$requires_grad

  if (s_t$requires_grad) {
    s_orig <- scalar_col
    grad_fn <- function(grad_out) {
      list(scalar_col = matrix(rowSums(grad_out), nrow(grad_out), 1L))
    }
    s_t$grad_fn <- grad_fn
    ag_record(s_t, grad_fn, list(scalar_col = scalar_col))
  }
  ag_mul(s_t, mat)
}

# Same for bias (beta): add [F,1] broadcast to [F,N]
.ag_add_broadcast_col <- function(scalar_col, mat) {
  s_data <- if (is_ag_tensor(scalar_col)) scalar_col$data else scalar_col
  m_data <- if (is_ag_tensor(mat))        mat$data        else mat
  n      <- ncol(m_data)
  s_exp  <- matrix(as.numeric(s_data), nrow(s_data), n)
  s_t    <- ag_tensor(s_exp)
  s_t$requires_grad <- is_ag_tensor(scalar_col) && scalar_col$requires_grad

  if (s_t$requires_grad) {
    s_orig <- scalar_col
    grad_fn <- function(grad_out) {
      list(scalar_col = matrix(rowSums(grad_out), nrow(grad_out), 1L))
    }
    s_t$grad_fn <- grad_fn
    ag_record(s_t, grad_fn, list(scalar_col = scalar_col))
  }
  ag_add(s_t, mat)
}

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
  env$running_mean <- matrix(0.0, num_features, 1L)
  env$running_var  <- matrix(1.0, num_features, 1L)
  env$num_features <- num_features
  env$eps          <- eps
  env$momentum     <- momentum
  env$training     <- TRUE
  env$name         <- paste0("batch_norm(", num_features, ")")

  env$forward <- function(x) {
    x_data <- if (is_ag_tensor(x)) x$data else x
    n      <- ncol(x_data)

    if (env$training) {
      mu  <- rowMeans(x_data)
      var <- rowMeans((x_data - mu)^2)
      # update running stats (no gradient through these assignments)
      env$running_mean <- (1 - env$momentum) * env$running_mean +
                           env$momentum * matrix(mu,  env$num_features, 1L)
      env$running_var  <- (1 - env$momentum) * env$running_var  +
                           env$momentum * matrix(var, env$num_features, 1L)
    } else {
      mu  <- as.numeric(env$running_mean)
      var <- as.numeric(env$running_var)
    }

    std   <- sqrt(var + env$eps)           # [F] numeric vector

    # Normalise: (x - mu) / std  — pure numeric, no grad needed here
    mu_m  <- matrix(mu,  num_features, n)
    std_m <- matrix(std, num_features, n)

    # x_hat as ag_tensor that propagates gradient from x
    x_hat <- ag_tensor((x_data - mu_m) / std_m)
    x_hat$requires_grad <- is_ag_tensor(x) && x$requires_grad

    if (x_hat$requires_grad) {
      std_snap <- std_m
      x_ref    <- x
      grad_fn  <- function(grad_out) list(x = grad_out / std_snap)
      x_hat$grad_fn <- grad_fn
      ag_record(x_hat, grad_fn, list(x = x))
    }

    # gamma * x_hat + beta  with column broadcast [F,1] -> [F,N]
    scaled <- .ag_mul_broadcast_col(env$gamma, x_hat)
    .ag_add_broadcast_col(env$beta, scaled)
  }

  env$parameters <- function() list(gamma = env$gamma, beta = env$beta)

  class(env) <- c("ag_batch_norm", "ag_layer")
  env
}

#' @export
ag_train.ag_batch_norm <- function(model) { model$training <- TRUE;  invisible(model) }
#' @export
ag_eval.ag_batch_norm  <- function(model) { model$training <- FALSE; invisible(model) }

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
    idx_int <- as.integer(if (is_ag_tensor(idx)) idx$data else idx)
    n       <- length(idx_int)

    # read current weight data (via $data, works even when gradcheck replaces it)
    W_data  <- env$weight$data           # [dim, vocab_size]
    out_data <- W_data[, idx_int + 1L, drop = FALSE]   # [dim, n]

    out <- ag_tensor(out_data)
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
      ag_record(out, grad_fn, list(weight = env$weight))
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
