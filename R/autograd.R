# Dynamic computational graph with autograd for ggmlR
# PyTorch-style: R-level tape records operations during forward pass,
# backward() traverses tape and computes analytical gradients via closures.
#
# Hybrid approach:
#   - Training:   dynamic graph (this file) — full R control
#   - Inference:  static ggml graph (nn_build_graph / ggml_predict)
#
# Key design: ag_tensor uses environment (reference semantics) so optimizer
# updates to $data are visible to all references, just like PyTorch tensors.
#
# GPU support (Phase 1):
#   - Forward pass dispatches compute-heavy ops to ggml backend.
#   - Tensors with device="gpu" still keep $data as R matrix for backward.
#   - The ggml backend is used for the actual arithmetic (matrix multiply etc.)
#     via .ag_run_op() which runs a tiny single-node graph per operation.
#   - This allows GPU acceleration while keeping backward fully in R.
#   - .ag_data(t) always returns an R matrix regardless of device.
#
# Usage:
#   w <- ag_param(matrix(runif(4*3), 4, 3))
#   x <- ag_tensor(matrix(runif(3*8), 3, 8))
#
#   with_grad_tape({
#     h    <- ag_matmul(w, x)
#     h    <- ag_relu(h)
#     loss <- ag_mse_loss(h, y)
#   })
#
#   grads <- backward(loss)
#   optimizer$step(grads)
#   optimizer$zero_grad()

# ============================================================================
# Global tape
# ============================================================================

.ag_tape <- new.env(parent = emptyenv())
.ag_tape$enabled  <- FALSE
.ag_tape$nodes    <- list()

.ag_id_counter <- new.env(parent = emptyenv())
.ag_id_counter$n <- 0L

ag_next_id <- function() {
  .ag_id_counter$n <- .ag_id_counter$n + 1L
  .ag_id_counter$n
}

# ============================================================================
# ag_tensor class  (environment = reference semantics)
# ============================================================================

#' Create a dynamic tensor (no gradient tracking)
#'
#' ag_tensor is backed by an R environment so all references to the same
#' tensor see updates (like PyTorch tensors).
#'
#' @param data Numeric matrix or vector
#' @param device \code{"cpu"} (default) or \code{"gpu"}. When \code{"gpu"},
#'   compute operations will be dispatched to the ggml backend.
#' @param dtype Floating-point precision: \code{"f32"} (default), \code{"f16"},
#'   or \code{"bf16"}. Ignored on CPU; controls upload precision on GPU.
#' @return An ag_tensor object (environment)
#' @export
ag_tensor <- function(data, device = .ag_device_state$device,
                      dtype = .ag_device_state$dtype) {
  if (is.vector(data) && !is.list(data)) data <- matrix(data, ncol = 1L)
  e <- new.env(parent = emptyenv())
  e$id            <- ag_next_id()
  e$data          <- data       # numeric matrix — always kept for backward
  e$grad          <- NULL       # filled by backward
  e$requires_grad <- FALSE
  e$grad_fn       <- NULL
  e$device        <- device
  e$dtype         <- dtype
  class(e) <- "ag_tensor"
  e
}

#' Create a parameter tensor (gradient tracked)
#'
#' @param data Numeric matrix or vector
#' @param device \code{"cpu"} (default) or \code{"gpu"}
#' @param dtype Floating-point precision: \code{"f32"} (default), \code{"f16"},
#'   or \code{"bf16"}. Ignored on CPU; controls upload precision on GPU.
#' @return An ag_tensor with requires_grad = TRUE
#' @export
ag_param <- function(data, device = .ag_device_state$device,
                     dtype = .ag_device_state$dtype) {
  t <- ag_tensor(data, device = device, dtype = dtype)
  t$requires_grad <- TRUE
  t
}

#' Check if object is an ag_tensor
#' @keywords internal
is_ag_tensor <- function(x) inherits(x, "ag_tensor")

#' Print method for ag_tensor
#'
#' @param x An \code{ag_tensor}
#' @param ... Ignored
#' @export
print.ag_tensor <- function(x, ...) {
  d <- .ag_data(x)
  dtype_str <- if (!is.null(x$dtype) && x$dtype != "f32") paste0(" [", x$dtype, "]") else ""
  cat("ag_tensor [", paste(dim(d), collapse = "x"), "]",
      if (!is.null(x$device) && x$device == "gpu") " [gpu]" else "",
      dtype_str,
      if (x$requires_grad) " (requires_grad)" else "",
      "\n", sep = "")
  print(d)
  if (!is.null(x$grad)) {
    cat("  grad:\n")
    print(x$grad)
  }
  invisible(x)
}

# ============================================================================
# Tape recording
# ============================================================================

ag_record <- function(output, grad_fn, inputs) {
  if (!.ag_tape$enabled) return(invisible(NULL))
  any_grad <- any(vapply(inputs, function(i) isTRUE(i$requires_grad), logical(1)))
  if (!any_grad) return(invisible(NULL))
  .ag_tape$nodes <- c(.ag_tape$nodes, list(list(
    output_id = output$id,
    grad_fn   = grad_fn,
    inputs    = inputs
  )))
  invisible(NULL)
}

#' Run code with gradient tape enabled
#'
#' Records all ag_* operations inside \code{expr} for later \code{backward()}.
#' When the default device is \code{"gpu"}, the ggml context is reset at the
#' start of each tape.
#'
#' @param expr Expression to evaluate under gradient tape
#' @return Value of last expression in expr (invisibly)
#' @export
#' @examples
#' \donttest{
#' w <- ag_param(matrix(c(1, 0, 0, 1), 2, 2))
#' x <- ag_tensor(matrix(c(1, 2), 2, 1))
#' y <- ag_tensor(matrix(c(1, 2), 2, 1))
#' with_grad_tape({
#'   out  <- ag_matmul(w, x)
#'   loss <- ag_mse_loss(out, y)
#' })
#' backward(loss)
#' }
with_grad_tape <- function(expr) {
  .ag_tape$enabled <- TRUE
  .ag_tape$nodes   <- list()

  if (.ag_device_state$device != "cpu") {
    .ag_reset_ggml_ctx()
  }

  on.exit({
    .ag_tape$enabled <- FALSE
  })
  eval(substitute(expr), envir = parent.frame())
}

# ============================================================================
# Operations
# ============================================================================

#' Matrix multiplication
#'
#' Computes \code{A \%*\% B} and records the operation on the gradient tape.
#'
#' @param A ag_tensor or numeric matrix of shape \code{[m, k]}
#' @param B ag_tensor or numeric matrix of shape \code{[k, n]}
#' @return ag_tensor of shape \code{[m, n]}
#' @export
ag_matmul <- function(A, B) {
  a_data <- .ag_data(A)
  b_data <- .ag_data(B)
  device <- .ag_result_device(A, B)

  if (device == "gpu") {
    # Dispatch to ggml backend
    # ggml_mul_mat(ctx, src0, src1) = src0^T @ src1 in ggml column-major
    # For R-semantics [m,k] %*% [k,n]: pass A^T as src0 so result = A @ B
    # A: [m,k] -> A^T in ggml is [k,m] (ggml ne0=m,ne1=k -> ne0'=k,ne1'=m)
    # But ggml_mul_mat(A_ggml, B_ggml): ne0(A)==ne0(B) required
    # A_ggml has ne0=m (rows in R), B_ggml has ne0=k (rows in R) -> mismatch
    # Correct: ggml_mul_mat wants: src0[ne0,ne1] src1[ne0,ne2] -> [ne1,ne2]
    # So we need src0 to have ne0 = k (shared dim).
    # A[m,k] in R -> stored as ne0=m, ne1=k in ggml (col-major: first dim = rows)
    # For A %*% B where A[m,k], B[k,n]:
    #   We need ggml_mul_mat(B_transposed_view, A) but that gets complex.
    #   Simpler: use ggml_out_prod(A,B) = A @ B^T, or just transpose result.
    #   Easiest correct route: compute in R and wrap in ag_tensor with gpu device.
    out <- ag_tensor(.ag_gpu_matmul(a_data, b_data), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(a_data %*% b_data)
  }

  out$requires_grad <- (is_ag_tensor(A) && A$requires_grad) ||
                       (is_ag_tensor(B) && B$requires_grad)

  if (out$requires_grad) {
    a_snap <- a_data
    b_snap <- b_data
    A_ref  <- A
    B_ref  <- B
    grad_fn <- function(grad_out) {
      list(
        A = if (is_ag_tensor(A_ref) && A_ref$requires_grad) grad_out %*% t(b_snap) else NULL,
        B = if (is_ag_tensor(B_ref) && B_ref$requires_grad) t(a_snap) %*% grad_out else NULL
      )
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(A = A, B = B))
  }
  out
}

#' Element-wise addition with broadcasting
#'
#' Computes \code{A + B}. If \code{B} is \code{[m, 1]} and \code{A} is
#' \code{[m, n]}, \code{B} is broadcast across columns (useful for bias
#' vectors).
#'
#' @param A ag_tensor or numeric matrix
#' @param B ag_tensor or numeric matrix (may be \code{[m,1]} or \code{[1,n]}
#'   for broadcasting)
#' @return ag_tensor
#' @export
ag_add <- function(A, B) {
  a_data <- .ag_data(A)
  b_data <- .ag_data(B)
  device <- .ag_result_device(A, B)

  b_orig <- b_data

  # Broadcasting: if b is [m, 1] and a is [m, n], broadcast
  needs_broadcast <- !is.null(dim(b_data)) && !is.null(dim(a_data)) &&
    ((ncol(b_data) == 1L && ncol(a_data) > 1L) ||
     (nrow(b_data) == 1L && nrow(a_data) > 1L))

  if (needs_broadcast) {
    if (ncol(b_data) == 1L && ncol(a_data) > 1L) {
      b_data <- matrix(b_data[, 1L], nrow = nrow(b_data), ncol = ncol(a_data))
    } else {
      b_data <- matrix(b_data[1L, ], nrow = nrow(a_data), ncol = ncol(b_data), byrow = TRUE)
    }
  }

  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_add(a_data, b_data), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(a_data + b_data, device = device)
  }
  out$requires_grad <- (is_ag_tensor(A) && A$requires_grad) ||
                       (is_ag_tensor(B) && B$requires_grad)

  if (out$requires_grad) {
    A_ref <- A
    B_ref <- B
    grad_fn <- function(grad_out) {
      ga <- if (is_ag_tensor(A_ref) && A_ref$requires_grad) grad_out else NULL
      gb <- NULL
      if (is_ag_tensor(B_ref) && B_ref$requires_grad) {
        if (!is.null(dim(b_orig)) && ncol(b_orig) == 1L && ncol(grad_out) > 1L) {
          gb <- matrix(rowSums(grad_out), ncol = 1L)
        } else if (!is.null(dim(b_orig)) && nrow(b_orig) == 1L && nrow(grad_out) > 1L) {
          gb <- matrix(colSums(grad_out), nrow = 1L)
        } else {
          gb <- grad_out
        }
      }
      list(A = ga, B = gb)
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(A = A, B = B))
  }
  out
}

#' Element-wise subtraction
#'
#' @param A ag_tensor or numeric matrix
#' @param B ag_tensor or numeric matrix
#' @return ag_tensor
#' @export
ag_sub <- function(A, B) {
  a_data <- .ag_data(A)
  b_data <- .ag_data(B)
  device <- .ag_result_device(A, B)

  b_orig <- b_data

  # Broadcasting: expand b to match a shape for GPU/elementwise computation
  needs_broadcast <- !is.null(dim(b_data)) && !is.null(dim(a_data)) &&
    ((ncol(b_data) == 1L && ncol(a_data) > 1L) ||
     (nrow(b_data) == 1L && nrow(a_data) > 1L))

  if (needs_broadcast && device != "gpu") {
    if (ncol(b_data) == 1L && ncol(a_data) > 1L) {
      b_data <- matrix(b_data[, 1L], nrow = nrow(b_data), ncol = ncol(a_data))
    } else {
      b_data <- matrix(b_data[1L, ], nrow = nrow(a_data), ncol = ncol(b_data), byrow = TRUE)
    }
  }

  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_sub(a_data, b_orig), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(a_data - b_data, device = device)
  }
  out$requires_grad <- (is_ag_tensor(A) && A$requires_grad) ||
                       (is_ag_tensor(B) && B$requires_grad)

  if (out$requires_grad) {
    A_ref <- A; B_ref <- B
    grad_fn <- function(grad_out) {
      ga <- if (is_ag_tensor(A_ref) && A_ref$requires_grad) {
        # reduce if A was broadcast-expanded (unlikely but handle symmetrically)
        g <- grad_out
        if (!is.null(dim(a_data)) && nrow(a_data) == 1L && nrow(g) > 1L)
          g <- matrix(colSums(g), 1L)
        if (!is.null(dim(a_data)) && ncol(a_data) == 1L && ncol(g) > 1L)
          g <- matrix(rowSums(g), ncol = 1L)
        g
      } else NULL
      gb <- if (is_ag_tensor(B_ref) && B_ref$requires_grad) {
        g <- -grad_out
        # reduce along broadcast dims
        if (!is.null(dim(b_orig)) && nrow(b_orig) == 1L && nrow(g) > 1L)
          g <- matrix(colSums(g), 1L)
        if (!is.null(dim(b_orig)) && ncol(b_orig) == 1L && ncol(g) > 1L)
          g <- matrix(rowSums(g), ncol = 1L)
        g
      } else NULL
      list(A = ga, B = gb)
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(A = A, B = B))
  }
  out
}

#' Element-wise multiplication
#'
#' @param A ag_tensor or numeric matrix
#' @param B ag_tensor or numeric matrix
#' @return ag_tensor
#' @export
ag_mul <- function(A, B) {
  a_data <- .ag_data(A)
  b_data <- .ag_data(B)
  device <- .ag_result_device(A, B)

  # Save original shapes before any broadcast expansion (needed for backward reduce).
  a_orig <- a_data
  b_orig <- b_data

  # CPU broadcast: expand smaller tensor to match larger before element-wise multiply.
  # R does not broadcast matrices automatically ([d,s] * [1,s] fails or recycles wrong).
  if (device != "gpu") {
    nr_a <- nrow(a_data); nc_a <- ncol(a_data)
    nr_b <- nrow(b_data); nc_b <- ncol(b_data)
    nr   <- max(nr_a, nr_b)
    nc   <- max(nc_a, nc_b)
    if (nr_a < nr) a_data <- a_data[rep(seq_len(nr_a), length.out = nr), , drop = FALSE]
    if (nc_a < nc) a_data <- a_data[, rep(seq_len(nc_a), length.out = nc), drop = FALSE]
    if (nr_b < nr) b_data <- b_data[rep(seq_len(nr_b), length.out = nr), , drop = FALSE]
    if (nc_b < nc) b_data <- b_data[, rep(seq_len(nc_b), length.out = nc), drop = FALSE]
  }

  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_mul(a_data, b_data), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(a_data * b_data, device = device)
  }
  out$requires_grad <- (is_ag_tensor(A) && A$requires_grad) ||
                       (is_ag_tensor(B) && B$requires_grad)

  if (out$requires_grad) {
    # Snapshots of expanded data (for grad_out * other computation).
    a_snap <- a_data; b_snap <- b_data
    A_ref  <- A;     B_ref  <- B
    grad_fn <- function(grad_out) {
      nr_g <- nrow(grad_out); nc_g <- ncol(grad_out)
      # Expand snap via rep-indexing to match grad_out shape, then reduce back.
      .mul_grad <- function(snap_self_orig, snap_other) {
        other_exp <- snap_other[
          rep(seq_len(nrow(snap_other)), length.out = nr_g),
          rep(seq_len(ncol(snap_other)), length.out = nc_g),
          drop = FALSE
        ]
        g <- grad_out * other_exp
        if (nrow(snap_self_orig) == 1L && nr_g > 1L)
          g <- matrix(colSums(g), 1L, nc_g)
        if (ncol(snap_self_orig) == 1L && nc_g > 1L)
          g <- matrix(rowSums(g), nrow(g), 1L)
        g
      }
      list(
        A = if (is_ag_tensor(A_ref) && A_ref$requires_grad) .mul_grad(a_orig, b_snap) else NULL,
        B = if (is_ag_tensor(B_ref) && B_ref$requires_grad) .mul_grad(b_orig, a_snap) else NULL
      )
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(A = A, B = B))
  }
  out
}

#' Scale tensor by a scalar constant
#'
#' @param x ag_tensor
#' @param scalar Numeric scalar (not tracked for gradients)
#' @return ag_tensor
#' @export
ag_scale <- function(x, scalar) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_scale(x_data, scalar), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(x_data * scalar, device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    x_ref <- x
    grad_fn <- function(grad_out) list(x = grad_out * scalar)
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' ReLU activation
#'
#' Applies the rectified linear unit: \eqn{\max(0, x)}.
#'
#' @param x ag_tensor
#' @return ag_tensor
#' @export
ag_relu <- function(x) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_relu(x_data), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(pmax(x_data, 0), device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad

  if (out$requires_grad) {
    mask  <- (x_data > 0) * 1.0
    x_ref <- x
    grad_fn <- function(grad_out) list(x = grad_out * mask)
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Sigmoid activation
#'
#' Applies \eqn{1 / (1 + e^{-x})}.
#'
#' @param x ag_tensor
#' @return ag_tensor
#' @export
ag_sigmoid <- function(x) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    s <- .ag_gpu_sigmoid(x_data)
  } else {
    s <- 1.0 / (1.0 + exp(-x_data))
  }
  out <- ag_tensor(s, device = device, dtype = .ag_device_state$dtype)
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad

  if (out$requires_grad) {
    s_snap <- s
    grad_fn <- function(grad_out) list(x = grad_out * s_snap * (1.0 - s_snap))
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Tanh activation
#'
#' @param x ag_tensor
#' @return ag_tensor
#' @export
ag_tanh <- function(x) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    t_val <- .ag_gpu_tanh(x_data)
  } else {
    t_val <- tanh(x_data)
  }
  out <- ag_tensor(t_val, device = device, dtype = .ag_device_state$dtype)
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad

  if (out$requires_grad) {
    t_snap <- t_val
    grad_fn <- function(grad_out) list(x = grad_out * (1.0 - t_snap^2))
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Softmax activation (column-wise)
#'
#' Applies numerically stable softmax along rows so that each column (one
#' sample) sums to 1.
#'
#' @param x ag_tensor of shape \code{[classes, batch_size]}
#' @return ag_tensor of the same shape as \code{x}
#' @export
ag_softmax <- function(x) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    p <- .ag_gpu_softmax(x_data)
  } else {
    # Numerically stable softmax (column-wise)
    mx <- apply(x_data, 2, max)
    mx <- matrix(mx, nrow = nrow(x_data), ncol = ncol(x_data), byrow = TRUE)
    e  <- exp(x_data - mx)
    s  <- matrix(colSums(e), nrow = nrow(x_data), ncol = ncol(x_data), byrow = TRUE)
    p  <- e / s
  }

  out <- ag_tensor(p, device = device, dtype = .ag_device_state$dtype)
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad

  if (out$requires_grad) {
    p_snap <- p
    grad_fn <- function(grad_out) {
      dot   <- colSums(p_snap * grad_out)
      dot_m <- matrix(dot, nrow = nrow(p_snap), ncol = ncol(p_snap), byrow = TRUE)
      list(x = p_snap * (grad_out - dot_m))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

# ============================================================================
# Loss functions
# ============================================================================

#' Mean Squared Error loss
#'
#' @param pred ag_tensor [units, batch_size]
#' @param target ag_tensor or matrix [units, batch_size]
#' @return scalar ag_tensor
#' @export
ag_mse_loss <- function(pred, target) {
  p_data <- .ag_data(pred)
  t_data <- .ag_data(target)
  device <- if (is_ag_tensor(pred)) pred$device else "cpu"

  diff     <- p_data - t_data
  n        <- length(diff)
  out      <- ag_tensor(matrix(sum(diff^2) / n), device = device)
  out$requires_grad <- is_ag_tensor(pred) && pred$requires_grad

  if (out$requires_grad) {
    diff_snap <- diff
    pred_ref  <- pred
    grad_fn <- function(grad_out) {
      list(pred = (2.0 / n) * diff_snap * as.numeric(grad_out))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(pred = pred))
  }
  out
}

#' Categorical Cross-Entropy loss
#'
#' Generic CE: \code{-sum(target * log(pred)) / batch_size}.
#' The gradient w.r.t. \code{pred} is \code{-target / pred / n}.
#' Use \code{ag_softmax_cross_entropy_loss()} for the numerically stable
#' combined softmax + CE (fused gradient \code{(p - y) / n}).
#'
#' @param pred ag_tensor [classes, batch_size]  probabilities (any, not just softmax)
#' @param target matrix [classes, batch_size]   one-hot (or soft) labels
#' @return scalar ag_tensor
#' @export
ag_cross_entropy_loss <- function(pred, target) {
  p_data <- .ag_data(pred)
  t_data <- .ag_data(target)
  device <- if (is_ag_tensor(pred)) pred$device else "cpu"

  eps     <- 1e-7
  p_clamp <- pmax(pmin(p_data, 1 - eps), eps)
  n       <- ncol(p_data)
  out     <- ag_tensor(matrix(-sum(t_data * log(p_clamp)) / n), device = device)
  out$requires_grad <- is_ag_tensor(pred) && pred$requires_grad

  if (out$requires_grad) {
    p_snap <- p_clamp
    t_snap <- t_data
    grad_fn <- function(grad_out) {
      list(pred = (-t_snap / p_snap) / n * as.numeric(grad_out))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(pred = pred))
  }
  out
}

#' Fused softmax + cross-entropy loss (numerically stable)
#'
#' Combines softmax and CE in one op using the fused gradient \code{(p - y) / n}.
#' More numerically stable than chaining \code{ag_softmax} + \code{ag_cross_entropy_loss}.
#' Use this when your last layer outputs raw logits.
#'
#' @param logits ag_tensor [classes, batch_size]  raw (pre-softmax) scores
#' @param target matrix [classes, batch_size]     one-hot labels
#' @return scalar ag_tensor
#' @export
ag_softmax_cross_entropy_loss <- function(logits, target) {
  l_data <- .ag_data(logits)
  t_data <- .ag_data(target)
  device <- if (is_ag_tensor(logits)) logits$device else "cpu"

  # Auto-convert 0-based integer indices to one-hot matrix [classes, seq_len].
  # Accepts: integer vector, numeric vector without dim, or integer matrix [1, seq_len].
  if (is.integer(t_data) ||
      (is.numeric(t_data) && is.null(dim(t_data))) ||
      (!is.null(dim(t_data)) && nrow(t_data) == 1L && nrow(l_data) > 1L)) {
    idx    <- as.integer(t_data)          # 0-based indices, length = seq_len
    n_cls  <- nrow(l_data)
    n_seq  <- ncol(l_data)
    oh     <- matrix(0.0, n_cls, n_seq)
    for (i in seq_along(idx)) oh[idx[i] + 1L, i] <- 1.0
    t_data <- oh
  }

  # Numerically stable softmax
  mx     <- apply(l_data, 2, max)
  mx_m   <- matrix(mx, nrow(l_data), ncol(l_data), byrow = TRUE)
  e      <- exp(l_data - mx_m)
  s      <- matrix(colSums(e), nrow(l_data), ncol(l_data), byrow = TRUE)
  p      <- e / s

  eps    <- 1e-7
  p_c    <- pmax(p, eps)
  n      <- ncol(l_data)
  out    <- ag_tensor(matrix(-sum(t_data * log(p_c)) / n), device = device)
  out$requires_grad <- is_ag_tensor(logits) && logits$requires_grad

  if (out$requires_grad) {
    p_snap <- p
    t_snap <- t_data
    grad_fn <- function(grad_out) {
      list(logits = (p_snap - t_snap) / n * as.numeric(grad_out))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(logits = logits))
  }
  out
}

# ============================================================================
# Backward pass
# ============================================================================

#' Run backward pass from a scalar loss tensor
#'
#' Traverses the gradient tape in reverse and accumulates gradients into
#' \code{tensor$grad} for all leaf tensors with \code{requires_grad = TRUE}.
#'
#' @param loss Scalar ag_tensor
#' @return Named environment: tensor id -> gradient matrix (for use by optimizer$step)
#' @export
#' @examples
#' \donttest{
#' w <- ag_param(matrix(runif(4), 2, 2))
#' x <- ag_tensor(matrix(c(1, 2), 2, 1))
#' y <- ag_tensor(matrix(c(0, 1), 2, 1))
#' with_grad_tape({
#'   out  <- ag_matmul(w, x)
#'   loss <- ag_mse_loss(out, y)
#' })
#' grads <- backward(loss)
#' }
backward <- function(loss) {
  if (!is_ag_tensor(loss)) stop("backward() requires an ag_tensor")

  grads <- new.env(hash = TRUE, parent = emptyenv())
  assign(as.character(loss$id), matrix(1.0), envir = grads)

  nodes <- rev(.ag_tape$nodes)

  for (node in nodes) {
    grad_out <- get0(as.character(node$output_id), envir = grads)
    if (is.null(grad_out)) next

    input_grads <- node$grad_fn(grad_out)

    for (nm in names(node$inputs)) {
      inp <- node$inputs[[nm]]
      if (!is_ag_tensor(inp) || !isTRUE(inp$requires_grad)) next
      ig  <- input_grads[[nm]]
      if (is.null(ig)) next

      key      <- as.character(inp$id)
      existing <- get0(key, envir = grads)
      if (is.null(existing)) {
        assign(key, ig, envir = grads)
      } else {
        assign(key, existing + ig, envir = grads)
      }
    }
  }

  # Write gradients back to leaf tensor $grad fields
  for (node in .ag_tape$nodes) {
    for (inp in node$inputs) {
      if (!is_ag_tensor(inp) || !isTRUE(inp$requires_grad)) next
      key <- as.character(inp$id)
      g   <- get0(key, envir = grads)
      if (!is.null(g)) {
        inp$grad <- if (is.null(inp$grad)) g else inp$grad + g
      }
    }
  }

  invisible(grads)
}

# ============================================================================
# Optimizers
# ============================================================================

#' Create an SGD optimizer
#'
#' @param params Named list of ag_param tensors
#' @param lr Learning rate (default 0.01)
#' @param momentum Momentum factor (default 0)
#' @return An optimizer environment
#' @export
#' @examples
#' \donttest{
#' w <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_sgd(list(w = w), lr = 0.01)
#' }
optimizer_sgd <- function(params, lr = 0.01, momentum = 0.0) {
  stopifnot(is.list(params))
  env <- new.env(parent = emptyenv())
  env$params   <- params
  env$lr       <- lr
  env$momentum <- momentum
  env$velocity <- lapply(params, function(p) {
    d <- .ag_data(p)
    matrix(0.0, nrow(d), ncol(d))
  })

  # step: update param $data in-place (reference semantics via environment)
  env$step <- function(grads) {
    for (nm in names(env$params)) {
      p   <- env$params[[nm]]
      key <- as.character(p$id)
      g   <- get0(key, envir = grads)
      if (is.null(g)) next
      if (env$momentum > 0) {
        env$velocity[[nm]] <- env$momentum * env$velocity[[nm]] + g
        p$data <- p$data - env$lr * env$velocity[[nm]]
      } else {
        p$data <- p$data - env$lr * g
      }
    }
  }

  env$zero_grad <- function() {
    for (nm in names(env$params)) {
      env$params[[nm]]$grad <- NULL
    }
    .ag_tape$nodes <- list()
  }

  class(env) <- "ag_optimizer_sgd"
  env
}

#' Create an Adam optimizer
#'
#' @param params Named list of ag_param tensors
#' @param lr Learning rate (default 1e-3)
#' @param beta1 First moment decay (default 0.9)
#' @param beta2 Second moment decay (default 0.999)
#' @param eps Stability constant (default 1e-8)
#' @return An optimizer environment
#' @export
#' @examples
#' \donttest{
#' w <- ag_param(matrix(runif(4), 2, 2))
#' opt <- optimizer_adam(list(w = w), lr = 1e-3)
#' }
optimizer_adam <- function(params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  stopifnot(is.list(params))
  env <- new.env(parent = emptyenv())
  env$params <- params
  env$lr     <- lr
  env$beta1  <- beta1
  env$beta2  <- beta2
  env$eps    <- eps
  env$t      <- 0L
  env$m      <- lapply(params, function(p) { d <- .ag_data(p); matrix(0.0, nrow(d), ncol(d)) })
  env$v      <- lapply(params, function(p) { d <- .ag_data(p); matrix(0.0, nrow(d), ncol(d)) })

  env$step <- function(grads) {
    env$t <- env$t + 1L
    for (nm in names(env$params)) {
      p   <- env$params[[nm]]
      key <- as.character(p$id)
      g   <- get0(key, envir = grads)
      if (is.null(g)) next

      env$m[[nm]] <- env$beta1 * env$m[[nm]] + (1 - env$beta1) * g
      env$v[[nm]] <- env$beta2 * env$v[[nm]] + (1 - env$beta2) * g^2

      m_hat <- env$m[[nm]] / (1 - env$beta1^env$t)
      v_hat <- env$v[[nm]] / (1 - env$beta2^env$t)

      p$data <- p$data - env$lr * m_hat / (sqrt(v_hat) + env$eps)
    }
  }

  env$zero_grad <- function() {
    for (nm in names(env$params)) {
      env$params[[nm]]$grad <- NULL
    }
    .ag_tape$nodes <- list()
  }

  class(env) <- "ag_optimizer_adam"
  env
}

#' @export
print.ag_optimizer_sgd <- function(x, ...) {
  cat("SGD optimizer | lr =", x$lr, "| momentum =", x$momentum,
      "| params:", length(x$params), "\n")
  invisible(x)
}

#' @export
print.ag_optimizer_adam <- function(x, ...) {
  cat("Adam optimizer | lr =", x$lr,
      "| beta1 =", x$beta1, "| beta2 =", x$beta2,
      "| step =", x$t,
      "| params:", length(x$params), "\n")
  invisible(x)
}

# ============================================================================
# Convenience: dense layer with parameter management
# ============================================================================

#' Create a dense layer with learnable parameters
#'
#' Returns a closure-based layer.  Because ag_param uses environment semantics,
#' the optimizer updates W and b in-place, and forward() always uses the latest
#' weights.
#'
#' @param in_features Input dimension
#' @param out_features Output dimension
#' @param activation "relu", "sigmoid", "tanh", "softmax", or NULL
#' @return List with \code{W}, \code{b}, \code{forward(x)}, \code{params()}
#' @export
#' @examples
#' \donttest{
#' layer <- ag_linear(4L, 8L, activation = "relu")
#' x     <- ag_tensor(matrix(runif(4 * 16), 4, 16))
#' out   <- layer$forward(x)
#' }
ag_linear <- function(in_features, out_features, activation = NULL) {
  limit <- sqrt(6.0 / (in_features + out_features))
  W <- ag_param(matrix(runif(out_features * in_features, -limit, limit),
                        out_features, in_features))
  b <- ag_param(matrix(0.0, out_features, 1L))

  forward <- function(x) {
    out <- ag_add(ag_matmul(W, x), b)
    act <- if (is.null(activation)) "none" else activation
    switch(act,
      "relu"    = ag_relu(out),
      "sigmoid" = ag_sigmoid(out),
      "tanh"    = ag_tanh(out),
      "softmax" = ag_softmax(out),
      out
    )
  }

  list(W = W, b = b, forward = forward,
       params = function() list(W = W, b = b))
}

# Null-coalescing helper (internal)
`%||%` <- function(a, b) if (!is.null(a)) a else b

# ============================================================================
# Missing ops: ag_sum, ag_mean, ag_log, ag_exp,
#              ag_reshape, ag_transpose, ag_clamp, ag_pow
# ============================================================================

#' Sum all elements (or along a dim): out = sum(x)
#'
#' @param x ag_tensor
#' @param dim NULL (all), 1 (row-wise), or 2 (col-wise)
#' @param keepdim Logical: keep size-1 dimensions
#' @return scalar (or reduced) ag_tensor
#' @export
ag_sum <- function(x, dim = NULL, keepdim = FALSE) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    if (is.null(dim)) {
      out_data <- .ag_gpu_sum_all(x_data)
    } else if (dim == 1L) {
      out_data <- .ag_gpu_sum_rows(x_data)   # [nrow,1] — keepdim=TRUE same shape
    } else {
      out_data <- .ag_gpu_sum_cols(x_data)   # [1,ncol] — keepdim=TRUE same shape
    }
  } else if (is.null(dim)) {
    out_data <- matrix(sum(x_data))
  } else if (dim == 1L) {
    # dim=1: reduce rows → [nrow,1]
    out_data <- matrix(rowSums(x_data), nrow(x_data), 1L)
  } else {
    # dim=2: reduce cols → [1,ncol]; keepdim keeps shape [1,ncol]
    out_data <- matrix(colSums(x_data), 1L, ncol(x_data))
  }
  out <- ag_tensor(out_data, device = device, dtype = .ag_device_state$dtype)
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    orig_shape <- dim(x_data)
    dim_arg    <- dim
    grad_fn <- function(grad_out) {
      if (is.null(dim_arg)) {
        list(x = matrix(as.numeric(grad_out), orig_shape[1L], orig_shape[2L]))
      } else if (dim_arg == 1L) {
        # grad_out [nrow,1] → broadcast to [nrow,ncol]
        g <- matrix(grad_out, orig_shape[1L], 1L)
        list(x = matrix(g, orig_shape[1L], orig_shape[2L]))
      } else {
        # grad_out [1,ncol] → broadcast to [nrow,ncol]
        g <- matrix(grad_out, 1L, orig_shape[2L])
        list(x = matrix(rep(g, each = orig_shape[1L]), orig_shape[1L], orig_shape[2L]))
      }
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Mean of elements (or along a dim)
#'
#' @param x ag_tensor
#' @param dim NULL (all), 1 (row-wise), or 2 (col-wise)
#' @param keepdim Logical
#' @return ag_tensor
#' @export
ag_mean <- function(x, dim = NULL, keepdim = FALSE) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  n_all  <- length(x_data)
  if (device == "gpu") {
    if (is.null(dim)) {
      out_data <- .ag_gpu_mean_all(x_data)
      n_div    <- n_all
    } else if (dim == 1L) {
      out_data <- .ag_gpu_mean_rows(x_data)   # [nrow,1]
      n_div    <- ncol(x_data)
    } else {
      out_data <- .ag_gpu_mean_cols(x_data)   # [1,ncol]
      n_div    <- nrow(x_data)
    }
  } else if (is.null(dim)) {
    out_data <- matrix(mean(x_data))
    n_div    <- n_all
  } else if (dim == 1L) {
    # dim=1: reduce rows → [nrow,1]
    out_data <- matrix(rowMeans(x_data), nrow(x_data), 1L)
    n_div    <- ncol(x_data)
  } else {
    # dim=2: reduce cols → [1,ncol]
    out_data <- matrix(colMeans(x_data), 1L, ncol(x_data))
    n_div    <- nrow(x_data)
  }
  out <- ag_tensor(out_data, device = device)
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    orig_shape <- dim(x_data)
    dim_arg    <- dim
    grad_fn <- function(grad_out) {
      if (is.null(dim_arg)) {
        list(x = matrix(as.numeric(grad_out) / n_div, orig_shape[1L], orig_shape[2L]))
      } else if (dim_arg == 1L) {
        # grad_out [nrow,1] → broadcast to [nrow,ncol]
        g <- matrix(grad_out, orig_shape[1L], 1L)
        list(x = matrix(g / n_div, orig_shape[1L], orig_shape[2L]))
      } else {
        # grad_out [1,ncol] → broadcast to [nrow,ncol]
        g <- matrix(grad_out, 1L, orig_shape[2L])
        list(x = matrix(rep(g, each = orig_shape[1L]) / n_div,
                        orig_shape[1L], orig_shape[2L]))
      }
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Element-wise natural logarithm
#'
#' @param x ag_tensor
#' @return ag_tensor
#' @export
ag_log <- function(x) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_log(x_data), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(log(x_data), device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    x_snap <- x_data
    grad_fn <- function(grad_out) list(x = grad_out / x_snap)
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Element-wise exponential
#'
#' @param x ag_tensor
#' @return ag_tensor
#' @export
ag_exp <- function(x) {
  x_data  <- .ag_data(x)
  device  <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    e_val <- .ag_gpu_exp(x_data)
  } else {
    e_val <- exp(x_data)
  }
  out <- ag_tensor(e_val, device = device, dtype = .ag_device_state$dtype)
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    e_snap <- e_val
    grad_fn <- function(grad_out) list(x = grad_out * e_snap)
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Reshape tensor
#'
#' @param x ag_tensor
#' @param nrow New number of rows (use -1 to infer)
#' @param ncol New number of columns (use -1 to infer)
#' @return ag_tensor with new shape, same data
#' @export
ag_reshape <- function(x, nrow, ncol) {
  x_data     <- .ag_data(x)
  device     <- if (is_ag_tensor(x)) x$device else "cpu"
  orig_shape <- dim(x_data)
  n          <- length(x_data)
  nrow       <- if (nrow == -1L) n %/% ncol else as.integer(nrow)
  ncol       <- if (ncol == -1L) n %/% nrow else as.integer(ncol)
  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_reshape(x_data, nrow, ncol), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(matrix(x_data, nrow, ncol), device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    grad_fn <- function(grad_out) {
      list(x = matrix(grad_out, orig_shape[1L], orig_shape[2L]))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Transpose a tensor
#'
#' @param x ag_tensor
#' @return ag_tensor with rows and columns swapped
#' @export
ag_transpose <- function(x) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_transpose(x_data), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(t(x_data), device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    grad_fn <- function(grad_out) list(x = t(grad_out))
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Element-wise clamp
#'
#' Clamps values to \code{[lo, hi]}.  Gradient is 1 inside the interval,
#' 0 at the boundary (straight-through estimator).
#'
#' @param x ag_tensor
#' @param lo Lower bound (default \code{-Inf})
#' @param hi Upper bound (default \code{Inf})
#' @return ag_tensor
#' @export
ag_clamp <- function(x, lo = -Inf, hi = Inf) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  lo_fin <- if (is.finite(lo)) lo else -3.402823e+38
  hi_fin <- if (is.finite(hi)) hi else  3.402823e+38
  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_clamp(x_data, lo_fin, hi_fin), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(pmin(pmax(x_data, lo), hi), device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    mask <- (x_data > lo & x_data < hi) * 1.0
    grad_fn <- function(grad_out) list(x = grad_out * mask)
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

#' Element-wise power
#'
#' @param x ag_tensor
#' @param p Numeric exponent (scalar, not tracked for gradients)
#' @return ag_tensor
#' @export
ag_pow <- function(x, p) {
  x_data <- .ag_data(x)
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    out <- ag_tensor(.ag_gpu_pow(x_data, p), device = "gpu", dtype = .ag_device_state$dtype)
  } else {
    out <- ag_tensor(x_data ^ p, device = device)
  }
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    x_snap <- x_data
    grad_fn <- function(grad_out) list(x = grad_out * p * x_snap ^ (p - 1))
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x))
  }
  out
}

# ============================================================================
# Device utility helpers
# ============================================================================

# Return "gpu" if any input tensor is on GPU, else "cpu".
.ag_result_device <- function(...) {
  args <- list(...)
  for (a in args) {
    if (is_ag_tensor(a) && isTRUE(a$device == "gpu")) return("gpu")
  }
  "cpu"
}

# Return dtype of result: prefer non-f32 dtype from inputs, else global default.
.ag_result_dtype <- function(...) {
  args <- list(...)
  for (a in args) {
    if (is_ag_tensor(a) && !is.null(a$dtype) && a$dtype != "f32") return(a$dtype)
  }
  if (length(args) > 0 && is_ag_tensor(args[[1L]]) && !is.null(args[[1L]]$dtype))
    return(args[[1L]]$dtype)
  .ag_device_state$dtype
}

# ============================================================================
# Gradient checker
# ============================================================================

#' Numerical gradient check (like torch.autograd.gradcheck)
#'
#' Compares analytical gradients (from \code{backward()}) with finite-difference
#' numerical gradients for all input tensors with \code{requires_grad = TRUE}.
#'
#' @param fn A function that takes a list of ag_tensor inputs and returns a
#'   scalar ag_tensor loss (must be used inside \code{with_grad_tape}).
#' @param inputs Named list of ag_tensor objects.  Only those with
#'   \code{requires_grad = TRUE} are checked.
#' @param eps Finite-difference step size (default 1e-5).
#' @param atol Absolute tolerance for pass/fail (default 1e-4).
#' @param verbose Print per-element comparison (default FALSE).
#' @param quiet Suppress per-parameter and overall status lines (default FALSE).
#'   Useful when calling from \code{testthat} tests to keep output clean.
#' @return Invisibly \code{TRUE} if all gradients match, \code{FALSE} otherwise.
#'   When \code{quiet = FALSE} (default), prints a summary report.
#' @export
#' @examples
#' \donttest{
#' W <- ag_param(matrix(runif(6), 2, 3))
#' x <- ag_tensor(matrix(runif(3), 3, 1))
#' ag_gradcheck(
#'   fn = function(ins) ag_mse_loss(ag_relu(ag_matmul(ins$W, ins$x)),
#'                                   matrix(0, 2, 1)),
#'   inputs = list(W = W, x = x)
#' )
#' }
ag_gradcheck <- function(fn, inputs, eps = 1e-5, atol = 1e-4, verbose = FALSE,
                          quiet = FALSE) {
  # ---- analytical gradients ----
  with_grad_tape({
    loss <- fn(inputs)
  })
  anal_grads_env <- backward(loss)

  all_ok <- TRUE
  results <- list()

  for (nm in names(inputs)) {
    inp <- inputs[[nm]]
    if (!is_ag_tensor(inp) || !isTRUE(inp$requires_grad)) next

    anal_g <- get0(as.character(inp$id), envir = anal_grads_env)
    if (is.null(anal_g)) {
      if (!quiet) cat(sprintf("[gradcheck] '%s': no analytical gradient found\n", nm))
      all_ok <- FALSE
      next
    }

    # ---- numerical gradients (central differences) ----
    x_flat     <- as.numeric(inp$data)
    num_g      <- numeric(length(x_flat))
    inp_shape  <- dim(inp$data)

    for (k in seq_along(x_flat)) {
      # +eps
      x_flat[k] <- x_flat[k] + eps
      inp$data   <- matrix(x_flat, inp_shape[1L], inp_shape[2L])
      with_grad_tape({ lp <- fn(inputs) })
      f_plus <- as.numeric(.ag_data(lp))

      # -eps
      x_flat[k] <- x_flat[k] - 2 * eps
      inp$data   <- matrix(x_flat, inp_shape[1L], inp_shape[2L])
      with_grad_tape({ lm <- fn(inputs) })
      f_minus <- as.numeric(.ag_data(lm))

      num_g[k]   <- (f_plus - f_minus) / (2 * eps)
      x_flat[k]  <- x_flat[k] + eps  # restore
    }
    inp$data <- matrix(x_flat, inp_shape[1L], inp_shape[2L])

    num_g_mat  <- matrix(num_g, nrow(anal_g), ncol(anal_g))
    max_err    <- max(abs(anal_g - num_g_mat))
    pass       <- max_err < atol

    if (!pass) all_ok <- FALSE

    if (!quiet) {
      cat(sprintf("[gradcheck] '%s': max_err = %.2e  %s\n",
                  nm, max_err, if (pass) "PASS" else "FAIL"))
    }

    if (verbose) {
      cat("  analytical:\n"); print(round(anal_g, 6))
      cat("  numerical:\n");  print(round(num_g_mat, 6))
    }

    results[[nm]] <- list(analytical = anal_g, numerical = num_g_mat,
                          max_err = max_err, pass = pass)
  }

  if (!quiet) cat(sprintf("[gradcheck] Overall: %s\n", if (all_ok) "PASS" else "FAIL"))
  invisible(all_ok)
}
