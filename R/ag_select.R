# Selection ops for ag_* tensors laid out [classes, batch]: every op works down
# the columns (one column = one sample), like ag_softmax. Indices are 0-based,
# as everywhere else in the package.

#' Column-wise argmax
#'
#' Index of the largest entry in every column of \code{x}, for action selection
#' over a \code{[n_actions, batch]} matrix of logits or log-probabilities.
#'
#' Ties go to the FIRST index on every path (R, CPU kernel, Vulkan shader),
#' matching \code{which.max}. \code{NaN} is never selected; a column with nothing
#' above \code{-Inf} (all \code{NaN} / all \code{-Inf}) gives \code{0}.
#'
#' Not differentiable. On the GPU this reads the result back to the host, so it
#' is a synchronisation point: fine where the environment needs the action, but
#' keep it out of the training step.
#'
#' @param x ag_tensor or numeric matrix, \code{[n, batch]}.
#' @return Integer vector of length \code{ncol(x)}, 0-based.
#' @export
#' @examples
#' ag_argmax(matrix(c(1, 3, 3,  5, 2, 5), 3, 2))   # 1 0
ag_argmax <- function(x) {
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    x_data <- .ag_operand(x)
    idx <- .ag_run_op(
      op_fn = function(ctx, ptrs) {
        a <- ptrs[[1L]]
        # ggml_argmax is F32-only (CPU asserts, Vulkan has one f32 pipeline);
        # an f16 / bf16 compute dtype uploads x in that type.
        if (ggml_tensor_type(a) != GGML_TYPE_F32) a <- ggml_cast(ctx, a, GGML_TYPE_F32)
        ggml_argmax(ctx, a)
      },
      inputs    = list(x_data),
      out_shape = c(1L, .ag_ncol(x_data))
    )
    return(as.integer(idx))
  }
  m <- .ag_as_matrix(.ag_data(x))
  if (is.null(dim(m))) m <- matrix(m, ncol = 1L)
  m[is.na(m)] <- -Inf                      # NaN and NA: never selected
  # max.col works on rows; ties.method = "first" is the documented rule.
  max.col(t(m), ties.method = "first") - 1L
}

#' Column-wise log-softmax
#'
#' \code{log(softmax(x))} down every column, computed without forming
#' \code{softmax} first: masked entries keep finite log-probabilities instead of
#' \code{log(0) = -Inf}, so \code{p * logp} stays finite (entropy terms).
#'
#' Mask illegal entries by ADDING a large but FINITE negative number (e.g.
#' \code{-1e4}, which also fits f16). \code{-Inf} is not supported: the shift
#' below multiplies x by p, and \code{0 * -Inf} is \code{NaN}. A column with
#' every entry masked is finite but meaningless (uniform over masked actions);
#' callers must leave at least one entry unmasked.
#'
#' @details Stability shift. Any per-column c with \code{max - log(n) <= c <= max}
#'   keeps \code{exp(x - c)} from overflowing and the largest term from
#'   underflowing. \code{c = colSums(x * softmax(x))} satisfies it
#'   (\code{lse - c} is the entropy, at most \code{log(n)}, and \code{lse >= max}),
#'   and on the device it is three ggml ops -- no max reduction, which ggml lacks
#'   along this axis. The result does not depend on c, and the gradient treats it
#'   as a constant: \code{dx = g - exp(logp) * colSums(g)}.
#'
#'   f16 compute dtype: \code{colSums(exp(x - c))} is at most n and is stored in
#'   the compute type, so n must stay below 65504 (the f16 maximum); larger
#'   vocabularies need \code{ag_dtype("f32")}. Inputs of magnitude ~1e4 are
#'   themselves quantised to a few units in f16 -- a property of the input.
#'
#' @param x ag_tensor or numeric matrix, \code{[n, batch]}.
#' @return ag_tensor of the same shape.
#' @export
#' @examples
#' lp <- ag_log_softmax(ag_tensor(matrix(c(1, 2, 3, 0, -1e4, 0), 3, 2)))
ag_log_softmax <- function(x) {
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    y <- .ag_gpu_log_softmax(.ag_operand(x))
  } else {
    m  <- .ag_data(x)
    mx <- matrix(apply(m, 2, max), nrow(m), ncol(m), byrow = TRUE)
    z  <- m - mx
    y  <- z - matrix(log(colSums(exp(z))), nrow(m), ncol(m), byrow = TRUE)
  }

  out <- .ag_wrap_result(y, device)
  out$requires_grad <- is_ag_tensor(x) && x$requires_grad
  if (out$requires_grad) {
    y_snap <- y
    grad_fn <- function(grad_out) {
      p <- exp(.ag_as_matrix(y_snap))
      list(x = grad_out - p * matrix(colSums(grad_out), nrow(p), ncol(p), byrow = TRUE))
    }
    out$grad_fn <- grad_fn
    ag_record(out, grad_fn, list(x = x), op = "log_softmax", y_snap = y_snap)
  }
  out
}

# logp = (x - c) - log(colSums(exp(x - c))), c = colSums(x * softmax(x)).
# sum_rows reduces ne[0] (R's rows) -> [1, ncol], and ggml_sub broadcasts that
# back over [n, ncol] -- the same pattern as the softmax backward rule.
.ag_gpu_log_softmax <- function(x_data) {
  .ag_run_op(
    op_fn = function(ctx, ptrs) {
      x  <- ptrs[[1L]]
      c0 <- ggml_sum_rows(ctx, ggml_mul(ctx, x, ggml_soft_max(ctx, x)))
      z  <- ggml_sub(ctx, x, c0)
      ggml_sub(ctx, z, ggml_log(ctx, ggml_sum_rows(ctx, ggml_exp(ctx, z))))
    },
    inputs    = list(x_data),
    out_shape = .ag_dim(x_data),
    resident  = .ag_is_handle(x_data)
  )
}

#' Pick one entry per column
#'
#' \code{out[1, j] = x[index[j] + 1, j]}: the log-probability of the action
#' taken in sample j, for example. Differentiable in \code{x}; the gradient of
#' column j lands on the picked row only.
#'
#' Built as \code{colSums(x * onehot(index))}, so the graph backward covers it
#' with no rule of its own. The one-hot is \code{[n, batch]} and is uploaded on
#' every call: cheap for tens of classes (games), wasteful for thousands --
#' there a dedicated op with an I32 index tensor is the right tool (TODO.md).
#'
#' @param x ag_tensor or numeric matrix, \code{[n, batch]}.
#' @param index Integer-valued vector of length \code{ncol(x)}, \strong{0-based},
#'   each in \code{0..nrow(x)-1}. A 1-based index is an error (out of range for
#'   \code{n}) or, worse, silently the neighbouring row -- so the range is checked
#'   and \code{NA} / non-integers are rejected.
#' @return ag_tensor \code{[1, batch]}.
#' @export
#' @examples
#' ag_gather(matrix(1:6 + 0, 3, 2), c(2L, 0L))   # 3 4
ag_gather <- function(x, index) {
  d <- .ag_sel_dim(x)
  n <- d[1L]; b <- d[2L]
  if (!is.numeric(index))
    stop("ag_gather: index must be numeric (0-based integers)", call. = FALSE)
  if (length(index) != b)
    stop(sprintf("ag_gather: index has length %d, x has %d columns", length(index), b),
         call. = FALSE)
  if (anyNA(index))
    stop("ag_gather: index contains NA", call. = FALSE)
  if (any(index != round(index)))
    stop("ag_gather: index must be whole numbers (0-based)", call. = FALSE)
  if (any(index < 0) || any(index > n - 1L))
    stop(sprintf("ag_gather: index out of range 0..%d (indices are 0-based)", n - 1L),
         call. = FALSE)

  onehot <- matrix(0, n, b)
  onehot[cbind(as.integer(index) + 1L, seq_len(b))] <- 1
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  ag_sum(ag_mul(x, ag_tensor(onehot, device = device)), dim = 2L)
}

#' Column-wise maximum
#'
#' \code{ag_gather(x, ag_argmax(x))}: the largest entry of every column, as a
#' \code{[1, batch]} ag_tensor. Differentiable; the gradient goes to the
#' selected entry only. With ties that is the FIRST maximum (the
#' \code{ag_argmax} rule), and it is not split between the tied entries --
#' a convention, which some frameworks resolve differently.
#'
#' Calls \code{ag_argmax}, so on the GPU it is a synchronisation point.
#'
#' @param x ag_tensor or numeric matrix, \code{[n, batch]}.
#' @return ag_tensor \code{[1, batch]}.
#' @export
ag_max <- function(x) ag_gather(x, ag_argmax(x))

# Shape of x without downloading it: a resident tensor answers from its handle.
.ag_sel_dim <- function(x) {
  h <- if (is_ag_tensor(x)) .ag_handle_of(x) else NULL
  d <- if (!is.null(h)) .ag_dim(h) else dim(.ag_data(x))
  if (is.null(d)) c(length(.ag_data(x)), 1L) else d
}

#' Sample one index per column from logits
#'
#' Categorical sampling by the Gumbel-max trick:
#' \code{argmax(x + G)} with \code{G = -log(-log(u))}, \code{u ~ U(0, 1)},
#' draws index i of column j with probability \code{softmax(x[, j])[i]}. \code{x}
#' may be logits or log-probabilities (they differ by a per-column constant).
#'
#' The noise comes from \code{stats::runif}, so \code{set.seed} reproduces the
#' draw, and the CPU and GPU paths see the same noise. On the GPU only the noise
#' goes up and only the indices come down: \code{x} stays on the device, and the
#' add and argmax are one launch. Like \code{ag_argmax} it is a synchronisation
#' point and not differentiable; get the log-probability of the drawn action with
#' \code{ag_gather(ag_log_softmax(x), a)}.
#'
#' Mask illegal entries as for \code{ag_log_softmax}: add a large but FINITE
#' negative number (\code{-1e4}). Gumbel noise exceeds 1e4 with probability
#' about \code{exp(-1e4)}, so a masked entry is never drawn while one column
#' entry is unmasked.
#'
#' Near-ties of \code{x + G} are rounded in f32 on the GPU and in double on the
#' CPU, so the two paths can differ in a rare column even with the same seed.
#'
#' @param x ag_tensor or numeric matrix, \code{[n, batch]}.
#' @return Integer vector of length \code{ncol(x)}, 0-based.
#' @export
#' @examples
#' set.seed(1)
#' ag_sample_categorical(matrix(c(0, 0, -1e4,  5, 0, 0), 3, 2))
ag_sample_categorical <- function(x) {
  d <- .ag_sel_dim(x)
  g <- matrix(-log(-log(stats::runif(d[1L] * d[2L]))), d[1L], d[2L])
  device <- if (is_ag_tensor(x)) x$device else "cpu"
  if (device == "gpu") {
    idx <- .ag_run_op(
      op_fn = function(ctx, ptrs) {
        # Add in f32: ggml_argmax is F32-only, and an f16 sum would quantise
        # x + G and bias near-ties towards whichever entry rounds up.
        f32 <- function(a) {
          if (ggml_tensor_type(a) != GGML_TYPE_F32) ggml_cast(ctx, a, GGML_TYPE_F32) else a
        }
        ggml_argmax(ctx, ggml_add(ctx, f32(ptrs[[1L]]), f32(ptrs[[2L]])))
      },
      inputs    = list(.ag_operand(x), g),
      out_shape = c(1L, d[2L])
    )
    return(as.integer(idx))
  }
  m <- .ag_as_matrix(.ag_data(x))
  if (is.null(dim(m))) m <- matrix(m, ncol = 1L)
  m <- m + g
  m[is.na(m)] <- -Inf
  max.col(t(m), ties.method = "first") - 1L
}

#' Sample from a diagonal Gaussian
#'
#' \code{mu + exp(log_std) * eps} with \code{eps ~ N(0, 1)} from
#' \code{stats::rnorm} (so \code{set.seed} reproduces it). Differentiable in
#' \code{mu} and \code{log_std} -- the reparameterisation used by SAC; for PPO
#' collection the result is just the action.
#'
#' \code{log_std} is either the shape of \code{mu} (state-dependent) or one
#' column \code{[n, 1]} broadcast over the batch (state-independent). On the GPU
#' \code{mu} and \code{log_std} stay on the device; only \code{eps} is uploaded.
#'
#' @param mu ag_tensor or numeric matrix, \code{[n, batch]}.
#' @param log_std ag_tensor or numeric matrix, \code{[n, batch]} or \code{[n, 1]}.
#' @return ag_tensor \code{[n, batch]}.
#' @export
#' @examples
#' set.seed(1)
#' a <- ag_sample_normal(matrix(0, 2, 3), matrix(log(0.5), 2, 1))
ag_sample_normal <- function(mu, log_std) {
  d  <- .ag_sel_dim(mu)
  ds <- .ag_sel_dim(log_std)
  if (ds[1L] != d[1L] || !(ds[2L] %in% c(1L, d[2L])))
    stop(sprintf("ag_sample_normal: log_std is [%d, %d], expected [%d, %d] or [%d, 1]",
                 ds[1L], ds[2L], d[1L], d[2L], d[1L]), call. = FALSE)
  device <- .ag_result_device(mu, log_std)
  eps <- ag_tensor(matrix(stats::rnorm(d[1L] * d[2L]), d[1L], d[2L]), device = device)
  ag_add(mu, ag_mul(eps, ag_exp(log_std)))
}
