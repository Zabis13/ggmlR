# Element-wise ops composed from existing ag_* primitives. Each one inherits
# the device path (CPU closure / resident GPU handle) and the backward of the
# ops it is built from, so no new kernel or graph rule is needed.

#' Element-wise division
#'
#' \code{A / B}. A numeric scalar \code{B} becomes \code{ag_scale(A, 1 / B)};
#' otherwise \code{A * B^-1}, broadcasting \code{B} as \code{\link{ag_mul}} does.
#'
#' @param A ag_tensor or numeric matrix
#' @param B ag_tensor, numeric matrix, or numeric scalar
#' @return ag_tensor
#' @export
#' @examples
#' as.matrix(ag_div(ag_tensor(matrix(1:4, 2, 2)), 2))
ag_div <- function(A, B) {
  if (!is_ag_tensor(B) && is.numeric(B) && length(B) == 1L && is.null(dim(B)))
    return(ag_scale(A, 1 / B))
  ag_mul(A, ag_pow(B, -1))
}

#' @rdname ag_exp
#' @details \code{ag_sqrt()}: \code{x^0.5}; the gradient is infinite at 0.
#' Returns: ag_tensor
#' @export
ag_sqrt <- function(x) ag_pow(x, 0.5)

#' @rdname ag_exp
#' @details \code{ag_abs()}: \code{relu(x) + relu(-x)}; the gradient at 0 is 0.
#' Returns: ag_tensor
#' @export
ag_abs <- function(x) ag_add(ag_relu(x), ag_relu(ag_scale(x, -1)))

#' Element-wise minimum and maximum of two tensors
#'
#' \code{minimum(A, B) = A - relu(A - B)}, \code{maximum(A, B) = A + relu(B - A)}.
#' The result has the shape of \code{A}; \code{B} may broadcast over it as in
#' \code{\link{ag_sub}}. On a tie the whole gradient goes to \code{A}.
#' Typical use is the PPO clipped objective
#' \code{ag_minimum(ag_mul(r, adv), ag_mul(ag_clamp(r, 1 - eps, 1 + eps), adv))}.
#'
#' @param A ag_tensor or numeric matrix
#' @param B ag_tensor or numeric matrix (same shape, \code{[m,1]}, \code{[1,n]}
#'   or \code{[1,1]})
#' @return ag_tensor
#' @export
ag_minimum <- function(A, B) ag_sub(A, ag_relu(ag_sub(A, B)))

#' @rdname ag_minimum
#' @export
ag_maximum <- function(A, B) ag_add(A, ag_relu(ag_scale(ag_sub(A, B), -1)))

#' Element-wise selection by a mask
#'
#' \code{B + cond * (A - B)}: \code{A} where \code{cond} is 1, \code{B} where it
#' is 0. \code{cond} is a constant (no gradient flows into it). \code{A} and
#' \code{B} must be finite: \code{Inf - Inf} gives \code{NaN} -- mask with a
#' large finite number (\code{-1e4}) instead, as for \code{\link{ag_log_softmax}}.
#'
#' @param cond logical or 0/1 numeric matrix, or an ag_tensor holding 0/1
#' @param A ag_tensor or numeric matrix
#' @param B ag_tensor or numeric matrix, same shape as \code{A}
#' @return ag_tensor
#' @export
#' @examples
#' as.matrix(ag_where(matrix(c(TRUE, FALSE), 1, 2),
#'                    ag_tensor(matrix(1, 1, 2)), ag_tensor(matrix(5, 1, 2))))
ag_where <- function(cond, A, B) {
  if (!is_ag_tensor(cond)) {
    d <- dim(cond)
    cond <- matrix(as.numeric(cond), nrow = if (is.null(d)) 1L else d[1L])
  }
  ag_add(B, ag_mul(cond, ag_sub(A, B)))
}

#' @rdname ag_exp
#' @details \code{ag_softplus()}: \code{log(1 + exp(x))} in the stable form
#' \code{relu(x) - log(sigmoid(|x|))}: \code{sigmoid(|x|)} lies in
#' \code{[0.5, 1]}, so nothing overflows or underflows. Gradient
#' \code{sigmoid(x)}.
#' Returns: ag_tensor
#' @export
ag_softplus <- function(x) ag_sub(ag_relu(x), ag_log(ag_sigmoid(ag_abs(x))))

#' Column-wise log-sum-exp
#'
#' \code{log(colSums(exp(x)))} for a \code{[n, batch]} tensor, the package's
#' layout (one column = one sample). Computed as \code{x[1, ] - log_softmax(x)[1, ]},
#' so it inherits the stability of \code{\link{ag_log_softmax}} and needs no max
#' reduction or host synchronisation. Gradient \code{softmax(x)}.
#'
#' @param x ag_tensor or numeric matrix, \code{[n, batch]}
#' @return ag_tensor \code{[1, batch]}
#' @export
#' @examples
#' as.matrix(ag_logsumexp(ag_tensor(matrix(c(0, 0, 1000, 1000), 2, 2))))
ag_logsumexp <- function(x) {
  ag_slice_rows(ag_sub(x, ag_log_softmax(x)), 0L, 1L)
}

#' Stop gradient
#'
#' Returns the value of \code{x} as a tensor that is not recorded on the tape:
#' nothing flows back through it. Use it for TD targets, advantages, or the
#' old-policy term of a KL penalty.
#'
#' When \code{x} is resident on the device the result shares its buffer (no
#' download, no copy), as \code{detach()} does in PyTorch. A later in-place
#' update of \code{x} -- an optimizer step on a parameter -- is therefore
#' visible through the result; copy the weights instead when a value must be
#' frozen across steps (e.g. a target network).
#'
#' @param x ag_tensor or numeric matrix
#' @return ag_tensor with \code{requires_grad = FALSE}
#' @export
#' @examples
#' W <- ag_param(matrix(1:4, 2, 2))
#' ag_detach(W)$requires_grad
ag_detach <- function(x) {
  if (!is_ag_tensor(x)) return(ag_tensor(x))
  h <- .ag_handle_of(x)
  if (!is.null(h)) return(.ag_tensor_from_handle(h, dtype = x$dtype))
  ag_tensor(.ag_data(x), device = x$device, dtype = x$dtype)
}
