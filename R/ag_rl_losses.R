# Losses and distribution terms for reinforcement learning, composed from
# existing ag_* primitives (no new kernel or graph rule). Per-sample results are
# [1, batch], one column per sample, the package's layout.

# Shared reduction: "none" keeps the shape, "mean" / "sum" reduce everything.
.ag_rl_reduce <- function(x, reduction) {
  switch(reduction,
         none = x,
         mean = ag_mean(x),
         sum  = ag_sum(x))
}

#' Huber and smooth-L1 losses
#'
#' Element-wise, with \code{d = |pred - target|} and \code{q = min(d, delta)}:
#' \deqn{huber = 0.5 q^2 + delta (d - q)}
#' i.e. \code{0.5 d^2} for \code{d <= delta} and \code{delta (d - 0.5 delta)}
#' beyond. \code{ag_smooth_l1_loss} is \code{huber(delta = beta) / beta} (the
#' Fast R-CNN form, slope 1 in the linear part); \code{beta = 0} is plain L1.
#'
#' The gradient is \code{clip(pred - target, -delta, delta)} for Huber, also at
#' \code{d = delta} (the loss is C1 there) and 0 at \code{d = 0}.
#'
#' @param pred ag_tensor \code{[n, batch]}
#' @param target ag_tensor or numeric matrix, same shape
#' @param delta Positive threshold (Huber)
#' @param beta Non-negative threshold (smooth-L1); 0 gives L1
#' @param reduction \code{"mean"} (default, like \code{\link{ag_mse_loss}}),
#'   \code{"sum"}, or \code{"none"} (element-wise, shape of \code{pred})
#' @return ag_tensor: scalar, or \code{[n, batch]} for \code{"none"}
#' @export
#' @examples
#' as.matrix(ag_huber_loss(ag_tensor(matrix(c(0.5, 3), 1, 2)), matrix(0, 1, 2),
#'                         reduction = "none"))
ag_huber_loss <- function(pred, target, delta = 1,
                          reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)
  if (!is.numeric(delta) || length(delta) != 1L || !(delta > 0))
    stop("ag_huber_loss: delta must be a positive number", call. = FALSE)
  .ag_rl_reduce(.ag_huber_elem(pred, target, delta), reduction)
}

#' @rdname ag_huber_loss
#' @export
ag_smooth_l1_loss <- function(pred, target, beta = 1,
                              reduction = c("mean", "sum", "none")) {
  reduction <- match.arg(reduction)
  if (!is.numeric(beta) || length(beta) != 1L || !(beta >= 0))
    stop("ag_smooth_l1_loss: beta must be a non-negative number", call. = FALSE)
  if (beta == 0) return(.ag_rl_reduce(ag_abs(ag_sub(pred, target)), reduction))
  ag_scale(.ag_rl_reduce(.ag_huber_elem(pred, target, beta), reduction), 1 / beta)
}

.ag_huber_elem <- function(pred, target, delta) {
  d <- ag_abs(ag_sub(pred, target))
  q <- ag_minimum(d, matrix(delta, 1L, 1L))
  ag_add(ag_scale(ag_mul(q, q), 0.5), ag_scale(ag_sub(d, q), delta))
}

#' Entropy of a categorical distribution given by logits
#'
#' \code{-sum(p * log p)} down every column, computed as
#' \code{-colSums(exp(logp) * logp)} with \code{logp = \link{ag_log_softmax}(logits)}:
#' masked entries (a large FINITE negative logit such as \code{-1e4}) give
#' \code{0 * finite}, never \code{0 * -Inf}. \code{-Inf} logits are not
#' supported (see \code{\link{ag_log_softmax}}).
#'
#' @param logits ag_tensor or numeric matrix \code{[n, batch]}
#' @param reduction \code{"none"} (default, \code{[1, batch]}), \code{"mean"}
#'   or \code{"sum"}
#' @return ag_tensor \code{[1, batch]}, or scalar
#' @export
#' @examples
#' as.matrix(ag_entropy(ag_tensor(matrix(0, 4, 2))))   # log(4)
ag_entropy <- function(logits, reduction = c("none", "mean", "sum")) {
  reduction <- match.arg(reduction)
  lp <- ag_log_softmax(logits)
  h  <- ag_scale(ag_sum(ag_mul(ag_exp(lp), lp), dim = 2L), -1)
  .ag_rl_reduce(h, reduction)
}

#' Diagonal normal distribution: log-density and entropy
#'
#' For \code{x, mu: [n, batch]} and \code{log_std: [n, batch]} or \code{[n, 1]}
#' (state-independent, broadcast over the batch, as in
#' \code{\link{ag_sample_normal}}), per sample:
#' \deqn{log p = sum_i [-0.5 ((x_i - mu_i) / sigma_i)^2 - log sigma_i] - n/2 log(2 pi)}
#' \deqn{H = sum_i log sigma_i + n/2 (1 + log(2 pi))}
#'
#' Both include the constant: callers must not add \code{-n/2 log(2 pi)}
#' themselves. (It cancels in a PPO ratio, but not in a SAC entropy term.)
#'
#' \code{ag_normal_entropy} does not depend on \code{x}; with a \code{[n, 1]}
#' \code{log_std} it is expanded to \code{[1, batch]} so it has the same shape
#' as \code{ag_normal_log_prob} and \code{\link{ag_entropy}}.
#'
#' @param x ag_tensor or numeric matrix \code{[n, batch]} (usually the action,
#'   a constant)
#' @param mu ag_tensor or numeric matrix \code{[n, batch]}
#' @param log_std ag_tensor or numeric matrix \code{[n, batch]} or \code{[n, 1]}
#' @param batch Number of samples for \code{ag_normal_entropy}; defaults to
#'   \code{ncol(log_std)}. Needed only to expand a \code{[n, 1]} \code{log_std}.
#' @param reduction \code{"none"} (default, \code{[1, batch]}), \code{"mean"}
#'   or \code{"sum"}
#' @return ag_tensor \code{[1, batch]}, or scalar
#' @export
#' @examples
#' as.matrix(ag_normal_log_prob(matrix(0, 2, 1), matrix(0, 2, 1),
#'                              matrix(0, 2, 1)))   # -log(2 pi)
ag_normal_log_prob <- function(x, mu, log_std,
                               reduction = c("none", "mean", "sum")) {
  reduction <- match.arg(reduction)
  d  <- .ag_sel_dim(mu)
  ds <- .ag_sel_dim(log_std)
  if (ds[1L] != d[1L] || !(ds[2L] %in% c(1L, d[2L])))
    stop(sprintf("ag_normal_log_prob: log_std is [%d, %d], expected [%d, %d] or [%d, 1]",
                 ds[1L], ds[2L], d[1L], d[2L], d[1L]), call. = FALSE)
  z  <- ag_mul(ag_sub(x, mu), ag_exp(ag_scale(log_std, -1)))
  sq <- ag_scale(ag_sum(ag_mul(z, z), dim = 2L), -0.5)        # [1, batch]
  lp <- ag_sub(sq, ag_sum(log_std, dim = 2L))                  # [1,1] broadcasts
  lp <- ag_add(lp, matrix(-0.5 * d[1L] * log(2 * pi), 1L, 1L))
  .ag_rl_reduce(lp, reduction)
}

#' @rdname ag_normal_log_prob
#' @export
ag_normal_entropy <- function(log_std, batch = NULL,
                              reduction = c("none", "mean", "sum")) {
  reduction <- match.arg(reduction)
  ds <- .ag_sel_dim(log_std)
  if (is.null(batch)) batch <- ds[2L]
  if (!(ds[2L] %in% c(1L, batch)))
    stop(sprintf("ag_normal_entropy: log_std is [%d, %d], expected [%d, %d] or [%d, 1]",
                 ds[1L], ds[2L], ds[1L], batch, ds[1L]), call. = FALSE)
  h <- ag_add(ag_sum(log_std, dim = 2L),
              matrix(0.5 * ds[1L] * (1 + log(2 * pi)), 1L, 1L))
  if (ds[2L] == 1L && batch > 1L) h <- ag_mul(matrix(1, 1L, batch), h)
  .ag_rl_reduce(h, reduction)
}
