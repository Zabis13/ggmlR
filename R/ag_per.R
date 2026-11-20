# Prioritized experience replay: the sampler, independent of where the
# transitions are stored (TorchRL's Sampler / Storage split). The sum / min
# tree is in C (src/r_per_tree.c); random numbers come from R's RNG.

#' Prioritized replay sampler
#'
#' Proportional prioritized experience replay (Schaul et al. 2016) over slot
#' indices \code{0 .. capacity - 1}; it stores priorities only, not
#' transitions, so it pairs with any storage (\code{\link{ag_replay_buffer}}
#' uses it with \code{prioritized = TRUE}).
#'
#' \itemize{
#'   \item priority \eqn{p_i = |\delta_i| + eps}, probability
#'     \eqn{P(i) = p_i^\alpha / \sum_k p_k^\alpha};
#'   \item \code{$add(slots)} gives new (or overwritten) slots the largest
#'     priority seen so far (1 at the start), so each is drawn at least once
#'     with high probability;
#'   \item \code{$sample(batch, beta)} draws stratified: the total mass is cut
#'     into \code{batch} equal segments, one draw in each. It returns the
#'     0-based slots and the importance weights
#'     \eqn{w_i = (N P(i))^{-\beta} / \max_j w_j}, normalised by the maximum
#'     over the WHOLE sampler (smallest priority), not over the batch, as in the
#'     paper and OpenAI baselines; N is the number of slots with a priority;
#'   \item \code{$update(slots, td_errors)} sets \eqn{p = |\delta| + eps} for
#'     the slots a batch returned;
#'   \item \code{$set_alpha(a)} rebuilds the tree in one pass (alpha schedules);
#'     \code{beta} is an argument of every draw, so annealing it is up to the
#'     caller.
#' }
#' \code{alpha = 0} samples uniformly, \code{beta = 0} gives weights 1.
#'
#' The tree lives in C memory: it does not survive \code{saveRDS}; use
#' \code{$priorities()} / \code{$load_priorities()} (\code{ag_replay_buffer}
#' saves them with the transitions).
#'
#' @param capacity Number of slots.
#' @param alpha Priority exponent, \code{>= 0} (default 0.6).
#' @param eps Added to \code{|td_error|} so no priority is 0 (default 1e-6).
#' @return An \code{ag_per_sampler} environment with the methods above, plus
#'   \code{$size()} (slots with a priority), \code{$alpha}, \code{$eps},
#'   \code{$capacity}.
#' @export
#' @examples
#' s <- ag_per_sampler(8L, alpha = 0.6)
#' s$add(0:7)
#' b <- s$sample(4L, beta = 0.4)
#' s$update(b$idx, td_errors = c(0.1, 2, 0.5, 1))
ag_per_sampler <- function(capacity, alpha = 0.6, eps = 1e-6) {
  capacity <- as.integer(capacity)
  if (length(capacity) != 1L || is.na(capacity) || capacity < 1L)
    stop("ag_per_sampler: capacity must be a positive integer", call. = FALSE)
  .ag_per_check_alpha(alpha)
  if (!is.numeric(eps) || length(eps) != 1L || is.na(eps) || eps <= 0)
    stop("ag_per_sampler: eps must be a positive number", call. = FALSE)

  env <- new.env(parent = emptyenv())
  env$capacity <- capacity
  env$alpha    <- alpha
  env$eps      <- eps
  env$tree     <- .Call("R_per_new", capacity, as.double(alpha), PACKAGE = "ggmlR")

  slots_int <- function(slots) {
    s <- as.integer(slots)
    if (anyNA(s) || any(s < 0L) || any(s >= env$capacity))
      stop("ag_per_sampler: slots must be in 0 .. capacity - 1", call. = FALSE)
    s
  }
  stats <- function() .Call("R_per_stats", env$tree, PACKAGE = "ggmlR")

  env$add <- function(slots) {
    s <- slots_int(slots)
    .Call("R_per_set", env$tree, s, stats()[3L], PACKAGE = "ggmlR")
    invisible(env)
  }

  env$update <- function(slots, td_errors) {
    s <- slots_int(slots)
    td <- as.numeric(td_errors)
    if (length(td) != length(s) || anyNA(td) || any(!is.finite(td)))
      stop("ag_per_sampler: td_errors must be finite, one per slot", call. = FALSE)
    .Call("R_per_set", env$tree, s, abs(td) + env$eps, PACKAGE = "ggmlR")
    invisible(env)
  }

  env$sample <- function(batch, beta = 0.4) {
    batch <- as.integer(batch)
    if (length(batch) != 1L || is.na(batch) || batch < 1L)
      stop("ag_per_sampler: batch must be a positive integer", call. = FALSE)
    if (!is.numeric(beta) || length(beta) != 1L || is.na(beta) || beta < 0)
      stop("ag_per_sampler: beta must be a number >= 0", call. = FALSE)
    st <- stats()
    if (st[4L] < 1) stop("ag_per_sampler: sampling from an empty sampler", call. = FALSE)
    idx  <- .Call("R_per_sample", env$tree, stats::runif(batch), PACKAGE = "ggmlR")
    leaf <- .Call("R_per_leaf", env$tree, idx, PACKAGE = "ggmlR")
    # w / max w = ((N * leaf/total) / (N * min/total))^-beta = (leaf / min)^-beta
    list(idx = idx, weights = (leaf / st[2L])^(-beta))
  }

  env$set_alpha <- function(alpha) {
    .ag_per_check_alpha(alpha)
    .Call("R_per_set_alpha", env$tree, as.double(alpha), PACKAGE = "ggmlR")
    env$alpha <- alpha
    invisible(env)
  }

  env$size <- function() as.integer(stats()[4L])

  # Raw priorities (|td| + eps), NA for slots without one; and back.
  env$priorities <- function() .Call("R_per_raw", env$tree, PACKAGE = "ggmlR")
  env$load_priorities <- function(p) {
    p <- as.numeric(p)
    if (length(p) != env$capacity)
      stop("ag_per_sampler: priorities must have length capacity", call. = FALSE)
    if (any(!is.na(p) & !(p > 0 & is.finite(p))))
      stop("ag_per_sampler: priorities must be positive and finite (or NA)", call. = FALSE)
    .Call("R_per_set", env$tree, seq_len(env$capacity) - 1L, p, PACKAGE = "ggmlR")
    if (any(!is.na(p)))
      .Call("R_per_set_max", env$tree, max(p, na.rm = TRUE), PACKAGE = "ggmlR")
    invisible(env)
  }

  class(env) <- "ag_per_sampler"
  env
}

.ag_per_check_alpha <- function(alpha) {
  if (!is.numeric(alpha) || length(alpha) != 1L || is.na(alpha) || alpha < 0)
    stop("ag_per_sampler: alpha must be a number >= 0", call. = FALSE)
}

#' @export
print.ag_per_sampler <- function(x, ...) {
  cat("<ag_per_sampler> ", x$size(), " / ", x$capacity, " slots, alpha ", x$alpha,
      ", eps ", x$eps, "\n", sep = "")
  invisible(x)
}
