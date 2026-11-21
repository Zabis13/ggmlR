# Returns, GAE and n-step targets for reinforcement learning.
#
# Plain R on host matrices: no gradient flows through these, the rollout data
# already lives on the host, and the recursion runs backwards in time (ggml has
# no scan op). Vectorised over environments, looped over time.
#
# Layout: [T, n_envs] -- time down the rows, one column per environment.
#
# Episode ends follow Gymnasium: `terminated` (true terminal state, value 0)
# and `truncated` (time limit etc., the episode was cut: bootstrap from the
# value of the FINAL observation). A step that is both counts as terminated.

.ag_rl_mat <- function(x, name, d = NULL, logical = FALSE, na_ok = FALSE) {
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (is.null(dim(x))) x <- matrix(x, ncol = if (is.null(d)) 1L else d[2L])
  if (logical) {
    if (!(is.logical(x) || is.numeric(x)))
      stop(name, " must be logical or 0/1", call. = FALSE)
    if (anyNA(x)) stop(name, " contains NA", call. = FALSE)
    if (is.numeric(x) && !all(x %in% c(0, 1)))
      stop(name, " must be logical or 0/1", call. = FALSE)
    x <- matrix(as.numeric(x), nrow(x), ncol(x))
  } else {
    if (!is.numeric(x)) stop(name, " must be numeric", call. = FALSE)
    if (!na_ok && anyNA(x)) stop(name, " contains NA", call. = FALSE)
    storage.mode(x) <- "double"
  }
  if (!is.null(d) && !identical(as.integer(dim(x)), as.integer(d)))
    stop(sprintf("%s is [%s], expected [%d, %d] (time x envs)", name,
                 paste(dim(x), collapse = ", "), d[1L], d[2L]), call. = FALSE)
  x
}

# Common inputs: term / done flags as 0/1 matrices, next_values checked where read.
.ag_rl_flags <- function(d, terminated, truncated, next_values, caller) {
  term  <- .ag_rl_mat(terminated, "terminated", d, logical = TRUE)
  trunc <- .ag_rl_mat(truncated,  "truncated",  d, logical = TRUE)
  if (is.null(trunc)) trunc <- term * 0
  trunc <- trunc * (1 - term)                 # both -> terminated
  nv <- .ag_rl_mat(next_values, "next_values", d, na_ok = TRUE)
  if (any(trunc == 1)) {
    if (is.null(nv))
      stop(caller, ": truncated steps need next_values (the value of the final ",
           "observation)", call. = FALSE)
    if (anyNA(nv[trunc == 1]))
      stop(caller, ": next_values is NA on a truncated step", call. = FALSE)
  }
  list(term = term, trunc = trunc, done = pmax(term, trunc), nv = nv)
}

.ag_rl_scalar <- function(x, name, lo = 0, hi = 1) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x < lo || x > hi)
    stop(name, " must be a number in [", lo, ", ", hi, "]", call. = FALSE)
  x
}

# Value that follows step t: 0 if terminated, next_values on a truncated step,
# else `cont` (values[t+1], or last_value at the end of the rollout).
.ag_rl_next <- function(f, t, cont) {
  v <- cont
  v[f$done[t, ] == 1] <- 0                 # last_value / values never leak past an end
  tr <- f$trunc[t, ] == 1
  if (any(tr)) v[tr] <- f$nv[t, tr]
  v
}

.ag_rl_last <- function(last_value, n) {
  if (is.null(last_value)) last_value <- 0
  if (!is.numeric(last_value) || !(length(last_value) %in% c(1L, n)) || anyNA(last_value))
    stop("last_value must be numeric, one per environment", call. = FALSE)
  rep_len(as.numeric(last_value), n)
}

#' Generalised advantage estimation and discounted returns
#'
#' \code{ag_gae} computes GAE(\eqn{\lambda}) (Schulman et al. 2016) backwards
#' over a rollout:
#' \deqn{\delta_t = r_t + \gamma V_{next} - V(s_t), \quad
#'       A_t = \delta_t + \gamma \lambda (1 - done_t) A_{t+1}}
#' and \code{returns = advantages + values}, as in Stable-Baselines3 and
#' CleanRL. \code{ag_discounted_returns} is the bootstrapped Monte-Carlo
#' return \eqn{G_t = r_t + \gamma G_{next}}; it equals \code{ag_gae} with
#' \code{lambda = 1} plus \code{values}.
#'
#' \eqn{V_{next}} after step t is
#' \itemize{
#'   \item 0 when \code{terminated[t]} (true terminal state);
#'   \item \code{next_values[t]} when \code{truncated[t]}: the value of the
#'     FINAL observation of the cut episode -- not of the first observation
#'     after an automatic reset, which belongs to the next episode (Gymnasium's
#'     \code{final_obs} / SB3's \code{terminal_observation});
#'   \item \code{values[t + 1]} inside an episode, and \code{last_value} after
#'     the last step of the rollout (not used when that step ended an episode).
#' }
#' The recursion is cut at every episode end. A step both terminated and
#' truncated counts as terminated.
#'
#' Without \code{truncated}, \code{terminated} plays the role of CleanRL's
#' \code{dones}: a time-limit cut is then treated as a terminal state, which
#' biases the values of states near the limit downwards.
#'
#' \code{turn_sign} is for two-player self-play, where the next state is seen
#' by the other player: \eqn{V_{next}} (including \code{next_values} and
#' \code{last_value}) and the carried \eqn{A_{t+1}} / \eqn{G_{t+1}} are
#' multiplied by it. Use -1 for alternating zero-sum moves, 1 (default)
#' otherwise.
#'
#' @param rewards Numeric \code{[T, n_envs]} (a vector for one environment).
#' @param values Numeric \code{[T, n_envs]}, \eqn{V(s_t)}.
#' @param terminated Logical or 0/1 \code{[T, n_envs]}, or \code{NULL} (no
#'   terminal steps; then \code{continues} is required).
#' @param truncated Logical or 0/1 \code{[T, n_envs]}, or \code{NULL}.
#' @param next_values Numeric \code{[T, n_envs]}: value of the final
#'   observation, read only on truncated steps (other entries may be NA).
#'   Required when any step is truncated.
#' @param last_value Value of the state after the rollout, one per environment
#'   (default 0).
#' @param gamma Discount factor in \code{[0, 1]}.
#' @param lambda GAE \eqn{\lambda} in \code{[0, 1]}: 0 gives the one-step TD
#'   error, 1 the Monte-Carlo return minus \code{values}.
#' @param turn_sign 1 (default) or -1, one number or one per environment.
#' @param continues \code{ag_gae} only. Numeric \code{[T, n_envs]} in
#'   \code{[0, 1]}, or \code{NULL} (all 1): soft probability \eqn{c_t} that the
#'   episode goes on after step t, e.g. a learned continue head in imagined
#'   rollouts (DreamerV3). It scales the discount of step t:
#'   \eqn{\delta_t = r_t + \gamma c_t V_{next} - V(s_t)},
#'   \eqn{A_t = \delta_t + \gamma \lambda c_t (1 - done_t) A_{t+1}}, so
#'   \code{returns} is the DreamerV3 \eqn{\lambda}-return
#'   \eqn{R_t = r_t + \gamma c_t ((1 - \lambda) V_{next} + \lambda R_{t+1})}.
#'   Combines with \code{terminated} / \code{truncated} by multiplication.
#' @return \code{ag_gae}: list of \code{advantages} and \code{returns}, both
#'   \code{[T, n_envs]}. \code{ag_discounted_returns}: the \code{[T, n_envs]}
#'   matrix of returns.
#' @seealso \code{\link{ag_nstep_targets}}
#' @export
#' @examples
#' r <- matrix(1, 4, 2); v <- matrix(0.5, 4, 2)
#' term <- matrix(c(FALSE, TRUE, FALSE, FALSE), 4, 2)
#' ag_gae(r, v, term, last_value = c(0.5, 0.5), gamma = 0.99, lambda = 0.95)$advantages
ag_gae <- function(rewards, values, terminated = NULL, truncated = NULL,
                   next_values = NULL, last_value = 0, gamma = 0.99, lambda = 0.95,
                   turn_sign = 1, continues = NULL) {
  r <- .ag_rl_mat(rewards, "rewards")
  d <- dim(r)
  v <- .ag_rl_mat(values, "values", d)
  if (is.null(terminated)) {
    if (is.null(continues))
      stop("ag_gae: give terminated, continues, or both", call. = FALSE)
    terminated <- matrix(FALSE, d[1L], d[2L])
  }
  f <- .ag_rl_flags(d, terminated, truncated, next_values, "ag_gae")
  cm <- if (is.null(continues)) matrix(1, d[1L], d[2L]) else
    .ag_rl_mat(continues, "continues", d)
  if (any(cm < 0 | cm > 1))
    stop("ag_gae: continues must be in [0, 1]", call. = FALSE)
  gamma  <- .ag_rl_scalar(gamma, "gamma")
  lambda <- .ag_rl_scalar(lambda, "lambda")
  s    <- .ag_rl_sign(turn_sign, d[2L])
  cont <- .ag_rl_last(last_value, d[2L])
  adv <- matrix(0, d[1L], d[2L])
  a <- numeric(d[2L])
  for (t in rev(seq_len(d[1L]))) {
    nxt   <- s * cm[t, ] * .ag_rl_next(f, t, cont)
    delta <- r[t, ] + gamma * nxt - v[t, ]
    a     <- delta + gamma * lambda * (1 - f$done[t, ]) * cm[t, ] * s * a
    adv[t, ] <- a
    cont  <- v[t, ]
  }
  list(advantages = adv, returns = adv + v)
}

#' @rdname ag_gae
#' @export
ag_discounted_returns <- function(rewards, terminated, truncated = NULL,
                                  next_values = NULL, last_value = 0, gamma = 0.99,
                                  turn_sign = 1) {
  r <- .ag_rl_mat(rewards, "rewards")
  d <- dim(r)
  f <- .ag_rl_flags(d, terminated, truncated, next_values, "ag_discounted_returns")
  gamma <- .ag_rl_scalar(gamma, "gamma")
  s <- .ag_rl_sign(turn_sign, d[2L])
  g <- .ag_rl_last(last_value, d[2L])
  out <- matrix(0, d[1L], d[2L])
  for (t in rev(seq_len(d[1L]))) {
    g <- r[t, ] + gamma * s * .ag_rl_next(f, t, g)
    out[t, ] <- g
  }
  out
}

.ag_rl_sign <- function(x, n) {
  if (!is.numeric(x) || !(length(x) %in% c(1L, n)) || anyNA(x) || !all(x %in% c(-1, 1)))
    stop("turn_sign must be 1 or -1, one number or one per environment", call. = FALSE)
  rep_len(as.numeric(x), n)
}

#' n-step bootstrapped targets
#'
#' For every step t (DQN-style replay targets):
#' \deqn{y_t = \sum_{k=0}^{m-1} \gamma^k r_{t+k} + \gamma^m (1 - terminated_{t+m-1})\, next\_values_{t+m-1}}
#' where the window length \eqn{m \le n} stops early at the first episode end
#' (terminated or truncated) or at the end of the rollout. The bootstrap is
#' always \code{next_values} of the LAST step of the window: 0 after a
#' terminated step, the value of the final observation after a truncated one,
#' \eqn{V(s_{t+m})} otherwise. \code{n = 1} gives the one-step TD target.
#'
#' Unlike \code{\link{ag_gae}}, \code{next_values} is read on every step:
#' \code{next_values[t]} is \eqn{V(s_{t+1})} (target network on the next
#' observation, the final observation when the episode ended at t). It may be
#' NA only on terminated steps.
#'
#' @inheritParams ag_gae
#' @param next_values Numeric \code{[T, n_envs]}, \eqn{V(s_{t+1})}.
#' @param n Number of steps, a positive integer.
#' @return Numeric \code{[T, n_envs]}.
#' @export
#' @examples
#' r <- matrix(1, 5, 1); nv <- matrix(10, 5, 1)
#' ag_nstep_targets(r, nv, terminated = c(FALSE, FALSE, TRUE, FALSE, FALSE),
#'                  gamma = 0.9, n = 3)
ag_nstep_targets <- function(rewards, next_values, terminated, truncated = NULL,
                             gamma = 0.99, n = 1L) {
  r <- .ag_rl_mat(rewards, "rewards")
  d <- dim(r)
  term  <- .ag_rl_mat(terminated, "terminated", d, logical = TRUE)
  trunc <- .ag_rl_mat(truncated,  "truncated",  d, logical = TRUE)
  if (is.null(trunc)) trunc <- term * 0
  done <- pmax(term, trunc)
  nv <- .ag_rl_mat(next_values, "next_values", d, na_ok = TRUE)
  if (anyNA(nv[term == 0]))
    stop("ag_nstep_targets: next_values is NA on a step that is not terminated",
         call. = FALSE)
  nv[term == 1] <- 0                       # never read; keeps NA out of 0 * NA
  gamma <- .ag_rl_scalar(gamma, "gamma")
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 1 || n != round(n))
    stop("n must be a positive integer", call. = FALSE)
  n <- as.integer(n)
  nT <- d[1L]
  out <- matrix(0, nT, d[2L])
  for (t in seq_len(nT)) {
    y <- numeric(d[2L]); alive <- rep(TRUE, d[2L]); disc <- 1
    for (k in 0:(n - 1L)) {
      s <- t + k
      if (s > nT) break
      y <- y + alive * disc * r[s, ]
      stop_here <- alive & (done[s, ] == 1 | k == n - 1L | s == nT)
      y <- y + stop_here * disc * gamma * (1 - term[s, ]) * nv[s, ]
      alive <- alive & !stop_here
      if (!any(alive)) break
      disc <- disc * gamma
    }
    out[t, ] <- y
  }
  out
}
