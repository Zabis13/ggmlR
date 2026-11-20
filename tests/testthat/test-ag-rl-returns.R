# Returns, GAE and n-step targets (R/ag_rl_returns.R).
# References are written differently from the code under test: forward sums
# per environment instead of a backward recursion vectorised over envs.

# Value after step t of env j, as documented.
next_after <- function(t, j, v, term, trunc, nv, last) {
  if (term[t, j]) return(0)
  if (trunc[t, j]) return(nv[t, j])
  if (t == nrow(v)) last[j] else v[t + 1L, j]
}

gae_naive <- function(r, v, term, trunc, nv, last, g, l, s = rep(1, ncol(r))) {
  nT <- nrow(r); A <- r * 0
  done <- term | trunc
  delta <- r * 0
  for (j in seq_len(ncol(r))) for (t in seq_len(nT))
    delta[t, j] <- r[t, j] + g * s[j] * next_after(t, j, v, term, trunc, nv, last) - v[t, j]
  for (j in seq_len(ncol(r))) for (t in seq_len(nT)) {
    acc <- 0; w <- 1; k <- t
    repeat {
      acc <- acc + w * delta[k, j]
      if (done[k, j] || k == nT) break
      w <- w * g * l * s[j]; k <- k + 1L
    }
    A[t, j] <- acc
  }
  A
}

ret_naive <- function(r, term, trunc, nv, last, g, s = rep(1, ncol(r))) {
  nT <- nrow(r); G <- r * 0; done <- term | trunc
  v0 <- r * 0                                    # only read at t == nT via last
  for (j in seq_len(ncol(r))) for (t in seq_len(nT)) {
    acc <- 0; w <- 1; k <- t
    repeat {
      acc <- acc + w * r[k, j]
      if (done[k, j] || k == nT) break
      w <- w * g * s[j]; k <- k + 1L
    }
    G[t, j] <- acc + w * g * s[j] * next_after(k, j, v0, term, trunc, nv, last)
  }
  G
}

nstep_naive <- function(r, nv, term, trunc, g, n) {
  nT <- nrow(r); Y <- r * 0; done <- term | trunc
  for (j in seq_len(ncol(r))) for (t in seq_len(nT)) {
    acc <- 0
    for (k in 0:(n - 1L)) {
      s <- t + k
      acc <- acc + g^k * r[s, j]
      if (done[s, j] || k == n - 1L || s == nT) {
        if (!term[s, j]) acc <- acc + g^(k + 1L) * nv[s, j]
        break
      }
    }
    Y[t, j] <- acc
  }
  Y
}

rl_data <- function(seed = 11, nT = 12L, n = 3L) {
  set.seed(seed)
  term  <- matrix(runif(nT * n) < 0.15, nT, n)
  trunc <- matrix(runif(nT * n) < 0.15, nT, n)
  trunc[3L, 1L] <- term[3L, 1L] <- TRUE          # both: counts as terminated
  term[nT, 2L] <- TRUE; trunc[nT, 3L] <- TRUE    # episode ends on the last step
  list(r = matrix(rnorm(nT * n), nT, n), v = matrix(rnorm(nT * n), nT, n),
       term = term, trunc = trunc,
       nv = matrix(rnorm(nT * n), nT, n), last = rnorm(n))
}

test_that("GAE and returns match the forward-sum references", {
  d <- rl_data()
  tr <- d$trunc & !d$term
  for (gl in list(c(0.99, 0.95), c(0.9, 0.5), c(1, 1))) {
    got <- ag_gae(d$r, d$v, d$term, d$trunc, d$nv, d$last, gl[1], gl[2])
    ref <- gae_naive(d$r, d$v, d$term, tr, d$nv, d$last, gl[1], gl[2])
    expect_equal(got$advantages, ref, tolerance = 1e-12)
    expect_equal(got$returns, ref + d$v, tolerance = 1e-12)
    expect_equal(ag_discounted_returns(d$r, d$term, d$trunc, d$nv, d$last, gl[1]),
                 ret_naive(d$r, d$term, tr, d$nv, d$last, gl[1]), tolerance = 1e-12)
  }
})

test_that("limits: lambda = 0 is the TD error, lambda = 1 is MC minus V, gamma = 0", {
  d <- rl_data(12)
  tr <- d$trunc & !d$term
  td <- d$r * 0
  for (j in 1:3) for (t in seq_len(nrow(d$r)))
    td[t, j] <- d$r[t, j] + 0.9 * next_after(t, j, d$v, d$term, tr, d$nv, d$last) - d$v[t, j]
  expect_equal(ag_gae(d$r, d$v, d$term, d$trunc, d$nv, d$last, 0.9, 0)$advantages, td,
               tolerance = 1e-12)
  mc <- ag_discounted_returns(d$r, d$term, d$trunc, d$nv, d$last, 0.9)
  expect_equal(ag_gae(d$r, d$v, d$term, d$trunc, d$nv, d$last, 0.9, 1)$advantages,
               mc - d$v, tolerance = 1e-12)
  expect_equal(ag_gae(d$r, d$v, d$term, d$trunc, d$nv, d$last, 0, 0.7)$advantages,
               d$r - d$v, tolerance = 1e-12)
})

test_that("episode ends: terminal is 0, truncated bootstraps next_values, last_value unused after an end", {
  r <- matrix(1, 3, 1); v <- matrix(0, 3, 1)
  # terminated at t = 2: no bootstrap from v[3]
  expect_equal(as.numeric(ag_discounted_returns(r, c(0, 1, 0), last_value = 5, gamma = 0.5)),
               c(1.5, 1, 1 + 0.5 * 5))
  # truncated at t = 2: bootstrap the final observation's value (10), not v[3]
  expect_equal(as.numeric(ag_discounted_returns(r, c(0, 0, 0), truncated = c(0, 1, 0),
                                                next_values = c(NA, 10, NA), gamma = 0.5)),
               c(1 + 0.5 * 6, 6, 1))
  # both flags: terminated wins, next_values ignored
  expect_equal(ag_discounted_returns(r, c(0, 1, 0), truncated = c(0, 1, 0),
                                     next_values = c(NA, 10, NA), gamma = 0.5),
               ag_discounted_returns(r, c(0, 1, 0), gamma = 0.5))
  # an episode ending on the last step does not read last_value
  d <- rl_data(13)
  a1 <- ag_gae(d$r, d$v, d$term, d$trunc, d$nv, d$last, 0.99, 0.95)
  l2 <- d$last; l2[2:3] <- 1e6                   # envs 2, 3 end on the last step
  a2 <- ag_gae(d$r, d$v, d$term, d$trunc, d$nv, l2, 0.99, 0.95)
  expect_equal(a1$advantages[, 2:3], a2$advantages[, 2:3])
  # without truncated, terminated behaves as CleanRL's dones
  expect_equal(ag_gae(d$r, d$v, d$term | d$trunc, last_value = d$last)$advantages,
               ag_gae(d$r, d$v, d$term | d$trunc, matrix(FALSE, 12, 3),
                      last_value = d$last)$advantages)
})

test_that("n-step targets match the reference; windows crossing an episode end", {
  d <- rl_data(14)
  for (n in c(1L, 2L, 4L, 20L))
    expect_equal(ag_nstep_targets(d$r, d$nv, d$term, d$trunc, 0.9, n),
                 nstep_naive(d$r, d$nv, d$term, d$trunc & !d$term, 0.9, n),
                 tolerance = 1e-12, info = paste("n", n))
  # n = 1 is the one-step TD target
  expect_equal(ag_nstep_targets(d$r, d$nv, d$term, d$trunc, 0.9, 1L),
               d$r + 0.9 * (1 - d$term) * d$nv, tolerance = 1e-12)
  # hand-made: r = 1, gamma = 0.5, n = 3, episode end at step 3
  r <- rep(1, 5); nv <- c(20, 30, 40, 50, 60)
  # terminated at 3: window from t = 1 stops at 3 (last position), from t = 2
  # at its middle position, from t = 3 at its first; no bootstrap
  y <- ag_nstep_targets(r, nv, c(0, 0, 1, 0, 0), gamma = 0.5, n = 3L)
  expect_equal(as.numeric(y)[1:3], c(1.75, 1.5, 1))
  # truncated at 3: same windows, bootstrap nv[3] = 40 from the end of the window
  y <- ag_nstep_targets(r, nv, c(0, 0, 0, 0, 0), truncated = c(0, 0, 1, 0, 0),
                        gamma = 0.5, n = 3L)
  expect_equal(as.numeric(y)[1:3], c(1.75 + 0.125 * 40, 1.5 + 0.25 * 40, 1 + 0.5 * 40))
  # rollout end inside the window: bootstrap next_values of the last step
  expect_equal(as.numeric(y)[4:5], c(1.5 + 0.25 * 60, 1 + 0.5 * 60))
  # NA next_values allowed only on terminated steps
  expect_silent(ag_nstep_targets(r, c(1, 1, NA, 1, 1), c(0, 0, 1, 0, 0), gamma = 0.5, n = 2L))
  expect_error(ag_nstep_targets(r, c(1, 1, NA, 1, 1), c(0, 0, 0, 0, 0), gamma = 0.5),
               "NA on a step")
})

test_that("turn_sign matches the forward references", {
  d <- rl_data(15)
  tr <- d$trunc & !d$term
  s <- c(-1, 1, -1)
  expect_equal(ag_gae(d$r, d$v, d$term, d$trunc, d$nv, d$last, 0.95, 0.9, turn_sign = s)$advantages,
               gae_naive(d$r, d$v, d$term, tr, d$nv, d$last, 0.95, 0.9, s), tolerance = 1e-12)
  expect_equal(ag_discounted_returns(d$r, d$term, d$trunc, d$nv, d$last, 0.95, turn_sign = s),
               ret_naive(d$r, d$term, tr, d$nv, d$last, 0.95, s), tolerance = 1e-12)
  expect_error(ag_gae(d$r, d$v, d$term, turn_sign = 0.5), "turn_sign")
})

# rltoolsR's ppo_gae (R/ppo.R), copied verbatim: [n_env, n_t] layout, the
# truncation bootstrap folded into the reward (reward += gamma * sign * V(final)),
# done = terminated | truncated. ag_gae must give the same advantages.
ppo_gae <- function(b_rew, b_val, b_done, last_val, gamma, lambda, turn_sign) {
  n_t <- ncol(b_rew)
  adv <- matrix(0, nrow(b_rew), n_t)
  next_val <- last_val
  next_adv <- numeric(nrow(b_rew))
  for (t in n_t:1) {
    nd <- 1 - b_done[, t]
    delta <- b_rew[, t] + gamma * nd * turn_sign * next_val - b_val[, t]
    adv[, t] <- delta + gamma * lambda * nd * turn_sign * next_adv
    next_val <- b_val[, t]
    next_adv <- adv[, t]
  }
  adv
}

test_that("ag_gae equals rltoolsR's ppo_gae with its reward-folded truncation", {
  d <- rl_data(16)
  tr <- d$trunc & !d$term
  for (sg in c(1, -1)) {
    nv0 <- d$nv; nv0[!tr] <- 0
    rew <- d$r + 0.99 * sg * nv0                  # rltoolsR's truncation bootstrap
    ref <- ppo_gae(t(rew), t(d$v), t((d$term | d$trunc) * 1), d$last, 0.99, 0.95, sg)
    got <- ag_gae(d$r, d$v, d$term, d$trunc, d$nv, d$last, 0.99, 0.95, turn_sign = sg)
    expect_equal(got$advantages, t(ref), tolerance = 1e-12, info = paste("sign", sg))
  }
})

test_that("bad input is refused", {
  r <- matrix(0, 4, 2)
  expect_error(ag_gae(r, matrix(0, 4, 3), r), "values is \\[4, 3\\]")
  expect_error(ag_gae(r, r, r, truncated = r + 2), "truncated must be logical or 0/1")
  expect_error(ag_gae(r, r, r, truncated = matrix(c(1, 0), 4, 2)), "need next_values")
  expect_error(ag_gae(r, r, r, gamma = 1.5), "gamma")
  expect_error(ag_gae(r, r, r, last_value = 1:3), "last_value")
  expect_error(ag_nstep_targets(r, r, r, n = 0), "positive integer")
})
