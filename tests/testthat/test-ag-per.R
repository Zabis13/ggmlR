# Prioritized replay sampler (R/ag_per.R, src/r_per_tree.c).

test_that("draws are proportional to p^alpha (stratified, so nearly exact)", {
  s <- ag_per_sampler(7L, alpha = 1)             # capacity not a power of two
  s$add(0:6)
  p <- c(1, 2, 3, 4, 5, 6, 7)
  s$update(0:6, p - s$eps)
  set.seed(1)
  b <- s$sample(70000L, beta = 0)
  freq <- tabulate(b$idx + 1L, 7L) / 70000
  expect_equal(freq, p / sum(p), tolerance = 1e-3)
  expect_equal(b$weights, rep(1, 70000L))         # beta = 0: no correction
  # alpha = 0.5
  s$set_alpha(0.5)
  b <- s$sample(70000L, beta = 0)
  expect_equal(tabulate(b$idx + 1L, 7L) / 70000, sqrt(p) / sum(sqrt(p)), tolerance = 1e-3)
})

test_that("alpha = 0 is uniform; equal priorities give one draw per segment", {
  s <- ag_per_sampler(4L, alpha = 0)
  s$add(0:3); s$update(0:3, c(0.1, 5, 2, 9))
  set.seed(2)
  b <- s$sample(4L, beta = 1)
  expect_identical(sort(b$idx), 0:3)             # stratified: each segment once
  expect_equal(b$weights, rep(1, 4))
})

test_that("importance weights: (N P)^-beta over the max over the whole sampler", {
  s <- ag_per_sampler(10L, alpha = 0.6, eps = 1e-6)
  s$add(0:5)                                     # 6 of 10 slots used
  td <- c(0.5, 2, 0.1, 3, 1, 0.05)
  s$update(0:5, td)
  pa <- (abs(td) + 1e-6)^0.6
  P <- pa / sum(pa); N <- 6
  wmax <- (N * min(P))^-0.4
  set.seed(3)
  b <- s$sample(50L, beta = 0.4)
  expect_true(all(b$idx %in% 0:5))
  expect_equal(b$weights, (N * P[b$idx + 1L])^-0.4 / wmax, tolerance = 1e-12)
  expect_true(all(b$weights <= 1 + 1e-12))
})

test_that("new slots get the largest priority seen; unused slots are never drawn", {
  s <- ag_per_sampler(8L)
  s$add(0:1)
  expect_equal(s$priorities()[1:2], c(1, 1))     # 1 at the start
  s$update(0L, 5)
  s$add(2L)
  expect_equal(s$priorities()[3], 5 + s$eps)
  expect_true(all(is.na(s$priorities()[4:8])))
  expect_identical(s$size(), 3L)
  set.seed(4)
  expect_true(all(s$sample(1000L)$idx %in% 0:2))
})

test_that("set_alpha equals a fresh sampler built with that alpha", {
  td <- c(0.3, 1.2, 0.7, 2.5, 0.01)
  a <- ag_per_sampler(5L, alpha = 0.6); a$add(0:4); a$update(0:4, td)
  a$set_alpha(0.9)
  b <- ag_per_sampler(5L, alpha = 0.9); b$add(0:4); b$update(0:4, td)
  set.seed(5); ra <- a$sample(64L, 0.5)
  set.seed(5); rb <- b$sample(64L, 0.5)
  expect_identical(ra$idx, rb$idx)
  expect_equal(ra$weights, rb$weights)
})

test_that("draws are reproducible with set.seed; priorities round trip", {
  s <- ag_per_sampler(16L); s$add(0:15)
  s$update(0:15, seq(0.1, 1.6, by = 0.1))
  set.seed(6); x <- s$sample(32L)
  set.seed(6); expect_identical(s$sample(32L), x)
  s2 <- ag_per_sampler(16L); s2$load_priorities(s$priorities())
  expect_equal(s2$priorities(), s$priorities())
  set.seed(6); expect_identical(s2$sample(32L)$idx, x$idx)
  s2$add(0L)                                     # max carried over by the load
  expect_equal(s2$priorities()[1], max(s$priorities()))
})

test_that("a capacity of 1 and large capacities work", {
  s <- ag_per_sampler(1L); s$add(0L)
  expect_identical(s$sample(3L)$idx, c(0L, 0L, 0L))
  big <- ag_per_sampler(100000L, alpha = 1); big$add(0:99999)
  big$update(99999L, 1e6)                        # one dominant slot
  set.seed(7)
  expect_gt(mean(big$sample(1000L)$idx == 99999L), 0.85)
})

test_that("bad input is refused", {
  expect_error(ag_per_sampler(0L), "capacity")
  expect_error(ag_per_sampler(4L, alpha = -1), "alpha")
  expect_error(ag_per_sampler(4L, eps = 0), "eps")
  s <- ag_per_sampler(4L)
  expect_error(s$sample(2L), "empty")
  expect_error(s$add(4L), "0 .. capacity - 1")
  s$add(0L)
  expect_error(s$update(0L, NA), "finite")
  expect_error(s$update(0:1, 1), "one per slot")
  expect_error(s$sample(2L, beta = -1), "beta")
  expect_error(s$load_priorities(c(1, 0, NA, NA)), "positive")
})
