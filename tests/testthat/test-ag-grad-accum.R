# Gradient accumulation on the autograd path (ag_*).
#
# Accumulation is not a new mechanism: backward() already adds into each leaf's
# $grad, so running k micro-batches without zero_grad() between them leaves the
# sum there. What these tests pin down is where that sum is and is not visible,
# because the two places a gradient lives do NOT agree:
#
#   p$grad                  -- accumulates across backward() calls
#   backward()'s return env -- holds only the last call's gradients
#
# optimizer$step() reads the second one, so the obvious accumulation loop (skip
# zero_grad, call step every k) silently trains on the last micro-batch instead
# of the sum. That is recorded here as a known gap, not as correct behaviour.

# ============================================================================
# What accumulation does today
# ============================================================================

test_that("backward() accumulates into $grad across calls", {
  set.seed(101)
  W  <- ag_param(matrix(rnorm(6), 2, 3))
  y  <- matrix(0, 2, 1)
  x1 <- matrix(rnorm(3), 3, 1)
  x2 <- matrix(rnorm(3), 3, 1)

  single <- function(x) {
    W$grad <- NULL
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
    backward(l)
    W$grad
  }
  g1 <- single(x1)
  g2 <- single(x2)

  # now the same two, accumulated
  W$grad <- NULL
  with_grad_tape({ la <- ag_mse_loss(ag_matmul(W, ag_tensor(x1)), y) })
  backward(la)
  expect_equal(W$grad, g1, tolerance = 1e-12)

  with_grad_tape({ lb <- ag_mse_loss(ag_matmul(W, ag_tensor(x2)), y) })
  backward(lb)
  expect_equal(W$grad, g1 + g2, tolerance = 1e-12)
})

test_that("zero_grad() clears the accumulator", {
  set.seed(102)
  W   <- ag_param(matrix(rnorm(6), 2, 3))
  opt <- optimizer_sgd(list(W = W), lr = 0.01)
  x   <- matrix(rnorm(3), 3, 1)
  y   <- matrix(0, 2, 1)

  with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
  backward(l)
  expect_false(is.null(W$grad))

  opt$zero_grad()
  expect_null(W$grad)
})

test_that("k micro-batches accumulate to the full-batch gradient", {
  # The property that makes accumulation worth having: summing the per-example
  # gradients of a SUM loss reproduces the gradient of the whole batch. (With
  # ag_mse_loss averaging over columns, the k-fold sum needs dividing by k --
  # which is exactly the averaging the optimizer will have to do.)
  set.seed(103)
  d_in <- 3L; d_out <- 2L; k <- 4L
  W    <- ag_param(matrix(rnorm(d_out * d_in), d_out, d_in))
  X    <- matrix(rnorm(d_in * k), d_in, k)
  Y    <- matrix(0, d_out, k)

  # one full batch
  W$grad <- NULL
  with_grad_tape({ lf <- ag_mse_loss(ag_matmul(W, ag_tensor(X)), Y) })
  backward(lf)
  g_full <- W$grad

  # k single-column micro-batches, accumulated then averaged
  W$grad <- NULL
  for (i in seq_len(k)) {
    with_grad_tape({
      li <- ag_mse_loss(ag_matmul(W, ag_tensor(X[, i, drop = FALSE])),
                        Y[, i, drop = FALSE])
    })
    backward(li)
  }
  expect_equal(W$grad / k, g_full, tolerance = 1e-10)
})

# ============================================================================
# The gap: $grad and the backward() return value disagree
# ============================================================================

test_that("backward() returns only the last call's gradients, not the sum", {
  # Documents the trap. If this ever starts failing because the return value
  # began accumulating too, the optimizer's own accumulation must be revisited
  # -- it would then be counting the same gradients twice.
  set.seed(104)
  W  <- ag_param(matrix(rnorm(6), 2, 3))
  y  <- matrix(0, 2, 1)
  x1 <- matrix(rnorm(3), 3, 1)
  x2 <- matrix(rnorm(3), 3, 1)

  with_grad_tape({ l1 <- ag_mse_loss(ag_matmul(W, ag_tensor(x1)), y) })
  backward(l1)
  with_grad_tape({ l2 <- ag_mse_loss(ag_matmul(W, ag_tensor(x2)), y) })
  g_ret <- backward(l2)

  from_env <- get0(as.character(W$id), envir = g_ret)
  expect_false(isTRUE(all.equal(W$grad, from_env)))
})

test_that("optimizer$step(grads) uses the passed env, so it misses the sum", {
  # The naive accumulation loop -- no zero_grad, step every k -- updates by the
  # LAST micro-batch only. Same weights, same data, two ways of stepping:
  # accumulate-then-step must not equal step-on-last, yet today it does.
  set.seed(105)
  x1 <- matrix(rnorm(3), 3, 1)
  x2 <- matrix(rnorm(3), 3, 1)
  y  <- matrix(0, 2, 1)
  W0 <- matrix(rnorm(6), 2, 3)

  run <- function(use_sum) {
    W   <- ag_param(W0)
    opt <- optimizer_sgd(list(W = W), lr = 0.1)
    with_grad_tape({ l1 <- ag_mse_loss(ag_matmul(W, ag_tensor(x1)), y) })
    backward(l1)
    with_grad_tape({ l2 <- ag_mse_loss(ag_matmul(W, ag_tensor(x2)), y) })
    g2 <- backward(l2)

    if (use_sum) {
      # what accumulation SHOULD apply: the mean of the two micro-batches
      env <- new.env(parent = emptyenv())
      assign(as.character(W$id), W$grad / 2, envir = env)
      opt$step(env)
    } else {
      opt$step(g2)      # what the naive loop actually applies
    }
    ggmlR:::.ag_data(W)
  }

  expect_false(isTRUE(all.equal(run(TRUE), run(FALSE), tolerance = 1e-8)))
})

# ============================================================================
# accumulate_steps: the optimizer does the bookkeeping
# ============================================================================

test_that("step() with no argument reads the accumulated $grad", {
  set.seed(106)
  x  <- matrix(rnorm(3), 3, 1)
  y  <- matrix(0, 2, 1)
  W0 <- matrix(rnorm(6), 2, 3)

  W   <- ag_param(W0)
  opt <- optimizer_sgd(list(W = W), lr = 0.1)
  with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
  g <- backward(l)
  opt$step()                       # no argument
  from_grad <- ggmlR:::.ag_data(W)

  W2   <- ag_param(W0)
  opt2 <- optimizer_sgd(list(W = W2), lr = 0.1)
  with_grad_tape({ l2 <- ag_mse_loss(ag_matmul(W2, ag_tensor(x)), y) })
  g2 <- backward(l2)
  opt2$step(g2)                    # explicit env, the old spelling
  from_env <- ggmlR:::.ag_data(W2)

  # one backward, so both sources hold the same gradient
  expect_equal(from_grad, from_env, tolerance = 1e-12)
})

test_that("accumulate_steps delays the update until the k-th step()", {
  set.seed(107)
  W   <- ag_param(matrix(rnorm(6), 2, 3))
  opt <- optimizer_sgd(list(W = W), lr = 0.1, accumulate_steps = 3L)
  x   <- matrix(rnorm(3), 3, 1)
  y   <- matrix(0, 2, 1)
  before <- ggmlR:::.ag_data(W)

  for (i in 1:2) {
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
    backward(l)
    expect_false(opt$step())                       # reports "not yet"
    expect_equal(ggmlR:::.ag_data(W), before)      # and changes nothing
  }

  with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
  backward(l)
  expect_true(opt$step())
  expect_false(isTRUE(all.equal(ggmlR:::.ag_data(W), before)))
})

test_that("k accumulated micro-batches equal one full batch of size k", {
  # The property the feature exists for. Same data, same lr, one update either
  # way -- the weights must land in the same place.
  set.seed(108)
  d_in <- 3L; d_out <- 2L; k <- 4L
  W0 <- matrix(rnorm(d_out * d_in), d_out, d_in)
  X  <- matrix(rnorm(d_in * k), d_in, k)
  Y  <- matrix(0, d_out, k)

  W_full <- ag_param(W0)
  o_full <- optimizer_sgd(list(W = W_full), lr = 0.1)
  with_grad_tape({ lf <- ag_mse_loss(ag_matmul(W_full, ag_tensor(X)), Y) })
  backward(lf)
  o_full$step()

  W_acc <- ag_param(W0)
  o_acc <- optimizer_sgd(list(W = W_acc), lr = 0.1, accumulate_steps = k)
  for (i in seq_len(k)) {
    with_grad_tape({
      li <- ag_mse_loss(ag_matmul(W_acc, ag_tensor(X[, i, drop = FALSE])),
                        Y[, i, drop = FALSE])
    })
    backward(li)
    o_acc$step()
  }

  expect_equal(ggmlR:::.ag_data(W_acc), ggmlR:::.ag_data(W_full),
               tolerance = 1e-10)
})

test_that("average = FALSE applies the plain sum", {
  set.seed(109)
  W0 <- matrix(rnorm(6), 2, 3)
  x  <- matrix(rnorm(3), 3, 1)
  y  <- matrix(0, 2, 1)
  k  <- 2L

  run <- function(avg) {
    W   <- ag_param(W0)
    opt <- optimizer_sgd(list(W = W), lr = 0.1,
                         accumulate_steps = k, average = avg)
    for (i in seq_len(k)) {
      with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
      backward(l)
      opt$step()
    }
    ggmlR:::.ag_data(W)
  }

  # identical micro-batches: the summed update moves exactly k times as far
  W_ref <- ag_param(W0)
  with_grad_tape({ l <- ag_mse_loss(ag_matmul(W_ref, ag_tensor(x)), y) })
  backward(l)
  g_one <- W_ref$grad

  expect_equal(run(TRUE),  W0 - 0.1 * g_one,     tolerance = 1e-10)
  expect_equal(run(FALSE), W0 - 0.1 * k * g_one, tolerance = 1e-10)
})

test_that("Adam advances its bias-correction counter once per real update", {
  # t must count updates, not step() calls: counting the skipped ones would
  # make the correction think k times as many steps had happened.
  set.seed(110)
  W   <- ag_param(matrix(rnorm(6), 2, 3))
  opt <- optimizer_adam(list(W = W), lr = 1e-2, accumulate_steps = 3L)
  x   <- matrix(rnorm(3), 3, 1)
  y   <- matrix(0, 2, 1)

  for (i in 1:3) {
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
    backward(l)
    opt$step()
  }
  expect_equal(opt$t, 1L)

  for (i in 1:3) {
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
    backward(l)
    opt$step()
  }
  expect_equal(opt$t, 2L)
})

test_that("accumulate_steps = 1 is unchanged and step(grads) still works", {
  # Backwards compatibility: dp_train() and the mlr3 learners pass an env.
  set.seed(111)
  W0 <- matrix(rnorm(6), 2, 3)
  x  <- matrix(rnorm(3), 3, 1)
  y  <- matrix(0, 2, 1)

  W   <- ag_param(W0)
  opt <- optimizer_adam(list(W = W), lr = 1e-2)
  with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, ag_tensor(x)), y) })
  g <- backward(l)
  expect_true(opt$step(g))
  expect_equal(opt$t, 1L)
  expect_false(isTRUE(all.equal(ggmlR:::.ag_data(W), W0)))
})

test_that("accumulate_steps rejects a non-positive value", {
  w <- ag_param(matrix(1, 2, 2))
  expect_error(optimizer_sgd(list(w = w), accumulate_steps = 0L),
               "positive integer")
  expect_error(optimizer_adam(list(w = w), accumulate_steps = -2L),
               "positive integer")
})
