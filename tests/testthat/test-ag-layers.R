# Tests for ag_sequential, ag_dropout, ag_batch_norm, ag_embedding

src <- function() {
  if (!exists("ag_tensor", mode = "function")) {
    source("R/autograd.R")
    source("R/ag_layers.R")
  }
}
src()

# ============================================================================
# ag_sequential
# ============================================================================

test_that("ag_sequential: forward shape correct", {
  set.seed(1)
  model <- ag_sequential(
    ag_linear(4L, 8L, activation = "relu"),
    ag_linear(8L, 3L, activation = "softmax")
  )
  x   <- ag_tensor(matrix(runif(4 * 16), 4, 16))
  out <- model$forward(x)
  expect_equal(dim(out$data), c(3L, 16L))
})

test_that("ag_sequential: parameters() collects all params", {
  model <- ag_sequential(
    ag_linear(4L, 8L),
    ag_linear(8L, 2L)
  )
  params <- model$parameters()
  # 2 layers × 2 params (W, b) = 4
  expect_equal(length(params), 4L)
  expect_true(all(sapply(params, function(p) p$requires_grad)))
})

test_that("ag_sequential: train/eval mode propagates to sub-layers", {
  model <- ag_sequential(
    ag_linear(4L, 8L),
    ag_dropout(0.5),
    ag_linear(8L, 2L)
  )
  ag_eval(model)
  expect_false(model$training)
  expect_false(model$layers[[2L]]$training)  # dropout

  ag_train(model)
  expect_true(model$training)
  expect_true(model$layers[[2L]]$training)
})

test_that("ag_sequential: backward reduces loss", {
  set.seed(2)
  model  <- ag_sequential(
    ag_linear(4L, 8L, activation = "relu"),
    ag_linear(8L, 2L)
  )
  params <- model$parameters()
  opt    <- optimizer_adam(params, lr = 0.01)

  n     <- 32L
  x_mat <- matrix(runif(4 * n), 4, n)
  y_mat <- matrix(0, 2, n)
  y_mat[1L, ] <- 1  # all class 0

  losses <- numeric(10L)
  for (i in seq_len(10L)) {
    with_grad_tape({
      out  <- model$forward(ag_tensor(x_mat))
      loss <- ag_mse_loss(out, y_mat)
    })
    grads <- backward(loss)
    opt$step(grads)
    opt$zero_grad()
    losses[i] <- loss$data[1L]
  }
  expect_lt(mean(losses[8:10]), mean(losses[1:3]))
})

# ============================================================================
# ag_dropout
# ============================================================================

test_that("ag_dropout: eval mode is identity", {
  set.seed(10)
  drop <- ag_dropout(0.5)
  ag_eval(drop)
  x   <- ag_tensor(matrix(c(1, 2, 3, 4), 2, 2))
  out <- drop$forward(x)
  expect_equal(out$data, x$data)
})

test_that("ag_dropout: train mode changes some values", {
  set.seed(11)
  drop <- ag_dropout(0.5)  # training=TRUE by default
  x    <- ag_tensor(matrix(rep(1, 100), 10, 10))
  out  <- drop$forward(x)
  # Some values should be 0, some 2 (inverted dropout scale = 1/(1-0.5)=2)
  vals <- as.numeric(out$data)
  expect_true(any(vals == 0))
  expect_true(any(abs(vals - 2) < 1e-9))
})

test_that("ag_dropout: rate=0 is always identity", {
  drop <- ag_dropout(0.0)
  x    <- ag_tensor(matrix(1:4, 2, 2))
  expect_equal(drop$forward(x)$data, x$data)
})

test_that("ag_dropout: gradcheck passes in train mode", {
  set.seed(12)
  W    <- ag_param(matrix(runif(8, -1, 1), 2, 4))
  x    <- ag_tensor(matrix(runif(4, -1, 1), 4, 1))
  drop <- ag_dropout(0.0)   # rate=0 -> deterministic identity, safe for gradcheck
  ok   <- ag_gradcheck(
    fn = function(ins) ag_mse_loss(drop$forward(ag_matmul(ins$W, x)),
                                    matrix(0, 2, 1)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ============================================================================
# ag_batch_norm
# ============================================================================

test_that("ag_batch_norm: output has approx zero mean and unit var (train)", {
  set.seed(20)
  bn  <- ag_batch_norm(4L)
  x   <- ag_tensor(matrix(rnorm(4 * 64, mean = 5, sd = 3), 4, 64))
  out <- bn$forward(x)
  mu  <- rowMeans(out$data)
  v   <- rowMeans((out$data - mu)^2)
  # After BN (gamma=1, beta=0): mean ~ 0, var ~ 1
  expect_lt(max(abs(mu)), 1e-5)
  expect_equal(v, rep(1, 4), tolerance = 0.05)
})

test_that("ag_batch_norm: running stats update in train mode", {
  set.seed(21)
  bn <- ag_batch_norm(2L)
  expect_equal(as.numeric(bn$running_mean), c(0, 0))

  x <- ag_tensor(matrix(c(10, 10, 10, 10, 20, 20, 20, 20), 2, 4))
  bn$forward(x)
  # running_mean should have moved toward [10, 20]
  expect_gt(bn$running_mean[1L], 0)
  expect_gt(bn$running_mean[2L], 0)
})

test_that("ag_batch_norm: eval mode uses running stats", {
  set.seed(22)
  bn <- ag_batch_norm(2L)

  # Train for several batches to build up running stats
  for (i in 1:20) {
    x <- ag_tensor(matrix(rnorm(2 * 32, mean = c(5, -5), sd = 1), 2, 32))
    bn$forward(x)
  }
  running_mean_saved <- bn$running_mean

  # Switch to eval and verify output uses running stats (same input -> same output)
  ag_eval(bn)
  x1 <- ag_tensor(matrix(c(5, -5), 2, 1))
  x2 <- ag_tensor(matrix(c(5, -5), 2, 1))
  out1 <- bn$forward(x1)
  out2 <- bn$forward(x2)
  expect_equal(out1$data, out2$data)
  # running stats should NOT change in eval mode
  expect_equal(bn$running_mean, running_mean_saved)
})

test_that("ag_batch_norm: gradcheck passes (gamma/beta)", {
  set.seed(23)
  bn  <- ag_batch_norm(3L)
  x_d <- matrix(rnorm(3 * 8), 3, 8)

  # gradcheck by swapping bn$gamma with the checked param tensor
  ok_gamma <- ag_gradcheck(
    fn = function(ins) {
      orig_g    <- bn$gamma
      bn$gamma  <- ins$gamma
      out       <- bn$forward(ag_tensor(x_d))
      bn$gamma  <- orig_g
      ag_sum(out)
    },
    inputs = list(gamma = bn$gamma), atol = 1e-3, quiet = TRUE
  )
  expect_true(ok_gamma)
})

test_that("ag_batch_norm: gradcheck passes for x in train mode", {
  # The train-mode gradient must account for mu and var depending on x.
  # Treating them as constants (grad_out / std) passes the gamma/beta check
  # above but is wrong here -- this is the regression guard.
  set.seed(24)
  bn <- ag_batch_norm(3L)
  ag_train(bn)
  x  <- ag_param(matrix(rnorm(3 * 6), 3, 6))

  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_mul(bn$forward(ins$x), ag_tensor(
      matrix(seq(0.3, by = 0.17, length.out = 18), 3, 6)))),
    inputs = list(x = x), atol = 1e-3, quiet = TRUE
  )
  expect_true(ok)
})

test_that("ag_batch_norm: eval mode keeps the running-stats gradient", {
  # In eval mode mu/sigma come from the running statistics and are genuine
  # constants w.r.t. x, so dx = grad_out * gamma / std exactly.  The train-mode
  # correction must NOT be applied here.
  set.seed(25)
  bn <- ag_batch_norm(3L)
  bn$running_mean <- matrix(c(0.5, -0.2, 1.3), 3, 1)
  bn$running_var  <- matrix(c(2.0,  0.5, 4.0), 3, 1)
  ag_eval(bn)
  x <- ag_param(matrix(rnorm(3 * 4), 3, 4))

  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_mul(bn$forward(ins$x), ag_tensor(
      matrix(seq(0.4, by = 0.11, length.out = 12), 3, 4)))),
    inputs = list(x = x), atol = 1e-3, quiet = TRUE
  )
  expect_true(ok)

  # and the analytical value is the closed form grad_out * gamma / std
  w <- matrix(seq(0.4, by = 0.11, length.out = 12), 3, 4)
  with_grad_tape({
    loss <- ag_sum(ag_mul(bn$forward(x), ag_tensor(w)))
  })
  g   <- backward(loss)
  std <- sqrt(as.numeric(bn$running_var) + bn$eps)
  expect_equal(get0(as.character(x$id), envir = g), w / std,
               tolerance = 1e-6)
})

test_that("ag_batch_norm: train gradient is not the naive grad_out/std", {
  # Mirror of the ag_layer_norm guard: with a constant grad_out and gamma=1 the
  # exact batch gradient cancels to (almost) zero, while the 'treat mu/sigma as
  # constants' shortcut would leave 1/std behind.
  set.seed(26)
  bn <- ag_batch_norm(4L)
  ag_train(bn)
  x  <- ag_param(matrix(rnorm(4 * 5), 4, 5))

  with_grad_tape({ loss <- ag_sum(bn$forward(x)) })
  g_x <- get0(as.character(x$id), envir = backward(loss))

  expect_true(max(abs(g_x)) < 1e-8)
})

test_that("ag_batch_norm: gradcheck passes for beta", {
  # .ag_add_broadcast_col was only ever exercised indirectly; beta itself had
  # no finite-difference coverage.
  set.seed(27)
  bn  <- ag_batch_norm(3L)
  ag_train(bn)
  x_d <- matrix(rnorm(3 * 6), 3, 6)
  w   <- matrix(seq(0.2, by = 0.13, length.out = 18), 3, 6)

  ok_beta <- ag_gradcheck(
    fn = function(ins) {
      orig_b  <- bn$beta
      bn$beta <- ins$beta
      out     <- bn$forward(ag_tensor(x_d))
      bn$beta <- orig_b
      ag_sum(ag_mul(out, ag_tensor(w)))
    },
    inputs = list(beta = bn$beta), atol = 1e-3, quiet = TRUE
  )
  expect_true(ok_beta)
})

test_that("ag_batch_norm: gradient is finite for N=1 and for near-zero variance", {
  # N=1 -> var is exactly 0, everything rides on eps.
  bn <- ag_batch_norm(3L)
  ag_train(bn)
  x1 <- ag_param(matrix(c(1.0, -2.0, 0.5), 3, 1))
  with_grad_tape({ loss1 <- ag_sum(bn$forward(x1)) })
  g1 <- backward(loss1)
  expect_true(all(is.finite(get0(as.character(x1$id), envir = g1))))

  # N=2 with nearly equal values -> var ~ eps, the numerically nastiest zone.
  bn2 <- ag_batch_norm(2L)
  ag_train(bn2)
  x2 <- ag_param(matrix(c(1.0, 1.0 + 1e-6, -3.0, -3.0 - 1e-6), 2, 2,
                        byrow = TRUE))
  with_grad_tape({ loss2 <- ag_sum(bn2$forward(x2)) })
  g2 <- backward(loss2)
  expect_true(all(is.finite(get0(as.character(x2$id), envir = g2))))
})

# ============================================================================
# ag_embedding
# ============================================================================

test_that("ag_embedding: output shape correct", {
  emb <- ag_embedding(10L, 4L)
  idx <- matrix(c(0L, 3L, 7L, 2L), 2L, 2L)
  out <- emb$forward(idx)
  expect_equal(dim(out$data), c(4L, 4L))  # [dim, seq_len*batch]
})

test_that("ag_embedding: same index -> same output", {
  emb <- ag_embedding(10L, 4L)
  idx <- matrix(c(0L, 0L), 1L, 2L)
  out <- emb$forward(idx)
  expect_equal(out$data[, 1L], out$data[, 2L])
})

test_that("ag_embedding: gradcheck passes", {
  set.seed(30)
  emb <- ag_embedding(5L, 3L)
  idx <- matrix(c(0L, 2L, 4L), 1L, 3L)

  # gradcheck: point ins$weight directly at emb$weight so forward() sees updates
  ok <- ag_gradcheck(
    fn = function(ins) {
      # temporarily swap emb$weight to the checked tensor
      orig  <- emb$weight
      emb$weight <- ins$weight
      out   <- emb$forward(idx)
      emb$weight <- orig
      ag_sum(out)
    },
    inputs = list(weight = emb$weight), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("ag_embedding: optimizer updates weight", {
  set.seed(31)
  emb  <- ag_embedding(5L, 3L)
  opt  <- optimizer_adam(emb$parameters(), lr = 0.1)
  idx  <- matrix(c(0L, 1L, 2L), 1L, 3L)
  W0   <- emb$weight$data

  with_grad_tape({
    out  <- emb$forward(idx)
    loss <- ag_sum(ag_pow(out, 2))   # push embeddings toward 0
  })
  grads <- backward(loss)
  opt$step(grads)

  expect_false(identical(emb$weight$data, W0))
})

# ============================================================================
# Integration: sequential with dropout + batch_norm
# ============================================================================

test_that("integration: sequential model with BN and dropout trains", {
  set.seed(40)
  model <- ag_sequential(
    ag_linear(4L, 16L, activation = "relu"),
    ag_batch_norm(16L),
    ag_dropout(0.2),
    ag_linear(16L, 2L)
  )
  params <- model$parameters()
  opt    <- optimizer_adam(params, lr = 0.01)

  n     <- 64L
  x_mat <- matrix(runif(4 * n), 4, n)
  y_mat <- rbind(rep(1, n), rep(0, n))   # all class 0

  losses <- numeric(15L)
  for (i in seq_len(15L)) {
    with_grad_tape({
      out  <- model$forward(ag_tensor(x_mat))
      loss <- ag_mse_loss(out, y_mat)
    })
    grads <- backward(loss)
    opt$step(grads)
    opt$zero_grad()
    losses[i] <- loss$data[1L]
  }
  expect_lt(mean(losses[13:15]), mean(losses[1:3]))
})
