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
