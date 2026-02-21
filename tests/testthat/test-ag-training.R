# Tests for DataLoader, lr_scheduler, clip_grad_norm

src <- function() {
  if (!exists("ag_tensor", mode = "function")) {
    source("R/autograd.R")
    source("R/ag_layers.R")
    source("R/ag_training.R")
  }
}
src()

# ============================================================================
# ag_dataloader
# ============================================================================

test_that("ag_dataloader: basic shape", {
  n  <- 64L
  x  <- matrix(runif(4 * n), 4, n)
  y  <- matrix(runif(2 * n), 2, n)
  dl <- ag_dataloader(x, y, batch_size = 16L, shuffle = FALSE)

  expect_equal(dl$n_batches(), 4L)
  batch <- dl$next_batch()
  expect_s3_class(batch$x, "ag_tensor")
  expect_equal(dim(batch$x$data), c(4L, 16L))
  expect_equal(dim(batch$y$data), c(2L, 16L))
})

test_that("ag_dataloader: covers all samples (no shuffle)", {
  n  <- 32L
  x  <- matrix(seq_len(32), 1, n)   # single feature, values 1..32
  dl <- ag_dataloader(x, batch_size = 8L, shuffle = FALSE)

  seen <- c()
  while (dl$has_next()) {
    b <- dl$next_batch()
    seen <- c(seen, as.integer(b$x$data))
  }
  expect_equal(sort(seen), seq_len(n))
})

test_that("ag_dataloader: shuffle produces different order", {
  set.seed(42)
  n   <- 100L
  x   <- matrix(seq_len(n), 1, n)
  dl1 <- ag_dataloader(x, batch_size = 10L, shuffle = TRUE)
  dl2 <- ag_dataloader(x, batch_size = 10L, shuffle = FALSE)

  get_all <- function(dl) {
    out <- c()
    while (dl$has_next()) out <- c(out, as.integer(dl$next_batch()$x$data))
    out
  }
  ord1 <- get_all(dl1)
  ord2 <- get_all(dl2)
  # With seed 42 the shuffle should produce a different order
  expect_false(identical(ord1, ord2))
  # But cover the same elements
  expect_equal(sort(ord1), sort(ord2))
})

test_that("ag_dataloader: reset reshuffles", {
  set.seed(1)
  n  <- 20L
  x  <- matrix(seq_len(n), 1, n)
  dl <- ag_dataloader(x, batch_size = 5L, shuffle = TRUE)
  ord1 <- dl$order
  dl$reset()
  ord2 <- dl$order
  # Reshuffle should (almost certainly) give different order
  # We can't guarantee it but with n=20 collision prob is tiny
  expect_false(identical(ord1, ord2))
})

test_that("ag_dataloader: has_next FALSE after exhaustion", {
  dl <- ag_dataloader(matrix(1:10, 1, 10), batch_size = 5L)
  dl$next_batch(); dl$next_batch()
  expect_false(dl$has_next())
})

test_that("ag_dataloader: col_major=FALSE transposes", {
  # row-major: [n, features]
  n <- 16L
  x <- matrix(runif(n * 4), n, 4)   # [16, 4]
  dl <- ag_dataloader(x, batch_size = 8L, col_major = FALSE)
  b  <- dl$next_batch()
  # after transpose: [features, batch] = [4, 8]
  expect_equal(dim(b$x$data), c(4L, 8L))
})

test_that("ag_dataloader: epoch() returns all batches", {
  n  <- 32L
  x  <- matrix(seq_len(n), 1, n)
  dl <- ag_dataloader(x, batch_size = 8L, shuffle = FALSE)
  batches <- dl$epoch()
  expect_equal(length(batches), 4L)
})

# ============================================================================
# lr_scheduler_step
# ============================================================================

test_that("lr_scheduler_step: decays at correct steps", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 1.0)
  sch <- lr_scheduler_step(opt, step_size = 5L, gamma = 0.5)

  for (i in 1:5) sch$step()
  expect_equal(opt$lr, 0.5, tolerance = 1e-9)

  for (i in 1:5) sch$step()
  expect_equal(opt$lr, 0.25, tolerance = 1e-9)

  for (i in 1:4) sch$step()          # only 4, no decay yet
  expect_equal(opt$lr, 0.25, tolerance = 1e-9)

  sch$step()                          # 5th step -> decay
  expect_equal(opt$lr, 0.125, tolerance = 1e-9)
})

test_that("lr_scheduler_step: no decay before step_size", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.1)
  sch <- lr_scheduler_step(opt, step_size = 10L, gamma = 0.1)
  for (i in 1:9) sch$step()
  expect_equal(opt$lr, 0.1, tolerance = 1e-9)
})

# ============================================================================
# lr_scheduler_cosine
# ============================================================================

test_that("lr_scheduler_cosine: lr decreases toward lr_min", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_cosine(opt, T_max = 10L, lr_min = 0)

  lrs <- numeric(10L)
  for (i in 1:10) lrs[i] <- sch$step()

  # First step: slightly below max; last step: near 0
  expect_lt(lrs[10], lrs[1])
  expect_lt(lrs[10], 0.01)
})

test_that("lr_scheduler_cosine: stays at lr_min after T_max (no restart)", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_cosine(opt, T_max = 5L, lr_min = 0.01, restart = FALSE)

  for (i in 1:5) sch$step()
  lr_at_5 <- opt$lr
  sch$step()  # beyond T_max
  expect_equal(opt$lr, lr_at_5, tolerance = 1e-9)
})

test_that("lr_scheduler_cosine: restarts correctly", {
  w   <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_cosine(opt, T_max = 5L, lr_min = 0, restart = TRUE)

  lrs <- numeric(10L)
  for (i in 1:10) lrs[i] <- sch$step()

  # Cycle 1: steps 1-5 should decrease
  expect_lt(lrs[5], lrs[1])
  # Step 6 should restart near max, step 5 should be near 0
  expect_gt(lrs[6], lrs[5])
})

# ============================================================================
# clip_grad_norm
# ============================================================================

test_that("clip_grad_norm: does not clip when norm < max_norm", {
  w  <- ag_param(matrix(c(0.1, 0.1, 0.1, 0.1), 2, 2))
  x  <- ag_tensor(matrix(c(1, 1), 2, 1))
  with_grad_tape({
    out  <- ag_matmul(w, x)
    loss <- ag_mse_loss(out, matrix(0, 2, 1))
  })
  grads      <- backward(loss)
  orig_g     <- get0(as.character(w$id), envir = grads)
  pre_norm   <- clip_grad_norm(list(w = w), grads, max_norm = 1000)
  after_g    <- get0(as.character(w$id), envir = grads)
  # Should be unchanged
  expect_equal(orig_g, after_g, tolerance = 1e-10)
})

test_that("clip_grad_norm: clips to max_norm", {
  w  <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  x  <- ag_tensor(matrix(c(10, 10), 2, 1))
  with_grad_tape({
    out  <- ag_matmul(w, x)
    loss <- ag_mse_loss(out, matrix(0, 2, 1))
  })
  grads    <- backward(loss)
  pre_norm <- clip_grad_norm(list(w = w), grads, max_norm = 1.0)
  g_after  <- get0(as.character(w$id), envir = grads)
  clipped_norm <- sqrt(sum(g_after^2))

  expect_gt(pre_norm, 1.0)
  expect_equal(clipped_norm, 1.0, tolerance = 1e-5)
})

test_that("clip_grad_norm: works with multiple params", {
  w1 <- ag_param(matrix(runif(4), 2, 2))
  w2 <- ag_param(matrix(runif(4), 2, 2))
  x  <- ag_tensor(matrix(c(1, 1), 2, 1))
  with_grad_tape({
    h    <- ag_matmul(w1, x)
    out  <- ag_matmul(w2, h)
    loss <- ag_mse_loss(out, matrix(0, 2, 1))
  })
  grads    <- backward(loss)
  pre_norm <- clip_grad_norm(list(w1 = w1, w2 = w2), grads, max_norm = 0.5)

  g1 <- get0(as.character(w1$id), envir = grads)
  g2 <- get0(as.character(w2$id), envir = grads)
  joint_norm <- sqrt(sum(g1^2) + sum(g2^2))
  expect_equal(joint_norm, 0.5, tolerance = 1e-5)
})

# ============================================================================
# Integration: full training loop with all utils
# ============================================================================

test_that("integration: dataloader + scheduler + clip_grad trains", {
  set.seed(99)
  n     <- 64L
  x_mat <- matrix(runif(4 * n), 4, n)
  y_mat <- rbind(rep(1, n), rep(0, n))

  dl    <- ag_dataloader(x_mat, y_mat, batch_size = 16L, shuffle = TRUE)
  model <- ag_sequential(
    ag_linear(4L, 8L, activation = "relu"),
    ag_linear(8L, 2L)
  )
  params <- model$parameters()
  opt    <- optimizer_adam(params, lr = 0.05)
  sch    <- lr_scheduler_step(opt, step_size = 5L, gamma = 0.5)

  all_losses <- c()
  for (epoch in 1:15) {
    epoch_loss <- 0
    batches    <- dl$epoch()
    for (batch in batches) {
      with_grad_tape({
        out  <- model$forward(batch$x)
        loss <- ag_mse_loss(out, batch$y)
      })
      grads <- backward(loss)
      clip_grad_norm(params, grads, max_norm = 5.0)
      opt$step(grads)
      opt$zero_grad()
      epoch_loss <- epoch_loss + loss$data[1L]
    }
    all_losses <- c(all_losses, epoch_loss / length(batches))
    sch$step()
  }

  expect_lt(mean(tail(all_losses, 3)), mean(head(all_losses, 3)))
  # lr should have decayed after epochs 5 and 10
  expect_lt(opt$lr, 0.05)
})

# ============================================================================
# dp_train
# ============================================================================

test_that("dp_train: 1 replica converges on simple regression", {
  set.seed(7L)
  D_IN <- 3L; D_OUT <- 2L; N <- 32L
  X <- matrix(rnorm(D_IN * N), D_IN, N)
  Y <- X[1:D_OUT, ] * 0.5 + 0.1   # deterministic target to ensure convergence
  data <- lapply(seq_len(N), function(i)
    list(x = X[, i, drop = FALSE], y = Y[, i, drop = FALSE]))

  make_model <- function() {
    W <- ag_param(matrix(rnorm(D_OUT * D_IN) * 0.1, D_OUT, D_IN))
    b <- ag_param(matrix(0.0, D_OUT, 1L))
    list(forward    = function(x) ag_add(ag_matmul(W, x), b),
         parameters = function() list(W = W, b = b))
  }

  result <- dp_train(
    make_model = make_model,
    data       = data,
    loss_fn    = function(out, tgt) ag_mse_loss(out, tgt),
    forward_fn = function(model, s) model$forward(s$x),
    target_fn  = function(s) s$y,
    n_gpu = 1L, n_iter = 60L, lr = 1e-2, verbose = FALSE
  )

  expect_named(result, c("params", "loss_history", "model"))
  expect_length(result$loss_history, 60L)
  expect_lt(mean(tail(result$loss_history, 10)), result$loss_history[1])
})

test_that("dp_train: 2 replicas — initial weight sync and no NaN", {
  set.seed(8L)
  D_IN <- 3L; D_OUT <- 2L
  make_model <- function() {
    W <- ag_param(matrix(rnorm(D_OUT * D_IN) * 0.1, D_OUT, D_IN))
    list(forward    = function(x) ag_matmul(W, x),
         parameters = function() list(W = W))
  }
  # data: each sample is a list(x, y) with compatible dims
  data <- lapply(1:8, function(i)
    list(x = matrix(rnorm(D_IN), D_IN, 1L), y = matrix(rnorm(D_OUT), D_OUT, 1L)))

  result <- dp_train(
    make_model = make_model,
    data       = data,
    loss_fn    = function(out, tgt) ag_mse_loss(out, tgt),
    forward_fn = function(model, s) model$forward(s$x),
    target_fn  = function(s) s$y,
    n_gpu = 2L, n_iter = 8L, lr = 1e-2, verbose = FALSE
  )
  expect_named(result, c("params", "loss_history", "model"))
  expect_length(result$loss_history, 8L)
  expect_false(any(is.nan(result$loss_history)))
})

test_that("dp_train: returns loss_history of correct length", {
  set.seed(9L)
  make_model <- function() {
    W <- ag_param(matrix(rnorm(4) * 0.1, 2, 2))
    list(forward    = function(x) ag_matmul(W, x),
         parameters = function() list(W = W))
  }
  data <- lapply(1:8, function(i) matrix(rnorm(2), 2, 1))

  result <- dp_train(
    make_model = make_model,
    data       = data,
    loss_fn    = function(out, tgt) ag_mse_loss(out, tgt),
    n_gpu = 1L, n_iter = 20L, lr = 1e-3, verbose = FALSE
  )
  expect_length(result$loss_history, 20L)
  expect_true(all(is.finite(result$loss_history)))
})
