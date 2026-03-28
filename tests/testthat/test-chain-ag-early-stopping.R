# Chain test: Autograd manual early stopping with weight restore
# Pattern from titanic_classification.R variant 4:
#   ag_sequential + optimizer_adam + manual val_loss monitoring +
#   best-weights snapshot + restore on plateau
#
# Uses synthetic data (no external files).

# Helper: predict with ag model (softmax, chunked)
ag_predict <- function(model, x_col) {
  n <- ncol(x_col)
  out <- matrix(0.0, 2L, n)
  for (s in seq(1L, n, by = 32L)) {
    e <- min(s + 31L, n)
    xb <- ag_tensor(x_col[, s:e, drop = FALSE])
    lg <- model$forward(xb)$data
    ev <- exp(lg - matrix(apply(lg, 2, max), nrow = nrow(lg), ncol = ncol(lg), byrow = TRUE))
    cs <- matrix(colSums(ev), nrow = nrow(ev), ncol = ncol(ev), byrow = TRUE)
    out[, s:e] <- ev / cs
  }
  out
}

# Compute BCE loss on predictions
bce_loss <- function(probs, targets) {
  eps <- 1e-7
  p <- pmin(pmax(probs, eps), 1 - eps)
  -mean(targets[1,] * log(p[1,]) + targets[2,] * log(p[2,]))
}

# ── Manual early stopping: stops and restores weights ──────

test_that("chain ag-early-stopping: stops training and restores best weights", {
  set.seed(42)
  n <- 120L
  # Linearly separable
  x_all <- rbind(matrix(rnorm(n, -1.5, 0.5), n/2, 2),
                 matrix(rnorm(n,  1.5, 0.5), n/2, 2))
  y_all <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
                 matrix(c(0,1), n/2, 2, byrow = TRUE))
  # Split 80/20
  idx_val <- c(1:12, 61:72)
  idx_tr  <- setdiff(1:n, idx_val)

  x_tr <- t(x_all[idx_tr, ])  # [2, n_tr]
  y_tr <- t(y_all[idx_tr, ])  # [2, n_tr]
  x_vl <- t(x_all[idx_val, ])
  y_vl <- t(y_all[idx_val, ])

  m <- ag_sequential(
    ag_linear(2L, 16L, activation = "relu"),
    ag_linear(16L, 2L)
  )
  params <- m$parameters()
  opt <- optimizer_adam(params, lr = 1e-2)

  # Early stopping state
  patience <- 10L
  best_val_loss <- Inf
  best_weights <- NULL
  wait <- 0L
  stopped_epoch <- NA_integer_
  n_tr <- ncol(x_tr)
  BS <- 16L

  ag_train(m)
  for (ep in seq_len(200L)) {
    perm <- sample(n_tr)
    for (b in seq_len(ceiling(n_tr / BS))) {
      idx <- perm[((b-1L)*BS+1L):min(b*BS, n_tr)]
      xb <- ag_tensor(x_tr[, idx, drop = FALSE])
      yb <- y_tr[, idx, drop = FALSE]
      with_grad_tape({ loss <- ag_softmax_cross_entropy_loss(m$forward(xb), yb) })
      grads <- backward(loss)
      opt$step(grads)
      opt$zero_grad()
    }

    # Val loss
    ag_eval(m)
    vl <- bce_loss(ag_predict(m, x_vl), y_vl)
    ag_train(m)

    if (vl < best_val_loss - 1e-4) {
      best_val_loss <- vl
      best_weights <- lapply(params, function(p) p$data)
      wait <- 0L
    } else {
      wait <- wait + 1L
      if (wait >= patience) {
        stopped_epoch <- ep
        break
      }
    }
  }

  # Should have stopped early (not run all 200)
  expect_true(!is.na(stopped_epoch))
  expect_true(stopped_epoch < 200L)

  # Best weights should be saved
  expect_false(is.null(best_weights))

  # Restore best weights
  for (nm in names(params)) params[[nm]]$data <- best_weights[[nm]]

  # Predictions after restore should be valid
  ag_eval(m)
  probs <- ag_predict(m, x_vl)
  expect_equal(nrow(probs), 2L)
  expect_equal(ncol(probs), ncol(x_vl))
  expect_true(all(is.finite(probs)))
  expect_true(all(probs >= 0 & probs <= 1))

  # Restored model should have decent accuracy on separable data
  pred_class <- apply(probs, 2, which.max)
  true_class <- apply(y_vl, 2, which.max)
  acc <- mean(pred_class == true_class)
  expect_true(acc > 0.7)
})

# ── Scheduler + clip_grad + dataloader full pipeline ───────

test_that("chain ag-pipeline: scheduler + clip_grad + dataloader trains", {
  set.seed(42)
  n <- 80L
  x_all <- rbind(matrix(rnorm(n, -2, 0.5), n/2, 2),
                 matrix(rnorm(n,  2, 0.5), n/2, 2))
  y_all <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
                 matrix(c(0,1), n/2, 2, byrow = TRUE))

  x_cm <- t(x_all)  # [2, n]
  y_cm <- t(y_all)  # [2, n]

  m <- ag_sequential(
    ag_linear(2L, 16L, activation = "relu"),
    ag_batch_norm(16L),
    ag_dropout(0.2),
    ag_linear(16L, 2L)
  )
  params <- m$parameters()
  opt <- optimizer_adam(params, lr = 1e-2)
  sch <- lr_scheduler_cosine(opt, T_max = 30L, lr_min = 1e-4)
  dl <- ag_dataloader(x_cm, y_cm, batch_size = 16L, shuffle = TRUE)

  losses <- numeric(30L)
  ag_train(m)
  for (ep in seq_len(30L)) {
    ep_loss <- 0
    nb <- 0L
    for (batch in dl$epoch()) {
      with_grad_tape({
        loss <- ag_softmax_cross_entropy_loss(m$forward(batch$x), batch$y$data)
      })
      grads <- backward(loss)
      clip_grad_norm(params, grads, max_norm = 5.0)
      opt$step(grads)
      opt$zero_grad()
      ep_loss <- ep_loss + loss$data[1]
      nb <- nb + 1L
    }
    losses[ep] <- ep_loss / nb
    sch$step()
  }

  # Loss should decrease
  expect_true(mean(losses[26:30]) < mean(losses[1:5]))

  # LR should have decayed
  expect_true(opt$lr < 1e-2)

  # Predictions should be valid
  ag_eval(m)
  probs <- ag_predict(m, x_cm)
  expect_true(all(is.finite(probs)))
})

# ── Raw ag_param + dp_train ────────────────────────────────

test_that("chain ag-raw-params: dp_train converges on separable data", {
  set.seed(42)
  n <- 60L
  x_all <- rbind(matrix(rnorm(n, -2, 0.3), n/2, 2),
                 matrix(rnorm(n,  2, 0.3), n/2, 2))
  y_all <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
                 matrix(c(0,1), n/2, 2, byrow = TRUE))

  dp_data <- lapply(seq_len(n), function(i)
    list(x = matrix(x_all[i, ], 2, 1),
         y = matrix(y_all[i, ], 2, 1)))

  make_model <- function() {
    W1 <- ag_param(matrix(rnorm(16 * 2) * 0.5, 16, 2))
    b1 <- ag_param(matrix(0, 16, 1))
    W2 <- ag_param(matrix(rnorm(2 * 16) * 0.5, 2, 16))
    b2 <- ag_param(matrix(0, 2, 1))
    list(
      forward = function(x) {
        h <- ag_relu(ag_add(ag_matmul(W1, x), b1))
        ag_add(ag_matmul(W2, h), b2)
      },
      parameters = function() list(W1=W1, b1=b1, W2=W2, b2=b2)
    )
  }

  res <- dp_train(
    make_model = make_model,
    data       = dp_data,
    loss_fn    = function(out, tgt) ag_softmax_cross_entropy_loss(out, tgt),
    forward_fn = function(model, s) model$forward(ag_tensor(s$x)),
    target_fn  = function(s) s$y,
    n_gpu      = 1L,
    n_iter     = 500L,
    lr         = 1e-3,
    verbose    = FALSE
  )

  # Loss should have decreased
  lh <- res$loss_history
  expect_true(mean(tail(lh, 50)) < mean(head(lh, 50)))
})
