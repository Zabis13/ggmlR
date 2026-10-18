# Tests for ggml_trainer(): a long-lived training context stepped by hand.

cleanup_model <- function(model) {
  ggml_backend_sched_free(model$compilation$sched)
  ggml_backend_free(model$compilation$backend)
  if (!is.null(model$compilation$cpu_backend)) {
    ggml_backend_free(model$compilation$cpu_backend)
  }
}

make_data <- function(n = 64L, nfeat = 4L, seed = 1L) {
  set.seed(seed)
  x <- matrix(runif(n * nfeat), nrow = n, ncol = nfeat)
  y <- matrix(0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > nfeat / 2) 1L else 2L] <- 1
  list(x = x, y = y)
}

build_model <- function(nfeat = 4L, seed = 1L, optimizer = "adam") {
  set.seed(seed)   # weights are drawn at compile time
  m <- ggml_model_sequential() |>
    ggml_layer_dense(8, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  m$input_shape <- nfeat
  ggml_compile(m, optimizer = optimizer, loss = "categorical_crossentropy")
}

test_that("stepping by hand trains the model", {
  ggml_set_n_threads(2L)
  d <- make_data()
  m <- build_model()

  tr <- ggml_trainer(m, batch_size = 16L)
  losses <- numeric(0)
  # Four passes over the same four batches: enough for the loss to move.
  for (rep in 1:4) {
    for (i in 1:4) {
      idx <- ((i - 1L) * 16L + 1L):(i * 16L)
      losses <- c(losses,
                  tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE]))
    }
  }

  expect_length(losses, 16L)
  expect_true(all(is.finite(losses)))
  expect_lt(mean(tail(losses, 4)), mean(head(losses, 4)))
  expect_equal(tr$n_steps(), 16L)

  tr$free()
  cleanup_model(m)
})

test_that("the context survives across steps rather than restarting", {
  ggml_set_n_threads(2L)
  d <- make_data()
  m <- build_model()

  tr <- ggml_trainer(m, batch_size = 16L)
  # Same batch twice. If the optimizer context were rebuilt per step, the second
  # loss would equal the first; a live context has already moved the weights.
  l1 <- tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])
  l2 <- tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])
  expect_false(isTRUE(all.equal(l1, l2)))

  tr$free()
  cleanup_model(m)
})

test_that("$eval does not train", {
  ggml_set_n_threads(2L)
  d <- make_data()
  m <- build_model()

  tr <- ggml_trainer(m, batch_size = 16L)
  e1 <- tr$eval(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])
  e2 <- tr$eval(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])

  expect_equal(e1, e2)          # weights untouched, so the loss repeats
  expect_equal(tr$n_steps(), 0L)

  tr$free()
  cleanup_model(m)
})

test_that("$model returns the trained weights", {
  ggml_set_n_threads(2L)
  d <- make_data()
  m <- build_model()

  tr <- ggml_trainer(m, batch_size = 16L)
  for (i in 1:4) {
    idx <- ((i - 1L) * 16L + 1L):(i * 16L)
    tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE])
  }
  trained <- tr$model()

  # The returned model is usable for inference, which is the point of handing
  # the weights back at all.
  preds <- ggml_predict(trained, d$x[1:16, , drop = FALSE], batch_size = 16L)
  expect_equal(dim(preds), c(16L, 2L))
  expect_true(all(is.finite(preds)))

  tr$free()
  cleanup_model(m)
})

test_that("a wrong batch size is refused", {
  ggml_set_n_threads(2L)
  d <- make_data()
  m <- build_model()

  tr <- ggml_trainer(m, batch_size = 16L)
  expect_error(tr$step(d$x[1:5, , drop = FALSE], d$y[1:5, , drop = FALSE]),
               "exactly 16")

  tr$free()
  cleanup_model(m)
})

test_that("free is idempotent and methods refuse to run afterwards", {
  ggml_set_n_threads(2L)
  d <- make_data()
  m <- build_model()

  tr <- ggml_trainer(m, batch_size = 16L)
  tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])

  expect_silent(tr$free())
  expect_silent(tr$free())      # second call is a no-op, not a double free
  expect_error(tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE]),
               "has been freed")
  expect_error(tr$model(), "has been freed")

  cleanup_model(m)
})

test_that("set_lr changes the step size", {
  ggml_set_n_threads(2L)
  d <- make_data()

  run_with_lr <- function(lr) {
    m <- build_model()
    tr <- ggml_trainer(m, batch_size = 16L)
    tr$set_lr(lr)
    for (i in 1:4) {
      idx <- ((i - 1L) * 16L + 1L):(i * 16L)
      tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE])
    }
    out <- tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])
    tr$free()
    cleanup_model(m)
    out
  }

  # Same data and seed, so any difference is the learning rate.
  expect_false(isTRUE(all.equal(run_with_lr(1e-4), run_with_lr(1e-1))))
})

test_that("gradient accumulation delays the weight update", {
  ggml_set_n_threads(2L)
  d <- make_data()
  m <- build_model()

  # nbatch_logical = 2 * batch_size: the optimizer steps every second call, so
  # the first two losses come from identical weights. The period lives inside
  # opt_ctx -- nothing on the R side counts it.
  tr <- ggml_trainer(m, batch_size = 16L, nbatch_logical = 32L)
  l1 <- tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])
  l2 <- tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])
  l3 <- tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])

  expect_equal(l1, l2)                          # no update between these two
  expect_false(isTRUE(all.equal(l2, l3)))       # update landed before the third

  tr$free()
  cleanup_model(m)
})

test_that("non-sequential input and uncompiled models are refused", {
  expect_error(ggml_trainer(list(), batch_size = 8L), "sequential")

  m <- ggml_model_sequential() |> ggml_layer_dense(4L, activation = "relu")
  m$input_shape <- 4L
  expect_error(ggml_trainer(m, batch_size = 8L), "compiled")
})

test_that("Adam moments carry over between steps", {
  # Regression. Calling ggml_fit() once per batch rebuilds the optimizer
  # context on every call, so the Adam moments and the bias-correction counter
  # restart and every step is the optimizer's first. A trainer must keep them.
  #
  # Rebuilding a trainer from $model() after every step reproduces that reset.
  # After one step the two runs agree -- both took a first Adam step from the
  # same weights, which also shows the rebuild carries the weights over -- and
  # after three they must not.
  ggml_set_n_threads(1L)
  d <- make_data()
  x <- d$x[1:16, , drop = FALSE]
  y <- d$y[1:16, , drop = FALSE]

  live_after <- function(k) {
    m <- build_model()
    tr <- ggml_trainer(m, batch_size = 16L)
    for (i in seq_len(k)) tr$step(x, y)
    out <- tr$eval(x, y)
    tr$free()
    cleanup_model(m)
    out
  }
  reset_after <- function(k) {
    m <- build_model()
    cur <- m
    for (i in seq_len(k)) {
      tr <- ggml_trainer(cur, batch_size = 16L)
      tr$step(x, y)
      cur <- tr$model()
      tr$free()
    }
    tr <- ggml_trainer(cur, batch_size = 16L)
    out <- tr$eval(x, y)
    tr$free()
    cleanup_model(m)
    out
  }

  expect_equal(live_after(1L), reset_after(1L))
  expect_false(isTRUE(all.equal(live_after(3L), reset_after(3L))))
})

# ---- functional models and loss masks --------------------------------------

build_functional <- function(seed = 1L, loss = "mse", n_out = 3L) {
  set.seed(seed)
  inp <- ggml_input(shape = 4L)
  h   <- ggml_layer_dense(inp, 8L, activation = "relu")
  out <- ggml_layer_dense(h, n_out)
  ggml_compile(ggml_model(inputs = inp, outputs = out),
               optimizer = "adam", loss = loss)
}

make_regression <- function(n = 64L, seed = 1L) {
  set.seed(seed)
  x <- matrix(runif(n * 4L), nrow = n, ncol = 4L)
  y <- cbind(rowSums(x), x[, 1] - x[, 2], x[, 3])
  list(x = x, y = y)
}

test_that("a functional model trains through the trainer", {
  ggml_set_n_threads(2L)
  d <- make_regression()
  m <- build_functional()

  set.seed(1L)
  tr <- ggml_trainer(m, batch_size = 16L)
  losses <- numeric(0)
  for (rep in 1:4) {
    for (i in 1:4) {
      idx <- ((i - 1L) * 16L + 1L):(i * 16L)
      losses <- c(losses,
                  tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE]))
    }
  }
  expect_true(all(is.finite(losses)))
  expect_lt(mean(tail(losses, 4)), mean(head(losses, 4)))

  trained <- tr$model()
  preds <- ggml_predict(trained, d$x[1:16, , drop = FALSE], batch_size = 16L)
  expect_equal(dim(preds), c(16L, 3L))
  expect_true(all(is.finite(preds)))

  tr$free()
  cleanup_model(m)
})

test_that("a masked output does not train, whatever its target", {
  # Two runs whose targets differ by 100 on every masked cell and nowhere else
  # must end with identical weights: a zero weight removes the term and its
  # gradient exactly. One thread and no dropout, so nothing else can make the
  # runs differ -- and the unmasked control shows the test can tell.
  ggml_set_n_threads(1L)
  d <- make_regression()
  set.seed(2L)
  mask <- matrix(rbinom(length(d$y), 1L, 0.4), nrow = nrow(d$y))
  y_far <- d$y + 100 * (1 - mask)

  final_preds <- function(y, loss_mask, w) {
    m <- build_functional()
    set.seed(1L)
    tr <- ggml_trainer(m, batch_size = 16L, loss_mask = loss_mask)
    for (i in 1:4) {
      idx <- ((i - 1L) * 16L + 1L):(i * 16L)
      tr$step(d$x[idx, , drop = FALSE], y[idx, , drop = FALSE],
              if (is.null(w)) NULL else w[idx, , drop = FALSE])
    }
    p <- ggml_predict(tr$model(), d$x[1:16, , drop = FALSE], batch_size = 16L)
    tr$free()
    cleanup_model(m)
    p
  }

  expect_equal(final_preds(d$y, 3L, mask), final_preds(y_far, 3L, mask))
  expect_false(isTRUE(all.equal(final_preds(d$y, NULL, NULL),
                                final_preds(y_far, NULL, NULL))))
})

test_that("per-sample weights are accepted with loss_mask = 1", {
  ggml_set_n_threads(2L)
  d <- make_regression()
  m <- build_functional()

  tr <- ggml_trainer(m, batch_size = 16L, loss_mask = 1L)
  l <- tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE],
               rep(c(1, 0.5), 8))
  expect_true(is.finite(l))
  # a weight of zero everywhere leaves nothing to fit
  expect_equal(tr$eval(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE],
                       rep(0, 16)), 0)

  tr$free()
  cleanup_model(m)
})

test_that("loss weights are refused where they cannot apply", {
  ggml_set_n_threads(2L)
  d <- make_regression()
  x <- d$x[1:16, , drop = FALSE]
  y <- d$y[1:16, , drop = FALSE]

  m <- build_functional()
  # a width that is neither 1 nor the number of outputs
  expect_error(ggml_trainer(m, batch_size = 16L, loss_mask = 2L),
               "number of outputs")
  # weights handed to a trainer built without them
  tr <- ggml_trainer(m, batch_size = 16L)
  expect_error(tr$step(x, y, matrix(1, 16, 3)), "without 'loss_mask'")
  tr$free()
  # weights of the wrong shape
  tr <- ggml_trainer(m, batch_size = 16L, loss_mask = 3L)
  expect_error(tr$step(x, y, matrix(1, 16, 2)), "16 x 3")
  tr$free()
  cleanup_model(m)

  # only mean squared error takes weights
  mc <- build_functional(loss = "categorical_crossentropy")
  expect_error(ggml_trainer(mc, batch_size = 16L, loss_mask = 3L), "mse")
  cleanup_model(mc)
})

test_that("ggml_fit refuses sample_weight on a functional model", {
  # It used to swallow the argument in `...` and train every output.
  d <- make_regression()
  m <- build_functional()
  expect_error(ggml_fit(m, d$x, d$y, batch_size = 16L, verbose = 0L,
                        sample_weight = matrix(1, 64, 3)),
               "ggml_trainer")
  cleanup_model(m)
})

test_that("a predict between steps neither crashes nor disturbs training", {
  # Regression. $model() hands out a model on the trainer's own scheduler. A
  # predict with it re-laid the scheduler's buffers while opt_ctx still believed
  # its own graph was allocated, and the next step wrote into them and
  # segfaulted. The trainer now re-allocates after $model() -- and that must not
  # cost the optimizer state either, so a run with predicts in the middle has to
  # give exactly the losses and weights of the same run without them.
  ggml_set_n_threads(1L)
  d <- make_regression()
  batches <- lapply(1:4, function(i) ((i - 1L) * 16L + 1L):(i * 16L))

  run <- function(with_predict) {
    m <- build_functional()
    set.seed(1L)
    tr <- ggml_trainer(m, batch_size = 16L)
    losses <- numeric(0)
    for (k in 1:8) {
      idx <- batches[[(k - 1L) %% 4L + 1L]]
      losses <- c(losses,
                  tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE]))
      if (with_predict && k %% 2L == 0L)
        ggml_predict(tr$model(), d$x[1:16, , drop = FALSE], batch_size = 16L)
    }
    p <- ggml_predict(tr$model(), d$x[1:16, , drop = FALSE], batch_size = 16L)
    tr$free()
    cleanup_model(m)
    list(losses = losses, preds = p)
  }

  plain <- run(FALSE)
  mixed <- run(TRUE)
  expect_equal(mixed$losses, plain$losses)
  expect_equal(mixed$preds, plain$preds)
})

# ---- $predict ---------------------------------------------------------------

test_that("$predict agrees with ggml_predict on the handed-out model", {
  # Forty rows against a batch of sixteen: the last batch is padded and trimmed.
  ggml_set_n_threads(1L)
  d <- make_regression()
  m <- build_functional()
  tr <- ggml_trainer(m, batch_size = 16L)
  for (i in 1:4) {
    idx <- ((i - 1L) * 16L + 1L):(i * 16L)
    tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE])
  }

  x40 <- d$x[1:40, , drop = FALSE]
  p <- tr$predict(x40)
  expect_equal(dim(p), c(40L, 3L))
  expect_equal(p, ggml_predict(tr$model(), x40, batch_size = 16L))

  tr$free()
  cleanup_model(m)
})

test_that("$predict takes fewer rows than a batch", {
  # A functional ggml_predict() refuses fewer rows than its batch_size; the
  # trainer's $predict() shrinks the batch to the rows instead.
  d  <- make_regression()
  m  <- build_functional()
  tr <- ggml_trainer(m, batch_size = 16L)
  tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])

  x5 <- d$x[1:5, , drop = FALSE]
  p  <- tr$predict(x5)
  expect_equal(dim(p), c(5L, 3L))
  expect_equal(p, ggml_predict(tr$model(), x5, batch_size = 5L))
  expect_equal(dim(tr$predict(d$x[1L, , drop = FALSE])), c(1L, 3L))

  tr$free()
  cleanup_model(m)
})

test_that("$predict reads the current weights, not a snapshot", {
  ggml_set_n_threads(1L)
  d <- make_regression()
  m <- build_functional()
  tr <- ggml_trainer(m, batch_size = 16L)
  x <- d$x[1:16, , drop = FALSE]
  y <- d$y[1:16, , drop = FALSE]

  p0 <- tr$predict(x)
  tr$step(x, y)
  p1 <- tr$predict(x)
  tr$step(x, y)
  p2 <- tr$predict(x)
  expect_false(isTRUE(all.equal(p0, p1)))
  expect_false(isTRUE(all.equal(p1, p2)))

  tr$free()
  cleanup_model(m)
})

test_that("$predict between steps leaves training exactly as it was", {
  # Losses at every step and the final weights (read through predictions) must
  # match a run without predicts: a side effect on the training buffers or on
  # the Adam moments would show up in either.
  ggml_set_n_threads(1L)
  d <- make_regression()
  batches <- lapply(1:4, function(i) ((i - 1L) * 16L + 1L):(i * 16L))

  run <- function(with_predict) {
    m <- build_functional()
    set.seed(1L)
    tr <- ggml_trainer(m, batch_size = 16L, loss_mask = 3L)
    losses <- numeric(0)
    for (k in 1:8) {
      idx <- batches[[(k - 1L) %% 4L + 1L]]
      losses <- c(losses,
                  tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE]))
      if (with_predict) tr$predict(d$x[1:40, , drop = FALSE])
    }
    p <- tr$predict(d$x[1:16, , drop = FALSE])
    tr$free()
    cleanup_model(m)
    list(losses = losses, preds = p)
  }

  plain <- run(FALSE)
  mixed <- run(TRUE)
  expect_equal(mixed$losses, plain$losses)
  expect_equal(mixed$preds, plain$preds)
})

test_that("$predict leaves models handed out by $model() on their scheduler", {
  ggml_set_n_threads(1L)
  d <- make_regression()
  m <- build_functional()
  tr <- ggml_trainer(m, batch_size = 16L)
  handed <- tr$model()
  tr$predict(d$x[1:16, , drop = FALSE])
  expect_identical(handed$compilation$sched, m$compilation$sched)
  expect_identical(tr$model()$compilation$sched, m$compilation$sched)

  tr$free()
  expect_error(tr$predict(d$x[1:16, , drop = FALSE]), "has been freed")
  cleanup_model(m)
})

test_that("$predict works for a sequential model", {
  ggml_set_n_threads(1L)
  d <- make_data()
  m <- build_model()
  tr <- ggml_trainer(m, batch_size = 16L)
  tr$step(d$x[1:16, , drop = FALSE], d$y[1:16, , drop = FALSE])
  expect_equal(tr$predict(d$x[1:40, , drop = FALSE]),
               ggml_predict(tr$model(), d$x[1:40, , drop = FALSE],
                            batch_size = 16L))
  tr$free()
  cleanup_model(m)
})

# ---- $snapshot / $update_target ---------------------------------------------

snapshot_weights <- function(m) {
  lapply(ggmlR:::.weight_tensors(m), ggml_backend_tensor_get_data)
}

train_steps <- function(tr, d, k) {
  for (i in seq_len(k)) {
    idx <- ((i - 1L) %% 4L * 16L + 1L):((i - 1L) %% 4L * 16L + 16L)
    tr$step(d$x[idx, , drop = FALSE], d$y[idx, , drop = FALSE])
  }
}

test_that("a snapshot starts equal to the trainer and stays put afterwards", {
  ggml_set_n_threads(1L)
  d <- make_regression()
  x <- d$x[1:16, , drop = FALSE]
  m <- build_functional()
  tr <- ggml_trainer(m, batch_size = 16L)
  train_steps(tr, d, 2L)

  target <- tr$snapshot()
  p_snap <- ggml_predict(target, x, batch_size = 16L)
  expect_equal(p_snap, tr$predict(x))

  train_steps(tr, d, 3L)
  expect_equal(ggml_predict(target, x, batch_size = 16L), p_snap)
  expect_false(isTRUE(all.equal(tr$predict(x), p_snap)))

  # it outlives the trainer
  tr$free()
  expect_equal(ggml_predict(target, x, batch_size = 16L), p_snap)

  ggml_snapshot_free(target)
  expect_silent(ggml_snapshot_free(target))   # second call is a no-op
  cleanup_model(m)
})

test_that("update_target: tau = 1 copies, tau = 0 keeps, tau = 0.5 averages", {
  ggml_set_n_threads(1L)
  d <- make_regression()
  x <- d$x[1:16, , drop = FALSE]
  m <- build_functional()
  tr <- ggml_trainer(m, batch_size = 16L)
  target <- tr$snapshot()
  train_steps(tr, d, 3L)

  old  <- snapshot_weights(target)
  live <- snapshot_weights(tr$model())

  tr$update_target(target, tau = 0)
  expect_equal(snapshot_weights(target), old)

  tr$update_target(target, tau = 0.5)
  mid <- snapshot_weights(target)
  for (k in names(mid))
    expect_equal(mid[[k]], 0.5 * (old[[k]] + live[[k]]), tolerance = 1e-6,
                 info = k)

  tr$update_target(target, tau = 1)
  expect_equal(ggml_predict(target, x, batch_size = 16L), tr$predict(x))

  ggml_snapshot_free(target)
  tr$free()
  cleanup_model(m)
})

test_that("update_target refuses a target that does not match", {
  ggml_set_n_threads(1L)
  m1 <- build_functional(n_out = 3L)
  m2 <- build_functional(n_out = 2L)
  tr1 <- ggml_trainer(m1, batch_size = 16L)
  tr2 <- ggml_trainer(m2, batch_size = 16L)
  other <- tr2$snapshot()

  expect_error(tr1$update_target(other), "same weights")
  expect_error(tr1$update_target(tr1$model()), "live snapshot")
  own <- tr1$snapshot()
  expect_error(tr1$update_target(own, tau = 2), "\\[0, 1\\]")

  ggml_snapshot_free(own)
  expect_error(tr1$update_target(own), "live snapshot")
  ggml_snapshot_free(other)
  tr1$free(); tr2$free()
  cleanup_model(m1); cleanup_model(m2)
})

test_that("a sequential model snapshots and updates too", {
  ggml_set_n_threads(1L)
  d <- make_data()
  x <- d$x[1:16, , drop = FALSE]
  m <- build_model()
  tr <- ggml_trainer(m, batch_size = 16L)
  target <- tr$snapshot()
  p0 <- ggml_predict(target, x, batch_size = 16L)

  tr$step(x, d$y[1:16, , drop = FALSE])
  expect_equal(ggml_predict(target, x, batch_size = 16L), p0)
  tr$update_target(target)
  expect_equal(ggml_predict(target, x, batch_size = 16L), tr$predict(x))

  ggml_snapshot_free(target)
  tr$free()
  cleanup_model(m)
})

test_that("taking and releasing a snapshot leaves the trainer's weights alone", {
  ggml_set_n_threads(1L)
  d <- make_regression()
  x <- d$x[1:16, , drop = FALSE]
  m <- build_functional()
  tr <- ggml_trainer(m, batch_size = 16L)
  train_steps(tr, d, 2L)

  before <- snapshot_weights(tr$model())
  target <- tr$snapshot()
  expect_identical(snapshot_weights(tr$model()), before)
  # the copy is equal in value but distinct in memory
  expect_equal(snapshot_weights(target), before)
  expect_false(identical(ggmlR:::.weight_tensors(target)[[1]],
                         ggmlR:::.weight_tensors(tr$model())[[1]]))

  ggml_snapshot_free(target)
  expect_identical(snapshot_weights(tr$model()), before)
  expect_error(ggml_predict(target, x, batch_size = 16L), "released")

  # and training goes on as if nothing happened
  expect_true(is.finite(tr$step(x, d$y[1:16, , drop = FALSE])))

  tr$free()
  cleanup_model(m)
})

# ---- a trainer built on a loaded model --------------------------------------

test_that("a trainer on a loaded model predicts from the trained weights", {
  ggml_set_n_threads(1L)
  d <- make_regression()
  x <- d$x[1:16, , drop = FALSE]

  m0  <- build_functional()
  tr0 <- ggml_trainer(m0, batch_size = 16L)
  train_steps(tr0, d, 3L)
  f0 <- tempfile(fileext = ".ggml")
  ggml_save_model(tr0$model(), f0)
  tr0$free()

  loaded  <- ggml_load_model(f0)
  p_saved <- ggml_predict(loaded, x, batch_size = 16L)

  tr <- ggml_trainer(loaded, batch_size = 16L)
  expect_equal(tr$predict(x), p_saved)        # starts from the loaded weights

  train_steps(tr, d, 5L)
  p_live <- tr$predict(x)
  # Bit-identical predictions after training mean the steps were not seen.
  expect_false(isTRUE(all.equal(p_live, p_saved)))

  # All three paths out of the trainer agree with the live weights.
  target <- tr$snapshot()
  expect_equal(ggml_predict(target, x, batch_size = 16L), p_live)
  ggml_snapshot_free(target)
  trained <- tr$model()
  expect_equal(ggml_predict(trained, x, batch_size = 16L), p_live)

  # Saved and loaded again, it is the trained net that comes back.
  f1 <- tempfile(fileext = ".ggml")
  ggml_save_model(trained, f1)
  reloaded <- ggml_load_model(f1)
  expect_equal(ggml_predict(reloaded, x, batch_size = 16L), p_live)

  tr$free()
  cleanup_model(reloaded)
  cleanup_model(loaded)
  cleanup_model(m0)
  unlink(c(f0, f1))
})
