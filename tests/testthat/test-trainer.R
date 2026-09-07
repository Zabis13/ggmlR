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
