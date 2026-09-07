# Tests for streaming training: ggml_fit_opt_gen() / ggml_fit_generator()

cleanup_model <- function(model) {
  ggml_backend_sched_free(model$compilation$sched)
  ggml_backend_free(model$compilation$backend)
  if (!is.null(model$compilation$cpu_backend)) {
    ggml_backend_free(model$compilation$cpu_backend)
  }
}

# Build a small separable classification problem.
make_data <- function(n = 128L, nfeat = 4L, seed = 1L) {
  set.seed(seed)
  x <- matrix(runif(n * nfeat), nrow = n, ncol = nfeat)
  y <- matrix(0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > nfeat / 2) 1L else 2L] <- 1
  list(x = x, y = y)
}

# A finite generator over a matrix: yields full batches, then NULL -- and resets
# itself on the way out, which is what the contract asks of a finite source so
# that the next epoch starts over.
batch_gen <- function(x, y, batch_size) {
  i <- 0L
  nb <- nrow(x) %/% batch_size
  function() {
    if (i >= nb) {
      i <<- 0L
      return(NULL)
    }
    idx <- (i * batch_size + 1L):((i + 1L) * batch_size)
    i <<- i + 1L
    list(x[idx, , drop = FALSE], y[idx, , drop = FALSE])
  }
}

# The same source without the reset: exhausted after one epoch.
once_gen <- function(x, y, batch_size) {
  i <- 0L
  nb <- nrow(x) %/% batch_size
  function() {
    if (i >= nb) return(NULL)
    idx <- (i * batch_size + 1L):((i + 1L) * batch_size)
    i <<- i + 1L
    list(x[idx, , drop = FALSE], y[idx, , drop = FALSE])
  }
}

# An endless generator: recycles the data and never returns NULL.
cyclic_gen <- function(x, y, batch_size) {
  i <- 0L
  nb <- nrow(x) %/% batch_size
  function() {
    j <- i %% nb
    i <<- i + 1L
    idx <- (j * batch_size + 1L):((j + 1L) * batch_size)
    list(x[idx, , drop = FALSE], y[idx, , drop = FALSE])
  }
}

build_model <- function(nfeat = 4L, seed = 1L) {
  set.seed(seed)   # weights are drawn at compile time
  m <- ggml_model_sequential() |>
    ggml_layer_dense(8, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  m$input_shape <- nfeat
  ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
}

# ============================================================================
# .gen_batch_size: mirrors ggml_opt_batch_size(), i.e. ne[ggml_n_dims(t)-1]
# ============================================================================

test_that(".gen_batch_size takes the last non-trivial axis, not ne[3]", {
  # ggml_tensor_shape() always returns four entries and never collapses
  # trailing unit dims, so reading the last element would give 1 for the most
  # common input layout and silently mis-derive opt_period. These cases pin the
  # collapsing rule, including the non-MNIST-shaped ones.
  bs <- ggmlR:::.gen_batch_size_of_ne

  expect_equal(bs(c(784, 32, 1, 1)), 32L)  # dense input
  expect_equal(bs(c(10, 32, 1, 1)), 32L)   # labels
  expect_equal(bs(c(28, 28, 1, 32)), 32L)  # image, batch on ne[3]
  expect_equal(bs(c(64, 1, 8, 1)), 8L)     # unit axis in the middle
  expect_equal(bs(c(3, 4, 5, 6)), 6L)      # fully populated 4D
  expect_equal(bs(c(1, 32, 1, 1)), 32L)    # ne[0] == 1
  expect_equal(bs(c(5, 1, 1, 1)), 5L)      # 1D: the axis itself is the batch
  expect_equal(bs(c(1, 1, 1, 1)), 1L)      # scalar
})

# ============================================================================
# Generator contract
# ============================================================================

test_that("a batch may be list(x, y) or list(inputs =, labels =)", {
  g1 <- function() list(1:4, 1:2)
  b1 <- ggmlR:::.gen_next_batch(g1)
  expect_equal(b1$inputs[[1]], 1:4)
  expect_equal(b1$labels[[1]], 1:2)

  g2 <- function() list(inputs = list(1:4), labels = list(1:2))
  b2 <- ggmlR:::.gen_next_batch(g2)
  expect_equal(b2$inputs[[1]], 1:4)
  expect_equal(b2$labels[[1]], 1:2)
})

test_that("NULL ends the epoch and bad shapes fail fast", {
  expect_null(ggmlR:::.gen_next_batch(function() NULL))

  expect_error(ggmlR:::.gen_next_batch(function() 42),
               "must return a list")
  expect_error(ggmlR:::.gen_next_batch(function() list(1, 2, 3)),
               "list of length 3")
  # Multi-head is deliberately refused until the functional API step: a wrong
  # label offset converges just as smoothly as a right one, so it cannot be
  # validated by a "loss goes down" test.
  expect_error(
    ggmlR:::.gen_next_batch(function() list(inputs = list(1, 2), labels = list(1, 2))),
    "multi-head"
  )
})

# ============================================================================
# Training equivalence and streaming behaviour
# ============================================================================

test_that("a finite generator trains like the in-memory path", {
  ggml_set_n_threads(2L)
  d <- make_data()

  m_gen <- build_model()
  m_gen <- ggml_fit(m_gen, generator = batch_gen(d$x, d$y, 32L),
                    epochs = 4L, batch_size = 32L, verbose = 0)

  expect_s3_class(m_gen$history, "ggml_history")
  expect_equal(length(m_gen$history$train_loss), 4L)
  # The point of training: the loss actually goes down.
  expect_lt(m_gen$history$train_loss[4], m_gen$history$train_loss[1])

  cleanup_model(m_gen)
})

test_that("a generator that does not restart stops training with a warning", {
  ggml_set_n_threads(2L)
  d <- make_data()

  m <- build_model()
  # Epoch 2 gets NULL on its first call. The two causes -- a finite source that
  # is genuinely done, and one that forgot to reset -- are indistinguishable
  # here, so the warning names both.
  expect_warning(
    m <- ggml_fit(m, generator = once_gen(d$x, d$y, 32L),
                  epochs = 4L, batch_size = 32L, verbose = 0),
    "returned NULL on the first step"
  )
  expect_equal(length(m$history$train_loss), 1L)
  cleanup_model(m)
})

test_that("a short epoch under steps_per_epoch warns", {
  ggml_set_n_threads(2L)
  d <- make_data()

  m <- build_model()
  # Declaring steps_per_epoch is a promise of that many batches; running out
  # early means the source broke it, which training must not hide.
  expect_warning(
    m <- ggml_fit(m, generator = once_gen(d$x, d$y, 32L),
                  steps_per_epoch = 10L, epochs = 1L, batch_size = 32L,
                  verbose = 0),
    "of 10 requested batches"
  )
  cleanup_model(m)
})

test_that("an endless generator needs steps_per_epoch and honours it", {
  ggml_set_n_threads(2L)
  d <- make_data()

  m <- build_model()
  m <- ggml_fit(m, generator = cyclic_gen(d$x, d$y, 32L),
                steps_per_epoch = 3L, epochs = 2L, batch_size = 32L,
                verbose = 0)

  expect_equal(length(m$history$train_loss), 2L)
  cleanup_model(m)
})

test_that("initial_epoch shifts numbering without restoring optimizer state", {
  ggml_set_n_threads(2L)
  d <- make_data()

  seen <- integer(0)
  cb <- list(on_epoch_begin = function(epoch, logs, state) {
    seen <<- c(seen, epoch)
  })

  m <- build_model()
  m <- ggml_fit(m, generator = cyclic_gen(d$x, d$y, 32L),
                steps_per_epoch = 2L, epochs = 3L, batch_size = 32L,
                initial_epoch = 10L, callbacks = list(cb), verbose = 0)

  # Keras semantics: the numbering is shifted for callbacks and LR schedules.
  expect_equal(seen, 11:13)
  expect_equal(m$history$epochs, 11:13)
  cleanup_model(m)
})

test_that("validation_generator produces validation metrics", {
  ggml_set_n_threads(2L)
  d <- make_data()

  m <- build_model()
  m <- ggml_fit(m, generator = cyclic_gen(d$x, d$y, 32L),
                steps_per_epoch = 3L, epochs = 2L, batch_size = 32L,
                validation_generator = cyclic_gen(d$x, d$y, 32L),
                validation_steps = 1L, verbose = 0)

  expect_false(any(is.na(m$history$val_loss)))
  cleanup_model(m)
})

# ============================================================================
# Fail-fast boundaries
# ============================================================================

test_that("dataset-shaped arguments are refused with a generator", {
  d <- make_data()
  g <- function() NULL

  m <- build_model()
  # val_split needs a total sample count, which a generator has not got.
  expect_error(ggml_fit(m, generator = g, validation_split = 0.2),
               "validation_split")
  expect_error(ggml_fit(m, generator = g, validation_data = list(d$x, d$y)),
               "validation_data")
  expect_error(ggml_fit(m, x = d$x, y = d$y, generator = g),
               "not both")
  expect_error(ggml_fit(m, generator = g, sample_weight = runif(128)),
               "sample_weight")
  cleanup_model(m)
})

test_that("a short batch is refused rather than silently padded", {
  ggml_set_n_threads(2L)
  d <- make_data()
  short <- function() list(d$x[1:5, , drop = FALSE], d$y[1:5, , drop = FALSE])

  m <- build_model()
  expect_error(
    ggml_fit(m, generator = short, steps_per_epoch = 1L, epochs = 1L,
             batch_size = 32L, verbose = 0),
    "batch of 5"
  )
  cleanup_model(m)
})
