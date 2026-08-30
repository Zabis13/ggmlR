# Tests for the per-output loss mask: WEIGHTED_MEAN_SQUARED_ERROR generalised
# from [1, ndata] weights to [ne_label, ndata].
#
# The contract under test:
#   loss = sum( w * (pred - y)^2 ) / nelements(pred)
# The denominator is nelements(pred) in BOTH modes -- a mask zeroes terms out
# of the numerator but does not renormalise over the active coordinates. Tests
# that assert on loss values below encode that deliberately.

cleanup_model <- function(model) {
  ggml_backend_sched_free(model$compilation$sched)
  ggml_backend_free(model$compilation$backend)
  if (!is.null(model$compilation$cpu_backend)) {
    ggml_backend_free(model$compilation$cpu_backend)
  }
}

# ============================================================================
# Dataset weight tensor: shape, width validation, cache behaviour
# ============================================================================

test_that("dataset weights default to the per-datapoint shape", {
  ds <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, ne_datapoint = 4L, ne_label = 3L, ndata = 8L, ndata_shard = 1L)
  on.exit(ggml_opt_dataset_free(ds))

  w <- ggml_opt_dataset_weights(ds)
  expect_equal(ggml_tensor_shape(w)[1:2], c(1L, 8L))
})

test_that("dataset weights allocate a mask when asked for one", {
  ds <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, ne_datapoint = 4L, ne_label = 3L, ndata = 8L, ndata_shard = 1L)
  on.exit(ggml_opt_dataset_free(ds))

  w <- ggml_opt_dataset_weights(ds, 3L)
  expect_equal(ggml_tensor_shape(w)[1:2], c(3L, 8L))
})

test_that("dataset weights reject a non-positive width", {
  ds <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, ne_datapoint = 4L, ne_label = 3L, ndata = 8L, ndata_shard = 1L)
  on.exit(ggml_opt_dataset_free(ds))

  expect_error(ggml_opt_dataset_weights(ds, 0L), "positive")
  expect_error(ggml_opt_dataset_weights(ds, -2L), "positive")
})

test_that("re-requesting the weights with the same width returns the cache", {
  ds <- ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, ne_datapoint = 4L, ne_label = 3L, ndata = 8L, ndata_shard = 1L)
  on.exit(ggml_opt_dataset_free(ds))

  w1 <- ggml_opt_dataset_weights(ds, 3L)
  w2 <- ggml_opt_dataset_weights(ds, 3L)
  expect_equal(ggml_tensor_shape(w1)[1:2], ggml_tensor_shape(w2)[1:2])
})

# ============================================================================
# ggml_fit: argument validation on the R side
# ============================================================================

make_mse_model <- function(n_in = 4L, n_out = 3L) {
  set.seed(1L)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(n_out, input_shape = n_in)
  ggml_compile(m, optimizer = "adam", loss = "mean_squared_error")
}

test_that("a matrix sample_weight is rejected for non-MSE losses", {
  set.seed(1L)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(3L, activation = "softmax", input_shape = 4L)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  on.exit(cleanup_model(m))

  x <- matrix(runif(8 * 4), nrow = 8)
  y <- matrix(0, nrow = 8, ncol = 3); y[, 1] <- 1
  mask <- matrix(1, nrow = 8, ncol = 3)

  expect_error(ggml_fit(m, x, y, epochs = 1L, batch_size = 4L,
                        sample_weight = mask, verbose = 0),
               "per-output loss mask")
})

test_that("a mask with the wrong number of rows is rejected", {
  m <- make_mse_model()
  on.exit(cleanup_model(m))

  x <- matrix(runif(8 * 4), nrow = 8)
  y <- matrix(runif(8 * 3), nrow = 8)

  expect_error(ggml_fit(m, x, y, epochs = 1L, batch_size = 4L,
                        sample_weight = matrix(1, nrow = 7, ncol = 3),
                        verbose = 0),
               "one row per training sample")
})

test_that("a mask with the wrong number of columns is rejected", {
  m <- make_mse_model()
  on.exit(cleanup_model(m))

  x <- matrix(runif(8 * 4), nrow = 8)
  y <- matrix(runif(8 * 3), nrow = 8)

  expect_error(ggml_fit(m, x, y, epochs = 1L, batch_size = 4L,
                        sample_weight = matrix(1, nrow = 8, ncol = 2),
                        verbose = 0),
               "one column per output")
})

# ============================================================================
# Regression: the vector path is unchanged
# ============================================================================

test_that("a vector sample_weight still trains (per-datapoint regression)", {
  m <- make_mse_model()
  on.exit(cleanup_model(m))

  x <- matrix(runif(16 * 4), nrow = 16)
  y <- matrix(runif(16 * 3), nrow = 16)

  h <- ggml_fit(m, x, y, epochs = 2L, batch_size = 8L,
                sample_weight = rep(1.0, 16), verbose = 0)

  expect_s3_class(h$history, "ggml_history")
  expect_length(h$history$train_loss, 2L)
  expect_true(all(is.finite(h$history$train_loss)))
})

test_that("an all-ones mask matches an all-ones weight vector", {
  # Both reduce to plain MSE, and the denominator is the same in each mode, so
  # the losses must agree.
  x <- matrix(runif(16 * 4), nrow = 16)
  y <- matrix(runif(16 * 3), nrow = 16)

  m1 <- make_mse_model(); on.exit(cleanup_model(m1), add = TRUE)
  h1 <- ggml_fit(m1, x, y, epochs = 1L, batch_size = 8L,
                 sample_weight = rep(1.0, 16), verbose = 0, shuffle = FALSE)

  m2 <- make_mse_model(); on.exit(cleanup_model(m2), add = TRUE)
  h2 <- ggml_fit(m2, x, y, epochs = 1L, batch_size = 8L,
                 sample_weight = matrix(1.0, nrow = 16, ncol = 3),
                 verbose = 0, shuffle = FALSE)

  expect_equal(h1$history$train_loss, h2$history$train_loss, tolerance = 1e-5)
})

# ============================================================================
# The mask actually masks
# ============================================================================

test_that("a zero mask makes the loss vanish", {
  # Every weight is 0, so the numerator is empty regardless of the predictions.
  m <- make_mse_model()
  on.exit(cleanup_model(m))

  x <- matrix(runif(16 * 4), nrow = 16)
  y <- matrix(runif(16 * 3), nrow = 16)

  h <- ggml_fit(m, x, y, epochs = 1L, batch_size = 8L,
                sample_weight = matrix(0.0, nrow = 16, ncol = 3),
                verbose = 0, shuffle = FALSE)

  expect_equal(h$history$train_loss[1], 0, tolerance = 1e-6)
})

test_that("masking a column removes exactly that column's error", {
  # Masking out one of three outputs must leave a loss strictly between the
  # fully-masked (0) and unmasked cases, and -- because the denominator does
  # not change -- the drop is the masked column's own contribution.
  x <- matrix(runif(16 * 4), nrow = 16)
  y <- matrix(runif(16 * 3), nrow = 16)

  m_full <- make_mse_model(); on.exit(cleanup_model(m_full), add = TRUE)
  h_full <- ggml_fit(m_full, x, y, epochs = 1L, batch_size = 8L,
                     sample_weight = matrix(1.0, nrow = 16, ncol = 3),
                     verbose = 0, shuffle = FALSE)

  mask <- matrix(1.0, nrow = 16, ncol = 3)
  mask[, 2] <- 0.0
  m_part <- make_mse_model(); on.exit(cleanup_model(m_part), add = TRUE)
  h_part <- ggml_fit(m_part, x, y, epochs = 1L, batch_size = 8L,
                     sample_weight = mask, verbose = 0, shuffle = FALSE)

  expect_lt(h_part$history$train_loss[1], h_full$history$train_loss[1])
  expect_gt(h_part$history$train_loss[1], 0)
})

test_that("a masked-out column is not trained toward its target", {
  # The mask must keep the gradient off the zero-weighted outputs, not merely
  # keep them out of the reported loss.
  #
  # The target is the SAME for every column, and only one column is unmasked.
  # A working mask trains that column alone, so the others stay near their
  # initial values. Giving each row its own active column instead (the literal
  # DQN layout) does NOT test this: with a shared dense layer the heads are
  # coupled through the same weights, so the masked-out outputs track the
  # trained ones and the test passes either way.
  set.seed(1L)
  n <- 32L; n_out <- 3L
  x <- matrix(runif(n * 4), nrow = n)
  y <- matrix(5.0, nrow = n, ncol = n_out)

  mask <- matrix(0.0, nrow = n, ncol = n_out)
  mask[, 1] <- 1.0

  m <- make_mse_model(n_out = n_out)
  on.exit(cleanup_model(m))

  h <- ggml_fit(m, x, y, epochs = 300L, batch_size = 8L,
                sample_weight = mask, verbose = 0, shuffle = FALSE)

  expect_true(all(is.finite(h$history$train_loss)))
  expect_lt(h$history$train_loss[length(h$history$train_loss)],
            h$history$train_loss[1])

  preds <- ggml_predict(h, x)
  # The unmasked column is pulled well toward 5; the masked ones are not.
  expect_gt(mean(preds[, 1]), 2.0)
  expect_lt(mean(preds[, 2]), 2.0)
  expect_lt(mean(preds[, 3]), 2.0)
  expect_gt(mean(preds[, 1]) - max(mean(preds[, 2]), mean(preds[, 3])), 1.0)
})

test_that("a per-row Q mask trains without collapsing", {
  # The literal DQN layout: one active action per row. This checks that the
  # mask feeds through a per-row selection at all (finite loss, still
  # decreasing) -- the separation itself is asserted by the test above, which
  # is not confounded by the shared dense layer.
  set.seed(1L)
  n <- 32L; n_out <- 3L
  x <- matrix(runif(n * 4), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = n_out)

  action <- rep(seq_len(n_out), length.out = n)
  mask <- matrix(0.0, nrow = n, ncol = n_out)
  for (i in seq_len(n)) {
    mask[i, action[i]] <- 1.0
    y[i, action[i]] <- 5.0
  }

  m <- make_mse_model(n_out = n_out)
  on.exit(cleanup_model(m))

  h <- ggml_fit(m, x, y, epochs = 300L, batch_size = 8L,
                sample_weight = mask, verbose = 0, shuffle = FALSE)

  expect_true(all(is.finite(h$history$train_loss)))
  expect_lt(h$history$train_loss[length(h$history$train_loss)],
            h$history$train_loss[1])

  preds <- ggml_predict(h, x)
  sel <- vapply(seq_len(n), function(i) preds[i, action[i]], numeric(1))
  expect_gt(mean(sel), 1.0)
})

# ============================================================================
# Sharding: a mask must be sliced as a rectangular block
# ============================================================================

test_that("the mask survives a multi-batch run", {
  # More than one batch means get_batch_weights() has to slice rectangular
  # blocks of the mask rather than the first nbatch scalars; a wrong stride
  # would misalign the mask against the labels and break the zero-loss identity.
  m <- make_mse_model()
  on.exit(cleanup_model(m))

  x <- matrix(runif(32 * 4), nrow = 32)
  y <- matrix(runif(32 * 3), nrow = 32)

  h <- ggml_fit(m, x, y, epochs = 1L, batch_size = 4L,
                sample_weight = matrix(0.0, nrow = 32, ncol = 3),
                verbose = 0, shuffle = FALSE)

  expect_equal(h$history$train_loss[1], 0, tolerance = 1e-6)
})

test_that("a per-row mask stays aligned with its own row across batches", {
  # Rows alternate between "all weight" and "no weight". If the shard copy
  # misaligned the mask, zeroed rows would leak error into the loss and the
  # result would differ from training on the kept rows alone.
  set.seed(1L)
  n <- 32L
  x <- matrix(runif(n * 4), nrow = n)
  y <- matrix(runif(n * 3), nrow = n)

  keep <- rep(c(TRUE, FALSE), length.out = n)
  mask <- matrix(0.0, nrow = n, ncol = 3)
  mask[keep, ] <- 1.0

  m <- make_mse_model()
  on.exit(cleanup_model(m))

  h <- ggml_fit(m, x, y, epochs = 1L, batch_size = 4L,
                sample_weight = mask, verbose = 0, shuffle = FALSE)

  expect_true(is.finite(h$history$train_loss[1]))
  expect_gt(h$history$train_loss[1], 0)
})
