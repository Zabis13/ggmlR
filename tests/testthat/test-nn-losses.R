# Tests for the loss functions beyond cross-entropy and MSE: MAE, Huber and
# binary cross-entropy.
#
# Each is a new GGML_OPT_LOSS_TYPE_* built out of graph nodes in ggml-opt.cpp,
# so the checks are that the graph value matches the closed-form definition,
# that training actually descends, and that ggml_evaluate() agrees with fit.

cleanup_loss_model <- function(model) {
  if (!is.null(model$compilation$buffer)) {
    ggml_backend_buffer_free(model$compilation$buffer)
  }
  if (!is.null(model$compilation$ctx_weights)) {
    ggml_free(model$compilation$ctx_weights)
  }
  if (!is.null(model$compilation$sched)) {
    ggml_backend_sched_free(model$compilation$sched)
  }
  if (!is.null(model$compilation$backend)) {
    ggml_backend_free(model$compilation$backend)
  }
  if (!is.null(model$compilation$cpu_backend)) {
    ggml_backend_free(model$compilation$cpu_backend)
  }
}

# ---------------------------------------------------------------------------
# Constants and name resolution
# ---------------------------------------------------------------------------

test_that("the new loss-type constants are distinct", {
  ids <- c(ggml_opt_loss_type_mean(),
           ggml_opt_loss_type_sum(),
           ggml_opt_loss_type_cross_entropy(),
           ggml_opt_loss_type_mse(),
           ggml_opt_loss_type_weighted_mse(),
           ggml_opt_loss_type_mae(),
           ggml_opt_loss_type_huber(),
           ggml_opt_loss_type_binary_cross_entropy())
  expect_length(unique(ids), length(ids))
  expect_true(all(vapply(ids, is.integer, logical(1))))
})

test_that("every accepted loss name maps to a loss type", {
  # NN_LOSSES and nn_loss_type_of() must not drift apart: a name accepted by
  # validation that the mapper does not know aborts at fit time instead.
  for (nm in ggmlR:::NN_LOSSES) {
    expect_silent(ggmlR:::nn_loss_type_of(nm))
  }
})

test_that("the new names resolve to their own loss types", {
  expect_equal(ggmlR:::nn_loss_type_of("mae"), ggml_opt_loss_type_mae())
  expect_equal(ggmlR:::nn_loss_type_of("mean_absolute_error"),
               ggml_opt_loss_type_mae())
  expect_equal(ggmlR:::nn_loss_type_of("huber"), ggml_opt_loss_type_huber())
  expect_equal(ggmlR:::nn_loss_type_of("binary_crossentropy"),
               ggml_opt_loss_type_binary_cross_entropy())
})

test_that("binary cross-entropy is not treated as a logits loss", {
  # A categorical-CE head has its softmax stripped so ggml_cross_entropy_loss()
  # can apply its own. Binary CE consumes probabilities instead, so stripping
  # its sigmoid would feed logits to log().
  expect_true(ggmlR:::nn_loss_is_ce("categorical_crossentropy"))
  expect_false(ggmlR:::nn_loss_is_ce("binary_crossentropy"))
  expect_false(ggmlR:::nn_loss_is_ce("mae"))
  expect_false(ggmlR:::nn_loss_is_ce("huber"))
})

# ---------------------------------------------------------------------------
# The R-side loss used by ggml_evaluate() matches the closed form
# ---------------------------------------------------------------------------

test_that("nn_head_loss computes MAE, Huber and BCE in closed form", {
  set.seed(4L)
  p <- matrix(runif(20, 0.05, 0.95), 5, 4)
  y <- matrix(rbinom(20, 1, 0.5), 5, 4)

  expect_equal(ggmlR:::nn_head_loss("mae", p, y), mean(abs(y - p)))

  e   <- abs(y - p)
  ref <- mean(ifelse(e <= 1, 0.5 * e^2, e - 0.5))
  expect_equal(ggmlR:::nn_head_loss("huber", p, y), ref)

  expect_equal(ggmlR:::nn_head_loss("binary_crossentropy", p, y),
               mean(-(y * log(p) + (1 - y) * log(1 - p))))
})

test_that("Huber is quadratic near zero and linear far out", {
  # The two regimes must actually differ, otherwise the clamp is a no-op and
  # Huber would silently be MSE (or MAE) everywhere.
  small <- matrix(c(0.1, -0.2), 1, 2)
  zero  <- matrix(0, 1, 2)
  h_small <- ggmlR:::nn_head_loss("huber", small, zero)
  expect_equal(h_small, mean(0.5 * small^2))

  big <- matrix(c(5, -5), 1, 2)
  h_big <- ggmlR:::nn_head_loss("huber", big, zero)
  expect_equal(h_big, mean(abs(big) - 0.5))
  # Far out it must be much cheaper than MSE would be -- that is the point.
  expect_lt(h_big, mean(big^2) / 2)
})

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

make_reg <- function(n = 128L) {
  set.seed(42L)
  x <- matrix(runif(n * 4L, -1, 1), nrow = n)
  y <- matrix(rowSums(x[, 1:2]) - x[, 3], ncol = 1L)
  list(x = x, y = y)
}

fit_with_loss <- function(loss, d, epochs = 30L, units = 16L) {
  set.seed(1L)
  inp <- ggml_input(shape = 4L, name = "in")
  h   <- inp |> ggml_layer_dense(units, activation = "relu")
  out <- h   |> ggml_layer_dense(ncol(d$y))
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = loss, backend = "cpu")
  ggml_fit(m, d$x, d$y, epochs = epochs, batch_size = 32L, verbose = 0L)
}

test_that("MAE training descends", {
  skip_on_cran()
  d <- make_reg()
  m <- fit_with_loss("mae", d, epochs = 60L)
  h <- m$history
  expect_true(all(is.finite(h$train_loss)))
  # MAE's gradient has constant magnitude, so it converges more slowly than
  # MSE at the same learning rate -- the check is that it descends and that the
  # model actually learned the relation, not that it reaches a fixed fraction.
  expect_lt(tail(h$train_loss, 1L), h$train_loss[1L])
  p <- ggml_predict(m, d$x, batch_size = 32L)
  expect_gt(cor(as.vector(p), as.vector(d$y)), 0.9)
  cleanup_loss_model(m)
})

test_that("Huber training descends", {
  skip_on_cran()
  d <- make_reg()
  m <- fit_with_loss("huber", d, epochs = 60L)
  h <- m$history
  expect_true(all(is.finite(h$train_loss)))
  expect_lt(tail(h$train_loss, 1L), h$train_loss[1L])
  p <- ggml_predict(m, d$x, batch_size = 32L)
  expect_gt(cor(as.vector(p), as.vector(d$y)), 0.9)
  cleanup_loss_model(m)
})

test_that("binary cross-entropy training descends", {
  skip_on_cran()
  set.seed(42L)
  n <- 128L
  x <- matrix(runif(n * 4L, -1, 1), nrow = n)
  # Two independent labels: BCE treats each output as its own Bernoulli, which
  # is what distinguishes it from categorical cross-entropy.
  y <- cbind(as.integer(rowSums(x[, 1:2]) > 0),
             as.integer(x[, 3] - x[, 4] > 0)) * 1.0

  set.seed(1L)
  inp <- ggml_input(shape = 4L, name = "in")
  h   <- inp |> ggml_layer_dense(16L, activation = "relu")
  # sigmoid, not softmax: the outputs must be independent probabilities.
  out <- h   |> ggml_layer_dense(2L, activation = "sigmoid")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "binary_crossentropy",
                    backend = "cpu")
  m <- ggml_fit(m, x, y, epochs = 30L, batch_size = 32L, verbose = 0L)

  hh <- m$history
  expect_true(all(is.finite(hh$train_loss)))
  expect_lt(tail(hh$train_loss, 1L), hh$train_loss[1L])
  # Predictions stay inside the unit interval, as probabilities must.
  p <- ggml_predict(m, x, batch_size = 32L)
  expect_true(all(p >= 0 & p <= 1))

  cleanup_loss_model(m)
})

test_that("MAE is more robust to an outlier than MSE", {
  skip_on_cran()
  # The reason MAE exists: one wild target should not drag the fit. Training
  # both on the same corrupted data, MAE's predictions on the CLEAN points
  # should stay closer to the clean relationship.
  set.seed(42L)
  n <- 128L
  x <- matrix(runif(n * 4L, -1, 1), nrow = n)
  y_clean <- matrix(rowSums(x[, 1:2]) - x[, 3], ncol = 1L)
  y <- y_clean
  y[1:3, 1] <- 60          # a few gross outliers

  d <- list(x = x, y = y)
  m_mae <- fit_with_loss("mae", d, epochs = 40L)
  m_mse <- fit_with_loss("mse", d, epochs = 40L)

  keep <- 4:n
  err_mae <- mean(abs(ggml_predict(m_mae, x, batch_size = 32L)[keep, 1] -
                        y_clean[keep, 1]))
  err_mse <- mean(abs(ggml_predict(m_mse, x, batch_size = 32L)[keep, 1] -
                        y_clean[keep, 1]))

  expect_lt(err_mae, err_mse)

  cleanup_loss_model(m_mae); cleanup_loss_model(m_mse)
})

# ---------------------------------------------------------------------------
# evaluate() and multi-output
# ---------------------------------------------------------------------------

test_that("ggml_evaluate reports a finite loss for the new losses", {
  skip_on_cran()
  # Previously any loss the R-side evaluator did not know returned NA, so this
  # guards the mapping as much as the value.
  d <- make_reg()
  for (loss in c("mae", "huber")) {
    m <- fit_with_loss(loss, d, epochs = 5L)
    ev <- ggml_evaluate(m, d$x, d$y, batch_size = 32L)
    expect_true(is.finite(ev$loss), info = loss)
    expect_gte(ev$loss, 0)
    cleanup_loss_model(m)
  }
})

test_that("a sequential model trains with the new losses", {
  skip_on_cran()
  # The sequential path maps the loss name through its own call site; before
  # this it had a duplicate switch that knew only CE and MSE.
  d <- make_reg()
  set.seed(1L)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(1L)
  m <- ggml_compile(m, optimizer = "adam", loss = "huber")
  m <- ggml_fit(m, d$x, d$y, epochs = 20L, batch_size = 32L, verbose = 0L)

  expect_true(all(is.finite(m$history$train_loss)))
  expect_lt(tail(m$history$train_loss, 1L), m$history$train_loss[1L])

  ev <- ggml_evaluate(m, d$x, d$y, batch_size = 32L)
  expect_true(is.finite(ev$loss))

  cleanup_loss_model(m)
})

test_that("the new losses mix with cross-entropy across output heads", {
  skip_on_cran()
  # Per-head loss resolution has to cope with a head whose softmax is stripped
  # (CE) sitting next to one whose activation must survive (Huber).
  set.seed(42L)
  n <- 128L
  x  <- matrix(runif(n * 4L, -1, 1), nrow = n)
  cls <- as.integer(rowSums(x) > 0)
  y_c <- cbind(1 - cls, cls) * 1.0
  y_r <- matrix(rowSums(x), ncol = 1L)

  set.seed(1L)
  inp   <- ggml_input(shape = 4L, name = "in")
  trunk <- inp   |> ggml_layer_dense(16L, activation = "relu", name = "trunk")
  head_c <- trunk |> ggml_layer_dense(2L, activation = "softmax", name = "cls")
  head_r <- trunk |> ggml_layer_dense(1L, name = "reg")

  m <- ggml_model(inputs = inp, outputs = list(head_c, head_r))
  m <- ggml_compile(m, optimizer = "adam",
                    loss = list(cls = "categorical_crossentropy", reg = "huber"),
                    backend = "cpu")
  m <- ggml_fit(m, x, list(cls = y_c, reg = y_r),
                epochs = 25L, batch_size = 32L, verbose = 0L)
  h <- m$history

  expect_true(all(is.finite(h$train_cls_loss)))
  expect_true(all(is.finite(h$train_reg_loss)))
  expect_lt(tail(h$train_reg_loss, 1L), h$train_reg_loss[1L])

  cleanup_loss_model(m)
})

test_that("an unknown loss name is rejected at compile time", {
  inp <- ggml_input(shape = 4L)
  out <- inp |> ggml_layer_dense(1L)
  m <- ggml_model(inputs = inp, outputs = out)
  expect_error(ggml_compile(m, loss = "not_a_loss", backend = "cpu"),
               "Unsupported loss")
})
