# Tests for Keras-compatible API: compile(), fit(), evaluate(), predict()

test_that("compile() dispatch works for sequential model", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 3L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_true(m$compiled)
})

test_that("compile() dispatch works for functional model", {
  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_dense(3L, activation = "softmax")
  m   <- ggml_model(inputs = x, outputs = out)
  m   <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_true(m$compiled)
})

test_that("fit() and evaluate() dispatch for sequential model", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1

  m <- fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)
  result <- evaluate(m, x, y, verbose = FALSE)
  expect_true(is.list(result) || is.numeric(result))
})

test_that("predict() dispatch for sequential model", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 2L, activation = "softmax", input_shape = 4L)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1
  m <- fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  p <- predict(m, x)
  expect_true(is.matrix(p) || is.numeric(p))
  expect_equal(nrow(p), 10)
})

test_that("fit() and predict() dispatch for functional model", {
  set.seed(42)
  x_in <- ggml_input(shape = 4L)
  out  <- x_in |> ggml_layer_dense(2L, activation = "softmax")
  m    <- ggml_model(inputs = x_in, outputs = out)
  m    <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  # Use batch_size <= n_samples
  n <- 32L
  x <- matrix(rnorm(4 * n), n, 4)
  y <- matrix(0, n, 2)
  y[cbind(1:n, sample(1:2, n, replace = TRUE))] <- 1

  m <- fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = FALSE)
  p <- predict(m, x)
  expect_true(is.matrix(p) || is.numeric(p))
})
