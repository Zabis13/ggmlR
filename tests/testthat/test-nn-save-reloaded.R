# Saving a functional model that was loaded but never fitted in this session.
#
# ggml_save_model() used to read the weights only from node_weights -- the live
# tensors ggml_fit()/ggml_trainer() leave behind. A model fresh from
# ggml_load_model() keeps them in node_weights_data instead, so it was written
# WITHOUT weights, and the next load silently initialised new ones. Nothing
# errored; the predictions simply changed.

test_that("a reloaded functional model saves its weights again", {
  set.seed(7L)
  n <- 32L
  x <- matrix(runif(n * 4L), n, 4L)
  y <- matrix(runif(n * 2L), n, 2L)

  inp <- ggml_input(shape = 4L)
  out <- inp |> ggml_layer_dense(6L, activation = "relu") |> ggml_layer_dense(2L)
  m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                    optimizer = "adam", loss = "mse", backend = "cpu")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 16L, verbose = 0L)
  p0 <- ggml_predict(m, x, batch_size = 16L)

  f1 <- tempfile(fileext = ".rds"); f2 <- tempfile(fileext = ".rds")
  on.exit(unlink(c(f1, f2)), add = TRUE)

  ggml_save_model(m, f1)
  m1 <- ggml_load_model(f1, backend = "cpu")
  ggml_save_model(m1, f2)                      # never fitted: weights as data
  m2 <- ggml_load_model(f2, backend = "cpu")

  expect_false(is.null(readRDS(f2)$node_weights_data))
  expect_equal(ggml_predict(m2, x, batch_size = 16L), p0, tolerance = 1e-6)
})

test_that("weights given as node_weights_data are what gets saved", {
  # The route an exporter takes: build the graph, hand it weights as plain
  # vectors keyed by node id, save -- no fit in between.
  inp <- ggml_input(shape = 3L)
  out <- ggml_layer_dense(inp, 2L, name = "head")
  m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                    optimizer = "adam", loss = "mse", backend = "cpu")

  W <- matrix(c(1, 2, 3, 4, 5, 6), 2, 3)       # R [out, in]
  b <- c(0.5, -0.5)
  m$node_weights_data <- setNames(
    list(list(weight = as.numeric(t(W)), bias = b)), out$id)

  x <- matrix(c(1, 0, 2, -1, 1, 1), 2, 3, byrow = TRUE)
  ref <- x %*% t(W) + matrix(b, nrow(x), 2L, byrow = TRUE)
  expect_equal(ggml_predict(m, x, batch_size = 2L), ref, tolerance = 1e-6)

  f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
  ggml_save_model(m, f)
  expect_equal(ggml_predict(ggml_load_model(f, backend = "cpu"), x, batch_size = 2L),
               ref, tolerance = 1e-6)
})
