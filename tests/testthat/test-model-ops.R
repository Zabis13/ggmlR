# Tests for model operations: save/load, freeze/unfreeze, predict_classes

# ============================================================================
# Sequential model: predict_classes
# ============================================================================

test_that("ggml_predict_classes returns integer class indices", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 3L, activation = "softmax")

  data(iris)
  x <- as.matrix(iris[1:10, 1:4])
  x <- scale(x)
  y <- model.matrix(~ Species - 1, iris[1:10, ])

  m <- ggml_compile(m, optimizer = "adam", loss = "cross_entropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  classes <- ggml_predict_classes(m, x)
  expect_type(classes, "integer")
  expect_length(classes, 10)
  expect_true(all(classes >= 1L & classes <= 3L))
})

# ============================================================================
# freeze / unfreeze weights (sequential)
# ============================================================================

test_that("ggml_freeze_weights and ggml_unfreeze_weights work for sequential", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")

  m_frozen <- ggml_freeze_weights(m, from = 1, to = 1)
  expect_false(m_frozen$layers[[1]]$trainable)
  expect_true(m_frozen$layers[[2]]$trainable)

  m_unfrozen <- ggml_unfreeze_weights(m_frozen, from = 1, to = 1)
  expect_true(m_unfrozen$layers[[1]]$trainable)
})

# ============================================================================
# freeze / unfreeze weights (functional)
# ============================================================================

test_that("ggml_freeze_weights and ggml_unfreeze_weights work for functional", {
  x   <- ggml_input(shape = 4L)
  h   <- x |> ggml_layer_dense(8L, name = "d1")
  out <- h |> ggml_layer_dense(2L, name = "d2")
  m   <- ggml_model(inputs = x, outputs = out)

  m_frozen <- ggml_freeze_weights(m, layer_names = "d1")
  expect_s3_class(m_frozen, "ggml_functional_model")
  # d1 alone is marked; d2 stays trainable
  expect_false(m_frozen$frozen_nodes[[h$id]])
  expect_null(m_frozen$frozen_nodes[[out$id]])

  m_unfrozen <- ggml_unfreeze_weights(m_frozen, layer_names = "d1")
  expect_s3_class(m_unfrozen, "ggml_functional_model")
  expect_null(m_unfrozen$frozen_nodes[[h$id]])
})

# ============================================================================
# save / load weights (sequential)
# ============================================================================

test_that("ggml_save_weights and ggml_load_weights roundtrip", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1

  m <- ggml_compile(m, optimizer = "adam", loss = "cross_entropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  pred_before <- ggml_predict(m, x)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_weights(m, tmp)

  # Load into a fresh model with same architecture
  m2 <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")
  m2 <- ggml_compile(m2, optimizer = "adam", loss = "cross_entropy")
  m2 <- ggml_load_weights(m2, tmp)

  pred_after <- ggml_predict(m2, x)
  expect_equal(pred_before, pred_after, tolerance = 1e-5)
})

# ============================================================================
# save / load model (sequential)
# ============================================================================

test_that("ggml_save_model and ggml_load_model roundtrip for sequential", {
  set.seed(42)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")

  x <- matrix(rnorm(40), 10, 4)
  y <- matrix(0, 10, 2)
  y[cbind(1:10, sample(1:2, 10, replace = TRUE))] <- 1

  m <- ggml_compile(m, optimizer = "adam", loss = "cross_entropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 10L, verbose = FALSE)

  pred_before <- ggml_predict(m, x)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_model(m, tmp)

  m2 <- ggml_load_model(tmp)
  pred_after <- ggml_predict(m2, x)
  expect_equal(pred_before, pred_after, tolerance = 1e-5)
})

test_that("ggml_save_model keeps ggml_freeze_weights marks on a functional model", {
  set.seed(42)
  x   <- ggml_input(shape = 4L)
  h   <- x |> ggml_layer_dense(8L, activation = "relu", name = "d1")
  out <- h |> ggml_layer_dense(2L, name = "d2")
  m   <- ggml_model(inputs = x, outputs = out)
  m   <- ggml_freeze_weights(m, layer_names = "d1")
  m   <- ggml_compile(m, optimizer = "adam", loss = "mse")

  xs <- matrix(rnorm(40), 10, 4)
  ys <- matrix(rnorm(20), 10, 2)
  m  <- ggml_fit(m, xs, ys, epochs = 1L, batch_size = 10L, verbose = FALSE)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_model(m, tmp)

  m2 <- ggml_load_model(tmp)
  expect_identical(m2$frozen_nodes, m$frozen_nodes)
  expect_false(m2$frozen_nodes[[h$id]])

  # Trained on after loading, d1 must not move while d2 does.
  d1_before <- readRDS(tmp)$node_weights_data[[h$id]]$weight
  d2_before <- readRDS(tmp)$node_weights_data[[out$id]]$weight
  m2 <- ggml_fit(m2, xs, ys, epochs = 3L, batch_size = 10L, verbose = FALSE)
  ggml_save_model(m2, tmp)
  expect_equal(readRDS(tmp)$node_weights_data[[h$id]]$weight, d1_before)
  expect_false(isTRUE(all.equal(readRDS(tmp)$node_weights_data[[out$id]]$weight,
                                d2_before)))
})

# ============================================================================
# file format and integrity on load
# ============================================================================

# A small compiled, trained sequential model and the file it was saved to.
saved_sequential <- function() {
  set.seed(1)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(2L)
  m <- ggml_compile(m, optimizer = "adam", loss = "mse")
  x <- matrix(rnorm(40), 10, 4)
  m <- ggml_fit(m, x, matrix(rnorm(20), 10, 2), epochs = 1L, batch_size = 10L,
                verbose = FALSE)
  f <- tempfile(fileext = ".rds")
  ggml_save_model(m, f)
  list(x = x, file = f)
}

test_that("ggml_save_model writes the current format version", {
  s <- saved_sequential(); on.exit(unlink(s$file))
  expect_identical(readRDS(s$file)$version, ggmlR:::NN_MODEL_FORMAT)
})

test_that("ggml_load_model refuses a file from a newer format", {
  s <- saved_sequential(); on.exit(unlink(s$file))
  d <- readRDS(s$file); d$version <- ggmlR:::NN_MODEL_FORMAT + 1L
  saveRDS(d, s$file)
  expect_error(ggml_load_model(s$file), "written by a newer ggmlR")
})

test_that("ggml_load_model names a truncated or foreign file", {
  s <- saved_sequential(); on.exit(unlink(s$file))
  raw <- readBin(s$file, "raw", file.size(s$file))
  writeBin(raw[seq_len(length(raw) %/% 2L)], s$file)
  expect_error(ggml_load_model(s$file), "cannot read")

  saveRDS(list(a = 1), s$file)
  expect_error(ggml_load_model(s$file), "not a file written by ggml_save_model")
})

test_that("a saved weight of the wrong size stops before it is written", {
  s <- saved_sequential(); on.exit(unlink(s$file))
  d <- readRDS(s$file)
  w <- d$layers[[1L]]$weights_data$weight
  d$layers[[1L]]$weights_data$weight <- w[-length(w)]
  saveRDS(d, s$file)
  m <- ggml_load_model(s$file)
  expect_error(ggml_predict(m, s$x), "layer 1 \\(dense\\): saved 'weight' has")
})

test_that("a saved weight that is missing stops instead of re-initialising", {
  s <- saved_sequential(); on.exit(unlink(s$file))
  d <- readRDS(s$file)
  d$layers[[2L]]$weights_data$bias <- NULL
  saveRDS(d, s$file)
  m <- ggml_load_model(s$file)
  expect_error(ggml_predict(m, s$x), "layer 2 \\(dense\\): the saved weights lack bias")
})

test_that("a functional model's saved weights are checked too", {
  set.seed(1)
  x   <- ggml_input(shape = 4L)
  h   <- x |> ggml_layer_dense(8L, activation = "relu", name = "d1")
  out <- h |> ggml_layer_dense(2L, name = "d2")
  m   <- ggml_compile(ggml_model(inputs = x, outputs = out),
                      optimizer = "adam", loss = "mse")
  xs  <- matrix(rnorm(40), 10, 4)
  m   <- ggml_fit(m, xs, matrix(rnorm(20), 10, 2), epochs = 1L,
                  batch_size = 10L, verbose = FALSE)
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f))
  ggml_save_model(m, f)

  d <- readRDS(f)
  w <- d$node_weights_data[[h$id]]$weight
  d$node_weights_data[[h$id]]$weight <- c(w, 0)
  saveRDS(d, f)
  expect_error(ggml_predict(ggml_load_model(f), xs, batch_size = 10L),
               "node 'd1' \\(dense\\): saved 'weight' has")
})

# ============================================================================
# freezing what is not a ggml_* model
# ============================================================================

test_that("ggml_freeze_weights names what it supports for anything else", {
  expect_error(ggml_freeze_weights(ag_dropout(0.1)),
               "works on models from ggml_model_sequential\\(\\) and ggml_model\\(\\).*optimizer")
  expect_error(ggml_unfreeze_weights(list()), "got an object of class 'list'")
})

# ============================================================================
# weight decay and frozen weights
# ============================================================================

test_that("weight_decay is checked, compiled in, and saved", {
  m <- ggml_model_sequential() |> ggml_layer_dense(2L, input_shape = 4L)
  expect_error(ggml_compile(m, loss = "mse", weight_decay = 2), "\\[0, 1\\]")
  expect_error(ggml_compile(m, loss = "mse", weight_decay = -0.1), "\\[0, 1\\]")

  s <- saved_sequential(); on.exit(unlink(s$file))
  m <- ggml_compile(ggml_load_model(s$file), optimizer = "adamw", loss = "mse",
                    weight_decay = 0.1)
  expect_equal(m$compilation$weight_decay, 0.1)
  m <- ggml_fit(m, s$x, matrix(0, 10, 2), epochs = 1L, batch_size = 10L,
                verbose = FALSE)
  ggml_save_model(m, s$file)
  expect_equal(ggml_load_model(s$file)$compilation$weight_decay, 0.1)
})

test_that("weight decay moves trainable weights but never frozen ones", {
  s <- saved_sequential(); on.exit(unlink(s$file))
  start <- readRDS(s$file)$layers
  y <- matrix(rnorm(20), 10, 2)

  # Load, freeze layer 1, train with the given decay, return the saved layers.
  train <- function(wd) {
    m  <- ggml_freeze_weights(ggml_load_model(s$file), from = 1L, to = 1L)
    tr <- ggml_trainer(m, batch_size = 10L, weight_decay = wd)
    on.exit(tr$free())
    tr$set_lr(1e-2)
    for (i in 1:20) tr$step(s$x, y)
    f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
    ggml_save_model(tr$model(), f)
    readRDS(f)$layers
  }
  plain <- train(0)
  decay <- train(0.5)

  # Frozen layer: untouched with decay as without it.
  expect_equal(decay[[1L]]$weights_data$weight, start[[1L]]$weights_data$weight)
  expect_equal(decay[[1L]]$weights_data$bias,   start[[1L]]$weights_data$bias)
  # Trainable layer: decay is really applied -- the same run without it differs,
  # and the decayed weights are smaller.
  expect_false(isTRUE(all.equal(decay[[2L]]$weights_data$weight,
                                plain[[2L]]$weights_data$weight)))
  expect_lt(sum(decay[[2L]]$weights_data$weight^2),
            sum(plain[[2L]]$weights_data$weight^2))
})
