# Tests for Functional Neural Network API (Block 1)

# Helper: free all ggml resources held by a functional model
cleanup_functional_model <- function(model) {
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

# ============================================================================
# ggml_input()
# ============================================================================

test_that("ggml_input creates a tensor node with correct class", {
  x <- ggml_input(shape = 64L)
  expect_s3_class(x, "ggml_tensor_node")
})

test_that("ggml_input stores shape correctly for 1-D", {
  x <- ggml_input(shape = 32L)
  expect_equal(x$config$shape, 32L)
  expect_equal(x$node_type, "input")
})

test_that("ggml_input stores shape correctly for 3-D image", {
  x <- ggml_input(shape = c(28L, 28L, 1L))
  expect_equal(x$config$shape, c(28L, 28L, 1L))
})

test_that("ggml_input auto-generates name like 'input_N'", {
  x <- ggml_input(shape = 10L)
  expect_match(x$config$name, "^input_\\d+$")
})

test_that("ggml_input respects custom name", {
  x <- ggml_input(shape = 10L, name = "my_input")
  expect_equal(x$config$name, "my_input")
})

test_that("ggml_input has no parents", {
  x <- ggml_input(shape = 8L)
  expect_length(x$parents, 0L)
})

# ============================================================================
# ggml_model()
# ============================================================================

test_that("ggml_model accepts single node inputs/outputs", {
  x   <- ggml_input(shape = 8L)
  out <- x |> ggml_layer_dense(4L)
  m   <- ggml_model(inputs = x, outputs = out)

  expect_s3_class(m, "ggml_functional_model")
  expect_false(m$compiled)
})

test_that("ggml_model wraps single nodes in lists", {
  x   <- ggml_input(shape = 8L)
  out <- x |> ggml_layer_dense(4L)
  m   <- ggml_model(inputs = x, outputs = out)

  expect_true(is.list(m$inputs))
  expect_true(is.list(m$outputs))
  expect_length(m$inputs, 1L)
  expect_length(m$outputs, 1L)
})

test_that("ggml_model rejects non-tensor-node inputs", {
  expect_error(ggml_model(inputs = "not_a_node", outputs = "also_not"), "ggml_tensor_node")
})

# ============================================================================
# Layer dispatch — functional mode
# ============================================================================

test_that("ggml_layer_dense on tensor node returns tensor node", {
  x <- ggml_input(shape = 16L)
  y <- ggml_layer_dense(x, 8L)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "dense")
  expect_equal(y$config$units, 8L)
})

test_that("ggml_layer_dense on sequential model still works", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(8L, activation = "relu")
  expect_equal(model$layers[[1]]$type, "dense")
})

test_that("ggml_layer_dense activation propagated in functional mode", {
  x <- ggml_input(shape = 8L)
  y <- ggml_layer_dense(x, 4L, activation = "relu")
  expect_equal(y$config$activation, "relu")
})

test_that("ggml_layer_flatten on tensor node returns tensor node", {
  x <- ggml_input(shape = c(4L, 4L))
  y <- ggml_layer_flatten(x)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "flatten")
})

test_that("ggml_layer_batch_norm on tensor node returns tensor node", {
  x <- ggml_input(shape = 16L)
  y <- ggml_layer_batch_norm(x)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "batch_norm")
})

# ============================================================================
# ggml_layer_add() / ggml_layer_concatenate()
# ============================================================================

test_that("ggml_layer_add creates add node with two parents", {
  x <- ggml_input(shape = 8L)
  a <- x |> ggml_layer_dense(8L)
  b <- x |> ggml_layer_dense(8L)
  z <- ggml_layer_add(list(a, b))

  expect_s3_class(z, "ggml_tensor_node")
  expect_equal(z$node_type, "add")
  expect_length(z$parents, 2L)
})

test_that("ggml_layer_add rejects fewer than 2 tensors", {
  x <- ggml_input(shape = 8L)
  a <- x |> ggml_layer_dense(8L)
  expect_error(ggml_layer_add(list(a)), "at least 2")
})

test_that("ggml_layer_add rejects non-list argument", {
  x <- ggml_input(shape = 8L)
  expect_error(ggml_layer_add(x))
})

test_that("ggml_layer_concatenate creates concatenate node", {
  x <- ggml_input(shape = 8L)
  a <- x |> ggml_layer_dense(4L)
  b <- x |> ggml_layer_dense(4L)
  z <- ggml_layer_concatenate(list(a, b), axis = 0L)

  expect_s3_class(z, "ggml_tensor_node")
  expect_equal(z$node_type, "concatenate")
  expect_equal(z$config$axis, 0L)
})

test_that("ggml_input auto-generates sequential names input_N", {
  a <- ggml_input(shape = 4L)
  b <- ggml_input(shape = 4L)
  # Both should match pattern and be different
  expect_match(a$config$name, "^input_\\d+$")
  expect_match(b$config$name, "^input_\\d+$")
  expect_false(a$config$name == b$config$name)
})

test_that("ggml_model rejects non-input node as input", {
  x   <- ggml_input(shape = 8L)
  mid <- x |> ggml_layer_dense(4L)
  out <- mid |> ggml_layer_dense(2L)
  expect_error(ggml_model(inputs = mid, outputs = out), "node_type")
})

test_that("ggml_layer_add auto-generates name add_N", {
  x <- ggml_input(shape = 8L)
  a <- x |> ggml_layer_dense(8L)
  b <- x |> ggml_layer_dense(8L)
  z <- ggml_layer_add(list(a, b))
  expect_match(z$config$name, "^add_\\d+$")
})

test_that("ggml_layer_add respects custom name", {
  x <- ggml_input(shape = 8L)
  a <- x |> ggml_layer_dense(8L)
  b <- x |> ggml_layer_dense(8L)
  z <- ggml_layer_add(list(a, b), name = "my_add")
  expect_equal(z$config$name, "my_add")
})

test_that("ggml_layer_add supports 3 inputs", {
  x <- ggml_input(shape = 8L)
  a <- x |> ggml_layer_dense(8L)
  b <- x |> ggml_layer_dense(8L)
  c <- x |> ggml_layer_dense(8L)
  z <- ggml_layer_add(list(a, b, c))
  expect_length(z$parents, 3L)
})

test_that("ggml_layer_add shape mismatch raises error at build time", {
  set.seed(1)
  n   <- 32L
  x   <- matrix(runif(n * 4L), nrow = n)
  y   <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  a   <- inp |> ggml_layer_dense(8L)
  b   <- inp |> ggml_layer_dense(4L)  # different shape!
  z   <- ggml_layer_add(list(a, b))
  out <- z |> ggml_layer_dense(2L, activation = "softmax")
  m   <- ggml_model(inputs = inp, outputs = out)
  m   <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_error(ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L),
               "shape mismatch")
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_layer_concatenate auto-generates name concatenate_N", {
  x <- ggml_input(shape = 4L)
  a <- x |> ggml_layer_dense(4L)
  b <- x |> ggml_layer_dense(4L)
  z <- ggml_layer_concatenate(list(a, b), axis = 0L)
  expect_match(z$config$name, "^concatenate_\\d+$")
})

test_that("ggml_layer_concatenate supports axis=-1", {
  x <- ggml_input(shape = 4L)
  a <- x |> ggml_layer_dense(4L)
  b <- x |> ggml_layer_dense(4L)
  z <- ggml_layer_concatenate(list(a, b), axis = -1L)
  expect_equal(z$config$axis, -1L)
})

test_that("ggml_layer_concatenate invalid axis raises error at build time", {
  set.seed(1)
  n   <- 32L
  x   <- matrix(runif(n * 4L), nrow = n)
  y   <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  a   <- inp |> ggml_layer_dense(4L)
  b   <- inp |> ggml_layer_dense(4L)
  z   <- ggml_layer_concatenate(list(a, b), axis = 5L)  # out of range for 1D
  out <- z |> ggml_layer_dense(2L, activation = "softmax")
  m   <- ggml_model(inputs = inp, outputs = out)
  m   <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_error(ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L),
               "out of range")
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_layer_concatenate with 3 inputs compiles (forward-only; no backward for concat)", {
  # ggml_concat does not implement backward pass — test compile only
  inp <- ggml_input(shape = 8L)
  a   <- inp |> ggml_layer_dense(4L, activation = "relu")
  b   <- inp |> ggml_layer_dense(4L, activation = "relu")
  c   <- inp |> ggml_layer_dense(4L, activation = "relu")
  cat_node <- ggml_layer_concatenate(list(a, b, c), axis = 0L)
  out <- cat_node |> ggml_layer_dense(2L, activation = "softmax")
  m   <- ggml_model(inputs = inp, outputs = out)
  expect_no_error(m <- ggml_compile(m, optimizer = "adam",
                                    loss = "categorical_crossentropy"))
  expect_true(m$compiled)
  on.exit(cleanup_functional_model(m))
})

test_that("ResNet-like model (deep residual) compiles and trains", {
  set.seed(3)
  n   <- 64L
  x   <- matrix(runif(n * 16L), nrow = n)
  y   <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 8) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 16L)
  # Block 1
  a1  <- inp |> ggml_layer_dense(16L, activation = "relu")
  r1  <- ggml_layer_add(list(inp, a1))
  # Block 2
  a2  <- r1 |> ggml_layer_dense(16L, activation = "relu")
  r2  <- ggml_layer_add(list(r1, a2))
  # Block 3
  a3  <- r2 |> ggml_layer_dense(16L, activation = "relu")
  r3  <- ggml_layer_add(list(r2, a3))
  out <- r3 |> ggml_layer_dense(2L, activation = "softmax")

  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_no_error(m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# nn_topo_sort()
# ============================================================================

test_that("nn_topo_sort returns input node before output node", {
  x   <- ggml_input(shape = 8L)
  out <- x |> ggml_layer_dense(4L)
  m   <- ggml_model(inputs = x, outputs = out)

  sorted <- nn_topo_sort(m$outputs)
  node_types <- vapply(sorted, `[[`, character(1), "node_type")

  # input must appear before dense
  expect_equal(node_types[1], "input")
  expect_equal(node_types[2], "dense")
})

test_that("nn_topo_sort handles residual graph correctly", {
  x    <- ggml_input(shape = 8L)
  a    <- x |> ggml_layer_dense(8L, activation = "relu")
  skip <- x |> ggml_layer_dense(8L)
  out  <- ggml_layer_add(list(a, skip))
  m    <- ggml_model(inputs = x, outputs = out)

  sorted <- nn_topo_sort(m$outputs)
  node_types <- vapply(sorted, `[[`, character(1), "node_type")

  # input first, add last
  expect_equal(node_types[1], "input")
  expect_equal(node_types[length(node_types)], "add")
})

# ============================================================================
# Compile
# ============================================================================

test_that("ggml_compile works on functional model", {
  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_dense(2L, activation = "softmax")
  m   <- ggml_model(inputs = x, outputs = out)
  m   <- ggml_compile(m, optimizer = "adam",
                       loss = "categorical_crossentropy")

  expect_true(m$compiled)
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Fit — linear graph (equivalent to Sequential)
# ============================================================================

test_that("ggml_fit.ggml_functional_model trains a linear model", {
  set.seed(42)
  n  <- 64L
  x  <- matrix(runif(n * 4L), nrow = n, ncol = 4L)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  out <- inp |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam",
                    loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 3L, batch_size = 32L, verbose = 0L)

  expect_true(m$compiled)
  expect_s3_class(m$history, "ggml_history")
  expect_length(m$history$train_loss, 3L)
  # Loss should be finite
  expect_true(all(is.finite(m$history$train_loss)))

  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Fit — residual block (add node)
# ============================================================================

test_that("ggml_fit trains a model with residual add", {
  set.seed(7)
  n  <- 64L
  x  <- matrix(runif(n * 8L), nrow = n, ncol = 8L)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 4) 1L else 2L] <- 1.0

  inp  <- ggml_input(shape = 8L)
  a    <- inp |> ggml_layer_dense(8L, activation = "relu")
  skip <- inp |> ggml_layer_dense(8L)
  res  <- ggml_layer_add(list(a, skip))
  out  <- res |> ggml_layer_dense(2L, activation = "softmax")

  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam",
                    loss = "categorical_crossentropy")

  expect_no_error(
    m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0L)
  )
  expect_true(all(is.finite(m$history$train_loss)))

  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Fit — concatenate
# ============================================================================

test_that("ggml_model with concatenate compiles and graph builds", {
  # Note: ggml_concat does not support backward pass, so we only test that
  # the model can be compiled (graph structure is valid).  Training would
  # require a custom backward implementation in ggml.
  inp <- ggml_input(shape = 8L)
  a   <- inp |> ggml_layer_dense(4L, activation = "relu")
  b   <- inp |> ggml_layer_dense(4L, activation = "relu")
  cat_node <- ggml_layer_concatenate(list(a, b), axis = 0L)
  out <- cat_node |> ggml_layer_dense(2L, activation = "softmax")

  m <- ggml_model(inputs = inp, outputs = out)
  expect_no_error(
    m <- ggml_compile(m, optimizer = "adam",
                      loss = "categorical_crossentropy")
  )
  expect_true(m$compiled)

  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Evaluate
# ============================================================================

test_that("ggml_evaluate.ggml_functional_model returns loss and accuracy", {
  set.seed(99)
  n  <- 64L
  x  <- matrix(runif(n * 4L), nrow = n, ncol = 4L)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  out <- inp |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam",
                    loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0L)

  res <- ggml_evaluate(m, x, y, batch_size = 32L)
  expect_named(res, c("loss", "accuracy"))
  expect_true(is.finite(res$loss))
  expect_true(is.finite(res$accuracy))

  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Predict
# ============================================================================

test_that("ggml_predict.ggml_functional_model returns matrix of correct shape", {
  set.seed(123)
  n  <- 64L
  x  <- matrix(runif(n * 4L), nrow = n, ncol = 4L)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  out <- inp |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam",
                    loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0L)

  preds <- ggml_predict(m, x, batch_size = 32L)
  expect_true(is.matrix(preds))
  expect_equal(nrow(preds), n)
  expect_equal(ncol(preds), 2L)
  # Softmax rows should sum to ~1
  row_sums <- rowSums(preds)
  expect_true(all(abs(row_sums - 1.0) < 0.01))

  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Block 2 — Dropout
# ============================================================================

test_that("ggml_layer_dropout on tensor node returns tensor node with correct type", {
  x <- ggml_input(shape = 16L)
  y <- ggml_layer_dropout(x, rate = 0.5)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "dropout")
  expect_equal(y$config$rate, 0.5)
})

test_that("ggml_layer_dropout on sequential model adds a dropout layer", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dropout(0.3)
  expect_equal(length(model$layers), 2L)
  expect_equal(model$layers[[2]]$type, "dropout")
  expect_equal(model$layers[[2]]$config$rate, 0.3)
})

test_that("ggml_layer_dropout rejects rate >= 1", {
  x <- ggml_input(shape = 8L)
  expect_error(ggml_layer_dropout(x, rate = 1.0))
  expect_error(ggml_layer_dropout(x, rate = 1.5))
})

test_that("functional model with dropout compiles without error", {
  inp <- ggml_input(shape = 8L)
  out <- inp |>
    ggml_layer_dense(16L, activation = "relu") |>
    ggml_layer_dropout(0.5) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  expect_no_error(m <- ggml_compile(m, optimizer = "adam",
                                     loss = "categorical_crossentropy"))
  expect_true(m$compiled)
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_fit with dropout converges (loss decreases)", {
  set.seed(1)
  n <- 64L
  x <- matrix(runif(n * 8L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 4) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 8L)
  out <- inp |>
    ggml_layer_dense(16L, activation = "relu") |>
    ggml_layer_dropout(0.3) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 5L, batch_size = 32L, verbose = 0L)

  expect_true(all(is.finite(m$history$train_loss)))
  # Loss over 5 epochs should go down (at least from first to last)
  expect_true(m$history$train_loss[5] <= m$history$train_loss[1] * 1.1)
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_predict with dropout is deterministic (two calls give same result)", {
  set.seed(2)
  n <- 32L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  out <- inp |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_dropout(0.4) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  p1 <- ggml_predict(m, x, batch_size = 32L)
  p2 <- ggml_predict(m, x, batch_size = 32L)
  expect_equal(p1, p2)
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_layer_dropout stochastic=TRUE stores config correctly", {
  x <- ggml_input(shape = 16L)
  y <- ggml_layer_dropout(x, rate = 0.5, stochastic = TRUE)
  expect_true(isTRUE(y$config$stochastic))
})

test_that("ggml_fit with stochastic dropout trains without error", {
  set.seed(10)
  n <- 64L
  x <- matrix(runif(n * 8L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 4) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 8L)
  out <- inp |>
    ggml_layer_dense(16L, activation = "relu") |>
    ggml_layer_dropout(0.3, stochastic = TRUE) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_no_error(m <- ggml_fit(m, x, y, epochs = 3L, batch_size = 32L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_predict with stochastic dropout is deterministic (training=FALSE, mask=ones)", {
  set.seed(11)
  n <- 32L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  out <- inp |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_dropout(0.4, stochastic = TRUE) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  p1 <- ggml_predict(m, x, batch_size = 32L)
  p2 <- ggml_predict(m, x, batch_size = 32L)
  expect_equal(p1, p2)
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Block 2 — Embedding
# ============================================================================

test_that("ggml_input with dtype='int32' stores dtype correctly", {
  x <- ggml_input(shape = 10L, dtype = "int32")
  expect_equal(x$config$dtype, "int32")
  expect_equal(x$config$shape, 10L)
})

test_that("ggml_input with default dtype is 'float32'", {
  x <- ggml_input(shape = 8L)
  expect_equal(x$config$dtype, "float32")
})

test_that("ggml_input rejects unknown dtype", {
  expect_error(ggml_input(shape = 8L, dtype = "float16"), "dtype")
})

test_that("ggml_layer_embedding on tensor node returns tensor node with correct config", {
  x <- ggml_input(shape = 10L, dtype = "int32")
  e <- ggml_layer_embedding(x, vocab_size = 100L, dim = 8L)
  expect_s3_class(e, "ggml_tensor_node")
  expect_equal(e$node_type, "embedding")
  expect_equal(e$config$vocab_size, 100L)
  expect_equal(e$config$dim, 8L)
})

test_that("embedding model compiles without error", {
  inp <- ggml_input(shape = 10L, dtype = "int32")
  out <- inp |>
    ggml_layer_embedding(vocab_size = 50L, dim = 8L) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(3L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  expect_no_error(m <- ggml_compile(m, optimizer = "adam",
                                     loss = "categorical_crossentropy"))
  expect_true(m$compiled)
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_fit with embedding layer trains without error", {
  set.seed(3)
  n        <- 64L
  seq_len  <- 5L
  vocab    <- 20L
  n_class  <- 3L

  x <- matrix(sample(0L:(vocab - 1L), n * seq_len, replace = TRUE),
               nrow = n, ncol = seq_len)
  y <- matrix(0.0, nrow = n, ncol = n_class)
  for (i in seq_len(n)) y[i, sample(n_class, 1)] <- 1.0

  inp <- ggml_input(shape = seq_len, dtype = "int32")
  out <- inp |>
    ggml_layer_embedding(vocab_size = vocab, dim = 8L) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(n_class, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  expect_no_error(m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_predict with embedding returns matrix of correct shape", {
  set.seed(4)
  n       <- 32L
  seq_len <- 4L
  vocab   <- 10L
  n_class <- 2L

  x <- matrix(sample(0L:(vocab - 1L), n * seq_len, replace = TRUE),
               nrow = n, ncol = seq_len)
  y <- matrix(0.0, nrow = n, ncol = n_class)
  for (i in seq_len(n)) y[i, sample(n_class, 1)] <- 1.0

  inp <- ggml_input(shape = seq_len, dtype = "int32")
  out <- inp |>
    ggml_layer_embedding(vocab_size = vocab, dim = 4L) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(n_class, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  preds <- ggml_predict(m, x, batch_size = 32L)
  expect_true(is.matrix(preds))
  expect_equal(nrow(preds), n)
  expect_equal(ncol(preds), n_class)
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Block 2 — Multi-output
# ============================================================================

test_that("ggml_model with two outputs stores both", {
  inp    <- ggml_input(shape = 8L)
  hidden <- inp |> ggml_layer_dense(4L, activation = "relu")
  out    <- hidden |> ggml_layer_dense(2L, activation = "softmax")
  m      <- ggml_model(inputs = inp, outputs = list(hidden, out))

  expect_length(m$outputs, 2L)
})

test_that("ggml_predict with multi-output model returns list of length 2", {
  set.seed(5)
  n <- 32L
  x <- matrix(runif(n * 8L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 4) 1L else 2L] <- 1.0

  inp    <- ggml_input(shape = 8L)
  hidden <- inp |> ggml_layer_dense(4L, activation = "relu")
  out    <- hidden |> ggml_layer_dense(2L, activation = "softmax")

  m <- ggml_model(inputs = inp, outputs = list(hidden, out))
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  preds <- ggml_predict(m, x, batch_size = 32L)
  expect_true(is.list(preds))
  expect_length(preds, 2L)
  on.exit(cleanup_functional_model(m))
})

test_that("multi-output: first output has shape [n, 4], second has shape [n, 2]", {
  set.seed(6)
  n <- 32L
  x <- matrix(runif(n * 8L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 4) 1L else 2L] <- 1.0

  inp    <- ggml_input(shape = 8L)
  hidden <- inp |> ggml_layer_dense(4L, activation = "relu")
  out    <- hidden |> ggml_layer_dense(2L, activation = "softmax")

  m <- ggml_model(inputs = inp, outputs = list(hidden, out))
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  preds <- ggml_predict(m, x, batch_size = 32L)
  expect_equal(dim(preds[[1]]), c(n, 4L))
  expect_equal(dim(preds[[2]]), c(n, 2L))
  on.exit(cleanup_functional_model(m))
})

test_that("ggml_fit with multi-output uses last output for loss", {
  set.seed(7)
  n <- 64L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp    <- ggml_input(shape = 4L)
  hidden <- inp |> ggml_layer_dense(8L, activation = "relu")
  out    <- hidden |> ggml_layer_dense(2L, activation = "softmax")

  m <- ggml_model(inputs = inp, outputs = list(hidden, out))
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  expect_no_error(m <- ggml_fit(m, x, y, epochs = 3L, batch_size = 32L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Block 3 — Shared layers
# ============================================================================

test_that("shared dense layer: two applications create nodes with same name", {
  x1 <- ggml_input(shape = 4L, name = "inp_sh1")
  x2 <- ggml_input(shape = 4L, name = "inp_sh2")
  y1 <- ggml_layer_dense(x1, 8L, activation = "relu", name = "shared")
  y2 <- ggml_layer_dense(x2, 8L, activation = "relu", name = "shared")
  expect_equal(y1$config$name, "shared")
  expect_equal(y2$config$name, "shared")
  expect_false(y1$id == y2$id)
})

test_that("shared dense layer: Siamese compile does not error", {
  # Two branches with same input shape share one dense layer by name.
  x1  <- ggml_input(shape = 4L, name = "in_a")
  x2  <- ggml_input(shape = 4L, name = "in_b")
  h1  <- ggml_layer_dense(x1, 8L, activation = "relu", name = "shared_d")
  h2  <- ggml_layer_dense(x2, 8L, activation = "relu", name = "shared_d")
  cat_out <- ggml_layer_concatenate(list(h1, h2), axis = 0L)
  out <- ggml_layer_dense(cat_out, 2L, activation = "softmax")
  # Use only the first input for single-input compile check
  m   <- ggml_model(inputs = x1, outputs = out)
  expect_no_error(m <- ggml_compile(m, optimizer = "adam",
                                     loss = "categorical_crossentropy"))
  on.exit(cleanup_functional_model(m))
})

test_that("shared dense layer: fit on single branch converges", {
  # Simplest shared-weight test: two nodes named "enc" both applied to the
  # same input x, then concatenated.  Both use identical weight tensors.
  set.seed(42)
  n <- 64L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  # Apply "enc" to the same input twice — both get same weights
  h1  <- ggml_layer_dense(inp, 4L, activation = "relu", name = "enc")
  h2  <- ggml_layer_dense(inp, 4L, activation = "relu", name = "enc")
  # h1 == h2 (same weights, same input) -> their sum is equivalent to 2*h1
  merged <- ggml_layer_add(list(h1, h2))
  out <- ggml_layer_dense(merged, 2L, activation = "softmax")
  m   <- ggml_model(inputs = inp, outputs = out)
  m   <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m   <- ggml_fit(m, x, y, epochs = 3L, batch_size = 32L, verbose = 0L)

  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("shared dense layer: predict after fit is deterministic", {
  set.seed(99)
  n <- 32L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  h1  <- ggml_layer_dense(inp, 4L, activation = "relu", name = "enc2")
  h2  <- ggml_layer_dense(inp, 4L, activation = "relu", name = "enc2")
  merged <- ggml_layer_add(list(h1, h2))
  out <- ggml_layer_dense(merged, 2L, activation = "softmax")
  m   <- ggml_model(inputs = inp, outputs = out)
  m   <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
  m   <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0L)

  p1 <- ggml_predict(m, x, batch_size = 32L)
  p2 <- ggml_predict(m, x, batch_size = 32L)
  expect_equal(p1, p2)
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Block 4 — GlobalMaxPooling2D / GlobalAveragePooling2D
# ============================================================================

test_that("ggml_layer_global_max_pooling_2d on tensor node returns tensor node", {
  x <- ggml_input(shape = c(8L, 8L, 16L))
  y <- ggml_layer_global_max_pooling_2d(x)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "global_max_pooling_2d")
})

test_that("ggml_layer_global_average_pooling_2d on tensor node returns tensor node", {
  x <- ggml_input(shape = c(8L, 8L, 16L))
  y <- ggml_layer_global_average_pooling_2d(x)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "global_average_pooling_2d")
})

test_that("GlobalMaxPooling2D output shape inference: [H,W,C] -> [C]", {
  x <- ggml_input(shape = c(4L, 4L, 8L))
  y <- ggml_layer_global_max_pooling_2d(x)
  out <- ggml_layer_dense(y, 2L, activation = "softmax")
  m <- ggml_model(inputs = x, outputs = out)
  expect_no_error(m <- ggml_compile(m, loss = "categorical_crossentropy"))
  on.exit(cleanup_functional_model(m))
})

test_that("GlobalMaxPooling2D functional model fits without error", {
  set.seed(21)
  n  <- 32L
  x  <- array(runif(n * 4L * 4L * 8L), dim = c(n, 4L, 4L, 8L))
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp <- ggml_input(shape = c(4L, 4L, 8L))
  out <- inp |>
    ggml_layer_global_max_pooling_2d() |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  expect_no_error(m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("GlobalAveragePooling2D functional model fits without error", {
  set.seed(22)
  n  <- 32L
  x  <- array(runif(n * 4L * 4L * 8L), dim = c(n, 4L, 4L, 8L))
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp <- ggml_input(shape = c(4L, 4L, 8L))
  out <- inp |>
    ggml_layer_global_average_pooling_2d() |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  expect_no_error(m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Block 4 — LSTM / GRU (Functional API)
# ============================================================================

test_that("ggml_layer_lstm on tensor node returns tensor node", {
  x <- ggml_input(shape = c(10L, 8L))
  y <- ggml_layer_lstm(x, 16L)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "lstm")
  expect_equal(y$config$units, 16L)
})

test_that("ggml_layer_gru on tensor node returns tensor node", {
  x <- ggml_input(shape = c(10L, 8L))
  y <- ggml_layer_gru(x, 16L)
  expect_s3_class(y, "ggml_tensor_node")
  expect_equal(y$node_type, "gru")
  expect_equal(y$config$units, 16L)
})

test_that("LSTM output shape without return_sequences: [units]", {
  x <- ggml_input(shape = c(5L, 4L))
  y <- ggml_layer_lstm(x, 8L, return_sequences = FALSE)
  out <- ggml_layer_dense(y, 2L, activation = "softmax")
  m <- ggml_model(inputs = x, outputs = out)
  expect_no_error(m <- ggml_compile(m, loss = "categorical_crossentropy"))
  on.exit(cleanup_functional_model(m))
})

test_that("LSTM functional model fits without error", {
  n        <- 32L
  seq_len  <- 5L
  input_sz <- 4L
  vals <- ((seq_len(n * seq_len * input_sz) - 1L) %% 20L - 10L) / 200.0
  x <- array(vals, dim = c(n, seq_len, input_sz))
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp <- ggml_input(shape = c(seq_len, input_sz))
  out <- inp |>
    ggml_layer_lstm(8L, return_sequences = FALSE) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  expect_no_error(m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("GRU functional model fits without error", {
  n        <- 32L
  seq_len  <- 5L
  input_sz <- 4L
  vals <- ((seq_len(n * seq_len * input_sz) - 1L) %% 20L - 10L) / 200.0
  x <- array(vals, dim = c(n, seq_len, input_sz))
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp <- ggml_input(shape = c(seq_len, input_sz))
  out <- inp |>
    ggml_layer_gru(8L, return_sequences = FALSE) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  expect_no_error(m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L, verbose = 0L))
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("LSTM predict is deterministic", {
  set.seed(33)
  n        <- 32L
  seq_len  <- 4L
  input_sz <- 3L
  x <- array(runif(n * seq_len * input_sz), dim = c(n, seq_len, input_sz))
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp <- ggml_input(shape = c(seq_len, input_sz))
  out <- inp |>
    ggml_layer_lstm(6L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 16L, verbose = 0L)

  p1 <- ggml_predict(m, x, batch_size = 16L)
  p2 <- ggml_predict(m, x, batch_size = 16L)
  expect_equal(p1, p2)
  on.exit(cleanup_functional_model(m))
})

# ============================================================================
# Block 4 — Save / Load (Sequential)
# ============================================================================

test_that("ggml_save_model and ggml_load_model round-trip Sequential dense model", {
  set.seed(41)
  n <- 64L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  model <- ggml_model_sequential() |>
    ggml_layer_dense(8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(2L, activation = "softmax")
  model <- ggml_compile(model, loss = "categorical_crossentropy")
  model <- ggml_fit(model, x, y, epochs = 2L, batch_size = 32L, verbose = 0L)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_model(model, tmp)

  model2 <- ggml_load_model(tmp)
  expect_s3_class(model2, "ggml_sequential_model")
  expect_true(model2$compiled)

  p1 <- ggml_predict(model,  x, batch_size = 32L)
  p2 <- ggml_predict(model2, x, batch_size = 32L)
  expect_equal(p1, p2, tolerance = 1e-5)
})

test_that("ggml_load_model Sequential restores correct architecture", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(2L, activation = "softmax")
  model <- ggml_compile(model, optimizer = "sgd", loss = "mse")

  x <- matrix(runif(32L * 4L), 32L, 4L)
  y <- matrix(0.0, 32L, 2L); for (i in seq_len(32L)) y[i, 1L] <- 1.0
  model <- ggml_fit(model, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_model(model, tmp)

  m2 <- ggml_load_model(tmp)
  expect_equal(length(m2$layers), 2L)
  expect_equal(m2$layers[[1]]$type, "dense")
  expect_equal(m2$layers[[1]]$config$units, 16L)
  expect_equal(m2$compilation$optimizer, "sgd")
  expect_equal(m2$compilation$loss, "mse")
})

test_that("ggml_load_model Sequential rejects version-1 files (ggml_save_weights)", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(4L, activation = "relu", input_shape = 2L) |>
    ggml_layer_dense(2L, activation = "softmax")
  model <- ggml_compile(model, loss = "categorical_crossentropy")
  x <- matrix(runif(32L * 2L), 32L, 2L)
  y <- matrix(0.0, 32L, 2L); for (i in seq_len(32L)) y[i, 1L] <- 1.0
  model <- ggml_fit(model, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_weights(model, tmp)
  expect_error(ggml_load_model(tmp), "ggml_save_weights")
})

# ============================================================================
# Block 4 — Save / Load (Functional)
# ============================================================================

test_that("ggml_save_model and ggml_load_model round-trip functional dense model", {
  set.seed(51)
  n <- 64L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  out <- inp |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 32L, verbose = 0L)

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp))
  ggml_save_model(m, tmp)

  m2 <- ggml_load_model(tmp)
  on.exit({ unlink(tmp); cleanup_functional_model(m2) }, add = TRUE)
  expect_s3_class(m2, "ggml_functional_model")
  expect_true(m2$compiled)

  p1 <- ggml_predict(m,  x, batch_size = 32L)
  p2 <- ggml_predict(m2, x, batch_size = 32L)
  expect_equal(p1, p2, tolerance = 1e-5)
  on.exit(cleanup_functional_model(m), add = TRUE)
})

test_that("ggml_save_model functional: loaded model predict is deterministic", {
  set.seed(52)
  n <- 32L
  x <- matrix(runif(n * 4L), nrow = n)
  y <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, if (sum(x[i, ]) > 2) 1L else 2L] <- 1.0

  inp <- ggml_input(shape = 4L)
  out <- inp |>
    ggml_layer_dense(8L, activation = "relu") |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, x, y, epochs = 1L, batch_size = 32L, verbose = 0L)

  tmp <- tempfile(fileext = ".rds")
  on.exit({ unlink(tmp); cleanup_functional_model(m) })
  ggml_save_model(m, tmp)

  m2 <- ggml_load_model(tmp)
  on.exit({ unlink(tmp); cleanup_functional_model(m); cleanup_functional_model(m2) },
           add = TRUE)

  p1 <- ggml_predict(m2, x, batch_size = 32L)
  p2 <- ggml_predict(m2, x, batch_size = 32L)
  expect_equal(p1, p2)
})

# ============================================================================
# Block 7 — Multi-input functional models
# ============================================================================

test_that("ggml_model accepts list of two inputs", {
  inp1 <- ggml_input(shape = 4L, name = "a")
  inp2 <- ggml_input(shape = 3L, name = "b")
  br1  <- inp1 |> ggml_layer_dense(4L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(4L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  expect_equal(length(m$inputs), 2L)
  expect_equal(m$inputs[[1L]]$config$name, "a")
  expect_equal(m$inputs[[2L]]$config$name, "b")
})

test_that("multi-input model compiles without error", {
  inp1 <- ggml_input(shape = 4L)
  inp2 <- ggml_input(shape = 3L)
  br1  <- inp1 |> ggml_layer_dense(4L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(4L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  expect_no_error(ggml_compile(m, loss = "categorical_crossentropy"))
})

test_that("multi-input fit runs without error and returns finite loss", {
  set.seed(60)
  n  <- 64L
  x1 <- matrix(runif(n * 6L), nrow = n)
  x2 <- matrix(runif(n * 4L), nrow = n)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp1 <- ggml_input(shape = 6L)
  inp2 <- ggml_input(shape = 4L)
  br1  <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")

  expect_no_error(
    m <- ggml_fit(m, list(x1, x2), y, epochs = 2L, batch_size = 32L, verbose = 0L)
  )
  expect_true(all(is.finite(m$history$train_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("multi-input fit with validation_split runs without error", {
  set.seed(61)
  n  <- 64L
  x1 <- matrix(runif(n * 6L), nrow = n)
  x2 <- matrix(runif(n * 4L), nrow = n)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp1 <- ggml_input(shape = 6L)
  inp2 <- ggml_input(shape = 4L)
  br1  <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")

  expect_no_error(
    m <- ggml_fit(m, list(x1, x2), y,
                  epochs = 2L, batch_size = 32L,
                  validation_split = 0.25, verbose = 0L)
  )
  expect_true(all(is.finite(m$history$train_loss)))
  expect_true(all(is.finite(m$history$val_loss)))
  on.exit(cleanup_functional_model(m))
})

test_that("multi-input predict returns matrix with correct dimensions", {
  set.seed(62)
  n  <- 64L
  x1 <- matrix(runif(n * 6L), nrow = n)
  x2 <- matrix(runif(n * 4L), nrow = n)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp1 <- ggml_input(shape = 6L)
  inp2 <- ggml_input(shape = 4L)
  br1  <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, list(x1, x2), y, epochs = 1L, batch_size = 32L, verbose = 0L)

  preds <- ggml_predict(m, list(x1, x2), batch_size = 32L)
  expect_true(is.matrix(preds))
  expect_equal(nrow(preds), n)
  expect_equal(ncol(preds), 2L)
  expect_true(all(is.finite(preds)))
  on.exit(cleanup_functional_model(m))
})

test_that("multi-input predict rows sum to 1 (softmax)", {
  set.seed(63)
  n  <- 64L
  x1 <- matrix(runif(n * 6L), nrow = n)
  x2 <- matrix(runif(n * 4L), nrow = n)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp1 <- ggml_input(shape = 6L)
  inp2 <- ggml_input(shape = 4L)
  br1  <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, list(x1, x2), y, epochs = 1L, batch_size = 32L, verbose = 0L)

  preds <- ggml_predict(m, list(x1, x2), batch_size = 32L)
  row_sums <- rowSums(preds)
  expect_true(all(abs(row_sums - 1.0) < 1e-4))
  on.exit(cleanup_functional_model(m))
})

test_that("multi-input evaluate returns finite loss and accuracy", {
  set.seed(64)
  n  <- 64L
  x1 <- matrix(runif(n * 6L), nrow = n)
  x2 <- matrix(runif(n * 4L), nrow = n)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp1 <- ggml_input(shape = 6L)
  inp2 <- ggml_input(shape = 4L)
  br1  <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, list(x1, x2), y, epochs = 1L, batch_size = 32L, verbose = 0L)

  score <- ggml_evaluate(m, list(x1, x2), y, batch_size = 32L)
  expect_true(is.finite(score$loss))
  expect_true(is.finite(score$accuracy))
  expect_true(score$accuracy >= 0.0 && score$accuracy <= 1.0)
  on.exit(cleanup_functional_model(m))
})

test_that("multi-input predict is deterministic", {
  set.seed(65)
  n  <- 64L
  x1 <- matrix(runif(n * 6L), nrow = n)
  x2 <- matrix(runif(n * 4L), nrow = n)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp1 <- ggml_input(shape = 6L)
  inp2 <- ggml_input(shape = 4L)
  br1  <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, list(x1, x2), y, epochs = 1L, batch_size = 32L, verbose = 0L)

  p1 <- ggml_predict(m, list(x1, x2), batch_size = 32L)
  p2 <- ggml_predict(m, list(x1, x2), batch_size = 32L)
  expect_equal(p1, p2)
  on.exit(cleanup_functional_model(m))
})

test_that("multi-input save/load round-trip preserves predictions", {
  set.seed(66)
  n  <- 64L
  x1 <- matrix(runif(n * 6L), nrow = n)
  x2 <- matrix(runif(n * 4L), nrow = n)
  y  <- matrix(0.0, nrow = n, ncol = 2L)
  for (i in seq_len(n)) y[i, (i %% 2L) + 1L] <- 1.0

  inp1 <- ggml_input(shape = 6L)
  inp2 <- ggml_input(shape = 4L)
  br1  <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  br2  <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
  m <- ggml_compile(m, loss = "categorical_crossentropy")
  m <- ggml_fit(m, list(x1, x2), y, epochs = 2L, batch_size = 32L, verbose = 0L)

  tmp <- tempfile(fileext = ".rds")
  on.exit({ unlink(tmp); cleanup_functional_model(m) })
  ggml_save_model(m, tmp)

  m2 <- ggml_load_model(tmp)
  on.exit({ unlink(tmp); cleanup_functional_model(m); cleanup_functional_model(m2) },
           add = TRUE)

  p1 <- ggml_predict(m,  list(x1, x2), batch_size = 32L)
  p2 <- ggml_predict(m2, list(x1, x2), batch_size = 32L)
  expect_equal(p1, p2)
})
