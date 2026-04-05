# Tests for print/summary S3 methods

# ============================================================================
# Autograd print methods
# ============================================================================

test_that("print.ag_tensor works", {
  x <- ag_tensor(matrix(1:6, 2, 3))
  out <- capture.output(print(x))
  expect_true(any(grepl("ag_tensor", out)))
})

test_that("print.ag_optimizer_adam works", {
  w <- ag_param(matrix(1:4, 2, 2))
  opt <- optimizer_adam(list(w), lr = 0.01)
  out <- capture.output(print(opt))
  expect_true(length(out) > 0)
})

test_that("print.ag_optimizer_sgd works", {
  w <- ag_param(matrix(1:4, 2, 2))
  opt <- optimizer_sgd(list(w), lr = 0.01)
  out <- capture.output(print(opt))
  expect_true(length(out) > 0)
})

test_that("print.ag_sequential works", {
  l1 <- ag_linear(4, 8)
  l2 <- ag_linear(8, 2)
  seq_model <- ag_sequential(l1, l2)
  out <- capture.output(print(seq_model))
  expect_true(length(out) > 0)
})

test_that("print.ag_dataloader works", {
  # ag_dataloader expects col-major: [features, samples]
  x <- matrix(rnorm(40), 4, 10)
  y <- matrix(rnorm(10), 1, 10)
  dl <- ag_dataloader(x, y, batch_size = 5)
  out <- capture.output(print(dl))
  expect_true(length(out) > 0)
})

test_that("print.lr_scheduler_step works", {
  w <- ag_param(matrix(1:4, 2, 2))
  opt <- optimizer_adam(list(w), lr = 0.01)
  sched <- lr_scheduler_step(opt, step_size = 10, gamma = 0.1)
  out <- capture.output(print(sched))
  expect_true(length(out) > 0)
})

test_that("print.lr_scheduler_cosine works", {
  w <- ag_param(matrix(1:4, 2, 2))
  opt <- optimizer_adam(list(w), lr = 0.01)
  sched <- lr_scheduler_cosine(opt, T_max = 100)
  out <- capture.output(print(sched))
  expect_true(length(out) > 0)
})

# ============================================================================
# Sequential model print/summary
# ============================================================================

test_that("print.ggml_sequential_model works", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")
  out <- capture.output(print(m))
  expect_true(length(out) > 0)
})

test_that("summary.ggml_sequential_model works", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(units = 8L, activation = "relu", input_shape = 4L) |>
    ggml_layer_dense(units = 2L, activation = "softmax")
  out <- capture.output(summary(m))
  expect_true(length(out) > 0)
})

# ============================================================================
# Functional model print
# ============================================================================

test_that("print.ggml_functional_model works", {
  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_dense(2L)
  m   <- ggml_model(inputs = x, outputs = out)
  outp <- capture.output(print(m))
  expect_true(length(outp) > 0)
})

# ============================================================================
# Training history
# ============================================================================

test_that("print.ggml_history works", {
  h <- structure(list(
    epochs = 1:3,
    train_loss = c(1.0, 0.5, 0.2),
    train_accuracy = c(0.3, 0.6, 0.9),
    val_loss = c(1.1, 0.6, 0.3),
    val_accuracy = c(0.2, 0.5, 0.8)
  ), class = "ggml_history")
  out <- capture.output(print(h))
  expect_true(length(out) > 0)
})

test_that("plot.ggml_history works without error", {
  h <- structure(list(
    epochs = 1:3,
    train_loss = c(1.0, 0.5, 0.2),
    train_accuracy = c(0.3, 0.6, 0.9),
    val_loss = c(1.1, 0.6, 0.3),
    val_accuracy = c(0.2, 0.5, 0.8)
  ), class = "ggml_history")
  expect_no_error(plot(h))
})

# ============================================================================
# ONNX print
# ============================================================================

test_that("print.onnx_model works on mock object", {
  m <- structure(list(
    graph_name = "test",
    n_nodes = 5L,
    n_inputs = 1L,
    n_outputs = 1L,
    opset = 13L
  ), class = "onnx_model")
  out <- capture.output(print(m))
  expect_true(length(out) > 0)
})
