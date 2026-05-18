# Tests for ggml_dense() + ggml_apply() shared-layer workflow (C4 gap).
# These functional-API building blocks were exported without direct test
# coverage. Distinct from inline ggml_layer_dense(): a ggml_dense() layer
# object can be applied to multiple inputs sharing the same weights.

test_that("ggml_dense() builds a reusable ggml_layer object", {
  enc <- ggml_dense(8L, activation = "relu", name = "enc")
  expect_s3_class(enc, "ggml_layer")
  expect_equal(enc$node_type, "dense")
  expect_equal(enc$config$units, 8L)
  expect_equal(enc$name, "enc")
  expect_true(enc$trainable)
})

test_that("ggml_apply() requires a tensor node and a layer object", {
  enc <- ggml_dense(4L)
  expect_error(ggml_apply(42, enc), "ggml_tensor_node")

  x <- ggml_input(shape = 4L)
  expect_error(ggml_apply(x, list()), "ggml_layer")
})

test_that("ggml_apply() returns a tensor node carrying the layer sharing key", {
  x   <- ggml_input(shape = 4L)
  enc <- ggml_dense(3L, activation = "relu")
  out <- ggml_apply(x, enc)

  expect_s3_class(out, "ggml_tensor_node")
  expect_equal(out$layer_id, enc$layer_id)   # sharing key == layer identity
  expect_equal(out$node_type, "dense")
  expect_identical(out$parents[[1]], x)
})

test_that("a shared ggml_dense() layer applied to two inputs reuses one layer_id", {
  shared <- ggml_dense(5L, activation = "relu")
  x1 <- ggml_input(shape = 4L)
  x2 <- ggml_input(shape = 4L)

  o1 <- ggml_apply(x1, shared)
  o2 <- ggml_apply(x2, shared)

  # both applications must reference the SAME layer object (weight sharing)
  expect_equal(o1$layer_id, o2$layer_id)
  expect_equal(o1$layer_id, shared$layer_id)
  # but they are distinct graph nodes
  expect_false(identical(o1$id, o2$id))
})

test_that("shared single-input functional model with ggml_apply predicts", {
  set.seed(7)
  shared <- ggml_dense(2L, activation = "softmax")
  x   <- ggml_input(shape = 4L)
  out <- ggml_apply(x, shared)

  m <- ggml_model(inputs = x, outputs = out)
  m <- compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n  <- 32L
  xa <- matrix(rnorm(4 * n), n, 4)
  p  <- predict(m, xa, batch_size = 32L)
  expect_true(is.matrix(p) || is.numeric(p))
})

test_that("multi-input shared-layer model builds (predict is a known limitation)", {
  set.seed(7)
  shared <- ggml_dense(2L, activation = "softmax")
  x1 <- ggml_input(shape = 4L)
  x2 <- ggml_input(shape = 4L)
  o1 <- ggml_apply(x1, shared)
  o2 <- ggml_apply(x2, shared)

  # Model construction with two inputs sharing one layer must succeed.
  m <- ggml_model(inputs = list(x1, x2), outputs = list(o1, o2))
  expect_s3_class(m, "ggml_functional_model")
  expect_equal(o1$layer_id, o2$layer_id)

  # NOTE: predict() on a multi-input + shared-layer model currently aborts in
  # the backend ("tensor buffer not set" for the 2nd output). Single-input
  # multi-output predict works (see test-nn-functional.R); multi-INPUT shared
  # predict is an unimplemented path, not a regression. Skip until supported.
  skip("multi-input shared-layer predict not implemented (backend buffer not set)")
})
