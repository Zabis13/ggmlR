# Tests for functional API layers: batch_norm, embedding, lstm, gru

# ============================================================================
# ggml_batch_norm (standalone layer constructor)
# ============================================================================

test_that("ggml_batch_norm creates ggml_layer with correct type", {
  l <- ggml_batch_norm()
  expect_s3_class(l, "ggml_layer")
  expect_equal(l$node_type, "batch_norm")
})

test_that("ggml_batch_norm with custom eps", {
  l <- ggml_batch_norm(eps = 1e-3)
  expect_equal(l$config$eps, 1e-3)
})

# ============================================================================
# ggml_embedding (standalone layer constructor)
# ============================================================================

test_that("ggml_embedding creates ggml_layer", {
  l <- ggml_embedding(vocab_size = 100L, dim = 32L)
  expect_s3_class(l, "ggml_layer")
  expect_equal(l$node_type, "embedding")
  expect_equal(l$config$vocab_size, 100L)
  expect_equal(l$config$dim, 32L)
})

# ============================================================================
# ggml_lstm (standalone layer constructor)
# ============================================================================

test_that("ggml_lstm creates ggml_layer", {
  l <- ggml_lstm(units = 32L)
  expect_s3_class(l, "ggml_layer")
  expect_equal(l$node_type, "lstm")
  expect_equal(l$config$units, 32L)
  expect_false(l$config$return_sequences)
})

test_that("ggml_lstm return_sequences option", {
  l <- ggml_lstm(units = 16L, return_sequences = TRUE)
  expect_true(l$config$return_sequences)
})

# ============================================================================
# ggml_gru (standalone layer constructor)
# ============================================================================

test_that("ggml_gru creates ggml_layer", {
  l <- ggml_gru(units = 32L)
  expect_s3_class(l, "ggml_layer")
  expect_equal(l$node_type, "gru")
  expect_equal(l$config$units, 32L)
})

test_that("ggml_gru return_sequences option", {
  l <- ggml_gru(units = 16L, return_sequences = TRUE)
  expect_true(l$config$return_sequences)
})

# ============================================================================
# nn_topo_sort
# ============================================================================

test_that("nn_topo_sort returns correct topological order", {
  x   <- ggml_input(shape = 4L)
  h   <- x |> ggml_layer_dense(8L)
  out <- h |> ggml_layer_dense(2L)
  m   <- ggml_model(inputs = x, outputs = out)

  order <- nn_topo_sort(m$outputs)
  expect_true(is.list(order))
  expect_true(length(order) >= 3)  # at least input + 2 dense
})

# ============================================================================
# ggml_layer_conv_1d / ggml_layer_conv_2d in functional API
# ============================================================================

test_that("ggml_layer_conv_1d creates tensor node in functional API", {
  x <- ggml_input(shape = c(10L, 1L))  # length x channels
  out <- x |> ggml_layer_conv_1d(filters = 4L, kernel_size = 3L)
  expect_s3_class(out, "ggml_tensor_node")
})

test_that("ggml_layer_conv_2d creates tensor node in functional API", {
  x <- ggml_input(shape = c(8L, 8L, 1L))
  out <- x |> ggml_layer_conv_2d(filters = 4L, kernel_size = 3L)
  expect_s3_class(out, "ggml_tensor_node")
})
