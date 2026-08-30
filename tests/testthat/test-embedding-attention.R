# End-to-end: embedding feeding a sequence layer (attention, GRU).
#
# The embedding build emits ggml [dim, seq_len, N]. Under the package's
# R-to-ggml rule -- an R shape c(a, b) is the tensor [b, a, N] -- the R shape
# describing that is c(seq_len, dim), the same layout ggml_input(c(seq, feat))
# produces. The shape function used to report c(dim, seq_len) instead, which
# inverted the two axes for every consumer that reads the feature width off
# psh[2]:
#   * attention rejected a correct graph ("d_model is X but the input has Y")
#   * GRU/LSTM sized their weights off the embedding axis instead of the
#     sequence axis -- silently: the model compiled, trained and predicted.
# These tests pin the convention so it cannot drift back.

cleanup_fn_model <- function(m) {
  ggml_backend_sched_free(m$compilation$sched)
  ggml_backend_free(m$compilation$backend)
  if (!is.null(m$compilation$cpu_backend)) {
    ggml_backend_free(m$compilation$cpu_backend)
  }
}

# ============================================================================
# The convention itself
# ============================================================================

test_that("an embedding reports the same shape convention as a sequence input", {
  slen <- 5L
  dim     <- 17L        # deliberately different from slen

  inp <- ggml_input(shape = slen, dtype = "int32")
  emb <- inp |> ggml_layer_embedding(vocab_size = 40L, dim = dim)

  expect_equal(ggmlR:::nn_functional_output_shape(emb, list(slen)),
               c(slen, dim))

  seq_in <- ggml_input(shape = c(slen, dim))
  expect_equal(ggmlR:::nn_functional_output_shape(seq_in, list(NULL)),
               c(slen, dim))
})

test_that("the sequential shape pass agrees with the functional one", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(6L, input_shape = 4L) |>
    ggml_layer_embedding(vocab_size = 40L, dim = 17L)
  model <- ggmlR:::nn_infer_shapes(model)

  # dense gives 6 features, which the embedding treats as its sequence length.
  expect_equal(model$layers[[2]]$output_shape, c(6L, 17L))
})

# ============================================================================
# attention accepts the embedding directly
# ============================================================================

test_that("embedding feeds attention without a permute", {
  slen <- 6L
  d_model <- 16L

  inp <- ggml_input(shape = slen, dtype = "int32")
  emb <- inp |> ggml_layer_embedding(vocab_size = 20L, dim = d_model)
  att <- emb |> ggml_layer_attention(d_model = d_model, n_heads = 2L)

  expect_equal(
    ggmlR:::nn_functional_output_shape(
      att, list(ggmlR:::nn_functional_output_shape(emb, list(slen)))),
    c(slen, d_model))
})

test_that("an embedding -> attention model compiles and trains", {
  set.seed(1L)
  vocab <- 20L; slen <- 6L; d_model <- 16L; n_cls <- 2L

  inp <- ggml_input(shape = slen, dtype = "int32")
  out <- inp |>
    ggml_layer_embedding(vocab_size = vocab, dim = d_model) |>
    ggml_layer_attention(d_model = d_model, n_heads = 2L) |>
    ggml_layer_sequence_pooling(mode = "mean") |>
    ggml_layer_dense(n_cls, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  # Learnable task: the class says whether token 1 occurs in the sequence.
  n <- 64L
  x <- matrix(sample(2:(vocab - 1L), n * slen, replace = TRUE), nrow = n)
  lab <- sample(c(0L, 1L), n, replace = TRUE)
  for (i in seq_len(n)) if (lab[i] == 1L) x[i, sample(slen, 1)] <- 1L
  y <- matrix(0, nrow = n, ncol = n_cls)
  y[cbind(seq_len(n), lab + 1L)] <- 1

  fit <- ggml_fit(m, x, y, epochs = 40L, batch_size = 8L, verbose = 0)
  on.exit(cleanup_fn_model(fit))

  tl <- fit$history$train_loss
  expect_true(all(is.finite(tl)))
  expect_lt(tl[length(tl)], tl[1])

  preds <- ggml_predict(fit, x)
  expect_equal(dim(preds), c(n, n_cls))
})

# ============================================================================
# GRU sizes its weights off the sequence axis, not the embedding axis
# ============================================================================

test_that("GRU after an embedding reads the feature width, not the length", {
  # The regression this guards: with the axes inverted GRU built W_zh as
  # [seq_len x 2*units] instead of [dim x 2*units] and unrolled over the
  # embedding axis. Nothing failed -- the model compiled and trained -- so only
  # an explicit shape assertion catches it. slen and dim differ so the two
  # readings cannot coincide.
  slen <- 5L
  dim     <- 17L
  units   <- 4L

  inp <- ggml_input(shape = slen, dtype = "int32")
  emb <- inp |> ggml_layer_embedding(vocab_size = 40L, dim = dim)

  psh <- ggmlR:::nn_functional_output_shape(emb, list(slen))
  expect_equal(psh[1], slen)   # GRU reads this as the sequence length
  expect_equal(psh[2], dim)       # ... and this as the input width

  gru <- emb |> ggml_layer_gru(units, return_sequences = TRUE)
  expect_equal(ggmlR:::nn_functional_output_shape(gru, list(psh)),
               c(slen, units))
})

test_that("an embedding -> GRU model compiles and trains", {
  set.seed(1L)
  vocab <- 20L; slen <- 5L; dim <- 17L; units <- 8L; n_cls <- 2L

  inp <- ggml_input(shape = slen, dtype = "int32")
  out <- inp |>
    ggml_layer_embedding(vocab_size = vocab, dim = dim) |>
    ggml_layer_gru(units, return_sequences = FALSE) |>
    ggml_layer_dense(n_cls, activation = "softmax")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")

  n <- 48L
  x <- matrix(sample(0L:(vocab - 1L), n * slen, replace = TRUE), nrow = n)
  y <- matrix(0, nrow = n, ncol = n_cls)
  y[cbind(seq_len(n), sample(c(1L, 2L), n, replace = TRUE))] <- 1

  fit <- ggml_fit(m, x, y, epochs = 5L, batch_size = 8L, verbose = 0)
  on.exit(cleanup_fn_model(fit))

  expect_true(all(is.finite(fit$history$train_loss)))
  expect_equal(dim(ggml_predict(fit, x)), c(n, n_cls))
})
