# return_sequences = TRUE must produce [units, seq_len, N], not [units, seq*N].
#
# The regression: the per-step hidden states were concatenated straight along
# dim 1, which folds the batch into the sequence axis. Nothing failed at build
# time, and a consumer reading only ne[0] (attention) still worked -- so the
# fault surfaced only downstream, where sequence_pooling reduced the wrong axis
# and its ggml_reshape_2d() aborted the process on the element count
# (GGML_ASSERT(ggml_nelements(a) == ne0*ne1), ggml-ops-builders.c).
#
# An abort takes the whole R session down rather than failing a test, so these
# are guard rails: if the stacking regresses, the suite dies here instead of
# somewhere unrelated.

cleanup_fn_model <- function(m) {
  ggml_backend_sched_free(m$compilation$sched)
  ggml_backend_free(m$compilation$backend)
  if (!is.null(m$compilation$cpu_backend)) {
    ggml_backend_free(m$compilation$cpu_backend)
  }
}

# ============================================================================
# The stacking helper itself
# ============================================================================

test_that("nn_stack_time_steps builds [units, seq_len, N]", {
  units <- 3L; nbatch <- 2L; nsteps <- 4L

  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Step t is filled with the value t, so the layout is readable in the output.
  steps <- lapply(seq_len(nsteps), function(i) {
    t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, units, nbatch)
    ggml_set_f32(t, as.numeric(rep(i, units * nbatch)))
    ggml_set_input(t)
    t
  })

  out <- ggmlR:::nn_stack_time_steps(ctx, steps, units, nbatch)
  ggml_set_output(out)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_graph_compute(backend, ggml_build_forward_expand(ctx, out))

  expect_equal(ggml_tensor_shape(out)[1:3], c(units, nsteps, nbatch))
  expect_equal(ggml_nelements(out), units * nsteps * nbatch)

  # Steps advance along the sequence axis: the first `units` values are step 1,
  # the next `units` step 2, and so on. A plain dim-1 concat would instead put
  # a whole batch of step 1 first.
  expect_equal(head(ggml_get_f32(out), units * nsteps),
               as.numeric(rep(seq_len(nsteps), each = units)),
               tolerance = 1e-6)
})

test_that("mean pooling over the stacked steps averages the sequence", {
  units <- 3L; nbatch <- 2L; nsteps <- 4L

  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  steps <- lapply(seq_len(nsteps), function(i) {
    t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, units, nbatch)
    ggml_set_f32(t, as.numeric(rep(i, units * nbatch)))
    ggml_set_input(t)
    t
  })
  stacked <- ggmlR:::nn_stack_time_steps(ctx, steps, units, nbatch)

  # Exactly what the "mean" branch of sequence_pooling does.
  tt  <- ggml_cont(ctx, ggml_transpose(ctx, stacked))
  out <- ggml_reshape_2d(ctx, ggml_mean(ctx, tt), units, nbatch)
  ggml_set_output(out)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)
  ggml_backend_graph_compute(backend, ggml_build_forward_expand(ctx, out))

  expect_equal(ggml_tensor_shape(out)[1:2], c(units, nbatch))
  # mean(1..4) = 2.5 in every cell.
  expect_equal(ggml_get_f32(out), rep(2.5, units * nbatch), tolerance = 1e-5)
})

# ============================================================================
# End to end, both layers and both downstream consumers
# ============================================================================

rnn_fixture <- function() {
  set.seed(1L)
  n <- 32L; vocab <- 20L; slen <- 6L; dim <- 16L
  list(n = n, vocab = vocab, slen = slen, dim = dim,
       x = matrix(sample(0L:(vocab - 1L), n * slen, replace = TRUE), nrow = n),
       y = local({
         yy <- matrix(0, nrow = n, ncol = 2L)
         yy[cbind(seq_len(n), sample(1:2, n, replace = TRUE))] <- 1
         yy
       }))
}

fit_seq_model <- function(fx, make_rnn, tail = c("pooling", "attention")) {
  tail <- match.arg(tail)
  inp <- ggml_input(shape = fx$slen, dtype = "int32")
  emb <- inp |> ggml_layer_embedding(vocab_size = fx$vocab, dim = fx$dim)
  h   <- make_rnn(emb)
  h   <- if (tail == "attention") {
    h |> ggml_layer_attention(d_model = fx$dim, n_heads = 2L) |>
      ggml_layer_sequence_pooling(mode = "mean")
  } else {
    h |> ggml_layer_sequence_pooling(mode = "mean")
  }
  out <- h |> ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                    optimizer = "adam", loss = "categorical_crossentropy")
  ggml_fit(m, fx$x, fx$y, epochs = 2L, batch_size = 8L, verbose = 0)
}

test_that("LSTM return_sequences feeds sequence_pooling", {
  fx <- rnn_fixture()
  fit <- fit_seq_model(fx, function(e) {
    e |> ggml_layer_lstm(fx$dim, return_sequences = TRUE)
  }, "pooling")
  on.exit(cleanup_fn_model(fit))

  expect_true(all(is.finite(fit$history$train_loss)))
  expect_equal(dim(ggml_predict(fit, fx$x)), c(fx$n, 2L))
})

test_that("GRU return_sequences feeds sequence_pooling", {
  # The same defect lived in both layers; only the downstream consumer differed,
  # which is why an early check that paired LSTM with pooling and GRU with
  # attention looked like an LSTM-only bug.
  fx <- rnn_fixture()
  fit <- fit_seq_model(fx, function(e) {
    e |> ggml_layer_gru(fx$dim, return_sequences = TRUE)
  }, "pooling")
  on.exit(cleanup_fn_model(fit))

  expect_true(all(is.finite(fit$history$train_loss)))
  expect_equal(dim(ggml_predict(fit, fx$x)), c(fx$n, 2L))
})

test_that("LSTM return_sequences feeds attention", {
  fx <- rnn_fixture()
  fit <- fit_seq_model(fx, function(e) {
    e |> ggml_layer_lstm(fx$dim, return_sequences = TRUE)
  }, "attention")
  on.exit(cleanup_fn_model(fit))

  expect_true(all(is.finite(fit$history$train_loss)))
  expect_equal(dim(ggml_predict(fit, fx$x)), c(fx$n, 2L))
})

test_that("return_sequences = FALSE is unaffected", {
  fx <- rnn_fixture()
  inp <- ggml_input(shape = fx$slen, dtype = "int32")
  out <- inp |>
    ggml_layer_embedding(vocab_size = fx$vocab, dim = fx$dim) |>
    ggml_layer_lstm(8L, return_sequences = FALSE) |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                    optimizer = "adam", loss = "categorical_crossentropy")
  fit <- ggml_fit(m, fx$x, fx$y, epochs = 2L, batch_size = 8L, verbose = 0)
  on.exit(cleanup_fn_model(fit))

  expect_true(all(is.finite(fit$history$train_loss)))
})

test_that("the declared shape matches what the stack produces", {
  fx <- rnn_fixture()
  units <- 8L

  inp <- ggml_input(shape = fx$slen, dtype = "int32")
  emb <- inp |> ggml_layer_embedding(vocab_size = fx$vocab, dim = fx$dim)
  rnn <- emb |> ggml_layer_lstm(units, return_sequences = TRUE)

  emb_sh <- ggmlR:::nn_functional_output_shape(emb, list(fx$slen))
  expect_equal(ggmlR:::nn_functional_output_shape(rnn, list(emb_sh)),
               c(fx$slen, units))
})
