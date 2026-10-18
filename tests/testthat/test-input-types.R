# Input element types, and the sequential embedding that depends on them.
#
# Every path that feeds a model writes through ggml_backend_tensor_set_data(),
# which converts there: integers into float tensors, whole numbers into I32
# index tensors, and NA or fractions into an error.

seq_tokens <- function(n = 16L, seq_len = 5L, vocab = 10L, seed = 1L) {
  set.seed(seed)
  list(x = matrix(sample(0:(vocab - 1L), n * seq_len, replace = TRUE), n, seq_len),
       y = matrix(rnorm(n * 2L), n, 2L))
}

build_seq_embedding <- function(vocab = 10L, seq_len = 5L, backend = "auto") {
  m <- ggml_model_sequential() |>
    ggml_layer_embedding(vocab, 4L, input_shape = seq_len) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L)
  ggml_compile(m, optimizer = "adam", loss = "mse", backend = backend)
}

embedding_table <- function(m)
  ggml_backend_tensor_get_data(m$layers[[1L]]$weights$weight)

# ============================================================================
# sequential embedding
# ============================================================================

test_that("a sequential embedding trains, from an integer index matrix", {
  d <- seq_tokens()
  m <- ggml_fit(build_seq_embedding(), d$x, d$y, epochs = 1L, batch_size = 16L,
                verbose = 0L)
  e1 <- embedding_table(m)
  expect_length(e1, 10L * 4L)
  m <- ggml_fit(m, d$x, d$y, epochs = 5L, batch_size = 16L, verbose = 0L)
  expect_false(isTRUE(all.equal(e1, embedding_table(m))))
})

test_that("freezing a sequential embedding keeps its table", {
  d <- seq_tokens()
  m <- ggml_fit(build_seq_embedding(), d$x, d$y, epochs = 1L, batch_size = 16L,
                verbose = 0L)
  e1 <- embedding_table(m)
  m  <- ggml_freeze_weights(m, from = 1L, to = 1L)
  m  <- ggml_fit(m, d$x, d$y, epochs = 5L, batch_size = 16L, verbose = 0L)
  expect_equal(embedding_table(m), e1)
})

test_that("a sequential embedding survives save and load", {
  d <- seq_tokens()
  m <- ggml_fit(build_seq_embedding(), d$x, d$y, epochs = 2L, batch_size = 16L,
                verbose = 0L)
  p <- ggml_predict(m, d$x, batch_size = 16L)
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f))
  ggml_save_model(m, f)
  expect_equal(ggml_predict(ggml_load_model(f), d$x, batch_size = 16L), p,
               tolerance = 1e-5)
})

test_that("a sequential embedding trains through ggml_trainer", {
  d  <- seq_tokens()
  m  <- build_seq_embedding()
  tr <- ggml_trainer(m, batch_size = 16L)
  on.exit(tr$free())
  tr$step(d$x, d$y)
  e1 <- embedding_table(tr$model())
  for (i in 1:5) tr$step(d$x, d$y)
  expect_false(isTRUE(all.equal(e1, embedding_table(tr$model()))))
  expect_equal(dim(tr$predict(d$x[1:3, , drop = FALSE])), c(3L, 2L))
})

test_that("an embedding anywhere but first is refused with a reason", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(5L, input_shape = 5L) |>
    ggml_layer_embedding(10L, 4L) |>
    ggml_layer_flatten() |>
    ggml_layer_dense(2L)
  m <- ggml_compile(m, optimizer = "adam", loss = "mse")
  d <- seq_tokens()
  expect_error(ggml_fit(m, d$x, d$y, epochs = 1L, batch_size = 16L, verbose = 0L),
               "must be the first layer")
})

# ============================================================================
# index validation
# ============================================================================

test_that("indices with NA or fractions stop instead of being truncated", {
  d <- seq_tokens()
  m <- build_seq_embedding()

  x_frac <- d$x + 0.0; x_frac[1L, 1L] <- 1.5
  expect_error(ggml_fit(m, x_frac, d$y, epochs = 1L, batch_size = 16L, verbose = 0L),
               "whole numbers")

  x_na <- d$x; x_na[2L, 3L] <- NA
  expect_error(ggml_fit(m, x_na, d$y, epochs = 1L, batch_size = 16L, verbose = 0L),
               "NA")
})

test_that("a functional int32 input refuses fractions too", {
  inp <- ggml_input(shape = 5L, dtype = "int32")
  out <- inp |> ggml_layer_embedding(10L, 4L) |> ggml_layer_flatten() |>
    ggml_layer_dense(2L)
  m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                    optimizer = "adam", loss = "mse")
  d <- seq_tokens()
  x_frac <- d$x + 0.0; x_frac[1L, 1L] <- 2.5
  expect_error(ggml_fit(m, x_frac, d$y, epochs = 1L, batch_size = 16L, verbose = 0L),
               "whole numbers")
})

# ============================================================================
# integer data into float inputs
# ============================================================================

test_that("an integer matrix trains and predicts like the same doubles", {
  set.seed(3)
  xi <- matrix(sample(0:9, 64L, replace = TRUE), 16L, 4L)
  xd <- xi + 0.0
  y  <- matrix(rnorm(32L), 16L, 2L)

  run <- function(x) {
    set.seed(4)
    m <- ggml_model_sequential() |> ggml_layer_dense(2L, input_shape = 4L)
    m <- ggml_compile(m, optimizer = "adam", loss = "mse")
    m <- ggml_fit(m, x, y, epochs = 2L, batch_size = 16L, verbose = 0L)
    ggml_predict(m, x, batch_size = 16L)
  }
  expect_equal(run(xi), run(xd))
})

test_that("ggml_trainer takes integer batches", {
  set.seed(5)
  xi <- matrix(sample(0:9, 64L, replace = TRUE), 16L, 4L)
  y  <- matrix(rnorm(32L), 16L, 2L)
  m  <- ggml_model_sequential() |> ggml_layer_dense(2L, input_shape = 4L)
  m  <- ggml_compile(m, optimizer = "adam", loss = "mse")
  tr <- ggml_trainer(m, batch_size = 16L)
  on.exit(tr$free())
  expect_true(is.finite(tr$step(xi, y)))
  expect_equal(dim(tr$predict(xi)), c(16L, 2L))
})

# ============================================================================
# vocabulary range
# ============================================================================

test_that("an index outside the vocabulary stops fit, predict and the trainer", {
  d <- seq_tokens()
  m <- ggml_fit(build_seq_embedding(), d$x, d$y, epochs = 1L, batch_size = 16L,
                verbose = 0L)
  x_hi <- d$x; x_hi[1L, 1L] <- 10L        # vocab is 10: rows 0..9
  x_lo <- d$x; x_lo[1L, 1L] <- -1L

  expect_error(ggml_fit(m, x_hi, d$y, epochs = 1L, batch_size = 16L, verbose = 0L),
               "out of range")
  expect_error(ggml_predict(m, x_hi, batch_size = 16L), "out of range")
  expect_error(ggml_predict(m, x_lo, batch_size = 16L), "out of range")

  tr <- ggml_trainer(m, batch_size = 16L)
  on.exit(tr$free())
  expect_error(tr$step(x_hi, d$y), "out of range")
  expect_error(tr$predict(x_hi), "out of range")
})

test_that("a functional int32 input is range-checked too", {
  inp <- ggml_input(shape = 5L, dtype = "int32")
  out <- inp |> ggml_layer_embedding(10L, 4L) |> ggml_layer_flatten() |>
    ggml_layer_dense(2L)
  m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                    optimizer = "adam", loss = "mse")
  d <- seq_tokens()
  x_hi <- d$x; x_hi[3L, 2L] <- 12L
  expect_error(ggml_fit(m, x_hi, d$y, epochs = 1L, batch_size = 16L, verbose = 0L),
               "out of range")
})

# ============================================================================
# sequential and functional embeddings are the same computation
# ============================================================================

# Same weights, same tokens: any difference is an axis-order bug in one of the
# two lookups ([dim, seq_len, N]), which training alone would only show as a
# worse model. 20 rows at batch 16 also covers the functional predict padding
# its last batch with index 0, which must pass the range check and be dropped.
# Both on the CPU: on Vulkan the tiled matmul keeps f16 tiles for some batch
# shapes, so the sequential remainder (a graph of 4 rows) and the functional
# padded batch (16 rows) differ by ~2e-5 in the last rows -- the backend's
# batch-shape noise, not the lookup this test is about.
test_that("sequential and functional embeddings agree on the same weights", {
  d <- seq_tokens(n = 20L)
  ms <- ggml_fit(build_seq_embedding(backend = "cpu"), d$x[1:16, ], d$y[1:16, ],
                 epochs = 2L, batch_size = 16L, verbose = 0L)
  E <- embedding_table(ms)
  W <- ggml_backend_tensor_get_data(ms$layers[[3L]]$weights$weight)
  b <- ggml_backend_tensor_get_data(ms$layers[[3L]]$weights$bias)

  inp <- ggml_input(shape = 5L, dtype = "int32")
  emb <- ggml_layer_embedding(inp, 10L, 4L)
  out <- emb |> ggml_layer_flatten() |> ggml_layer_dense(2L)
  mf  <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                      optimizer = "adam", loss = "mse", backend = "cpu")
  mf$node_weights_data <- setNames(list(list(weight = E),
                                        list(weight = W, bias = b)),
                                   c(emb$id, out$id))

  ps <- ggml_predict(ms, d$x, batch_size = 16L)
  pf <- ggml_predict(mf, d$x, batch_size = 16L)
  expect_equal(dim(pf), c(20L, 2L))
  expect_equal(pf, ps, tolerance = 1e-5)
})

test_that("integers go into a float tensor directly, NA as NaN", {
  be  <- ggml_backend_cpu_init()
  ctx <- ggml_init(1024L * 1024L, no_alloc = TRUE)
  t   <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  on.exit({ ggml_backend_buffer_free(buf); ggml_free(ctx) })

  ggml_backend_tensor_set_data(t, c(1L, -2L, NA, 7L))
  expect_equal(ggml_backend_tensor_get_data(t), c(1, -2, NaN, 7))
  ggml_backend_tensor_set_data(t, c(TRUE, FALSE, TRUE, NA))
  expect_equal(ggml_backend_tensor_get_data(t), c(1, 0, 1, NaN))
})
