# Tests for ggml_layer_positional_embedding -- the learned per-position vector
# that lets an order-blind stack tell one position from another.
#
# The load-bearing test is the last one: a task where the positions carry
# IDENTICAL content, so position is the only signal there is. Everything else
# (a copy task, a reversal) can be solved from the content alone, and passes
# with or without the layer.

cleanup_pe <- function(model) {
  cp <- model$compilation
  if (is.null(cp)) return(invisible(NULL))
  if (!is.null(cp$buffer))      ggml_backend_buffer_free(cp$buffer)
  if (!is.null(cp$ctx_weights)) ggml_free(cp$ctx_weights)
  if (!is.null(cp$sched))       ggml_backend_sched_free(cp$sched)
  if (!is.null(cp$backend))     ggml_backend_free(cp$backend)
  if (!is.null(cp$cpu_backend)) ggml_backend_free(cp$cpu_backend)
}

test_that("the layer keeps the sequence shape", {
  N <- 4L; S <- 5L; D <- 8L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_positional_embedding()
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  m <- ggml_fit(m, xa, y, epochs = 1L, batch_size = N, verbose = 0L)

  expect_equal(ncol(ggml_predict(m, xa, batch_size = N)), S * D)
  cleanup_pe(m)
})

test_that("a non-sequence input is rejected", {
  N <- 8L; D <- 6L
  x <- ggml_input(shape = D, name = "x")
  o <- x |> ggml_layer_positional_embedding()
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  d <- matrix(runif(N * D), N, D)
  expect_error(ggml_fit(m, d, d, epochs = 1L, batch_size = N, verbose = 0L),
               "sequence input")
})

test_that("the table holds one vector per position", {
  N <- 8L; S <- 4L; D <- 6L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_positional_embedding(name = "pe")
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  m <- ggml_fit(m, xa, y, epochs = 1L, batch_size = N, verbose = 0L)

  nw  <- m$node_weights
  hit <- names(nw)[vapply(nw, function(e) !is.null(e$pos), logical(1))]
  expect_length(hit, 1L)
  expect_equal(ggml_nelements(nw[[hit[[1L]]]]$pos), S * D)
  cleanup_pe(m)
})

test_that("the same table is added to every sample in the batch", {
  # A zero input returns the table itself, so two samples must come back equal.
  N <- 4L; S <- 3L; D <- 6L
  xa <- array(0, dim = c(N, S, D))
  y  <- matrix(0, N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_positional_embedding()
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  m <- ggml_fit(m, xa, y, epochs = 1L, batch_size = N, verbose = 0L)
  p <- ggml_predict(m, xa, batch_size = N)

  expect_equal(p[1L, ], p[2L, ], tolerance = 1e-6)
  # ... and it is not all zeros, which would pass the check above vacuously.
  expect_gt(max(abs(p[1L, ])), 0)
  cleanup_pe(m)
})

test_that("identical positions come out distinguishable", {
  # This is the property the layer exists for. Feed a sample whose positions
  # carry the SAME vector: without the layer the output rows are identical and
  # nothing downstream can tell the positions apart; with it they differ.
  # Checked directly rather than through a training run, whose outcome varies
  # with the initialization.
  N <- 4L; S <- 4L; D <- 6L
  base <- matrix(runif(N * D), N, D)
  xa <- array(0, dim = c(N, S, D))
  for (t in seq_len(S)) xa[, t, ] <- base
  y <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_positional_embedding()
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  m <- ggml_fit(m, xa, y, epochs = 1L, batch_size = N, verbose = 0L)
  got <- matrix(ggml_predict(m, xa, batch_size = N)[1L, ], S, D, byrow = TRUE)

  # Every pair of positions must differ somewhere.
  for (i in seq_len(S - 1L)) {
    for (j in (i + 1L):S) {
      expect_gt(max(abs(got[i, ] - got[j, ])), 0)
    }
  }
  # The input rows were identical, so the difference is the layer's doing.
  expect_equal(as.numeric(xa[1L, 1L, ]), as.numeric(xa[1L, 2L, ]))
  cleanup_pe(m)
})

test_that("a position-only task trains", {
  # The target scales with the position index and the content is constant
  # across positions, so this only fits because the layer supplies the order.
  set.seed(3L)
  N <- 32L; S <- 3L; D <- 4L
  base <- matrix(runif(N * D), N, D)
  xa <- array(0, dim = c(N, S, D))
  yv <- array(0, dim = c(N, S, D))
  for (t in seq_len(S)) {
    xa[, t, ] <- base
    yv[, t, ] <- base * t
  }
  y <- matrix(aperm(yv, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_positional_embedding() |>
    ggml_layer_dense(D, time_distributed = TRUE)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  l <- ggml_fit(m, xa, y, epochs = 20L, batch_size = 8L,
                verbose = 0L)$history$train_loss

  expect_false(any(is.na(l)))
  expect_lt(l[[length(l)]], l[[1L]])
  cleanup_pe(m)
})
