# Tests for ggml_layer_sequence_pooling -- collapsing c(seq_len, d_model) to
# d_model so a head can sit on an encoder.
#
# The numbers are checked through a dense head: a pooling layer on its own has
# no trainable parameters, and ggml_opt refuses to build a graph without any.
# Reading the head's weights back and applying them in R gives an exact
# reference for what the pooling must have produced.

cleanup_sp <- function(model) {
  cp <- model$compilation
  if (is.null(cp)) return(invisible(NULL))
  if (!is.null(cp$buffer))      ggml_backend_buffer_free(cp$buffer)
  if (!is.null(cp$ctx_weights)) ggml_free(cp$ctx_weights)
  if (!is.null(cp$sched))       ggml_backend_sched_free(cp$sched)
  if (!is.null(cp$backend))     ggml_backend_free(cp$backend)
  if (!is.null(cp$cpu_backend)) ggml_backend_free(cp$cpu_backend)
}

# Fit one epoch (to populate the head), then return the prediction together
# with the head's weights so the caller can rebuild it in R.
pool_and_head <- function(xa, mode, N, S, D) {
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_sequence_pooling(mode = mode) |> ggml_layer_dense(D)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  m <- ggml_fit(m, xa, matrix(0, N, D), epochs = 1L, batch_size = N,
                verbose = 0L)

  nw <- m$node_weights
  k  <- names(nw)[vapply(nw, function(e) !is.null(e$weight), logical(1))][[1L]]
  res <- list(pred = ggml_predict(m, xa, batch_size = N),
              W = matrix(ggml_backend_tensor_get_data(nw[[k]]$weight), D, D),
              b = ggml_backend_tensor_get_data(nw[[k]]$bias))
  cleanup_sp(m)
  res
}

test_that("mean pooling averages over the positions", {
  set.seed(1L)
  N <- 8L; S <- 4L; D <- 6L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  r  <- pool_and_head(xa, "mean", N, S, D)

  pooled <- t(vapply(seq_len(N),
                     function(i) colMeans(matrix(xa[i, , ], S, D)),
                     numeric(D)))
  expect_equal(r$pred, pooled %*% r$W + rep(r$b, each = N), tolerance = 1e-5)
})

test_that("first pooling takes position 1 and drops the rest", {
  set.seed(2L)
  N <- 8L; S <- 4L; D <- 6L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  r  <- pool_and_head(xa, "first", N, S, D)

  pooled <- t(vapply(seq_len(N), function(i) xa[i, 1L, ], numeric(D)))
  expect_equal(r$pred, pooled %*% r$W + rep(r$b, each = N), tolerance = 1e-5)
})

test_that("the output width is d_model, not seq_len * d_model", {
  # This is the point of pooling over flatten: the head does not grow with the
  # sequence.
  set.seed(3L)
  N <- 4L; S <- 5L; D <- 6L
  xa <- array(runif(N * S * D), dim = c(N, S, D))

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_sequence_pooling() |> ggml_layer_dense(2L)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  m <- ggml_fit(m, xa, matrix(0, N, 2L), epochs = 1L, batch_size = N,
                verbose = 0L)

  expect_equal(ncol(ggml_predict(m, xa, batch_size = N)), 2L)
  cleanup_sp(m)
})

test_that("a non-sequence input is rejected", {
  N <- 8L; D <- 6L
  x <- ggml_input(shape = D, name = "x")
  o <- x |> ggml_layer_sequence_pooling() |> ggml_layer_dense(D)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  d <- matrix(runif(N * D), N, D)
  expect_error(ggml_fit(m, d, d, epochs = 1L, batch_size = N, verbose = 0L),
               "sequence input")
})

test_that("mode is validated", {
  x <- ggml_input(shape = c(4L, 6L), name = "x")
  expect_error(ggml_layer_sequence_pooling(x, mode = "median"), "arg")
})

test_that("an encoder with a pooled head trains", {
  set.seed(5L)
  N <- 16L; S <- 4L; D <- 6L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  # Target depends on the whole sequence, so mean pooling is the right summary.
  y  <- t(vapply(seq_len(N),
                 function(i) colMeans(matrix(xa[i, , ], S, D)),
                 numeric(D)))

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = 2L) |>
    ggml_layer_sequence_pooling() |> ggml_layer_dense(D)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  l <- ggml_fit(m, xa, y, epochs = 6L, batch_size = 8L,
                verbose = 0L)$history$train_loss

  expect_false(any(is.na(l)))
  expect_lt(l[[length(l)]], l[[1L]])
  cleanup_sp(m)
})

test_that("two heads can share one pooled encoder", {
  # The ADI shape: a value head and a policy head over the same summary.
  set.seed(7L)
  N <- 8L; S <- 4L; D <- 6L; A <- 3L
  xa <- array(runif(N * S * D), dim = c(N, S, D))

  x <- ggml_input(shape = c(S, D), name = "x")
  h <- x |> ggml_layer_attention(D, n_heads = 2L) |>
    ggml_layer_sequence_pooling()
  v <- h |> ggml_layer_dense(1L, name = "value")
  p <- h |> ggml_layer_dense(A, name = "policy")
  m <- ggml_compile(ggml_model(x, list(v, p)), loss = "mse", backend = "cpu")

  l <- ggml_fit(m, xa, list(matrix(runif(N), N, 1L), matrix(runif(N * A), N, A)),
                epochs = 2L, batch_size = 4L, verbose = 0L)$history$train_loss
  expect_false(any(is.na(l)))
  cleanup_sp(m)
})
