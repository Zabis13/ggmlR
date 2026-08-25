# Tests for ggml_layer_rms_norm in both APIs. The reference is computed here in
# plain R, so a scale put on the wrong axis shows up as a numeric mismatch
# rather than as a shape error -- the way the functional batch_norm's
# [1, D, 1] reshape hid a broken sequence path behind a working flat one.

cleanup_model <- function(model) {
  cp <- model$compilation
  if (is.null(cp)) return(invisible(NULL))
  if (!is.null(cp$buffer))      ggml_backend_buffer_free(cp$buffer)
  if (!is.null(cp$ctx_weights)) ggml_free(cp$ctx_weights)
  if (!is.null(cp$sched))       ggml_backend_sched_free(cp$sched)
  if (!is.null(cp$backend))     ggml_backend_free(cp$backend)
  if (!is.null(cp$cpu_backend)) ggml_backend_free(cp$cpu_backend)
}

# RMS-normalize a row: x / sqrt(mean(x^2) + eps). gamma/beta default to 1/0.
ref_rms_norm <- function(x, eps = 1e-5) {
  x / sqrt(mean(x^2) + eps)
}

test_that("rms_norm matches a reference implementation on a flat input", {
  set.seed(3L)
  N <- 4L; D <- 8L
  x <- ggml_input(shape = D, name = "x")
  o <- x |> ggml_layer_rms_norm(name = "rn")
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  xa <- matrix(runif(N * D, 1, 3), N, D)
  # One epoch to populate the weights with gamma = 1, beta = 0.
  m  <- ggml_fit(m, xa, xa, epochs = 1L, batch_size = N, verbose = 0L)
  p  <- ggml_predict(m, xa, batch_size = N)

  for (i in seq_len(N)) {
    expect_equal(as.numeric(p[i, ]), ref_rms_norm(xa[i, ]), tolerance = 1e-2)
  }
  cleanup_model(m)
})

test_that("rms_norm normalizes along the feature axis of a sequence", {
  set.seed(5L)
  N <- 3L; S <- 4L; D <- 6L
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_rms_norm(name = "rn")
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  xa <- array(runif(N * S * D, 1, 3), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)
  m  <- ggml_fit(m, xa, y, epochs = 1L, batch_size = N, verbose = 0L)
  p  <- ggml_predict(m, xa, batch_size = N)

  # Each of the S positions is normalized over its own D features, so every
  # row of the result has RMS 1. A scale on the sequence axis would break this.
  for (i in seq_len(N)) {
    got <- matrix(p[i, ], S, D, byrow = TRUE)
    for (t in seq_len(S)) {
      expect_equal(as.numeric(got[t, ]), ref_rms_norm(xa[i, t, ]),
                   tolerance = 1e-2)
    }
  }
  cleanup_model(m)
})

test_that("rms_norm trains: the loss falls", {
  set.seed(7L)
  N <- 16L; S <- 4L; D <- 6L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_rms_norm() |> ggml_layer_dense(D, time_distributed = TRUE)
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  h <- ggml_fit(m, xa, y, epochs = 6L, batch_size = 8L, verbose = 0L)
  l <- h$history$train_loss
  expect_false(any(is.na(l)))
  expect_lt(l[[length(l)]], l[[1L]])
  cleanup_model(m)
})

test_that("rms_norm works in the sequential API and survives a save/load", {
  set.seed(11L)
  N <- 8L; D <- 6L
  d <- matrix(runif(N * D), N, D)

  m <- ggml_model_sequential() |>
    ggml_layer_dense(D, input_shape = D) |>
    ggml_layer_rms_norm() |>
    ggml_layer_dense(D)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d, d, epochs = 2L, batch_size = 8L, verbose = 0L)

  p1 <- ggml_predict(m, d, batch_size = 8L)
  f  <- tempfile(fileext = ".rds")
  on.exit(unlink(f), add = TRUE)
  ggml_save_model(m, f)
  m2 <- ggml_load_model(f)
  p2 <- ggml_predict(m2, d, batch_size = 8L)

  expect_equal(p1, p2, tolerance = 1e-6)
})

test_that("rms_norm keeps no running statistics, unlike batch_norm", {
  N <- 8L; D <- 4L
  mk <- function(kind) {
    m <- ggml_model_sequential() |> ggml_layer_dense(D, input_shape = D)
    m <- if (kind == "rms") ggml_layer_rms_norm(m) else ggml_layer_batch_norm(m)
    m <- ggml_compile(m, loss = "mse", backend = "cpu")
    d <- matrix(runif(N * D), N, D)
    ggml_fit(m, d, d, epochs = 1L, batch_size = N, verbose = 0L)
  }
  norm_layer <- function(m, type)
    m$layers[[which(vapply(m$layers, function(l) l$type == type, logical(1)))]]

  expect_null(norm_layer(mk("rms"), "rms_norm")$weights$running_mean)
  expect_false(is.null(norm_layer(mk("bn"), "batch_norm")$weights$running_mean))
})
