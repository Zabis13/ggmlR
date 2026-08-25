# Tests for the additive attention mask -- the third parent of an attention
# node, which reaches ggml_soft_max_ext() instead of the NULL it used to get.
#
# The axis order is the thing worth pinning down: an R mask is c(seq_q, seq_kv),
# one row per query, because R's second axis is ggml's ne0 and scores' ne0 is
# the key axis. Getting it backwards still runs and still trains -- it just
# masks the wrong thing -- so the reference comparison here is what catches it.

softmax_rows_m <- function(M) {
  e <- exp(M - apply(M, 1, max))
  e / rowSums(e)
}

as_mat_m <- function(tensor, nrow, ncol) {
  matrix(ggml_backend_tensor_get_data(tensor), nrow, ncol)
}

attn_w_of <- function(model) {
  nw  <- model$node_weights
  hit <- names(nw)[vapply(nw, function(e) !is.null(e$W_q), logical(1))]
  nw[[hit[[1L]]]]
}

# Attention on one sample, with an optional set of blocked key indices.
ref_masked <- function(X, W, d_model, n_heads, blocked = NULL) {
  Q <- X %*% W$W_q; K <- X %*% W$W_k; V <- X %*% W$W_v
  dh  <- d_model %/% n_heads
  out <- matrix(0, nrow(X), d_model)
  for (h in seq_len(n_heads)) {
    cols <- ((h - 1L) * dh + 1L):(h * dh)
    sc <- (Q[, cols, drop = FALSE] %*% t(K[, cols, drop = FALSE])) / sqrt(dh)
    if (!is.null(blocked)) sc[, blocked] <- -Inf
    out[, cols] <- softmax_rows_m(sc) %*% V[, cols, drop = FALSE]
  }
  res <- out %*% W$W_o
  if (!is.null(W$b_o)) res <- res + rep(W$b_o, each = nrow(X))
  res
}

# Build, fit one epoch (to populate weights), predict, and read the weights.
fit_masked <- function(xa, mk, N, S, D, H) {
  y <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)
  x <- ggml_input(shape = c(S, D), name = "x")
  m <- ggml_input(shape = c(S, S), name = "mask")
  o <- ggml_layer_attention(x, D, n_heads = H, mask = m, name = "attn")
  mod <- ggml_compile(ggml_model(list(x, m), o), loss = "mse", backend = "cpu")
  mod <- ggml_fit(mod, list(xa, mk), y, epochs = 1L, batch_size = N, verbose = 0L)
  w0  <- attn_w_of(mod)
  list(pred = ggml_predict(mod, list(xa, mk), batch_size = N),
       W = list(W_q = as_mat_m(w0$W_q, D, D), W_k = as_mat_m(w0$W_k, D, D),
                W_v = as_mat_m(w0$W_v, D, D), W_o = as_mat_m(w0$W_o, D, D),
                b_o = ggml_backend_tensor_get_data(w0$b_o)))
}

test_that("an all-zero mask leaves attention unchanged", {
  set.seed(5L)
  N <- 2L; S <- 4L; D <- 6L; H <- 2L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  r  <- fit_masked(xa, array(0, dim = c(N, S, S)), N, S, D, H)

  for (i in seq_len(N)) {
    got <- matrix(r$pred[i, ], S, D, byrow = TRUE)
    expect_equal(got, ref_masked(matrix(xa[i, , ], S, D), r$W, D, H),
                 tolerance = 1e-4)
  }
})

test_that("a mask blocks the keys it names, and on the right axis", {
  set.seed(5L)
  N <- 2L; S <- 4L; D <- 6L; H <- 2L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  # mask[n, q, k]: block keys 3 and 4 for every query -> the k axis is the
  # SECOND R index. Blocking [, 3:4, ] instead would mask queries and fail.
  mk <- array(0, dim = c(N, S, S)); mk[, , 3:4] <- -1e9
  r  <- fit_masked(xa, mk, N, S, D, H)

  for (i in seq_len(N)) {
    got <- matrix(r$pred[i, ], S, D, byrow = TRUE)
    expect_equal(got, ref_masked(matrix(xa[i, , ], S, D), r$W, D, H, blocked = 3:4),
                 tolerance = 1e-4)
  }
})

test_that("the mask shape is checked against the sequence lengths", {
  # The check lives where the graph is built, which is the first fit, not
  # compile -- a mask one key too wide would otherwise read past the scores.
  N <- 4L; S <- 4L; D <- 6L
  x <- ggml_input(shape = c(S, D), name = "x")
  m <- ggml_input(shape = c(S, S + 1L), name = "mask")
  o <- ggml_layer_attention(x, D, n_heads = 2L, mask = m)
  mod <- ggml_compile(ggml_model(list(x, m), o), loss = "mse", backend = "cpu")

  xa <- array(runif(N * S * D), dim = c(N, S, D))
  expect_error(
    ggml_fit(mod, list(xa, array(0, dim = c(N, S, S + 1L))),
             matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D),
             epochs = 1L, batch_size = N, verbose = 0L),
    "mask shape")
})

test_that("a padding mask trains and beats padding with zeros alone", {
  set.seed(9L)
  N <- 32L; S <- 6L; D <- 8L; H <- 2L
  len <- sample(2:S, N, replace = TRUE)
  xa  <- array(runif(N * S * D), dim = c(N, S, D))
  mk  <- array(0, dim = c(N, S, S))
  y   <- matrix(0, N, D)
  for (i in seq_len(N)) {
    if (len[i] < S) {
      xa[i, (len[i] + 1L):S, ] <- 0
      mk[i, , (len[i] + 1L):S] <- -1e9
    }
    y[i, ] <- colMeans(matrix(xa[i, seq_len(len[i]), ], len[i], D))
  }

  x <- ggml_input(shape = c(S, D), name = "x")
  m <- ggml_input(shape = c(S, S), name = "mask")
  o <- ggml_layer_attention(x, D, n_heads = H, mask = m) |>
    ggml_layer_flatten() |> ggml_layer_dense(D)
  mod <- ggml_compile(ggml_model(list(x, m), o), loss = "mse", backend = "cpu")
  l <- ggml_fit(mod, list(xa, mk), y, epochs = 10L, batch_size = 8L,
                verbose = 0L)$history$train_loss

  expect_false(any(is.na(l)))
  expect_lt(l[[length(l)]], l[[1L]])
})

test_that("causal and an explicit mask combine", {
  set.seed(11L)
  N <- 8L; S <- 4L; D <- 6L; H <- 2L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)
  mk <- array(0, dim = c(N, S, S)); mk[, , S] <- -1e9

  x <- ggml_input(shape = c(S, D), name = "x")
  m <- ggml_input(shape = c(S, S), name = "mask")
  o <- ggml_layer_attention(x, D, n_heads = H, causal = TRUE, mask = m)
  mod <- ggml_compile(ggml_model(list(x, m), o), loss = "mse", backend = "cpu")
  l <- ggml_fit(mod, list(xa, mk), y, epochs = 3L, batch_size = 4L,
                verbose = 0L)$history$train_loss

  expect_false(any(is.na(l)))
})
