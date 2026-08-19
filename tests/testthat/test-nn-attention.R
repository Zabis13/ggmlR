# Tests for multi-head attention in the functional API, and for the
# position-wise (time-distributed) dense layer that completes a transformer
# block. The reference values are computed here in plain R, so a wrong axis or
# a mis-sliced head shows up as a numeric mismatch rather than as a shape error.

cleanup_attn_model <- function(model) {
  if (!is.null(model$compilation$buffer)) {
    ggml_backend_buffer_free(model$compilation$buffer)
  }
  if (!is.null(model$compilation$ctx_weights)) {
    ggml_free(model$compilation$ctx_weights)
  }
  if (!is.null(model$compilation$sched)) {
    ggml_backend_sched_free(model$compilation$sched)
  }
  if (!is.null(model$compilation$backend)) {
    ggml_backend_free(model$compilation$backend)
  }
  if (!is.null(model$compilation$cpu_backend)) {
    ggml_backend_free(model$compilation$cpu_backend)
  }
}

# Softmax over each row of a matrix, the direction attention normalises in:
# one query's weights over all keys.
softmax_rows <- function(M) {
  e <- exp(M - apply(M, 1, max))
  e / rowSums(e)
}

# A model's weights live in a context that ggml_predict() does not keep alive,
# so they are read straight after a fit, which leaves node_weights populated
# with live tensors.
attn_weights_of <- function(model) {
  nw  <- model$node_weights
  hit <- names(nw)[vapply(nw, function(e) !is.null(e$W_q), logical(1))]
  nw[[hit[[1L]]]]
}

as_mat <- function(tensor, nrow, ncol) {
  matrix(ggml_backend_tensor_get_data(tensor), nrow = nrow, ncol = ncol)
}

# Reference multi-head attention. Q/K/V come from `q_src`/`kv_src`, which are
# the same matrix for self-attention. `causal` masks keys after the query.
ref_attention <- function(q_src, kv_src, W, d_model, n_heads, causal = FALSE) {
  Q  <- q_src  %*% W$W_q
  K  <- kv_src %*% W$W_k
  V  <- kv_src %*% W$W_v
  dh <- d_model %/% n_heads
  out <- matrix(0, nrow(q_src), d_model)
  for (h in seq_len(n_heads)) {
    cols <- ((h - 1L) * dh + 1L):(h * dh)
    sc <- (Q[, cols, drop = FALSE] %*% t(K[, cols, drop = FALSE])) / sqrt(dh)
    if (causal) sc[upper.tri(sc)] <- -Inf
    out[, cols] <- softmax_rows(sc) %*% V[, cols, drop = FALSE]
  }
  res <- out %*% W$W_o
  if (!is.null(W$b_o)) res <- res + rep(W$b_o, each = nrow(q_src))
  res
}

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

test_that("ggml_attention rejects a d_model that the heads cannot divide", {
  expect_error(ggml_attention(10L, n_heads = 4L), "divisible")
  expect_error(ggml_attention(0L, n_heads = 1L), "positive integer")
  expect_error(ggml_attention(8L, n_heads = 0L), "positive integer")
})

test_that("attention rejects an input that is not a sequence", {
  x <- ggml_input(shape = 16L)               # flat, no sequence axis
  o <- x |> ggml_layer_attention(d_model = 16L, n_heads = 2L)
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  # Shapes are inferred when the graph is built, not at compile time.
  expect_error(ggml_predict(m, matrix(0, 2L, 16L), batch_size = 2L),
               "sequence input")
})

test_that("attention rejects a d_model that disagrees with the input width", {
  x <- ggml_input(shape = c(6L, 8L))
  o <- x |> ggml_layer_attention(d_model = 16L, n_heads = 2L)
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  expect_error(ggml_predict(m, array(0, dim = c(2L, 6L, 8L)), batch_size = 2L),
               "but the input has")
})

test_that("the layer object and the pipe form build the same node", {
  x <- ggml_input(shape = c(5L, 8L))
  a <- x |> ggml_layer_attention(8L, n_heads = 2L, name = "a")
  b <- ggml_apply(x, ggml_attention(8L, n_heads = 2L, name = "b"))
  expect_identical(a$node_type, "attention")
  expect_identical(b$node_type, "attention")
  expect_identical(a$config$d_model, b$config$d_model)
  expect_identical(a$config$n_heads, b$config$n_heads)
})

test_that("applying one attention object twice shares its weights", {
  attn <- ggml_attention(8L, n_heads = 2L, name = "shared")
  x1 <- ggml_input(shape = c(4L, 8L), name = "x1")
  x2 <- ggml_input(shape = c(4L, 8L), name = "x2")
  o1 <- ggml_apply(x1, attn)
  o2 <- ggml_apply(x2, attn)
  # The sharing key is the layer object's identity, not its name.
  expect_identical(o1$layer_id, o2$layer_id)
  expect_false(identical(o1$id, o2$id))
})

test_that("ggml_apply rejects a list for a layer taking one input", {
  x1 <- ggml_input(shape = 8L)
  x2 <- ggml_input(shape = 8L)
  expect_error(ggml_apply(list(x1, x2), ggml_dense(4L)),
               "takes a single input")
  expect_error(ggml_apply(list(x1, x2, x1), ggml_attention(8L)),
               "one input .* or two")
})

# ---------------------------------------------------------------------------
# Numerics -- against a reference implementation in plain R
# ---------------------------------------------------------------------------

test_that("self-attention matches a reference implementation", {
  set.seed(7L)
  S <- 4L; D <- 6L; H <- 2L; N <- 2L
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = H, name = "attn")
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  xa <- array(runif(N * S * D), dim = c(N, S, D))
  # One epoch only to populate node_weights with live tensors; the values are
  # read back and used as the reference's weights, so the training is irrelevant.
  m <- ggml_fit(m, xa, matrix(0, N, S * D), epochs = 1L, batch_size = N,
                verbose = 0L)
  ww <- attn_weights_of(m)
  W  <- list(W_q = as_mat(ww$W_q, D, D), W_k = as_mat(ww$W_k, D, D),
             W_v = as_mat(ww$W_v, D, D), W_o = as_mat(ww$W_o, D, D),
             b_o = ggml_backend_tensor_get_data(ww$b_o))

  p <- ggml_predict(m, xa, batch_size = N)
  for (i in seq_len(N)) {
    Xs  <- matrix(xa[i, , ], S, D)
    got <- matrix(p[i, ], S, D, byrow = TRUE)
    expect_equal(got, ref_attention(Xs, Xs, W, D, H), tolerance = 1e-5)
  }

  cleanup_attn_model(m)
})

test_that("causal attention masks keys after the query", {
  set.seed(11L)
  S <- 5L; D <- 4L; H <- 2L; N <- 2L
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = H, causal = TRUE, name = "attn")
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  xa <- array(runif(N * S * D), dim = c(N, S, D))
  m  <- ggml_fit(m, xa, matrix(0, N, S * D), epochs = 1L, batch_size = N,
                 verbose = 0L)
  ww <- attn_weights_of(m)
  W  <- list(W_q = as_mat(ww$W_q, D, D), W_k = as_mat(ww$W_k, D, D),
             W_v = as_mat(ww$W_v, D, D), W_o = as_mat(ww$W_o, D, D),
             b_o = ggml_backend_tensor_get_data(ww$b_o))

  p <- ggml_predict(m, xa, batch_size = N)
  for (i in seq_len(N)) {
    Xs  <- matrix(xa[i, , ], S, D)
    got <- matrix(p[i, ], S, D, byrow = TRUE)
    expect_equal(got, ref_attention(Xs, Xs, W, D, H, causal = TRUE),
                 tolerance = 1e-5)
    # An unmasked model would give a different answer -- otherwise the test
    # would pass with the mask silently dropped.
    expect_false(isTRUE(all.equal(got, ref_attention(Xs, Xs, W, D, H),
                                  tolerance = 1e-5)))
  }

  cleanup_attn_model(m)
})

test_that("causal attention makes earlier positions independent of later ones", {
  # The property the mask exists for: a query may only read keys at or before
  # its own position, so perturbing the END of the sequence must leave every
  # earlier position bit-identical. This is a stronger check than the reference
  # comparison above -- it holds whatever the weights are.
  set.seed(17L)
  S <- 5L; D <- 4L; N <- 2L
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = 2L, causal = TRUE)
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  xa <- array(runif(N * S * D), dim = c(N, S, D))
  xb <- xa
  xb[, S, ] <- xb[, S, ] + 5          # change only the LAST position

  # A compiled model re-randomises its weights on every ggml_predict() until it
  # has been fitted once, so the two calls below would otherwise be comparing
  # two different models rather than two inputs.
  m <- ggml_fit(m, xa, matrix(0, N, S * D), epochs = 1L, batch_size = N,
                verbose = 0L)

  pa <- ggml_predict(m, xa, batch_size = N)
  pb <- ggml_predict(m, xb, batch_size = N)
  ra <- matrix(pa[1L, ], S, D, byrow = TRUE)
  rb <- matrix(pb[1L, ], S, D, byrow = TRUE)

  # Every position before the last is untouched.
  for (i in seq_len(S - 1L)) {
    expect_equal(ra[i, ], rb[i, ], tolerance = 1e-5)
  }
  # The last position did change, so the model is not simply ignoring its input.
  expect_false(isTRUE(all.equal(ra[S, ], rb[S, ], tolerance = 1e-5)))

  cleanup_attn_model(m)
})

test_that("a non-causal model does propagate a late change backwards", {
  # The counterpart of the test above: without the mask, every position sees
  # the whole sequence, so the same perturbation must reach position 1. Without
  # this, the causal test could pass on a model that ignores its input.
  set.seed(17L)
  S <- 5L; D <- 4L; N <- 2L
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = 2L, causal = FALSE)
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  xa <- array(runif(N * S * D), dim = c(N, S, D))
  xb <- xa
  xb[, S, ] <- xb[, S, ] + 5
  m <- ggml_fit(m, xa, matrix(0, N, S * D), epochs = 1L, batch_size = N,
                verbose = 0L)

  ra <- matrix(ggml_predict(m, xa, batch_size = N)[1L, ], S, D, byrow = TRUE)
  rb <- matrix(ggml_predict(m, xb, batch_size = N)[1L, ], S, D, byrow = TRUE)

  expect_false(isTRUE(all.equal(ra[1L, ], rb[1L, ], tolerance = 1e-5)))

  cleanup_attn_model(m)
})

test_that("cross-attention reads keys and values from the second input", {
  set.seed(13L)
  Sq <- 3L; Sk <- 6L; D <- 6L; H <- 2L; N <- 2L
  q  <- ggml_input(shape = c(Sq, D), name = "q")
  kv <- ggml_input(shape = c(Sk, D), name = "kv")
  o  <- ggml_apply(list(q, kv), ggml_attention(D, n_heads = H, name = "xattn"))
  m  <- ggml_model(inputs = list(q, kv), outputs = o)
  m  <- ggml_compile(m, loss = "mse", backend = "cpu")

  qa <- matrix(runif(N * Sq * D), N)
  ka <- matrix(runif(N * Sk * D), N)
  m  <- ggml_fit(m, list(qa, ka), matrix(0, N, Sq * D), epochs = 1L,
                 batch_size = N, verbose = 0L)
  ww <- attn_weights_of(m)
  W  <- list(W_q = as_mat(ww$W_q, D, D), W_k = as_mat(ww$W_k, D, D),
             W_v = as_mat(ww$W_v, D, D), W_o = as_mat(ww$W_o, D, D),
             b_o = ggml_backend_tensor_get_data(ww$b_o))

  p <- ggml_predict(m, list(qa, ka), batch_size = N)
  # The queries decide the output length, not the context.
  expect_equal(ncol(p), Sq * D)
  for (i in seq_len(N)) {
    Qs  <- matrix(qa[i, ], Sq, D, byrow = TRUE)
    Ks  <- matrix(ka[i, ], Sk, D, byrow = TRUE)
    got <- matrix(p[i, ], Sq, D, byrow = TRUE)
    expect_equal(got, ref_attention(Qs, Ks, W, D, H), tolerance = 1e-5)
  }

  cleanup_attn_model(m)
})

test_that("causal is rejected on a cross-attention layer", {
  # Masking by position compares indices in two unrelated sequences.
  q  <- ggml_input(shape = c(3L, 4L), name = "q")
  kv <- ggml_input(shape = c(6L, 4L), name = "kv")
  o  <- ggml_apply(list(q, kv), ggml_attention(4L, n_heads = 2L, causal = TRUE))
  m  <- ggml_model(inputs = list(q, kv), outputs = o)
  m  <- ggml_compile(m, loss = "mse", backend = "cpu")
  expect_error(ggml_predict(m, list(matrix(0, 2L, 12L), matrix(0, 2L, 24L)),
                            batch_size = 2L),
               "self-attention")
})

test_that("the context input must share d_model with the queries", {
  q  <- ggml_input(shape = c(3L, 8L), name = "q")
  kv <- ggml_input(shape = c(6L, 4L), name = "kv")   # wrong width
  o  <- ggml_apply(list(q, kv), ggml_attention(8L, n_heads = 2L))
  m  <- ggml_model(inputs = list(q, kv), outputs = o)
  m  <- ggml_compile(m, loss = "mse", backend = "cpu")
  expect_error(ggml_predict(m, list(matrix(0, 2L, 24L), matrix(0, 2L, 24L)),
                            batch_size = 2L),
               "context input")
})

test_that("n_heads changes the result, so the heads are really split", {
  # One head attends over the whole feature vector; four attend over quarters.
  # Identical output would mean the head split never happened.
  set.seed(23L)
  S <- 4L; D <- 8L; N <- 2L
  xa <- array(runif(N * S * D), dim = c(N, S, D))

  build <- function(h) {
    set.seed(1L)   # same initial weights for both
    x <- ggml_input(shape = c(S, D), name = "x")
    o <- x |> ggml_layer_attention(D, n_heads = h, name = "a")
    mm <- ggml_model(inputs = x, outputs = o)
    ggml_compile(mm, loss = "mse", backend = "cpu")
  }
  m1 <- build(1L); p1 <- ggml_predict(m1, xa, batch_size = N)
  m4 <- build(4L); p4 <- ggml_predict(m4, xa, batch_size = N)

  expect_equal(dim(p1), dim(p4))
  expect_false(isTRUE(all.equal(p1, p4, tolerance = 1e-4)))

  cleanup_attn_model(m1); cleanup_attn_model(m4)
})

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

test_that("attention trains on a task that needs cross-position mixing", {
  skip_on_cran()
  # Every output position must equal the mean over the whole sequence, which a
  # position-wise layer cannot represent -- only a layer that mixes positions.
  set.seed(3L)
  N <- 256L; S <- 6L; D <- 8L
  xa <- array(runif(N * S * D, -1, 1), dim = c(N, S, D))
  ya <- t(vapply(seq_len(N), function(i) {
    Xs <- matrix(xa[i, , ], S, D)
    as.vector(t(matrix(rep(colMeans(Xs), each = S), S, D)))
  }, numeric(S * D)))

  set.seed(1L)
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = 2L, name = "attn")
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, optimizer = "adam", loss = "mse", backend = "cpu")
  m <- ggml_fit(m, xa, ya, epochs = 40L, batch_size = 32L, verbose = 0L)
  h <- m$history

  expect_true(all(is.finite(h$train_loss)))
  expect_lt(tail(h$train_loss, 1L), h$train_loss[1L] / 2)

  cleanup_attn_model(m)
})

test_that("attention layers stack", {
  skip_on_cran()
  # The output keeps the c(seq_len, d_model) shape of the input, which is what
  # lets one attention layer feed another.
  set.seed(2L)
  N <- 96L; S <- 5L; D <- 8L
  xb  <- array(runif(N * S * D, -1, 1), dim = c(N, S, D))
  lab <- as.integer(apply(xb, 1, mean) > 0)
  yb  <- cbind(1 - lab, lab) * 1.0

  set.seed(1L)
  i2 <- ggml_input(shape = c(S, D), name = "in")
  a1 <- i2 |> ggml_layer_attention(D, n_heads = 2L, name = "mha1")
  a2 <- a1 |> ggml_layer_attention(D, n_heads = 2L, name = "mha2")
  o2 <- a2 |> ggml_layer_flatten() |> ggml_layer_dense(2L, activation = "softmax")
  m2 <- ggml_model(inputs = i2, outputs = o2)
  m2 <- ggml_compile(m2, optimizer = "adam",
                     loss = "categorical_crossentropy", backend = "cpu")
  m2 <- ggml_fit(m2, xb, yb, epochs = 10L, batch_size = 32L, verbose = 0L)

  expect_true(all(is.finite(m2$history$train_loss)))

  cleanup_attn_model(m2)
})

# ---------------------------------------------------------------------------
# Position-wise (time-distributed) dense
# ---------------------------------------------------------------------------

test_that("time_distributed keeps the sequence axis in the inferred shape", {
  # The contract this changes: an ordinary dense layer flattens a sequence into
  # a single vector, a time-distributed one keeps one output per position.
  td <- ggmlR:::nn_functional_output_shape(
    list(node_type = "dense",
         config = list(units = 7L, time_distributed = TRUE)),
    list(c(5L, 4L)))
  flat <- ggmlR:::nn_functional_output_shape(
    list(node_type = "dense",
         config = list(units = 7L, time_distributed = FALSE)),
    list(c(5L, 4L)))

  expect_equal(td, c(5L, 7L))     # c(seq_len, units)
  expect_equal(flat, 7L)          # collapsed to a bare width
})

test_that("time_distributed sizes the kernel from the features alone", {
  # [features, units], not [seq_len*features, units] -- that is what makes the
  # weights shared across positions rather than one kernel per position.
  expect_equal(ggmlR:::nn_dense_fan_in(list(config = list(time_distributed = TRUE)),
                               c(5L, 4L)), 4)
  expect_equal(ggmlR:::nn_dense_fan_in(list(config = list(time_distributed = FALSE)),
                               c(5L, 4L)), 20)
})

test_that("a time-distributed dense builds a [units, seq, batch] tensor", {
  # The built tensor, not just the inferred shape: both the position and the
  # batch axis have to survive into the graph.
  S <- 5L; Din <- 4L; U <- 7L; N <- 3L
  x <- ggml_input(shape = c(S, Din), name = "x")
  o <- x |> ggml_layer_dense(U, time_distributed = TRUE, name = "td")
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  gi <- ggmlR:::nn_build_functional_graph(m, batch_size = N, training = FALSE)
  ne <- ggml_tensor_shape(gi$outputs[[1L]])
  expect_equal(ne[1L], U)
  expect_equal(ne[2L], S)
  expect_equal(ne[3L], N)

  ggml_free(gi$ctx_compute)
  cleanup_attn_model(m)
})

test_that("a time-distributed dense applies one kernel at every position", {
  set.seed(5L)
  S <- 4L; Din <- 3L; U <- 5L; N <- 2L
  x <- ggml_input(shape = c(S, Din), name = "x")
  o <- x |> ggml_layer_dense(U, activation = "relu", time_distributed = TRUE,
                             name = "td")
  m <- ggml_model(inputs = x, outputs = o)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")

  xa <- array(runif(N * S * Din, -1, 1), dim = c(N, S, Din))
  m  <- ggml_fit(m, xa, matrix(0, N, S * U), epochs = 1L, batch_size = N,
                 verbose = 0L)
  nw <- m$node_weights
  hit <- names(nw)[vapply(nw, function(e) !is.null(e$weight), logical(1))]
  ww <- nw[[hit[[1L]]]]
  W  <- as_mat(ww$weight, Din, U)
  b  <- ggml_backend_tensor_get_data(ww$bias)
  # Shared across positions: the kernel maps features to units, once.
  expect_equal(dim(W), c(Din, U))

  p <- ggml_predict(m, xa, batch_size = N)
  for (i in seq_len(N)) {
    Xs  <- matrix(xa[i, , ], S, Din)
    ref <- pmax(Xs %*% W + rep(b, each = S), 0)
    expect_equal(matrix(p[i, ], S, U, byrow = TRUE), ref, tolerance = 1e-5)
  }

  cleanup_attn_model(m)
})

test_that("time_distributed is rejected on a sequential model", {
  m <- ggml_model_sequential()
  expect_error(ggml_layer_dense(m, 8L, input_shape = 4L,
                                time_distributed = TRUE),
               "functional API")
})

test_that("a full transformer block trains", {
  skip_on_cran()
  # Attention sublayer + residual, then a position-wise feed-forward sublayer
  # + residual -- the complete encoder block, not just the attention half.
  set.seed(2L)
  N <- 256L; S <- 6L; D <- 8L
  xb  <- array(runif(N * S * D, -1, 1), dim = c(N, S, D))
  lab <- as.integer(apply(xb, 1, mean) > 0)
  yb  <- cbind(1 - lab, lab) * 1.0

  set.seed(1L)
  inp <- ggml_input(shape = c(S, D), name = "in")
  at  <- inp |> ggml_layer_attention(D, n_heads = 2L, name = "mha")
  h1  <- ggml_layer_add(list(inp, at))
  f1  <- h1 |> ggml_layer_dense(D * 2L, activation = "relu",
                                time_distributed = TRUE, name = "ff1")
  f2  <- f1 |> ggml_layer_dense(D, time_distributed = TRUE, name = "ff2")
  h2  <- ggml_layer_add(list(h1, f2))
  out <- h2 |> ggml_layer_flatten() |>
    ggml_layer_dense(2L, activation = "softmax", name = "cls")
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, optimizer = "adam",
                    loss = "categorical_crossentropy", backend = "cpu")
  m <- ggml_fit(m, xb, yb, epochs = 25L, batch_size = 32L, verbose = 0L)
  h <- m$history

  expect_true(all(is.finite(h$train_loss)))
  expect_lt(tail(h$train_loss, 1L), h$train_loss[1L])
  expect_gt(tail(h$train_accuracy, 1L), 0.8)

  cleanup_attn_model(m)
})
