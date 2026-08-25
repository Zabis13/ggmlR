# Tests for rope = TRUE in the attention layer -- rotary position embedding,
# applied to queries and keys inside the head split.
#
# The decisive test is the symmetry one: plain attention is order-blind, so the
# first position of a sequence and the last position of its reversal produce
# the SAME output. RoPE breaks that tie. Checking merely that rope changes the
# numbers would also pass if it rotated by the wrong positions.

cleanup_rope <- function(model) {
  cp <- model$compilation
  if (is.null(cp)) return(invisible(NULL))
  if (!is.null(cp$buffer))      ggml_backend_buffer_free(cp$buffer)
  if (!is.null(cp$ctx_weights)) ggml_free(cp$ctx_weights)
  if (!is.null(cp$sched))       ggml_backend_sched_free(cp$sched)
  if (!is.null(cp$backend))     ggml_backend_free(cp$backend)
  if (!is.null(cp$cpu_backend)) ggml_backend_free(cp$cpu_backend)
}

# One fit + predict with a fixed seed, so two calls differ only in the argument
# under test.
run_attn <- function(dat, rope, N, S, D, H, backend = "cpu", seed = 11L) {
  set.seed(seed)
  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = H, rope = rope)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = backend)
  m <- ggml_fit(m, dat, matrix(aperm(dat, c(1L, 3L, 2L)), N, S * D),
                epochs = 1L, batch_size = N, verbose = 0L)
  p <- ggml_predict(m, dat, batch_size = N)
  cleanup_rope(m)
  p
}

test_that("rope needs an even head width", {
  # A head of 5 features cannot be rotated in pairs.
  expect_error(ggml_attention(10L, n_heads = 2L, rope = TRUE), "even head width")
  expect_silent(ggml_attention(8L, n_heads = 2L, rope = TRUE))
})

test_that("rope breaks the order symmetry of plain attention", {
  set.seed(3L)
  N <- 4L; S <- 4L; D <- 8L; H <- 2L
  tok <- matrix(runif(S * D), S, D)
  fwd <- array(0, dim = c(N, S, D))
  rev_ <- array(0, dim = c(N, S, D))
  for (i in seq_len(N)) {
    fwd[i, , ]  <- tok
    rev_[i, , ] <- tok[S:1, ]
  }

  # Position 1 of the forward sequence and position S of the reversed one hold
  # the same token and see the same set of others. Order is the only thing that
  # separates them.
  pick <- function(p, row) matrix(p[1L, ], S, D, byrow = TRUE)[row, ]

  without <- c(pick(run_attn(fwd, FALSE, N, S, D, H), 1L),
               pick(run_attn(rev_, FALSE, N, S, D, H), S))
  expect_equal(without[seq_len(D)], without[D + seq_len(D)], tolerance = 1e-6)

  with_r <- c(pick(run_attn(fwd, TRUE, N, S, D, H), 1L),
              pick(run_attn(rev_, TRUE, N, S, D, H), S))
  expect_gt(max(abs(with_r[seq_len(D)] - with_r[D + seq_len(D)])), 1e-4)
})

test_that("rope trains: gradients flow through the rotation", {
  set.seed(5L)
  N <- 12L; S <- 4L; D <- 8L; H <- 2L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = H, rope = TRUE) |>
    ggml_layer_dense(D, time_distributed = TRUE)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  l <- ggml_fit(m, xa, y, epochs = 5L, batch_size = 6L,
                verbose = 0L)$history$train_loss

  expect_false(any(is.na(l)))
  expect_lt(l[[length(l)]], l[[1L]])
  cleanup_rope(m)
})

test_that("rope combines with causal masking", {
  set.seed(7L)
  N <- 4L; S <- 4L; D <- 8L; H <- 2L
  xa <- array(runif(N * S * D), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D), name = "x")
  o <- x |> ggml_layer_attention(D, n_heads = H, rope = TRUE, causal = TRUE)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  l <- ggml_fit(m, xa, y, epochs = 2L, batch_size = 4L,
                verbose = 0L)$history$train_loss

  expect_false(any(is.na(l)))
  cleanup_rope(m)
})

test_that("the Vulkan path agrees with the CPU one", {
  skip_on_cran()
  dev <- tryCatch(ggml_backend_dev_by_type(ggml_backend_device_type_gpu()),
                  error = function(e) NULL)
  skip_if(is.null(dev), "no Vulkan device")

  set.seed(3L)
  N <- 4L; S <- 4L; D <- 8L; H <- 2L
  xa <- array(runif(N * S * D), dim = c(N, S, D))

  expect_equal(run_attn(xa, TRUE, N, S, D, H, backend = "vulkan", seed = 21L),
               run_attn(xa, TRUE, N, S, D, H, backend = "cpu",    seed = 21L),
               tolerance = 1e-2)
})
