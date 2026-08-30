# Backward pass for GELU (GGML_OP_GELU_BACK, a ggmlR extension).
#
# Upstream ggml has no gradient for GELU: ggml_compile(activation = "gelu")
# built a forward graph and then ggml_fit aborted the process from
# ggml-graph.c's unary default branch. The op added here implements the
# derivative of the tanh approximation ggml_gelu() computes:
#   g(x)  = 0.5*x*(1 + tanh(u)),  u = c*x*(1 + a*x^2)
#   g'(x) = 0.5*(1 + tanh(u)) + 0.5*x*(1 - tanh(u)^2)*c*(1 + 3*a*x^2)
# GELU_QUICK and GELU_ERF are different functions and remain unsupported.

GELU_A <- 0.044715
GELU_C <- 0.79788456080286535587989211986876

gelu_ref <- function(x) 0.5 * x * (1 + tanh(GELU_C * x * (1 + GELU_A * x * x)))

gelu_grad_ref <- function(x) {
  u  <- GELU_C * x * (1 + GELU_A * x * x)
  th <- tanh(u)
  0.5 * (1 + th) + 0.5 * x * (1 - th * th) * GELU_C * (1 + 3 * GELU_A * x * x)
}

# Runs the op on a backend. Tensors are allocated on the backend and filled
# through ggml_backend_tensor_set_data(), not ggml_set_f32(): the latter writes
# into the context's own CPU memory, which a Vulkan buffer never sees -- the
# kernel would then read zeros and every result would come back 0.
run_gelu_back <- function(x, dy, backend = c("cpu", "vulkan")) {
  backend <- match.arg(backend)

  ctx <- ggml_init(64 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  if (backend == "vulkan") {
    b <- ggml_vulkan_init(0L)
  } else {
    b <- ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(b, 2L)
  }
  on.exit(ggml_backend_free(b), add = TRUE)

  g <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(dy))
  ggml_set_input(g)
  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(x))
  ggml_set_input(t)

  out <- ggml_gelu_back(ctx, g, t)
  ggml_set_output(out)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
  ggml_backend_tensor_set_data(g, dy)
  ggml_backend_tensor_set_data(t, x)

  ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
  ggml_backend_tensor_get_data(out)
}

# ============================================================================
# The op itself, against the analytic derivative
# ============================================================================

test_that("ggml_gelu_back matches the analytic derivative", {
  x  <- c(-6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6)
  dy <- rep(1, length(x))

  expect_equal(run_gelu_back(x, dy), gelu_grad_ref(x), tolerance = 1e-5)
})

test_that("ggml_gelu_back scales by the incoming gradient", {
  x  <- c(-2, -0.5, 0, 0.5, 2)
  dy <- c(0.5, -1, 2, 3, -0.25)

  expect_equal(run_gelu_back(x, dy), dy * gelu_grad_ref(x), tolerance = 1e-5)
})

test_that("ggml_gelu_back agrees with a finite difference of ggml_gelu", {
  # Ties the gradient to the forward op actually shipped, not just to the
  # formula written in this file: if gelu.comp/ggml_gelu_f32 ever switches to
  # the erf form, this is what notices.
  x <- c(-3, -1, -0.25, 0.25, 1, 3)
  h <- 1e-4
  fd <- (gelu_ref(x + h) - gelu_ref(x - h)) / (2 * h)

  expect_equal(run_gelu_back(x, rep(1, length(x))), fd, tolerance = 1e-4)
})

test_that("ggml_gelu_back stays finite in the saturating tails", {
  # exp(2u) overflows for large x if the derivative is written naively; the
  # kernels spell tanh out as 2/(exp(2u)+1) to avoid that.
  x <- c(-60, -30, 30, 60)
  out <- run_gelu_back(x, rep(1, length(x)))

  expect_true(all(is.finite(out)))
  expect_equal(out, gelu_grad_ref(x), tolerance = 1e-5)
})

# ============================================================================
# Vulkan parity
# ============================================================================

test_that("the Vulkan kernel matches the CPU one", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")

  set.seed(1L)
  x  <- c(seq(-6, 6, by = 0.5), runif(32, -4, 4))
  dy <- runif(length(x), -2, 2)

  expect_equal(run_gelu_back(x, dy, "vulkan"),
               run_gelu_back(x, dy, "cpu"),
               tolerance = 1e-5)
})

# ============================================================================
# End to end: a model with gelu now trains
# ============================================================================

test_that("a sequential model with gelu trains instead of aborting", {
  set.seed(1L)
  n <- 64L
  x <- matrix(runif(n * 8), nrow = n)
  y <- matrix(runif(n * 2), nrow = n)

  m <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "gelu", input_shape = 8L) |>
    ggml_layer_dense(2L)
  m <- ggml_compile(m, optimizer = "adam", loss = "mean_squared_error")

  fit <- ggml_fit(m, x, y, epochs = 10L, batch_size = 8L, verbose = 0)
  on.exit({
    ggml_backend_sched_free(fit$compilation$sched)
    ggml_backend_free(fit$compilation$backend)
  })

  tl <- fit$history$train_loss
  expect_true(all(is.finite(tl)))
  expect_lt(tl[length(tl)], tl[1])
})

test_that("a transformer block with activation = gelu trains", {
  set.seed(1L)
  n <- 32L; vocab <- 20L; slen <- 6L; d_model <- 16L

  x <- matrix(sample(0L:(vocab - 1L), n * slen, replace = TRUE), nrow = n)
  y <- matrix(0, nrow = n, ncol = 2L)
  y[cbind(seq_len(n), sample(1:2, n, replace = TRUE))] <- 1

  inp <- ggml_input(shape = slen, dtype = "int32")
  out <- inp |>
    ggml_layer_embedding(vocab_size = vocab, dim = d_model) |>
    ggml_layer_transformer_block(d_model = d_model, n_heads = 2L,
                                 ff_dim = d_model * 2L, activation = "gelu") |>
    ggml_layer_sequence_pooling(mode = "mean") |>
    ggml_layer_dense(2L, activation = "softmax")
  m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                    optimizer = "adam", loss = "categorical_crossentropy")

  fit <- ggml_fit(m, x, y, epochs = 5L, batch_size = 8L, verbose = 0)
  on.exit({
    ggml_backend_sched_free(fit$compilation$sched)
    ggml_backend_free(fit$compilation$backend)
  })

  expect_true(all(is.finite(fit$history$train_loss)))
  expect_equal(dim(ggml_predict(fit, x)), c(n, 2L))
})
