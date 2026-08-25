# Tests for GGML_OP_NORM_BACK -- the LayerNorm backward that upstream ggml does
# not have, so ggml_norm() would otherwise be inference-only.
#
# The reference is the closed form, computed here in R:
#     dL/dx = (dz - mean(dz) - y * mean(dz * y)) / sigma,  y = (x - mu) / sigma
# Checking against it (rather than CPU-vs-GPU alone) is what catches a formula
# that is wrong the same way in both kernels.

ref_norm_back <- function(dz, x, eps = 1e-5) {
  mu <- mean(x)
  s  <- sqrt(mean((x - mu)^2) + eps)
  y  <- (x - mu) / s
  (dz - mean(dz) - y * mean(dz * y)) / s
}

# Run ggml_norm_back on one backend and return the result as a plain vector.
run_norm_back <- function(backend, dz, x, K, R, eps = 1e-5) {
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  g <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, R)
  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, R)
  o <- ggml_norm_back(ctx, g, t, eps)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)

  ggml_backend_tensor_set_data(g, dz)
  ggml_backend_tensor_set_data(t, x)
  ggml_backend_graph_compute(backend, ggml_build_forward_expand(ctx, o))
  ggml_backend_tensor_get_data(o)
}

test_that("norm_back matches the closed form on CPU", {
  set.seed(1L)
  K <- 8L; R <- 3L
  dz <- rnorm(K * R); x <- rnorm(K * R)

  b <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(b), add = TRUE)
  got <- run_norm_back(b, dz, x, K, R)

  want <- unlist(lapply(seq_len(R), function(r) {
    i <- ((r - 1L) * K + 1L):(r * K)
    ref_norm_back(dz[i], x[i])
  }))
  expect_equal(got, want, tolerance = 1e-5)
})

test_that("the Vulkan shader agrees with the CPU kernel", {
  skip_on_cran()
  dev <- tryCatch(ggml_backend_dev_by_type(ggml_backend_device_type_gpu()),
                  error = function(e) NULL)
  skip_if(is.null(dev), "no Vulkan device")

  set.seed(2L)
  K <- 16L; R <- 4L
  dz <- rnorm(K * R); x <- rnorm(K * R)

  b_cpu <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(b_cpu), add = TRUE)
  b_gpu <- ggml_backend_dev_init(dev)
  on.exit(ggml_backend_free(b_gpu), add = TRUE)

  expect_equal(run_norm_back(b_gpu, dz, x, K, R),
               run_norm_back(b_cpu, dz, x, K, R), tolerance = 1e-5)
})

test_that("a wider row still reduces correctly", {
  # The shader reduces in shared memory over BLOCK_SIZE threads, so a row
  # longer than one thread's stride exercises the loop, not just the tail.
  set.seed(3L)
  K <- 600L; R <- 2L
  dz <- rnorm(K * R); x <- rnorm(K * R)

  b <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(b), add = TRUE)
  got <- run_norm_back(b, dz, x, K, R)

  want <- unlist(lapply(seq_len(R), function(r) {
    i <- ((r - 1L) * K + 1L):(r * K)
    ref_norm_back(dz[i], x[i])
  }))
  expect_equal(got, want, tolerance = 1e-4)
})

test_that("layer_norm trains: the loss falls and stays finite", {
  set.seed(5L)
  N <- 32L; S <- 4L; D <- 8L
  xa <- array(runif(N * S * D, 1, 3), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  x <- ggml_input(shape = c(S, D))
  o <- x |> ggml_layer_layer_norm() |> ggml_layer_dense(D, time_distributed = TRUE)
  m <- ggml_compile(ggml_model(x, o), loss = "mse", backend = "cpu")
  l <- ggml_fit(m, xa, y, epochs = 8L, batch_size = 8L, verbose = 0L)$history$train_loss

  expect_false(any(is.na(l)))
  expect_lt(l[[length(l)]], l[[1L]])
})

test_that("layer_norm centres its output, rms_norm does not", {
  set.seed(7L)
  N <- 8L; S <- 2L; D <- 6L
  # Off-centre input: the two normalizations only differ once the mean is not 0.
  xa <- array(runif(N * S * D, 5, 7), dim = c(N, S, D))
  y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)

  out <- function(layer) {
    x <- ggml_input(shape = c(S, D))
    m <- ggml_compile(ggml_model(x, layer(x)), loss = "mse", backend = "cpu")
    m <- ggml_fit(m, xa, y, epochs = 1L, batch_size = N, verbose = 0L)
    matrix(ggml_predict(m, xa, batch_size = N)[1L, ], S, D, byrow = TRUE)
  }

  # Centred to f32 precision; the input's mean is ~6, so the residual is
  # relative to that, not to 1.
  expect_lt(abs(mean(out(ggml_layer_layer_norm)[1L, ])), 1e-2)
  expect_gt(abs(mean(out(ggml_layer_rms_norm)[1L, ])), 0.1)
})
