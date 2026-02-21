# Tests for ag_* GPU device support (Phase 1)
#
# All tests that require a real GPU backend are wrapped in:
#   skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
#
# CPU path tests run unconditionally and verify that the new device parameter
# does not break existing behaviour.

# ============================================================================
# Helper: reset device to CPU after each test
# ============================================================================

reset_to_cpu <- function() {
  ag_device("cpu")
}

# ============================================================================
# CPU-path smoke tests (always run)
# ============================================================================

test_that("ag_tensor device field defaults to cpu", {
  x <- ag_tensor(matrix(1:4, 2, 2))
  expect_equal(x$device, "cpu")
  expect_equal(x$data, matrix(1:4, 2, 2))
})

test_that("ag_param device field defaults to cpu", {
  p <- ag_param(matrix(1:4, 2, 2))
  expect_equal(p$device, "cpu")
  expect_true(p$requires_grad)
})

test_that("ag_default_device returns cpu by default", {
  reset_to_cpu()
  expect_equal(ag_default_device(), "cpu")
})

test_that("ag_device('cpu') returns previous device invisibly", {
  reset_to_cpu()
  prev <- ag_device("cpu")
  expect_equal(prev, "cpu")
})

test_that("ag_matmul CPU path unchanged after refactor", {
  A <- ag_param(matrix(c(1, 0, 0, 1), 2, 2))
  B <- ag_tensor(matrix(c(1, 2, 3, 4), 2, 2))
  out <- ag_matmul(A, B)
  expect_equal(ggmlR:::.ag_data(out), ggmlR:::.ag_data(B))
})

test_that("ag_relu CPU path unchanged after refactor", {
  x <- ag_param(matrix(c(-2, -1, 0, 1, 2, 3), 2, 3))
  with_grad_tape({
    out  <- ag_relu(x)
    loss <- ag_mse_loss(out, matrix(0, 2, 3))
  })
  grads <- backward(loss)
  g <- get0(as.character(x$id), envir = grads)
  expect_equal(g[1, 1], 0)   # -2 -> grad 0
  expect_equal(g[2, 1], 0)   # -1 -> grad 0
  expect_gt(abs(g[2, 3]), 0) # 3 -> grad nonzero
})

test_that("full training loop (CPU) still reduces loss", {
  set.seed(42)
  n     <- 32L
  x_mat <- matrix(sample(c(0, 1), 2 * n, replace = TRUE), 2, n)
  y_mat <- matrix(as.numeric(xor(x_mat[1,], x_mat[2,])), 1, n)

  l1  <- ag_linear(2L, 4L, activation = "relu")
  l2  <- ag_linear(4L, 1L, activation = "sigmoid")
  opt <- optimizer_adam(c(l1$params(), l2$params()), lr = 0.05)

  losses <- numeric(30L)
  for (i in seq_len(30L)) {
    x <- ag_tensor(x_mat)
    y <- ag_tensor(y_mat)
    with_grad_tape({
      h    <- l1$forward(x)
      out  <- l2$forward(h)
      loss <- ag_mse_loss(out, y)
    })
    grads <- backward(loss)
    opt$step(grads)
    opt$zero_grad()
    losses[i] <- ggmlR:::.ag_data(loss)[1]
  }

  expect_lt(mean(losses[21:30]), mean(losses[1:10]))
})

test_that("ag_to_device returns same tensor if already on target device", {
  x <- ag_tensor(matrix(1:4, 2, 2))
  y <- ag_to_device(x, "cpu")
  expect_equal(x$id, y$id)
})

# ============================================================================
# GPU tests (skip if no backend available)
# ============================================================================

test_that("ag_device('gpu') does not error when backend available", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  expect_silent(ag_device("gpu"))
  reset_to_cpu()
})

test_that("ag_tensor(x, device='gpu') has device='gpu'", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  ag_device("gpu")
  x <- ag_tensor(matrix(1:4, 2, 2), device = "gpu")
  expect_equal(x$device, "gpu")
  expect_false(is.null(x$data))   # data always kept
  reset_to_cpu()
})

test_that("ag_param(x, device='gpu') keeps $data as source-of-truth", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  ag_device("gpu")
  d <- matrix(1:4, 2, 2)
  p <- ag_param(d, device = "gpu")
  expect_equal(p$device, "gpu")
  expect_equal(p$data, d)
  expect_true(p$requires_grad)
  reset_to_cpu()
})

test_that("ag_matmul GPU forward equals CPU forward (tol=1e-4)", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(7)
  a_mat <- matrix(runif(6), 2, 3)
  b_mat <- matrix(runif(6), 3, 2)
  expected <- a_mat %*% b_mat

  ag_device("gpu")
  A   <- ag_param(a_mat, device = "gpu")
  B   <- ag_tensor(b_mat, device = "gpu")
  with_grad_tape({
    out <- ag_matmul(A, B)
  })
  result <- ggmlR:::.ag_data(out)

  expect_equal(result, expected, tolerance = 1e-4)
  reset_to_cpu()
})

test_that("ag_relu GPU forward equals CPU forward", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(11)
  x_mat    <- matrix(runif(12, -1, 1), 3, 4)
  expected <- pmax(x_mat, 0)

  ag_device("gpu")
  x <- ag_param(x_mat, device = "gpu")
  with_grad_tape({
    out <- ag_relu(x)
  })
  result <- ggmlR:::.ag_data(out)

  expect_equal(result, expected, tolerance = 1e-6)
  reset_to_cpu()
})

test_that("backward on GPU tensors matches CPU backward (tol=1e-4)", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(99)
  w_mat <- matrix(runif(6, -1, 1), 2, 3)
  x_mat <- matrix(runif(3), 3, 1)
  y_mat <- matrix(runif(2), 2, 1)

  # CPU reference
  W_cpu <- ag_param(w_mat)
  with_grad_tape({
    out_cpu  <- ag_matmul(W_cpu, ag_tensor(x_mat))
    loss_cpu <- ag_mse_loss(out_cpu, y_mat)
  })
  grads_cpu <- backward(loss_cpu)
  g_cpu     <- get0(as.character(W_cpu$id), envir = grads_cpu)

  # GPU
  ag_device("gpu")
  W_gpu <- ag_param(w_mat, device = "gpu")
  with_grad_tape({
    out_gpu  <- ag_matmul(W_gpu, ag_tensor(x_mat, device = "gpu"))
    loss_gpu <- ag_mse_loss(out_gpu, ag_tensor(y_mat, device = "gpu"))
  })
  grads_gpu <- backward(loss_gpu)
  g_gpu     <- get0(as.character(W_gpu$id), envir = grads_gpu)

  expect_equal(g_gpu, g_cpu, tolerance = 1e-4)
  reset_to_cpu()
})

test_that("ag_gradcheck passes for GPU tensors (matmul + relu)", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(55)
  ag_device("gpu")

  W <- ag_param(matrix(runif(6, -0.5, 0.5), 2, 3), device = "gpu")
  x <- ag_tensor(matrix(runif(3), 3, 1), device = "gpu")

  result <- ag_gradcheck(
    fn = function(ins) {
      ag_mse_loss(ag_relu(ag_matmul(ins$W, ins$x)), matrix(0, 2, 1))
    },
    inputs = list(W = W, x = x),
    atol   = 1e-3,
    quiet  = TRUE
  )

  expect_true(result)
  reset_to_cpu()
})

test_that("training loop on GPU reduces loss over 10 epochs", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(77)
  ag_device("gpu")

  n     <- 32L
  x_mat <- matrix(runif(4 * n), 4, n)
  y_mat <- matrix(runif(2 * n), 2, n)

  W <- ag_param(matrix(runif(8, -0.5, 0.5), 2, 4), device = "gpu")
  b <- ag_param(matrix(0, 2, 1), device = "gpu")
  opt <- optimizer_adam(list(W = W, b = b), lr = 0.01)

  losses <- numeric(10L)
  for (i in seq_len(10L)) {
    x <- ag_tensor(x_mat, device = "gpu")
    y <- ag_tensor(y_mat, device = "gpu")
    with_grad_tape({
      h    <- ag_relu(ag_add(ag_matmul(W, x), b))
      loss <- ag_mse_loss(h, y)
    })
    grads <- backward(loss)
    opt$step(grads)
    opt$zero_grad()
    losses[i] <- as.numeric(ggmlR:::.ag_data(loss))
  }

  expect_lt(losses[10L], losses[1L])
  reset_to_cpu()
})

test_that("ag_to_device(tensor, 'cpu') correctly copies GPU data", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(13)
  d <- matrix(runif(6), 2, 3)

  ag_device("gpu")
  gpu_t <- ag_tensor(d, device = "gpu")

  cpu_t <- ag_to_device(gpu_t, "cpu")
  expect_equal(cpu_t$device, "cpu")
  expect_equal(cpu_t$data, d, tolerance = 1e-6)
  reset_to_cpu()
})

test_that("ag_softmax GPU forward equals CPU forward", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(21)
  x_mat <- matrix(runif(12, -2, 2), 3, 4)
  # CPU reference: column-wise softmax
  mx  <- apply(x_mat, 2, max)
  mx  <- matrix(mx, 3, 4, byrow = TRUE)
  e   <- exp(x_mat - mx)
  expected <- e / matrix(colSums(e), 3, 4, byrow = TRUE)

  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_softmax(x) })
  result <- ggmlR:::.ag_data(out)

  expect_equal(result, expected, tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_add GPU with [m,1] broadcast equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(22)
  a_mat <- matrix(runif(12), 3, 4)
  b_mat <- matrix(runif(3),  3, 1)
  expected <- a_mat + as.vector(b_mat)

  ag_device("gpu")
  A <- ag_tensor(a_mat, device = "gpu")
  B <- ag_param(b_mat, device = "gpu")
  with_grad_tape({ out <- ag_add(A, B) })
  result <- ggmlR:::.ag_data(out)

  expect_equal(result, expected, tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_dtype('bf16') + ag_matmul GPU result close to f32", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(51)
  a_mat <- matrix(runif(6), 2, 3)
  b_mat <- matrix(runif(6), 3, 2)
  expected <- a_mat %*% b_mat

  ag_device("gpu"); ag_dtype("bf16")
  A <- ag_param(a_mat, device = "gpu")
  B <- ag_tensor(b_mat, device = "gpu")
  expect_equal(A$dtype, "bf16")
  with_grad_tape({ out <- ag_matmul(A, B) })
  result <- ggmlR:::.ag_data(out)

  # bf16 has ~3 decimal digits of precision
  expect_equal(result, expected, tolerance = 1e-2)
  expect_equal(out$dtype, "bf16")
  ag_dtype("f32"); reset_to_cpu()
})

test_that("ag_dtype('f16') + ag_relu GPU result close to f32", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(52)
  x_mat <- matrix(runif(8, -1, 1), 2, 4)
  expected <- pmax(x_mat, 0)

  ag_device("gpu"); ag_dtype("f16")
  x <- ag_param(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_relu(x) })
  result <- ggmlR:::.ag_data(out)

  expect_equal(result, expected, tolerance = 1e-2)
  ag_dtype("f32"); reset_to_cpu()
})

test_that("ag_default_dtype returns f32 by default", {
  ag_dtype("f32")
  expect_equal(ag_default_dtype(), "f32")
})

test_that("ag_dtype switches and returns previous", {
  ag_dtype("f32")
  prev <- ag_dtype("bf16")
  expect_equal(prev, "f32")
  expect_equal(ag_default_dtype(), "bf16")
  ag_dtype("f32")
})

test_that("ag_sum GPU dim=1 (rowSums) equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(31)
  x_mat <- matrix(runif(12), 3, 4)
  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_sum(x, dim = 1L) })
  expect_equal(ggmlR:::.ag_data(out), matrix(rowSums(x_mat), 3, 1), tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_sum GPU dim=2 (colSums) equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(32)
  x_mat <- matrix(runif(12), 3, 4)
  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_sum(x, dim = 2L) })
  expect_equal(ggmlR:::.ag_data(out), matrix(colSums(x_mat), 1, 4), tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_mean GPU dim=1 (rowMeans) equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(33)
  x_mat <- matrix(runif(12), 3, 4)
  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_mean(x, dim = 1L) })
  expect_equal(ggmlR:::.ag_data(out), matrix(rowMeans(x_mat), 3, 1), tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_mean GPU dim=2 (colMeans) equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(34)
  x_mat <- matrix(runif(12), 3, 4)
  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_mean(x, dim = 2L) })
  expect_equal(ggmlR:::.ag_data(out), matrix(colMeans(x_mat), 1, 4), tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_pow GPU p=2 (sqr) equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(41)
  x_mat <- matrix(runif(6, 0.1, 2), 2, 3)
  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_pow(x, 2) })
  expect_equal(ggmlR:::.ag_data(out), x_mat^2, tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_pow GPU p=0.5 (sqrt) equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(42)
  x_mat <- matrix(runif(6, 0.1, 2), 2, 3)
  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_pow(x, 0.5) })
  expect_equal(ggmlR:::.ag_data(out), x_mat^0.5, tolerance = 1e-5)
  reset_to_cpu()
})

test_that("ag_pow GPU p=3 (general) equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(43)
  x_mat <- matrix(runif(6, 0.1, 2), 2, 3)
  ag_device("gpu")
  x <- ag_tensor(x_mat, device = "gpu")
  with_grad_tape({ out <- ag_pow(x, 3) })
  expect_equal(ggmlR:::.ag_data(out), x_mat^3, tolerance = 1e-4)
  reset_to_cpu()
})

test_that("ag_add GPU with [1,n] broadcast equals CPU", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  set.seed(23)
  a_mat <- matrix(runif(12), 3, 4)
  b_mat <- matrix(runif(4),  1, 4)
  expected <- a_mat + rep(b_mat, each = 3)

  ag_device("gpu")
  A <- ag_tensor(a_mat, device = "gpu")
  B <- ag_param(b_mat, device = "gpu")
  with_grad_tape({ out <- ag_add(A, B) })
  result <- ggmlR:::.ag_data(out)

  expect_equal(result, expected, tolerance = 1e-5)
  reset_to_cpu()
})
