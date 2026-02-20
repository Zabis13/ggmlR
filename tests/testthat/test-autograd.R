# Tests for R-level dynamic graph / autograd

test_that("ag_tensor creates tensor with correct shape", {
  x <- ag_tensor(matrix(1:6, 2, 3))
  expect_s3_class(x, "ag_tensor")
  expect_equal(dim(x$data), c(2L, 3L))
  expect_false(x$requires_grad)
  expect_null(x$grad)
})

test_that("ag_param sets requires_grad = TRUE", {
  w <- ag_param(matrix(1:4, 2, 2))
  expect_true(w$requires_grad)
})

test_that("ag_matmul forward pass is correct", {
  A <- ag_param(matrix(c(1, 0, 0, 1), 2, 2))  # identity
  B <- ag_tensor(matrix(c(1, 2, 3, 4), 2, 2))
  out <- ag_matmul(A, B)
  expect_equal(out$data, B$data)
})

test_that("backward computes correct gradient for matmul", {
  # W filled column-major: matrix(c(1,2,3,4), 2,2) = [[1,3],[2,4]]
  # W %*% [1,1]^T = [1+3, 2+4] = [4, 6]
  W <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  x <- ag_tensor(matrix(c(1, 1), 2, 1))
  y_true <- matrix(c(4, 6), 2, 1)  # exact match -> zero loss

  with_grad_tape({
    out  <- ag_matmul(W, x)
    loss <- ag_mse_loss(out, y_true)
  })
  grads <- backward(loss)

  # loss should be 0
  expect_lt(abs(loss$data[1]), 1e-10)
  # grad should be near 0
  g <- get0(as.character(W$id), envir = grads)
  expect_lt(max(abs(g)), 1e-10)
})

test_that("backward: nonzero gradient for mse_loss", {
  W <- ag_param(matrix(c(1, 0, 0, 1), 2, 2))
  x <- ag_tensor(matrix(c(2, 3), 2, 1))
  y_true <- matrix(c(0, 0), 2, 1)

  with_grad_tape({
    out  <- ag_matmul(W, x)
    loss <- ag_mse_loss(out, y_true)
  })
  grads <- backward(loss)

  g <- get0(as.character(W$id), envir = grads)
  expect_false(is.null(g))
  expect_equal(dim(g), dim(W$data))
  # Analytical: d/dW MSE = 2/n * (Wx - y) x^T
  # n = 2, Wx = [2,3], y=0 -> residual = [2,3]
  # grad = 2/2 * [2,3] %*% t([2,3]) = [[4,6],[6,9]]
  expected_g <- matrix(c(4, 6, 6, 9), 2, 2)
  expect_equal(g, expected_g, tolerance = 1e-6)
})

test_that("relu backward: gradient is zero below 0", {
  # column-major fill: matrix(c(-1,2,-3,4), 2,2) = [[-1,-3],[2,4]]
  # [1,1]=-1 (neg), [2,1]=2 (pos), [1,2]=-3 (neg), [2,2]=4 (pos)
  x <- ag_param(matrix(c(-1, 2, -3, 4), 2, 2))
  with_grad_tape({
    out  <- ag_relu(x)
    loss <- ag_mse_loss(out, matrix(0, 2, 2))
  })
  grads <- backward(loss)
  g <- get0(as.character(x$id), envir = grads)
  # grad 0 where x < 0: [1,1] and [1,2]
  expect_equal(g[1, 1], 0)
  expect_equal(g[1, 2], 0)
  # grad nonzero where x > 0: [2,1] and [2,2]
  expect_gt(abs(g[2, 1]), 0)
  expect_gt(abs(g[2, 2]), 0)
})

test_that("sigmoid backward is correct", {
  x <- ag_param(matrix(c(0), 1, 1))
  with_grad_tape({
    out  <- ag_sigmoid(x)
    loss <- ag_mse_loss(out, matrix(0, 1, 1))
  })
  grads <- backward(loss)
  g <- get0(as.character(x$id), envir = grads)
  # sigmoid(0) = 0.5, d_sigmoid = 0.5*0.5 = 0.25
  # d_mse/d_out = 2/1*(0.5 - 0) = 1
  # grad = 1 * 0.25 = 0.25
  expect_equal(as.numeric(g), 0.25, tolerance = 1e-6)
})

test_that("cross_entropy_loss backward: grad shape matches pred", {
  # 3 classes, 4 samples
  set.seed(1)
  logits <- ag_param(matrix(runif(3 * 4), 3, 4))
  with_grad_tape({
    pred <- ag_softmax(logits)
    # one-hot targets
    y    <- matrix(c(1,0,0, 0,1,0, 0,0,1, 1,0,0), 3, 4)
    loss <- ag_cross_entropy_loss(pred, y)
  })
  grads <- backward(loss)
  g_logits <- get0(as.character(logits$id), envir = grads)
  # softmax + CE grad flows through softmax into logits
  # grad for pred: (p - y)/n
  # grad for logits from softmax JVP
  expect_equal(dim(g_logits), c(3L, 4L))
})

test_that("optimizer_sgd updates parameters", {
  W <- ag_param(matrix(c(1, 0, 0, 1), 2, 2))
  x <- ag_tensor(matrix(c(2, 3), 2, 1))
  y_true <- matrix(c(0, 0), 2, 1)

  W_orig <- W$data

  opt <- optimizer_sgd(list(W = W), lr = 0.1)

  with_grad_tape({
    out  <- ag_matmul(W, x)
    loss <- ag_mse_loss(out, y_true)
  })
  grads <- backward(loss)
  opt$step(grads)

  expect_false(identical(opt$params$W$data, W_orig))
})

test_that("optimizer_adam updates parameters", {
  W <- ag_param(matrix(c(1, 0, 0, 1), 2, 2))
  x <- ag_tensor(matrix(c(1, 1), 2, 1))
  y_true <- matrix(c(0, 0), 2, 1)

  opt <- optimizer_adam(list(W = W), lr = 0.01)
  W_orig <- W$data

  with_grad_tape({
    out  <- ag_matmul(W, x)
    loss <- ag_mse_loss(out, y_true)
  })
  grads <- backward(loss)
  opt$step(grads)

  expect_false(identical(opt$params$W$data, W_orig))
  expect_equal(opt$t, 1L)
})

test_that("ag_linear forward has correct output shape", {
  layer <- ag_linear(4L, 8L, activation = "relu")
  x     <- ag_tensor(matrix(runif(4 * 16), 4, 16))
  out   <- layer$forward(x)
  expect_equal(dim(out$data), c(8L, 16L))
  expect_true(all(out$data >= 0))  # ReLU
})

test_that("full training loop reduces loss", {
  set.seed(123)
  # XOR-like: 2 inputs -> 1 output
  n     <- 64L
  x_mat <- matrix(sample(c(0, 1), 2 * n, replace = TRUE), 2, n)
  y_mat <- matrix(as.numeric(xor(x_mat[1,], x_mat[2,])), 1, n)

  l1 <- ag_linear(2L, 4L, activation = "relu")
  l2 <- ag_linear(4L, 1L, activation = "sigmoid")

  params <- c(l1$params(), l2$params())
  opt    <- optimizer_adam(params, lr = 0.05)

  losses <- numeric(50L)
  for (i in seq_len(50L)) {
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
    losses[i] <- loss$data[1]
  }

  # Loss should decrease over training (average last 10 vs first 10)
  expect_lt(mean(losses[41:50]), mean(losses[1:10]))
})
