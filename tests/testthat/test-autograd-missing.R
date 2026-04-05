# Tests for autograd ops not covered elsewhere: ag_tanh

test_that("ag_tanh forward pass is correct", {
  x <- ag_tensor(matrix(c(0, 1, -1, 2), 2, 2))
  y <- ag_tanh(x)
  expect_equal(y$data, tanh(x$data), tolerance = 1e-6)
})

test_that("ag_tanh backward computes correct gradient", {
  x <- ag_param(matrix(c(0.5, -0.5, 1.0, -1.0), 2, 2))

  with_grad_tape({
    y <- ag_tanh(x)
    loss <- ag_sum(y)
  })
  grads <- backward(loss)
  g <- get0(as.character(x$id), envir = grads)
  expect_false(is.null(g))
  # d/dx tanh(x) = 1 - tanh(x)^2
  expected <- 1 - tanh(x$data)^2
  expect_equal(g, expected, tolerance = 1e-5)
})
