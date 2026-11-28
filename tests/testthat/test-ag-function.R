# ag_function(): user-defined differentiable ops (R/ag_function.R)

m <- function(v, nr = 2L) matrix(v, nrow = nr)

pow_fn <- ag_function(
  forward = function(ctx, x, p) {
    ctx$save_for_backward(x = x)
    ctx$p <- p
    x^p
  },
  backward = function(ctx, grad_output) {
    x <- ctx$saved$x
    list(x = grad_output * ctx$p * x^(ctx$p - 1))
  })

# y = A %*% B with both gradients
matmul_fn <- ag_function(
  forward = function(ctx, A, B) {
    ctx$save_for_backward(A = A, B = B)
    A %*% B
  },
  backward = function(ctx, grad_output) {
    list(A = grad_output %*% t(ctx$saved$B),
         B = t(ctx$saved$A) %*% grad_output)
  })

test_that("forward value and gradient match the built-in op", {
  ag_device("cpu")
  x0 <- m(c(1.2, -0.7, 0.5, 2.1))
  x1 <- ag_param(x0); x2 <- ag_param(x0)
  # with_grad_tape() starts a fresh tape, so each backward follows its own tape
  with_grad_tape(l1 <- ag_sum(pow_fn(x1, 3)))
  backward(l1)
  with_grad_tape(l2 <- ag_sum(ag_pow(x2, 3)))
  backward(l2)
  expect_equal(as.matrix(pow_fn(ag_tensor(x0), 3)), x0^3, tolerance = 1e-6)
  expect_equal(ag_grad(x1), ag_grad(x2), tolerance = 1e-6)
  expect_equal(ag_grad(x1), 3 * x0^2, tolerance = 1e-6)
})

test_that("gradcheck: two tensor inputs, chained with built-in ops", {
  ag_device("cpu")
  set.seed(3)
  A <- ag_param(matrix(rnorm(6), 2, 3))
  B <- ag_param(matrix(rnorm(12), 3, 4))
  expect_true(ag_gradcheck(
    fn = function(ins) ag_sum(ag_tanh(matmul_fn(ins$A, ins$B))),
    inputs = list(A = A, B = B), atol = 1e-4, quiet = TRUE))
  # positional and named calls bind the same way
  expect_true(ag_gradcheck(
    fn = function(ins) ag_sum(ag_scale(matmul_fn(B = ins$B, A = ins$A), 2)),
    inputs = list(A = A, B = B), atol = 1e-4, quiet = TRUE))
})

test_that("a non-tensor argument and a constant input get no gradient", {
  ag_device("cpu")
  A <- ag_param(m(c(1, 2, 3, 4)))
  Bc <- ag_tensor(m(c(0.5, -1, 2, 1)))      # requires_grad = FALSE
  seen <- NULL
  f <- ag_function(
    forward = function(ctx, A, B) { seen <<- ctx$needs_input_grad; ctx$B <- B; A * B },
    backward = function(ctx, grad_output) list(A = grad_output * ctx$B, B = NULL))
  with_grad_tape(l <- ag_sum(f(A, Bc)))
  expect_identical(seen, c(A = TRUE, B = FALSE))
  backward(l)
  expect_equal(ag_grad(A), m(c(0.5, -1, 2, 1)), tolerance = 1e-6)
  expect_null(Bc$grad)
})

test_that("used twice on the tape, gradients accumulate", {
  ag_device("cpu")
  x0 <- m(c(1, 2, 3, 4)); x <- ag_param(x0)
  with_grad_tape(l <- ag_add(ag_sum(pow_fn(x, 2)), ag_sum(pow_fn(x, 3))))
  backward(l)
  expect_equal(ag_grad(x), 2 * x0 + 3 * x0^2, tolerance = 1e-6)
})

test_that("no tape, or no input requiring a gradient: forward only", {
  ag_device("cpu")
  y <- pow_fn(ag_tensor(m(1:4)), 2)
  expect_false(y$requires_grad)
  expect_null(y$grad_fn)
})

test_that("backward results are checked: names and shapes", {
  ag_device("cpu")
  x <- ag_param(m(c(1, 2, 3, 4)))
  bad_shape <- ag_function(function(ctx, x) x * 2,
                           function(ctx, grad_output) list(x = matrix(1, 1, 4)))
  with_grad_tape(l <- ag_sum(bad_shape(x)))
  expect_error(backward(l), "gradient for 'x' has shape \\(1, 4\\); expected \\(2, 2\\)")

  bad_name <- ag_function(function(ctx, x) x * 2,
                          function(ctx, grad_output) list(y = grad_output))
  with_grad_tape(l <- ag_sum(bad_name(x)))
  expect_error(backward(l), "unknown: y")

  # a plain vector of the right length is reshaped
  flat <- ag_function(function(ctx, x) x * 2,
                      function(ctx, grad_output) list(x = as.vector(grad_output) * 2))
  x2 <- ag_param(m(c(1, 2, 3, 4)))
  with_grad_tape(l <- ag_sum(flat(x2)))
  backward(l)
  expect_equal(ag_grad(x2), m(rep(2, 4)), tolerance = 1e-6)
})

test_that("ag_function validates its arguments", {
  expect_error(ag_function("f", function(ctx, g) NULL), "forward must be")
  expect_error(ag_function(function(ctx, x) x, "b"), "backward must be")
  expect_error(ag_function(function() 1, function(ctx, g) NULL), "ctx as its first")
  expect_error(ag_function(function(ctx, x) x, function(ctx) NULL), "\\(ctx, grad_output\\)")
  f <- ag_function(function(ctx, x) "a", function(ctx, g) list())
  expect_error(f(ag_tensor(m(1:4))), "numeric matrix or vector")
})

test_that("ag_function refuses to be recorded by ag_capture", {
  reg <- .ag_capture_reg
  old <- reg$recording
  reg$recording <- TRUE
  on.exit(reg$recording <- old)
  expect_error(pow_fn(ag_tensor(m(1:4)), 2), "cannot be recorded by ag_capture")
})

test_that("on the GPU: value and gradient match the CPU", {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
  on.exit(ag_device("cpu"))
  ag_device("gpu")
  set.seed(5)
  a0 <- matrix(rnorm(6), 2, 3); b0 <- matrix(rnorm(12), 3, 4)
  A <- ag_param(a0); B <- ag_param(b0)
  # built-in ops on both sides, so grad_output reaches backward from a resident tape
  with_grad_tape(l <- ag_sum(ag_tanh(matmul_fn(ag_relu(A), B))))
  backward(l)
  h <- pmax(a0, 0); y <- h %*% b0; g <- 1 - tanh(y)^2
  expect_equal(ag_grad(A), (g %*% t(b0)) * (a0 > 0), tolerance = 1e-4)
  expect_equal(ag_grad(B), t(h) %*% g, tolerance = 1e-4)
})
