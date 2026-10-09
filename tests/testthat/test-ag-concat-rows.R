# ag_concat_rows(): rbind(a, b) as ggml_concat along ne0. Tested: the backend
# concatenates along that axis in f32 (first, on the GPU), forward against
# rbind, shape and dtype errors, the gradient against finite differences, an
# operand without gradient, the GPU graph backward against the CPU, and use
# inside ag_capture.

skip_if_no_gpu_cat <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

set.seed(1L)
A0 <- matrix(rnorm(3 * 4), 3, 4)
B0 <- matrix(rnorm(1 * 4), 1, 4)
W0 <- matrix(rnorm(2 * 4), 2, 4)
T0 <- matrix(rnorm(2 * 4), 2, 4)

# loss = mse(W %*% rbind(A, B), T): 3 + 1 rows, as obs + action
cat_loss <- function(A, B, W) ag_mse_loss(ag_matmul(W, ag_concat_rows(A, B)), T0)

test_that("the GPU backend concatenates f32 along ne0 (rows)", {
  skip_if_no_gpu_cat()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  out <- ag_concat_rows(ag_tensor(A0), ag_tensor(B0))
  expect_equal(as.matrix(out), rbind(A0, B0), tolerance = 1e-6)
})

test_that("ag_concat_rows stacks rows and checks the shapes", {
  ag_device("cpu")
  expect_equal(as.matrix(ag_concat_rows(ag_tensor(A0), ag_tensor(B0))), rbind(A0, B0))
  expect_equal(as.matrix(ag_concat_rows(ag_tensor(B0), ag_tensor(A0))), rbind(B0, A0))
  expect_error(ag_concat_rows(ag_tensor(A0), ag_tensor(matrix(0, 1, 5))),
               "a has 4 columns, b has 5")
  expect_error(ag_concat_rows(ag_tensor(A0, dtype = "f32"), ag_tensor(B0, dtype = "f16")),
               "dtypes must match")
})

test_that("the gradient matches finite differences, for each operand", {
  ag_device("cpu")
  A <- ag_param(A0); B <- ag_param(B0); W <- ag_tensor(W0)
  expect_true(ag_gradcheck(fn = function(ins) cat_loss(ins$A, ins$B, W),
                           inputs = list(A = A, B = B), atol = 1e-4, quiet = TRUE))
})

test_that("an operand without gradient gets none", {
  ag_device("cpu")
  A <- ag_tensor(A0)                      # constant, e.g. observations
  B <- ag_param(B0)                       # e.g. the actor's action
  with_grad_tape(loss <- cat_loss(A, B, ag_tensor(W0)))
  backward(loss)
  expect_null(A$grad)
  ref <- (2 / length(T0)) * crossprod(W0, W0 %*% rbind(A0, B0) - T0)
  expect_equal(ag_grad(B), ref[4, , drop = FALSE], tolerance = 1e-6)
})

test_that("on the GPU forward and graph backward equal the CPU", {
  skip_if_no_gpu_cat()
  on.exit(ag_device("cpu"), add = TRUE)
  run <- function(dev, graph = TRUE, a_grad = TRUE) {
    ag_device(dev)
    old <- ggmlR:::ag_backward_graph(graph)
    on.exit(ggmlR:::ag_backward_graph(old), add = TRUE)
    A <- if (a_grad) ag_param(A0) else ag_tensor(A0)
    B <- ag_param(B0); W <- ag_param(W0)
    with_grad_tape(loss <- cat_loss(A, B, W))
    backward(loss)
    list(path = ggmlR:::ag_backward_path(), loss = as.numeric(as.matrix(loss)),
         gA = if (a_grad) ag_grad(A), gB = ag_grad(B), gW = ag_grad(W), A = A)
  }
  for (a_grad in c(TRUE, FALSE)) {
    cpu <- run("cpu", a_grad = a_grad)
    gpu <- run("gpu", graph = TRUE, a_grad = a_grad)
    info <- paste("a_grad", a_grad)
    expect_identical(gpu$path, "graph", info = info)
    expect_equal(gpu$loss, cpu$loss, tolerance = 1e-5, info = info)
    expect_equal(gpu$gA, cpu$gA, tolerance = 1e-5, info = info)
    expect_equal(gpu$gB, cpu$gB, tolerance = 1e-5, info = info)
    expect_equal(gpu$gW, cpu$gW, tolerance = 1e-5, info = info)
    if (!a_grad) expect_null(gpu$A$grad)
  }
})

test_that("ag_concat_rows works inside ag_capture", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L) "gpu")) {
    ag_device(dev)
    W <- ag_param(W0)
    f <- ag_capture(function(a, b) ag_matmul(W, ag_concat_rows(a, b)), params = list(W))
    expect_equal(f(a = A0, b = B0), W0 %*% rbind(A0, B0), tolerance = 1e-5, info = dev)
    A1 <- A0 + 1
    expect_equal(f(a = A1, b = B0), W0 %*% rbind(A1, B0), tolerance = 1e-5, info = dev)
  }
})
