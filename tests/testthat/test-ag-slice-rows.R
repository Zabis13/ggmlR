# ag_slice_rows(): rows from+1..from+n as a device view made contiguous.
# Tested: forward against a host reference, bounds, the gradient against
# finite differences (CPU), the GPU graph backward against the CPU closures,
# and use inside ag_capture together with ag_get_rows (a packed batch split
# into fields).

skip_if_no_gpu_slice <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

set.seed(1L)
X0 <- matrix(rnorm(6 * 4), 6, 4)
W0 <- matrix(rnorm(3 * 2), 3, 2)
T0 <- matrix(rnorm(3 * 4), 3, 4)

# loss = mse(W %*% X[3:4, ], T): the gradient of X is nonzero in rows 3:4 only
slice_loss <- function(X, W) ag_mse_loss(ag_matmul(W, ag_slice_rows(X, 2, 2)), T0)

test_that("ag_slice_rows returns the rows and checks the range", {
  ag_device("cpu")
  x <- ag_tensor(X0)
  expect_equal(as.matrix(ag_slice_rows(x, 0, 1)), X0[1, , drop = FALSE])
  expect_equal(as.matrix(ag_slice_rows(x, 2, 3)), X0[3:5, ])
  expect_equal(as.matrix(ag_slice_rows(x, 0, 6)), X0)
  expect_error(ag_slice_rows(x, 4, 3), "outside x")
  expect_error(ag_slice_rows(x, -1, 2), "from")
  expect_error(ag_slice_rows(x, 0, 0), "n must")
  expect_error(ag_slice_rows(x, 1.5, 2), "from")
})

test_that("the gradient matches finite differences", {
  ag_device("cpu")
  X <- ag_param(X0)
  ok <- ag_gradcheck(fn = function(ins) slice_loss(ins$X, ag_tensor(W0)),
                     inputs = list(X = X), atol = 1e-4, quiet = TRUE)
  expect_true(ok)
  # and is zero outside the slice
  with_grad_tape(loss <- slice_loss(X, ag_tensor(W0)))
  backward(loss)
  g <- ag_grad(X)
  expect_equal(g[-(3:4), ], matrix(0, 4, 4))
  expect_false(all(g[3:4, ] == 0))
})

test_that("on the GPU forward and graph backward equal the CPU", {
  skip_if_no_gpu_slice()
  on.exit(ag_device("cpu"), add = TRUE)
  run <- function(dev, graph = TRUE) {
    ag_device(dev)
    old <- ggmlR:::ag_backward_graph(graph)
    on.exit(ggmlR:::ag_backward_graph(old), add = TRUE)
    X <- ag_param(X0); W <- ag_param(W0)
    with_grad_tape(loss <- slice_loss(X, W))
    backward(loss)
    list(path = ggmlR:::ag_backward_path(), loss = as.numeric(as.matrix(loss)),
         gX = ag_grad(X), gW = ag_grad(W))
  }
  cpu <- run("cpu")
  gpu <- run("gpu", graph = TRUE)
  expect_identical(gpu$path, "graph")
  expect_equal(gpu$loss, cpu$loss, tolerance = 1e-5)
  expect_equal(gpu$gX, cpu$gX, tolerance = 1e-5)
  expect_equal(gpu$gW, cpu$gW, tolerance = 1e-5)
  expect_equal(gpu$gX[-(3:4), ], matrix(0, 4, 4))
  closures <- run("gpu", graph = FALSE)
  expect_equal(closures$gX, cpu$gX, tolerance = 1e-5)
})

test_that("a packed batch splits into fields inside ag_capture", {
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(2L)
  # columns: [obs (3) | act (2) | reward (1)], 20 transitions
  packed <- matrix(rnorm(6 * 20), 6, 20)
  idx <- matrix(c(19, 0, 7, 7), ncol = 1)
  b <- packed[, idx + 1]
  for (dev in c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L) "gpu")) {
    ag_device(dev)
    buf <- ag_buffer(6, 20)
    ag_buffer_write(buf, packed, 0)
    f <- ag_capture(function(idx) {
      x <- ag_get_rows(buf, idx)
      list(obs = ag_slice_rows(x, 0, 3), act = ag_slice_rows(x, 3, 2),
           reward = ag_slice_rows(x, 5, 1))
    }, params = list(buf))
    out <- f(idx = idx)
    expect_equal(out$obs, b[1:3, ], tolerance = 1e-6, info = dev)
    expect_equal(out$act, b[4:5, ], tolerance = 1e-6, info = dev)
    expect_equal(out$reward, b[6, , drop = FALSE], tolerance = 1e-6, info = dev)
    ag_buffer_free(buf)
  }
})
