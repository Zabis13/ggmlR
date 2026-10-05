# ag_capture(): a forward recorded once per input shape and replayed as one
# graph launch. Tested: replay equals the uncaptured forward (and the CPU),
# a replay is ONE launch with no R-level ops, re-recording on a new shape and
# on a changed parameter pointer, weights updated in place are seen by the
# replay, and the guard refuses host data frozen into the graph.

skip_if_no_gpu_cap <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}
launches <- function() ggmlR:::ag_launch_count()[["total"]]

mlp <- function(W1, b1, W2) function(x)
  list(h = ag_relu(ag_add(ag_matmul(W1, x), b1)),
       y = ag_matmul(W2, ag_relu(ag_add(ag_matmul(W1, x), b1))))

test_that("ag_capture on the CPU just runs fn", {
  ag_device("cpu")
  set.seed(1)
  W <- ag_param(matrix(rnorm(6), 2, 3))
  f <- ag_capture(function(x) ag_relu(ag_matmul(W, x)), params = list(W))
  x <- matrix(rnorm(12), 3, 4)
  expect_equal(f(x = x), pmax(as.matrix(W) %*% x, 0))
})

test_that("captured forward equals the uncaptured one and the CPU", {
  skip_if_no_gpu_cap()
  set.seed(2)
  w1 <- matrix(rnorm(8 * 5), 8, 5); b1 <- matrix(rnorm(8), 8, 1)
  w2 <- matrix(rnorm(3 * 8), 3, 8); x <- matrix(rnorm(5 * 6), 5, 6)
  ref <- pmax(w1 %*% x + as.vector(b1), 0)

  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  prev_prec <- ag_matmul_precision("f32")
  on.exit(ag_matmul_precision(prev_prec), add = TRUE)
  W1 <- ag_param(w1); B1 <- ag_param(b1); W2 <- ag_param(w2)
  f <- ag_capture(mlp(W1, B1, W2), params = list(W1, B1, W2))
  on.exit(ag_capture_free(f), add = TRUE)

  out1 <- f(x = x)                       # records, then replays
  out2 <- f(x = x)                       # replay only
  expect_named(out1, c("h", "y"))
  expect_equal(out1$h, ref, tolerance = 1e-5)
  expect_equal(out1$y, w2 %*% ref, tolerance = 1e-4)
  expect_identical(out2, out1)

  # A replay is one launch and dispatches no ag_* op in R.
  ggmlR:::ag_launch_count(reset = TRUE)
  x2 <- matrix(rnorm(5 * 6), 5, 6)
  out3 <- f(x = x2)
  expect_equal(launches(), 1)
  expect_equal(out3$h, pmax(w1 %*% x2 + as.vector(b1), 0), tolerance = 1e-5)
})

test_that("a new input shape records again; the old recording still works", {
  skip_if_no_gpu_cap()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(3)
  w <- matrix(rnorm(12), 4, 3)
  W <- ag_param(w)
  f <- ag_capture(function(x) ag_matmul(W, x), params = list(W))
  on.exit(ag_capture_free(f), add = TRUE)
  a <- matrix(rnorm(3 * 2), 3, 2); b <- matrix(rnorm(3 * 7), 3, 7)
  expect_equal(f(x = a), w %*% a, tolerance = 1e-5)
  expect_equal(f(x = b), w %*% b, tolerance = 1e-5)
  expect_equal(f(x = a), w %*% a, tolerance = 1e-5)
  expect_length(ls(attr(f, "captures")), 2L)
})

test_that("a weight updated in place is seen by the replay", {
  skip_if_no_gpu_cap()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(4)
  W <- ag_param(matrix(rnorm(6), 2, 3))
  f <- ag_capture(function(x) ag_matmul(W, x), params = list(W))
  on.exit(ag_capture_free(f), add = TRUE)
  x <- matrix(rnorm(3 * 5), 3, 5)
  f(x = x)
  # One optimizer step writes into W's existing device tensor.
  opt <- optimizer_sgd(list(W = W), lr = 0.1)
  with_grad_tape({ loss <- ag_sum(ag_matmul(W, ag_tensor(x))) })
  opt$step(backward(loss))
  w_new <- as.matrix(W)
  expect_equal(f(x = x), w_new %*% x, tolerance = 1e-5)
})

test_that("the guard refuses host data uploaded inside fn", {
  skip_if_no_gpu_cap()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  W <- ag_param(matrix(1, 2, 3))
  f <- ag_capture(function(x) ag_add(ag_matmul(W, x), ag_tensor(matrix(1, 2, 4))))
  expect_error(f(x = matrix(1, 3, 4)), "outside the graph")
  # Reading a value back inside fn is refused too.
  g <- ag_capture(function(x) { as.matrix(x); ag_matmul(W, x) })
  expect_error(g(x = matrix(1, 3, 4)), "outside the graph")
  # A failed recording leaves the session usable.
  h <- ag_capture(function(x) ag_matmul(W, x))
  expect_equal(h(x = matrix(1, 3, 4)), matrix(3, 2, 4), tolerance = 1e-6)
  ag_capture_free(h)
})

test_that("a device switch releases recordings; the next call records again", {
  skip_if_no_gpu_cap()
  ag_device("gpu")
  set.seed(5)
  w <- matrix(rnorm(6), 2, 3)
  W <- ag_param(w)
  f <- ag_capture(function(x) ag_matmul(W, x), params = list(W))
  x <- matrix(rnorm(9), 3, 3)
  f(x = x)
  cap <- get(ls(attr(f, "captures"))[1], envir = attr(f, "captures"))
  ag_device("cpu")
  expect_false(cap$alive)
  # Back on the GPU: W was materialised to the host by the switch; the capture
  # makes it resident again (params) and records anew.
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  expect_equal(f(x = x), w %*% x, tolerance = 1e-5)
  expect_true(get(ls(attr(f, "captures"))[1], envir = attr(f, "captures"))$alive)
  ag_capture_free(f)
})

test_that("an unchanged input is not re-sent, a changed one is", {
  skip_if_no_gpu_cap()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(6)
  w <- matrix(rnorm(12), 4, 3)
  W <- ag_param(w)
  f <- ag_capture(function(x, s) list(y = ag_mul(ag_matmul(W, x), s), w = W),
                  params = list(W))
  on.exit(ag_capture_free(f), add = TRUE)
  x <- matrix(rnorm(15), 3, 5)
  s1 <- matrix(c(1, 2, 3, 4), 4, 1); s2 <- matrix(c(-1, 0.5, 2, 0), 4, 1)
  f(x = x, s = s1)
  ups <- function() {
    r <- ggmlR:::.ag_xfer$counts[["up capture inputs"]]
    if (is.null(r)) 0L else r$n
  }
  ggmlR:::ag_xfer_count(TRUE)
  on.exit(ggmlR:::ag_xfer_count(FALSE), add = TRUE)
  out <- f(x = x, s = s1)
  expect_equal(ups(), 0L)                                 # nothing changed
  out2 <- f(x = x, s = s2)
  expect_equal(ups(), 1L)                                 # one transfer for s
  expect_equal(out$y, (w %*% x) * as.vector(s1), tolerance = 1e-5)
  expect_equal(out2$y, (w %*% x) * as.vector(s2), tolerance = 1e-5)
  # a parameter returned as an output comes back through the joined download
  expect_equal(out2$w, w, tolerance = 1e-6)
})
