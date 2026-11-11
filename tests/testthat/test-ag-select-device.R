# Action selection that stays on the device (R/ag_select.R): ag_rand(), and the
# tensor-valued forms of ag_argmax(), ag_sample_categorical() and ag_gather().
# Together they let a recorded graph (ag_capture) pick an action and return
# its log-probability with nothing but a two-number key going up.

skip_if_no_gpu_seldev <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

host <- function(t) as.matrix(t)

test_that("ag_rand on the CPU device: shape, key, stream, moments", {
  ag_device("cpu")
  k <- c(0.25, 0.75)
  u <- ag_rand(k, "uniform", c(3, 1000))
  expect_true(inherits(u, "ag_tensor"))
  expect_identical(dim(host(u)), c(3L, 1000L))
  expect_identical(host(ag_rand(k, "uniform", c(3, 1000))), host(u))
  expect_lt(mean(host(ag_rand(k, "uniform", c(3, 1000), stream = 1L)) == host(u)), 0.01)
  expect_lt(mean(host(ag_rand(c(0.25, 0.5), "uniform", c(3, 1000))) == host(u)), 0.01)
  expect_true(all(host(u) > 0 & host(u) < 1))
  z <- host(ag_rand(k, "normal", c(4, 20000)))
  expect_lt(abs(mean(z)), 0.02)
  expect_equal(var(as.numeric(z)), 1, tolerance = 0.03)
  # a key held in a tensor, and set.seed driving the key
  expect_identical(host(ag_rand(ag_tensor(matrix(k, 2, 1)), "uniform", c(3, 1000))), host(u))
  set.seed(5); a <- host(ag_rand(runif(2), "gumbel", c(2, 8)))
  set.seed(5); b <- host(ag_rand(runif(2), "gumbel", c(2, 8)))
  expect_identical(a, b)
  expect_error(ag_rand(1, "uniform", c(2, 2)), "key")
  expect_error(ag_rand(k, "uniform", 4), "dims")
})

test_that("ag_rand draws the same numbers on the GPU and the CPU device", {
  skip_if_no_gpu_seldev()
  k <- c(0.3, 0.6)
  ag_device("cpu")
  u_cpu <- host(ag_rand(k, "uniform", c(5, 700), stream = 2L))
  z_cpu <- host(ag_rand(k, "normal", c(5, 700)))
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  expect_identical(host(ag_rand(k, "uniform", c(5, 700), stream = 2L)), u_cpu)
  expect_equal(host(ag_rand(k, "normal", c(5, 700))), z_cpu, tolerance = 1e-4)
  # resident key -> resident result
  kt <- ag_tensor(matrix(k, 2, 1))
  expect_identical(host(ag_rand(kt, "uniform", c(5, 700), stream = 2L)), u_cpu)
})

test_that("ag_argmax(as_tensor = TRUE) and ag_gather(index tensor), both devices", {
  set.seed(11)
  m <- matrix(rnorm(7 * 40), 7, 40)
  m[c(2, 5), 3] <- 9                     # a tie: first index wins
  ref <- apply(m, 2, which.max) - 1L
  ref_pick <- m[cbind(ref + 1L, seq_len(ncol(m)))]
  check <- function() {
    a <- ag_argmax(ag_tensor(m), as_tensor = TRUE)
    expect_true(inherits(a, "ag_tensor"))
    expect_identical(dim(host(a)), c(1L, 40L))
    expect_equal(as.numeric(host(a)), ref, tolerance = 0)
    expect_equal(as.numeric(host(ag_gather(ag_tensor(m), a))), ref_pick, tolerance = 1e-6)
    # any index tensor, not only the argmax
    idx <- (seq_len(40) * 3L) %% 7L
    g <- ag_gather(ag_tensor(m), ag_tensor(matrix(idx, 1)))
    expect_equal(as.numeric(host(g)), m[cbind(idx + 1L, 1:40)], tolerance = 1e-6)
  }
  ag_device("cpu")
  check()
  skip_if_no_gpu_seldev()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  check()
})

test_that("ag_sample_categorical(key): distribution, mask, CPU/GPU agreement", {
  B <- 20000L
  logit <- c(1.5, 0, -1, -1e4)             # the last action is masked
  m <- matrix(logit, 4, B)
  p <- exp(logit - max(logit)); p <- p / sum(p)
  k <- c(0.11, 0.87)
  draw <- function() as.integer(host(ag_sample_categorical(ag_tensor(m), key = k)))
  ag_device("cpu")
  a_cpu <- draw()
  expect_true(inherits(ag_sample_categorical(ag_tensor(m), key = k), "ag_tensor"))
  expect_false(any(a_cpu == 3L))
  expect_equal(as.numeric(table(factor(a_cpu, 0:2))) / B, p[1:3], tolerance = 0.03)
  expect_identical(draw(), a_cpu)
  expect_lt(mean(as.integer(host(ag_sample_categorical(ag_tensor(m), key = k,
                                                       stream = 1L))) == a_cpu), 0.9)
  skip_if_no_gpu_seldev()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  a_gpu <- draw()
  # same noise bits; only a near-tie of x + G can round differently
  expect_gte(mean(a_gpu == a_cpu), 0.999)
})

test_that("a recorded graph selects an action and its logp from a key", {
  skip_if_no_gpu_seldev()
  set.seed(21)
  w <- matrix(rnorm(6 * 5), 6, 5)
  x <- matrix(rnorm(5 * 32), 5, 32)
  logp_ref <- apply(w %*% x, 2, function(v) { z <- v - max(v); z - log(sum(exp(z))) })

  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  # f32 x f32 matmul runs on f16 tiles by default; ask for exact f32
  old_prec <- ag_matmul_precision("f32")
  on.exit(ag_matmul_precision(old_prec), add = TRUE)
  W <- ag_param(w)
  act <- function(greedy) function(x, key) {
    lp <- ag_log_softmax(ag_matmul(W, x))
    a <- if (greedy) ag_argmax(lp, as_tensor = TRUE) else ag_sample_categorical(lp, key = key)
    list(a = a, logp = ag_gather(lp, a))
  }
  f <- ag_capture(act(FALSE), params = list(W))
  on.exit(ag_capture_free(f), add = TRUE)
  k1 <- matrix(c(0.2, 0.9), 2, 1); k2 <- matrix(c(0.4, 0.1), 2, 1)
  r1 <- f(x = x, key = k1)
  r2 <- f(x = x, key = k2)
  r1b <- f(x = x, key = k1)
  expect_identical(dim(r1$a), c(1L, 32L))
  expect_true(all(r1$a == round(r1$a) & r1$a >= 0 & r1$a <= 5))
  expect_identical(r1b$a, r1$a)                    # same key, same draw
  expect_false(identical(r2$a, r1$a))              # new key, new draw
  for (r in list(r1, r2))
    expect_equal(as.numeric(r$logp), logp_ref[cbind(as.integer(r$a) + 1L, 1:32)],
                 tolerance = 1e-4)

  g <- ag_capture(act(TRUE), params = list(W))
  on.exit(ag_capture_free(g), add = TRUE)
  rg <- g(x = x, key = k1)
  expect_equal(as.integer(rg$a), apply(logp_ref, 2, which.max) - 1L)
  expect_equal(as.numeric(rg$logp), apply(logp_ref, 2, max), tolerance = 1e-4)

  # the CPU device with the same key picks the same actions
  ag_device("cpu")
  Wc <- ag_param(w)
  lp <- ag_log_softmax(ag_matmul(Wc, ag_tensor(x)))
  a_cpu <- as.integer(host(ag_sample_categorical(lp, key = k1)))
  expect_gte(mean(a_cpu == as.integer(r1$a)), 0.95)
})
