# ag_sample_categorical (Gumbel-max, noise from runif) and ag_sample_normal
# (mu + exp(log_std) * rnorm). Both take their noise from R's RNG, so the CPU
# and GPU paths see identical noise under the same seed.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

test_that("ag_sample_categorical frequencies match softmax (CPU)", {
  set.seed(42)
  x <- c(1, 0, -1, 2)
  B <- 20000L
  a <- ag_sample_categorical(matrix(x, 4, B))
  expect_true(is.integer(a))
  expect_length(a, B)
  p <- exp(x) / sum(exp(x))
  f <- tabulate(a + 1L, 4L) / B
  # 4 binomial standard errors per class
  expect_true(all(abs(f - p) < 4 * sqrt(p * (1 - p) / B)))
})

test_that("ag_sample_categorical never draws a masked entry", {
  set.seed(1)
  x <- matrix(c(0, -1e4, 0, -1e4), 4, 5000)
  a <- ag_sample_categorical(x)
  expect_true(all(a %in% c(0L, 2L)))
})

test_that("ag_sample_categorical is reproducible under set.seed", {
  x <- matrix(rnorm(30), 3, 10)
  set.seed(7); a1 <- ag_sample_categorical(x)
  set.seed(7); a2 <- ag_sample_categorical(x)
  expect_identical(a1, a2)
})

test_that("ag_sample_categorical GPU matches CPU under the same seed", {
  skip_if_no_gpu()
  set.seed(3)
  x <- matrix(rnorm(9 * 512), 9, 512)
  set.seed(11); a_cpu <- ag_sample_categorical(ag_tensor(x, device = "cpu"))
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(11); a_gpu <- ag_sample_categorical(ag_tensor(x, device = "gpu"))
  # Near-ties round in f32 on the GPU and in double on the CPU.
  expect_gte(mean(a_cpu == a_gpu), 0.995)
  # Masking holds on the device too.
  xm <- x; xm[c(2, 5), ] <- xm[c(2, 5), ] - 1e4
  set.seed(12); a_m <- ag_sample_categorical(ag_tensor(xm, device = "gpu"))
  expect_false(any(a_m %in% c(1L, 4L)))
})

test_that("ag_sample_normal moments and reproducibility (CPU)", {
  mu <- matrix(c(1, -2), 2, 20000)
  ls <- matrix(log(c(0.5, 2)), 2, 1)
  set.seed(5); s1 <- as.matrix(ag_sample_normal(mu, ls))
  set.seed(5); s2 <- as.matrix(ag_sample_normal(mu, ls))
  expect_identical(s1, s2)
  expect_equal(rowMeans(s1), c(1, -2), tolerance = 0.05)
  expect_equal(apply(s1, 1, sd), c(0.5, 2), tolerance = 0.05)
})

test_that("ag_sample_normal is differentiable in mu and log_std", {
  mu <- ag_param(matrix(0, 2, 4))
  ls <- ag_param(matrix(0, 2, 1))
  set.seed(9)
  eps <- matrix(rnorm(8), 2, 4)
  set.seed(9)
  with_grad_tape({
    loss <- ag_sum(ag_sample_normal(mu, ls))
  })
  g <- backward(loss)
  expect_equal(as.matrix(get0(as.character(mu$id), envir = g)), matrix(1, 2, 4))
  # d/dls sum(mu + exp(ls) * eps) = rowSums(eps) at ls = 0
  expect_equal(as.numeric(get0(as.character(ls$id), envir = g)), rowSums(eps), tolerance = 1e-6)
})

test_that("ag_sample_normal rejects a mismatched log_std", {
  expect_error(ag_sample_normal(matrix(0, 2, 3), matrix(0, 3, 1)), "log_std")
})

test_that("ag_sample_normal GPU matches CPU under the same seed", {
  skip_if_no_gpu()
  mu <- matrix(rnorm(6 * 64), 6, 64)
  ls <- matrix(rnorm(6, sd = 0.3), 6, 1)
  set.seed(21); s_cpu <- as.matrix(ag_sample_normal(mu, ls))
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(21)
  s_gpu <- as.matrix(ag_sample_normal(ag_tensor(mu, device = "gpu"),
                                      ag_tensor(ls, device = "gpu")))
  expect_equal(s_gpu, s_cpu, tolerance = 1e-5)
})
