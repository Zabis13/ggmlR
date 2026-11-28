# ag_permute(): permute the sample axes packed in the rows (R/autograd.R)

ref_permute <- function(m, shape, perm)
  matrix(aperm(array(m, c(shape, ncol(m))), c(perm, length(shape) + 1L)),
         prod(shape), ncol(m))

# every permutation of 1:k (k <= 4)
perms <- function(k) {
  if (k == 1L) return(list(1L))
  out <- list()
  for (p in perms(k - 1L)) for (i in 0:(k - 1L))
    out[[length(out) + 1L]] <- append(p, k, after = i)
  out
}

SHAPES <- list(c(5L), c(2L, 3L), c(2L, 3L, 4L), c(2L, 3L, 2L, 4L))

test_that("CPU forward matches aperm for every permutation, 1 to 4 axes", {
  ag_device("cpu")
  set.seed(1)
  for (s in SHAPES) {
    m <- matrix(rnorm(prod(s) * 3), prod(s), 3)
    for (p in perms(length(s)))
      expect_equal(as.matrix(ag_permute(ag_tensor(m), s, p)), ref_permute(m, s, p))
  }
  # more axes than the GPU takes: fine on the CPU
  s5 <- c(2L, 1L, 3L, 2L, 2L); m <- matrix(rnorm(24 * 2), 24, 2)
  expect_equal(as.matrix(ag_permute(ag_tensor(m), s5, c(5L, 3L, 1L, 4L, 2L))),
               ref_permute(m, s5, c(5L, 3L, 1L, 4L, 2L)))
})

test_that("gradcheck on the CPU (3 and 4 axes)", {
  ag_device("cpu")
  set.seed(2)
  W  <- ag_param(matrix(rnorm(24 * 2), 24, 2))
  wt <- matrix(rnorm(48), 24, 2)     # non-uniform weights so the order matters
  for (case in list(list(c(2L, 3L, 4L), c(3L, 1L, 2L)),
                    list(c(2L, 3L, 2L, 2L), c(4L, 2L, 1L, 3L))))
    expect_true(ag_gradcheck(
      fn = function(ins) ag_sum(ag_mul(ag_permute(ins$W, case[[1]], case[[2]]), wt)),
      inputs = list(W = W), atol = 1e-4, quiet = TRUE))
})

test_that("permute then inverse permute is the identity, value and gradient", {
  ag_device("cpu")
  s <- c(2L, 3L, 4L); p <- c(2L, 3L, 1L)
  x0 <- matrix(as.numeric(1:48), 24, 2); x <- ag_param(x0)
  wt <- matrix(seq_len(48) / 10, 24, 2)
  with_grad_tape({
    y <- ag_permute(ag_permute(x, s, p), s[p], order(p))
    l <- ag_sum(ag_mul(y, wt))
  })
  expect_equal(as.matrix(y), x0)
  backward(l)
  expect_equal(ag_grad(x), wt)
})

test_that("ag_permute validates its arguments", {
  ag_device("cpu")
  x <- ag_tensor(matrix(0, 6, 2))
  expect_error(ag_permute(x, c(2, 3), c(1, 1)), "permutation of 1:2")
  expect_error(ag_permute(x, c(2, 3), c(1, 2, 3)), "permutation of 1:2")
  expect_error(ag_permute(x, c(2, 4), c(2, 1)), "6 rows; prod\\(shape\\) is 8")
  expect_error(ag_permute(x, c(0, 3), c(2, 1)), "positive integers")
})

skip_if_no_vk <- function()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")

test_that("GPU forward matches the CPU for every permutation, 1 to 4 axes (5D)", {
  skip_if_no_vk()
  on.exit(ag_device("cpu"))
  ag_device("gpu")
  set.seed(3)
  for (s in SHAPES) {
    m <- matrix(rnorm(prod(s) * 3), prod(s), 3)
    for (p in perms(length(s)))
      expect_equal(as.matrix(ag_permute(ag_tensor(m), s, p)), ref_permute(m, s, p),
                   tolerance = 1e-6, info = paste(c(s, "|", p), collapse = " "))
  }
  expect_error(ag_permute(ag_tensor(matrix(0, 32, 1)), rep(2L, 5), 5:1),
               "at most 4 axes on the GPU")
})

test_that("GPU gradient (graph backward rule) matches the CPU, 3 and 4 axes", {
  skip_if_no_vk()
  on.exit(ag_device("cpu"))
  set.seed(4)
  x0 <- matrix(rnorm(48 * 3), 48, 3); wt <- matrix(rnorm(48 * 3), 48, 3)
  grad_on <- function(dev, s, p) {
    ag_device(dev)
    x <- ag_param(x0)
    with_grad_tape(l <- ag_sum(ag_mul(ag_tanh(ag_permute(x, s, p)), wt)))
    backward(l)
    ag_grad(x)
  }
  for (case in list(list(c(4L, 3L, 4L), c(3L, 1L, 2L)),
                    list(c(2L, 3L, 2L, 4L), c(4L, 1L, 3L, 2L)))) {
    g_cpu <- grad_on("cpu", case[[1]], case[[2]])
    g_gpu <- grad_on("gpu", case[[1]], case[[2]])
    expect_equal(g_gpu, g_cpu, tolerance = 1e-4)
  }
})

test_that("GPU 5D 0<->2 tile copy: sizes not multiples of the 32 tile, every batch", {
  # A permute whose source axis 0 lands at position 2 selects the tiled
  # copy_transpose_02 shader; it used to copy only ne4 index 0 (the first sample).
  # ne0 = 37 and ne2 = 33 straddle the 32x32 tile; all dims > 1 and distinct so a
  # wrong product in the shader's z loop or the dispatch cannot hide.
  skip_if_no_vk()
  on.exit(ag_device("cpu"))
  ag_device("gpu")
  set.seed(6)
  s <- c(37L, 5L, 33L, 3L); B <- 4L
  m <- matrix(rnorm(prod(s) * B), prod(s), B)
  for (p in list(c(3L, 2L, 1L, 4L), c(2L, 3L, 1L, 4L), c(4L, 2L, 1L, 3L),
                 c(3L, 4L, 1L, 2L), c(2L, 1L, 4L, 3L), c(1L, 2L, 3L, 4L)))
    expect_equal(as.matrix(ag_permute(ag_tensor(m), s, p)), ref_permute(m, s, p),
                 tolerance = 1e-6, info = paste(p, collapse = ""))
  # the same 0<->2 swap with 3 axes (4D view): ne3 = B is the batch
  s3 <- c(37L, 5L, 33L)
  m3 <- matrix(rnorm(prod(s3) * B), prod(s3), B)
  expect_equal(as.matrix(ag_permute(ag_tensor(m3), s3, c(3L, 2L, 1L))),
               ref_permute(m3, s3, c(3L, 2L, 1L)), tolerance = 1e-6)
})
