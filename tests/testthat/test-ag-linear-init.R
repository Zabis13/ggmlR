# ag_linear(init =, gain =). The default must not move: the reference below was
# captured from the package BEFORE init existed (set.seed(42); ag_linear(5, 3);
# ag_linear(4, 1, "tanh")), not recomputed by the new code.

ref_W1 <- c(0.7184651426223132, 0.75703682257106242, -0.37041719148036939,
            0.57235207758912954, 0.24551044046789627, 0.033075154112140681,
            0.4097829814357663, -0.63277601528613614, 0.27191862337257944,
            0.35518262479513529, -0.073193390575315598, 0.37951355242278262,
            0.75287441670874333, -0.42360970230837125, -0.065310747164618932)
ref_W2 <- c(0.96402351897227101, 1.0477416096881784, -0.83804320221027861,
            -0.054778649729977813)

w_of <- function(layer) ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(layer$W))

# Not expect_identical: runif's a + (b - a) * u rounds 1 ULP differently on
# FMA builds (CI), so only a tolerance is portable; an RNG/formula change
# still moves values by orders of magnitude more than 1e-12.
test_that("default init matches the pre-init reference", {
  ag_device("cpu")
  set.seed(42)
  l1 <- ag_linear(5L, 3L)
  l2 <- ag_linear(4L, 1L, "tanh")          # activation still 3rd positional
  expect_equal(as.numeric(w_of(l1)), ref_W1, tolerance = 1e-12)
  expect_equal(as.numeric(w_of(l2)), ref_W2, tolerance = 1e-12)
  expect_identical(dim(w_of(l1)), c(3L, 5L))
})

test_that("orthogonal init is orthogonal on every shape, scaled by gain", {
  ag_device("cpu")
  for (sh in list(c(4L, 4L), c(3L, 7L), c(7L, 3L), c(1L, 6L), c(6L, 1L))) {
    out <- sh[1L]; inp <- sh[2L]
    W <- w_of(ag_linear(inp, out, init = "orthogonal", gain = 2))
    expect_identical(dim(W), c(out, inp))
    info <- paste0(out, "x", inp)
    if (out <= inp) {
      expect_equal(W %*% t(W), 4 * diag(out), tolerance = 1e-10, info = info)
    } else {
      expect_equal(t(W) %*% W, 4 * diag(inp), tolerance = 1e-10, info = info)
    }
  }
})

test_that("orthogonal init is reproducible from set.seed", {
  ag_device("cpu")
  set.seed(7); a <- w_of(ag_linear(5L, 3L, init = "orthogonal"))
  set.seed(7); b <- w_of(ag_linear(5L, 3L, init = "orthogonal"))
  expect_identical(a, b)
})

test_that("function and matrix inits are used as is, without drawing RNG", {
  ag_device("cpu")
  M <- matrix(1:6 / 10, 2, 3)
  set.seed(3); before <- runif(1)
  set.seed(3)
  W <- w_of(ag_linear(3L, 2L, init = M))
  expect_identical(runif(1), before)                 # stream untouched
  expect_equal(W, M, ignore_attr = TRUE)
  W2 <- w_of(ag_linear(3L, 2L, init = function(out, inp) matrix(0.5, out, inp)))
  expect_equal(W2, matrix(0.5, 2, 3), ignore_attr = TRUE)
})

test_that("bad inits are refused", {
  ag_device("cpu")
  M <- matrix(0.1, 2, 3)
  expect_error(ag_linear(3L, 2L, init = M, gain = 2), "gain applies to string")
  expect_error(ag_linear(3L, 2L, init = function(o, i) M, gain = 2), "gain applies")
  expect_error(ag_linear(3L, 2L, init = t(M)), "3x2, expected 2x3")
  expect_error(ag_linear(3L, 2L, init = function(o, i) t(M)), "expected 2x3")
  bad <- M; bad[1, 1] <- NA
  expect_error(ag_linear(3L, 2L, init = bad), "NA, NaN or Inf")
  bad[1, 1] <- Inf
  expect_error(ag_linear(3L, 2L, init = bad), "NA, NaN or Inf")
  expect_error(ag_linear(3L, 2L, init = "he_normal"), "should be one of")
  expect_error(ag_linear(3L, 2L, init = list(1)), "init must be")
  expect_error(ag_linear(3L, 2L, gain = NA), "gain must be")
})
