# Component 2 of the resident contract: a tensor whose value lives on the
# device and is materialised only when asked for.
#
# Component 1 gave operations a way to hand back a device handle. This is the
# other half: an ag_tensor that HOLDS one, so a result can become a tensor
# without a download. The contract fields ($ptr/$shape/$ctx_gen/$data_gen) and
# .ag_data()'s ability to read them already existed -- what is new is creating
# a tensor in that state, and the tests below are about whether such a tensor
# is indistinguishable from an ordinary one everywhere it matters.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns        <- asNamespace("ggmlR")
run_op    <- get(".ag_run_op",            envir = ns)
from_h    <- get(".ag_tensor_from_handle", envir = ns)
set_h     <- get(".ag_data_set_handle",   envir = ns)
handle_of <- get(".ag_handle_of",         envir = ns)
ag_data   <- get(".ag_data",              envir = ns)
ag_set    <- get(".ag_data_set",          envir = ns)
is_handle <- get(".ag_is_handle",         envir = ns)

# An op that leaves its result on the device, for building fixtures.
resident_scale <- function(m, k) {
  run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], k),
         inputs = list(m), out_shape = dim(m), resident = TRUE)
}

test_that("a tensor built from a handle has no host copy until read", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m <- matrix(rnorm(64), 8, 8)
  t <- from_h(resident_scale(m, 2))

  # rule 4: $data NULL means "not materialised", never "empty"
  expect_null(t$data)
  expect_false(is.null(t$ptr))
  expect_equal(t$shape, c(8L, 8L))

  # ... and reading it produces the value, from the device
  expect_equal(ag_data(t), m * 2, tolerance = 1e-4)
})

test_that("materialising caches, so a second read does not download again", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m <- matrix(rnorm(64), 8, 8)
  t <- from_h(resident_scale(m, 3))

  v1 <- ag_data(t)
  expect_false(is.null(t$data))                 # cached now
  expect_identical(t$data_gen, t$ctx_gen)       # tagged with its generation
  v2 <- ag_data(t)
  expect_identical(v1, v2)
})

test_that("installing a handle on an existing tensor drops the host copy", {
  # Two values with no way to tell which is current is exactly the failure the
  # contract exists to prevent, so the old matrix must go.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m   <- matrix(rnorm(64), 8, 8)
  t   <- ag_tensor(m, device = "gpu")
  expect_false(is.null(t$data))

  set_h(t, resident_scale(m, 5))
  expect_null(t$data)
  expect_equal(ag_data(t), m * 5, tolerance = 1e-4)
})

test_that(".ag_handle_of round-trips a resident tensor without downloading", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m <- matrix(rnorm(64), 8, 8)
  t <- from_h(resident_scale(m, 2))

  h <- handle_of(t)
  expect_true(is_handle(h))
  expect_equal(h$shape, c(8L, 8L))
  expect_null(t$data)                # asking for the handle materialised nothing

  # and it can be fed straight back into an op
  out <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 2),
                inputs = list(h), out_shape = c(8L, 8L))
  expect_equal(out, m * 4, tolerance = 1e-4)
})

test_that(".ag_handle_of returns NULL for a host-side tensor", {
  # Callers are meant to write `.ag_handle_of(x) %||% .ag_data(x)`, so the
  # no-residency case has to be a plain NULL rather than an error.
  ag_device("cpu")
  t <- ag_tensor(matrix(rnorm(4), 2, 2))
  expect_null(handle_of(t))
  expect_null(handle_of(42))          # and a non-tensor is not an error either
})

test_that(".ag_data_set on a resident tensor drops residency", {
  # Writing a new value must invalidate the device copy, or the two disagree
  # silently -- rule 2 of the contract.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m <- matrix(rnorm(64), 8, 8)
  t <- from_h(resident_scale(m, 2))
  expect_false(is.null(t$ptr))

  ag_set(t, matrix(1, 8, 8))
  expect_null(t$ptr)                  # residency dropped
  expect_null(t$ctx_gen)
  expect_equal(ag_data(t), matrix(1, 8, 8))
  expect_null(handle_of(t))
})

test_that("a stale handle is refused at install time", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m <- matrix(rnorm(64), 8, 8)
  h <- resident_scale(m, 2)
  get(".ag_residency_reset", envir = ns)()

  t <- ag_tensor(m, device = "gpu")
  expect_error(set_h(t, h), "generation")
})

test_that("a tensor whose buffer was freed says so rather than reading it", {
  # No host copy to fall back on, so the only safe answer is an error naming
  # the reset. Reading the freed pointer would return plausible garbage.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m <- matrix(rnorm(64), 8, 8)
  t <- from_h(resident_scale(m, 2))
  get(".ag_residency_reset", envir = ns)()
  expect_error(ag_data(t), "freed by a tape reset")
})

test_that("a resident tensor behaves like any other in ag_* operations", {
  # The equivalence that makes component 2 usable: whatever holds the value,
  # the arithmetic must agree.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  a <- matrix(rnorm(64), 8, 8)
  b <- matrix(rnorm(64), 8, 8)

  resident <- from_h(resident_scale(a, 1))   # same values, device-side
  host     <- ag_tensor(a, device = "gpu")   # same values, host-side
  B        <- ag_tensor(b, device = "gpu")

  expect_equal(ag_data(ag_matmul(resident, B)),
               ag_data(ag_matmul(host, B)), tolerance = 1e-3)
  expect_equal(ag_data(ag_add(resident, B)),
               ag_data(ag_add(host, B)), tolerance = 1e-4)
})

test_that("gradients through a resident tensor match finite differences", {
  # The check that matters most. Three silent bugs have been found in this area
  # and none of them showed up as a wrong forward pass, so equivalence of
  # values proves nothing about gradients -- differentiate against a reference.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  set.seed(42L)
  d <- 4L
  w_val <- matrix(rnorm(d * d) * 0.3, d, d)
  x_val <- matrix(rnorm(d * d) * 0.3, d, d)
  y     <- matrix(0.0, d, d)

  loss_of <- function(w_mat) {
    W <- ag_param(w_mat, device = "gpu")
    X <- ag_tensor(x_val, device = "gpu")
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), y) })
    list(loss = ag_data(l)[1L], W = W, l = l)
  }

  r <- loss_of(w_val)
  backward(r$l)
  # A resident gradient is a handle; the comparison below is against finite
  # differences computed in R, so it is materialised here.
  analytic <- get(".ag_as_matrix", envir = ns)(r$W$grad)

  eps <- 1e-3
  fd   <- matrix(0, d, d)
  for (i in seq_len(d)) for (j in seq_len(d)) {
    wp <- w_val; wp[i, j] <- wp[i, j] + eps
    wm <- w_val; wm[i, j] <- wm[i, j] - eps
    fd[i, j] <- (loss_of(wp)$loss - loss_of(wm)$loss) / (2 * eps)
  }
  expect_equal(analytic, fd, tolerance = 1e-2)
})
