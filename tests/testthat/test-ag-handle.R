# Component 1 of the resident contract: .ag_run_op accepts and returns handles.
#
# The property being established is equivalence, not speed: a chain computed
# through handles must produce exactly what the same chain produces through R
# matrices. Everything else in the redesign rests on that, and this is the area
# where three silent bugs have already been found (doubled gradients, device
# leak, batch_norm) -- all of which passed a "forward looks right" check.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns  <- asNamespace("ggmlR")
run_op    <- get(".ag_run_op",       envir = ns)
mk_handle <- get(".ag_handle",       envir = ns)
is_handle <- get(".ag_is_handle",    envir = ns)
h_live    <- get(".ag_handle_live",  envir = ns)
h_to_r    <- get(".ag_handle_to_r",  envir = ns)
as_mat    <- get(".ag_as_matrix",    envir = ns)

test_that("a matrix operand still works and still returns a matrix", {
  # The bridge must not change existing behaviour: every caller not yet
  # converted goes through this path.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  a <- matrix(rnorm(64), 8, 8)
  b <- matrix(rnorm(64), 8, 8)
  out <- run_op(function(ctx, p) ggml_add(ctx, p[[1L]], p[[2L]]),
                inputs = list(a, b), out_shape = c(8L, 8L))
  expect_true(is.matrix(out))
  expect_equal(out, a + b, tolerance = 1e-4)
})

test_that("resident = TRUE returns a live handle, not a matrix", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  a <- matrix(rnorm(64), 8, 8)
  b <- matrix(rnorm(64), 8, 8)
  h <- run_op(function(ctx, p) ggml_add(ctx, p[[1L]], p[[2L]]),
              inputs = list(a, b), out_shape = c(8L, 8L), resident = TRUE)
  expect_true(is_handle(h))
  expect_true(h_live(h))
  expect_equal(h$shape, c(8L, 8L))
  expect_equal(h_to_r(h), a + b, tolerance = 1e-4)
})

test_that("a handle can be an operand without being re-uploaded", {
  # The point of the type: the second op consumes the first op's result while
  # it is still on the device.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  a <- matrix(rnorm(64), 8, 8)
  b <- matrix(rnorm(64), 8, 8)
  c_ <- matrix(rnorm(64), 8, 8)

  h1 <- run_op(function(ctx, p) ggml_add(ctx, p[[1L]], p[[2L]]),
               inputs = list(a, b), out_shape = c(8L, 8L), resident = TRUE)
  out <- run_op(function(ctx, p) ggml_add(ctx, p[[1L]], p[[2L]]),
                inputs = list(h1, c_), out_shape = c(8L, 8L))
  expect_equal(out, a + b + c_, tolerance = 1e-4)
})

test_that("a resident chain equals the same chain through R matrices", {
  # Equivalence on a chain long enough that an off-by-one in the plumbing would
  # show: the whole redesign is only sound if these two agree.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  d <- 16L
  x <- matrix(rnorm(d * d), d, d)
  k <- matrix(rnorm(d * d), d, d)

  add_op <- function(ctx, p) ggml_add(ctx, p[[1L]], p[[2L]])

  # resident: one download at the end
  h <- run_op(add_op, list(x, k), c(d, d), resident = TRUE)
  for (i in 1:5) h <- run_op(add_op, list(h, k), c(d, d), resident = TRUE)
  resident_out <- h_to_r(h)

  # per-op: a download after every step, as today
  m <- run_op(add_op, list(x, k), c(d, d))
  for (i in 1:5) m <- run_op(add_op, list(m, k), c(d, d))

  expect_equal(resident_out, m, tolerance = 1e-4)
  expect_equal(resident_out, x + 6 * k, tolerance = 1e-3)
})

test_that("a stale handle is refused rather than read", {
  # Rule 5 of the data contract: a pointer is only valid with its generation.
  # Reading a freed buffer would return plausible garbage, which is the failure
  # mode the check exists to prevent.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  a <- matrix(rnorm(64), 8, 8)
  h <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 2),
              inputs = list(a), out_shape = c(8L, 8L), resident = TRUE)
  expect_true(h_live(h))

  # A tape reset frees the contexts and bumps the generation.
  get(".ag_residency_reset", envir = ns)()
  expect_false(h_live(h))
  expect_error(h_to_r(h), "freed by a tape reset")
  expect_error(
    run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 2),
           inputs = list(h), out_shape = c(8L, 8L)),
    "device handle from the pass pool")
})

test_that("a handle is not usable as an arithmetic operand", {
  # Rule 3: no Ops methods, ever. If `h * 2` silently worked, a stale-value bug
  # would become a wrong answer instead of an error.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  a <- matrix(rnorm(64), 8, 8)
  h <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 2),
              inputs = list(a), out_shape = c(8L, 8L), resident = TRUE)
  expect_error(h * 2)
  expect_error(h + h)
})

test_that(".ag_as_matrix bridges both forms", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  m <- matrix(rnorm(64), 8, 8)
  expect_identical(as_mat(m), m)          # matrix passes through untouched
  h <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 1),
              inputs = list(m), out_shape = c(8L, 8L), resident = TRUE)
  expect_equal(as_mat(h), m, tolerance = 1e-4)
})

test_that("shape accessors work on both forms without downloading", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  nr <- get(".ag_nrow", envir = ns); nc <- get(".ag_ncol", envir = ns)
  dm <- get(".ag_dim",  envir = ns)
  m <- matrix(rnorm(12), 3, 4)
  h <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 1),
              inputs = list(m), out_shape = c(3L, 4L), resident = TRUE)
  expect_equal(nr(m), 3L); expect_equal(nr(h), 3L)
  expect_equal(nc(m), 4L); expect_equal(nc(h), 4L)
  expect_equal(dm(h), c(3L, 4L))
})

test_that("existing ag_* operations are unaffected", {
  # Component 1 changes .ag_run_op but converts no helper, so every ag_* result
  # must be bit-identical in behaviour to before. This is the regression guard
  # for the rest of the redesign.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  A <- ag_tensor(matrix(rnorm(64), 8, 8))
  B <- ag_tensor(matrix(rnorm(64), 8, 8))
  expect_equal(.ag_data_test <- as.matrix(get(".ag_data", envir = ns)(ag_matmul(A, B))),
               get(".ag_data", envir = ns)(A) %*% get(".ag_data", envir = ns)(B),
               tolerance = 1e-3)
})
