# ag_tape_memory(): does the tape report describe the tape that is actually there?
#
# The report exists to decide whether activation lifetime management is worth
# doing, so the property that matters is not "it prints something" but that the
# numbers move with the tape: more nodes, bigger operands and shared storage
# each have a right answer, and a report that got them wrong would argue for
# the wrong change.

test_that("an empty tape reports zero", {
  ag_device("cpu")
  tape <- get(".ag_tape", envir = asNamespace("ggmlR"))
  tape$nodes <- list()
  r <- ag_tape_memory(quiet = TRUE)
  expect_equal(r$nodes, 0L)
  expect_equal(r$bytes_total, 0)
})

test_that("nodes are counted and the tape holds memory", {
  ag_device("cpu")
  w <- ag_param(matrix(runif(64), 8, 8))
  x <- ag_tensor(matrix(runif(64), 8, 8))
  with_grad_tape({
    h    <- ag_relu(ag_matmul(w, x))
    loss <- ag_mse_loss(h, matrix(0, 8, 8))
  })
  backward(loss)

  r <- ag_tape_memory(quiet = TRUE)
  expect_gt(r$nodes, 0L)
  expect_gt(r$bytes_total, 0)
  # the parts must add up to the whole: the walk assigns every object to
  # exactly one category
  expect_equal(r$bytes_total,
               r$bytes_snapshots + r$bytes_fields + r$bytes_inputs)
})

test_that("a longer tape holds more than a shorter one", {
  ag_device("cpu")
  build <- function(n_ops) {
    w <- ag_param(matrix(runif(1024), 32, 32))
    x <- ag_tensor(matrix(runif(1024), 32, 32))
    with_grad_tape({
      h <- x
      for (i in seq_len(n_ops)) h <- ag_relu(ag_matmul(w, h))
      loss <- ag_mse_loss(h, matrix(0, 32, 32))
    })
    backward(loss)
    ag_tape_memory(quiet = TRUE)
  }
  short <- build(1L)
  long  <- build(4L)
  expect_gt(long$nodes, short$nodes)
  expect_gt(long$bytes_total, short$bytes_total)
})

test_that("bigger operands weigh more", {
  ag_device("cpu")
  build <- function(d) {
    w <- ag_param(matrix(runif(d * d), d, d))
    x <- ag_tensor(matrix(runif(d * d), d, d))
    with_grad_tape({ loss <- ag_mse_loss(ag_matmul(w, x), matrix(0, d, d)) })
    backward(loss)
    ag_tape_memory(quiet = TRUE)
  }
  small <- build(16L)
  big   <- build(64L)
  expect_equal(small$nodes, big$nodes)          # same shape of tape
  expect_gt(big$bytes_total, small$bytes_total) # different weight
})

test_that("shared storage is counted once, not once per reference", {
  # ag_matmul captures a_snap/b_snap in the closure AND records the same two
  # matrices as fields for the graph path. They are one object each, so a
  # correct report must not charge for them twice. object.size() alone does:
  # verified separately that a matrix under two names in a list reports double.
  ag_device("cpu")
  d <- 64L
  w <- ag_param(matrix(runif(d * d), d, d))
  x <- ag_tensor(matrix(runif(d * d), d, d))
  with_grad_tape({ loss <- ag_mse_loss(ag_matmul(w, x), matrix(0, d, d)) })
  backward(loss)

  r    <- ag_tape_memory(quiet = TRUE)
  one  <- as.numeric(object.size(matrix(0, d, d)))
  # The tape holds a handful of d x d matrices. Without deduplication the
  # matmul snapshots alone would be charged twice; the bound below is loose
  # enough to survive an extra intermediate but tight enough to fail if every
  # snapshot were counted twice.
  expect_lt(r$bytes_total, 8 * one)

  # The operands are real and must be charged as operands: w and x are the two
  # inputs of the matmul, and their matrices ARE the closure's a_snap/b_snap
  # (same address). Reporting them as activations would claim the tape can free
  # memory that the parameter keeps alive anyway.
  expect_gte(r$bytes_inputs, 2 * one * 0.9)
})

test_that("shared storage is charged to the operand, not to the activation", {
  # The distinction the report exists to make: how much would clearing the tape
  # actually free? A weight referenced by a closure is not freeable, so it must
  # not inflate the activation figure.
  ag_device("cpu")
  d <- 64L
  w <- ag_param(matrix(runif(d * d), d, d))
  x <- ag_tensor(matrix(runif(d * d), d, d))
  with_grad_tape({ loss <- ag_mse_loss(ag_matmul(w, x), matrix(0, d, d)) })
  backward(loss)

  r   <- ag_tape_memory(quiet = TRUE)
  one <- as.numeric(object.size(matrix(0, d, d)))
  # Both matmul operands land in the operand category, so the activations left
  # over are the genuinely tape-only intermediates -- strictly less than the
  # operands here, and in particular not the whole tape.
  expect_gt(r$bytes_inputs, 0)
  expect_lt(r$bytes_snapshots, r$bytes_total)
  expect_lt(r$bytes_snapshots, 3 * one)
})

test_that("the per-op breakdown names the ops on the tape", {
  ag_device("cpu")
  w <- ag_param(matrix(runif(64), 8, 8))
  x <- ag_tensor(matrix(runif(64), 8, 8))
  with_grad_tape({
    h    <- ag_relu(ag_matmul(w, x))
    loss <- ag_mse_loss(h, matrix(0, 8, 8))
  })
  backward(loss)

  r <- ag_tape_memory(quiet = TRUE)
  expect_true(is.data.frame(r$by_op))
  expect_true(all(c("op", "nodes", "mb") %in% names(r$by_op)))
  expect_true("matmul" %in% r$by_op$op)
  expect_equal(sum(r$by_op$nodes), r$nodes)
  # sorted most expensive first
  expect_false(is.unsorted(rev(r$by_op$mb)))
})

test_that("quiet suppresses the report but not the figures", {
  # Every other test here runs quiet, so a regression that started printing
  # again would only show up as noise in the test log -- which is exactly the
  # kind of thing nobody fixes. This asserts it instead.
  ag_device("cpu")
  w <- ag_param(matrix(runif(64), 8, 8))
  x <- ag_tensor(matrix(runif(64), 8, 8))
  with_grad_tape({ loss <- ag_mse_loss(ag_matmul(w, x), matrix(0, 8, 8)) })
  backward(loss)

  expect_silent(r <- ag_tape_memory(quiet = TRUE))
  expect_gt(r$bytes_total, 0)
  expect_output(ag_tape_memory(), "Gradient tape")
})

test_that("a device-resident tensor is skipped rather than downloaded", {
  # A diagnostic must not move data: a tensor whose value lives only on the
  # device has no host matrix, and reporting it would mean a download.
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(runif(256), 16, 16))
  x <- ag_tensor(matrix(runif(256), 16, 16))
  with_grad_tape({ loss <- ag_mse_loss(ag_matmul(w, x), matrix(0, 16, 16)) })
  backward(loss)

  expect_no_error(r <- ag_tape_memory(quiet = TRUE))
  expect_gt(r$nodes, 0L)
})
