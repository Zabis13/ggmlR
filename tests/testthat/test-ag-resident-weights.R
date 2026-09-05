# Stage 3.1 of full residency: a parameter lives on the device across steps.
#
# Before this, ag_param() kept its weight as an R matrix and every operation
# that touched it uploaded it again -- measured at 74-98% of a forward pass
# (inst/scripts/proto_ag_weight_cache.R). The weight now goes into the
# persistent pool once, which is the pool with_grad_tape() does NOT free.
#
# Two things have to hold, and they pull in opposite directions:
#   1. the weight stays resident -- across tape resets AND across optimizer
#      steps, since a read-modify-write through .ag_data_set() would silently
#      drop the pointer and put us back where we started;
#   2. the numbers are unchanged -- residency is a transport decision, and any
#      difference in the values means it went wrong.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns          <- asNamespace("ggmlR")
dev_state   <- get(".ag_device_state",         envir = ns)
t_scope     <- get(".ag_tensor_scope",         envir = ns)
ptr_live    <- get(".ag_ptr_is_live",          envir = ns)
ag_data     <- get(".ag_data",                 envir = ns)
tape_mem    <- get(".ag_tape_mem",             envir = ns)
write_res   <- get(".ag_data_write_resident",  envir = ns)
store_w     <- get(".ag_opt_store_weight",     envir = ns)
data_set    <- get(".ag_data_set",             envir = ns)

test_that("a gpu parameter is resident in the persistent pool", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(1:6 / 10, 2L, 3L))

  expect_false(is.null(w$ptr))
  expect_identical(t_scope(w), "persistent")
  expect_true(ptr_live(w))
  # The value is unchanged by the trip to the device.
  expect_equal(ag_data(w), matrix(1:6 / 10, 2L, 3L), tolerance = 1e-6)
})

test_that("a cpu parameter is untouched", {
  ag_device("cpu")
  w <- ag_param(matrix(1:4, 2L, 2L))
  expect_null(w$ptr)
  expect_identical(t_scope(w), "pass")   # the default, meaning "no pool"
  expect_equal(ag_data(w), matrix(1:4, 2L, 2L))
})

test_that("a resident weight survives a tape reset", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(rnorm(16), 4L, 4L))
  before <- ag_data(w)

  # This is the event that happens once per training step, and the one that
  # used to destroy every resident tensor.
  for (i in seq_len(5L)) with_grad_tape({ NULL })

  expect_true(ptr_live(w))
  expect_identical(t_scope(w), "persistent")
  expect_equal(ag_data(w), before, tolerance = 1e-6)
})

test_that("writing a resident weight keeps it resident", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w   <- ag_param(matrix(0, 3L, 3L))
  ptr <- w$ptr

  write_res(w, matrix(1:9, 3L, 3L))

  # Same buffer, new numbers: the pointer is what is being written to, so it
  # must not change.
  expect_identical(w$ptr, ptr)
  expect_true(ptr_live(w))
  expect_equal(ag_data(w), matrix(1:9, 3L, 3L), tolerance = 1e-6)
})

test_that("a resident write refuses a shape change", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(0, 2L, 2L))
  expect_error(write_res(w, matrix(0, 3L, 3L)), "cannot change a tensor's shape")
})

test_that("a resident write drops the materialisation cache", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(1, 2L, 2L))
  expect_equal(ag_data(w), matrix(1, 2L, 2L), tolerance = 1e-6)  # caches $data

  write_res(w, matrix(7, 2L, 2L))

  # A stale cache here would serve the pre-write value and nothing would say so.
  expect_equal(ag_data(w), matrix(7, 2L, 2L), tolerance = 1e-6)
})

test_that(".ag_opt_store_weight keeps residency, .ag_data_set drops it", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(0, 2L, 2L))
  store_w(w, matrix(2, 2L, 2L))
  expect_true(ptr_live(w))
  expect_equal(ag_data(w), matrix(2, 2L, 2L), tolerance = 1e-6)

  # The contrast matters: .ag_data_set is still the right call for a host-side
  # value (ag_save loading a checkpoint), and it must still drop the pointer.
  data_set(w, matrix(3, 2L, 2L))
  expect_null(w$ptr)
  expect_equal(ag_data(w), matrix(3, 2L, 2L))
})

test_that("a weight stays resident across optimizer steps", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w   <- ag_param(matrix(rnorm(9), 3L, 3L))
  opt <- optimizer_adam(list(w = w), lr = 0.01)

  for (i in seq_len(5L)) {
    w$grad <- matrix(rnorm(9), 3L, 3L)
    opt$step()
    # The read-modify-write inside step() is where residency used to be lost:
    # .ag_data_set() drops $ptr, so the weight would go host-side on step 1 and
    # be re-uploaded on every step after it.
    expect_true(ptr_live(w))
    expect_identical(t_scope(w), "persistent")
  }
})

test_that("resident and host optimizers agree numerically", {
  skip_if_no_gpu()

  w0 <- matrix(seq(0.1, 0.9, length.out = 9L), 3L, 3L)
  g  <- matrix(seq(-0.4, 0.4, length.out = 9L), 3L, 3L)

  ag_device("cpu")
  wc <- ag_param(w0)
  oc <- optimizer_adam(list(w = wc), lr = 0.05)
  for (i in seq_len(3L)) { wc$grad <- g; oc$step() }
  host <- ag_data(wc)

  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  wg <- ag_param(w0)
  og <- optimizer_adam(list(w = wg), lr = 0.05)
  for (i in seq_len(3L)) { wg$grad <- g; og$step() }
  dev <- ag_data(wg)

  # Residency is a transport decision; the arithmetic is still the host's at
  # this stage, so the two paths must agree to f32 round-trip precision.
  expect_equal(dev, host, tolerance = 1e-5)
})

test_that("a resident weight is rescued when the device is released", {
  skip_if_no_gpu()

  d <- matrix(seq(0.1, 1.6, length.out = 16L), 4L, 4L)
  ag_device("gpu")
  w <- ag_param(d)
  expect_null(w$data)          # the value exists only in the device buffer

  # Releasing the device frees the persistent pool. Without a rescue the weight
  # would be gone outright -- no host copy, no readable pointer -- which for a
  # trained model is the most expensive failure in this whole area.
  ag_device("cpu")

  expect_null(w$ptr)
  expect_equal(ag_data(w), d, tolerance = 1e-6)
})

test_that("a tape reset does not drag weights off the device", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(rnorm(9), 3L, 3L))
  with_grad_tape({ NULL })     # pass-scope reset

  # The rescue is per pool: a pass reset must leave persistent weights alone,
  # or the persistent pool would buy nothing.
  expect_false(is.null(w$ptr))
  expect_true(ptr_live(w))
  expect_null(w$data)
})

test_that("weights updated on the device are rescued with their new values", {
  skip_if_no_gpu()

  ag_device("gpu")
  w   <- ag_param(matrix(0.5, 2L, 2L))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  for (i in seq_len(3L)) { w$grad <- matrix(0.2, 2L, 2L); opt$step() }
  trained <- ag_data(w)
  expect_false(isTRUE(all.equal(trained, matrix(0.5, 2L, 2L))))

  ag_device("cpu")

  # What comes back must be the trained weight, not the value it was created
  # with: the in-place device updates wrote into the buffer being rescued.
  expect_equal(ag_data(w), trained, tolerance = 1e-6)
})

test_that("the persistent pool holds the weights and the pass pool does not grow", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w <- ag_param(matrix(rnorm(64), 8L, 8L))
  m <- tape_mem()
  expect_gt(m$p_buffer_bytes, 0)

  held <- m$p_buffer_bytes
  for (i in seq_len(3L)) with_grad_tape({ NULL })

  # Tape resets free the pass pool only; the weight's memory is untouched.
  expect_identical(tape_mem()$p_buffer_bytes, held)
})
