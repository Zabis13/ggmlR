# Stage 3.3: the Adam step runs on the device.
#
# Weights, moments and the gradient all live in device buffers, and the step
# updates them in place -- no numbers cross to the host and back. The baseline
# for this (inst/scripts/measure_ag_step_transfers.R) was 10 crossings per
# step, 4 of them the optimizer's own read-modify-write on the weights.
#
# The property that matters is equality, not speed: a device step that computes
# something subtly different from Adam would still train, just worse, and
# nothing would say so. Every test here is either an equality against the host
# path or an invariant that a broken in-place update would violate.
#
# ⚠️ DIAGNOSING A FAILURE of "stays resident across steps". Two different causes
# produce it, and $ptr tells them apart:
#   * pointer LIVE, numbers wrong  -> the ggml_cpy into the destination is the
#     problem (wrong context for the copy node, or the copy never ran).
#   * pointer DEAD (.ag_ptr_is_live FALSE) -> the copy node was built in the
#     pass context and tied to the pass generation, so the tape reset at the
#     next with_grad_tape() invalidated it. The node must come from `ctx`, the
#     residency context the operands are in.
# Stage 3.1 already proved that "buffer stays, host cache is dropped" works
# without a graph, so a failure here is about the graph path specifically.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns        <- asNamespace("ggmlR")
dev_state <- get(".ag_device_state",  envir = ns)
ag_data   <- get(".ag_data",          envir = ns)
t_scope   <- get(".ag_tensor_scope",  envir = ns)
ptr_live  <- get(".ag_ptr_is_live",   envir = ns)
is_handle <- get(".ag_is_handle",     envir = ns)
h_live    <- get(".ag_handle_live",   envir = ns)
h_scope   <- get(".ag_handle_scope",  envir = ns)
as_mat    <- get(".ag_as_matrix",     envir = ns)
tape_mem  <- get(".ag_tape_mem",      envir = ns)
run_op    <- get(".ag_run_op",        envir = ns)
mk_handle <- get(".ag_handle",        envir = ns)
r_to_gpu  <- get(".ag_r_to_gpu",      envir = ns)

test_that("run_op(out=) writes in place and returns the same handle", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  dst <- mk_handle(r_to_gpu(matrix(0, 4L, 4L), scope = "persistent"),
                   c(4L, 4L), scope = "persistent")
  ptr <- dst$ptr
  a   <- matrix(1:16 / 16, 4L, 4L)

  got <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 2),
                inputs = list(a), out_shape = c(4L, 4L),
                scope = "persistent", out = dst)

  # Same buffer, new contents: an allocation here would grow a pool that cannot
  # be reset while the weights in it are live.
  expect_identical(got$ptr, ptr)
  expect_equal(as_mat(got), a * 2, tolerance = 1e-5)
})

test_that("run_op(out=) refuses a shape mismatch", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  dst <- mk_handle(r_to_gpu(matrix(0, 2L, 2L), scope = "persistent"),
                   c(2L, 2L), scope = "persistent")
  expect_error(
    run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 2),
           inputs = list(matrix(1, 3L, 3L)), out_shape = c(3L, 3L),
           scope = "persistent", out = dst),
    "shape mismatch")
})

test_that("adam moments are resident when the parameters are", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w   <- ag_param(matrix(rnorm(9), 3L, 3L))
  opt <- optimizer_adam(list(w = w), lr = 0.01)

  expect_true(opt$resident)
  expect_true(is_handle(opt$m$w))
  expect_true(is_handle(opt$v$w))
  expect_identical(h_scope(opt$m$w), "persistent")
  expect_equal(as_mat(opt$m$w), matrix(0, 3L, 3L), tolerance = 1e-6)
})

test_that("adam moments stay host-side on the cpu", {
  ag_device("cpu")
  w   <- ag_param(matrix(rnorm(4), 2L, 2L))
  opt <- optimizer_adam(list(w = w), lr = 0.01)

  expect_false(opt$resident)
  expect_true(is.matrix(opt$m$w))
})

test_that("the device step matches the host step numerically", {
  skip_if_no_gpu()

  w0 <- matrix(seq(-0.4, 0.4, length.out = 12L), 3L, 4L)
  gs <- lapply(1:4, function(i) matrix(sin(seq_len(12) + i) * 0.1, 3L, 4L))

  ag_device("cpu")
  wc <- ag_param(w0)
  oc <- optimizer_adam(list(w = wc), lr = 0.05)
  for (g in gs) { wc$grad <- g; oc$step() }
  host <- ag_data(wc)

  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  wg <- ag_param(w0)
  og <- optimizer_adam(list(w = wg), lr = 0.05)
  for (g in gs) { wg$grad <- g; og$step() }
  dev <- ag_data(wg)

  # Four steps of bias-corrected Adam: the correction, the square root and the
  # epsilon all have to be in the right places for these to agree.
  expect_equal(dev, host, tolerance = 1e-4)
})

test_that("the weight stays resident across device steps", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w   <- ag_param(matrix(rnorm(16), 4L, 4L))
  opt <- optimizer_adam(list(w = w), lr = 0.01)
  ptr <- w$ptr

  for (i in seq_len(6L)) {
    w$grad <- matrix(rnorm(16), 4L, 4L)
    opt$step()
    # See the diagnosis note at the top of this file if this fails.
    expect_true(ptr_live(w))
    expect_identical(t_scope(w), "persistent")
    expect_identical(w$ptr, ptr)          # in place: same buffer every step
    expect_true(h_live(opt$m$w))
  }
  expect_false(any(is.na(ag_data(w))))
})

test_that("device steps do not grow the persistent pool", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w   <- ag_param(matrix(rnorm(64), 8L, 8L))
  opt <- optimizer_adam(list(w = w), lr = 0.01)

  w$grad <- matrix(rnorm(64), 8L, 8L)
  opt$step()
  held <- tape_mem()$p_buffer_bytes

  for (i in seq_len(10L)) {
    w$grad <- matrix(rnorm(64), 8L, 8L)
    opt$step()
  }

  # The persistent pool has no collector -- it cannot be reset while the weights
  # in it are live -- so any per-step allocation here is a leak that only shows
  # up on long runs.
  expect_identical(tape_mem()$p_buffer_bytes, held)
})

test_that("the weight survives tape resets between device steps", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  w   <- ag_param(matrix(rnorm(9), 3L, 3L))
  opt <- optimizer_adam(list(w = w), lr = 0.01)

  for (i in seq_len(4L)) {
    with_grad_tape({ NULL })              # the per-step pass-pool reset
    w$grad <- matrix(rnorm(9), 3L, 3L)
    opt$step()
    expect_true(ptr_live(w))
    expect_true(h_live(opt$m$w))
  }
})

test_that("a real training loop converges on the device", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  set.seed(3L)

  W <- ag_param(matrix(rnorm(8 * 4) * 0.1, 4L, 8L))
  x <- ag_tensor(matrix(rnorm(8 * 16), 8L, 16L))
  y <- ag_tensor(matrix(rnorm(4 * 16) * 0.1, 4L, 16L))
  opt <- optimizer_adam(list(W = W), lr = 0.05)

  first <- NULL; last <- NULL
  for (i in seq_len(25L)) {
    with_grad_tape({ loss <- ag_mse_loss(ag_matmul(W, x), y) })
    backward(loss)
    opt$step()
    opt$zero_grad()
    l <- as.numeric(ag_data(loss))
    if (is.null(first)) first <- l
    last <- l
  }

  # The end-to-end check: an in-place update that wrote to the wrong buffer, or
  # a moment that reset every step, would leave the loss flat rather than error.
  expect_true(is.finite(last))
  expect_lt(last, first)
})

test_that("gradient accumulation still averages correctly on the device", {
  skip_if_no_gpu()

  g1 <- matrix(0.2, 2L, 2L); g2 <- matrix(0.6, 2L, 2L)
  w0 <- matrix(0.5, 2L, 2L)

  ag_device("cpu")
  wc <- ag_param(w0)
  oc <- optimizer_adam(list(w = wc), lr = 0.1, accumulate_steps = 2L,
                       average = TRUE)
  wc$grad <- g1; oc$step(); wc$grad <- g1 + g2; oc$step()
  host <- ag_data(wc)

  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  wg <- ag_param(w0)
  og <- optimizer_adam(list(w = wg), lr = 0.1, accumulate_steps = 2L,
                       average = TRUE)
  wg$grad <- g1; og$step(); wg$grad <- g1 + g2; og$step()

  # The 1/n divide moved onto the device too; doing it on the host would have
  # meant downloading the gradient this stage keeps resident.
  expect_equal(ag_data(wg), host, tolerance = 1e-4)
})

test_that("an optimizer outliving the device falls back to the host", {
  skip_if_no_gpu()

  ag_device("gpu")
  w   <- ag_param(matrix(rnorm(4), 2L, 2L))
  opt <- optimizer_adam(list(w = w), lr = 0.01)
  w$grad <- matrix(0.1, 2L, 2L)
  opt$step()
  expect_true(opt$resident)

  # Releasing the device frees the persistent pool; the moments the optimizer
  # holds are now pointers into freed memory. The step must notice rather than
  # read them.
  ag_device("cpu")
  w$grad <- matrix(0.1, 2L, 2L)
  expect_no_error(opt$step())
  expect_false(any(is.na(ag_data(w))))
})
