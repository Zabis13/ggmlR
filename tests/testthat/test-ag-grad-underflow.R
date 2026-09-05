# Do gradients underflow in f16, i.e. does the ag_* path need loss scaling?
#
# Today it does not, and these tests are a tripwire for the day that changes.
#
# Mixed-precision training needs a GradScaler when the BACKWARD pass runs in
# f16: gradients below the f16 denormal minimum (~6e-8) flush to zero, the
# update never happens, and nothing about it is visible except a model that
# does not learn. That is the PyTorch situation. It is not this one.
#
# On the ag_* path backward() is a chain of R closures computing in double.
# ag_dtype() governs how a forward tensor is uploaded into a ggml buffer; it
# does not reach $grad, which stays double whatever the dtype is. So there is
# no underflow to scale away, and a scaler would be code with no caller.
#
# If backward ever moves into f16 kernels -- a graph backward that keeps the
# whole pass on the GPU in f16, say -- these tests fail, and the loss-scaling
# item in TODO.md section 3 comes back.

# Gradient small enough that f16 could not hold it: 1e-11 is well under the
# f16 denormal minimum of ~6e-8.
tiny_grad_case <- function() {
  W <- ag_param(matrix(c(1e-5, 1e-6, 1e-7, 1e-8), 2, 2))
  x <- ag_tensor(matrix(c(1e-3, 1e-3), 2, 1))
  with_grad_tape({
    loss <- ag_mse_loss(ag_matmul(W, x), matrix(0.0, 2, 1))
  })
  backward(loss)
  W$grad
}

test_that("gradients below the f16 denormal minimum survive on CPU", {
  local_cpu_device()
  g <- tiny_grad_case()

  expect_false(any(g == 0))
  expect_true(all(abs(g) < 6e-8))          # genuinely in the danger zone
  expect_equal(storage.mode(g), "double")  # and held in double, not f16
})

test_that("$grad stays double when ag_dtype() asks for f16", {
  local_cpu_device()
  prev <- ag_dtype("f16")
  withr::defer(ag_dtype(prev))

  g <- tiny_grad_case()
  expect_false(any(g == 0))
  expect_equal(storage.mode(g), "double")
})

test_that("f16 on the GPU does not flush small gradients to zero", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  local_cpu_device()
  ag_device("gpu")
  prev <- ag_dtype("f16")
  withr::defer(ag_dtype(prev))

  g <- tiny_grad_case()
  expect_false(any(g == 0))
  # loose tolerance: the operands do round-trip through f16, so the value is
  # approximate -- the point is that it is not zero.
  expect_true(all(abs(g) > 1e-13))
})

# ============================================================================
# Optimizer states and the master weight copy
# ============================================================================

# The other half of mixed-precision training: even with f16 gradients, the
# optimizer must keep its moments and a master copy of the weights in higher
# precision, or small updates vanish when added to a large weight. On the ag_*
# path that holds by construction -- m, v and the parameter values are all R
# doubles, and ag_dtype() does not reach them -- so there is nothing to build.
# These tests fail if any of the three ever moves into f16.

test_that("Adam moments stay double under ag_dtype('f16')", {
  local_cpu_device()
  prev <- ag_dtype("f16")
  withr::defer(ag_dtype(prev))

  W   <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  opt <- optimizer_adam(list(W = W), lr = 1e-3)

  expect_equal(storage.mode(opt$m$W), "double")
  expect_equal(storage.mode(opt$v$W), "double")
})

test_that("small updates accumulate into a large weight (master copy is f32+)", {
  # 100 steps of 1e-7 on a weight of 1.0. In f16 the spacing at 1.0 is ~1e-3,
  # so every one of those updates would be lost and the weight would still read
  # exactly 1. Getting 0.99999 back is the master-copy property.
  local_cpu_device()
  prev <- ag_dtype("f16")
  withr::defer(ag_dtype(prev))

  W   <- ag_param(matrix(1.0, 1, 1))
  opt <- optimizer_sgd(list(W = W), lr = 1e-7)
  for (i in 1:100) {
    W$grad <- matrix(1.0, 1, 1)
    opt$step()
  }

  expect_equal(as.numeric(ggmlR:::.ag_data(W)), 1 - 100e-7, tolerance = 1e-12)
})
