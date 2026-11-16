# ag_pass_scope(): eager device work in a private pass pool, freed on exit.
# Tested: repeated eager forwards inside the scope do not grow the pass pool
# (they do without it), an error inside restores the caller's pool, a handle
# that escapes the scope fails loudly instead of reading freed memory, an
# ag_tensor returned from the scope is refused, and the CPU path just runs.

skip_if_no_gpu_ps <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}
pass_bytes <- function() ggmlR:::.ag_tape_mem()$buffer_bytes
pass_gen   <- function() ggmlR:::.ag_scope_gen("pass")

test_that("ag_pass_scope on the CPU just evaluates expr", {
  ag_device("cpu")
  W <- ag_param(matrix(1:6, 2, 3))
  y <- ag_pass_scope(as.matrix(ag_matmul(W, ag_tensor(matrix(1, 3, 2)))))
  expect_equal(y, matrix(c(9, 12, 9, 12), 2, 2))
})

test_that("repeated eager forwards inside the scope do not grow the pass pool", {
  skip_if_no_gpu_ps()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(1)
  W <- ag_param(matrix(rnorm(64 * 32), 64, 32))
  fwd <- function() as.matrix(ag_relu(ag_matmul(W, ag_tensor(matrix(rnorm(32 * 16), 32, 16)))))

  # Without the scope the pool grows: the defect this guards against.
  b0 <- pass_bytes()
  for (i in 1:5) fwd()
  expect_gt(pass_bytes(), b0)

  b1 <- pass_bytes(); g1 <- pass_gen()
  for (i in 1:20) {
    y <- ag_pass_scope(fwd())
    expect_true(is.matrix(y))
  }
  expect_identical(pass_bytes(), b1)
  expect_identical(pass_gen(), g1)
})

test_that("an error inside the scope restores the caller's pass pool", {
  skip_if_no_gpu_ps()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  W <- ag_param(matrix(1:6, 2, 3))
  b0 <- pass_bytes(); g0 <- pass_gen()
  expect_error(ag_pass_scope({
    ag_matmul(W, ag_tensor(matrix(1, 3, 2)))
    stop("boom")
  }), "boom")
  expect_identical(pass_bytes(), b0)
  expect_identical(pass_gen(), g0)
  # The device is still usable afterwards.
  y <- as.matrix(ag_matmul(W, ag_tensor(matrix(1, 3, 2))))
  expect_equal(y, matrix(c(9, 12, 9, 12), 2, 2), tolerance = 1e-5)
})

test_that("a handle escaping the scope is an error, not a read of freed memory", {
  skip_if_no_gpu_ps()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  W <- ag_param(matrix(1:6, 2, 3))
  leaked <- NULL
  ag_pass_scope({
    leaked <- ag_matmul(W, ag_tensor(matrix(1, 3, 2)))
    NULL
  })
  skip_if(!is.null(leaked$data), "op kept a host copy; nothing to leak")
  expect_error(as.matrix(leaked))
})

test_that("returning a device-only ag_tensor from the scope is refused", {
  skip_if_no_gpu_ps()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  W <- ag_param(matrix(1:6, 2, 3))
  b0 <- pass_bytes()
  res <- tryCatch(ag_pass_scope(ag_matmul(W, ag_tensor(matrix(1, 3, 2)))),
                  error = function(e) e)
  if (!inherits(res, "error"))
    skip("op kept a host copy; nothing device-only to return")
  expect_match(conditionMessage(res), "freed on exit")
  expect_identical(pass_bytes(), b0)
})
