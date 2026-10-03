# Extracted from test-ag-grad-underflow.R:58

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "ggmlR", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
tiny_grad_case <- function() {
  W <- ag_param(matrix(c(1e-5, 1e-6, 1e-7, 1e-8), 2, 2))
  x <- ag_tensor(matrix(c(1e-3, 1e-3), 2, 1))
  with_grad_tape({
    loss <- ag_mse_loss(ag_matmul(W, x), matrix(0.0, 2, 1))
  })
  backward(loss)
  W$grad
}

# test -------------------------------------------------------------------------
skip_if_no_gpu()
local_cpu_device()
ag_device("gpu")
prev <- ag_dtype("f16")
withr::defer(ag_dtype(prev))
g <- tiny_grad_case()
expect_false(any(g == 0))
