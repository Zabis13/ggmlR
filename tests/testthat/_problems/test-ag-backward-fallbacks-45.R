# Extracted from test-ag-backward-fallbacks.R:45

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "ggmlR", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
skip_if_no_gpu_fb <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}
fallbacks <- function(reset = FALSE) ggmlR:::ag_backward_fallbacks(reset)
small_tape <- function() {
  set.seed(41)
  W <- ag_param(matrix(rnorm(6), 2, 3))
  x <- ag_tensor(matrix(rnorm(12), 3, 4))
  with_grad_tape({ loss <- ag_mse_loss(ag_matmul(W, x), matrix(0, 2, 4)) })
  list(W = W, x = x, loss = loss)
}

# test -------------------------------------------------------------------------
skip_if_no_gpu_fb()
ag_device("gpu")
on.exit(ag_device("cpu"), add = TRUE)
local_mocked_bindings(.ag_bwd_run_graph = function(loss, nodes) stop("boom"),
                        .package = "ggmlR")
fallbacks(reset = TRUE)
tp <- small_tape()
withr::local_options(ggmlR.ag_backward_strict = TRUE)
expect_error(backward(tp$loss), "strict mode, no fallback\\): boom")
withr::local_options(ggmlR.ag_backward_strict = FALSE)
tp <- small_tape()
backward(tp$loss)
