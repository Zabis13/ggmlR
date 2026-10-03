# Extracted from test-ag-defer.R:213

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "ggmlR", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}
ns          <- asNamespace("ggmlR")
defer_on    <- get("ag_defer_forward", envir = ns)
defer_len   <- get(".ag_defer_len",    envir = ns)
ag_data     <- get(".ag_data",         envir = ns)
as_mat      <- get(".ag_as_matrix",    envir = ns)
is_handle   <- get(".ag_is_handle",    envir = ns)
h_pending   <- get(".ag_handle_pending", envir = ns)
with_defer <- function(on, expr) {
  old <- defer_on(on)
  on.exit(defer_on(old), add = TRUE)
  force(expr)
}
run_chain <- function(seed = 11L, d = 5L, b = 3L, depth = 3L) {
  set.seed(seed)
  Wt <- lapply(seq_len(depth), function(i) matrix(rnorm(d * d) * 0.3, d, d))
  X  <- matrix(rnorm(d * b) * 0.3, d, b)
  list(Wt = Wt, X = X, d = d, b = b)
}
forward_only <- function(cfg) {
  h <- ag_tensor(cfg$X)
  for (W in cfg$Wt) h <- ag_relu(ag_matmul(ag_tensor(W), h))
  ag_data(h)
}
train_grads <- function(cfg) {
  Wp <- lapply(cfg$Wt, ag_param)
  X  <- ag_tensor(cfg$X)
  Y  <- matrix(0.0, cfg$d, cfg$b)
  with_grad_tape({
    h <- X
    for (W in Wp) h <- ag_relu(ag_matmul(W, h))
    loss <- ag_mse_loss(h, Y)
  })
  backward(loss)
  list(loss = ag_data(loss),
       grads = lapply(Wp, function(p) as_mat(p$grad)))
}

# test -------------------------------------------------------------------------
old <- defer_on(NA)
expect_false(isTRUE(old))
