# Optimizers must update every parameter regardless of how the list is named.
#
# They used to loop over names(params): an unnamed list was never updated at
# all, and a duplicated name stepped its first parameter twice and the other
# never -- both silently. The reference here is the same run with a properly
# named list; every other spelling must land on the same weights, exactly.

skip_if_no_gpu_opt <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

# Three steps of a small linear fit; returns the final weights and the initial
# ones. `make_list(W, b)` decides how the optimizer is handed its parameters.
opt_run <- function(make_list, make_opt) {
  set.seed(7L)
  W <- ag_param(matrix(runif(6, -1, 1), 2, 3))
  b <- ag_param(matrix(runif(2, -1, 1), 2, 1))
  x <- ag_tensor(matrix(runif(12, -1, 1), 3, 4))
  y <- ag_tensor(matrix(runif(8, -1, 1), 2, 4))
  W0 <- as.matrix(W); b0 <- as.matrix(b)
  opt <- make_opt(make_list(W, b))
  for (k in 1:3) {
    loss <- NULL
    with_grad_tape({ loss <- ag_mse_loss(ag_add(ag_matmul(W, x), b), y) })
    opt$step(backward(loss))
    opt$zero_grad()
  }
  list(W = as.matrix(W), b = as.matrix(b), W0 = W0, b0 = b0)
}

opt_makers <- list(
  adam = function(p) optimizer_adam(p, lr = 0.05),
  sgd  = function(p) optimizer_sgd(p, lr = 0.05, momentum = 0.9))

check_param_lists <- function() {
  for (nm in names(opt_makers)) {
    mk  <- opt_makers[[nm]]
    ref <- opt_run(function(W, b) list(W = W, b = b), mk)
    expect_false(isTRUE(all.equal(ref$W, ref$W0)), info = nm)
    expect_false(isTRUE(all.equal(ref$b, ref$b0)), info = nm)

    unnamed <- opt_run(function(W, b) list(W, b), mk)
    expect_identical(unnamed$W, ref$W, info = paste(nm, "unnamed"))
    expect_identical(unnamed$b, ref$b, info = paste(nm, "unnamed"))

    dupname <- opt_run(function(W, b) list(p = W, p = b), mk)
    expect_identical(dupname$W, ref$W, info = paste(nm, "duplicate names"))
    expect_identical(dupname$b, ref$b, info = paste(nm, "duplicate names"))

    # Same tensor twice: dropped once at construction, so it steps once.
    twice <- NULL
    expect_warning(
      twice <- opt_run(function(W, b) list(W = W, b = b, W_again = W), mk),
      "listed more than once")
    expect_identical(twice$W, ref$W, info = paste(nm, "same tensor twice"))
    expect_identical(twice$b, ref$b, info = paste(nm, "same tensor twice"))
  }
}

test_that("optimizers update unnamed and duplicate-named params (CPU)", {
  ag_device("cpu")
  check_param_lists()
})

test_that("optimizers update unnamed and duplicate-named params (GPU)", {
  skip_if_no_gpu_opt()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  check_param_lists()
})
