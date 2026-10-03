# ag_graph_mode() / ag_local_mode() and the launch counter.
#
# Graph mode only changes WHEN ops run (one fused graph at the next sync point
# instead of one launch per op), never WHAT they compute -- so the tests compare
# results with it on and off, check that it actually cuts launches, and check
# the sync points that guard against running a queued op on changed data.

skip_if_no_gpu_gm <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}
launches <- function() ggmlR:::ag_launch_count()[["total"]]
as_m <- function(t) ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(t))

test_that("ag_graph_mode() gets, sets and validates", {
  old <- ag_graph_mode(FALSE)
  on.exit(ag_graph_mode(old), add = TRUE)
  expect_false(ag_graph_mode())
  expect_false(ag_graph_mode(TRUE))      # returns the previous value
  expect_true(ag_graph_mode())
  expect_error(ag_graph_mode(NA), "TRUE or FALSE")
  expect_error(ag_graph_mode("yes"), "TRUE or FALSE")
  expect_true(ag_graph_mode())           # a bad value changes nothing
  expect_error(ag_graph_mode(TRUE, check = 1), "`check`")
})

test_that("ag_local_mode() restores on normal exit and on error", {
  old_g <- ag_graph_mode(FALSE); old_p <- ag_matmul_precision("default")
  on.exit({ ag_graph_mode(old_g); ag_matmul_precision(old_p) }, add = TRUE)

  f <- function() {
    ag_local_mode(graph = TRUE, matmul_precision = "f32")
    list(ag_graph_mode(), ag_matmul_precision())
  }
  expect_identical(f(), list(TRUE, "f32"))
  expect_false(ag_graph_mode()); expect_identical(ag_matmul_precision(), "default")

  g <- function() { ag_local_mode(graph = TRUE, matmul_precision = "f32"); stop("boom") }
  expect_error(g(), "boom")
  expect_false(ag_graph_mode()); expect_identical(ag_matmul_precision(), "default")

  h <- function() ag_local_mode(graph = TRUE, matmul_precision = "f16")
  expect_error(h(), "should be one of")
  expect_false(ag_graph_mode())          # validated before anything changed

  only <- function() { ag_local_mode(matmul_precision = "f32"); ag_graph_mode() }
  expect_false(only())                   # NULL leaves the other setting alone
})

test_that("ag_launch_count() counts and resets", {
  skip_if_no_gpu_gm()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(FALSE); on.exit(ag_graph_mode(old), add = TRUE)
  ggmlR:::ag_launch_count(reset = TRUE)
  x <- ag_tensor(matrix(runif(12), 3, 4))
  y <- as_m(ag_relu(ag_add(x, x)))
  v <- ggmlR:::ag_launch_count(reset = TRUE)
  expect_true("total" %in% names(v))
  expect_gte(v[["total"]], 2)            # add, relu: one launch each
  expect_identical(ggmlR:::ag_launch_count()[["total"]], 0)
})

# The PPO-shaped loss from test-ag-backward-graph.R, small.
gm_ppo <- function(seed = 31L) {
  set.seed(seed)
  B <- 64L; nobs <- 9L; nh <- 16L; na <- 9L
  W1 <- matrix(rnorm(nh * nobs, sd = 0.3), nh, nobs); b1 <- matrix(0, nh, 1)
  Wp <- matrix(rnorm(na * nh, sd = 0.3), na, nh);     bp <- matrix(0, na, 1)
  obs <- matrix(runif(nobs * B, -1, 1), nobs, B)
  legal <- matrix(rbinom(na * B, 1, 0.7), na, B); legal[1, ] <- 1
  act <- apply(legal, 2, function(l) which(l == 1)[1L]) - 1L
  old <- matrix(log(runif(B, 0.1, 0.9)), 1); adv <- matrix(rnorm(B), 1)
  function() {
    ps <- lapply(list(W1, b1, Wp, bp), ag_param)
    with_grad_tape({
      h    <- ag_relu(ag_add(ag_matmul(ps[[1]], ag_tensor(obs)), ps[[2]]))
      lg   <- ag_add(ag_matmul(ps[[3]], h), ps[[4]])
      logp <- ag_log_softmax(ag_add(lg, ag_tensor(-1e4 * (1 - legal))))
      r    <- ag_exp(ag_sub(ag_gather(logp, act), ag_tensor(old)))
      s1   <- ag_mul(r, ag_tensor(adv))
      s2   <- ag_mul(ag_clamp(r, 0.8, 1.2), ag_tensor(adv))
      loss <- ag_scale(ag_mean(ag_sub(s1, ag_relu(ag_sub(s1, s2)))), -1)
    })
    g <- backward(loss)
    list(loss = as.numeric(as_m(loss)),
         grads = lapply(ps, function(p) as_m(get0(as.character(p$id), envir = g))))
  }
}

test_that("graph mode gives the same loss and gradients, with fewer launches", {
  skip_if_no_gpu_gm()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_matmul_precision("f32"); on.exit(ag_matmul_precision(old), add = TRUE)
  run <- gm_ppo()
  res <- list()
  for (mode in c(FALSE, TRUE)) {
    prev <- ag_graph_mode(mode)
    ggmlR:::ag_launch_count(reset = TRUE)
    res[[as.character(mode)]] <- c(run(), launches = launches())
    ag_graph_mode(prev)
  }
  off <- res[["FALSE"]]; on <- res[["TRUE"]]
  expect_equal(on$loss, off$loss, tolerance = 1e-6)
  for (i in seq_along(off$grads))
    expect_equal(on$grads[[i]], off$grads[[i]], tolerance = 1e-6, info = paste("param", i))
  expect_lt(on$launches, off$launches)
})

test_that("writing a parameter runs the queue first (graph mode)", {
  skip_if_no_gpu_gm()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(TRUE); on.exit(ag_graph_mode(old), add = TRUE)
  set.seed(32)
  W0 <- matrix(rnorm(12), 4, 3); X <- matrix(rnorm(15), 3, 5)
  lin <- ag_linear(3L, 4L, init = W0)
  y <- ag_matmul(lin$W, ag_tensor(X))             # queued, not read yet
  sd <- ag_state_dict(list(params = function() list(W = lin$W, b = lin$b)))
  sd$parameters$W <- sd$parameters$W * 0 + 7      # overwrite W before reading y
  ag_load_state_dict(list(params = function() list(W = lin$W, b = lin$b)), sd)
  expect_equal(as_m(y), W0 %*% X, tolerance = 1e-5)  # y was computed with the OLD W
})

test_that("check mode runs each queued op at once and labels the queue", {
  skip_if_no_gpu_gm()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(TRUE); on.exit(ag_graph_mode(old, check = FALSE), add = TRUE)
  x <- ag_tensor(matrix(runif(12), 3, 4))

  ggmlR:::ag_launch_count(reset = TRUE)
  y <- ag_relu(ag_add(x, x))
  expect_identical(ggmlR:::.ag_defer$labels, c("ag_add", "ag_relu"))   # queued, labelled
  invisible(as_m(y))
  fused <- launches()

  ag_graph_mode(TRUE, check = TRUE)
  ggmlR:::ag_launch_count(reset = TRUE)
  y <- ag_relu(ag_add(x, x))
  expect_length(ggmlR:::.ag_defer$labels, 0L)                          # ran already
  invisible(as_m(y))
  expect_gt(launches(), fused)
})
