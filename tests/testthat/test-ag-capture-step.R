# ag_capture_step(): forward + graph backward recorded as one graph, the Adam
# step as a second, replayed together. Tested: the trajectory equals the
# uncaptured loop (GPU, graph mode, fused Adam) and the CPU, including a
# learning-rate change mid-run; a replay is two launches; t and opt$state()
# advance; shapes are fixed; host data in fn is refused; after a device switch
# the dead optimizer state is an error, not a replay of a freed graph.

skip_if_no_gpu_cs <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}
launches <- function() ggmlR:::ag_launch_count()[["total"]]

cs_data <- function() {
  set.seed(21L)
  list(w1 = matrix(rnorm(8 * 16) * 0.3, 16L, 8L), w2 = matrix(rnorm(16 * 4) * 0.3, 4L, 16L),
       x = matrix(rnorm(8 * 32), 8L, 32L), y = matrix(rnorm(4 * 32) * 0.1, 4L, 32L))
}

cs_model <- function(d) {
  ps <- list(W1 = ag_param(d$w1), b1 = ag_param(matrix(0, 16L, 1L)),
             W2 = ag_param(d$w2), b2 = ag_param(matrix(0, 4L, 1L)))
  fn <- function(x, y) {
    h <- ag_relu(ag_add(ag_matmul(ps$W1, x), ps$b1))
    ag_mse_loss(ag_add(ag_matmul(ps$W2, h), ps$b2), y)
  }
  list(ps = ps, fn = fn)
}

# Betas exact in float, so the CPU comparison is element-wise (see
# test-ag-adam-fused.R: the kernel's 1 - fl(beta2)).
cs_opt <- function(ps) optimizer_adam(ps, lr = 0.02, beta1 = 0.875, beta2 = 1 - 2^-11)

# Ordinary loop, the reference; lr halved before step 5.
cs_plain <- function(device, steps = 8L) {
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  x <- ag_tensor(d$x); y <- ag_tensor(d$y)
  losses <- numeric(0)
  for (k in seq_len(steps)) {
    if (k == 5L) opt$lr <- 0.01
    loss <- NULL
    with_grad_tape(loss <- m$fn(x, y))
    opt$step(backward(loss))
    opt$zero_grad()
    losses <- c(losses, as.numeric(as.matrix(loss)))
  }
  list(w = lapply(m$ps, as.matrix), losses = losses, t = opt$t)
}

cs_captured <- function(device, steps = 8L) {
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  train <- ag_capture_step(m$fn, optimizer = opt)
  on.exit(ag_capture_free(train), add = TRUE)
  losses <- numeric(0)
  for (k in seq_len(steps)) {
    if (k == 5L) opt$lr <- 0.01
    losses <- c(losses, as.numeric(train(x = d$x, y = d$y)))
  }
  list(w = lapply(m$ps, as.matrix), losses = losses, t = opt$t, opt = opt)
}

test_that("ag_capture_step on the CPU runs the ordinary step", {
  on.exit(ag_device("cpu"), add = TRUE)
  ref <- cs_plain("cpu")
  got <- cs_captured("cpu")
  expect_equal(got$losses, ref$losses)
  expect_equal(got$w, ref$w)
  expect_identical(got$t, 8L)
})

test_that("captured step matches the uncaptured GPU loop and the CPU", {
  skip_if_no_gpu_cs()
  on.exit(ag_device("cpu"), add = TRUE)
  cpu <- cs_plain("cpu")
  ref <- cs_plain("gpu")
  got <- cs_captured("gpu")
  expect_identical(got$t, 8L)
  expect_equal(got$losses, ref$losses, tolerance = 1e-5)
  for (nm in names(ref$w)) expect_equal(got$w[[nm]], ref$w[[nm]], tolerance = 1e-5, info = nm)
  expect_equal(got$losses, cpu$losses, tolerance = 1e-4)
  for (nm in names(cpu$w)) expect_equal(got$w[[nm]], cpu$w[[nm]], tolerance = 1e-4, info = nm)
  # The moments advanced on the device are what state() reports.
  st <- got$opt$state()
  expect_identical(as.integer(st$t), 8L)
})

test_that("a captured step is two launches; outputs are named", {
  skip_if_no_gpu_cs()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  train <- ag_capture_step(function(x, y) {
    loss <- m$fn(x, y)
    list(loss = loss, l2 = ag_scale(loss, 2))
  }, optimizer = opt)
  on.exit(ag_capture_free(train), add = TRUE)
  first <- train(x = d$x, y = d$y)              # records, then replays
  expect_named(first, c("loss", "l2"))
  expect_equal(first$l2, 2 * first$loss, tolerance = 1e-6)
  ggmlR:::ag_launch_count(reset = TRUE)
  ggmlR:::ag_xfer_count(TRUE)
  on.exit(ggmlR:::ag_xfer_count(FALSE), add = TRUE)
  second <- train(x = d$x, y = d$y)
  expect_equal(launches(), 2)
  # One upload (inputs + AdamW hyperparameters share one buffer), one download.
  cnt <- ggmlR:::.ag_xfer$counts
  expect_identical(sort(ls(cnt)), c("down capture outputs", "up capture inputs"))
  expect_equal(cnt[["up capture inputs"]]$n, 1L)
  expect_equal(cnt[["down capture outputs"]]$n, 1L)
  expect_lt(as.numeric(second$loss), as.numeric(first$loss))
})

test_that("input shapes are fixed by the first call", {
  skip_if_no_gpu_cs()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  train <- ag_capture_step(m$fn, optimizer = opt)
  on.exit(ag_capture_free(train), add = TRUE)
  train(x = d$x, y = d$y)
  expect_error(train(x = d$x[, 1:16], y = d$y[, 1:16]), "shapes are fixed")
})

test_that("host data computed inside fn is refused", {
  skip_if_no_gpu_cs()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  train <- ag_capture_step(function(x, y)
    m$fn(x, ag_add(y, ag_tensor(matrix(1, 4L, 32L)))), optimizer = opt)
  expect_error(train(x = d$x, y = d$y), "outside the graph")
})

test_that("after a device switch the dead optimizer state is an error", {
  skip_if_no_gpu_cs()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  train <- ag_capture_step(m$fn, optimizer = opt)
  on.exit(ag_capture_free(train), add = TRUE)
  train(x = d$x, y = d$y)
  ag_device("cpu"); ag_device("gpu")
  expect_error(train(x = d$x, y = d$y), "not on the device")
})

# ---------------------------------------------------------------------------
# Clipping by global norm (max_grad_norm).

# Reference: the ordinary loop with clip_grad_norm(); the limit `mx` may be a
# vector, one per step (a schedule). Returns the pre-clipping norms too.
cs_plain_clip <- function(device, mx, seed = 21L, steps = length(mx)) {
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  d <- cs_data(); if (seed != 21L) { set.seed(seed); d$x <- matrix(rnorm(8 * 32), 8L, 32L) }
  m <- cs_model(d); opt <- cs_opt(m$ps)
  x <- ag_tensor(d$x); y <- ag_tensor(d$y)
  losses <- norms <- numeric(0)
  for (k in seq_len(steps)) {
    loss <- NULL
    with_grad_tape(loss <- m$fn(x, y))
    grads <- backward(loss)
    norms <- c(norms, clip_grad_norm(m$ps, grads, mx[k]))
    opt$step(grads)
    opt$zero_grad()
    losses <- c(losses, as.numeric(as.matrix(loss)))
  }
  list(w = lapply(m$ps, as.matrix), losses = losses, norms = norms)
}

cs_captured_clip <- function(device, mx, seed = 21L, steps = length(mx)) {
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  d <- cs_data(); if (seed != 21L) { set.seed(seed); d$x <- matrix(rnorm(8 * 32), 8L, 32L) }
  m <- cs_model(d); opt <- cs_opt(m$ps)
  train <- ag_capture_step(m$fn, optimizer = opt, max_grad_norm = mx[1L])
  on.exit(ag_capture_free(train), add = TRUE)
  losses <- norms <- numeric(0)
  caps <- list()
  for (k in seq_len(steps)) {
    ag_capture_set(train, max_grad_norm = mx[k])
    out <- train(x = d$x, y = d$y)
    caps[[k]] <- attr(train, "captures")$step
    losses <- c(losses, as.numeric(out$loss))
    norms  <- c(norms, as.numeric(out$grad_norm))
  }
  list(w = lapply(m$ps, as.matrix), losses = losses, norms = norms, caps = caps)
}

test_that("max_grad_norm and ag_capture_set validate their argument", {
  ag_device("cpu")
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  for (bad in list(0, -1, Inf, NA_real_, c(1, 2), "1"))
    expect_error(ag_capture_step(m$fn, optimizer = opt, max_grad_norm = bad),
                 "positive finite")
  plain <- ag_capture_step(m$fn, optimizer = opt)
  expect_error(ag_capture_set(plain, max_grad_norm = 1), "without max_grad_norm")
  clipped <- ag_capture_step(m$fn, optimizer = opt, max_grad_norm = 1)
  expect_error(ag_capture_set(clipped, max_grad_norm = 0), "positive finite")
})

test_that("clipping on the CPU equals clip_grad_norm in the ordinary loop", {
  on.exit(ag_device("cpu"), add = TRUE)
  mx <- rep(0.05, 6L)
  ref <- cs_plain_clip("cpu", mx)
  got <- cs_captured_clip("cpu", mx)
  expect_equal(got$losses, ref$losses)
  expect_equal(got$norms, ref$norms)
  expect_equal(got$w, ref$w)
})

test_that("captured clipping matches the ordinary loop: active, inactive, scheduled", {
  skip_if_no_gpu_cs()
  on.exit(ag_device("cpu"), add = TRUE)
  cases <- list(active   = rep(0.05, 6L),             # clips on every step
                inactive = rep(1e3, 6L),              # never clips
                schedule = c(rep(1e3, 3L), rep(0.05, 3L)))
  for (nm in names(cases)) {
    mx  <- cases[[nm]]
    cpu <- cs_plain_clip("cpu", mx)
    ref <- cs_plain_clip("gpu", mx)
    got <- cs_captured_clip("gpu", mx)
    # The pre-clipping norm, whether or not the step clipped.
    expect_equal(got$norms, ref$norms, tolerance = 1e-5, info = nm)
    expect_equal(got$losses, ref$losses, tolerance = 1e-5, info = nm)
    for (w in names(ref$w))
      expect_equal(got$w[[w]], ref$w[[w]], tolerance = 1e-5, info = paste(nm, w))
    expect_equal(got$losses, cpu$losses, tolerance = 1e-4, info = nm)
    if (nm == "active") expect_true(all(ref$norms > 0.05))
    if (nm == "inactive") expect_true(all(ref$norms < 1e3))
    # ag_capture_set changed the limit without recording again.
    expect_true(all(vapply(got$caps, identical, logical(1), got$caps[[1L]])), info = nm)
  }
  # Never clipping must equal not clipping at all.
  plain <- cs_captured("gpu", steps = 6L)
  got   <- cs_captured_clip("gpu", rep(1e3, 6L))
  expect_equal(got$losses[1:4], plain$losses[1:4], tolerance = 1e-6)
})

test_that("captured clipping matches the ordinary loop on several seeds", {
  skip_if_no_gpu_cs()
  on.exit(ag_device("cpu"), add = TRUE)
  for (seed in c(101L, 202L, 303L)) {
    mx  <- rep(0.1, 8L)
    ref <- cs_plain_clip("gpu", mx, seed = seed)
    got <- cs_captured_clip("gpu", mx, seed = seed)
    expect_equal(got$norms, ref$norms, tolerance = 1e-5, info = seed)
    for (w in names(ref$w))
      expect_equal(got$w[[w]], ref$w[[w]], tolerance = 1e-5, info = paste(seed, w))
  }
})

test_that("a non-finite gradient norm is an error and changes nothing", {
  skip_if_no_gpu_cs()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  d <- cs_data(); m <- cs_model(d); opt <- cs_opt(m$ps)
  train <- ag_capture_step(m$fn, optimizer = opt, max_grad_norm = 0.5)
  on.exit(ag_capture_free(train), add = TRUE)
  train(x = d$x, y = d$y)
  ggmlR:::ag_launch_count(reset = TRUE)
  train(x = d$x, y = d$y)
  expect_equal(launches(), 2)              # the check between graphs adds none

  before <- list(w = lapply(m$ps, as.matrix), st = opt$state(), t = opt$t)
  bad <- d$x; bad[1L, 1L] <- Inf
  expect_error(train(x = bad, y = d$y), "update was not applied")
  expect_identical(lapply(m$ps, as.matrix), before$w)
  expect_identical(opt$state(), before$st)
  expect_identical(opt$t, before$t)

  # The next step with good data runs normally.
  out <- train(x = d$x, y = d$y)
  expect_true(is.finite(out$grad_norm[1L]))
  expect_identical(opt$t, before$t + 1L)
})
