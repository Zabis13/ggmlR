# ag_state_dict() / ag_load_state_dict() and opt$state() / opt$load_state().
#
# The property that matters: resume == continuing. Train k steps, snapshot,
# train k more (reference); separately rebuild from the snapshot and train the
# same k more. On one device and dtype the two runs must agree -- bit for bit on
# the CPU (no RNG in the step), within float tolerance on the GPU and across
# devices.

skip_if_no_gpu_sd <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

sd_data <- local({
  set.seed(501L)
  list(x = matrix(rnorm(4 * 16), 4, 16), y = matrix(rnorm(2 * 16), 2, 16))
})

sd_build <- function(seed, h = 8L) {
  set.seed(seed)
  ag_sequential(ag_linear(4L, h, "relu"), ag_batch_norm(h), ag_linear(h, 2L))
}

sd_train <- function(model, opt, k) {
  ag_train(model)
  for (i in seq_len(k)) {
    with_grad_tape({
      loss <- ag_mse_loss(model$forward(ag_tensor(sd_data$x)), ag_tensor(sd_data$y))
    })
    opt$step(backward(loss))
    opt$zero_grad()
  }
  invisible(model)
}

weights <- function(model) {
  sd <- ag_state_dict(model)
  c(sd$parameters, sd$buffers)
}

opt_makers <- list(
  adam = function(m) optimizer_adam(m$parameters(), lr = 0.01),
  sgd  = function(m) optimizer_sgd(m$parameters(), lr = 0.05, momentum = 0.9))

# Continue k steps from a snapshot taken on `dev_from`, resumed on `dev_to`.
resume_vs_continue <- function(make_opt, dev_from, dev_to, k = 3L) {
  ag_device(dev_from)
  m1 <- sd_build(1L); o1 <- make_opt(m1)
  sd_train(m1, o1, k)
  snap <- list(model = ag_state_dict(m1), opt = o1$state())
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
  saveRDS(snap, f)                               # through a file, as a checkpoint
  sd_train(m1, o1, k)
  cont <- weights(m1)

  ag_device(dev_to)
  snap <- readRDS(f)
  m2 <- sd_build(999L); o2 <- make_opt(m2)       # different initial weights
  ag_load_state_dict(m2, snap$model)
  o2$load_state(snap$opt)
  sd_train(m2, o2, k)
  list(cont = cont, res = weights(m2))
}

max_diff <- function(a, b) max(mapply(function(x, y) max(abs(x - y)), a, b))

test_that("resume equals continuing, bit for bit on the CPU", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (nm in names(opt_makers)) {
    r <- resume_vs_continue(opt_makers[[nm]], "cpu", "cpu")
    expect_identical(r$res, r$cont, info = nm)
  }
})

test_that("resume equals continuing on the GPU, and across devices", {
  skip_if_no_gpu_sd()
  on.exit(ag_device("cpu"), add = TRUE)
  for (nm in names(opt_makers)) {
    for (path in list(c("gpu", "gpu"), c("cpu", "gpu"), c("gpu", "cpu"))) {
      r <- resume_vs_continue(opt_makers[[nm]], path[1L], path[2L])
      expect_lt(max_diff(r$res, r$cont), 1e-4)
    }
  }
})

test_that("a state taken before the first step loads, and loading is idempotent", {
  ag_device("cpu")
  m <- sd_build(1L)
  o <- optimizer_adam(m$parameters(), lr = 0.01)
  st0 <- o$state()
  expect_identical(st0$t, 0L)
  expect_true(all(vapply(st0$slots$m, function(x) all(x == 0), logical(1))))

  m2 <- sd_build(1L); o2 <- optimizer_adam(m2$parameters(), lr = 0.01)
  expect_no_error(o2$load_state(st0))

  sd_train(m, o, 2L)
  sd <- ag_state_dict(m); st <- o$state()
  m3 <- sd_build(7L); o3 <- optimizer_adam(m3$parameters(), lr = 0.01)
  ag_load_state_dict(m3, sd); o3$load_state(st)
  once <- list(weights(m3), o3$state())
  ag_load_state_dict(m3, sd); o3$load_state(st)
  expect_identical(list(weights(m3), o3$state()), once)
  expect_identical(o3$t, 2L)
})

test_that("ag_load_state_dict checks everything before writing", {
  ag_device("cpu")
  m  <- sd_build(1L)
  sd <- ag_state_dict(sd_build(2L))
  before <- weights(m)

  bad <- sd
  nm2 <- names(bad$parameters)[2L]
  bad$parameters[[nm2]] <- matrix(0, 3, 3)       # first entry fine, second not
  expect_error(ag_load_state_dict(m, bad), "shape mismatch")
  expect_identical(weights(m), before)            # nothing written

  bad <- sd; bad$parameters[[1L]][1, 1] <- NaN
  expect_error(ag_load_state_dict(m, bad), "NA, NaN or Inf")

  bad <- sd; bad$parameters$extra_W <- matrix(0, 1, 1)
  expect_error(ag_load_state_dict(m, bad), "not in model: extra_W")
  bad <- sd; bad$parameters[[1L]] <- NULL
  expect_error(ag_load_state_dict(m, bad), "missing in state")
  expect_identical(weights(m), before)

  # strict = FALSE: skip one-sided names, still shape-check shared ones
  part <- sd; part$parameters <- part$parameters[1L]; part$parameters$extra_W <- matrix(0, 1, 1)
  ag_load_state_dict(m, part, strict = FALSE)
  w <- weights(m)
  expect_identical(w[[1L]], sd$parameters[[1L]])
  expect_identical(w[-1L], before[-1L])
  part$parameters[[1L]] <- matrix(0, 2, 2)
  expect_error(ag_load_state_dict(m, part, strict = FALSE), "shape mismatch")

  # a different architecture is refused by shape
  expect_error(ag_load_state_dict(sd_build(1L, h = 6L), sd), "shape mismatch")
})

test_that("ag_load_state_dict restores batch-norm running statistics", {
  ag_device("cpu")
  m <- sd_build(1L); sd_train(m, optimizer_sgd(m$parameters(), lr = 0.01), 3L)
  sd <- ag_state_dict(m)
  expect_true(any(sd$buffers[[1L]] != 0))         # stats actually moved
  m2 <- sd_build(5L)
  ag_load_state_dict(m2, sd)
  expect_identical(ag_state_dict(m2)$buffers, sd$buffers)
})

test_that("opt$load_state refuses mismatched states", {
  ag_device("cpu")
  m <- sd_build(1L)
  adam <- optimizer_adam(m$parameters(), lr = 0.01)
  sgd  <- optimizer_sgd(m$parameters(), lr = 0.05)
  st <- adam$state()

  expect_error(sgd$load_state(st), "state is for a 'adam' optimizer")
  expect_error(optimizer_adam(m$parameters()[-1L])$load_state(st), "parameters, optimizer has")
  renamed <- m$parameters(); names(renamed)[1L] <- "other"
  expect_error(optimizer_adam(renamed, lr = 0.01)$load_state(st), "names differ")
  unnamed <- unname(m$parameters())
  expect_no_error(optimizer_adam(unnamed, lr = 0.01)$load_state(st))   # forms only
  bad <- st; bad$slots$m[[1L]] <- matrix(0, 1, 1)
  expect_error(adam$load_state(bad), "has shape")
  bad <- st; bad$version <- 99L
  expect_error(adam$load_state(bad), "unsupported state version")
  expect_error(adam$load_state(list(1)), "not an optimizer state")
  expect_warning(optimizer_adam(m$parameters(), lr = 0.5)$load_state(st),
                 "lr = 0.5 \\(state 0.01\\)")
  # classed, still a warning, with the changed values as data
  w <- tryCatch(optimizer_adam(m$parameters(), lr = 0.5, eps = 1e-6)$load_state(st),
                warning = function(w) w)
  expect_s3_class(w, c("ggmlR_opt_hyper_warning", "warning", "condition"), exact = TRUE)
  expect_named(w$changed, c("lr", "eps"))
  expect_identical(w$changed$lr, list(current = 0.5, state = 0.01))
  # muffling the class leaves the load done and other conditions alone
  o <- optimizer_adam(m$parameters(), lr = 0.5)
  withCallingHandlers(o$load_state(st),
                      ggmlR_opt_hyper_warning = function(w) invokeRestart("muffleWarning"))
  expect_identical(o$t, st$t)
})

test_that("a container written by ag_save_model loads as a state dict", {
  ag_device("cpu")
  m <- sd_build(1L)
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
  ag_save_model(m, f)
  m2 <- sd_build(3L)
  ag_load_state_dict(m2, readRDS(f))
  expect_identical(weights(m2), weights(m))
})
