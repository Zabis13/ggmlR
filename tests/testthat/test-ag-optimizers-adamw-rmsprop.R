# optimizer_adamw() and optimizer_rmsprop() (R/autograd.R, R/ag_rmsprop.R):
# reference formulas (torch.optim semantics written out in R), decoupled vs L2
# decay, lazy slots, state round trip, memory multiplier, GPU vs CPU, and
# AdamW through ag_capture_step.

# ---- references: the torch.optim algorithms, written out --------------------

adamw_ref <- function(w, gs, lr, b1, b2, eps, wd) {
  m <- v <- w * 0
  for (t in seq_along(gs)) {
    g <- gs[[t]]
    w <- w * (1 - lr * wd)
    m <- b1 * m + (1 - b1) * g
    v <- b2 * v + (1 - b2) * g^2
    w <- w - lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
  }
  w
}

rmsprop_ref <- function(w, gs, lr, alpha, eps, wd, mom, centered) {
  v <- ga <- buf <- w * 0
  for (g in gs) {
    if (wd != 0) g <- g + wd * w
    v <- alpha * v + (1 - alpha) * g^2
    if (centered) { ga <- alpha * ga + (1 - alpha) * g; den <- sqrt(v - ga^2) + eps }
    else den <- sqrt(v) + eps
    if (mom > 0) { buf <- mom * buf + g / den; w <- w - lr * buf }
    else w <- w - lr * g / den
  }
  w
}

opt_data <- function() {
  set.seed(31)
  list(w0 = matrix(rnorm(12), 3, 4),
       gs = lapply(1:5, function(i) matrix(rnorm(12), 3, 4)))
}

# Feed a fixed gradient sequence through opt$step().
run_steps <- function(make_opt, d) {
  p <- ag_param(d$w0)
  opt <- make_opt(list(w = p))
  for (g in d$gs) { p$grad <- g; opt$step(); opt$zero_grad() }
  list(w = as.matrix(p), opt = opt)
}

rms_cfgs <- list(
  plain     = list(wd = 0,    mom = 0,   centered = FALSE),
  l2        = list(wd = 0.1,  mom = 0,   centered = FALSE),
  momentum  = list(wd = 0,    mom = 0.9, centered = FALSE),
  centered  = list(wd = 0,    mom = 0,   centered = TRUE),
  all       = list(wd = 0.05, mom = 0.8, centered = TRUE))

make_rms <- function(cf) function(ps)
  optimizer_rmsprop(ps, lr = 0.01, alpha = 0.9, eps = 1e-8,
                    weight_decay = cf$wd, momentum = cf$mom, centered = cf$centered)

# ---- AdamW --------------------------------------------------------------------

test_that("AdamW host step matches the reference", {
  d <- opt_data()
  got <- run_steps(function(ps) optimizer_adamw(ps, lr = 0.01, weight_decay = 0.1), d)
  expect_equal(got$w, adamw_ref(d$w0, d$gs, 0.01, 0.9, 0.999, 1e-8, 0.1),
               tolerance = 1e-10)
  expect_s3_class(got$opt, c("ag_optimizer_adamw", "ag_optimizer_adam"), exact = TRUE)
})

test_that("decoupled decay: zero gradient shrinks the weight by 1 - lr * wd", {
  w0 <- matrix(c(1, -2, 3, 0.5), 2, 2)
  p <- ag_param(w0)
  opt <- optimizer_adamw(list(p), lr = 0.1, weight_decay = 0.2)
  p$grad <- w0 * 0
  opt$step()
  expect_equal(as.matrix(p), w0 * (1 - 0.1 * 0.2), tolerance = 1e-12)
  # L2 (RMSprop's weight_decay) moves it differently: the decay enters g
  p2 <- ag_param(w0)
  opt2 <- optimizer_rmsprop(list(p2), lr = 0.1, weight_decay = 0.2)
  p2$grad <- w0 * 0
  opt2$step()
  expect_false(isTRUE(all.equal(as.matrix(p2), w0 * (1 - 0.1 * 0.2))))
})

test_that("optimizer_adam has no weight_decay; bad values are refused", {
  p <- ag_param(matrix(1, 2, 2))
  expect_error(optimizer_adam(list(p), weight_decay = 0.1), "unused argument")
  expect_error(optimizer_adamw(list(p), weight_decay = -1), "non-negative")
  expect_error(optimizer_rmsprop(list(p), alpha = 1.5), "alpha")
})

test_that("AdamW state round trip; Adam states stay free of weight_decay", {
  d <- opt_data()
  mk <- function(ps) optimizer_adamw(ps, lr = 0.01, weight_decay = 0.1)
  a <- run_steps(mk, list(w0 = d$w0, gs = d$gs[1:3]))
  st <- a$opt$state()
  expect_identical(st$type, "adamw")
  expect_equal(st$hyper$weight_decay, 0.1)
  p <- ag_param(a$w); b <- mk(list(w = p))
  expect_silent(b$load_state(st))
  for (g in d$gs[4:5]) { p$grad <- g; b$step(); b$zero_grad() }
  expect_equal(as.matrix(p), adamw_ref(d$w0, d$gs, 0.01, 0.9, 0.999, 1e-8, 0.1),
               tolerance = 1e-10)
  sa <- optimizer_adam(list(ag_param(d$w0)))$state()
  expect_null(sa$hyper$weight_decay)
  expect_error(optimizer_adam(list(ag_param(d$w0)))$load_state(st), "'adamw'")
})

# ---- RMSprop ------------------------------------------------------------------

test_that("RMSprop host step matches the reference for every option set", {
  d <- opt_data()
  for (nm in names(rms_cfgs)) {
    cf <- rms_cfgs[[nm]]
    got <- run_steps(make_rms(cf), d)
    expect_equal(got$w, rmsprop_ref(d$w0, d$gs, 0.01, 0.9, 1e-8, cf$wd, cf$mom,
                                    cf$centered), tolerance = 1e-10, info = nm)
  }
})

test_that("RMSprop creates only the slots it uses", {
  p <- ag_param(matrix(1, 2, 2))
  slots <- function(o) names(o$state()$slots)
  expect_identical(slots(optimizer_rmsprop(list(p))), "square_avg")
  expect_identical(slots(optimizer_rmsprop(list(p), momentum = 0.9)),
                   c("square_avg", "momentum_buffer"))
  expect_identical(slots(optimizer_rmsprop(list(p), centered = TRUE)),
                   c("square_avg", "grad_avg"))
  o <- optimizer_rmsprop(list(p))
  expect_null(o$momentum_buffer)
  expect_null(o$grad_avg)
})

test_that("RMSprop state round trip continues exactly, without warnings", {
  d <- opt_data()
  cf <- rms_cfgs$all
  a <- run_steps(make_rms(cf), list(w0 = d$w0, gs = d$gs[1:3]))
  st <- a$opt$state()
  p <- ag_param(a$w); b <- make_rms(cf)(list(w = p))
  expect_silent(b$load_state(st))
  for (g in d$gs[4:5]) { p$grad <- g; b$step(); b$zero_grad() }
  expect_equal(as.matrix(p), rmsprop_ref(d$w0, d$gs, 0.01, 0.9, 1e-8, cf$wd, cf$mom,
                                         cf$centered), tolerance = 1e-10)
})

test_that("RMSprop works with an lr scheduler", {
  p <- ag_param(matrix(1, 2, 2))
  opt <- optimizer_rmsprop(list(p), lr = 0.1)
  sch <- lr_scheduler_step(opt, step_size = 1L, gamma = 0.5)
  sch$step(); sch$step()
  expect_equal(opt$lr, 0.025)
})

# ---- memory multiplier ----------------------------------------------------------

test_that("memory multiplier equals the slots each optimizer really holds", {
  p <- ag_param(matrix(1, 2, 2))
  mult <- ggmlR:::.ag_opt_state_multiplier
  objs <- list(
    adam     = optimizer_adam(list(p)),
    adamw    = optimizer_adamw(list(p)),
    sgd      = optimizer_sgd(list(p), momentum = 0.9),
    rmsprop  = optimizer_rmsprop(list(p)),
    rmsprop_momentum = optimizer_rmsprop(list(p), momentum = 0.9),
    rmsprop_centered = optimizer_rmsprop(list(p), centered = TRUE),
    rmsprop_momentum_centered = optimizer_rmsprop(list(p), momentum = 0.9, centered = TRUE))
  for (nm in names(objs)) {
    n_slots <- length(objs[[nm]]$state()$slots)
    expect_equal(mult(objs[[nm]]), n_slots, info = nm)
    expect_equal(mult(nm), n_slots, info = nm)
  }
  r <- ag_estimate_training_memory(list(c(10, 10)), optimizer = objs$rmsprop_centered,
                                   activation_frac = 0, quiet = TRUE)
  expect_equal(r$optimizer, 100 * 8 * 2)
  expect_error(mult("nadam"), "rmsprop_centered")
})

# ---- GPU ----------------------------------------------------------------------

skip_if_no_gpu_opt <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

run_steps_dev <- function(make_opt, d, device) {
  ag_device(device)
  on.exit(ag_device("cpu"), add = TRUE)
  r <- run_steps(make_opt, d)
  list(w = as.matrix(r$w), resident = isTRUE(r$opt$resident))
}

test_that("AdamW and RMSprop on the GPU match the CPU (device step)", {
  skip_if_no_gpu_opt()
  d <- opt_data()
  makers <- c(list(adamw = function(ps) optimizer_adamw(ps, lr = 0.01, weight_decay = 0.1)),
              lapply(rms_cfgs, make_rms))
  for (fused in c("1", "0")) {
    withr::local_envvar(GGMLR_AG_ADAM_FUSED = fused)
    for (nm in names(makers)) {
      if (fused == "0" && nm != "adamw") next     # the switch only affects Adam
      cpu <- run_steps_dev(makers[[nm]], d, "cpu")
      gpu <- run_steps_dev(makers[[nm]], d, "gpu")
      info <- paste(nm, "fused", fused)
      expect_true(gpu$resident, info = info)
      expect_equal(gpu$w, cpu$w, tolerance = 1e-5, info = info)
    }
  }
})

# ---- ag_capture_step ------------------------------------------------------------

cap_data <- function() {
  set.seed(21L)
  list(w1 = matrix(rnorm(8 * 16) * 0.3, 16L, 8L), x = matrix(rnorm(8 * 32), 8L, 32L),
       y = matrix(rnorm(16 * 32) * 0.1, 16L, 32L))
}
cap_model <- function(d) {
  ps <- list(W1 = ag_param(d$w1), b1 = ag_param(matrix(0.5, 16L, 1L)))
  list(ps = ps, fn = function(x, y) ag_mse_loss(ag_add(ag_matmul(ps$W1, x), ps$b1), y))
}
# betas exact in float (see test-ag-capture-step.R)
cap_opt <- function(ps) optimizer_adamw(ps, lr = 0.02, beta1 = 0.875,
                                        beta2 = 1 - 2^-11, weight_decay = 0.5)

cap_run <- function(device, captured, steps = 6L) {
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  d <- cap_data(); m <- cap_model(d); opt <- cap_opt(m$ps)
  if (captured) {
    train <- ag_capture_step(m$fn, optimizer = opt)
    on.exit(ag_capture_free(train), add = TRUE)
  }
  x <- ag_tensor(d$x); y <- ag_tensor(d$y)
  for (k in seq_len(steps)) {
    if (k == 4L) opt$lr <- 0.01                  # schedule: decay follows lr
    if (captured) train(x = d$x, y = d$y)
    else {
      loss <- NULL
      with_grad_tape(loss <- m$fn(x, y))
      opt$step(backward(loss)); opt$zero_grad()
    }
  }
  lapply(m$ps, as.matrix)
}

test_that("AdamW decay reaches ag_capture_step, lr change included", {
  skip_if_no_gpu_opt()
  on.exit(ag_device("cpu"), add = TRUE)
  cpu <- cap_run("cpu", FALSE)
  ref <- cap_run("gpu", FALSE)
  got <- cap_run("gpu", TRUE)
  for (nm in names(ref)) {
    expect_equal(got[[nm]], ref[[nm]], tolerance = 1e-5, info = nm)
    expect_equal(got[[nm]], cpu[[nm]], tolerance = 1e-4, info = nm)
  }
  # and the decay did something: without it the result differs
  ag_device("cpu")
  expect_false(isTRUE(all.equal(cpu$W1, {
    d <- cap_data(); m <- cap_model(d)
    opt <- optimizer_adam(m$ps, lr = 0.02, beta1 = 0.875, beta2 = 1 - 2^-11)
    x <- ag_tensor(d$x); y <- ag_tensor(d$y)
    for (k in 1:6) { if (k == 4L) opt$lr <- 0.01
      loss <- NULL; with_grad_tape(loss <- m$fn(x, y))
      opt$step(backward(loss)); opt$zero_grad() }
    as.matrix(m$ps$W1)
  }, tolerance = 1e-6)))
})

test_that("ag_capture_step refuses RMSprop", {
  p <- ag_param(matrix(1, 2, 2))
  expect_error(ag_capture_step(function(x) ag_sum(ag_mul(p, x)),
                               optimizer = optimizer_rmsprop(list(p))),
               "only optimizer_adam\\(\\) and optimizer_adamw\\(\\)")
})
