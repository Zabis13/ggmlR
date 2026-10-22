# Device Adam under ag_dtype("f16"): F32 moments, F32 master weights, F32
# gradient -- for both the fused and the per-op step.
#
# Before: the step ran in the weight's type. (1 - beta2) * g^2 underflowed to 0
# in f16 for |g| below ~8e-3, eps = 1e-8 is 0 in f16, and m / 0 made the weight
# Inf at the FIRST step (measured on the autograd vignette model: every
# non-finite weight sat on an element with v == 0; eps = 1e-3 hid it). An
# F32 step on a weight rounded back to f16 every time would still lose updates
# smaller than the f16 spacing, so the step runs on an F32 master and copies
# it into the f16 weight.
#
# Everything that reads a device handle happens INSIDE with_gpu_dtype():
# leaving it switches to the CPU, which frees the persistent pool.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns        <- asNamespace("ggmlR")
h_to_mat  <- get(".ag_as_matrix", envir = ns)
handle_of <- get(".ag_handle_of", envir = ns)
as_mat    <- function(x) if (inherits(x, "ag_tensor")) as.matrix(x) else h_to_mat(x)
type_of   <- function(h) ggml_type_name(ggml_tensor_type(h$ptr))

with_fused <- function(on, code) {
  old <- Sys.getenv("GGMLR_AG_ADAM_FUSED", unset = NA)
  Sys.setenv(GGMLR_AG_ADAM_FUSED = if (on) "1" else "0")
  on.exit(if (is.na(old)) Sys.unsetenv("GGMLR_AG_ADAM_FUSED")
          else Sys.setenv(GGMLR_AG_ADAM_FUSED = old), add = TRUE)
  force(code)
}

with_gpu_dtype <- function(dtype, code) {
  ag_device("gpu")
  old <- ag_dtype(dtype)
  on.exit({ ag_dtype(old); ag_device("cpu") }, add = TRUE)
  force(code)
}

paths <- c(fused = TRUE, per_op = FALSE)

# n steps of Adam on one parameter (initial value w0, gradients grad_at(k)),
# optionally resumed from `state`. Returns host values only.
run_adam <- function(fused, w0, grad_at, steps, lr, dtype = "f16", state = NULL) {
  with_gpu_dtype(dtype, with_fused(fused, {
    W   <- ag_param(w0)
    opt <- optimizer_adam(list(W = W), lr = lr)
    restored <- NULL
    if (!is.null(state)) {
      opt$load_state(state)
      restored <- opt$state()$master[[1]]
    }
    for (k in steps) { W$grad <- grad_at(k); opt$step() }
    st <- opt$state()
    list(w = as_mat(W), master = st$master[[1]], state = st, restored = restored,
         w_type = type_of(handle_of(W)),
         m_type = type_of(opt$m[[1]]), v_type = type_of(opt$v[[1]]),
         master_type = if (is.null(W$master)) NA_character_ else type_of(W$master))
  }))
}

test_that("state is F32 under f16: moments and master", {
  skip_if_no_gpu()
  for (nm in names(paths)) {
    r <- run_adam(paths[[nm]], matrix(0.5, 3L, 4L), function(k) matrix(0.1, 3L, 4L),
                  1L, 1e-3)
    expect_identical(r$w_type, "f16", info = nm)
    expect_identical(r$m_type, "f32", info = nm)
    expect_identical(r$v_type, "f32", info = nm)
    expect_identical(r$master_type, "f32", info = nm)
  }
})

test_that("the first step from small gradients stays finite", {
  # |g| = 1e-3: (1 - beta2) * g^2 = 1e-9 is below the smallest f16, and so is
  # eps = 1e-8. In f16 that was m / 0 at step 1.
  skip_if_no_gpu()
  w0 <- matrix(seq(-0.5, 0.5, length.out = 12L), 3L, 4L)
  for (nm in names(paths)) {
    r <- run_adam(paths[[nm]], w0, function(k) matrix(1e-3, 3L, 4L), 1L, 1e-3)
    expect_true(all(is.finite(r$w)), info = nm)
    expect_true(all(is.finite(r$master)), info = nm)
    expect_false(isTRUE(all.equal(r$w, w0)), info = nm)   # it moved
  }
})

test_that("updates smaller than the f16 spacing accumulate in the master", {
  # Constant gradient: Adam's bias-corrected step is lr / (1 + eps), here 1e-4
  # -- a tenth of the f16 spacing at 1.0 (2^-10). Rounded back to f16 each
  # step the weight would never leave 1; through the master it drifts by
  # 50 * 1e-4.
  skip_if_no_gpu()
  for (nm in names(paths)) {
    r <- run_adam(paths[[nm]], matrix(1, 2L, 2L), function(k) matrix(1, 2L, 2L),
                  1:50, 1e-4)
    expect_true(all(abs(r$master - (1 - 50 * 1e-4)) < 1e-5), info = nm)
    # The working weight is the master rounded to f16 (spacing 2^-11 below 1).
    expect_true(all(abs(r$w - r$master) <= 2^-12), info = nm)
    expect_true(all(r$w < 1), info = nm)
  }
})

test_that("fused and per-op steps agree under f16", {
  # Not bit-identical (the kernel takes beta2 as float, see test-ag-adam-fused.R);
  # close on the F32 master.
  skip_if_no_gpu()
  w0 <- matrix(seq(-0.4, 0.4, length.out = 15L), 5L, 3L)
  g  <- function(k) matrix(sin(seq_len(15L) * 0.7 + k) * 1e-2, 5L, 3L)
  a  <- run_adam(TRUE,  w0, g, 1:20, 1e-3)
  b  <- run_adam(FALSE, w0, g, 1:20, 1e-3)
  expect_true(max(abs(a$master - b$master)) < 1e-5)
  expect_true(max(abs(a$w - b$w)) <= 2^-10)
})

test_that("training through backward() under f16 stays finite and learns", {
  # Resident f16 gradients from the backward graph reach the step and are cast
  # to F32 there. Same model shape as the autograd vignette, smaller.
  skip_if_no_gpu()
  data(iris)
  x <- t(scale(as.matrix(iris[, 1:4])))
  y <- t(model.matrix(~ Species - 1, iris))
  for (nm in names(paths)) {
    losses <- with_gpu_dtype("f16", with_fused(paths[[nm]], {
      set.seed(3)
      model <- ag_sequential(ag_linear(4L, 32L, activation = "relu"),
                             ag_batch_norm(32L), ag_linear(32L, 3L))
      opt <- optimizer_adam(model$parameters(), lr = 1e-2)
      ag_train(model)
      out <- numeric(0)
      for (k in 1:40) {
        ii <- sample(150L, 32L)
        with_grad_tape({
          loss <- ag_softmax_cross_entropy_loss(
            model$forward(ag_tensor(x[, ii, drop = FALSE])), y[, ii, drop = FALSE])
        })
        out <- c(out, as.matrix(loss)[1])
        opt$step(backward(loss)); opt$zero_grad()
      }
      out
    }))
    expect_true(all(is.finite(losses)), info = nm)
    expect_lt(mean(tail(losses, 5)), mean(head(losses, 5)))
  }
})

test_that("optimizer state carries the master and restores it", {
  # Resume after 10 of 20 steps from opt$state(), into a fresh f16 parameter
  # holding the ROUNDED weight: the master in the state is what makes the
  # resumed run continue the straight one instead of restarting from f16.
  skip_if_no_gpu()
  w0 <- matrix(1, 2L, 3L)
  g  <- function(k) matrix(1 + 0.1 * k, 2L, 3L)
  for (nm in names(paths)) {
    straight <- run_adam(paths[[nm]], w0, g, 1:20, 1e-4)
    half     <- run_adam(paths[[nm]], w0, g, 1:10, 1e-4)
    # The master is more precise than the weight it was rounded into.
    expect_false(isTRUE(all.equal(half$master, half$w, tolerance = 0)), info = nm)

    resumed <- run_adam(paths[[nm]], half$w, g, 11:20, 1e-4, state = half$state)
    expect_equal(resumed$restored, half$master, tolerance = 0, info = nm)
    expect_true(max(abs(resumed$master - straight$master)) < 1e-6, info = nm)
    expect_true(max(abs(resumed$w - straight$w)) <= 2^-11, info = nm)
  }
})

test_that("a state without master weights still loads (older states)", {
  skip_if_no_gpu()
  half <- run_adam(TRUE, matrix(1, 2L, 2L), function(k) matrix(1, 2L, 2L), 1:5, 1e-4)
  st <- half$state
  st$master <- NULL
  r <- run_adam(TRUE, half$w, function(k) matrix(1, 2L, 2L), 6L, 1e-4, state = st)
  expect_null(r$restored)          # nothing restored before the step ...
  expect_true(all(is.finite(r$w))) # ... and the step from the f16 weight works
})

test_that("F32 is unchanged: the weight is its own master, the state has none", {
  skip_if_no_gpu()
  for (nm in names(paths)) {
    r <- run_adam(paths[[nm]], matrix(0.5, 3L, 2L), function(k) matrix(0.1, 3L, 2L),
                  1L, 1e-3, dtype = "f32")
    expect_identical(r$w_type, "f32", info = nm)
    expect_true(is.na(r$master_type), info = nm)
    expect_null(r$master, info = nm)
    # One Adam step from zero moments with a constant gradient: w - lr.
    expect_equal(r$w, matrix(0.5 - 1e-3, 3L, 2L), tolerance = 1e-6, info = nm)
  }
})
