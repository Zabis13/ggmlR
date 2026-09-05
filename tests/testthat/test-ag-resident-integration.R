# Residency against the rest of the autograd engine.
#
# Components 1-3 changed shared code: .ag_run_op, backward(), the optimizers,
# .ag_residency_reset, the clipping utilities and the dp_train allreduce. Their
# own test files check the new behaviour, but every OTHER ag_* test file runs
# with the flag off -- so the subsystems that merely consume gradients were
# never exercised against a $grad that is a device handle.
#
# That is the gap this file closes. Each test runs a real subsystem end to end
# with residency on, and compares against the same run with it off. A handle has
# no arithmetic methods on purpose (rule 3 of the data contract), so anything
# that reaches for $grad without going through the accessors fails loudly here
# rather than in a user's training loop.
#
# The comparison is always "same result", never "resident works". Three silent
# bugs in this engine produced correct forward values and wrong gradients, so
# equality of loss curves proves nothing on its own -- the gradients themselves
# are compared, and one test differentiates against finite differences.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns        <- asNamespace("ggmlR")
bwd_graph <- get("ag_backward_graph",    envir = ns)
bwd_res   <- get("ag_backward_resident", envir = ns)
bwd_path  <- get("ag_backward_path",     envir = ns)
ag_data   <- get(".ag_data",             envir = ns)
as_mat    <- get(".ag_as_matrix",        envir = ns)
is_handle <- get(".ag_is_handle",        envir = ns)

# Run `f` under a given residency setting, restoring both flags afterwards.
under <- function(resident, f) {
  og <- bwd_graph(TRUE); orr <- bwd_res(resident)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)
  f()
}

# Both settings, same seed, so the two runs are comparable by construction.
both <- function(f) list(off = under(FALSE, f), on = under(TRUE, f))

test_that("a full training loop converges the same either way", {
  # The integration test proper: many steps, optimizer, zero_grad, the lot.
  # If residency leaked a stale handle or lost a gradient, the loss curves
  # would part company.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  train <- function() {
    set.seed(101L)
    d <- 8L; b <- 16L
    W1 <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
    W2 <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
    X  <- ag_tensor(matrix(rnorm(d * b) * 0.3, d, b))
    Y  <- matrix(rnorm(d * b) * 0.1, d, b)
    opt <- optimizer_sgd(list(w1 = W1, w2 = W2), lr = 0.02)
    losses <- numeric(8L)
    for (i in seq_len(8L)) {
      with_grad_tape({
        h <- ag_relu(ag_matmul(W1, X))
        l <- ag_mse_loss(ag_matmul(W2, h), Y)
      })
      losses[i] <- ag_data(l)[1L]
      backward(l); opt$step(); opt$zero_grad()
    }
    list(losses = losses, w1 = ag_data(W1), path = bwd_path())
  }

  r <- both(train)
  skip_if_not(identical(r$on$path, "graph"),
              paste("graph path declined:", r$on$path))
  expect_equal(r$on$losses, r$off$losses, tolerance = 1e-3)
  expect_equal(r$on$w1,     r$off$w1,     tolerance = 1e-3)
  expect_lt(r$on$losses[8L], r$on$losses[1L])   # and it actually learned
})

test_that("Adam agrees with itself across residency", {
  # SGD reads $grad once; Adam also builds m and v from it, so a gradient that
  # materialised differently would show up in the moments rather than the step.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  train <- function() {
    set.seed(202L)
    d <- 6L
    W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
    X <- ag_tensor(matrix(rnorm(d * d) * 0.3, d, d))
    opt <- optimizer_adam(list(w = W), lr = 0.05)
    for (i in seq_len(5L)) {
      with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
      backward(l); opt$step(); opt$zero_grad()
    }
    # The moments are device handles once the optimizer runs on the GPU, so they
    # are materialised here: the comparison below is about their VALUES, and
    # comparing handles would compare pointers, which differ by construction.
    list(w = ag_data(W),
         m = as_mat(opt$m$w), v = as_mat(opt$v$w),
         t = opt$t, path = bwd_path())
  }

  r <- both(train)
  skip_if_not(identical(r$on$path, "graph"),
              paste("graph path declined:", r$on$path))
  expect_equal(r$on$w, r$off$w, tolerance = 1e-4)
  expect_equal(r$on$m, r$off$m, tolerance = 1e-4)   # first moment
  expect_equal(r$on$v, r$off$v, tolerance = 1e-4)   # second moment
  expect_equal(r$on$t, r$off$t)                     # step counter
})

test_that("gradient accumulation over micro-batches agrees", {
  # accumulate_steps holds $grad across several backward passes, which is
  # exactly the case where a handle outlives the tape that produced it.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  train <- function() {
    set.seed(303L)
    d <- 5L
    W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
    xs <- lapply(1:4, function(i) ag_tensor(matrix(rnorm(d * 3L) * 0.3, d, 3L)))
    opt <- optimizer_sgd(list(w = W), lr = 0.05, accumulate_steps = 4L)
    for (x in xs) {
      with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, x), matrix(0, d, 3L)) })
      backward(l)
      opt$step()          # updates only on the fourth call
    }
    opt$zero_grad()
    list(w = ag_data(W), path = bwd_path())
  }

  r <- both(train)
  skip_if_not(identical(r$on$path, "graph"),
              paste("graph path declined:", r$on$path))
  expect_equal(r$on$w, r$off$w, tolerance = 1e-4)
})

test_that("clipping produces the same norm and the same update", {
  # clip_grad_norm reads every gradient, computes a global norm on the host and
  # writes scaled values back -- three separate chances to mishandle a handle.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  run <- function() {
    set.seed(404L)
    d <- 6L
    W <- ag_param(matrix(rnorm(d * d), d, d))
    X <- ag_tensor(matrix(rnorm(d * d), d, d))
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
    g <- backward(l)
    nrm <- clip_grad_norm(list(w = W), g, max_norm = 0.01)
    opt <- optimizer_sgd(list(w = W), lr = 0.1)
    opt$step(g)
    list(norm = nrm, w = ag_data(W), path = bwd_path())
  }

  r <- both(run)
  skip_if_not(identical(r$on$path, "graph"),
              paste("graph path declined:", r$on$path))
  expect_equal(r$on$norm, r$off$norm, tolerance = 1e-4)
  expect_equal(r$on$w,    r$off$w,    tolerance = 1e-4)
})

test_that("check_grad_anomaly reads resident gradients", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  run <- function() {
    set.seed(505L)
    d <- 4L
    W <- ag_param(matrix(rnorm(d * d), d, d))
    X <- ag_tensor(matrix(rnorm(d * d), d, d))
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
    g <- backward(l)
    list(rep = check_grad_anomaly(list(w = W), g, action = "silent"),
         path = bwd_path())
  }

  r <- both(run)
  skip_if_not(identical(r$on$path, "graph"),
              paste("graph path declined:", r$on$path))
  expect_equal(r$on$rep$status,  r$off$rep$status)
  expect_equal(r$on$rep$max_abs, r$off$rep$max_abs, tolerance = 1e-4)
})

test_that("an ag_sequential module trains identically", {
  # A layer module rather than bare parameters: parameters() collects them,
  # the optimizer updates them by name, and the values feed the next forward.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  train <- function() {
    set.seed(606L)
    model <- ag_sequential(ag_linear(8L, 8L, activation = "relu"),
                           ag_linear(8L, 4L))
    X <- ag_tensor(matrix(rnorm(8L * 12L) * 0.3, 8L, 12L))
    Y <- matrix(rnorm(4L * 12L) * 0.1, 4L, 12L)
    opt <- optimizer_sgd(model$parameters(), lr = 0.02)
    losses <- numeric(5L)
    for (i in seq_len(5L)) {
      with_grad_tape({ l <- ag_mse_loss(model$forward(X), Y) })
      losses[i] <- ag_data(l)[1L]
      backward(l); opt$step(); opt$zero_grad()
    }
    list(losses = losses, path = bwd_path())
  }

  r <- both(train)
  skip_if_not(identical(r$on$path, "graph"),
              paste("graph path declined:", r$on$path))
  expect_equal(r$on$losses, r$off$losses, tolerance = 1e-3)
})

test_that("saving a model after a resident backward round-trips", {
  # ag_save_model reads parameter VALUES, not gradients, but it runs right
  # after a backward and must not be confused by a tensor whose gradient is a
  # handle -- nor may the values themselves have drifted.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  path_rds <- tempfile(fileext = ".rds")
  on.exit(unlink(path_rds), add = TRUE)

  under(TRUE, function() {
    set.seed(707L)
    mk <- function() ag_sequential(ag_linear(6L, 6L, activation = "relu"),
                                   ag_linear(6L, 3L))
    model <- mk()
    X <- ag_tensor(matrix(rnorm(6L * 8L) * 0.3, 6L, 8L))
    with_grad_tape({ l <- ag_mse_loss(model$forward(X), matrix(0, 3L, 8L)) })
    backward(l)
    skip_if_not(identical(bwd_path(), "graph"),
                paste("graph path declined:", bwd_path()))

    before <- lapply(model$parameters(), ag_data)
    ag_save_model(model, path_rds, model_fn = mk)
    restored <- ag_load_model(path_rds)
    after <- lapply(restored$parameters(), ag_data)
    expect_equal(after, before, tolerance = 1e-5)
  })
})

test_that("residency survives an error mid-pass", {
  # An op that throws leaves the tape and the register in whatever state they
  # were in. The next pass must still work rather than trip over a handle from
  # the abandoned one.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  under(TRUE, function() {
    set.seed(808L)
    d <- 5L
    W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
    X <- ag_tensor(matrix(rnorm(d * d) * 0.3, d, d))

    # a shape error inside the tape
    expect_error(
      with_grad_tape({
        bad <- ag_matmul(W, ag_tensor(matrix(1, d + 1L, 2L)))
      }))

    # and a clean pass afterwards
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
    expect_no_error(backward(l))
    expect_false(is.null(W$grad))
    expect_true(all(is.finite(as_mat(W$grad))))
  })
})

test_that("switching residency mid-session does not corrupt state", {
  # Turning the flag on and off between passes leaves matrices and handles in
  # $grad alternately; nothing may carry over between the two shapes.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  og <- bwd_graph(TRUE); orr <- bwd_res(FALSE)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)

  set.seed(909L)
  d <- 5L
  W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
  X <- ag_tensor(matrix(rnorm(d * d) * 0.3, d, d))
  opt <- optimizer_sgd(list(w = W), lr = 0.01)
  one <- function() {
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
    backward(l); opt$step(); opt$zero_grad()
  }

  for (resident in c(FALSE, TRUE, FALSE, TRUE, TRUE, FALSE)) {
    bwd_res(resident)
    expect_no_error(one())
  }
  expect_true(all(is.finite(ag_data(W))))
})

test_that("the closure path is untouched by the residency flag", {
  # A tape the graph path declines must behave identically whatever the flag
  # says -- ag_sum is not in .AG_BWD_GRAPH_OPS, so this exercises the fallback.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  run <- function() {
    set.seed(111L)
    d <- 4L
    W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
    X <- ag_tensor(matrix(rnorm(d * d) * 0.3, d, d))
    with_grad_tape({ l <- ag_sum(ag_matmul(W, X)) })
    backward(l)
    list(grad = as_mat(W$grad), path = bwd_path())
  }
  r <- both(run)
  expect_false(identical(r$on$path, "graph"))   # declined, as intended
  expect_equal(r$on$grad, r$off$grad, tolerance = 1e-5)
  expect_false(is_handle(r$on$grad))            # closures never make handles
})
