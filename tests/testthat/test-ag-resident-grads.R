# Component 3 of the resident contract: gradients stay on the device.
#
# The graph backward downloaded every leaf gradient and then installed it on
# the leaf tensor -- two stages that measured 43-64% of a backward pass between
# them (inst/scripts/measure_ag_residency_on_backward.R). Neither needs the
# numbers: the optimizer does, eventually, and that is one download per step
# instead of one per leaf per pass.
#
# What must hold: gradients computed this way are the SAME gradients. Every
# test below is an equality against the non-resident path or against finite
# differences, because three silent bugs in this area all produced correct
# forward values and wrong gradients.

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

# Run a small dense model and return the parameter gradients as plain matrices.
run_model <- function(seed = 7L, d = 6L, b = 4L) {
  set.seed(seed)
  W1 <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
  W2 <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
  X  <- ag_tensor(matrix(rnorm(d * b) * 0.3, d, b))
  Y  <- matrix(0.0, d, b)
  with_grad_tape({
    h    <- ag_relu(ag_matmul(W1, X))
    out  <- ag_matmul(W2, h)
    loss <- ag_mse_loss(out, Y)
  })
  backward(loss)
  list(W1 = W1, W2 = W2, path = bwd_path())
}

with_settings <- function(graph, resident, expr) {
  og <- bwd_graph(graph); orr <- bwd_res(resident)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)
  force(expr)
}

test_that("resident gradients equal downloaded ones", {
  # The equivalence the whole component rests on.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  a <- with_settings(TRUE, FALSE, run_model())
  b <- with_settings(TRUE, TRUE,  run_model())
  skip_if_not(identical(a$path, "graph") && identical(b$path, "graph"),
              paste("graph path declined:", a$path, "/", b$path))

  expect_equal(as_mat(b$W1$grad), as_mat(a$W1$grad), tolerance = 1e-4)
  expect_equal(as_mat(b$W2$grad), as_mat(a$W2$grad), tolerance = 1e-4)
})

test_that("and equal the closure path too", {
  # The graph path is the one being changed; the closures are the reference
  # implementation that has not moved.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  clo <- with_settings(FALSE, FALSE, run_model())
  res <- with_settings(TRUE,  TRUE,  run_model())
  skip_if_not(identical(res$path, "graph"),
              paste("graph path declined:", res$path))

  expect_equal(as_mat(res$W1$grad), as_mat(clo$W1$grad), tolerance = 1e-3)
  expect_equal(as_mat(res$W2$grad), as_mat(clo$W2$grad), tolerance = 1e-3)
})

test_that("$grad really is a handle, not a downloaded matrix", {
  # If this fails the component silently did nothing: the tests above would
  # still pass while every gradient was being downloaded as before.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  r <- with_settings(TRUE, TRUE, run_model())
  skip_if_not(identical(r$path, "graph"), paste("graph path declined:", r$path))
  expect_true(is_handle(r$W1$grad))
  expect_true(is_handle(r$W2$grad))

  # and switching it off gives matrices again
  r2 <- with_settings(TRUE, FALSE, run_model())
  expect_false(is_handle(r2$W1$grad))
})

test_that("an optimizer step through resident gradients matches", {
  # .ag_opt_grad_for is where the one download per step happens; the weights it
  # produces must be the same either way.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  step_with <- function(resident) {
    r <- with_settings(TRUE, resident, run_model())
    opt <- optimizer_sgd(list(w1 = r$W1, w2 = r$W2), lr = 0.05)
    opt$step()
    list(w1 = ag_data(r$W1), w2 = ag_data(r$W2), path = r$path)
  }
  a <- step_with(FALSE)
  b <- step_with(TRUE)
  skip_if_not(identical(b$path, "graph"), paste("graph path declined:", b$path))

  expect_equal(b$w1, a$w1, tolerance = 1e-4)
  expect_equal(b$w2, a$w2, tolerance = 1e-4)
})

test_that("gradient accumulation across two backwards still sums", {
  # Two passes before zero_grad() is where residency had a real bug: a handle
  # installed by the first pass points at memory that with_grad_tape() frees at
  # the start of the second ("buffer freed by a tape reset, generation 12 < 13").
  #
  # The fix is not to materialise later but to not go resident at all when a
  # leaf already carries a gradient -- residency lasts exactly as long as the
  # tape that produced it. This test is the guard on that: it must sum
  # correctly, whatever the flag says.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  acc <- function(resident) {
    og <- bwd_graph(TRUE); orr <- bwd_res(resident)
    on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)
    set.seed(11L)
    d <- 5L
    W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
    X <- ag_tensor(matrix(rnorm(d * 3L) * 0.3, d, 3L))
    Y <- matrix(0.0, d, 3L)
    one <- function() {
      with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), Y) })
      backward(l)
    }
    one(); g1 <- as_mat(W$grad)
    one(); g2 <- as_mat(W$grad)
    list(single = g1, doubled = g2, path = bwd_path())
  }

  r <- acc(TRUE)
  skip_if_not(identical(r$path, "graph"), paste("graph path declined:", r$path))
  # the second pass accumulates onto the first: same gradient, so twice it
  expect_equal(r$doubled, 2 * r$single, tolerance = 1e-4)
})

test_that("a resident gradient is rescued by the tape reset that would free it", {
  # The rule that makes residency safe across passes. A $grad handle points into
  # a buffer the NEXT with_grad_tape() frees, and a gradient has no host
  # fallback -- unlike a value, it exists nowhere else. So .ag_residency_reset
  # materialises registered gradients before freeing anything.
  #
  # Asserted directly because the failure mode is delayed: without the rescue
  # the error appears at the next read, several frames from its cause (it did:
  # "buffer freed by a tape reset, generation 13 < 14").
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  og <- bwd_graph(TRUE); orr <- bwd_res(TRUE)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)
  set.seed(13L)
  d <- 4L
  W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
  X <- ag_tensor(matrix(rnorm(d * d) * 0.3, d, d))
  with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
  backward(l)
  skip_if_not(identical(bwd_path(), "graph"),
              paste("graph path declined:", bwd_path()))

  expect_true(is_handle(W$grad))          # resident, as intended
  before <- as_mat(W$grad)                # what it should still be worth

  # A reset frees the buffer the handle points into.
  get(".ag_residency_reset", envir = ns)()

  expect_false(is_handle(W$grad))         # rescued on the way out
  expect_true(is.matrix(W$grad))
  expect_equal(W$grad, before, tolerance = 1e-6)
})

test_that("clipping works on resident gradients", {
  # clip_grad_norm reads $grad through the env and does host arithmetic on it.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  og <- bwd_graph(TRUE); orr <- bwd_res(TRUE)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)
  set.seed(3L)
  d <- 4L
  W <- ag_param(matrix(rnorm(d * d), d, d))
  X <- ag_tensor(matrix(rnorm(d * d), d, d))
  with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
  g <- backward(l)
  skip_if_not(identical(bwd_path(), "graph"),
              paste("graph path declined:", bwd_path()))

  expect_no_error(nrm <- clip_grad_norm(list(w = W), g, max_norm = 1e-6))
  expect_true(is.finite(nrm))
})

test_that("resident gradients match finite differences", {
  # The reference that does not depend on any other path being right.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  og <- bwd_graph(TRUE); orr <- bwd_res(TRUE)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)

  set.seed(5L)
  d <- 4L
  w_val <- matrix(rnorm(d * d) * 0.3, d, d)
  x_val <- matrix(rnorm(d * d) * 0.3, d, d)
  y     <- matrix(0.0, d, d)

  loss_of <- function(w_mat) {
    W <- ag_param(w_mat, device = "gpu")
    X <- ag_tensor(x_val, device = "gpu")
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), y) })
    list(loss = ag_data(l)[1L], W = W, l = l)
  }

  r <- loss_of(w_val)
  backward(r$l)
  analytic <- as_mat(r$W$grad)

  eps <- 1e-3
  fd  <- matrix(0, d, d)
  for (i in seq_len(d)) for (j in seq_len(d)) {
    wp <- w_val; wp[i, j] <- wp[i, j] + eps
    wm <- w_val; wm[i, j] <- wm[i, j] - eps
    fd[i, j] <- (loss_of(wp)$loss - loss_of(wm)$loss) / (2 * eps)
  }
  expect_equal(analytic, fd, tolerance = 1e-2)
})

test_that("zero_grad drops the rescue register", {
  # Fix A: rescuing a gradient the optimizer has already consumed is a download
  # of numbers nobody will read. Measured at 14.5 ms of a 78 ms step -- four
  # gradients at ~9 ms each, all of them spent.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  og <- bwd_graph(TRUE); orr <- bwd_res(TRUE)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)
  st <- get(".ag_device_state", envir = ns)

  r <- run_model()
  skip_if_not(identical(r$path, "graph"), paste("graph path declined:", r$path))
  expect_gt(length(st$pending_grads), 0L)      # registered while it matters

  opt <- optimizer_sgd(list(w1 = r$W1, w2 = r$W2), lr = 0.01)
  opt$step()
  opt$zero_grad()
  expect_equal(length(st$pending_grads), 0L)   # and dropped once consumed
})

test_that("the register does not grow across passes", {
  # Without keying by id, a loop that never calls zero_grad() would append the
  # same tensors every pass and the reset would materialise each of them once
  # per registration.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  og <- bwd_graph(TRUE); orr <- bwd_res(TRUE)
  on.exit({ bwd_graph(og); bwd_res(orr) }, add = TRUE)
  st <- get(".ag_device_state", envir = ns)

  set.seed(17L)
  d <- 4L
  W <- ag_param(matrix(rnorm(d * d) * 0.3, d, d))
  X <- ag_tensor(matrix(rnorm(d * d) * 0.3, d, d))
  one <- function() {
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, X), matrix(0, d, d)) })
    backward(l)
  }
  one()
  skip_if_not(identical(bwd_path(), "graph"),
              paste("graph path declined:", bwd_path()))
  n1 <- length(st$pending_grads)
  one(); one()
  expect_lte(length(st$pending_grads), n1)
})

test_that("the flag defaults to on", {
  # Resident gradients are the default now that the two cross-tape holders are
  # fixed (dp_train's replica loop and ag_checkpoint's accumulation); the tests
  # above pin the behaviour that made that safe. GGMLR_AG_RESIDENT_GRADS=0 is
  # the way back to host-side gradients.
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- bwd_res(NA)
  expect_true(isTRUE(old))
})
