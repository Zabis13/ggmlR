# Deferred forward: build the nodes now, compute them once.
#
# The claim under test is equality, not speed. Deferral changes WHEN a value is
# computed, never what it is, so every test below compares the deferred path
# against the immediate one on the same inputs. The measured gain is 1.0-1.3x on
# a step (inst/scripts/proto_ag_forward_graph.R) -- this exists as groundwork for
# the fused forward+backward graph, and the thing that must not break on the way
# there is the arithmetic.
#
# The failure mode being guarded is specific and quiet: a read that skips the
# barrier returns whatever the buffer held, which is not an error and not a
# stale-pointer trap -- just wrong numbers. So the tests exercise the read paths
# (value, gradient, print, rescue-at-reset) rather than only the happy chain.

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

# Every test restores the previous setting: the gate is global, and a test that
# left it on would silently change the path of every test file after it.
with_defer <- function(on, expr) {
  old <- defer_on(on)
  on.exit(defer_on(old), add = TRUE)
  force(expr)
}

# A chain deep enough that deferral has something to batch, small enough to
# compare exactly.
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

test_that("a deferred forward computes what the immediate one computes", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  ref <- with_defer(FALSE, forward_only(cfg))
  got <- with_defer(TRUE,  forward_only(cfg))

  # Same device, same kernels, same order of operations -- the only difference
  # is how many computes it took, so this is exact, not approximate.
  expect_equal(got, ref)
})

test_that("deferral leaves gradients and loss unchanged", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  ref <- with_defer(FALSE, train_grads(cfg))
  got <- with_defer(TRUE,  train_grads(cfg))

  expect_equal(got$loss, ref$loss)
  for (i in seq_along(ref$grads))
    expect_equal(got$grads[[i]], ref$grads[[i]], tolerance = 1e-6)
})

test_that("the queue actually fills, then drains on read", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  # Without this the equality tests above would still pass with deferral doing
  # nothing at all -- the most likely way for this feature to be silently dead.
  with_defer(TRUE, {
    h <- ag_tensor(cfg$X)
    for (W in cfg$Wt) h <- ag_relu(ag_matmul(ag_tensor(W), h))
    expect_gt(defer_len(), 0L)
    invisible(ag_data(h))
    expect_equal(defer_len(), 0L)
  })
})

test_that("backward() settles a queue nobody read", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  # The tape's loss is never read here, so nothing before backward() has forced
  # the forward to be computed.
  #
  # ⚠️ This asserts the GRADIENTS, not an empty queue. It used to require
  # defer_len() == 0 afterwards, which was right while backward() drained
  # unconditionally and is wrong now: on the fused path backward() deliberately
  # leaves its own roots queued, and the barrier fires when a gradient is read.
  # Either way the numbers must be finite and correct, which is what matters
  # here -- see the fusion tests below for the queue-state assertions.
  with_defer(TRUE, {
    Wp <- lapply(cfg$Wt, ag_param)
    X  <- ag_tensor(cfg$X)
    Y  <- matrix(0.0, cfg$d, cfg$b)
    with_grad_tape({
      h <- X
      for (W in Wp) h <- ag_relu(ag_matmul(W, h))
      loss <- ag_mse_loss(h, Y)
    })
    backward(loss)
    for (p in Wp) expect_true(all(is.finite(as_mat(p$grad))))
    expect_equal(defer_len(), 0L)   # reading the gradients above drained it
  })
})

test_that("a full training step matches step for step", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  # The end-to-end shape: several optimizer steps, where a missed barrier would
  # compound rather than show up once. Loss trajectories must agree.
  losses <- function(defer) {
    with_defer(defer, {
      set.seed(3L)
      W <- ag_param(matrix(rnorm(16L) * 0.3, 4L, 4L))
      X <- ag_tensor(matrix(rnorm(8L) * 0.3, 4L, 2L))
      Y <- matrix(0.5, 4L, 2L)
      opt <- optimizer_adam(list(W = W), lr = 0.05)
      vapply(seq_len(4L), function(i) {
        with_grad_tape({
          out  <- ag_relu(ag_matmul(W, X))
          loss <- ag_mse_loss(out, Y)
        })
        backward(loss)
        opt$step(); opt$zero_grad()
        as.numeric(ag_data(loss))
      }, numeric(1))
    })
  }

  expect_equal(losses(TRUE), losses(FALSE), tolerance = 1e-6)
})

test_that("the optimizer step is never deferred", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  # Adam writes into the weight it owns (out=) and allocates from the persistent
  # pool, both of which .ag_defer_ok refuses. This pins that: a weight left
  # queued after step() would be read by the NEXT forward as an operand, and the
  # five iterations TODO records on the Adam step all had this shape.
  with_defer(TRUE, {
    set.seed(5L)
    W <- ag_param(matrix(rnorm(16L) * 0.3, 4L, 4L))
    X <- ag_tensor(matrix(rnorm(8L) * 0.3, 4L, 2L))
    Y <- matrix(0.0, 4L, 2L)
    opt <- optimizer_adam(list(W = W), lr = 0.1)
    before <- ag_data(W)
    with_grad_tape({
      loss <- ag_mse_loss(ag_matmul(W, X), Y)
    })
    backward(loss)
    opt$step()
    expect_equal(defer_len(), 0L)
    expect_false(isTRUE(all.equal(ag_data(W), before)))
  })
})

test_that("a reset with a live queue does not fabricate values", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  # The ordering trap in .ag_residency_reset: the rescue that copies resident
  # values back to the host runs while a queue may still be pending, and reading
  # an uncomputed node would write plausible-looking garbage into $data as
  # though it were authoritative. Draining before the rescue is what prevents
  # it; this test fails loudly if that order is ever reversed.
  cfg <- run_chain()
  with_defer(TRUE, {
    h <- ag_tensor(cfg$X)
    for (W in cfg$Wt) h <- ag_relu(ag_matmul(ag_tensor(W), h))
    # Leave it queued on purpose, then force the reset a new tape performs.
    expect_gt(defer_len(), 0L)
    with_grad_tape({ NULL })
    expect_equal(defer_len(), 0L)
  })
})

test_that("deferral is off unless asked for", {
  # A gate that quietly defaulted to on would change every path in the package
  # at once. No GPU needed: this is about the gate, not the device.
  old <- defer_on(NA)
  expect_false(isTRUE(old))
})

# ---------------------------------------------------------------------------
# Fused forward + backward: one graph for the whole step.
#
# What made this possible was not a change to the backward -- it never read a
# forward value, only pointers -- but the loss becoming resident. While
# ag_mse_loss divided its scalar on the host, the forward had to be computed
# before backward() could be called at all.
#
# The risk is the same one deferral carries everywhere: a value read before the
# barrier is plausible garbage rather than an error. So these tests compare
# against the unfused path rather than checking that anything merely runs.
# ---------------------------------------------------------------------------

bwd_path_of <- get("ag_backward_path", envir = ns)

test_that("the loss stays on the device until someone reads it", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  # The pointed part: a resident loss is what removes the crossing AND the
  # ordering constraint. If this regresses to a host matrix, fusion silently
  # stops happening and only the timing would show it.
  with_defer(TRUE, {
    W <- ag_param(matrix(1, 3L, 3L))
    X <- ag_tensor(matrix(1, 3L, 2L))
    with_grad_tape({
      loss <- ag_mse_loss(ag_matmul(W, X), matrix(0, 3L, 2L))
    })
    expect_true(is.null(loss$data))
    expect_false(is.null(loss$ptr))
    # W %*% X is 3x2 with every entry 3 (a sum of three 1*1 terms), so the
    # squared error is 54 over n = 6 entries -> 9. Worth spelling out: the
    # first version of this line asserted 81, from reading the product as 9
    # per entry and forgetting the division by n. The resident loss folds that
    # division into the graph (ggml_scale, 1/n), so an expectation that skips
    # it is testing the wrong formula rather than the code.
    expect_equal(as.numeric(ag_data(loss)), 9, tolerance = 1e-5)
  })
})

test_that("a fused step gives the same gradients as an unfused one", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  ref <- with_defer(FALSE, train_grads(cfg))
  got <- with_defer(TRUE,  train_grads(cfg))

  expect_equal(got$loss, ref$loss, tolerance = 1e-6)
  for (i in seq_along(ref$grads))
    expect_equal(got$grads[[i]], ref$grads[[i]], tolerance = 1e-6)
})

test_that("backward() leaves the graph queued instead of computing it", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  # Without this the equality above would pass with fusion never engaging --
  # the same "silently dead feature" failure the queue-length test guards for
  # the forward. The queue must still be non-empty AFTER backward() returns.
  with_defer(TRUE, {
    Wp <- lapply(cfg$Wt, ag_param)
    X  <- ag_tensor(cfg$X)
    with_grad_tape({
      h <- X
      for (W in Wp) h <- ag_relu(ag_matmul(W, h))
      loss <- ag_mse_loss(h, matrix(0, cfg$d, cfg$b))
    })
    backward(loss)
    skip_if_not(identical(bwd_path_of(), "graph"),
                "backward did not take the graph path")
    expect_gt(defer_len(), 0L)
    # Reading a gradient is the barrier.
    invisible(as_mat(Wp[[1L]]$grad))
    expect_equal(defer_len(), 0L)
  })
})

test_that("a fused step that is never read does not lose its gradients", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  # The rescue ordering, now reachable on the ordinary path: with fusion,
  # backward() does not drain, so a step whose gradients go unread arrives at
  # the next tape with everything still queued. .ag_residency_reset must drain
  # BEFORE materialising the pending gradients, or it rescues uncomputed
  # buffers and writes garbage into $grad as though it were authoritative.
  with_defer(TRUE, {
    Wp <- lapply(cfg$Wt, ag_param)
    X  <- ag_tensor(cfg$X)
    with_grad_tape({
      h <- X
      for (W in Wp) h <- ag_relu(ag_matmul(W, h))
      loss <- ag_mse_loss(h, matrix(0, cfg$d, cfg$b))
    })
    backward(loss)
    with_grad_tape({ NULL })          # forces the reset while the queue is live
    g <- as_mat(Wp[[1L]]$grad)
    expect_true(all(is.finite(g)))
    expect_false(all(g == 0))
  })
})

test_that("fusion survives a chain deep enough to span contexts", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  # .ag_ctx_ensure retires a context and opens a new one when a request does not
  # fit, so a long queue spans several. The barrier has to flush all of them:
  # a tensor left without a backend buffer is not an error, it simply computes
  # nothing, and the result reads as garbage rather than as a failure.
  cfg <- run_chain(seed = 23L, d = 4L, b = 2L, depth = 24L)
  ref <- with_defer(FALSE, train_grads(cfg))
  got <- with_defer(TRUE,  train_grads(cfg))

  expect_equal(got$loss, ref$loss, tolerance = 1e-5)
  for (i in seq_along(ref$grads))
    expect_equal(got$grads[[i]], ref$grads[[i]], tolerance = 1e-5)
})

test_that("training converges the same way with fusion on", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  # End to end: the loss trajectory over several Adam steps. A fusion bug that
  # affected only one operand would show here as a drift rather than as an
  # error, which is how the Adam ordering defect was caught.
  losses <- function(defer) {
    with_defer(defer, {
      set.seed(9L)
      W <- ag_param(matrix(rnorm(16L) * 0.3, 4L, 4L))
      X <- ag_tensor(matrix(rnorm(8L) * 0.3, 4L, 2L))
      Y <- matrix(0.5, 4L, 2L)
      opt <- optimizer_adam(list(W = W), lr = 0.05)
      vapply(seq_len(6L), function(i) {
        with_grad_tape({
          loss <- ag_mse_loss(ag_relu(ag_matmul(W, X)), Y)
        })
        backward(loss)
        opt$step(); opt$zero_grad()
        as.numeric(ag_data(loss))
      }, numeric(1))
    })
  }
  a <- losses(TRUE); b <- losses(FALSE)
  expect_equal(a, b, tolerance = 1e-6)
  expect_lt(a[6L], a[1L])          # and it actually trained
})

test_that("the barrier fires at the first gradient read, not later", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cfg <- run_chain()

  # The symmetric invariant to the one relaxed in "backward() settles a queue
  # nobody read". That test no longer requires an empty queue after backward(),
  # because leaving it queued IS fusion -- so on its own it would catch a
  # performance regression and miss a correctness one: a compute that never
  # happens because the read went through a path with no barrier.
  #
  # This pins the other half: the queue must be empty the moment a gradient has
  # been read, and the value read must be the computed one. "Drains eventually"
  # is not enough -- the optimizer is the first consumer, and it reads $grad.
  with_defer(TRUE, {
    Wp <- lapply(cfg$Wt, ag_param)
    X  <- ag_tensor(cfg$X)
    with_grad_tape({
      h <- X
      for (W in Wp) h <- ag_relu(ag_matmul(W, h))
      loss <- ag_mse_loss(h, matrix(0, cfg$d, cfg$b))
    })
    backward(loss)

    g1 <- as_mat(Wp[[1L]]$grad)
    # Empty IMMEDIATELY after the first read, not after some later one.
    expect_equal(defer_len(), 0L)
    expect_true(all(is.finite(g1)))
    expect_false(all(g1 == 0))

    # And the value is stable: reading again cannot change it, which it would
    # if the first read had returned an uncomputed buffer.
    expect_equal(as_mat(Wp[[1L]]$grad), g1)
  })
})

test_that("a host-path loss drains the queue before it reads pred", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  # CE and SCE are host-side: they open with .ag_data(pred), which materialises
  # the whole forward. Mixed with a deferred forward that is a topological
  # hazard of the same family as the Adam "one step ahead" defect -- a consumer
  # reading operands the queue has not computed.
  #
  # It is closed by construction rather than by a special case: .ag_data on a
  # resident tensor goes through .ag_gpu_to_r, whose first statement is a drain.
  # This test pins that construction, since the guarantee is invisible at the
  # CE call site and would be easy to lose while refactoring the accessors.
  with_defer(TRUE, {
    set.seed(31L)
    W <- ag_param(matrix(rnorm(9L) * 0.3, 3L, 3L))
    X <- ag_tensor(matrix(abs(rnorm(6L)) + 0.5, 3L, 2L))
    Y <- matrix(c(1, 0, 0, 0, 1, 0), 3L, 2L)

    with_grad_tape({
      h    <- ag_softmax(ag_matmul(W, X))
      loss <- ag_cross_entropy_loss(h, Y)
    })
    # Reading pred inside CE drained everything the forward had queued.
    expect_equal(defer_len(), 0L)
    lv <- as.numeric(ag_data(loss))
    expect_true(is.finite(lv))
    expect_gt(lv, 0)

    backward(loss)
    g <- as_mat(W$grad)
    expect_true(all(is.finite(g)))
    expect_false(all(g == 0))
  })
})
