# Tests for ag_checkpoint() -- trading recompute for activation memory.
#
# Checkpointing is only worth having if its gradients are indistinguishable
# from the ordinary tape's, so almost every test here compares against a
# non-checkpointed run of the same model rather than against hand-written
# numbers. The failure modes it must rule out are all silent:
#
#   * the segment's parameters get NO gradient (the node was dropped from the
#     tape because its visible inputs did not require one);
#   * the segment's parameters get a DOUBLED gradient (the boundary tensor was
#     not detached, so the gradient arrived through the outer tape as well as
#     through the replay);
#   * layers BEFORE the checkpoint get no gradient (the boundary was detached
#     too aggressively and the chain broke);
#   * a segment containing dropout replays with a different mask, so backward
#     uses a mask the forward never saw.
#
# None of these produce an error, which is why each has a test.

ag_checkpoint <- ggmlR:::ag_checkpoint

# Build the same two-layer model twice, once plain and once with the first
# layer wrapped in a checkpoint, and return both gradient sets.
both_ways <- function(seed = 4L, n = 4L) {
  mk <- function() {
    set.seed(seed)
    list(w1 = ag_param(matrix(runif(n * n, -1, 1), n, n)),
         w2 = ag_param(matrix(runif(n * n, -1, 1), n, n)),
         x  = ag_tensor(matrix(runif(n * n, -1, 1), n, n)),
         y  = ag_tensor(matrix(runif(n * n, -1, 1), n, n)))
  }

  v <- mk()
  with_grad_tape({
    h    <- ag_relu(ag_matmul(v$w1, v$x))
    o    <- ag_matmul(v$w2, h)
    loss <- ag_mse_loss(o, v$y)
  })
  backward(loss)
  ref <- list(w1 = v$w1$grad, w2 = v$w2$grad)

  u <- mk()
  with_grad_tape({
    h    <- ag_checkpoint(function(inp) ag_relu(ag_matmul(u$w1, inp)), u$x)
    o    <- ag_matmul(u$w2, h)
    loss <- ag_mse_loss(o, u$y)
  })
  backward(loss)
  got <- list(w1 = u$w1$grad, w2 = u$w2$grad)

  list(ref = ref, got = got)
}

test_that("checkpointed gradients equal the ordinary ones", {
  r <- both_ways()

  # The segment's own parameter: this is the one that gets no gradient at all
  # if the checkpoint node is dropped from the tape.
  expect_false(is.null(r$got$w1))
  expect_equal(r$got$w1, r$ref$w1, tolerance = 1e-12)

  # The layer after the checkpoint, unaffected by any of this.
  expect_equal(r$got$w2, r$ref$w2, tolerance = 1e-12)
})

test_that("the segment's gradient is not doubled", {
  # A doubled gradient is what happens when the boundary tensor is not detached
  # and the gradient reaches w1 both through the outer tape and through the
  # replay. Equality above would already catch it, but state it directly: the
  # ratio to the reference must be 1, not 2.
  r <- both_ways()
  ratio <- max(abs(r$got$w1)) / max(abs(r$ref$w1))
  expect_equal(ratio, 1, tolerance = 1e-9)
})

test_that("gradients flow through a checkpoint to earlier layers", {
  # The opposite failure: detaching so thoroughly that nothing upstream of the
  # checkpoint trains. w0 sits BEFORE the checkpointed segment.
  mk <- function() {
    set.seed(7L)
    list(w0 = ag_param(matrix(runif(16, -1, 1), 4, 4)),
         w1 = ag_param(matrix(runif(16, -1, 1), 4, 4)),
         x  = ag_tensor(matrix(runif(16, -1, 1), 4, 4)),
         y  = ag_tensor(matrix(runif(16, -1, 1), 4, 4)))
  }

  v <- mk()
  with_grad_tape({
    a    <- ag_matmul(v$w0, v$x)
    h    <- ag_relu(ag_matmul(v$w1, a))
    loss <- ag_mse_loss(h, v$y)
  })
  backward(loss)
  ref_w0 <- v$w0$grad

  u <- mk()
  with_grad_tape({
    a    <- ag_matmul(u$w0, u$x)
    h    <- ag_checkpoint(function(inp) ag_relu(ag_matmul(u$w1, inp)), a)
    loss <- ag_mse_loss(h, u$y)
  })
  backward(loss)

  expect_false(is.null(u$w0$grad))
  expect_equal(u$w0$grad, ref_w0, tolerance = 1e-12)
})

test_that("a checkpointed segment stores no activations", {
  # The whole point: the tape must be smaller. Measured on the tape itself
  # rather than on gc(), so the number means "what the tape holds" and nothing
  # else.
  n <- 64L
  mk <- function() {
    set.seed(9L)
    list(w1 = ag_param(matrix(runif(n * n, -1, 1), n, n)),
         w2 = ag_param(matrix(runif(n * n, -1, 1), n, n)),
         x  = ag_tensor(matrix(runif(n * n, -1, 1), n, n)),
         y  = ag_tensor(matrix(runif(n * n, -1, 1), n, n)))
  }
  tape_bytes <- function() {
    sum(vapply(ggmlR:::.ag_tape$nodes,
               function(nd) as.numeric(utils::object.size(nd)), numeric(1)))
  }

  v <- mk()
  with_grad_tape({
    h    <- ag_relu(ag_matmul(v$w1, v$x))
    o    <- ag_matmul(v$w2, h)
    loss <- ag_mse_loss(o, v$y)
  })
  plain <- tape_bytes()

  u <- mk()
  with_grad_tape({
    h    <- ag_checkpoint(function(inp) ag_relu(ag_matmul(u$w1, inp)), u$x)
    o    <- ag_matmul(u$w2, h)
    loss <- ag_mse_loss(o, u$y)
  })
  checkpointed <- tape_bytes()

  expect_lt(checkpointed, plain)
})


test_that("checkpoints nest", {
  # Depth > 1 exercises the tape stack itself: the inner replay saves and
  # restores .ag_tape$nodes while the OUTER replay is already doing the same.
  # Getting that wrong loses either the outer segment's nodes (no gradient for
  # the outer parameters) or the inner ones (no gradient for the inner).
  # Nothing errors in either case, so the test compares against a plain run.
  mk <- function() {
    set.seed(13L)
    list(w1 = ag_param(matrix(runif(16, -1, 1), 4, 4)),
         w2 = ag_param(matrix(runif(16, -1, 1), 4, 4)),
         w3 = ag_param(matrix(runif(16, -1, 1), 4, 4)),
         x  = ag_tensor(matrix(runif(16, -1, 1), 4, 4)),
         y  = ag_tensor(matrix(runif(16, -1, 1), 4, 4)))
  }

  v <- mk()
  with_grad_tape({
    h1   <- ag_relu(ag_matmul(v$w1, v$x))
    h2   <- ag_relu(ag_matmul(v$w2, h1))
    o    <- ag_matmul(v$w3, h2)
    loss <- ag_mse_loss(o, v$y)
  })
  backward(loss)
  ref <- list(w1 = v$w1$grad, w2 = v$w2$grad, w3 = v$w3$grad)

  # A checkpoint whose segment itself contains a checkpoint.
  u <- mk()
  with_grad_tape({
    h2 <- ag_checkpoint(function(inp) {
      inner <- ag_checkpoint(function(z) ag_relu(ag_matmul(u$w1, z)), inp)
      ag_relu(ag_matmul(u$w2, inner))
    }, u$x)
    o    <- ag_matmul(u$w3, h2)
    loss <- ag_mse_loss(o, u$y)
  })
  backward(loss)

  expect_false(is.null(u$w1$grad))   # innermost segment
  expect_false(is.null(u$w2$grad))   # outer segment
  expect_equal(u$w1$grad, ref$w1, tolerance = 1e-10)
  expect_equal(u$w2$grad, ref$w2, tolerance = 1e-10)
  expect_equal(u$w3$grad, ref$w3, tolerance = 1e-10)
})

test_that("several checkpoints in sequence each get their gradient", {
  # The common usage: every Nth block wrapped. Each segment replays
  # independently during one backward pass, so a shared-state bug in the replay
  # (a tape not restored, an RNG left advanced) shows up as the later segments
  # disagreeing with the reference while the first one looks fine.
  n <- 4L
  mk <- function() {
    set.seed(17L)
    list(w = lapply(1:4, function(i) ag_param(matrix(runif(n * n, -1, 1), n, n))),
         x = ag_tensor(matrix(runif(n * n, -1, 1), n, n)),
         y = ag_tensor(matrix(runif(n * n, -1, 1), n, n)))
  }

  v <- mk()
  with_grad_tape({
    h <- v$x
    for (i in 1:4) h <- ag_relu(ag_matmul(v$w[[i]], h))
    loss <- ag_mse_loss(h, v$y)
  })
  backward(loss)
  ref <- lapply(v$w, function(p) p$grad)

  # NOTE the local(): a checkpoint segment is replayed LATER, during
  # backward, so it must not close over a loop variable. Writing
  # `wi <- u$w[[i]]` in the loop body leaves every segment sharing one binding,
  # and by replay time they all see the last layer -- earlier ones then get no
  # gradient at all while the loss still looks fine. This is R closure capture,
  # not a checkpoint bug, but it is the way users will hit it first.
  u <- mk()
  with_grad_tape({
    h <- u$x
    for (i in 1:4) {
      seg <- local({
        wi <- u$w[[i]]
        function(inp) ag_relu(ag_matmul(wi, inp))
      })
      h <- if (i %% 2L == 0L) ag_checkpoint(seg, h) else seg(h)
    }
    loss <- ag_mse_loss(h, u$y)
  })
  backward(loss)

  for (i in 1:4) {
    expect_false(is.null(u$w[[i]]$grad), info = paste("layer", i))
    expect_equal(u$w[[i]]$grad, ref[[i]], tolerance = 1e-10,
                 info = paste("layer", i))
  }
})

test_that("a stochastic segment replays with the same random draw", {
  # ag_dropout draws its mask from runif(). If the replay drew a fresh one, the
  # backward would apply a mask the forward never used -- wrong gradients with
  # nothing to signal it. The check is against a plain run with the same seed:
  # if the RNG were not restored, these would differ.
  mk <- function() {
    set.seed(11L)
    list(w  = ag_param(matrix(runif(16, -1, 1), 4, 4)),
         dp = ag_dropout(0.5),
         x  = ag_tensor(matrix(runif(16, -1, 1), 4, 4)),
         y  = ag_tensor(matrix(runif(16, -1, 1), 4, 4)))
  }

  v <- mk()
  set.seed(99L)
  with_grad_tape({
    h    <- v$dp$forward(ag_matmul(v$w, v$x))
    loss <- ag_mse_loss(h, v$y)
  })
  backward(loss)
  ref <- v$w$grad

  u <- mk()
  set.seed(99L)
  with_grad_tape({
    h    <- ag_checkpoint(function(inp) u$dp$forward(ag_matmul(u$w, inp)), u$x)
    loss <- ag_mse_loss(h, u$y)
  })
  backward(loss)

  expect_equal(u$w$grad, ref, tolerance = 1e-12)
})

test_that("a failing segment leaves the tape and RNG untouched", {
  # A checkpoint that throws must not disable the tape for everything after it,
  # nor advance the random stream -- both would corrupt an entire training run
  # while looking like an unrelated bug much later.
  x <- ag_tensor(matrix(runif(4), 2, 2))
  set.seed(5L)
  runif(1L)
  seed_before <- .Random.seed

  with_grad_tape({
    expect_error(
      ag_checkpoint(function(inp) stop("boom"), x),
      "boom")
    expect_true(ggmlR:::.ag_tape$enabled)
  })

  expect_identical(.Random.seed, seed_before)
})

test_that("ag_checkpoint outside a tape just runs the function", {
  x <- ag_tensor(matrix(c(1, 2, 3, 4), 2, 2))
  out <- ag_checkpoint(function(inp) ag_relu(inp), x)
  expect_true(ggmlR:::is_ag_tensor(out))
  expect_equal(ggmlR:::.ag_data(out), matrix(c(1, 2, 3, 4), 2, 2))
})
