# Tests for the graph backward path (R/ag_backward_graph.R).
#
# The path computes backward() as ONE ggml graph instead of one R closure per
# tape node. It is an optimisation, so the thing worth testing is not that it
# runs but that it computes THE SAME GRADIENTS as the closures it replaces --
# a fast wrong backward would train a model to nothing while looking healthy.
#
# So every numeric test here is a comparison against the closure path on the
# same tape, not against hand-written expected values. The closure path is the
# reference implementation; if the two ever disagree, this file says so.
#
# The second thing tested is the fallback. Stage 1 emits matmul, add and
# mse_loss only, and a tape containing anything else must fall back WHOLESALE
# rather than mixing paths. That is a correctness property, not a performance
# one: a half-graph backward would silently drop gradients for the ops it
# skipped.

# The graph path's controls are internal (no @export), so reach them the way a
# probe does. `:::` keeps the test working under plain testthat::test_dir() as
# well as under test_check(), which runs inside the package namespace -- the
# unqualified name only resolves in the latter, and that difference is exactly
# what made these tests pass locally and fail on the installed package.
ag_backward_graph <- ggmlR:::ag_backward_graph
ag_backward_path  <- ggmlR:::ag_backward_path
`%||%`            <- ggmlR:::`%||%`

skip_if_no_gpu <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

# Run the same tape twice -- once through each backward path -- and return both
# gradient sets. Seeding before each build keeps the two tapes identical.
both_paths <- function(build) {
  old <- ag_backward_graph(FALSE)
  on.exit(ag_backward_graph(old %||% FALSE), add = TRUE)

  ag_backward_graph(FALSE)
  ref <- build()
  ref_grads <- backward(ref$loss)
  ref_path  <- ag_backward_path()

  ag_backward_graph(TRUE)
  got <- build()
  got_grads <- backward(got$loss)
  got_path  <- ag_backward_path()

  list(ref = ref_grads, ref_params = ref$params, ref_path = ref_path,
       got = got_grads, got_params = got$params, got_path = got_path)
}

# Gradients come back as device handles when residency is on, so they are read
# through the accessor rather than as fields. A handle has no arithmetic on
# purpose (rule 3 of the data contract), which is why `a - b` on one fails
# loudly instead of quietly computing the wrong thing -- helpful in the engine,
# but here it just means the test has to materialise first.
.bwd_as_matrix <- get(".ag_as_matrix", envir = asNamespace("ggmlR"))
.bwd_dim       <- get(".ag_dim",       envir = asNamespace("ggmlR"))

# Max absolute difference between the gradients of matching parameters.
grad_maxdiff <- function(r) {
  stopifnot(length(r$ref_params) == length(r$got_params))
  max(vapply(seq_along(r$ref_params), function(i) {
    a <- r$ref_params[[i]]$grad
    b <- r$got_params[[i]]$grad
    if (is.null(a) && is.null(b)) return(0)
    if (is.null(a) || is.null(b)) return(Inf)   # one path produced no gradient
    max(abs(.bwd_as_matrix(a) - .bwd_as_matrix(b)))
  }, numeric(1)))
}

test_that("graph backward matches the closure backward on a matmul chain", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  build <- function() {
    set.seed(11L)
    w1 <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    w2 <- ag_param(matrix(runif(6,  -1, 1), 3, 2))
    x  <- ag_tensor(matrix(runif(4 * 5, -1, 1), 5, 4))
    y  <- ag_tensor(matrix(runif(5 * 2, -1, 1), 5, 2))
    loss <- NULL
    with_grad_tape({
      h    <- ag_matmul(x, w1)
      o    <- ag_matmul(h, w2)
      loss <- ag_mse_loss(o, y)
    })
    list(loss = loss, params = list(w1, w2))
  }

  r <- both_paths(build)

  expect_identical(r$ref_path, "closures")
  expect_identical(r$got_path, "graph")
  # f16 accumulation on Vulkan: the tolerance is the device's, not the rule's.
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("graph backward matches the closure backward with a bias add", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # No broadcast: the bias is full-shape, which is the add case stage 1 emits.
  build <- function() {
    set.seed(12L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    b <- ag_param(matrix(runif(15, -1, 1), 5, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      o    <- ag_add(ag_matmul(x, w), b)
      loss <- ag_mse_loss(o, y)
    })
    list(loss = loss, params = list(w, b))
  }

  r <- both_paths(build)

  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
})


test_that("graph backward handles a column-broadcast bias", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # b is [m,1] broadcast across columns -- the shape ag_linear uses, and the one
  # that needs the transposed reduction (sum_rows cannot reduce ne[1] directly).
  build <- function() {
    set.seed(22L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    b <- ag_param(matrix(runif(5,  -1, 1), 5, 1))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_add(ag_matmul(x, w), b), y)
    })
    list(loss = loss, params = list(w, b))
  }

  r <- both_paths(build)

  expect_identical(r$got_path, "graph")
  # The reduction must also produce the right SHAPE: a [1,m] gradient for a
  # [m,1] parameter would broadcast silently in the optimizer.
  expect_identical(.bwd_dim(r$got_params[[2]]$grad), c(5L, 1L))
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("graph backward handles a row-broadcast bias", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  build <- function() {
    set.seed(23L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    b <- ag_param(matrix(runif(3,  -1, 1), 1, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_add(ag_matmul(x, w), b), y)
    })
    list(loss = loss, params = list(w, b))
  }

  r <- both_paths(build)

  expect_identical(r$got_path, "graph")
  expect_identical(.bwd_dim(r$got_params[[2]]$grad), c(1L, 3L))
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("graph backward matches closures on a real ag_linear stack", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # The case stage 1 exists for: two ag_linear layers with no activation, so
  # every tape node is matmul/add/mse_loss and the whole pass qualifies. The
  # bias is [out,1], i.e. the column broadcast.
  build <- function() {
    set.seed(31L)
    l1 <- ag_linear(6L, 4L)
    l2 <- ag_linear(4L, 2L)
    x  <- ag_tensor(matrix(runif(6 * 8, -1, 1), 6, 8))
    y  <- ag_tensor(matrix(runif(2 * 8, -1, 1), 2, 8))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(l2$forward(l1$forward(x)), y)
    })
    list(loss = loss, params = list(l1$W, l1$b, l2$W, l2$b))
  }

  r <- both_paths(build)

  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
  expect_identical(.bwd_dim(r$got_params[[2]]$grad), c(4L, 1L))
  expect_identical(.bwd_dim(r$got_params[[4]]$grad), c(2L, 1L))
})


test_that("graph backward matches closures for relu, sigmoid and tanh", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # All three reduce to dx = g * mult with a multiplier the forward pass already
  # computed, so they share one emission rule. Testing them together is what
  # catches a rule wired to the wrong multiplier -- each has a different one.
  for (act in list(list(f = ag_relu,    nm = "relu"),
                   list(f = ag_sigmoid, nm = "sigmoid"),
                   list(f = ag_tanh,    nm = "tanh"))) {
    build <- function() {
      set.seed(41L)
      w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
      x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
      y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
      loss <- NULL
      with_grad_tape({
        loss <- ag_mse_loss(act$f(ag_matmul(x, w)), y)
      })
      list(loss = loss, params = list(w))
    }

    r <- both_paths(build)

    expect_identical(r$got_path, "graph", info = act$nm)
    expect_lt(grad_maxdiff(r), 1e-3)
  }
})

test_that("graph backward matches closures on an activated ag_linear stack", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # The realistic case: two ag_linear layers WITH activations, so the tape mixes
  # matmul, broadcast add, two different activations and the loss -- and still
  # has to run entirely as one graph.
  build <- function() {
    set.seed(31L)
    l1 <- ag_linear(6L, 4L, activation = "relu")
    l2 <- ag_linear(4L, 2L, activation = "tanh")
    x  <- ag_tensor(matrix(runif(6 * 8, -1, 1), 6, 8))
    y  <- ag_tensor(matrix(runif(2 * 8, -1, 1), 2, 8))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(l2$forward(l1$forward(x)), y)
    })
    list(loss = loss, params = list(l1$W, l1$b, l2$W, l2$b))
  }

  r <- both_paths(build)

  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("an op outside the covered set still forces a fallback", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # ag_sub records no op description, so the tape must decline AS A WHOLE even
  # though every other node on it is covered. That is the guarantee that keeps a
  # partly covered tape from silently dropping the gradients it cannot emit.
  #
  # This test has had to be rewritten twice, as relu and then softmax gained
  # emitters -- so pick the op deliberately: it must be one that is genuinely
  # outside .AG_BWD_GRAPH_OPS, and it should be re-pointed rather than deleted
  # when ag_sub is eventually covered too.
  build <- function() {
    set.seed(51L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    b <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_sub(ag_matmul(x, w), b), y)
    })
    list(loss = loss, params = list(w))
  }

  r <- both_paths(build)

  expect_match(r$got_path, "^closures \\(")
  expect_lt(grad_maxdiff(r), 1e-6)   # same code ran twice: no device tolerance
})


test_that("graph backward matches closures for the three losses", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # All three share one emission rule (loss_const): a matrix the forward pass
  # already built, times a scalar. Only gmat and gscale differ, so testing all
  # three together is what catches a rule wired to the wrong pair.
  build_sce <- function() {
    set.seed(61L)
    w <- ag_param(matrix(runif(12, -1, 1), 3, 4))
    x <- ag_tensor(matrix(runif(24, -1, 1), 4, 6))
    y <- c(0L, 2L, 1L, 0L, 1L, 2L)
    loss <- NULL
    with_grad_tape({
      loss <- ag_softmax_cross_entropy_loss(ag_matmul(w, x), y)
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build_sce)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)

  build_ce <- function() {
    set.seed(62L)
    w  <- ag_param(matrix(runif(12, -1, 1), 3, 4))
    x  <- ag_tensor(matrix(runif(24, -1, 1), 4, 6))
    tg <- matrix(0, 3, 6)
    for (i in 1:6) tg[((i - 1L) %% 3L) + 1L, i] <- 1
    loss <- NULL
    with_grad_tape({
      loss <- ag_cross_entropy_loss(ag_sigmoid(ag_matmul(w, x)), tg)
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build_ce)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("graph backward matches closures for transpose, softmax and scale", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # transpose: dx = t(g), emitted as cont(transpose(g)).
  build_t <- function() {
    set.seed(71L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 3, 5))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_transpose(ag_matmul(x, w)), y)
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build_t)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)

  # softmax: the column coupling makes this four nodes, not an elementwise one.
  build_s <- function() {
    set.seed(72L)
    w <- ag_param(matrix(runif(12, -1, 1), 3, 4))
    x <- ag_tensor(matrix(runif(24, -1, 1), 4, 6))
    y <- ag_tensor(matrix(runif(18, -1, 1), 3, 6))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_softmax(ag_matmul(w, x)), y)
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build_s)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)

  # scale: a scalar multiplier, so nothing is uploaded for the rule at all.
  build_sc <- function() {
    set.seed(75L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_scale(ag_matmul(x, w), 0.3), y)
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build_sc)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("multi-head attention runs entirely as one graph", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # ag_multihead_attention is composed from the primitives above -- its head
  # slicing and concatenation go through selector matrices, i.e. ag_matmul, and
  # the causal mask is an ag_add of a constant. So it needs no emitter of its
  # own, and this test is what proves that claim rather than assuming it: a
  # single uncovered node anywhere in those ~31 tape entries forces a fallback.
  for (causal in c(FALSE, TRUE)) {
    build <- function() {
      set.seed(73L)
      at <- ag_multihead_attention(8L, 2L)
      x  <- ag_tensor(matrix(runif(48, -1, 1), 8, 6))
      y  <- ag_tensor(matrix(runif(48, -1, 1), 8, 6))
      loss <- NULL
      with_grad_tape({
        loss <- ag_mse_loss(at$forward(x, causal_mask = causal), y)
      })
      list(loss = loss, params = at$parameters())
    }

    r <- both_paths(build)

    expect_identical(r$got_path, "graph", info = paste("causal =", causal))
    expect_lt(grad_maxdiff(r), 1e-3)
  }
})


test_that("graph backward matches closures for elementwise mul and dropout", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # Both operands tracked: each gradient needs the OTHER operand's forward
  # value, so a rule that used the wrong snapshot would still produce
  # correctly-shaped nonsense. Two distinct parameters catch that.
  build_mul <- function() {
    set.seed(81L)
    w1 <- ag_param(matrix(runif(15, -1, 1), 5, 3))
    w2 <- ag_param(matrix(runif(15, -1, 1), 5, 3))
    y  <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_mul(w1, w2), y)
    })
    list(loss = loss, params = list(w1, w2))
  }
  r <- both_paths(build_mul)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)

  # ag_dropout is one ag_mul against a constant mask, so covering mul covers it.
  # The mask is random, so both paths must build the tape under the same seed --
  # both_paths reseeds inside build().
  build_dp <- function() {
    set.seed(82L)
    l  <- ag_linear(4L, 3L)
    dp <- ag_dropout(0.3)
    x  <- ag_tensor(matrix(runif(20, -1, 1), 4, 5))
    y  <- ag_tensor(matrix(runif(15, -1, 1), 3, 5))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(dp$forward(l$forward(x)), y)
    })
    list(loss = loss, params = list(l$W, l$b))
  }
  r <- both_paths(build_dp)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("ag_batch_norm still falls back to closures", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # Documents a known gap rather than a desired behaviour: batch_norm is a
  # hybrid -- its normalisation is one monolithic closure (grad_out / std) and
  # its gamma/beta go through broadcast helpers that record no op either. Two
  # of its four tape nodes are therefore uncovered. When an emitter is added,
  # this expectation flips to "graph"; until then the fallback is what keeps
  # its gradients right.
  build <- function() {
    set.seed(83L)
    bn <- ag_batch_norm(4L)
    x  <- ag_tensor(matrix(runif(24, -1, 1), 4, 6))
    y  <- ag_tensor(matrix(runif(24, -1, 1), 4, 6))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(bn$forward(x), y)
    })
    list(loss = loss, params = list(bn$gamma, bn$beta))
  }

  r <- both_paths(build)

  expect_match(r$got_path, "^closures \\(")
  expect_lt(grad_maxdiff(r), 1e-6)
})

test_that("the graph path is off unless enabled", {
  # Default state must be the closure path: stage 1 covers three ops, so an
  # unflagged session has to behave exactly as before.
  old <- ag_backward_graph(FALSE)
  on.exit(ag_backward_graph(old %||% FALSE), add = TRUE)

  set.seed(14L)
  w <- ag_param(matrix(runif(4), 2, 2))
  x <- ag_tensor(matrix(c(1, 2), 1, 2))
  y <- ag_tensor(matrix(c(0, 1), 1, 2))
  with_grad_tape({
    loss <- ag_mse_loss(ag_matmul(x, w), y)
  })
  backward(loss)

  expect_identical(ag_backward_path(), "closures")
  expect_false(is.null(w$grad))
})
