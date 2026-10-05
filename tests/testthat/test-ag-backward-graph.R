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

  # ag_pow records no op description, so the tape must decline AS A WHOLE even
  # though every other node on it is covered. That is the guarantee that keeps a
  # partly covered tape from silently dropping the gradients it cannot emit.
  #
  # This test has had to be rewritten twice, as relu and then softmax gained
  # emitters -- so pick the op deliberately: it must be one that is genuinely
  # outside .AG_BWD_GRAPH_OPS, and it should be re-pointed rather than deleted
  # when ag_pow is eventually covered too (ag_sub was, in the PPO wave).
  build <- function() {
    set.seed(51L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    b <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(ag_add(ag_pow(ag_matmul(x, w), 2), b), y)
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


# ---------------------------------------------------------------------------
# PPO wave: exp, clamp (elemwise_mul), sub, log, sum/mean, and loss_const below
# the root. Closures are the reference, but closures and graph share the tape,
# so log, sub and the branching case are also checked against central finite
# differences of a plain-R loss -- a defect common to both paths shows there.
# ---------------------------------------------------------------------------

# Central differences of scalar f(m) w.r.t. every entry of m, in double on host.
fd_grad <- function(f, m, h = 1e-4) {
  g <- m
  for (i in seq_along(m)) {
    mp <- m; mp[i] <- mp[i] + h
    mm <- m; mm[i] <- mm[i] - h
    g[i] <- (f(mp) - f(mm)) / (2 * h)
  }
  g
}

# Graph-path gradient of one parameter, materialised.
graph_grad <- function(build, k = 1L) {
  old <- ag_backward_graph(TRUE)
  on.exit(ag_backward_graph(old %||% FALSE), add = TRUE)
  b <- build()
  backward(b$loss)
  list(g = .bwd_as_matrix(b$params[[k]]$grad), path = ag_backward_path())
}

test_that("graph backward matches closures for exp, clamp and log", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # clamp: entries exactly on both limits, where the mask must be 0 like the
  # closure's strict inequality.
  wc <- matrix(c(-0.5, 0.5, -0.2, 0.3, -0.9, 0.8, 0.5, -0.5, 0.1, 0.0,
                 0.7, -0.7, 0.25, -0.25, 0.45), 5, 3)
  cases <- list(
    list(nm = "exp",   f = function(w) ag_exp(w),
         w = function() matrix(runif(15, -1, 1), 5, 3)),
    list(nm = "clamp", f = function(w) ag_clamp(w, -0.5, 0.5),
         w = function() wc),
    list(nm = "log",   f = function(w) ag_log(w),
         w = function() matrix(runif(15, 0.5, 2), 5, 3)))

  for (cs in cases) {
    build <- function() {
      set.seed(61L)
      w <- ag_param(cs$w())
      y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
      loss <- NULL
      with_grad_tape({ loss <- ag_mse_loss(cs$f(w), y) })
      list(loss = loss, params = list(w))
    }
    r <- both_paths(build)
    expect_identical(r$got_path, "graph", info = cs$nm)
    expect_lt(grad_maxdiff(r), 1e-3)
  }
})

test_that("graph backward for log matches finite differences", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  set.seed(62L)
  w0 <- matrix(runif(15, 0.5, 2), 5, 3)
  y0 <- matrix(runif(15, -1, 1), 5, 3)
  build <- function() {
    w <- ag_param(w0)
    loss <- NULL
    with_grad_tape({ loss <- ag_mse_loss(ag_log(w), ag_tensor(y0)) })
    list(loss = loss, params = list(w))
  }
  r <- graph_grad(build)
  expect_identical(r$path, "graph")
  expect_equal(r$g, fd_grad(function(m) mean((log(m) - y0)^2), w0),
               tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("graph backward for log at x <= 0 behaves like the closure", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # dx = g / x: Inf at 0, finite negative below. The forward is -Inf / NaN;
  # only the gradient rule is compared, so the loss is a plain sum.
  build <- function() {
    w <- ag_param(matrix(c(0, -1, 2, 0.5, -0.25, 4), 2, 3))
    loss <- NULL
    with_grad_tape({ loss <- ag_sum(ag_log(w)) })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build)
  expect_identical(r$got_path, "graph")
  expect_equal(.bwd_as_matrix(r$got_params[[1]]$grad),
               .bwd_as_matrix(r$ref_params[[1]]$grad), tolerance = 1e-5)
})

test_that("graph backward for sub matches closures and finite differences", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # Asymmetric 5x3: a wrong reduction axis or a lost sign cannot cancel out.
  set.seed(63L)
  x0 <- matrix(runif(15, -1, 1), 5, 3)
  y0 <- matrix(runif(15, -1, 1), 5, 3)
  for (bs in list(c(5L, 3L), c(1L, 3L), c(5L, 1L))) {
    b0 <- matrix(runif(prod(bs), -1, 1), bs[1L], bs[2L])
    build <- function() {
      a <- ag_param(x0)
      b <- ag_param(b0)
      loss <- NULL
      with_grad_tape({ loss <- ag_mse_loss(ag_sub(a, b), ag_tensor(y0)) })
      list(loss = loss, params = list(a, b))
    }
    info <- paste(bs, collapse = "x")
    r <- both_paths(build)
    expect_identical(r$got_path, "graph", info = info)
    expect_lt(grad_maxdiff(r), 1e-3)

    bexp <- function(m) matrix(m, 5L, 3L, byrow = bs[1L] == 1L)
    fd <- fd_grad(function(m) mean((x0 - bexp(m) - y0)^2), b0)
    expect_equal(.bwd_as_matrix(r$got_params[[2]]$grad), fd,
                 tolerance = 1e-3, ignore_attr = TRUE, info = info)
  }
})

test_that("graph backward matches closures for sum and mean on every dim", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  for (red in list(list(f = ag_sum,  nm = "sum"), list(f = ag_mean, nm = "mean"))) {
    for (d in list(NULL, 1L, 2L)) {
      ysh <- if (is.null(d)) c(1L, 1L) else if (d == 1L) c(5L, 1L) else c(1L, 3L)
      build <- function() {
        set.seed(64L)
        w <- ag_param(matrix(runif(15, -1, 1), 5, 3))
        y <- ag_tensor(matrix(runif(prod(ysh), -1, 1), ysh[1L], ysh[2L]))
        loss <- NULL
        with_grad_tape({ loss <- ag_mse_loss(red$f(w, dim = d), y) })
        list(loss = loss, params = list(w))
      }
      info <- paste(red$nm, if (is.null(d)) "all" else d)
      r <- both_paths(build)
      expect_identical(r$got_path, "graph", info = info)
      expect_lt(grad_maxdiff(r), 1e-3)
    }
  }
})

test_that("a loss below the tape root stays on the graph path", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # vf_loss as in PPO: scaled and added, so its loss_const is not the root.
  build <- function() {
    set.seed(65L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      h <- ag_matmul(x, w)
      loss <- ag_add(ag_scale(ag_mse_loss(h, y), 0.5),
                     ag_scale(ag_sum(ag_exp(h)), -0.01))
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("graph backward sums the gradients of a value used twice", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # h feeds exp and both operands of mul: three contributions to dh.
  set.seed(66L)
  w0 <- matrix(runif(12, -0.5, 0.5), 4, 3)
  x0 <- matrix(runif(20, -1, 1), 5, 4)
  y0 <- matrix(runif(15, -1, 1), 5, 3)
  build <- function() {
    w <- ag_param(w0)
    loss <- NULL
    with_grad_tape({
      h <- ag_matmul(ag_tensor(x0), w)
      loss <- ag_mse_loss(ag_add(ag_exp(h), ag_mul(h, h)), ag_tensor(y0))
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)

  fd <- fd_grad(function(m) { h <- x0 %*% m; mean((exp(h) + h * h - y0)^2) }, w0)
  expect_equal(.bwd_as_matrix(r$got_params[[1]]$grad), fd,
               tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("nodes recorded after the loss do not touch its gradients", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # One node consumes the loss, one is unrelated to it but shares w.
  build <- function() {
    set.seed(67L)
    w <- ag_param(matrix(runif(12, -1, 1), 4, 3))
    x <- ag_tensor(matrix(runif(20, -1, 1), 5, 4))
    y <- ag_tensor(matrix(runif(15, -1, 1), 5, 3))
    loss <- NULL
    with_grad_tape({
      loss  <- ag_mse_loss(ag_matmul(x, w), y)
      after <- ag_scale(loss, 3)
      other <- ag_sum(ag_exp(ag_matmul(x, w)))
    })
    list(loss = loss, params = list(w))
  }
  r <- both_paths(build)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)
})

test_that("a non-scalar loss is refused by name", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_backward_graph(TRUE)
  on.exit(ag_backward_graph(old %||% FALSE), add = TRUE)

  w <- ag_param(matrix(runif(6), 2, 3))
  out <- NULL
  with_grad_tape({ out <- ag_exp(w) })
  try(backward(out), silent = TRUE)
  expect_identical(ag_backward_path(), "closures (loss is not a scalar)")
})

test_that("a PPO-shaped loss runs as one graph and matches closures", {
  skip_if_no_gpu()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # The loss of rltoolsR's ppo_update() with a 2-layer net: the tape is the
  # same op mix (softmax, clamp, log, sum(dim), exp, sub, relu, scale, mse
  # below the root). A fallback here is a silent speed loss, hence the path.
  B <- 256L; nobs <- 9L; nh <- 16L; na <- 9L
  set.seed(68L)
  W1 <- matrix(rnorm(nh * nobs, sd = 0.3), nh, nobs); b1 <- matrix(0, nh, 1)
  Wp <- matrix(rnorm(na * nh, sd = 0.3), na, nh);     bp <- matrix(0, na, 1)
  Wv <- matrix(rnorm(nh, sd = 0.3), 1, nh);           bv <- matrix(0, 1, 1)
  obs   <- matrix(runif(nobs * B, -1, 1), nobs, B)
  legal <- matrix(rbinom(na * B, 1, 0.7), na, B); legal[1, ] <- 1
  act   <- apply(legal, 2, function(l) which(l == 1)[1L])
  oh    <- matrix(0, na, B); oh[cbind(act, seq_len(B))] <- 1
  old0  <- matrix(log(runif(B, 0.1, 0.9)), 1)
  adv0  <- matrix(rnorm(B), 1)
  ret0  <- matrix(rnorm(B), 1)
  clip  <- 0.2

  build <- function() {
    ps <- lapply(list(W1, b1, Wp, bp, Wv, bv), ag_param)
    loss <- NULL
    with_grad_tape({
      h      <- ag_relu(ag_add(ag_matmul(ps[[1]], ag_tensor(obs)), ps[[2]]))
      logits <- ag_add(ag_matmul(ps[[3]], h), ps[[4]])
      value  <- ag_add(ag_matmul(ps[[5]], h), ps[[6]])
      p      <- ag_softmax(ag_add(logits, ag_tensor(-30 * (1 - legal))))
      logp   <- ag_log(ag_clamp(p, 1e-8, 1))
      logp_a <- ag_sum(ag_mul(logp, ag_tensor(oh)), dim = 2L)
      ratio  <- ag_exp(ag_sub(logp_a, ag_tensor(old0)))
      A      <- ag_tensor(adv0)
      s1     <- ag_mul(ratio, A)
      s2     <- ag_mul(ag_clamp(ratio, 1 - clip, 1 + clip), A)
      surr   <- ag_sub(s1, ag_relu(ag_sub(s1, s2)))
      pg     <- ag_scale(ag_sum(surr), -1 / B)
      vf     <- ag_mse_loss(value, ag_tensor(ret0))
      ent    <- ag_scale(ag_sum(ag_mul(p, logp)), -1 / B)
      loss   <- ag_add(ag_add(pg, ag_scale(vf, 0.5)), ag_scale(ent, -0.01))
    })
    list(loss = loss, params = ps)
  }

  r <- both_paths(build)
  expect_identical(r$got_path, "graph")
  expect_lt(grad_maxdiff(r), 1e-3)

  # Timing is reported, not asserted: it depends on the device.
  t_of <- function(on) {
    old <- ag_backward_graph(on)
    on.exit(ag_backward_graph(old %||% FALSE), add = TRUE)
    b <- build(); backward(b$loss)                      # warm-up
    system.time(for (i in 1:5) { b <- build(); backward(b$loss) })[["elapsed"]] / 5
  }
  tc <- t_of(FALSE); tg <- t_of(TRUE)
  message(sprintf("PPO loss build+backward: closures %.1f ms, graph %.1f ms (%.2fx)",
                  1000 * tc, 1000 * tg, tc / tg))
})

# ag_mul with a broadcast operand: [m,1], [1,n] and [1,1] against [m,n], the
# small one first and second, both requiring a gradient. Before the rule was
# emitted the whole tape fell back to closures ("closures (mul: broadcast)") --
# PPO's Gaussian policy multiplies [act,B] by exp(-log_std), log_std [act,1].
for (bc in list(c(4L, 1L), c(1L, 3L), c(1L, 1L))) for (small_first in c(FALSE, TRUE)) {
  test_that(sprintf("graph backward handles ag_mul broadcast [%d,%d], small %s",
                    bc[1], bc[2], if (small_first) "first" else "second"), {
    skip_if_no_gpu()
    ag_device("gpu")
    on.exit(ag_device("cpu"), add = TRUE)

    build <- function() {
      set.seed(31L)
      w <- ag_param(matrix(runif(15, -1, 1), 3, 5))
      s <- ag_param(matrix(runif(prod(bc), 0.5, 1.5), bc[1], bc[2]))
      x <- ag_tensor(matrix(runif(20, -1, 1), 4, 5))
      y <- ag_tensor(matrix(runif(12, -1, 1), 4, 3))
      loss <- NULL
      with_grad_tape({
        h <- ag_matmul(x, ag_transpose(w))                    # [4,3]
        p <- if (small_first) ag_mul(s, h) else ag_mul(h, s)
        loss <- ag_mse_loss(p, y)
      })
      list(loss = loss, params = list(w, s))
    }

    r <- both_paths(build)
    expect_identical(r$got_path, "graph")
    expect_identical(.bwd_dim(r$got_params[[2]]$grad), bc)
    expect_lt(grad_maxdiff(r), 1e-3)
  })
}
