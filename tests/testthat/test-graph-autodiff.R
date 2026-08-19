# Tests for the low-level graph autodiff: building a graph that carries
# gradient storage, adding the backward pass, and reading a gradient back.
#
# This path used to be unreachable from R. ggml_build_forward_expand() goes
# through ggml_new_graph(), which is hardcoded to grads = false, so every graph
# R could produce had cgraph->grads == NULL -- and ggml_build_backward_expand()
# asserts on exactly that. The binding was registered and impossible to call
# successfully for any input.
#
# These checks deliberately use trivial ops (mul, add, sqr) whose derivatives
# are obvious by hand. Whether a particular complicated kernel differentiates
# correctly is a separate question, tested next to that kernel.

test_that("a plain forward graph is rejected by the backward builder", {
  # The failure mode being fixed: without gradient storage ggml asserts, and a
  # failed GGML_ASSERT aborts the R process rather than raising a condition.
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3, 4))
  ggml_set_param(a)
  loss <- ggml_sum(ctx, a)

  plain <- ggml_build_forward_expand(ctx, loss)
  expect_error(ggml_build_backward_expand(ctx, plain),
               "no gradient storage")
})

test_that("d(sum(a * b))/da equals b", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  a_v <- c(1, 2, 3, 4)
  b_v <- c(5, 6, 7, 8)
  ggml_set_f32(a, a_v)
  ggml_set_f32(b, b_v)
  ggml_set_param(a)

  loss  <- ggml_sum(ctx, ggml_mul(ctx, a, b))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)   # seeds d(loss)/d(loss) = 1
  ggml_graph_compute(ctx, graph)

  g <- ggml_graph_get_grad(graph, a)
  expect_false(is.null(g))
  expect_equal(ggml_get_f32(g), b_v, tolerance = 1e-5)
})

test_that("d(sum(a^2))/da equals 2a", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  a_v <- c(-2, -0.5, 0, 1.5, 3)
  ggml_set_f32(a, a_v)
  ggml_set_param(a)

  loss  <- ggml_sum(ctx, ggml_sqr(ctx, a))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)   # seeds d(loss)/d(loss) = 1
  ggml_graph_compute(ctx, graph)

  expect_equal(ggml_get_f32(ggml_graph_get_grad(graph, a)),
               2 * a_v, tolerance = 1e-5)
})

test_that("both operands of an add receive a gradient", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(1, 2, 3))
  ggml_set_f32(b, c(4, 5, 6))
  ggml_set_param(a)
  ggml_set_param(b)

  loss  <- ggml_sum(ctx, ggml_add(ctx, a, b))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)   # seeds d(loss)/d(loss) = 1
  ggml_graph_compute(ctx, graph)

  # d(sum(a + b)) is 1 for every element of both.
  expect_equal(ggml_get_f32(ggml_graph_get_grad(graph, a)), rep(1, 3),
               tolerance = 1e-5)
  expect_equal(ggml_get_f32(ggml_graph_get_grad(graph, b)), rep(1, 3),
               tolerance = 1e-5)
})

test_that("a tensor that is not a param gets no gradient", {
  # ggml_set_param() is what decides; an ordinary input stays untracked.
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(1, 2, 3))
  ggml_set_f32(b, c(4, 5, 6))
  ggml_set_param(a)          # b deliberately left alone

  loss  <- ggml_sum(ctx, ggml_mul(ctx, a, b))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)   # seeds d(loss)/d(loss) = 1
  ggml_graph_compute(ctx, graph)

  expect_false(is.null(ggml_graph_get_grad(graph, a)))
  expect_null(ggml_graph_get_grad(graph, b))
})

test_that("gradients match a numeric gradient through a chain of ops", {
  # Several ops composed, checked against a central difference -- the same
  # standard the SSM kernels are held to, on math simple enough that a
  # disagreement points at the machinery rather than at the formula.
  set.seed(3L)
  a_v <- runif(6, -1, 1)
  b_v <- runif(6, -1, 1)

  # loss(a) = sum((a * b)^2)
  fwd <- function(av) {
    ctx <- ggml_init(16 * 1024 * 1024)
    on.exit(ggml_free(ctx))
    a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(av))
    b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(b_v))
    ggml_set_f32(a, av); ggml_set_f32(b, b_v)
    out <- ggml_sum(ctx, ggml_sqr(ctx, ggml_mul(ctx, a, b)))
    ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, out))
    ggml_get_f32(out)[1]
  }

  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(a_v))
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(b_v))
  ggml_set_f32(a, a_v); ggml_set_f32(b, b_v)
  ggml_set_param(a)
  loss  <- ggml_sum(ctx, ggml_sqr(ctx, ggml_mul(ctx, a, b)))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)   # seeds d(loss)/d(loss) = 1
  ggml_graph_compute(ctx, graph)
  analytic <- ggml_get_f32(ggml_graph_get_grad(graph, a))

  eps <- 1e-3
  numeric_g <- vapply(seq_along(a_v), function(i) {
    hi <- a_v; hi[i] <- hi[i] + eps
    lo <- a_v; lo[i] <- lo[i] - eps
    (fwd(hi) - fwd(lo)) / (2 * eps)
  }, numeric(1))

  expect_equal(analytic, numeric_g, tolerance = 1e-3)
})
