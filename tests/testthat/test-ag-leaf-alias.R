# A computed device handle enters a NEW graph as a leaf alias (same memory,
# op NONE), never as the computed node itself.
#
# Putting the node in directly made ggml_build_forward_expand() walk src[] and
# re-run the whole ancestry from the leaves' CURRENT values. Measured: a per-op
# Adam step copied b's gradient after W's step had moved W and got b's gradient
# for the NEW W (-9%), so GPU Adam with backward() gradients was wrong for every
# parameter after the first.
#
# The class of defect, tested here independently of Adam: a node is computed, a
# leaf it depends on changes, and anything built from the node afterwards must
# see the node's value -- not a re-computation from the changed leaf. The
# direct measure of re-computation is the node count of the consuming graph
# (re-running ancestry adds nodes to the same launch, not launches).

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns         <- asNamespace("ggmlR")
run_op     <- get(".ag_run_op",          envir = ns)
mk_handle  <- get(".ag_handle",          envir = ns)
r_to_gpu   <- get(".ag_r_to_gpu",        envir = ns)
as_mat     <- get(".ag_as_matrix",       envir = ns)
xfer_up    <- get(".ag_xfer_up",         envir = ns)
computed   <- get(".ag_handle_computed", envir = ns)
defer      <- get(".ag_defer",           envir = ns)
launch_env <- get(".ag_launch",          envir = ns)
last_nodes <- function() as.integer(launch_env$last_nodes)

# A persistent leaf, like a weight: overwritten in place below.
leaf <- function(m) mk_handle(r_to_gpu(m, scope = "persistent"), dim(m),
                              scope = "persistent")
scale_op <- function(s) function(ctx, p) ggml_scale(ctx, p[[1L]], s)

W0 <- matrix(seq(0.1, 1.2, by = 0.1), 3L, 4L)
W1 <- W0 * -3                                  # the "after the step" value

test_that("per-op: a child built after the leaf changed uses the node's value", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(FALSE); on.exit(ag_graph_mode(old), add = TRUE)

  W <- leaf(W0)
  h <- run_op(scale_op(2), inputs = list(W), out_shape = dim(W0), resident = TRUE)
  expect_true(computed(h))
  xfer_up(W$ptr, as.numeric(W1), "test")       # the leaf moves

  child <- run_op(scale_op(10), inputs = list(h), out_shape = dim(W0), resident = TRUE)
  # One node: the child's own scale. Two would mean h's scale was re-run.
  expect_identical(last_nodes(), 1L)
  expect_equal(as_mat(h), 2 * W0, tolerance = 1e-6)          # node unchanged
  expect_equal(as_mat(child), 20 * W0, tolerance = 1e-6)     # child from OLD value
})

test_that("graph mode: a pending input is computed once, in its own graph", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(TRUE); on.exit(ag_graph_mode(old), add = TRUE)

  W <- leaf(W0)
  a <- run_op(scale_op(2), inputs = list(W), out_shape = dim(W0), resident = TRUE)
  b <- run_op(scale_op(3), inputs = list(a), out_shape = dim(W0), resident = TRUE)
  expect_false(computed(a))                    # same epoch: must stay a node
  expect_false(computed(b))

  launch_env$by_site <- list()
  expect_equal(as_mat(b), 6 * W0, tolerance = 1e-6)
  expect_identical(launch_env$by_site[["deferred graph"]], 1)
  expect_identical(last_nodes(), 2L)           # a and b, each once
  expect_true(computed(a)); expect_true(computed(b))

  # After that drain, a is finished data: a graph built from it later is a
  # single node, and a changed leaf does not reach it.
  xfer_up(W$ptr, as.numeric(W1), "test")
  c <- run_op(scale_op(10), inputs = list(a), out_shape = dim(W0), resident = TRUE)
  expect_equal(as_mat(c), 20 * W0, tolerance = 1e-6)
  expect_identical(last_nodes(), 1L)
})

test_that("check mode: a handle queued and drained at once counts as computed", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(TRUE, check = TRUE)
  on.exit(ag_graph_mode(old, check = FALSE), add = TRUE)

  W <- leaf(W0)
  a <- run_op(scale_op(2), inputs = list(W), out_shape = dim(W0), resident = TRUE)
  expect_true(computed(a))                     # epoch read before the push
  xfer_up(W$ptr, as.numeric(W1), "test")
  c <- run_op(scale_op(10), inputs = list(a), out_shape = dim(W0), resident = TRUE)
  expect_equal(as_mat(c), 20 * W0, tolerance = 1e-6)
})

test_that("a failed drain does not mark its handles computed", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(TRUE); on.exit(ag_graph_mode(old), add = TRUE)

  W <- leaf(W0)
  a <- run_op(scale_op(2), inputs = list(W), out_shape = dim(W0), resident = TRUE)
  ep <- defer$epoch
  # The compute reports failure as a status, not an error.
  local_mocked_bindings(.ag_graph_compute = function(backend, graph, site) 1L,
                        .package = "ggmlR")
  expect_error(get(".ag_defer_drain", envir = ns)(), "status 1")
  expect_identical(defer$epoch, ep)
  expect_false(computed(a))
})

test_that("epoch bookkeeping of .ag_handle_computed", {
  ep <- defer$epoch
  p <- NULL                                    # pointer is not dereferenced here
  expect_true(computed(mk_handle(p, c(1L, 1L))))
  expect_false(computed(mk_handle(p, c(1L, 1L), pending = TRUE, epoch = ep)))
  expect_true(computed(mk_handle(p, c(1L, 1L), pending = TRUE, epoch = ep - 1L)))
  expect_false(computed(mk_handle(p, c(1L, 1L), pending = TRUE, epoch = NULL)))
})

test_that("a handle from a reset pool is refused, not aliased", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(FALSE); on.exit(ag_graph_mode(old), add = TRUE)

  h <- run_op(scale_op(2), inputs = list(W0), out_shape = dim(W0), resident = TRUE)
  with_grad_tape({ NULL })                     # resets the pass pool
  expect_error(run_op(scale_op(1), inputs = list(h), out_shape = dim(W0),
                      resident = TRUE), "pool")
})

test_that("aliases of views and of views of views read the right bytes", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(FALSE); on.exit(ag_graph_mode(old), add = TRUE)

  W  <- leaf(W0)
  s  <- run_op(scale_op(2), inputs = list(W), out_shape = dim(W0), resident = TRUE)
  # A transpose is a view of s (non-contiguous); a transpose of it is a view
  # of a view, collapsed by ggml to s's buffer.
  t1 <- run_op(function(ctx, p) ggml_transpose(ctx, p[[1L]]), inputs = list(s),
               out_shape = rev(dim(W0)), resident = TRUE)
  t2 <- run_op(function(ctx, p) ggml_transpose(ctx, p[[1L]]), inputs = list(t1),
               out_shape = dim(W0), resident = TRUE)
  xfer_up(W$ptr, as.numeric(W1), "test")       # s has its own memory: unaffected

  c1 <- run_op(function(ctx, p) ggml_cont(ctx, p[[1L]]), inputs = list(t1),
               out_shape = rev(dim(W0)), resident = TRUE)
  expect_identical(last_nodes(), 1L)
  expect_equal(as_mat(c1), t(2 * W0), tolerance = 1e-6)
  c2 <- run_op(function(ctx, p) ggml_cont(ctx, p[[1L]]), inputs = list(t2),
               out_shape = dim(W0), resident = TRUE)
  expect_identical(last_nodes(), 1L)
  expect_equal(as_mat(c2), 2 * W0, tolerance = 1e-6)
})

test_that("backward snapshots are aliases: no forward re-run, pre-step values", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_graph_mode(FALSE); on.exit(ag_graph_mode(old), add = TRUE)
  set.seed(2L)
  x  <- ag_tensor(matrix(rnorm(4 * 6), 4L, 6L))
  y  <- ag_tensor(matrix(rnorm(3 * 6), 3L, 6L))
  Wv <- matrix(rnorm(3 * 4), 3L, 4L)

  grad_of <- function(W, change = NULL) {
    loss <- NULL
    with_grad_tape({ loss <- ag_mse_loss(ag_relu(ag_matmul(W, x)), y) })
    # Move the weight between forward and backward: the backward must use the
    # forward's recorded activations, not re-compute them from the new W.
    if (!is.null(change)) get(".ag_data_write_resident", envir = ns)(W, change)
    g <- backward(loss)
    as_mat(get(as.character(W$id), envir = g))
  }
  ref <- grad_of(ag_param(Wv))
  got <- grad_of(ag_param(Wv), change = Wv * 5)
  # dL/dW = dL/d(relu) * relu' %*% t(x): with the snapshots kept, relu' and the
  # activations are the ones of the forward that actually ran.
  expect_equal(got, ref, tolerance = 1e-6)
})

test_that("graph mode via ag_tensor: pending state survives storage in the tensor", {
  # The public path stores a result's pointer in the ag_tensor and rebuilds the
  # handle from it when the next op takes it as an operand. That rebuild used
  # to drop `pending`, so ag_add() aliased ag_matmul()'s node before it was
  # computed ("source tensor has no backend memory").
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old  <- ag_graph_mode(TRUE);         on.exit(ag_graph_mode(old), add = TRUE)
  oldp <- ag_matmul_precision("f32");  on.exit(ag_matmul_precision(oldp), add = TRUE)
  handle_of <- get(".ag_handle_of", envir = ns)

  set.seed(4L)
  Wv <- matrix(rnorm(3 * 4), 3L, 4L)
  xv <- matrix(rnorm(4 * 6), 4L, 6L)
  bv <- matrix(rnorm(3), 3L, 1L)
  W <- ag_param(Wv); b <- ag_param(bv); x <- ag_tensor(xv)

  y1 <- ag_matmul(W, x)
  expect_false(computed(handle_of(y1)))        # queued, not computed
  y2 <- ag_add(y1, b)
  expect_equal(as.matrix(y2), Wv %*% xv + as.vector(bv), tolerance = 1e-5)
  expect_true(computed(handle_of(y1)))         # its drain has run

  # A consumer built after the weight moved sees y1 as computed, not W' x.
  get(".ag_data_write_resident", envir = ns)(W, Wv * -2)
  y3 <- ag_add(y1, b)
  expect_equal(as.matrix(y3), Wv %*% xv + as.vector(bv), tolerance = 1e-5)
})
