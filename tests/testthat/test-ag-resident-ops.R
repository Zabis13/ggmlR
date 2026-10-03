# Resident (device-side) paths for sub, mul, scale, sum/mean, exp, clamp -- the
# ops a PPO loss is built from. On the GPU each must keep its result on the
# device (so graph mode can fuse it) and must compute exactly what the closure
# path computes on the CPU, forward and backward, including broadcasts and the
# clamp boundary. Checked three ways: CPU, GPU per-op, GPU in graph mode.

skip_if_no_gpu_ro <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}
as_m <- function(t) ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(t))

# Run f(params...) on `dev` (graph mode `graph`); return the forward value and
# the gradient of every param for loss = sum(out * W), W fixed and non-uniform
# so the incoming gradient differs per entry.
run_op <- function(f, inputs, dev, graph = FALSE) {
  ag_device(dev)
  old_g <- ag_graph_mode(graph); old_p <- ag_matmul_precision("f32")
  on.exit({ ag_graph_mode(old_g); ag_matmul_precision(old_p); ag_device("cpu") }, add = TRUE)
  ps <- lapply(inputs, ag_param)
  out <- NULL
  with_grad_tape({
    out  <- do.call(f, ps)
    Wt   <- ag_tensor(matrix(seq_len(length(as_m(out))) / 7, nrow(as_m(out))))
    loss <- ag_sum(ag_mul(out, Wt))
  })
  resident <- !is.null(ggmlR:::.ag_handle_of(out))
  backward(loss)
  list(val = as_m(out), grads = lapply(ps, ag_grad), resident = resident)
}

set.seed(51)
M  <- matrix(rnorm(15), 5, 3)
R1 <- matrix(rnorm(3), 1, 3)          # row vector, broadcast down rows
C1 <- matrix(rnorm(5), 5, 1)          # column vector, broadcast across columns
EDGE <- matrix(c(-0.5, 0.5, -0.2, 0.3, -0.9, 0.8, 0.5, -0.5, 0.1, 0, 0.7, -0.7, 0.25, -0.25, 0.45), 5, 3)

cases <- list(
  sub_same   = list(f = ag_sub, x = list(M, M * 0.3)),
  sub_row    = list(f = ag_sub, x = list(M, R1)),
  sub_col    = list(f = ag_sub, x = list(M, C1)),
  mul_same   = list(f = ag_mul, x = list(M, M * 0.3)),
  mul_row    = list(f = ag_mul, x = list(M, R1)),
  mul_col    = list(f = ag_mul, x = list(M, C1)),
  mul_small1 = list(f = ag_mul, x = list(R1, M)),     # smaller operand FIRST
  mul_small2 = list(f = ag_mul, x = list(C1, M)),
  scale      = list(f = function(x) ag_scale(x, -0.37), x = list(M)),
  sum_all    = list(f = function(x) ag_sum(x), x = list(M)),
  sum_rows   = list(f = function(x) ag_sum(x, dim = 1L), x = list(M)),
  sum_cols   = list(f = function(x) ag_sum(x, dim = 2L), x = list(M)),
  mean_all   = list(f = function(x) ag_mean(x), x = list(M)),
  mean_rows  = list(f = function(x) ag_mean(x, dim = 1L), x = list(M)),
  mean_cols  = list(f = function(x) ag_mean(x, dim = 2L), x = list(M)),
  exp        = list(f = ag_exp, x = list(M)),
  clamp_edge = list(f = function(x) ag_clamp(x, -0.5, 0.5), x = list(EDGE)),
  clamp_inf  = list(f = function(x) ag_clamp(x, -Inf, 0.2), x = list(M)))

test_that("resident ops match the CPU closures, per-op and in graph mode", {
  skip_if_no_gpu_ro()
  for (nm in names(cases)) {
    cs  <- cases[[nm]]
    ref <- run_op(cs$f, cs$x, "cpu")
    for (graph in c(FALSE, TRUE)) {
      got  <- run_op(cs$f, cs$x, "gpu", graph)
      info <- paste(nm, if (graph) "graph" else "per-op")
      expect_true(got$resident, info = info)                     # stayed on the device
      expect_equal(got$val, ref$val, tolerance = 1e-5, ignore_attr = TRUE, info = info)
      for (i in seq_along(ref$grads))
        expect_equal(got$grads[[i]], ref$grads[[i]], tolerance = 1e-5,
                     ignore_attr = TRUE, info = paste(info, "grad", i))
    }
  }
})

test_that("clamp does not overwrite its input inside one graph", {
  skip_if_no_gpu_ro()
  # PPO uses ratio both clamped and unclamped; ggml_clamp is in place, so a
  # fused graph without the copy would hand the clamped values to both.
  ag_device("gpu")
  old <- ag_graph_mode(TRUE)
  on.exit({ ag_graph_mode(old); ag_device("cpu") }, add = TRUE)
  x <- ag_tensor(M * 3)
  y <- ag_add(ag_clamp(x, -0.5, 0.5), ag_scale(x, 0))  # x read after the clamp
  z <- ag_scale(x, 1)
  expect_equal(as_m(z), M * 3, tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(as_m(y), pmin(pmax(M * 3, -0.5), 0.5), tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("a PPO-shaped loss collapses to a few launches in graph mode", {
  skip_if_no_gpu_ro()
  ag_device("gpu")
  old_p <- ag_matmul_precision("f32")
  on.exit({ ag_matmul_precision(old_p); ag_device("cpu") }, add = TRUE)
  set.seed(52)
  B <- 64L; na <- 9L; nh <- 16L
  W1 <- matrix(rnorm(nh * 9, sd = 0.3), nh, 9); Wp <- matrix(rnorm(na * nh, sd = 0.3), na, nh)
  obs <- matrix(runif(9 * B, -1, 1), 9, B)
  legal <- matrix(rbinom(na * B, 1, 0.7), na, B); legal[1, ] <- 1
  act <- apply(legal, 2, function(l) which(l == 1)[1L]) - 1L
  old <- matrix(log(runif(B, 0.1, 0.9)), 1); adv <- matrix(rnorm(B), 1)
  step <- function() {
    ps <- lapply(list(W1, Wp), ag_param)
    with_grad_tape({
      lg   <- ag_matmul(ps[[2]], ag_relu(ag_matmul(ps[[1]], ag_tensor(obs))))
      logp <- ag_log_softmax(ag_add(lg, ag_tensor(-1e4 * (1 - legal))))
      r    <- ag_exp(ag_sub(ag_gather(logp, act), ag_tensor(old)))
      s1   <- ag_mul(r, ag_tensor(adv)); s2 <- ag_mul(ag_clamp(r, 0.8, 1.2), ag_tensor(adv))
      p    <- ag_exp(logp)
      ent  <- ag_scale(ag_mean(ag_sum(ag_mul(p, logp), dim = 2L)), -1)
      loss <- ag_add(ag_scale(ag_mean(ag_sub(s1, ag_relu(ag_sub(s1, s2)))), -1),
                     ag_scale(ent, -0.01))
    })
    backward(loss)
    list(loss = as.numeric(as_m(loss)), g = lapply(ps, ag_grad))
  }
  res <- list()
  for (graph in c(FALSE, TRUE)) {
    prev <- ag_graph_mode(graph)
    ggmlR:::ag_launch_count(reset = TRUE)
    r <- step()
    r$launches <- ggmlR:::ag_launch_count()[["total"]]
    r$path <- ggmlR:::ag_backward_path()
    ag_graph_mode(prev)
    res[[as.character(graph)]] <- r
  }
  off <- res[["FALSE"]]; on <- res[["TRUE"]]
  expect_identical(on$path, "graph")
  expect_equal(on$loss, off$loss, tolerance = 1e-6)
  for (i in 1:2) expect_equal(on$g[[i]], off$g[[i]], tolerance = 1e-6, ignore_attr = TRUE)
  # a whole forward + backward in a handful of graphs, not one per op
  expect_lte(on$launches, 5)
  expect_lt(on$launches * 5, off$launches)
})
