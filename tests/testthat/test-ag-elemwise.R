# Element-wise ops composed from ag_* primitives (R/ag_elemwise.R)

m <- function(v, nr = 2L) matrix(v, nrow = nr)

test_that("forward matches base R", {
  a <- m(c(-2, -0.5, 0.7, 3)); b <- m(c(1, -1, 0.7, 2))
  A <- ag_tensor(a); B <- ag_tensor(b)
  expect_equal(as.matrix(ag_div(A, B)), a / b, tolerance = 1e-6)
  expect_equal(as.matrix(ag_div(A, 4)), a / 4, tolerance = 1e-6)
  expect_equal(as.matrix(ag_sqrt(ag_abs(A))), sqrt(abs(a)), tolerance = 1e-6)
  expect_equal(as.matrix(ag_abs(A)), abs(a), tolerance = 1e-6)
  expect_equal(as.matrix(ag_minimum(A, B)), pmin(a, b), tolerance = 1e-6)
  expect_equal(as.matrix(ag_maximum(A, B)), pmax(a, b), tolerance = 1e-6)
  cond <- a > 0
  expect_equal(as.matrix(ag_where(cond, A, B)), ifelse(cond, a, b), tolerance = 1e-6)
  expect_equal(as.matrix(ag_softplus(A)), log1p(exp(a)), tolerance = 1e-6)
  expect_equal(as.numeric(as.matrix(ag_logsumexp(A))),
               log(colSums(exp(a))), tolerance = 1e-6)
})

test_that("softplus and logsumexp are stable for large inputs", {
  x <- ag_tensor(m(c(-1000, 1000, 1000, 1000)))
  expect_equal(as.numeric(as.matrix(ag_softplus(x))), c(0, 1000, 1000, 1000),
               tolerance = 1e-6)
  expect_equal(as.numeric(as.matrix(ag_logsumexp(x))), c(1000, 1000 + log(2)),
               tolerance = 1e-6)
})

test_that("minimum / maximum broadcast B over A", {
  a <- m(c(-2, -0.5, 0.7, 3)); b <- matrix(0, 1, 1)
  expect_equal(as.matrix(ag_minimum(ag_tensor(a), b)), pmin(a, 0), tolerance = 1e-6)
  expect_equal(as.matrix(ag_maximum(ag_tensor(a), b)), pmax(a, 0), tolerance = 1e-6)
})

test_that("gradcheck: elementwise ops", {
  set.seed(7)
  # values kept away from 0 and from ties, where the ops are not smooth
  W <- ag_param(m(c(-1.3, -0.6, 0.8, 1.7)))
  V <- ag_param(m(c(0.9, -1.1, 1.4, 1.2)))
  # two-input ops get W and V; one-input ops only W (an unused param has no
  # gradient, which ag_gradcheck counts as a failure)
  fns2 <- list(
    function(ins) ag_sum(ag_div(ins$W, ins$V)),
    function(ins) ag_sum(ag_minimum(ins$W, ins$V)),
    function(ins) ag_sum(ag_maximum(ins$W, ins$V)),
    function(ins) ag_sum(ag_where(m(c(1, 0, 0, 1)), ins$W, ins$V))
  )
  fns1 <- list(
    function(ins) ag_sum(ag_div(ins$W, 3)),
    function(ins) ag_sum(ag_sqrt(ag_abs(ins$W))),
    function(ins) ag_sum(ag_softplus(ins$W)),
    function(ins) ag_sum(ag_mul(ag_logsumexp(ins$W), matrix(c(1, 2), 1, 2)))
  )
  for (f in fns2)
    expect_true(ag_gradcheck(fn = f, inputs = list(W = W, V = V),
                             atol = 1e-4, quiet = TRUE))
  for (f in fns1)
    expect_true(ag_gradcheck(fn = f, inputs = list(W = W),
                             atol = 1e-4, quiet = TRUE))
})

# ---- GPU: same values and gradients as the CPU, per-op and in graph mode ----

skip_if_no_gpu_ew <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}
as_m_ew <- function(t) ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(t))

# Forward value and gradient of every param for loss = sum(out * Wt), Wt fixed
# and non-uniform so the incoming gradient differs per entry.
run_ew <- function(f, inputs, dev, graph = FALSE) {
  ag_device(dev)
  old_g <- ag_graph_mode(graph); old_p <- ag_matmul_precision("f32")
  on.exit({ ag_graph_mode(old_g); ag_matmul_precision(old_p); ag_device("cpu") },
          add = TRUE)
  ps <- lapply(inputs, ag_param)
  out <- NULL
  with_grad_tape({
    out  <- do.call(f, ps)
    om   <- as_m_ew(out)
    Wt   <- ag_tensor(matrix(seq_along(om) / 7, nrow(om)))
    loss <- ag_sum(ag_mul(out, Wt))
  })
  resident <- !is.null(ggmlR:::.ag_handle_of(out))
  backward(loss)
  list(val = as_m_ew(out), grads = lapply(ps, ag_grad), resident = resident)
}

test_that("elementwise ops on the GPU match the CPU", {
  skip_if_no_gpu_ew()
  set.seed(71)
  # away from 0 and from ties, where the ops are not smooth
  X <- matrix(sample(c(-1, 1), 15, TRUE) * runif(15, 0.3, 2), 5, 3)
  Y <- X + sample(c(-1, 1), 15, TRUE) * runif(15, 0.2, 1)
  P <- matrix(runif(15, 0.5, 2), 5, 3)        # positive, for div / sqrt
  S <- matrix(0.4, 1, 1)
  MASK <- matrix(rbinom(15, 1, 0.5), 5, 3)
  cases <- list(
    div      = list(f = ag_div, x = list(X, P)),
    div_num  = list(f = function(a) ag_div(a, 3), x = list(X)),
    sqrt     = list(f = ag_sqrt, x = list(P)),
    abs      = list(f = ag_abs, x = list(X)),
    minimum  = list(f = ag_minimum, x = list(X, Y)),
    maximum  = list(f = ag_maximum, x = list(X, Y)),
    min_scal = list(f = function(a) ag_minimum(a, S), x = list(X)),
    where    = list(f = function(a, b) ag_where(MASK, a, b), x = list(X, Y)),
    softplus = list(f = ag_softplus, x = list(X * 5)),
    lse      = list(f = ag_logsumexp, x = list(X * 5)))
  for (nm in names(cases)) {
    cs  <- cases[[nm]]
    ref <- run_ew(cs$f, cs$x, "cpu")
    for (graph in c(FALSE, TRUE)) {
      got  <- run_ew(cs$f, cs$x, "gpu", graph)
      info <- paste(nm, if (graph) "graph" else "per-op")
      expect_true(got$resident, info = info)
      expect_equal(got$val, ref$val, tolerance = 1e-5, ignore_attr = TRUE, info = info)
      for (i in seq_along(ref$grads))
        expect_equal(got$grads[[i]], ref$grads[[i]], tolerance = 1e-5,
                     ignore_attr = TRUE, info = paste(info, "grad", i))
    }
  }
})

test_that("ag_detach stops the gradient and keeps the value", {
  w <- m(c(-1.3, -0.6, 0.8, 1.7))
  W <- ag_param(w)
  D <- ag_detach(W)
  expect_false(D$requires_grad)
  expect_equal(as.matrix(D), w, tolerance = 1e-6)
  # d/dW sum(W * stop(W)) = W, not 2W
  with_grad_tape({ loss <- ag_sum(ag_mul(W, ag_detach(W))) })
  backward(loss)
  expect_equal(ag_grad(W), w, tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("ag_detach on the GPU stays resident and matches the CPU", {
  skip_if_no_gpu_ew()
  set.seed(72)
  X <- matrix(runif(15, -2, 2), 5, 3)
  f <- function(a) ag_mul(a, ag_detach(ag_scale(a, 2)))
  ref <- run_ew(f, list(X), "cpu")
  for (graph in c(FALSE, TRUE)) {
    got  <- run_ew(f, list(X), "gpu", graph)
    info <- if (graph) "graph" else "per-op"
    expect_true(got$resident, info = info)
    expect_equal(got$val, ref$val, tolerance = 1e-5, ignore_attr = TRUE, info = info)
    expect_equal(got$grads[[1]], ref$grads[[1]], tolerance = 1e-5,
                 ignore_attr = TRUE, info = info)
  }
})
