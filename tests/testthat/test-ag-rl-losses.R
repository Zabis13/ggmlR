# RL losses and distribution terms (R/ag_rl_losses.R)

huber_ref <- function(d, delta) ifelse(abs(d) <= delta, 0.5 * d^2,
                                       delta * (abs(d) - 0.5 * delta))

test_that("huber / smooth-L1 match the analytic form, incl. d = 0 and |d| = delta", {
  # d = pred - target covers 0, both boundaries, inside and beyond
  p <- matrix(c(0, 1, -1, 0.3, -2.5, 4), 2, 3); t <- matrix(0, 2, 3)
  P <- ag_tensor(p)
  for (delta in c(1, 0.5, 2)) {
    ref <- huber_ref(p, delta)
    expect_equal(as.matrix(ag_huber_loss(P, t, delta, "none")), ref, tolerance = 1e-6)
    expect_equal(as.numeric(as.matrix(ag_huber_loss(P, t, delta))), mean(ref),
                 tolerance = 1e-6)
    expect_equal(as.numeric(as.matrix(ag_huber_loss(P, t, delta, "sum"))), sum(ref),
                 tolerance = 1e-6)
    expect_equal(as.matrix(ag_smooth_l1_loss(P, t, delta, "none")), ref / delta,
                 tolerance = 1e-6)
  }
  expect_equal(as.matrix(ag_smooth_l1_loss(P, t, 0, "none")), abs(p), tolerance = 1e-6)
  expect_error(ag_huber_loss(P, t, 0), "positive")
  expect_error(ag_smooth_l1_loss(P, t, -1), "non-negative")
})

test_that("huber gradient is clip(d, -delta, delta), also on the boundary", {
  p <- matrix(c(0, 1, -1, 0.3, -2.5, 4), 2, 3)
  for (delta in c(1, 0.5)) {
    P <- ag_param(p)
    with_grad_tape({ loss <- ag_huber_loss(P, matrix(0, 2, 3), delta, "sum") })
    backward(loss)
    expect_equal(ag_grad(P), pmin(pmax(p, -delta), delta), tolerance = 1e-6,
                 ignore_attr = TRUE, info = paste("delta", delta))
  }
})

test_that("categorical entropy: value, uniform case, masked logits stay finite", {
  x <- matrix(c(1, 2, 3, 0.5, -1, 0), 3, 2)
  p <- apply(x, 2, function(v) exp(v) / sum(exp(v)))
  ref <- -colSums(p * log(p))
  h <- as.matrix(ag_entropy(ag_tensor(x)))
  expect_equal(dim(h), c(1L, 2L))
  expect_equal(as.numeric(h), ref, tolerance = 1e-6)
  expect_equal(as.numeric(as.matrix(ag_entropy(ag_tensor(x), "mean"))), mean(ref),
               tolerance = 1e-6)
  expect_equal(as.numeric(as.matrix(ag_entropy(ag_tensor(matrix(0, 4, 1))))), log(4),
               tolerance = 1e-6)
  # a masked action (-1e4) contributes 0, not NaN; large logits do not overflow
  xm <- matrix(c(0, 0, -1e4, 1000, 1000, 1000), 3, 2)
  hm <- as.numeric(as.matrix(ag_entropy(ag_tensor(xm))))
  expect_true(all(is.finite(hm)))
  expect_equal(hm, c(log(2), log(3)), tolerance = 1e-6)
})

test_that("normal log_prob / entropy match the analytic form, both log_std shapes", {
  set.seed(3)
  x  <- matrix(rnorm(6), 3, 2); mu <- matrix(rnorm(6), 3, 2)
  ls <- matrix(rnorm(6, sd = 0.3), 3, 2); ls1 <- matrix(c(-0.2, 0.1, 0.4), 3, 1)
  # full [n, batch] log_std
  expect_equal(as.numeric(as.matrix(ag_normal_log_prob(x, ag_tensor(mu), ag_tensor(ls)))),
               colSums(dnorm(x, mu, exp(ls), log = TRUE)), tolerance = 1e-6)
  # [n, 1] log_std broadcast over the batch
  ls1b <- ls1[, c(1, 1)]
  lp1 <- as.matrix(ag_normal_log_prob(x, ag_tensor(mu), ag_tensor(ls1)))
  expect_equal(dim(lp1), c(1L, 2L))
  expect_equal(as.numeric(lp1), colSums(dnorm(x, mu, exp(ls1b), log = TRUE)),
               tolerance = 1e-6)
  # the -n/2 log(2 pi) constant is included exactly once
  expect_equal(as.numeric(as.matrix(ag_normal_log_prob(matrix(0, 3, 1), matrix(0, 3, 1),
                                                       matrix(0, 3, 1)))),
               -1.5 * log(2 * pi), tolerance = 1e-6)
  # entropy: sum(log_std) + n/2 (1 + log 2 pi); [n, 1] expanded to [1, batch]
  c0 <- 1.5 * (1 + log(2 * pi))
  expect_equal(as.numeric(as.matrix(ag_normal_entropy(ag_tensor(ls)))),
               colSums(ls) + c0, tolerance = 1e-6)
  h1 <- as.matrix(ag_normal_entropy(ag_tensor(ls1), batch = 4L))
  expect_equal(dim(h1), c(1L, 4L))
  expect_equal(as.numeric(h1), rep(sum(ls1) + c0, 4), tolerance = 1e-6)
  expect_error(ag_normal_log_prob(x, mu, matrix(0, 2, 2)), "log_std is")
  expect_error(ag_normal_entropy(ls, batch = 3L), "log_std is")
})

test_that("gradcheck: RL losses", {
  # pred - target kept away from 0 and from +-delta, where Huber is not smooth
  P  <- ag_param(matrix(c(-1.7, -0.4, 0.3, 2.2), 2, 2))
  L  <- ag_param(matrix(c(1.0, -0.5, 0.2, 0.7), 2, 2))     # logits
  MU <- ag_param(matrix(c(0.1, -0.3, 0.5, 0.2), 2, 2))
  LS <- ag_param(matrix(c(-0.2, 0.3, 0.1, -0.4), 2, 2))
  L1 <- ag_param(matrix(c(-0.2, 0.3), 2, 1))
  X  <- matrix(c(0.4, -0.6, 0.9, -0.1), 2, 2)
  w  <- matrix(c(1, 2), 1, 2)
  checks <- list(
    list(function(ins) ag_huber_loss(ins$P, matrix(0, 2, 2), 1), list(P = P)),
    list(function(ins) ag_smooth_l1_loss(ins$P, matrix(0, 2, 2), 0.5, "sum"), list(P = P)),
    list(function(ins) ag_sum(ag_mul(ag_entropy(ins$L), w)), list(L = L)),
    list(function(ins) ag_sum(ag_mul(ag_normal_log_prob(X, ins$MU, ins$LS), w)),
         list(MU = MU, LS = LS)),
    list(function(ins) ag_sum(ag_mul(ag_normal_log_prob(X, ins$MU, ins$L1), w)),
         list(MU = MU, L1 = L1)),
    list(function(ins) ag_sum(ag_mul(ag_normal_entropy(ins$L1, batch = 2L), w)),
         list(L1 = L1)))
  for (i in seq_along(checks))
    expect_true(ag_gradcheck(fn = checks[[i]][[1]], inputs = checks[[i]][[2]],
                             atol = 1e-4, quiet = TRUE), info = paste("check", i))
})

# ---- GPU: same values and gradients as the CPU, per-op and in graph mode ----

skip_if_no_gpu_rl <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}
as_m_rl <- function(t) ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(t))

run_rl <- function(f, inputs, dev, graph = FALSE) {
  ag_device(dev)
  old_g <- ag_graph_mode(graph); old_p <- ag_matmul_precision("f32")
  on.exit({ ag_graph_mode(old_g); ag_matmul_precision(old_p); ag_device("cpu") },
          add = TRUE)
  ps <- lapply(inputs, ag_param)
  out <- NULL
  with_grad_tape({
    out  <- do.call(f, ps)
    om   <- as_m_rl(out)
    Wt   <- ag_tensor(matrix(seq_along(om) / 7, nrow(om)))
    loss <- ag_sum(ag_mul(out, Wt))
  })
  resident <- !is.null(ggmlR:::.ag_handle_of(out))
  backward(loss)
  list(val = as_m_rl(out), grads = lapply(ps, ag_grad), resident = resident)
}

test_that("RL losses on the GPU match the CPU", {
  skip_if_no_gpu_rl()
  set.seed(73)
  X  <- matrix(sample(c(-1, 1), 15, TRUE) * runif(15, 0.2, 3), 5, 3)
  T0 <- matrix(0, 5, 3)
  A  <- matrix(rnorm(15), 5, 3)
  LS <- matrix(rnorm(15, sd = 0.3), 5, 3)
  L1 <- matrix(rnorm(5, sd = 0.3), 5, 1)
  cases <- list(
    huber      = list(f = function(p) ag_huber_loss(p, T0, 1, "none"), x = list(X)),
    huber_mean = list(f = function(p) ag_huber_loss(p, T0, 0.5), x = list(X)),
    smooth_l1  = list(f = function(p) ag_smooth_l1_loss(p, T0, 0.5, "none"), x = list(X)),
    l1         = list(f = function(p) ag_smooth_l1_loss(p, T0, 0, "none"), x = list(X)),
    entropy    = list(f = ag_entropy, x = list(X * 3)),
    logp       = list(f = function(mu, ls) ag_normal_log_prob(A, mu, ls), x = list(X, LS)),
    logp_n1    = list(f = function(mu, ls) ag_normal_log_prob(A, mu, ls), x = list(X, L1)),
    ent_n      = list(f = function(ls) ag_normal_entropy(ls, batch = 3L), x = list(L1)))
  for (nm in names(cases)) {
    cs  <- cases[[nm]]
    ref <- run_rl(cs$f, cs$x, "cpu")
    for (graph in c(FALSE, TRUE)) {
      got  <- run_rl(cs$f, cs$x, "gpu", graph)
      info <- paste(nm, if (graph) "graph" else "per-op")
      expect_true(got$resident, info = info)
      expect_equal(got$val, ref$val, tolerance = 1e-5, ignore_attr = TRUE, info = info)
      for (i in seq_along(ref$grads))
        expect_equal(got$grads[[i]], ref$grads[[i]], tolerance = 1e-5,
                     ignore_attr = TRUE, info = paste(info, "grad", i))
    }
  }
})
