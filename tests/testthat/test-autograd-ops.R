# Tests for new ops and gradcheck
# Every op is tested: (a) forward value, (b) gradcheck passes

source_autograd <- function() {
  # loaded by devtools in package context; source directly in standalone tests
  if (!exists("ag_tensor", mode = "function")) source("R/autograd.R")
}
source_autograd()

# ---- helpers ----------------------------------------------------------------

# Tiny wrapper: reset tape counter between tests so IDs are stable
reset_tape <- function() {
  .ag_tape$nodes   <- list()
  .ag_tape$enabled <- FALSE
}

# ---- ag_sum -----------------------------------------------------------------

test_that("ag_sum(all): forward correct", {
  x <- ag_tensor(matrix(1:6, 2, 3))
  expect_equal(as.numeric(ag_sum(x)$data), 21)
})

test_that("ag_sum(dim=1, keepdim=FALSE): shape and value", {
  x <- ag_tensor(matrix(1:6, 2, 3))  # [[1,3,5],[2,4,6]]
  s <- ag_sum(x, dim = 1L)
  expect_equal(dim(s$data), c(2L, 1L))
  expect_equal(as.numeric(s$data), c(9, 12))
})

test_that("ag_sum(dim=2): shape and value", {
  x <- ag_tensor(matrix(1:6, 2, 3))
  s <- ag_sum(x, dim = 2L)
  expect_equal(dim(s$data), c(1L, 3L))
  expect_equal(as.numeric(s$data), c(3, 7, 11))
})

test_that("gradcheck: ag_sum(all)", {
  set.seed(1)
  W <- ag_param(matrix(runif(6, -1, 1), 2, 3))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_relu(ins$W)),
    inputs = list(W = W), atol = 1e-4, verbose = FALSE, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: ag_sum(dim=1)", {
  set.seed(2)
  W <- ag_param(matrix(runif(6, 0.5, 1.5), 2, 3))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_sum(ins$W, dim = 1L)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_mean ----------------------------------------------------------------

test_that("ag_mean(all): forward", {
  x <- ag_tensor(matrix(1:4, 2, 2))
  expect_equal(as.numeric(ag_mean(x)$data), 2.5)
})

test_that("gradcheck: ag_mean", {
  set.seed(3)
  W <- ag_param(matrix(runif(4, -1, 1), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_mean(ag_sigmoid(ins$W)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_log -----------------------------------------------------------------

test_that("ag_log forward", {
  x   <- ag_tensor(matrix(c(1, exp(1), exp(2)), 1, 3))
  out <- ag_log(x)
  expect_equal(as.numeric(out$data), c(0, 1, 2), tolerance = 1e-6)
})

test_that("gradcheck: ag_log", {
  set.seed(4)
  W <- ag_param(matrix(runif(4, 0.5, 2), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_log(ins$W)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_exp -----------------------------------------------------------------

test_that("ag_exp forward", {
  x   <- ag_tensor(matrix(c(0, 1, 2), 1, 3))
  out <- ag_exp(x)
  expect_equal(as.numeric(out$data), exp(c(0, 1, 2)), tolerance = 1e-6)
})

test_that("gradcheck: ag_exp", {
  set.seed(5)
  W <- ag_param(matrix(runif(4, -0.5, 0.5), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_exp(ins$W)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_reshape -------------------------------------------------------------

test_that("ag_reshape: data unchanged, shape changes", {
  x   <- ag_tensor(matrix(1:6, 2, 3))
  out <- ag_reshape(x, 3L, 2L)
  expect_equal(dim(out$data), c(3L, 2L))
  expect_equal(as.numeric(out$data), as.numeric(x$data))
})

test_that("gradcheck: ag_reshape", {
  set.seed(6)
  W <- ag_param(matrix(runif(6, -1, 1), 2, 3))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_reshape(ins$W, 3L, 2L)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_transpose -----------------------------------------------------------

test_that("ag_transpose forward", {
  x   <- ag_tensor(matrix(1:6, 2, 3))
  out <- ag_transpose(x)
  expect_equal(dim(out$data), c(3L, 2L))
  expect_equal(out$data, t(x$data))
})

test_that("gradcheck: ag_transpose", {
  set.seed(7)
  W <- ag_param(matrix(runif(4, -1, 1), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_relu(ag_transpose(ins$W))),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_clamp ---------------------------------------------------------------

test_that("ag_clamp forward", {
  x   <- ag_param(matrix(c(-2, 0, 3), 1, 3))
  out <- ag_clamp(x, lo = -1, hi = 2)
  expect_equal(as.numeric(out$data), c(-1, 0, 2))
})

test_that("gradcheck: ag_clamp (interior points only)", {
  set.seed(8)
  # Use values safely inside (lo, hi) so grad = 1 everywhere
  W <- ag_param(matrix(runif(4, -0.5, 0.5), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_clamp(ins$W, lo = -1, hi = 1)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_pow -----------------------------------------------------------------

test_that("ag_pow forward", {
  x   <- ag_tensor(matrix(c(2, 3, 4), 1, 3))
  out <- ag_pow(x, 2)
  expect_equal(as.numeric(out$data), c(4, 9, 16))
})

test_that("gradcheck: ag_pow (p=2)", {
  set.seed(9)
  W <- ag_param(matrix(runif(4, 0.5, 2), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_pow(ins$W, 2)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: ag_pow (p=0.5, sqrt)", {
  set.seed(10)
  W <- ag_param(matrix(runif(4, 0.5, 2), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_pow(ins$W, 0.5)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- gradcheck: composite chain ---------------------------------------------

test_that("gradcheck: matmul -> relu -> sum", {
  set.seed(11)
  W <- ag_param(matrix(runif(6, -1, 1), 2, 3))
  x <- ag_tensor(matrix(runif(3, -1, 1), 3, 1))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_relu(ag_matmul(ins$W, x))),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: add with bias broadcast", {
  set.seed(12)
  W <- ag_param(matrix(runif(8, -1, 1), 4, 2))
  b <- ag_param(matrix(runif(4, -0.5, 0.5), 4, 1))
  x <- ag_tensor(matrix(runif(2, -1, 1), 2, 1))
  ok <- ag_gradcheck(
    fn = function(ins) ag_mse_loss(ag_add(ag_matmul(ins$W, x), ins$b),
                                    matrix(0, 4, 1)),
    inputs = list(W = W, b = b), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: sigmoid + generic cross_entropy", {
  # ag_cross_entropy_loss has exact grad -t/p/n, works with any pred (not just softmax)
  set.seed(13)
  W <- ag_param(matrix(runif(6, -0.5, 0.5), 3, 2))
  x <- ag_tensor(matrix(runif(2, -1, 1), 2, 1))
  y <- matrix(c(1, 0, 0), 3, 1)
  ok <- ag_gradcheck(
    fn = function(ins) ag_cross_entropy_loss(ag_sigmoid(ag_matmul(ins$W, x)), y),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: fused softmax_cross_entropy_loss", {
  set.seed(131)
  W <- ag_param(matrix(runif(6, -0.5, 0.5), 3, 2))
  x <- ag_tensor(matrix(runif(2, -1, 1), 2, 1))
  y <- matrix(c(1, 0, 0), 3, 1)
  ok <- ag_gradcheck(
    fn = function(ins) ag_softmax_cross_entropy_loss(ag_matmul(ins$W, x), y),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: log(exp(x)) ~ identity gradient", {
  set.seed(14)
  W <- ag_param(matrix(runif(4, -0.5, 0.5), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_log(ag_exp(ins$W))),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: pow(x,2) vs mul(x,x)", {
  set.seed(15)
  W  <- ag_param(matrix(runif(4, 0.5, 2), 2, 2))
  W2 <- ag_param(W$data + 0)  # same values, separate param
  ok_pow <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_pow(ins$W, 2)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  ok_mul <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_mul(ins$W2, ins$W2)),
    inputs = list(W2 = W2), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok_pow)
  expect_true(ok_mul)
})

# ---- ag_sub -----------------------------------------------------------------

test_that("ag_sub: forward value correct", {
  A <- ag_tensor(matrix(c(5, 3, 2, 1), 2, 2))
  B <- ag_tensor(matrix(c(1, 2, 1, 0), 2, 2))
  C <- ag_sub(A, B)
  expect_equal(C$data, matrix(c(4, 1, 1, 1), 2, 2))
})

test_that("gradcheck: ag_sub", {
  set.seed(20)
  A <- ag_param(matrix(runif(4, -1, 1), 2, 2))
  B <- ag_param(matrix(runif(4, -1, 1), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_mse_loss(ag_sub(ins$A, ins$B), matrix(0, 2, 2)),
    inputs = list(A = A, B = B), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_scale ---------------------------------------------------------------

test_that("ag_scale: forward value correct", {
  x <- ag_tensor(matrix(c(2, 4, 6, 8), 2, 2))
  s <- ag_scale(x, 0.5)
  expect_equal(s$data, matrix(c(1, 2, 3, 4), 2, 2))
})

test_that("gradcheck: ag_scale", {
  set.seed(21)
  W <- ag_param(matrix(runif(4, -1, 1), 2, 2))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_scale(ins$W, 3.0)),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_softmax forward -----------------------------------------------------

test_that("ag_softmax: output sums to 1 per column", {
  set.seed(22)
  x   <- ag_tensor(matrix(runif(3 * 8, -2, 2), 3, 8))
  out <- ag_softmax(x)
  col_sums <- colSums(out$data)
  expect_equal(col_sums, rep(1.0, 8), tolerance = 1e-7)
})

test_that("ag_softmax: all outputs positive", {
  x   <- ag_tensor(matrix(c(-10, 0, 10), 3, 1))
  out <- ag_softmax(x)
  expect_true(all(out$data > 0))
})

test_that("gradcheck: ag_softmax", {
  set.seed(23)
  W <- ag_param(matrix(runif(6, -1, 1), 3, 2))
  x <- ag_tensor(matrix(runif(2, -1, 1), 2, 1))
  ok <- ag_gradcheck(
    fn = function(ins) ag_sum(ag_softmax(ag_matmul(ins$W, x))),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_softmax_cross_entropy_loss forward ----------------------------------

test_that("ag_softmax_cross_entropy_loss: value is positive scalar", {
  set.seed(24)
  logits <- ag_param(matrix(runif(3 * 4, -2, 2), 3, 4))
  y      <- matrix(c(1,0,0, 0,1,0, 0,0,1, 1,0,0), 3, 4)
  with_grad_tape({
    loss <- ag_softmax_cross_entropy_loss(logits, y)
  })
  expect_equal(dim(loss$data), c(1L, 1L))
  expect_gt(as.numeric(loss$data), 0)
})

test_that("ag_softmax_cross_entropy_loss: perfect logits -> low loss", {
  # Large positive logits on correct class -> near-zero loss
  logits <- ag_tensor(matrix(c(10, -10, -10,
                                -10, 10, -10), 3, 2))
  y      <- matrix(c(1, 0, 0,
                     0, 1, 0), 3, 2)
  loss   <- ag_softmax_cross_entropy_loss(logits, y)
  expect_lt(as.numeric(loss$data), 0.01)
})

# ---- print methods ----------------------------------------------------------

test_that("print.ag_tensor works without error", {
  x <- ag_param(matrix(c(1, 2, 3, 4), 2, 2))
  expect_output(print(x), "ag_tensor")
  expect_output(print(x), "requires_grad")
})

test_that("print.ag_optimizer_sgd works without error", {
  w   <- ag_param(matrix(runif(4), 2, 2))
  opt <- optimizer_sgd(list(w = w), lr = 0.05, momentum = 0.9)
  expect_output(print(opt), "SGD")
  expect_output(print(opt), "0.05")
})

test_that("print.ag_optimizer_adam works without error", {
  w   <- ag_param(matrix(runif(4), 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 1e-3)
  expect_output(print(opt), "Adam")
})

test_that("print.ag_sequential works without error", {
  if (!exists("ag_sequential", mode = "function")) source("R/ag_layers.R")
  model <- ag_sequential(
    ag_linear(4L, 8L),
    ag_linear(8L, 2L)
  )
  expect_output(print(model), "ag_sequential")
  expect_output(print(model), "2 layers")
})

test_that("print.ag_dataloader works without error", {
  if (!exists("ag_dataloader", mode = "function")) source("R/ag_training.R")
  dl <- ag_dataloader(matrix(runif(4 * 32), 4, 32), batch_size = 8L)
  expect_output(print(dl), "ag_dataloader")
  expect_output(print(dl), "n=32")
})

test_that("print.lr_scheduler_step works without error", {
  if (!exists("lr_scheduler_step", mode = "function")) source("R/ag_training.R")
  w   <- ag_param(matrix(runif(4), 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_step(opt, step_size = 5L)
  expect_output(print(sch), "lr_scheduler_step")
})

test_that("print.lr_scheduler_cosine works without error", {
  if (!exists("lr_scheduler_cosine", mode = "function")) source("R/ag_training.R")
  w   <- ag_param(matrix(runif(4), 2, 2))
  opt <- optimizer_adam(list(w = w), lr = 0.1)
  sch <- lr_scheduler_cosine(opt, T_max = 10L)
  expect_output(print(sch), "lr_scheduler_cosine")
})

# ---- ag_mul broadcast -------------------------------------------------------

test_that("ag_mul broadcast [d,s] * [1,s]: forward correct", {
  A <- ag_tensor(matrix(c(1,2,3,4,5,6), 3, 2))   # [3,2]
  B <- ag_tensor(matrix(c(2, 3), 1, 2))            # [1,2]
  C <- ag_mul(A, B)
  expect_equal(dim(C$data), c(3L, 2L))
  expect_equal(C$data, matrix(c(2,4,6, 12,15,18), 3, 2))
})

test_that("ag_mul broadcast [d,s] * [d,1]: forward correct", {
  A <- ag_tensor(matrix(c(1,2,3,4,5,6), 3, 2))   # [3,2]
  B <- ag_tensor(matrix(c(10, 20, 30), 3, 1))      # [3,1]
  C <- ag_mul(A, B)
  expect_equal(dim(C$data), c(3L, 2L))
  expect_equal(C$data, matrix(c(10,40,90, 40,100,180), 3, 2))
})

test_that("gradcheck: ag_mul broadcast [d,s] * [1,s]", {
  set.seed(50)
  A <- ag_param(matrix(rnorm(12), 4, 3))
  B <- ag_param(matrix(rnorm(3),  1, 3))
  ok <- ag_gradcheck(
    fn     = function(ins) ag_mse_loss(ag_mul(ins$A, ins$B), matrix(0, 4, 3)),
    inputs = list(A = A, B = B), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck: ag_mul broadcast [d,s] * [d,1]", {
  set.seed(51)
  A <- ag_param(matrix(rnorm(12), 4, 3))
  B <- ag_param(matrix(rnorm(4),  4, 1))
  ok <- ag_gradcheck(
    fn     = function(ins) ag_mse_loss(ag_mul(ins$A, ins$B), matrix(0, 4, 3)),
    inputs = list(A = A, B = B), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})

# ---- ag_softmax_cross_entropy_loss: integer targets -------------------------

test_that("ag_softmax_cross_entropy_loss: integer targets == one-hot", {
  set.seed(60)
  logits <- matrix(rnorm(16 * 8), 16, 8)
  idx    <- c(2L, 0L, 5L, 1L, 3L, 7L, 4L, 6L)   # 0-based

  oh <- matrix(0.0, 16, 8)
  for (i in seq_along(idx)) oh[idx[i] + 1L, i] <- 1.0

  lt1 <- ag_param(logits); lt2 <- ag_param(logits)
  with_grad_tape({ l1 <- ag_softmax_cross_entropy_loss(lt1, idx) })
  with_grad_tape({ l2 <- ag_softmax_cross_entropy_loss(lt2, oh)  })

  expect_equal(as.numeric(l1$data), as.numeric(l2$data), tolerance = 1e-6)
})

test_that("ag_softmax_cross_entropy_loss: integer targets give sensible loss", {
  # Uniform logits -> loss ~ log(vocab_size)
  set.seed(61)
  logits  <- ag_param(matrix(0.0, 8, 4))
  targets <- c(0L, 1L, 2L, 3L)
  with_grad_tape({ loss <- ag_softmax_cross_entropy_loss(logits, targets) })
  expect_equal(as.numeric(loss$data), log(8), tolerance = 1e-5)
})

test_that("gradcheck: ag_softmax_cross_entropy_loss with integer targets", {
  set.seed(62)
  W   <- ag_param(matrix(runif(24, -0.5, 0.5), 8, 3))
  x   <- ag_tensor(matrix(runif(3), 3, 1))
  idx <- 2L   # single position
  ok  <- ag_gradcheck(
    fn     = function(ins) ag_softmax_cross_entropy_loss(ag_matmul(ins$W, x), idx),
    inputs = list(W = W), atol = 1e-4, quiet = TRUE
  )
  expect_true(ok)
})
