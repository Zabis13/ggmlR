# Selection ops (R/ag_select.R). The rule under test is the same on every path:
# R reference (which.max), the ggml CPU kernel, the Vulkan shader, and
# ag_argmax() on both devices -- ties to the FIRST index, NaN never selected,
# a column with nothing above -Inf gives 0.

skip_if_no_gpu_sel <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

# Reference: first maximum, NaN/NA ignored, 0 when nothing beats -Inf.
ref_argmax <- function(m) {
  apply(m, 2, function(col) {
    col[is.na(col)] <- -Inf
    which.max(col) - 1L      # which.max of all -Inf is 1 -> 0
  })
}

# ggml_argmax on a given backend, bypassing the ag_* layer: this is what pins
# the CPU kernel and the shader themselves.
raw_argmax <- function(be, m) {
  cs <- ggml_init(64e6, no_alloc = TRUE)
  on.exit(ggml_free(cs), add = TRUE, after = FALSE)
  x <- ggml_new_tensor_2d(cs, GGML_TYPE_F32, nrow(m), ncol(m))
  buf <- ggml_backend_alloc_ctx_tensors(cs, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(x, as.numeric(m))
  g <- ggml_init(16e6, no_alloc = TRUE)
  on.exit(ggml_free(g), add = TRUE, after = FALSE)
  a <- ggml_argmax(g, x)
  ggml_set_output(a)
  gr <- ggml_build_forward_expand(g, a)
  s <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit(ggml_backend_sched_free(s), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(s, gr)
  ggml_backend_sched_graph_compute(s, gr)
  as.integer(ggml_backend_tensor_get_data(a))
}

# Columns built to catch every tie pattern, for n classes:
#   ties at (1, n-1)  -- the second one sits in a lower-numbered shader thread
#                        whenever n-1 >= subgroup size (the old wrong branch)
#   ties at (0, n-1), ties at every position, distinct random values,
#   ties at (n %/% 2, n %/% 2 + 32, n %/% 2 + 64) where they fit,
#   leading NaN before the true maximum, all NaN, all -Inf, values ~ +-1e4.
tie_matrix <- function(n) {
  set.seed(n)
  cols <- list()
  add <- function(v) cols[[length(cols) + 1L]] <<- v
  base <- function() runif(n, -1, 1)
  v <- base(); v[c(2L, n)[c(2L, n) <= n]] <- 5; add(v)
  v <- base(); v[unique(c(1L, n))] <- 5; add(v)
  add(rep(3, n))
  add(rnorm(n))
  p <- n %/% 2 + 1L + c(0L, 32L, 64L); p <- p[p <= n]
  v <- base(); v[p] <- 7; add(v)
  v <- base(); v[1L] <- NaN; v[n] <- 9; add(v)
  add(rep(NaN, n))
  add(rep(-Inf, n))
  v <- runif(n, -1e4, 1e4); v[n] <- 1e4 + 1; add(v)
  do.call(cbind, cols)
}

sizes <- c(1L, 2L, 31L, 32L, 33L, 64L, 65L, 1000L)

test_that("ag_argmax on the CPU device follows the first-index rule", {
  ag_device("cpu")
  for (n in sizes) {
    m <- tie_matrix(n)
    expect_identical(ag_argmax(m), ref_argmax(m), info = paste("n =", n))
    expect_identical(ag_argmax(ag_tensor(m)), ref_argmax(m), info = paste("n =", n))
  }
})

test_that("the ggml CPU argmax kernel takes the first of equal maxima", {
  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be), add = TRUE)
  for (n in sizes) {
    m <- tie_matrix(n)
    expect_identical(raw_argmax(be, m), ref_argmax(m), info = paste("n =", n))
  }
})

test_that("the Vulkan argmax shader takes the first of equal maxima", {
  skip_if_no_gpu_sel()
  be <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(be), add = TRUE)
  for (n in sizes) {
    m <- tie_matrix(n)
    expect_identical(raw_argmax(be, m), ref_argmax(m), info = paste("n =", n))
  }
})

test_that("ag_argmax on the GPU device matches the reference", {
  skip_if_no_gpu_sel()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  for (n in sizes) {
    m <- tie_matrix(n)
    expect_identical(ag_argmax(ag_tensor(m)), ref_argmax(m), info = paste("n =", n))
  }
  # a single column, and a non-F32 compute dtype (cast before ggml_argmax)
  expect_identical(ag_argmax(ag_tensor(matrix(c(1, 4, 4), 3, 1))), 1L)
  old <- ag_dtype("f16")
  on.exit(ag_dtype(old), add = TRUE)
  # Small integers: exact in f16, so rounding cannot create or break a tie.
  m <- matrix(c(1, 3, 3,  2, 2, 0,  0, 0, 0,  -1e4, 5, -1e4), 3, 4)
  expect_identical(ag_argmax(ag_tensor(m)), c(1L, 0L, 0L, 1L))
})


# ---------------------------------------------------------------------------
# ag_log_softmax
# ---------------------------------------------------------------------------

ref_log_softmax <- function(m) {
  apply(m, 2, function(v) { z <- v - max(v); z - log(sum(exp(z))) })
}

# A masked policy head: legal entries random, illegal ones -1e4 (finite).
lsm_cases <- function() {
  set.seed(91L)
  legal <- matrix(rbinom(9 * 16, 1, 0.6), 9, 16); legal[1, ] <- 1
  list(
    plain  = matrix(rnorm(9 * 16, sd = 3), 9, 16),
    masked = matrix(rnorm(9 * 16), 9, 16) - 1e4 * (1 - legal),
    big    = matrix(runif(9 * 16, -1e4, 1e4), 9, 16),
    one_class  = matrix(rnorm(5), 1, 5),
    one_column = matrix(rnorm(7), 7, 1),
    all_masked = matrix(-1e4, 4, 3))
}

check_lsm <- function(lp, m, tol, info) {
  lp <- ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(lp))
  expect_true(all(is.finite(lp)), info = info)
  expect_equal(lp, ref_log_softmax(m), tolerance = tol, ignore_attr = TRUE, info = info)
  p <- exp(lp)
  expect_equal(colSums(p), rep(1, ncol(m)), tolerance = tol, info = info)
  expect_true(all(is.finite(p * lp)), info = info)      # entropy term, no NaN
}

test_that("ag_log_softmax on the CPU matches the reference", {
  ag_device("cpu")
  cs <- lsm_cases()
  for (nm in names(cs)) check_lsm(ag_log_softmax(ag_tensor(cs[[nm]])), cs[[nm]], 1e-10, nm)
})

test_that("ag_log_softmax on the GPU matches the reference (f32 and f16)", {
  skip_if_no_gpu_sel()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  cs <- lsm_cases()
  for (nm in names(cs)) check_lsm(ag_log_softmax(ag_tensor(cs[[nm]])), cs[[nm]], 1e-4, nm)

  old <- ag_dtype("f16")
  on.exit(ag_dtype(old), add = TRUE)
  for (nm in c("plain", "masked", "one_column"))
    check_lsm(ag_log_softmax(ag_tensor(cs[[nm]])), cs[[nm]], 2e-2, paste("f16", nm))
})

# Loss used for the gradient checks: a weighted sum of log-probabilities, so
# every entry of g differs and colSums(g) != 0 -- a missing p * colSums(g) term
# cannot hide.
lsm_loss <- function(x, wts) ag_sum(ag_mul(ag_log_softmax(x), ag_tensor(wts)))

fd_grad_sel <- function(f, m, h = 1e-5) {
  g <- m
  for (i in seq_along(m)) {
    mp <- m; mp[i] <- mp[i] + h
    mm <- m; mm[i] <- mm[i] - h
    g[i] <- (f(mp) - f(mm)) / (2 * h)
  }
  g
}

lsm_setup <- function() {
  set.seed(92L)
  list(x = matrix(rnorm(20, sd = 2), 5, 4),
       w = matrix(runif(20, -1, 1), 5, 4))
}

test_that("ag_log_softmax backward matches finite differences (CPU)", {
  ag_device("cpu")
  s <- lsm_setup()
  x <- ag_param(s$x)
  with_grad_tape({ l <- lsm_loss(x, s$w) })
  g <- backward(l)
  got <- ggmlR:::.ag_as_matrix(get0(as.character(x$id), envir = g))
  fd <- fd_grad_sel(function(m) sum(ref_log_softmax(m) * s$w), s$x)
  expect_equal(got, fd, tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("ag_log_softmax graph backward matches closures and finite differences", {
  skip_if_no_gpu_sel()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  bg <- ggmlR:::ag_backward_graph
  old <- bg(TRUE)
  on.exit(bg(if (is.null(old)) FALSE else old), add = TRUE)

  s <- lsm_setup()
  run <- function(graph) {
    bg(graph)
    x <- ag_param(s$x)
    with_grad_tape({ l <- lsm_loss(x, s$w) })
    g <- backward(l)
    list(g = ggmlR:::.ag_as_matrix(get0(as.character(x$id), envir = g)),
         path = ggmlR:::ag_backward_path())
  }
  rc <- run(FALSE)
  rg <- run(TRUE)
  expect_identical(rg$path, "graph")
  expect_equal(rg$g, rc$g, tolerance = 1e-4)
  fd <- fd_grad_sel(function(m) sum(ref_log_softmax(m) * s$w), s$x)
  expect_equal(rg$g, fd, tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("ag_log_softmax keeps tiny probabilities exact (spread 50-100)", {
  # The point of not computing log(softmax(x)): exp(-100) is ~3.7e-44, a
  # denormal in f32 that a GPU flushes to zero, so log(softmax) would give -Inf
  # or a value pinned near the denormal floor. log-softmax must stay at ~-100.
  m <- cbind(c(0, -50, -75, -100, -10), c(3, -97, -47, -72, 3))
  ref <- ref_log_softmax(m)                          # double precision
  ag_device("cpu")
  expect_equal(ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(ag_log_softmax(ag_tensor(m)))),
               ref, tolerance = 1e-12, ignore_attr = TRUE)
  skip_if_no_gpu_sel()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  got <- ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(ag_log_softmax(ag_tensor(m))))
  expect_true(all(is.finite(got)))
  expect_lt(max(abs(got - ref)), 1e-4)              # absolute, at |logp| ~ 100
})


# ---------------------------------------------------------------------------
# ag_gather / ag_max
# ---------------------------------------------------------------------------

as_m <- function(t) ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(t))

test_that("ag_gather validates its index", {
  x <- ag_tensor(matrix(1:6 + 0, 3, 2))
  expect_error(ag_gather(x, c(0, NA)),   "NA")
  expect_error(ag_gather(x, c(0, 1.5)),  "whole numbers")
  expect_error(ag_gather(x, c(0, 3)),    "out of range 0..2")     # 1-based slip
  expect_error(ag_gather(x, c(-1, 0)),   "out of range")
  expect_error(ag_gather(x, 0),          "length 1")
  expect_error(ag_gather(x, c("0", "1")), "numeric")
})

for (dev in c("cpu", "gpu")) {
  test_that(paste("ag_gather and ag_max values and gradients,", dev), {
    if (dev == "gpu") skip_if_no_gpu_sel()
    ag_device(dev)
    on.exit(ag_device("cpu"), add = TRUE)

    m   <- matrix(c(1, 5, 5,  -2, 0, 7,  4, 4, 4,  -1e4, 3, -1e4), 3, 4)
    idx <- c(2L, 0L, 1L, 1L)
    expect_equal(as.numeric(as_m(ag_gather(ag_tensor(m), idx))),
                 m[cbind(idx + 1L, 1:4)])
    expect_equal(as.numeric(as_m(ag_max(ag_tensor(m)))), apply(m, 2, max))

    # gather: dL/dx = onehot(idx) * w (L = sum(w * gather))
    w <- matrix(c(2, -3, 0.5, 1), 1, 4)
    x <- ag_param(m)
    with_grad_tape({ l <- ag_sum(ag_mul(ag_gather(x, idx), ag_tensor(w))) })
    g <- as_m(get0(as.character(x$id), envir = backward(l)))
    want <- matrix(0, 3, 4); want[cbind(idx + 1L, 1:4)] <- w
    expect_equal(g, want, tolerance = 1e-6, ignore_attr = TRUE)

    # max: the whole gradient goes to the FIRST of tied maxima, not split
    x2 <- ag_param(m)
    with_grad_tape({ l2 <- ag_sum(ag_max(x2)) })
    g2 <- as_m(get0(as.character(x2$id), envir = backward(l2)))
    first <- matrix(0, 3, 4); first[cbind(c(1L, 2L, 0L, 1L) + 1L, 1:4)] <- 1
    expect_equal(g2, first, tolerance = 1e-6, ignore_attr = TRUE)
  })
}

test_that("a PPO loss written with log_softmax + gather stays on the graph path", {
  skip_if_no_gpu_sel()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)

  # Same loss two ways: the hand-rolled form rltoolsR uses today
  # (softmax -> clamp -> log, onehot mul + sum) and the new functions. Values
  # and gradients must agree; the new form must not fall back to closures.
  B <- 64L; na <- 9L
  set.seed(93L)
  L0    <- matrix(rnorm(na * B), na, B)
  legal <- matrix(rbinom(na * B, 1, 0.7), na, B); legal[1, ] <- 1
  act   <- apply(legal, 2, function(l) which(l == 1)[1L]) - 1L
  oh    <- matrix(0, na, B); oh[cbind(act + 1L, 1:B)] <- 1
  old   <- matrix(log(runif(B, 0.1, 0.9)), 1)
  A     <- matrix(rnorm(B), 1)
  mask  <- -1e4 * (1 - legal)

  run <- function(new_api) {
    lg <- ag_param(L0)
    with_grad_tape({
      z <- ag_add(lg, ag_tensor(mask))
      if (new_api) {
        logp   <- ag_log_softmax(z)
        logp_a <- ag_gather(logp, act)
      } else {
        logp   <- ag_log(ag_clamp(ag_softmax(z), 1e-8, 1))
        logp_a <- ag_sum(ag_mul(logp, ag_tensor(oh)), dim = 2L)
      }
      ratio <- ag_exp(ag_sub(logp_a, ag_tensor(old)))
      loss  <- ag_scale(ag_sum(ag_mul(ratio, ag_tensor(A))), -1 / B)
    })
    g <- backward(loss)
    list(loss = as.numeric(as_m(loss)), path = ggmlR:::ag_backward_path(),
         g = as_m(get0(as.character(lg$id), envir = g)))
  }
  r_old <- run(FALSE)
  r_new <- run(TRUE)
  expect_identical(r_new$path, "graph")
  expect_equal(r_new$loss, r_old$loss, tolerance = 1e-4)
  expect_equal(r_new$g, r_old$g, tolerance = 1e-4)
})
