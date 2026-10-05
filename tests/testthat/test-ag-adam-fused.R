# Fused device Adam: every parameter's step as ONE graph of native
# GGML_OP_OPT_STEP_ADAMW nodes (.ag_adam_step_device_fused), instead of seven
# per-op launches per parameter. Measured on PPO before the change: the per-op
# step was 448 of the update's 456 launches.
#
# The fused kernel computes the same formula as the host step, but rounds
# differently -- lr * (m * (1/bc1)) against (lr / bc1) * m, about 1 ULP a step
# -- so the checks below use tolerances, not identity. Error grows with the step
# count, and a relative error is meaningless for a weight near zero, hence the
# mixed criterion: |d| <= max(abs, rel * |host|).
#
# Tied weights, unnamed lists and duplicate names are covered by
# test-ag-optimizer-params.R, which runs through the fused path by default.

skip_if_no_gpu <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

ns       <- asNamespace("ggmlR")
launches <- get("ag_launch_count", envir = ns)
h_to_mat <- get(".ag_as_matrix",   envir = ns)
# ag_tensor -> as.matrix() (downloads a device tensor); handle/matrix -> as is
as_mat <- function(x) if (inherits(x, "ag_tensor")) as.matrix(x) else h_to_mat(x)

# Run `code` with GGMLR_AG_ADAM_FUSED set to "1" or "0", restoring it after.
with_fused <- function(on, code) {
  old <- Sys.getenv("GGMLR_AG_ADAM_FUSED", unset = NA)
  Sys.setenv(GGMLR_AG_ADAM_FUSED = if (on) "1" else "0")
  on.exit(if (is.na(old)) Sys.unsetenv("GGMLR_AG_ADAM_FUSED")
          else Sys.setenv(GGMLR_AG_ADAM_FUSED = old), add = TRUE)
  force(code)
}

expect_adam_close <- function(dev, host, rel, abs, info = NULL) {
  d   <- abs(dev - host)
  lim <- pmax(abs, rel * abs(host))
  expect_true(all(d <= lim),
              info = paste(info, sprintf("max|d| = %.3g, worst d/lim = %.3g",
                                         max(d), max(d / lim))))
}

# Parameters of different shapes and gradient scales: a matrix with ordinary
# gradients, a column vector with tiny ones (v ~ 1e-12, close to eps) and a
# 1x1 with huge ones (g^2 ~ 1e6).
shapes <- list(a = c(3L, 4L), b = c(5L, 1L), c = c(1L, 1L))
scales <- c(a = 0.1, b = 1e-6, c = 1e3)
w_init <- lapply(shapes, function(s) matrix(seq(-0.4, 0.4, length.out = prod(s)), s[1], s[2]))
grad_at <- function(nm, k) {
  s <- shapes[[nm]]
  matrix(sin(seq_len(prod(s)) * 0.7 + k) * scales[[nm]], s[1], s[2])
}

# n Adam steps on `device`; snapshots of every weight after the steps in `at`.
# `lr_at(k)` gives the learning rate for step k (a scheduler stand-in).
run_adam <- function(device, n, at, lr_at = function(k) 0.01, ...) {
  ag_device(device)
  ps  <- lapply(w_init, ag_param)
  opt <- optimizer_adam(ps, lr = lr_at(1L), ...)
  snaps <- list()
  for (k in seq_len(n)) {
    opt$lr <- lr_at(k)
    for (nm in names(ps)) ps[[nm]]$grad <- grad_at(nm, k)
    opt$step()
    if (k %in% at) snaps[[as.character(k)]] <- lapply(ps, as_mat)
  }
  list(snaps = snaps, opt = opt, ps = ps)
}

# The kernel takes beta1/beta2 as floats and forms 1.0f - beta itself, so the
# weight of g^2 in v is 1 - fl(beta2), not 1 - beta2. The tolerances below are
# DERIVED from that, not fitted: e(beta2) = |(1 - fl(beta2)) - (1 - beta2)| /
# (1 - beta2) = 9.5e-7 (0.99), 1.29e-5 (0.999), 1.66e-4 (0.9999).
f32 <- function(x) readBin(writeBin(x, raw(), size = 4), "double", size = 4, n = length(x))

test_that("ggml_opt_step_adamw: one CPU step matches the float-beta model", {
  w0 <- c(0.5, -0.25, 1, 0); g <- c(0.1, -0.2, 0.3, 1e-6)
  m0 <- c(0.01, 0, -0.02, 0); v0 <- c(1e-4, 0, 2e-4, 0)
  lr <- 0.01; b1 <- 0.9; eps <- 1e-8; t <- 3L

  for (b2 in c(0.99, 0.999, 0.9999)) {
    info <- paste("beta2 =", b2)
    ctx <- ggml_init(1024 * 1024)
    mk <- function(x) { t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(x)); ggml_set_f32(t, x); t }
    w <- mk(w0); gr <- mk(g); m <- mk(m0); v <- mk(v0)
    hp <- mk(c(lr, b1, b2, eps, 0, 1 / (1 - b1^t), 1 / (1 - b2^t)))
    # No ggml_set_param: the binding sets the flag only while building the node.
    node <- ggml_opt_step_adamw(ctx, w, gr, m, v, hp)
    ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, node))
    mk_ <- ggml_get_f32(m); vk <- ggml_get_f32(v); wk <- ggml_get_f32(w)
    ggml_free(ctx)

    # Model of the kernel: every input rounded to float, the formula in double.
    F <- lapply(list(w0 = w0, g = g, m0 = m0, v0 = v0, lr = lr, b1 = b1, b2 = b2,
                     eps = eps, b1h = 1 / (1 - b1^t), b2h = 1 / (1 - b2^t)), f32)
    m_mod <- F$b1 * F$m0 + F$g * (1 - F$b1)
    v_mod <- F$b2 * F$v0 + F$g^2 * (1 - F$b2)
    w_mod <- F$w0 - F$lr * (m_mod * F$b1h) / (sqrt(v_mod * F$b2h) + F$eps)
    # Against the model only float rounding of a short chain remains.
    near <- function(a, b) all(abs(a - b) <= pmax(1e-6 * abs(b), 1e-30))
    expect_true(near(mk_, m_mod), info = paste(info, "m"))
    expect_true(near(vk, v_mod), info = paste(info, "v"))
    expect_true(near(wk, w_mod), info = paste(info, "w"))

    # Against the exact double formula, v may deviate by the derived amount:
    # e(beta2) on the g^2 term, the rounding of beta2 on the v0 term, plus the
    # same float rounding as above. Nothing larger.
    v_ex  <- b2 * v0 + (1 - b2) * g^2
    e_b2  <- abs((1 - f32(b2)) - (1 - b2)) / (1 - b2)
    bound <- e_b2 * (1 - b2) * g^2 + abs(f32(b2) - b2) * v0 + pmax(1e-6 * abs(v_ex), 1e-30)
    expect_true(all(abs(vk - v_ex) <= bound), info = paste(info, "v vs exact"))
  }

  # The model is not just a looser bound: for beta2 = 0.9999 the kernel's v is
  # closer to 1 - fl(beta2) than to 1 - beta2 (element 3, g = 0.3: the effect
  # is ~100 float ulps of v).
  b2 <- 0.9999
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  mk <- function(x) { t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(x)); ggml_set_f32(t, x); t }
  w <- mk(w0); gr <- mk(g); m <- mk(m0); v <- mk(v0)
  hp <- mk(c(lr, b1, b2, eps, 0, 1 / (1 - b1^t), 1 / (1 - b2^t)))
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, ggml_opt_step_adamw(ctx, w, gr, m, v, hp)))
  vk3 <- ggml_get_f32(v)[3]
  v_mod3 <- f32(b2) * f32(v0[3]) + f32(g[3])^2 * (1 - f32(b2))
  v_ex3  <- b2 * v0[3] + (1 - b2) * g[3]^2
  expect_lt(abs(vk3 - v_mod3), abs(vk3 - v_ex3))
})

test_that("ggml_opt_step_adamw refuses what the kernels would abort on", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  f32 <- function(n = 4L) ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  hp  <- f32(7L)

  w16 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 4L)
  expect_error(ggml_opt_step_adamw(ctx, w16, f32(), f32(), f32(), hp), "F32-only")
  expect_error(ggml_opt_step_adamw(ctx, f32(), f32(5L), f32(), f32(), hp), "same shape")
  expect_error(ggml_opt_step_adamw(ctx, f32(), f32(), f32(), f32(), f32(6L)), "7 elements")
})

test_that("fused step matches the host Adam at steps 1, 2, 10 and 100", {
  skip_if_no_gpu()
  on.exit(ag_device("cpu"), add = TRUE)
  at <- c(1L, 2L, 10L, 100L)
  host <- run_adam("cpu", 100L, at)$snaps
  dev  <- with_fused(TRUE, run_adam("gpu", 100L, at)$snaps)

  for (k in as.character(at)) for (nm in names(shapes)) {
    # Early steps must agree to rounding; later ones are allowed the
    # accumulated per-step ULP drift.
    tol <- if (as.integer(k) <= 2L) c(1e-5, 1e-6) else c(1e-4, 1e-5)
    expect_adam_close(dev[[k]][[nm]], host[[k]][[nm]], rel = tol[1], abs = tol[2],
                      info = paste0("step ", k, ", param ", nm))
  }
})

test_that("fused step is ONE launch per step, per-op path is many", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)

  count_step <- function(on) with_fused(on, {
    ps  <- lapply(w_init, ag_param)
    opt <- optimizer_adam(ps, lr = 0.01)
    for (nm in names(ps)) ps[[nm]]$grad <- grad_at(nm, 1L)
    launches(reset = TRUE)
    opt$step()
    cnt <- launches(reset = TRUE)
    # Counted is not computed: a node Vulkan skips (unbound view) still counts
    # as a launch. Every weight must have moved.
    moved <- vapply(names(ps), function(nm)
      !isTRUE(all.equal(as_mat(ps[[nm]]), w_init[[nm]])), logical(1))
    list(cnt = cnt, moved = moved)
  })

  fused <- count_step(TRUE)
  expect_true(all(fused$moved))
  fused <- fused$cnt
  expect_identical(unname(fused["adam step"]), 1)
  expect_true(is.na(fused["per-op"]) || fused["per-op"] == 0)

  # Guards the switch itself, and makes a silent return to the per-op path
  # visible: there it is several launches per parameter.
  perop <- count_step(FALSE)
  expect_true(all(perop$moved))
  perop <- perop$cnt
  expect_true(is.na(perop["adam step"]))
  expect_gte(unname(perop["per-op"]), 2 * length(shapes))
})

test_that("a learning-rate change mid-run reaches the fused step", {
  skip_if_no_gpu()
  on.exit(ag_device("cpu"), add = TRUE)
  lr_at <- function(k) if (k <= 3L) 0.05 else 0.005
  host <- run_adam("cpu", 6L, 6L, lr_at = lr_at)$snaps[["6"]]
  dev  <- with_fused(TRUE, run_adam("gpu", 6L, 6L, lr_at = lr_at)$snaps[["6"]])
  for (nm in names(shapes))
    expect_adam_close(dev[[nm]], host[[nm]], rel = 1e-4, abs = 1e-5, info = nm)

  # And the change is not a no-op: a constant rate lands elsewhere.
  const <- with_fused(TRUE, run_adam("gpu", 6L, 6L)$snaps[["6"]])
  expect_false(isTRUE(all.equal(const$a, dev$a)))
})

test_that("gradient accumulation feeds the fused step the averaged gradient", {
  skip_if_no_gpu()
  on.exit(ag_device("cpu"), add = TRUE)
  host <- run_adam("cpu", 8L, 8L, accumulate_steps = 2L)$snaps[["8"]]
  dev  <- with_fused(TRUE, run_adam("gpu", 8L, 8L, accumulate_steps = 2L)$snaps[["8"]])
  for (nm in names(shapes))
    expect_adam_close(dev[[nm]], host[[nm]], rel = 1e-4, abs = 1e-5, info = nm)
})

test_that("clip_grad_norm before the fused step matches the host", {
  skip_if_no_gpu()
  on.exit(ag_device("cpu"), add = TRUE)
  clipped_run <- function(device) {
    ag_device(device)
    ps  <- lapply(w_init, ag_param)
    opt <- optimizer_adam(ps, lr = 0.01)
    norms <- numeric(0)
    for (k in 1:5) {
      grads <- new.env()
      for (nm in names(ps)) assign(as.character(ps[[nm]]$id), grad_at(nm, k), envir = grads)
      norms <- c(norms, clip_grad_norm(ps, grads, max_norm = 1))  # c's 1e3 forces a clip
      opt$step(grads)
    }
    list(w = lapply(ps, as_mat), norms = norms)
  }
  host <- clipped_run("cpu")
  dev  <- with_fused(TRUE, clipped_run("gpu"))
  expect_true(all(host$norms > 1))
  for (nm in names(shapes))
    expect_adam_close(dev$w[[nm]], host$w[[nm]], rel = 1e-4, abs = 1e-5, info = nm)
})

# Training through backward() in graph mode, against an INDEPENDENT reference
# (host Adam on the CPU), not only against the other device path: two fast
# paths agreeing proves nothing if they share a defect, and disagreeing does
# not say which one is wrong.
#
# Two regimes, because the fused kernel has one known systematic difference:
# it forms 1 - beta2 in float, so the effective beta2 is 1 - fl(beta2)
# (0.99900001287 for 0.999; the (1 - beta2) term is off by 1.29e-5). On a ReLU
# network that small difference can flip a unit's derivative for a sample near
# zero and move single elements by ~1e-4 (measured on PPO tictactoe). So:
#   * betas EXACT in float (0.875, 1 - 2^-11): element-wise, tight -- catches
#     real defects of either path, independent of beta rounding;
#   * beta2 = 0.999 on a ReLU net: aggregates only (losses, parameter norms),
#     documenting the known deviation without flaking on element flips.
gm_train <- function(device, fused, relu, beta1, beta2) with_fused(fused, {
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  set.seed(11L)
  W1 <- ag_param(matrix(rnorm(8 * 16) * 0.3, 16L, 8L))
  b1 <- ag_param(matrix(0, 16L, 1L))
  W2 <- ag_param(matrix(rnorm(16 * 4) * 0.3, 4L, 16L))
  b2 <- ag_param(matrix(0, 4L, 1L))
  x <- ag_tensor(matrix(rnorm(8 * 32), 8L, 32L))
  y <- ag_tensor(matrix(rnorm(4 * 32) * 0.1, 4L, 32L))
  ps  <- list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  opt <- optimizer_adam(ps, lr = 0.02, beta1 = beta1, beta2 = beta2)
  losses <- numeric(0)
  for (k in 1:20) {
    loss <- NULL
    with_grad_tape({
      h <- ag_add(ag_matmul(W1, x), b1)
      if (relu) h <- ag_relu(h)
      loss <- ag_mse_loss(ag_add(ag_matmul(W2, h), b2), y)
    })
    opt$step(backward(loss))
    opt$zero_grad()
    losses <- c(losses, as.numeric(as_mat(loss)))
  }
  list(w = lapply(ps, as_mat), losses = losses)
})

test_that("graph mode, betas exact in float: both paths match the CPU element-wise", {
  skip_if_no_gpu()
  on.exit(ag_device("cpu"), add = TRUE)
  bt  <- c(0.875, 1 - 2^-11)
  ref <- gm_train("cpu", FALSE, relu = TRUE, bt[1], bt[2])
  expect_lt(ref$losses[20], ref$losses[1])
  for (on in c(TRUE, FALSE)) {
    got  <- gm_train("gpu", on, relu = TRUE, bt[1], bt[2])
    info <- if (on) "fused" else "per-op"
    expect_equal(got$losses, ref$losses, tolerance = 1e-5, info = info)
    for (nm in names(ref$w))
      expect_adam_close(got$w[[nm]], ref$w[[nm]], rel = 1e-5, abs = 1e-6,
                        info = paste(info, nm))
  }
})

test_that("graph mode, beta2 = 0.999 on a ReLU net: aggregates match the CPU", {
  skip_if_no_gpu()
  on.exit(ag_device("cpu"), add = TRUE)
  ref   <- gm_train("cpu", FALSE, relu = TRUE, 0.9, 0.999)
  norms <- function(w) vapply(w, function(m) sqrt(sum(m^2)), 0)
  for (on in c(TRUE, FALSE)) {
    got  <- gm_train("gpu", on, relu = TRUE, 0.9, 0.999)
    info <- if (on) "fused" else "per-op"
    expect_equal(got$losses, ref$losses, tolerance = 1e-3, info = info)
    expect_equal(norms(got$w), norms(ref$w), tolerance = 1e-3, info = info)
  }
})

test_that("a fused step leaves no PARAM flag: later graphs are bit-identical", {
  skip_if_no_gpu()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  set.seed(5L)
  x <- ag_tensor(matrix(rnorm(8 * 6), 8L, 6L))
  y <- ag_tensor(matrix(rnorm(3 * 6), 3L, 6L))
  fwd_bwd <- function(W) {
    loss <- NULL
    with_grad_tape({ loss <- ag_mse_loss(ag_matmul(W, x), y) })
    g <- backward(loss)
    list(loss = as_mat(loss), grad = as_mat(get(as.character(W$id), envir = g)))
  }

  W <- ag_param(matrix(rnorm(3 * 8), 3L, 8L))
  opt <- optimizer_adam(list(W = W), lr = 0.01)
  with_fused(TRUE, opt$step(list2env(setNames(list(fwd_bwd(W)$grad),
                                              as.character(W$id)))))
  after <- fwd_bwd(W)

  # A fresh parameter with the same values never went through the step, so
  # it never carried the flag. A lasting flag would make W a graph node
  # instead of a leaf; the results must not notice the difference.
  fresh <- fwd_bwd(ag_param(as_mat(W)))
  expect_identical(after$loss, fresh$loss)
  expect_identical(after$grad, fresh$grad)
})

test_that("checkpoints move between the per-op and fused paths", {
  skip_if_no_gpu()
  on.exit(ag_device("cpu"), add = TRUE)
  ag_device("gpu")

  # 3 steps on one path, state() -> load_state() into a fresh optimizer on the
  # other path, 3 more steps; against 6 uninterrupted steps on the second path.
  cross <- function(first, second) {
    w0  <- w_init$a
    W   <- ag_param(w0)
    opt <- optimizer_adam(list(W = W), lr = 0.02)
    with_fused(first, for (k in 1:3) { W$grad <- grad_at("a", k); opt$step() })
    st <- opt$state(); w3 <- as_mat(W)

    W2   <- ag_param(w3)
    opt2 <- optimizer_adam(list(W = W2), lr = 0.02)
    opt2$load_state(st)
    with_fused(second, for (k in 4:6) { W2$grad <- grad_at("a", k); opt2$step() })

    Wr   <- ag_param(w0)
    optr <- optimizer_adam(list(W = Wr), lr = 0.02)
    with_fused(second, for (k in 1:6) { Wr$grad <- grad_at("a", k); optr$step() })
    list(resumed = as_mat(W2), straight = as_mat(Wr), t = opt2$t)
  }
  for (dir in list(c(FALSE, TRUE), c(TRUE, FALSE))) {
    r <- cross(dir[1], dir[2])
    info <- paste(if (dir[1]) "fused" else "per-op", "->", if (dir[2]) "fused" else "per-op")
    expect_identical(r$t, 6L, info = info)
    expect_adam_close(r$resumed, r$straight, rel = 1e-5, abs = 1e-6, info = info)
  }
})

test_that("non-F32 weights fall back to the per-op step instead of failing", {
  skip_if_no_gpu()
  ag_device("gpu")
  old <- ag_dtype("f16")
  on.exit({ ag_dtype(old); ag_device("cpu") }, add = TRUE)

  W <- ag_param(matrix(seq(-0.5, 0.5, length.out = 12L), 3L, 4L))
  opt <- optimizer_adam(list(W = W), lr = 0.05)
  w0 <- as_mat(W)
  cnt <- with_fused(TRUE, {
    launches(reset = TRUE)
    for (k in 1:3) { W$grad <- matrix(0.1, 3L, 4L); opt$step() }
    launches(reset = TRUE)
  })
  expect_true(is.na(cnt["adam step"]))     # refused, all-or-nothing
  expect_true(all(is.finite(as_mat(W))))
  expect_false(isTRUE(all.equal(as_mat(W), w0)))
})
