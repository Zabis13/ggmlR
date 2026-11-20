# Recurrent layers (R/ag_rnn.R): parity with PyTorch on fixed weights (gate
# order), sequence layer vs a loop of cell calls, episode-start masks,
# gradients through time, initialisation, state dicts. CPU and GPU.

rnn_devices <- function() c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() > 0L) "gpu")

# The fixed weights and inputs of the torch reference (in 3, hidden 2, B 2, T 3)
rnn_fill <- function(r, c, k) matrix(0.5 * sin(k * seq_len(r * c)), r, c)
rnn_ref_weights <- function(G, H = 2L, inp = 3L)
  list(weight_ih = rnn_fill(G * H, inp, 1.3), weight_hh = rnn_fill(G * H, H, 1.7),
       bias_ih = rnn_fill(G * H, 1, 0.9), bias_hh = rnn_fill(G * H, 1, 1.1))
rnn_ref_xs <- function() lapply(1:3, function(t) rnn_fill(3L, 2L, 0.7 + t))

# Outputs of libtorch in float64 (R torch: aten lstm_cell / gru_cell, the ops
# nn.LSTM / nn.GRU step with, on the weights above, h0 = rnn_fill(2, 2, 2.1),
# c0 = rnn_fill(2, 2, 2.9)); per step a [2, 2] matrix, column-major
torch_lstm_out <- c(-0.258069777105276, 0.0593352908887937, -0.269664026960704,
                    -0.28002291115209, -0.287806955075623, 0.0101741604545775,
                    -0.424034690913231, -0.248623878393122, -0.517030433868715,
                    -0.165665345075415, -0.47394853720904, 0.0735491250582585)
torch_lstm_c <- c(-0.907530625909279, -0.234471210268482, -0.909620744935996,
                  0.136675982140622)
torch_gru_out <- c(-0.150085190119018, -0.181329767894821, -0.253762050968135,
                   0.258577470607834, -0.387152077353313, -0.339760804817866,
                   -0.387967770750058, -0.0669981539229034, -0.466316346293651,
                   -0.226043945578586, -0.531401043615298, 0.0165054641824168)

rnn_set <- function(model, w) {
  sd <- ag_state_dict(model)
  sd$parameters <- w
  ag_load_state_dict(model, sd)
  model
}
rnn_mat <- function(x) as.matrix(x)

test_that("LSTM and GRU match PyTorch on fixed weights (gate order, both biases)", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rnn_devices()) {
    ag_device(dev)
    tol <- if (dev == "gpu") 1e-4 else 1e-12
    xs <- lapply(rnn_ref_xs(), ag_tensor)
    lstm <- rnn_set(ag_lstm(3L, 2L), rnn_ref_weights(4L))
    r <- lstm$forward(xs, state = list(h = rnn_fill(2, 2, 2.1), c = rnn_fill(2, 2, 2.9)))
    expect_equal(unlist(lapply(r$outputs, rnn_mat)), torch_lstm_out, tolerance = tol, info = dev)
    expect_equal(as.numeric(rnn_mat(r$state$c)), torch_lstm_c, tolerance = tol, info = dev)
    # one step: the first column block
    s1 <- lstm$cell$forward(xs[[1]], list(h = rnn_fill(2, 2, 2.1), c = rnn_fill(2, 2, 2.9)))
    expect_equal(as.numeric(rnn_mat(s1$h)), torch_lstm_out[1:4], tolerance = tol, info = dev)

    gru <- rnn_set(ag_gru(3L, 2L), rnn_ref_weights(3L))
    r <- gru$forward(xs, state = list(h = rnn_fill(2, 2, 2.1)))
    expect_equal(unlist(lapply(r$outputs, rnn_mat)), torch_gru_out, tolerance = tol, info = dev)
    s1 <- gru$cell$forward(xs[[1]], list(h = rnn_fill(2, 2, 2.1)))
    expect_equal(as.numeric(rnn_mat(s1$h)), torch_gru_out[1:4], tolerance = tol, info = dev)
  }
})

test_that("the sequence layer equals a loop of cell calls", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rnn_devices()) {
    ag_device(dev)
    for (mk in list(ag_lstm, ag_gru)) {
      set.seed(1)
      layer <- mk(3L, 4L)
      xs <- lapply(1:5, function(t) ag_tensor(matrix(rnorm(3 * 2), 3, 2)))
      starts <- matrix(0, 5, 2); starts[3, 2] <- 1
      r <- layer$forward(xs, starts = starts)
      s <- NULL
      for (t in 1:5) {
        s <- layer$cell$forward(xs[[t]], s, starts[t, ])
        expect_equal(rnn_mat(r$outputs[[t]]), rnn_mat(s$h), tolerance = 1e-6,
                     info = paste(dev, layer$kind, t))
      }
      expect_equal(rnn_mat(r$state$h), rnn_mat(s$h), tolerance = 1e-6)
    }
  }
})

test_that("starts zero the state at the episode boundary, first step and mid-sequence", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in rnn_devices()) {
    ag_device(dev)
    tol <- if (dev == "gpu") 1e-5 else 1e-10
    for (mk in list(ag_lstm, ag_gru)) {
      set.seed(2)
      layer <- mk(3L, 4L)
      xs <- lapply(1:6, function(t) ag_tensor(matrix(rnorm(3 * 3), 3, 3)))
      s0 <- lapply(if (layer$kind == "lstm") c(h = 1, c = 1) else c(h = 1),
                   function(v) matrix(rnorm(4 * 3), 4, 3))
      info <- paste(dev, layer$kind)
      # first step: starts = 1 means the given state is ignored for that copy
      st <- matrix(0, 6, 3); st[1, 2] <- 1
      a <- layer$forward(xs, state = s0, starts = st)
      fresh <- layer$forward(xs)                       # zero state for everyone
      kept  <- layer$forward(xs, state = s0)           # no reset
      for (t in 1:6) {
        expect_equal(rnn_mat(a$outputs[[t]])[, 2], rnn_mat(fresh$outputs[[t]])[, 2],
                     tolerance = tol, info = info)
        expect_equal(rnn_mat(a$outputs[[t]])[, -2], rnn_mat(kept$outputs[[t]])[, -2],
                     tolerance = tol, info = info)
      }
      # mid-sequence: copy 3 starts a new episode at step 4 -- from there it
      # equals a fresh run on steps 4..6; before, and the other copies, untouched
      st <- matrix(0, 6, 3); st[4, 3] <- 1
      b <- layer$forward(xs, state = s0, starts = st)
      tail_run <- layer$forward(xs[4:6])
      for (t in 1:3)
        expect_equal(rnn_mat(b$outputs[[t]]), rnn_mat(kept$outputs[[t]]), tolerance = tol,
                     info = info)
      for (t in 4:6) {
        expect_equal(rnn_mat(b$outputs[[t]])[, 3], rnn_mat(tail_run$outputs[[t - 3]])[, 3],
                     tolerance = tol, info = info)
        expect_equal(rnn_mat(b$outputs[[t]])[, 1:2], rnn_mat(kept$outputs[[t]])[, 1:2],
                     tolerance = tol, info = info)
      }
    }
  }
})

test_that("gradients through time match finite differences", {
  ag_device("cpu")
  for (mk in list(ag_lstm, ag_gru)) {
    set.seed(3)
    layer <- mk(2L, 2L, init = "torch")
    x1 <- ag_param(matrix(rnorm(4), 2, 2)); x2 <- ag_param(matrix(rnorm(4), 2, 2))
    h0 <- ag_param(matrix(rnorm(4), 2, 2))
    tgt <- matrix(c(0.3, -0.2, 0.1, 0.4), 2, 2)
    fn <- function(ins) {
      s0 <- if (layer$kind == "lstm") list(h = ins$h0, c = ins$h0) else list(h = ins$h0)
      r <- layer$forward(list(ins$x1, ins$x2), state = s0, starts = matrix(c(0, 0, 0, 1), 2, 2))
      ag_add(ag_mse_loss(r$outputs[[1]], tgt), ag_mse_loss(r$outputs[[2]], tgt))
    }
    ins <- c(layer$parameters(), list(x1 = x1, x2 = x2, h0 = h0))
    expect_true(ag_gradcheck(fn, ins, eps = 1e-6, atol = 1e-6, quiet = TRUE), info = layer$kind)
  }
})

test_that("initialisation: rl (orthogonal blocks, forget bias 1) and torch ranges", {
  set.seed(4)
  l <- ag_lstm_cell(3L, 5L)
  w <- lapply(l$parameters(), rnn_mat)
  expect_equal(dim(w$weight_ih), c(20L, 3L))
  expect_equal(dim(w$weight_hh), c(20L, 5L))
  for (g in 0:3) {
    blk <- w$weight_hh[g * 5 + 1:5, ]
    expect_equal(crossprod(blk), diag(5), tolerance = 1e-6)
  }
  expect_equal(as.numeric(w$bias_ih), c(rep(0, 5), rep(1, 5), rep(0, 10)))
  expect_equal(as.numeric(w$bias_hh), rep(0, 20))
  g <- lapply(ag_gru_cell(3L, 5L)$parameters(), rnn_mat)
  expect_equal(as.numeric(g$bias_ih), rep(0, 15))
  t <- lapply(ag_gru_cell(3L, 4L, init = "torch")$parameters(), rnn_mat)
  expect_true(all(abs(unlist(t)) <= 1 / sqrt(4)))
  expect_true(all(unlist(t) != 0))
})

test_that("parameters() carry the torch names and survive a state dict round trip", {
  set.seed(5)
  a <- ag_lstm(3L, 2L)
  expect_identical(names(a$parameters()), c("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
  expect_identical(a$parameters()$weight_ih, a$cell$weight_ih)   # one set of weights
  b <- ag_lstm(3L, 2L)
  ag_load_state_dict(b, ag_state_dict(a))
  xs <- lapply(1:2, function(t) matrix(rnorm(6), 3, 2))
  expect_equal(rnn_mat(b$forward(xs)$state$h), rnn_mat(a$forward(xs)$state$h))
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f), add = TRUE)
  ag_save_model(a, f)
  c2 <- ag_load_model(f, model_fn = function() ag_lstm(3L, 2L))
  expect_equal(rnn_mat(c2$forward(xs)$state$h), rnn_mat(a$forward(xs)$state$h))
})

test_that("bad input is refused", {
  expect_error(ag_lstm_cell(0L, 2L), "positive whole")
  cell <- ag_gru_cell(3L, 2L)
  expect_error(cell$forward(matrix(0, 4, 2)), "expected input_size = 3")
  expect_error(cell$forward(matrix(0, 3, 2), list(c = matrix(0, 2, 2))), "list\\(h\\)")
  expect_error(cell$forward(matrix(0, 3, 2), list(h = matrix(0, 2, 2)), starts = c(1, 2)),
               "0/1")
  layer <- ag_lstm(3L, 2L)
  expect_error(layer$forward(matrix(0, 3, 2)), "list")
  expect_error(layer$forward(list(matrix(0, 3, 2)), starts = matrix(0, 2, 2)), "one row per step")
})

# ---- starts as a tensor (inputs of ag_capture_step) -------------------------

# outputs and gradients of a short LSTM run with starts given as `as`
rnn_starts_run <- function(as_tensor, graph) {
  old <- ggmlR:::ag_backward_graph(graph)
  on.exit(ggmlR:::ag_backward_graph(old), add = TRUE)
  set.seed(51)
  layer <- ag_lstm(3L, 4L, init = "torch")
  xs <- lapply(1:4, function(t) ag_param(matrix(rnorm(6), 3, 2)))
  h0 <- ag_param(matrix(rnorm(8), 4, 2)); c0 <- ag_param(matrix(rnorm(8), 4, 2))
  st <- matrix(c(0, 0, 1, 0, 0, 1, 0, 0), 4, 2)          # copy 1 at step 3, copy 2 at 2
  if (as_tensor) st <- ag_tensor(st)
  tgt <- matrix(0.1, 4, 2)
  with_grad_tape({
    r <- layer$forward(xs, state = list(h = h0, c = c0), starts = st)
    loss <- ag_add(ag_mse_loss(r$outputs[[4]], tgt), ag_mse_loss(r$outputs[[2]], tgt))
  })
  backward(loss)
  list(h = lapply(r$outputs, as.matrix),
       g = lapply(c(layer$parameters(), list(h0 = h0, c0 = c0), xs),
                  function(p) as.matrix(ag_grad(p))),
       path = ggmlR:::.ag_bwd$last_path)
}

test_that("starts as a [T, B] tensor equals starts as a matrix (CPU, GPU, graph backward)", {
  on.exit({ ag_device("cpu"); ag_matmul_precision("default") }, add = TRUE)
  for (dev in rnn_devices()) for (graph in c(FALSE, TRUE)) {
    ag_device(dev); ag_matmul_precision("f32")
    m <- rnn_starts_run(FALSE, graph)
    t <- rnn_starts_run(TRUE, graph)
    info <- paste(dev, graph)
    if (dev == "gpu" && graph)
      expect_false(startsWith(t$path, "closures"), info = paste(info, t$path))
    expect_equal(t$h, m$h, tolerance = 1e-6, info = info)
    expect_equal(t$g, m$g, tolerance = 1e-5, info = info)
  }
  ag_device("cpu")
  cell <- ag_lstm_cell(3L, 2L)
  expect_error(cell$forward(matrix(0, 3, 2), list(h = matrix(0, 2, 2), c = matrix(0, 2, 2)),
                            starts = ag_tensor(matrix(0, 2, 2))), "must be \\[1, 2\\]")
})

test_that("ag_capture_step trains an LSTM whose starts are an input", {
  on.exit(ag_device("cpu"), add = TRUE)
  run <- function(dev, captured) {
    ag_device(dev)
    ag_local_mode(graph = TRUE, matmul_precision = "f32")
    set.seed(52)
    layer <- ag_lstm(3L, 4L, init = "torch")
    head <- ag_linear(4L, 1L)
    ps <- c(layer$parameters(), list(hW = head$W, hb = head$b))
    opt <- optimizer_adam(ps, lr = 0.01)
    d <- list(x = matrix(rnorm(3 * 3 * 2), 9, 2), st = matrix(c(1, 0, 1, 1, 0, 0), 3, 2),
              h = matrix(0.2, 4, 2), c = matrix(-0.1, 4, 2), y = matrix(c(0.5, -0.5), 1, 2))
    fn <- function(x, st, h, c, y) {
      xs <- lapply(0:2, function(t) ag_slice_rows(x, 3L * t, 3L))
      r <- layer$forward(xs, state = list(h = h, c = c), starts = st)
      ag_mse_loss(head$forward(r$outputs[[3]]), y)
    }
    losses <- numeric(0)
    if (captured) {
      train <- ag_capture_step(fn, optimizer = opt)
      on.exit(ag_capture_free(train), add = TRUE)
      for (k in 1:4) losses <- c(losses, as.numeric(do.call(train, d)))
    } else {
      args <- lapply(d, ag_tensor)
      for (k in 1:4) {
        loss <- NULL
        with_grad_tape(loss <- do.call(fn, args))
        opt$step(backward(loss)); opt$zero_grad()
        losses <- c(losses, as.numeric(as.matrix(loss)))
      }
    }
    list(losses = losses, w = lapply(ps, as.matrix))
  }
  for (dev in rnn_devices()) {
    ref <- run(dev, FALSE)
    got <- run(dev, TRUE)
    expect_equal(got$losses, ref$losses, tolerance = 1e-5, info = dev)
    expect_equal(got$w, ref$w, tolerance = 1e-5, info = dev)
  }
})
