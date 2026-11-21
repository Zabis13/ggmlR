# 2-D convolution (R/ag_conv.R): forward and gradients against a direct loop
# over the PyTorch definition, float64, in torch's layout (x [B, C, H, W],
# weight [OC, C, KH, KW], output and its gradient weights G [B, OC, OH, OW])
# and converted here, so the test also pins the layout (image W-H-C, weight
# C-KH-KW). The loop itself is anchored to a few libtorch values (torch 0.16.3,
# float64, same inputs; it matched libtorch to 1e-14 on every value).
# CPU and GPU, f16, gradcheck, layer.

conv_has_gpu <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
conv_devices <- function() c("cpu", if (conv_has_gpu()) "gpu")

conv_cfgs <- list(
  list(name = "3x3, batch 2",                B = 2L, C = 2L, H = 7L,  W = 6L,  OC = 3L, k = c(3L, 3L), s = c(1L, 1L), p = c(0L, 0L), d = c(1L, 1L)),
  list(name = "non-square kernel and image", B = 3L, C = 3L, H = 9L,  W = 11L, OC = 2L, k = c(2L, 4L), s = c(2L, 1L), p = c(1L, 2L), d = c(1L, 1L)),
  list(name = "padding and dilation",        B = 2L, C = 2L, H = 10L, W = 9L,  OC = 2L, k = c(3L, 3L), s = c(1L, 1L), p = c(2L, 2L), d = c(2L, 2L)),
  list(name = "uncovered tail",              B = 2L, C = 1L, H = 11L, W = 10L, OC = 2L, k = c(4L, 3L), s = c(3L, 4L), p = c(0L, 0L), d = c(1L, 1L)),
  list(name = "Nature-CNN conv1, stride 4",  B = 2L, C = 4L, H = 20L, W = 20L, OC = 3L, k = c(8L, 8L), s = c(4L, 4L), p = c(0L, 0L), d = c(1L, 1L)))

# Forward and the gradients of sum(y * G) by the definition, torch layout.
conv_loop <- function(cf, x, w, b, G) {
  B <- cf$B; C <- cf$C; OC <- cf$OC
  OH <- dim(G)[3L]; OW <- dim(G)[4L]
  y <- array(rep(b, each = B), c(B, OC, OH, OW))
  dx <- array(0, dim(x)); dw <- array(0, dim(w))
  for (oh in seq_len(OH)) for (ow in seq_len(OW)) {
    g <- matrix(G[, , oh, ow], B, OC)
    for (kh in seq_len(cf$k[1L])) for (kw in seq_len(cf$k[2L])) {
      ih <- (oh - 1L) * cf$s[1L] - cf$p[1L] + (kh - 1L) * cf$d[1L] + 1L
      iw <- (ow - 1L) * cf$s[2L] - cf$p[2L] + (kw - 1L) * cf$d[2L] + 1L
      if (ih < 1L || ih > cf$H || iw < 1L || iw > cf$W) next
      xs <- matrix(x[, , ih, iw], B, C); wk <- matrix(w[, , kh, kw], OC, C)
      y[, , oh, ow] <- y[, , oh, ow] + xs %*% t(wk)
      dx[, , ih, iw] <- dx[, , ih, iw] + g %*% wk
      dw[, , kh, kw] <- dw[, , kh, kw] + crossprod(g, xs)
    }
  }
  list(y = y, dx = dx, dw = dw, db = apply(G, 2L, sum))
}

# Inputs: the seed and draw order the libtorch anchors were computed with.
conv_ref_gen <- function() {
  set.seed(20261120)
  lapply(conv_cfgs, function(cf) {
    oh <- (c(cf$H, cf$W) + 2L * cf$p - cf$d * (cf$k - 1L) - 1L) %/% cf$s + 1L
    x <- array(runif(cf$B * cf$C * cf$H * cf$W, -1, 1), c(cf$B, cf$C, cf$H, cf$W))
    w <- array(rnorm(cf$OC * cf$C * prod(cf$k), sd = 0.3), c(cf$OC, cf$C, cf$k))
    b <- rnorm(cf$OC)
    G <- array(rnorm(cf$B * cf$OC * prod(oh)), c(cf$B, cf$OC, oh))
    c(cf, list(x = x, w = w, b = b, G = G), conv_loop(cf, x, w, b, G))
  })
}
conv_refs <- conv_ref_gen()
conv_ref <- function() conv_refs

test_that("the loop reference matches libtorch", {
  r <- conv_ref()
  a <- r[[2L]]
  expect_equal(c(a$y[1, 1, 1, 1], a$y[3, 2, 5, 12]), c(-1.67512801840765, 1.72170655333497), tolerance = 1e-12)
  expect_equal(c(a$dx[1, 1, 2, 3], a$dx[3, 3, 9, 11]), c(0.361043210217603, 0.4309000404157), tolerance = 1e-12)
  expect_equal(c(a$dw[1, 1, 1, 1], a$dw[2, 3, 2, 4]), c(6.64869062876027, -4.27913448958988), tolerance = 1e-12)
  expect_equal(a$db, c(-4.38904024213406, -5.80119266434984), tolerance = 1e-12)
  a <- r[[5L]]
  expect_equal(c(a$y[1, 1, 1, 1], a$y[2, 3, 4, 4]), c(3.4698294282794, -1.80612621210221), tolerance = 1e-12)
  expect_equal(c(a$dx[1, 1, 2, 3], a$dx[2, 4, 20, 20]), c(-0.74218650738974, 0.0792149218615202), tolerance = 1e-12)
  expect_equal(c(a$dw[1, 1, 1, 1], a$dw[3, 4, 8, 8]), c(3.4538073430239, 2.80176823337274), tolerance = 1e-12)
  expect_equal(a$db, c(2.16911892623756, 1.07077876764477, 6.37846898327965), tolerance = 1e-12)
})

# torch [B, C, H, W] -> ggmlR [C*H*W, B]; torch [OC, C, KH, KW] -> [OC, C*KH*KW]
conv_img <- function(a) matrix(aperm(a, c(4L, 3L, 2L, 1L)), ncol = dim(a)[1L])
conv_wt  <- function(w) matrix(aperm(w, c(1L, 4L, 3L, 2L)), dim(w)[1L])

# Forward and the gradients of sum(y * G), on the current device. The matmul
# precision defaults to "f32" for the strict comparisons: with "default" an
# fp16/coopmat Vulkan device keeps its matmul tiles in f16 (~4e-4 relative),
# exactly as for ag_linear; that mode has its own test below.
conv_run <- function(r, graph = FALSE, precision = "f32") {
  old <- ggmlR:::ag_backward_graph(graph); old_p <- ag_matmul_precision(precision)
  on.exit({ ggmlR:::ag_backward_graph(old); ag_matmul_precision(old_p) }, add = TRUE)
  x <- ag_param(conv_img(r$x)); w <- ag_param(conv_wt(r$w)); b <- ag_param(matrix(r$b))
  y <- NULL
  with_grad_tape({
    y <- ag_conv2d(x, w, b, in_shape = c(r$C, r$H, r$W), kernel_size = r$k,
                   stride = r$s, padding = r$p, dilation = r$d)
    loss <- ag_sum(ag_mul(y, ag_tensor(conv_img(r$G))))
  })
  backward(loss)
  list(y = as.matrix(y), dx = as.matrix(ag_grad(x)), dw = as.matrix(ag_grad(w)),
       db = as.matrix(ag_grad(b)), path = ggmlR:::.ag_bwd$last_path)
}

conv_expect_ref <- function(got, r, tol, info) {
  expect_equal(got$y,  conv_img(r$y),  tolerance = tol, info = paste(info, "y"))
  expect_equal(got$dx, conv_img(r$dx), tolerance = tol, info = paste(info, "dx"))
  expect_equal(got$dw, conv_wt(r$dw),  tolerance = tol, info = paste(info, "dw"))
  expect_equal(as.numeric(got$db), as.numeric(r$db), tolerance = tol,
               info = paste(info, "db"))
}

test_that("forward and gradients match libtorch (stride, padding, dilation, odd shapes)", {
  on.exit(ag_device("cpu"), add = TRUE)
  for (dev in conv_devices()) {
    ag_device(dev)
    # expect_equal's tolerance is relative: f32 on the GPU, float64 on the CPU
    tol <- if (dev == "gpu") 1e-4 else 1e-10
    for (r in conv_ref()) conv_expect_ref(conv_run(r), r, tol, paste(dev, r$name))
  }
})

test_that("GPU and CPU agree", {
  skip_if_not(conv_has_gpu(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  for (r in conv_ref()) {
    ag_device("cpu"); cpu <- conv_run(r)
    ag_device("gpu"); gpu <- conv_run(r)
    for (nm in c("y", "dx", "dw", "db"))
      expect_equal(gpu[[nm]], cpu[[nm]], tolerance = 1e-4, info = paste(r$name, nm))
  }
})

test_that("the default matmul precision follows ag_matmul_precision(), as ag_linear", {
  skip_if_not(conv_has_gpu(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  ag_device("gpu")
  for (r in conv_ref())
    conv_expect_ref(conv_run(r, precision = "default"), r, 2e-3, paste("default", r$name))
})

test_that("an f16 compute dtype casts to F32 before im2col, both directions", {
  skip_if_not(conv_has_gpu(), "no Vulkan device")
  on.exit({ ag_dtype("f32"); ag_device("cpu") }, add = TRUE)
  r <- conv_ref()[[2L]]
  ag_device("gpu"); ag_dtype("f16")
  got <- conv_run(r)
  for (nm in c("y", "dx", "dw", "db")) expect_true(all(is.finite(got[[nm]])), info = nm)
  conv_expect_ref(got, r, 1e-2, "f16")
})

test_that("gradients match finite differences", {
  ag_device("cpu")
  set.seed(31)
  x <- ag_param(matrix(rnorm(2 * 5 * 4 * 2), 40, 2))
  w <- ag_param(matrix(rnorm(3 * 2 * 2 * 3), 3))
  b <- ag_param(matrix(rnorm(3)))
  G <- matrix(rnorm(3 * 3 * 2 * 2), ncol = 2)                 # OH 3, OW 2
  fn <- function(ins)
    ag_sum(ag_mul(ag_conv2d(ins$x, ins$w, ins$b, in_shape = c(2, 5, 4),
                            kernel_size = c(2, 3), stride = c(2, 1), padding = c(1, 0),
                            dilation = c(1, 1)), G))
  expect_true(ag_gradcheck(fn, list(x = x, w = w, b = b), eps = 1e-6, atol = 1e-6,
                           quiet = TRUE))
})

test_that("without a bias, and with constant (no-grad) operands", {
  ag_device("cpu")
  r <- conv_ref()[[1L]]
  x <- ag_tensor(conv_img(r$x)); w <- ag_param(conv_wt(r$w))
  with_grad_tape({
    y <- ag_conv2d(x, w, in_shape = c(r$C, r$H, r$W), kernel_size = r$k)
    loss <- ag_sum(ag_mul(y, ag_tensor(conv_img(r$G))))
  })
  backward(loss)
  expect_equal(as.matrix(y), conv_img(r$y) - rep(r$b, each = prod(dim(r$y)[3:4])),
               tolerance = 1e-10)
  expect_equal(as.matrix(ag_grad(w)), conv_wt(r$dw), tolerance = 1e-10)
})

test_that("the layer: output size, init, size checks", {
  set.seed(32)
  l <- ag_conv2d_layer(4L, 32L, kernel_size = 8L, in_hw = c(84L, 84L), stride = 4L)
  expect_identical(l$out_shape, c(32L, 20L, 20L))
  expect_identical(l$out_features, 32L * 20L * 20L)
  W <- as.matrix(l$weight)
  expect_identical(dim(W), c(32L, 256L))
  expect_equal(tcrossprod(W) / 2, diag(32), tolerance = 1e-8)   # orthogonal rows, gain sqrt(2)
  expect_equal(as.numeric(as.matrix(l$bias)), rep(0, 32))
  t <- ag_conv2d_layer(3L, 5L, kernel_size = c(3L, 2L), in_hw = c(6L, 7L), init = "torch")
  k <- 1 / sqrt(3 * 3 * 2)
  expect_true(all(abs(unlist(lapply(t$parameters(), as.matrix))) <= k))
  expect_identical(names(t$parameters()), c("weight", "bias"))
  nb <- ag_conv2d_layer(1L, 2L, 3L, c(5L, 5L), bias = FALSE)
  expect_identical(names(nb$parameters()), "weight")
  y <- nb$forward(matrix(runif(25 * 3), 25, 3))
  expect_identical(dim(as.matrix(y)), c(2L * 3L * 3L, 3L))
  expect_error(nb$forward(matrix(0, 24, 1)), "needs 25")
  expect_error(ag_conv2d_layer(1L, 2L, 5L, c(3L, 3L)), "output would be")
  expect_error(ag_conv2d_layer(1L, 2L, 3L, c(5L, 5L), stride = 0L), "stride")
})

test_that("a state dict round trip keeps the layer's output", {
  set.seed(33)
  a <- ag_conv2d_layer(2L, 3L, 3L, c(6L, 5L), padding = 1L)
  b <- ag_conv2d_layer(2L, 3L, 3L, c(6L, 5L), padding = 1L)
  ag_load_state_dict(b, ag_state_dict(a))
  x <- matrix(runif(60 * 2), 60, 2)
  expect_equal(as.matrix(b$forward(x)), as.matrix(a$forward(x)))
})

# ---- graph backward and ag_capture_step ------------------------------------

test_that("the graph backward takes conv2d and matches libtorch and the closures", {
  skip_if_not(conv_has_gpu(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  ag_device("gpu")
  for (r in conv_ref()) {
    gr <- conv_run(r, graph = TRUE)
    expect_false(startsWith(gr$path, "closures"), info = paste(r$name, gr$path))
    conv_expect_ref(gr, r, 1e-4, paste("graph", r$name))
    cl <- conv_run(r, graph = FALSE)
    expect_true(startsWith(cl$path, "closures"), info = r$name)
    for (nm in c("dx", "dw", "db"))
      expect_equal(gr[[nm]], cl[[nm]], tolerance = 1e-5, info = paste(r$name, nm))
  }
})

# conv (2 -> 3, 3x3, padding 1, on 6x5) -> relu -> linear -> mse
cc_data <- function() {
  set.seed(41L)
  list(wc = matrix(rnorm(3 * 18) * 0.3, 3L), wl = matrix(rnorm(2 * 90) * 0.1, 2L),
       x = matrix(runif(60 * 8), 60L, 8L), y = matrix(rnorm(2 * 8) * 0.1, 2L, 8L))
}
cc_model <- function(d) {
  ps <- list(Wc = ag_param(d$wc), bc = ag_param(matrix(0.1, 3L, 1L)),
             Wl = ag_param(d$wl), bl = ag_param(matrix(0, 2L, 1L)))
  fn <- function(x, y) {
    h <- ag_relu(ag_conv2d(x, ps$Wc, ps$bc, in_shape = c(2, 6, 5), kernel_size = 3,
                           padding = 1))
    ag_mse_loss(ag_add(ag_matmul(ps$Wl, h), ps$bl), y)
  }
  list(ps = ps, fn = fn)
}
cc_opt <- function(ps) optimizer_adam(ps, lr = 0.02, beta1 = 0.875, beta2 = 1 - 2^-11)

cc_plain <- function(device, steps = 6L) {
  ag_device(device)
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  d <- cc_data(); m <- cc_model(d); opt <- cc_opt(m$ps)
  x <- ag_tensor(d$x); y <- ag_tensor(d$y)
  losses <- numeric(0)
  for (k in seq_len(steps)) {
    loss <- NULL
    with_grad_tape(loss <- m$fn(x, y))
    opt$step(backward(loss))
    opt$zero_grad()
    losses <- c(losses, as.numeric(as.matrix(loss)))
  }
  list(w = lapply(m$ps, as.matrix), losses = losses)
}

test_that("ag_capture_step trains through conv2d like the ordinary loop", {
  skip_if_not(conv_has_gpu(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  cpu <- cc_plain("cpu")
  ref <- cc_plain("gpu")
  ag_device("gpu")
  ag_local_mode(graph = TRUE, matmul_precision = "f32")
  d <- cc_data(); m <- cc_model(d); opt <- cc_opt(m$ps)
  train <- ag_capture_step(m$fn, optimizer = opt)
  on.exit(ag_capture_free(train), add = TRUE)
  losses <- vapply(1:6, function(k) as.numeric(train(x = d$x, y = d$y)), numeric(1L))
  expect_equal(losses, ref$losses, tolerance = 1e-5)
  expect_equal(losses, cpu$losses, tolerance = 1e-4)
  for (nm in names(ref$w)) {
    expect_equal(as.matrix(m$ps[[nm]]), ref$w[[nm]], tolerance = 1e-5, info = nm)
    expect_equal(as.matrix(m$ps[[nm]]), cpu$w[[nm]], tolerance = 1e-4, info = nm)
  }
  # the batch is fixed by the recording: another size is refused, not re-recorded
  expect_error(train(x = d$x[, 1:4], y = d$y[, 1:4]))
})
