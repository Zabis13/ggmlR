# ag_conv_transpose2d(): PyTorch ConvTranspose2d semantics (checked against a direct
# loop over the PyTorch definition, torch weight layout [IC, OC, KH, KW]), the
# adjoint of ag_conv2d, finite-difference gradients; on the GPU the forward and
# the graph backward against the CPU, and an ag_capture_step recording.

ct_ref <- function(x, w, s, p, d, op) {
  IC <- dim(x)[1]; H <- dim(x)[2]; W <- dim(x)[3]
  OC <- dim(w)[2]; KH <- dim(w)[3]; KW <- dim(w)[4]
  OH <- (H - 1) * s[1] - 2 * p[1] + d[1] * (KH - 1) + op[1] + 1
  OW <- (W - 1) * s[2] - 2 * p[2] + d[2] * (KW - 1) + op[2] + 1
  y <- array(0, c(OC, OH, OW))
  for (ic in 1:IC) for (i in 1:H) for (j in 1:W) for (kh in 1:KH) for (kw in 1:KW) {
    r <- (i - 1) * s[1] - p[1] + (kh - 1) * d[1] + 1
    c <- (j - 1) * s[2] - p[2] + (kw - 1) * d[2] + 1
    if (r >= 1 && r <= OH && c >= 1 && c <= OW)
      y[, r, c] <- y[, r, c] + x[ic, i, j] * w[ic, , kh, kw]
  }
  y
}
ct_col <- function(a) as.vector(aperm(a, c(3, 2, 1)))   # [C, H, W] -> W fastest

ct_cases <- list(
  list(k = c(3, 3), s = c(2, 2), p = c(1, 1), op = c(1, 1), d = c(1, 1), hw = c(4, 5)),
  list(k = c(4, 4), s = c(2, 2), p = c(1, 1), op = c(0, 0), d = c(1, 1), hw = c(3, 3)),
  list(k = c(3, 2), s = c(1, 3), p = c(0, 1), op = c(0, 2), d = c(2, 1), hw = c(4, 3)),
  list(k = c(1, 1), s = c(1, 1), p = c(0, 0), op = c(0, 0), d = c(1, 1), hw = c(2, 2)))

ct_case_data <- function(cs, seed) {
  set.seed(seed)
  IC <- 3L; OC <- 2L; B <- 2L
  wt <- array(rnorm(IC * OC * prod(cs$k)), c(IC, OC, cs$k))
  xs <- lapply(seq_len(B), function(b) array(rnorm(IC * prod(cs$hw)), c(IC, cs$hw)))
  list(IC = IC, OC = OC, B = B, wt = wt, xs = xs,
       W = matrix(aperm(wt, c(1, 4, 3, 2)), IC), X = sapply(xs, ct_col),
       bias = matrix(rnorm(OC), OC, 1L),
       oh = (cs$hw - 1) * cs$s - 2 * cs$p + cs$d * (cs$k - 1) + cs$op + 1)
}

ct_run <- function(cs, d, X = d$X, W = d$W, bias = d$bias)
  ag_conv_transpose2d(X, W, bias, in_shape = c(d$IC, cs$hw), kernel_size = cs$k,
                      stride = cs$s, padding = cs$p, output_padding = cs$op,
                      dilation = cs$d)

test_that("ag_conv_transpose2d matches the PyTorch definition and output size (CPU)", {
  local_cpu_device()
  for (i in seq_along(ct_cases)) {
    cs <- ct_cases[[i]]; d <- ct_case_data(cs, i)
    y <- as.matrix(ct_run(cs, d))
    ref <- sapply(d$xs, function(x) {
      r <- ct_ref(x, d$wt, cs$s, cs$p, cs$d, cs$op)
      ct_col(r) + rep(as.numeric(d$bias), each = prod(dim(r)[2:3]))
    })
    expect_identical(nrow(y), as.integer(d$OC * prod(d$oh)), info = i)
    expect_equal(y, ref, tolerance = 1e-12, info = i)
  }
  # the usual decoder step doubles H and W
  l <- ag_conv_transpose2d_layer(8L, 3L, kernel_size = 3L, in_hw = c(5L, 7L), stride = 2L,
                                 padding = 1L, output_padding = 1L)
  expect_identical(l$out_shape, c(3L, 10L, 14L))
})

test_that("ag_conv_transpose2d is the adjoint of ag_conv2d (CPU)", {
  local_cpu_device()
  for (i in seq_along(ct_cases)) {
    cs <- ct_cases[[i]]; d <- ct_case_data(cs, 10 + i)
    y <- as.matrix(ct_run(cs, d, bias = NULL))
    G <- matrix(rnorm(length(y)), nrow(y), d$B)
    c2 <- as.matrix(ag_conv2d(G, d$W, in_shape = c(d$OC, d$oh), kernel_size = cs$k,
                              stride = cs$s, padding = cs$p, dilation = cs$d))
    expect_equal(sum(y * G), sum(d$X * c2), tolerance = 1e-12, info = i)
  }
})

test_that("ag_conv_transpose2d gradients match finite differences (CPU)", {
  local_cpu_device()
  for (i in seq_along(ct_cases)) {
    cs <- ct_cases[[i]]; d <- ct_case_data(cs, 20 + i)
    G <- matrix(rnorm(d$OC * prod(d$oh) * d$B), ncol = d$B)
    x <- ag_param(d$X); w <- ag_param(d$W); b <- ag_param(d$bias)
    with_grad_tape(loss <- ag_sum(ag_mul(ct_run(cs, d, x, w, b), ag_tensor(G))))
    backward(loss)
    f <- function(X = d$X, W = d$W, bias = d$bias)
      sum(as.matrix(ct_run(cs, d, X, W, bias)) * G)
    num <- function(M, arg) {
      out <- M * 0
      for (j in seq_along(M)) {
        a <- M; a[j] <- a[j] + 1e-6; m <- M; m[j] <- m[j] - 1e-6
        out[j] <- (do.call(f, setNames(list(a), arg)) - do.call(f, setNames(list(m), arg))) / 2e-6
      }
      out
    }
    expect_equal(as.matrix(x$grad), num(d$X, "X"), tolerance = 1e-6, info = i)
    expect_equal(as.matrix(w$grad), num(d$W, "W"), tolerance = 1e-6, info = i)
    expect_equal(as.matrix(b$grad), num(d$bias, "bias"), tolerance = 1e-6, info = i)
  }
})

test_that("ag_conv_transpose2d refuses groups, a large output_padding and bad weights", {
  local_cpu_device()
  X <- matrix(0, 12, 1)
  expect_error(ag_conv_transpose2d(X, matrix(0, 3, 2), in_shape = c(3, 2, 2), kernel_size = 1,
                                   groups = 2), "only groups = 1")
  expect_error(ag_conv_transpose2d(X, matrix(0, 3, 18), in_shape = c(3, 2, 2), kernel_size = 3,
                                   stride = 2, output_padding = 2), "smaller than stride")
  expect_error(ag_conv_transpose2d(X, matrix(0, 4, 18), in_shape = c(3, 2, 2), kernel_size = 3),
               "weight has 4 rows")
  expect_error(ag_conv_transpose2d(X, matrix(0, 3, 10), in_shape = c(3, 2, 2), kernel_size = 3),
               "not a multiple")
})

test_that("ag_conv_transpose2d_layer: torch init takes fan_in from the weight's second axis", {
  local_cpu_device()
  set.seed(3L)
  l <- ag_conv_transpose2d_layer(8L, 3L, kernel_size = 3L, in_hw = c(5L, 5L), init = "torch")
  bound <- 1 / sqrt(3 * 3 * 3)              # OC*KH*KW, not IC*KH*KW
  w <- as.matrix(l$weight)
  expect_identical(dim(w), c(8L, 27L))
  expect_lte(max(abs(w)), bound)
  expect_gt(max(abs(w)), 0.9 * bound)
  expect_lte(max(abs(as.matrix(l$bias))), bound)
  expect_named(l$parameters(), c("weight", "bias"))
})

ct_gpu_grads <- function(cs, d, G, graph) {
  old <- ggmlR:::ag_backward_graph(graph)
  withr::defer(ggmlR:::ag_backward_graph(old))
  x <- ag_param(d$X); w <- ag_param(d$W); b <- ag_param(d$bias)
  with_grad_tape(loss <- ag_sum(ag_mul(ct_run(cs, d, x, w, b), ag_tensor(G))))
  backward(loss)
  list(path = ggmlR:::ag_backward_path(), x = ggmlR:::.ag_as_matrix(x$grad),
       w = ggmlR:::.ag_as_matrix(w$grad), b = ggmlR:::.ag_as_matrix(b$grad))
}

test_that("ag_conv_transpose2d on the GPU: forward and graph backward match the CPU", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  for (i in seq_along(ct_cases)) {
    cs <- ct_cases[[i]]; d <- ct_case_data(cs, 30 + i)
    G <- matrix(rnorm(d$OC * prod(d$oh) * d$B), ncol = d$B)
    ag_device("cpu")
    ref_y <- as.matrix(ct_run(cs, d))
    ref_g <- ct_gpu_grads(cs, d, G, graph = FALSE)
    ag_device("gpu"); ag_dtype("f32")
    ag_local_mode(graph = TRUE, matmul_precision = "f32")
    expect_equal(as.matrix(ct_run(cs, d)), ref_y, tolerance = 1e-5, info = i)
    got <- ct_gpu_grads(cs, d, G, graph = TRUE)
    expect_identical(got$path, "graph", info = i)
    for (nm in c("x", "w", "b"))
      expect_equal(got[[nm]], ref_g[[nm]], tolerance = 1e-4, info = paste(i, nm))
  }
})

test_that("ag_capture_step records ag_conv_transpose2d and follows the uncaptured loop", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  run <- function(captured) {
    ag_device("gpu"); ag_dtype("f32")
    ag_local_mode(graph = TRUE, matmul_precision = "f32")
    set.seed(8L)
    up <- ag_conv_transpose2d_layer(4L, 2L, kernel_size = 3L, in_hw = c(4L, 4L),
                                    stride = 2L, padding = 1L, output_padding = 1L,
                                    init = "torch")
    X <- matrix(rnorm(4 * 16 * 6), 64, 6); Y <- matrix(rnorm(2 * 64 * 6) * 0.1, 128, 6)
    fn  <- function(x, y) ag_mse_loss(up$forward(x), y)
    opt <- optimizer_adam(up$parameters(), lr = 0.02)
    losses <- numeric(0)
    if (captured) {
      tr <- ag_capture_step(fn, optimizer = opt)
      on.exit(ag_capture_free(tr), add = TRUE)
      for (k in 1:5) losses <- c(losses, as.numeric(tr(x = X, y = Y)))
    } else {
      x <- ag_tensor(X); y <- ag_tensor(Y)
      for (k in 1:5) {
        loss <- NULL
        with_grad_tape(loss <- fn(x, y))
        opt$step(backward(loss)); opt$zero_grad()
        losses <- c(losses, as.numeric(as.matrix(loss)))
      }
    }
    losses
  }
  ref <- run(FALSE)
  got <- run(TRUE)
  expect_equal(got, ref, tolerance = 1e-4)
  expect_lt(got[5], got[1])
})
