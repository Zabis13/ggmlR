# QLinearConv integer path (GGML_OP_QCONV_I32) on the CPU: edge shapes of the
# im2col kernel, checked bit for bit against an R reference of the same
# arithmetic -- int16-saturated channel pairs (VPMADDUBSW), x_zp padding,
# per-channel weight scale, round-half-even, uint8 clamp.
#
# The shapes target the im2col layout: odd C_in (the pair slot holding 0*0),
# a Kv tail, padding on every edge, stride 2, batch > 1, and outputs smaller
# than the thread count (threads with no rows and no output channels must
# still pass the barrier). Weights and activations are drawn from the full
# int8/uint8 range so pairs genuinely saturate; the test asserts that they do,
# because without saturation an exact f32 fallback would pass too.

run_onnx_q <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  onnx_run(m, inputs)[[1]]
}

# Reference: x is ONNX [N, C_in, H, W] row-major, w is [C_out, C_in, KH, KW].
qconv_ref_acc <- function(x, w, dims, bias, x_zp, pad, stride) {
  N <- dims$N; C <- dims$C_in; H <- dims$H; W <- dims$W
  OC <- dims$C_out; K <- dims$k
  OH <- (H + 2 * pad - K) %/% stride + 1
  OW <- (W + 2 * pad - K) %/% stride + 1
  xa <- array(x, c(W, H, C, N))            # R column-major view of row-major
  wa <- array(w, c(K, K, C, OC))
  acc <- array(0L, c(OW, OH, OC, N))
  n_sat <- 0L
  for (n in seq_len(N)) for (oc in seq_len(OC)) {
    sum_w <- sum(wa[, , , oc])
    for (oh in seq_len(OH)) for (ow in seq_len(OW)) {
      s <- 0
      for (kh in seq_len(K)) for (kw in seq_len(K)) {
        ih <- (oh - 1) * stride - pad + kh
        iw <- (ow - 1) * stride - pad + kw
        is_pad <- ih < 1 || ih > H || iw < 1 || iw > W
        for (ic in seq(1, C, by = 2)) {
          xv <- function(c) if (c > C) 0 else if (is_pad) x_zp else xa[iw, ih, c, n]
          wv <- function(c) if (c > C) 0 else wa[kw, kh, c, oc]
          pair <- xv(ic) * wv(ic) + xv(ic + 1) * wv(ic + 1)
          if (pair > 32767 || pair < -32768) n_sat <- n_sat + 1L
          s <- s + min(max(pair, -32768), 32767)
        }
      }
      acc[ow, oh, oc, n] <- s + bias[oc] - x_zp * sum_w
    }
  }
  list(acc = acc, n_sat = n_sat, OH = OH, OW = OW)
}

make_qconv_model <- function(dims, w, w_scale, bias, x_zp, y_scale, y_zp,
                             pad, stride) {
  N <- dims$N; C <- dims$C_in; OC <- dims$C_out; K <- dims$k
  OH <- (dims$H + 2 * pad - K) %/% stride + 1
  OW <- (dims$W + 2 * pad - K) %/% stride + 1
  f32 <- function(nm, v) .onnx_tensor(nm, length(v), 1L,
                                      unlist(lapply(v, .float_bytes)))
  u8  <- function(nm, v) .onnx_tensor(nm, 1L, 2L, as.raw(v))
  inits <- list(
    f32("x_sc", 1),
    u8("x_zp", x_zp),
    .onnx_tensor("W", c(OC, C, K, K), 3L, as.raw(ifelse(w < 0, w + 256, w))),
    f32("w_sc", w_scale),
    .onnx_tensor("w_zp", OC, 3L, as.raw(rep(0, OC))),
    f32("y_sc", y_scale),
    u8("y_zp", y_zp),
    .onnx_tensor("B", OC, 6L, writeBin(as.integer(bias), raw(), size = 4,
                                       endian = "little")))
  node <- .onnx_node("QLinearConv",
    c("X", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp", "B"), "Y",
    attrs = list(.onnx_attr_ints("kernel_shape", c(K, K)),
                 .onnx_attr_ints("pads", rep(pad, 4)),
                 .onnx_attr_ints("strides", c(stride, stride))))
  graph <- .onnx_graph("qconv_im2col", list(node),
    list(.onnx_value_info("X", 1L, c(N, C, dims$H, dims$W))),
    list(.onnx_value_info("Y", 1L, c(N, OC, OH, OW))),
    inits)
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

check_qconv_case <- function(N, C_in, C_out, H, W, k, pad, stride, seed,
                             per_channel = TRUE, device = "cpu") {
  set.seed(seed)
  dims <- list(N = N, C_in = C_in, C_out = C_out, H = H, W = W, k = k)
  x    <- sample(0:255, N * C_in * H * W, replace = TRUE)
  w    <- sample(-128:127, C_out * C_in * k * k, replace = TRUE)
  bias <- sample(-2000:2000, C_out, replace = TRUE)
  # Guarantee saturating pairs on tiny shapes: channels 1-2 of image 1 at 255
  # against filter 1's channels 1-2 at 127 give 64770 per pair, every tap.
  x[seq_len(min(2, C_in) * H * W)] <- 255
  w[seq_len(min(2, C_in) * k * k)] <- 127
  x_zp <- 7
  y_zp <- 128

  ref <- qconv_ref_acc(x, w, dims, bias, x_zp, pad, stride)
  # Saturation must actually occur, or this test cannot tell the integer path
  # from the exact f32 fallback.
  expect_gt(ref$n_sat, 0L)

  # Powers of two only: mult = x_sc * w_sc[oc] / y_sc is then exact in float,
  # and so is acc * mult, so the reference needs no float emulation. y_sc is
  # sized so outputs stay inside [0, 255] and every element is informative.
  w_scale <- if (per_channel) 2^sample(-1:1, C_out, replace = TRUE) else 1
  ws_oc   <- rep_len(w_scale, C_out)
  scaled  <- sweep(ref$acc, 3, ws_oc, `*`)
  y_scale <- 2^ceiling(log2(max(abs(scaled)) / 100))

  path <- make_qconv_model(dims, w, w_scale, bias, x_zp, y_scale, y_zp,
                           pad, stride)
  got <- as.numeric(run_onnx_q(path, list(X = as.numeric(x)), device))

  want <- pmin(pmax(round(scaled / y_scale) + y_zp, 0), 255)
  expect_equal(length(got), length(want))
  expect_identical(got, as.numeric(want))
}

test_that("qconv_i32 im2col: odd C_in, 3x3, padding on every edge", {
  check_qconv_case(N = 1, C_in = 3, C_out = 5, H = 5, W = 6,
                   k = 3, pad = 1, stride = 1, seed = 1)
})

test_that("qconv_i32 im2col: stride 2, batch 2", {
  check_qconv_case(N = 2, C_in = 4, C_out = 3, H = 7, W = 5,
                   k = 3, pad = 1, stride = 2, seed = 2)
})

test_that("qconv_i32 im2col: 1x1 with odd C_in", {
  check_qconv_case(N = 1, C_in = 7, C_out = 2, H = 4, W = 3,
                   k = 1, pad = 0, stride = 1, seed = 3)
})

test_that("qconv_i32 im2col: long K (vector loop + tail), W_out not a multiple of 4", {
  check_qconv_case(N = 1, C_in = 33, C_out = 6, H = 3, W = 9,
                   k = 3, pad = 1, stride = 1, seed = 4)
})

test_that("qconv_i32 im2col: one output element, fewer rows and channels than threads", {
  check_qconv_case(N = 1, C_in = 2, C_out = 1, H = 1, W = 1,
                   k = 3, pad = 1, stride = 1, seed = 5)
})

test_that("qconv_i32 im2col: per-tensor weight scale", {
  check_qconv_case(N = 1, C_in = 5, C_out = 4, H = 4, W = 4,
                   k = 3, pad = 1, stride = 1, seed = 6, per_channel = FALSE)
})

# The Vulkan shader against the same reference, with the batch axis carrying
# real work: ggml_vk_qconv_i32 runs one flat grid over batch*C_out*H_out*W_out,
# and its supports_op guard used to size that grid without the batch. 51 is
# MaskRCNN's mask-head batch. The guard itself cannot be tripped here on a
# device whose x-axis limit is 2^32-1 (RX 9070 / RADV); this checks that the
# batched dispatch is right, bit for bit.
test_that("qconv_i32 Vulkan: batch 51 matches the reference", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  check_qconv_case(N = 51, C_in = 5, C_out = 4, H = 6, W = 6,
                   k = 3, pad = 1, stride = 1, seed = 7, device = "vulkan")
})
