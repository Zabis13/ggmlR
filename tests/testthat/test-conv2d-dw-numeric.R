# Numeric depthwise-conv2d tests, ported from upstream ggml tests/test-conv2d-dw.cpp
#
# Upstream computes a reference depthwise convolution on the CPU and compares it
# against ggml_conv_2d_dw_direct for several (channels, kernel, stride, pad,
# dilation) configurations. We reproduce the reference in R and check the same
# WHCN-layout configurations. (The im2col variant ggml_conv_2d_dw requires an
# F16 kernel; the direct variant used here matches upstream and takes F32.)

library(ggmlR)

# Direct R port of conv_2d_dw_reference() from the upstream test.
# src/knl are flat F32 in WHCN / [KW,KH,1,C] order; 0-based indexing as in C.
conv_2d_dw_reference <- function(src_w, src_h, src, knl_w, knl_h, knl,
                                 channels, batch, stride, pad, dilation) {
  dst_w <- (src_w + 2 * pad - dilation * (knl_w - 1) - 1) %/% stride + 1
  dst_h <- (src_h + 2 * pad - dilation * (knl_h - 1) - 1) %/% stride + 1
  dst <- numeric(dst_w * dst_h * channels * batch)
  for (b in 0:(batch - 1)) {
    sb <- b * src_w * src_h * channels
    db <- b * dst_w * dst_h * channels
    for (c in 0:(channels - 1)) {
      for (y in 0:(dst_h - 1)) {
        for (x in 0:(dst_w - 1)) {
          s <- 0
          for (ky in 0:(knl_h - 1)) {
            for (kx in 0:(knl_w - 1)) {
              sx <- x * stride + kx * dilation - pad
              sy <- y * stride + ky * dilation - pad
              if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
                s <- s + src[sb + c * src_w * src_h + sy * src_w + sx + 1] *
                         knl[c * knl_w * knl_h + ky * knl_w + kx + 1]
              }
            }
          }
          dst[db + c * dst_w * dst_h + y * dst_w + x + 1] <- s
        }
      }
    }
  }
  dst
}

# Run ggml_conv_2d_dw_direct and the R reference on identical inputs.
run_dw <- function(channels, k, stride, pad, dilation) {
  batch <- 2L; src_w <- 8L; src_h <- 6L
  ctx <- ggml_init(64 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  src <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, src_w, src_h, channels, batch)
  knl <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, k, 1L, channels)
  r   <- ggml_conv_2d_dw_direct(ctx, knl, src, stride, stride, pad, pad,
                                dilation, dilation)

  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be), add = TRUE)
  ggml_backend_cpu_set_n_threads(be, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, be)

  sv <- seq(-1, 1, length.out = src_w * src_h * channels * batch)
  kv <- seq(-1, 1, length.out = k * k * channels)
  ggml_backend_tensor_set_data(src, sv)
  ggml_backend_tensor_set_data(knl, kv)

  gf <- ggml_build_forward_expand(ctx, r)
  ggml_backend_graph_compute(be, gf)

  list(got = ggml_backend_tensor_get_data(r),
       exp = conv_2d_dw_reference(src_w, src_h, sv, k, k, kv,
                                  channels, batch, stride, pad, dilation))
}

# Upstream WHCN configurations: (channels, kernel, stride, pad, dilation)
configs <- list(
  c(3,  1, 1, 0, 1),
  c(42, 3, 2, 1, 1),
  c(8,  5, 1, 2, 2)
)

for (cfg in configs) {
  ch <- cfg[1]; k <- cfg[2]; s <- cfg[3]; p <- cfg[4]; d <- cfg[5]
  test_that(sprintf("conv_2d_dw_direct matches reference (ch=%d k=%d s=%d p=%d d=%d)",
                    ch, k, s, p, d), {
    res <- run_dw(as.integer(ch), as.integer(k), as.integer(s),
                  as.integer(p), as.integer(d))
    expect_length(res$got, length(res$exp))
    expect_equal(res$got, res$exp, tolerance = 1e-4)
  })
}
