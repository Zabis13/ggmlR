# ggml_im2col_back (the R binding of GGML_OP_IM2COL_BACK) and the F32 im2col it
# inverts: both against loops written in R, the backward as the adjoint of the
# forward, CPU and Vulkan. The column order (ic, kh, kw), kw fastest, is the
# row-major flattening of a PyTorch kernel [OC, IC, KH, KW] that ag_conv2d
# relies on.

im2col_backends <- function()
  c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L) "vulkan")

im2col_out_size <- function(i, k, s, p, d) (i + 2 * p - d * (k - 1) - 1) %/% s + 1

# One graph on one backend: `build(ctx, tensors)` returns the output node.
# `inputs` is a list of list(shape, data). Returns list(out, supported).
im2col_run <- function(backend, inputs, build) {
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  b <- if (backend == "vulkan") ggml_vulkan_init(0L) else {
    cb <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cb, 2L); cb
  }
  on.exit(ggml_backend_free(b), add = TRUE, after = FALSE)
  ts <- lapply(inputs, function(i) {
    t <- ggml_new_tensor(ctx, GGML_TYPE_F32, length(i$shape), i$shape)
    ggml_set_input(t)
    t
  })
  out <- build(ctx, ts)
  ggml_set_output(out)
  supported <- ggml_backend_dev_supports_op(ggml_backend_get_device(b), out)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, b)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  for (i in seq_along(ts)) ggml_backend_tensor_set_data(ts[[i]], inputs[[i]]$data)
  ggml_backend_graph_compute(b, ggml_build_forward_expand(ctx, out))
  list(out = ggml_backend_tensor_get_data(out), supported = supported)
}

# The (column row, image index) pairs of a configuration, 1-based; image index
# NA where the tap reads the zero padding.
im2col_map <- function(cs) {
  IW <- cs$img[1]; IH <- cs$img[2]; IC <- cs$img[3]; N <- cs$img[4]
  KW <- cs$k[1]; KH <- cs$k[2]
  OW <- im2col_out_size(IW, KW, cs$s[1], cs$p[1], cs$d[1])
  OH <- im2col_out_size(IH, KH, cs$s[2], cs$p[2], cs$d[2])
  g <- expand.grid(kw = 0:(KW - 1), kh = 0:(KH - 1), ic = 0:(IC - 1),
                   ow = 0:(OW - 1), oh = 0:(OH - 1), n = 0:(N - 1))
  ix <- g$ow * cs$s[1] - cs$p[1] + g$kw * cs$d[1]
  iy <- g$oh * cs$s[2] - cs$p[2] + g$kh * cs$d[2]
  inside <- ix >= 0 & ix < IW & iy >= 0 & iy < IH
  img <- ifelse(inside, 1 + ix + IW * (iy + IH * (g$ic + IC * g$n)), NA)
  list(img = img, OW = OW, OH = OH, K = IC * KH * KW)   # rows of g are in column order
}

im2col_ref <- function(x, m) { v <- x[m$img]; v[is.na(v)] <- 0; v }
im2col_back_ref <- function(gc, m, n_img) {
  keep <- !is.na(m$img)
  out <- numeric(n_img)
  s <- rowsum(gc[keep], m$img[keep])
  out[as.integer(rownames(s))] <- s[, 1]
  out
}

im2col_case_run <- function(cs, backend) {
  m <- im2col_map(cs)
  kshape <- c(cs$k[1], cs$k[2], cs$img[3], 2)            # [KW, KH, IC, OC]
  colshape <- c(m$K, m$OW, m$OH, cs$img[4])
  x  <- runif(prod(cs$img), -1, 1)
  gc <- runif(prod(colshape), -1, 1)
  kern <- list(shape = kshape, data = runif(prod(kshape)))
  fwd <- im2col_run(backend, list(kern, list(shape = cs$img, data = x)), function(ctx, t)
    ggml_im2col(ctx, t[[1]], t[[2]], cs$s[1], cs$s[2], cs$p[1], cs$p[2], cs$d[1], cs$d[2],
                TRUE, GGML_TYPE_F32))
  bwd <- im2col_run(backend, list(list(shape = colshape, data = gc), kern), function(ctx, t)
    ggml_im2col_back(ctx, t[[1]], t[[2]], cs$img, cs$s[1], cs$s[2], cs$p[1], cs$p[2],
                     cs$d[1], cs$d[2]))
  list(m = m, x = x, gc = gc, fwd = fwd, bwd = bwd)
}

im2col_cases <- list(
  list(name = "overlap, batch 2",            img = c(7, 5, 3, 2),  k = c(3, 3), s = c(1, 1), p = c(0, 0), d = c(1, 1)),
  list(name = "non-square image and kernel", img = c(9, 6, 2, 3),  k = c(3, 2), s = c(2, 1), p = c(1, 0), d = c(1, 1)),
  list(name = "stride < kernel, padding",    img = c(8, 8, 2, 2),  k = c(4, 4), s = c(2, 2), p = c(1, 1), d = c(1, 1)),
  list(name = "dilation 2",                  img = c(10, 9, 2, 2), k = c(3, 3), s = c(1, 1), p = c(0, 0), d = c(2, 2)),
  list(name = "padding and dilation",        img = c(9, 7, 2, 2),  k = c(3, 2), s = c(2, 1), p = c(2, 1), d = c(2, 3)),
  list(name = "uncovered tail (11 / 4)",     img = c(11, 10, 1, 2), k = c(4, 3), s = c(4, 3), p = c(0, 0), d = c(1, 1)),
  list(name = "Nature-CNN conv1 shape",      img = c(20, 20, 4, 2), k = c(8, 8), s = c(4, 4), p = c(0, 0), d = c(1, 1)),
  list(name = "8x8 kernel, stride 1",        img = c(14, 12, 2, 2), k = c(8, 8), s = c(1, 1), p = c(0, 0), d = c(1, 1)))

test_that("im2col (F32) and im2col_back match the definition, on every backend", {
  set.seed(11)
  for (be in im2col_backends()) for (cs in im2col_cases) {
    info <- paste(be, cs$name)
    r <- im2col_case_run(cs, be)
    expect_true(r$fwd$supported, info = info)
    expect_true(r$bwd$supported, info = info)
    expect_equal(r$fwd$out, im2col_ref(r$x, r$m), tolerance = 1e-6, info = info)
    expect_equal(r$bwd$out, im2col_back_ref(r$gc, r$m, length(r$x)), tolerance = 1e-6,
                 info = info)
    # the adjoint: <im2col(x), g> = <x, im2col_back(g)>
    expect_equal(sum(r$fwd$out * r$gc), sum(r$x * r$bwd$out), tolerance = 1e-5, info = info)
  }
})

test_that("Vulkan im2col_back agrees with the CPU (f32 tolerance)", {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L, "Vulkan not available")
  for (cs in im2col_cases) {
    set.seed(12); cpu <- im2col_case_run(cs, "cpu")
    set.seed(12); vk  <- im2col_case_run(cs, "vulkan")
    expect_equal(vk$bwd$out, cpu$bwd$out, tolerance = 1e-6, info = cs$name)
    expect_equal(vk$fwd$out, cpu$fwd$out, tolerance = 1e-6, info = cs$name)
  }
})

test_that("ggml_im2col_back refuses a bad shape and non-F32 sources", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  g <- ggml_new_tensor(ctx, GGML_TYPE_F32, 4, c(9, 3, 3, 1))
  k <- ggml_new_tensor(ctx, GGML_TYPE_F32, 4, c(3, 3, 1, 1))
  expect_error(ggml_im2col_back(ctx, g, k, c(5, 5, 1), 1, 1, 0, 0, 1, 1), "4 positive")
  k16 <- ggml_new_tensor(ctx, GGML_TYPE_F16, 4, c(3, 3, 1, 1))
  expect_error(ggml_im2col_back(ctx, g, k16, c(5, 5, 1, 1), 1, 1, 0, 0, 1, 1), "F32")
})
