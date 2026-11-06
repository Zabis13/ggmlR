# CPU leaky_relu F32: ggml_vec_leaky_relu_f32 has an AVX / SSE2 and-mask path
# (DIVERGENCE from upstream, whose scalar two-ternary loop GCC -O2 turns into
# branches). It must give exactly the bits of the scalar formula
#     y = (x > 0 ? x : 0) + ns * (x < 0 ? x : 0)
# so NaN -> 0 and -0 -> +0, on every length (vector body + scalar tail).
#
# Values are chosen so the f32 result is exact: halves and powers of two times
# ns = 0.25, plus NaN, +-0, +-Inf and an f32 denormal (2^-140).

run_cpu <- function(x, shape, ns) {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[1], shape[2])
  out <- ggml_leaky_relu(ctx, a, negative_slope = ns)
  gf <- ggml_build_forward_expand(ctx, out)
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE, after = FALSE)
  ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(a, x)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

leaky_ref <- function(x, ns) {
  p <- ifelse(!is.na(x) & x > 0, x, 0)
  m <- ifelse(!is.na(x) & x < 0, x, 0)
  p + ns * m          # +0 + ns*0 = +0, so -0 and NaN both come out +0
}

special <- c(NaN, -0, 0, Inf, -Inf, 2^-140, -2^-140, 1.5, -1.5)

test_that("CPU leaky_relu is bit-exact on every length (SIMD body + tail)", {
  ns <- 0.25
  for (n in c(1L, 3L, 4L, 5L, 7L, 8L, 9L, 15L, 16L, 17L, 1000003L)) {
    x <- ((seq_len(n) %% 13L) - 6L) * 0.5
    x[seq_len(min(n, length(special)))] <- special[seq_len(min(n, length(special)))]
    got <- run_cpu(x, c(n, 1L), ns)
    ref <- leaky_ref(x, ns)
    expect_identical(got, ref, info = paste("n =", n))
    # -0 must come out +0 (identical() does not tell the sign of zero)
    expect_true(all(1 / got[got == 0] > 0), info = paste("sign of zero, n =", n))
    expect_false(anyNA(got), info = paste("NaN leaked, n =", n))
  }
})

test_that("CPU leaky_relu is bit-exact per row (rows shorter than a vector)", {
  ns <- 0.25
  for (shape in list(c(7L, 5L), c(9L, 4L), c(33L, 3L))) {
    n <- prod(shape)
    x <- ((seq_len(n) %% 11L) - 5L) * 0.5
    x[seq_along(special)] <- special
    expect_identical(run_cpu(x, shape, ns), leaky_ref(x, ns),
                     info = paste(shape, collapse = "x"))
  }
})

test_that("CPU leaky_relu with a non-exact slope matches the f32 product", {
  # 0.1 is not exact in f32; the C side rounds it to float, then multiplies in
  # f32. Products of two f32 values are exact in double, so rounding the R
  # double product through an F32 tensor gives the reference bits.
  set.seed(5)
  n <- 4099L
  x <- rnorm(n)
  got <- run_cpu(x, c(n, 1L), 0.1)
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))
  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  ggml_set_f32(t, x); xf <- ggml_get_f32(t)                 # x rounded to f32
  ggml_set_f32(t, rep(0.1, n)); nsf <- ggml_get_f32(t)[1]   # 0.1f
  ggml_set_f32(t, ifelse(xf > 0, xf, 0) + nsf * ifelse(xf < 0, xf, 0))
  expect_identical(got, ggml_get_f32(t))
})
