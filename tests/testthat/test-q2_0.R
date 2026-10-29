# Q2_0 (new in ggml 0.27): blocks of 64 weights, one FP16 scale d = the
# block's largest |x|, 2 bits per weight decoding to (code - 1) * d, so the
# levels are -d, 0, d, 2d. The reference quantizer rounds x / d, which lies in
# [-1, 1], so it never emits the 2d level: every weight lands on -d, 0 or d,
# within d / 2 of where it started.
#
# The CPU matmul goes through ggml_vec_dot_q2_0_q8_0 (AVX2 on x86, generic
# elsewhere). The AVX2 kernel walks the blocks two at a time and finishes an
# odd count with a single-block tail, so the shapes below have both.

GGML_TYPE_Q2_0 <- 42L
QK2_0 <- 64L
Q2_0_BLOCK_BYTES <- 2L + QK2_0 %/% 4L  # fp16 d + 16 bytes of 2-bit codes

test_that("Q2_0 quantizes each weight to -d, 0 or d of its block", {
  set.seed(21)
  n_blocks <- 5L
  n <- n_blocks * QK2_0
  x <- runif(n, -3, 3)

  q <- quantize_q2_0(x, 1, n, NULL)
  expect_type(q, "raw")
  expect_length(q, n_blocks * Q2_0_BLOCK_BYTES)

  y <- dequantize_row_q2_0(q, n)
  expect_length(y, n)

  for (b in seq_len(n_blocks)) {
    idx <- ((b - 1L) * QK2_0 + 1L):(b * QK2_0)
    d <- max(abs(x[idx]))
    # d goes through FP16: 11 significant bits
    tol <- d * 2^-10
    levels <- round(y[idx] / d)
    expect_true(all(levels %in% c(-1, 0, 1)), label = sprintf("block %d levels", b))
    expect_lt(max(abs(y[idx] - levels * d)), tol + 1e-7)
    expect_lte(max(abs(x[idx] - y[idx])), d / 2 + tol)
  }
})

test_that("Q2_0 keeps an all-zero block at zero", {
  n <- QK2_0
  y <- dequantize_row_q2_0(quantize_q2_0(rep(0, n), 1, n, NULL), n)
  expect_equal(y, rep(0, n))
})

test_that("quantize_row_q2_0_ref matches quantize_q2_0", {
  set.seed(22)
  n <- 4L * QK2_0
  x <- rnorm(n)
  expect_identical(quantize_row_q2_0_ref(x, n), quantize_q2_0(x, 1, n, NULL))
})

# A: weights [K, M] in Q2_0; B: activations [K, N] f32; out = A * B -> [M, N].
# The reference multiplies the dequantized weights in double precision; the
# CPU kernel also rounds B to Q8_0, which is what the tolerance absorbs.
run_q2_0_mm_cpu <- function(a_q, b, M, K, N) {
  ctx <- ggml_init(64L * 1024 * 1024)
  on.exit(ggml_free(ctx))
  ggml_set_no_alloc(ctx, TRUE)
  a   <- ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, K, M)
  bt  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)
  out <- ggml_mul_mat(ctx, a, bt)

  backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, a_q)
  ggml_backend_tensor_set_data(bt, as.vector(b))
  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  matrix(ggml_backend_tensor_get_data(out), nrow = M, ncol = N)
}

for (shape in list(c(M = 16L, K = 5L * QK2_0, N = 3L),   # odd block count: AVX2 tail
                   c(M = 32L, K = 8L * QK2_0, N = 7L))) { # even: pairs only
  M <- shape[["M"]]; K <- shape[["K"]]; N <- shape[["N"]]
  test_that(sprintf("Q2_0 CPU matmul matches the dequantized reference (M=%d K=%d N=%d)", M, K, N), {
    set.seed(K + N)
    a <- matrix(rnorm(M * K), nrow = M, ncol = K)
    b <- matrix(rnorm(K * N), nrow = K, ncol = N)

    a_q <- quantize_q2_0(as.vector(t(a)), M, K, NULL)
    a_deq <- matrix(dequantize_row_q2_0(a_q, M * K), nrow = M, ncol = K, byrow = TRUE)
    ref <- a_deq %*% b

    got <- run_q2_0_mm_cpu(a_q, b, M, K, N)
    expect_gt(cor(as.vector(got), as.vector(ref)), 0.9999)
    expect_lt(max(abs(got - ref)) / max(abs(ref)), 0.02)
  })
}
