# Tests for quantize_* functions (n_rows/n_per_row API) not covered in test-quants.R

test_that("quantize_tq2_0 works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test")
  n <- 256
  original <- runif(n, -1, 1)

  ggml_quantize_init(35L)  # TQ2_0
  q <- quantize_tq2_0(original, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_tq2_0(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

test_that("quantize_iq2_s works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test")
  n <- 256
  original <- runif(n, -1, 1)

  ggml_quantize_init(22L)  # IQ2_S
  q <- quantize_iq2_s(original, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq2_s(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

test_that("quantize_iq3_xxs works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test")
  n <- 256
  original <- runif(n, -1, 1)

  ggml_quantize_init(18L)  # IQ3_XXS
  q <- quantize_iq3_xxs(original, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq3_xxs(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

test_that("quantize_iq3_s works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test")
  n <- 256
  original <- runif(n, -1, 1)

  ggml_quantize_init(21L)  # IQ3_S
  q <- quantize_iq3_s(original, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq3_s(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

test_that("quantize_iq4_nl works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test")
  n <- 256
  original <- runif(n, -1, 1)

  ggml_quantize_init(20L)  # IQ4_NL
  q <- quantize_iq4_nl(original, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq4_nl(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

test_that("quantize_iq4_xs works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test")
  n <- 256
  original <- runif(n, -1, 1)

  ggml_quantize_init(23L)  # IQ4_XS
  q <- quantize_iq4_xs(original, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq4_xs(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

# ============================================================================
# New quant types from ggml-0.11.0 (Q1_0, NVFP4)
# Block sizes: Q1_0 = 128, NVFP4 = 64
# ============================================================================

test_that("quantize_q1_0 round-trips correctly", {
  set.seed(1)
  n <- 128 * 4  # 4 blocks
  original <- runif(n, -1, 1)

  q <- quantize_q1_0(original, 1, n, NULL)
  expect_type(q, "raw")
  # Each Q1_0 block: 2 bytes (fp16 d) + 16 bytes (qs) = 18 bytes; 4 blocks = 72
  expect_length(q, 4 * 18)

  d <- dequantize_row_q1_0(q, n)
  expect_length(d, n)
  # Q1_0 stores only the sign of each weight: dequantized values must have the
  # same sign as the original (zeros excluded).
  nonzero <- abs(original) > 1e-6
  expect_true(all(sign(d[nonzero]) == sign(original[nonzero])))
  # All dequantized magnitudes equal |d| (the block delta = mean(|x|))
  block_means <- vapply(seq_len(4), function(b) {
    idx <- ((b - 1) * 128 + 1):(b * 128)
    mean(abs(original[idx]))
  }, numeric(1))
  for (b in seq_len(4)) {
    idx <- ((b - 1) * 128 + 1):(b * 128)
    expect_equal(abs(d[idx]), rep(block_means[b], 128), tolerance = 1e-2)
  }
})

test_that("quantize_q1_0 handles all-positive and all-negative blocks", {
  n <- 128
  pos <- runif(n, 0.1, 1.0)
  neg <- runif(n, -1.0, -0.1)

  qp <- quantize_q1_0(pos, 1, n, NULL)
  qn <- quantize_q1_0(neg, 1, n, NULL)
  dp <- dequantize_row_q1_0(qp, n)
  dn <- dequantize_row_q1_0(qn, n)

  expect_true(all(dp > 0))
  expect_true(all(dn < 0))
})

test_that("quantize_nvfp4 round-trips with reasonable error", {
  set.seed(2)
  n <- 64 * 4  # 4 NVFP4 blocks
  original <- runif(n, -1, 1)

  q <- quantize_nvfp4(original, 1, n, NULL)
  expect_type(q, "raw")
  # Each NVFP4 block: 4 bytes (4× UE4M3 sub-scales) + 32 bytes (qs) = 36; 4 blocks = 144
  expect_length(q, 4 * 36)

  d <- dequantize_row_nvfp4(q, n)
  expect_length(d, n)

  # NVFP4 is 4-bit E2M1: max rel error is reasonable on uniformly-distributed input
  err <- abs(d - original)
  expect_true(mean(err) < 0.1)
  expect_true(max(err) < 0.5)
})

test_that("quantize_nvfp4 preserves zero block", {
  n <- 64
  zeros <- rep(0.0, n)
  q <- quantize_nvfp4(zeros, 1, n, NULL)
  d <- dequantize_row_nvfp4(q, n)
  expect_equal(d, zeros, tolerance = 1e-6)
})

test_that("ggml_quantize_chunk dispatches Q1_0 and NVFP4", {
  # type indices: NVFP4 = 40, Q1_0 = 41
  src_q1 <- runif(128 * 2, -1, 1)
  out_q1 <- .Call("R_ggml_quantize_chunk", 41L, src_q1, 2, 128)
  expect_type(out_q1, "raw")
  expect_length(out_q1, 2 * 18)  # 2 blocks × 18 bytes

  src_nv <- runif(64 * 2, -1, 1)
  out_nv <- .Call("R_ggml_quantize_chunk", 40L, src_nv, 2, 64)
  expect_type(out_nv, "raw")
  expect_length(out_nv, 2 * 36)  # 2 blocks × 36 bytes
})

test_that("Q1_0/NVFP4 registered in type_traits", {
  # Sanity: the C-side ggml_get_type_traits should return non-null type_name for
  # both new types. We can't call it directly from R, but we can verify via the
  # roundtrip that lengths match what the type registration implies.
  q <- quantize_q1_0(rep(0.5, 128), 1, 128, NULL)
  expect_equal(length(q), 18L)  # confirms block size 128 → 18 bytes

  q2 <- quantize_nvfp4(rep(0.5, 64), 1, 64, NULL)
  expect_equal(length(q2), 36L)  # confirms block size 64 → 36 bytes
})
