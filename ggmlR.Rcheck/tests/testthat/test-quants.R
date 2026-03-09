# Tests for low-level quantization functions

library(ggmlR)

# ============================================================================
# Basic quantization (q4_0, q4_1, q5_0, q5_1, q8_0)
# ============================================================================

test_that("basic quants roundtrip works", {
  n <- 64
  original <- runif(n, -1, 1)

  # Q4_0
  q <- quantize_row_q4_0_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_q4_0(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.9)

 # Q4_1
  q <- quantize_row_q4_1_ref(original, n)
  d <- dequantize_row_q4_1(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.9)

  # Q5_0
  q <- quantize_row_q5_0_ref(original, n)
  d <- dequantize_row_q5_0(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.9)

  # Q5_1
  q <- quantize_row_q5_1_ref(original, n)
  d <- dequantize_row_q5_1(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.9)

  # Q8_0 (higher precision)
  q <- quantize_row_q8_0_ref(original, n)
  d <- dequantize_row_q8_0(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.99)

  # Q8_1
  q <- quantize_row_q8_1_ref(original, n)
  expect_type(q, "raw")
})

test_that("quantize with imatrix works", {
  n <- 64
  original <- runif(n, -1, 1)
  imatrix <- rep(1.0, n)

  # With imatrix
  q <- quantize_q4_0(original, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_q4_0(q, n)
  expect_length(d, n)

  # Without imatrix
  q <- quantize_q4_0(original, 1, n, NULL)
  expect_type(q, "raw")

  # Multiple rows
  q <- quantize_q4_0(rep(original, 2), 2, n, NULL)
  expect_type(q, "raw")
})

test_that("ggml_quant_block_info works", {
  info <- ggml_quant_block_info(GGML_TYPE_Q4_0)
  expect_equal(info$type_name, "q4_0")
  expect_true(info$is_quantized)
  expect_equal(info$block_size, 32L)

  info <- ggml_quant_block_info(GGML_TYPE_F32)
  expect_equal(info$type_name, "f32")
  expect_false(info$is_quantized)
})

# ============================================================================
# K-quants (q2_K through q8_K) - use 256-element blocks
# ============================================================================

test_that("K-quants roundtrip works", {
  n <- 256
  original <- runif(n, -1, 1)

  # Q2_K
  q <- quantize_q2_K(original, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_q2_K(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.7)

  # Q3_K
  q <- quantize_q3_K(original, 1, n, NULL)
  d <- dequantize_row_q3_K(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.8)

  # Q4_K
  q <- quantize_q4_K(original, 1, n, NULL)
  d <- dequantize_row_q4_K(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.9)

  # Q5_K
  q <- quantize_q5_K(original, 1, n, NULL)
  d <- dequantize_row_q5_K(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.95)

  # Q6_K
  q <- quantize_q6_K(original, 1, n, NULL)
  d <- dequantize_row_q6_K(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.98)
})

test_that("K-quants row_ref works", {
  n <- 256
  original <- runif(n, -1, 1)

  expect_type(quantize_row_q2_K_ref(original, n), "raw")
  expect_type(quantize_row_q3_K_ref(original, n), "raw")
  expect_type(quantize_row_q4_K_ref(original, n), "raw")
  expect_type(quantize_row_q5_K_ref(original, n), "raw")
  expect_type(quantize_row_q6_K_ref(original, n), "raw")

  # Q8_K dequantize
  q <- quantize_row_q8_K_ref(original, n)
  d <- dequantize_row_q8_K(q, n)
  expect_length(d, n)
  expect_gt(cor(original, d), 0.99)
})

# ============================================================================
# Ternary quants (TQ1_0, TQ2_0)
# ============================================================================

test_that("ternary quants work", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test - set GGMLR_SLOW_TESTS=1 to run")
  n <- 256
  original <- runif(n, -1, 1)

  # TQ1_0 (type 34)
  ggml_quantize_init(34L)
  q <- quantize_row_tq1_0_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_tq1_0(q, n)
  expect_length(d, n)
  ggml_quantize_free()

  # TQ2_0 (type 35)
  ggml_quantize_init(35L)
  q <- quantize_row_tq2_0_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_tq2_0(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

# ============================================================================
# IQ quants (IQ3, IQ4) - require ggml_quantize_init
# ============================================================================

test_that("IQ3/IQ4 quants work", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test - set GGMLR_SLOW_TESTS=1 to run")
  n <- 256
  original <- runif(n, -1, 1)

  # IQ3_XXS (type 18)
  ggml_quantize_init(18L)
  q <- quantize_row_iq3_xxs_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_iq3_xxs(q, n)
  expect_length(d, n)
  ggml_quantize_free()

  # IQ3_S (type 21)
  ggml_quantize_init(21L)
  q <- quantize_row_iq3_s_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_iq3_s(q, n)
  expect_length(d, n)
  ggml_quantize_free()

  # IQ4_NL (type 20)
  ggml_quantize_init(20L)
  q <- quantize_row_iq4_nl_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_iq4_nl(q, n)
  expect_length(d, n)
  ggml_quantize_free()

  # IQ4_XS (type 23)
  ggml_quantize_init(23L)
  q <- quantize_row_iq4_xs_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_iq4_xs(q, n)
  expect_length(d, n)
  ggml_quantize_free()

  # IQ2_S (type 22)
  ggml_quantize_init(22L)
  q <- quantize_row_iq2_s_ref(original, n)
  expect_type(q, "raw")
  d <- dequantize_row_iq2_s(q, n)
  expect_length(d, n)
  ggml_quantize_free()
})

# ============================================================================
# IQ2/IQ1 quants - require iq2xs_init_impl + importance matrix
# ============================================================================

test_that("IQ2_XXS quant with imatrix works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test - set GGMLR_SLOW_TESTS=1 to run")
  n <- 256
  original <- runif(n, -1, 1)
  imatrix <- rep(1.0, n)

  iq2xs_init_impl(16L)
  ggml_quantize_init(16L)
  q <- quantize_iq2_xxs(original, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq2_xxs(q, n)
  expect_length(d, n)
  ggml_quantize_free()
  iq2xs_free_impl(16L)
})

test_that("IQ2_XS quant with imatrix works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test - set GGMLR_SLOW_TESTS=1 to run")
  n <- 256
  original <- runif(n, -1, 1)
  imatrix <- rep(1.0, n)

  iq2xs_init_impl(17L)
  ggml_quantize_init(17L)
  q <- quantize_iq2_xs(original, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq2_xs(q, n)
  expect_length(d, n)
  ggml_quantize_free()
  iq2xs_free_impl(17L)
})

test_that("IQ1_S quant with imatrix works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test - set GGMLR_SLOW_TESTS=1 to run")
  n <- 256
  original <- runif(n, -1, 1)
  imatrix <- rep(1.0, n)

  iq2xs_init_impl(19L)
  ggml_quantize_init(19L)
  q <- quantize_iq1_s(original, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq1_s(q, n)
  expect_length(d, n)
  ggml_quantize_free()
  iq2xs_free_impl(19L)
})

test_that("IQ1_M quant with imatrix works", {
  skip_if(Sys.getenv("GGMLR_SLOW_TESTS") != "1", "Slow test - set GGMLR_SLOW_TESTS=1 to run")
  n <- 256
  original <- runif(n, -1, 1)
  imatrix <- rep(1.0, n)

  iq2xs_init_impl(29L)
  ggml_quantize_init(29L)
  q <- quantize_iq1_m(original, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq1_m(q, n)
  expect_length(d, n)
  ggml_quantize_free()
  iq2xs_free_impl(29L)
})

# ============================================================================
# MXFP4
# ============================================================================

test_that("MXFP4 quantize/dequantize works", {
  n <- 64
  original <- runif(n, -1, 1)

  q <- quantize_row_mxfp4_ref(original, n)
  expect_type(q, "raw")

  d <- dequantize_row_mxfp4(q, n)
  expect_length(d, n)
})

# ============================================================================
# Additional quantize functions
# ============================================================================

test_that("quantize functions with imatrix work", {
  n <- 64
  original <- runif(n, -1, 1)

  expect_type(quantize_q4_1(original, 1, n, NULL), "raw")
  expect_type(quantize_q5_0(original, 1, n, NULL), "raw")
  expect_type(quantize_q5_1(original, 1, n, NULL), "raw")
  expect_type(quantize_q8_0(original, 1, n, NULL), "raw")
})

test_that("different quant sizes comparison", {
  n <- 256
  original <- runif(n, -1, 1)

  q4 <- quantize_q4_0(original, 1, n, NULL)
  q8 <- quantize_q8_0(original, 1, n, NULL)

  # Q8 should be larger than Q4
 expect_gt(length(q8), length(q4))
})
