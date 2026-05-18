# IQ-quants degenerate-input tests covering 9 fixes ported from upstream
# ggml-0.11.0 (Этап 4). Pathological inputs that previously triggered
# ASSERT/abort or used uninitialised L[] should now produce a quantised
# output (graceful fallback) and dequantise without crashing.
#
# This file is listed in tests/testthat.R `heavy` — it runs in dev
# (NOT_CRAN=true) but is skipped on CRAN because IQ quants init large
# grid tables and take noticeable CPU time.

test_that("IQ1_S handles all-zero block without aborting (besti<0 fallback)", {
  # IQ1_S type = 19; block size for the quantize_iq1_s API is 256 elements.
  # IQ1_S requires an imatrix (upstream GGML_ASSERT in quant impl), so pass
  # a uniform-weight matrix to exercise the degenerate-input path.
  n <- 256
  zeros <- rep(0, n)
  imatrix <- rep(1, n)

  ggml_quantize_init(19L)
  q <- quantize_iq1_s(zeros, 1, n, imatrix)
  expect_type(q, "raw")

  d <- dequantize_row_iq1_s(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})

test_that("IQ1_S handles uniform-magnitude block (degenerate scale case)", {
  n <- 256
  # All equal positive values — pre-fix code could hit besti1/besti2<0 or
  # best_shift==0 because all candidates score the same. Should now degrade
  # gracefully.
  x <- rep(0.25, n)
  imatrix <- rep(1, n)

  ggml_quantize_init(19L)
  q <- quantize_iq1_s(x, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq1_s(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  ggml_quantize_free()
})

test_that("IQ1_M handles all-zero block (besti<0 / best_k<0 fallback)", {
  n <- 256
  zeros <- rep(0, n)
  imatrix <- rep(1, n)

  ggml_quantize_init(29L)  # IQ1_M
  q <- quantize_iq1_m(zeros, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq1_m(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})

test_that("IQ1_M handles uniform-magnitude block", {
  n <- 256
  x <- rep(-0.5, n)
  imatrix <- rep(1, n)

  ggml_quantize_init(29L)
  q <- quantize_iq1_m(x, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq1_m(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  ggml_quantize_free()
})

test_that("IQ4_NL handles all-zero input (sumq2 > 0 guard)", {
  # IQ4_NL: pre-fix code computed d = sumqx/sumq2 unconditionally → div/0
  # when input is all-zero. Now guarded with sumq2 > 0 check.
  n <- 256
  zeros <- rep(0, n)

  ggml_quantize_init(20L)  # IQ4_NL
  q <- quantize_iq4_nl(zeros, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq4_nl(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})

test_that("IQ4_XS handles all-zero input (sumq2 > 0 guard)", {
  n <- 256
  zeros <- rep(0, n)

  ggml_quantize_init(23L)  # IQ4_XS
  q <- quantize_iq4_xs(zeros, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq4_xs(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})

test_that("IQ3_S handles all-zero input (make_qp_quants eff_max<=0 guard)", {
  n <- 256
  zeros <- rep(0, n)

  ggml_quantize_init(21L)  # IQ3_S
  q <- quantize_iq3_s(zeros, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq3_s(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})

test_that("IQ3_XXS handles all-zero input (memset(L,0) above guard)", {
  # IQ3_XXS: pre-fix code had memset(L, 0, 32) BELOW the early-exit guard,
  # so on degenerate blocks the loop could use uninitialised L[]. Now
  # memset is above the guard.
  n <- 256
  zeros <- rep(0, n)

  ggml_quantize_init(18L)  # IQ3_XXS
  q <- quantize_iq3_xxs(zeros, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq3_xxs(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})

test_that("IQ2_S handles all-zero input (memset(L,0) above guard)", {
  n <- 256
  zeros <- rep(0, n)

  ggml_quantize_init(22L)  # IQ2_S
  q <- quantize_iq2_s(zeros, 1, n, NULL)
  expect_type(q, "raw")
  d <- dequantize_row_iq2_s(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})

test_that("IQ2_XS handles all-zero input (memset(L,0) above guard)", {
  # IQ2_XS requires imatrix (upstream GGML_ASSERT).
  n <- 256
  zeros <- rep(0, n)
  imatrix <- rep(1, n)

  ggml_quantize_init(17L)  # IQ2_XS
  q <- quantize_iq2_xs(zeros, 1, n, imatrix)
  expect_type(q, "raw")
  d <- dequantize_row_iq2_xs(q, n)
  expect_length(d, n)
  expect_false(any(is.nan(d)))
  expect_false(any(is.infinite(d)))
  ggml_quantize_free()
})
