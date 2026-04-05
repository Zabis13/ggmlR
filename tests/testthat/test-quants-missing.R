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
