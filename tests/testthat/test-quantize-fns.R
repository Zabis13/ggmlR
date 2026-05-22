# Quantization accuracy tests, ported from upstream ggml tests/test-quantize-fns.cpp
#
# For each quant type we measure the round-trip error
#   data -> quantize -> dequantize -> data'
# using the SAME synthetic data, the SAME RMSE metric and the SAME per-type
# error thresholds as upstream. This is stricter than the cor() > 0.9 checks in
# test-quants.R and will catch precision regressions those would miss.
#
# Not ported: the dot-product error part of the upstream test, which needs a
# ggml_vec_dot binding that ggmlR does not expose.

library(ggmlR)

# --- upstream constants (test-quantize-fns.cpp) ---------------------------
MAX_QUANTIZATION_TOTAL_ERROR          <- 0.002
MAX_QUANTIZATION_TOTAL_ERROR_BINARY   <- 0.025
MAX_QUANTIZATION_TOTAL_ERROR_TERNARY  <- 0.01
MAX_QUANTIZATION_TOTAL_ERROR_2BITS    <- 0.0075
MAX_QUANTIZATION_TOTAL_ERROR_3BITS    <- 0.0040
MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS <- 0.0050
MAX_QUANTIZATION_TOTAL_ERROR_FP4      <- 0.0030

# upstream: generate_data(offset): dst[i] = 0.1 + 2*cos(i + offset)
generate_data <- function(offset, n) 0.1 + 2 * cos(seq.int(0, n - 1) + offset)

# upstream: array_rmse = sqrt(sum(diff^2)) / n   (note: /n, not /sqrt(n))
array_rmse <- function(a, b) sqrt(sum((a - b)^2)) / length(a)

# block size is 32 for legacy quants, 256 for K/IQ/TQ/fp4 quants
test_size <- 32 * 128  # 4096, multiple of 256 -> valid for every type

# One row of data, shared across all types (mirrors upstream test_data)
test_data <- generate_data(0.0, test_size)

# ggml_type enum values, needed by IQ types whose ggml_quantize_init() builds
# the lookup grids (mirrors the upstream ggml_quantize_init(ei) call per type).
GGML_TYPE_IQ3_XXS <- 18L
GGML_TYPE_IQ3_S   <- 21L
GGML_TYPE_IQ2_S   <- 22L

# (label, quantize-fn, dequantize-fn, threshold, init_type|NA)
# quantize-fn: function(x, n) -> raw ; dequantize-fn: function(raw, n) -> numeric
qfn  <- function(f)  function(x, n) f(x, 1, n, NULL)
qref <- function(f)  function(x, n) f(x, n)

cases <- list(
  list("q4_0", qref(quantize_row_q4_0_ref), dequantize_row_q4_0, MAX_QUANTIZATION_TOTAL_ERROR, NA),
  list("q4_1", qref(quantize_row_q4_1_ref), dequantize_row_q4_1, MAX_QUANTIZATION_TOTAL_ERROR, NA),
  list("q5_0", qref(quantize_row_q5_0_ref), dequantize_row_q5_0, MAX_QUANTIZATION_TOTAL_ERROR, NA),
  list("q5_1", qref(quantize_row_q5_1_ref), dequantize_row_q5_1, MAX_QUANTIZATION_TOTAL_ERROR, NA),
  list("q8_0", qref(quantize_row_q8_0_ref), dequantize_row_q8_0, MAX_QUANTIZATION_TOTAL_ERROR, NA),

  list("q2_K", qfn(quantize_q2_K), dequantize_row_q2_K, MAX_QUANTIZATION_TOTAL_ERROR_2BITS, NA),
  list("q3_K", qfn(quantize_q3_K), dequantize_row_q3_K, MAX_QUANTIZATION_TOTAL_ERROR_3BITS, NA),
  list("q4_K", qfn(quantize_q4_K), dequantize_row_q4_K, MAX_QUANTIZATION_TOTAL_ERROR, NA),
  list("q5_K", qfn(quantize_q5_K), dequantize_row_q5_K, MAX_QUANTIZATION_TOTAL_ERROR, NA),
  list("q6_K", qfn(quantize_q6_K), dequantize_row_q6_K, MAX_QUANTIZATION_TOTAL_ERROR, NA),

  list("iq3_xxs", qfn(quantize_iq3_xxs), dequantize_row_iq3_xxs, MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS, GGML_TYPE_IQ3_XXS),
  list("iq3_s",   qfn(quantize_iq3_s),   dequantize_row_iq3_s,   MAX_QUANTIZATION_TOTAL_ERROR_3BITS, GGML_TYPE_IQ3_S),
  list("iq2_s",   qfn(quantize_iq2_s),   dequantize_row_iq2_s,   MAX_QUANTIZATION_TOTAL_ERROR_2BITS, GGML_TYPE_IQ2_S),
  list("iq4_nl",  qfn(quantize_iq4_nl),  dequantize_row_iq4_nl,  MAX_QUANTIZATION_TOTAL_ERROR, NA),
  list("iq4_xs",  qfn(quantize_iq4_xs),  dequantize_row_iq4_xs,  MAX_QUANTIZATION_TOTAL_ERROR, NA),

  list("tq1_0", qfn(quantize_tq1_0), dequantize_row_tq1_0, MAX_QUANTIZATION_TOTAL_ERROR_TERNARY, NA),
  list("tq2_0", qfn(quantize_tq2_0), dequantize_row_tq2_0, MAX_QUANTIZATION_TOTAL_ERROR_TERNARY, NA),

  list("mxfp4", qfn(quantize_mxfp4), dequantize_row_mxfp4, MAX_QUANTIZATION_TOTAL_ERROR_FP4, NA),
  list("nvfp4", qfn(quantize_nvfp4), dequantize_row_nvfp4, MAX_QUANTIZATION_TOTAL_ERROR_FP4, NA),

  list("q1_0", qfn(quantize_q1_0), dequantize_row_q1_0, MAX_QUANTIZATION_TOTAL_ERROR_BINARY, NA)
)

for (case in cases) {
  label  <- case[[1]]
  quant  <- case[[2]]
  dequant <- case[[3]]
  thresh <- case[[4]]
  init_type <- case[[5]]

  test_that(sprintf("%s round-trip RMSE is below upstream threshold", label), {
    if (!is.na(init_type)) {
      ggml_quantize_init(init_type)
      on.exit(ggml_quantize_free(), add = TRUE)
    }

    raw <- quant(test_data, test_size)
    expect_type(raw, "raw")
    out <- dequant(raw, test_size)
    expect_length(out, test_size)

    err <- array_rmse(test_data, out)
    expect_true(is.finite(err))
    expect_lt(err, thresh)
  })
}
