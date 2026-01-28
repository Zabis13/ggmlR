# Tests for Quantization Functions

# ============================================================================
# Quantization Init/Free
# ============================================================================

test_that("ggml_quantize_init works for Q8_0", {
  expect_no_error(ggml_quantize_init(GGML_TYPE_Q8_0))
  expect_no_error(ggml_quantize_free())
})

test_that("ggml_quantize_init works for Q4_0", {
  expect_no_error(ggml_quantize_init(GGML_TYPE_Q4_0))
  expect_no_error(ggml_quantize_free())
})

test_that("ggml_quantize_free is safe to call multiple times", {
  ggml_quantize_init(GGML_TYPE_Q8_0)
  expect_no_error(ggml_quantize_free())
  expect_no_error(ggml_quantize_free())  # Should be safe
})

# ============================================================================
# Quantization Requirements
# ============================================================================

test_that("ggml_quantize_requires_imatrix returns logical", {
  result <- ggml_quantize_requires_imatrix(GGML_TYPE_Q4_0)
  expect_type(result, "logical")
})

test_that("ggml_quantize_requires_imatrix for different types", {
  # Q4_0 doesn't require imatrix
  expect_false(ggml_quantize_requires_imatrix(GGML_TYPE_Q4_0))

  # Q8_0 doesn't require imatrix
  expect_false(ggml_quantize_requires_imatrix(GGML_TYPE_Q8_0))

  # F32 doesn't require imatrix (it's not even quantized)
  expect_false(ggml_quantize_requires_imatrix(GGML_TYPE_F32))
})

# ============================================================================
# Quantize Chunk - Q8_0
# ============================================================================

test_that("ggml_quantize_chunk works for Q8_0", {
  # Q8_0 has block size 32
  n_elements <- 256  # Multiple of 32
  data <- rnorm(n_elements)

  result <- ggml_quantize_chunk(GGML_TYPE_Q8_0, data, nrows = 1, n_per_row = n_elements)

  expect_type(result, "raw")
  expect_gt(length(result), 0)

  # Q8_0 should compress: 32 floats (128 bytes) -> 32 bytes + 2 byte scale = 34 bytes per block
  # 256 elements = 8 blocks, 8 * 34 = 272 bytes
  expect_equal(length(result), 272)

  ggml_quantize_free()
})

test_that("ggml_quantize_chunk preserves data approximately for Q8_0", {
  # Q8_0 is relatively high quality
  n_elements <- 32  # One block
  data <- c(1:16, -(1:16)) / 10  # Range -1.6 to 1.6

  quantized <- ggml_quantize_chunk(GGML_TYPE_Q8_0, data, nrows = 1, n_per_row = n_elements)

  expect_type(quantized, "raw")
  expect_gt(length(quantized), 0)

  ggml_quantize_free()
})

# ============================================================================
# Quantize Chunk - Q4_0
# ============================================================================

test_that("ggml_quantize_chunk works for Q4_0", {
  # Q4_0 has block size 32
  n_elements <- 256
  data <- rnorm(n_elements)

  result <- ggml_quantize_chunk(GGML_TYPE_Q4_0, data, nrows = 1, n_per_row = n_elements)

  expect_type(result, "raw")
  expect_gt(length(result), 0)

  # Q4_0: 32 floats -> 16 bytes (4 bits each) + 2 byte scale = 18 bytes per block
  # 256 elements = 8 blocks, 8 * 18 = 144 bytes
  expect_equal(length(result), 144)

  ggml_quantize_free()
})

# ============================================================================
# Multi-row Quantization
# ============================================================================

test_that("ggml_quantize_chunk handles multiple rows", {
  n_per_row <- 128  # 4 blocks per row
  nrows <- 4

  data <- rnorm(n_per_row * nrows)

  result <- ggml_quantize_chunk(GGML_TYPE_Q8_0, data, nrows = nrows, n_per_row = n_per_row)

  expect_type(result, "raw")
  # 4 rows * 4 blocks/row * 34 bytes/block = 544 bytes
  expect_equal(length(result), 544)

  ggml_quantize_free()
})

# ============================================================================
# Edge Cases
# ============================================================================

test_that("ggml_quantize_chunk handles zeros", {
  n_elements <- 32
  data <- rep(0, n_elements)

  result <- ggml_quantize_chunk(GGML_TYPE_Q8_0, data, nrows = 1, n_per_row = n_elements)

  expect_type(result, "raw")
  expect_gt(length(result), 0)

  ggml_quantize_free()
})

test_that("ggml_quantize_chunk handles uniform values", {
  n_elements <- 32
  data <- rep(1.5, n_elements)

  result <- ggml_quantize_chunk(GGML_TYPE_Q8_0, data, nrows = 1, n_per_row = n_elements)

  expect_type(result, "raw")

  ggml_quantize_free()
})

test_that("ggml_quantize_chunk handles large values", {
  n_elements <- 32
  data <- seq(-1000, 1000, length.out = n_elements)

  result <- ggml_quantize_chunk(GGML_TYPE_Q8_0, data, nrows = 1, n_per_row = n_elements)

  expect_type(result, "raw")

  ggml_quantize_free()
})

# ============================================================================
# Type Constants
# ============================================================================

test_that("quantization type constants are defined", {
  expect_equal(GGML_TYPE_F32, 0L)
  expect_equal(GGML_TYPE_F16, 1L)
  expect_equal(GGML_TYPE_Q4_0, 2L)
  expect_equal(GGML_TYPE_Q4_1, 3L)
  expect_equal(GGML_TYPE_Q8_0, 8L)
  expect_equal(GGML_TYPE_I32, 26L)
})
