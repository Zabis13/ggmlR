# Low-level quantization functions

# ============================================================================
# Dequantize Row Functions
# Convert quantized data back to float
# ============================================================================

#' Dequantize Row (Q4_0)
#'
#' Converts Q4_0 quantized data back to float values.
#'
#' @param raw_data Raw vector containing quantized data
#' @param n_elements Number of elements to dequantize
#' @return Numeric vector of dequantized values
#' @export
#' @family quantization
dequantize_row_q4_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q4_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q4_0
#' @export
dequantize_row_q4_1 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q4_1", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q4_0
#' @export
dequantize_row_q5_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q5_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q4_0
#' @export
dequantize_row_q5_1 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q5_1", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q4_0
#' @export
dequantize_row_q8_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q8_0", raw_data, as.numeric(n_elements))
}

# K-quants dequantize

#' Dequantize Row (K-quants)
#'
#' Converts K-quant quantized data back to float values.
#' K-quants (q2_K through q8_K) provide better quality/size tradeoffs.
#'
#' @param raw_data Raw vector containing quantized data
#' @param n_elements Number of elements to dequantize
#' @return Numeric vector of dequantized values
#' @export
#' @family quantization
dequantize_row_q2_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q2_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q2_K
#' @export
dequantize_row_q3_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q3_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q2_K
#' @export
dequantize_row_q4_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q4_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q2_K
#' @export
dequantize_row_q5_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q5_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q2_K
#' @export
dequantize_row_q6_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q6_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_q2_K
#' @export
dequantize_row_q8_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q8_K", raw_data, as.numeric(n_elements))
}

# Ternary quants dequantize

#' Dequantize Row (Ternary)
#'
#' Converts ternary quantized data back to float values.
#' TQ1_0 and TQ2_0 are extreme compression formats.
#'
#' @param raw_data Raw vector containing quantized data
#' @param n_elements Number of elements to dequantize
#' @return Numeric vector of dequantized values
#' @export
#' @family quantization
dequantize_row_tq1_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_tq1_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_tq1_0
#' @export
dequantize_row_tq2_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_tq2_0", raw_data, as.numeric(n_elements))
}

# IQ quants dequantize

#' Dequantize Row (IQ)
#'
#' Converts IQ (integer quantization) data back to float values.
#' IQ formats provide high compression with importance-matrix-aware quantization.
#'
#' @param raw_data Raw vector containing quantized data
#' @param n_elements Number of elements to dequantize
#' @return Numeric vector of dequantized values
#' @export
#' @family quantization
dequantize_row_iq2_xxs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq2_xxs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq2_xs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq2_xs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq2_s <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq2_s", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq3_xxs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq3_xxs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq3_s <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq3_s", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq4_nl <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq4_nl", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq4_xs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq4_xs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq1_s <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq1_s", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row_iq2_xxs
#' @export
dequantize_row_iq1_m <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq1_m", raw_data, as.numeric(n_elements))
}

# MXFP4 dequantize

#' Dequantize Row (MXFP4)
#'
#' Converts MXFP4 (microscaling FP4) quantized data back to float values.
#'
#' @param raw_data Raw vector containing quantized data
#' @param n_elements Number of elements to dequantize
#' @return Numeric vector of dequantized values
#' @export
#' @family quantization
dequantize_row_mxfp4 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_mxfp4", raw_data, as.numeric(n_elements))
}

# ============================================================================
# Quantize Functions (with importance matrix support)
# ============================================================================

#' Quantize Data (Q4_0)
#'
#' Quantizes float data to Q4_0 format with optional importance matrix.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_rows Number of rows
#' @param n_per_row Number of elements per row
#' @param imatrix Optional importance matrix (numeric vector or NULL)
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_q4_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q4_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q4_0
#' @export
quantize_q4_1 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q4_1", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q4_0
#' @export
quantize_q5_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q5_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q4_0
#' @export
quantize_q5_1 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q5_1", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q4_0
#' @export
quantize_q8_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q8_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# K-quants

#' Quantize Data (K-quants)
#'
#' Quantizes float data to K-quant format with optional importance matrix.
#' K-quants provide better quality/size tradeoffs than basic quants.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_rows Number of rows
#' @param n_per_row Number of elements per row
#' @param imatrix Optional importance matrix (numeric vector or NULL)
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_q2_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q2_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q2_K
#' @export
quantize_q3_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q3_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q2_K
#' @export
quantize_q4_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q4_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q2_K
#' @export
quantize_q5_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q5_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_q2_K
#' @export
quantize_q6_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q6_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# Ternary quants

#' Quantize Data (Ternary)
#'
#' Quantizes float data to ternary format with optional importance matrix.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_rows Number of rows
#' @param n_per_row Number of elements per row
#' @param imatrix Optional importance matrix (numeric vector or NULL)
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_tq1_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_tq1_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_tq1_0
#' @export
quantize_tq2_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_tq2_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# IQ quants

#' Quantize Data (IQ)
#'
#' Quantizes float data to IQ format. IQ formats require importance matrix
#' initialization before use (see iq2xs_init_impl, iq3xs_init_impl).
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_rows Number of rows
#' @param n_per_row Number of elements per row
#' @param imatrix Optional importance matrix (numeric vector or NULL)
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_iq2_xxs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq2_xxs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq2_xs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq2_xs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq2_s <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq2_s", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq3_xxs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq3_xxs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq3_s <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq3_s", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq1_s <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq1_s", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq1_m <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq1_m", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq4_nl <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq4_nl", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_iq2_xxs
#' @export
quantize_iq4_xs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq4_xs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# MXFP4

#' Quantize Data (MXFP4)
#'
#' Quantizes float data to MXFP4 (microscaling FP4) format.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_rows Number of rows
#' @param n_per_row Number of elements per row
#' @param imatrix Optional importance matrix (numeric vector or NULL)
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_mxfp4 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_mxfp4", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# ============================================================================
# Quantize Row Reference Functions
# Basic row-level quantization without importance matrix
# ============================================================================

#' Quantize Row Reference (Basic)
#'
#' Basic row-level quantization without importance matrix.
#' These are reference implementations.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_elements Number of elements to quantize
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_row_q4_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q4_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q4_0_ref
#' @export
quantize_row_q4_1_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q4_1_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q4_0_ref
#' @export
quantize_row_q5_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q5_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q4_0_ref
#' @export
quantize_row_q5_1_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q5_1_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q4_0_ref
#' @export
quantize_row_q8_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q8_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q4_0_ref
#' @export
quantize_row_q8_1_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q8_1_ref", as.numeric(src_data), as.numeric(n_elements))
}

# K-quants row ref

#' Quantize Row Reference (K-quants)
#'
#' Basic row-level K-quant quantization without importance matrix.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_elements Number of elements to quantize
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_row_q2_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q2_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q2_K_ref
#' @export
quantize_row_q3_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q3_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q2_K_ref
#' @export
quantize_row_q4_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q4_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q2_K_ref
#' @export
quantize_row_q5_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q5_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q2_K_ref
#' @export
quantize_row_q6_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q6_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_q2_K_ref
#' @export
quantize_row_q8_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q8_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

# Ternary row ref

#' Quantize Row Reference (Ternary)
#'
#' Basic row-level ternary quantization.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_elements Number of elements to quantize
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_row_tq1_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_tq1_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_tq1_0_ref
#' @export
quantize_row_tq2_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_tq2_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

# IQ row ref

#' Quantize Row Reference (IQ)
#'
#' Basic row-level IQ quantization.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_elements Number of elements to quantize
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_row_iq3_xxs_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq3_xxs_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_iq3_xxs_ref
#' @export
quantize_row_iq4_nl_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq4_nl_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_iq3_xxs_ref
#' @export
quantize_row_iq4_xs_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq4_xs_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_iq3_xxs_ref
#' @export
quantize_row_iq3_s_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq3_s_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_iq3_xxs_ref
#' @export
quantize_row_iq2_s_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq2_s_ref", as.numeric(src_data), as.numeric(n_elements))
}

# MXFP4 row ref

#' Quantize Row Reference (MXFP4)
#'
#' Basic row-level MXFP4 quantization.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_elements Number of elements to quantize
#' @return Raw vector of quantized data
#' @export
#' @family quantization
quantize_row_mxfp4_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_mxfp4_ref", as.numeric(src_data), as.numeric(n_elements))
}

# ============================================================================
# IQ Init/Free Functions
# ============================================================================

#' Initialize IQ2 Quantization Tables
#'
#' Initializes lookup tables for IQ2 quantization types.
#' Must be called before using iq2_xxs, iq2_xs, or iq2_s quantization.
#'
#' @param type GGML type constant (e.g., GGML_TYPE_IQ2_XXS())
#' @return NULL invisibly
#' @export
#' @family quantization
iq2xs_init_impl <- function(type) {
  invisible(.Call("R_iq2xs_init_impl", as.integer(type)))
}

#' Free IQ2 Quantization Tables
#'
#' Frees lookup tables for IQ2 quantization types.
#'
#' @param type GGML type constant
#' @return NULL invisibly
#' @export
#' @family quantization
iq2xs_free_impl <- function(type) {
  invisible(.Call("R_iq2xs_free_impl", as.integer(type)))
}

#' Initialize IQ3 Quantization Tables
#'
#' Initializes lookup tables for IQ3 quantization types.
#' Must be called before using iq3_xxs or iq3_s quantization.
#'
#' @param grid_size Grid size for IQ3 (typically 256)
#' @return NULL invisibly
#' @export
#' @family quantization
iq3xs_init_impl <- function(grid_size) {
  invisible(.Call("R_iq3xs_init_impl", as.integer(grid_size)))
}

#' Free IQ3 Quantization Tables
#'
#' Frees lookup tables for IQ3 quantization types.
#'
#' @param grid_size Grid size for IQ3
#' @return NULL invisibly
#' @export
#' @family quantization
iq3xs_free_impl <- function(grid_size) {
  invisible(.Call("R_iq3xs_free_impl", as.integer(grid_size)))
}

# ============================================================================
# Quantization Info
# ============================================================================

#' Get Quantization Block Info
#'
#' Returns information about a quantization type including name,
#' type size, block size, and whether it's quantized.
#'
#' @param type GGML type constant
#' @return List with type_name, type_size, block_size, is_quantized
#' @export
#' @family quantization
ggml_quant_block_info <- function(type) {
  .Call("R_ggml_quant_block_info", as.integer(type))
}
