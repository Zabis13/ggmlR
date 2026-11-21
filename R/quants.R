# Low-level quantization functions

# ============================================================================
# Dequantize Row Functions
# Convert quantized data back to float
# ============================================================================

#' Dequantize rows
#'
#' Convert quantized data back to float values. One function per quantization
#' type:
#' \describe{
#'   \item{Basic}{\code{q4_0}, \code{q4_1}, \code{q5_0}, \code{q5_1},
#'     \code{q8_0}.}
#'   \item{K-quants}{\code{q2_K} through \code{q8_K}; better quality/size
#'     tradeoffs.}
#'   \item{Ternary}{\code{tq1_0}, \code{tq2_0}; extreme compression formats.}
#'   \item{IQ}{\code{iq2_xxs}, \code{iq2_xs}, \code{iq2_s}, \code{iq3_xxs},
#'     \code{iq3_s}, \code{iq4_nl}, \code{iq4_xs}, \code{iq1_s},
#'     \code{iq1_m}; high compression with importance-matrix-aware
#'     quantization.}
#'   \item{Other}{\code{mxfp4} (microscaling FP4), \code{q1_0}
#'     (\code{n_elements} must be a multiple of 128), \code{q2_0} and
#'     \code{nvfp4} (multiples of 64).}
#' }
#'
#' @param raw_data Raw vector containing quantized data
#' @param n_elements Number of elements to dequantize
#' @return Numeric vector of dequantized values
#' @seealso \code{\link{quantize_data}}, \link{ggmlR-quantization}
#' @name dequantize_row
NULL

#' @rdname dequantize_row
#' @export
dequantize_row_q4_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q4_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q4_1 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q4_1", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q5_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q5_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q5_1 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q5_1", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q8_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q8_0", raw_data, as.numeric(n_elements))
}

# K-quants dequantize

#' @rdname dequantize_row
#' @export
dequantize_row_q2_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q2_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q3_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q3_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q4_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q4_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q5_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q5_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q6_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q6_K", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q8_K <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q8_K", raw_data, as.numeric(n_elements))
}

# Ternary quants dequantize

#' @rdname dequantize_row
#' @export
dequantize_row_tq1_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_tq1_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_tq2_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_tq2_0", raw_data, as.numeric(n_elements))
}

# IQ quants dequantize

#' @rdname dequantize_row
#' @export
dequantize_row_iq2_xxs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq2_xxs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq2_xs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq2_xs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq2_s <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq2_s", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq3_xxs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq3_xxs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq3_s <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq3_s", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq4_nl <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq4_nl", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq4_xs <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq4_xs", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq1_s <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq1_s", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_iq1_m <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_iq1_m", raw_data, as.numeric(n_elements))
}

# MXFP4 dequantize

#' @rdname dequantize_row
#' @export
dequantize_row_mxfp4 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_mxfp4", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q1_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q1_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_q2_0 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_q2_0", raw_data, as.numeric(n_elements))
}

#' @rdname dequantize_row
#' @export
dequantize_row_nvfp4 <- function(raw_data, n_elements) {
  .Call("R_dequantize_row_nvfp4", raw_data, as.numeric(n_elements))
}

# ============================================================================
# Quantize Functions (with importance matrix support)
# ============================================================================

#' Quantize data
#'
#' Quantize float data, row by row, with an optional importance matrix. One
#' function per quantization type:
#' \describe{
#'   \item{Basic}{\code{q4_0}, \code{q4_1}, \code{q5_0}, \code{q5_1},
#'     \code{q8_0}.}
#'   \item{K-quants}{\code{q2_K} through \code{q6_K}; better quality/size
#'     tradeoffs than basic quants.}
#'   \item{Ternary}{\code{tq1_0}, \code{tq2_0}.}
#'   \item{IQ}{\code{iq2_xxs}, \code{iq2_xs}, \code{iq2_s}, \code{iq3_xxs},
#'     \code{iq3_s}, \code{iq1_s}, \code{iq1_m}, \code{iq4_nl},
#'     \code{iq4_xs}. IQ formats require lookup table initialization before
#'     use (see \code{\link{iq2xs_init_impl}}, \code{\link{iq3xs_init_impl}}).}
#'   \item{MXFP4}{Microscaling FP4.}
#'   \item{Q1_0}{1-bit-per-weight sign quantization; \code{n_per_row} must be
#'     a multiple of 128; \code{imatrix} is ignored.}
#'   \item{Q2_0}{2 bits per weight, blocks of 64 with one FP16 scale \code{d}
#'     (the block's largest absolute value). The 2-bit codes decode to
#'     \code{-d, 0, d, 2d}; since \code{d} is the block maximum, this
#'     quantizer only produces \code{-d, 0} and \code{d}. \code{n_per_row}
#'     must be a multiple of 64; \code{imatrix} is ignored.}
#'   \item{NVFP4}{NVIDIA FP4 with UE4M3 per-sub-block scale; \code{n_per_row}
#'     must be a multiple of 64; \code{imatrix} is ignored.}
#' }
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_rows Number of rows
#' @param n_per_row Number of elements per row
#' @param imatrix Optional importance matrix (numeric vector or NULL)
#' @return Raw vector of quantized data
#' @seealso \code{\link{dequantize_row}}, \code{\link{quantize_row_ref}},
#'   \link{ggmlR-quantization}
#' @name quantize_data
NULL

#' @rdname quantize_data
#' @export
quantize_q4_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q4_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q4_1 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q4_1", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q5_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q5_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q5_1 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q5_1", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q8_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q8_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# K-quants

#' @rdname quantize_data
#' @export
quantize_q2_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q2_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q3_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q3_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q4_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q4_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q5_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q5_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q6_K <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q6_K", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# Ternary quants

#' @rdname quantize_data
#' @export
quantize_tq1_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_tq1_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_tq2_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_tq2_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# IQ quants

#' @rdname quantize_data
#' @export
quantize_iq2_xxs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq2_xxs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq2_xs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq2_xs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq2_s <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq2_s", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq3_xxs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq3_xxs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq3_s <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq3_s", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq1_s <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq1_s", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq1_m <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq1_m", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq4_nl <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq4_nl", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_iq4_xs <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_iq4_xs", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# MXFP4

#' @rdname quantize_data
#' @export
quantize_mxfp4 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_mxfp4", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q1_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q1_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_q2_0 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_q2_0", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

#' @rdname quantize_data
#' @export
quantize_nvfp4 <- function(src_data, n_rows, n_per_row, imatrix = NULL) {
  .Call("R_quantize_nvfp4", as.numeric(src_data), as.numeric(n_rows),
        as.numeric(n_per_row), imatrix)
}

# ============================================================================
# Quantize Row Reference Functions
# Basic row-level quantization without importance matrix
# ============================================================================

#' Quantize row (reference implementations)
#'
#' Basic row-level quantization without importance matrix. These are the
#' reference implementations: \code{q4_0}, \code{q4_1}, \code{q5_0},
#' \code{q5_1}, \code{q8_0}, \code{q8_1}; K-quants \code{q2_K} through
#' \code{q8_K}; ternary \code{tq1_0}, \code{tq2_0}; IQ \code{iq3_xxs},
#' \code{iq4_nl}, \code{iq4_xs}, \code{iq3_s}, \code{iq2_s}; \code{mxfp4},
#' \code{q1_0}, \code{q2_0} (\code{n_elements} a multiple of 64) and
#' \code{nvfp4}.
#'
#' @param src_data Numeric vector of float values to quantize
#' @param n_elements Number of elements to quantize
#' @return Raw vector of quantized data
#' @seealso \code{\link{quantize_data}}, \link{ggmlR-quantization}
#' @name quantize_row_ref
NULL

#' @rdname quantize_row_ref
#' @export
quantize_row_q4_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q4_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q4_1_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q4_1_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q5_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q5_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q5_1_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q5_1_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q8_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q8_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q8_1_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q8_1_ref", as.numeric(src_data), as.numeric(n_elements))
}

# K-quants row ref

#' @rdname quantize_row_ref
#' @export
quantize_row_q2_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q2_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q3_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q3_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q4_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q4_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q5_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q5_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q6_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q6_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q8_K_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q8_K_ref", as.numeric(src_data), as.numeric(n_elements))
}

# Ternary row ref

#' @rdname quantize_row_ref
#' @export
quantize_row_tq1_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_tq1_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_tq2_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_tq2_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

# IQ row ref

#' @rdname quantize_row_ref
#' @export
quantize_row_iq3_xxs_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq3_xxs_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_iq4_nl_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq4_nl_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_iq4_xs_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq4_xs_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_iq3_s_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq3_s_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_iq2_s_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_iq2_s_ref", as.numeric(src_data), as.numeric(n_elements))
}

# MXFP4 row ref

#' @rdname quantize_row_ref
#' @export
quantize_row_mxfp4_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_mxfp4_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q1_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q1_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_q2_0_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_q2_0_ref", as.numeric(src_data), as.numeric(n_elements))
}

#' @rdname quantize_row_ref
#' @export
quantize_row_nvfp4_ref <- function(src_data, n_elements) {
  .Call("R_quantize_row_nvfp4_ref", as.numeric(src_data), as.numeric(n_elements))
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
#' @seealso \link{ggmlR-quantization}
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
#' @seealso \link{ggmlR-quantization}
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
#' @seealso \link{ggmlR-quantization}
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
#' @seealso \link{ggmlR-quantization}
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
#' @seealso \link{ggmlR-quantization}
ggml_quant_block_info <- function(type) {
  .Call("R_ggml_quant_block_info", as.integer(type))
}
