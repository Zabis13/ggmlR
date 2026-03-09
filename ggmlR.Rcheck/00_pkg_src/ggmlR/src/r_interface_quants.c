// R interface for low-level GGML quantization functions

#include <R.h>
#include <Rinternals.h>
#include <string.h>
#include "ggml.h"
#include "ggml-quants.h"

// ============================================================================
// Dequantize Row Functions
// These convert quantized blocks back to float arrays
// ============================================================================

// Helper macro for dequantize functions
#define IMPL_DEQUANTIZE_ROW(type_name, block_type) \
SEXP R_dequantize_row_##type_name(SEXP raw_data, SEXP n_elements) { \
    if (TYPEOF(raw_data) != RAWSXP) { \
        Rf_error("raw_data must be a raw vector"); \
    } \
    int64_t k = (int64_t)Rf_asReal(n_elements); \
    if (k <= 0) { \
        Rf_error("n_elements must be positive"); \
    } \
    const block_type *x = (const block_type *)RAW(raw_data); \
    SEXP result = PROTECT(Rf_allocVector(REALSXP, k)); \
    float *y = (float *)R_alloc(k, sizeof(float)); \
    dequantize_row_##type_name(x, y, k); \
    for (int64_t i = 0; i < k; i++) { \
        REAL(result)[i] = (double)y[i]; \
    } \
    UNPROTECT(1); \
    return result; \
}

// Basic quantization types
IMPL_DEQUANTIZE_ROW(q4_0, block_q4_0)
IMPL_DEQUANTIZE_ROW(q4_1, block_q4_1)
IMPL_DEQUANTIZE_ROW(q5_0, block_q5_0)
IMPL_DEQUANTIZE_ROW(q5_1, block_q5_1)
IMPL_DEQUANTIZE_ROW(q8_0, block_q8_0)

// K-quants
IMPL_DEQUANTIZE_ROW(q2_K, block_q2_K)
IMPL_DEQUANTIZE_ROW(q3_K, block_q3_K)
IMPL_DEQUANTIZE_ROW(q4_K, block_q4_K)
IMPL_DEQUANTIZE_ROW(q5_K, block_q5_K)
IMPL_DEQUANTIZE_ROW(q6_K, block_q6_K)
IMPL_DEQUANTIZE_ROW(q8_K, block_q8_K)

// Ternary quants
IMPL_DEQUANTIZE_ROW(tq1_0, block_tq1_0)
IMPL_DEQUANTIZE_ROW(tq2_0, block_tq2_0)

// IQ quants
IMPL_DEQUANTIZE_ROW(iq2_xxs, block_iq2_xxs)
IMPL_DEQUANTIZE_ROW(iq2_xs, block_iq2_xs)
IMPL_DEQUANTIZE_ROW(iq2_s, block_iq2_s)
IMPL_DEQUANTIZE_ROW(iq3_xxs, block_iq3_xxs)
IMPL_DEQUANTIZE_ROW(iq3_s, block_iq3_s)
IMPL_DEQUANTIZE_ROW(iq4_nl, block_iq4_nl)
IMPL_DEQUANTIZE_ROW(iq4_xs, block_iq4_xs)
IMPL_DEQUANTIZE_ROW(iq1_s, block_iq1_s)
IMPL_DEQUANTIZE_ROW(iq1_m, block_iq1_m)

// MXFP4
IMPL_DEQUANTIZE_ROW(mxfp4, block_mxfp4)

// ============================================================================
// Quantize Functions (with importance matrix support)
// These convert float arrays to quantized format
// ============================================================================

// Helper macro for quantize functions - uses uppercase GGML type
#define IMPL_QUANTIZE(type_name, ggml_type) \
SEXP R_quantize_##type_name(SEXP src_data, SEXP n_rows, SEXP n_per_row, SEXP imatrix) { \
    if (TYPEOF(src_data) != REALSXP) { \
        Rf_error("src_data must be a numeric vector"); \
    } \
    int64_t nrows = (int64_t)Rf_asReal(n_rows); \
    int64_t npr = (int64_t)Rf_asReal(n_per_row); \
    if (nrows <= 0 || npr <= 0) { \
        Rf_error("n_rows and n_per_row must be positive"); \
    } \
    int64_t total = nrows * npr; \
    if (Rf_length(src_data) < total) { \
        Rf_error("src_data too short: need %lld elements", (long long)total); \
    } \
    float *src = (float *)R_alloc(total, sizeof(float)); \
    for (int64_t i = 0; i < total; i++) { \
        src[i] = (float)REAL(src_data)[i]; \
    } \
    const float *imat = NULL; \
    if (!Rf_isNull(imatrix)) { \
        if (TYPEOF(imatrix) != REALSXP) { \
            Rf_error("imatrix must be numeric or NULL"); \
        } \
        float *imat_tmp = (float *)R_alloc(Rf_length(imatrix), sizeof(float)); \
        for (int i = 0; i < Rf_length(imatrix); i++) { \
            imat_tmp[i] = (float)REAL(imatrix)[i]; \
        } \
        imat = imat_tmp; \
    } \
    size_t dst_size = ggml_row_size(ggml_type, nrows * npr); \
    if (dst_size == 0) { \
        dst_size = nrows * npr * sizeof(float); \
    } \
    SEXP result = PROTECT(Rf_allocVector(RAWSXP, dst_size)); \
    void *dst = RAW(result); \
    size_t actual_size = quantize_##type_name(src, dst, nrows, npr, imat); \
    if (actual_size < (size_t)Rf_length(result)) { \
        SEXP trimmed = PROTECT(Rf_allocVector(RAWSXP, actual_size)); \
        memcpy(RAW(trimmed), RAW(result), actual_size); \
        UNPROTECT(2); \
        return trimmed; \
    } \
    UNPROTECT(1); \
    return result; \
}

// Basic types
IMPL_QUANTIZE(q4_0, GGML_TYPE_Q4_0)
IMPL_QUANTIZE(q4_1, GGML_TYPE_Q4_1)
IMPL_QUANTIZE(q5_0, GGML_TYPE_Q5_0)
IMPL_QUANTIZE(q5_1, GGML_TYPE_Q5_1)
IMPL_QUANTIZE(q8_0, GGML_TYPE_Q8_0)

// K-quants
IMPL_QUANTIZE(q2_K, GGML_TYPE_Q2_K)
IMPL_QUANTIZE(q3_K, GGML_TYPE_Q3_K)
IMPL_QUANTIZE(q4_K, GGML_TYPE_Q4_K)
IMPL_QUANTIZE(q5_K, GGML_TYPE_Q5_K)
IMPL_QUANTIZE(q6_K, GGML_TYPE_Q6_K)

// Ternary
IMPL_QUANTIZE(tq1_0, GGML_TYPE_TQ1_0)
IMPL_QUANTIZE(tq2_0, GGML_TYPE_TQ2_0)

// IQ types
IMPL_QUANTIZE(iq2_xxs, GGML_TYPE_IQ2_XXS)
IMPL_QUANTIZE(iq2_xs, GGML_TYPE_IQ2_XS)
IMPL_QUANTIZE(iq2_s, GGML_TYPE_IQ2_S)
IMPL_QUANTIZE(iq3_xxs, GGML_TYPE_IQ3_XXS)
IMPL_QUANTIZE(iq3_s, GGML_TYPE_IQ3_S)
IMPL_QUANTIZE(iq1_s, GGML_TYPE_IQ1_S)
IMPL_QUANTIZE(iq1_m, GGML_TYPE_IQ1_M)
IMPL_QUANTIZE(iq4_nl, GGML_TYPE_IQ4_NL)
IMPL_QUANTIZE(iq4_xs, GGML_TYPE_IQ4_XS)

// MXFP4
IMPL_QUANTIZE(mxfp4, GGML_TYPE_MXFP4)

// ============================================================================
// Quantize Row Reference Functions
// These are the basic row-level quantization without imatrix
// ============================================================================

// Helper macro for quantize_row_ref functions - uses uppercase GGML type
#define IMPL_QUANTIZE_ROW_REF(type_name, block_type, ggml_type) \
SEXP R_quantize_row_##type_name##_ref(SEXP src_data, SEXP n_elements) { \
    if (TYPEOF(src_data) != REALSXP) { \
        Rf_error("src_data must be a numeric vector"); \
    } \
    int64_t k = (int64_t)Rf_asReal(n_elements); \
    if (k <= 0) { \
        Rf_error("n_elements must be positive"); \
    } \
    if (Rf_length(src_data) < k) { \
        Rf_error("src_data too short"); \
    } \
    float *src = (float *)R_alloc(k, sizeof(float)); \
    for (int64_t i = 0; i < k; i++) { \
        src[i] = (float)REAL(src_data)[i]; \
    } \
    int64_t blck_size = ggml_blck_size(ggml_type); \
    if (blck_size == 0) blck_size = 32; \
    int64_t n_blocks = (k + blck_size - 1) / blck_size; \
    size_t dst_size = n_blocks * sizeof(block_type); \
    SEXP result = PROTECT(Rf_allocVector(RAWSXP, dst_size)); \
    block_type *dst = (block_type *)RAW(result); \
    quantize_row_##type_name##_ref(src, dst, k); \
    UNPROTECT(1); \
    return result; \
}

// Basic types
IMPL_QUANTIZE_ROW_REF(q4_0, block_q4_0, GGML_TYPE_Q4_0)
IMPL_QUANTIZE_ROW_REF(q4_1, block_q4_1, GGML_TYPE_Q4_1)
IMPL_QUANTIZE_ROW_REF(q5_0, block_q5_0, GGML_TYPE_Q5_0)
IMPL_QUANTIZE_ROW_REF(q5_1, block_q5_1, GGML_TYPE_Q5_1)
IMPL_QUANTIZE_ROW_REF(q8_0, block_q8_0, GGML_TYPE_Q8_0)
IMPL_QUANTIZE_ROW_REF(q8_1, block_q8_1, GGML_TYPE_Q8_1)

// K-quants
IMPL_QUANTIZE_ROW_REF(q2_K, block_q2_K, GGML_TYPE_Q2_K)
IMPL_QUANTIZE_ROW_REF(q3_K, block_q3_K, GGML_TYPE_Q3_K)
IMPL_QUANTIZE_ROW_REF(q4_K, block_q4_K, GGML_TYPE_Q4_K)
IMPL_QUANTIZE_ROW_REF(q5_K, block_q5_K, GGML_TYPE_Q5_K)
IMPL_QUANTIZE_ROW_REF(q6_K, block_q6_K, GGML_TYPE_Q6_K)
IMPL_QUANTIZE_ROW_REF(q8_K, block_q8_K, GGML_TYPE_Q8_K)

// Ternary
IMPL_QUANTIZE_ROW_REF(tq1_0, block_tq1_0, GGML_TYPE_TQ1_0)
IMPL_QUANTIZE_ROW_REF(tq2_0, block_tq2_0, GGML_TYPE_TQ2_0)

// IQ types
IMPL_QUANTIZE_ROW_REF(iq3_xxs, block_iq3_xxs, GGML_TYPE_IQ3_XXS)
IMPL_QUANTIZE_ROW_REF(iq4_nl, block_iq4_nl, GGML_TYPE_IQ4_NL)
IMPL_QUANTIZE_ROW_REF(iq4_xs, block_iq4_xs, GGML_TYPE_IQ4_XS)
IMPL_QUANTIZE_ROW_REF(iq3_s, block_iq3_s, GGML_TYPE_IQ3_S)
IMPL_QUANTIZE_ROW_REF(iq2_s, block_iq2_s, GGML_TYPE_IQ2_S)

// MXFP4
IMPL_QUANTIZE_ROW_REF(mxfp4, block_mxfp4, GGML_TYPE_MXFP4)

// ============================================================================
// IQ Init/Free Functions
// ============================================================================

SEXP R_iq2xs_init_impl(SEXP type) {
    enum ggml_type t = (enum ggml_type)Rf_asInteger(type);
    iq2xs_init_impl(t);
    return R_NilValue;
}

SEXP R_iq2xs_free_impl(SEXP type) {
    enum ggml_type t = (enum ggml_type)Rf_asInteger(type);
    iq2xs_free_impl(t);
    return R_NilValue;
}

SEXP R_iq3xs_init_impl(SEXP grid_size) {
    int gs = Rf_asInteger(grid_size);
    iq3xs_init_impl(gs);
    return R_NilValue;
}

SEXP R_iq3xs_free_impl(SEXP grid_size) {
    int gs = Rf_asInteger(grid_size);
    iq3xs_free_impl(gs);
    return R_NilValue;
}

// ============================================================================
// Convenience function to get quantization block info
// ============================================================================

SEXP R_ggml_quant_block_info(SEXP type) {
    enum ggml_type t = (enum ggml_type)Rf_asInteger(type);

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 4));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));

    SET_STRING_ELT(names, 0, Rf_mkChar("type_name"));
    SET_STRING_ELT(names, 1, Rf_mkChar("type_size"));
    SET_STRING_ELT(names, 2, Rf_mkChar("block_size"));
    SET_STRING_ELT(names, 3, Rf_mkChar("is_quantized"));

    SET_VECTOR_ELT(result, 0, Rf_mkString(ggml_type_name(t)));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger((int)ggml_type_size(t)));
    SET_VECTOR_ELT(result, 2, Rf_ScalarInteger((int)ggml_blck_size(t)));
    SET_VECTOR_ELT(result, 3, Rf_ScalarLogical(ggml_is_quantized(t)));

    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}
