/* r_interface_gguf.c — GGUF file reading for R */

#include <R.h>
#include <Rinternals.h>
#include <string.h>
#include "ggml.h"
#include "gguf.h"

/* ── finalizer ─────────────────────────────────────────────── */

static void gguf_finalizer(SEXP ptr) {
    struct gguf_context *gf = (struct gguf_context *)R_ExternalPtrAddr(ptr);
    if (gf) {
        /* also free the ggml_context stored in the tag */
        SEXP tag = R_ExternalPtrTag(ptr);
        if (tag != R_NilValue) {
            struct ggml_context *ggml_ctx =
                (struct ggml_context *)R_ExternalPtrAddr(tag);
            if (ggml_ctx) ggml_free(ggml_ctx);
        }
        gguf_free(gf);
        R_ClearExternalPtr(ptr);
    }
}

/* ── R_gguf_load(path) ─────────────────────────────────────── */

SEXP R_gguf_load(SEXP r_path) {
    const char *path = CHAR(STRING_ELT(r_path, 0));

    struct ggml_context *ggml_ctx = NULL;
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx      = &ggml_ctx,
    };

    struct gguf_context *gf = gguf_init_from_file(path, params);
    if (!gf) {
        Rf_error("Failed to load GGUF file: %s", path);
    }

    /* wrap gguf_context as externalptr, store ggml_context in tag for cleanup */
    SEXP tag = R_NilValue;
    if (ggml_ctx) {
        tag = PROTECT(R_MakeExternalPtr(ggml_ctx, R_NilValue, R_NilValue));
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(gf, tag, R_NilValue));
    R_RegisterCFinalizerEx(ptr, gguf_finalizer, TRUE);

    UNPROTECT(ggml_ctx ? 2 : 1);
    return ptr;
}

/* ── R_gguf_free(ptr) ──────────────────────────────────────── */

SEXP R_gguf_free(SEXP r_ptr) {
    gguf_finalizer(r_ptr);
    return R_NilValue;
}

/* ── R_gguf_info(ptr) ──────────────────────────────────────── */

SEXP R_gguf_info(SEXP r_ptr) {
    struct gguf_context *gf = (struct gguf_context *)R_ExternalPtrAddr(r_ptr);
    if (!gf) Rf_error("GGUF context already freed");

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 3));
    SEXP names  = PROTECT(Rf_allocVector(STRSXP, 3));

    SET_STRING_ELT(names, 0, Rf_mkChar("version"));
    SET_STRING_ELT(names, 1, Rf_mkChar("n_tensors"));
    SET_STRING_ELT(names, 2, Rf_mkChar("n_kv"));

    SET_VECTOR_ELT(result, 0, Rf_ScalarInteger((int)gguf_get_version(gf)));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger((int)gguf_get_n_tensors(gf)));
    SET_VECTOR_ELT(result, 2, Rf_ScalarInteger((int)gguf_get_n_kv(gf)));

    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

/* ── R_gguf_metadata(ptr) ──────────────────────────────────── */

SEXP R_gguf_metadata(SEXP r_ptr) {
    struct gguf_context *gf = (struct gguf_context *)R_ExternalPtrAddr(r_ptr);
    if (!gf) Rf_error("GGUF context already freed");

    int64_t n_kv = gguf_get_n_kv(gf);
    SEXP result = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)n_kv));
    SEXP names  = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)n_kv));

    for (int64_t i = 0; i < n_kv; i++) {
        const char *key = gguf_get_key(gf, i);
        SET_STRING_ELT(names, (R_xlen_t)i, Rf_mkChar(key));

        enum gguf_type type = gguf_get_kv_type(gf, i);
        switch (type) {
            case GGUF_TYPE_UINT8:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarInteger(gguf_get_val_u8(gf, i)));
                break;
            case GGUF_TYPE_INT8:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarInteger(gguf_get_val_i8(gf, i)));
                break;
            case GGUF_TYPE_UINT16:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarInteger(gguf_get_val_u16(gf, i)));
                break;
            case GGUF_TYPE_INT16:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarInteger(gguf_get_val_i16(gf, i)));
                break;
            case GGUF_TYPE_UINT32:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarReal((double)gguf_get_val_u32(gf, i)));
                break;
            case GGUF_TYPE_INT32:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarInteger(gguf_get_val_i32(gf, i)));
                break;
            case GGUF_TYPE_FLOAT32:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarReal(gguf_get_val_f32(gf, i)));
                break;
            case GGUF_TYPE_UINT64:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarReal((double)gguf_get_val_u64(gf, i)));
                break;
            case GGUF_TYPE_INT64:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarReal((double)gguf_get_val_i64(gf, i)));
                break;
            case GGUF_TYPE_FLOAT64:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarReal(gguf_get_val_f64(gf, i)));
                break;
            case GGUF_TYPE_BOOL:
                SET_VECTOR_ELT(result, i,
                    Rf_ScalarLogical(gguf_get_val_bool(gf, i)));
                break;
            case GGUF_TYPE_STRING:
                SET_VECTOR_ELT(result, i,
                    Rf_mkString(gguf_get_val_str(gf, i)));
                break;
            case GGUF_TYPE_ARRAY: {
                enum gguf_type arr_type = gguf_get_arr_type(gf, i);
                size_t arr_n = gguf_get_arr_n(gf, i);
                if (arr_type == GGUF_TYPE_STRING) {
                    SEXP sv = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)arr_n));
                    for (size_t j = 0; j < arr_n; j++)
                        SET_STRING_ELT(sv, (R_xlen_t)j,
                            Rf_mkChar(gguf_get_arr_str(gf, i, j)));
                    SET_VECTOR_ELT(result, i, sv);
                    UNPROTECT(1);
                } else if (arr_type == GGUF_TYPE_FLOAT32) {
                    const float *d = (const float *)gguf_get_arr_data(gf, i);
                    SEXP rv = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)arr_n));
                    for (size_t j = 0; j < arr_n; j++)
                        REAL(rv)[j] = (double)d[j];
                    SET_VECTOR_ELT(result, i, rv);
                    UNPROTECT(1);
                } else if (arr_type == GGUF_TYPE_INT32) {
                    const int32_t *d = (const int32_t *)gguf_get_arr_data(gf, i);
                    SEXP iv = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t)arr_n));
                    for (size_t j = 0; j < arr_n; j++)
                        INTEGER(iv)[j] = (int)d[j];
                    SET_VECTOR_ELT(result, i, iv);
                    UNPROTECT(1);
                } else {
                    /* unsupported array type — store as string description */
                    char buf[64];
                    snprintf(buf, sizeof(buf), "<array[%zu] of %s>",
                             arr_n, gguf_type_name(arr_type));
                    SET_VECTOR_ELT(result, i, Rf_mkString(buf));
                }
                break;
            }
            default: {
                char buf[64];
                snprintf(buf, sizeof(buf), "<type %d>", (int)type);
                SET_VECTOR_ELT(result, i, Rf_mkString(buf));
                break;
            }
        }
    }

    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

/* ── R_gguf_tensor_names(ptr) ──────────────────────────────── */

SEXP R_gguf_tensor_names(SEXP r_ptr) {
    struct gguf_context *gf = (struct gguf_context *)R_ExternalPtrAddr(r_ptr);
    if (!gf) Rf_error("GGUF context already freed");

    int64_t n = gguf_get_n_tensors(gf);
    SEXP result = PROTECT(Rf_allocVector(STRSXP, (R_xlen_t)n));
    for (int64_t i = 0; i < n; i++) {
        SET_STRING_ELT(result, (R_xlen_t)i,
            Rf_mkChar(gguf_get_tensor_name(gf, i)));
    }
    UNPROTECT(1);
    return result;
}

/* ── R_gguf_tensor_info(ptr, name) ─────────────────────────── */

SEXP R_gguf_tensor_info(SEXP r_ptr, SEXP r_name) {
    struct gguf_context *gf = (struct gguf_context *)R_ExternalPtrAddr(r_ptr);
    if (!gf) Rf_error("GGUF context already freed");

    /* get ggml_context from tag */
    SEXP tag = R_ExternalPtrTag(r_ptr);
    if (tag == R_NilValue) Rf_error("No tensor data loaded (no_alloc was true?)");
    struct ggml_context *ggml_ctx =
        (struct ggml_context *)R_ExternalPtrAddr(tag);
    if (!ggml_ctx) Rf_error("ggml context already freed");

    const char *name = CHAR(STRING_ELT(r_name, 0));
    struct ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (!t) Rf_error("Tensor '%s' not found", name);

    int n_dims = ggml_n_dims(t);

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 4));
    SEXP rnames = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(rnames, 0, Rf_mkChar("name"));
    SET_STRING_ELT(rnames, 1, Rf_mkChar("shape"));
    SET_STRING_ELT(rnames, 2, Rf_mkChar("type"));
    SET_STRING_ELT(rnames, 3, Rf_mkChar("size_bytes"));

    SET_VECTOR_ELT(result, 0, Rf_mkString(ggml_get_name(t)));

    SEXP shape = PROTECT(Rf_allocVector(INTSXP, n_dims));
    for (int d = 0; d < n_dims; d++)
        INTEGER(shape)[d] = (int)t->ne[d];
    SET_VECTOR_ELT(result, 1, shape);

    SET_VECTOR_ELT(result, 2, Rf_mkString(ggml_type_name(t->type)));
    SET_VECTOR_ELT(result, 3, Rf_ScalarReal((double)ggml_nbytes(t)));

    Rf_setAttrib(result, R_NamesSymbol, rnames);
    UNPROTECT(3);
    return result;
}

/* ── R_gguf_tensor_data(ptr, name) — dequantize to f32 ─────── */

SEXP R_gguf_tensor_data(SEXP r_ptr, SEXP r_name) {
    struct gguf_context *gf = (struct gguf_context *)R_ExternalPtrAddr(r_ptr);
    if (!gf) Rf_error("GGUF context already freed");

    SEXP tag = R_ExternalPtrTag(r_ptr);
    if (tag == R_NilValue) Rf_error("No tensor data loaded");
    struct ggml_context *ggml_ctx =
        (struct ggml_context *)R_ExternalPtrAddr(tag);
    if (!ggml_ctx) Rf_error("ggml context already freed");

    const char *name = CHAR(STRING_ELT(r_name, 0));
    struct ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (!t) Rf_error("Tensor '%s' not found", name);

    int64_t n_elem = ggml_nelements(t);

    /* dequantize to float */
    float *buf = (float *)malloc(n_elem * sizeof(float));
    if (!buf) Rf_error("Out of memory for %lld floats", (long long)n_elem);

    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, n_elem * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, buf, n_elem);
    } else if (ggml_is_quantized(t->type)) {
        /* generic dequant via to_float from type traits */
        const int64_t ne0 = t->ne[0];
        const int64_t n_rows = n_elem / ne0;
        const size_t row_size = ggml_row_size(t->type, ne0);
        const struct ggml_type_traits *tt = ggml_get_type_traits(t->type);
        if (!tt->to_float) {
            free(buf);
            Rf_error("No dequantize function for type: %s", ggml_type_name(t->type));
        }
        for (int64_t r = 0; r < n_rows; r++) {
            const void *src = (const char *)t->data + r * row_size;
            tt->to_float(src, buf + r * ne0, ne0);
        }
    } else {
        free(buf);
        Rf_error("Unsupported tensor type: %s", ggml_type_name(t->type));
    }

    /* copy to R numeric vector */
    SEXP result = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)n_elem));
    for (int64_t i = 0; i < n_elem; i++)
        REAL(result)[i] = (double)buf[i];
    free(buf);

    /* set dim attribute */
    int n_dims = ggml_n_dims(t);
    SEXP dim = PROTECT(Rf_allocVector(INTSXP, n_dims));
    for (int d = 0; d < n_dims; d++)
        INTEGER(dim)[d] = (int)t->ne[d];
    Rf_setAttrib(result, R_DimSymbol, dim);

    UNPROTECT(2);
    return result;
}
