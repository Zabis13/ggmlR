/* r_onnx.c — R interface for ONNX loader
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <string.h>

#include "onnx_loader.h"
#include "onnx_ggml.h"
#include "../ggml.h"
#include "../ggml-backend.h"

/* ── Pointers + finalizers ──────────────────────────────────────── */

static void finalizer_onnx_model(SEXP ptr) {
    onnx_model_t *m = (onnx_model_t *)R_ExternalPtrAddr(ptr);
    if (m) {
        onnx_free(m);
        R_ClearExternalPtr(ptr);
    }
}

static void finalizer_onnx_ctx(SEXP ptr) {
    onnx_ggml_ctx_t *c = (onnx_ggml_ctx_t *)R_ExternalPtrAddr(ptr);
    if (c) {
        /* Free the onnx model too since ctx doesn't own it */
        if (c->onnx) onnx_free(c->onnx);
        onnx_ggml_free(c);
        R_ClearExternalPtr(ptr);
    }
}

/* ── R_onnx_load(path) ──────────────────────────────────────────── */

SEXP R_onnx_load(SEXP path_) {
    const char *path = CHAR(STRING_ELT(path_, 0));
    onnx_model_t *m = onnx_load(path);
    if (!m) {
        Rf_error("onnx_load: failed to load '%s'", path);
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(m, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, finalizer_onnx_model, TRUE);
    UNPROTECT(1);
    return ptr;
}

/* ── R_onnx_summary(model_ptr) ──────────────────────────────────── */

SEXP R_onnx_summary(SEXP model_ptr_) {
    onnx_model_t *m = (onnx_model_t *)R_ExternalPtrAddr(model_ptr_);
    if (!m) Rf_error("onnx_summary: NULL model pointer");

    /* Return a list with model info */
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 7));
    SEXP names  = PROTECT(Rf_allocVector(STRSXP, 7));

    /* ir_version */
    SET_STRING_ELT(names, 0, Rf_mkChar("ir_version"));
    SET_VECTOR_ELT(result, 0, Rf_ScalarInteger((int)m->ir_version));

    /* opset_version */
    SET_STRING_ELT(names, 1, Rf_mkChar("opset_version"));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger((int)m->opset_version));

    /* producer */
    SET_STRING_ELT(names, 2, Rf_mkChar("producer"));
    SET_VECTOR_ELT(result, 2, Rf_mkString(m->producer_name));

    /* graph_name */
    SET_STRING_ELT(names, 3, Rf_mkChar("graph_name"));
    SET_VECTOR_ELT(result, 3, Rf_mkString(m->graph_name));

    /* n_nodes */
    SET_STRING_ELT(names, 4, Rf_mkChar("n_nodes"));
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(m->n_nodes));

    /* n_initializers */
    SET_STRING_ELT(names, 5, Rf_mkChar("n_initializers"));
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(m->n_initializers));

    /* ops — unique op types used */
    /* Count unique ops */
    int n_unique = 0;
    char unique_ops[512][128];
    for (int i = 0; i < m->n_nodes; i++) {
        int found = 0;
        for (int j = 0; j < n_unique; j++) {
            if (strcmp(unique_ops[j], m->nodes[i].op_type) == 0) {
                found = 1;
                break;
            }
        }
        if (!found && n_unique < 512) {
            snprintf(unique_ops[n_unique], 128, "%s", m->nodes[i].op_type);
            n_unique++;
        }
    }
    SEXP ops_vec = PROTECT(Rf_allocVector(STRSXP, n_unique));
    for (int i = 0; i < n_unique; i++) {
        SET_STRING_ELT(ops_vec, i, Rf_mkChar(unique_ops[i]));
    }
    SET_STRING_ELT(names, 6, Rf_mkChar("ops"));
    SET_VECTOR_ELT(result, 6, ops_vec);

    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(3);
    return result;
}

/* ── R_onnx_build(model_ptr, device, n_threads, dtype) ──────────── */

SEXP R_onnx_build(SEXP model_ptr_, SEXP device_, SEXP n_threads_, SEXP dtype_) {
    onnx_model_t *m = (onnx_model_t *)R_ExternalPtrAddr(model_ptr_);
    if (!m) Rf_error("onnx_build: NULL model pointer");

    const char *device = Rf_isNull(device_) ? NULL : CHAR(STRING_ELT(device_, 0));
    int n_threads = Rf_asInteger(n_threads_);

    /* Parse dtype: "f16" → GGML_TYPE_F16, anything else → GGML_TYPE_F32 */
    enum ggml_type model_dtype = GGML_TYPE_F32;
    if (!Rf_isNull(dtype_)) {
        const char *dtype_str = CHAR(STRING_ELT(dtype_, 0));
        if (strcmp(dtype_str, "f16") == 0 || strcmp(dtype_str, "fp16") == 0 ||
            strcmp(dtype_str, "float16") == 0) {
            model_dtype = GGML_TYPE_F16;
        }
    }

    onnx_ggml_ctx_t *ctx = onnx_ggml_build(m, device, n_threads, model_dtype);
    if (!ctx) {
        Rf_error("onnx_build: failed to build ggml graph");
        return R_NilValue;
    }

    /* Transfer onnx model ownership to ctx — clear the R model pointer
     * so the finalizer won't double-free */
    R_ClearExternalPtr(model_ptr_);

    SEXP ptr = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, finalizer_onnx_ctx, TRUE);
    UNPROTECT(1);
    return ptr;
}

/* ── R_onnx_run(ctx_ptr, input_names, input_data_list) ──────────── */

SEXP R_onnx_run(SEXP ctx_ptr_, SEXP input_names_, SEXP input_data_) {
    onnx_ggml_ctx_t *ctx = (onnx_ggml_ctx_t *)R_ExternalPtrAddr(ctx_ptr_);
    if (!ctx) Rf_error("onnx_run: NULL context pointer");

    int n_inputs = Rf_length(input_names_);
    const char **names = (const char **)R_alloc(n_inputs, sizeof(char *));
    const float **data  = (const float **)R_alloc(n_inputs, sizeof(float *));

    for (int i = 0; i < n_inputs; i++) {
        names[i] = CHAR(STRING_ELT(input_names_, i));
        SEXP vec = VECTOR_ELT(input_data_, i);
        /* R REAL is double — convert to float for ggml */
        int64_t nel = Rf_length(vec);
        float *fdata = (float *)R_alloc(nel, sizeof(float));
        double *rdata = REAL(vec);
        for (int64_t j = 0; j < nel; j++)
            fdata[j] = (float)rdata[j];
        data[i] = fdata;
    }

    int status = onnx_ggml_run(ctx, names, data, n_inputs);
    if (status != 0) {
        Rf_error("onnx_run: inference failed");
    }

    /* Collect outputs */
    int n_outputs = ctx->onnx->n_outputs;
    SEXP result = PROTECT(Rf_allocVector(VECSXP, n_outputs));
    SEXP out_names = PROTECT(Rf_allocVector(STRSXP, n_outputs));

    for (int i = 0; i < n_outputs; i++) {
        struct ggml_tensor *t = onnx_ggml_output(ctx, i);
        SET_STRING_ELT(out_names, i, Rf_mkChar(ctx->onnx->outputs[i].name));

        if (t) {
            int64_t nel = ggml_nelements(t);
            SEXP vec = PROTECT(Rf_allocVector(REALSXP, nel));
            float *buf = (float *)R_alloc(nel, sizeof(float));
            ggml_backend_tensor_get(t, buf, 0, nel * sizeof(float));
            double *rdata = REAL(vec);
            for (int64_t j = 0; j < nel; j++) {
                rdata[j] = (double)buf[j];
            }

            /* Set dim attribute in ONNX order (reverse ggml ne[]) */
            int ndims = ggml_n_dims(t);
            if (ndims > 1) {
                SEXP dim = PROTECT(Rf_allocVector(INTSXP, ndims));
                for (int d = 0; d < ndims; d++) {
                    INTEGER(dim)[d] = (int)t->ne[ndims - 1 - d];
                }
                Rf_setAttrib(vec, R_DimSymbol, dim);
                UNPROTECT(1);
            }

            SET_VECTOR_ELT(result, i, vec);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(result, i, R_NilValue);
        }
    }

    Rf_setAttrib(result, R_NamesSymbol, out_names);
    UNPROTECT(2);
    return result;
}

/* ── R_onnx_override_input_shapes(model_ptr, names, shapes) ─────── */

SEXP R_onnx_override_input_shapes(SEXP model_ptr_, SEXP names_, SEXP shapes_) {
    onnx_model_t *m = (onnx_model_t *)R_ExternalPtrAddr(model_ptr_);
    if (!m) Rf_error("onnx_override_input_shapes: NULL model pointer");

    int n = Rf_length(names_);
    for (int k = 0; k < n; k++) {
        const char *name = CHAR(STRING_ELT(names_, k));
        SEXP shape = VECTOR_ELT(shapes_, k);
        int ndims = Rf_length(shape);

        /* Find this input in model */
        int found = 0;
        for (int i = 0; i < m->n_inputs; i++) {
            if (strcmp(m->inputs[i].name, name) == 0) {
                /* Override dims */
                m->inputs[i].n_dims = ndims > ONNX_MAX_DIMS ? ONNX_MAX_DIMS : ndims;
                for (int d = 0; d < m->inputs[i].n_dims; d++) {
                    m->inputs[i].dims[d] = (int64_t)INTEGER(shape)[d];
                }
                found = 1;
                break;
            }
        }
        if (!found) {
            Rf_warning("onnx_load: input_shapes contains unknown input '%s'", name);
        }
    }
    return R_NilValue;
}

/* ── R_onnx_inputs(ctx_ptr) — list model inputs info ────────────── */

SEXP R_onnx_inputs(SEXP ctx_ptr_) {
    onnx_ggml_ctx_t *ctx = (onnx_ggml_ctx_t *)R_ExternalPtrAddr(ctx_ptr_);
    if (!ctx) Rf_error("onnx_inputs: NULL context pointer");

    /* Filter out inputs that are initializers */
    int n = 0;
    for (int i = 0; i < ctx->onnx->n_inputs; i++) {
        if (!onnx_find_initializer(ctx->onnx, ctx->onnx->inputs[i].name))
            n++;
    }

    SEXP result = PROTECT(Rf_allocVector(VECSXP, n));
    SEXP names  = PROTECT(Rf_allocVector(STRSXP, n));
    int idx = 0;
    for (int i = 0; i < ctx->onnx->n_inputs; i++) {
        onnx_value_info_t *vi = &ctx->onnx->inputs[i];
        if (onnx_find_initializer(ctx->onnx, vi->name)) continue;

        SET_STRING_ELT(names, idx, Rf_mkChar(vi->name));

        /* Shape vector from ggml tensor (has resolved dynamic dims → 1).
         * Return in ONNX order (reversed from ggml ne[]). */
        struct ggml_tensor *t = NULL;
        for (int k = ctx->tensor_map_size - 1; k >= 0; k--) {
            if (strcmp(ctx->tensor_map_keys[k], vi->name) == 0) {
                t = ctx->tensor_map_vals[k]; break;
            }
        }
        int nd = vi->n_dims;
        SEXP shape = PROTECT(Rf_allocVector(INTSXP, nd));
        if (t) {
            for (int d = 0; d < nd; d++)
                INTEGER(shape)[d] = (int)t->ne[nd - 1 - d];
        } else {
            for (int d = 0; d < nd; d++)
                INTEGER(shape)[d] = (int)vi->dims[d];
        }
        SET_VECTOR_ELT(result, idx, shape);
        UNPROTECT(1);
        idx++;
    }

    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

/* ── R_onnx_device_info(ctx_ptr) — scheduler/backend diagnostics ─── */

SEXP R_onnx_device_info(SEXP ctx_ptr_) {
    onnx_ggml_ctx_t *ctx = (onnx_ggml_ctx_t *)R_ExternalPtrAddr(ctx_ptr_);
    if (!ctx) Rf_error("onnx_device_info: NULL context pointer");
    if (!ctx->graph) Rf_error("onnx_device_info: NULL graph");

    /* 8 fields: backends, n_backends, n_splits, n_nodes, gpu_ops, cpu_ops, cpu_only_ops, actual_backend */
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 8));
    SEXP names  = PROTECT(Rf_allocVector(STRSXP, 8));

    /* backends — character vector of backend names */
    int n_be = ctx->backend_gpu ? 2 : 1;
    SEXP be_vec = PROTECT(Rf_allocVector(STRSXP, n_be));
    int be_idx = 0;
    if (ctx->backend_gpu)
        SET_STRING_ELT(be_vec, be_idx++, Rf_mkChar(ggml_backend_name(ctx->backend_gpu)));
    SET_STRING_ELT(be_vec, be_idx, Rf_mkChar(ggml_backend_name(ctx->backend_cpu)));
    SET_STRING_ELT(names, 0, Rf_mkChar("backends"));
    SET_VECTOR_ELT(result, 0, be_vec);
    UNPROTECT(1); /* be_vec */

    /* n_backends */
    SET_STRING_ELT(names, 1, Rf_mkChar("n_backends"));
    SET_VECTOR_ELT(result, 1, Rf_ScalarInteger(n_be));

    /* n_splits */
    SET_STRING_ELT(names, 2, Rf_mkChar("n_splits"));
    SET_VECTOR_ELT(result, 2, Rf_ScalarInteger(
        ctx->sched ? ggml_backend_sched_get_n_splits(ctx->sched) : 0));

    /* n_nodes */
    int n_nodes = ggml_graph_n_nodes(ctx->graph);
    SET_STRING_ELT(names, 3, Rf_mkChar("n_nodes"));
    SET_VECTOR_ELT(result, 3, Rf_ScalarInteger(n_nodes));

    /* gpu_ops, cpu_ops, cpu_only_ops */
    int n_gpu = 0, n_cpu = 0;

    /* Collect unique CPU-only op names */
    char cpu_op_names[128][64];
    int cpu_op_counts[128];
    int n_unique_cpu_ops = 0;

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor *node = ggml_graph_node(ctx->graph, i);
        if (ctx->backend_gpu && ggml_backend_supports_op(ctx->backend_gpu, node)) {
            n_gpu++;
        } else {
            n_cpu++;
            const char *opname = ggml_op_name(node->op);
            int found = 0;
            for (int j = 0; j < n_unique_cpu_ops; j++) {
                if (strcmp(cpu_op_names[j], opname) == 0) {
                    cpu_op_counts[j]++;
                    found = 1;
                    break;
                }
            }
            if (!found && n_unique_cpu_ops < 128) {
                snprintf(cpu_op_names[n_unique_cpu_ops], 64, "%s", opname);
                cpu_op_counts[n_unique_cpu_ops] = 1;
                n_unique_cpu_ops++;
            }
        }
    }

    SET_STRING_ELT(names, 4, Rf_mkChar("gpu_ops"));
    SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(n_gpu));

    SET_STRING_ELT(names, 5, Rf_mkChar("cpu_ops"));
    SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(n_cpu));

    /* cpu_only_ops — named integer vector: op_name => count */
    SEXP cpu_vec = PROTECT(Rf_allocVector(INTSXP, n_unique_cpu_ops));
    SEXP cpu_names = PROTECT(Rf_allocVector(STRSXP, n_unique_cpu_ops));
    for (int i = 0; i < n_unique_cpu_ops; i++) {
        INTEGER(cpu_vec)[i] = cpu_op_counts[i];
        SET_STRING_ELT(cpu_names, i, Rf_mkChar(cpu_op_names[i]));
    }
    Rf_setAttrib(cpu_vec, R_NamesSymbol, cpu_names);
    SET_STRING_ELT(names, 6, Rf_mkChar("cpu_only_ops"));
    SET_VECTOR_ELT(result, 6, cpu_vec);
    UNPROTECT(2); /* cpu_vec, cpu_names */

    /* actual_backend — check where the last graph node's buffer actually lives */
    {
        const char *actual = "unknown";
        struct ggml_tensor *last = n_nodes > 0 ? ggml_graph_node(ctx->graph, n_nodes - 1) : NULL;
        if (last && last->buffer) {
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(last->buffer);
            const char *buft_name = ggml_backend_buft_name(buft);
            actual = buft_name ? buft_name : "unknown";
        } else if (last && !last->buffer) {
            actual = "no_buffer";
        }
        SET_STRING_ELT(names, 7, Rf_mkChar("actual_backend"));
        SET_VECTOR_ELT(result, 7, Rf_mkString(actual));
    }

    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2); /* result, names */
    return result;
}
