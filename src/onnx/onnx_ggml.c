/* onnx_ggml.c — Map ONNX ops to ggml ops and run inference
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ggml.h"
#include "onnx_ops_internal.h"
#include "../ggml.h"
#include "../ggml-alloc.h"
#include "../ggml-backend.h"
#include "../ggml-cpu.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>


/* Check if Vulkan is available at compile time */
#ifdef GGML_USE_VULKAN
#include "../ggml-vulkan.h"
#endif

/* ── Tensor name map ────────────────────────────────────────────── */

void tmap_put_nd(onnx_ggml_ctx_t *c, const char *name,
                        struct ggml_tensor *t, int onnx_ndims) {
    if (c->tensor_map_size >= c->tensor_map_cap) {
        c->tensor_map_cap = c->tensor_map_cap ? c->tensor_map_cap * 2 : 256;
        c->tensor_map_keys = realloc(c->tensor_map_keys,
                                      c->tensor_map_cap * sizeof(*c->tensor_map_keys));
        c->tensor_map_vals = realloc(c->tensor_map_vals,
                                      c->tensor_map_cap * sizeof(*c->tensor_map_vals));
        c->tensor_map_ndims = realloc(c->tensor_map_ndims,
                                       c->tensor_map_cap * sizeof(*c->tensor_map_ndims));
        c->tensor_map_onnx_ne = realloc(c->tensor_map_onnx_ne,
                                         c->tensor_map_cap * sizeof(*c->tensor_map_onnx_ne));
    }
    int idx = c->tensor_map_size;
    strncpy(c->tensor_map_keys[idx], name, ONNX_MAX_NAME - 1);
    c->tensor_map_keys[idx][ONNX_MAX_NAME - 1] = '\0';
    c->tensor_map_vals[idx] = t;
    c->tensor_map_ndims[idx] = onnx_ndims;
    /* Default: reconstruct ONNX shape from ggml ne (reversed, ≤5D) */
    memset(c->tensor_map_onnx_ne[idx], 0, sizeof(c->tensor_map_onnx_ne[idx]));
    int nd = onnx_ndims < GGML_MAX_DIMS ? onnx_ndims : GGML_MAX_DIMS;
    for (int d = 0; d < nd; d++)
        c->tensor_map_onnx_ne[idx][d] = t->ne[nd - 1 - d];
    c->tensor_map_size++;
}

/* Store explicit ONNX shape (for Reshape, Expand etc. where >4D is collapsed) */
void tmap_put_shape(onnx_ggml_ctx_t *c, const char *name,
                           struct ggml_tensor *t, const int64_t *onnx_shape, int onnx_ndims) {
    tmap_put_nd(c, name, t, onnx_ndims);
    /* Overwrite the auto-generated shape with the explicit one */
    int idx = c->tensor_map_size - 1;
    memset(c->tensor_map_onnx_ne[idx], 0, sizeof(c->tensor_map_onnx_ne[idx]));
    int nd = onnx_ndims < ONNX_MAX_DIMS ? onnx_ndims : ONNX_MAX_DIMS;
    for (int d = 0; d < nd; d++)
        c->tensor_map_onnx_ne[idx][d] = onnx_shape[d];
}

/* Get full ONNX shape. Returns ndims, fills shape[]. */
int tmap_get_shape(onnx_ggml_ctx_t *c, const char *name,
                          int64_t *shape, int max_dims) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0) {
            int nd = c->tensor_map_ndims[i];
            if (nd > max_dims) nd = max_dims;
            memcpy(shape, c->tensor_map_onnx_ne[i], nd * sizeof(int64_t));
            return nd;
        }
    }
    return 0;
}

void tmap_put(onnx_ggml_ctx_t *c, const char *name, struct ggml_tensor *t) {
    tmap_put_nd(c, name, t, ggml_n_dims(t));
}

int tmap_get_ndims(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0)
            return c->tensor_map_ndims[i];
    }
    return 4;
}

struct ggml_tensor *tmap_get(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0)
            return c->tensor_map_vals[i];
    }
    return NULL;
}

/* Helper: squeeze trailing unit dims (5D→4D when ne[4]==1, etc.)
 * Keeps ggml tensors compact; real ONNX ndims tracked via tmap out_nd. */
int onnx_squeeze_ndims(const int64_t *ne, int ndims) {
    while (ndims > 1 && ne[ndims - 1] == 1) ndims--;
    return ndims;
}

/* Helper: reshape tensor to given ne[] with appropriate ndims.
 * Squeezes trailing 1s for ggml compatibility. */
struct ggml_tensor *onnx_reshape_nd(struct ggml_context *ctx,
                                           struct ggml_tensor *a,
                                           const int64_t *ne, int ndims) {
    ndims = onnx_squeeze_ndims(ne, ndims);
    switch (ndims) {
        case 1: return ggml_reshape_1d(ctx, a, ne[0]);
        case 2: return ggml_reshape_2d(ctx, a, ne[0], ne[1]);
        case 3: return ggml_reshape_3d(ctx, a, ne[0], ne[1], ne[2]);
        case 4: return ggml_reshape_4d(ctx, a, ne[0], ne[1], ne[2], ne[3]);
        default: return ggml_reshape_5d(ctx, a, ne[0], ne[1], ne[2], ne[3], ne[4]);
    }
}

/* Helper: create new tensor with given ne[] and appropriate ndims. */
struct ggml_tensor *onnx_new_tensor_nd(struct ggml_context *ctx,
                                              enum ggml_type type,
                                              const int64_t *ne, int ndims) {
    ndims = onnx_squeeze_ndims(ne, ndims);
    switch (ndims) {
        case 1: return ggml_new_tensor_1d(ctx, type, ne[0]);
        case 2: return ggml_new_tensor_2d(ctx, type, ne[0], ne[1]);
        case 3: return ggml_new_tensor_3d(ctx, type, ne[0], ne[1], ne[2]);
        case 4: return ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        default: return ggml_new_tensor_5d(ctx, type, ne[0], ne[1], ne[2], ne[3], ne[4]);
    }
}

/* Helper: product of ne[0..ndims-1] */
int64_t ne_product(const int64_t *ne, int ndims) {
    int64_t p = 1;
    for (int d = 0; d < ndims; d++) p *= ne[d];
    return p;
}

/* ── Compile-time value map (for shape propagation) ─────────────── */


void cval_put(onnx_ggml_ctx_t *c, const char *name,
                     const int64_t *vals, int n) {
    if (n > ONNX_MAX_DIMS) n = ONNX_MAX_DIMS;
    if (c->cval_size >= c->cval_cap) {
        c->cval_cap = c->cval_cap ? c->cval_cap * 2 : 256;
        c->cval_keys = realloc(c->cval_keys, c->cval_cap * sizeof(*c->cval_keys));
        c->cval_data = realloc(c->cval_data, c->cval_cap * sizeof(*c->cval_data));
        c->cval_lens = realloc(c->cval_lens, c->cval_cap * sizeof(*c->cval_lens));
    }
    strncpy(c->cval_keys[c->cval_size], name, ONNX_MAX_NAME - 1);
    c->cval_keys[c->cval_size][ONNX_MAX_NAME - 1] = '\0';
    memcpy(c->cval_data[c->cval_size], vals, n * sizeof(int64_t));
    c->cval_lens[c->cval_size] = n;
    c->cval_size++;
}

int cval_get(onnx_ggml_ctx_t *c, const char *name,
                    int64_t *out, int max_n) {
    for (int i = c->cval_size - 1; i >= 0; i--) {
        if (strcmp(c->cval_keys[i], name) == 0) {
            int n = c->cval_lens[i];
            if (n > max_n) n = max_n;
            memcpy(out, c->cval_data[i], n * sizeof(int64_t));
            return n;
        }
    }
    return 0;
}

/* ── Find Constant node's tensor by output name ────────────────── */

const onnx_initializer_t *find_constant_tensor(const onnx_model_t *m,
                                                        const char *name) {
    for (int i = 0; i < m->n_nodes; i++) {
        const onnx_node_t *nd = &m->nodes[i];
        if (strcmp(nd->op_type, "Constant") != 0) continue;
        if (nd->n_outputs > 0 && strcmp(nd->outputs[0], name) == 0) {
            const onnx_attr_t *va = onnx_node_find_attr(nd, "value");
            if (va && va->tensor) return va->tensor;
        }
    }
    return NULL;
}

/* ── Create a scalar constant tensor (no_alloc safe) ────────────── */
/* Returns a 1-element f32 tensor in ctx_weight, value filled during build. */
struct ggml_tensor *make_scalar(onnx_ggml_ctx_t *c, float val) {
    struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
    struct ggml_tensor *t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 1);
    ggml_set_input(t);
    if (c->n_const_fills < ONNX_MAX_DEFERRED) {
        c->const_fill_ptrs[c->n_const_fills] = t;
        c->const_fill_vals[c->n_const_fills] = val;
        c->n_const_fills++;
    }
    return t;
}

/* ── ONNX dtype → ggml type ─────────────────────────────────────── */

enum ggml_type onnx_dtype_to_ggml(int32_t dt) {
    switch (dt) {
        case ONNX_DTYPE_FLOAT:    return GGML_TYPE_F32;
        case ONNX_DTYPE_FLOAT16:  return GGML_TYPE_F16;
        case ONNX_DTYPE_BFLOAT16: return GGML_TYPE_BF16;
        case ONNX_DTYPE_INT32:    return GGML_TYPE_I32;
        case ONNX_DTYPE_INT64:    return GGML_TYPE_I32; /* downcast to i32 */
        case ONNX_DTYPE_DOUBLE:   return GGML_TYPE_F32; /* downcast to f32 */
        default:                  return GGML_TYPE_F32;
    }
}

/* ── Size of ONNX data type in bytes ────────────────────────────── */

size_t onnx_dtype_size(int32_t dt) {
    switch (dt) {
        case ONNX_DTYPE_FLOAT:    return 4;
        case ONNX_DTYPE_DOUBLE:   return 8;
        case ONNX_DTYPE_FLOAT16:  return 2;
        case ONNX_DTYPE_BFLOAT16: return 2;
        case ONNX_DTYPE_INT32:    return 4;
        case ONNX_DTYPE_INT64:    return 8;
        case ONNX_DTYPE_INT8:     return 1;
        case ONNX_DTYPE_UINT8:    return 1;
        case ONNX_DTYPE_INT16:    return 2;
        case ONNX_DTYPE_BOOL:     return 1;
        default:                  return 4;
    }
}

/* ── Create ggml tensors for initializers (weights) ─────────────── */

static int create_initializer_tensors(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_initializers; i++) {
        onnx_initializer_t *init = &c->onnx->initializers[i];
        enum ggml_type type = onnx_dtype_to_ggml(init->data_type);

        /* Reverse ONNX dims → ggml ne[].
         * ONNX is row-major: dims[0]=outermost (batch/OC), dims[last]=innermost.
         * ggml is column-major: ne[0]=innermost, ne[last]=outermost.
         * So ne[i] = dims[ndims-1-i]. */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        int ndims = init->n_dims;
        if (ndims > GGML_MAX_DIMS) ndims = GGML_MAX_DIMS;
        if (ndims == 0) {
            /* Scalar initializer */
            ndims = 1;
            ne[0] = 1;
        } else {
            for (int d = 0; d < ndims; d++)
                ne[d] = init->dims[ndims - 1 - d];
        }

        /* FP16 promotion: convert large F32 weight tensors to F16 for
         * faster Vulkan compute.
         * ndims >= 2: Conv (4D) and Linear/MatMul (2D) weights → F16.
         * ndims == 1: bias, BN/LN gamma/beta/mean/var — kept F32 (precision-sensitive).
         * Small tensors (< ONNX_FP16_MIN_ELEMENTS) and INT types are never converted. */
        int64_t n_elem = 1;
        for (int d = 0; d < ndims; d++) n_elem *= ne[d];
        if (c->model_dtype == GGML_TYPE_F16 &&
            type == GGML_TYPE_F32 &&
            init->n_dims >= 2 &&
            n_elem >= ONNX_FP16_MIN_ELEMENTS) {
            type = GGML_TYPE_F16;
        }

        /* Allocate weight tensors in ctx_weight so they get a dedicated
         * buffer that the scheduler never touches or aliases. */
        struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
        struct ggml_tensor *t = onnx_new_tensor_nd(wctx, type, ne, ndims);
        if (!t) return -1;
        ggml_set_name(t, init->name);
        ggml_set_input(t);
        if (onnx_trace_nodes() && init->n_dims == 0)
            fprintf(stderr, "[rank0] initializer %s (scalar, stored as nd1)\n", init->name);
        tmap_put_nd(c, init->name, t, init->n_dims > 0 ? init->n_dims : 1);
        (void)0;

        /* Register cval for small initializers (shape constants, indices, scalars).
         * Supports INT64, INT32, and F32 (cast to int64 for cval map). */
        if (n_elem <= ONNX_MAX_DIMS) {
            const void *src = init->raw_data ? init->raw_data
                            : init->decoded_data ? init->decoded_data : NULL;
            if (src) {
                int64_t vals[ONNX_MAX_DIMS];
                if (init->data_type == ONNX_DTYPE_INT64) {
                    memcpy(vals, src, (size_t)n_elem * sizeof(int64_t));
                    cval_put(c, init->name, vals, (int)n_elem);
                } else if (init->data_type == ONNX_DTYPE_INT32) {
                    int32_t tmp_i32;
                    for (int64_t j = 0; j < n_elem; j++) {
                        memcpy(&tmp_i32, (const char *)src + j * sizeof(int32_t), sizeof(int32_t));
                        vals[j] = (int64_t)tmp_i32;
                    }
                    cval_put(c, init->name, vals, (int)n_elem);
                } else if (init->data_type == ONNX_DTYPE_FLOAT) {
                    float tmp_f32;
                    for (int64_t j = 0; j < n_elem; j++) {
                        memcpy(&tmp_f32, (const char *)src + j * sizeof(float), sizeof(float));
                        vals[j] = (int64_t)tmp_f32;
                    }
                    cval_put(c, init->name, vals, (int)n_elem);
                }
            }
        }
    }
    return 0;
}

/* ── Load weight data into tensors ──────────────────────────────── */

/* Upload one initializer's payload into its tensor, converting dtype on the
 * way (ONNX carries types ggml has no equivalent for -- INT64 indices, doubles
 * -- and downcasts them here).
 *
 * Split out of load_weights() so that Constant nodes can reuse it: their data
 * lives in a node attribute, not in graph.initializer, so they never appear in
 * the array load_weights() walks, yet they need exactly this conversion. */
int onnx_upload_initializer(struct ggml_tensor *t,
                            const onnx_initializer_t *init) {
    const void *data = NULL;
    size_t data_size = 0;

    if (init->raw_data && init->raw_size > 0) {
        data = init->raw_data;
        data_size = init->raw_size;
    } else if (init->decoded_data && init->decoded_size > 0) {
        data = init->decoded_data;
        data_size = init->decoded_size;
    }

    if (data && data_size > 0) {
        size_t tsize = ggml_nbytes(t);

        /* Sanity check: raw_data size vs expected from ONNX dtype */
        size_t expected = (size_t)ggml_nelements(t) * onnx_dtype_size(init->data_type);
        if (data_size < expected && init->data_type != ONNX_DTYPE_INT8 &&
            init->data_type != ONNX_DTYPE_UINT8) {
            fprintf(stderr, "ONNX WARNING: initializer '%s' raw_data %llu bytes "
                    "< expected %llu (dtype %d, nel %lld)\n",
                    init->name[0] ? init->name : "?",
                    (unsigned long long)data_size, (unsigned long long)expected,
                    init->data_type, (long long)ggml_nelements(t));
        }

        /* With reversed dims, ONNX row-major data maps directly to ggml
         * column-major layout — no transposition needed. */

        /* INT8/UINT8 → F32 conversion: raw bytes are 1-byte ints,
         * but ggml tensor is F32 (4 bytes per element) */
        if ((init->data_type == ONNX_DTYPE_INT8 ||
             init->data_type == ONNX_DTYPE_UINT8) &&
            t->type == GGML_TYPE_F32) {
            int64_t n_elem = ggml_nelements(t);
            size_t src_elems = data_size; /* 1 byte per element */
            if ((int64_t)src_elems > n_elem) src_elems = (size_t)n_elem;
            float *buf = (float *)malloc(n_elem * sizeof(float));
            if (!buf) return -1;
            const uint8_t *src = (const uint8_t *)data;
            if (init->data_type == ONNX_DTYPE_INT8) {
                for (size_t j = 0; j < src_elems; j++)
                    buf[j] = (float)((int8_t)src[j]);
            } else {
                for (size_t j = 0; j < src_elems; j++)
                    buf[j] = (float)src[j];
            }
            for (size_t j = src_elems; j < (size_t)n_elem; j++)
                buf[j] = 0.0f;
            ggml_backend_tensor_set(t, buf, 0, n_elem * sizeof(float));
            free(buf);
        }
        /* INT64 → I32 downcast */
        else if (init->data_type == ONNX_DTYPE_INT64 &&
                 t->type == GGML_TYPE_I32) {
            int64_t n_elem = ggml_nelements(t);
            size_t src_elems = data_size / 8;
            if ((int64_t)src_elems > n_elem) src_elems = (size_t)n_elem;
            int32_t *buf = (int32_t *)malloc(n_elem * sizeof(int32_t));
            if (!buf) return -1;
            int64_t tmp_i64;
            for (size_t j = 0; j < src_elems; j++) {
                memcpy(&tmp_i64, (const char *)data + j * sizeof(int64_t), sizeof(int64_t));
                buf[j] = (int32_t)tmp_i64;
            }
            for (size_t j = src_elems; j < (size_t)n_elem; j++)
                buf[j] = 0;
            ggml_backend_tensor_set(t, buf, 0, n_elem * sizeof(int32_t));
            free(buf);
        }
        /* DOUBLE → F32 downcast */
        else if (init->data_type == ONNX_DTYPE_DOUBLE &&
                 t->type == GGML_TYPE_F32) {
            int64_t n_elem = ggml_nelements(t);
            size_t src_elems = data_size / 8;
            if ((int64_t)src_elems > n_elem) src_elems = (size_t)n_elem;
            float *buf = (float *)malloc(n_elem * sizeof(float));
            if (!buf) return -1;
            double tmp_f64;
            for (size_t j = 0; j < src_elems; j++) {
                memcpy(&tmp_f64, (const char *)data + j * sizeof(double), sizeof(double));
                buf[j] = (float)tmp_f64;
            }
            for (size_t j = src_elems; j < (size_t)n_elem; j++)
                buf[j] = 0.0f;
            ggml_backend_tensor_set(t, buf, 0, n_elem * sizeof(float));
            free(buf);
        }
        /* F32 source data → F16 tensor (FP16 inference mode) */
        else if (init->data_type == ONNX_DTYPE_FLOAT &&
                 t->type == GGML_TYPE_F16) {
            int64_t n_elem = ggml_nelements(t);
            size_t src_elems = data_size / sizeof(float);
            if ((int64_t)src_elems > n_elem) src_elems = (size_t)n_elem;
            ggml_fp16_t *buf = (ggml_fp16_t *)malloc(n_elem * sizeof(ggml_fp16_t));
            if (!buf) return -1;
            float *aligned_f32 = (float *)malloc(src_elems * sizeof(float));
            if (!aligned_f32) { free(buf); return -1; }
            memcpy(aligned_f32, data, src_elems * sizeof(float));
            ggml_fp32_to_fp16_row(aligned_f32, buf, (int64_t)src_elems);
            free(aligned_f32);
            /* Zero-fill any remaining elements */
            for (size_t j = src_elems; j < (size_t)n_elem; j++)
                buf[j] = ggml_fp32_to_fp16(0.0f);
            ggml_backend_tensor_set(t, buf, 0, n_elem * sizeof(ggml_fp16_t));
            free(buf);
        }
        else {
            size_t copy_size = data_size < tsize ? data_size : tsize;
            ggml_backend_tensor_set(t, data, 0, copy_size);
        }
    }
    return 0;
}

static int load_weights(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_initializers; i++) {
        onnx_initializer_t *init = &c->onnx->initializers[i];
        struct ggml_tensor *t = tmap_get(c, init->name);
        if (!t) continue;
        /* Skip tensors not in the graph (e.g. shape constants for Reshape) —
         * they have no buffer allocated by the scheduler */
        if (!t->buffer) continue;

        if (onnx_upload_initializer(t, init) != 0) return -1;
    }
    return 0;
}

/* ── Create input placeholder tensors ───────────────────────────── */

static int create_input_tensors(onnx_ggml_ctx_t *c) {
    struct ggml_context *ictx = c->ctx;

    for (int i = 0; i < c->onnx->n_inputs; i++) {
        onnx_value_info_t *vi = &c->onnx->inputs[i];
        /* Skip if already created as initializer */
        if (tmap_get(c, vi->name)) continue;

        enum ggml_type type = onnx_dtype_to_ggml(vi->elem_type);
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        if (onnx_trace_nodes() && vi->n_dims == 0)
            fprintf(stderr, "[rank0] value_info %s (scalar, stored as nd1)\n", vi->name);
        int ndims = vi->n_dims > 0 ? vi->n_dims : 1;
        if (ndims > GGML_MAX_DIMS) ndims = GGML_MAX_DIMS;
        /* Reverse ONNX dims → ggml ne[] (row-major → column-major) */
        for (int d = 0; d < ndims; d++) {
            int64_t dim = vi->dims[ndims - 1 - d];
            if (dim <= 0) dim = 1; /* symbolic/dynamic dim → default 1 */
            ne[d] = dim;
        }

        struct ggml_tensor *t = onnx_new_tensor_nd(ictx, type, ne, ndims);
        if (!t) return -1;
        ggml_set_name(t, vi->name);
        ggml_set_input(t);
        tmap_put_nd(c, vi->name, t, vi->n_dims > 0 ? vi->n_dims : 1);
    }
    return 0;
}

/* ── Map ONNX node → ggml op ────────────────────────────────────── */

struct ggml_tensor *get_input(onnx_ggml_ctx_t *c, const onnx_node_t *n, int idx) {
    if (idx >= n->n_inputs) return NULL;
    if (n->inputs[idx][0] == '\0') return NULL; /* optional empty input */
    return tmap_get(c, n->inputs[idx]);
}

/* Current node being processed — for diagnostic messages */
const onnx_node_t *g_current_node = NULL;

/* Context of the node being mapped -- lets helpers without a ctx parameter
 * (onnx_broadcast_prepare) look up ONNX ranks for diagnostics. */
onnx_ggml_ctx_t *g_current_ctx = NULL;

/* Runtime gate for per-node graph tracing (ONNX_TRACE_NODES=1). */
int onnx_trace_nodes(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ONNX_TRACE_NODES");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

/* Segmented execution for data-dependent shapes, gated by ONNX_SEGMENTS.
 * Default ON; ONNX_SEGMENTS=0 forces the original single-pass path so that
 * the two can be compared on one build.  A model with no data-dependent op
 * takes the single-pass path either way -- the gate only decides what
 * happens when the pre-pass DID find cut points. */
int onnx_use_segments(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ONNX_SEGMENTS");
        cached = (e && *e && *e == '0') ? 0 : 1;
    }
    return cached;
}

/* Report an op the builder has no handler for.  A model can contain dozens of
 * nodes of the same unsupported type -- MaskRCNN alone hits TopK, ReduceMin,
 * Less, Not and And 44 times -- so each op type is reported only once.  The
 * list is reset per model load by onnx_reset_unsupported_warnings(). */
#define ONNX_MAX_WARNED_OPS 64
static char g_warned_ops[ONNX_MAX_WARNED_OPS][ONNX_MAX_NAME];
static int  g_n_warned_ops = 0;

void onnx_reset_unsupported_warnings(void) {
    g_n_warned_ops = 0;
}

void onnx_warn_unsupported_op(const char *op) {
    for (int i = 0; i < g_n_warned_ops; i++)
        if (strcmp(g_warned_ops[i], op) == 0) return;
    if (g_n_warned_ops < ONNX_MAX_WARNED_OPS) {
        strncpy(g_warned_ops[g_n_warned_ops], op, ONNX_MAX_NAME - 1);
        g_warned_ops[g_n_warned_ops][ONNX_MAX_NAME - 1] = '\0';
        g_n_warned_ops++;
    }
    fprintf(stderr, "onnx_ggml: unsupported op '%s'\n", op);
}
/* Broadcast compatibility, in ggml axis order: every axis of `b` must either
 * be 1 or match `a`.  Shared by every path in onnx_broadcast_prepare so the
 * paths cannot drift apart on what "broadcastable" means. */
static int bcast_fits(const int64_t *b_ne, const int64_t *a_ne) {
    for (int d = 0; d < GGML_MAX_DIMS; d++)
        if (b_ne[d] != 1 && b_ne[d] != a_ne[d]) return 0;
    return 1;
}


/* ── Broadcast helper for binary ops ────────────────────────────── */
/* Reshape b so that it is broadcastable into a (ggml requires b->ne[d] == 1 or a->ne[d]).
 * Returns b (possibly reshaped). If a and b need swapping, sets *swapped=1. */
void onnx_broadcast_prepare(struct ggml_context *ctx,
                                    struct ggml_tensor **pa,
                                    struct ggml_tensor **pb) {
    struct ggml_tensor *a = *pa, *b = *pb;

    /* Swap so that a has more (or equal) elements — ggml requires a >= b */
    if (ggml_nelements(a) < ggml_nelements(b)) {
        struct ggml_tensor *tmp = a; a = b; b = tmp;
        *pa = a; *pb = b;
    }

    /* Check if b is already broadcastable into a */
    int ok = bcast_fits(b->ne, a->ne);
    if (ok) {
        if (onnx_trace_nodes() && g_current_node && g_current_ctx) {
            int nd_a = -1, nd_b = -1;
            if (g_current_node->n_inputs > 0)
                nd_a = tmap_get_ndims(g_current_ctx, g_current_node->inputs[0]);
            if (g_current_node->n_inputs > 1)
                nd_b = tmap_get_ndims(g_current_ctx, g_current_node->inputs[1]);
            fprintf(stderr, "[bcast] %s op=%s a.ne=[%lld,%lld,%lld,%lld] b.ne=[%lld,%lld,%lld,%lld] onnx_nd=(%d,%d) -> passthrough\n",
                    g_current_node->outputs[0], g_current_node->op_type,
                    (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                    (long long)b->ne[0],(long long)b->ne[1],(long long)b->ne[2],(long long)b->ne[3],
                    nd_a, nd_b);
        }
        return;
    }

    /* ONNX broadcast: numpy-style, right-aligned in ONNX dim order.
     * ggml dims are reversed vs ONNX, so ONNX right-align = ggml left-align (dim 0).
     *
     * b's non-trivial dims (those != 1) must be placed to match a's dims,
     * left-aligned in ggml order. b may have fewer dims than a — the extra
     * higher dims get padded with 1.
     *
     * Example (ggml order):
     *   a = [W, H, C, N]   (4D)
     *   b = [1, 1, C]      (3D, ggml_n_dims sees it as 3 or less)
     *   → reshape b to [1, 1, C, 1] → broadcast OK
     *
     * But ggml_n_dims drops trailing 1s, so b=[C] when originally [C,1,1].
     * We need to figure out which dim of a each dim of b corresponds to.
     *
     * Strategy: b's dim 0 aligns with a's dim 0, dim 1 with dim 1, etc.
     * (This is ONNX right-align = ggml left-align.) Pad higher dims with 1.
     */

    /* Count non-trivial dims of b */
    int nd_b = GGML_MAX_DIMS;
    while (nd_b > 0 && b->ne[nd_b-1] == 1) nd_b--;
    if (nd_b == 0) return; /* scalar, already broadcastable */

    /* Count non-trivial dims of a */
    int nd_a = GGML_MAX_DIMS;
    while (nd_a > 0 && a->ne[nd_a-1] == 1) nd_a--;
    if (nd_a == 0) nd_a = 1;

    /* Left-aligned: b[0]→a[0], b[1]→a[1], ... works when b has fewer or equal dims */
    int64_t new_ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
    int left_ok = 1;
    for (int d = 0; d < nd_b; d++) {
        new_ne[d] = b->ne[d];
        if (b->ne[d] != 1 && b->ne[d] != a->ne[d]) left_ok = 0;
    }

    if (left_ok) {
        /* Check if reshape is needed */
        int changed = 0;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (new_ne[d] != b->ne[d]) { changed = 1; break; }
        }
        if (changed) {
            int nd = GGML_MAX_DIMS;
            while (nd > 1 && new_ne[nd-1] == 1) nd--;
            *pb = onnx_reshape_nd(ctx, b, new_ne, nd);
        }
        if (onnx_trace_nodes() && g_current_node)
            fprintf(stderr, "[bcast] %s op=%s a.ne=[%lld,%lld,%lld,%lld] b.ne=[%lld,%lld,%lld,%lld] -> left_align\n",
                    g_current_node->outputs[0], g_current_node->op_type,
                    (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                    (long long)(*pb)->ne[0],(long long)(*pb)->ne[1],(long long)(*pb)->ne[2],(long long)(*pb)->ne[3]);
        return;
    }

    /* Left-align didn't work. Try right-aligning b within a's dims:
     * b's highest dim (nd_b-1) aligns with a's highest dim (nd_a-1).
     * This handles cases like a=[W,H,C,N], b=[C,N] → b goes to dims [2,3]. */
    int offset = nd_a - nd_b;
    if (offset < 0) offset = 0;

    for (int d = 0; d < GGML_MAX_DIMS; d++) new_ne[d] = 1;
    int right_ok = 1;
    for (int d = 0; d < nd_b; d++) {
        int ad = d + offset;
        new_ne[ad] = b->ne[d];
        if (b->ne[d] != 1 && b->ne[d] != a->ne[ad]) right_ok = 0;
    }

    if (right_ok) {
        int64_t nel = ne_product(new_ne, GGML_MAX_DIMS);
        if (nel == ggml_nelements(b)) {
            int nd = GGML_MAX_DIMS;
            while (nd > 1 && new_ne[nd-1] == 1) nd--;
            *pb = onnx_reshape_nd(ctx, b, new_ne, nd);
        if (onnx_trace_nodes() && g_current_node)
            fprintf(stderr, "[bcast] %s op=%s a.ne=[%lld,%lld,%lld,%lld] b.ne=[%lld,%lld,%lld,%lld] -> right_align(off=%d)\n",
                    g_current_node->outputs[0], g_current_node->op_type,
                    (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                    (long long)(*pb)->ne[0],(long long)(*pb)->ne[1],(long long)(*pb)->ne[2],(long long)(*pb)->ne[3],
                    offset);
        }
        return;
    }

    /* Last resort: try matching each b dim to an a dim by value */
    for (int d = 0; d < GGML_MAX_DIMS; d++) new_ne[d] = 1;
    int b_idx = 0;
    for (int d = 0; d < GGML_MAX_DIMS && b_idx < nd_b; d++) {
        if (b->ne[b_idx] == a->ne[d] || b->ne[b_idx] == 1) {
            new_ne[d] = b->ne[b_idx];
            b_idx++;
        }
    }
    if (b_idx == nd_b) {
        int64_t nel = ne_product(new_ne, GGML_MAX_DIMS);
        if (nel == ggml_nelements(b)) {
            int nd = GGML_MAX_DIMS;
            while (nd > 1 && new_ne[nd-1] == 1) nd--;
            if (onnx_trace_nodes() && g_current_node)
                fprintf(stderr, "[bcast] %s op=%s a.ne=[%lld,%lld,%lld,%lld] b.ne=[%lld,%lld,%lld,%lld] -> match_by_value\n",
                        g_current_node->outputs[0], g_current_node->op_type,
                        (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                        (long long)(*pb)->ne[0],(long long)(*pb)->ne[1],(long long)(*pb)->ne[2],(long long)(*pb)->ne[3]);
            *pb = onnx_reshape_nd(ctx, b, new_ne, nd);
            return;
        }
    }


    /* Neither a broadcasts into b nor b into a.
     * Both need expansion to a common shape: max(a.ne[d], b.ne[d]) per dim.
     * Use ggml_repeat on each tensor to expand to the target shape. */
    {
        if (onnx_trace_nodes() && g_current_node && g_current_ctx) {
            int nda = g_current_node->n_inputs > 0 ? tmap_get_ndims(g_current_ctx, g_current_node->inputs[0]) : -1;
            int ndb = g_current_node->n_inputs > 1 ? tmap_get_ndims(g_current_ctx, g_current_node->inputs[1]) : -1;
            fprintf(stderr, "[bcast] %s op=%s a.ne=[%lld,%lld,%lld,%lld] b.ne=[%lld,%lld,%lld,%lld] onnx_nd=(%d,%d) -> EXPAND_BOTH\n",
                    g_current_node->outputs[0], g_current_node->op_type,
                    (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                    (long long)b->ne[0],(long long)b->ne[1],(long long)b->ne[2],(long long)b->ne[3],
                    nda, ndb);
        }
        int64_t target[GGML_MAX_DIMS];
        int need_a = 0, need_b = 0;
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            target[d] = (a->ne[d] > b->ne[d]) ? a->ne[d] : b->ne[d];
            /* Verify broadcast compatibility: each dim must be 1 or equal to target */
            if (a->ne[d] != 1 && a->ne[d] != target[d]) {
                return;
            }
            if (b->ne[d] != 1 && b->ne[d] != target[d]) {
                return;
            }
            if (a->ne[d] != target[d]) need_a = 1;
            if (b->ne[d] != target[d]) need_b = 1;
        }

        int tgt_nd = GGML_MAX_DIMS;
        while (tgt_nd > 1 && target[tgt_nd-1] == 1) tgt_nd--;
        struct ggml_tensor *tgt = onnx_new_tensor_nd(ctx, a->type, target, tgt_nd);
        if (need_a) {
            if (!ggml_can_repeat(a, tgt))
                fprintf(stderr, "[broadcast] repeat_a FAIL: a='%s' ne=[%lld,%lld,%lld,%lld] tgt=[%lld,%lld,%lld,%lld] node=%s\n",
                        a->name, (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                        (long long)tgt->ne[0],(long long)tgt->ne[1],(long long)tgt->ne[2],(long long)tgt->ne[3],
                        g_current_node ? g_current_node->op_type : "?");
            *pa = ggml_repeat(ctx, a, tgt);
        }
        if (need_b) {
            if (!ggml_can_repeat(b, tgt))
                fprintf(stderr, "[broadcast] repeat_b FAIL: b='%s' ne=[%lld,%lld,%lld,%lld] tgt=[%lld,%lld,%lld,%lld] node=%s\n",
                        b->name, (long long)b->ne[0],(long long)b->ne[1],(long long)b->ne[2],(long long)b->ne[3],
                        (long long)tgt->ne[0],(long long)tgt->ne[1],(long long)tgt->ne[2],(long long)tgt->ne[3],
                        g_current_node ? g_current_node->op_type : "?");
            *pb = ggml_repeat(ctx, b, tgt);
        }
    }
}

static int map_node(onnx_ggml_ctx_t *c, const onnx_node_t *n) {
    g_current_node = n;
    g_current_ctx  = c;
    struct ggml_tensor *out = NULL;
    int out_nd = -1;  /* output ONNX ndims; -1 = inherit from input (generic path) */

    /* Node trace: set ONNX_TRACE_NODES=1 to dump each node's inputs/shapes */
    if (onnx_trace_nodes()) {
        fprintf(stderr, "[node] %s op=%s inputs=[", n->outputs[0], n->op_type);
        for (int i = 0; i < n->n_inputs; i++) {
            struct ggml_tensor *ti = tmap_get(c, n->inputs[i]);
            if (ti)
                fprintf(stderr, "%s%s[%lld,%lld,%lld,%lld,%lld](nd%d)", i?", ":"", n->inputs[i],
                        (long long)ti->ne[0],(long long)ti->ne[1],(long long)ti->ne[2],(long long)ti->ne[3],(long long)ti->ne[4],
                        tmap_get_ndims(c, n->inputs[i]));
            else
                fprintf(stderr, "%s%s(NULL)", i?", ":"", n->inputs[i]);
        }
        fprintf(stderr, "]\n");
    }

    struct ggml_tensor *a = get_input(c, n, 0);
    struct ggml_tensor *b = get_input(c, n, 1);

    const char *op = n->op_type;

    /* Dispatch to op group handlers.
     * Each returns: 1 = handled, 0 = not this group, -1 = error.
     * Handlers that register outputs themselves return 1 directly (no goto needed).
     * Handlers that set *out and *out_nd fall through to generic registration. */
    {
        int r;
        /* Which group claimed the node, and what it returned.  Logged here at
         * the branch point rather than inside a handler: a handler that bails
         * out before its own trace (a rejected axis, a missing input) prints
         * nothing at all, which leaves no way to tell "not reached" apart from
         * "reached and refused". */
        r = map_node_basic  (c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> basic r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_tensor (c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> tensor r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_nn     (c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> nn r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_quant  (c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> quant r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_special(c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> special r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1; if (r > 0) goto reg_output;
        onnx_warn_unsupported_op(op);
        return -1;
    }

reg_output:
    /* Register outputs.
     * If the handler set out_nd, use it (authoritative).
     * Otherwise fall back to inheriting from first input. */
    if (out) {
        int out_ndims;
        if (out_nd > 0) {
            out_ndims = out_nd;
        } else {
            out_ndims = ggml_n_dims(out);
            /* ggml_n_dims() cannot see ONNX dims of size 1 (it drops trailing
             * ggml axes, i.e. leading ONNX ones), so the ONNX rank is taken
             * from the inputs.  Under ONNX broadcasting the result has the
             * rank of the widest operand, so every input is considered --
             * looking only at inputs[0] loses the rank whenever the first
             * operand is the smaller one, as in `Add(bias[768], x[1,128,768])`.
             *
             * NOTE: ggml_n_dims stays the floor on purpose.  Lowering the rank
             * to the operands alone desynchronises it from the ggml shape --
             * a tensor stored as ne=[1,N] with ONNX rank 1 reconstructs as
             * [1] and loses N-1 elements in the next Unsqueeze/Reshape.  Rank
             * and shape have to move together; see TODO for the MaskRCNN case. */
            for (int i = 0; i < n->n_inputs; i++) {
                if (n->inputs[i][0] == '\0') continue;
                int in_nd = tmap_get_ndims(c, n->inputs[i]);
                if (in_nd > out_ndims) out_ndims = in_nd;
            }
        }
        if (onnx_trace_nodes()) {
            int in_nd0 = (n->n_inputs > 0 && n->inputs[0][0] != '\0')
                       ? tmap_get_ndims(c, n->inputs[0]) : -1;
            fprintf(stderr, "[rank] %s op=%s out_nd=%d ggml_n_dims=%d in0_nd=%d -> stored=%d ne=[%lld,%lld,%lld,%lld]\n",
                    n->outputs[0], n->op_type, out_nd, (int)ggml_n_dims(out),
                    in_nd0, out_ndims,
                    (long long)out->ne[0], (long long)out->ne[1],
                    (long long)out->ne[2], (long long)out->ne[3]);
        }
        for (int i = 0; i < n->n_outputs; i++) {
            if (n->outputs[i][0] != '\0') {
                ggml_set_name(out, n->outputs[i]);
                tmap_put_nd(c, n->outputs[i], out, out_ndims);
            }
        }
    }

    return 0;
}

/* ── Allocate scheduler buffers and load all static data ─────────── */
/* Called from build() on first load and from run() before each compute
 * (reset + realloc) so that intermediate buffer aliasing cannot corrupt
 * weight data between runs.                                            */

/* Reload weights, constants, shapes into already-allocated buffers.
 * Used on repeated runs where compute may have overwritten weight data
 * via intermediate buffer aliasing. */
/* Fill strided Slice outputs (step != 1) by reading src and copying with stride */
static void fill_strided_slices(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->n_slice_fills; i++) {
        struct ggml_tensor *src = c->slice_fill_src[i];
        struct ggml_tensor *dst = c->slice_fill_dst[i];
        if (!src || !src->buffer || !dst || !dst->buffer) continue;

        /* Both buffers are indexed as float below, so anything else would read
         * and write the wrong element size -- past the end of the block for a
         * narrower type. */
        if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            fprintf(stderr, "[onnx] strided Slice '%s': only F32 is supported "
                            "(src %s, dst %s) -- skipped\n",
                    ggml_get_name(dst), ggml_type_name(src->type),
                    ggml_type_name(dst->type));
            continue;
        }

        size_t src_bytes = ggml_nbytes(src);
        float *src_buf = (float *)malloc(src_bytes);
        if (!src_buf) continue;
        ggml_backend_tensor_get(src, src_buf, 0, src_bytes);

        size_t dst_n = ggml_nelements(dst);
        float *dst_buf = (float *)malloc(dst_n * sizeof(float));
        if (!dst_buf) { free(src_buf); continue; }

        /* dst may hold more elements than the source can supply; the write
         * below fills exactly dst_n, so the buffer sizes must agree. */
        if (dst_n * sizeof(float) > ggml_nbytes(dst)) {
            free(src_buf); free(dst_buf); continue;
        }

        int64_t *st = c->slice_fill_starts[i];
        int64_t *sp = c->slice_fill_steps[i];
        int64_t *one = c->slice_fill_out_ne[i];

        /* Compute strides for source and output */
        int64_t src_stride[GGML_MAX_DIMS], out_stride[GGML_MAX_DIMS];
        src_stride[0] = 1; out_stride[0] = 1;
        for (int d = 1; d < GGML_MAX_DIMS; d++) {
            src_stride[d] = src_stride[d-1] * src->ne[d-1];
            out_stride[d] = out_stride[d-1] * one[d-1];
        }
        /* The source index is built from the recorded starts and steps, which
         * come from the model.  A negative step, a start past the end, or a
         * shape that disagrees with what was recorded all put `si` outside the
         * source -- and reading outside a malloc'd block corrupts the
         * allocator's metadata, which is only noticed at some later malloc,
         * far from here.  Clamp instead, so a bad slice yields wrong numbers
         * rather than an unexplained crash elsewhere. */
        int64_t src_n = (int64_t)ggml_nelements(src);
        int64_t n_oob = 0;
        for (int64_t di = 0; di < (int64_t)dst_n; di++) {
            int64_t si = 0;
            int64_t rem = di;
            for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
                if (out_stride[d] <= 0) continue;
                int64_t coord = rem / out_stride[d];
                rem -= coord * out_stride[d];
                si += (st[d] + coord * sp[d]) * src_stride[d];
            }
            if (si < 0 || si >= src_n) { n_oob++; si = 0; }
            dst_buf[di] = src_buf[si];
        }
        if (n_oob > 0)
            fprintf(stderr, "[onnx] strided Slice '%s': %lld of %lld source "
                            "indices out of range -- clamped\n",
                    ggml_get_name(dst), (long long)n_oob, (long long)dst_n);

        ggml_backend_tensor_set(dst, dst_buf, 0, dst_n * sizeof(float));
        free(src_buf);
        free(dst_buf);
    }
}

/* ── Pre-pass: detect RelPosBias2D (pos_embed) subgraphs ────────── */

/* Build CPU-side copy of concat(W_h, W_w) weights for rel_pos_bias kernel.
 * W_h ONNX shape: [C, rel_h], W_w ONNX shape: [C, rel_w].
 * Output layout (col-major, stride = rel_h+rel_w):
 *   w_cpu[r + c * stride]  r in [0, rel_h) → W_h
 *                           r in [rel_h, rel_h+rel_w) → W_w */
static float *build_w_cpu(const onnx_initializer_t *wh_init,
                           const onnx_initializer_t *ww_init,
                           int C, int rel_h, int rel_w) {
    int stride = rel_h + rel_w;
    float *buf = (float *)malloc((size_t)C * stride * sizeof(float));
    if (!buf) return NULL;

    /* Get raw float pointers for W_h and W_w */
    const float *wh = wh_init->decoded_data ? (const float *)wh_init->decoded_data
                                            : (const float *)wh_init->raw_data;
    const float *ww = ww_init->decoded_data ? (const float *)ww_init->decoded_data
                                            : (const float *)ww_init->raw_data;
    if (!wh || !ww) { free(buf); return NULL; }

    /* ONNX layout: W_h[c, r] stored row-major → W_h[c * rel_h + r]
     * ggml kernel expects col-major: w_cpu[r + c * stride] */
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < rel_h; r++)
            buf[r + c * stride] = wh[c * rel_h + r];
        for (int r = 0; r < rel_w; r++)
            buf[rel_h + r + c * stride] = ww[c * rel_w + r];
    }
    return buf;
}

/* Each pos_embed block in BoTNet consists of ~60-80 nodes with output names
 * containing "/pos_embed/".  Structure:
 *   MatMul_0: input0 = first_Reshape_output, input1 = W_h (initializer)
 *   MatMul_1: input1 = W_w (initializer)
 *   ...many Reshape/Pad/Flatten/Slice/Expand/Transpose nodes...
 *   Final Reshape: output = block output consumed by attention Add
 *
 * The pre-pass finds block boundaries and extracts:
 *   - x input: input0 of the first Reshape (before MatMul_0)
 *   - W_h, W_w: initializer inputs of the two MatMul ops
 *   - H, W, C from W_h shape [C, 2*H-1], W_w shape [C, 2*W-1]
 *   - B from x shape  [B, H*W, C]
 *   - output name of final Reshape
 */
static int copy_segment_boundaries(onnx_ggml_ctx_t *c, int seg);

/* Record the measured size of a data-dependent op's output. */
static void resolved_put(onnx_ggml_ctx_t *c, const char *name, int64_t size) {
    for (int i = 0; i < c->n_resolved; i++)
        if (strcmp(c->resolved_names[i], name) == 0) {
            c->resolved_sizes[i] = size;
            return;
        }
    if (c->n_resolved >= ONNX_MAX_RESOLVED) return;
    strncpy(c->resolved_names[c->n_resolved], name, ONNX_MAX_NAME - 1);
    c->resolved_names[c->n_resolved][ONNX_MAX_NAME - 1] = '\0';
    c->resolved_sizes[c->n_resolved] = size;
    c->n_resolved++;
}

/* The measured size for `name`, or -1 if it has not been measured yet. */
int64_t onnx_resolved_size(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = 0; i < c->n_resolved; i++)
        if (strcmp(c->resolved_names[i], name) == 0)
            return c->resolved_sizes[i];
    return -1;
}

/* Measure what the cut ops of segment `seg` actually produced.
 *
 * The segment has just been computed, so the INPUT of each of its cut ops now
 * holds real data -- which is exactly what decides the op's output size.  For
 * NonZero that is the count of non-zero elements; the value is recorded so
 * that re-mapping the op in the next segment builds it at its true size
 * instead of the all-elements-non-zero guess.
 *
 * Ops whose size cannot be measured this way are left unrecorded and keep
 * their build-time guess. */
static int resolve_segment_sizes(onnx_ggml_ctx_t *c, int seg) {
    const onnx_segment_t *sg = &c->segments[seg];

    for (int j = 0; j < sg->n_cut_nodes; j++) {
        const onnx_node_t *cn = &c->onnx->nodes[sg->cut_nodes[j]];
        if (strcmp(cn->op_type, "NonZero") != 0) continue;   /* NMS/TopK: stage 2c */

        struct ggml_tensor *src = tmap_get(c, cn->inputs[0]);
        if (!src || !src->buffer) continue;

        int64_t nel = ggml_nelements(src);
        if (nel <= 0) continue;

        /* Read the input back and count what is actually non-zero. */
        size_t nbytes = ggml_nbytes(src);
        void *buf = malloc(nbytes);
        if (!buf) return -1;
        ggml_backend_tensor_get(src, buf, 0, nbytes);

        int64_t nnz = 0;
        if (src->type == GGML_TYPE_F32) {
            const float *v = (const float *)buf;
            for (int64_t k = 0; k < nel; k++) if (v[k] != 0.0f) nnz++;
        } else if (src->type == GGML_TYPE_I32) {
            const int32_t *v = (const int32_t *)buf;
            for (int64_t k = 0; k < nel; k++) if (v[k] != 0) nnz++;
        } else {
            /* Unhandled element type -- keep the guess rather than record a
             * count derived from bytes we cannot interpret. */
            free(buf);
            continue;
        }
        free(buf);

        resolved_put(c, cn->outputs[0], nnz);
        if (onnx_trace_nodes())
            fprintf(stderr, "[resolve] %s: NonZero over '%s' (%lld elems) -> nnz=%lld\n",
                    cn->outputs[0], cn->inputs[0], (long long)nel, (long long)nnz);
    }
    return 0;
}

/* Is this tensor read by a segment later than `seg`? */
static int tensor_crosses_boundary(onnx_ggml_ctx_t *c, int seg, const char *nm) {
    for (int s2 = seg + 1; s2 < c->n_segments; s2++)
        for (int j = c->segments[s2].first_node;
             j <= c->segments[s2].last_node; j++)
            for (int a = 0; a < c->onnx->nodes[j].n_inputs; a++)
                if (strcmp(c->onnx->nodes[j].inputs[a], nm) == 0)
                    return 1;
    return 0;
}

/* Expand a segment's graph to everything it must actually produce.
 *
 * Two kinds of result matter: the inputs of its cut ops (needed to resolve
 * the data-dependent shapes) and any tensor a later segment reads.  Building
 * only towards the cut ops leaves the second kind out of the graph entirely,
 * so it never gets computed and never gets a buffer -- and the copy that
 * carries it across the boundary then has nothing to copy.  The final
 * segment additionally produces the model outputs. */
/* Dump a segment's graph: which nodes it will actually compute, and where each
 * came from.  Answers the question a failing node raises -- was it pulled into
 * this segment at all, and if so by which of the three expansion roots (a cut
 * op's input, a tensor crossing the boundary, or a model output). */
static void trace_segment_graph(onnx_ggml_ctx_t *c, int seg) {
    /* Its own switch (ONNX_TRACE_SGRAPH): a segment can hold hundreds of nodes
     * and this would drown the ordinary trace. */
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("ONNX_TRACE_SGRAPH");
        on = (e && *e && *e != '0') ? 1 : 0;
    }
    if (!on || !c->graph) return;
    int n = ggml_graph_n_nodes(c->graph);
    fprintf(stderr, "[sgraph] segment %d: %d nodes, range %d..%d\n",
            seg, n, c->segments[seg].first_node, c->segments[seg].last_node);
    for (int i = 0; i < n; i++) {
        struct ggml_tensor *t = ggml_graph_node(c->graph, i);
        fprintf(stderr, "[sgraph]   %d '%s' op=%s type=%s src0=%s(%s) src1=%s(%s)\n",
                i, ggml_get_name(t), ggml_op_name(t->op), ggml_type_name(t->type),
                t->src[0] ? ggml_get_name(t->src[0]) : "-",
                t->src[0] ? ggml_type_name(t->src[0]->type) : "-",
                t->src[1] ? ggml_get_name(t->src[1]) : "-",
                t->src[1] ? ggml_type_name(t->src[1]->type) : "-");
    }
}

static void build_segment_graph(onnx_ggml_ctx_t *c, int seg) {
    const onnx_segment_t *sg = &c->segments[seg];
    onnx_model_t *onnx = c->onnx;

    for (int j = 0; j < sg->n_cut_nodes; j++) {
        const onnx_node_t *cn = &onnx->nodes[sg->cut_nodes[j]];
        for (int a = 0; a < cn->n_inputs; a++) {
            struct ggml_tensor *t = tmap_get(c, cn->inputs[a]);
            if (t) { ggml_set_output(t); ggml_build_forward_expand(c->graph, t); }
        }
    }

    for (int i = sg->first_node; i <= sg->last_node; i++) {
        for (int o = 0; o < onnx->nodes[i].n_outputs; o++) {
            const char *nm = onnx->nodes[i].outputs[o];
            if (nm[0] == '\0') continue;
            if (!tensor_crosses_boundary(c, seg, nm)) continue;
            struct ggml_tensor *t = tmap_get(c, nm);
            if (t) { ggml_set_output(t); ggml_build_forward_expand(c->graph, t); }
        }
    }

    if (seg == c->n_segments - 1) {
        for (int i = 0; i < onnx->n_outputs; i++) {
            struct ggml_tensor *t = tmap_get(c, onnx->outputs[i].name);
            if (t) { ggml_set_output(t); ggml_build_forward_expand(c->graph, t); }
        }
    }

    trace_segment_graph(c, seg);
}

/* Give a buffer to weight tensors added to ctx_weight since the last call.
 *
 * alloc_ctx_tensors walks the whole context but sizes and allocates ONLY
 * tensors whose ->data is still NULL (ggml-alloc.c: `if (t->data == NULL &&
 * t->view_src == NULL)`), so calling it again after a segment has added
 * tensors allocates just those.  It returns a fresh buffer for them, or NULL
 * when there was nothing left to allocate -- NULL is the normal "no new
 * tensors" answer here, not a failure.
 *
 * Every non-NULL buffer is recorded so onnx_ggml_free can release it: the
 * count of buffers kept must equal the count freed, which is what the
 * [wbuf] trace lets a run confirm. */
static int alloc_new_weight_tensors(onnx_ggml_ctx_t *c) {
    ggml_backend_t backend = c->backend_gpu ? c->backend_gpu : c->backend_cpu;
    if (!backend || !c->ctx_weight) return 0;

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c->ctx_weight, backend);
    if (!buf) return 0;   /* nothing new to allocate */

    if (c->n_extra_weight_bufs >= ONNX_MAX_WEIGHT_BUFS) {
        fprintf(stderr, "[onnx] too many weight buffers (>%d)\n", ONNX_MAX_WEIGHT_BUFS);
        ggml_backend_buffer_free(buf);
        return -1;
    }
    c->extra_weight_bufs[c->n_extra_weight_bufs++] = buf;
    if (onnx_trace_nodes())
        fprintf(stderr, "[wbuf] segment %d: extra weight buffer %d (%zu bytes)\n",
                c->cur_segment, c->n_extra_weight_bufs - 1,
                ggml_backend_buffer_get_size(buf));
    return 0;
}

/* ── Data-dependent shape segmentation (pre-pass) ────────────────── */

/* Is this op's output shape a function of its input VALUES rather than only
 * its input shapes?  See ONNX_SHAPE_DEPENDENT_OPS in onnx_ggml.h. */
static int op_is_shape_dependent(const char *op) {
    static const char *ops[] = ONNX_SHAPE_DEPENDENT_OPS;
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); i++)
        if (strcmp(op, ops[i]) == 0) return 1;
    return 0;
}

/* Split onnx->nodes[] into segments cut at data-dependent ops.
 *
 * A cut op can only be resolved once its own inputs have been computed, so
 * cut ops are grouped into waves: a cut op joins the current wave when none
 * of its transitive inputs is produced by another cut op still in that same
 * wave.  Each wave ends one segment.
 *
 * Rather than walking the dependency graph transitively -- which needs a
 * producer map over every tensor name and is the expensive part -- the pass
 * uses the fact that nodes are already in topological order: a cut op
 * belongs to the current wave unless some node between the previous cut and
 * itself consumes a tensor produced by a cut op of the current wave.  That
 * is a linear scan carrying a small set of "tainted" tensor names.
 *
 * Sets c->n_segments = 0 when the model has no data-dependent op at all, so
 * the caller takes the original single-pass path.
 * Returns 0 on success, -1 if the model needs more segments than fit. */
static int detect_shape_dependent_segments(onnx_ggml_ctx_t *c) {
    onnx_model_t *onnx = c->onnx;
    c->n_segments = 0;

    /* Cheap first look: most models have no such op and skip everything. */
    int any = 0;
    for (int i = 0; i < onnx->n_nodes; i++)
        if (op_is_shape_dependent(onnx->nodes[i].op_type)) { any = 1; break; }
    if (!any) {
        if (onnx_trace_nodes())
            fprintf(stderr, "[segment] no data-dependent ops -- single-pass path\n");
        return 0;
    }

    /* Names produced by cut ops of the wave being accumulated.  A node
     * consuming one of these cannot be built before that wave has run, so it
     * forces the wave to close. */
    static char tainted[ONNX_MAX_DEFERRED][ONNX_MAX_NAME];
    int n_tainted = 0;

    int seg_first = 0;
    int cut_nodes[ONNX_MAX_DEFERRED];
    int n_cut = 0;

    for (int i = 0; i < onnx->n_nodes; i++) {
        const onnx_node_t *n = &onnx->nodes[i];

        /* Does this node consume anything the current wave produces? */
        int consumes_tainted = 0;
        for (int a = 0; a < n->n_inputs && !consumes_tainted; a++) {
            if (n->inputs[a][0] == '\0') continue;
            for (int t = 0; t < n_tainted; t++)
                if (strcmp(n->inputs[a], tainted[t]) == 0) { consumes_tainted = 1; break; }
        }

        /* If so, the wave has to be executed before this node can be built:
         * close the segment just before it. */
        if (consumes_tainted && n_cut > 0) {
            if (c->n_segments >= ONNX_MAX_SEGMENTS) {
                fprintf(stderr, "[onnx] more than %d data-dependent segments needed\n",
                        ONNX_MAX_SEGMENTS);
                return -1;
            }
            onnx_segment_t *s = &c->segments[c->n_segments++];
            s->first_node  = seg_first;
            s->last_node   = i - 1;
            s->n_cut_nodes = n_cut;
            for (int j = 0; j < n_cut; j++) s->cut_nodes[j] = cut_nodes[j];

            seg_first  = i;
            n_cut      = 0;
            n_tainted  = 0;
        }

        if (op_is_shape_dependent(n->op_type)) {
            if (n_cut < ONNX_MAX_DEFERRED) cut_nodes[n_cut++] = i;
            for (int o = 0; o < n->n_outputs; o++) {
                if (n->outputs[o][0] == '\0') continue;
                if (n_tainted < ONNX_MAX_DEFERRED) {
                    strncpy(tainted[n_tainted], n->outputs[o], ONNX_MAX_NAME - 1);
                    tainted[n_tainted][ONNX_MAX_NAME - 1] = '\0';
                    n_tainted++;
                }
            }
        }
    }

    /* Final segment: everything after the last wave, with no cut after it. */
    if (c->n_segments >= ONNX_MAX_SEGMENTS) {
        fprintf(stderr, "[onnx] more than %d data-dependent segments needed\n",
                ONNX_MAX_SEGMENTS);
        return -1;
    }
    {
        onnx_segment_t *s = &c->segments[c->n_segments++];
        s->first_node  = seg_first;
        s->last_node   = onnx->n_nodes - 1;
        s->n_cut_nodes = n_cut;
        for (int j = 0; j < n_cut; j++) s->cut_nodes[j] = cut_nodes[j];
    }

    /* Boundary census: tensors produced in one segment and read in a later
     * one.  These have to be copied somewhere the scheduler will not reuse,
     * so what matters is not how many exist in total but how many are alive
     * at once -- a boundary is only needed from the segment that produces it
     * until the last one that reads it.  The peak of that count is what the
     * copy buffer must hold; the total is only an upper bound. */
    if (onnx_trace_nodes()) {
        int n_boundary = 0;
        int live[ONNX_MAX_SEGMENTS];   /* boundaries alive during segment s */
        for (int s = 0; s < ONNX_MAX_SEGMENTS; s++) live[s] = 0;

        for (int s = 0; s < c->n_segments; s++) {
            for (int i = c->segments[s].first_node; i <= c->segments[s].last_node; i++) {
                for (int o = 0; o < onnx->nodes[i].n_outputs; o++) {
                    const char *nm = onnx->nodes[i].outputs[o];
                    if (nm[0] == '\0') continue;

                    /* Last segment that reads this tensor, -1 if none does. */
                    int last_use = -1;
                    for (int s2 = s + 1; s2 < c->n_segments; s2++) {
                        int used = 0;
                        for (int j = c->segments[s2].first_node;
                             j <= c->segments[s2].last_node && !used; j++)
                            for (int a = 0; a < onnx->nodes[j].n_inputs; a++)
                                if (strcmp(onnx->nodes[j].inputs[a], nm) == 0) {
                                    used = 1; break;
                                }
                        if (used) last_use = s2;
                    }
                    if (last_use < 0) continue;   /* stays inside its segment */

                    n_boundary++;
                    for (int s2 = s; s2 <= last_use; s2++) live[s2]++;
                }
            }
        }

        int peak = 0, peak_seg = 0;
        for (int s = 0; s < c->n_segments; s++)
            if (live[s] > peak) { peak = live[s]; peak_seg = s; }

        fprintf(stderr, "[segment] %d tensors cross a segment boundary; "
                        "peak %d live at once (segment %d)\n",
                n_boundary, peak, peak_seg);
        fprintf(stderr, "[segment] live boundaries per segment:");
        for (int s = 0; s < c->n_segments; s++)
            fprintf(stderr, " %d:%d", s, live[s]);
        fprintf(stderr, "\n");
    }

    if (onnx_trace_nodes()) {
        fprintf(stderr, "[segment] %d segments over %d nodes\n",
                c->n_segments, onnx->n_nodes);
        for (int s = 0; s < c->n_segments; s++) {
            const onnx_segment_t *sg = &c->segments[s];
            fprintf(stderr, "[segment] %d: nodes %d..%d (%d), %d cut op(s):",
                    s, sg->first_node, sg->last_node,
                    sg->last_node - sg->first_node + 1, sg->n_cut_nodes);
            for (int j = 0; j < sg->n_cut_nodes; j++)
                fprintf(stderr, " %s(%s)",
                        onnx->nodes[sg->cut_nodes[j]].outputs[0],
                        onnx->nodes[sg->cut_nodes[j]].op_type);
            fprintf(stderr, "\n");
        }
    }
    return 0;
}

static int detect_pos_embed_blocks(onnx_ggml_ctx_t *c) {
    onnx_model_t *onnx = c->onnx;
    c->n_pos_embed_blocks = 0;

    /* Pass 1: find contiguous ranges of nodes with /pos_embed/ in outputs */
    int block_start = -1;
    int matmul_count = 0;
    char wh_name[ONNX_MAX_NAME] = {0};
    char ww_name[ONNX_MAX_NAME] = {0};
    char first_reshape_input[ONNX_MAX_NAME] = {0};

    for (int i = 0; i < onnx->n_nodes; i++) {
        onnx_node_t *nd = &onnx->nodes[i];
        int is_pos_embed = 0;

        /* Check if any output name contains /pos_embed/ or /attn/ba/ */
        for (int o = 0; o < nd->n_outputs; o++) {
            if (strstr(nd->outputs[o], "/pos_embed/") ||
                strstr(nd->outputs[o], "/attn/ba/")) {
                is_pos_embed = 1;
                break;
            }
        }

        if (is_pos_embed) {
            if (block_start < 0) {
                /* Start of a new block */
                block_start = i;
                matmul_count = 0;
                wh_name[0] = ww_name[0] = first_reshape_input[0] = '\0';
            }

            /* Track MatMul ops to extract W_h, W_w */
            if (strcmp(nd->op_type, "MatMul") == 0) {
                matmul_count++;
                if (matmul_count == 1) {
                    /* First MatMul: input1 = W_h */
                    snprintf(wh_name, ONNX_MAX_NAME, "%s", nd->inputs[1]);
                    /* input0 is the Reshape output; find what feeds it */
                } else if (matmul_count == 2) {
                    /* Second MatMul: input1 = W_w */
                    snprintf(ww_name, ONNX_MAX_NAME, "%s", nd->inputs[1]);
                }
            }

            /* First Reshape in block: its input0 is the real x */
            if (strcmp(nd->op_type, "Reshape") == 0 && first_reshape_input[0] == '\0') {
                snprintf(first_reshape_input, ONNX_MAX_NAME, "%s", nd->inputs[0]);
            }
        } else if (block_start >= 0) {
            /* End of block (current node is outside pos_embed) */
            int block_end = i - 1;

            if (matmul_count >= 2 && wh_name[0] && ww_name[0] &&
                first_reshape_input[0] && c->n_pos_embed_blocks < ONNX_MAX_POS_EMBED) {

                int bi = c->n_pos_embed_blocks;

                snprintf(c->pos_embed_blocks[bi].x_input_name, ONNX_MAX_NAME,
                         "%s", first_reshape_input);
                snprintf(c->pos_embed_blocks[bi].wh_name, ONNX_MAX_NAME,
                         "%s", wh_name);
                snprintf(c->pos_embed_blocks[bi].ww_name, ONNX_MAX_NAME,
                         "%s", ww_name);
                /* Output: last node's first output */
                snprintf(c->pos_embed_blocks[bi].output_name, ONNX_MAX_NAME,
                         "%s", onnx->nodes[block_end].outputs[0]);
                c->pos_embed_blocks[bi].first_node_idx = block_start;
                c->pos_embed_blocks[bi].last_node_idx  = block_end;

                /* Extract H, W, C from W_h and W_w initializer shapes */
                const onnx_initializer_t *wh_init = onnx_find_initializer(onnx, wh_name);
                const onnx_initializer_t *ww_init = onnx_find_initializer(onnx, ww_name);

                if (wh_init && ww_init && wh_init->n_dims == 2 && ww_init->n_dims == 2) {
                    /* ONNX shape: W_h [C, 2*H-1], W_w [C, 2*W-1] */
                    int C     = (int)wh_init->dims[0];
                    int rel_h = (int)wh_init->dims[1]; /* 2*H-1 */
                    int rel_w = (int)ww_init->dims[1]; /* 2*W-1 */
                    int H = (rel_h + 1) / 2;
                    int W = (rel_w + 1) / 2;

                    /* B (the head count) cannot be read here: this pre-pass
                     * runs before any node is mapped, so the x tensor does not
                     * exist yet and tmap_get returns NULL, leaving B at 1.
                     * BoTNet has four heads, so the kernel then filled a
                     * quarter of its output and left the rest untouched.
                     * Recorded as 0 and resolved at emit time, where the
                     * tensor is real. */
                    int B = 0;

                    c->pos_embed_blocks[bi].params.H     = H;
                    c->pos_embed_blocks[bi].params.W     = W;
                    c->pos_embed_blocks[bi].params.B     = B;
                    c->pos_embed_blocks[bi].params.C     = C;
                    c->pos_embed_blocks[bi].params.rel_h = rel_h;
                    c->pos_embed_blocks[bi].params.rel_w = rel_w;
                    c->pos_embed_blocks[bi].params.w_cpu_stride = rel_h + rel_w;

                    /* Verify W_h, W_w are F32 */
                    if (wh_init->data_type != 1 /* ONNX_DTYPE_FLOAT */ ||
                        ww_init->data_type != 1) {
                        fprintf(stderr, "[onnx] pos_embed block %d: W_h/W_w not F32 "
                                "(types: %d, %d) — skipping\n",
                                bi, wh_init->data_type, ww_init->data_type);
                        block_start = -1;
                        continue;
                    }

                    c->pos_embed_blocks[bi].params.w_cpu =
                        build_w_cpu(wh_init, ww_init, C, rel_h, rel_w);
                    c->n_pos_embed_blocks++;

                    if (onnx_trace_nodes())
                        fprintf(stderr, "[posembed] block %d accepted: nodes %d..%d "
                                "H=%d W=%d B=%d C=%d rel=(%d,%d) w_cpu=%s\n"
                                "           wh='%s' ww='%s'\n",
                                bi, block_start, block_end, H, W, B, C, rel_h, rel_w,
                                c->pos_embed_blocks[bi].params.w_cpu ? "ok" : "NULL",
                                wh_name, ww_name);
                } else if (onnx_trace_nodes()) {
                    /* The block's node range was already written into the slot
                     * above; failing here leaves it there without incrementing
                     * the count, so nothing skips those nodes and nothing
                     * replaces them either. */
                    fprintf(stderr, "[posembed] block %d REJECTED: nodes %d..%d "
                            "wh='%s'(%s nd=%d) ww='%s'(%s nd=%d)\n",
                            bi, block_start, block_end,
                            wh_name, wh_init ? "found" : "MISSING",
                            wh_init ? wh_init->n_dims : -1,
                            ww_name, ww_init ? "found" : "MISSING",
                            ww_init ? ww_init->n_dims : -1);
                }
            }
            block_start = -1;
        }
    }

    /* Handle case where last node is inside a pos_embed block */
    if (block_start >= 0 && matmul_count >= 2 && wh_name[0] && ww_name[0] &&
        first_reshape_input[0] && c->n_pos_embed_blocks < ONNX_MAX_POS_EMBED) {
        int block_end = onnx->n_nodes - 1;
        int bi = c->n_pos_embed_blocks;

        snprintf(c->pos_embed_blocks[bi].x_input_name, ONNX_MAX_NAME,
                 "%s", first_reshape_input);
        snprintf(c->pos_embed_blocks[bi].wh_name, ONNX_MAX_NAME, "%s", wh_name);
        snprintf(c->pos_embed_blocks[bi].ww_name, ONNX_MAX_NAME, "%s", ww_name);
        snprintf(c->pos_embed_blocks[bi].output_name, ONNX_MAX_NAME,
                 "%s", onnx->nodes[block_end].outputs[0]);
        c->pos_embed_blocks[bi].first_node_idx = block_start;
        c->pos_embed_blocks[bi].last_node_idx  = block_end;

        const onnx_initializer_t *wh_init = onnx_find_initializer(onnx, wh_name);
        const onnx_initializer_t *ww_init = onnx_find_initializer(onnx, ww_name);
        if (wh_init && ww_init && wh_init->n_dims == 2 && ww_init->n_dims == 2) {
            int C     = (int)wh_init->dims[0];
            int rel_h = (int)wh_init->dims[1];
            int rel_w = (int)ww_init->dims[1];
            int H = (rel_h + 1) / 2;
            int W = (rel_w + 1) / 2;
            int B = 1;
            struct ggml_tensor *xt = tmap_get(c, first_reshape_input);
            if (xt) B = (int)xt->ne[2];

            c->pos_embed_blocks[bi].params = (rel_pos_bias_params_t){
                H, W, B, C, rel_h, rel_w, NULL, rel_h + rel_w};

            if (wh_init->data_type != 1 || ww_init->data_type != 1) {
                fprintf(stderr, "[onnx] pos_embed block %d: W_h/W_w not F32 — skipping\n", bi);
            } else {
                c->pos_embed_blocks[bi].params.w_cpu =
                    build_w_cpu(wh_init, ww_init, C, rel_h, rel_w);
                c->n_pos_embed_blocks++;
            }
        }
    }

    (void)0;  /* n_pos_embed_blocks set silently */

    return 0;
}

/* Check if node index falls inside a pos_embed block (should be skipped) */
static int is_pos_embed_node(onnx_ggml_ctx_t *c, int node_idx) {
    for (int b = 0; b < c->n_pos_embed_blocks; b++) {
        if (node_idx >= c->pos_embed_blocks[b].first_node_idx &&
            node_idx <= c->pos_embed_blocks[b].last_node_idx) {
            return 1;
        }
    }
    return 0;
}

/* Check if node_idx is the last node of a pos_embed block, return block index or -1 */
static int pos_embed_block_end(onnx_ggml_ctx_t *c, int node_idx) {
    for (int b = 0; b < c->n_pos_embed_blocks; b++) {
        if (node_idx == c->pos_embed_blocks[b].last_node_idx) {
            return b;
        }
    }
    return -1;
}

static int sched_alloc_and_fill_on(onnx_ggml_ctx_t *c, ggml_backend_sched_t sch) {

    /* Free orphan-input buffers from a previous run before re-allocating.
     *
     * Detaching the tensor is not optional bookkeeping.  These buffers come
     * from the CPU backend, so freeing one returns the memory to malloc; a
     * tensor still holding data into it makes the next set_model_inputs()
     * write over the allocator's own structures, which surfaces later as
     * "malloc(): unaligned tcache chunk detected" at an unrelated allocation.
     * ggml_backend_tensor_alloc() also asserts buffer == NULL, so a tensor
     * left attached could not be given a replacement buffer either. */
    for (int i = 0; i < c->n_orphan_input_bufs; i++) {
        struct ggml_tensor *t = c->orphan_input_tensors[i];
        if (t && t->buffer == c->orphan_input_bufs[i]) {
            t->buffer = NULL;
            t->data   = NULL;
        }
        c->orphan_input_tensors[i] = NULL;
        if (c->orphan_input_bufs[i]) {
            ggml_backend_buffer_free(c->orphan_input_bufs[i]);
            c->orphan_input_bufs[i] = NULL;
        }
    }
    c->n_orphan_input_bufs = 0;

    /* GPU-first: pre-assign all graph nodes and their leaf sources to GPU.
     * Weight tensors already have ->buffer set (from weight_buf), so sched
     * will skip them.  Only non-weight inputs and intermediates get assigned. */
    if (c->backend_gpu) {
        int n_nodes = ggml_graph_n_nodes(c->graph);
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor *node = ggml_graph_node(c->graph, i);
            if (ggml_backend_supports_op(c->backend_gpu, node)) {
                ggml_backend_sched_set_tensor_backend(sch, node, c->backend_gpu);
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j]) {
                        /* Tell sched about ALL sources — including weight tensors
                         * that already have ->buffer on GPU.  Without this, sched
                         * doesn't know their backend and may insert spurious copies
                         * or fall back to CPU. */
                        ggml_backend_sched_set_tensor_backend(
                            sch, node->src[j], c->backend_gpu);
                    }
                }
            }
        }
    }

    /* Allocate graph buffers via scheduler.
     * Tensors with ->buffer already set (weights in weight_buf) are skipped.
     * Only input placeholders and intermediate compute tensors get allocated. */
    if (!ggml_backend_sched_alloc_graph(sch, c->graph)) return -1;


    /* Ensure all ONNX input tensors have buffers.
     * When the graph has no compute ops (all outputs are standalone/weight tensors,
     * e.g. Shape→ConstantOfShape→NonZero chain), the scheduler may not allocate
     * buffers for input tensors.  Allocate them on the CPU backend. */
    for (int i = 0; i < c->onnx->n_inputs; i++) {
        struct ggml_tensor *t = tmap_get(c, c->onnx->inputs[i].name);
        if (t && !t->buffer) {
            /* Check this is a real input (not an initializer) */
            int is_init = 0;
            for (int j = 0; j < c->onnx->n_initializers; j++) {
                if (strcmp(c->onnx->inputs[i].name, c->onnx->initializers[j].name) == 0) {
                    is_init = 1;
                    break;
                }
            }
            if (!is_init) {
                /* Allocate a small buffer on CPU for this orphan input */
                ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(c->backend_cpu,
                                                                       ggml_nbytes(t) + 64);
                if (buf) {
                    /* Track before attaching, and give up the buffer rather
                     * than attach one nothing will ever free: an untracked
                     * buffer leaks on every run, and its tensor would still be
                     * pointing at it after the next free sweep clears the ones
                     * that were tracked. */
                    if (c->n_orphan_input_bufs < ONNX_MAX_DEFERRED) {
                        int k = c->n_orphan_input_bufs++;
                        c->orphan_input_bufs[k]    = buf;
                        c->orphan_input_tensors[k] = t;
                        ggml_backend_tensor_alloc(buf, t,
                                (char *)ggml_backend_buffer_get_base(buf));
                    } else {
                        fprintf(stderr, "ONNX WARNING: more than %d orphan inputs, "
                                "'%s' left without a buffer\n",
                                ONNX_MAX_DEFERRED, c->onnx->inputs[i].name);
                        ggml_backend_buffer_free(buf);
                    }
                }
            }
        }
    }

    /* Fill strided Slice outputs — their src may live in sched buffer,
     * so this must happen after sched alloc. */
    fill_strided_slices(c);

    return 0;
}

/* The common case: allocate on the context's own scheduler. */
static int sched_alloc_and_fill(onnx_ggml_ctx_t *c) {
    return sched_alloc_and_fill_on(c, c->sched);
}

/* Map ONNX nodes [node_lo, node_hi] onto ggml ops.
 *
 * The whole range is the single-pass path; segmented execution calls this
 * once per segment, which is the only reason the bounds are parameters.
 *
 * Returns -1 when the ggml context runs out of room, 0 otherwise.  An op that
 * cannot be built is still skipped silently -- that is a model this build does
 * not support, and the rest of the graph may well be fine -- but an exhausted
 * context is not selective: everything after it fails too, and each failure
 * hands back a NULL tensor that only crashes once something reads it. */
static int map_node_range(onnx_ggml_ctx_t *c, int node_lo, int node_hi) {
    onnx_model_t *onnx = c->onnx;
    for (int i = node_lo; i <= node_hi; i++) {
        /* Skip nodes inside pos_embed blocks (handled by fused custom op) */
        if (c->n_pos_embed_blocks > 0 && is_pos_embed_node(c, i)) {
            int bi = pos_embed_block_end(c, i);
            if (onnx_trace_nodes())
                fprintf(stderr, "[posembed] node %d '%s' inside a block; "
                        "block_end=%d (%s)\n", i, onnx->nodes[i].outputs[0], bi,
                        bi >= 0 ? "emitting fused op" : "skipped");
            if (bi >= 0) {
                /* Last node of block — emit fused RelPosBias2D op */
                rel_pos_bias_params_t *p = &c->pos_embed_blocks[bi].params;
                int HW = p->H * p->W;

                /* Get input tensors */
                struct ggml_tensor *x_t  = tmap_get(c, c->pos_embed_blocks[bi].x_input_name);
                struct ggml_tensor *wh_t = tmap_get(c, c->pos_embed_blocks[bi].wh_name);
                struct ggml_tensor *ww_t = tmap_get(c, c->pos_embed_blocks[bi].ww_name);

                if (!x_t || !wh_t || !ww_t) {
                    fprintf(stderr, "[onnx] pos_embed block %d: missing input tensor "
                            "(x=%p W_h=%p W_w=%p) — skipping\n",
                            bi, (void*)x_t, (void*)wh_t, (void*)ww_t);
                    continue;
                }

                /* Head count, now that x exists: ggml [C, H*W, B]. */
                if (p->B <= 0) p->B = (int)x_t->ne[2];
                if (p->B <= 0) p->B = 1;

                /* Build wcat = concat(W_h, W_w) along axis 0 (rel dim).
                 * ggml_rel_pos_bias handles both CPU and Vulkan dispatch. */
                struct ggml_tensor *wcat = ggml_concat(c->ctx, wh_t, ww_t, 0);

                /* Emit fused Vulkan-capable op */
                struct ggml_tensor *out = wcat ? ggml_rel_pos_bias(c->ctx,
                    x_t, wcat, p->H, p->W) : NULL;

                /* A ggml context that has run out of room returns NULL and
                 * only logs -- in a release build there is no assert to stop
                 * on.  Naming or registering that NULL crashes here, and
                 * letting it into tmap crashes further away, in whichever op
                 * consumes it. */
                if (!out) {
                    fprintf(stderr, "[onnx] out of context memory building "
                            "pos_embed block %d\n", bi);
                    return -1;
                }
                ggml_set_name(out, c->pos_embed_blocks[bi].output_name);
                tmap_put_nd(c, c->pos_embed_blocks[bi].output_name, out, 3);
                if (onnx_trace_nodes())
                    fprintf(stderr, "[posembed] block %d -> fused op as '%s' "
                            "ne=[%lld,%lld,%lld] (x='%s' wcat.ne0=%lld)\n",
                            bi, c->pos_embed_blocks[bi].output_name,
                            (long long)out->ne[0], (long long)out->ne[1],
                            (long long)out->ne[2],
                            c->pos_embed_blocks[bi].x_input_name,
                            (long long)wcat->ne[0]);

            }
            continue;
        }
        if (map_node(c, &onnx->nodes[i]) != 0) {
            /* Non-fatal: skip unsupported/invalid ops silently */
            (void)0;
        }
    }
    return 0;
}

/* Fill the deferred tensors: Shape outputs, ConstantOfShape values, NonZero
 * index lists, EyeLike identities and NMS parameter triples.
 *
 * Each op registers its tensor at build time and the data is written here,
 * once the tensor has a buffer.  Segmented execution mapped only part of the
 * model at load, so every segment adds new entries to these lists and this
 * has to run again after each one -- filling only at load leaves later
 * segments' tensors holding whatever their buffer happened to contain, which
 * for a NonZero index list means out-of-range rows.
 *
 * Entries whose tensor still has no buffer are skipped, so re-running is
 * harmless: already-filled tensors are simply written the same values again.
 */
static void fill_deferred_tensors(onnx_ggml_ctx_t *c) {
    if (onnx_trace_nodes())
        fprintf(stderr, "[fill] segment %d: shape=%d const=%d cinit=%d nonzero=%d eye=%d nms=%d\n",
                c->cur_segment, c->n_shape_tensors, c->n_const_fills,
                c->n_cinit_fills, c->n_nonzero_fills, c->n_eye_fills,
                c->n_nms_deferred);

    /* Constant node payloads first: they are plain data with no dependency on
     * anything else here, and other deferred fills may read a shape or an
     * index out of one. */
    for (int i = 0; i < c->n_cinit_fills; i++) {
        struct ggml_tensor *t = c->cinit_fill_ptrs[i];
        if (!t || !t->buffer) continue;
        onnx_upload_initializer(t, c->cinit_fill_srcs[i]);
    }
    /* Fill Shape op output tensors with ONNX dims */
    for (int i = 0; i < c->n_shape_tensors; i++) {
        struct ggml_tensor *t = c->shape_tensor_ptrs[i];
        if (!t || !t->buffer) continue;
        int nd = (int)c->shape_tensors_ne[i][0];
        /* Clamp before it is used as a length: nd indexes both the local array
         * and shape_tensors_ne[i][1..], and a stored value past either bound
         * writes over the stack, which shows up later as heap corruption
         * somewhere unrelated. */
        if (nd > ONNX_MAX_DIMS - 1) nd = ONNX_MAX_DIMS - 1;
        if (nd < 0) nd = 0;
        size_t fill_sz = (size_t)nd * sizeof(int32_t);
        size_t t_sz = ggml_nbytes(t);
        if (fill_sz > t_sz) continue;
        int32_t dims[ONNX_MAX_DIMS];
        for (int d = 0; d < nd; d++)
            dims[d] = (int32_t)c->shape_tensors_ne[i][d + 1];
        ggml_backend_tensor_set(t, dims, 0, fill_sz);
    }

    /* Fill ConstantOfShape tensors with constant value */
    for (int i = 0; i < c->n_const_fills; i++) {
        struct ggml_tensor *t = c->const_fill_ptrs[i];
        if (!t || !t->buffer) continue;
        float val = c->const_fill_vals[i];
        size_t n = ggml_nelements(t);
        float *buf = (float *)malloc(n * sizeof(float));
        if (buf) {
            for (size_t j = 0; j < n; j++) buf[j] = val;
            ggml_backend_tensor_set(t, buf, 0, n * sizeof(float));
            free(buf);
        }
    }

    /* Fill NonZero output tensors.
     * At build time we assumed all elements are non-zero (ConstantOfShape
     * with value != 0), so nnz == total_elements of src.
     * Output layout (ggml): [nnz, input_ndims] of F32.
     * Row d contains the d-th coordinate of each non-zero element
     * (ONNX dim order, i.e. row-major unravel). */
    for (int i = 0; i < c->n_nonzero_fills; i++) {
        struct ggml_tensor *dst = c->nonzero_fill_dst[i];
        struct ggml_tensor *src = c->nonzero_fill_src[i];
        int nd = c->nonzero_fill_ndims[i];
        if (onnx_trace_nodes())
            fprintf(stderr, "[nzfill] %d/%d dst=%s buffer=%s src=%s nd=%d "
                            "dst.ne=[%lld,%lld] src.ne=[%lld,%lld]\n",
                    i, c->n_nonzero_fills,
                    dst ? ggml_get_name(dst) : "(null)",
                    (dst && dst->buffer) ? "yes" : "NO",
                    src ? ggml_get_name(src) : "(null)", nd,
                    dst ? (long long)dst->ne[0] : -1, dst ? (long long)dst->ne[1] : -1,
                    src ? (long long)src->ne[0] : -1, src ? (long long)src->ne[1] : -1);
        if (!dst || !dst->buffer) continue;
        if (!src) continue;

        int64_t n_src = (int64_t)ggml_nelements(src);
        /* Compute src shape in ONNX order (reversed from ggml ne).
         * nd comes from the op's recorded ONNX rank and is clamped here: it
         * indexes a fixed-size array below, and an ONNX rank larger than
         * ONNX_MAX_DIMS would write past it -- a stack overwrite that would
         * surface far from here, as a corrupted heap in unrelated code. */
        if (nd > ONNX_MAX_DIMS) nd = ONNX_MAX_DIMS;
        if (nd < 1) nd = 1;
        int src_ggml_dims = (int)ggml_n_dims(src);
        int64_t onnx_shape[ONNX_MAX_DIMS];
        for (int d = 0; d < nd; d++) {
            int gd = nd - 1 - d;  /* ggml dim for ONNX dim d */
            onnx_shape[d] = (gd < src_ggml_dims) ? src->ne[gd] : 1;
            /* A zero extent would make the unravel below divide by zero. */
            if (onnx_shape[d] < 1) onnx_shape[d] = 1;
        }

        /* How many entries the output holds.  dst was built either at the
         * measured size or at the all-non-zero guess; either way it is dst
         * that says how many coordinates fit. */
        int64_t nnz = dst->ne[0];
        if (nnz > n_src) nnz = n_src;

        /* Read the input so only the genuinely non-zero positions are listed.
         * Enumerating every position regardless (the old behaviour) is right
         * only when the guess holds, and produces indices for zero entries
         * otherwise. */
        int have_src_data = 0;
        void *srcbuf = NULL;
        if (src->buffer) {
            srcbuf = malloc(ggml_nbytes(src));
            if (srcbuf) {
                ggml_backend_tensor_get(src, srcbuf, 0, ggml_nbytes(src));
                have_src_data = (src->type == GGML_TYPE_F32 ||
                                 src->type == GGML_TYPE_I32);
            }
        }

        /* Build index matrix: for flat index k, unravel to nd coordinates */
        size_t fill_bytes = (size_t)nnz * nd * sizeof(float);
        size_t dst_bytes = ggml_nbytes(dst);
        if (fill_bytes > dst_bytes) { free(srcbuf); continue; }
        float *buf = (float *)malloc(fill_bytes);
        if (buf) {
            int64_t written = 0;
            for (int64_t k = 0; k < n_src && written < nnz; k++) {
                if (have_src_data) {
                    int is_nz = (src->type == GGML_TYPE_F32)
                        ? (((const float *)srcbuf)[k] != 0.0f)
                        : (((const int32_t *)srcbuf)[k] != 0);
                    if (!is_nz) continue;
                }
                int64_t rem = k;
                for (int d = nd - 1; d >= 0; d--) {
                    buf[d * nnz + written] = (float)(rem % onnx_shape[d]);
                    rem /= onnx_shape[d];
                }
                written++;
            }
            /* Fewer non-zeros than the tensor holds: pad with zeros so no
             * stale coordinate is left behind. */
            for (int64_t k = written; k < nnz; k++)
                for (int d = 0; d < nd; d++)
                    buf[d * nnz + k] = 0.0f;
            ggml_backend_tensor_set(dst, buf, 0, fill_bytes);
            free(buf);
        }
        free(srcbuf);
    }

    /* Fill EyeLike tensors with identity matrix */
    for (int i = 0; i < c->n_eye_fills; i++) {
        struct ggml_tensor *t = c->eye_fill_ptrs[i];
        if (!t || !t->buffer) continue;
        int cols = c->eye_fill_cols[i];
        int rows = c->eye_fill_rows[i];
        int k    = c->eye_fill_k[i];
        size_t n = (size_t)cols * rows;
        float *buf = (float *)calloc(n, sizeof(float));
        if (buf) {
            for (int r = 0; r < rows; r++) {
                int c_idx = r + k;
                if (c_idx >= 0 && c_idx < cols)
                    buf[r * cols + c_idx] = 1.0f;
            }
            ggml_backend_tensor_set(t, buf, 0, n * sizeof(float));
            free(buf);
        }
    }

    /* Fill NMS param tensors with [max_boxes, iou_thresh, score_thresh] */
    for (int i = 0; i < c->n_nms_deferred; i++) {
        struct ggml_tensor *t = c->nms_param_tensors[i];
        if (!t || !t->buffer) continue;
        float params[3];
        params[0] = (float)c->nms_max_boxes[i];
        memcpy(&params[1], &c->nms_iou_thresh[i], sizeof(float));
        memcpy(&params[2], &c->nms_score_thresh[i], sizeof(float));
        ggml_backend_tensor_set(t, params, 0, 3 * sizeof(float));
    }
}

/* ── Build full graph ───────────────────────────────────────────── */

onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device, int n_threads,
                                  enum ggml_type model_dtype) {
    onnx_reset_unsupported_warnings();
    onnx_ggml_ctx_t *c = calloc(1, sizeof(onnx_ggml_ctx_t));
    if (!c) return NULL;
    c->onnx = onnx;
    c->model_dtype = (model_dtype == GGML_TYPE_F16) ? GGML_TYPE_F16 : GGML_TYPE_F32;

    /* Estimate memory: rough heuristic based on file size */
    size_t mem_size = onnx->mmap_size * 2 + 256 * 1024 * 1024;

    /* Segmented models build one graph per segment in this context and ggml
     * never reclaims within a context, so the graphs accumulate.  Room for
     * them has to be reserved up front: GGML_DEFAULT_GRAPH_SIZE nodes' worth
     * of metadata per segment, plus the tensors each segment's mapping adds. */
    {
        int n_seg_estimate = 0;
        for (int i = 0; i < onnx->n_nodes; i++)
            if (op_is_shape_dependent(onnx->nodes[i].op_type)) n_seg_estimate++;
        if (n_seg_estimate > ONNX_MAX_SEGMENTS) n_seg_estimate = ONNX_MAX_SEGMENTS;
        if (n_seg_estimate > 0)
            mem_size += (size_t)(n_seg_estimate + 1) * ggml_graph_overhead();
    }

    /* ctx_weight: separate context for weight tensors (initializers).
     * These get a dedicated GPU buffer that the scheduler never aliases. */
    {
        /* Weight context needs space for tensor metadata only (no_alloc=true).
         * Includes initializers + Constant/Shape/ConstantOfShape/EyeLike/scalar
         * tensors that are also placed here during map_node.
         * Estimate ~512 bytes per ggml_tensor struct. */
        size_t n_weight_tensors = (size_t)onnx->n_initializers + (size_t)onnx->n_nodes;
        size_t weight_meta = n_weight_tensors * 512 + 64 * 1024;
        struct ggml_init_params wp = {
            .mem_size   = weight_meta,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        c->ctx_weight = ggml_init(wp);
        if (!c->ctx_weight) { free(c); return NULL; }
    }

    /* ctx: context for inputs, graph ops, and intermediate tensors */
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    c->ctx = ggml_init(params);
    if (!c->ctx) { ggml_free(c->ctx_weight); free(c); return NULL; }

    /* ctx_boundary: copies of tensors that outlive their segment.  Metadata
     * only (no_alloc), like ctx_weight -- the data buffer comes later. */
    {
        struct ggml_init_params bp = {
            .mem_size   = (size_t)ONNX_MAX_BOUNDARY * 2 * 512 + 64 * 1024,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        c->ctx_boundary = ggml_init(bp);
        if (!c->ctx_boundary) { ggml_free(c->ctx); ggml_free(c->ctx_weight); free(c); return NULL; }
    }
    c->cur_segment = -1;

    /* Create tensors for initializers (in ctx_weight) and inputs (in ctx) */
    if (create_initializer_tensors(c) != 0) goto fail;
    if (create_input_tensors(c) != 0) goto fail;

    /* Pre-pass: detect pos_embed subgraphs for RelPosBias2D fusion */
    detect_pos_embed_blocks(c);

    /* Pre-pass: cut the graph at data-dependent shape ops.  STAGE 1 -- the
     * segmentation is computed and reported but not yet acted on, so every
     * model still takes the single-pass path below and behaviour is
     * unchanged.  The execution loop lands in stage 2. */
    if (detect_shape_dependent_segments(c) != 0) goto fail;

    /* Map ONNX nodes to ggml ops.
     *
     * Segmented models map only the FIRST segment here; the rest are mapped
     * in onnx_ggml_run, each after the preceding one has been computed and
     * its data-dependent shapes are known.  Mapping them now would defeat
     * the purpose, since the shapes they need do not exist yet. */
    c->cur_segment = 0;
    if (c->n_segments > 1 && onnx_use_segments()) {
        if (map_node_range(c, c->segments[0].first_node,
                              c->segments[0].last_node) != 0) goto fail;
        if (onnx_trace_nodes())
            fprintf(stderr, "[segment] built segment 0 (nodes %d..%d) at load\n",
                    c->segments[0].first_node, c->segments[0].last_node);
    } else {
        if (map_node_range(c, 0, onnx->n_nodes - 1) != 0) goto fail;
    }

    /* Boundary sizes.  The pre-pass counts boundaries but cannot size them --
     * intermediate tensors have no value_info and do not exist yet.  Now that
     * mapping is done they are in tmap, so report the bytes that segmented
     * execution would have to copy, and the heaviest few: if a handful of
     * tensors dominates, that is where any later optimisation belongs. */
    if (onnx_trace_nodes() && c->n_segments > 1) {
        size_t total_bytes = 0;
        size_t top_bytes[5] = {0};
        char   top_name[5][ONNX_MAX_NAME] = {{0}};

        for (int s = 0; s < c->n_segments; s++) {
            for (int i = c->segments[s].first_node; i <= c->segments[s].last_node; i++) {
                for (int o = 0; o < onnx->nodes[i].n_outputs; o++) {
                    const char *nm = onnx->nodes[i].outputs[o];
                    if (nm[0] == '\0') continue;
                    int crosses = 0;
                    for (int s2 = s + 1; s2 < c->n_segments && !crosses; s2++)
                        for (int j = c->segments[s2].first_node;
                             j <= c->segments[s2].last_node && !crosses; j++)
                            for (int a = 0; a < onnx->nodes[j].n_inputs; a++)
                                if (strcmp(onnx->nodes[j].inputs[a], nm) == 0) {
                                    crosses = 1; break;
                                }
                    if (!crosses) continue;

                    struct ggml_tensor *t = tmap_get(c, nm);
                    if (!t) continue;
                    size_t nb = ggml_nbytes(t);
                    total_bytes += nb;
                    for (int k = 0; k < 5; k++) {
                        if (nb > top_bytes[k]) {
                            for (int m = 4; m > k; m--) {
                                top_bytes[m] = top_bytes[m-1];
                                memcpy(top_name[m], top_name[m-1], ONNX_MAX_NAME);
                            }
                            top_bytes[k] = nb;
                            strncpy(top_name[k], nm, ONNX_MAX_NAME - 1);
                            top_name[k][ONNX_MAX_NAME - 1] = '\0';
                            break;
                        }
                    }
                }
            }
        }
        fprintf(stderr, "[segment] boundary bytes total %.1f MB; heaviest:",
                total_bytes / (1024.0 * 1024.0));
        for (int k = 0; k < 5 && top_bytes[k] > 0; k++)
            fprintf(stderr, " %s=%.2fMB", top_name[k], top_bytes[k] / (1024.0 * 1024.0));
        fprintf(stderr, "\n");
    }

    /* Build forward graph.
     *
     * A segment's graph ends at its own cut ops rather than at the model
     * outputs: those outputs are produced by later segments that have not
     * been mapped yet.  Everything a cut op needs is upstream of it, so
     * expanding from the cut ops pulls in exactly this segment's work. */
    {
        c->graph = ggml_new_graph(c->ctx);
        if (c->n_segments > 1 && onnx_use_segments()) {
            build_segment_graph(c, 0);
            /* Remember it: the segment loop reassigns c->graph as it goes, so
             * this pointer is the only way back to segment 0 on a later run. */
            c->seg0_graph  = c->graph;
            c->seg_graphs[0] = c->graph;
        } else {
            for (int i = 0; i < onnx->n_outputs; i++) {
                struct ggml_tensor *t = tmap_get(c, onnx->outputs[i].name);
                if (t) {
                    ggml_set_output(t);
                    ggml_build_forward_expand(c->graph, t);
                }
            }
        }
    }

    /* Choose backends — always have CPU, optionally add Vulkan */
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) goto fail;
    if (n_threads < 1) n_threads = 1;
    ggml_backend_cpu_set_n_threads(c->backend_cpu, n_threads);

    c->backend_gpu = NULL;
    int use_vulkan = 0;
    if (device == NULL || strcmp(device, "vulkan") == 0) {
#ifdef GGML_USE_VULKAN
        use_vulkan = 1;
#else
        if (device && strcmp(device, "vulkan") == 0) {
            fprintf(stderr, "onnx_ggml: Vulkan not available, falling back to CPU\n");
        }
#endif
    }

    if (use_vulkan) {
#ifdef GGML_USE_VULKAN
        c->backend_gpu = ggml_backend_vk_init(0);
        if (!c->backend_gpu) {
            fprintf(stderr, "onnx_ggml: Vulkan init failed, using CPU only\n");
        }
#endif
    }

    /* Allocate weight buffer on preferred backend and load weights once.
     * After this, all tensors in ctx_weight have ->buffer set, so the
     * scheduler will skip them and never alias intermediate results
     * over weight data. */
    {
        ggml_backend_t weight_backend = c->backend_gpu ? c->backend_gpu : c->backend_cpu;
        c->weight_buf = ggml_backend_alloc_ctx_tensors(c->ctx_weight, weight_backend);
        if (onnx_trace_nodes())
            fprintf(stderr, "[wbuf] initial weight_buf=%s\n",
                    c->weight_buf ? "allocated" : "NULL (deferred fills will be skipped)");
        /* weight_buf may be NULL if there are no initializers — that's OK */
        if (c->weight_buf) {
            /* Load initializer weights */
            if (load_weights(c) != 0) goto fail;

            /* Constant node payloads are filled by fill_deferred_tensors()
             * below, which runs them through the same dtype conversion as
             * initializers.  A separate loader used to sit here and handled
             * only F32->F16, falling back to a raw memcpy for everything
             * else: an INT64 index landed in an I32 tensor as the low half of
             * whatever eight bytes the file held, or as nothing at all when
             * the value arrived packed rather than as raw_data. */

            fill_deferred_tensors(c);

            /* Note: strided Slice fills are deferred to first run
             * (sched_alloc_and_fill) because their src may be in sched buffer */
        }
    }

    /* Allocate host-visible pinned staging buffer for fast input transfer.
     * When ggml_backend_tensor_set is called with a pointer inside this buffer,
     * Vulkan detects pinned memory and does direct DMA (no staging copy). */
#ifdef GGML_USE_VULKAN
    if (c->backend_gpu) {
        ggml_backend_buffer_type_t hbuft = ggml_backend_vk_host_buffer_type();
        if (hbuft) {
            /* Estimate max input size: sum of all non-initializer inputs */
            size_t total_input_bytes = 0;
            for (int i = 0; i < c->onnx->n_inputs; i++) {
                onnx_value_info_t *vi = &c->onnx->inputs[i];
                if (tmap_get(c, vi->name) == NULL) continue; /* skip if not created */
                struct ggml_tensor *t = tmap_get(c, vi->name);
                if (t->buffer) continue; /* skip weights (already have buffer) */
                total_input_bytes += ggml_nbytes(t);
            }
            if (total_input_bytes > 0) {
                /* Add alignment padding */
                total_input_bytes += 4096;
                c->pinned_buf = ggml_backend_buft_alloc_buffer(hbuft, total_input_bytes);
                if (c->pinned_buf) {
                    c->pinned_ptr  = ggml_backend_buffer_get_base(c->pinned_buf);
                    c->pinned_size = total_input_bytes;
                }
            }
        }
    }
#endif

    /* Create scheduler with CPU fallback */
    {
        ggml_backend_t backends[2];
        int n_backends = 0;
        if (c->backend_gpu) {
            backends[n_backends++] = c->backend_gpu;
        }
        backends[n_backends++] = c->backend_cpu;

        c->sched = ggml_backend_sched_new(
            backends, NULL, n_backends,
            GGML_DEFAULT_GRAPH_SIZE,
            false,  /* parallel */
            true    /* op_offload — let sched pick best backend per op */
        );
        if (!c->sched) goto fail;
    }

    return c;

fail:
    onnx_ggml_free(c);
    return NULL;
}

/* ── Node ring buffer (ONNX_TRACE_RING=1) ─────────────────────────────
 *
 * Keeps the last RING_N nodes with their values, and dumps them when the run
 * dies.  A full per-node trace answers the same question, but a corrupting
 * run produces thousands of lines before it falls over and the interesting
 * three are wherever the terminal buffer happens to have kept them; a ring
 * holds exactly the tail that matters and costs nothing until it is dumped.
 *
 * Values are printed first4 + last4.  Corruption shows at the edges far more
 * often than in the middle -- an off-by-one write, a stale buffer whose head
 * still holds the previous run's data, an index list running past its table --
 * and min/max/mean (what ONNX_DIFF_DEBUG reports) hides exactly that: a single
 * wild element at the end moves the max and nothing else.
 *
 * Dumped from r_ggml_abort, because a ggml assertion reaches R through
 * Rf_error, which longjmps: there is no return to any code that could print
 * afterwards, and atexit never runs since R itself does not exit.  The hook is
 * a plain function pointer so that r_ggml_io.c, which knows nothing about
 * ONNX, does not have to link against this file. */

#define ONNX_RING_N     3
#define ONNX_RING_EDGE  4

typedef struct {
    char    name[64];
    int     op;
    int     idx;
    int64_t ne[4];
    int64_t nel;
    char    type[16];
    int     phase;          /* 0 = before compute (ask), 1 = after */
    int     have_vals;
    double  first[ONNX_RING_EDGE];
    double  last[ONNX_RING_EDGE];
} onnx_ring_slot_t;

static onnx_ring_slot_t g_ring[ONNX_RING_N];
static int g_ring_pos   = 0;    /* next slot to write */
static int g_ring_count = 0;    /* how many slots are live */

int onnx_trace_ring(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ONNX_TRACE_RING");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

/* Read the leading and trailing ONNX_RING_EDGE values of a tensor.
 *
 * Reads only the two edges, never the whole tensor: a node in the middle of
 * MaskRCNN can hold millions of elements, and pulling all of them back from
 * the backend for every node would change the timing enough to move the very
 * bug being chased. */
static int onnx_ring_read_edges(const struct ggml_tensor *t,
                                double *first, double *last) {
    int64_t nel = ggml_nelements(t);
    if (nel <= 0 || !t->buffer || !t->data) return 0;

    size_t esz;
    switch (t->type) {
        case GGML_TYPE_F32: esz = sizeof(float);       break;
        case GGML_TYPE_I32: esz = sizeof(int32_t);     break;
        case GGML_TYPE_F16: esz = sizeof(ggml_fp16_t); break;
        default: return 0;   /* quantised types have no element-wise read */
    }
    /* A non-contiguous tensor is read only at its very start, where the
     * strides have not diverged from a flat layout yet.  Skipping such tensors
     * outright, as this did, silenced the trace over most of a transformer:
     * permutes and views are half its nodes, and the answer to "where did the
     * NaN come from" was in the half that was not looked at. */
    int strided = !ggml_is_contiguous(t);

    int n = nel < ONNX_RING_EDGE ? (int)nel : ONNX_RING_EDGE;
    char raw[ONNX_RING_EDGE * 8];

    for (int half = 0; half < 2; half++) {
        /* half 0 = leading n elements, half 1 = trailing n.  For a strided
         * tensor the tail offset means nothing, so both halves read the head
         * and the caller simply sees the same values twice. */
        int64_t off = (half && !strided) ? (nel - n) : 0;
        double *out = half ? last : first;
        ggml_backend_tensor_get((struct ggml_tensor *)t, raw,
                                (size_t)off * esz, (size_t)n * esz);
        for (int i = 0; i < n; i++) {
            switch (t->type) {
                case GGML_TYPE_F32: out[i] = ((float *)raw)[i];   break;
                case GGML_TYPE_I32: out[i] = ((int32_t *)raw)[i]; break;
                default: out[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)raw)[i]); break;
            }
        }
        for (int i = n; i < ONNX_RING_EDGE; i++) out[i] = 0.0;
    }
    return n;
}

static void onnx_ring_record(const struct ggml_tensor *t, int idx, int phase) {
    onnx_ring_slot_t *s = &g_ring[g_ring_pos];

    snprintf(s->name, sizeof(s->name), "%s", t->name);
    snprintf(s->type, sizeof(s->type), "%s", ggml_type_name(t->type));
    s->op    = (int)t->op;
    s->idx   = idx;
    s->phase = phase;
    s->nel   = ggml_nelements(t);
    for (int d = 0; d < 4; d++) s->ne[d] = t->ne[d];
    s->have_vals = onnx_ring_read_edges(t, s->first, s->last);

    g_ring_pos = (g_ring_pos + 1) % ONNX_RING_N;
    if (g_ring_count < ONNX_RING_N) g_ring_count++;
}

static void onnx_ring_print_edges(const char *label, const onnx_ring_slot_t *s) {
    if (!s->have_vals) {
        fprintf(stderr, "      %s: <not readable>\n", label);
        return;
    }
    fprintf(stderr, "      %s: first", label);
    for (int i = 0; i < s->have_vals; i++) fprintf(stderr, " %g", s->first[i]);
    /* Only worth printing the tail when it is not the head over again. */
    if (s->nel > s->have_vals) {
        fprintf(stderr, "  last");
        for (int i = 0; i < s->have_vals; i++) fprintf(stderr, " %g", s->last[i]);
    }
    fprintf(stderr, "\n");
}

/* Dump the ring oldest-first, so it reads in execution order. */
void onnx_ring_dump(void) {
    if (!g_ring_count) return;

    fprintf(stderr, "\n=== last %d graph nodes before the failure ===\n",
            g_ring_count);
    int start = (g_ring_pos - g_ring_count + ONNX_RING_N) % ONNX_RING_N;
    for (int k = 0; k < g_ring_count; k++) {
        const onnx_ring_slot_t *s = &g_ring[(start + k) % ONNX_RING_N];
        fprintf(stderr, "  [-%d] node %d '%s' op=%d %s ne=[%lld,%lld,%lld,%lld] "
                        "nel=%lld (%s)\n",
                g_ring_count - 1 - k, s->idx, s->name, s->op, s->type,
                (long long)s->ne[0], (long long)s->ne[1],
                (long long)s->ne[2], (long long)s->ne[3],
                (long long)s->nel, s->phase ? "computed" : "pre-compute");
        onnx_ring_print_edges("vals", s);
    }
    fprintf(stderr, "=== end of ring ===\n\n");
}

/* Does this tensor's readable edge hold a NaN or an infinity?
 *
 * Only the edges are examined, for the same reason the ring reads only edges:
 * a full scan of every node changes the timing of the run.  A NaN that sits
 * strictly in the middle of a tensor and touches neither end is therefore
 * missed -- acceptable, because a NaN produced by a bad weight or a blank
 * buffer covers the whole tensor rather than one interior element. */
static int onnx_edge_has_nan(const struct ggml_tensor *t) {
    double first[ONNX_RING_EDGE], last[ONNX_RING_EDGE];
    int n = onnx_ring_read_edges(t, first, last);
    if (!n) return 0;
    for (int i = 0; i < n; i++) {
        if (first[i] != first[i] || last[i] != last[i]) return 1;
        if (first[i] > 3.4e38 || first[i] < -3.4e38) return 1;
        if (last[i]  > 3.4e38 || last[i]  < -3.4e38) return 1;
    }
    return 0;
}

/* Report the first node that turns finite inputs into a NaN output.
 *
 * A NaN anywhere downstream of the first one is just the first one spreading,
 * so a per-node NaN log is mostly noise; what identifies the defect is the one
 * node whose sources are all clean and whose result is not.  Reported once and
 * then left alone, since everything after it is a consequence. */
static int g_nan_reported = 0;
/* Node counter shared by every scheduler the ring is attached to, so the
 * numbering runs continuously across segments instead of restarting. */
static int g_ring_node_idx = 0;

static bool onnx_ring_eval_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    int *idx = (int *)user_data;

    if (ask) {
        /* Record the srcs, so the ring shows what this node was fed. */
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) continue;
            onnx_ring_record(t->src[s], *idx, 0);
        }
        return true;
    }
    onnx_ring_record(t, *idx, 1);

    /* First node to manufacture a NaN out of clean inputs.  Everything after
     * it inherits the NaN, so only this one names the defect. */
    if (!g_nan_reported && onnx_edge_has_nan(t)) {
        int src_clean = 1;
        for (int s = 0; s < GGML_MAX_SRC; s++)
            if (t->src[s] && onnx_edge_has_nan(t->src[s])) src_clean = 0;
        if (src_clean) {
            g_nan_reported = 1;
            fprintf(stderr, "\n=== first NaN: node %d '%s' op=%d, inputs are clean ===\n",
                    *idx, t->name, (int)t->op);
            onnx_ring_dump();
        }
    }

    (*idx)++;
    return true;
}

/* Attach the ring to one scheduler.
 *
 * Every scheduler that computes anything needs its own attachment: the ring
 * used to be installed on ctx->sched alone, which covered the whole model
 * while one scheduler ran it, but stopped at segment 0 once the later segments
 * moved to schedulers of their own -- so the trace went quiet over exactly the
 * thousand nodes worth looking at. */
static void onnx_ring_attach(ggml_backend_sched_t sch) {
    if (!sch) return;
    ggml_backend_sched_set_eval_callback(sch, onnx_ring_eval_cb, &g_ring_node_idx);
}

#ifdef ONNX_DIFF_DEBUG
/* Per-node eval callback: writes node name + output stats (min/max/mean/v0) to a log
 * file. Set ONNX_DIFF_LOG env var to the output path before running. */
typedef struct { int idx; FILE *fp; } diff_cb_state_t;

static bool onnx_diff_eval_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    if (ask) return true;  /* only post-compute */
    diff_cb_state_t *st = (diff_cb_state_t *)user_data;
    int64_t nel = ggml_nelements(t);
    if (nel <= 0 || t->type != GGML_TYPE_F32) { st->idx++; return true; }
    /* read up to 1024 elements */
    int64_t n = nel < 1024 ? nel : 1024;
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) { st->idx++; return true; }
    ggml_backend_tensor_get(t, buf, 0, n * sizeof(float));
    float v0 = buf[0];
    float mn = buf[0], mx = buf[0], sm = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
        sm += buf[i];
    }
    float mean = sm / n;
    free(buf);
    fprintf(st->fp, "%d\t%s\top=%d\tne=[%lld,%lld,%lld,%lld]\tv0=%g\tmin=%g\tmax=%g\tmean=%g\n",
            st->idx, t->name, (int)t->op,
            (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3],
            v0, mn, mx, mean);
    fflush(st->fp);
    st->idx++;
    return true;
}
#endif

#ifdef ONNX_NAN_DEBUG
/* Per-node eval callback: called before (ask=true) and after (ask=false) each node.
 * Checks inputs before, output after. Stops at first NaN. */
/* Scan entire tensor for NaN/Inf, return count. buf must hold nelements floats. */
static int onnx_scan_nan(struct ggml_tensor *t, int *n_nan, int *n_inf) {
    /* Only meaningful for F32 — integer types read as float give false NaN */
    if (t->type != GGML_TYPE_F32) { *n_nan = 0; *n_inf = 0; return 0; }
    int64_t n = ggml_nelements(t);
    if (n <= 0) return 0;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return 0;
    ggml_backend_tensor_get(t, buf, 0, (size_t)n * sizeof(float));
    *n_nan = 0; *n_inf = 0;
    for (int64_t i = 0; i < n; i++) {
        if (buf[i] != buf[i]) (*n_nan)++;
        else if (buf[i] > 3.4e38f || buf[i] < -3.4e38f) (*n_inf)++;
    }
    free(buf);
    return *n_nan + *n_inf;
}

static bool onnx_nan_eval_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    int *idx = (int *)user_data;
    if (ask) {
        /* Before compute: scan all src inputs for NaN */
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) continue;
            int n_nan = 0, n_inf = 0;
            if (onnx_scan_nan(t->src[s], &n_nan, &n_inf) > 0) {
                fprintf(stderr, "[NaN-IN] node %d '%s' op=%d: src[%d] '%s' "
                        "ne=[%lld,%lld,%lld,%lld] nan=%d inf=%d\n",
                        *idx, t->name, (int)t->op, s, t->src[s]->name,
                        (long long)t->src[s]->ne[0], (long long)t->src[s]->ne[1],
                        (long long)t->src[s]->ne[2], (long long)t->src[s]->ne[3],
                        n_nan, n_inf);
            }
        }
        return true;
    }
    /* After compute: scan output */
    int n_nan = 0, n_inf = 0;
    if (onnx_scan_nan(t, &n_nan, &n_inf) > 0) {
        fprintf(stderr, "[NaN-OUT] node %d '%s' op=%d ne=[%lld,%lld,%lld,%lld]: nan=%d inf=%d\n",
                *idx, t->name, (int)t->op,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3], n_nan, n_inf);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) continue;
            float sv = 0.0f;
            struct ggml_tensor *src = t->src[s];
            if (ggml_nelements(src) > 0) {
                if (src->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(src, &sv, 0, sizeof(float));
                } else if (src->type == GGML_TYPE_F16) {
                    ggml_fp16_t hv = 0;
                    ggml_backend_tensor_get(src, &hv, 0, sizeof(ggml_fp16_t));
                    sv = ggml_fp16_to_fp32(hv);
                } else if (src->type == GGML_TYPE_BF16) {
                    ggml_bf16_t bv = {0};
                    ggml_backend_tensor_get(src, &bv, 0, sizeof(ggml_bf16_t));
                    sv = ggml_bf16_to_fp32(bv);
                } else if (src->type == GGML_TYPE_I32) {
                    int32_t iv = 0;
                    ggml_backend_tensor_get(src, &iv, 0, sizeof(int32_t));
                    sv = (float)iv;
                }
            }
            fprintf(stderr, "  src[%d] '%s' type=%d ne=[%lld,%lld,%lld,%lld] v0=%g\n",
                    s, src->name, (int)src->type,
                    (long long)src->ne[0], (long long)src->ne[1],
                    (long long)src->ne[2], (long long)src->ne[3], sv);
        }
        (*idx)++;
        return false;
    }
    if (*idx < 5) {
        float v0 = 0.0f;
        if (ggml_nelements(t) > 0) {
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, &v0, 0, sizeof(float));
            } else if (t->type == GGML_TYPE_F16) {
                ggml_fp16_t hv = 0;
                ggml_backend_tensor_get(t, &hv, 0, sizeof(ggml_fp16_t));
                v0 = ggml_fp16_to_fp32(hv);
            } else if (t->type == GGML_TYPE_BF16) {
                ggml_bf16_t bv = {0};
                ggml_backend_tensor_get(t, &bv, 0, sizeof(ggml_bf16_t));
                v0 = ggml_bf16_to_fp32(bv);
            } else if (t->type == GGML_TYPE_I32) {
                int32_t iv = 0;
                ggml_backend_tensor_get(t, &iv, 0, sizeof(int32_t));
                v0 = (float)iv;
            }
        }
        fprintf(stderr, "[OK] node %d '%s' op=%d type=%d v0=%g\n",
                *idx, t->name, (int)t->op, (int)t->type, v0);
    }
    (*idx)++;
    return true;  /* continue */
}
#endif

/* Write the caller's data into the model's input tensors.
 *
 * Called once per segment rather than once per run: each segment reallocates
 * the scheduler's buffers, and an input tensor living in one of them comes
 * back pointing at fresh, uninitialised memory.  Segment 0 would then run on
 * the real data and every later segment on garbage -- which shows up as an
 * embedding lookup with a wild index rather than as an obviously wrong
 * number. */
static int set_model_inputs(onnx_ggml_ctx_t *ctx,
                            const char **input_names, const float **input_data,
                            const int64_t *input_lens, int n_inputs) {
    /* Set input data.
     * When a pinned staging buffer is available, copy data there first so that
     * ggml_backend_tensor_set detects pinned source and does direct DMA
     * (skipping the internal staging copy). */
    size_t pinned_offset = 0;
    for (int i = 0; i < n_inputs; i++) {
        struct ggml_tensor *t = tmap_get(ctx, input_names[i]);
        if (!t) {
            fprintf(stderr, "onnx_ggml: input '%s' not found\n", input_names[i]);
            return -1;
        }
        /* Never read more elements than the caller actually supplied.
         *
         * The tensor's own size is not a safe bound: segmented execution can
         * rebuild an input-facing tensor at a different size between calls,
         * and reading ggml_nelements(t) from a shorter caller array walks off
         * the end of it -- memory owned by R, corrupted silently and blamed on
         * a later allocation somewhere else entirely. */
        int64_t avail = input_lens ? input_lens[i] : ggml_nelements(t);

        if (t->type == GGML_TYPE_I32) {
            /* Input is integer (e.g. token IDs for Gather/embedding).
             * Caller passes float — convert to int32. */
            int64_t nel = ggml_nelements(t);
            if (nel > avail) nel = avail;
            size_t  nbytes = nel * sizeof(int32_t);
            if (ctx->pinned_ptr && pinned_offset + nbytes <= ctx->pinned_size) {
                int32_t *dst = (int32_t *)((char *)ctx->pinned_ptr + pinned_offset);
                for (int64_t j = 0; j < nel; j++)
                    dst[j] = (int32_t)input_data[i][j];
                ggml_backend_tensor_set(t, dst, 0, nbytes);
                pinned_offset += nbytes;
            } else {
                int32_t *ibuf = (int32_t *)malloc(nbytes);
                if (!ibuf) return -1;
                for (int64_t j = 0; j < nel; j++)
                    ibuf[j] = (int32_t)input_data[i][j];
                ggml_backend_tensor_set(t, ibuf, 0, nbytes);
                free(ibuf);
            }
        } else {
            size_t nbytes = ggml_nbytes(t);
            size_t avail_bytes = (size_t)avail * sizeof(float);
            if (nbytes > avail_bytes) nbytes = avail_bytes;
            if (ctx->pinned_ptr && pinned_offset + nbytes <= ctx->pinned_size) {
                void *dst = (char *)ctx->pinned_ptr + pinned_offset;
                memcpy(dst, input_data[i], nbytes);
                ggml_backend_tensor_set(t, dst, 0, nbytes);
                pinned_offset += nbytes;
            } else {
                ggml_backend_tensor_set(t, input_data[i], 0, nbytes);
            }
        }
    }
    return 0;
}

/* ── Segment graph cache ───────────────────────────────────────────
 *
 * A segmented model rebuilds itself on every inference: the loop re-maps each
 * segment's nodes, allocates their tensors afresh and builds a new graph.
 * Since a ggml context releases nothing before it is destroyed, that growth is
 * permanent -- roberta added ~195 constants and ~193 shape tensors per call
 * and its weight context ran dry on the fourth.  The graphs are identical
 * between runs whenever the data-dependent sizes come out the same, so they
 * are kept and reused instead.
 *
 * The key is those sizes.  A NonZero that finds a different number of non-zero
 * elements changes every shape downstream of the cut, so its graphs are no
 * longer valid; anything else about the input may differ freely.  This is the
 * same bargain a shape-keyed engine cache makes: reuse while the shapes hold,
 * rebuild when they move. */

/* Do the sizes just measured match the ones the cached graphs were built for?
 *
 * Order matters as much as content: resolve_segment_sizes appends in segment
 * order, so a differing count or a differing name at the same index both mean
 * the resolution went differently this time. */
/* The scheduler segment `s` runs on, created on first use.
 *
 * Each segment keeps its own so that the allocation made for its graph is
 * still there next time: a scheduler re-reserves its arena whenever it is
 * handed a graph unlike the last one it saw, and segments differ, so a shared
 * scheduler would tear down each segment's placement as the following segment
 * arrived.  Segment 0 keeps using ctx->sched, which the build already made. */
static ggml_backend_sched_t seg_sched(onnx_ggml_ctx_t *c, int s) {
    if (s == 0) return c->sched;
    if (s >= ONNX_MAX_SEGMENTS) return c->sched;
    if (c->seg_scheds[s]) return c->seg_scheds[s];

    ggml_backend_t backends[2];
    int n_backends = 0;
    if (c->backend_gpu) backends[n_backends++] = c->backend_gpu;
    backends[n_backends++] = c->backend_cpu;

    c->seg_scheds[s] = ggml_backend_sched_new(backends, NULL, n_backends,
                                              GGML_DEFAULT_GRAPH_SIZE,
                                              false, true);
    if (!c->seg_scheds[s]) {
        fprintf(stderr, "[onnx] could not create scheduler for segment %d\n", s);
        return NULL;
    }
    if (onnx_trace_ring()) onnx_ring_attach(c->seg_scheds[s]);
    if (onnx_trace_nodes())
        fprintf(stderr, "[segcache] new scheduler for segment %d\n", s);
    return c->seg_scheds[s];
}

static int seg_cache_key_matches(const onnx_ggml_ctx_t *c) {
    if (c->n_seg_key != c->n_resolved) return 0;
    for (int i = 0; i < c->n_resolved; i++) {
        if (c->seg_key_sizes[i] != c->resolved_sizes[i]) return 0;
        if (strcmp(c->seg_key_names[i], c->resolved_names[i]) != 0) return 0;
    }
    return 1;
}

/* Record the sizes the cached graphs correspond to. */
static void seg_cache_store_key(onnx_ggml_ctx_t *c) {
    int n = c->n_resolved < ONNX_MAX_RESOLVED ? c->n_resolved : ONNX_MAX_RESOLVED;
    for (int i = 0; i < n; i++) {
        c->seg_key_sizes[i] = c->resolved_sizes[i];
        strncpy(c->seg_key_names[i], c->resolved_names[i], ONNX_MAX_NAME - 1);
        c->seg_key_names[i][ONNX_MAX_NAME - 1] = '\0';
    }
    c->n_seg_key = n;
}


int onnx_ggml_run(onnx_ggml_ctx_t *ctx,
                  const char **input_names, const float **input_data,
                  const int64_t *input_lens, int n_inputs) {
    if (!ctx->graph || !ctx->sched) return -1;

    /* Put segment 0's graph back, if the segment loop has moved on from it.
     *
     * ctx->graph is reassigned per segment as that loop runs, so on entry to
     * any run after the first it names whichever segment finished last, and
     * computing that would start the model from its middle.  On the first run
     * it already IS segment 0's graph and the scheduler has yet to allocate,
     * so this both is unnecessary and would double-allocate: sched asserts
     * !is_alloc, and the block below allocates too. */
    if (ctx->seg0_graph && ctx->graph != ctx->seg0_graph) {
        /* Segment 0 owns ctx->sched, and its placement there is untouched:
         * the later segments now allocate on schedulers of their own, so
         * nothing has re-reserved this arena since segment 0 last ran.  Only
         * the graph pointer has to come back. */
        ctx->graph = ctx->seg0_graph;
        /* Back to segment 0 in the bookkeeping too: cur_segment still held the
         * last segment of the previous run, which made every diagnostic from
         * here on name the wrong one. */
        ctx->cur_segment = 0;
        fill_deferred_tensors(ctx);
        if (onnx_trace_nodes())
            fprintf(stderr, "[segcache] restored segment 0 graph (%d nodes)\n",
                    ggml_graph_n_nodes(ctx->graph));
    } else if (!ctx->is_allocated) {
        /* First run: allocate compute buffers and fill small deferred tensors.
         * Weights live in a separate weight_buf that sched never touches,
         * so subsequent runs need NO reload — just set inputs and compute. */
        if (sched_alloc_and_fill(ctx) != 0) return -1;
        ctx->is_allocated = 1;
    }

    if (set_model_inputs(ctx, input_names, input_data, input_lens, n_inputs) != 0) return -1;

#ifdef ONNX_DIFF_DEBUG
    diff_cb_state_t diff_st = {0, NULL};
    const char *diff_log = getenv("ONNX_DIFF_LOG");
    if (diff_log) {
        diff_st.fp = fopen(diff_log, "w");
        if (diff_st.fp)
            ggml_backend_sched_set_eval_callback(ctx->sched, onnx_diff_eval_cb, &diff_st);
    }
#endif
#ifdef ONNX_NAN_DEBUG
    /* Install per-node eval callback for NaN tracing */
    static int nan_node_idx = 0;
    nan_node_idx = 0;
    ggml_backend_sched_set_eval_callback(ctx->sched, onnx_nan_eval_cb, &nan_node_idx);
#endif

    /* Ring tracing (ONNX_TRACE_RING=1), gated at runtime rather than behind a
     * #define so that a misbehaving run can be examined on the build that
     * misbehaved -- rebuilding to switch diagnostics on changes the binary
     * under investigation.
     *
     * It earns its place: the NaN that reached roberta's output was traced to
     * a single node in one run, after four separate readings of the code had
     * each proposed a cause that turned out to be wrong.  Costs nothing while
     * the variable is unset. */
    if (onnx_trace_ring()) {
        g_ring_node_idx = 0;
        g_nan_reported = 0;
        r_ggml_abort_hook = onnx_ring_dump;
        onnx_ring_attach(ctx->sched);
    }


    if (onnx_trace_nodes())
        fprintf(stderr, "[segment] %d: computing %d graph nodes (first compute)\n",
                ctx->cur_segment, ggml_graph_n_nodes(ctx->graph));

    enum ggml_status status = ggml_backend_sched_graph_compute(ctx->sched, ctx->graph);

    /* Segmented models: the compute above finished segment 0 only.  Walk the
     * remaining segments, each time saving what later segments still need,
     * mapping the next batch of nodes now that the shapes they depend on are
     * known, and computing again.
     *
     * STAGE 2a -- the shapes read back are still the build-time guesses
     * (NonZero keeps assuming every element is non-zero), so this exercises
     * the segment machinery without yet fixing any shape.  Reading the real
     * sizes is stage 2b. */
    if (status == GGML_STATUS_SUCCESS &&
        ctx->n_segments > 1 && onnx_use_segments()) {
        /* Whether this run can reuse the graphs the last one built.  Decided
         * per segment rather than once up front: the sizes are measured as the
         * run proceeds, so the earliest a mismatch can be seen is after the
         * segment that produced it. */
        int reuse = (ctx->n_seg_graphs == ctx->n_segments);

        for (int s = 1; s < ctx->n_segments; s++) {
            /* The segment just computed has given its cut ops real inputs, so
             * their true output sizes can be measured now. */
            if (resolve_segment_sizes(ctx, s - 1) != 0) return -1;

            /* A size that came out differently invalidates every graph from
             * this cut onwards, so drop the cache and build from here. */
            if (reuse && !seg_cache_key_matches(ctx)) {
                reuse = 0;
                if (onnx_trace_nodes())
                    fprintf(stderr, "[segcache] miss at segment %d: "
                            "resolved sizes changed, rebuilding\n", s);
            }

            if (reuse) {
                /* Same shapes as last time, so this segment's graph and the
                 * placement its own scheduler made for it both still hold.
                 * Nothing is re-mapped and nothing is re-allocated -- that
                 * work is what grew the ggml context on every call -- only the
                 * values that change per run are written before computing. */
                ggml_backend_sched_t sch = seg_sched(ctx, s);
                if (!sch) return -1;
                ctx->cur_segment = s;
                ctx->graph = ctx->seg_graphs[s];
                fill_deferred_tensors(ctx);
                if (set_model_inputs(ctx, input_names, input_data,
                                     input_lens, n_inputs) != 0) return -1;
                if (onnx_trace_nodes())
                    fprintf(stderr, "[segcache] reuse segment %d (%d nodes)\n",
                            s, ggml_graph_n_nodes(ctx->graph));
                status = ggml_backend_sched_graph_compute(sch, ctx->graph);
                if (status != GGML_STATUS_SUCCESS) break;
                continue;
            }

            /* Re-map the cut ops themselves.  They were first mapped as part
             * of their own segment, necessarily before their inputs existed,
             * so they were built at the guessed size; mapping them again now
             * rebuilds them at the measured one.  tmap_put_nd appends and
             * tmap_get searches backwards, so the new tensor simply shadows
             * the old for every consumer that follows. */
            for (int j = 0; j < ctx->segments[s - 1].n_cut_nodes; j++) {
                int ni = ctx->segments[s - 1].cut_nodes[j];
                if (onnx_resolved_size(ctx, ctx->onnx->nodes[ni].outputs[0]) < 0)
                    continue;   /* nothing measured -- leave the original */
                if (map_node_range(ctx, ni, ni) != 0) return -1;
            }

            if (copy_segment_boundaries(ctx, s - 1) != 0) return -1;

            ctx->cur_segment = s;
            if (map_node_range(ctx, ctx->segments[s].first_node,
                                    ctx->segments[s].last_node) != 0) return -1;

            /* Weight-like tensors this segment added (Constant, scalars,
             * NonZero/NMS outputs) still need a buffer of their own. */
            if (alloc_new_weight_tensors(ctx) != 0) return -1;

            /* A fresh graph for this segment: up to its own cut ops, or to
             * the model outputs for the last one. */
            /* Each segment needs its own graph, and ggml contexts do not free
             * anything until the whole context goes -- so the graphs, and the
             * tensors every segment adds, accumulate.  A context that runs out
             * of room returns NULL here rather than growing, and continuing
             * past that point corrupts the run in ways that surface much
             * later (a graph pointing at freed scheduler buffers takes the
             * whole R process down during cleanup). */
            ctx->graph = ggml_new_graph(ctx->ctx);
            if (!ctx->graph) {
                fprintf(stderr, "[onnx] out of context memory building segment "
                                "%d/%d -- model too large for segmented execution\n",
                        s, ctx->n_segments);
                return -1;
            }
            build_segment_graph(ctx, s);

            /* Keep it for the next run.  Graph and tensors both live in
             * ctx->ctx, which outlives the run, so the pointer stays good for
             * as long as the resolved sizes it was built for hold. */
            if (s < ONNX_MAX_SEGMENTS) ctx->seg_graphs[s] = ctx->graph;

            /* Build the placement on this segment's own scheduler, the one a
             * cached run will compute on.  Allocating on the shared scheduler
             * here and computing on the per-segment one later would leave the
             * graph's tensors pointing into an arena nobody is maintaining. */
            ggml_backend_sched_t sch = seg_sched(ctx, s);
            if (!sch) return -1;
            ggml_backend_sched_reset(sch);
            if (sched_alloc_and_fill_on(ctx, sch) != 0) return -1;

            /* This segment registered its own Shape/ConstantOfShape/NonZero/
             * EyeLike/NMS tensors, which only now have buffers to write into.
             * Without this their contents are undefined -- a NonZero index
             * list full of garbage indexes rows outside the embedding table. */
            fill_deferred_tensors(ctx);

            /* The reallocation above handed the input tensors new memory, so
             * the caller's data has to be written again -- otherwise this
             * segment reads whatever the fresh buffer happened to hold. */
            if (set_model_inputs(ctx, input_names, input_data, input_lens, n_inputs) != 0)
                return -1;

            if (onnx_trace_nodes())
                fprintf(stderr, "[segment] %d: computing %d graph nodes\n",
                        s, ggml_graph_n_nodes(ctx->graph));

            status = ggml_backend_sched_graph_compute(sch, ctx->graph);
            if (status != GGML_STATUS_SUCCESS) break;
        }

        /* Publish the cache only for a run that finished.  A run that broke
         * partway left tmap and the boundary copies describing a state the
         * saved graphs do not match, and reusing them would compute over it. */
        if (status == GGML_STATUS_SUCCESS) {
            ctx->n_seg_graphs = ctx->n_segments;
            seg_cache_store_key(ctx);
            if (onnx_trace_nodes())
                fprintf(stderr, "[segcache] stored %d segment graphs, "
                        "key = %d resolved sizes\n",
                        ctx->n_seg_graphs, ctx->n_seg_key);
        }
    }

#ifdef ONNX_DIFF_DEBUG
    if (diff_st.fp) {
        fclose(diff_st.fp);
        ggml_backend_sched_set_eval_callback(ctx->sched, NULL, NULL);
    }
#endif
#ifdef ONNX_NAN_DEBUG
    /* Clear callback after compute */
    ggml_backend_sched_set_eval_callback(ctx->sched, NULL, NULL);
#endif

    return (status == GGML_STATUS_SUCCESS) ? 0 : -1;
}

/* Copy the tensors segment `seg` leaves behind into ctx_boundary.
 *
 * A segment's results live in scheduler buffers, which the next segment's
 * allocation reuses.  Anything a later segment reads therefore has to be
 * copied first into a context the scheduler never touches -- the same trick
 * ctx_weight uses for weights.  The copy replaces the original in tmap, so
 * consumers pick it up with no knowledge that a segment boundary was
 * crossed.
 *
 * Only tensors actually read later are copied (the boundary census showed
 * 669 of them for MaskRCNN, peaking at 406 live at once); everything else
 * dies with its segment. */
static int copy_segment_boundaries(onnx_ggml_ctx_t *c, int seg) {
    onnx_model_t *onnx = c->onnx;
    const onnx_segment_t *sg = &c->segments[seg];
    int n_copied = 0;
    size_t bytes = 0;

    for (int i = sg->first_node; i <= sg->last_node; i++) {
        for (int o = 0; o < onnx->nodes[i].n_outputs; o++) {
            const char *nm = onnx->nodes[i].outputs[o];
            if (nm[0] == '\0') continue;

            /* Same test the graph builder uses to decide what to compute --
             * shared so the two cannot disagree about what a boundary is. */
            if (!tensor_crosses_boundary(c, seg, nm)) continue;

            struct ggml_tensor *src = tmap_get(c, nm);
            if (!src) continue;
            if (!src->buffer) {
                /* Produced by this segment but never given memory -- it was
                 * folded away at build time (a cval-only shape tensor, say).
                 * Nothing to copy, and nothing to repoint: the later segment
                 * will rebuild it the same way.  Silently skipping would
                 * instead leave tmap pointing into this segment's scheduler
                 * buffer, which the next allocation reuses. */
                if (onnx_trace_nodes())
                    fprintf(stderr, "[segment] %d: boundary '%s' has no buffer "
                                    "-- not copied\n", seg, nm);
                continue;
            }

            /* Tensors living in ctx_weight or ctx_boundary already survive
             * the segment -- the scheduler never allocates over those buffers,
             * which is the whole reason weights are kept there.  Copying one
             * anyway would be worse than useless: the copy replaces it in
             * tmap, while the deferred-fill lists still point at the original,
             * so a NonZero output (built in ctx_weight) would be filled in the
             * tensor nobody reads and read from the tensor nobody fills. */
            if (src->buffer == c->weight_buf || src->buffer == c->boundary_buf) {
                if (onnx_trace_nodes())
                    fprintf(stderr, "[segment] %d: boundary '%s' is persistent "
                                    "-- not copied\n", seg, nm);
                continue;
            }
            {
                int persistent = 0;
                for (int k = 0; k < c->n_extra_weight_bufs; k++)
                    if (src->buffer == c->extra_weight_bufs[k]) { persistent = 1; break; }
                if (persistent) {
                    if (onnx_trace_nodes())
                        fprintf(stderr, "[segment] %d: boundary '%s' is persistent "
                                        "-- not copied\n", seg, nm);
                    continue;
                }
            }

            /* Copy the full ne[], all GGML_MAX_DIMS of it.  ggml_n_dims stops
             * at the last axis larger than 1, so sizing the copy by it drops
             * every trailing unit axis: roberta's 214 is [128,1,1,1,1] with
             * ONNX rank 4, and a copy built from ggml_n_dims(src)==1 comes
             * back rank-1.  The Shape ops in the next segment read that form
             * and compute positions from the wrong dimensions, which lands as
             * out-of-range rows in get_rows.  The ONNX rank is carried
             * separately through tmap below; the ggml shape has to be exact. */
            struct ggml_tensor *dst =
                ggml_new_tensor(c->ctx_boundary, src->type,
                                GGML_MAX_DIMS, src->ne);
            if (!dst) return -1;
            ggml_set_input(dst);
            ggml_set_name(dst, nm);
            /* Check the bound BEFORE writing: the slot is indexed by the
             * current count, so testing afterwards lets the last write land
             * one past the end of the array. */
            if (c->n_boundary_pending >= ONNX_MAX_BOUNDARY) {
                fprintf(stderr, "[onnx] too many boundary tensors (>%d)\n",
                        ONNX_MAX_BOUNDARY);
                c->n_boundary_pending = 0;
                return -1;
            }
            c->boundary_pending[c->n_boundary_pending].src = src;
            c->boundary_pending[c->n_boundary_pending].dst = dst;
            c->n_boundary_pending++;
            n_copied++;
            bytes += ggml_nbytes(src);
        }
    }

    /* Give the copies memory, then move the data across and repoint tmap. */
    if (n_copied > 0) {
        ggml_backend_t backend = c->backend_gpu ? c->backend_gpu : c->backend_cpu;
        ggml_backend_buffer_t buf =
            ggml_backend_alloc_ctx_tensors(c->ctx_boundary, backend);
        if (buf) {
            if (c->n_extra_weight_bufs >= ONNX_MAX_WEIGHT_BUFS) {
                fprintf(stderr, "[onnx] too many boundary buffers\n");
                ggml_backend_buffer_free(buf);
                return -1;
            }
            c->extra_weight_bufs[c->n_extra_weight_bufs++] = buf;
        }
        for (int k = 0; k < c->n_boundary_pending; k++) {
            struct ggml_tensor *src = c->boundary_pending[k].src;
            struct ggml_tensor *dst = c->boundary_pending[k].dst;
            const char *nm = ggml_get_name(dst);
            if (!dst->buffer) {
                /* The copy was queued but got no memory, so the data cannot
                 * be moved and tmap still points at the segment's own buffer,
                 * which is about to be reused -- a later read would return
                 * garbage or fault.  Fail loudly instead. */
                fprintf(stderr, "[onnx] boundary '%s' got no buffer\n", nm);
                return -1;
            }
            size_t nb = ggml_nbytes(src);
            void *tmp = malloc(nb);
            if (!tmp) return -1;
            ggml_backend_tensor_get(src, tmp, 0, nb);
            ggml_backend_tensor_set(dst, tmp, 0, nb);
            free(tmp);
            /* Repoint the name at the copy, preserving the ONNX rank the
             * original carried -- read BEFORE the put, since tmap_get_ndims
             * would otherwise find the entry just added. */
            int nd = tmap_get_ndims(c, nm);
            tmap_put_nd(c, nm, dst, nd);
        }
    }

    /* Clear the queue unconditionally.
     *
     * Resetting only when something was copied leaves stale entries behind
     * whenever a segment queues nothing -- and those entries point at tensors
     * belonging to a context that has since been torn down.  The next segment
     * that does copy something then walks the whole queue, reading and writing
     * through those dangling pointers: memory belonging to something else,
     * corrupted quietly, and blamed on whichever malloc happens to notice
     * first. */
    c->n_boundary_pending = 0;

    if (onnx_trace_nodes())
        fprintf(stderr, "[segment] %d: copied %d boundary tensors (%.2f MB)\n",
                seg, n_copied, bytes / (1024.0 * 1024.0));
    return 0;
}

struct ggml_tensor *onnx_ggml_output(onnx_ggml_ctx_t *ctx, int index) {
    if (!ctx->onnx || index < 0 || index >= ctx->onnx->n_outputs)
        return NULL;
    return tmap_get(ctx, ctx->onnx->outputs[index].name);
}

void onnx_ggml_free(onnx_ggml_ctx_t *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_orphan_input_bufs; i++) {
        if (ctx->orphan_input_bufs[i]) ggml_backend_buffer_free(ctx->orphan_input_bufs[i]);
    }
    if (ctx->sched)       ggml_backend_sched_free(ctx->sched);
    /* Segment 0 shares ctx->sched, so the loop starts at 1 and cannot free it
     * a second time. */
    for (int i = 1; i < ONNX_MAX_SEGMENTS; i++) {
        if (ctx->seg_scheds[i]) ggml_backend_sched_free(ctx->seg_scheds[i]);
    }
    if (ctx->pinned_buf)  ggml_backend_buffer_free(ctx->pinned_buf);
    if (ctx->weight_buf)  ggml_backend_buffer_free(ctx->weight_buf);
    /* Buffers from the per-segment alloc_ctx_tensors calls (see the field's
     * comment): each one covers the weight tensors added by one segment, and
     * freeing only weight_buf would leak all of them. */
    for (int i = 0; i < ctx->n_extra_weight_bufs; i++) {
        if (ctx->extra_weight_bufs[i])
            ggml_backend_buffer_free(ctx->extra_weight_bufs[i]);
    }
    ctx->n_extra_weight_bufs = 0;
    if (ctx->boundary_buf) ggml_backend_buffer_free(ctx->boundary_buf);
    if (ctx->backend_gpu) ggml_backend_free(ctx->backend_gpu);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->ctx_boundary) ggml_free(ctx->ctx_boundary);
    if (ctx->ctx_weight)  ggml_free(ctx->ctx_weight);
    if (ctx->ctx)         ggml_free(ctx->ctx);
    free(ctx->tensor_map_keys);
    free(ctx->tensor_map_vals);
    free(ctx->tensor_map_ndims);
    free(ctx->tensor_map_onnx_ne);
    free(ctx->cval_keys);
    free(ctx->cval_data);
    free(ctx->cval_lens);
    for (int i = 0; i < ctx->n_pos_embed_blocks; i++)
        free(ctx->pos_embed_blocks[i].params.w_cpu);
    free(ctx->pos_embed_params);
    free(ctx->roi_align_params);
    free(ctx->nms_params);
    /* Note: onnx model is NOT freed here — caller manages it */
    free(ctx);
}
