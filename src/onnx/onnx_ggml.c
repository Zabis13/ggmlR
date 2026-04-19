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

static int load_weights(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_initializers; i++) {
        onnx_initializer_t *init = &c->onnx->initializers[i];
        struct ggml_tensor *t = tmap_get(c, init->name);
        if (!t) continue;
        /* Skip tensors not in the graph (e.g. shape constants for Reshape) —
         * they have no buffer allocated by the scheduler */
        if (!t->buffer) continue;

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
    int ok = 1;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        if (b->ne[d] != 1 && b->ne[d] != a->ne[d]) { ok = 0; break; }
    }
    if (ok) return;

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
            *pb = onnx_reshape_nd(ctx, b, new_ne, nd);
            return;
        }
    }

    /* Neither a broadcasts into b nor b into a.
     * Both need expansion to a common shape: max(a.ne[d], b.ne[d]) per dim.
     * Use ggml_repeat on each tensor to expand to the target shape. */
    {
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
    struct ggml_tensor *out = NULL;
    int out_nd = -1;  /* output ONNX ndims; -1 = inherit from input (generic path) */

    /* Debug: uncomment to trace node processing
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
    */

    struct ggml_tensor *a = get_input(c, n, 0);
    struct ggml_tensor *b = get_input(c, n, 1);

    const char *op = n->op_type;

    /* Dispatch to op group handlers.
     * Each returns: 1 = handled, 0 = not this group, -1 = error.
     * Handlers that register outputs themselves return 1 directly (no goto needed).
     * Handlers that set *out and *out_nd fall through to generic registration. */
    {
        int r;
        r = map_node_basic  (c, n, a, b, &out, &out_nd); if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_tensor (c, n, a, b, &out, &out_nd); if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_nn     (c, n, a, b, &out, &out_nd); if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_quant  (c, n, a, b, &out, &out_nd); if (r < 0) return -1; if (r > 0) goto reg_output;
        r = map_node_special(c, n, a, b, &out, &out_nd); if (r < 0) return -1; if (r > 0) goto reg_output;
        fprintf(stderr, "onnx_ggml: unsupported op '%s'\n", op);
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
            if (n->n_inputs > 0 && n->inputs[0][0] != '\0') {
                int in_nd = tmap_get_ndims(c, n->inputs[0]);
                if (in_nd > out_ndims) out_ndims = in_nd;
            }
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

        size_t src_bytes = ggml_nbytes(src);
        float *src_buf = (float *)malloc(src_bytes);
        if (!src_buf) continue;
        ggml_backend_tensor_get(src, src_buf, 0, src_bytes);

        size_t dst_n = ggml_nelements(dst);
        float *dst_buf = (float *)malloc(dst_n * sizeof(float));
        if (!dst_buf) { free(src_buf); continue; }

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
        for (int64_t di = 0; di < (int64_t)dst_n; di++) {
            int64_t si = 0;
            int64_t rem = di;
            for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
                int64_t coord = rem / out_stride[d];
                rem -= coord * out_stride[d];
                si += (st[d] + coord * sp[d]) * src_stride[d];
            }
            dst_buf[di] = src_buf[si];
        }

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

                    /* B: from x tensor shape. x is [B, H*W, C] in ONNX.
                     * Try to get from tmap (input tensor already created) */
                    int B = 1;
                    struct ggml_tensor *xt = tmap_get(c, first_reshape_input);
                    if (xt) {
                        /* ggml: ne[0]=C, ne[1]=H*W, ne[2]=B */
                        B = (int)xt->ne[2];
                    }

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

                    fprintf(stderr, "[pos_embed] block %d detected: output='%s' H=%d W=%d B=%d C=%d\n",
                            bi, c->pos_embed_blocks[bi].output_name, H, W, B, C);
                    c->n_pos_embed_blocks++;
                } else {
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

static int sched_alloc_and_fill(onnx_ggml_ctx_t *c) {

    /* GPU-first: pre-assign all graph nodes and their leaf sources to GPU.
     * Weight tensors already have ->buffer set (from weight_buf), so sched
     * will skip them.  Only non-weight inputs and intermediates get assigned. */
    if (c->backend_gpu) {
        int n_nodes = ggml_graph_n_nodes(c->graph);
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor *node = ggml_graph_node(c->graph, i);
            if (ggml_backend_supports_op(c->backend_gpu, node)) {
                ggml_backend_sched_set_tensor_backend(c->sched, node, c->backend_gpu);
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j]) {
                        /* Tell sched about ALL sources — including weight tensors
                         * that already have ->buffer on GPU.  Without this, sched
                         * doesn't know their backend and may insert spurious copies
                         * or fall back to CPU. */
                        ggml_backend_sched_set_tensor_backend(
                            c->sched, node->src[j], c->backend_gpu);
                    }
                }
            }
        }
    }

    /* Allocate graph buffers via scheduler.
     * Tensors with ->buffer already set (weights in weight_buf) are skipped.
     * Only input placeholders and intermediate compute tensors get allocated. */
    if (!ggml_backend_sched_alloc_graph(c->sched, c->graph)) return -1;


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
                    ggml_backend_tensor_alloc(buf, t, (char *)ggml_backend_buffer_get_base(buf));
                }
            }
        }
    }

    /* Fill strided Slice outputs — their src may live in sched buffer,
     * so this must happen after sched alloc. */
    fill_strided_slices(c);

    return 0;
}

/* ── Build full graph ───────────────────────────────────────────── */

onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device, int n_threads,
                                  enum ggml_type model_dtype) {
    onnx_ggml_ctx_t *c = calloc(1, sizeof(onnx_ggml_ctx_t));
    if (!c) return NULL;
    c->onnx = onnx;
    c->model_dtype = (model_dtype == GGML_TYPE_F16) ? GGML_TYPE_F16 : GGML_TYPE_F32;

    /* Estimate memory: rough heuristic based on file size */
    size_t mem_size = onnx->mmap_size * 2 + 256 * 1024 * 1024;

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

    /* Create tensors for initializers (in ctx_weight) and inputs (in ctx) */
    if (create_initializer_tensors(c) != 0) goto fail;
    if (create_input_tensors(c) != 0) goto fail;

    /* Pre-pass: detect pos_embed subgraphs for RelPosBias2D fusion */
    detect_pos_embed_blocks(c);

    /* Map ONNX nodes to ggml ops */
    for (int i = 0; i < onnx->n_nodes; i++) {
        /* Skip nodes inside pos_embed blocks (handled by fused custom op) */
        if (c->n_pos_embed_blocks > 0 && is_pos_embed_node(c, i)) {
            int bi = pos_embed_block_end(c, i);
            if (bi >= 0) {
                /* Last node of block — emit fused RelPosBias2D op */
                const rel_pos_bias_params_t *p = &c->pos_embed_blocks[bi].params;
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

                /* Build wcat = concat(W_h, W_w) along axis 0 (rel dim).
                 * ggml_rel_pos_bias handles both CPU and Vulkan dispatch. */
                struct ggml_tensor *wcat = ggml_concat(c->ctx, wh_t, ww_t, 0);

                /* Emit fused Vulkan-capable op */
                struct ggml_tensor *out = ggml_rel_pos_bias(c->ctx,
                    x_t, wcat, p->H, p->W);

                ggml_set_name(out, c->pos_embed_blocks[bi].output_name);
                tmap_put_nd(c, c->pos_embed_blocks[bi].output_name, out, 3);

            }
            continue;
        }
        if (map_node(c, &onnx->nodes[i]) != 0) {
            /* Non-fatal: skip unsupported/invalid ops silently */
            (void)0;
        }
    }

    /* Build forward graph from output tensors */
    {
        c->graph = ggml_new_graph(c->ctx);
        for (int i = 0; i < onnx->n_outputs; i++) {
            struct ggml_tensor *t = tmap_get(c, onnx->outputs[i].name);
            if (t) {
                ggml_set_output(t);
                ggml_build_forward_expand(c->graph, t);
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
        /* weight_buf may be NULL if there are no initializers — that's OK */
        if (c->weight_buf) {
            /* Load initializer weights */
            if (load_weights(c) != 0) goto fail;

            /* Load Constant op tensor data */
            for (int i = 0; i < c->onnx->n_nodes; i++) {
                onnx_node_t *nd = &c->onnx->nodes[i];
                if (strcmp(nd->op_type, "Constant") != 0) continue;
                const onnx_attr_t *va = onnx_node_find_attr(nd, "value");
                if (!va || !va->tensor) continue;
                onnx_initializer_t *ti = va->tensor;
                struct ggml_tensor *t = tmap_get(c, nd->outputs[0]);
                if (!t || !t->buffer) continue;

                const void *data = NULL;
                size_t data_size = 0;
                if (ti->raw_data && ti->raw_size > 0) {
                    data = ti->raw_data;
                    data_size = ti->raw_size;
                } else if (ti->decoded_data && ti->decoded_size > 0) {
                    data = ti->decoded_data;
                    data_size = ti->decoded_size;
                }
                if (data && data_size > 0) {
                    /* F32 source → F16 tensor (FP16 mode for large constants) */
                    if (ti->data_type == ONNX_DTYPE_FLOAT &&
                        t->type == GGML_TYPE_F16) {
                        int64_t n_elem = ggml_nelements(t);
                        size_t src_elems = data_size / sizeof(float);
                        if ((int64_t)src_elems > n_elem) src_elems = (size_t)n_elem;
                        ggml_fp16_t *buf = (ggml_fp16_t *)malloc(n_elem * sizeof(ggml_fp16_t));
                        if (buf) {
                            float *aligned_f32 = (float *)malloc(src_elems * sizeof(float));
                            if (aligned_f32) {
                                memcpy(aligned_f32, data, src_elems * sizeof(float));
                                ggml_fp32_to_fp16_row(aligned_f32, buf, (int64_t)src_elems);
                                free(aligned_f32);
                            }
                            for (size_t j = src_elems; j < (size_t)n_elem; j++)
                                buf[j] = ggml_fp32_to_fp16(0.0f);
                            ggml_backend_tensor_set(t, buf, 0, n_elem * sizeof(ggml_fp16_t));
                            free(buf);
                        }
                    } else {
                        size_t tsize = ggml_nbytes(t);
                        size_t copy_size = data_size < tsize ? data_size : tsize;
                        ggml_backend_tensor_set(t, data, 0, copy_size);
                    }
                }
            }

            /* Fill Shape op output tensors with ONNX dims */
            for (int i = 0; i < c->n_shape_tensors; i++) {
                struct ggml_tensor *t = c->shape_tensor_ptrs[i];
                if (!t || !t->buffer) continue;
                int nd = (int)c->shape_tensors_ne[i][0];
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
                if (!dst || !dst->buffer) continue;
                if (!src) continue;

                int64_t nnz = (int64_t)ggml_nelements(src);
                /* Compute src shape in ONNX order (reversed from ggml ne) */
                int src_ggml_dims = (int)ggml_n_dims(src);
                int64_t onnx_shape[ONNX_MAX_DIMS];
                for (int d = 0; d < nd; d++) {
                    int gd = nd - 1 - d;  /* ggml dim for ONNX dim d */
                    onnx_shape[d] = (gd < src_ggml_dims) ? src->ne[gd] : 1;
                }

                /* Build index matrix: for flat index k, unravel to nd coordinates */
                size_t fill_bytes = (size_t)nnz * nd * sizeof(float);
                size_t dst_bytes = ggml_nbytes(dst);
                if (fill_bytes > dst_bytes) continue;
                float *buf = (float *)malloc(fill_bytes);
                if (buf) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int64_t rem = k;
                        for (int d = nd - 1; d >= 0; d--) {
                            buf[d * nnz + k] = (float)(rem % onnx_shape[d]);
                            rem /= onnx_shape[d];
                        }
                    }
                    ggml_backend_tensor_set(dst, buf, 0, fill_bytes);
                    free(buf);
                }
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

int onnx_ggml_run(onnx_ggml_ctx_t *ctx,
                  const char **input_names, const float **input_data,
                  int n_inputs) {
    if (!ctx->graph || !ctx->sched) return -1;

    /* First run: allocate compute buffers and fill small deferred tensors.
     * Weights live in a separate weight_buf that sched never touches,
     * so subsequent runs need NO reload — just set inputs and compute. */
    if (!ctx->is_allocated) {
        if (sched_alloc_and_fill(ctx) != 0) return -1;
        ctx->is_allocated = 1;
    }

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
        if (t->type == GGML_TYPE_I32) {
            /* Input is integer (e.g. token IDs for Gather/embedding).
             * Caller passes float — convert to int32. */
            int64_t nel = ggml_nelements(t);
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


    enum ggml_status status = ggml_backend_sched_graph_compute(ctx->sched, ctx->graph);

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

struct ggml_tensor *onnx_ggml_output(onnx_ggml_ctx_t *ctx, int index) {
    if (!ctx->onnx || index < 0 || index >= ctx->onnx->n_outputs)
        return NULL;
    return tmap_get(ctx, ctx->onnx->outputs[index].name);
}

void onnx_ggml_free(onnx_ggml_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->sched)       ggml_backend_sched_free(ctx->sched);
    if (ctx->pinned_buf)  ggml_backend_buffer_free(ctx->pinned_buf);
    if (ctx->weight_buf)  ggml_backend_buffer_free(ctx->weight_buf);
    if (ctx->backend_gpu) ggml_backend_free(ctx->backend_gpu);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
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
