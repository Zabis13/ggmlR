/* onnx_ggml.c — Map ONNX ops to ggml ops and run inference
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ggml.h"
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

static void tmap_put_nd(onnx_ggml_ctx_t *c, const char *name,
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
static void tmap_put_shape(onnx_ggml_ctx_t *c, const char *name,
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
static int tmap_get_shape(onnx_ggml_ctx_t *c, const char *name,
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

static void tmap_put(onnx_ggml_ctx_t *c, const char *name, struct ggml_tensor *t) {
    tmap_put_nd(c, name, t, ggml_n_dims(t));
}

static int tmap_get_ndims(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0)
            return c->tensor_map_ndims[i];
    }
    return 4;
}

static struct ggml_tensor *tmap_get(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0)
            return c->tensor_map_vals[i];
    }
    return NULL;
}

/* Helper: squeeze trailing unit dims (5D→4D when ne[4]==1, etc.)
 * Keeps ggml tensors compact; real ONNX ndims tracked via tmap out_nd. */
static int onnx_squeeze_ndims(const int64_t *ne, int ndims) {
    while (ndims > 1 && ne[ndims - 1] == 1) ndims--;
    return ndims;
}

/* Helper: reshape tensor to given ne[] with appropriate ndims.
 * Squeezes trailing 1s for ggml compatibility. */
static struct ggml_tensor *onnx_reshape_nd(struct ggml_context *ctx,
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
static struct ggml_tensor *onnx_new_tensor_nd(struct ggml_context *ctx,
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
static int64_t ne_product(const int64_t *ne, int ndims) {
    int64_t p = 1;
    for (int d = 0; d < ndims; d++) p *= ne[d];
    return p;
}

/* ── Compile-time value map (for shape propagation) ─────────────── */


static void cval_put(onnx_ggml_ctx_t *c, const char *name,
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

static int cval_get(onnx_ggml_ctx_t *c, const char *name,
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

static const onnx_initializer_t *find_constant_tensor(const onnx_model_t *m,
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
static struct ggml_tensor *make_scalar(onnx_ggml_ctx_t *c, float val) {
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

static enum ggml_type onnx_dtype_to_ggml(int32_t dt) {
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

static size_t onnx_dtype_size(int32_t dt) {
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
         * faster Vulkan compute. Only for ndims >= 3 (Conv/pooling weights).
         * 2D weights (MatMul/embedding) stay F32 because Vulkan's MMQ
         * (integer dot product) path for F32→Q8_1 is faster than F16 matmul.
         * Small tensors (bias, BN params) and INT types are never converted. */
        int64_t n_elem = 1;
        for (int d = 0; d < ndims; d++) n_elem *= ne[d];
        if (c->model_dtype == GGML_TYPE_F16 &&
            type == GGML_TYPE_F32 &&
            init->n_dims >= 3 &&
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
                    const int32_t *i32 = (const int32_t *)src;
                    for (int64_t j = 0; j < n_elem; j++)
                        vals[j] = (int64_t)i32[j];
                    cval_put(c, init->name, vals, (int)n_elem);
                } else if (init->data_type == ONNX_DTYPE_FLOAT) {
                    const float *f32 = (const float *)src;
                    for (int64_t j = 0; j < n_elem; j++)
                        vals[j] = (int64_t)f32[j];
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
                const int64_t *src = (const int64_t *)data;
                for (size_t j = 0; j < src_elems; j++)
                    buf[j] = (int32_t)src[j];
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
                const double *src = (const double *)data;
                for (size_t j = 0; j < src_elems; j++)
                    buf[j] = (float)src[j];
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
                ggml_fp32_to_fp16_row((const float *)data, buf, (int64_t)src_elems);
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

static struct ggml_tensor *get_input(onnx_ggml_ctx_t *c, const onnx_node_t *n, int idx) {
    if (idx >= n->n_inputs) return NULL;
    if (n->inputs[idx][0] == '\0') return NULL; /* optional empty input */
    return tmap_get(c, n->inputs[idx]);
}

/* Current node being processed — for diagnostic messages */
static const onnx_node_t *g_current_node = NULL;

/* ── Broadcast helper for binary ops ────────────────────────────── */
/* Reshape b so that it is broadcastable into a (ggml requires b->ne[d] == 1 or a->ne[d]).
 * Returns b (possibly reshaped). If a and b need swapping, sets *swapped=1. */
static void onnx_broadcast_prepare(struct ggml_context *ctx,
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
                fprintf(stderr, "[onnx_broadcast] WARN: incompatible dim %d: a=%lld, b=%lld"
                        " (a='%s' [%lld,%lld,%lld,%lld], b='%s' [%lld,%lld,%lld,%lld])"
                        " node=%s out=%s\n",
                        d, (long long)a->ne[d], (long long)b->ne[d],
                        a->name, (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
                        b->name, (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3],
                        g_current_node ? g_current_node->op_type : "?",
                        g_current_node && g_current_node->n_outputs > 0 ? g_current_node->outputs[0] : "?");
                return;
            }
            if (b->ne[d] != 1 && b->ne[d] != target[d]) {
                fprintf(stderr, "[onnx_broadcast] WARN: incompatible dim %d: a=%lld, b=%lld"
                        " (a='%s' [%lld,%lld,%lld,%lld], b='%s' [%lld,%lld,%lld,%lld])"
                        " node=%s out=%s\n",
                        d, (long long)a->ne[d], (long long)b->ne[d],
                        a->name, (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
                        b->name, (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3],
                        g_current_node ? g_current_node->op_type : "?",
                        g_current_node && g_current_node->n_outputs > 0 ? g_current_node->outputs[0] : "?");
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

/* ── Elementwise binary ─────────────────────────────────────── */
    if (strcmp(op, "Add") == 0) {
        if (!a || !b) return -1;
        /* Ensure matching types for binary op */
        if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_F32)
            a = ggml_cast(c->ctx, a, GGML_TYPE_F32);
        else if (b->type == GGML_TYPE_I32 && a->type == GGML_TYPE_F32)
            b = ggml_cast(c->ctx, b, GGML_TYPE_F32);
        struct ggml_tensor *ta = a, *tb = b;
        onnx_broadcast_prepare(c->ctx, &ta, &tb);
        out = ggml_add(c->ctx, ta, tb);

        /* cval propagation for Add */
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 > 0) {
                int64_t result[ONNX_MAX_DIMS];
                int nr = na2 > nb2 ? na2 : nb2;
                for (int j = 0; j < nr; j++) {
                    int64_t va = cv_a[j < na2 ? j : 0];
                    int64_t vb = cv_b[j < nb2 ? j : 0];
                    result[j] = va + vb;
                }
                cval_put(c, n->outputs[0], result, nr);
            }
        }
    }
    else if (strcmp(op, "Sub") == 0) {
        if (!a || !b) return -1;
        if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_F32)
            a = ggml_cast(c->ctx, a, GGML_TYPE_F32);
        else if (b->type == GGML_TYPE_I32 && a->type == GGML_TYPE_F32)
            b = ggml_cast(c->ctx, b, GGML_TYPE_F32);
        /* Sub is not commutative — only reshape b for broadcast */
        struct ggml_tensor *ta = a, *tb = b;
        if (ggml_nelements(a) >= ggml_nelements(b)) {
            onnx_broadcast_prepare(c->ctx, &ta, &tb);
            out = ggml_sub(c->ctx, ta, tb);
        } else {
            /* a smaller: sub(a,b) = -(b-a) */
            onnx_broadcast_prepare(c->ctx, &tb, &ta);
            out = ggml_neg(c->ctx, ggml_sub(c->ctx, tb, ta));
        }

        /* cval propagation for Sub */
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 > 0) {
                int64_t result[ONNX_MAX_DIMS];
                int nr = na2 > nb2 ? na2 : nb2;
                for (int j = 0; j < nr; j++) {
                    int64_t va = cv_a[j < na2 ? j : 0];
                    int64_t vb = cv_b[j < nb2 ? j : 0];
                    result[j] = va - vb;
                }
                cval_put(c, n->outputs[0], result, nr);
            }
        }
    }
    else if (strcmp(op, "Mul") == 0) {
        if (!a || !b) return -1;
        if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_F32)
            a = ggml_cast(c->ctx, a, GGML_TYPE_F32);
        else if (b->type == GGML_TYPE_I32 && a->type == GGML_TYPE_F32)
            b = ggml_cast(c->ctx, b, GGML_TYPE_F32);
        struct ggml_tensor *ta = a, *tb = b;
        onnx_broadcast_prepare(c->ctx, &ta, &tb);
        out = ggml_mul(c->ctx, ta, tb);

        /* cval propagation for Mul (element-wise or scalar broadcast) */
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 > 0) {
                int64_t result[ONNX_MAX_DIMS];
                int nr = na2 > nb2 ? na2 : nb2;
                for (int j = 0; j < nr; j++) {
                    int64_t va = cv_a[j < na2 ? j : 0];
                    int64_t vb = cv_b[j < nb2 ? j : 0];
                    result[j] = va * vb;
                }
                cval_put(c, n->outputs[0], result, nr);
            }
        }
    }
    else if (strcmp(op, "Div") == 0) {
        if (!a || !b) return -1;
        if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_F32)
            a = ggml_cast(c->ctx, a, GGML_TYPE_F32);
        else if (b->type == GGML_TYPE_I32 && a->type == GGML_TYPE_F32)
            b = ggml_cast(c->ctx, b, GGML_TYPE_F32);
        struct ggml_tensor *ta = a, *tb = b;
        if (ggml_nelements(a) >= ggml_nelements(b)) {
            onnx_broadcast_prepare(c->ctx, &ta, &tb);
            out = ggml_div(c->ctx, ta, tb);
        } else {
            onnx_broadcast_prepare(c->ctx, &ta, &tb);
            out = ggml_div(c->ctx, ta, tb);
        }

        /* cval propagation for Div (integer division) */
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 > 0) {
                int64_t result[ONNX_MAX_DIMS];
                int nr = na2 > nb2 ? na2 : nb2;
                for (int j = 0; j < nr; j++) {
                    int64_t va = cv_a[j < na2 ? j : 0];
                    int64_t vb = cv_b[j < nb2 ? j : 0];
                    result[j] = (vb != 0) ? va / vb : 0;
                }
                cval_put(c, n->outputs[0], result, nr);
            }
        }
    }

    /* ── MatMul / Gemm ──────────────────────────────────────────── */
    else if (strcmp(op, "MatMul") == 0) {
        if (!a || !b) return -1;
        /* Debug: uncomment to trace MatMul shapes
        fprintf(stderr, "[MatMul] '%s': A.ne=[%lld,%lld,%lld,%lld,%lld] nd_a=%d type=%d cont=%d  "
                "B.ne=[%lld,%lld,%lld,%lld,%lld] nd_b=%d type=%d cont=%d\n",
                n->outputs[0],
                (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3], (long long)a->ne[4],
                tmap_get_ndims(c, n->inputs[0]), (int)a->type, ggml_is_contiguous(a),
                (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3], (long long)b->ne[4],
                tmap_get_ndims(c, n->inputs[1]), (int)b->type, ggml_is_contiguous(b));
        */
        /* ONNX MatMul: A[...,M,K] @ B[...,K,N] → [...,M,N]
         *
         * Source of truth: reconstruct ONNX shapes from ggml ne[] + tmap_get_ndims.
         * ONNX dim order is reversed relative to ggml:
         *   ONNX shape[i] = ne[ndims-1-i]
         *
         * For ONNX A[...,M,K]: last ONNX dim = K → A.ne[0]=K, second-to-last = M → A.ne[1]=M
         * For ONNX B[...,K,N]: last ONNX dim = N → B.ne[0]=N, second-to-last = K → B.ne[1]=K
         *
         * ggml_mul_mat(w, x): contracts w.ne[0]==x.ne[0]
         *   result = [w.ne[1], x.ne[1], x.ne[2], x.ne[3]]
         *   w.ne[2..3] must divide x.ne[2..3] (broadcast)
         *
         * Goal: make w with ne[0]=K, ne[1]=N, batch... (transposed B)
         *        and x with ne[0]=K, ne[1]=M, batch... (A as-is)
         *   result = [N, M, batch...] ✓
         */

        /* Recover ONNX ndims */
        int nd_a = tmap_get_ndims(c, n->inputs[0]);
        int nd_b = tmap_get_ndims(c, n->inputs[1]);
        if (nd_a < 2) nd_a = 2;
        if (nd_b < 2) nd_b = 2;

        /* ONNX shapes from ne[] (reversed).
         * ONNX A: [..., M, K]  → A.ne[0]=K, A.ne[1]=M
         * ONNX B: [..., K, N]  → B.ne[0]=N, B.ne[1]=K */
        int64_t K_a = a->ne[0];  /* K from A's last ONNX dim */
        int64_t K_b = b->ne[1];  /* K from B's second-to-last ONNX dim */

        /* Fast path: A.ne[0]==K and B.ne[1]==K (normal ONNX→ggml mapping) */
        if (K_a == K_b) {
            /* B: ne=[N,K,batch...]. Transpose to ne=[K,N,batch...] */
            struct ggml_tensor *bt = ggml_cont(c->ctx,
                ggml_permute(c->ctx, b, 1, 0, 2, 3));
            /* bt.ne=[K,N,batch_b...], a.ne=[K,M,batch_a...]
             * ggml_mul_mat(bt, a) needs bt.ne[2..3] | a.ne[2..3] */
            if (bt->ne[2] <= a->ne[2] && bt->ne[3] <= a->ne[3]) {
                out = ggml_mul_mat(c->ctx, bt, a);
            } else if (a->ne[2] <= bt->ne[2] && a->ne[3] <= bt->ne[3]) {
                out = ggml_mul_mat(c->ctx, a, bt);
                out = ggml_cont(c->ctx, ggml_permute(c->ctx, out, 1, 0, 2, 3));
            } else {
                out = ggml_mul_mat(c->ctx, bt, a);
            }
        }
        /* Fallback: ne[] layout doesn't match expected ONNX mapping.
         * This happens when upstream Transpose/Reshape reordered dims.
         * Reshape both tensors to canonical [K,M,batch] / [N,K,batch] layout.
         *
         * Reconstruct full ONNX shape for A and B, extract M,K,N and batch,
         * then reshape into clean ggml layout. */
        else {
            /* Try: maybe B has K at ne[0] instead of ne[1]
             * (upstream Transpose already moved it) */
            int64_t K_b0 = b->ne[0];
            if (K_b0 == K_a) {
                /* B already has ne[0]=K — use as-is for mul_mat first arg */
                struct ggml_tensor *tb = b;
                if (tb->ne[2] <= a->ne[2] && tb->ne[3] <= a->ne[3]) {
                    out = ggml_mul_mat(c->ctx, tb, a);
                } else if (a->ne[2] <= tb->ne[2] && a->ne[3] <= tb->ne[3]) {
                    out = ggml_mul_mat(c->ctx, a, tb);
                    out = ggml_cont(c->ctx, ggml_permute(c->ctx, out, 1, 0, 2, 3));
                } else {
                    out = ggml_mul_mat(c->ctx, tb, a);
                }
            }
            /* Last resort: reshape B to put K in the right place.
             * Reconstruct ONNX B shape, identify K (must equal K_a),
             * figure out which ggml dim holds K, and permute accordingly. */
            else {
                /* Check all 4 dims of B for K match */
                int k_dim = -1;
                for (int d = 0; d < 4; d++) {
                    if (b->ne[d] == K_a) { k_dim = d; break; }
                }
                if (k_dim >= 0 && k_dim != 0) {
                    /* Permute B to move K to ne[0] */
                    int p[4] = {0, 1, 2, 3};
                    p[0] = k_dim; p[k_dim] = 0;
                    struct ggml_tensor *tb = ggml_cont(c->ctx,
                        ggml_permute(c->ctx, b, p[0], p[1], p[2], p[3]));
                    if (tb->ne[2] <= a->ne[2] && tb->ne[3] <= a->ne[3]) {
                        out = ggml_mul_mat(c->ctx, tb, a);
                    } else if (a->ne[2] <= tb->ne[2] && a->ne[3] <= tb->ne[3]) {
                        out = ggml_mul_mat(c->ctx, a, tb);
                        out = ggml_cont(c->ctx, ggml_permute(c->ctx, out, 1, 0, 2, 3));
                    } else {
                        out = ggml_mul_mat(c->ctx, tb, a);
                    }
                } else if (k_dim == 0) {
                    /* K already at ne[0] */
                    if (b->ne[2] <= a->ne[2] && b->ne[3] <= a->ne[3]) {
                        out = ggml_mul_mat(c->ctx, b, a);
                    } else {
                        out = ggml_mul_mat(c->ctx, a, b);
                        out = ggml_cont(c->ctx, ggml_permute(c->ctx, out, 1, 0, 2, 3));
                    }
                } else {
                    fprintf(stderr, "onnx_ggml: MatMul '%s': cannot find K=%lld in B.ne=[%lld,%lld,%lld,%lld]\n",
                            n->outputs[0], (long long)K_a,
                            (long long)b->ne[0], (long long)b->ne[1],
                            (long long)b->ne[2], (long long)b->ne[3]);
                    return -1;
                }
            }
        }
        out_nd = nd_a > nd_b ? nd_a : nd_b;
    }
    else if (strcmp(op, "Gemm") == 0) {
        if (!a || !b) return -1;
        int64_t transA = onnx_attr_int(n, "transA", 0);
        int64_t transB = onnx_attr_int(n, "transB", 0);
        float alpha = onnx_attr_float(n, "alpha", 1.0f);
        float beta  = onnx_attr_float(n, "beta", 1.0f);

        /* With reversed dims: A[M,K] → ne=[K,M], B[K,N] → ne=[N,K].
         * Need ta with ne[0]=K (contraction), tb with ne[0]=K.
         * Default: A already has K at ne[0]. B needs transpose.
         * transA flips A dims, transB flips B dims. */
        struct ggml_tensor *ta = transA ? ggml_cont(c->ctx, ggml_transpose(c->ctx, a)) : a;
        struct ggml_tensor *tb = transB ? b : ggml_cont(c->ctx, ggml_transpose(c->ctx, b));

        out = ggml_mul_mat(c->ctx, tb, ta);

        if (alpha != 1.0f)
            out = ggml_scale(c->ctx, out, alpha);

        struct ggml_tensor *bias = get_input(c, n, 2);
        if (bias) {
            if (beta != 1.0f)
                bias = ggml_scale(c->ctx, bias, beta);
            out = ggml_add(c->ctx, out, bias);
        }
    }

    /* ── Activations ────────────────────────────────────────────── */
    else if (strcmp(op, "Relu") == 0) {
        if (!a) return -1;
        out = ggml_relu(c->ctx, a);
    }
    else if (strcmp(op, "Sigmoid") == 0) {
        if (!a) return -1;
        out = ggml_sigmoid(c->ctx, a);
    }
    else if (strcmp(op, "Tanh") == 0) {
        if (!a) return -1;
        out = ggml_tanh(c->ctx, a);
    }
    else if (strcmp(op, "Gelu") == 0) {
        if (!a) return -1;
        out = ggml_gelu(c->ctx, a);
    }
    else if (strcmp(op, "Softmax") == 0) {
        if (!a) return -1;
        /* Default axis=1 for opset < 13, axis=-1 for opset >= 13.
         * Most models specify it explicitly; use 1 as safe default. */
        int64_t axis = onnx_attr_int(n, "axis", 1);
        /* Use actual ONNX ndims for correct axis mapping */
        int nd = tmap_get_ndims(c, n->inputs[0]);
        if (nd <= 0) nd = (int)ggml_n_dims(a);
        if (nd < 1) nd = 1;
        if (axis < 0) axis += nd;
        /* ONNX axis → ggml dim (reversed) */
        int ggml_d = nd - 1 - (int)axis;
        if (ggml_d < 0) ggml_d = 0;

        if (ggml_d == 0) {
            /* Softmax over ne[0] — ggml_soft_max does this natively */
            out = ggml_soft_max(c->ctx, a);
        } else {
            /* ggml_soft_max normalizes over ne[0]. We need to normalize over ggml_d.
             * Strategy: flatten dims [0..ggml_d] into ne[0], keeping the rest.
             * Then softmax, then reshape back.
             * E.g. ONNX [N,C] with axis=1: nd=2, ggml_d=0 → fast path above.
             * E.g. ONNX [N,H,W,C] with axis=3: nd=4, ggml_d=0 → fast path.
             * E.g. ONNX [N,C,H,W] with axis=1: nd=4, ggml_d=2 →
             *      softmax_dim = ne[0]*ne[1]*ne[2] = W*H*C, batch = ne[3] = N */
            int64_t softmax_dim = 1;
            for (int d = 0; d <= ggml_d; d++)
                softmax_dim *= a->ne[d];
            int64_t batch_dim = ggml_nelements(a) / softmax_dim;

            struct ggml_tensor *flat = ggml_reshape_2d(c->ctx, a, softmax_dim,
                                                        batch_dim > 0 ? batch_dim : 1);
            struct ggml_tensor *sm = ggml_soft_max(c->ctx, flat);
            /* Reshape back to original shape */
            out = onnx_reshape_nd(c->ctx, sm, a->ne, ggml_n_dims(a));
        }
    }
    else if (strcmp(op, "LeakyRelu") == 0) {
        if (!a) return -1;
        float alpha = onnx_attr_float(n, "alpha", 0.01f);
        out = ggml_leaky_relu(c->ctx, a, alpha, false);
    }
    else if (strcmp(op, "Elu") == 0) {
        if (!a) return -1;
        out = ggml_elu(c->ctx, a);
    }
    else if (strcmp(op, "Silu") == 0 || strcmp(op, "SiLU") == 0) {
        if (!a) return -1;
        out = ggml_silu(c->ctx, a);
    }

    /* ── Math ───────────────────────────────────────────────────── */
    else if (strcmp(op, "Sqrt") == 0) {
        if (!a) return -1;
        out = ggml_sqrt(c->ctx, a);
    }
    else if (strcmp(op, "Exp") == 0) {
        if (!a) return -1;
        out = ggml_exp(c->ctx, a);
    }
    else if (strcmp(op, "Log") == 0) {
        if (!a) return -1;
        out = ggml_log(c->ctx, a);
    }
    else if (strcmp(op, "Abs") == 0) {
        if (!a) return -1;
        out = ggml_abs(c->ctx, a);
    }
    else if (strcmp(op, "Neg") == 0) {
        if (!a) return -1;
        out = ggml_neg(c->ctx, a);
    }
    else if (strcmp(op, "Floor") == 0) {
        if (!a) return -1;
        out = ggml_floor(c->ctx, a);
    }
    else if (strcmp(op, "Ceil") == 0) {
        if (!a) return -1;
        out = ggml_ceil(c->ctx, a);
    }
    else if (strcmp(op, "Clip") == 0) {
        if (!a) return -1;
        /* min/max can be inputs (opset 11+) or attributes */
        float min_val = -3.402823e+38f;
        float max_val =  3.402823e+38f;
        struct ggml_tensor *min_t = get_input(c, n, 1);
        struct ggml_tensor *max_t = get_input(c, n, 2);
        /* Read scalar value from tensor inputs (initializer or Constant) */
        if (min_t && n->n_inputs > 1 && n->inputs[1][0] != '\0') {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (!mi) mi = find_constant_tensor(c->onnx, n->inputs[1]);
            if (mi && mi->raw_data && mi->raw_size >= 4 &&
                mi->data_type == ONNX_DTYPE_FLOAT) {
                memcpy(&min_val, mi->raw_data, sizeof(float));
            }
        } else if (!min_t) {
            min_val = onnx_attr_float(n, "min", min_val);
        }
        if (max_t && n->n_inputs > 2 && n->inputs[2][0] != '\0') {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[2]);
            if (!mi) mi = find_constant_tensor(c->onnx, n->inputs[2]);
            if (mi && mi->raw_data && mi->raw_size >= 4 &&
                mi->data_type == ONNX_DTYPE_FLOAT) {
                memcpy(&max_val, mi->raw_data, sizeof(float));
            }
        } else if (!max_t) {
            max_val = onnx_attr_float(n, "max", max_val);
        }
        out = ggml_clamp(c->ctx, a, min_val, max_val);
    }

    /* ── Shape ops ──────────────────────────────────────────────── */
    else if (strcmp(op, "Reshape") == 0) {
        if (!a || !b) return -1;
        /* b is the shape tensor — from initializer, Constant, or computed shape */
        int64_t shape[ONNX_MAX_DIMS];
        int ndims = 0;

        /* Try 1: static initializer or Constant node */
        const onnx_initializer_t *shape_init = onnx_find_initializer(c->onnx, n->inputs[1]);
        if (!shape_init)
            shape_init = find_constant_tensor(c->onnx, n->inputs[1]);
        if (shape_init) {
            if (shape_init->raw_data && shape_init->data_type == ONNX_DTYPE_INT64) {
                ndims = (int)(shape_init->raw_size / sizeof(int64_t));
                if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
                memcpy(shape, shape_init->raw_data, ndims * sizeof(int64_t));
            } else if (shape_init->decoded_data) {
                ndims = (int)(shape_init->decoded_size / sizeof(int64_t));
                if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
                memcpy(shape, shape_init->decoded_data, ndims * sizeof(int64_t));
            }
        }
        /* Try 2: compile-time value map (for dynamic Shape→Slice→Concat chains) */
        if (ndims == 0) {
            ndims = cval_get(c, n->inputs[1], shape, ONNX_MAX_DIMS);
        }
        if (ndims == 0) {
            fprintf(stderr, "[onnx] Reshape: cannot resolve shape tensor '%s'\n", n->inputs[1]);
            return -1;
        }

        /* Resolve -1 and 0 dims (in ONNX order).
         * shape[d]==0 means keep original ONNX dim d.
         * We can get ONNX dim d from a->ne[ndims_a-1-d] (reversed). */
        int64_t total = ggml_nelements(a);
        int64_t product = 1;
        int neg_idx = -1;
        int ndims_a = ggml_n_dims(a);
        for (int d = 0; d < ndims; d++) {
            if (shape[d] == 0) {
                /* Map ONNX dim d → ggml ne index (reversed) */
                int ggml_d = ndims_a - 1 - d;
                shape[d] = (ggml_d >= 0 && ggml_d < ndims_a) ? a->ne[ggml_d] : 1;
            }
            if (shape[d] == -1) neg_idx = d;
            else product *= shape[d];
        }
        if (neg_idx >= 0 && product > 0)
            shape[neg_idx] = total / product;

        /* Collapse >5D ONNX shape into 5D by merging leading ONNX dims. */
        int orig_ndims = ndims; /* save for tmap_put_nd */
        int64_t orig_shape[ONNX_MAX_DIMS];
        memcpy(orig_shape, shape, orig_ndims * sizeof(int64_t));
        if (ndims > GGML_MAX_DIMS) {
            int64_t merged = 1;
            for (int d = 0; d < ndims - (GGML_MAX_DIMS - 1); d++)
                merged *= shape[d];
            int64_t tmp[GGML_MAX_DIMS];
            tmp[0] = merged;
            for (int d = 1; d < GGML_MAX_DIMS; d++)
                tmp[d] = shape[ndims - (GGML_MAX_DIMS - 1) + d - 1];
            memcpy(shape, tmp, sizeof(tmp));
            ndims = GGML_MAX_DIMS;
        }

        /* Reverse ONNX shape → ggml ne order for reshape */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < ndims; d++)
            ne[d] = shape[ndims - 1 - d];

        /* Verify element count match */
        {
            int64_t total_out = ne_product(ne, ndims);
            if (total_out != total) {
                fprintf(stderr, "onnx_ggml: Reshape element count mismatch: %s "
                        "input=%lld output=%lld\n",
                        n->outputs[0], (long long)total, (long long)total_out);
                return -1;
            }
        }
        out = onnx_reshape_nd(c->ctx, a, ne, ndims);
        /* Register with ONNX shape for correct axis mapping downstream.
         * If >5D was collapsed, use collapsed ndims (not orig) so downstream
         * ops don't see dimensions beyond what ggml can represent. */
        int reg_ndims = (orig_ndims > GGML_MAX_DIMS) ? ndims : orig_ndims;
        int64_t *reg_shape = (orig_ndims > GGML_MAX_DIMS) ? shape : orig_shape;
        if (out) {
            for (int i = 0; i < n->n_outputs; i++) {
                if (n->outputs[i][0] != '\0') {
                    ggml_set_name(out, n->outputs[i]);
                    tmap_put_shape(c, n->outputs[i], out, reg_shape, reg_ndims);
                }
            }
            /* Propagate cval through Reshape (flat order unchanged) */
            {
                int64_t cv[ONNX_MAX_DIMS];
                int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
                if (ncv > 0)
                    cval_put(c, n->outputs[0], cv, ncv);
            }
            return 0;
        }
    }
    else if (strcmp(op, "Transpose") == 0) {
        if (!a) return -1;
        int64_t perm[ONNX_MAX_DIMS];
        int n_perm = onnx_attr_ints(n, "perm", perm, ONNX_MAX_DIMS);
        int nd = tmap_get_ndims(c, n->inputs[0]);
        if (nd <= 0) nd = ggml_n_dims(a);
        (void)0;
        /* Check for identity perm first */
        int is_identity = 0;
        if (n_perm > 0) {
            is_identity = 1;
            for (int i = 0; i < n_perm; i++) {
                if (perm[i] != i) { is_identity = 0; break; }
            }
        }
        /* If perm has more dims than tensor (Reshape collapsed leading dims),
         * remap perm to match actual tensor ndims. */
        if (n_perm > nd && nd >= 2) {
            int n_merged = n_perm - nd; /* number of leading ONNX dims collapsed into dim 0 */
            /* Build collapsed perm: remap dim references and skip merged output dims */
            int64_t cperm[ONNX_MAX_DIMS];
            int cp = 0;
            for (int i = 0; i < n_perm; i++) {
                int64_t src = perm[i];
                /* Remap source: dims 0..n_merged → 0, dims n_merged+k → k+1... no:
                 * dims 0..n_merged all map to collapsed dim 0,
                 * dim n_merged+k maps to dim k (since leading merged into 0, rest shift) */
                int64_t mapped_src;
                if (src <= n_merged) mapped_src = 0;
                else mapped_src = src - n_merged;
                /* Skip output positions that refer to within-merged reshuffling
                 * (output dims 0..n_merged that all point to merged source) */
                if (i <= n_merged && mapped_src == 0 && cp > 0 && cperm[cp-1] == 0)
                    continue; /* duplicate merged dim in output — skip */
                if (cp < ONNX_MAX_DIMS) cperm[cp++] = mapped_src;
            }
            /* If we got exactly nd entries, use collapsed perm */
            if (cp == nd) {
                memcpy(perm, cperm, nd * sizeof(int64_t));
                n_perm = nd;
            }
        }
        if (is_identity) {
            out = a;
        } else if (nd >= 5 && n_perm >= 5) {
            /* 5D Transpose via squeeze-batch → 4D permute → restore 5D.
             * Find a unit ONNX *input* dim to squeeze out temporarily. */
            int batch_in = -1; /* ONNX input dim index with size==1 */
            for (int i = 0; i < nd && i < 5; i++) {
                int ggml_d = nd - 1 - i;
                if (ggml_d >= 0 && ggml_d < GGML_MAX_DIMS && a->ne[ggml_d] == 1) {
                    batch_in = i; break;
                }
            }
            if (batch_in < 0) {
                fprintf(stderr, "[onnx] Transpose 5D: no unit dim to squeeze\n");
                batch_in = 0;
            }
            /* Find where batch_in lands in output: perm[batch_out]==batch_in */
            int batch_out = 0;
            for (int i = 0; i < n_perm; i++) {
                if (perm[i] == batch_in) { batch_out = i; break; }
            }

            /* Build 4D perm: skip batch_out in output, renumber refs past batch_in */
            int64_t perm4[4];
            int p4 = 0;
            for (int i = 0; i < n_perm; i++) {
                if (i == batch_out) continue; /* skip the output dim that receives batch */
                int64_t v = perm[i];
                if (v > batch_in) v--;  /* renumber: input dims shift down */
                else if (v == batch_in) v = 0; /* shouldn't happen (filtered above) */
                perm4[p4++] = v;
            }

            /* Build 4D ONNX input shape (squeeze out batch_in dim) */
            int64_t onnx_in4[4];
            int s4 = 0;
            for (int i = 0; i < nd && i < 5; i++) {
                if (i == batch_in) continue;
                onnx_in4[s4++] = a->ne[nd - 1 - i];
            }

            /* Reshape to 4D ggml (reverse onnx_in4) */
            int64_t ne4[4];
            for (int d = 0; d < 4; d++) ne4[d] = onnx_in4[3 - d];
            struct ggml_tensor *a4 = ggml_reshape_4d(c->ctx, a, ne4[0], ne4[1], ne4[2], ne4[3]);

            /* Compute 4D ggml permute axes from 4D ONNX perm */
            int ax[4] = {0, 1, 2, 3};
            for (int i = 0; i < 4; i++) {
                int ggml_dst = 3 - i;
                int ggml_src = 3 - (int)perm4[i];
                ax[ggml_src] = ggml_dst;
            }

            struct ggml_tensor *permuted;
            if (ax[0] == 0 && ax[1] == 1 && ax[2] == 2 && ax[3] == 3)
                permuted = a4;
            else
                permuted = ggml_cont(c->ctx, ggml_permute(c->ctx, a4, ax[0], ax[1], ax[2], ax[3]));

            /* Restore 5D: build output ONNX shape, insert batch=1 at batch_out */
            int64_t onnx_out4[4];
            for (int i = 0; i < 4; i++) onnx_out4[i] = onnx_in4[perm4[i]];

            int64_t onnx_out5[5];
            s4 = 0;
            for (int i = 0; i < 5; i++) {
                if (i == batch_out) onnx_out5[i] = 1;
                else onnx_out5[i] = onnx_out4[s4++];
            }

            /* Reverse to ggml ne order */
            int64_t ne5[5];
            for (int d = 0; d < 5; d++) ne5[d] = onnx_out5[4 - d];
            /* Use onnx_reshape_nd to squeeze trailing 1s (ggml max 4D ops) */
            out = onnx_reshape_nd(c->ctx, permuted, ne5, 5);
            out_nd = 5;
        } else if (n_perm == 0 || nd == 2) {
            /* Default: reverse all dims → standard transpose */
            out = ggml_cont(c->ctx, ggml_transpose(c->ctx, a));
        } else {
            /* 4D (or less) Transpose via ggml_permute.
             * Convert ONNX perm to ggml permute axes.
             * ONNX perm[i]=j means output ONNX dim i ← input ONNX dim j.
             * ggml ax[k] means output ggml dim k ← input ggml dim ax[k].
             * ONNX dim i → ggml dim (reversed). */
            int onnx_nd = n_perm > nd ? n_perm : nd;
            if (n->n_inputs > 0) {
                int in_nd = tmap_get_ndims(c, n->inputs[0]);
                if (in_nd > onnx_nd) onnx_nd = in_nd;
            }

            int ax[4] = {0, 1, 2, 3};
            for (int i = 0; i < n_perm; i++) {
                int ggml_dst = onnx_nd - 1 - i;
                int ggml_src = onnx_nd - 1 - (int)perm[i];
                if (ggml_dst < 0) ggml_dst = 0;
                if (ggml_dst > 3) ggml_dst = 3;
                if (ggml_src < 0) ggml_src = 0;
                if (ggml_src > 3) ggml_src = 3;
                ax[ggml_src] = ggml_dst;
            }
            if (ax[0] == 0 && ax[1] == 1 && ax[2] == 2 && ax[3] == 3) {
                out = a;
            } else {
                out = ggml_cont(c->ctx, ggml_permute(c->ctx, a, ax[0], ax[1], ax[2], ax[3]));
            }
        }
        /* Propagate cval through Transpose (permute flat elements).
         * cval stores values in ggml flat order (col-major: dim0 fastest).
         * Transpose swaps dims, so we need to remap element indices. */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0 && nd <= 4) {
                /* Build effective perm — default (n_perm==0) reverses all dims */
                int64_t eff_perm[4];
                int eff_n = nd;
                if (n_perm > 0) {
                    for (int d = 0; d < nd; d++)
                        eff_perm[d] = (d < n_perm) ? perm[d] : d;
                } else {
                    for (int d = 0; d < nd; d++)
                        eff_perm[d] = nd - 1 - d;
                }

                /* Input/output shapes in ONNX order */
                int64_t shape_in[4] = {1,1,1,1};
                for (int d = 0; d < nd; d++) shape_in[d] = a->ne[nd-1-d];
                int64_t shape_out[4] = {1,1,1,1};
                for (int d = 0; d < eff_n; d++)
                    shape_out[d] = shape_in[eff_perm[d]];

                /* Output ggml ne (reversed from ONNX) */
                int64_t out_ne0[4] = {1,1,1,1};
                for (int d = 0; d < nd; d++)
                    out_ne0[d] = shape_out[nd-1-d];

                int64_t result[ONNX_MAX_DIMS];
                int ncv_out = 1;
                for (int d = 0; d < nd; d++) ncv_out *= (int)shape_out[d];
                if (ncv_out == ncv && ncv_out <= ONNX_MAX_DIMS) {
                    /* Iterate output in ggml flat order (dim0 fastest) */
                    for (int oi = 0; oi < ncv_out; oi++) {
                        /* Decompose oi into ggml output indices [g0,g1,g2,g3] */
                        int gi[4] = {0};
                        int tmp = oi;
                        for (int d = 0; d < nd; d++) {
                            gi[d] = tmp % (int)out_ne0[d];
                            tmp /= (int)out_ne0[d];
                        }
                        /* Convert ggml output → ONNX output indices */
                        int onnx_out[4] = {0};
                        for (int d = 0; d < nd; d++)
                            onnx_out[d] = gi[nd-1-d];
                        /* Map ONNX output → ONNX input via inverse perm */
                        int onnx_in[4] = {0};
                        for (int d = 0; d < eff_n; d++)
                            onnx_in[(int)eff_perm[d]] = onnx_out[d];
                        /* Convert ONNX input → ggml input indices */
                        int gi_in[4] = {0};
                        for (int d = 0; d < nd; d++)
                            gi_in[d] = onnx_in[nd-1-d];
                        /* Compute flat ggml input index */
                        int ii = 0;
                        for (int d = nd-1; d >= 0; d--)
                            ii = ii * (int)a->ne[d] + gi_in[d];
                        result[oi] = cv[ii];
                    }
                    cval_put(c, n->outputs[0], result, ncv_out);
                }
            }
        }
        out_nd = nd;  /* Transpose preserves ndims */
    }
    else if (strcmp(op, "Flatten") == 0) {
        if (!a) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 1);
        int nd_a = tmap_get_ndims(c, n->inputs[0]);
        if (nd_a <= 0) nd_a = ggml_n_dims(a);
        if (axis < 0) axis += nd_a;

        /* ONNX Flatten: output shape = [product(dims[:axis]), product(dims[axis:])] */
        if (axis == 0) {
            /* Special case: [1, total] */
            int64_t total = ggml_nelements(a);
            out = ggml_reshape_2d(c->ctx, a, total, 1);
        } else if (axis >= nd_a) {
            /* All dims go to first part: [total, 1] */
            int64_t total = ggml_nelements(a);
            out = ggml_reshape_2d(c->ctx, a, 1, total);
        } else {
            /* Compute product of ONNX dims [axis:] → ggml dims [0..nd_a-1-axis]
             * and ONNX dims [:axis] → ggml dims [nd_a-axis..nd_a-1] */
            int64_t inner = 1, outer = 1;
            for (int d = 0; d < nd_a; d++) {
                int onnx_d = nd_a - 1 - d;
                if (onnx_d >= axis) inner *= a->ne[d];
                else outer *= a->ne[d];
            }
            /* ggml [inner, outer] = ONNX [outer, inner] */
            out = ggml_reshape_2d(c->ctx, a, inner, outer);
        }
        /* Propagate cval through Flatten (flat order unchanged in ggml) */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0)
                cval_put(c, n->outputs[0], cv, ncv);
        }
        /* Register with ONNX ndims=2 */
        if (out) {
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, 2);
            return 0;
        }
    }
    else if (strcmp(op, "Unsqueeze") == 0) {
        if (!a) return -1;
        /* Get axes from attribute (opset < 13) or second input (opset >= 13) */
        int64_t axes[ONNX_MAX_DIMS];
        int n_axes = onnx_attr_ints(n, "axes", axes, ONNX_MAX_DIMS);
        if (n_axes == 0 && n->n_inputs > 1) {
            const onnx_initializer_t *axes_init = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (axes_init && axes_init->raw_data && axes_init->data_type == ONNX_DTYPE_INT64) {
                n_axes = (int)(axes_init->raw_size / sizeof(int64_t));
                if (n_axes > ONNX_MAX_DIMS) n_axes = ONNX_MAX_DIMS;
                memcpy(axes, axes_init->raw_data, n_axes * sizeof(int64_t));
            }
            /* Fallback: try cval (from Constant nodes) */
            if (n_axes == 0) {
                n_axes = cval_get(c, n->inputs[1], axes, ONNX_MAX_DIMS);
            }
        }

        /* Use stored ONNX ndims (ggml_n_dims drops trailing 1s) */
        int nd_in = tmap_get_ndims(c, n->inputs[0]);
        if (nd_in <= 0) {
            nd_in = GGML_MAX_DIMS;
            while (nd_in > 1 && a->ne[nd_in - 1] == 1) nd_in--;
        }
        /* Check if ONNX input had more dims (trailing 1s) by looking at
         * total expected output ndims vs axes. If max axis >= nd_in + n_axes,
         * we need more input dims. */
        int nd_out = nd_in + n_axes;
        if (nd_out > GGML_MAX_DIMS) nd_out = GGML_MAX_DIMS;

        /* Normalize negative axes (relative to output ndims) */
        for (int i = 0; i < n_axes; i++)
            if (axes[i] < 0) axes[i] += nd_out;

        /* Build input ONNX dims from ggml ne (reversed) */
        int64_t onnx_in[ONNX_MAX_DIMS];
        for (int i = 0; i < nd_in; i++)
            onnx_in[i] = a->ne[nd_in - 1 - i];

        /* Build output ONNX shape by inserting 1s at axes positions */
        int64_t onnx_out[ONNX_MAX_DIMS];
        int in_idx = 0;
        for (int o = 0; o < nd_out; o++) {
            int is_new = 0;
            for (int j = 0; j < n_axes; j++)
                if (axes[j] == o) { is_new = 1; break; }
            if (is_new)
                onnx_out[o] = 1;
            else
                onnx_out[o] = (in_idx < nd_in) ? onnx_in[in_idx++] : 1;
        }

        /* Reverse to ggml ne order */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < nd_out && d < GGML_MAX_DIMS; d++)
            ne[d] = onnx_out[nd_out - 1 - d];

        out = onnx_reshape_nd(c->ctx, a, ne, nd_out);
        /* Preserve ONNX ndims only for real data tensors.
         * For scalar-like tensors (all dims==1), let squeeze determine ndims
         * so shape-tensor chains (Unsqueeze→Concat→Reshape) work correctly. */
        {
            int has_nonunit = 0;
            for (int d = 0; d < nd_out; d++)
                if (ne[d] > 1) { has_nonunit = 1; break; }
            if (has_nonunit) out_nd = nd_out;
        }

        /* cval propagation: Unsqueeze preserves values */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0)
                cval_put(c, n->outputs[0], cv, ncv);
        }
    }
    else if (strcmp(op, "Squeeze") == 0) {
        if (!a) return -1;
        /* Get axes from attribute (opset < 13) or second input (opset >= 13) */
        int64_t axes[ONNX_MAX_DIMS];
        int n_axes = onnx_attr_ints(n, "axes", axes, ONNX_MAX_DIMS);
        if (n_axes == 0 && n->n_inputs > 1) {
            const onnx_initializer_t *axes_init = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (axes_init && axes_init->raw_data && axes_init->data_type == ONNX_DTYPE_INT64) {
                n_axes = (int)(axes_init->raw_size / sizeof(int64_t));
                if (n_axes > ONNX_MAX_DIMS) n_axes = ONNX_MAX_DIMS;
                memcpy(axes, axes_init->raw_data, n_axes * sizeof(int64_t));
            }
            /* Fallback: try cval (from Constant nodes) */
            if (n_axes == 0) {
                n_axes = cval_get(c, n->inputs[1], axes, ONNX_MAX_DIMS);
            }
        }

        /* Use stored ONNX ndims when available, fallback to 4 */
        int nd_in = tmap_get_ndims(c, n->inputs[0]);
        if (nd_in <= 0) {
            nd_in = GGML_MAX_DIMS;
            while (nd_in > 1 && a->ne[nd_in - 1] == 1) nd_in--;
        }
        if (nd_in > GGML_MAX_DIMS) nd_in = GGML_MAX_DIMS;

        /* Normalize negative axes */
        for (int i = 0; i < n_axes; i++)
            if (axes[i] < 0) axes[i] += nd_in;

        /* Build ONNX dims from ggml ne (reversed, using nd_in) */
        int64_t onnx_in[ONNX_MAX_DIMS];
        for (int i = 0; i < ONNX_MAX_DIMS; i++) onnx_in[i] = 1;
        for (int i = 0; i < nd_in; i++)
            onnx_in[i] = a->ne[nd_in - 1 - i];

        int64_t onnx_out[ONNX_MAX_DIMS];
        int nd_out = 0;
        for (int i = 0; i < nd_in; i++) {
            int squeeze = 0;
            if (n_axes == 0) {
                squeeze = (onnx_in[i] == 1);
            } else {
                for (int j = 0; j < n_axes; j++)
                    if (axes[j] == i) { squeeze = 1; break; }
            }
            if (!squeeze)
                onnx_out[nd_out++] = onnx_in[i];
        }
        if (nd_out == 0) { nd_out = 1; onnx_out[0] = 1; }

        /* Reverse to ggml ne order */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < nd_out && d < GGML_MAX_DIMS; d++)
            ne[d] = onnx_out[nd_out - 1 - d];

        out = onnx_reshape_nd(c->ctx, a, ne, nd_out);
        {
            int has_nonunit = 0;
            for (int d = 0; d < nd_out; d++)
                if (ne[d] > 1) { has_nonunit = 1; break; }
            if (has_nonunit) out_nd = nd_out;
        }
    }

    /* ── Concat ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Concat") == 0) {
        if (!a) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        /* Determine effective ONNX ndims for axis mapping.
         * Use tmap stored ndims (from value_info or previous ops),
         * falling back to ggml_n_dims. */
        int eff_nd = ggml_n_dims(a);
        for (int i = 0; i < n->n_inputs; i++) {
            int stored = tmap_get_ndims(c, n->inputs[i]);
            if (stored > eff_nd) eff_nd = stored;
            struct ggml_tensor *ti = get_input(c, n, i);
            if (ti) {
                int ndi = ggml_n_dims(ti);
                if (ndi > eff_nd) eff_nd = ndi;
            }
        }
        if (eff_nd < 1) eff_nd = 1;
        int nd = eff_nd;
        int onnx_axis = (int)axis;
        if (onnx_axis < 0) onnx_axis = eff_nd + onnx_axis;
        int dim = eff_nd - 1 - onnx_axis;
        if (dim < 0) dim = 0;
        if (dim > GGML_MAX_DIMS - 1) dim = GGML_MAX_DIMS - 1;
        /* Concat supports N inputs — chain pairwise */
        out = a;
        for (int i = 1; i < n->n_inputs; i++) {
            struct ggml_tensor *inp = get_input(c, n, i);
            if (!inp) continue;
            /* ggml_concat requires matching types — cast if needed */
            if (out->type != inp->type) {
                if (inp->type == GGML_TYPE_I32 && out->type == GGML_TYPE_F32)
                    inp = ggml_cast(c->ctx, inp, GGML_TYPE_F32);
                else if (out->type == GGML_TYPE_I32 && inp->type == GGML_TYPE_F32) {
                    out = ggml_cast(c->ctx, out, GGML_TYPE_F32);
                }
            }
            out = ggml_concat(c->ctx, out, inp, dim);
        }
        out_nd = nd; /* preserve ONNX ndims for correct axis mapping downstream */

        /* Propagate compile-time values for 1D Concat (shape tensor concatenation) */
        if (onnx_axis == 0 && nd == 1) {
            int64_t merged[ONNX_MAX_DIMS];
            int total = 0;
            int all_known = 1;
            for (int i = 0; i < n->n_inputs; i++) {
                int64_t part[ONNX_MAX_DIMS];
                int np = cval_get(c, n->inputs[i], part, ONNX_MAX_DIMS);
                if (np == 0) { all_known = 0; break; }
                for (int j = 0; j < np && total < ONNX_MAX_DIMS; j++)
                    merged[total++] = part[j];
            }
            if (all_known && total > 0)
                cval_put(c, n->outputs[0], merged, total);
        }
    }

    /* ── Gather ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Gather") == 0) {
        if (!a || !b) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        int b_nd = tmap_get_ndims(c, n->inputs[1]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);
        if (b_nd <= 0) b_nd = (int)ggml_n_dims(b);

        /* Case 1: scalar/shape indexing — data is small 1D (shape tensor,
         * constants) and both have cval → create scalar constant.
         * This avoids ggml_get_rows which is designed for embedding lookup. */
        int64_t cv_data[ONNX_MAX_DIMS], cv_idx[ONNX_MAX_DIMS];
        int ncv_data = cval_get(c, n->inputs[0], cv_data, ONNX_MAX_DIMS);
        int ncv_idx  = cval_get(c, n->inputs[1], cv_idx, ONNX_MAX_DIMS);
        if (ncv_data > 0 && ncv_idx > 0) {
            /* Both compile-time known: resolve at graph-build time */
            int64_t result[ONNX_MAX_DIMS];
            int nr = 0;
            for (int j = 0; j < ncv_idx; j++) {
                int64_t idx = cv_idx[j];
                if (idx < 0) idx += ncv_data;
                if (idx >= 0 && idx < ncv_data && nr < ONNX_MAX_DIMS)
                    result[nr++] = cv_data[idx];
            }
            /* Create scalar/small constant tensor in ctx_weight */
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            if (nr == 0) nr = 1;
            out = ggml_new_tensor_1d(wctx, a->type, nr);
            if (out) {
                ggml_set_input(out);
                ggml_set_name(out, n->outputs[0]);
                tmap_put_nd(c, n->outputs[0], out, nr > 1 ? 1 : 1);
                cval_put(c, n->outputs[0], result, nr);
                /* Register for deferred fill (const_fill can't handle multi-value,
                 * so stash as shape tensor) */
                if (c->n_shape_tensors < ONNX_MAX_DEFERRED) {
                    c->shape_tensors_ne[c->n_shape_tensors][0] = nr;
                    for (int j = 0; j < nr; j++)
                        c->shape_tensors_ne[c->n_shape_tensors][j + 1] = result[j];
                    c->shape_tensor_ptrs[c->n_shape_tensors] = out;
                    c->n_shape_tensors++;
                }
            }
            return 0;
        }

        /* Case 2: embedding lookup — ggml_get_rows */
        if (b->type != GGML_TYPE_I32) {
            b = ggml_cast(c->ctx, b, GGML_TYPE_I32);
        }

        /* Normalize negative axis */
        if (axis < 0) axis += a_nd;

        if (axis == 0 && a_nd > 2) {
            /* Gather axis=0 on rank>2: ggml_get_rows only handles 2D.
             * ONNX axis=0 selects along the first ONNX dim = last ggml dim.
             * Flatten all dims except the last into row_size, gather, reshape back. */
            int g_nd = ggml_n_dims(a);
            if (g_nd < 2) g_nd = 2;
            int64_t row_size = 1;
            for (int d = 0; d < g_nd - 1; d++)
                row_size *= a->ne[d];
            struct ggml_tensor *a2d = ggml_reshape_2d(c->ctx, a, row_size, a->ne[g_nd - 1]);
            struct ggml_tensor *gathered = ggml_get_rows(c->ctx, a2d, b);
            /* Reshape back: replace the gathered dim with n_indices */
            int64_t n_idx = (int64_t)ggml_nelements(b);
            int64_t back_ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
            for (int d = 0; d < g_nd - 1; d++)
                back_ne[d] = a->ne[d];
            back_ne[g_nd - 1] = n_idx;
            out = onnx_reshape_nd(c->ctx, gathered, back_ne, g_nd);
        } else {
            out = ggml_get_rows(c->ctx, a, b);
        }

        /* Register output with correct ONNX ndims.
         * ONNX Gather(data, indices, axis=0):
         * output_shape = indices_shape + data_shape[1:]
         * E.g. data [V,D] (2D) + indices [B,S] (2D) → [B,S,D] (3D). */
        if (out) {
            int out_nd = b_nd + (a_nd > 1 ? a_nd - 1 : 0);
            if (out_nd < 1) out_nd = 1;
            if (out_nd > 4) out_nd = 4;
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, out_nd);
        }
        return 0; /* already registered */
    }

    /* ── ScatterElements ────────────────────────────────────────── */
    else if (strcmp(op, "ScatterElements") == 0) {
        if (!a || !b) return -1;
        /* ONNX ScatterElements(data, indices, updates, axis=0, reduction)
         * data:    [D0, D1, ...] — base tensor
         * indices: [I0, I1, ...] — index tensor (same shape as updates)
         * updates: [I0, I1, ...] — values to scatter
         * Input mapping: a=data, b=indices, c_upd=updates (input[2]) */
        struct ggml_tensor *c_upd = NULL;
        if (n->n_inputs > 2 && n->inputs[2][0] != '\0')
            c_upd = tmap_get(c, n->inputs[2]);
        if (!c_upd) { fprintf(stderr, "[onnx] ScatterElements: missing updates\n"); return -1; }

        int64_t axis = onnx_attr_int(n, "axis", 0);
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);
        if (axis < 0) axis += a_nd;

        /* ONNX axis → ggml dim (reversed) */
        int ggml_axis = a_nd - 1 - (int)axis;
        if (ggml_axis < 0) ggml_axis = 0;

        /* Determine reduction: 0=none, 1=add */
        int reduction = 0;
        {
            char red_str[64] = {0};
            int red_len = onnx_attr_str(n, "reduction", red_str, sizeof(red_str));
            if (red_len > 0) {
                if (strcmp(red_str, "add") == 0) reduction = 1;
                else if (strcmp(red_str, "none") == 0) reduction = 0;
                else {
                    fprintf(stderr, "[onnx] ScatterElements: reduction='%s' not supported\n", red_str);
                    return -1;
                }
            }
        }

        /* Cast indices to I32 if needed */
        if (b->type != GGML_TYPE_I32) {
            b = ggml_cast(c->ctx, b, GGML_TYPE_I32);
        }
        /* Cast updates to F32 if needed */
        if (c_upd->type != GGML_TYPE_F32) {
            c_upd = ggml_cast(c->ctx, c_upd, GGML_TYPE_F32);
        }
        /* Cast data to F32 if needed */
        if (a->type != GGML_TYPE_F32) {
            a = ggml_cast(c->ctx, a, GGML_TYPE_F32);
        }

        out = ggml_scatter_elements(c->ctx, a, c_upd, b, reduction, ggml_axis);
        if (out) {
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, a_nd);
        }
        return 0;
    }

    /* ── Slice ──────────────────────────────────────────────────── */
    else if (strcmp(op, "Slice") == 0) {
        if (!a) return -1;
        /* Inputs: data, starts, ends, [axes], [steps] — all from initializers */
        int64_t starts[4] = {0}, ends[4] = {0};
        int64_t axes_arr[4] = {0, 1, 2, 3}, steps[4] = {1, 1, 1, 1};
        int n_slices = 0;
        int has_axes = 0;

        /* Helper macro: read int64 values from initializer, Constant, or cval */
        #define READ_SLICE_INPUT(idx, dst, cnt) do { \
            if (n->n_inputs > (idx) && n->inputs[idx][0] != '\0') { \
                const onnx_initializer_t *_si = onnx_find_initializer(c->onnx, n->inputs[idx]); \
                if (!_si) _si = find_constant_tensor(c->onnx, n->inputs[idx]); \
                if (_si && _si->raw_data && _si->data_type == ONNX_DTYPE_INT64) { \
                    int _n = (int)(_si->raw_size / sizeof(int64_t)); \
                    if (_n > 4) _n = 4; \
                    memcpy(dst, _si->raw_data, _n * sizeof(int64_t)); \
                    cnt = _n; \
                } else { \
                    int _n = cval_get(c, n->inputs[idx], dst, 4); \
                    if (_n > 0) cnt = _n; \
                } \
            } \
        } while(0)

        /* Read starts (input 1) */
        READ_SLICE_INPUT(1, starts, n_slices);
        /* Read ends (input 2) */
        { int _d_ends = 0; READ_SLICE_INPUT(2, ends, _d_ends); (void)_d_ends; }
        /* Read axes (input 3, optional) */
        if (n->n_inputs > 3 && n->inputs[3][0] != '\0') {
            int _d_axes = 0;
            READ_SLICE_INPUT(3, axes_arr, _d_axes);
            (void)_d_axes;
            has_axes = 1;
        }
        /* Read steps (input 4, optional) */
        if (n->n_inputs > 4 && n->inputs[4][0] != '\0') {
            int _dummy_steps = 0;
            READ_SLICE_INPUT(4, steps, _dummy_steps);
            (void)_dummy_steps;
        }
        #undef READ_SLICE_INPUT

        /* Determine ONNX ndims — prefer stored ndims over ggml_n_dims
         * (ggml_n_dims ignores trailing 1-dims, e.g. [24,128,4,1] → 3 not 4) */
        int nd_onnx = tmap_get_ndims(c, n->inputs[0]);
        if (nd_onnx <= 0) nd_onnx = ggml_n_dims(a);
        if (nd_onnx < 1) nd_onnx = 1;

        /* Convert ONNX axes to ggml dims and compute view params.
         * ONNX dim d → ggml dim (nd_onnx-1-d).
         * We build offset, output ne[], and normalized starts/steps in ggml order. */
        int64_t out_ne[GGML_MAX_DIMS], offsets[GGML_MAX_DIMS] = {0};
        int64_t norm_starts[GGML_MAX_DIMS] = {0}, norm_steps[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < GGML_MAX_DIMS; d++)
            out_ne[d] = a->ne[d];

        for (int i = 0; i < n_slices; i++) {
            int onnx_ax = has_axes ? (int)axes_arr[i] : i;
            if (onnx_ax < 0) onnx_ax += nd_onnx;
            int ggml_d = nd_onnx - 1 - onnx_ax;
            if (ggml_d < 0 || ggml_d > GGML_MAX_DIMS - 1) continue;

            int64_t dim_size = a->ne[ggml_d];
            int64_t s = starts[i], e = ends[i], st = steps[i];
            if (st == 0) st = 1; /* safety */

            /* Normalize negative indices */
            if (s < 0) s += dim_size;
            if (e < 0) e += dim_size;

            /* Clamp per ONNX spec (different for positive vs negative step) */
            if (st > 0) {
                if (s < 0) s = 0;
                if (s > dim_size) s = dim_size;
                if (e < 0) e = 0;
                if (e > dim_size) e = dim_size;
            } else {
                /* step < 0: clamp to [-1, dim_size-1] */
                if (s < -1) s = -1;
                if (s > dim_size - 1) s = dim_size - 1;
                if (e < -1) e = -1;
                if (e > dim_size - 1) e = dim_size - 1;
            }

            int64_t len;
            if (st > 0) {
                len = (e - s + st - 1) / st; /* ceil((e-s)/st) */
            } else {
                /* step < 0: len = ceil((s - e) / |st|) — e.g. s=2,e=-1,st=-1 → len=3 */
                len = (s - e + (-st) - 1) / (-st);
            }
            if (len < 0) len = 0;

            offsets[ggml_d] = (st > 0) ? s : 0; /* for step>0 view; step<0 uses deferred */
            norm_starts[ggml_d] = s;
            norm_steps[ggml_d] = st;
            out_ne[ggml_d] = len;
        }

        /* Only step=1 supported via ggml_view — check */
        int all_step1 = 1;
        for (int i = 0; i < n_slices; i++)
            if (steps[i] != 1) { all_step1 = 0; break; }

        if (all_step1) {
            size_t offset_bytes = 0;
            for (int d = 0; d < GGML_MAX_DIMS; d++)
                offset_bytes += offsets[d] * a->nb[d];
            /* Use ONNX ndims for view dimension (not ggml_n_dims which drops trailing 1s) */
            int nd = nd_onnx;
            if (nd > GGML_MAX_DIMS) nd = GGML_MAX_DIMS;

            switch (nd) {
                case 1:
                    out = ggml_view_1d(c->ctx, a, out_ne[0], offset_bytes);
                    break;
                case 2:
                    out = ggml_view_2d(c->ctx, a, out_ne[0], out_ne[1],
                                       a->nb[1], offset_bytes);
                    break;
                case 3:
                    out = ggml_view_3d(c->ctx, a, out_ne[0], out_ne[1], out_ne[2],
                                       a->nb[1], a->nb[2], offset_bytes);
                    break;
                case 4:
                    out = ggml_view_4d(c->ctx, a, out_ne[0], out_ne[1],
                                       out_ne[2], out_ne[3],
                                       a->nb[1], a->nb[2], a->nb[3],
                                       offset_bytes);
                    break;
                default:
                    out = ggml_view_5d(c->ctx, a, out_ne[0], out_ne[1],
                                       out_ne[2], out_ne[3], out_ne[4],
                                       a->nb[1], a->nb[2], a->nb[3], a->nb[4],
                                       offset_bytes);
                    break;
            }
            /* Make contiguous so downstream ops work correctly */
            out = ggml_cont(c->ctx, out);
        } else {
            /* step != 1: deferred strided copy after alloc */
            if (c->n_slice_fills >= ONNX_MAX_DEFERRED) {
                fprintf(stderr, "onnx_ggml: too many strided Slice ops (max 64)\n");
                return -1;
            }
            /* Create output tensor with correct shape */
            int nd = nd_onnx;
            if (nd > GGML_MAX_DIMS) nd = GGML_MAX_DIMS;
            out = onnx_new_tensor_nd(c->ctx, a->type, out_ne, nd);
            ggml_set_input(out);

            int sf = c->n_slice_fills;
            c->slice_fill_src[sf] = a;
            c->slice_fill_dst[sf] = out;
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                c->slice_fill_starts[sf][d] = norm_starts[d];
                c->slice_fill_steps[sf][d]  = norm_steps[d];
                c->slice_fill_out_ne[sf][d] = out_ne[d];
            }
            c->slice_fill_ndims[sf] = nd_onnx;
            c->n_slice_fills++;
        }

        /* Propagate compile-time values through Slice (for shape tensors).
         * Supports strided/reverse slicing and multi-dim tensors. */
        {
            int64_t src_vals[ONNX_MAX_DIMS];
            int nv = cval_get(c, n->inputs[0], src_vals, ONNX_MAX_DIMS);
            if (nv > 0 && n_slices > 0) {
                /* Total output elements */
                int64_t total_out = ne_product(out_ne, GGML_MAX_DIMS);

                if (total_out > 0 && total_out <= ONNX_MAX_DIMS) {
                    int64_t result[ONNX_MAX_DIMS];
                    /* Compute strides for source and output in ggml order */
                    int64_t src_stride[GGML_MAX_DIMS], out_stride[GGML_MAX_DIMS];
                    src_stride[0] = 1; out_stride[0] = 1;
                    for (int d = 1; d < GGML_MAX_DIMS; d++) {
                        src_stride[d] = src_stride[d-1] * a->ne[d-1];
                        out_stride[d] = out_stride[d-1] * out_ne[d-1];
                    }
                    /* Iterate all output elements via flat index */
                    for (int64_t di = 0; di < total_out; di++) {
                        int64_t si = 0;
                        int64_t rem = di;
                        for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
                            int64_t coord = rem / out_stride[d];
                            rem -= coord * out_stride[d];
                            si += (norm_starts[d] + coord * norm_steps[d]) * src_stride[d];
                        }
                        if ((size_t)si < (size_t)nv)
                            result[di] = src_vals[si];
                    }
                    cval_put(c, n->outputs[0], result, (int)total_out);
                }
            }
        }
    }

    /* ── Split ──────────────────────────────────────────────────── */
    else if (strcmp(op, "Split") == 0) {
        if (!a) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        /* Use original ONNX ndims for >4D axis mapping */
        int nd = ggml_n_dims(a);
        if (n->n_inputs > 0) {
            int in_nd = tmap_get_ndims(c, n->inputs[0]);
            if (in_nd > nd) nd = in_nd;
        }
        if (nd < 1) nd = 1;
        if (axis < 0) axis += nd;
        int ggml_d = nd - 1 - (int)axis;
        if (ggml_d < 0) ggml_d = 0;
        if (ggml_d > GGML_MAX_DIMS - 1) ggml_d = GGML_MAX_DIMS - 1;

        int64_t dim_size = a->ne[ggml_d];
        int n_out = n->n_outputs;

        /* Get split sizes: from input 1 (opset 13+) or attribute */
        int64_t splits[ONNX_MAX_OUTPUTS];
        int n_splits = onnx_attr_ints(n, "split", splits, ONNX_MAX_OUTPUTS);

        if (n_splits == 0 && n->n_inputs > 1 && n->inputs[1][0] != '\0') {
            const onnx_initializer_t *si = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (si && si->raw_data && si->data_type == ONNX_DTYPE_INT64) {
                n_splits = (int)(si->raw_size / sizeof(int64_t));
                if (n_splits > ONNX_MAX_OUTPUTS) n_splits = ONNX_MAX_OUTPUTS;
                memcpy(splits, si->raw_data, n_splits * sizeof(int64_t));
            }
        }

        /* Default: equal split */
        if (n_splits == 0 && n_out > 0) {
            int64_t chunk = dim_size / n_out;
            for (int i = 0; i < n_out; i++)
                splits[i] = chunk;
            splits[n_out - 1] = dim_size - chunk * (n_out - 1);
            n_splits = n_out;
        }

        /* Create view for each split output */
        int64_t offset = 0;
        for (int i = 0; i < n_splits && i < n_out; i++) {
            int64_t out_ne[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; d++)
                out_ne[d] = a->ne[d];
            out_ne[ggml_d] = splits[i];

            size_t offset_bytes = offset * a->nb[ggml_d];
            int vnd = GGML_MAX_DIMS;
            while (vnd > 1 && out_ne[vnd-1] == 1) vnd--;

            struct ggml_tensor *view;
            switch (vnd) {
                case 1:
                    view = ggml_view_1d(c->ctx, a, out_ne[0], offset_bytes);
                    break;
                case 2:
                    view = ggml_view_2d(c->ctx, a, out_ne[0], out_ne[1],
                                        a->nb[1], offset_bytes);
                    break;
                case 3:
                    view = ggml_view_3d(c->ctx, a, out_ne[0], out_ne[1], out_ne[2],
                                        a->nb[1], a->nb[2], offset_bytes);
                    break;
                case 4:
                    view = ggml_view_4d(c->ctx, a, out_ne[0], out_ne[1],
                                        out_ne[2], out_ne[3],
                                        a->nb[1], a->nb[2], a->nb[3],
                                        offset_bytes);
                    break;
                default:
                    view = ggml_view_5d(c->ctx, a, out_ne[0], out_ne[1],
                                        out_ne[2], out_ne[3], out_ne[4],
                                        a->nb[1], a->nb[2], a->nb[3], a->nb[4],
                                        offset_bytes);
                    break;
            }
            view = ggml_cont(c->ctx, view);

            if (n->outputs[i][0] != '\0') {
                ggml_set_name(view, n->outputs[i]);
                tmap_put_nd(c, n->outputs[i], view, nd);
            }
            offset += splits[i];
        }
        return 0; /* outputs already registered */
    }

    /* ── Resize / Upsample ──────────────────────────────────────── */
    else if (strcmp(op, "Resize") == 0 || strcmp(op, "Upsample") == 0) {
        if (!a) return -1;

        /* mode attribute */
        char mode_str[32] = "nearest";
        onnx_attr_str(n, "mode", mode_str, sizeof(mode_str));
        enum ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST;
        if (strcmp(mode_str, "linear") == 0 || strcmp(mode_str, "bilinear") == 0)
            mode = GGML_SCALE_MODE_BILINEAR;

        /* Target sizes: from "sizes" input (input 3) or "scales" input (input 2).
         * Resize: inputs = [X, roi, scales, sizes]
         * Upsample: inputs = [X, scales] */
        int64_t target_ne[4];
        for (int d = 0; d < 4; d++)
            target_ne[d] = a->ne[d];

        int got_target = 0;

        /* Try sizes input (Resize input 3) */
        if (n->n_inputs > 3 && n->inputs[3][0] != '\0') {
            const onnx_initializer_t *si = onnx_find_initializer(c->onnx, n->inputs[3]);
            if (si && si->raw_data && si->data_type == ONNX_DTYPE_INT64) {
                int nsz = (int)(si->raw_size / sizeof(int64_t));
                int64_t sizes[4];
                if (nsz > 4) nsz = 4;
                memcpy(sizes, si->raw_data, nsz * sizeof(int64_t));
                /* sizes are in ONNX order → reverse to ggml */
                for (int d = 0; d < nsz && d < 4; d++)
                    target_ne[d] = sizes[nsz - 1 - d];
                got_target = 1;
            }
        }

        /* Try scales input (Resize input 2, or Upsample input 1) */
        if (!got_target) {
            int scales_idx = (strcmp(op, "Upsample") == 0) ? 1 : 2;
            if (n->n_inputs > scales_idx && n->inputs[scales_idx][0] != '\0') {
                const onnx_initializer_t *sci = onnx_find_initializer(c->onnx, n->inputs[scales_idx]);
                if (sci && sci->raw_data && sci->data_type == ONNX_DTYPE_FLOAT) {
                    int nsc = (int)(sci->raw_size / sizeof(float));
                    float scales[4];
                    if (nsc > 4) nsc = 4;
                    memcpy(scales, sci->raw_data, nsc * sizeof(float));
                    /* scales in ONNX order → reverse to ggml */
                    for (int d = 0; d < nsc && d < 4; d++) {
                        int ggml_d = nsc - 1 - d;
                        target_ne[ggml_d] = (int64_t)(a->ne[ggml_d] * scales[d]);
                        if (target_ne[ggml_d] < 1) target_ne[ggml_d] = 1;
                    }
                    got_target = 1;
                }
            }
        }

        if (!got_target) {
            /* Try cval (compile-time value from Shape→Concat chain) for sizes */
            if (n->n_inputs > 3 && n->inputs[3][0] != '\0') {
                int64_t cv[ONNX_MAX_DIMS];
                int ncv = cval_get(c, n->inputs[3], cv, ONNX_MAX_DIMS);
                if (ncv > 0) {
                    /* cv is in ONNX order → reverse to ggml */
                    for (int d = 0; d < ncv && d < 4; d++)
                        target_ne[d] = cv[ncv - 1 - d];
                    got_target = 1;
                }
            }
        }
        if (!got_target) {
            /* Try cval for scales */
            int scales_idx = (strcmp(op, "Upsample") == 0) ? 1 : 2;
            if (n->n_inputs > scales_idx && n->inputs[scales_idx][0] != '\0') {
                int64_t cv[ONNX_MAX_DIMS];
                int ncv = cval_get(c, n->inputs[scales_idx], cv, ONNX_MAX_DIMS);
                if (ncv > 0) {
                    for (int d = 0; d < ncv && d < 4; d++) {
                        float s;
                        int32_t bits = (int32_t)cv[d];
                        memcpy(&s, &bits, sizeof(float));
                        int ggml_d = ncv - 1 - d;
                        target_ne[ggml_d] = (int64_t)(a->ne[ggml_d] * s);
                        if (target_ne[ggml_d] < 1) target_ne[ggml_d] = 1;
                    }
                    got_target = 1;
                }
            }
        }
        out = ggml_interpolate(c->ctx, a,
                               target_ne[0], target_ne[1],
                               target_ne[2], target_ne[3],
                               (uint32_t)mode);
    }

    /* ── Expand ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Expand") == 0) {
        if (!a || !b) return -1;
        /* b is the target shape tensor — read from initializer, Constant, or cval */
        int64_t shape[ONNX_MAX_DIMS];
        int ndims = 0;

        const onnx_initializer_t *shape_init = onnx_find_initializer(c->onnx, n->inputs[1]);
        if (!shape_init)
            shape_init = find_constant_tensor(c->onnx, n->inputs[1]);

        if (shape_init && shape_init->raw_data && shape_init->data_type == ONNX_DTYPE_INT64) {
            ndims = (int)(shape_init->raw_size / sizeof(int64_t));
            if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
            memcpy(shape, shape_init->raw_data, ndims * sizeof(int64_t));
        }

        /* Fallback: try compile-time value map */
        if (ndims == 0) {
            ndims = cval_get(c, n->inputs[1], shape, ONNX_MAX_DIMS);
        }
        if (ndims == 0) return -1;

        /* Resolve -1 (keep dim) using full ONNX shape of input tensor. */
        {
            int64_t a_onnx[ONNX_MAX_DIMS];
            int a_nd = tmap_get_shape(c, n->inputs[0], a_onnx, ONNX_MAX_DIMS);
            /* Align from the right: shape[ndims-1-k] ← a_onnx[a_nd-1-k] */
            for (int d = 0; d < ndims; d++) {
                if (shape[d] == -1) {
                    int a_d = d - (ndims - a_nd); /* right-aligned index into a_onnx */
                    if (a_d >= 0 && a_d < a_nd)
                        shape[d] = a_onnx[a_d];
                    else
                        shape[d] = 1;
                }
            }
        }
        /* Save resolved full ONNX shape before collapse */
        int orig_expand_ndims = ndims;
        int64_t orig_expand_shape[ONNX_MAX_DIMS];
        memcpy(orig_expand_shape, shape, ndims * sizeof(int64_t));

        /* Collapse >5D ONNX shape into 5D by merging leading ONNX dims. */
        if (ndims > GGML_MAX_DIMS) {
            int64_t merged = 1;
            for (int d = 0; d < ndims - (GGML_MAX_DIMS - 1); d++)
                merged *= shape[d];
            int64_t tmp[GGML_MAX_DIMS];
            tmp[0] = merged;
            for (int d = 1; d < GGML_MAX_DIMS; d++)
                tmp[d] = shape[ndims - (GGML_MAX_DIMS - 1) + d - 1];
            memcpy(shape, tmp, sizeof(tmp));
            ndims = GGML_MAX_DIMS;
        }

        /* Numpy-style broadcast: if rank(a) < rank(target), left-pad a
         * with 1s so ranks match, then apply broadcast rules. */
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) {
            a_nd = (int)ggml_n_dims(a);
        }
        if (a_nd < ndims && ndims <= GGML_MAX_DIMS) {
            int64_t a_ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
            for (int d = 0; d < a_nd; d++)
                a_ne[d] = a->ne[d];
            a = onnx_reshape_nd(c->ctx, a, a_ne, ndims);
        }

        /* Reverse ONNX shape → ggml ne order */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < ndims; d++)
            ne[d] = shape[ndims - 1 - d];

        /* Expand semantics: broadcast a into target shape.
         * For each dim: if shape[d]==1 → use a->ne[d],
         * if a->ne[d]==1 → use shape[d], otherwise must match. */
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (ne[d] == 1) ne[d] = a->ne[d];
        }

        struct ggml_tensor *target = onnx_new_tensor_nd(c->ctx, a->type, ne, ndims);
        if (ggml_are_same_shape(a, target)) {
            out = ggml_dup(c->ctx, a);  /* same shape: copy, not alias */
        } else {
            if (!ggml_can_repeat(a, target)) {
                fprintf(stderr, "[Expand] repeat FAIL: '%s' a.ne=[%lld,%lld,%lld,%lld] target=[%lld,%lld,%lld,%lld] shape_input='%s' ndims=%d\n",
                        n->outputs[0],
                        (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                        (long long)ne[0],(long long)ne[1],(long long)ne[2],(long long)ne[3],
                        n->inputs[1], ndims);
                fprintf(stderr, "  ONNX shape=[");
                for (int d = 0; d < ndims; d++) fprintf(stderr, "%lld%s", (long long)shape[d], d<ndims-1?",":"");
                fprintf(stderr, "] a_input='%s' a_ndims=%d\n", n->inputs[0], tmap_get_ndims(c, n->inputs[0]));
            }
            out = ggml_repeat(c->ctx, a, target);
        }
        /* Register with full resolved ONNX shape (before collapse) */
        if (out) {
            ggml_set_name(out, n->outputs[0]);
            tmap_put_shape(c, n->outputs[0], out, orig_expand_shape, orig_expand_ndims);
            return 0;
        }
    }

    /* ── Normalization ──────────────────────────────────────────── */
    else if (strcmp(op, "BatchNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *bias  = get_input(c, n, 2);
        struct ggml_tensor *mean  = get_input(c, n, 3);
        struct ggml_tensor *var   = get_input(c, n, 4);

        /* With reversed dims, input [N,C,H,W] → ggml [W,H,C,N].
         * BN params [C] → ggml ne[0]=C. Need reshape to [1,1,C,1]
         * for broadcast over [W,H,C,N]. For 2D input [N,C] → ggml [C,N],
         * params [C] broadcast fine (ne[0]=C matches). */
        int nd = ggml_n_dims(a);
        if (nd > 2) {
            /* Reshape channel params [C] → [1,1,C,1] for spatial broadcast */
            int64_t ch = ggml_nelements(mean);
            if (mean)  mean  = ggml_reshape_4d(c->ctx, mean,  1, 1, ch, 1);
            if (var)   var   = ggml_reshape_4d(c->ctx, var,   1, 1, ch, 1);
            if (scale) scale = ggml_reshape_4d(c->ctx, scale, 1, 1, ch, 1);
            if (bias)  bias  = ggml_reshape_4d(c->ctx, bias,  1, 1, ch, 1);
        }

        /* x_norm = (x - mean) / sqrt(var + eps) */
        out = ggml_sub(c->ctx, a, mean);
        struct ggml_tensor *eps_t = make_scalar(c, eps);
        struct ggml_tensor *std = ggml_sqrt(c->ctx, ggml_add(c->ctx, var, eps_t));
        out = ggml_div(c->ctx, out, std);
        if (scale) out = ggml_mul(c->ctx, out, scale);
        if (bias)  out = ggml_add(c->ctx, out, bias);
    }
    else if (strcmp(op, "LayerNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *bias  = get_input(c, n, 2);

        /* With reversed dims, ONNX last axis (normalized) = ggml ne[0].
         * ggml_norm normalizes over ne[0] — exactly what we need. */
        out = ggml_norm(c->ctx, a, eps);
        if (scale) out = ggml_mul(c->ctx, out, scale);
        if (bias)  out = ggml_add(c->ctx, out, bias);
    }
    else if (strcmp(op, "GroupNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        int64_t num_groups = onnx_attr_int(n, "num_groups", 1);
        out = ggml_group_norm(c->ctx, a, (int)num_groups, eps);
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *bias  = get_input(c, n, 2);
        /* Reshape [C] → [1,1,C,1] for broadcast over [W,H,C,N] */
        int nd = ggml_n_dims(a);
        if (nd > 2) {
            int64_t ch = ggml_nelements(scale ? scale : bias);
            if (scale) scale = ggml_reshape_4d(c->ctx, scale, 1, 1, ch, 1);
            if (bias)  bias  = ggml_reshape_4d(c->ctx, bias,  1, 1, ch, 1);
        }
        if (scale) out = ggml_mul(c->ctx, out, scale);
        if (bias)  out = ggml_add(c->ctx, out, bias);
    }
    else if (strcmp(op, "RMSNormalization") == 0) {
        if (!a) return -1;
        float eps = onnx_attr_float(n, "epsilon", 1e-5f);
        out = ggml_rms_norm(c->ctx, a, eps);
    }

    /* ── Pooling ────────────────────────────────────────────────── */
    else if (strcmp(op, "MaxPool") == 0 || strcmp(op, "AveragePool") == 0) {
        if (!a) return -1;
        int64_t kshape[2] = {1, 1}, strides[2] = {1, 1}, pads[4] = {0};
        onnx_attr_ints(n, "kernel_shape", kshape, 2);
        onnx_attr_ints(n, "strides", strides, 2);
        onnx_attr_ints(n, "pads", pads, 4);
        /* auto_pad: compute pads from input shape */
        char auto_pad[32] = "";
        onnx_attr_str(n, "auto_pad", auto_pad, sizeof(auto_pad));
        if (strcmp(auto_pad, "SAME_UPPER") == 0 || strcmp(auto_pad, "SAME_LOWER") == 0) {
            /* input is [W,H,C,N] in ggml. ONNX spatial: H=ne[1], W=ne[0] */
            for (int d = 0; d < 2; d++) {
                int64_t in_d = (d == 0) ? a->ne[1] : a->ne[0]; /* H, W */
                int64_t out_d = (in_d + strides[d] - 1) / strides[d];
                int64_t total_pad = (out_d - 1) * strides[d] + kshape[d] - in_d;
                if (total_pad < 0) total_pad = 0;
                pads[d]     = (strcmp(auto_pad, "SAME_LOWER") == 0) ?
                              (total_pad + 1) / 2 : total_pad / 2;
                pads[d + 2] = total_pad - pads[d];
            }
        }
        int64_t ceil_mode = onnx_attr_int(n, "ceil_mode", 0);
        enum ggml_op_pool pool_op = (strcmp(op, "MaxPool") == 0) ?
                                     GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;
        /* ggml_pool_2d uses symmetric padding; ONNX pads = [H_begin, W_begin, H_end, W_end].
         * Use total padding (begin+end) divided by 2, rounded up. */
        int p0 = (int)((pads[1] + pads[3] + 1) / 2); /* W pad */
        int p1 = (int)((pads[0] + pads[2] + 1) / 2); /* H pad */
        /* ceil_mode: add extra padding to emulate ceil division */
        if (ceil_mode) {
            p0 += (int)(strides[1] - 1);
            p1 += (int)(strides[0] - 1);
        }
        out = ggml_pool_2d(c->ctx, a, pool_op,
                           (int)kshape[1], (int)kshape[0],
                           (int)strides[1], (int)strides[0],
                           p0, p1);
    }
    else if (strcmp(op, "GlobalAveragePool") == 0) {
        if (!a) return -1;
        /* Pool over entire spatial dims */
        int h = (int)a->ne[1];
        int w = (int)a->ne[0];
        out = ggml_pool_2d(c->ctx, a, GGML_OP_POOL_AVG,
                           w, h, w, h, 0, 0);
    }

    /* ── Conv ───────────────────────────────────────────────────── */
    else if (strcmp(op, "Conv") == 0) {
        if (!a || !b) return -1;
        int64_t strides[2] = {1, 1}, pads[4] = {0}, dilations[2] = {1, 1};
        onnx_attr_ints(n, "strides", strides, 2);
        onnx_attr_ints(n, "pads", pads, 4);
        onnx_attr_ints(n, "dilations", dilations, 2);
        /* auto_pad */
        char auto_pad[32] = "";
        onnx_attr_str(n, "auto_pad", auto_pad, sizeof(auto_pad));
        if (strcmp(auto_pad, "SAME_UPPER") == 0 || strcmp(auto_pad, "SAME_LOWER") == 0) {
            for (int d = 0; d < 2; d++) {
                int64_t in_d = (d == 0) ? a->ne[1] : a->ne[0]; /* H, W */
                int64_t k_d = (d == 0) ? b->ne[1] : b->ne[0];
                int64_t out_d = (in_d + strides[d] - 1) / strides[d];
                int64_t eff_k = (k_d - 1) * dilations[d] + 1;
                int64_t total_pad = (out_d - 1) * strides[d] + eff_k - in_d;
                if (total_pad < 0) total_pad = 0;
                pads[d]     = (strcmp(auto_pad, "SAME_LOWER") == 0) ?
                              (total_pad + 1) / 2 : total_pad / 2;
                pads[d + 2] = total_pad - pads[d];
            }
        }

        int64_t groups = onnx_attr_int(n, "group", 1);
        /* Determine spatial dims from kernel_shape attr or ONNX ndims.
         * ggml_n_dims is unreliable when C_in=1 and C_out=1. */
        int64_t kshape[2] = {0, 0};
        int n_kshape = onnx_attr_ints(n, "kernel_shape", kshape, 2);
        int ndims_kernel;
        if (n_kshape > 0) {
            ndims_kernel = n_kshape + 2; /* spatial + C_in + C_out */
        } else {
            /* Fallback: use stored ONNX ndims, or ggml_n_dims */
            int onnx_nd = tmap_get_ndims(c, n->inputs[1]);
            ndims_kernel = (onnx_nd > 0) ? onnx_nd : ggml_n_dims(b);
        }

        if (ndims_kernel <= 3) {
            /* 1D conv — group not handled yet.
             * ggml_conv_1d requires F16 kernel (im2col hardcodes F16). */
            struct ggml_tensor *bk = b;
            if (bk->type != GGML_TYPE_F16)
                bk = ggml_cast(c->ctx, bk, GGML_TYPE_F16);
            out = ggml_conv_1d(c->ctx, bk, a,
                               (int)strides[0], (int)pads[0], (int)dilations[0]);
        } else if (groups == 1) {
            /* Standard 2D conv */
            out = ggml_conv_2d(c->ctx, b, a,
                               (int)strides[1], (int)strides[0],
                               (int)pads[1], (int)pads[0],
                               (int)dilations[1], (int)dilations[0]);
        } else {
            /* Grouped / depthwise 2D conv.
             * ggml layout (reversed from ONNX):
             *   input a = [W, H, C_in, N]
             *   kernel b = [KW, KH, C_in/groups, C_out]
             * C_in_per_group = C_in / groups
             * C_out_per_group = C_out / groups */
            int64_t C_in  = a->ne[2];
            int64_t C_out = b->ne[3];
            int64_t C_in_g  = C_in / groups;
            int64_t C_out_g = C_out / groups;

            if (groups == C_in && C_in_g == 1) {
                /* Depthwise conv: use direct kernel (avoids F16 im2col in ggml_conv_2d_dw
                 * which creates MUL_MAT with F16 src1 unsupported by CPU backend) */
                out = ggml_conv_2d_dw_direct(c->ctx, b, a,
                                      (int)strides[1], (int)strides[0],
                                      (int)pads[1], (int)pads[0],
                                      (int)dilations[1], (int)dilations[0]);
            } else {
                /* General grouped conv: split, conv each group, concat.
                 * Split input along dim2 (C_in), kernel along dim3 (C_out). */
                struct ggml_tensor *group_outs[512];
                if (groups > 512) { fprintf(stderr, "[onnx] Conv groups=%lld > 512\n", (long long)groups); return -1; }

                for (int64_t g = 0; g < groups; g++) {
                    /* View of input: [W, H, C_in_g, N] starting at channel g*C_in_g */
                    size_t off_a = g * C_in_g * a->nb[2];
                    struct ggml_tensor *a_g = ggml_view_4d(c->ctx, a,
                        a->ne[0], a->ne[1], C_in_g, a->ne[3],
                        a->nb[1], a->nb[2], a->nb[3], off_a);

                    /* View of kernel: [KW, KH, C_in_g, C_out_g] starting at filter g*C_out_g */
                    size_t off_b = g * C_out_g * b->nb[3];
                    struct ggml_tensor *b_g = ggml_view_4d(c->ctx, b,
                        b->ne[0], b->ne[1], b->ne[2], C_out_g,
                        b->nb[1], b->nb[2], b->nb[3], off_b);

                    group_outs[g] = ggml_conv_2d(c->ctx, b_g, a_g,
                        (int)strides[1], (int)strides[0],
                        (int)pads[1], (int)pads[0],
                        (int)dilations[1], (int)dilations[0]);
                }

                /* Concat all groups along dim2 (channel dim) */
                out = group_outs[0];
                for (int64_t g = 1; g < groups; g++) {
                    out = ggml_concat(c->ctx, out, group_outs[g], 2);
                }
            }
        }
        /* Add bias if present — reshape bias [C] to [1,1,C,1] for broadcast */
        struct ggml_tensor *bias = get_input(c, n, 2);
        if (bias) {
            /* With reversed dims, output is [W,H,C_out,N].
             * Bias is [C_out] (1D). ggml_add broadcasts: bias ne[0]=C_out
             * must match out ne[0]=W — doesn't match.
             * Need to reshape bias to [1,1,C_out,1] so it broadcasts over C dim. */
            int64_t c_out = ggml_nelements(bias);
            struct ggml_tensor *bias_4d = ggml_reshape_4d(c->ctx, bias, 1, 1, c_out, 1);
            out = ggml_add(c->ctx, out, bias_4d);
        }
    }

    /* ── ConvTranspose ──────────────────────────────────────────── */
    else if (strcmp(op, "ConvTranspose") == 0) {
        if (!a || !b) return -1;
        int64_t strides[2] = {1, 1}, pads[4] = {0}, dilations[2] = {1, 1};
        int64_t output_padding[2] = {0, 0};
        onnx_attr_ints(n, "strides", strides, 2);
        onnx_attr_ints(n, "pads", pads, 4);
        onnx_attr_ints(n, "dilations", dilations, 2);
        onnx_attr_ints(n, "output_padding", output_padding, 2);

        int ndims_kernel = ggml_n_dims(b);
        /* ggml conv ops require F16 kernel */
        struct ggml_tensor *bk = b;
        if (bk->type != GGML_TYPE_F16)
            bk = ggml_cast(c->ctx, bk, GGML_TYPE_F16);
        if (ndims_kernel <= 2) {
            /* 1D ConvTranspose */
            out = ggml_conv_transpose_1d(c->ctx, bk, a,
                                          (int)strides[0], (int)pads[0],
                                          (int)dilations[0]);
        } else {
            /* 2D ConvTranspose — ggml only supports p0, stride (no dilation/pad) */
            out = ggml_conv_transpose_2d_p0(c->ctx, bk, a, (int)strides[0]);

            /* If pads specified, crop output: pads = [top, left, bottom, right]
             * With reversed dims, output is [W_out, H_out, C, N].
             * Crop by creating a view that skips padding pixels. */
            if (pads[0] > 0 || pads[1] > 0 || pads[2] > 0 || pads[3] > 0) {
                int64_t h_out = out->ne[1] - pads[0] - pads[2];
                int64_t w_out = out->ne[0] - pads[1] - pads[3];
                if (h_out < 1) h_out = 1;
                if (w_out < 1) w_out = 1;
                size_t offset = pads[1] * out->nb[0] + pads[0] * out->nb[1];
                out = ggml_cont(c->ctx,
                    ggml_view_4d(c->ctx, out, w_out, h_out,
                                 out->ne[2], out->ne[3],
                                 out->nb[1], out->nb[2], out->nb[3],
                                 offset));
            }
        }

        /* Add bias if present */
        struct ggml_tensor *bias = get_input(c, n, 2);
        if (bias) {
            int64_t c_out = ggml_nelements(bias);
            struct ggml_tensor *bias_4d = ggml_reshape_4d(c->ctx, bias, 1, 1, c_out, 1);
            out = ggml_add(c->ctx, out, bias_4d);
        }
    }

    /* ── Reduction ──────────────────────────────────────────────── */
    else if (strcmp(op, "ReduceMean") == 0) {
        if (!a) return -1;
        /* Parse axes attribute (opset < 18) or input (opset 18+) */
        int64_t axes[ONNX_MAX_DIMS];
        int n_axes = 0;
        const onnx_attr_t *axes_attr = onnx_node_find_attr(n, "axes");
        if (axes_attr && axes_attr->n_ints > 0) {
            n_axes = axes_attr->n_ints;
            if (n_axes > ONNX_MAX_DIMS) n_axes = ONNX_MAX_DIMS;
            for (int d = 0; d < n_axes; d++) axes[d] = axes_attr->ints[d];
        }
        /* opset 18+: axes from second input */
        if (n_axes == 0 && n->n_inputs > 1) {
            n_axes = cval_get(c, n->inputs[1], axes, ONNX_MAX_DIMS);
        }
        int keepdims = onnx_attr_int(n, "keepdims", 1);
        (void)keepdims;
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);

        /* Normalize negative axes */
        for (int d = 0; d < n_axes; d++) {
            if (axes[d] < 0) axes[d] += a_nd;
        }

        if (n_axes == 0) {
            /* No axes: reduce all → scalar */
            out = ggml_mean(c->ctx, a);
        } else if (n_axes == 1) {
            /* Single axis reduction */
            int onnx_axis = (int)axes[0];
            /* ONNX axis → ggml dim: ggml_dim = a_nd - 1 - onnx_axis */
            int ggml_dim = a_nd - 1 - onnx_axis;
            if (ggml_dim < 0) ggml_dim = 0;

            if (ggml_dim == 0) {
                /* Reduce along ne[0]: sum_rows / ne[0] */
                struct ggml_tensor *s = ggml_sum_rows(c->ctx, a);
                float inv = 1.0f / (float)a->ne[0];
                out = ggml_scale(c->ctx, s, inv);
            } else {
                /* Reduce along other dim: permute to bring target dim to ne[0],
                 * sum_rows, scale, permute back */
                int perm[4] = {0, 1, 2, 3};
                perm[0] = ggml_dim; perm[ggml_dim] = 0;
                struct ggml_tensor *ap = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, a, perm[0], perm[1], perm[2], perm[3]));
                struct ggml_tensor *s = ggml_sum_rows(c->ctx, ap);
                float inv = 1.0f / (float)a->ne[ggml_dim];
                struct ggml_tensor *m = ggml_scale(c->ctx, s, inv);
                /* Permute back */
                out = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, m, perm[0], perm[1], perm[2], perm[3]));
            }
        } else {
            /* Multi-axis: fallback to full mean for now */
            out = ggml_mean(c->ctx, a);
        }
    }
    else if (strcmp(op, "ReduceSum") == 0) {
        if (!a) return -1;
        /* Parse axes (same logic as ReduceMean) */
        int64_t axes[ONNX_MAX_DIMS];
        int n_axes = 0;
        const onnx_attr_t *axes_attr = onnx_node_find_attr(n, "axes");
        if (axes_attr && axes_attr->n_ints > 0) {
            n_axes = axes_attr->n_ints;
            if (n_axes > ONNX_MAX_DIMS) n_axes = ONNX_MAX_DIMS;
            for (int d = 0; d < n_axes; d++) axes[d] = axes_attr->ints[d];
        }
        if (n_axes == 0 && n->n_inputs > 1) {
            n_axes = cval_get(c, n->inputs[1], axes, ONNX_MAX_DIMS);
        }
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);
        for (int d = 0; d < n_axes; d++) {
            if (axes[d] < 0) axes[d] += a_nd;
        }

        if (n_axes == 0) {
            out = ggml_sum(c->ctx, a);
        } else if (n_axes == 1) {
            int onnx_axis = (int)axes[0];
            int ggml_dim = a_nd - 1 - onnx_axis;
            if (ggml_dim < 0) ggml_dim = 0;

            if (ggml_dim == 0) {
                out = ggml_sum_rows(c->ctx, a);
            } else {
                int perm[4] = {0, 1, 2, 3};
                perm[0] = ggml_dim; perm[ggml_dim] = 0;
                struct ggml_tensor *ap = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, a, perm[0], perm[1], perm[2], perm[3]));
                struct ggml_tensor *s = ggml_sum_rows(c->ctx, ap);
                out = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, s, perm[0], perm[1], perm[2], perm[3]));
            }
        } else {
            out = ggml_sum(c->ctx, a);
        }
    }

    /* ── Identity / Dropout (inference mode) ────────────────────── */
    else if (strcmp(op, "Identity") == 0 || strcmp(op, "Dropout") == 0) {
        if (!a) return -1;
        out = a; /* pass-through */
    }

    /* ── Constant ──────────────────────────────────────────────── */
    else if (strcmp(op, "Constant") == 0) {
        /* Create tensor from attribute "value" (TensorProto) */
        const onnx_attr_t *val_attr = onnx_node_find_attr(n, "value");
        if (val_attr && val_attr->tensor) {
            onnx_initializer_t *t_init = val_attr->tensor;
            enum ggml_type type = onnx_dtype_to_ggml(t_init->data_type);

            int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
            int ndims = t_init->n_dims;
            if (ndims > GGML_MAX_DIMS) ndims = GGML_MAX_DIMS;
            if (ndims == 0) {
                /* Scalar constant: single element */
                ndims = 1;
                ne[0] = 1;
            } else {
                for (int d = 0; d < ndims; d++)
                    ne[d] = t_init->dims[ndims - 1 - d];
            }

            /* FP16 promotion for large Constant tensors (ndims >= 3 only) */
            {
                int onnx_nd = t_init->n_dims > 0 ? t_init->n_dims : 1;
                int64_t n_elem = ne_product(ne, ndims);
                if (c->model_dtype == GGML_TYPE_F16 &&
                    type == GGML_TYPE_F32 &&
                    onnx_nd >= 3 &&
                    n_elem >= ONNX_FP16_MIN_ELEMENTS) {
                    type = GGML_TYPE_F16;
                }
            }

            /* Constant tensors go into ctx_weight for dedicated buffer */
            {
                struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
                out = onnx_new_tensor_nd(wctx, type, ne, ndims);
            }
            if (out) {
                ggml_set_input(out);
                /* Data will be loaded into weight_buf during build */
                ggml_set_name(out, n->outputs[0]);
                tmap_put_nd(c, n->outputs[0], out, t_init->n_dims > 0 ? t_init->n_dims : 1);
                /* Stash init pointer for load_weights to find later */
                strncpy(t_init->name, n->outputs[0], ONNX_MAX_NAME - 1);
                t_init->name[ONNX_MAX_NAME - 1] = '\0';
                /* Register compile-time values for int64 constants (shape tensors) */
                if (t_init->data_type == ONNX_DTYPE_INT64) {
                    int64_t vals[ONNX_MAX_DIMS];
                    int nv = (int)ggml_nelements(out);
                    if (nv > ONNX_MAX_DIMS) nv = ONNX_MAX_DIMS;
                    const void *src = t_init->raw_data ? t_init->raw_data : t_init->decoded_data;
                    if (src) {
                        memcpy(vals, src, nv * sizeof(int64_t));
                        cval_put(c, n->outputs[0], vals, nv);
                    }
                }
            }
            return 0; /* already registered output */
        }
        /* value_float / value_int — scalar constants */
        float vf = onnx_attr_float(n, "value_float", 0.0f);
        int64_t vi = onnx_attr_int(n, "value_int", 0);
        const onnx_attr_t *vf_attr = onnx_node_find_attr(n, "value_float");
        if (vf_attr) {
            out = make_scalar(c, vf);
            /* cval for scalar float (cast to int64) */
            int64_t cv = (int64_t)vf;
            cval_put(c, n->outputs[0], &cv, 1);
        } else {
            out = make_scalar(c, (float)vi);
            /* cval for scalar int */
            cval_put(c, n->outputs[0], &vi, 1);
        }
    }

    /* ── Cast (type conversion) ──────────────────────────────────── */
    else if (strcmp(op, "Cast") == 0) {
        if (!a) return -1;
        /* ONNX "to" attribute: 1=F32, 6=I32, 7=I64, 10=F16, 16=BF16 */
        int64_t to = onnx_attr_int(n, "to", 1);
        if ((to == 6 || to == 7) && a->type == GGML_TYPE_F32) {
            /* F32 → I32: needed for index tensors (Gather, etc.) */
            out = ggml_cast(c->ctx, a, GGML_TYPE_I32);
        } else if (to == 1 && a->type == GGML_TYPE_I32) {
            /* I32 → F32 */
            out = ggml_cast(c->ctx, a, GGML_TYPE_F32);
        } else {
            /* Same type or unsupported — pass-through */
            out = a;
        }
        /* Propagate cval through Cast (values preserved as int64) */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0)
                cval_put(c, n->outputs[0], cv, ncv);
        }
    }

    /* ── Shape ──────────────────────────────────────────────────── */
    else if (strcmp(op, "Shape") == 0) {
        if (!a) return -1;
        /* Output is a 1D int64 tensor with the ONNX shape of the input.
         * ggml ne is reversed from ONNX, so un-reverse.
         * Use max(tmap_ndims, ggml_n_dims) — tmap_ndims preserves leading-1
         * dims for graph inputs (e.g. [1,128] → 2D, not 1D), while
         * ggml_n_dims is authoritative for intermediates that grew dims. */
        int nd = tmap_get_ndims(c, n->inputs[0]);
        int gnd = (int)ggml_n_dims(a);
        if (nd < gnd) nd = gnd;
        if (nd <= 0) {
            nd = 4;
            while (nd > 1 && a->ne[nd-1] == 1) nd--;
        }
        {
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            out = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, nd);
        }
        if (out) {
            ggml_set_input(out);
            /* Store ONNX dims — will fill data after sched alloc */
            ggml_set_name(out, n->outputs[0]);
            tmap_put(c, n->outputs[0], out);
            /* Stash dims for later loading */
            if (c->n_shape_tensors < ONNX_MAX_DEFERRED) {
                c->shape_tensors_ne[c->n_shape_tensors][0] = nd;
                for (int d = 0; d < nd; d++)
                    c->shape_tensors_ne[c->n_shape_tensors][d+1] = a->ne[nd - 1 - d];
                c->shape_tensor_ptrs[c->n_shape_tensors] = out;
                c->n_shape_tensors++;
            }
            /* Register compile-time values: ONNX dims of input tensor */
            int64_t shape_vals[ONNX_MAX_DIMS];
            for (int d = 0; d < nd; d++)
                shape_vals[d] = a->ne[nd - 1 - d]; /* un-reverse to ONNX order */
            cval_put(c, n->outputs[0], shape_vals, nd);
        }
        return 0; /* already registered */
    }

    /* ── Pow ────────────────────────────────────────────────────── */
    else if (strcmp(op, "Pow") == 0) {
        if (!a || !b) return -1;
        /* Common case: b is a scalar constant (e.g. x^2, x^0.5) */
        /* For now, use log-exp: a^b = exp(b * log(a))
         * Works for positive a; good enough for normalization patterns. */
        struct ggml_tensor *log_a = ggml_log(c->ctx, a);
        struct ggml_tensor *prod;
        if (ggml_nelements(b) < ggml_nelements(a))
            prod = ggml_mul(c->ctx, log_a, b);
        else
            prod = ggml_mul(c->ctx, b, log_a);
        out = ggml_exp(c->ctx, prod);
    }

    /* ── Erf (error function) ──────────────────────────────────── */
    else if (strcmp(op, "Erf") == 0) {
        if (!a) return -1;
        /* Approximate erf using: erf(x) ≈ tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
         * This is the standard fast approximation. */
        struct ggml_tensor *x3 = ggml_mul(c->ctx, a, ggml_mul(c->ctx, a, a));
        struct ggml_tensor *inner = ggml_add(c->ctx, a,
            ggml_scale(c->ctx, x3, 0.044715f));
        out = ggml_tanh(c->ctx, ggml_scale(c->ctx, inner, 0.7978845608f)); /* sqrt(2/pi) */
    }

    /* ── Sin / Cos ──────────────────────────────────────────────── */
    else if (strcmp(op, "Sin") == 0) {
        if (!a) return -1;
        out = ggml_sin(c->ctx, a);
    }
    else if (strcmp(op, "Cos") == 0) {
        if (!a) return -1;
        out = ggml_cos(c->ctx, a);
    }

    /* ── Tile (repeat tensor along axes) ──────────────────────── */
    else if (strcmp(op, "Tile") == 0) {
        if (!a || !b) return -1;
        /* b = repeats tensor (int64). Read from initializer, Constant, or cval. */
        int64_t repeats[ONNX_MAX_DIMS];
        int nreps = 0;

        const onnx_initializer_t *rep_init = onnx_find_initializer(c->onnx, n->inputs[1]);
        if (!rep_init)
            rep_init = find_constant_tensor(c->onnx, n->inputs[1]);

        if (rep_init && rep_init->raw_data && rep_init->data_type == ONNX_DTYPE_INT64) {
            nreps = (int)(rep_init->raw_size / sizeof(int64_t));
            if (nreps > ONNX_MAX_DIMS) nreps = ONNX_MAX_DIMS;
            memcpy(repeats, rep_init->raw_data, nreps * sizeof(int64_t));
        }

        if (nreps == 0)
            nreps = cval_get(c, n->inputs[1], repeats, ONNX_MAX_DIMS);
        if (nreps == 0) return -1;

        /* Target shape = a->ne[d] * repeats[d] (reversed ONNX→ggml) */
        int64_t ne[GGML_MAX_DIMS] = {a->ne[0], a->ne[1], a->ne[2], a->ne[3], a->ne[4]};
        for (int d = 0; d < nreps && d < GGML_MAX_DIMS; d++) {
            int ggml_d = nreps - 1 - d; /* reverse ONNX dim → ggml dim */
            if (ggml_d < GGML_MAX_DIMS)
                ne[ggml_d] *= repeats[d];
        }
        out = ggml_repeat_5d(c->ctx, a, ne[0], ne[1], ne[2], ne[3], ne[4]);
    }

    /* ── Where (conditional select) ────────────────────────────── */
    else if (strcmp(op, "Where") == 0) {
        if (!a || !b) return -1;
        struct ggml_tensor *cond_t = get_input(c, n, 2);
        if (!cond_t) return -1;
        /* Where(condition, X, Y): output = condition ? X : Y
         * a=condition, b=X, cond_t=Y.
         * Implement as: out = condition * X_clamped + (1 - condition) * Y_clamped
         * Clamp X,Y to [-1e9, 1e9] to avoid NaN from 0 * inf (IEEE 754).
         * -1e9 is sufficient for softmax (exp(-1e9) ≈ 0). */
        struct ggml_tensor *x_clamped = ggml_clamp(c->ctx, b, -1e9f, 1e9f);
        struct ggml_tensor *y_clamped = ggml_clamp(c->ctx, cond_t, -1e9f, 1e9f);
        struct ggml_tensor *neg_cond = ggml_neg(c->ctx, a);
        struct ggml_tensor *ones = make_scalar(c, 1.0f);
        struct ggml_tensor *inv_cond = ggml_add(c->ctx, neg_cond, ones);
        struct ggml_tensor *ta_cond = a;
        struct ggml_tensor *yes_part, *no_part;
        if (ggml_nelements(ta_cond) >= ggml_nelements(x_clamped))
            yes_part = ggml_mul(c->ctx, ta_cond, x_clamped);
        else
            yes_part = ggml_mul(c->ctx, x_clamped, ta_cond);
        if (ggml_nelements(inv_cond) >= ggml_nelements(y_clamped))
            no_part = ggml_mul(c->ctx, inv_cond, y_clamped);
        else
            no_part = ggml_mul(c->ctx, y_clamped, inv_cond);
        out = ggml_add(c->ctx, yes_part, no_part);

        /* cval propagation for Where(cond, X, Y):
         * if all three inputs have known compile-time values, compute output */
        if (n->n_inputs >= 3) {
            int64_t cv_cond[ONNX_MAX_DIMS], cv_x[ONNX_MAX_DIMS], cv_y[ONNX_MAX_DIMS];
            int nc = cval_get(c, n->inputs[0], cv_cond, ONNX_MAX_DIMS);
            int nx = cval_get(c, n->inputs[1], cv_x, ONNX_MAX_DIMS);
            int ny = cval_get(c, n->inputs[2], cv_y, ONNX_MAX_DIMS);
            if (nc > 0 && nx == nc && ny == nc) {
                int64_t result[ONNX_MAX_DIMS];
                for (int j = 0; j < nc; j++)
                    result[j] = cv_cond[j] ? cv_x[j] : cv_y[j];
                cval_put(c, n->outputs[0], result, nc);
            }
        }
    }

    /* ── Equal ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Equal") == 0) {
        if (!a || !b) return -1;
        /* Equal(A, B) → 0/1 float mask.
         * step(0.5 - abs(a - b)): if a==b → step(0.5)=1, else step(neg)=0. */
        struct ggml_tensor *ta = a, *tb = b;
        onnx_broadcast_prepare(c->ctx, &ta, &tb);
        struct ggml_tensor *diff = ggml_sub(c->ctx, ta, tb);
        struct ggml_tensor *absdiff = ggml_abs(c->ctx, diff);
        struct ggml_tensor *neg_abs = ggml_neg(c->ctx, absdiff);
        struct ggml_tensor *half = make_scalar(c, 0.5f);
        struct ggml_tensor *shifted = ggml_add(c->ctx, neg_abs, half);
        out = ggml_step(c->ctx, shifted);

        /* cval propagation for Equal */
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 == na2) {
                int64_t result[ONNX_MAX_DIMS];
                for (int j = 0; j < na2; j++)
                    result[j] = (cv_a[j] == cv_b[j]) ? 1 : 0;
                cval_put(c, n->outputs[0], result, na2);
            }
        }
    }

    /* ── EyeLike ──────────────────────────────────────────────── */
    else if (strcmp(op, "EyeLike") == 0) {
        if (!a) return -1;
        /* EyeLike(input) → identity matrix of same shape.
         * input is 2D [ne0, ne1] in ggml = ONNX [rows, cols].
         * Create as deferred fill (like ConstantOfShape). */
        int64_t k = onnx_attr_int(n, "k", 0);
        {
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            out = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, a->ne[0], a->ne[1]);
        }
        ggml_set_name(out, n->outputs[0]);
        ggml_set_input(out);

        /* Store for deferred fill after alloc */
        if (c->n_eye_fills < ONNX_MAX_DEFERRED) {
            c->eye_fill_ptrs[c->n_eye_fills] = out;
            c->eye_fill_rows[c->n_eye_fills] = (int)a->ne[1]; /* rows in ggml = ONNX cols */
            c->eye_fill_cols[c->n_eye_fills] = (int)a->ne[0]; /* cols in ggml = ONNX rows */
            c->eye_fill_k[c->n_eye_fills] = (int)k;
            c->n_eye_fills++;
        }
    }

    /* ── ConstantOfShape ───────────────────────────────────────── */
    else if (strcmp(op, "ConstantOfShape") == 0) {
        if (!a) return -1;
        /* Input a is a shape tensor (int64). Read shape from initializer,
         * Constant, or compile-time cval (e.g. from Shape op). */
        int64_t shape[ONNX_MAX_DIMS];
        int ndims = 0;
        const onnx_initializer_t *si = onnx_find_initializer(c->onnx, n->inputs[0]);
        if (!si) si = find_constant_tensor(c->onnx, n->inputs[0]);
        if (si) {
            if (si->raw_data && si->data_type == ONNX_DTYPE_INT64) {
                ndims = (int)(si->raw_size / sizeof(int64_t));
                if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
                memcpy(shape, si->raw_data, ndims * sizeof(int64_t));
            }
        } else {
            /* Fallback: try compile-time values (e.g. Shape op output) */
            ndims = cval_get(c, n->inputs[0], shape, ONNX_MAX_DIMS);
        }
        if (ndims == 0) {
            fprintf(stderr, "[onnx] ConstantOfShape: cannot resolve shape from '%s'\n",
                    n->inputs[0]);
            return -1;
        }

        /* Get fill value from attribute "value" (TensorProto, usually scalar) */
        float fill_val = 0.0f;
        const onnx_attr_t *va = onnx_node_find_attr(n, "value");
        if (va && va->tensor && va->tensor->raw_data && va->tensor->raw_size > 0) {
            int vdt = va->tensor->data_type;
            if (vdt == ONNX_DTYPE_INT64 && va->tensor->raw_size >= 8) {
                int64_t iv; memcpy(&iv, va->tensor->raw_data, sizeof(int64_t));
                fill_val = (float)iv;
            } else if (vdt == ONNX_DTYPE_INT32 && va->tensor->raw_size >= 4) {
                int32_t iv; memcpy(&iv, va->tensor->raw_data, sizeof(int32_t));
                fill_val = (float)iv;
            } else if (vdt == ONNX_DTYPE_DOUBLE && va->tensor->raw_size >= 8) {
                double dv; memcpy(&dv, va->tensor->raw_data, sizeof(double));
                fill_val = (float)dv;
            } else if (va->tensor->raw_size >= 4) {
                memcpy(&fill_val, va->tensor->raw_data, sizeof(float));
            }
        }

        /* Reverse ONNX shape → ggml ne */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        if (ndims > GGML_MAX_DIMS) ndims = GGML_MAX_DIMS;
        for (int d = 0; d < ndims; d++)
            ne[d] = shape[ndims - 1 - d];

        /* Create constant tensor in ctx_weight for dedicated buffer */
        {
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            out = onnx_new_tensor_nd(wctx, GGML_TYPE_F32, ne, ndims);
        }
        if (out) {
            ggml_set_input(out);
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, ndims);
            /* Store fill value for loading after sched alloc */
            if (c->n_const_fills < ONNX_MAX_DEFERRED) {
                c->const_fill_vals[c->n_const_fills] = fill_val;
                c->const_fill_ptrs[c->n_const_fills] = out;
                c->n_const_fills++;
            }
            /* cval propagation: all elements = fill_val (cast to int64) */
            int64_t total_elems = ne[0];
            for (int d = 1; d < ndims; d++) total_elems *= ne[d];
            if (total_elems <= ONNX_MAX_DIMS) {
                int64_t cvals[ONNX_MAX_DIMS];
                for (int j = 0; j < (int)total_elems; j++)
                    cvals[j] = (int64_t)fill_val;
                cval_put(c, n->outputs[0], cvals, (int)total_elems);
            }
        }
        return 0; /* already registered */
    }

    /* ── Pad ───────────────────────────────────────────────────── */
    else if (strcmp(op, "Pad") == 0) {
        if (!a) return -1;
        /* Read pads from input 1 (opset 11+) or attribute */
        int64_t pads_arr[8] = {0};
        int n_pads = 0;
        if (n->n_inputs > 1 && n->inputs[1][0] != '\0') {
            const onnx_initializer_t *pi = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (!pi) pi = find_constant_tensor(c->onnx, n->inputs[1]);
            if (pi && pi->raw_data && pi->data_type == ONNX_DTYPE_INT64) {
                n_pads = (int)(pi->raw_size / sizeof(int64_t));
                if (n_pads > 8) n_pads = 8;
                memcpy(pads_arr, pi->raw_data, n_pads * sizeof(int64_t));
            }
            /* Fallback: try cval (for dynamic Slice→Transpose→Cast chains) */
            if (n_pads == 0) {
                n_pads = cval_get(c, n->inputs[1], pads_arr, 8);
            }
        }
        /* pads format: [begin_0, begin_1, ..., end_0, end_1, ...]
         * For 4D: [b0,b1,b2,b3, e0,e1,e2,e3]. Only spatial padding supported.
         * ggml_pad adds padding at the end only. For symmetric:
         * ONNX dims [N,C,H,W] → ggml [W,H,C,N]. Pad H,W only. */
        int nd = n_pads / 2;
        if (nd > 4) nd = 4;

        /* Map ONNX pad dims to ggml: pad ggml_d = nd-1-onnx_d */
        int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        for (int d = 0; d < nd; d++) {
            int ggml_d = nd - 1 - d;
            int64_t begin = pads_arr[d];
            int64_t end = pads_arr[nd + d];
            int total = (int)(begin + end);
            switch (ggml_d) {
                case 0: p0 = total; break;
                case 1: p1 = total; break;
                case 2: p2 = total; break;
                case 3: p3 = total; break;
            }
        }
        out = ggml_pad(c->ctx, a, p0, p1, p2, p3);
    }

    /* ── Quantized ops (QLinear family) ──────────────────────────── */
    /* DequantizeLinear(x, x_scale, x_zero_point) → y = (x - zp) * scale
     * All inputs stored as F32 (int8 converted at load time). */
    else if (strcmp(op, "DequantizeLinear") == 0) {
        if (!a) return -1;
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *zp    = get_input(c, n, 2);
        if (!scale) return -1;
        out = a;
        if (zp) {
            /* Broadcast zp to match x shape */
            struct ggml_tensor *tx = out, *tzp = zp;
            onnx_broadcast_prepare(c->ctx, &tx, &tzp);
            out = ggml_sub(c->ctx, tx, tzp);
        }
        /* Broadcast scale to match shape */
        {
            struct ggml_tensor *tx = out, *ts = scale;
            onnx_broadcast_prepare(c->ctx, &tx, &ts);
            out = ggml_mul(c->ctx, tx, ts);
        }
    }

    /* QuantizeLinear(x, y_scale, y_zero_point) → y = round(x / scale) + zp
     * Output stored as F32 (representing quantized values). */
    else if (strcmp(op, "QuantizeLinear") == 0) {
        if (!a) return -1;
        struct ggml_tensor *scale = get_input(c, n, 1);
        struct ggml_tensor *zp    = get_input(c, n, 2);
        if (!scale) return -1;
        /* x / scale — broadcast scale */
        {
            struct ggml_tensor *tx = a, *ts = scale;
            onnx_broadcast_prepare(c->ctx, &tx, &ts);
            /* Division via reciprocal: x * (1/scale) would need custom scalar.
             * Instead: use ggml_div if available, or approximate.
             * ggml doesn't have div — use scale(x, 1/s) for scalar, or element-wise mul by reciprocal. */
            /* For per-tensor quant (scalar scale), use ggml_scale */
            if (ggml_nelements(scale) == 1) {
                /* Will be filled with actual value at runtime — we need the reciprocal.
                 * Create 1/scale via a divide workaround: sub(0, scale) then ... no.
                 * Simplest: trust that scale is a constant, extract at graph-build.
                 * But scale is a weight tensor, value not available yet.
                 * Solution: build x * (1.0) placeholder — will be correct after alloc. */
                /* Actually, ggml_div exists for element-wise division */
                out = ggml_div(c->ctx, tx, ts);
            } else {
                out = ggml_div(c->ctx, tx, ts);
            }
        }
        /* round — ggml doesn't have round op, approximate identity (values will be ~integer) */
        /* For QLinear pipeline correctness, skip rounding — output feeds back into DequantizeLinear */
        if (zp) {
            struct ggml_tensor *tx = out, *tzp = zp;
            onnx_broadcast_prepare(c->ctx, &tx, &tzp);
            out = ggml_add(c->ctx, tx, tzp);
        }
    }

    /* QLinearConv(x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, [bias])
     * → dequant x, dequant w, Conv, requant output */
    else if (strcmp(op, "QLinearConv") == 0) {
        /* Input layout: 0=x, 1=x_scale, 2=x_zp, 3=w, 4=w_scale, 5=w_zp, 6=y_scale, 7=y_zp, 8=bias */
        struct ggml_tensor *x       = get_input(c, n, 0);
        struct ggml_tensor *x_scale = get_input(c, n, 1);
        struct ggml_tensor *x_zp    = get_input(c, n, 2);
        struct ggml_tensor *w       = get_input(c, n, 3);
        struct ggml_tensor *w_scale = get_input(c, n, 4);
        struct ggml_tensor *w_zp    = get_input(c, n, 5);
        struct ggml_tensor *y_scale = get_input(c, n, 6);
        struct ggml_tensor *y_zp    = get_input(c, n, 7);
        struct ggml_tensor *bias    = get_input(c, n, 8);
        if (!x || !x_scale || !w || !w_scale || !y_scale) return -1;

        /* Dequantize x: (x - x_zp) * x_scale */
        struct ggml_tensor *dx = x;
        if (x_zp) {
            struct ggml_tensor *tx = dx, *tzp = x_zp;
            onnx_broadcast_prepare(c->ctx, &tx, &tzp);
            dx = ggml_sub(c->ctx, tx, tzp);
        }
        {
            struct ggml_tensor *tx = dx, *ts = x_scale;
            onnx_broadcast_prepare(c->ctx, &tx, &ts);
            dx = ggml_mul(c->ctx, tx, ts);
        }

        /* Dequantize w: (w - w_zp) * w_scale */
        struct ggml_tensor *dw = w;
        if (w_zp) {
            struct ggml_tensor *tw = dw, *tzp = w_zp;
            onnx_broadcast_prepare(c->ctx, &tw, &tzp);
            dw = ggml_sub(c->ctx, tw, tzp);
        }
        {
            struct ggml_tensor *tw = dw, *ts = w_scale;
            onnx_broadcast_prepare(c->ctx, &tw, &ts);
            dw = ggml_mul(c->ctx, tw, ts);
        }

        /* Conv with dequantized inputs — reuse Conv logic */
        int64_t strides[2] = {1, 1}, pads[4] = {0}, dilations[2] = {1, 1};
        onnx_attr_ints(n, "strides", strides, 2);
        onnx_attr_ints(n, "pads", pads, 4);
        onnx_attr_ints(n, "dilations", dilations, 2);
        char auto_pad[32] = "";
        onnx_attr_str(n, "auto_pad", auto_pad, sizeof(auto_pad));
        if (strcmp(auto_pad, "SAME_UPPER") == 0 || strcmp(auto_pad, "SAME_LOWER") == 0) {
            for (int d = 0; d < 2; d++) {
                int64_t in_d = (d == 0) ? dx->ne[1] : dx->ne[0];
                int64_t k_d = (d == 0) ? dw->ne[1] : dw->ne[0];
                int64_t out_d = (in_d + strides[d] - 1) / strides[d];
                int64_t eff_k = (k_d - 1) * dilations[d] + 1;
                int64_t total_pad = (out_d - 1) * strides[d] + eff_k - in_d;
                if (total_pad < 0) total_pad = 0;
                pads[d]     = (strcmp(auto_pad, "SAME_LOWER") == 0) ?
                              (total_pad + 1) / 2 : total_pad / 2;
                pads[d + 2] = total_pad - pads[d];
            }
        }
        int64_t groups = onnx_attr_int(n, "group", 1);
        int ndims_kernel = ggml_n_dims(dw);

        if (ndims_kernel <= 2) {
            /* ggml_conv_1d requires F16 kernel */
            struct ggml_tensor *dwk = dw;
            if (dwk->type != GGML_TYPE_F16)
                dwk = ggml_cast(c->ctx, dwk, GGML_TYPE_F16);
            out = ggml_conv_1d(c->ctx, dwk, dx,
                               (int)strides[0], (int)pads[0], (int)dilations[0]);
        } else if (groups == 1) {
            out = ggml_conv_2d(c->ctx, dw, dx,
                               (int)strides[1], (int)strides[0],
                               (int)pads[1], (int)pads[0],
                               (int)dilations[1], (int)dilations[0]);
        } else {
            int64_t C_in  = dx->ne[2];
            int64_t C_out = dw->ne[3];
            int64_t C_in_g  = C_in / groups;
            int64_t C_out_g = C_out / groups;
            if (groups == C_in && C_in_g == 1) {
                out = ggml_conv_2d_dw_direct(c->ctx, dw, dx,
                                      (int)strides[1], (int)strides[0],
                                      (int)pads[1], (int)pads[0],
                                      (int)dilations[1], (int)dilations[0]);
            } else {
                struct ggml_tensor *group_outs[512];
                if (groups > 512) { fprintf(stderr, "[onnx] QLinearConv groups=%lld > 512\n", (long long)groups); return -1; }
                for (int64_t g = 0; g < groups; g++) {
                    size_t off_a = g * C_in_g * dx->nb[2];
                    struct ggml_tensor *a_g = ggml_view_4d(c->ctx, dx,
                        dx->ne[0], dx->ne[1], C_in_g, dx->ne[3],
                        dx->nb[1], dx->nb[2], dx->nb[3], off_a);
                    size_t off_b = g * C_out_g * dw->nb[3];
                    struct ggml_tensor *b_g = ggml_view_4d(c->ctx, dw,
                        dw->ne[0], dw->ne[1], dw->ne[2], C_out_g,
                        dw->nb[1], dw->nb[2], dw->nb[3], off_b);
                    group_outs[g] = ggml_conv_2d(c->ctx, b_g, a_g,
                        (int)strides[1], (int)strides[0],
                        (int)pads[1], (int)pads[0],
                        (int)dilations[1], (int)dilations[0]);
                }
                out = group_outs[0];
                for (int64_t g = 1; g < groups; g++)
                    out = ggml_concat(c->ctx, out, group_outs[g], 2);
            }
        }
        /* Add bias (float, not quantized) */
        if (bias) {
            int64_t c_out = ggml_nelements(bias);
            struct ggml_tensor *bias_4d = ggml_reshape_4d(c->ctx, bias, 1, 1, c_out, 1);
            out = ggml_add(c->ctx, out, bias_4d);
        }

        /* Requantize output: round(conv_out / y_scale) + y_zp */
        if (ggml_nelements(y_scale) > 0) {
            struct ggml_tensor *tx = out, *ts = y_scale;
            onnx_broadcast_prepare(c->ctx, &tx, &ts);
            out = ggml_div(c->ctx, tx, ts);
        }
        if (y_zp) {
            struct ggml_tensor *tx = out, *tzp = y_zp;
            onnx_broadcast_prepare(c->ctx, &tx, &tzp);
            out = ggml_add(c->ctx, tx, tzp);
        }
    }

    /* QLinearAdd(a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp) */
    else if (strcmp(op, "QLinearAdd") == 0) {
        struct ggml_tensor *xa       = get_input(c, n, 0);
        struct ggml_tensor *a_scale  = get_input(c, n, 1);
        struct ggml_tensor *a_zp     = get_input(c, n, 2);
        struct ggml_tensor *xb       = get_input(c, n, 3);
        struct ggml_tensor *b_scale  = get_input(c, n, 4);
        struct ggml_tensor *b_zp     = get_input(c, n, 5);
        struct ggml_tensor *y_scale  = get_input(c, n, 6);
        struct ggml_tensor *y_zp     = get_input(c, n, 7);
        if (!xa || !a_scale || !xb || !b_scale || !y_scale) return -1;

        /* Dequant a */
        struct ggml_tensor *da = xa;
        if (a_zp) { struct ggml_tensor *t1=da, *t2=a_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); da=ggml_sub(c->ctx,t1,t2); }
        { struct ggml_tensor *t1=da, *t2=a_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); da=ggml_mul(c->ctx,t1,t2); }
        /* Dequant b */
        struct ggml_tensor *db = xb;
        if (b_zp) { struct ggml_tensor *t1=db, *t2=b_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); db=ggml_sub(c->ctx,t1,t2); }
        { struct ggml_tensor *t1=db, *t2=b_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); db=ggml_mul(c->ctx,t1,t2); }
        /* Add */
        { struct ggml_tensor *t1=da, *t2=db; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
        /* Requant */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
    }

    /* QLinearMatMul(a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp) */
    else if (strcmp(op, "QLinearMatMul") == 0) {
        struct ggml_tensor *xa       = get_input(c, n, 0);
        struct ggml_tensor *a_scale  = get_input(c, n, 1);
        struct ggml_tensor *a_zp     = get_input(c, n, 2);
        struct ggml_tensor *xb       = get_input(c, n, 3);
        struct ggml_tensor *b_scale  = get_input(c, n, 4);
        struct ggml_tensor *b_zp     = get_input(c, n, 5);
        struct ggml_tensor *y_scale  = get_input(c, n, 6);
        struct ggml_tensor *y_zp     = get_input(c, n, 7);
        if (!xa || !a_scale || !xb || !b_scale || !y_scale) return -1;

        /* Dequant a */
        struct ggml_tensor *da = xa;
        if (a_zp) { struct ggml_tensor *t1=da, *t2=a_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); da=ggml_sub(c->ctx,t1,t2); }
        { struct ggml_tensor *t1=da, *t2=a_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); da=ggml_mul(c->ctx,t1,t2); }
        /* Dequant b */
        struct ggml_tensor *db = xb;
        if (b_zp) { struct ggml_tensor *t1=db, *t2=b_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); db=ggml_sub(c->ctx,t1,t2); }
        { struct ggml_tensor *t1=db, *t2=b_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); db=ggml_mul(c->ctx,t1,t2); }
        /* MatMul — same as regular MatMul: transpose B then mul_mat */
        struct ggml_tensor *bt = ggml_cont(c->ctx, ggml_transpose(c->ctx, db));
        out = ggml_mul_mat(c->ctx, bt, da);
        /* Requant */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
    }

    /* QLinearSigmoid(x, x_scale, x_zp, y_scale, y_zp) */
    else if (strcmp(op, "QLinearSigmoid") == 0) {
        struct ggml_tensor *x       = get_input(c, n, 0);
        struct ggml_tensor *x_scale = get_input(c, n, 1);
        struct ggml_tensor *x_zp    = get_input(c, n, 2);
        struct ggml_tensor *y_scale = get_input(c, n, 3);
        struct ggml_tensor *y_zp    = get_input(c, n, 4);
        if (!x || !x_scale || !y_scale) return -1;

        /* Dequant */
        struct ggml_tensor *dx = x;
        if (x_zp) { struct ggml_tensor *t1=dx, *t2=x_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); dx=ggml_sub(c->ctx,t1,t2); }
        { struct ggml_tensor *t1=dx, *t2=x_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); dx=ggml_mul(c->ctx,t1,t2); }
        /* Sigmoid */
        out = ggml_sigmoid(c->ctx, dx);
        /* Requant */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
    }

    /* QLinearConcat — Microsoft extension: concat quantized tensors
     * Inputs: (y_scale, y_zp, x1, x1_scale, x1_zp, x2, x2_scale, x2_zp, ...) */
    else if (strcmp(op, "QLinearConcat") == 0) {
        struct ggml_tensor *y_scale = get_input(c, n, 0);
        struct ggml_tensor *y_zp    = get_input(c, n, 1);
        if (!y_scale) return -1;

        int64_t axis = onnx_attr_int(n, "axis", 0);
        /* Each subsequent group of 3: (tensor, scale, zp) */
        int n_tensors = (n->n_inputs - 2) / 3;
        if (n_tensors <= 0) return -1;

        /* Determine ggml concat dim from ONNX axis (reversed) */
        /* First dequant all inputs */
        struct ggml_tensor *dequants[64];
        if (n_tensors > 64) n_tensors = 64;
        int ndims_first = 0;
        for (int i = 0; i < n_tensors; i++) {
            int base = 2 + i * 3;
            struct ggml_tensor *xi   = get_input(c, n, base);
            struct ggml_tensor *si   = get_input(c, n, base + 1);
            struct ggml_tensor *zpi  = get_input(c, n, base + 2);
            if (!xi || !si) return -1;
            struct ggml_tensor *di = xi;
            if (zpi) { struct ggml_tensor *t1=di, *t2=zpi; onnx_broadcast_prepare(c->ctx,&t1,&t2); di=ggml_sub(c->ctx,t1,t2); }
            { struct ggml_tensor *t1=di, *t2=si; onnx_broadcast_prepare(c->ctx,&t1,&t2); di=ggml_mul(c->ctx,t1,t2); }
            dequants[i] = di;
            if (i == 0) ndims_first = ggml_n_dims(xi);
        }

        /* Convert ONNX axis to ggml dim (reversed) */
        int onnx_ndims = ndims_first > 0 ? ndims_first : 4;
        if (axis < 0) axis += onnx_ndims;
        int ggml_dim = onnx_ndims - 1 - (int)axis;
        if (ggml_dim < 0) ggml_dim = 0;

        out = dequants[0];
        for (int i = 1; i < n_tensors; i++)
            out = ggml_concat(c->ctx, out, dequants[i], ggml_dim);

        /* Requant */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
    }

    /* ── NonZero ────────────────────────────────────────────────── */
    else if (strcmp(op, "NonZero") == 0) {
        if (!a) return -1;
        /* NonZero returns indices of non-zero elements.
         * Output shape: ONNX [n_dims_input, nnz], ggml [nnz, n_dims_input].
         * For transformer position embeddings: input is ConstantOfShape(value=1),
         * so nnz = total_elements and result = arange(0, N).
         *
         * At graph-build time we need to know nnz to allocate output tensor.
         * If input is from ConstantOfShape with non-zero fill, nnz = total_elements.
         * Otherwise, defer to runtime fill. */
        int input_ndims = tmap_get_ndims(c, n->inputs[0]);
        if (input_ndims < 1) input_ndims = (int)ggml_n_dims(a);

        int64_t total_elems = (int64_t)ggml_nelements(a);

        /* Assume all elements are non-zero (common case: ConstantOfShape with value=1).
         * For truly sparse inputs this would need runtime sizing. */
        int64_t nnz = total_elems;

        /* Output: ONNX [input_ndims, nnz] → ggml [nnz, input_ndims] */
        {
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            out = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, nnz, (int64_t)input_ndims);
        }
        if (out) {
            ggml_set_input(out);
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, 2); /* ONNX 2D output */

            /* Register for deferred fill after sched alloc */
            if (c->n_nonzero_fills < ONNX_MAX_DEFERRED) {
                c->nonzero_fill_src[c->n_nonzero_fills] = a;
                c->nonzero_fill_dst[c->n_nonzero_fills] = out;
                c->nonzero_fill_ndims[c->n_nonzero_fills] = input_ndims;
                c->n_nonzero_fills++;
            }

            /* cval propagation for 1D all-nonzero case: [0, 1, 2, ..., N-1] */
            if (input_ndims == 1 && nnz <= ONNX_MAX_DIMS) {
                int64_t cvals[ONNX_MAX_DIMS];
                for (int j = 0; j < (int)nnz; j++)
                    cvals[j] = (int64_t)j;
                cval_put(c, n->outputs[0], cvals, (int)nnz);
            }
        }
        return 0; /* already registered */
    }

    /* ── RoiAlign ───────────────────────────────────────────────── */
    else if (strcmp(op, "RoiAlign") == 0) {
        /* Inputs: X [N,C,H,W], rois [num_rois,4], batch_indices [num_rois] */
        struct ggml_tensor *X    = get_input(c, n, 0);
        struct ggml_tensor *rois = get_input(c, n, 1);
        struct ggml_tensor *bi   = get_input(c, n, 2);
        if (!X || !rois) return -1;

        int oh = (int)onnx_attr_int(n, "output_height", 1);
        int ow = (int)onnx_attr_int(n, "output_width", 1);
        int sr = (int)onnx_attr_int(n, "sampling_ratio", 0);
        float ss = onnx_attr_float(n, "spatial_scale", 1.0f);
        char mode_str[16] = "avg";
        onnx_attr_str(n, "mode", mode_str, sizeof(mode_str));
        int mode = (strcmp(mode_str, "max") == 0) ? 1 : 0;

        /* X is ggml [W, H, C, N] */
        int C_feat = (int)X->ne[2];
        int num_rois_val = (int)rois->ne[1]; /* rois ggml [4, num_rois] */

        /* Allocate params (must outlive graph) */
        c->roi_align_params = (roi_align_params_t *)realloc(
            c->roi_align_params,
            (size_t)(c->n_roi_aligns + 1) * sizeof(roi_align_params_t));
        roi_align_params_t *p = &c->roi_align_params[c->n_roi_aligns];
        p->output_height  = oh;
        p->output_width   = ow;
        p->sampling_ratio = sr;
        p->spatial_scale  = ss;
        p->mode           = mode;

        /* Output: ggml [ow, oh, C, num_rois] */
        struct ggml_tensor *dummy = ggml_new_tensor_4d(c->ctx, GGML_TYPE_F32,
                                                        ow, oh, C_feat, num_rois_val);
        if (!bi) {
            /* Create zero batch_indices if missing */
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            bi = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, num_rois_val);
            ggml_set_input(bi);
            /* Will be zero-filled by default */
        }

        p->X = X; /* callback reads feature map from params */
        out = ggml_map_custom3(c->ctx, dummy, rois, bi,
                               roi_align_cpu, 1, p);
        /* Add X as dependency so scheduler keeps its buffer alive */
        out->src[3] = X;
        c->n_roi_aligns++;
        out_nd = 4;
    }

    /* ── NonMaxSuppression ─────────────────────────────────────── */
    else if (strcmp(op, "NonMaxSuppression") == 0) {
        /* Inputs: boxes [N,num_boxes,4], scores [N,num_classes,num_boxes],
         *         max_output_boxes_per_class (scalar), iou_threshold (scalar),
         *         score_threshold (scalar) */
        struct ggml_tensor *boxes  = get_input(c, n, 0);
        struct ggml_tensor *scores = get_input(c, n, 1);
        if (!boxes || !scores) return -1;

        int cpb = (int)onnx_attr_int(n, "center_point_box", 0);

        /* Get scalar params from raw initializer data */
        int max_boxes_val = 0;
        float iou_thresh_val = 0.0f;
        float score_thresh_val = 0.0f;

        /* max_output_boxes_per_class (INT64 scalar) */
        if (n->n_inputs > 2 && n->inputs[2][0]) {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[2]);
            if (mi && mi->raw_data && mi->raw_size >= 8 &&
                mi->data_type == ONNX_DTYPE_INT64) {
                int64_t v; memcpy(&v, mi->raw_data, sizeof(int64_t));
                max_boxes_val = (int)v;
            } else {
                int64_t cv[1] = {0};
                if (cval_get(c, n->inputs[2], cv, 1))
                    max_boxes_val = (int)cv[0];
            }
        }
        /* iou_threshold (FLOAT scalar) */
        if (n->n_inputs > 3 && n->inputs[3][0]) {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[3]);
            if (mi && mi->raw_data && mi->raw_size >= 4 &&
                mi->data_type == ONNX_DTYPE_FLOAT) {
                memcpy(&iou_thresh_val, mi->raw_data, sizeof(float));
            }
        }
        /* score_threshold (FLOAT scalar) */
        if (n->n_inputs > 4 && n->inputs[4][0]) {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[4]);
            if (mi && mi->raw_data && mi->raw_size >= 4 &&
                mi->data_type == ONNX_DTYPE_FLOAT) {
                memcpy(&score_thresh_val, mi->raw_data, sizeof(float));
            }
        }

        /* Allocate NMS params */
        c->nms_params = (nms_params_t *)realloc(
            c->nms_params,
            (size_t)(c->n_nms_ops + 1) * sizeof(nms_params_t));
        nms_params_t *p = &c->nms_params[c->n_nms_ops];
        p->center_point_box = cpb;
        p->scores = scores;

        /* boxes ggml [4, num_boxes, N], scores ggml [num_boxes, num_classes, N] */
        int num_boxes_val = (int)boxes->ne[1];
        int N_batch = (int)boxes->ne[2];
        int num_classes_val = (int)scores->ne[1];

        /* Max possible output: N * num_classes * max_boxes (or num_boxes if max=0) */
        int mb = max_boxes_val > 0 ? max_boxes_val : num_boxes_val;
        int max_selected = N_batch * num_classes_val * mb;
        if (max_selected > num_boxes_val * N_batch)
            max_selected = num_boxes_val * N_batch;

        /* Create params tensor [3] to pass scalar params at runtime */
        struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
        struct ggml_tensor *params_t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 3);
        ggml_set_input(params_t);
        ggml_set_name(params_t, "nms_params");

        /* Register deferred fill for params tensor */
        if (c->n_nms_deferred < ONNX_MAX_DEFERRED) {
            c->nms_param_tensors[c->n_nms_deferred] = params_t;
            c->nms_max_boxes[c->n_nms_deferred] = max_boxes_val;
            c->nms_iou_thresh[c->n_nms_deferred] = iou_thresh_val;
            c->nms_score_thresh[c->n_nms_deferred] = score_thresh_val;
            c->n_nms_deferred++;
        }

        /* Output: ggml [3, max_selected] */
        struct ggml_tensor *nms_dummy = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32,
                                                            3, max_selected);
        out = ggml_map_custom3(c->ctx, nms_dummy, boxes, params_t,
                               nms_cpu, 1, p);
        /* Add boxes and scores as dependencies */
        out->src[3] = boxes;
        out->src[4] = scores;
        c->n_nms_ops++;
        out_nd = 2;
    }

    /* ── Unknown op ─────────────────────────────────────────────── */
    else {
        fprintf(stderr, "onnx_ggml: unsupported op '%s'\n", op);
        return -1;
    }

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

        /* Check if any output name contains /pos_embed/ */
        for (int o = 0; o < nd->n_outputs; o++) {
            if (strstr(nd->outputs[o], "/pos_embed/")) {
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

                    /* Verify W_h, W_w are F32 */
                    if (wh_init->data_type != 1 /* ONNX_DTYPE_FLOAT */ ||
                        ww_init->data_type != 1) {
                        fprintf(stderr, "[onnx] pos_embed block %d: W_h/W_w not F32 "
                                "(types: %d, %d) — skipping\n",
                                bi, wh_init->data_type, ww_init->data_type);
                        block_start = -1;
                        continue;
                    }

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

            c->pos_embed_blocks[bi].params = (rel_pos_bias_params_t){H, W, B, C, rel_h, rel_w};

            if (wh_init->data_type != 1 || ww_init->data_type != 1) {
                fprintf(stderr, "[onnx] pos_embed block %d: W_h/W_w not F32 — skipping\n", bi);
            } else {
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

    /* Allocate params array for pos_embed custom ops (must outlive graph compute) */
    if (c->n_pos_embed_blocks > 0) {
        c->pos_embed_params = (rel_pos_bias_params_t *)calloc(
            c->n_pos_embed_blocks, sizeof(rel_pos_bias_params_t));
        if (!c->pos_embed_params) goto fail;
        for (int b = 0; b < c->n_pos_embed_blocks; b++)
            c->pos_embed_params[b] = c->pos_embed_blocks[b].params;
    }

    /* Map ONNX nodes to ggml ops */
    for (int i = 0; i < onnx->n_nodes; i++) {
        /* Skip nodes inside pos_embed blocks (handled by fused custom op) */
        if (c->n_pos_embed_blocks > 0 && is_pos_embed_node(c, i)) {
            int bi = pos_embed_block_end(c, i);
            if (bi >= 0) {
                /* Last node of block — emit fused RelPosBias2D op */
                rel_pos_bias_params_t *p = &c->pos_embed_params[bi];
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

                /* Concat W_h [rel_h, C] and W_w [rel_w, C] → [rel_h+rel_w, C] in ggml
                 * (ggml dim 0 concatenation) */
                struct ggml_tensor *wcat = ggml_concat(c->ctx, wh_t, ww_t, 0);

                /* Dummy tensor with output shape: ggml [HW, HW, B]
                 * = ONNX [B, HW, HW] — the attention bias matrix */
                struct ggml_tensor *dummy = ggml_new_tensor_3d(c->ctx, GGML_TYPE_F32,
                                                                HW, HW, p->B);

                /* Emit fused custom op */
                struct ggml_tensor *out = ggml_map_custom3(c->ctx,
                    dummy, x_t, wcat,
                    rel_pos_bias_2d_cpu,
                    1,  /* n_tasks = 1 (single-threaded for now) */
                    p);

                ggml_set_name(out, c->pos_embed_blocks[bi].output_name);
                tmap_put_nd(c, c->pos_embed_blocks[bi].output_name, out, 3);

            }
            continue;
        }
        if (map_node(c, &onnx->nodes[i]) != 0) {
            /* Non-fatal: skip unsupported ops with warning */
            fprintf(stderr, "[onnx] SKIP node %d: %s (outputs: %s)\n",
                    i, onnx->nodes[i].op_type,
                    onnx->nodes[i].n_outputs > 0 ? onnx->nodes[i].outputs[0] : "?");
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
            } else {
                fprintf(stderr, "[onnx] WARNING: output '%s' not found in tmap\n",
                        onnx->outputs[i].name);
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
                            ggml_fp32_to_fp16_row((const float *)data, buf, (int64_t)src_elems);
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

#ifdef ONNX_NAN_DEBUG
/* Per-node eval callback: called before (ask=true) and after (ask=false) each node.
 * Checks inputs before, output after. Stops at first NaN. */
static bool onnx_nan_eval_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    int *idx = (int *)user_data;
    if (ask) {
        /* Before compute: check src inputs */
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) continue;
            float sv;
            ggml_backend_tensor_get(t->src[s], &sv, 0, sizeof(float));
            if (sv != sv) {
                fprintf(stderr, "[NaN-IN] node %d '%s' op=%d: src[%d] '%s' is NaN!\n",
                        *idx, t->name, (int)t->op, s, t->src[s]->name);
            }
        }
        return true;  /* we want post-compute callback */
    }
    /* After compute: check output */
    float v;
    ggml_backend_tensor_get(t, &v, 0, sizeof(float));
    if (v != v) {
        fprintf(stderr, "[NaN-OUT] node %d '%s' op=%d ne=[%lld,%lld,%lld,%lld]: OUTPUT NaN!\n",
                *idx, t->name, (int)t->op,
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
        /* Dump src values */
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) continue;
            float sv;
            ggml_backend_tensor_get(t->src[s], &sv, 0, sizeof(float));
            fprintf(stderr, "  src[%d] '%s' ne=[%lld,%lld,%lld,%lld] v0=%g\n",
                    s, t->src[s]->name,
                    (long long)t->src[s]->ne[0], (long long)t->src[s]->ne[1],
                    (long long)t->src[s]->ne[2], (long long)t->src[s]->ne[3], sv);
        }
        (*idx)++;
        return false;  /* stop iteration */
    }
    if (*idx < 5) {
        fprintf(stderr, "[OK] node %d '%s' op=%d v0=%g\n", *idx, t->name, (int)t->op, v);
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

#ifdef ONNX_NAN_DEBUG
    /* Install per-node eval callback for NaN tracing */
    static int nan_node_idx = 0;
    nan_node_idx = 0;
    ggml_backend_sched_set_eval_callback(ctx->sched, onnx_nan_eval_cb, &nan_node_idx);
#endif

    enum ggml_status status = ggml_backend_sched_graph_compute(ctx->sched, ctx->graph);

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
    free(ctx->pos_embed_params);
    free(ctx->roi_align_params);
    free(ctx->nms_params);
    /* Note: onnx model is NOT freed here — caller manages it */
    free(ctx);
}
