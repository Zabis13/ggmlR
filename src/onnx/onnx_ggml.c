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
        c->tensor_map_empty = realloc(c->tensor_map_empty,
                                       c->tensor_map_cap * sizeof(*c->tensor_map_empty));
    }
    int idx = c->tensor_map_size;
    strncpy(c->tensor_map_keys[idx], name, ONNX_MAX_NAME - 1);
    c->tensor_map_keys[idx][ONNX_MAX_NAME - 1] = '\0';
    c->tensor_map_vals[idx] = t;
    c->tensor_map_ndims[idx] = onnx_ndims;
    c->tensor_map_empty[idx] = 0;
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

/* Mark the most recent entry for `name` as logically empty; see the field
 * comment in onnx_ggml.h.  Marking after tmap_put_nd() rather than passing a
 * flag through it keeps every existing call site unchanged. */
void tmap_mark_empty(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0) {
            c->tensor_map_empty[i] = 1;
            return;
        }
    }
}

int tmap_is_empty(onnx_ggml_ctx_t *c, const char *name) {
    if (!name || name[0] == '\0') return 0;
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0)
            return c->tensor_map_empty[i];
    }
    return 0;
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
    /* Every ggml_reshape_* asserts contiguity, and a reshape of a strided view
     * is not something ggml can express -- the bytes would have to move.  This
     * is the single point through which the ONNX layer reaches those builders
     * (Reshape, Squeeze, Unsqueeze, Flatten all arrive here), so the contract
     * is met once here rather than in each handler.
     *
     * Conditional, not unconditional: the overwhelming majority of inputs are
     * already contiguous and stay zero-copy; only a genuinely strided one is
     * materialised.  MaskRCNN's Squeeze of a [1,147,1,1] view is what found
     * this -- it aborted the process, since an assert in a vendored builder is
     * GGML_ABORT, not an error a caller can handle.  It went unnoticed while
     * an upstream Gather defect was collapsing that tensor to one element,
     * where a view is trivially contiguous. */
    if (!ggml_is_contiguous(a))
        a = ggml_cont(ctx, a);
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
        /* Widened to F32 on load, one byte or two per element in the file --
         * see the conversion in create_initializer_tensors. */
        case ONNX_DTYPE_INT8:
        case ONNX_DTYPE_UINT8:
        case ONNX_DTYPE_INT16:
        case ONNX_DTYPE_UINT16:
        case ONNX_DTYPE_BOOL:     return GGML_TYPE_F32;
        default:                  return GGML_TYPE_F32;
    }
}

/* Is this dtype one the loader actually converts?
 *
 * The mapping above answers "which ggml type" and has to answer something, so
 * an unhandled dtype used to leave as F32 and have its bytes copied verbatim:
 * a UINT64 initializer became eight-bytes-read-as-two-floats, with no error
 * anywhere.  That is the same silent corruption BOOL caused in GPT-NeoX's
 * causal mask, and the reason for naming the supported set in one place
 * rather than leaving each default branch to guess.
 *
 * UINT32/UINT64 are refused rather than converted because nothing in the
 * reference set exercises them: writing the conversion blind would add code
 * no test reaches.  STRING and UNDEFINED have no numeric meaning at all. */
int onnx_dtype_supported(int32_t dt) {
    switch (dt) {
        case ONNX_DTYPE_FLOAT:
        case ONNX_DTYPE_FLOAT16:
        case ONNX_DTYPE_BFLOAT16:
        case ONNX_DTYPE_DOUBLE:
        case ONNX_DTYPE_INT8:
        case ONNX_DTYPE_UINT8:
        case ONNX_DTYPE_INT16:
        case ONNX_DTYPE_UINT16:
        case ONNX_DTYPE_INT32:
        case ONNX_DTYPE_INT64:
        case ONNX_DTYPE_BOOL:     return 1;
        default:                  return 0;
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
        case ONNX_DTYPE_UINT16:   return 2;
        case ONNX_DTYPE_UINT32:   return 4;
        case ONNX_DTYPE_UINT64:   return 8;
        case ONNX_DTYPE_BOOL:     return 1;
        /* Falling through to 4 is how UINT16 and UINT64 used to be sized --
         * wrong by half and by double respectively, before any conversion got
         * a chance to look at them.  The remaining fall-through is UNDEFINED
         * and STRING, neither of which has a byte width to report; callers
         * that reach a real tensor of those types are stopped by
         * onnx_dtype_supported() instead. */
        default:                  return 4;
    }
}

/* ── Create ggml tensors for initializers (weights) ─────────────── */

static int create_initializer_tensors(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_initializers; i++) {
        onnx_initializer_t *init = &c->onnx->initializers[i];

        /* Refuse before building anything.  An unsupported dtype has no
         * conversion below, so the raw bytes would be copied into an F32
         * tensor and read at the wrong width -- wrong numbers, no error.
         * Name the tensor and the dtype: the number is from the ONNX spec's
         * TensorProto.DataType, which is what a model dump shows. */
        if (!onnx_dtype_supported(init->data_type)) {
            fprintf(stderr, "[onnx] initializer '%s': unsupported dtype %d, "
                            "refusing to load it as raw bytes\n",
                    init->name, (int)init->data_type);
            return -1;
        }

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

        /* Same refusal as in create_initializer_tensors, and needed
         * separately: a Constant node's payload lives in a node attribute and
         * arrives here without ever passing through onnx->initializers[]. */
        if (!onnx_dtype_supported(init->data_type)) {
            fprintf(stderr, "[onnx] tensor '%s': unsupported dtype %d, "
                            "refusing to load it as raw bytes\n",
                    init->name[0] ? init->name : "?", (int)init->data_type);
            return -1;
        }

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

        /* INT8/UINT8/BOOL → F32 conversion: raw bytes are 1 byte per element,
         * but ggml tensor is F32 (4 bytes per element).
         *
         * BOOL belongs here for the same reason as UINT8, and leaving it out
         * was silent rather than fatal: onnx_dtype_to_ggml has no case for it,
         * so it fell through to F32 and the raw bytes were copied verbatim.
         * Four consecutive bools then read as one float -- GPT-NeoX's causal
         * mask, 128x128 of 0x01/0x00, became 1.4e-45 and zeros, which is zero
         * for every practical purpose.  Where() therefore took the "false"
         * branch everywhere, filling the attention scores with -1e9, softmax
         * of a row of equal values came back uniform, and the NaN surfaced a
         * MatMul later, far from the cause. */
        if ((init->data_type == ONNX_DTYPE_INT8 ||
             init->data_type == ONNX_DTYPE_UINT8 ||
             init->data_type == ONNX_DTYPE_BOOL) &&
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
            } else if (init->data_type == ONNX_DTYPE_BOOL) {
                /* Any non-zero byte is true: the spec says 0 or 1, but an
                 * exporter is not obliged to normalize what it writes. */
                for (size_t j = 0; j < src_elems; j++)
                    buf[j] = src[j] ? 1.0f : 0.0f;
            } else {
                for (size_t j = 0; j < src_elems; j++)
                    buf[j] = (float)src[j];
            }
            for (size_t j = src_elems; j < (size_t)n_elem; j++)
                buf[j] = 0.0f;
            ggml_backend_tensor_set(t, buf, 0, n_elem * sizeof(float));
            free(buf);
        }
        /* INT16/UINT16 → F32: two bytes per element in the file, four in the
         * tensor.  Same shape of conversion as INT8/UINT8 above, only wider;
         * memcpy rather than a cast through a short* because raw_data points
         * into the mmap with no alignment guarantee. */
        else if ((init->data_type == ONNX_DTYPE_INT16 ||
                  init->data_type == ONNX_DTYPE_UINT16) &&
                 t->type == GGML_TYPE_F32) {
            int64_t n_elem = ggml_nelements(t);
            size_t src_elems = data_size / 2;
            if ((int64_t)src_elems > n_elem) src_elems = (size_t)n_elem;
            float *buf = (float *)malloc(n_elem * sizeof(float));
            if (!buf) return -1;
            for (size_t j = 0; j < src_elems; j++) {
                if (init->data_type == ONNX_DTYPE_INT16) {
                    int16_t v;
                    memcpy(&v, (const char *)data + j * 2, 2);
                    buf[j] = (float)v;
                } else {
                    uint16_t v;
                    memcpy(&v, (const char *)data + j * 2, 2);
                    buf[j] = (float)v;
                }
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
    if (idx < 0 || idx >= n->n_inputs) return NULL;
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

/* ONNX_TRACE_SUM: whole-tensor checksum on each [val] line.
 *
 * Separate from ONNX_TRACE_VALS because it reads every element of every node:
 * that is the point (three leading values hide a difference anywhere else),
 * and also the reason it is not always on. */
int onnx_trace_sum(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ONNX_TRACE_SUM");
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


/* Bring both operands of an elementwise binary op to a type ggml can compute.
 *
 * ggml's binary kernels are F32/F16 only, so two integer operands abort in
 * binary_op rather than returning anything.  ONNX reaches that case through
 * ordinary index arithmetic: MaskRCNN assigns each box an FPN level with
 * floor/clamp/Cast(to=INT64) and then Sub(level, min_level), which arrives
 * here as i32 minus i32.
 *
 * F32 holds every integer up to 2^24 exactly, and these are levels, counts
 * and indices -- an index that large would have overflowed the tensor it
 * indexes long before precision became the problem.  The result stays F32:
 * the ops that consume it (Equal, Where, get_rows) take F32 indices already,
 * and casting back would cost a node for nothing.
 *
 * Returns 1 if it converted anything, so a caller can tell.  A mixed pair
 * (one integer, one float) is handled too -- the same widening, one side. */
int onnx_binary_promote(struct ggml_context *ctx,
                        struct ggml_tensor **pa, struct ggml_tensor **pb) {
    struct ggml_tensor *a = *pa, *b = *pb;
    const int a_int = (a->type == GGML_TYPE_I32);
    const int b_int = (b->type == GGML_TYPE_I32);

    if (!a_int && !b_int) return 0;          /* already floating point */

    if (a_int) *pa = ggml_cast_numeric(ctx, a, GGML_TYPE_F32);
    if (b_int) *pb = ggml_cast_numeric(ctx, b, GGML_TYPE_F32);
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
        if (r < 0) return -1;
        if (r > 0) goto reg_output;
        r = map_node_tensor (c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> tensor r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1;
        if (r > 0) goto reg_output;
        r = map_node_nn     (c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> nn r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1;
        if (r > 0) goto reg_output;
        r = map_node_quant  (c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> quant r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1;
        if (r > 0) goto reg_output;
        r = map_node_special(c, n, a, b, &out, &out_nd);
        if (r != 0 && onnx_trace_nodes())
            fprintf(stderr, "[dispatch] %s op=%s -> special r=%d\n", n->outputs[0], op, r);
        if (r < 0) return -1;
        if (r > 0) goto reg_output;
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
        /* The model's own declaration, where it made one.  Inferring the rank
         * from the inputs propagates any earlier mistake down the whole chain:
         * MaskRCNN's mask branch reaches NonZero with rank 2 where the export
         * says 1, purely because a DequantizeLinear eight ops upstream was
         * read as rank 2 and everything after it inherited that.
         *
         * Only the rank is taken.  The dimensions in value_info may be
         * symbolic, and the real ones are in the tensor already.
         *
         * ggml_n_dims remains the floor for the same reason it was before:
         * a rank below the physical shape desynchronises the two, and the
         * reconstruction in tmap_put_nd would then drop elements.  So the
         * declaration can only ever raise or confirm, never lower. */
        {
            /* Only where the rank was INFERRED.  A handler that set out_nd
             * knows the shape it just built, and the declaration is about the
             * ONNX-level tensor, not that construction: NonMaxSuppression
             * builds ggml [3, N] and says rank 2, while value_info declares
             * [?, 3] -- overriding it there dropped the rank to 1 and sent the
             * next Gather down the wrong axis. */
            int declared = (out_nd > 0) ? 0 : onnx_declared_rank(c->onnx, n->outputs[0]);
            int floor_nd = (int)ggml_n_dims(out);
            if (declared > 0 && declared >= floor_nd && declared != out_ndims) {
                if (onnx_trace_nodes())
                    fprintf(stderr, "[rank/declared] %s: %d -> %d (value_info)\n",
                            n->outputs[0], out_ndims, declared);
                out_ndims = declared;
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

    /* Inherit the logically-empty mark.  Deliberately outside the `if (out)`
     * above: the group handlers keep their own local `out`, separate from the
     * `out_p` they were passed, and a branch that registers its outputs itself
     * returns 1 without ever writing through that pointer -- so `out` is NULL
     * here for every one of them.  Gather is one, which is why not one of this
     * model's 849 Gather nodes reaches the block above.  Only names are needed
     * here; the tensor is already in the map.
     *
     * The list is deliberately short: these are the ops that are a pure
     * pass-through, view or index of a single input, so "the input had no
     * rows" and "the output has no rows" are the same statement.  Anything
     * else has to decide for itself what an empty operand means, and gets no
     * mark by default.
     *
     * Concat is not here -- it aggregates, and handles the mark in its own
     * handler by dropping empty inputs outright.
     *
     * Which operands count depends on the op, not on how many there are.
     * Gather selects rows, so an empty INDEX means an empty result however
     * full the data is -- requiring both to be empty would never fire, as the
     * data side never is.  For the rest the condition is over every named
     * input: an op whose operands come from different branches is only empty
     * when all of them are.  In MaskRCNN both of NonMaxSuppression's operands
     * descend from the same NonZero, but a graph where they do not must not
     * have the mark spread across. */
    {
        static const char *const transit[] = {
            "Squeeze", "Unsqueeze", "Gather", "NonMaxSuppression"
        };
        int is_transit = 0;
        for (size_t k = 0; k < sizeof(transit) / sizeof(*transit); k++)
            if (strcmp(n->op_type, transit[k]) == 0) { is_transit = 1; break; }

        if (is_transit) {
            int n_named = 0, n_empty = 0;
            if (strcmp(n->op_type, "Gather") == 0) {
                /* Either operand empties the result, so this one is an OR
                 * rather than the AND used below: selecting no rows leaves
                 * nothing, and selecting from nothing leaves nothing too.
                 * MaskRCNN needs both directions -- the per-class branch
                 * starts by indexing full boxes with an empty NonZero, and
                 * ends by indexing the empty NMS result with a constant. */
                n_named = 1;
                n_empty = 0;
                for (int i = 0; i < 2 && i < n->n_inputs; i++)
                    if (n->inputs[i][0] != '\0' && tmap_is_empty(c, n->inputs[i]))
                        n_empty = 1;
            } else if (strcmp(n->op_type, "NonMaxSuppression") == 0) {
                /* Boxes and scores.  Inputs 2..4 are the limits and
                 * thresholds -- scalars off an initializer, never empty,
                 * so counting them would keep the condition from ever
                 * being met. */
                for (int i = 0; i < 2 && i < n->n_inputs; i++) {
                    if (n->inputs[i][0] == '\0') continue;
                    n_named++;
                    if (tmap_is_empty(c, n->inputs[i])) n_empty++;
                }
            } else {
                for (int i = 0; i < n->n_inputs; i++) {
                    if (n->inputs[i][0] == '\0') continue;
                    n_named++;
                    if (tmap_is_empty(c, n->inputs[i])) n_empty++;
                }
            }
            if (n_named > 0 && n_empty == n_named) {
                for (int i = 0; i < n->n_outputs; i++)
                    if (n->outputs[i][0] != '\0')
                        tmap_mark_empty(c, n->outputs[i]);
                if (onnx_trace_nodes())
                    fprintf(stderr, "[empty] %s op=%s: inherited from %d input(s)\n",
                            n->outputs[0], n->op_type, n_empty);
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
        /* NonMaxSuppression: the kernel already wrote how many boxes it kept
         * into op_params[NMS_COUNT_SLOT], so the measurement is a read rather
         * than a scan. */
        if (strcmp(cn->op_type, "NonMaxSuppression") == 0) {
            struct ggml_tensor *out = tmap_get(c, cn->outputs[0]);
            if (!out || !out->buffer) continue;
            int64_t sel = (int64_t)out->op_params[NMS_COUNT_SLOT];
            if (sel < 0) continue;
            resolved_put(c, cn->outputs[0], sel);
            if (onnx_trace_nodes())
                fprintf(stderr, "[resolve] %s: NMS selected %lld box(es)\n",
                        cn->outputs[0], (long long)sel);
            continue;
        }

        /* TopK: its own K is min(limit, however many candidates arrived), and
         * the candidate count is the input's first ONNX dimension -- known
         * only once the previous segment has run.  Measuring it here lets the
         * re-map below rebuild the node at the real K instead of the
         * build-time guess, which for MaskRCNN was 1 and collapsed every
         * downstream shape to a single box. */
        if (strcmp(cn->op_type, "TopK") == 0) {
            struct ggml_tensor *src = tmap_get(c, cn->inputs[0]);
            if (!src || !src->buffer) continue;
            int src_nd = tmap_get_ndims(c, cn->inputs[0]);
            if (src_nd <= 0) src_nd = (int)ggml_n_dims(src);
            /* The candidates lie along the axis this TopK ranks, so the
             * measurement has to ask the node which axis that is.  ONNX axis
             * `ax` of a rank-`nd` tensor is ggml dimension nd-1-ax, the same
             * mapping the TopK op itself computes before rotating that axis
             * into ne[0] and capping k at it.
             *
             * Reading ne[src_nd-1] unconditionally, as this did, is that
             * formula with axis pinned to 0.  MaskRCNN has both kinds: its
             * five early TopKs carry axis=1 over a rank-2 [N,1] input, where
             * the candidates sit on ggml ne[0] -- 9408, 2352, 588, 147 and 48
             * of them -- while ne[1] is the shape's trailing 1.  Measuring
             * that 1 set k=1, collapsed every downstream shape to a single
             * box, and left the detection branch with nothing: segments 10
             * onwards built zero nodes, no NMS kernel ever ran, and all four
             * model outputs came back empty.  Its two late TopKs carry axis=0,
             * which is where the old formula happened to be right.
             *
             * The distinction cannot be made from the source's shape alone --
             * counting from ggml_n_dims(src) instead of the tmap rank was
             * tried and broke TopK on a non-final axis (test-onnx-reduce.R),
             * because ggml_n_dims collapses trailing ones whether or not the
             * ONNX rank behind them is real. */
            int64_t ax = onnx_attr_int(cn, "axis", -1);
            if (ax < 0) ax += src_nd;
            int gd = src_nd - 1 - (int)ax;
            if (gd < 0) gd = 0;
            if (gd >= GGML_MAX_DIMS) gd = GGML_MAX_DIMS - 1;
            int64_t n_cand = src->ne[gd];
            /* A count of 1 that survives the right axis is the honest
             * measurement, and the collapse lies further upstream. */
            if (n_cand <= 0) continue;
            resolved_put(c, cn->outputs[0], n_cand);
            if (onnx_trace_nodes())
                fprintf(stderr, "[resolve] %s: TopK src='%s' src_nd=%d axis=%d gd=%d "
                                "ggml_nd=%d ne=[%lld,%lld,%lld,%lld] -> %lld candidate(s)\n",
                        cn->outputs[0], cn->inputs[0], src_nd, (int)ax, gd,
                        (int)ggml_n_dims(src),
                        (long long)src->ne[0], (long long)src->ne[1],
                        (long long)src->ne[2], (long long)src->ne[3],
                        (long long)n_cand);
            continue;
        }

        if (strcmp(cn->op_type, "NonZero") != 0) continue;

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

/* Is this name one the caller reads when the whole run is over?
 *
 * Deliberately NOT folded into tensor_crosses_boundary(). That one answers
 * "does a later segment's GRAPH read this", and build_segment_graph() asks it
 * to decide what to expand the graph towards; a model output answered yes
 * there would be pulled into every segment's graph rather than the last one's,
 * which is the case the `seg == n_segments - 1` branch already handles.
 *
 * The reader here is R, after the final compute, through tmap -- so the test
 * belongs only where the question is "must this survive the pool", which is
 * copy_segment_boundaries().
 *
 * Measured on MaskRCNN-12-int8 (CPU): '6568', '6570' and '6572' are built
 * during segment 20 of 23, and nothing downstream takes them as an input --
 * they are results, not intermediates. copy_segment_boundaries() therefore
 * skipped all three, segment 20's pool release dropped their tmap entries, and
 * onnx_ggml_run reported "3 of 4 model outputs were never built ... every path
 * to them was cut by an op that could not be mapped" -- a misleading message,
 * since first_failed_node was empty and every one of them had mapped fine
 * (r=1, shapes [4,51], [51], [51], matching ONNX Runtime). The fourth output,
 * '6887', is built in segment 23 and survived for no better reason than that
 * nothing releases a pool after it. */
static int tensor_is_model_output(onnx_ggml_ctx_t *c, const char *nm) {
    for (int i = 0; i < c->onnx->n_outputs; i++)
        if (strcmp(c->onnx->outputs[i].name, nm) == 0)
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
        /* The OUTPUT flag decides whether ggml-alloc may hand this tensor's
         * buffer to a later node, so it belongs in a dump meant to explain who
         * overwrote what.  view_src too: a tensor sitting on someone else's
         * storage is freed by that owner's refcount, not its own. */
        /* Identity by pointer, not by name.  Three separate cast results all
         * inherit the name "6566 (cast_numeric)", so a dump keyed on names
         * cannot say which one carries a flag or shares storage with which --
         * and that distinction is the whole question here.  data= is what
         * makes an overwrite visible; view_src says whose refcount actually
         * governs that storage. */
        /* %s%s for view_src, not %s%s%s: the third conversion had no argument
         * and read whatever the stack held next, then followed it as a char*.
         * On CPU that happened to be harmless (it printed "(null)" at the end
         * of every line, which is what a missing argument looks like); on
         * Vulkan it was a live address and the trace segfaulted at
         * 'memory not mapped' -- so the only tool that could show a segment's
         * graph could not be run on the backend under investigation. */
        fprintf(stderr, "[sgraph]   %d '%s' op=%s type=%s%s%s obj=%p data=%p"
                        " src0=%s(%s,%p) src1=%s(%s,%p)%s%s\n",
                i, ggml_get_name(t), ggml_op_name(t->op), ggml_type_name(t->type),
                (t->flags & GGML_TENSOR_FLAG_OUTPUT) ? " OUT" : "",
                (t->flags & GGML_TENSOR_FLAG_INPUT)  ? " IN"  : "",
                (void *)t, (void *)t->data,
                t->src[0] ? ggml_get_name(t->src[0]) : "-",
                t->src[0] ? ggml_type_name(t->src[0]->type) : "-",
                t->src[0] ? (void *)t->src[0]->data : NULL,
                t->src[1] ? ggml_get_name(t->src[1]) : "-",
                t->src[1] ? ggml_type_name(t->src[1]->type) : "-",
                t->src[1] ? (void *)t->src[1]->data : NULL,
                t->view_src ? " view_src=" : "",
                t->view_src ? ggml_get_name(t->view_src) : "");
    }
}

/* The storage a tensor actually lives in: a view owns nothing, so its
 * lifetime is decided by whoever it sits on.  Marking a view has no effect on
 * when its bytes are reused; the owner has to be marked instead. */
static struct ggml_tensor *storage_owner(struct ggml_tensor *t) {
    while (t && t->view_src) t = t->view_src;
    return t;
}

/* Does this tensor's METADATA live in the segment pool about to be released?
 *
 * The same address test segment_ctx_release() uses to decide which tmap
 * entries to drop, exposed so the two cannot disagree -- and they did. A view
 * (ggml_reshape and friends) allocates its struct from the current context
 * while inheriting ->buffer and ->data from whatever it looks at, so a reshape
 * of a persistent tensor built during a segment has persistent STORAGE and
 * pool-resident METADATA.
 *
 * copy_segment_boundaries() judged those by ->buffer alone and skipped them as
 * "already survives", then segment_ctx_release() dropped them by address.
 * Measured on MaskRCNN-12-int8: '1919' (Reshape over '1912') was reported
 * "is persistent -- not copied", the pool release took its tmap entry 7 lines
 * later, and Gather '1920' got inputs=[1919(NULL)] -- which killed the RPN
 * branch feeding 2157/2169, left all four outputs unbuilt on CPU, and on
 * Vulkan reached GGML_ASSERT(buffer != nullptr) in ggml_vk_tensor_subbuffer.
 *
 * Storage outlasting metadata is not a case tmap can represent: the entry has
 * to point at a struct that still exists, so such a tensor needs copying like
 * any other boundary crosser. */
static int tensor_in_one_pool(struct ggml_context *pool,
                              const struct ggml_tensor *t) {
    if (!pool) return 0;
    const char *base = (const char *)ggml_get_mem_buffer(pool);
    const size_t span = ggml_get_mem_size(pool);
    const char *p = (const char *)t;
    return base && p >= base && p < base + span;
}

/* Both live pools, because segment_ctx_release() is called for both: ctx_seg
 * and ctx_prev. Testing only ctx_seg would agree with it today -- the main
 * loop leaves ctx_prev unused -- and stop agreeing the moment it is put to
 * work, which is the kind of divergence this helper exists to prevent. */
static int tensor_in_pool(onnx_ggml_ctx_t *c, const struct ggml_tensor *t) {
    if (!t) return 0;
    return tensor_in_one_pool(c->ctx_seg,  t) ||
           tensor_in_one_pool(c->ctx_prev, t);
}

/* Keep alive anything the graph reads more than once.
 *
 * ggml-alloc hands a node's output the storage of an input it believes is
 * finished with, which is right when that input has one reader and wrong when
 * it has several.  MaskRCNN's TopK indices have three: the tensor feeds three
 * casts, one per Gather, and the second Gather's output was placed on top of
 * it -- so the third Gather read detection class numbers where indices should
 * have been, and index 80 ran off the end of an 80-row table.
 *
 * The OUTPUT flag is what tells the allocator not to do that.  It is a blunt
 * instrument: it holds the buffer for the whole graph rather than until the
 * last read, so peak memory goes up.  That trade is deliberate -- a wrong
 * answer costs more than a buffer, and a real liveness analysis is a larger
 * change than this defect warrants.
 *
 * This runs over the finished graph rather than over ONNX consumer counts: a
 * value with three consumers in the model may well have one in this segment,
 * and only what is actually in gf->nodes can overwrite anything.  Counting is
 * by storage owner, not by tensor object, because the readers here arrive
 * through views and would otherwise each look unique.
 *
 * Independent of tensor_crosses_boundary(), which answers a different
 * question -- who reads this in a LATER segment -- and stays as it is. */
static void hold_multi_consumer_tensors(onnx_ggml_ctx_t *c) {
    if (!c->graph) return;
    const int n = ggml_graph_n_nodes(c->graph);

    /* Counting goes in a side table, never in a field of the tensor: extra
     * and op_params belong to the backends, and borrowing one of them here
     * would be the same kind of shared-state defect this is fixing.
     *
     * Two entries per node is the worst case that matters (a node reads at
     * most GGML_MAX_SRC sources), so the table is bounded by the graph. */
    struct ggml_tensor **owners = (struct ggml_tensor **)
        malloc((size_t)n * GGML_MAX_SRC * sizeof(struct ggml_tensor *));
    int *counts = (int *)calloc((size_t)n * GGML_MAX_SRC, sizeof(int));
    if (!owners || !counts) { free(owners); free(counts); return; }
    int n_owners = 0;

    for (int i = 0; i < n; i++) {
        struct ggml_tensor *node = ggml_graph_node(c->graph, i);

        /* A node reading one owner through two of its srcs counts once: it
         * cannot overwrite a buffer it is itself still reading. */
        struct ggml_tensor *seen_here[GGML_MAX_SRC];
        int n_here = 0;

        for (int k = 0; k < GGML_MAX_SRC; k++) {
            struct ggml_tensor *src = node->src[k];
            if (!src) continue;
            struct ggml_tensor *own = storage_owner(src);
            if (!own) continue;

            int dup = 0;
            for (int q = 0; q < n_here; q++)
                if (seen_here[q] == own) { dup = 1; break; }
            if (dup) continue;
            seen_here[n_here++] = own;

            int slot = -1;
            for (int q = 0; q < n_owners; q++)
                if (owners[q] == own) { slot = q; break; }
            if (slot < 0) { slot = n_owners++; owners[slot] = own; }

            if (++counts[slot] == 2) ggml_set_output(own);
        }
    }

    free(owners);
    free(counts);
}

/* Record src[] of every node in the current graph, before the scheduler runs.
 *
 * Call after the graph is built and BEFORE ggml_backend_sched_alloc_graph(),
 * which is where the rewrite happens: alloc_graph -> split_graph, and
 * split_graph assigns node->src[j] = tensor_id_copy(...) for every source
 * that sits on a different backend. See the field comment in onnx_ggml.h for
 * what that costs a segmented model. */
static void snapshot_graph_srcs(onnx_ggml_ctx_t *c) {
    c->n_src_snap = 0;
    if (!c->graph) return;

    /* ON by default: it is what makes the FIRST run correct. Measured on
     * MaskRCNN-12-int8 / Vulkan:
     *   snapshot ON  -> 51 detections (max|d|=0 vs ORT), then a second run
     *                   aborts in ggml_vk_tensor_subbuffer
     *   snapshot OFF -> 52 detections, but two runs survive
     * Neither is right. This is a stopgap for the first run, not the fix; the
     * fix is a context per segment (see TODO), which removes the shared
     * tensor objects the scheduler's rewrite rides on.
     *
     * GGMLR_NO_SRC_SNAPSHOT=1 turns it off, which is how the two rows above
     * were measured. */
    {
        static int off = -1;
        if (off < 0) {
            const char *e = getenv("GGMLR_NO_SRC_SNAPSHOT");
            off = (e && *e && *e != '0') ? 1 : 0;
        }
        if (off) return;
    }

    /* Only when a second backend exists.  With the CPU alone the scheduler
     * never makes a cross-backend copy, so there is nothing to undo and the
     * snapshot would be pure overhead on the path that already works. */
    if (!c->backend_gpu) return;

    const int n = ggml_graph_n_nodes(c->graph);
    if (n > ONNX_MAX_SRC_SNAPSHOT) {
        fprintf(stderr, "[onnx] segment graph has %d nodes, over the %d the "
                        "src snapshot holds -- not restoring (a cross-backend "
                        "copy may outlive this segment)\n",
                n, ONNX_MAX_SRC_SNAPSHOT);
        return;
    }
    for (int i = 0; i < n; i++) {
        struct ggml_tensor *node = ggml_graph_node(c->graph, i);
        if (!node) continue;
        c->src_snap_node[c->n_src_snap] = node;
        for (int j = 0; j < GGML_MAX_SRC; j++)
            c->src_snap_src[c->n_src_snap][j] = node->src[j];
        c->n_src_snap++;
    }
}

/* Put back the src[] the scheduler replaced with its own copies.
 *
 * Only pointers the snapshot actually holds are written, and only where they
 * differ, so a node the scheduler left alone is not touched at all. */
static void restore_graph_srcs(onnx_ggml_ctx_t *c) {
    int n_restored = 0, n_kept = 0;
    for (int i = 0; i < c->n_src_snap; i++) {
        struct ggml_tensor *node = c->src_snap_node[i];
        if (!node) continue;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor *orig = c->src_snap_src[i][j];
            if (node->src[j] == orig) continue;

            /* ⚠️ Do NOT put back a source the GPU cannot read.
             *
             * The scheduler replaces src[j] with a copy on the consumer's
             * backend exactly when the producer sits on a different one -- a
             * CPU-only op such as the NMS custom op feeding a Vulkan CONT.
             * That copy is the only version of the data the Vulkan op can
             * address: restoring the original leaves a Vulkan op pointing at a
             * CPU tensor, and ggml_vk_tensor_subbuffer() then dereferences a
             * buffer that is not a Vulkan one (measured on MaskRCNN-12-int8,
             * node '2169 (permuted) (cont)': the input matches ORT exactly and
             * the run dies producing the output).
             *
             * So the snapshot's job is narrowed to what it was for -- undoing
             * rewrites WITHIN one backend -- and cross-backend copies are left
             * alone. A source with no buffer yet is not a scheduler copy and is
             * restored as before. */
            if (c->backend_gpu && orig && orig->buffer &&
                !ggml_backend_supports_buft(c->backend_gpu,
                                            ggml_backend_buffer_get_type(orig->buffer))) {
                n_kept++;
                continue;
            }

            node->src[j] = orig;
            n_restored++;
        }
    }
    if ((n_restored > 0 || n_kept > 0) && onnx_trace_nodes())
        fprintf(stderr, "[src-restore] %d source pointer(s) put back, %d "
                        "cross-backend copy(ies) kept after segment %d\n",
                n_restored, n_kept, c->cur_segment);
    c->n_src_snap = 0;
}

/* Give this segment a pool of its own, and point c->ctx at it.
 *
 * Every op builder allocates from c->ctx, so swapping the pointer is what puts
 * a segment's graph and intermediates somewhere they can be released; changing
 * the builders instead would mean touching every op in four files.
 *
 * The size is the model context's own estimate rather than a fraction of it: a
 * segment can hold most of the graph (MaskRCNN's segment 18 has 240 nodes), and
 * a pool too small fails by returning NULL tensors that only crash once
 * something reads them. It is freed at the end of the segment, so the peak is
 * one such pool at a time, not one per segment. */
static void segment_ctx_release(onnx_ggml_ctx_t *c, struct ggml_context *pool);

static int segment_ctx_begin(onnx_ggml_ctx_t *c, size_t mem_size) {
    struct ggml_init_params p = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    c->ctx_seg = ggml_init(p);
    if (!c->ctx_seg) {
        fprintf(stderr, "[onnx] could not create a context for segment %d\n",
                c->cur_segment);
        return -1;
    }
    if (!c->ctx_main) c->ctx_main = c->ctx;
    c->ctx = c->ctx_seg;
    return 0;
}

/* Release the segment's pool and every reference into it.
 *
 * tmap is truncated back to where the segment started: its entries point at
 * tensors that are about to stop existing, and tmap_get searches backwards, so
 * a stale entry would shadow the real tensor of the same name rather than
 * merely linger. Anything a later segment still needs was copied into
 * ctx_boundary before this runs, and that copy re-registered the name.
 *
 * ⚠️ Call only after the segment has computed AND after
 * copy_segment_boundaries() -- the copies read from these tensors. */
static void segment_ctx_release(onnx_ggml_ctx_t *c, struct ggml_context *pool) {
    if (!pool) return;

    /* Drop the tmap entries that point INTO this pool, and only those.
     *
     * Truncating back to the size tmap had when the segment began was tried
     * first and breaks the model outright: a segment registers tensors that
     * outlive it too -- constants and cut-op outputs in ctx_weight, and above
     * all the boundary copies, which copy_segment_boundaries() registers under
     * the original's name precisely so the next segment finds them. Cutting the
     * tail removed those as well, and MaskRCNN then failed at node 674
     * (Transpose) with all four outputs unbuilt.
     *
     * Ownership is decided by address: a tensor allocated from a context lives
     * inside that context's memory buffer, so a pointer within [base, base+size)
     * belongs to the pool about to be freed. Entries are compacted rather than
     * blanked -- tmap_get searches backwards and a NULL slot would still shadow
     * an earlier, valid entry of the same name. */
    const char *base = (const char *)ggml_get_mem_buffer(pool);
    const size_t span = ggml_get_mem_size(pool);
    int kept = 0, dropped = 0;
    for (int i = 0; i < c->tensor_map_size; i++) {
        const char *p = (const char *)c->tensor_map_vals[i];
        if (p >= base && p < base + span) { dropped++; continue; }
        if (kept != i) {
            memcpy(c->tensor_map_keys[kept], c->tensor_map_keys[i],
                   sizeof(c->tensor_map_keys[0]));
            c->tensor_map_vals[kept]  = c->tensor_map_vals[i];
            c->tensor_map_ndims[kept] = c->tensor_map_ndims[i];
            c->tensor_map_empty[kept] = c->tensor_map_empty[i];
            memcpy(c->tensor_map_onnx_ne[kept], c->tensor_map_onnx_ne[i],
                   sizeof(c->tensor_map_onnx_ne[0]));
        }
        kept++;
    }
    c->tensor_map_size = kept;
    if (dropped && onnx_trace_nodes())
        fprintf(stderr, "[segctx] segment %d: released pool, dropped %d tmap "
                        "entr%s, %d kept\n",
                c->cur_segment, dropped, dropped == 1 ? "y" : "ies", kept);

    /* The deferred-fill lists hold raw tensor pointers into the same pool, and
     * nothing above reaches them: tmap is compacted by address, they are not.
     * After ggml_free() the pool's memory is handed out again and those
     * pointers name whatever was built over them.
     *
     * Measured on MaskRCNN-12-int8 (CPU, first run) by printing
     * ggml_get_name() for a FIXED slot index across successive
     * fill_deferred_tensors() passes -- the pointer is written once, so a name
     * that changes is the memory underneath changing, not the list. Slot 42
     * (NonZero '3043') read 3042, then '6081 (cast_numeric)', then '3111',
     * then nothing at all; 202 of 220 slots moved this way, 567 changes in
     * total, and NOT ONE of them before the first pool release. The fill loop
     * skips a NULL src silently, so '3043' kept whatever its allocation held
     * -- 9.7e+21, which casts to INT_MIN -- and get_rows aborted with
     * GGML_ASSERT(i01 >= 0 && i01 < ne01).
     *
     * Dropped rather than skipped at fill time: a stale entry that happens to
     * point at a live tensor of the right size would be filled from the wrong
     * source and never look wrong. The op that owns the fill re-registers it
     * on the next map_node, which is what makes dropping safe -- these are
     * per-run registrations, not ownership records.
     *
     * ⚠️ Only lists whose entries own nothing. Deliberately excluded:
     *   - qconv_mult_values: malloc'd, and freed in teardown by COUNT, so
     *     compacting the list leaks the tail unless the free moves here too;
     *   - cut_carry_buf: malloc'd with its own free a few lines from where it
     *     is taken, inside one iteration of the segment loop;
     *   - src_snap_*: written back into nodes before the next graph is built,
     *     a protocol this function is not part of;
     *   - roi_aligns / nms_ops / qconv_ops: parameter blocks whose ADDRESSES
     *     were handed to kernels as userdata (see reset_deferred_fills). */
    {
        const int n_nz0 = c->n_nonzero_fills;
        const int n_sl0 = c->n_slice_fills;
        const int n_ey0 = c->n_eye_fills;
        const int n_sh0 = c->n_shape_tensors;
        const int n_cf0 = c->n_const_fills;
        const int n_ci0 = c->n_cinit_fills;
        const int n_nm0 = c->n_nms_deferred;
        int k;

        /* NonZero: two tensors per entry, either one enough to condemn it. */
        k = 0;
        for (int i = 0; i < n_nz0; i++) {
            if (tensor_in_one_pool(pool, c->nonzero_fill_dst[i]) ||
                tensor_in_one_pool(pool, c->nonzero_fill_src[i])) continue;
            if (k != i) {
                c->nonzero_fill_dst[k]   = c->nonzero_fill_dst[i];
                c->nonzero_fill_src[k]   = c->nonzero_fill_src[i];
                c->nonzero_fill_ndims[k] = c->nonzero_fill_ndims[i];
            }
            k++;
        }
        c->n_nonzero_fills = k;

        /* Strided Slice: same shape of entry, same test. */
        k = 0;
        for (int i = 0; i < n_sl0; i++) {
            if (tensor_in_one_pool(pool, c->slice_fill_dst[i]) ||
                tensor_in_one_pool(pool, c->slice_fill_src[i])) continue;
            if (k != i) {
                c->slice_fill_dst[k]   = c->slice_fill_dst[i];
                c->slice_fill_src[k]   = c->slice_fill_src[i];
                c->slice_fill_ndims[k] = c->slice_fill_ndims[i];
                memcpy(c->slice_fill_starts[k], c->slice_fill_starts[i],
                       sizeof(c->slice_fill_starts[0]));
                memcpy(c->slice_fill_steps[k], c->slice_fill_steps[i],
                       sizeof(c->slice_fill_steps[0]));
                memcpy(c->slice_fill_out_ne[k], c->slice_fill_out_ne[i],
                       sizeof(c->slice_fill_out_ne[0]));
            }
            k++;
        }
        c->n_slice_fills = k;

        /* EyeLike: one tensor, three scalars beside it. */
        k = 0;
        for (int i = 0; i < n_ey0; i++) {
            if (tensor_in_one_pool(pool, c->eye_fill_ptrs[i])) continue;
            if (k != i) {
                c->eye_fill_ptrs[k] = c->eye_fill_ptrs[i];
                c->eye_fill_rows[k] = c->eye_fill_rows[i];
                c->eye_fill_cols[k] = c->eye_fill_cols[i];
                c->eye_fill_k[k]    = c->eye_fill_k[i];
            }
            k++;
        }
        c->n_eye_fills = k;

        /* Shape outputs: the dims travel in the row beside the pointer. */
        k = 0;
        for (int i = 0; i < n_sh0; i++) {
            if (tensor_in_one_pool(pool, c->shape_tensor_ptrs[i])) continue;
            if (k != i) {
                c->shape_tensor_ptrs[k] = c->shape_tensor_ptrs[i];
                memcpy(c->shape_tensors_ne[k], c->shape_tensors_ne[i],
                       sizeof(c->shape_tensors_ne[0]));
            }
            k++;
        }
        c->n_shape_tensors = k;

        /* ConstantOfShape: pointer plus the scalar it is filled with. */
        k = 0;
        for (int i = 0; i < n_cf0; i++) {
            if (tensor_in_one_pool(pool, c->const_fill_ptrs[i])) continue;
            if (k != i) {
                c->const_fill_ptrs[k] = c->const_fill_ptrs[i];
                c->const_fill_vals[k] = c->const_fill_vals[i];
            }
            k++;
        }
        c->n_const_fills = k;

        /* Constant-node payloads: the second pointer is into the ONNX model,
         * which outlives every pool, so only the tensor decides. */
        k = 0;
        for (int i = 0; i < n_ci0; i++) {
            if (tensor_in_one_pool(pool, c->cinit_fill_ptrs[i])) continue;
            if (k != i) {
                c->cinit_fill_ptrs[k] = c->cinit_fill_ptrs[i];
                c->cinit_fill_srcs[k] = c->cinit_fill_srcs[i];
            }
            k++;
        }
        c->n_cinit_fills = k;

        /* NMS parameter tensors: the params themselves are scalars here; the
         * malloc'd blocks handed to the kernel are n_nms_ops, left alone. */
        k = 0;
        for (int i = 0; i < n_nm0; i++) {
            if (tensor_in_one_pool(pool, c->nms_param_tensors[i])) continue;
            if (k != i) {
                c->nms_param_tensors[k]     = c->nms_param_tensors[i];
                c->nms_max_boxes[k]         = c->nms_max_boxes[i];
                c->nms_iou_thresh[k]        = c->nms_iou_thresh[i];
                c->nms_score_thresh[k]      = c->nms_score_thresh[i];
                c->nms_have_score_thresh[k] = c->nms_have_score_thresh[i];
            }
            k++;
        }
        c->n_nms_deferred = k;

        const int nz = n_nz0 - c->n_nonzero_fills;
        const int sl = n_sl0 - c->n_slice_fills;
        const int ey = n_ey0 - c->n_eye_fills;
        const int sh = n_sh0 - c->n_shape_tensors;
        const int cf = n_cf0 - c->n_const_fills;
        const int ci = n_ci0 - c->n_cinit_fills;
        const int nm = n_nm0 - c->n_nms_deferred;

        if ((nz | sl | ey | sh | cf | ci | nm) && onnx_trace_nodes())
            fprintf(stderr, "[segctx] segment %d: dropped deferred fills into "
                            "the pool: nonzero=%d slice=%d eye=%d shape=%d "
                            "const=%d cinit=%d nms=%d\n",
                    c->cur_segment, nz, sl, ey, sh, cf, ci, nm);
    }
}

/* Close both live pools and put c->ctx back on the model context.
 *
 * Used where the segment walk is over for good -- the start of the next run,
 * and teardown -- rather than between segments, which is segment_ctx_begin()'s
 * job. */
static void segment_ctx_end(onnx_ggml_ctx_t *c) {
    if (!c->ctx_seg && !c->ctx_prev) return;

    if (c->ctx_main) c->ctx = c->ctx_main;

    if (c->ctx_seg) {
        segment_ctx_release(c, c->ctx_seg);
        ggml_free(c->ctx_seg);
        c->ctx_seg = NULL;
    }
    if (c->ctx_prev) {
        segment_ctx_release(c, c->ctx_prev);
        ggml_free(c->ctx_prev);
        c->ctx_prev = NULL;
    }
    c->ctx_main = NULL;

    /* The graph lived in a pool just freed, so the pointer is dangling.
     * Segment 0's graph is in the model context and is restored from
     * seg0_graph by the next run; anything else would be read from freed
     * memory, and the comparison against seg0_graph at the top of
     * onnx_ggml_run() is exactly such a read. */
    c->graph = c->seg0_graph;
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

        /* A cut op's OUTPUTS are not added here: a cut op's own segment is
         * where it gets computed, and the result is carried across by
         * copy_segment_boundaries (ordinary boundary tensors) or by the
         * cut_carry list (a cut op's own GGML_OP_CUSTOM output, saved around
         * the re-map that rebuilds it at its measured size).  See the
         * ordering note at the copy_segment_boundaries call site. */
    }

    for (int i = sg->first_node; i <= sg->last_node; i++) {
        for (int o = 0; o < onnx->nodes[i].n_outputs; o++) {
            const char *nm = onnx->nodes[i].outputs[o];
            if (nm[0] == '\0') continue;
            /* A model output produced HERE counts as well, and the boundary
             * test cannot see it: that test looks for a later node taking the
             * name as an input, and a result has no consumer inside the graph
             * at all. The branch below covers only outputs that survive to the
             * last segment, which on a segmented detector is not where most of
             * them are built.
             *
             * ⚠️ Pairs with the same predicate in copy_segment_boundaries().
             * Either half alone does nothing: not expanding the graph here
             * leaves the tensor uncomputed, so the copy there finds "no buffer"
             * and skips it; copying without expanding preserves a tensor that
             * was never filled. Measured on MaskRCNN-12-int8 -- '6568', '6570'
             * and '6572' are nodes 2815..2817, inside segment 21 of 23. */
            if (!tensor_crosses_boundary(c, seg, nm) &&
                !tensor_is_model_output(c, nm)) continue;
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

    hold_multi_consumer_tensors(c);

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

    /* ctx_host first, and on the CPU backend whatever the weights use.
     *
     * Tensors land in ctx_host after load too: every re-map of a cut op builds
     * a new NMS params block and a new NMS output there, and the one-shot
     * allocation at load time never saw them. Left without a buffer the output
     * cannot be measured, which is the whole reason it was moved out of c->ctx
     * -- see the note at its ggml_custom_4d. Returns NULL when nothing new was
     * added, which is the common case. */
    if (c->ctx_host && c->backend_cpu) {
        ggml_backend_buffer_t hbuf =
            ggml_backend_alloc_ctx_tensors(c->ctx_host, c->backend_cpu);
        if (hbuf) {
            if (c->n_extra_weight_bufs >= ONNX_MAX_WEIGHT_BUFS) {
                fprintf(stderr, "[onnx] too many weight buffers (>%d)\n",
                        ONNX_MAX_WEIGHT_BUFS);
                ggml_backend_buffer_free(hbuf);
                return -1;
            }
            c->extra_weight_bufs[c->n_extra_weight_bufs++] = hbuf;
            if (onnx_trace_nodes())
                fprintf(stderr, "[wbuf] segment %d: host buffer %d (%zu bytes)\n",
                        c->cur_segment, c->n_extra_weight_bufs - 1,
                        ggml_backend_buffer_get_size(hbuf));
        }
    }

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

/* Is this tensor's storage owned by an op the GPU cannot run?
 *
 * Asked along the whole view_src chain, not one step: a view sits on its
 * source's bytes, so the two can never be placed on different backends, and a
 * view of a view is just as bound.  Returns 0 for a leaf (GGML_OP_NONE) --
 * supports_op says false for those too, and a weight must keep its normal
 * placement. */
static int storage_op_unsupported(onnx_ggml_ctx_t *c, struct ggml_tensor *t) {
    if (!c->backend_gpu) return 0;
    /* GGML_MAX_SRC is an arbitrary but finite bound: view chains are short,
     * and a cycle here would otherwise hang the build. */
    for (int depth = 0; t && depth < 64; depth++) {
        if (t->op != GGML_OP_NONE &&
            !ggml_backend_supports_op(c->backend_gpu, t)) {
            return 1;
        }
        t = t->view_src;
    }
    return 0;
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
        /* Does this tensor's storage belong to an op the GPU cannot run?
         *
         * Two tensors that share storage cannot be placed on different
         * backends: ggml_permute returns a VIEW of its source, so pinning the
         * view to the GPU while the source runs on the host makes the device
         * read host memory -- measured, and it is not a soft failure:
         * "GPUVM fault at 0x651cc4a25000, PERMISSION_FAULTS: 3", then
         * vk::DeviceLostError and an aborted process.
         *
         * So the question is never "is THIS tensor supported" but "is the
         * storage it sits on owned by something unsupported", and that is
         * asked along the whole view_src chain rather than one step: a
         * RESHAPE of a PERMUTE of an unsupported op is two links away, and
         * a check that stops at the first would pin it exactly as before.
         * MaskRCNN has no chain deeper than one today (measured: 6 views on a
         * CPU-only tensor, none on those views), so the loop guards against
         * the case rather than one already seen.
         *
         * Leaves are excluded deliberately: supports_op answers false for
         * GGML_OP_NONE too, and treating a weight as unsupported would strip
         * it of the assignment the comment above asks for. */
        int n_nodes = ggml_graph_n_nodes(c->graph);
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor *node = ggml_graph_node(c->graph, i);
            /* A node whose storage belongs to an unsupported op is skipped
             * WITH its sources: pinning the consumer while leaving the owner
             * to the scheduler is the combination that faulted the device. */
            if (storage_op_unsupported(c, node)) continue;
            if (ggml_backend_supports_op(c->backend_gpu, node)) {
                ggml_backend_sched_set_tensor_backend(sch, node, c->backend_gpu);
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j]) {
                        /* A source that is itself an op the GPU cannot run must
                         * keep its own placement.
                         *
                         * This loop used to assign EVERY source of a supported
                         * node to the GPU.  That is right for weights and for
                         * intermediates, and wrong for an op the backend
                         * refuses: NonMaxSuppression is GGML_OP_CUSTOM, which
                         * Vulkan has no case for, but its consumer is a PERMUTE,
                         * which is supported -- so the NMS output was handed to
                         * the GPU because of who reads it.
                         *
                         * ggml_backend_sched_set_tensor_backend writes the id
                         * directly and marks the cause "usr", which the
                         * scheduler treats as authoritative: the pass that would
                         * otherwise place the node on the first backend that
                         * supports it never reconsiders.  Segment 14 of
                         * MaskRCNN then planned as n_splits=1 (gpu~1 cpu~0) --
                         * a graph holding a CPU-only op with no CPU split in it.
                         * The kernel was never called, its output tensor kept
                         * whatever the buffer already held (all ones, or the
                         * denormal 0x00000001), and every consumer downstream
                         * read that: box indices became 1 1 1, a Gather returned
                         * one row 304 times, the ARGSORT over the resulting
                         * constant vector produced an identity permutation, and
                         * the detector's box count diverged from the CPU run.
                         *
                         * The test is `is an op` AND `unsupported`, not merely
                         * `unsupported`: supports_op answers false for a leaf
                         * (GGML_OP_NONE) too, so testing support alone would
                         * strip the weights of the very assignment the comment
                         * below asks for.  Measured on MaskRCNN with
                         * ONNX_TRACE_PREASSIGN: 21 sources are skipped, all of
                         * them real ops (17 CUSTOM, 4 MAP_CUSTOM3), no leaf and
                         * no weight among them. */
                        struct ggml_tensor *sj = node->src[j];
                        int sj_unsupported = storage_op_unsupported(c, sj);
                        if (getenv("ONNX_TRACE_PREASSIGN")) {
                            struct ggml_tensor *s = sj;
                            if (sj_unsupported) {
                                /* Why it was refused, inferred from the tensor.
                                 *
                                 * The backend has no general "explain the last
                                 * refusal" channel -- only QCONV_I32 carries one
                                 * (QC_NO in ggml-vulkan-graph.cpp) -- and adding
                                 * one for every op would be a far larger change
                                 * than this measurement warrants.  The common
                                 * conditions are readable from outside, so they
                                 * are reconstructed here.
                                 *
                                 * The distinction that matters: an op with no
                                 * case at all is refused for every tensor and is
                                 * a known CPU-only op, while one refused over
                                 * layout or dtype is normally a GPU op that fell
                                 * out on THIS tensor -- unpredictable, and
                                 * invisible in any static list of unsupported
                                 * ops. */
                                /* No "leaf" case here: sj_unsupported already
                                 * requires sj to be an op, so a leaf never
                                 * reaches this print.  Claiming otherwise would
                                 * suggest the trace can report something it
                                 * cannot. */
                                const char *why = "op has no Vulkan case";
                                if (!ggml_is_contiguous(s)) {
                                    why = "dst not contiguous";
                                } else if (s->src[0] && !ggml_is_contiguous(s->src[0])) {
                                    why = "src0 not contiguous";
                                } else if (s->type != GGML_TYPE_F32 &&
                                           s->type != GGML_TYPE_F16) {
                                    why = "dst dtype";
                                } else if (s->src[0] && s->src[0]->type != s->type) {
                                    why = "src0/dst dtype mismatch";
                                }
                                fprintf(stderr,
                                    "[preassign] seg %d: '%s' op=%s <- src[%d] "
                                    "'%s' op=%s type=%s NOT supported (%s) "
                                    "leaf=%d buffer=%d cont=%d ne=[%lld,%lld]\n",
                                    c->cur_segment, node->name,
                                    ggml_op_name(node->op), j, s->name,
                                    ggml_op_name(s->op), ggml_type_name(s->type),
                                    why,
                                    s->op == GGML_OP_NONE ? 1 : 0,
                                    s->buffer ? 1 : 0,
                                    ggml_is_contiguous(s) ? 1 : 0,
                                    (long long)s->ne[0], (long long)s->ne[1]);
                            }
                        }
                        if (sj_unsupported) {
                            /* Leave it for the scheduler, which places it on a
                             * backend that can run it and puts a split boundary
                             * around it -- exactly what already happens for the
                             * 85 CPU-only ops nothing reads inside their own
                             * segment, and what makes those come out right.
                             *
                             * Skipping the source ALONE was tried first and
                             * faulted the device, because its view stayed
                             * pinned to the GPU.  The skip above, which drops
                             * any node sitting on unsupported storage, is the
                             * other half: without it this line reintroduces
                             * that fault. */
                            continue;
                        }
                        /* Tell sched about ALL other sources — including weight
                         * tensors that already have ->buffer on GPU.  Without
                         * this, sched doesn't know their backend and may insert
                         * spurious copies or fall back to CPU. */
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
            /* Remember the first node that failed, whatever the reason: a
             * group that declined it (r=-1) or no group claiming it at all.
             * Both leave the output unregistered, and everything downstream
             * then goes unbuilt -- so the first failure is the cause and the
             * rest are consequences.  Without this the empty-graph message
             * can only say "look for r=-1 yourself". */
            const onnx_node_t *fn = &onnx->nodes[i];
            if (c->first_failed_node[0] == '\0' && fn->n_outputs > 0) {
                /* snprintf rather than strncpy: it always terminates.  Both
                 * fields are sized to their source (ONNX_MAX_NAME, op_type[128])
                 * so nothing is truncated -- a field smaller than the source
                 * makes GCC warn about the truncation it would then do. */
                snprintf(c->first_failed_node, sizeof(c->first_failed_node),
                         "%s", fn->outputs[0]);
                snprintf(c->first_failed_op, sizeof(c->first_failed_op),
                         "%s", fn->op_type);
            }
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
/* Drop every deferred-fill registration.
 *
 * The registrations below hold raw tensor pointers recorded while the graph was
 * mapped.  Re-mapping replaces those tensors, which leaves the recorded
 * pointers dangling -- not NULL, just aimed at an arena that has since been
 * reused.  fill_deferred_tensors()'s `!t || !t->buffer` guard lets a dangling
 * pointer straight through and then writes through it, which is a segfault at
 * an address with no obvious connection to this code.
 *
 * Call this on every path that re-maps nodes, BEFORE that run's first
 * map_node_range(), and never on the cached path: there nothing is re-mapped,
 * the previous run's registrations still name the live tensors, and clearing
 * them would leave NonZero outputs holding whatever their buffer contained --
 * the phantom-index failure these fills exist to prevent.
 *
 * Left uncleared the arrays also only ever grow, re-registering the same
 * tensors on each rebuild until ONNX_MAX_DEFERRED silently drops the rest. */
static void reset_deferred_fills(onnx_ggml_ctx_t *c) {
    c->n_shape_tensors = 0;
    c->n_const_fills   = 0;
    c->n_cinit_fills   = 0;
    c->n_nonzero_fills = 0;
    c->n_eye_fills     = 0;
    /* NMS too: it is a deferred fill like the rest (nms_param_tensors holds a
     * params block written before each compute), and leaving it out made this
     * the one list that only ever grew.
     *
     * Measured on MaskRCNN-12-int8 with ONNX_TRACE_NODES, the [fill] counters
     * across one run: nonzero rose and fell with each segment, as a reset list
     * does, while nms went 0,2,4,5,6,7,8,9,10 and never came back. Run 2 then
     * re-registered every NMS on top of run 1's entries, and '2169' started
     * from the [3,1] the previous run's re-map had left instead of its
     * [3,147] -- one box in where 147 belong. The detection branch collapsed
     * behind it (TopK '6565' over 1 candidate, six NonZero measuring 0) and
     * the run died in get_rows with "index 3 out of range [0,1)".
     *
     * Truncating tmap to its post-load baseline does not cover this: these
     * pointers live in their own array and are read by fill_deferred_tensors
     * without consulting tmap at all. */
    c->n_nms_deferred  = 0;
    /* Strided Slice for the same reason: slice_fill_src/dst are registered per
     * map_node and consumed by fill_strided_slices(), so a rebuild that is not
     * cleared first leaves the previous run's source and destination pointers
     * in the list ahead of this run's. Found by listing every n_* counter in
     * the header rather than by hitting it -- the same audit that turned up
     * the NMS one, and the reason to do it as a list instead of one at a
     * time. */
    c->n_slice_fills   = 0;
    /* QConv requant multipliers: its own comment calls it "refilled from
     * fill_deferred_tensors() like every other deferred payload", so it
     * belongs with them here too. */
    c->n_qconv_mult    = 0;

    /* ⚠️ NOT reset here, though they look alike: n_roi_aligns, n_nms_ops and
     * n_qconv_ops count malloc'd parameter blocks whose ADDRESSES were handed
     * to kernels as userdata. Those have to stay valid for the life of the
     * graph, and the pointers are the only record of the allocation -- zeroing
     * the count leaks every block and lets the next registration hand a kernel
     * a record that is being overwritten. They are per-op ownership, not
     * per-run registration; the distinction is what separates this list from
     * that one. */
}

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

    /* Fill NMS param tensors with
     * [max_boxes, iou_thresh, score_thresh, have_score_thresh] */
    for (int i = 0; i < c->n_nms_deferred; i++) {
        struct ggml_tensor *t = c->nms_param_tensors[i];
        if (!t || !t->buffer) continue;
        float params[4];
        params[0] = (float)c->nms_max_boxes[i];
        memcpy(&params[1], &c->nms_iou_thresh[i], sizeof(float));
        memcpy(&params[2], &c->nms_score_thresh[i], sizeof(float));
        params[3] = (float)c->nms_have_score_thresh[i];
        ggml_backend_tensor_set(t, params, 0, 4 * sizeof(float));
    }

    /* Per-channel requantisation multipliers for QCONV_I32. Host-computed at
     * build time; refilled here because segmented execution blanks whatever
     * lives in a scheduler buffer. */
    for (int i = 0; i < c->n_qconv_mult; i++) {
        struct ggml_tensor *t = c->qconv_mult_tensors[i];
        if (!t || !t->buffer || !c->qconv_mult_values[i]) continue;
        ggml_backend_tensor_set(t, c->qconv_mult_values[i], 0,
                                (size_t)c->qconv_mult_n[i] * sizeof(float));
    }
}

/* ── Build full graph ───────────────────────────────────────────── */

onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device, int n_threads,
                                  enum ggml_type model_dtype) {
    onnx_reset_unsupported_warnings();
    onnx_ggml_ctx_t *c = calloc(1, sizeof(onnx_ggml_ctx_t));
    if (!c) return NULL;
    c->onnx = onnx;
    if (onnx_trace_nodes())
        fprintf(stderr, "[value_info] parsed %d declarations\n", onnx->n_value_info);
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

    /* What one tensor costs in a no_alloc context: the ggml_tensor struct plus
     * the ggml_object header ggml charges for every allocation. Used to size
     * the metadata-only contexts below in whole allocations. */
    const size_t tensor_cost = ggml_tensor_overhead();

    /* ctx_weight: separate context for weight tensors (initializers).
     * These get a dedicated GPU buffer that the scheduler never aliases. */
    {
        /* Weight context needs space for tensor metadata only (no_alloc=true).
         * Includes initializers + Constant/Shape/ConstantOfShape/EyeLike/scalar
         * tensors that are also placed here during map_node.
         *
         * Sized from ggml_tensor_overhead() rather than a 512-byte guess, and
         * with room for more than one tensor per node.
         *
         * ⚠️ The old formula was (n_init + n_nodes) * 512 + 64 KiB, and it came
         * up ONE OBJECT short on RoBERTa-SeqClass: "not enough space in the
         * context's memory pool (needed 816704, available 816640)" -- a 64-byte
         * shortfall, exactly sizeof(struct ggml_object), which aborted the model
         * on both backends. One tensor per node is not the real bound either:
         * ctx_weight is written from onnx_ops_nn.c, onnx_ops_special.c,
         * onnx_ops_tensor.c and onnx_ops_quant.c as well as from the initializer
         * pass, and a single node can place several tensors there (a quantised
         * conv contributes its requantisation multiplier on top of its weights).
         *
         * ggml also charges GGML_OBJECT_SIZE for the object header of EVERY
         * allocation and refuses one that would not fit whole, so the reserve
         * has to be a multiple of the per-tensor cost, not a round number of
         * kilobytes. */
        size_t n_weight_tensors = (size_t)onnx->n_initializers
                                + (size_t)onnx->n_nodes * 4;
        size_t weight_meta = (n_weight_tensors + 1024) * tensor_cost;
        struct ggml_init_params wp = {
            .mem_size   = weight_meta,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        c->ctx_weight = ggml_init(wp);
        if (!c->ctx_weight) { free(c); return NULL; }

        /* Host context: same shape as ctx_weight, but allocated on the CPU
         * backend so the tensors in it stay on the host.  Only small parameter
         * blocks that a CPU-only kernel reads belong here -- see the note on
         * ctx_host in onnx_ggml.h for why ggml_set_input() cannot do this job.
         *
         * Two tensors per node, not one: NMS places both its parameter block
         * and its output here, and a re-mapped cut op builds a fresh pair on
         * every run. Sized in units of ggml_tensor_overhead() for the same
         * reason as ctx_weight above -- a reserve that is not a whole number of
         * allocations can leave the pool one 64-byte object header short. */
        struct ggml_init_params hp = {
            .mem_size   = ((size_t)onnx->n_nodes * 2 + 1024) * tensor_cost,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        c->ctx_host = ggml_init(hp);
        if (!c->ctx_host) { ggml_free(c->ctx_weight); free(c); return NULL; }
    }

    /* ctx: context for inputs, graph ops, and intermediate tensors */
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    c->ctx = ggml_init(params);
    if (!c->ctx) { ggml_free(c->ctx_host); ggml_free(c->ctx_weight); free(c); return NULL; }

    /* Each segment gets a pool this size, one at a time.
     *
     * The same estimate the model context uses: a single segment can hold most
     * of the graph, and the pool is freed when the segment ends, so this is a
     * peak rather than a per-segment cost. Undersizing it does not degrade
     * gracefully -- ggml returns NULL tensors that only fail once read. */
    c->seg_ctx_size = mem_size;

    /* ctx_boundary: copies of tensors that outlive their segment.  Metadata
     * only (no_alloc), like ctx_weight -- the data buffer comes later.
     *
     * ⚠️ These ACCUMULATE: a context frees nothing until it is destroyed, so
     * every segment's boundary copies stay for the life of the model, and a
     * re-mapped cut op adds a fresh device mirror per run on top (see the
     * substitution in the segment loop). Budget per segment, not one queue's
     * worth, and in units of ggml_tensor_overhead() so the reserve is a whole
     * number of allocations -- a 64-byte shortfall is enough to abort a model,
     * as the old ctx_weight estimate did on RoBERTa-SeqClass. */
    {
        struct ggml_init_params bp = {
            .mem_size   = ((size_t)ONNX_MAX_BOUNDARY * 2
                           + (size_t)ONNX_MAX_SEGMENTS * 8
                           + 1024) * tensor_cost,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        c->ctx_boundary = ggml_init(bp);
        if (!c->ctx_boundary) { ggml_free(c->ctx); ggml_free(c->ctx_host); ggml_free(c->ctx_weight); free(c); return NULL; }
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
    /* "gpu" is a synonym for "vulkan": the rest of the package spells the
     * device that way (ag_device("gpu")), and an unrecognised name used to
     * fall through to CPU silently -- a benchmark asking for the GPU then
     * reported CPU timings as GPU ones, and a session spent a GPU bug hunt
     * on runs that were never on the GPU. */
    const int want_gpu = device == NULL
                      || strcmp(device, "vulkan") == 0
                      || strcmp(device, "gpu") == 0;
    if (want_gpu) {
#ifdef GGML_USE_VULKAN
        use_vulkan = 1;
#else
        if (device) {
            fprintf(stderr, "onnx_ggml: Vulkan not available, falling back to CPU\n");
        }
#endif
    } else if (device && strcmp(device, "cpu") != 0) {
        /* Anything else is a typo, not a device.  Saying so beats running on
         * a backend the caller did not ask for. */
        fprintf(stderr, "onnx_ggml: unknown device '%s' "
                        "(expected \"cpu\", \"gpu\" or \"vulkan\")\n", device);
        goto fail;
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

        /* Always the CPU backend, whatever the weights went to: the whole
         * point of ctx_host is that these tensors are read by host-only
         * kernels and must not cost a split input slot each. Returns NULL when
         * nothing was placed there, which is the normal case for a model with
         * no CPU-only custom ops. */
        c->host_buf = ggml_backend_alloc_ctx_tensors(c->ctx_host, c->backend_cpu);
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

#define ONNX_RING_N     8
#define ONNX_RING_EDGE  6

typedef struct {
    char    name[GGML_MAX_NAME];
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

/* ONNX_TRACE_VALS: print every node's output as it is computed, rather than
 * keeping the last few for a post-mortem.
 *
 * The ring exists because a corrupting run buries the interesting nodes under
 * thousands of lines.  That is the right default, but it answers only "what
 * did the last few nodes hold"; when the question is where a value first
 * departs from the reference -- comparing a whole run against ONNX Runtime,
 * say -- the whole sequence is the point, and it belongs in a file rather
 * than a terminal. */
int onnx_trace_vals(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ONNX_TRACE_VALS");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

/* ONNX_DUMP_NODES=<name>[,<name>...]: write those nodes' contents in full to
 * ONNX_DUMP_DIR, as little-endian float32 with no header.
 *
 * ONNX_TRACE_VALS cannot answer this.  It prints three values per node, from
 * the tensor's edges, deliberately: a MaskRCNN node can hold millions of
 * elements and pulling all of them back for every node would move the timing.
 * That is the right trade for "where does a run first go wrong", and the
 * wrong one for "do these 594 boxes match the reference" -- three numbers
 * from the edge of a list say nothing about element 300, and a value read
 * off that edge has already been mistaken for a difference once.
 *
 * So: whole tensors, but only for nodes named on the way in.  The names are
 * ONNX edge names, the same ones a reference run can be asked to return --
 * a ggml-internal name like "node_59 (cont)" has no counterpart to compare
 * against, and is rejected rather than silently dumped.
 *
 * Everything is written as f32, indices included: an I32 index fits a float
 * exactly at these sizes, and one dtype keeps the comparison one code path. */
#define ONNX_DUMP_MAX 64

static int g_dump_n = -1;
static char g_dump_names[ONNX_DUMP_MAX][ONNX_MAX_NAME];
static const char *g_dump_dir = NULL;

/* How many times each requested name has been computed this run.
 *
 * An ONNX edge can be computed more than once: MaskRCNN's graph is cut into
 * segments and a name near a cut shows up in two of them, with the same value
 * both times.  Only the first is written and the rest are counted --
 * overwriting would silently hand the comparison whichever copy ran last, and
 * that is a different question from the one being asked. */
static int g_dump_seen[ONNX_DUMP_MAX];

static void onnx_dump_init(void) {
    /* The seen-counts reset on every call, the name list is parsed once: a
     * second onnx_run() in the same process is a second run and should write
     * its own dump, not be mistaken for a repeat of the first. */
    memset(g_dump_seen, 0, sizeof(g_dump_seen));
    if (g_dump_n >= 0) return;
    g_dump_n = 0;
    const char *e = getenv("ONNX_DUMP_NODES");
    if (!e || !*e) return;
    g_dump_dir = getenv("ONNX_DUMP_DIR");
    if (!g_dump_dir || !*g_dump_dir) g_dump_dir = ".";

    const char *p = e;
    while (*p && g_dump_n < ONNX_DUMP_MAX) {
        const char *q = strchr(p, ',');
        size_t len = q ? (size_t)(q - p) : strlen(p);
        while (len > 0 && (*p == ' ' || *p == '\t')) { p++; len--; }
        while (len > 0 && (p[len-1] == ' ' || p[len-1] == '\t')) len--;
        if (len > 0 && len < ONNX_MAX_NAME) {
            memcpy(g_dump_names[g_dump_n], p, len);
            g_dump_names[g_dump_n][len] = '\0';
            g_dump_n++;
        }
        if (!q) break;
        p = q + 1;
    }
    fprintf(stderr, "[dump] %d node(s) requested, dir=%s\n", g_dump_n, g_dump_dir);
}

/* ONNX_INJECT_NODES=<name>[,<name>...] with ONNX_INJECT_DIR: after a named
 * node is computed, overwrite its contents with <dir>/<name>.ort.bin -- the
 * same flat little-endian f32 the reference runner writes.
 *
 * The mirror image of ONNX_DUMP_NODES, and it answers a question dumping
 * cannot: whether a disagreement downstream is ONE divergence being amplified
 * or several independent ones.  Dumping says two runs differ at node N and
 * again at node M; only replacing N's output with the reference's says
 * whether M would have agreed had N been right.
 *
 * Scope is deliberately small: f32 tensors, element count must match, and the
 * file is read once per node per run.  A mismatch is refused loudly rather
 * than padded or truncated, because a silent partial overwrite would answer
 * the question wrongly and look like a result. */
#define ONNX_INJECT_MAX 32

static int  g_inject_n = -1;
static char g_inject_names[ONNX_INJECT_MAX][ONNX_MAX_NAME];
static const char *g_inject_dir = NULL;
static int  g_inject_done[ONNX_INJECT_MAX];

static void onnx_inject_init(void) {
    if (g_inject_n >= 0) { memset(g_inject_done, 0, sizeof(g_inject_done)); return; }
    g_inject_n = 0;
    memset(g_inject_done, 0, sizeof(g_inject_done));
    const char *e = getenv("ONNX_INJECT_NODES");
    if (!e || !*e) return;
    g_inject_dir = getenv("ONNX_INJECT_DIR");
    if (!g_inject_dir || !*g_inject_dir) g_inject_dir = ".";

    const char *p = e;
    while (*p && g_inject_n < ONNX_INJECT_MAX) {
        const char *q = strchr(p, ',');
        size_t len = q ? (size_t)(q - p) : strlen(p);
        while (len > 0 && (*p == ' ' || *p == '\t')) { p++; len--; }
        while (len > 0 && (p[len-1] == ' ' || p[len-1] == '\t')) len--;
        if (len > 0 && len < ONNX_MAX_NAME) {
            memcpy(g_inject_names[g_inject_n], p, len);
            g_inject_names[g_inject_n][len] = '\0';
            g_inject_n++;
        }
        if (!q) break;
        p = q + 1;
    }
    if (g_inject_n > 0)
        fprintf(stderr, "[inject] %d node(s) requested, dir=%s\n",
                g_inject_n, g_inject_dir);
}

/* Whether any node is being replaced.  The eval callback is installed only
 * when something asks for it, and injection has to be on that list: without
 * it the requested names are parsed, the banner prints, and nothing is ever
 * replaced -- which reads exactly like "the substitution made no difference"
 * and would have been believed. */
static int onnx_inject_active(void) {
    return g_inject_n > 0;
}

static int onnx_inject_wanted(const char *name) {
    if (g_inject_n <= 0 || !name || !*name) return -1;
    for (int i = 0; i < g_inject_n; i++)
        if (strcmp(g_inject_names[i], name) == 0) return i;
    return -1;
}

/* Overwrite one computed tensor with the reference's values. */
static void onnx_inject_tensor(struct ggml_tensor *t) {
    int64_t nel = ggml_nelements(t);
    if (nel <= 0 || !t->buffer) return;
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "[inject] '%s': type %s is not F32 -- skipped\n",
                t->name, ggml_type_name(t->type));
        return;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.ort.bin", g_inject_dir, t->name);
    for (char *s = path + strlen(g_inject_dir) + 1; *s; s++)
        if (*s == '/') *s = '_';

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[inject] '%s': cannot open %s\n", t->name, path); return; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (bytes != (long)(nel * (int64_t)sizeof(float))) {
        fprintf(stderr, "[inject] '%s': %s holds %ld bytes, tensor wants %lld "
                        "-- refused\n", t->name, path, bytes,
                (long long)(nel * (int64_t)sizeof(float)));
        fclose(f);
        return;
    }

    float *buf = (float *)malloc((size_t)bytes);
    if (!buf) { fclose(f); return; }
    size_t got = fread(buf, 1, (size_t)bytes, f);
    fclose(f);
    if (got != (size_t)bytes) { free(buf); return; }

    ggml_backend_tensor_set(t, buf, 0, (size_t)bytes);
    fprintf(stderr, "[inject] '%s': replaced %lld values from the reference\n",
            t->name, (long long)nel);
    free(buf);
}

/* Whether any node was requested.
 *
 * This does NOT initialise: it used to, and the initialisation was then lost
 * to short-circuit evaluation.  The setup sites read
 *   trace_ring() || trace_live() || trace_vals() || dump_active()
 * so turning on any trace flag made the earlier term true and dump_active()
 * was never called -- the dump silently did nothing, but only in combination
 * with a trace, which is exactly when someone is looking at something else.
 * onnx_dump_init() is called on its own before the predicate instead. */
static int onnx_dump_active(void) {
    return g_dump_n > 0;
}

/* Which requested node this is, or -1. */
static int onnx_dump_wanted(const char *name) {
    if (g_dump_n <= 0 || !name || !*name) return -1;
    for (int i = 0; i < g_dump_n; i++)
        if (strcmp(g_dump_names[i], name) == 0) return i;
    return -1;
}

/* Pull one tensor back from the backend and write it as f32.
 *
 * Appends a manifest row rather than a header in the file: the comparison
 * side already reads flat f32 plus a manifest, and keeping that shape means
 * ref_compare_ops.R needs no second reader. */
static void onnx_dump_tensor(const struct ggml_tensor *t) {
    int64_t nel = ggml_nelements(t);
    if (nel <= 0 || !t->buffer) return;

    size_t nbytes = ggml_nbytes(t);
    void *raw = malloc(nbytes);
    if (!raw) return;
    ggml_backend_tensor_get((struct ggml_tensor *)t, raw, 0, nbytes);

    float *out = (float *)malloc((size_t)nel * sizeof(float));
    if (!out) { free(raw); return; }

    int ok = 1;
    switch (t->type) {
        case GGML_TYPE_F32:
            memcpy(out, raw, (size_t)nel * sizeof(float));
            break;
        case GGML_TYPE_I32:
            for (int64_t i = 0; i < nel; i++) out[i] = (float)((int32_t *)raw)[i];
            break;
        case GGML_TYPE_I64:
            for (int64_t i = 0; i < nel; i++) out[i] = (float)((int64_t *)raw)[i];
            break;
        case GGML_TYPE_F16:
            for (int64_t i = 0; i < nel; i++)
                out[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)raw)[i]);
            break;
        default:
            ok = 0;
            break;
    }

    if (ok) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.ggmlr.bin", g_dump_dir, t->name);
        /* '/' in an ONNX edge name would open a path; keep it to one file. */
        for (char *s = path + strlen(g_dump_dir) + 1; *s; s++)
            if (*s == '/') *s = '_';
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(out, sizeof(float), (size_t)nel, f);
            fclose(f);
            char mpath[1024];
            snprintf(mpath, sizeof(mpath), "%s/nodes.tsv", g_dump_dir);
            FILE *mf = fopen(mpath, "a");
            if (mf) {
                fprintf(mf, "%s\t%lld\t%lld,%lld,%lld,%lld\t%s\n",
                        t->name, (long long)nel,
                        (long long)t->ne[0], (long long)t->ne[1],
                        (long long)t->ne[2], (long long)t->ne[3],
                        ggml_type_name(t->type));
                fclose(mf);
            }
            fprintf(stderr, "[dump] '%s' n=%lld ne=[%lld,%lld,%lld,%lld]\n",
                    t->name, (long long)nel,
                    (long long)t->ne[0], (long long)t->ne[1],
                    (long long)t->ne[2], (long long)t->ne[3]);
        }
    } else {
        fprintf(stderr, "[dump] '%s' skipped: type %s not convertible\n",
                t->name, ggml_type_name(t->type));
    }

    free(out);
    free(raw);
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

/* ONNX_TRACE_LIVE=1: name every node on stderr as it is about to run.
 *
 * The ring holds the same information and reads far better, but it is printed
 * from the abort hook, and a run that dies of a plain segfault never reaches
 * an abort: the ring is written and then lost with the process.  This prints
 * unbuffered, before the node runs, so the last line to appear IS the node
 * that killed the run -- the one question the ring cannot answer for a
 * segfault.
 *
 * Beware of reading its silence as good news: it reads every node's edges
 * through ggml_backend_tensor_get, which perturbs allocator and memory-access
 * order, and a run that dies without it can survive with it.  That happened
 * here -- the dangling NMS userdata (onnx_ops_special.c) only crashed with the
 * trace off.  A bug that disappears under this flag has been masked, not
 * fixed, and the difference itself is the clue: it points at memory whose
 * lifetime is wrong. */
static int onnx_trace_live(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("ONNX_TRACE_LIVE");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

static bool onnx_ring_eval_cb(struct ggml_tensor *t, bool ask, void *user_data) {
    int *idx = (int *)user_data;

    if (ask) {
        if (onnx_trace_live()) {
            fprintf(stderr, "[live] %d '%s' op=%d(%s) type=%s ne=[%lld,%lld,%lld,%lld]"
                            " data=%p buf=%p\n",
                    *idx, t->name, (int)t->op, ggml_op_name(t->op),
                    ggml_type_name(t->type),
                    (long long)t->ne[0], (long long)t->ne[1],
                    (long long)t->ne[2], (long long)t->ne[3],
                    (void *)t->data, (void *)t->buffer);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (!t->src[s]) continue;
                fprintf(stderr, "[live]     src%d '%s' op=%s type=%s "
                                "ne=[%lld,%lld,%lld,%lld] data=%p\n",
                        s, t->src[s]->name, ggml_op_name(t->src[s]->op),
                        ggml_type_name(t->src[s]->type),
                        (long long)t->src[s]->ne[0], (long long)t->src[s]->ne[1],
                        (long long)t->src[s]->ne[2], (long long)t->src[s]->ne[3],
                        (void *)t->src[s]->data);
            }
            fflush(stderr);
        }
        /* Record the srcs, so the ring shows what this node was fed. */
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (!t->src[s]) continue;
            onnx_ring_record(t->src[s], *idx, 0);
        }
        return true;
    }
    onnx_ring_record(t, *idx, 1);

    {
        int di = onnx_dump_wanted(t->name);
        if (di >= 0) {
            if (g_dump_seen[di]++ == 0) {
                onnx_dump_tensor(t);
            } else {
                fprintf(stderr, "[dump] '%s' seen again (#%d), keeping the first\n",
                        t->name, g_dump_seen[di]);
            }
        }
    }

    /* Injection goes AFTER the dump on purpose: the dump then still records
     * what this build actually computed, and only the consumers downstream
     * see the reference's values.  Both can therefore run in one pass. */
    {
        int ii = onnx_inject_wanted(t->name);
        if (ii >= 0 && g_inject_done[ii]++ == 0)
            onnx_inject_tensor((struct ggml_tensor *)t);
    }

    if (onnx_trace_vals()) {
        /* The first three values of what this node just produced.  Integer
         * types print as integers: these carry shapes and indices, and %g
         * turning 128 into 1.28e+02 is exactly the wrong rendering for a
         * value whose whole meaning is which element it selects. */
        double v[ONNX_RING_EDGE], tail[ONNX_RING_EDGE];
        int nv = onnx_ring_read_edges(t, v, tail);
        fprintf(stderr, "[val] %d '%s' op=%s %s ne=[%lld,%lld,%lld,%lld] :",
                *idx, t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
        if (!nv) {
            fprintf(stderr, " <not readable>");
        } else {
            int n_show = nv < 3 ? nv : 3;
            for (int i = 0; i < n_show; i++) {
                if (t->type == GGML_TYPE_I32)
                    fprintf(stderr, " %lld", (long long)v[i]);
                else
                    fprintf(stderr, " %g", v[i]);
            }
        }
        /* ONNX_TRACE_SUM=1 adds a checksum over the WHOLE tensor.
         *
         * Three leading values are not evidence that two tensors are equal:
         * comparing backends on them called a 14x14x256 input identical when
         * its first three elements merely happened to be zero, and sent a
         * diagnosis in the wrong direction twice. This reads every element.
         *
         * Sum, min and max rather than a hash: a hash says only "differs",
         * while these say by how much and where to look. Sum is in double so
         * it does not itself lose the difference it is meant to expose. */
        if (onnx_trace_sum() && t->data) {
            const int64_t nel = ggml_nelements(t);
            const size_t  esz = ggml_type_size(t->type);
            if ((t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_I32) &&
                ggml_is_contiguous(t) && nel > 0) {
                void *buf = malloc((size_t)nel * esz);
                if (buf) {
                    ggml_backend_tensor_get((struct ggml_tensor *)t, buf, 0,
                                            (size_t)nel * esz);
                    double sum = 0.0, mn, mx;
                    if (t->type == GGML_TYPE_F32) {
                        const float *f = (const float *)buf;
                        mn = mx = f[0];
                        for (int64_t i = 0; i < nel; i++) {
                            sum += f[i];
                            if (f[i] < mn) mn = f[i];
                            if (f[i] > mx) mx = f[i];
                        }
                    } else {
                        const int32_t *q = (const int32_t *)buf;
                        mn = mx = q[0];
                        for (int64_t i = 0; i < nel; i++) {
                            sum += (double)q[i];
                            if (q[i] < mn) mn = q[i];
                            if (q[i] > mx) mx = q[i];
                        }
                    }
                    fprintf(stderr, " | n=%lld sum=%.9g min=%g max=%g",
                            (long long)nel, sum, mn, mx);
                    free(buf);
                }
            }
        }
        fprintf(stderr, "\n");
    }

    /* ONNX_TRACE_MAX=<name>: per-row maximum of one named 2D tensor.
     *
     * The edge trace above prints three leading values, which answers "what
     * is in here" but not "how big does it get" -- and a threshold question
     * is entirely about the latter.  MaskRCNN filters an [n_class, n_cand]
     * score tensor against 0.05 and keeps one branch per class; whether the
     * classes that kept nothing were far below the threshold or sitting just
     * under it is the difference between a structural fault and quantisation
     * noise, and neither the first three values nor the final detections can
     * tell the two apart.
     *
     * Off unless asked for by name: it reads the whole tensor back from the
     * backend, which is the cost the edge reader exists to avoid. */
    {
        static const char *want = NULL;
        static int want_init = 0;
        if (!want_init) { want = getenv("ONNX_TRACE_MAX"); want_init = 1; }
        if (want && *want && t->name[0] && strstr(t->name, want) &&
            t->type == GGML_TYPE_F32 && t->buffer && t->data &&
            ggml_is_contiguous(t) && t->ne[0] > 0 && t->ne[1] > 0 &&
            t->ne[2] == 1 && t->ne[3] == 1) {
            /* ONNX_TRACE_MAX_DIR=<dir>: also write the whole tensor there as
             * raw float32, named "<dir>/<tensor>.bin".  Aggregates answer
             * "how big does it get"; replaying an algorithm against the very
             * numbers it ran on needs all of them. */
            const char *dumpdir = getenv("ONNX_TRACE_MAX_DIR");
            if (dumpdir && *dumpdir) {
                char path[1024];
                snprintf(path, sizeof(path), "%s/%s.bin", dumpdir, t->name);
                FILE *f = fopen(path, "wb");
                if (f) {
                    int64_t nel = ggml_nelements(t);
                    float *all = (float *)malloc((size_t)nel * sizeof(float));
                    if (all) {
                        ggml_backend_tensor_get((struct ggml_tensor *)t, all, 0,
                                                (size_t)nel * sizeof(float));
                        fwrite(all, sizeof(float), (size_t)nel, f);
                        free(all);
                        fprintf(stderr, "[max] wrote %lld floats to %s\n",
                                (long long)nel, path);
                    }
                    fclose(f);
                }
            }
            /* Aggregate down ne[0], not across it.  The tensor is
             * [n_class, n_cand] in ggml order, so a contiguous row is one
             * candidate's distribution over classes -- and the threshold
             * question is the other way round: for each class, how high did
             * any candidate get.  Reducing the wrong way gives a per-candidate
             * softmax profile, which looks informative and answers nothing. */
            int64_t ncls = t->ne[0], ncand = t->ne[1];
            float *row = (float *)malloc((size_t)ncls * sizeof(float));
            double *mx = (double *)malloc((size_t)ncls * sizeof(double));
            int64_t *n_over = (int64_t *)calloc((size_t)ncls, sizeof(int64_t));
            if (row && mx && n_over) {
                for (int64_t k = 0; k < ncls; k++) mx[k] = -1e30;
                for (int64_t r = 0; r < ncand; r++) {
                    ggml_backend_tensor_get((struct ggml_tensor *)t, row,
                                            (size_t)r * ncls * sizeof(float),
                                            (size_t)ncls * sizeof(float));
                    for (int64_t k = 0; k < ncls; k++) {
                        if (row[k] > mx[k]) mx[k] = row[k];
                        if (row[k] > 0.05f) n_over[k]++;
                    }
                }
                fprintf(stderr, "[max] '%s' ne=[%lld,%lld] per-class over %lld candidates:\n",
                        t->name, (long long)ncls, (long long)ncand, (long long)ncand);
                for (int64_t k = 0; k < ncls; k++)
                    fprintf(stderr, "[max]   class %lld  max=%.8g  n>0.05=%lld\n",
                            (long long)k, mx[k], (long long)n_over[k]);
                fflush(stderr);
            }
            free(row); free(mx); free(n_over);
        }
    }

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

    /* Strided Slice outputs whose source is a model input.  This runs here,
     * after the upload above, because the copy reads the source with
     * ggml_backend_tensor_get -- at sched-alloc time that source still holds
     * whatever the buffer happened to contain.  Sources that are computed by
     * the graph are not fixed by this and take the graph path in
     * onnx_ops_tensor.c instead; what remains here is diagnosed there. */
    fill_strided_slices(ctx);

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
    onnx_dump_init();
    onnx_inject_init();
    if (onnx_trace_ring() || onnx_trace_live() || onnx_trace_vals() || onnx_dump_active() || onnx_inject_active())
        onnx_ring_attach(c->seg_scheds[s]);
    if (onnx_trace_nodes())
        fprintf(stderr, "[segcache] new scheduler for segment %d\n", s);
    return c->seg_scheds[s];
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
    /* Release the last segment of the PREVIOUS run.
     *
     * That pool was kept open so the caller could read the model's outputs out
     * of it after onnx_ggml_run() returned. By now it has: control is back
     * here for the next inference, and r_onnx.c copied the values into R
     * vectors before that. Freeing it here rather than at the end of the run
     * is what keeps segment pools from accumulating across inferences while
     * still letting the outputs survive the call that produced them. */
    segment_ctx_end(ctx);

    /* Forget the sizes measured by the previous run.
     *
     * These are measurements of THIS input, not properties of the model: how
     * many boxes NMS kept, how many elements NonZero found, how many
     * candidates a TopK had to rank. Carried into the next run they stop being
     * measurements and become predictions -- and a shape-dependent op builds
     * its output from them before anything has been measured again.
     *
     * Measured on MaskRCNN-12-int8 with ONNX_TRACE_NODES. NMS '2169' has
     * capacity 147 in both runs, so its inputs are identical; the trace reads
     *   run 1:  [NMS] 2169: capacity 147 -> measured 1 selected   (at the re-map,
     *           after the op ran and one box survived)
     *   run 2:  [NMS] 2169: capacity 147 -> measured 1 selected   (at the FIRST
     *           mapping of segment 9, before this run has measured anything)
     * so run 2 built a [3,1] output where 147 belong. Everything downstream
     * collapsed with it -- TopK '6565' ranking a single candidate, six NonZero
     * measuring 0 -- while the indices reaching them were computed at the real
     * sizes, and get_rows aborted with "index 3 out of range [0,1)". On Vulkan
     * the same mismatch is not checked: it read past the buffer and took the
     * device down with a GPUVM fault.
     *
     * Only the sizes are dropped, and they cost nothing to rediscover:
     * resolve_segment_sizes() takes them again after each segment computes.
     * Until it does, a cut op builds at its spec-derived capacity, which is
     * exactly what the first run does. */
    if (ctx->n_resolved > 0) {
        if (onnx_trace_nodes())
            fprintf(stderr, "[resolve] run start: forgetting %d measured "
                            "size%s from the previous run\n",
                    ctx->n_resolved, ctx->n_resolved == 1 ? "" : "s");
        ctx->n_resolved = 0;
    }

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

        /* Before the allocation, for the same reason as in the segment loop:
         * this is the last moment segment 0's nodes still point at the tensors
         * its graph was built from.  A cached run does not come through here —
         * its graph was allocated by the run that built it, and the restore at
         * the end of that run's segment 0 already put these pointers back. */
        snapshot_graph_srcs(ctx);
        if (sched_alloc_and_fill(ctx) != 0) return -1;
        ctx->is_allocated = 1;
    }

    /* Segment 0's graph was mapped during build and its registrations are the
     * ones just used above, so nothing is dropped here; the reset that matters
     * is the one guarding the rebuild path in the segment loop below. */

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
    onnx_dump_init();
    onnx_inject_init();
    if (onnx_trace_ring() || onnx_trace_live() || onnx_trace_vals() || onnx_dump_active() || onnx_inject_active()) {
        g_ring_node_idx = 0;
        g_nan_reported = 0;
        r_ggml_abort_hook = onnx_ring_dump;
        onnx_ring_attach(ctx->sched);
    }


    if (onnx_trace_nodes())
        fprintf(stderr, "[segment] %d: computing %d graph nodes (first compute)\n",
                ctx->cur_segment, ggml_graph_n_nodes(ctx->graph));

    enum ggml_status status = ggml_backend_sched_graph_compute(ctx->sched, ctx->graph);

    /* Segment 0 has computed; put its src[] back before any later segment is
     * built over the same tensors. */
    restore_graph_srcs(ctx);

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
        /* Every segment is rebuilt on every run.
         *
         * Caching the per-segment graphs was tried and removed. The cache held
         * the FIRST run's tensor objects while the scheduler went on handing
         * their memory to later allocations, so a cached segment computed over
         * whatever had landed there -- on Vulkan that ended in
         * GGML_ASSERT(buffer != nullptr) on the second run. Keeping the graphs
         * alive meant keeping their context alive, and a ggml context frees
         * nothing until it is destroyed, so the pool grew by one full set of
         * segment graphs per inference: a model loaded once for a few hundred
         * predictions ran out of context memory. The cache also could not tell
         * a verified size from an unverified one -- resolve_segment_sizes()
         * skips a tensor whose buffer is gone, and the accumulating size list
         * then kept the previous run's number, so the key matched on figures
         * the run had never measured (19 of 204 actually read on MaskRCNN).
         *
         * Rebuilding costs the map and allocate work on every run and buys a
         * segment graph whose tensors are the ones this run allocated. */
        int fills_reset = 0;

        for (int s = 1; s < ctx->n_segments; s++) {
            /* The segment just computed has given its cut ops real inputs, so
             * their true output sizes can be measured now. */
            if (resolve_segment_sizes(ctx, s - 1) != 0) return -1;

            /* Everything from here re-maps nodes, so the registrations held
             * from the previous run are about to be replaced by new tensors.
             * Drop them before the first map_node_range() below rather than
             * after: the ones recorded from here on belong to this run. */
            if (!fills_reset) {
                reset_deferred_fills(ctx);
                fills_reset = 1;
                if (onnx_trace_nodes())
                    fprintf(stderr, "[fill] reset deferred registrations "
                                    "at segment %d (rebuild)\n", s);
            }

            /* Re-map the cut ops themselves.  They were first mapped as part
             * of their own segment, necessarily before their inputs existed,
             * so they were built at the guessed size; mapping them again now
             * rebuilds them at the measured one.  tmap_put_nd appends and
             * tmap_get searches backwards, so the new tensor simply shadows
             * the old for every consumer that follows. */


            for (int j = 0; j < ctx->segments[s - 1].n_cut_nodes; j++) {
                int ni = ctx->segments[s - 1].cut_nodes[j];
                const char *onm = ctx->onnx->nodes[ni].outputs[0];
                int64_t rs = onnx_resolved_size(ctx, onm);
                /* Trace the re-map, not just its inputs.
                 *
                 * On Vulkan six NMS nodes keep their guessed shape while the
                 * size measured for them is right, so the fault lies between
                 * "measured" and "rebuilt" -- and every step in between is
                 * silent.  Printing the measured size, whether the node was
                 * skipped, and the shape tmap holds before and after says
                 * which of them it is: no measurement, no call, or a call
                 * whose new tensor does not end up shadowing the old one. */
                if (getenv("ONNX_TRACE_REMAP")) {
                    struct ggml_tensor *before = tmap_get(ctx, onm);
                    fprintf(stderr,
                        "[remap] seg %d cut %d '%s' op=%s resolved=%lld "
                        "before=[%lld,%lld] %s\n",
                        s, j, onm, ctx->onnx->nodes[ni].op_type, (long long)rs,
                        before ? (long long)before->ne[0] : -1,
                        before ? (long long)before->ne[1] : -1,
                        rs < 0 ? "-> SKIP (not measured)" : "-> remapping");
                }
                if (rs < 0)
                    continue;   /* nothing measured -- leave the original */

                /* ⚠️ Save the computed values before the re-map abandons them.
                 *
                 * map_node_range() below rebuilds this cut op at the measured
                 * size and shadows the old tensor in tmap. NonZero, TopK,
                 * Shape and EyeLike need no VALUES carried across it: their
                 * outputs are deferred fills, rewritten after every
                 * allocation.
                 *
                 * ⚠️ "Rewritten after every allocation" holds only while the
                 * registration naming them survives, and for a cut op at a
                 * segment boundary it did not: see the ctx_weight switch and
                 * the input hold below, both of which exist to keep the entry
                 * valid past the pool release. Read this paragraph as "no
                 * value carry needed", not as "nothing needed".
                 * NonMaxSuppression is the exception -- only its PARAMS are
                 * deferred; the output is an ordinary ggml_custom_4d node that
                 * nms_cpu evaluated when its own segment ran, and nothing
                 * evaluates the rebuilt one. Its buffer arrives empty and
                 * every later read gets whatever it happens to hold.
                 *
                 * Measured on MaskRCNN-12-int8, node 1170: rebuilt
                 * [3,1000] -> [3,979], and the second read came back as
                 * 1000,1,4 instead of 0,0,0 on Vulkan -- box indices 0,1,2
                 * became 0,3,6 and, through Gather -> ArgSort -> TopK -> box
                 * decode, 51 detections became 100. On CPU the abandoned
                 * buffer still held the right bytes, so only Vulkan showed it.
                 *
                 * ⚠️ GGML_OP_CUSTOM only. The first version tested
                 * `op != GGML_OP_NONE` ("anything the graph computes"), which
                 * also caught TopK -- its output reaches tmap through a
                 * RESHAPE -- and writing stale values over those deferred
                 * fills corrupted the run. Widen this only with a measurement
                 * showing another op in the same position. */
                struct ggml_tensor *old_out = tmap_get(ctx, onm);
                void  *carry_buf   = NULL;
                size_t carry_bytes = 0;
                const int carry_op = old_out && old_out->buffer &&
                                     old_out->op == GGML_OP_CUSTOM;
                if (carry_op) {
                    carry_bytes = ggml_nbytes(old_out);
                    carry_buf   = malloc(carry_bytes);
                    if (carry_buf)
                        ggml_backend_tensor_get(old_out, carry_buf, 0, carry_bytes);
                }

                /* Hold a rebuilt TopK's INPUT before rebuilding over it.
                 *
                 * TopK is the one cut op that has to be recomputed after the
                 * re-map (see the leaf note below), so it is the only one whose
                 * rebuilt chain -- argsort_top_k, the cont it needs because
                 * argsort_top_k returns a view, the cast/arange/get_rows the
                 * value gather gathers with -- is evaluated by a LATER
                 * segment's graph. Those nodes read this segment's input
                 * tensor, and this segment's pool is released a few lines down.
                 *
                 * Measured on MaskRCNN-12-int8. Holding the OUTPUT instead (the
                 * rebuilt node in ctx_weight) was tried first: '1908' did stay
                 * in tmap past the boundary, and the run still died -- CPU at
                 * ops-elemwise.cpp:351, GGML_ASSERT(ggml_nelements(dst) ==
                 * ggml_nelements(src0)), Vulkan with SIGSEGV inside
                 * ggml_vk_cpy -> ggml_vk_op_f32. Both land on the CPY that
                 * ggml_cont emits, the first node of the chain to touch the
                 * input, and the assert says it plainly: dst is sized at the
                 * new k while src0 points into memory that has been handed out
                 * again. The output was never the part that had to survive.
                 *
                 * The copy goes in ctx_weight (allocated on backend_gpu when
                 * there is one, so the chain stays on the backend that will
                 * compute it) and is substituted in tmap only for the duration
                 * of map_node_range, then put back.
                 *
                 * ⚠️ The restore is not optional. copy_segment_boundaries()
                 * carries a warning from an earlier attempt to hold cut-op
                 * inputs by copying them there: a copy REPLACES its original in
                 * tmap, so everything built afterwards is repointed at it, and
                 * MaskRCNN went from 68 detections to 134 with "TopK 6565:
                 * K=100 exceeds the 51 elements on the ranked axis". Here the
                 * substitution is visible to exactly one map_node_range call
                 * and to nothing else, which is what keeps that from happening.
                 *
                 * A failed copy is not fatal: it leaves the old behaviour (the
                 * rebuild over a pool tensor), which is no worse than not
                 * trying. */
                struct ggml_tensor *topk_in_saved = NULL;
                const char *topk_in_name = NULL;
                int topk_in_nd = 0;
                if ((strcmp(ctx->onnx->nodes[ni].op_type, "TopK")    == 0 ||
                     strcmp(ctx->onnx->nodes[ni].op_type, "NonZero") == 0) &&
                    ctx->ctx_weight && ctx->onnx->nodes[ni].n_inputs > 0) {
                    const char *inm = ctx->onnx->nodes[ni].inputs[0];
                    struct ggml_tensor *in = tmap_get(ctx, inm);
                    if (in && in->buffer) {
                        struct ggml_tensor *hold =
                            ggml_new_tensor(ctx->ctx_weight, in->type,
                                            GGML_MAX_DIMS, in->ne);
                        if (hold) {
                            ggml_set_input(hold);
                            ggml_set_name(hold, inm);
                            ggml_backend_t bk = ctx->backend_gpu
                                              ? ctx->backend_gpu : ctx->backend_cpu;
                            ggml_backend_buffer_t hb =
                                ggml_backend_alloc_ctx_tensors(ctx->ctx_weight, bk);
                            if (hb) {
                                if (ctx->n_extra_weight_bufs >= ONNX_MAX_WEIGHT_BUFS) {
                                    fprintf(stderr, "[onnx] too many weight buffers "
                                                    "(>%d)\n", ONNX_MAX_WEIGHT_BUFS);
                                    ggml_backend_buffer_free(hb);
                                    free(carry_buf);
                                    return -1;
                                }
                                ctx->extra_weight_bufs[ctx->n_extra_weight_bufs++] = hb;
                            }
                            if (hold->buffer) {
                                size_t nb = ggml_nbytes(in);
                                void *tmp = malloc(nb);
                                if (tmp) {
                                    ggml_backend_tensor_get(in, tmp, 0, nb);
                                    ggml_backend_tensor_set(hold, tmp, 0, nb);
                                    free(tmp);
                                    /* Read the rank before the put: tmap_get_ndims
                                     * searches backwards and would find the new
                                     * entry instead of the original's. */
                                    topk_in_nd    = tmap_get_ndims(ctx, inm);
                                    topk_in_saved = in;
                                    topk_in_name  = inm;
                                    tmap_put_nd(ctx, inm, hold, topk_in_nd);
                                    if (getenv("ONNX_TRACE_REMAP"))
                                        fprintf(stderr, "[remap]   held input '%s' "
                                                "(%zu bytes) in ctx_weight\n", inm, nb);
                                }
                            }
                        }
                    }
                }

                /* Build the rebuilt chain in ctx_weight too, not just over a
                 * held input.
                 *
                 * Holding the input alone was measured and is not enough: the
                 * rebuilt TopK still lands in c->ctx, which is the outgoing
                 * segment's pool while this loop runs, and segment_ctx_release()
                 * drops every tmap entry whose tensor lives inside it. '1908'
                 * therefore vanished from the map exactly as before the input
                 * was held, and segment 3's Gather '2099' got a NULL input --
                 * dets=NA, all four outputs unbuilt, on both backends.
                 *
                 * The two halves close different failures and neither is
                 * sufficient alone, which is why both are here:
                 *   - the held input above keeps the chain's src[] off memory
                 *     the pool is about to hand out (measured: CPU assert at
                 *     ops-elemwise.cpp:351, Vulkan SIGSEGV in ggml_vk_cpy);
                 *   - this switch keeps the chain's own nodes, and the tmap
                 *     entries naming them, out of the pool in the first place.
                 *
                 * ctx_weight rather than ctx_host: the next segment's graph
                 * computes these nodes like any others, so they belong on the
                 * backend that graph runs on. alloc_new_weight_tensors() runs
                 * once per segment and sizes exactly the tensors added since
                 * the last call, so the buffer arrives without new machinery.
                 *
                 * ⚠️ NonZero needs this switch too, and for a reason TopK's
                 * account does not cover -- it is not "recomputed later", it
                 * is never computed at all. Its output is a LEAF
                 * (ggml_set_input, op == GGML_OP_NONE) whose only source of
                 * values is fill_deferred_tensors(), so losing the fill loses
                 * the tensor; a rebuilt TopK merely gets evaluated again by
                 * the next graph, which is why the pool never hurt it.
                 *
                 * Measured on MaskRCNN-12-int8 (CPU, first run), NonZero
                 * '2157' cut at segment 8, with the re-map left in c->ctx:
                 *   [nzfill] 16/18 dst=2157 src=2156          <- the ORIGINAL,
                 *                                                filled fine
                 *   [resolve] 2157: 147 elems -> nnz=147
                 *   [node] 2157 op=NonZero                    <- re-map builds
                 *                                                a NEW tensor
                 *   boundary '2157' has no buffer -- not copied
                 *   released pool; dropped deferred fills: nonzero=4
                 * Three sources of values, all shut: no boundary copy (the
                 * rebuilt tensor has no buffer yet when the copy runs), no
                 * live fill (both the old entry and the new one point into
                 * the pool, so segment_ctx_release drops them), and nothing
                 * to recompute. Segment 9 then read '2157' as whatever the
                 * allocation held -- a float that casts to INT_MIN -- and
                 * get_rows aborted with index -2147483648 out of range
                 * [0,147).
                 *
                 * ⚠️ The INPUT hold above covers NonZero as well, and leaving
                 * it TopK-only makes this switch useless -- which is what a
                 * first attempt did. A NonZero entry names TWO tensors, its
                 * output AND its input, and fill_deferred_tensors() reads the
                 * input every time it runs, one segment LATER. Moving only the
                 * output to ctx_weight leaves nonzero_fill_src pointing into
                 * the pool, segment_ctx_release() drops the entry on that
                 * pointer alone, and the rescued output goes unfilled exactly
                 * as before.
                 *
                 * TopK's hold exists for a different read -- its rebuilt chain
                 * takes the input through src[] when a later graph computes it
                 * -- and NonZero's for the deferred fill. Same copy, same
                 * place, two reasons; either one alone is enough to need it. */
                struct ggml_context *saved_ctx = ctx->ctx;
                const char *remap_op = ctx->onnx->nodes[ni].op_type;
                const int weight_remap =
                    (strcmp(remap_op, "TopK")    == 0 ||
                     strcmp(remap_op, "NonZero") == 0) &&
                    ctx->ctx_weight != NULL;
                if (weight_remap) ctx->ctx = ctx->ctx_weight;
                const int map_rc = map_node_range(ctx, ni, ni);
                if (weight_remap) ctx->ctx = saved_ctx;

                /* Put the original back, so the substitution cannot outlive the
                 * rebuild. The rebuilt chain keeps the copy through its src[]
                 * pointers, which is the whole point; tmap goes back to naming
                 * whatever it named before. */
                if (topk_in_saved)
                    tmap_put_nd(ctx, topk_in_name, topk_in_saved, topk_in_nd);

                if (map_rc != 0) { free(carry_buf); return -1; }

                if (carry_buf) {
                    struct ggml_tensor *new_out = tmap_get(ctx, onm);

                    /* If the rebuilt output lives on a backend the compute
                     * cannot read, substitute a device-side tensor NOW, before
                     * this segment's own nodes are mapped.
                     *
                     * ⚠️ It has to happen here, not after cut_carry further
                     * down. map_node_range() for THIS segment runs between the
                     * two, and it builds the consumers -- '2169 (permuted)' is a
                     * view whose view_src is fixed at construction. Repointing
                     * tmap afterwards leaves that view addressing the host
                     * tensor, and Vulkan aborts in ggml_vk_tensor_subbuffer():
                     * measured on MaskRCNN-12-int8, where the mirror was made
                     * and the view still pointed at the CPU copy.
                     *
                     * NMS is the case: its output must be built in ctx_host
                     * (nms_cpu reads host memory only), which is allocated on
                     * the CPU backend, while its consumers run on the GPU. The
                     * scheduler inserts no copy of its own here, because the
                     * producer is not a node of the consuming segment's graph.
                     *
                     * The tensor has no buffer yet -- ctx_boundary is allocated
                     * on the line below, and the DATA arrives via cut_carry,
                     * which is pointed at the mirror instead of the original. */
                    if (new_out && ctx->backend_gpu && ctx->ctx_boundary &&
                        new_out->buffer == NULL) {
                        struct ggml_tensor *dev =
                            ggml_new_tensor(ctx->ctx_boundary, new_out->type,
                                            GGML_MAX_DIMS, new_out->ne);
                        if (dev) {
                            ggml_set_input(dev);
                            ggml_set_name(dev, onm);
                            memcpy(dev->op_params, new_out->op_params,
                                   sizeof(dev->op_params));
                            ggml_backend_buffer_t dbuf =
                                ggml_backend_alloc_ctx_tensors(ctx->ctx_boundary,
                                                               ctx->backend_gpu);
                            if (dbuf) {
                                if (ctx->n_extra_weight_bufs < ONNX_MAX_WEIGHT_BUFS)
                                    ctx->extra_weight_bufs[ctx->n_extra_weight_bufs++] = dbuf;
                                else
                                    ggml_backend_buffer_free(dbuf);
                            }
                            if (dev->buffer) {
                                int64_t osh[ONNX_MAX_DIMS];
                                const int ond = tmap_get_shape(ctx, onm, osh,
                                                               ONNX_MAX_DIMS);
                                /* ⚠️ Read the empty flag BEFORE the put and set
                                 * it again after: tmap_put_nd() appends a fresh
                                 * entry with tensor_map_empty = 0, and
                                 * tmap_is_empty() reads the LAST entry for a
                                 * name. Mirroring a logically-empty NMS output
                                 * without carrying the flag therefore un-marks
                                 * it, the Gather/Squeeze chain below stops
                                 * inheriting emptiness, and Concat keeps the 79
                                 * phantom per-class branches it is supposed to
                                 * drop -- measured on MaskRCNN-12-int8: 130
                                 * detections against ORT's 51, with 0 skipped
                                 * inputs where the CPU path skipped 79. */
                                const int was_empty = tmap_is_empty(ctx, onm);
                                if (ond > 0) tmap_put_shape(ctx, onm, dev, osh, ond);
                                else         tmap_put_nd(ctx, onm, dev,
                                                        (int)ggml_n_dims(dev));
                                if (was_empty) tmap_mark_empty(ctx, onm);
                                if (onnx_trace_nodes())
                                    fprintf(stderr, "[segment] cut output '%s' is on a "
                                            "backend the compute cannot read -- "
                                            "substituted a device tensor before "
                                            "mapping\n", onm);
                                new_out = dev;
                            }
                        }
                    }

                    if (new_out && new_out != old_out &&
                        new_out->type == old_out->type &&
                        ctx->n_cut_carry < ONNX_MAX_DEFERRED) {
                        /* The rebuilt tensor has no memory until the scheduler
                         * allocates it further down, so the write waits. */
                        ctx->cut_carry_dst[ctx->n_cut_carry]   = new_out;
                        ctx->cut_carry_buf[ctx->n_cut_carry]   = carry_buf;
                        ctx->cut_carry_bytes[ctx->n_cut_carry] = carry_bytes;
                        ctx->n_cut_carry++;
                        carry_buf = NULL;   /* the list owns it now */

                        /* Make the rebuilt tensor a LEAF, so nothing computes
                         * it a second time.
                         *
                         * build_segment_graph() deliberately does not add a cut
                         * op's output to the graph -- but that decides nothing:
                         * ggml_build_forward_expand() walks from a CONSUMER back
                         * through its sources, so 2171 = Gather(2169, ...) drags
                         * 2169 in regardless, and ggml_visit_parents_graph()
                         * (ggml-graph.c) files a tensor under leafs[] only when
                         * its op is GGML_OP_NONE. With the op still set, the
                         * kernel ran again over inputs belonging to the segment
                         * that produced them.
                         *
                         * Measured on MaskRCNN-12-int8, node 2169: the original
                         * [3,147] tensor selected 37 boxes (sum=1213, max=143),
                         * then the rebuilt [3,37] one ran again and selected 1,
                         * leaving its output all -1 -- the filler nms_cpu writes
                         * for unused rows -- which reached get_rows as "index -1
                         * out of range [0,147)". The values carried in above
                         * were overwritten by the recomputation they were meant
                         * to replace.
                         *
                         * Clearing src[] too: a leaf's sources are not visited,
                         * and leaving them set keeps pointers into a segment
                         * that is about to be released. */
                    }
                    free(carry_buf);
                }

                /* A rebuilt cut op becomes a leaf only if its value was carried
                 * across -- which is to say, only when there is nothing left to
                 * compute.
                 *
                 * The three kinds of cut op differ in what the re-map leaves
                 * behind, and the difference decides this:
                 *
                 *   NonMaxSuppression -- its result was final when its own
                 *     segment ran, so cut_carry copies the bytes into the
                 *     rebuilt tensor and a second evaluation could only undo
                 *     that. Leaf.
                 *   NonZero -- its output is a standalone tensor written by
                 *     fill_deferred_tensors() after every allocation, so it has
                 *     no op to suppress in the first place.
                 *   TopK -- REBUILT AT A NEW k, which is the whole reason the
                 *     re-map exists. Its output is a function of the input it
                 *     ranks, not a value carried over, so it has to stay a live
                 *     op and be recomputed. Making it a leaf was measured on
                 *     MaskRCNN-12-int8: '2241' kept whatever its buffer held,
                 *     '2492' came out all zeros, GET_ROWS then read row 0 for
                 *     all 48 rows, and the scores reaching NMS were the same
                 *     number 48 times -- 1 box selected where ONNX Runtime
                 *     keeps 11.
                 *
                 * Keyed on carry_op: freeze what was carried, leave TopK
                 * computable.
                 *
                 * ⚠️ This rule was tried once before and reverted, because TopK
                 * then kept live src[] into the segment pool: the pool was
                 * released at the end of the iteration and the recomputation
                 * landed on freed memory -- three tensors filled with 1, then
                 * segfaults at several addresses. The note left behind said it
                 * "needs the inputs held first", and freezing everything was
                 * the stand-in until then.
                 *
                 * Three things now make this test safe, and each was measured
                 * failing without the others: the input is copied into
                 * ctx_weight, the rebuilt chain is BUILT in ctx_weight, and
                 * TopK is left computable here. Holding only the input left
                 * '1908' out of tmap after the pool release (dets=NA, outputs
                 * unbuilt); building only in ctx_weight left the chain reading
                 * a freed input (CPU assert at ops-elemwise.cpp:351, Vulkan
                 * SIGSEGV in ggml_vk_cpy); freezing TopK as well leaves it
                 * holding whatever its buffer had, which is the empty-output
                 * failure described just above. */
                {
                    struct ggml_tensor *leaf = tmap_get(ctx, onm);
                    if (carry_op && leaf && leaf->op != GGML_OP_NONE) {
                        leaf->op = GGML_OP_NONE;
                        for (int q = 0; q < GGML_MAX_SRC; q++)
                            leaf->src[q] = NULL;
                    }
                }
                if (getenv("ONNX_TRACE_REMAP")) {
                    struct ggml_tensor *after = tmap_get(ctx, onm);
                    fprintf(stderr, "[remap]   after=[%lld,%lld]\n",
                        after ? (long long)after->ne[0] : -1,
                        after ? (long long)after->ne[1] : -1);
                }

                /* Which pool did the rebuilt tensor and its sources land in?
                 *
                 * The re-map runs while c->ctx is still the outgoing segment's
                 * pool, so a cut op that builds real graph ops (TopK is
                 * argsort_top_k over a cont of its input) puts BOTH the result
                 * and the intermediate permute/cont there -- and those have to
                 * survive the release. Printing the owner of each, by address
                 * range, says which tensors actually cross the boundary
                 * instead of leaving it to be reasoned about. */
                if (getenv("ONNX_TRACE_POOL")) {
                    struct ggml_tensor *nt = tmap_get(ctx, onm);
                    const char *seg_base = ctx->ctx_seg
                        ? (const char *)ggml_get_mem_buffer(ctx->ctx_seg) : NULL;
                    size_t seg_span = ctx->ctx_seg
                        ? ggml_get_mem_size(ctx->ctx_seg) : 0;
                    #define POOL_OF(t) ((!(t)) ? "null" : \
                        (seg_base && (const char *)(t) >= seg_base && \
                         (const char *)(t) < seg_base + seg_span) ? "SEG" : "main")
                    fprintf(stderr, "[pool] '%s' op=%s out=%s", onm,
                            ctx->onnx->nodes[ni].op_type, POOL_OF(nt));
                    if (nt)
                        for (int q = 0; q < GGML_MAX_SRC; q++)
                            if (nt->src[q])
                                fprintf(stderr, " src%d('%s')=%s", q,
                                        ggml_get_name(nt->src[q]),
                                        POOL_OF(nt->src[q]));
                    fprintf(stderr, "\n");
                    #undef POOL_OF
                }
            }

            /* Carry every tensor this segment produces that a later one reads
             * into memory the scheduler will not reuse.
             *
             * The cut_carry above is not a substitute: it saves the outputs of
             * CUT OPS only, and only those whose op is GGML_OP_CUSTOM.  An
             * ordinary tensor that happens to cross the boundary -- a Slice
             * lowered to cont(view(x)), say -- is neither, so nothing kept it
             * and its buffer was handed to the next allocation.
             *
             * Measured on MaskRCNN-12-int8 with this call absent: node 447
             * (Mul, 1999 * 2064) read 1924's bytes instead of 1999's, because
             * 1999 crosses a boundary and is a CONT.  Elementwise, -164 * 176
             * = -28864 where ORT has -0.00984 * 176 = -1.732; the error rides
             * the RPN box decode down to 39 detections against ORT's 51.
             *
             * Placed AFTER the re-map loop, not before: the loop rebuilds the
             * cut ops at their measured sizes, and the copy has to see the
             * tensors tmap ends up holding.  The rebuilt ones have no buffer
             * yet -- they are allocated further down -- so the !src->buffer
             * guard inside skips them and the cut_carry list keeps its
             * pointers.
             *
             * ⚠️ Moving this (and the pool release below) ahead of the re-map
             * was tried and reverted: the re-map reads its inputs through tmap,
             * and releasing the pool first strips the entries it needs --
             * "ggml_compute_forward_get_rows_f32: index -1 out of range
             * [0,48)" on the very first run. */
            if (copy_segment_boundaries(ctx, s - 1) != 0) return -1;
            segment_ctx_end(ctx);

            /* ⚠️ MEASURED AND REVERTED: re-mapping the whole owning segment
             * here (to rebuild the cut ops' consumers against the fresh
             * tensors) breaks the CPU path outright --
             * "ggml_compute_forward_get_rows_f32: index -332 out of range
             * [0,48) dst='2253' src0='2252'", then a segfault. Re-mapping a
             * range re-creates every tensor in it, including ones OTHER
             * segments already hold pointers to, and those consumers are left
             * addressing half-built objects. Whatever fixes the consumers has
             * to give them a new tensor without invalidating the old one for
             * everybody else -- see the per-segment context plan in TODO. */

            ctx->cur_segment = s;

            /* From here until segment_ctx_end() below, c->ctx is this
             * segment's own pool: the mapping's tensors and the graph built
             * over them are released once the segment has computed, instead of
             * accumulating for the life of the model. */
            if (segment_ctx_begin(ctx, ctx->seg_ctx_size) != 0) return -1;

            if (map_node_range(ctx, ctx->segments[s].first_node,
                                    ctx->segments[s].last_node) != 0) {
                segment_ctx_end(ctx);
                return -1;
            }

            /* Weight-like tensors this segment added (Constant, scalars,
             * NonZero/NMS outputs) still need a buffer of their own. */
            if (alloc_new_weight_tensors(ctx) != 0) { segment_ctx_end(ctx); return -1; }


            /* A fresh graph for this segment: up to its own cut ops, or to
             * the model outputs for the last one.  It lives in the segment
             * pool, so a pool too small shows up here as a NULL graph rather
             * than as a corrupted run later. */
            ctx->graph = ggml_new_graph(ctx->ctx);
            if (!ctx->graph) {
                fprintf(stderr, "[onnx] out of context memory building segment "
                                "%d/%d -- model too large for segmented execution\n",
                        s, ctx->n_segments);
                segment_ctx_end(ctx);
                return -1;
            }
            build_segment_graph(ctx, s);

            /* Take the snapshot BEFORE the allocation below: that is the last
             * moment the nodes still point at the tensors this graph was built
             * from.  sched_alloc_and_fill_on() -> ggml_backend_sched_alloc_graph()
             * -> split_graph() is where a src on another backend is replaced by
             * a scheduler copy, and the next segment inherits that. */
            snapshot_graph_srcs(ctx);

            /* Build the placement on this segment's own scheduler, the one it
             * also computes on.  Allocating on the shared scheduler here and
             * computing on the per-segment one would leave the graph's tensors
             * pointing into an arena nobody is maintaining. */
            ggml_backend_sched_t sch = seg_sched(ctx, s);
            if (!sch) { segment_ctx_end(ctx); return -1; }
            ggml_backend_sched_reset(sch);
            if (sched_alloc_and_fill_on(ctx, sch) != 0) {
                segment_ctx_end(ctx);
                return -1;
            }

            /* ⚠️ HERE, not after alloc_new_weight_tensors() above.
             *
             * NMS builds its output with ggml_custom_4d(c->ctx, ...), so the
             * rebuilt tensor's memory comes from the scheduler on the line
             * above -- alloc_new_weight_tensors only allocates ctx_weight.
             * Writing any earlier finds t->buffer still NULL and silently
             * does nothing, which is what the first attempt at this did.
             * See the note at the re-map for what the write is carrying. */
            for (int k = 0; k < ctx->n_cut_carry; k++) {
                struct ggml_tensor *t = ctx->cut_carry_dst[k];
                void *src = ctx->cut_carry_buf[k];
                if (t && t->buffer && src) {
                    size_t nb = ggml_nbytes(t);
                    if (nb > ctx->cut_carry_bytes[k]) nb = ctx->cut_carry_bytes[k];
                    ggml_backend_tensor_set(t, src, 0, nb);
                    if (onnx_trace_nodes())
                        fprintf(stderr, "[cut-carry] '%s': %zu bytes into the "
                                        "rebuilt tensor\n", ggml_get_name(t), nb);
                }
                free(src);
                ctx->cut_carry_buf[k] = NULL;
            }
            ctx->n_cut_carry = 0;


            /* This segment registered its own Shape/ConstantOfShape/NonZero/
             * EyeLike/NMS tensors, which only now have buffers to write into.
             * Without this their contents are undefined -- a NonZero index
             * list full of garbage indexes rows outside the embedding table. */
            fill_deferred_tensors(ctx);

            /* The reallocation above handed the input tensors new memory, so
             * the caller's data has to be written again -- otherwise this
             * segment reads whatever the fresh buffer happened to hold. */
            if (set_model_inputs(ctx, input_names, input_data, input_lens, n_inputs) != 0) {
                segment_ctx_end(ctx);
                return -1;
            }

            if (onnx_trace_nodes())
                fprintf(stderr, "[segment] %d: computing %d graph nodes\n",
                        s, ggml_graph_n_nodes(ctx->graph));

            status = ggml_backend_sched_graph_compute(sch, ctx->graph);

            /* Undo the scheduler's src[] rewrites now that this segment has
             * computed.  Unconditional, and before the status check: a run
             * that broke partway still leaves the graph mutated, and the
             * cached graphs outlive the run.  Leaving a copy in place would
             * hide the producer from the NEXT graph built over these tensors,
             * which is the defect this restores. */
            restore_graph_srcs(ctx);

            if (status != GGML_STATUS_SUCCESS) break;
        }

        /* ⚠️ The last segment's pool stays open on purpose.
         *
         * The loop releases a segment on the following iteration and there is
         * none after this one -- but the model's outputs are that segment's
         * tensors, and the caller reads them AFTER this function returns
         * (r_onnx.c calls onnx_ggml_output() then ggml_backend_tensor_get()).
         * Freeing here would hand back tensors whose memory is gone. It is
         * released at the start of the next run instead, by the
         * segment_ctx_end() near the top of onnx_ggml_run(). */
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

    /* Every op that map_node declines leaves its output unregistered, and if
     * one of those sits on the path to a model output, nothing downstream gets
     * built either: the output then keeps whatever its buffer happened to hold
     * and the run still reports success.
     *
     * Worth naming loudly because the symptoms point everywhere but here:
     * MaskRCNN's rejected TopK surfaced as a custom kernel that never ran, an
     * index of -1 in a gather, and a segfault in an unrelated segment, none of
     * which mentions a missing output.
     *
     * The test is whether the outputs resolve, not whether the graph has
     * nodes.  An empty graph is perfectly legitimate when nothing needs
     * computing -- Min with a single input, Transpose with an identity perm, a
     * Reshape that is only a view -- and those handlers hand the input tensor
     * straight back, correctly.  Warning there is a false alarm, the kind that
     * teaches the reader to skim past the real one.
     *
     * Asked here, after the segment loop, rather than before segment 0's
     * compute: a segmented model builds its outputs in the LAST segment, so
     * the earlier question reported an ordinary two-segment model as broken
     * (ConstantOfShape->NonZero cuts at NonZero, leaving segment 0 with the
     * cut op alone and the real output three nodes into segment 1). */
    if (status == GGML_STATUS_SUCCESS) {
        int n_missing = 0;
        for (int i = 0; i < ctx->onnx->n_outputs; i++)
            if (!tmap_get(ctx, ctx->onnx->outputs[i].name)) n_missing++;

        if (n_missing > 0) {
            fprintf(stderr, "[onnx] %d of %d model outputs were never built: "
                            "every path to them was cut by an op that could not "
                            "be mapped.\n", n_missing, ctx->onnx->n_outputs);
            for (int i = 0; i < ctx->onnx->n_outputs; i++)
                if (!tmap_get(ctx, ctx->onnx->outputs[i].name))
                    fprintf(stderr, "[onnx]   unresolved output: '%s'\n",
                            ctx->onnx->outputs[i].name);
            if (ctx->first_failed_node[0])
                fprintf(stderr, "[onnx]   first failure: node '%s' (%s) -- the "
                                "later ones follow from it\n",
                        ctx->first_failed_node, ctx->first_failed_op);
            fprintf(stderr, "[onnx]   outputs will be whatever their buffers held.\n");
        }
    }

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
             * shared so the two cannot disagree about what a boundary is.
             *
             * ⚠️ Do NOT widen this to keep a cut op's input alive for
             * resolve_segment_sizes(). Tried, measured, reverted: a copy
             * REPLACES its original in tmap (see the persistent-buffer note
             * below), so copying an input that no later segment reads still
             * repoints everything built over it afterwards. MaskRCNN-12-int8
             * went from 68 detections to 134 on run 2, with "TopK 6565: K=100
             * exceeds the 51 elements on the ranked axis". The measurement is
             * taken right after the segment computes instead -- it needs one
             * integer, not the bytes.
             *
             * A model output is the one safe widening, and for the reason the
             * warning above turns on: the danger is that a copy REPLACES the
             * original in tmap and repoints everything built over it, and
             * nothing is ever built over a model output -- it is terminal by
             * definition. It also has a reader the other test cannot see,
             * since tensor_crosses_boundary() looks for a later NODE taking it
             * as an input, and this one is read by R after the last compute.
             * See tensor_is_model_output() for the measurement. */
            if (!tensor_crosses_boundary(c, seg, nm) &&
                !tensor_is_model_output(c, nm)) continue;

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
            /* ...but only when the tensor STRUCT is out of the pool too: see
             * tensor_in_pool() for the view case, where persistent storage sits
             * behind metadata the release is about to free. */
            if ((src->buffer == c->weight_buf || src->buffer == c->boundary_buf)
                && !tensor_in_pool(c, src)) {
                if (onnx_trace_nodes())
                    fprintf(stderr, "[segment] %d: boundary '%s' is persistent "
                                    "-- not copied\n", seg, nm);
                continue;
            }
            {
                int persistent = 0;
                for (int k = 0; k < c->n_extra_weight_bufs; k++)
                    if (src->buffer == c->extra_weight_bufs[k]) { persistent = 1; break; }
                if (persistent && !tensor_in_pool(c, src)) {
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
    /* Carry buffers are normally consumed a few lines after they are filled;
     * these are the ones a run that failed in between left behind. */
    for (int i = 0; i < ctx->n_cut_carry; i++) free(ctx->cut_carry_buf[i]);
    ctx->n_cut_carry = 0;
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
    /* Before the backends below: a buffer belongs to the backend that made it
     * and freeing it afterwards touches a destroyed object. */
    if (ctx->host_buf)    ggml_backend_buffer_free(ctx->host_buf);
    if (ctx->backend_gpu) ggml_backend_free(ctx->backend_gpu);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    /* A run leaves its last segment's pool open so the caller can read the
     * outputs; teardown may well come next, with ctx still pointing into it.
     * Closing it here puts ctx back on the model context before that is freed
     * below -- otherwise the pool would be freed twice and the real context
     * not at all. */
    segment_ctx_end(ctx);

    if (ctx->ctx_boundary) ggml_free(ctx->ctx_boundary);
    if (ctx->ctx_host)    ggml_free(ctx->ctx_host);
    if (ctx->ctx_weight)  ggml_free(ctx->ctx_weight);
    if (ctx->ctx)         ggml_free(ctx->ctx);
    free(ctx->tensor_map_keys);
    free(ctx->tensor_map_vals);
    free(ctx->tensor_map_ndims);
    free(ctx->tensor_map_onnx_ne);
    free(ctx->tensor_map_empty);
    free(ctx->cval_keys);
    free(ctx->cval_data);
    free(ctx->cval_lens);
    for (int i = 0; i < ctx->n_pos_embed_blocks; i++)
        free(ctx->pos_embed_blocks[i].params.w_cpu);
    free(ctx->pos_embed_params);
    for (int i = 0; i < ctx->n_roi_aligns; i++) free(ctx->roi_align_params[i]);
    free(ctx->roi_align_params);
    /* Each entry is its own allocation (see onnx_ops_special.c), so the
     * entries go before the array that holds them. */
    for (int i = 0; i < ctx->n_nms_ops; i++) free(ctx->nms_params[i]);
    free(ctx->nms_params);
    for (int i = 0; i < ctx->n_qconv_ops; i++) free(ctx->qconv_params[i]);
    free(ctx->qconv_params);
    for (int i = 0; i < ctx->n_qconv_mult; i++) free(ctx->qconv_mult_values[i]);
    /* Note: onnx model is NOT freed here — caller manages it */
    free(ctx);
}
