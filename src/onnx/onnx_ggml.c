/* onnx_ggml.c — Map ONNX ops to ggml ops and run inference
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ggml.h"
#include "../ggml.h"
#include "../ggml-backend.h"
#include "../ggml-cpu.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Check if Vulkan is available at compile time */
#ifdef GGML_USE_VULKAN
#include "../ggml-vulkan.h"
#endif

/* ── Tensor name map ────────────────────────────────────────────── */

static void tmap_put(onnx_ggml_ctx_t *c, const char *name, struct ggml_tensor *t) {
    if (c->tensor_map_size >= c->tensor_map_cap) {
        c->tensor_map_cap = c->tensor_map_cap ? c->tensor_map_cap * 2 : 256;
        c->tensor_map_keys = realloc(c->tensor_map_keys,
                                      c->tensor_map_cap * sizeof(*c->tensor_map_keys));
        c->tensor_map_vals = realloc(c->tensor_map_vals,
                                      c->tensor_map_cap * sizeof(*c->tensor_map_vals));
    }
    strncpy(c->tensor_map_keys[c->tensor_map_size], name, ONNX_MAX_NAME - 1);
    c->tensor_map_keys[c->tensor_map_size][ONNX_MAX_NAME - 1] = '\0';
    c->tensor_map_vals[c->tensor_map_size] = t;
    c->tensor_map_size++;
}

static struct ggml_tensor *tmap_get(onnx_ggml_ctx_t *c, const char *name) {
    for (int i = c->tensor_map_size - 1; i >= 0; i--) {
        if (strcmp(c->tensor_map_keys[i], name) == 0)
            return c->tensor_map_vals[i];
    }
    return NULL;
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
        int64_t ne[4] = {1, 1, 1, 1};
        int ndims = init->n_dims > 0 ? init->n_dims : 1;
        if (ndims > 4) ndims = 4;
        for (int d = 0; d < ndims; d++) {
            ne[d] = init->dims[ndims - 1 - d];
        }

        struct ggml_tensor *t;
        switch (ndims) {
            case 1: t = ggml_new_tensor_1d(c->ctx, type, ne[0]); break;
            case 2: t = ggml_new_tensor_2d(c->ctx, type, ne[0], ne[1]); break;
            case 3: t = ggml_new_tensor_3d(c->ctx, type, ne[0], ne[1], ne[2]); break;
            default: t = ggml_new_tensor_4d(c->ctx, type, ne[0], ne[1], ne[2], ne[3]); break;
        }
        if (!t) return -1;
        ggml_set_name(t, init->name);
        ggml_set_input(t);
        tmap_put(c, init->name, t);
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
            size_t elem_size = onnx_dtype_size(init->data_type);

            /* With reversed dims, ONNX row-major data maps directly to ggml
             * column-major layout — no transposition needed. */
            {
                size_t copy_size = data_size < tsize ? data_size : tsize;
                ggml_backend_tensor_set(t, data, 0, copy_size);
            }
        }
    }
    return 0;
}

/* ── Create input placeholder tensors ───────────────────────────── */

static int create_input_tensors(onnx_ggml_ctx_t *c) {
    for (int i = 0; i < c->onnx->n_inputs; i++) {
        onnx_value_info_t *vi = &c->onnx->inputs[i];
        /* Skip if already created as initializer */
        if (tmap_get(c, vi->name)) continue;

        enum ggml_type type = onnx_dtype_to_ggml(vi->elem_type);
        int64_t ne[4] = {1, 1, 1, 1};
        int ndims = vi->n_dims > 0 ? vi->n_dims : 1;
        if (ndims > 4) ndims = 4;
        /* Reverse ONNX dims → ggml ne[] (row-major → column-major) */
        for (int d = 0; d < ndims; d++) {
            int64_t dim = vi->dims[ndims - 1 - d];
            if (dim <= 0) dim = 1; /* symbolic/dynamic dim → default 1 */
            ne[d] = dim;
        }

        struct ggml_tensor *t;
        switch (ndims) {
            case 1: t = ggml_new_tensor_1d(c->ctx, type, ne[0]); break;
            case 2: t = ggml_new_tensor_2d(c->ctx, type, ne[0], ne[1]); break;
            case 3: t = ggml_new_tensor_3d(c->ctx, type, ne[0], ne[1], ne[2]); break;
            default: t = ggml_new_tensor_4d(c->ctx, type, ne[0], ne[1], ne[2], ne[3]); break;
        }
        if (!t) return -1;
        ggml_set_name(t, vi->name);
        ggml_set_input(t);
        tmap_put(c, vi->name, t);
    }
    return 0;
}

/* ── Map ONNX node → ggml op ────────────────────────────────────── */

static struct ggml_tensor *get_input(onnx_ggml_ctx_t *c, const onnx_node_t *n, int idx) {
    if (idx >= n->n_inputs) return NULL;
    if (n->inputs[idx][0] == '\0') return NULL; /* optional empty input */
    return tmap_get(c, n->inputs[idx]);
}

static int map_node(onnx_ggml_ctx_t *c, const onnx_node_t *n) {
    struct ggml_tensor *out = NULL;
    struct ggml_tensor *a = get_input(c, n, 0);
    struct ggml_tensor *b = get_input(c, n, 1);

    const char *op = n->op_type;

    fprintf(stderr, "onnx_ggml: mapping op '%s'", op);
    if (a) fprintf(stderr, " a=[%lld,%lld,%lld,%lld]",
                   (long long)a->ne[0], (long long)a->ne[1],
                   (long long)a->ne[2], (long long)a->ne[3]);
    if (b) fprintf(stderr, " b=[%lld,%lld,%lld,%lld]",
                   (long long)b->ne[0], (long long)b->ne[1],
                   (long long)b->ne[2], (long long)b->ne[3]);
    fprintf(stderr, "\n");

    /* ── Elementwise binary ─────────────────────────────────────── */
    if (strcmp(op, "Add") == 0) {
        if (!a || !b) return -1;
        /* ggml_add requires b broadcastable into a — swap if a is smaller */
        if (ggml_nelements(a) < ggml_nelements(b))
            out = ggml_add(c->ctx, b, a);
        else
            out = ggml_add(c->ctx, a, b);
    }
    else if (strcmp(op, "Sub") == 0) {
        if (!a || !b) return -1;
        out = ggml_sub(c->ctx, a, b);
    }
    else if (strcmp(op, "Mul") == 0) {
        if (!a || !b) return -1;
        if (ggml_nelements(a) < ggml_nelements(b))
            out = ggml_mul(c->ctx, b, a);
        else
            out = ggml_mul(c->ctx, a, b);
    }
    else if (strcmp(op, "Div") == 0) {
        if (!a || !b) return -1;
        out = ggml_div(c->ctx, a, b);
    }

    /* ── MatMul / Gemm ──────────────────────────────────────────── */
    else if (strcmp(op, "MatMul") == 0) {
        if (!a || !b) return -1;
        /* ONNX MatMul(A[M,K], B[K,N]) → result [M,N].
         * With reversed dims: A.ne=[K,M], B.ne=[N,K].
         * ggml_mul_mat(a,b) contracts a.ne[0]==b.ne[0], result=[a.ne[1],b.ne[1]].
         * Need contraction on K: transpose B → B^T.ne=[K,N].
         * ggml_mul_mat(B^T, A): K==K ✓, result=[N,M] (= ONNX [M,N] reversed) ✓ */
        struct ggml_tensor *bt = ggml_cont(c->ctx, ggml_transpose(c->ctx, b));
        out = ggml_mul_mat(c->ctx, bt, a);
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
        out = ggml_soft_max(c->ctx, a);
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
        if (!min_t) min_val = onnx_attr_float(n, "min", min_val);
        if (!max_t) max_val = onnx_attr_float(n, "max", max_val);
        /* TODO: handle tensor min/max inputs */
        out = ggml_clamp(c->ctx, a, min_val, max_val);
    }

    /* ── Shape ops ──────────────────────────────────────────────── */
    else if (strcmp(op, "Reshape") == 0) {
        if (!a || !b) return -1;
        /* b is the shape tensor — for now we read from initializer */
        const onnx_initializer_t *shape_init = onnx_find_initializer(c->onnx, n->inputs[1]);
        if (!shape_init) return -1;

        int64_t shape[ONNX_MAX_DIMS];
        int ndims = 0;
        if (shape_init->raw_data && shape_init->data_type == ONNX_DTYPE_INT64) {
            ndims = (int)(shape_init->raw_size / sizeof(int64_t));
            if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
            memcpy(shape, shape_init->raw_data, ndims * sizeof(int64_t));
        } else if (shape_init->decoded_data) {
            ndims = (int)(shape_init->decoded_size / sizeof(int64_t));
            if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
            memcpy(shape, shape_init->decoded_data, ndims * sizeof(int64_t));
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

        /* Reverse ONNX shape → ggml ne order for reshape */
        int64_t ne[4] = {1, 1, 1, 1};
        for (int d = 0; d < ndims; d++)
            ne[d] = shape[ndims - 1 - d];

        fprintf(stderr, "  Reshape: [%lld] → [%lld,%lld,%lld,%lld]\n",
                (long long)total, (long long)ne[0], (long long)ne[1],
                (long long)ne[2], (long long)ne[3]);

        switch (ndims) {
            case 1: out = ggml_reshape_1d(c->ctx, a, ne[0]); break;
            case 2: out = ggml_reshape_2d(c->ctx, a, ne[0], ne[1]); break;
            case 3: out = ggml_reshape_3d(c->ctx, a, ne[0], ne[1], ne[2]); break;
            case 4: out = ggml_reshape_4d(c->ctx, a, ne[0], ne[1], ne[2], ne[3]); break;
            default: out = ggml_reshape_2d(c->ctx, a, ne[0], total / ne[0]); break;
        }
    }
    else if (strcmp(op, "Transpose") == 0) {
        if (!a) return -1;
        int64_t perm[4];
        int n_perm = onnx_attr_ints(n, "perm", perm, 4);
        int nd = ggml_n_dims(a);
        if (n_perm == 0 || nd == 2) {
            /* Default: reverse all dims → standard transpose */
            out = ggml_cont(c->ctx, ggml_transpose(c->ctx, a));
        } else {
            /* Convert ONNX perm to ggml permute axes.
             * ONNX perm[i]=j means output ONNX dim i ← input ONNX dim j.
             * ggml ax[k] means output ggml dim k ← input ggml dim ax[k].
             * ONNX dim i → ggml dim (nd-1-i).
             * So: ax[nd-1-i] = nd-1-perm[i]. */
            int ax[4] = {0, 1, 2, 3};
            for (int i = 0; i < n_perm && i < nd; i++) {
                int ggml_out = nd - 1 - i;
                int ggml_in  = nd - 1 - (int)perm[i];
                if (ggml_out >= 0 && ggml_out < 4 && ggml_in >= 0 && ggml_in < 4)
                    ax[ggml_out] = ggml_in;
            }
            out = ggml_cont(c->ctx, ggml_permute(c->ctx, a, ax[0], ax[1], ax[2], ax[3]));
        }
    }
    else if (strcmp(op, "Flatten") == 0) {
        if (!a) return -1;
        int64_t total = ggml_nelements(a);
        out = ggml_reshape_1d(c->ctx, a, total);
    }
    else if (strcmp(op, "Squeeze") == 0 || strcmp(op, "Unsqueeze") == 0) {
        if (!a) return -1;
        /* For now, pass through — reshape handled at boundaries */
        out = ggml_cont(c->ctx, a);
    }

    /* ── Concat ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Concat") == 0) {
        if (!a) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        int nd = GGML_MAX_DIMS;
        int onnx_axis = (int)axis;
        if (onnx_axis < 0) onnx_axis = nd + onnx_axis;
        int dim = nd - 1 - onnx_axis;
        /* Concat supports N inputs — chain pairwise */
        out = a;
        for (int i = 1; i < n->n_inputs; i++) {
            struct ggml_tensor *inp = get_input(c, n, i);
            if (!inp) continue;
            out = ggml_concat(c->ctx, out, inp, dim);
        }
    }

    /* ── Gather ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Gather") == 0) {
        if (!a || !b) return -1;
        out = ggml_get_rows(c->ctx, a, b);
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
        struct ggml_tensor *eps_t = ggml_new_f32(c->ctx, eps);
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
        enum ggml_op_pool pool_op = (strcmp(op, "MaxPool") == 0) ?
                                     GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;
        out = ggml_pool_2d(c->ctx, a, pool_op,
                           (int)kshape[1], (int)kshape[0],
                           (int)strides[1], (int)strides[0],
                           (int)pads[1], (int)pads[0]);
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

        int ndims_kernel = ggml_n_dims(b);
        if (ndims_kernel <= 2) {
            out = ggml_conv_1d(c->ctx, b, a,
                               (int)strides[0], (int)pads[0], (int)dilations[0]);
        } else {
            out = ggml_conv_2d(c->ctx, b, a,
                               (int)strides[1], (int)strides[0],
                               (int)pads[1], (int)pads[0],
                               (int)dilations[1], (int)dilations[0]);
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

    /* ── Reduction ──────────────────────────────────────────────── */
    else if (strcmp(op, "ReduceMean") == 0) {
        if (!a) return -1;
        out = ggml_mean(c->ctx, a);
    }
    else if (strcmp(op, "ReduceSum") == 0) {
        if (!a) return -1;
        out = ggml_sum(c->ctx, a);
    }

    /* ── Identity / Dropout (inference mode) ────────────────────── */
    else if (strcmp(op, "Identity") == 0 || strcmp(op, "Dropout") == 0) {
        if (!a) return -1;
        out = a; /* pass-through */
    }

    /* ── Constant (skip — should be in initializers) ────────────── */
    else if (strcmp(op, "Constant") == 0) {
        /* TODO: create tensor from attribute value */
        return 0; /* skip silently */
    }

    /* ── Unknown op ─────────────────────────────────────────────── */
    else {
        fprintf(stderr, "onnx_ggml: unsupported op '%s'\n", op);
        return -1;
    }

    /* Register outputs */
    if (out) {
        for (int i = 0; i < n->n_outputs; i++) {
            if (n->outputs[i][0] != '\0') {
                ggml_set_name(out, n->outputs[i]);
                tmap_put(c, n->outputs[i], out);
            }
        }
    }

    return 0;
}

/* ── Build full graph ───────────────────────────────────────────── */

onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device) {
    onnx_ggml_ctx_t *c = calloc(1, sizeof(onnx_ggml_ctx_t));
    if (!c) return NULL;
    c->onnx = onnx;

    /* Estimate memory: rough heuristic based on file size */
    size_t mem_size = onnx->mmap_size * 2 + 256 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    c->ctx = ggml_init(params);
    if (!c->ctx) { free(c); return NULL; }

    /* Create tensors for initializers and inputs */
    if (create_initializer_tensors(c) != 0) goto fail;
    if (create_input_tensors(c) != 0) goto fail;

    /* Map ONNX nodes to ggml ops */
    for (int i = 0; i < onnx->n_nodes; i++) {
        if (map_node(c, &onnx->nodes[i]) != 0) {
            /* Non-fatal: skip unsupported ops with warning */
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

    /* Allocate graph buffers via scheduler */
    if (!ggml_backend_sched_alloc_graph(c->sched, c->graph)) goto fail;

    /* Load weights into allocated tensors */
    if (load_weights(c) != 0) goto fail;

    return c;

fail:
    onnx_ggml_free(c);
    return NULL;
}

int onnx_ggml_run(onnx_ggml_ctx_t *ctx,
                  const char **input_names, const float **input_data,
                  int n_inputs) {
    /* Set input data — with reversed dims, row-major data maps directly
     * to ggml column-major layout, no transposition needed. */
    for (int i = 0; i < n_inputs; i++) {
        struct ggml_tensor *t = tmap_get(ctx, input_names[i]);
        if (!t) {
            fprintf(stderr, "onnx_ggml: input '%s' not found\n", input_names[i]);
            return -1;
        }
        size_t nbytes = ggml_nbytes(t);
        ggml_backend_tensor_set(t, input_data[i], 0, nbytes);
    }

    /* Compute via scheduler (handles CPU fallback for unsupported ops) */
    if (!ctx->graph || !ctx->sched) return -1;
    enum ggml_status status = ggml_backend_sched_graph_compute(ctx->sched, ctx->graph);
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
    if (ctx->backend_gpu) ggml_backend_free(ctx->backend_gpu);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->ctx)     ggml_free(ctx->ctx);
    free(ctx->tensor_map_keys);
    free(ctx->tensor_map_vals);
    /* Note: onnx model is NOT freed here — caller manages it */
    free(ctx);
}
