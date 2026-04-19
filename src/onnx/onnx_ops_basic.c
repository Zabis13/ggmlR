/* onnx_ops_basic.c — basic math ops: binary, matmul, activations
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ops_internal.h"

/* Returns 1 = handled, 0 = not this group's op, -1 = error */
int map_node_basic(onnx_ggml_ctx_t *c, const onnx_node_t *n,
                   struct ggml_tensor *a, struct ggml_tensor *b,
                   struct ggml_tensor **out_p, int *out_nd_p)
{
    const char *op = n->op_type;
    struct ggml_tensor *out = NULL;
    int out_nd = -1;

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
                /* Check all dims of B for K match (up to GGML_MAX_DIMS) */
                int k_dim = -1;
                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    if (b->ne[d] == K_a) { k_dim = d; break; }
                }
                if (k_dim >= 0 && k_dim != 0 && k_dim < 4) {
                    /* Permute B to move K to ne[0] (only dims 0-3 supported by ggml_permute) */
                    int p[5] = {0, 1, 2, 3, 4};
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

    else {
        return 0; /* not this group */
    }

    *out_p    = out;
    *out_nd_p = out_nd;
    return 1;
}
