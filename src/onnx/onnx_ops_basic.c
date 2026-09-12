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
        /* Ensure matching types for binary op, and a type ggml can compute:
         * two integers reach the kernel as i32 and abort there. */
        onnx_binary_promote(c->ctx, &a, &b);
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
        onnx_binary_promote(c->ctx, &a, &b);
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
        onnx_binary_promote(c->ctx, &a, &b);
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
        /* Integer Div truncates toward zero in ONNX -- 7/2 is 3, not 3.5.
         * Widening the operands to F32 to reach ggml's kernel does not change
         * that, so the truncation has to be put back afterwards; without it
         * the runtime disagreed with the cval folding a few lines below, which
         * has always done the division in integers. */
        const int int_div = (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_I32);
        onnx_binary_promote(c->ctx, &a, &b);
        struct ggml_tensor *ta = a, *tb = b;
        if (ggml_nelements(a) >= ggml_nelements(b)) {
            onnx_broadcast_prepare(c->ctx, &ta, &tb);
            out = ggml_div(c->ctx, ta, tb);
        } else {
            onnx_broadcast_prepare(c->ctx, &ta, &tb);
            out = ggml_div(c->ctx, ta, tb);
        }
        if (int_div) out = ggml_trunc(c->ctx, out);

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

    /* ── Min / Max (variadic, elementwise) ──────────────────────── */
    else if (strcmp(op, "Min") == 0 || strcmp(op, "Max") == 0) {
        if (!a) return -1;
        const int is_min = (op[1] == 'i');

        /* min(x,y) = (x + y - |x - y|) / 2 ;  max uses + |x - y|.
         *
         * ggml has no elementwise min/max over two tensors (ggml_max reduces
         * along an axis, which is a different operation), and the identity
         * above needs only add/sub/abs/scale -- all of which already run on
         * both CPU and Vulkan.  A new kernel would compute this in one node
         * instead of four, at the cost of a divergence from upstream ggml
         * plus a shader; that trade is not worth it for an op no model in the
         * reference set uses yet.
         *
         * Exact for every finite input: no comparison, no tie-breaking, and
         * the halving is a power of two.  A NaN operand propagates, which is
         * what the IEEE-style reading of the spec asks for.
         *
         * ONNX allows one input ("Min(x) = x"), which falls out of the loop
         * below running zero times. */
        struct ggml_tensor *acc = a;
        for (int i = 1; i < n->n_inputs; i++) {
            struct ggml_tensor *x = get_input(c, n, i);
            if (!x) return -1;

            onnx_binary_promote(c->ctx, &acc, &x);

            struct ggml_tensor *t1 = acc, *t2 = x;
            onnx_broadcast_prepare(c->ctx, &t1, &t2);
            struct ggml_tensor *sum  = ggml_add(c->ctx, t1, t2);

            /* |x - y| is symmetric, so the broadcast pair is reused as-is. */
            struct ggml_tensor *diff = ggml_sub(c->ctx, t1, t2);
            struct ggml_tensor *ad   = ggml_abs(c->ctx, diff);

            struct ggml_tensor *comb = is_min ? ggml_sub(c->ctx, sum, ad)
                                              : ggml_add(c->ctx, sum, ad);
            acc = ggml_scale(c->ctx, comb, 0.5f);
        }
        out = acc;

        /* Folded left, so the node count grows linearly with the number of
         * inputs rather than logarithmically.  Deliberate: the shape of the
         * graph only starts to matter somewhere past a handful of inputs, and
         * no model at hand has more than a few.  A balanced tree would be the
         * fix if one ever does. */

        /* Compile-time values, so shape arithmetic keeps folding through a
         * clamp written as Max(lo, Min(hi, x)) -- the form MaskRCNN uses for
         * FPN level assignment. */
        {
            int64_t cv_acc[ONNX_MAX_DIMS];
            int n_acc = cval_get(c, n->inputs[0], cv_acc, ONNX_MAX_DIMS);
            for (int i = 1; i < n->n_inputs && n_acc > 0; i++) {
                int64_t cv_x[ONNX_MAX_DIMS];
                int n_x = cval_get(c, n->inputs[i], cv_x, ONNX_MAX_DIMS);
                if (n_x <= 0) { n_acc = 0; break; }
                int nr = n_acc > n_x ? n_acc : n_x;
                for (int j = 0; j < nr; j++) {
                    int64_t va = cv_acc[j < n_acc ? j : 0];
                    int64_t vb = cv_x[j < n_x ? j : 0];
                    cv_acc[j] = is_min ? (va < vb ? va : vb)
                                       : (va > vb ? va : vb);
                }
                n_acc = nr;
            }
            if (n_acc > 0) cval_put(c, n->outputs[0], cv_acc, n_acc);
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
        /* ggml_permute shuffles ne[0]..ne[3] only, so an axis at ggml dim 4
         * (a 5D input normalized along its outermost ONNX dim) has no path
         * here.  Say so rather than index ax[4] off the end of a 4-element
         * array and permute by whatever was on the stack. */
        if (ggml_d > 3) {
            fprintf(stderr, "[onnx] Softmax %s: axis %lld of a rank-%d input maps "
                            "to ggml dim %d, which permute cannot reach\n",
                    n->outputs[0], (long long)axis, nd, ggml_d);
            return -1;
        }

        if (ggml_d == 0) {
            /* Softmax over ne[0] — ggml_soft_max does this natively */
            out = ggml_soft_max(c->ctx, a);
        } else {
            /* ggml_soft_max normalizes over ne[0], so bring the axis being
             * normalized there, softmax, and put it back.  Swapping 0 and
             * ggml_d is its own inverse, so the same permutation serves both
             * ways.
             *
             * The previous form flattened dims [0..ggml_d] into ne[0] and
             * softmaxed that.  Flattening puts the LOWER dims inside the
             * normalization group, so the sum ran over them too: for ONNX
             * [N,C,H,W] with axis=1 it normalized over W*H*C at once, where
             * the spec asks for C alone, separately for each (H,W).  On a
             * [1,3,2,2] tensor that is wrong by 0.65 -- verified against a
             * hand-computed softmax before this was replaced.
             *
             * No model in the reference set reaches here (they all carry
             * axis=-1, which is the ne[0] fast path above), which is why the
             * error survived: it is only reachable from an intermediate axis. */
            int ax[4] = {0, 1, 2, 3};
            ax[0] = ggml_d;
            ax[ggml_d] = 0;

            struct ggml_tensor *p = ggml_cont(c->ctx,
                ggml_permute(c->ctx, a, ax[0], ax[1], ax[2], ax[3]));
            struct ggml_tensor *sm = ggml_soft_max(c->ctx, p);
            out = ggml_cont(c->ctx,
                ggml_permute(c->ctx, sm, ax[0], ax[1], ax[2], ax[3]));
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

    /* ── HardSigmoid ────────────────────────────────────────────── */
    else if (strcmp(op, "HardSigmoid") == 0) {
        if (!a) return -1;
        /* max(0, min(1, alpha*x + beta)).  ONNX defaults are alpha=0.2,
         * beta=0.5; ggml_hardsigmoid hardcodes 1/6 and 0.5 (the HardSwish
         * flavour), so it only serves one point of the parameter space.
         * scale_bias + clamp expresses every alpha/beta and rides the same
         * two Vulkan kernels, so there is no reason to special-case the
         * default and then be wrong on any model that picks its own. */
        float alpha = onnx_attr_float(n, "alpha", 0.2f);
        float beta  = onnx_attr_float(n, "beta",  0.5f);
        out = ggml_clamp(c->ctx, ggml_scale_bias(c->ctx, a, alpha, beta),
                         0.0f, 1.0f);
    }

    /* ── HardSwish ──────────────────────────────────────────────── */
    else if (strcmp(op, "HardSwish") == 0) {
        if (!a) return -1;
        /* x * max(0, min(1, x/6 + 0.5)) -- the constants are fixed by the
         * ONNX spec, not attributes, and match ggml_hardswish exactly. */
        out = ggml_hardswish(c->ctx, a);
    }

    /* ── PRelu ──────────────────────────────────────────────────── */
    else if (strcmp(op, "PRelu") == 0) {
        if (!a || !b) return -1;
        /* x >= 0 ? x : slope*x, with slope a TENSOR (per channel, usually),
         * which is why ggml_leaky_relu -- scalar alpha only -- cannot serve.
         *
         * Written as relu(x) - slope*relu(-x): the first term keeps the
         * positive half and zeroes the rest, the second contributes only
         * where x < 0.  No comparison op and no select needed, so it stays
         * on ops that already have Vulkan kernels.
         *
         * ONNX broadcasts slope against x by the usual rules (a [C] slope
         * against [N,C,H,W]), so it goes through the same preparation every
         * other binary op here uses. */
        struct ggml_tensor *pos = ggml_relu(c->ctx, a);
        struct ggml_tensor *neg = ggml_relu(c->ctx, ggml_neg(c->ctx, a));
        struct ggml_tensor *sl  = b;
        onnx_broadcast_prepare(c->ctx, &neg, &sl);
        out = ggml_sub(c->ctx, pos, ggml_mul(c->ctx, neg, sl));
    }

    /* ── Range ──────────────────────────────────────────────────── */
    else if (strcmp(op, "Range") == 0) {
        /* start, limit, delta arrive as rank-0 tensors.  ggml_arange takes
         * plain floats, so the three must be known while the graph is being
         * built; they are shape arithmetic in every model seen so far, which
         * is exactly what cval tracks.  Say so plainly when they are not,
         * rather than build a range of the wrong length. */
        int64_t v[3];
        float f[3];
        int have = 1;
        for (int i = 0; i < 3 && have; i++) {
            if (n->n_inputs <= i || n->inputs[i][0] == '\0') { have = 0; break; }
            int nv = cval_get(c, n->inputs[i], v, 1);
            if (nv > 0) { f[i] = (float)v[0]; continue; }
            /* Not a tracked shape value: fall back to reading a constant. */
            const onnx_initializer_t *ii = onnx_find_initializer(c->onnx, n->inputs[i]);
            if (!ii) ii = find_constant_tensor(c->onnx, n->inputs[i]);
            if (ii && ii->raw_data) {
                switch (ii->data_type) {
                    case ONNX_DTYPE_FLOAT: f[i] = ((const float *)ii->raw_data)[0]; break;
                    case ONNX_DTYPE_INT64: f[i] = (float)((const int64_t *)ii->raw_data)[0]; break;
                    case ONNX_DTYPE_INT32: f[i] = (float)((const int32_t *)ii->raw_data)[0]; break;
                    default: have = 0; break;
                }
            } else have = 0;
        }
        if (!have) {
            fprintf(stderr, "[onnx] Range %s: start/limit/delta are computed at "
                            "runtime; only build-time values are supported\n",
                    n->outputs[0]);
            return -1;
        }
        if (f[2] == 0.0f) {
            fprintf(stderr, "[onnx] Range %s: delta is zero\n", n->outputs[0]);
            return -1;
        }
        out = ggml_arange(c->ctx, f[0], f[1], f[2]);
        out_nd = 1;
        /* A range of shape values is itself shape arithmetic downstream. */
        {
            /* count = ceil((limit-start)/delta), without pulling in math.h */
            double span = ((double)f[1] - (double)f[0]) / (double)f[2];
            int64_t cnt = (int64_t)span;
            if ((double)cnt < span) cnt++;
            if (cnt > 0 && cnt <= ONNX_MAX_DIMS) {
                int64_t vals[ONNX_MAX_DIMS];
                for (int64_t k = 0; k < cnt; k++)
                    vals[k] = (int64_t)(f[0] + (float)k * f[2]);
                cval_put(c, n->outputs[0], vals, (int)cnt);
            }
        }
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


    /* ── CumSum ─────────────────────────────────────────────────── */
    else if (strcmp(op, "CumSum") == 0) {
        if (!a) return -1;
        /* ggml_cumsum accumulates along ne[0], so the ONNX axis is rotated
         * there and back -- the same swap Softmax uses, and self-inverse for
         * the same reason.
         *
         * axis arrives as input 1 (a rank-0 tensor), not as an attribute. */
        int64_t axis = 0;
        {
            int64_t v[1];
            int got = 0;
            if (n->n_inputs > 1 && n->inputs[1][0] != '\0') {
                if (cval_get(c, n->inputs[1], v, 1) > 0) { axis = v[0]; got = 1; }
                else {
                    const onnx_initializer_t *ii = onnx_find_initializer(c->onnx, n->inputs[1]);
                    if (!ii) ii = find_constant_tensor(c->onnx, n->inputs[1]);
                    if (ii && ii->raw_data) {
                        if (ii->data_type == ONNX_DTYPE_INT64)      { axis = ((const int64_t *)ii->raw_data)[0]; got = 1; }
                        else if (ii->data_type == ONNX_DTYPE_INT32) { axis = ((const int32_t *)ii->raw_data)[0]; got = 1; }
                    }
                }
            }
            if (!got) {
                fprintf(stderr, "[onnx] CumSum %s: axis is not known at build time\n",
                        n->outputs[0]);
                return -1;
            }
        }

        int64_t exclusive = onnx_attr_int(n, "exclusive", 0);
        int64_t reverse   = onnx_attr_int(n, "reverse", 0);
        if (exclusive || reverse) {
            /* Both are expressible -- shift by one, or flip the axis -- but
             * neither has a caller yet, and an untested path that silently
             * returns the inclusive answer is worse than a refusal. */
            fprintf(stderr, "[onnx] CumSum %s: exclusive=%lld reverse=%lld not supported\n",
                    n->outputs[0], (long long)exclusive, (long long)reverse);
            return -1;
        }

        int nd = tmap_get_ndims(c, n->inputs[0]);
        if (nd <= 0) nd = (int)ggml_n_dims(a);
        if (nd < 1) nd = 1;
        if (axis < 0) axis += nd;
        int ggml_d = nd - 1 - (int)axis;
        if (ggml_d < 0) ggml_d = 0;
        if (ggml_d > 3) {
            fprintf(stderr, "[onnx] CumSum %s: axis %lld of a rank-%d input maps to "
                            "ggml dim %d, past what permute reaches\n",
                    n->outputs[0], (long long)axis, nd, ggml_d);
            return -1;
        }

        struct ggml_tensor *x = a;
        if (x->type != GGML_TYPE_F32) x = ggml_cast_numeric(c->ctx, x, GGML_TYPE_F32);

        if (ggml_d == 0) {
            out = ggml_cumsum(c->ctx, x);
        } else {
            int ax[4] = {0, 1, 2, 3};
            ax[0] = ggml_d; ax[ggml_d] = 0;
            struct ggml_tensor *p = ggml_cont(c->ctx,
                ggml_permute(c->ctx, x, ax[0], ax[1], ax[2], ax[3]));
            struct ggml_tensor *cs = ggml_cumsum(c->ctx, p);
            out = ggml_cont(c->ctx,
                ggml_permute(c->ctx, cs, ax[0], ax[1], ax[2], ax[3]));
        }
        out_nd = nd;
    }

    /* ── InstanceNormalization ──────────────────────────────────── */
    else if (strcmp(op, "InstanceNormalization") == 0) {
        if (!a || !b) return -1;
        /* Normalize each (batch, channel) plane over its spatial extent, then
         * scale and shift per channel.
         *
         * ONNX [N,C,H,W] is ggml [W,H,C,N], so the spatial axes are exactly
         * ne[0] and ne[1] -- the leading pair -- and collapsing them into one
         * makes this ggml_norm, which normalizes along ne[0].  That is the
         * whole op: no per-op kernel, and it lands on the same Vulkan shader
         * every other norm uses.
         *
         * scale (input 1) and bias (input 2) are per channel, i.e. along
         * ne[2] after the reshape back. */
        struct ggml_tensor *bias = NULL;
        if (n->n_inputs > 2 && n->inputs[2][0] != '\0')
            bias = tmap_get(c, n->inputs[2]);

        float eps = onnx_attr_float(n, "epsilon", 1e-5f);

        int nd = tmap_get_ndims(c, n->inputs[0]);
        if (nd <= 0) nd = (int)ggml_n_dims(a);
        if (nd < 3) {
            fprintf(stderr, "[onnx] InstanceNormalization %s: rank %d, expected at least 3\n",
                    n->outputs[0], nd);
            return -1;
        }

        struct ggml_tensor *x = a;
        if (x->type != GGML_TYPE_F32) x = ggml_cast_numeric(c->ctx, x, GGML_TYPE_F32);

        /* Spatial extent = every ggml axis below the channel axis. */
        int ch_d = nd - 2;              /* ONNX C is at ggml dim nd-1-1 */
        if (ch_d < 1 || ch_d > 3) {
            fprintf(stderr, "[onnx] InstanceNormalization %s: channel at ggml dim %d\n",
                    n->outputs[0], ch_d);
            return -1;
        }
        int64_t spatial = 1;
        for (int d = 0; d < ch_d; d++) spatial *= x->ne[d];
        int64_t C = x->ne[ch_d];
        int64_t N = 1;
        for (int d = ch_d + 1; d < GGML_MAX_DIMS; d++) N *= x->ne[d];

        struct ggml_tensor *flat = ggml_reshape_3d(c->ctx, x, spatial, C, N);
        struct ggml_tensor *nrm  = ggml_norm(c->ctx, flat, eps);

        /* scale/bias are [C]; as ggml tensors they are ne=[C], which lines up
         * with the middle axis of [spatial, C, N] by broadcast only after
         * being viewed as [1, C, 1]. */
        struct ggml_tensor *g = ggml_reshape_3d(c->ctx, b, 1, C, 1);
        out = ggml_mul(c->ctx, nrm, g);
        if (bias) {
            struct ggml_tensor *bt = ggml_reshape_3d(c->ctx, bias, 1, C, 1);
            out = ggml_add(c->ctx, out, bt);
        }
        out = onnx_reshape_nd(c->ctx, out, x->ne, nd > GGML_MAX_DIMS ? GGML_MAX_DIMS : nd);
        out_nd = nd;
    }

    /* ── Einsum ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Einsum") == 0) {
        if (!a || !b) return -1;
        /* Not a general einsum: the forms handled are the ones that are a
         * batched matmul written in index notation, which is what exporters
         * emit.  Anything else is refused by name rather than approximated.
         *
         * Mask2Former's mask head ends in "bqc,bchw->bqhw": queries against a
         * per-pixel embedding.  Folding hw into one axis makes it
         * [B,Q,C] x [B,C,HW] -> [B,Q,HW], a plain batched matmul, and the
         * result is reshaped back. */
        char eq[128] = {0};
        int eqn = onnx_attr_str(n, "equation", eq, sizeof(eq));
        if (eqn <= 0) {
            fprintf(stderr, "[onnx] Einsum %s: no equation attribute\n", n->outputs[0]);
            return -1;
        }
        /* strip spaces so "bqc, bchw -> bqhw" compares equal to the packed form */
        {
            int w = 0;
            for (int r = 0; eq[r]; r++) if (eq[r] != ' ') eq[w++] = eq[r];
            eq[w] = '\0';
        }

        int nd_a = tmap_get_ndims(c, n->inputs[0]);
        int nd_b = tmap_get_ndims(c, n->inputs[1]);
        if (nd_a <= 0) nd_a = (int)ggml_n_dims(a);
        if (nd_b <= 0) nd_b = (int)ggml_n_dims(b);

        if (strcmp(eq, "bqc,bchw->bqhw") == 0) {
            /* ggml: a = [C,Q,B], b = [W,H,C,B].  mul_mat contracts ne[0], so b
             * needs C there: fold [W,H] into one axis, then swap it with C. */
            int64_t W = b->ne[0], H = b->ne[1], C = b->ne[2], B = b->ne[3];
            if (a->ne[0] != C) {
                fprintf(stderr, "[onnx] Einsum %s: c mismatch (%lld vs %lld)\n",
                        n->outputs[0], (long long)a->ne[0], (long long)C);
                return -1;
            }
            struct ggml_tensor *b3 = ggml_reshape_3d(c->ctx, b, W * H, C, B);
            struct ggml_tensor *bt = ggml_cont(c->ctx,
                ggml_permute(c->ctx, b3, 1, 0, 2, 3));      /* [C, WH, B] */
            /* mul_mat([C,Q,B], [C,WH,B]) gives [Q, WH, B]: the output's fast
             * axis is Q, but bqhw wants the spatial pair there.  Swapping the
             * two before the reshape is what makes the values land in the
             * right places -- without it the numbers are all correct and all
             * in the wrong order, which a length check would not catch. */
            struct ggml_tensor *mm = ggml_mul_mat(c->ctx, a, bt);
            struct ggml_tensor *sw = ggml_cont(c->ctx,
                ggml_permute(c->ctx, mm, 1, 0, 2, 3));      /* [WH, Q, B] */
            out = ggml_reshape_4d(c->ctx, sw, W, H, a->ne[1], B);
            out_nd = 4;
        } else {
            fprintf(stderr, "[onnx] Einsum %s: equation '%s' not supported "
                            "(only batched-matmul forms are)\n", n->outputs[0], eq);
            return -1;
        }
    }
    /* ── Mod ────────────────────────────────────────────────────── */
    else if (strcmp(op, "Mod") == 0) {
        if (!a || !b) return -1;
        /* Two different operations share this op name, chosen by an attribute:
         *   fmod=0 (default) -- result takes the sign of the DIVISOR (Python %)
         *   fmod=1           -- result takes the sign of the DIVIDEND (C fmod)
         * They differ only in how the quotient is rounded, so one expression
         * covers both: a - b*round(a/b), with floor for the first and trunc
         * for the second.  Both were checked against R's %% and C's fmod over
         * a sign grid before this was written.
         *
         * The spec also ties fmod=0 to integer types and fmod=1 to floats,
         * but nothing here depends on the type: whatever ggml is holding,
         * the arithmetic is the same. */
        int64_t fmod_attr = onnx_attr_int(n, "fmod", 0);

        struct ggml_tensor *x = a, *y = b;
        onnx_broadcast_prepare(c->ctx, &x, &y);
        /* Both operands must be float for the division below; ONNX allows
         * integer input for fmod=0, which arrives here as I32. */
        if (x->type != GGML_TYPE_F32) x = ggml_cast_numeric(c->ctx, x, GGML_TYPE_F32);
        if (y->type != GGML_TYPE_F32) y = ggml_cast_numeric(c->ctx, y, GGML_TYPE_F32);

        struct ggml_tensor *q = ggml_div(c->ctx, x, y);
        q = fmod_attr ? ggml_trunc(c->ctx, q) : ggml_floor(c->ctx, q);
        out = ggml_sub(c->ctx, x, ggml_mul(c->ctx, y, q));

        /* cval propagation.  Mod is shape arithmetic far more often than it is
         * data arithmetic: Swin computes its window padding as
         * (window - size %% window), and without the value travelling on, Pad
         * receives zeros, pads nothing, and the block's 6-D window partition
         * then gets a shape that does not match its own element count. */
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 > 0) {
                int64_t result[ONNX_MAX_DIMS];
                int nr = na2 > nb2 ? na2 : nb2;
                if (nr > ONNX_MAX_DIMS) nr = ONNX_MAX_DIMS;
                for (int j = 0; j < nr; j++) {
                    int64_t va = cv_a[j < na2 ? j : 0];
                    int64_t vb = cv_b[j < nb2 ? j : 0];
                    if (vb == 0) { result[j] = 0; continue; }
                    int64_t r = va % vb;               /* C: sign of dividend */
                    if (!fmod_attr && r != 0 && ((r < 0) != (vb < 0)))
                        r += vb;                       /* fmod=0: sign of divisor */
                    result[j] = r;
                }
                cval_put(c, n->outputs[0], result, nr);
            }
        }
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
            size_t psz = 0;
            const void *pay = onnx_init_payload(mi, &psz);
            if (pay && psz >= 4 && mi->data_type == ONNX_DTYPE_FLOAT)
                memcpy(&min_val, pay, sizeof(float));
        } else if (!min_t) {
            min_val = onnx_attr_float(n, "min", min_val);
        }
        if (max_t && n->n_inputs > 2 && n->inputs[2][0] != '\0') {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[2]);
            if (!mi) mi = find_constant_tensor(c->onnx, n->inputs[2]);
            size_t psz = 0;
            const void *pay = onnx_init_payload(mi, &psz);
            if (pay && psz >= 4 && mi->data_type == ONNX_DTYPE_FLOAT)
                memcpy(&max_val, pay, sizeof(float));
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
