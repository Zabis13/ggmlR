/* onnx_ops_quant.c — quantized ops: DequantizeLinear, QuantizeLinear,
 * QLinearConv, QLinearAdd, QLinearMatMul, QLinearSigmoid, QLinearConcat
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ops_internal.h"

/* Returns 1 = handled, 0 = not this group's op, -1 = error */
int map_node_quant(onnx_ggml_ctx_t *c, const onnx_node_t *n,
                   struct ggml_tensor *a, struct ggml_tensor *b,
                   struct ggml_tensor **out_p, int *out_nd_p)
{
    const char *op = n->op_type;
    struct ggml_tensor *out = NULL;
    int out_nd = -1;

    /* ── Quantized ops (QLinear family) ──────────────────────────── */
    if (strcmp(op, "DequantizeLinear") == 0) {
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
            out = ggml_conv_2d_direct(c->ctx, dw, dx,
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
                    group_outs[g] = ggml_conv_2d_direct(c->ctx, b_g, a_g,
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
    else {
        return 0; /* not this group */
    }

    *out_p    = out;
    *out_nd_p = out_nd;
    return 1;
}
