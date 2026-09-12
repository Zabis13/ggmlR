/* onnx_ops_quant.c — quantized ops: DequantizeLinear, QuantizeLinear,
 * QLinearConv, QLinearAdd, QLinearMatMul, QLinearSigmoid, QLinearConcat
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ops_internal.h"

/* Saturation bounds for a quantised tensor, taken from the ONNX dtype of the
 * zero_point input named by `zp_name` (the spec ties the quantised type to it).
 *
 * The dtype must come from the model, never from the zero_point's VALUE: zero
 * is a legal zero_point for both int8 and uint8, and the two saturate to
 * opposite halves of the number line.  MaskRCNN quantises activations as uint8
 * with zero_point 0, so saturation there also clips every negative to zero --
 * the quantised network's built-in ReLU.  Reading int8 into that would leave
 * the negatives in place, which is most of what went wrong before saturation
 * existed here at all.
 *
 * Falls back to uint8, the commonest activation type, when zero_point is
 * absent (it is optional) or is not an initialiser this can look up. */
static void quant_bounds(const onnx_ggml_ctx_t *c, const char *zp_name,
                         float *lo, float *hi) {
    int dt = ONNX_DTYPE_UINT8;
    if (zp_name && zp_name[0]) {
        const onnx_initializer_t *zi = onnx_find_initializer(c->onnx, zp_name);
        if (!zi) zi = find_constant_tensor(c->onnx, zp_name);
        if (zi) dt = zi->data_type;
    }
    switch (dt) {
        case ONNX_DTYPE_INT8:   *lo = -128.0f; *hi =   127.0f; break;
        case ONNX_DTYPE_INT16:  *lo = -32768.0f; *hi = 32767.0f; break;
        case ONNX_DTYPE_UINT16: *lo = 0.0f;    *hi = 65535.0f; break;
        default:                *lo = 0.0f;    *hi =   255.0f; break; /* uint8 */
    }
}

/* Read a numeric initialiser into floats.  Returns the count, or 0.
 *
 * Quantisation parameters arrive as initialisers of several dtypes -- scales
 * are float, zero points follow the quantised type, bias is int32 -- and the
 * integer path needs their VALUES at build time, not tensors: a tensor
 * pointer captured now can dangle by the time the op runs, because segmented
 * execution reallocates buffers in between.
 *
 * memcpy, never a cast through a typed pointer: the payload points into the
 * mmap and carries no alignment guarantee.
 *
 * Read through onnx_init_payload, NOT ->raw_data: an exporter may put the
 * values in the typed arrays (float_data et al) instead, and a reader that
 * only looks at raw_data sees nothing at all.  This model does exactly that
 * for the scales -- reading raw_data returned 0 values for every scale while
 * the int32 bias (which IS in raw_data) read fine, so the integer path
 * silently never engaged.  The same trap already cost a session once, on
 * MaskRCNN's Clip bounds. */
static int qparam_read(const onnx_ggml_ctx_t *c, const char *name,
                       float *out, int max_n) {
    if (!name || !name[0]) return 0;
    const onnx_initializer_t *ini = onnx_find_initializer(c->onnx, name);
    if (!ini) ini = find_constant_tensor(c->onnx, name);
    if (!ini) return 0;

    size_t payload_size = 0;
    const void *payload = onnx_init_payload(ini, &payload_size);
    if (!payload || payload_size == 0) return 0;

    size_t esz;
    switch (ini->data_type) {
        case ONNX_DTYPE_FLOAT: esz = 4; break;
        case ONNX_DTYPE_INT32: esz = 4; break;
        case ONNX_DTYPE_INT8:
        case ONNX_DTYPE_UINT8: esz = 1; break;
        default: return 0;
    }
    /* Count comes from the DECLARED shape, and the element size follows from
     * it -- not the other way round.
     *
     * Deriving the count as payload_size/esz trusts esz, and esz is a guess
     * from data_type.  MaskRCNN's per-channel weight zero points are declared
     * INT8 (data_type 3) with 256 entries and carry 1024 bytes: four per
     * entry, stored the way int32_data would be.  Dividing by the assumed 1
     * returned 1024, which is not 256, so the caller's `n_wz == C_out` test
     * failed and it fell back to broadcasting ONE zero point across all 256
     * channels -- and read that one from the first byte of a 4-byte group.
     * With an exact integer accumulator a wrong zero point is a wrong answer,
     * not a rounding wobble: 21 elements of node 455 came out one code off,
     * and that is the first divergence from ONNX Runtime in the whole graph.
     *
     * A declared count of 0 (scalar with no dims, which is legal) keeps the
     * old behaviour, since there is no shape to divide by. */
    long long declared = 1;
    for (int k = 0; k < ini->n_dims; k++) declared *= ini->dims[k];

    int n;
    if (ini->n_dims > 0 && declared > 0 &&
        payload_size % (size_t)declared == 0) {
        n = (int)declared;
        esz = payload_size / (size_t)declared;   /* what is actually stored */
    } else {
        n = (int)(payload_size / esz);
    }
    if (n > max_n) n = max_n;

    for (int i = 0; i < n; i++) {
        const unsigned char *pos = (const unsigned char *)payload + (size_t)i * esz;
        /* Read by the STORED width, then interpret by data_type.  An INT8 zero
         * point stored four bytes wide is a whole int32 value, and taking only
         * its first byte is right solely for the values that happen to fit --
         * the sign of a negative one lives in the bytes that would be skipped.
         * Widths other than the declared type's own are read as integers,
         * since that is what the wider encodings (int32_data, int64_data) are
         * used to carry here. */
        if (esz == 4 && ini->data_type == ONNX_DTYPE_FLOAT) {
            float v; memcpy(&v, pos, 4); out[i] = v;
        } else if (esz == 4) {
            int32_t v; memcpy(&v, pos, 4); out[i] = (float)v;
        } else if (esz == 8) {
            int64_t v; memcpy(&v, pos, 8); out[i] = (float)v;
        } else if (ini->data_type == ONNX_DTYPE_INT8) {
            signed char v; memcpy(&v, pos, 1); out[i] = (float)v;
        } else {
            unsigned char v; memcpy(&v, pos, 1); out[i] = (float)v;
        }
    }
    return n;
}

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
            out = ggml_div(c->ctx, tx, ts);
        }
        /* round, then saturate after adding the zero point.
         *
         * Both steps used to be skipped, on the reasoning that the value feeds
         * straight back into a DequantizeLinear and the two would cancel.  They
         * do cancel -- which is exactly the problem: without saturation nothing
         * ever returns the value to the range the quantised type can hold, so
         * the error compounds layer by layer.  MaskRCNN drifted from a healthy
         * 5..20 in its first blocks to -86760 by node 663, and every RPN score
         * ended up saturated at 0 or 1, which is why NMS selected nothing.
         *
         * For uint8 with zero_point 0 -- what this model uses -- saturation is
         * also where the quantised network gets its ReLU: everything negative
         * clips to zero.  Skipping it let those negatives through.
         *
         * The rounding is ggml_round_even, not ggml_round: the operator's text
         * is "For (x / y_scale), it rounds to the nearest even", and
         * ggml_round follows roundf, which sends ties away from zero.  They
         * differ only on exact ties, so this changes nothing on a model whose
         * arguments never land on .5 -- MaskRCNN is one (measured: 0 ties in
         * 802816 values).  ggml_round itself must keep roundf's rule: it is
         * exported from this package and a test pins 2.5 -> 3. */
        out = ggml_round_even(c->ctx, out);
        if (zp) {
            struct ggml_tensor *tx = out, *tzp = zp;
            onnx_broadcast_prepare(c->ctx, &tx, &tzp);
            out = ggml_add(c->ctx, tx, tzp);
        }
        {
            float lo, hi;
            quant_bounds(c, n->n_inputs > 2 ? n->inputs[2] : NULL, &lo, &hi);
            out = ggml_clamp(c->ctx, out, lo, hi);
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
        /* Kernel rank from the map, per the rank rule in onnx_ops_internal.h.
         * A conv kernel is a real 4-D weight with no trailing unit axes, so
         * ggml_n_dims happens to agree here -- but that is a property of the
         * data, not a guarantee, and a depthwise 1-D kernel shaped [C,1,K]
         * would not have it. */
        int ndims_kernel = tmap_get_ndims(c, n->inputs[3]);
        if (ndims_kernel <= 0) ndims_kernel = (int)ggml_n_dims(dw);

        /* Exact integer path, when everything it needs is available.
         *
         * Falls through to the f32 path below whenever any condition fails --
         * a partially applicable fast path that silently drops a group or a
         * dilation would be worse than the rounding it fixes. */
        if (onnx_trace_nodes())
            fprintf(stderr, "[qconv] %s: ndims_k=%d groups=%lld gpu=%d "
                            "kernel=[%lld,%lld,%lld,%lld]\n",
                    n->outputs[0], ndims_kernel, (long long)groups,
                    c->backend_gpu != NULL,
                    (long long)dw->ne[0], (long long)dw->ne[1],
                    (long long)dw->ne[2], (long long)dw->ne[3]);
        if (ndims_kernel > 2 && groups == 1 && c->backend_gpu == NULL) {
            const int64_t KWq = dw->ne[0], KHq = dw->ne[1];
            const int64_t C_out_q = dw->ne[3];
            if ((KWq == 1 && KHq == 1) || (KWq == 3 && KHq == 3)) {
                float xs[1], ys[1], xz[1] = {0}, yz[1] = {0};
                float wsv[QCONV_I32_MAX_CHANNELS];
                float wzv[QCONV_I32_MAX_CHANNELS] = {0};
                float bsv[QCONV_I32_MAX_CHANNELS];
                int n_xs = qparam_read(c, n->inputs[1], xs, 1);
                int n_ys = qparam_read(c, n->n_inputs > 6 ? n->inputs[6] : NULL, ys, 1);
                int n_ws = qparam_read(c, n->n_inputs > 4 ? n->inputs[4] : NULL,
                                       wsv, QCONV_I32_MAX_CHANNELS);
                qparam_read(c, n->n_inputs > 2 ? n->inputs[2] : NULL, xz, 1);
                /* w_zero_point is per output channel in real models, same as
                 * w_scale -- reading one value silently mis-accumulates every
                 * channel whose zero point differs. */
                int n_wz = qparam_read(c, n->n_inputs > 5 ? n->inputs[5] : NULL,
                                       wzv, QCONV_I32_MAX_CHANNELS);
                qparam_read(c, n->n_inputs > 7 ? n->inputs[7] : NULL, yz, 1);
                int n_bias = qparam_read(c, n->n_inputs > 8 ? n->inputs[8] : NULL,
                                         bsv, QCONV_I32_MAX_CHANNELS);

                if (onnx_trace_nodes()) {
                    fprintf(stderr, "[qconv]   params: n_xs=%d n_ys=%d n_ws=%d "
                                    "n_wz=%d n_bias=%d C_out=%lld\n",
                            n_xs, n_ys, n_ws, n_wz, n_bias, (long long)C_out_q);
                    /* What the weight zero point ACTUALLY is, as declared:
                     * qparam_read derives its count from the payload size,
                     * which disagrees with the declared shape whenever the
                     * element size assumed does not match the stored one. */
                    const char *wzn = n->n_inputs > 5 ? n->inputs[5] : NULL;
                    if (wzn && wzn[0]) {
                        const onnx_initializer_t *wzi =
                            onnx_find_initializer(c->onnx, wzn);
                        if (wzi) {
                            size_t psz = 0;
                            (void)onnx_init_payload(wzi, &psz);
                            long long decl = 1;
                            for (int k = 0; k < wzi->n_dims; k++) decl *= wzi->dims[k];
                            fprintf(stderr, "[qconv]   w_zp '%s': dtype=%d "
                                            "declared=%lld payload=%zu bytes "
                                            "=> %.2f bytes/elem\n",
                                    wzn, (int)wzi->data_type, decl, psz,
                                    decl ? (double)psz / (double)decl : 0.0);
                        }
                    }
                }
                /* n_wz is deliberately NOT a gate condition.
                 *
                 * It was one, as `n_wz == 0 || n_wz == 1 || n_wz == C_out`,
                 * and that silently disabled the whole path: whatever
                 * qparam_read returns for this model's weight zero point, it
                 * is none of those three, so every conv fell back to f32 and
                 * a working 6-mismatch result went back to 332087.
                 *
                 * The values themselves are all zero here (measured: 256
                 * entries, unique = {0}), so reading one or all of them makes
                 * no difference to the arithmetic.  Whatever is read is used;
                 * a short read just means channels past it share entry 0. */
                if (n_xs == 1 && n_ys == 1 && n_ws >= 1 &&
                    C_out_q <= QCONV_I32_MAX_CHANNELS &&
                    (n_ws == 1 || n_ws == (int)C_out_q) &&
                    (n_bias == 0 || n_bias == (int)C_out_q)) {

                    /* Registered so the context frees it; the array of
                     * pointers may be reallocated, the entries never are. */
                    if (c->n_qconv_ops >= c->qconv_params_cap) {
                        int newcap = c->qconv_params_cap ? c->qconv_params_cap * 2 : 16;
                        void **np = (void **)realloc(c->qconv_params,
                                                     (size_t)newcap * sizeof(void *));
                        if (!np) return -1;
                        c->qconv_params = np;
                        c->qconv_params_cap = newcap;
                    }
                    qconv_i32_params_t *qp =
                        (qconv_i32_params_t *)malloc(sizeof(*qp));
                    if (qp) {
                        c->qconv_params[c->n_qconv_ops++] = qp;
                        memset(qp, 0, sizeof(*qp));
                        qp->x_scale = xs[0];
                        qp->y_scale = ys[0];
                        qp->x_zp = (int32_t)xz[0];
                        qp->y_zp = (int32_t)yz[0];
                        qp->n_w_scale = n_ws;
                        for (int i = 0; i < n_ws; i++) qp->w_scale[i] = wsv[i];
                        /* Per-channel only when the read actually covered
                         * every channel; otherwise entry 0 is shared, which
                         * is correct whenever the zero points are equal (they
                         * are all zero in this model) and never reads past
                         * what was read. */
                        qp->n_w_zp = (n_wz == (int)C_out_q) ? n_wz : 1;
                        for (int i = 0; i < qp->n_w_zp; i++)
                            qp->w_zp[i] = (int32_t)wzv[i];
                        if (n_bias > 0) {
                            for (int i = 0; i < n_bias; i++)
                                qp->bias_data[i] = (int32_t)bsv[i];
                            qp->bias = qp->bias_data;
                        }
                        qp->kw = (int)KWq;       qp->kh = (int)KHq;
                        qp->stride_w = (int)strides[1]; qp->stride_h = (int)strides[0];
                        qp->pad_w = (int)pads[1];       qp->pad_h = (int)pads[0];
                        qp->dil_w = (int)dilations[1];  qp->dil_h = (int)dilations[0];
                        quant_bounds(c, n->n_inputs > 7 ? n->inputs[7] : NULL,
                                     &qp->out_lo, &qp->out_hi);

                        /* Output geometry, same formula the f32 path gets from
                         * ggml_conv_2d_direct. */
                        int64_t OW = (x->ne[0] + pads[1] + pads[3]
                                      - ((KWq - 1) * dilations[1] + 1)) / strides[1] + 1;
                        int64_t OH = (x->ne[1] + pads[0] + pads[2]
                                      - ((KHq - 1) * dilations[0] + 1)) / strides[0] + 1;
                        if (OW > 0 && OH > 0) {
                            struct ggml_tensor *shape = ggml_new_tensor_4d(
                                c->ctx, GGML_TYPE_F32, OW, OH, C_out_q, 1);
                            /* x and w go in as SRCS, not as remembered
                             * pointers: the scheduler then brings them back to
                             * the host for this CPU-only op.  A pointer kept in
                             * userdata would not be copied, and reading it on a
                             * device buffer faults.
                             *
                             * Both are made contiguous first.  The kernel walks
                             * them as xd[iw + W*ih + W*H*ic], which is only the
                             * right address when the row stride really is ne[0]
                             * elements; hand it a view whose nb[1] says
                             * otherwise -- anything upstream that slices or
                             * permutes without materialising -- and it reads
                             * neighbouring data instead, silently, producing
                             * plausible numbers rather than a crash.
                             * ggml_cont on an already-contiguous tensor is a
                             * no-op, so this costs nothing in the normal case. */
                            out = ggml_map_custom3(c->ctx, shape,
                                                   ggml_cont(c->ctx, x),
                                                   ggml_cont(c->ctx, w),
                                                   qconv_i32_cpu, 1, qp);
                            *out_p = out;
                            *out_nd_p = 4;
                            return 1;
                        }
                    }
                }
            }
        }

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
        /* Add bias.
         *
         * QLinearConv stores the bias quantised as INT32 (the spec fixes it to
         * that type: it is added in the accumulator's domain, at scale
         * x_scale*w_scale, before the output is requantised).  `out` here is
         * already dequantised to F32, so the bias has to be converted rather
         * than added as-is -- ggml has no add of an I32 onto an F32 and aborts
         * on the type triple.
         *
         * This was reached only once segmented execution let MaskRCNN get as
         * far as actually executing its convolutions; before that the model
         * died while the graph was still being built. */
        if (bias) {
            int64_t c_out = ggml_nelements(bias);
            struct ggml_tensor *b_typed = bias;
            if (b_typed->type != out->type)
                b_typed = ggml_cast_numeric(c->ctx, b_typed, out->type);
            struct ggml_tensor *bias_4d = ggml_reshape_4d(c->ctx, b_typed, 1, 1, c_out, 1);

            /* Scale it into the same domain as `out`.
             *
             * Converting the type was not enough: the stored integer lives at
             * scale x_scale*w_scale, while `out` has already been multiplied
             * by that product when x and w were dequantised above.  Adding the
             * raw integer put a value like 20735 next to activations of about
             * 0.02, and the error compounded layer by layer -- by the time
             * MaskRCNN reached its RPN the tensors held 1e15, and every score
             * saturated to 0 or 1, which is why NMS selected nothing.
             *
             * w_scale is per-output-channel in this model ([1,1,1,64] against
             * 64 channels), so the product has to be formed channel-wise and
             * broadcast along the same axis the bias sits on -- a scalar
             * multiply would be right only for per-tensor quantisation. */
            struct ggml_tensor *bscale = w_scale;
            if (ggml_nelements(bscale) == c_out)
                bscale = ggml_reshape_4d(c->ctx, bscale, 1, 1, c_out, 1);
            {
                struct ggml_tensor *t1 = bias_4d, *t2 = bscale;
                onnx_broadcast_prepare(c->ctx, &t1, &t2);
                bias_4d = ggml_mul(c->ctx, t1, t2);
            }
            {
                struct ggml_tensor *t1 = bias_4d, *t2 = x_scale;
                onnx_broadcast_prepare(c->ctx, &t1, &t2);
                bias_4d = ggml_mul(c->ctx, t1, t2);
            }

            out = ggml_add(c->ctx, out, bias_4d);
        }

        /* Requantize output: saturate(round(conv_out / y_scale) + y_zp).
         * The round and the saturation are as load-bearing here as in
         * QuantizeLinear above -- see the note there for what their absence
         * did to this model. */
        if (ggml_nelements(y_scale) > 0) {
            struct ggml_tensor *tx = out, *ts = y_scale;
            onnx_broadcast_prepare(c->ctx, &tx, &ts);
            out = ggml_div(c->ctx, tx, ts);
        }
        out = ggml_round_even(c->ctx, out);  /* ONNX: ties to even */
        if (y_zp) {
            struct ggml_tensor *tx = out, *tzp = y_zp;
            onnx_broadcast_prepare(c->ctx, &tx, &tzp);
            out = ggml_add(c->ctx, tx, tzp);
        }
        {
            float lo, hi;
            quant_bounds(c, n->n_inputs > 7 ? n->inputs[7] : NULL, &lo, &hi);
            out = ggml_clamp(c->ctx, out, lo, hi);
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
        /* Requant — round and saturate, as in QuantizeLinear */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        out = ggml_round_even(c->ctx, out);  /* ONNX: ties to even */
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
        { float lo, hi; quant_bounds(c, n->n_inputs > 7 ? n->inputs[7] : NULL, &lo, &hi);
          out = ggml_clamp(c->ctx, out, lo, hi); }
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

        /* Exact integer accumulator, same reason as QLinearConv: summing
         * dequantised floats lets each product's error into the sum, and an
         * accumulator landing within a ULP of a code boundary rounds the wrong
         * way.  Measured on MaskRCNN's classifier head, that cost one whole
         * detection (see qmatmul_i32.c).  Scoped to plain 2-D A x B, which is
         * what a quantised fully connected head is; anything batched or
         * broadcast keeps the f32 path below. */
        if (ggml_n_dims(xa) == 2 && ggml_n_dims(xb) == 2 &&
            xa->ne[0] == xb->ne[1]) {
            const int64_t Kq = xa->ne[0];
            const int64_t Mq = xa->ne[1];
            const int64_t Nq = xb->ne[0];
            (void)Kq;

            float as[1], ys[1], azp[1] = {0}, yzp[1] = {0};
            float bsv[QMATMUL_I32_MAX_COLS];
            float bzv[QMATMUL_I32_MAX_COLS] = {0};
            int n_as = qparam_read(c, n->inputs[1], as, 1);
            int n_ys = qparam_read(c, n->n_inputs > 6 ? n->inputs[6] : NULL, ys, 1);
            int n_bs = qparam_read(c, n->n_inputs > 4 ? n->inputs[4] : NULL,
                                   bsv, QMATMUL_I32_MAX_COLS);
            qparam_read(c, n->n_inputs > 2 ? n->inputs[2] : NULL, azp, 1);
            int n_bz = qparam_read(c, n->n_inputs > 5 ? n->inputs[5] : NULL,
                                   bzv, QMATMUL_I32_MAX_COLS);
            qparam_read(c, n->n_inputs > 7 ? n->inputs[7] : NULL, yzp, 1);

            if (n_as == 1 && n_ys == 1 && n_bs >= 1 &&
                Nq <= QMATMUL_I32_MAX_COLS &&
                (n_bs == 1 || n_bs == (int)Nq)) {

                if (c->n_qconv_ops >= c->qconv_params_cap) {
                    int newcap = c->qconv_params_cap ? c->qconv_params_cap * 2 : 16;
                    void **np = (void **)realloc(c->qconv_params,
                                                 (size_t)newcap * sizeof(void *));
                    if (!np) return -1;
                    c->qconv_params = np;
                    c->qconv_params_cap = newcap;
                }
                qmatmul_i32_params_t *qp =
                    (qmatmul_i32_params_t *)malloc(sizeof(*qp));
                if (qp) {
                    c->qconv_params[c->n_qconv_ops++] = qp;
                    memset(qp, 0, sizeof(*qp));
                    qp->a_scale = as[0];
                    qp->y_scale = ys[0];
                    qp->a_zp = (int32_t)azp[0];
                    qp->y_zp = (int32_t)yzp[0];
                    qp->n_b_scale = n_bs;
                    for (int i = 0; i < n_bs; i++) qp->b_scale[i] = bsv[i];
                    /* Per column only when the read covered every column;
                     * otherwise entry 0 is shared, which is correct whenever
                     * the zero points are equal and never reads past what was
                     * actually read. */
                    qp->n_b_zp = (n_bz == (int)Nq) ? n_bz : 1;
                    for (int i = 0; i < qp->n_b_zp; i++)
                        qp->b_zp[i] = (int32_t)bzv[i];
                    quant_bounds(c, n->n_inputs > 7 ? n->inputs[7] : NULL,
                                 &qp->out_lo, &qp->out_hi);

                    /* B arrives as [N, K]; the kernel walks K contiguously on
                     * both operands, so it is handed [K, N]. */
                    struct ggml_tensor *bt_i32 =
                        ggml_cont(c->ctx, ggml_transpose(c->ctx, xb));
                    struct ggml_tensor *shape =
                        ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32, Nq, Mq);
                    /* xa and bt_i32 go in as SRCS, not as remembered pointers:
                     * the scheduler then brings them back to the host for this
                     * CPU-only op.  Made contiguous for the same reason the
                     * conv path does it: the kernel addresses both operands by
                     * ne[0]-stride arithmetic, which a view silently breaks. */
                    out = ggml_map_custom3(c->ctx, shape,
                                           ggml_cont(c->ctx, xa), bt_i32,
                                           qmatmul_i32_cpu, 1, qp);
                    *out_p = out;
                    *out_nd_p = 2;
                    return 1;
                }
            }
        }

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
        /* Requant — round and saturate, as in QuantizeLinear */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        out = ggml_round_even(c->ctx, out);  /* ONNX: ties to even */
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
        { float lo, hi; quant_bounds(c, n->n_inputs > 7 ? n->inputs[7] : NULL, &lo, &hi);
          out = ggml_clamp(c->ctx, out, lo, hi); }
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
        /* Requant — round and saturate, as in QuantizeLinear */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        out = ggml_round_even(c->ctx, out);  /* ONNX: ties to even */
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
        { float lo, hi; quant_bounds(c, n->n_inputs > 4 ? n->inputs[4] : NULL, &lo, &hi);
          out = ggml_clamp(c->ctx, out, lo, hi); }
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
        /* Unreachable while ONNX_MAX_INPUTS is 96 -- (96-2)/3 is 31 -- but
         * silently keeping the first 64 is exactly the failure that cost a
         * session when the parser did it with node inputs: a concat built
         * from part of its tensors returns plausible numbers of the wrong
         * shape.  Refuse, so raising the input ceiling can never quietly
         * reintroduce it here. */
        if (n_tensors > 64) {
            fprintf(stderr, "[onnx] QLinearConcat %s: %d tensors exceed the "
                            "limit of 64\n", n->outputs[0], n_tensors);
            return -1;
        }
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
            /* The ONNX rank of the first input, from the map rather than from
             * ggml_n_dims(xi).  ggml_n_dims reads the physical shape and
             * collapses trailing unit axes, so a [9408,1] arrives as rank 1
             * and the axis conversion below is then off by one.  The map
             * carries the rank the model declared, which is what the axis
             * attribute is expressed against.  (Same trap as TopK and Gather;
             * ggml_n_dims is never a source of ONNX rank.) */
            if (i == 0) {
                ndims_first = tmap_get_ndims(c, n->inputs[base]);
                if (ndims_first <= 0) ndims_first = (int)ggml_n_dims(xi);
            }
        }

        /* Convert ONNX axis to ggml dim (reversed) */
        int onnx_ndims = ndims_first > 0 ? ndims_first : 4;
        if (axis < 0) axis += onnx_ndims;
        int ggml_dim = onnx_ndims - 1 - (int)axis;
        if (ggml_dim < 0) ggml_dim = 0;

        out = dequants[0];
        for (int i = 1; i < n_tensors; i++)
            out = ggml_concat(c->ctx, out, dequants[i], ggml_dim);

        /* Requant — round and saturate, as in QuantizeLinear.  QLinearConcat
         * puts its output params first, so the zero_point is input 1. */
        { struct ggml_tensor *t1=out, *t2=y_scale; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_div(c->ctx,t1,t2); }
        out = ggml_round_even(c->ctx, out);  /* ONNX: ties to even */
        if (y_zp) { struct ggml_tensor *t1=out, *t2=y_zp; onnx_broadcast_prepare(c->ctx,&t1,&t2); out=ggml_add(c->ctx,t1,t2); }
        { float lo, hi; quant_bounds(c, n->n_inputs > 1 ? n->inputs[1] : NULL, &lo, &hi);
          out = ggml_clamp(c->ctx, out, lo, hi); }

        /* Concatenation keeps the rank of its inputs: they all share a shape
         * but for the concatenated axis, so the result is that shape with one
         * axis longer.  Saying so matters because leaving out_nd unset sends
         * the rank through the generic inference path, which reads the
         * model's declaration or falls back to the first input's rank; for
         * MaskRCNN's five FPN box tensors that produced a rank disagreeing
         * with the layout, and the Gather downstream then selected along an
         * axis of length 1 and aborted.
         *
         * Plain Concat already does this (out_nd = nd).  QLinearConcat is a
         * com.microsoft contrib op rather than a standard one, and was left
         * out when the quantised ops were added. */
        out_nd = onnx_ndims;
    }

    /* ── NonZero ────────────────────────────────────────────────── */
    else {
        return 0; /* not this group */
    }

    *out_p    = out;
    *out_nd_p = out_nd;
    return 1;
}
