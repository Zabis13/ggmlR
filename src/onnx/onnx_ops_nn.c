/* onnx_ops_nn.c — neural network ops: normalization, pooling, conv, reduce
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ops_internal.h"

/* Returns 1 = handled, 0 = not this group's op, -1 = error */
int map_node_nn(onnx_ggml_ctx_t *c, const onnx_node_t *n,
                struct ggml_tensor *a, struct ggml_tensor *b,
                struct ggml_tensor **out_p, int *out_nd_p)
{
    const char *op = n->op_type;
    struct ggml_tensor *out = NULL;
    int out_nd = -1;

    /* ── Normalization ──────────────────────────────────────────── */
    if (strcmp(op, "BatchNormalization") == 0) {
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
            /* Standard 2D conv — use direct kernel (GGML_OP_CONV_2D) instead of
             * ggml_conv_2d (IM2COL + MUL_MAT) to avoid dispatch overhead on GPU.
             * ggml_conv_2d_direct expects kernel [KW,KH,IC,OC] which matches
             * the reversed-dims layout already applied to ONNX weights. */
            struct ggml_tensor *bk = ggml_is_contiguous(b) ? b : ggml_cont(c->ctx, b);
            struct ggml_tensor *ak = ggml_is_contiguous(a) ? a : ggml_cont(c->ctx, a);
            out = ggml_conv_2d_direct(c->ctx, bk, ak,
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
                /* {NULL} so the compiler can see group_outs[0] is defined even
                 * where it cannot prove groups >= 1; the guard below makes an
                 * empty loop impossible in practice. */
                struct ggml_tensor *group_outs[512] = {NULL};
                if (groups < 1 || groups > 512) { fprintf(stderr, "[onnx] Conv groups=%lld out of range\n", (long long)groups); return -1; }

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

                    group_outs[g] = ggml_conv_2d_direct(c->ctx, b_g, a_g,
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
            /* Where the channel sits depends on the convolution's rank: a 2D
             * output is [W,H,C,N] and a 1D output is [L,C,N], so the [1,1,C,1]
             * bias that broadcasts over the former does not line up with the
             * latter -- ggml_add refuses it outright, which is how whisper's
             * first Conv1d failed to load. */
            struct ggml_tensor *bias_4d = (ndims_kernel <= 3)
                ? ggml_reshape_4d(c->ctx, bias, 1, c_out, 1, 1)
                : ggml_reshape_4d(c->ctx, bias, 1, 1, c_out, 1);
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

        /* Use ONNX ndims (from tmap) to determine 1D vs 2D — ggml_n_dims()
         * squeezes trailing 1s and misclassifies e.g. [IC,OC,K,1] as 2D. */
        int ndims_kernel = tmap_get_ndims(c, n->inputs[1]);
        if (ndims_kernel <= 0) ndims_kernel = (int)ggml_n_dims(b);
        /* ggml conv ops require F16 kernel */
        struct ggml_tensor *bk = b;
        if (bk->type != GGML_TYPE_F16)
            bk = ggml_cast(c->ctx, bk, GGML_TYPE_F16);
        if (ndims_kernel <= 3) {
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

        /* Ensure F32 output (kernel is F16, output may inherit that type) */
        if (out->type != GGML_TYPE_F32)
            out = ggml_cast_numeric(c->ctx, out, GGML_TYPE_F32);

        /* Add bias if present */
        struct ggml_tensor *bias = get_input(c, n, 2);
        if (bias) {
            int64_t c_out = ggml_nelements(bias);
            /* Where the channel sits depends on the convolution's rank: a 2D
             * output is [W,H,C,N] and a 1D output is [L,C,N], so the [1,1,C,1]
             * bias that broadcasts over the former does not line up with the
             * latter -- ggml_add refuses it outright, which is how whisper's
             * first Conv1d failed to load. */
            struct ggml_tensor *bias_4d = (ndims_kernel <= 3)
                ? ggml_reshape_4d(c->ctx, bias, 1, c_out, 1, 1)
                : ggml_reshape_4d(c->ctx, bias, 1, 1, c_out, 1);
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
            } else if (ggml_dim < 4) {
                /* Reduce along dims 1-3: permute to bring target dim to ne[0],
                 * sum_rows, scale, permute back.
                 * perm[5] sized for 5D safety; ggml_permute takes only 4 args (dims 0-3) */
                int perm[5] = {0, 1, 2, 3, 4};
                perm[0] = ggml_dim; perm[ggml_dim] = 0;
                struct ggml_tensor *ap = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, a, perm[0], perm[1], perm[2], perm[3]));
                struct ggml_tensor *s = ggml_sum_rows(c->ctx, ap);
                float inv = 1.0f / (float)a->ne[ggml_dim];
                struct ggml_tensor *m = ggml_scale(c->ctx, s, inv);
                out = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, m, perm[0], perm[1], perm[2], perm[3]));
            } else {
                /* ggml_dim==4 (5D, reducing ONNX batch dim): not supported via permute,
                 * fall back to full mean */
                out = ggml_mean(c->ctx, a);
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
            } else if (ggml_dim < 4) {
                int perm[5] = {0, 1, 2, 3, 4};
                perm[0] = ggml_dim; perm[ggml_dim] = 0;
                struct ggml_tensor *ap = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, a, perm[0], perm[1], perm[2], perm[3]));
                struct ggml_tensor *s = ggml_sum_rows(c->ctx, ap);
                out = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, s, perm[0], perm[1], perm[2], perm[3]));
            } else {
                /* ggml_dim==4 (5D, reducing ONNX batch dim): fall back to full sum */
                out = ggml_sum(c->ctx, a);
            }
        } else {
            out = ggml_sum(c->ctx, a);
        }
    }

    /* ── ReduceMax / ReduceMin ─────────────────────────────────── */
    /* ggml has sum_rows but no max/min row reduction, so the extremum is
     * taken with a pooling window that spans the whole axis -- the same
     * trick GlobalAveragePool uses above.  Pooling works on ne[0]/ne[1],
     * so the reduced axis is permuted into ne[0] first.  ReduceMin is
     * ReduceMax on the negated input, negated back. */
    else if (strcmp(op, "ReduceMax") == 0 || strcmp(op, "ReduceMin") == 0) {
        if (!a) return -1;
        int is_min = (strcmp(op, "ReduceMin") == 0);

        /* The reduction below is built from ggml_neg and ggml_pool_2d, both of
         * which are float-only.  ONNX allows integer inputs here -- MaskRCNN
         * reduces a Cast-to-int64 tensor of detection counts -- so convert
         * first and let the ONNX rank bookkeeping treat it as before.  The
         * values involved are counts and indices, well inside what F32
         * represents exactly. */
        if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16 &&
            a->type != GGML_TYPE_BF16)
            a = ggml_cast_numeric(c->ctx, a, GGML_TYPE_F32);

        int64_t axes[ONNX_MAX_DIMS];
        int n_axes = 0;
        const onnx_attr_t *axes_attr = onnx_node_find_attr(n, "axes");
        if (axes_attr && axes_attr->n_ints > 0) {
            n_axes = axes_attr->n_ints;
            if (n_axes > ONNX_MAX_DIMS) n_axes = ONNX_MAX_DIMS;
            for (int d = 0; d < n_axes; d++) axes[d] = axes_attr->ints[d];
        }
        if (n_axes == 0 && n->n_inputs > 1)
            n_axes = cval_get(c, n->inputs[1], axes, ONNX_MAX_DIMS);

        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);
        for (int d = 0; d < n_axes; d++)
            if (axes[d] < 0) axes[d] += a_nd;

        /* Reduce over ne[0] of `src`, keeping the other axes. */
        struct ggml_tensor *src = a;
        int ggml_dim = 0;
        int perm[4] = {0, 1, 2, 3};
        int permuted = 0;

        if (n_axes == 1) {
            ggml_dim = a_nd - 1 - (int)axes[0];
            if (ggml_dim < 0) ggml_dim = 0;
            if (ggml_dim > 3) ggml_dim = 3;
            if (ggml_dim != 0) {
                perm[0] = ggml_dim; perm[ggml_dim] = 0;
                src = ggml_cont(c->ctx,
                    ggml_permute(c->ctx, a, perm[0], perm[1], perm[2], perm[3]));
                permuted = 1;
            }
        } else if (n_axes > 1) {
            /* Several axes at once: flatten everything into one row, which is
             * also what a missing `axes` means (reduce the whole tensor). */
            src = ggml_reshape_2d(c->ctx, ggml_cont(c->ctx, a),
                                  ggml_nelements(a), 1);
        } else {
            src = ggml_reshape_2d(c->ctx, ggml_cont(c->ctx, a),
                                  ggml_nelements(a), 1);
        }

        if (is_min) src = ggml_neg(c->ctx, src);

        /* One output per row: the window spans ne[0] and is one element tall.
         * pool_2d rather than pool_1d because the Vulkan backend implements
         * POOL_2D only -- pool_1d would force the op onto the CPU. */
        struct ggml_tensor *pooled =
            ggml_pool_2d(c->ctx, src, GGML_OP_POOL_MAX,
                         (int)src->ne[0], 1,
                         (int)src->ne[0], 1, 0, 0);

        if (is_min) pooled = ggml_neg(c->ctx, pooled);

        if (permuted)
            pooled = ggml_cont(c->ctx,
                ggml_permute(c->ctx, pooled, perm[0], perm[1], perm[2], perm[3]));
        out = pooled;

        /* Compile-time folding, so shape arithmetic keeps working. */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0) {
                int64_t best = cv[0];
                for (int j = 1; j < ncv; j++) {
                    if (is_min ? (cv[j] < best) : (cv[j] > best)) best = cv[j];
                }
                cval_put(c, n->outputs[0], &best, 1);
            }
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
                ggml_set_name(out, n->outputs[0]);
                tmap_put_nd(c, n->outputs[0], out, t_init->n_dims > 0 ? t_init->n_dims : 1);
                /* Queue the payload for the deferred fill.
                 *
                 * Naming the attribute tensor after this node used to be the
                 * whole mechanism, on the assumption that load_weights would
                 * then find it -- but load_weights iterates
                 * onnx->initializers[], which the loader builds solely from
                 * graph.initializer, and a Constant node's tensor is never
                 * put there.  The name was set and nothing ever read it, so
                 * the tensor kept whatever its freshly allocated buffer held.
                 * The name is still assigned: find_constant_tensor() and the
                 * diagnostics identify these tensors by it. */
                strncpy(t_init->name, n->outputs[0], ONNX_MAX_NAME - 1);
                t_init->name[ONNX_MAX_NAME - 1] = '\0';
                if (c->n_cinit_fills < ONNX_MAX_CONSTANTS) {
                    c->cinit_fill_ptrs[c->n_cinit_fills] = out;
                    c->cinit_fill_srcs[c->n_cinit_fills] = t_init;
                    c->n_cinit_fills++;
                } else {
                    fprintf(stderr, "ONNX WARNING: too many Constant nodes "
                            "(> %d), '%s' will read uninitialised memory\n",
                            ONNX_MAX_CONSTANTS, n->outputs[0]);
                }
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
            return 1; /* already registered output */
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
            out = ggml_cast_numeric(c->ctx, a, GGML_TYPE_I32);
        } else if (to == 1 && a->type == GGML_TYPE_I32) {
            out = ggml_cast_numeric(c->ctx, a, GGML_TYPE_F32);
        } else if (to == 10 && a->type == GGML_TYPE_F32) {
            out = ggml_cast_numeric(c->ctx, a, GGML_TYPE_F16);
        } else if (to == 1 && a->type == GGML_TYPE_F16) {
            out = ggml_cast_numeric(c->ctx, a, GGML_TYPE_F32);
        } else if (to == 16 && a->type == GGML_TYPE_F32) {
            out = ggml_cast_numeric(c->ctx, a, GGML_TYPE_BF16);
        } else if (to == 1 && a->type == GGML_TYPE_BF16) {
            out = ggml_cast_numeric(c->ctx, a, GGML_TYPE_F32);
        } else if ((to == 2 || to == 3 || to == 4 || to == 5 || to == 9) &&
                   a->type == GGML_TYPE_F32) {
            /* Narrow integer targets (uint8, int8, uint16, int16) and bool.
             * ggml has no tensor type for any of them, so the value stays F32
             * and only its RANGE is made to match: truncate toward zero, as
             * the spec says float-to-int does, then clamp to what the target
             * type can hold.  A later op reading this sees the same numbers a
             * real uint8 tensor would have held.
             *
             * MaskRCNN casts Greater(scores, threshold) to uint8 and gathers
             * with it -- values are already 0 or 1, so neither step changes
             * anything there; they matter for the general case, where letting
             * an out-of-range value through unchanged is the silent corruption
             * this branch exists to avoid.
             *
             * BOOL is not a range but a predicate, so it is the one target
             * that is not a clamp: anything non-zero becomes 1. */
            if (to == 9) {
                /* Nonzero → 1, zero → 0.  step() on its own is the wrong
                 * shape: it maps 0 to 1 and every negative to 0, whereas bool
                 * cares about magnitude, not sign.
                 *
                 * step(|x| - eps) with a tiny eps would read better but is
                 * not safe: a subnormal eps flushes to zero under a backend
                 * running in flush-to-zero mode, and step(0) is 1 -- turning
                 * false into true precisely where it matters.  Scaling up
                 * first keeps the comparison in normal floating point: any
                 * |x| at or above 1e-30 lands at or above 1e8 after the
                 * scale, leaving the 0.5 threshold far below it, while a true
                 * zero stays zero. */
                struct ggml_tensor *ax = ggml_abs(c->ctx, a);
                struct ggml_tensor *big = ggml_scale(c->ctx, ax, 1e38f);
                out = ggml_step(c->ctx, ggml_scale_bias(c->ctx, big, 1.0f, -0.5f));
            } else {
                float lo, hi;
                switch (to) {
                    case 2: lo =      0.0f; hi =    255.0f; break;  /* uint8  */
                    case 3: lo =   -128.0f; hi =    127.0f; break;  /* int8   */
                    case 4: lo =      0.0f; hi =  65535.0f; break;  /* uint16 */
                    default: lo = -32768.0f; hi =  32767.0f; break; /* int16  */
                }
                out = ggml_clamp(c->ctx, ggml_trunc(c->ctx, a), lo, hi);
            }
        } else {
            /* Everything left is either a cast that changes nothing, or one
             * this handler cannot perform.  The two must not share a branch:
             * passing a tensor through unchanged is right for the first and
             * silently wrong for the second, and Cast is precisely the op
             * whose whole purpose is to change the type, so a no-op there
             * hands the next op a tensor of a type it did not ask for.
             *
             * Refuse rather than pass through.  The caller reports the node
             * by name, which is what turns "the numbers are wrong somewhere"
             * into a location -- the same reason TopK and Einsum below say so
             * out loud instead of returning something plausible. */
            enum ggml_type want = GGML_TYPE_COUNT;
            switch (to) {
                case 1:  want = GGML_TYPE_F32;  break;
                case 6:
                case 7:  want = GGML_TYPE_I32;  break;  /* i64 downcast to i32 */
                case 10: want = GGML_TYPE_F16;  break;
                case 16: want = GGML_TYPE_BF16; break;
                default: break;
            }
            if (want != GGML_TYPE_COUNT && want == a->type) {
                out = a;  /* genuine no-op: already the requested type */
            } else {
                fprintf(stderr, "[onnx] Cast %s: to=%lld from %s is not "
                                "implemented\n",
                        n->outputs[0], (long long)to, ggml_type_name(a->type));
                return -1;
            }
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
        /* The dims are stashed below into a [ONNX_MAX_DIMS + 1] row as
         * [count, dims...], so a rank beyond that would write past the row and
         * into the next one -- silent corruption of the context struct that
         * only shows up much later, elsewhere. */
        if (nd > ONNX_MAX_DIMS) nd = ONNX_MAX_DIMS;
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
        return 1; /* already registered */
    }

    /* ── Pow ────────────────────────────────────────────────────── */
    else if (strcmp(op, "Pow") == 0) {
        if (!a || !b) return -1;

        /* exp(b*log(a)) is the general form, but log(a) is NaN for every
         * negative a -- and the most common Pow in a transformer is the
         * Pow(x - mean, 2) inside LayerNorm, whose input is centred and so
         * half negative by construction.  That produced a NaN in half the
         * elements of the very first LayerNorm and carried it to the model's
         * output.
         *
         * A small integer exponent needs no logarithm at all: repeated
         * multiplication is exact, defined for negative bases, and cheaper.
         * Anything else keeps the old path, which remains correct wherever a
         * is positive. */
        int64_t exp_i = 0;
        int is_int_exp = 0;
        if (ggml_nelements(b) == 1) {
            const onnx_initializer_t *bi = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (!bi) bi = find_constant_tensor(c->onnx, n->inputs[1]);
            if (bi) {
                const void *src = bi->raw_data ? (const void *)bi->raw_data
                                               : (const void *)bi->decoded_data;
                double bv = 0.0;
                int got = 0;
                if (src) {
                    if (bi->data_type == ONNX_DTYPE_FLOAT) {
                        float f; memcpy(&f, src, sizeof(f)); bv = f; got = 1;
                    } else if (bi->data_type == ONNX_DTYPE_DOUBLE) {
                        double d; memcpy(&d, src, sizeof(d)); bv = d; got = 1;
                    } else if (bi->data_type == ONNX_DTYPE_INT64) {
                        int64_t v; memcpy(&v, src, sizeof(v)); bv = (double)v; got = 1;
                    } else if (bi->data_type == ONNX_DTYPE_INT32) {
                        int32_t v; memcpy(&v, src, sizeof(v)); bv = (double)v; got = 1;
                    }
                }
                /* Only exponents small enough to be worth unrolling, and only
                 * exact integers: 2.5 is not 2. */
                if (got && bv == (double)(int64_t)bv && bv >= 1 && bv <= 8) {
                    exp_i = (int64_t)bv;
                    is_int_exp = 1;
                }
            }
        }

        if (is_int_exp) {
            /* x^1 still has to be a node of its own: handing back `a` itself
             * would register this op's output name onto the input tensor. */
            out = (exp_i == 1) ? ggml_dup(c->ctx, a) : a;
            for (int64_t k = 1; k < exp_i; k++)
                out = ggml_mul(c->ctx, out, a);
            if (onnx_trace_nodes())
                fprintf(stderr, "[Pow] %s: integer exponent %lld, using repeated mul\n",
                        n->outputs[0], (long long)exp_i);
        } else {
            struct ggml_tensor *log_a = ggml_log(c->ctx, a);
            struct ggml_tensor *prod;
            if (ggml_nelements(b) < ggml_nelements(a))
                prod = ggml_mul(c->ctx, log_a, b);
            else
                prod = ggml_mul(c->ctx, b, log_a);
            out = ggml_exp(c->ctx, prod);
        }
    }

    /* ── Erf (error function) ──────────────────────────────────── */
    else if (strcmp(op, "Erf") == 0) {
        if (!a) return -1;
        /* Abramowitz & Stegun 7.1.26:
         *
         *   erf(|x|) = 1 - (a1 t + a2 t^2 + ... + a5 t^5) * exp(-x^2),
         *   t = 1 / (1 + p|x|),
         *
         * odd-extended by the sign of x.  Accurate to 1.4e-7 over [-6,6].
         *
         * The previous form, tanh(sqrt(2/pi)*(u + 0.044715 u^3)), came from
         * the tanh formulation of GELU, where the tanh stands in for
         * erf(u/sqrt(2)) -- the sqrt(2) included.  Used as erf it was wrong by
         * 0.166; corrected by feeding it x*sqrt(2) it was still only good to
         * 3.6e-4, and that residue was what kept bert, roberta and cait about
         * 1e-2 away from ONNX Runtime after every other defect was fixed. */
        static const float p_as  = 0.3275911f;
        static const float a1_as = 0.254829592f, a2_as = -0.284496736f,
                           a3_as = 1.421413741f, a4_as = -1.453152027f,
                           a5_as = 1.061405429f;

        struct ggml_tensor *ax = ggml_abs(c->ctx, a);
        struct ggml_tensor *den = ggml_scale_bias(c->ctx, ax, p_as, 1.0f);
        /* t = 1/den.  The numerator is built from the input itself -- x*0 + 1 --
         * rather than from a scalar: ggml_div needs both operands the same
         * shape, and onnx_broadcast_prepare cannot supply it here because it
         * SWAPS its arguments so the larger one comes first.  For a scalar over
         * a tensor that turns 1/den into den/1, which is what wrecked every
         * model using Erf when this was first written that way. */
        struct ggml_tensor *ones = ggml_scale_bias(c->ctx, a, 0.0f, 1.0f);
        struct ggml_tensor *t = ggml_div(c->ctx, ones, den);

        /* Horner: (((a5 t + a4) t + a3) t + a2) t + a1, then one more t. */
        struct ggml_tensor *poly = ggml_scale_bias(c->ctx, t, a5_as, a4_as);
        poly = ggml_scale_bias(c->ctx, ggml_mul(c->ctx, poly, t), 1.0f, a3_as);
        poly = ggml_scale_bias(c->ctx, ggml_mul(c->ctx, poly, t), 1.0f, a2_as);
        poly = ggml_scale_bias(c->ctx, ggml_mul(c->ctx, poly, t), 1.0f, a1_as);
        poly = ggml_mul(c->ctx, poly, t);

        struct ggml_tensor *gauss = ggml_exp(c->ctx,
            ggml_neg(c->ctx, ggml_sqr(c->ctx, a)));
        /* 1 - poly*exp(-x^2), via -(poly*gauss) + 1 so no constant tensor is
         * needed for the subtraction. */
        struct ggml_tensor *mag = ggml_scale_bias(c->ctx,
            ggml_mul(c->ctx, poly, gauss), -1.0f, 1.0f);

        out = ggml_mul(c->ctx, ggml_sgn(c->ctx, a), mag);
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
        /* All three arms end up in mul/add below, so none of them may stay
         * integer.  Two calls because the helper takes a pair; the condition
         * is the one most likely to be integer, coming from Equal or Less. */
        onnx_binary_promote(c->ctx, &a, &b);
        onnx_binary_promote(c->ctx, &a, &cond_t);
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
         * step(0.5 - abs(a - b)): if a==b → step(0.5)=1, else step(neg)=0.
         *
         * The comparison is expressed as arithmetic, so an integer operand
         * has to be widened exactly as for Add or Sub: comparing indices is
         * the common case here, not the exotic one -- MaskRCNN compares an
         * FPN level against a constant level to pick a feature map. */
        onnx_binary_promote(c->ctx, &a, &b);
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

    /* ── Less / LessOrEqual / Greater / GreaterOrEqual ─────────── */
    /* Comparisons produce a 0/1 float mask, the same representation Equal and
     * Where already use.  ggml_step(x) is 1 for x > 0 and 0 otherwise, so a
     * strict comparison is a step of the difference; the "or equal" forms add
     * half a unit of slack so that an exact tie lands on the 1 side.  That
     * slack assumes the compared values are not closer together than 0.5,
     * which holds for the shape arithmetic and index masks these ops are used
     * for -- the same assumption Equal makes with its 0.5 window. */
    else if (strcmp(op, "Less") == 0 || strcmp(op, "LessOrEqual") == 0 ||
             strcmp(op, "Greater") == 0 || strcmp(op, "GreaterOrEqual") == 0) {
        if (!a || !b) return -1;
        int is_less  = (op[0] == 'L');
        int or_equal = (strstr(op, "OrEqual") != NULL);

        /* Same widening as Equal: the comparison is built out of sub/step. */
        onnx_binary_promote(c->ctx, &a, &b);
        struct ggml_tensor *ta = a, *tb = b;
        onnx_broadcast_prepare(c->ctx, &ta, &tb);
        /* Less: b - a > 0 ;  Greater: a - b > 0.
         *
         * Both are written with `ta` first and the Less case negated, rather
         * than as sub(tb, ta): ggml_sub broadcasts its SECOND operand into the
         * first and asserts ggml_can_repeat(b, a), so a scalar threshold in
         * the first position aborts the process outright.  Comparing a tensor
         * against a scalar bound is the common case -- MaskRCNN's Less(147
         * boxes, scalar) hit it -- and it went unseen only while an upstream
         * Gather defect was collapsing that tensor to a single element, where
         * the two operands happened to be the same size.  Same reasoning as
         * the Not case below, which says it outright. */
        struct ggml_tensor *diff = ggml_sub(c->ctx, ta, tb);
        if (is_less) diff = ggml_neg(c->ctx, diff);
        if (or_equal)
            diff = ggml_add(c->ctx, diff, make_scalar(c, 0.5f));
        out = ggml_step(c->ctx, diff);

        /* Compile-time values, so shape arithmetic keeps folding. */
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 == na2) {
                int64_t result[ONNX_MAX_DIMS];
                for (int j = 0; j < na2; j++) {
                    int r = is_less ? (or_equal ? cv_a[j] <= cv_b[j] : cv_a[j] < cv_b[j])
                                    : (or_equal ? cv_a[j] >= cv_b[j] : cv_a[j] > cv_b[j]);
                    result[j] = r ? 1 : 0;
                }
                cval_put(c, n->outputs[0], result, na2);
            }
        }
    }

    /* ── Not ───────────────────────────────────────────────────── */
    else if (strcmp(op, "Not") == 0) {
        if (!a) return -1;
        /* 1 - x, written as -(x - 1) so the full-size tensor stays the first
         * operand -- ggml broadcasts the second into the first, not the other
         * way round. */
        out = ggml_neg(c->ctx, ggml_sub(c->ctx, a, make_scalar(c, 1.0f)));
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0) {
                int64_t result[ONNX_MAX_DIMS];
                for (int j = 0; j < ncv; j++) result[j] = cv[j] ? 0 : 1;
                cval_put(c, n->outputs[0], result, ncv);
            }
        }
    }

    /* ── And / Or / Xor ────────────────────────────────────────── */
    /* On 0/1 masks: And is the product, Or is 1 - (1-a)(1-b), and Xor is
     * |a - b|. */
    else if (strcmp(op, "And") == 0 || strcmp(op, "Or") == 0 ||
             strcmp(op, "Xor") == 0) {
        if (!a || !b) return -1;
        /* These take BOOL operands, which arrive as F32 0/1 from the loader --
         * except when they come from an integer-producing op, so widen here
         * too rather than assume. */
        onnx_binary_promote(c->ctx, &a, &b);
        struct ggml_tensor *ta = a, *tb = b;
        onnx_broadcast_prepare(c->ctx, &ta, &tb);
        if (strcmp(op, "And") == 0) {
            out = ggml_mul(c->ctx, ta, tb);
        } else if (strcmp(op, "Or") == 0) {
            /* 1 - (1-a)(1-b), with the full-size tensor kept as the first
             * operand of every sub so ggml broadcasts the scalar into it. */
            struct ggml_tensor *one = make_scalar(c, 1.0f);
            struct ggml_tensor *na_ = ggml_neg(c->ctx, ggml_sub(c->ctx, ta, one));
            struct ggml_tensor *nb_ = ggml_neg(c->ctx, ggml_sub(c->ctx, tb, one));
            struct ggml_tensor *prod = ggml_mul(c->ctx, na_, nb_);
            out = ggml_neg(c->ctx, ggml_sub(c->ctx, prod, one));
        } else {
            out = ggml_abs(c->ctx, ggml_sub(c->ctx, ta, tb));
        }
        {
            int64_t cv_a[ONNX_MAX_DIMS], cv_b[ONNX_MAX_DIMS];
            int na2 = cval_get(c, n->inputs[0], cv_a, ONNX_MAX_DIMS);
            int nb2 = cval_get(c, n->inputs[1], cv_b, ONNX_MAX_DIMS);
            if (na2 > 0 && nb2 == na2) {
                int64_t result[ONNX_MAX_DIMS];
                for (int j = 0; j < na2; j++) {
                    int x = cv_a[j] != 0, y = cv_b[j] != 0;
                    result[j] = (strcmp(op, "And") == 0) ? (x && y)
                              : (strcmp(op, "Or") == 0)  ? (x || y)
                                                         : (x != y);
                }
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

        if (onnx_trace_nodes()) {
            /* Where the value actually lives decides whether it is read at all:
             * the branch above requires raw_data, while a TensorProto may carry
             * the value in int64_data/float_data instead, which the loader
             * decodes into decoded_data.  Printing both sizes says which of the
             * two this model uses, and so whether the value was read or the
             * initialiser 0.0f survived. */
            const onnx_initializer_t *vt = va ? va->tensor : NULL;
            fprintf(stderr, "[ConstantOfShape] %s: shape_src='%s' fill_val=%g has_value_attr=%d "
                    "dtype=%d raw=%zu decoded=%zu resolved=[",
                    n->outputs[0], n->inputs[0], (double)fill_val, va ? 1 : 0,
                    vt ? vt->data_type : -1,
                    vt ? vt->raw_size : (size_t)0,
                    vt ? vt->decoded_size : (size_t)0);
            for (int d = 0; d < ndims; d++)
                fprintf(stderr, "%lld%s", (long long)shape[d], d < ndims - 1 ? "," : "");
            fprintf(stderr, "]\n");
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

            /* Inherit emptiness across Shape.
             *
             * The generic transit rule one level up cannot carry the mark here:
             * it asks whether this node's own inputs are empty, and a shape
             * tensor never is -- it holds a length, and a length of zero is
             * still a number.  What decides the result is the tensor whose
             * shape was measured, one node back.
             *
             * MaskRCNN builds per-class labels as ConstantOfShape(Shape(boxes)),
             * one branch per class.  For the 79 classes where nothing passed the
             * score threshold the boxes are empty, so the labels must be empty
             * too -- otherwise each contributes a phantom entry and the labels
             * concat comes out 127 long against 48 for boxes and scores, which
             * then reads the class of every detection off the wrong row.
             *
             * Deliberately narrow: only a ConstantOfShape fed directly by a
             * Shape, and only when that Shape's own input is marked empty.
             * Shape itself stays out of the transit list, where it would push
             * the mark through 258 unrelated reshape chains. */
            for (int k = 0; k < c->onnx->n_nodes; k++) {
                const onnx_node_t *pn = &c->onnx->nodes[k];
                if (strcmp(pn->op_type, "Shape") != 0) continue;
                if (pn->n_outputs < 1 || strcmp(pn->outputs[0], n->inputs[0]) != 0) continue;
                if (pn->n_inputs < 1 || pn->inputs[0][0] == '\0') break;
                if (tmap_is_empty(c, pn->inputs[0])) {
                    for (int i = 0; i < n->n_outputs; i++)
                        if (n->outputs[i][0] != '\0')
                            tmap_mark_empty(c, n->outputs[i]);
                    if (onnx_trace_nodes())
                        fprintf(stderr, "[empty] %s op=ConstantOfShape: inherited via "
                                "Shape('%s')\n", n->outputs[0], pn->inputs[0]);
                }
                break;
            }
        }
        return 1; /* already registered */
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
                /* pads holds a begin and an end for every axis, so more than
                 * 8 means a rank above 4 -- which the mapping below cannot
                 * express anyway.  Truncating silently kept the first four
                 * axes and dropped the rest, padding the wrong dimensions by
                 * the right amounts; refuse instead. */
                if (n_pads > 8) {
                    fprintf(stderr, "[onnx] Pad %s: %d pad values (rank %d) "
                                    "exceed the rank-4 limit\n",
                            n->outputs[0], n_pads, n_pads / 2);
                    return -1;
                }
                /* memcpy, not a cast through int64_t*: raw_data points into
                 * the mmap and carries no alignment guarantee. */
                memcpy(pads_arr, pi->raw_data, (size_t)n_pads * sizeof(int64_t));
            }
            /* Fallback: try cval (for dynamic Slice→Transpose→Cast chains) */
            if (n_pads == 0) {
                n_pads = cval_get(c, n->inputs[1], pads_arr, 8);
            }
        }
        /* pads format: [begin_0, begin_1, ..., end_0, end_1, ...]
         * For 4D: [b0,b1,b2,b3, e0,e1,e2,e3].
         * ONNX dims [N,C,H,W] → ggml [W,H,C,N], so ggml_d = nd-1-onnx_d.
         *
         * begin and end are kept apart and passed to ggml_pad_ext, which takes
         * a left and a right amount per axis.  They used to be SUMMED and
         * handed to ggml_pad, which appends only: the output had the right
         * shape and the data sat at the wrong offset, shifted left by begin.
         * Every begin-side pad was silently a translation of the image, and
         * the three configurations [0,1,0,1], [0,0,0,2] and [0,2,0,0] all
         * produced the identical wrong answer, which is what makes a summing
         * bug so easy to miss: the total is preserved, only the position is
         * lost. */
        int nd = n_pads / 2;
        if (nd > 4) nd = 4;  /* unreachable: n_pads > 8 is refused above */

        int lp[4] = {0, 0, 0, 0}, rp[4] = {0, 0, 0, 0};
        for (int d = 0; d < nd; d++) {
            int ggml_d = nd - 1 - d;
            if (ggml_d < 0 || ggml_d > 3) continue;
            lp[ggml_d] = (int)pads_arr[d];
            rp[ggml_d] = (int)pads_arr[nd + d];
        }
        if (onnx_trace_nodes()) {
            fprintf(stderr, "[Pad] %s: n_pads=%d pads=[", n->outputs[0], n_pads);
            for (int d = 0; d < n_pads; d++)
                fprintf(stderr, "%lld%s", (long long)pads_arr[d], d < n_pads-1 ? "," : "");
            fprintf(stderr, "] -> ggml lp=[%d,%d,%d,%d] rp=[%d,%d,%d,%d]\n",
                    lp[0], lp[1], lp[2], lp[3], rp[0], rp[1], rp[2], rp[3]);
        }
        out = ggml_pad_ext(c->ctx, a, lp[0], rp[0], lp[1], rp[1],
                           lp[2], rp[2], lp[3], rp[3]);
    }

    /* ── Quantized ops (QLinear family) ──────────────────────────── */
    /* DequantizeLinear(x, x_scale, x_zero_point) → y = (x - zp) * scale
     * All inputs stored as F32 (int8 converted at load time). */
    else {
        return 0; /* not this group */
    }

    *out_p    = out;
    *out_nd_p = out_nd;
    return 1;
}
