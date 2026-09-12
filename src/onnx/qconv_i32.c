/* qconv_i32.c — QLinearConv with an exact integer accumulator.
 * Copyright (c) 2026 ggmlR authors. MIT License.
 *
 * The default path dequantises x and w to F32, convolves, then requantises.
 * That is arithmetically reasonable and still wrong at the edges: the error
 * of each of the 64..2048 products enters the sum, so an output whose exact
 * accumulator lands within a float ULP of a quantisation boundary can round
 * to the neighbouring code.
 *
 * Measured on MaskRCNN-12-int8, node 7 (a 1x1 conv, 64 -> 256):
 *   f32 path      802815 / 802816 agree with ONNX Runtime  (one element off)
 *   this path     802816 / 802816
 * The one element had accumulator 396, giving 135.500006 -- six millionths
 * above the boundary between codes 135 and 136.  It is one element in 800k,
 * and it propagated: 1 -> 20 -> 8704 -> 332087 by the RPN input, ending in a
 * different top-1000 and 25 of 51 boxes matching.
 *
 * The rule, straight from the operator spec:
 *   acc[i32] = sum_k (xq_k - xzp) * (wq_k - wzp)  +  bias[i32]
 *   y        = round_half_even(acc * (xs * ws[oc] / ys)) + yzp,  clamped
 * Two properties matter.  The sum is exact, so nothing rounds until the end.
 * And w_scale is PER OUTPUT CHANNEL in real models (256 values for node 7),
 * so the multiplier is not one constant.
 *
 * Scope: 1x1 and 3x3, stride/pad/dilation as given, group == 1.  Anything
 * else falls back to the f32 path -- the caller checks before selecting this.
 */

#include "qconv_i32.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Everything this kernel needs that is not a tensor argument.
 *
 * Held in the op's userdata, allocated once per graph build and owned by the
 * ONNX context -- NOT freed here.  The scales are copied by value rather than
 * kept as tensor pointers: a pointer to a tensor captured at build time can
 * dangle by the time the op runs, because segmented execution resets and
 * reallocates buffers in between.  That exact mistake is why nms_cpu has to
 * null-check a tensor it was handed at build time. */

void qconv_i32_cpu(struct ggml_tensor *dst,
                   const struct ggml_tensor *a,   /* dummy: shape only */
                   const struct ggml_tensor *b,   /* x, quantised, F32-stored */
                   const struct ggml_tensor *c,   /* w, quantised, F32-stored */
                   int ith, int nth, void *userdata) {
    (void)a;
    const qconv_i32_params_t *p = (const qconv_i32_params_t *)userdata;

    /* Every pointer is checked before it is followed: under segmented
     * execution a tensor that existed at build time may have no data now, and
     * reading it then is a bare segfault in a worker thread with no message,
     * because it never reaches GGML_ABORT. */
    if (!p || !b || !c || !dst || !b->data || !c->data || !dst->data) {
        if (ith == 0)
            fprintf(stderr, "[qconv_i32] missing tensor (p=%p x=%p w=%p dst=%p)"
                            " -- output left untouched\n",
                    (const void *)p, (const void *)b, (const void *)c,
                    (const void *)dst);
        return;
    }

    /* This kernel reads host memory.  A non-NULL ->data is not enough: on a
     * device backend it is an offset into VRAM, which passes a NULL check and
     * then faults on the first read. */
    if ((b->buffer  && !ggml_backend_buffer_is_host(b->buffer)) ||
        (c->buffer  && !ggml_backend_buffer_is_host(c->buffer)) ||
        (dst->buffer && !ggml_backend_buffer_is_host(dst->buffer))) {
        if (ith == 0)
            fprintf(stderr, "[qconv_i32] '%s': inputs are not on the host -- "
                            "this kernel is CPU-only\n", dst->name);
        return;
    }

    const float *xd = (const float *)b->data;
    const float *wd = (const float *)c->data;
    float       *od = (float *)dst->data;

    const int64_t W_in  = b->ne[0], H_in  = b->ne[1], C_in = b->ne[2];
    const int64_t W_out = dst->ne[0], H_out = dst->ne[1], C_out = dst->ne[2];
    const int64_t KW = p->kw, KH = p->kh;

    /* Rows are split across threads; each output element is independent. */
    const int64_t total = H_out * C_out;
    const int64_t per   = (total + nth - 1) / nth;
    const int64_t begin = per * ith;
    const int64_t end   = begin + per < total ? begin + per : total;

    for (int64_t idx = begin; idx < end; idx++) {
        const int64_t oc = idx / H_out;
        const int64_t oh = idx % H_out;
        /* One multiplier per output channel: w_scale is per-channel. */
        const float mult = p->x_scale * p->w_scale[p->n_w_scale > 1 ? oc : 0]
                         / p->y_scale;
        const int32_t wzp = p->w_zp[p->n_w_zp > 1 ? oc : 0];
        const int32_t bias = p->bias ? p->bias[oc] : 0;

        for (int64_t ow = 0; ow < W_out; ow++) {
            int32_t acc = bias;

            for (int64_t ic = 0; ic < C_in; ic++) {
                for (int64_t kh = 0; kh < KH; kh++) {
                    const int64_t ih = oh * p->stride_h - p->pad_h + kh * p->dil_h;
                    if (ih < 0 || ih >= H_in) continue;   /* zero padding */
                    for (int64_t kw = 0; kw < KW; kw++) {
                        const int64_t iw = ow * p->stride_w - p->pad_w + kw * p->dil_w;
                        if (iw < 0 || iw >= W_in) continue;

                        /* x is stored as F32 but holds integers; the cast is
                         * exact for the 8-bit range. */
                        const int32_t xq =
                            (int32_t)xd[iw + W_in * ih + W_in * H_in * ic];
                        const int32_t wq =
                            (int32_t)wd[kw + KW * kh + KW * KH * ic
                                        + KW * KH * C_in * oc];
                        acc += (xq - p->x_zp) * (wq - wzp);
                    }
                }
            }

            /* rintf is round-half-to-even, which is what the spec asks for
             * ("it rounds to the nearest even").  roundf would send ties away
             * from zero and reintroduce the very off-by-one this exists to
             * remove. */
            float v = rintf((float)acc * mult) + (float)p->y_zp;
            if (v < p->out_lo) v = p->out_lo;
            if (v > p->out_hi) v = p->out_hi;
            od[ow + W_out * oh + W_out * H_out * oc] = v;
        }
    }
}
