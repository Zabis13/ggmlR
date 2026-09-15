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
 *
 * Why the accumulator is deliberately NOT exact.
 *
 * An exact int32 sum disagrees with ONNX Runtime, and the reason is in ONNX
 * Runtime, not here.  Its AVX2 symmetric convolution kernel
 * (ConvSymKernelAvx2.asm) multiplies with VPMADDUBSW:
 *
 *     vpmaddubsw ymm3,ymm2,ymm0    ; uint8 * int8, add adjacent pair,
 *                                  ; SATURATE the pair to int16
 *
 * A pair of products reaches 255*127*2 = 64770 against an int16 limit of
 * 32767, so the pair sum genuinely clips.  Microsoft ship a debug hook for
 * exactly this (CheckSaturationForVPMADDUBSW, guarded by
 * ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER) -- it is a known lossy step, traded
 * for the speed of the instruction.
 *
 * Measured on MaskRCNN node 455, four outputs that disagreed:
 *   oc=191 oh=1  ow=25   exact -37737 -> 119   saturated -35057 -> 120 (ORT)
 *   oc=191 oh=1  ow=26   exact -39636 -> 119   saturated -36116 -> 120 (ORT)
 *   oc=191 oh=1  ow=0    exact -30074 -> 121   saturated -26956 -> 122 (ORT)
 *   oc=191 oh=27 ow=26   exact  23770 -> 133   saturated  24380 -> 134 (ORT)
 * One pair in 1152 saturates at each, which is enough to move the output a
 * whole code.  Emulating the clip reproduces ONNX Runtime on all four.
 *
 * Consequences worth knowing before changing any of this:
 *   - this kernel is now LESS accurate than a plain integer sum, on purpose.
 *   - it tracks the AVX2 path specifically.  On AVX-VNNI ONNX Runtime uses
 *     VPDPBUSDS, which accumulates into int32 and effectively never clips, so
 *     on such a machine this emulation can itself disagree with ONNX Runtime.
 *   - the pairing is positional: pair different channels together and a
 *     different pair saturates.  The loop nest below is ordered to match
 *     MlasConvSymPackW (kernel position outer, input channels inner, packed
 *     four at a time) for that reason, not for speed.
 *
 * This is a mirror, not a preference: the clip is here only because ONNX
 * Runtime has it.  When ONNX Runtime stops saturating -- AVX2 replaced by a
 * VNNI-only path, or the kernel reworked -- delete the clip and let the sum
 * be exact again.  The test is the reference check, not the calendar: if
 * inst/scripts/ref_check_vs_onnxruntime.sh starts DISAGREEING on the int8
 * models with the clip in place, that is the signal the other side moved.
 *
 * Ruled out by measurement before this was found, so nobody repeats them:
 * the input is bit-identical to the reference; the requantisation grouping of
 * (x_scale, w_scale, y_scale) makes no difference; and the row/column-sum
 * decomposition is exact in integer arithmetic -- computed on the real taps
 * it returns the same -37737 as the direct sum, with the intermediate sums
 * peaking at 7e4 against an int32 limit of 2.1e9.
 */

#include "qconv_i32.h"
#include "../ggml-impl.h"      /* ggml_get_op_params_* */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* GGML_OP_QCONV_I32 on the host.
 *
 * Reads everything from the tensor it is handed: scalars out of op_params, the
 * per-output-channel tables out of src[2..4]. It used to take a
 * qconv_i32_params_t by userdata, which carried the tables as fixed 4096-entry
 * arrays copied per node -- 48 KB each, 3 MB across a MaskRCNN graph, and a
 * second copy of values the graph already held.
 *
 * src[4] (bias) is NULL when the convolution has none.
 */
void qconv_i32_compute(struct ggml_tensor *dst, int ith, int nth) {
    const struct ggml_tensor *b  = dst ? dst->src[0] : NULL;  /* x */
    const struct ggml_tensor *c  = dst ? dst->src[1] : NULL;  /* w */
    const struct ggml_tensor *ts = dst ? dst->src[2] : NULL;  /* w_scale */
    const struct ggml_tensor *tz = dst ? dst->src[3] : NULL;  /* w_zp    */
    const struct ggml_tensor *tb = dst ? dst->src[4] : NULL;  /* bias, may be NULL */
    const struct ggml_tensor *tm = dst ? dst->src[5] : NULL;  /* mult, per channel */

    /* Every pointer is checked before it is followed: under segmented
     * execution a tensor that existed at build time may have no data now, and
     * reading it then is a bare segfault in a worker thread with no message,
     * because it never reaches GGML_ABORT. */
    if (!b || !c || !ts || !dst || !tm ||
        !b->data || !c->data || !ts->data || !dst->data || !tm->data ||
        (tz && !tz->data) || (tb && !tb->data)) {
        if (ith == 0)
            fprintf(stderr, "[qconv_i32] missing tensor (x=%p w=%p ws=%p wz=%p"
                            " dst=%p) -- output left untouched\n",
                    (const void *)b, (const void *)c, (const void *)ts,
                    (const void *)tz, (const void *)dst);
        return;
    }

    /* op_params, in the order ggml_qconv_i32() wrote them. */
    const int32_t stride_w = ggml_get_op_params_i32(dst, 0);
    const int32_t stride_h = ggml_get_op_params_i32(dst, 1);
    const int32_t pad_w    = ggml_get_op_params_i32(dst, 2);
    const int32_t pad_h    = ggml_get_op_params_i32(dst, 3);
    const int32_t dil_w    = ggml_get_op_params_i32(dst, 4);
    const int32_t dil_h    = ggml_get_op_params_i32(dst, 5);
    /* Slots 6 and 7 hold x_scale and y_scale. The kernel no longer reads them:
     * both are folded into the precomputed mult in src[5]. They stay in
     * op_params because they describe the op. */
    const int32_t x_zp     = ggml_get_op_params_i32(dst, 8);
    const int32_t y_zp     = ggml_get_op_params_i32(dst, 9);
    const float   out_lo   = ggml_get_op_params_f32(dst, 10);
    const float   out_hi   = ggml_get_op_params_f32(dst, 11);

    /* Table lengths come off the tensors rather than a separate field, so the
     * two cannot disagree. Length 1 means the exporter quantised per tensor. */
    const float  *mult_t    = (const float *)tm->data;
    const int64_t n_w_scale = ts->ne[0];
    (void)n_w_scale;   /* w_scale reaches the kernel only through mult now */
    const int64_t n_w_zp    = tz ? tz->ne[0] : 0;

    /* A zero point reaches this kernel as F32 (INT8 in the file, widened on
     * load) and a bias as I32 (the spec fixes that type). Both hold whole
     * numbers, so each is read at whatever type it actually carries: casting
     * the pointer instead would reinterpret a float's bit pattern as an
     * integer -- not a wrong number but a wild one, and silently. */
    const int   wz_is_f32 = (tz && tz->type == GGML_TYPE_F32);
    const void *w_zp_raw  = tz ? tz->data : NULL;
    const int   bi_is_f32 = (tb && tb->type == GGML_TYPE_F32);
    const void *bias_raw  = tb ? tb->data : NULL;

    #define QCONV_WZP(i)  (wz_is_f32 ? (int32_t)((const float   *)w_zp_raw)[i] \
                                     :          ((const int32_t *)w_zp_raw)[i])
    #define QCONV_BIAS(i) (bi_is_f32 ? (int32_t)((const float   *)bias_raw)[i] \
                                     :          ((const int32_t *)bias_raw)[i])

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

    /* The index arithmetic below -- xd[iw + W_in*ih + W_in*H_in*ic] -- is only
     * the right address when each axis really is packed at the next one's
     * stride.  A view whose nb[] says otherwise reads neighbouring data and
     * returns plausible numbers rather than crashing, so say so out loud
     * instead of trusting it.  ggml_is_contiguous is not the test: it can be
     * false for reasons that do not affect this arithmetic, and the arithmetic
     * is what matters, so compare the strides this kernel actually assumes.
     *
     * This did NOT explain MaskRCNN node 455: measured, the check never fires
     * and wrapping both operands in ggml_cont changed no output bit.  The
     * guard stays because reading a view here would be silent and wrong, not
     * because it is the bug that was being chased. */
    {
        const size_t es = ggml_type_size(b->type);
        const int x_bad = b->nb[0] != es ||
                          b->nb[1] != es * (size_t)b->ne[0] ||
                          b->nb[2] != es * (size_t)b->ne[0] * (size_t)b->ne[1];
        const size_t ew = ggml_type_size(c->type);
        const int w_bad = c->nb[0] != ew ||
                          c->nb[1] != ew * (size_t)c->ne[0] ||
                          c->nb[2] != ew * (size_t)c->ne[0] * (size_t)c->ne[1];
        if ((x_bad || w_bad) && ith == 0)
            fprintf(stderr, "[qconv_i32] '%s': STRIDE MISMATCH -- results are "
                            "wrong. x nb=[%zu,%zu,%zu] expected [%zu,%zu,%zu]; "
                            "w nb=[%zu,%zu,%zu] expected [%zu,%zu,%zu]\n",
                    dst->name, b->nb[0], b->nb[1], b->nb[2],
                    es, es * (size_t)b->ne[0],
                    es * (size_t)b->ne[0] * (size_t)b->ne[1],
                    c->nb[0], c->nb[1], c->nb[2],
                    ew, ew * (size_t)c->ne[0],
                    ew * (size_t)c->ne[0] * (size_t)c->ne[1]);
    }
    const int64_t W_out = dst->ne[0], H_out = dst->ne[1], C_out = dst->ne[2];
    const int64_t KW = c->ne[0], KH = c->ne[1];

    /* Rows are split across threads; each output element is independent. */
    const int64_t total = H_out * C_out;
    const int64_t per   = (total + nth - 1) / nth;
    const int64_t begin = per * ith;
    const int64_t end   = begin + per < total ? begin + per : total;

    for (int64_t idx = begin; idx < end; idx++) {
        const int64_t oc = idx / H_out;
        const int64_t oh = idx % H_out;
        /* One multiplier per output channel: w_scale is per-channel.
         *
         * FLOAT, deliberately, and not double.  Computing the multiplier and
         * the product in double was tried, on the theory that an exact
         * accumulator deserves an exact finish: measured on MaskRCNN node 455
         * it took the disagreement with ONNX Runtime from 21 elements to
         * 44622, and the classifier logits from 25962 to 64591.  ONNX Runtime
         * itself requantises in float, so extra precision here does not move
         * toward the reference, it moves away from it.  Do not "fix" this. */
        /* Read, not recomputed: see ggml_qconv_i32() in ggml.h for why the
         * multiplier is precomputed. Recomputing it here would put this kernel
         * one ulp away from the shader again. */
        const float mult = mult_t[oc];
        const int32_t wzp  = w_zp_raw ? QCONV_WZP(n_w_zp > 1 ? oc : 0) : 0;
        const int32_t bias = bias_raw ? QCONV_BIAS(oc) : 0;

        /* sum(wq) over the whole filter for this output channel: it depends on
         * oc alone, so it is hoisted out of the ow loop.  ONNX Runtime folds
         * the same quantity into column_sums_ once at pre-pack time. */
        int32_t sum_w_oc = 0;
        if (x_zp != 0) {
            for (int64_t kh = 0; kh < KH; kh++)
                for (int64_t kw = 0; kw < KW; kw++)
                    for (int64_t ic = 0; ic < C_in; ic++)
                        sum_w_oc += (int32_t)wd[kw + KW * kh + KW * KH * ic
                                                + KW * KH * C_in * oc];
        }

        for (int64_t ow = 0; ow < W_out; ow++) {
            /* MLAS order: the kernel position is the OUTER loop and the input
             * channels the inner one, packed four at a time -- that is what
             * MlasConvSymPackW lays out and what vpbroadcastd then reads as one
             * dword.  The loop nest is shaped to match because the pairing
             * below is positional: pair up a different pair of channels and a
             * different pair saturates. */
            int32_t sum_pairs = 0;

            for (int64_t kh = 0; kh < KH; kh++) {
                const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
                const int h_pad = (ih < 0 || ih >= H_in);
                for (int64_t kw = 0; kw < KW; kw++) {
                    const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                    const int pad = h_pad || iw < 0 || iw >= W_in;

                    for (int64_t ic = 0; ic < C_in; ic += 2) {
                        /* Padding rows are vectors of x_zp (qlinearconv.cc
                         * fills padding_data with X_zero_point_value), so the
                         * product is xq*wq with xq = x_zp -- the -x_zp*sum(wq)
                         * term below then cancels it, which is why skipping
                         * the tap entirely gives the same answer. */
                        const int32_t xa = pad ? x_zp :
                            (int32_t)xd[iw + W_in * ih + W_in * H_in * ic];
                        const int32_t xb = (ic + 1 < C_in)
                            ? (pad ? x_zp :
                               (int32_t)xd[iw + W_in * ih + W_in * H_in * (ic + 1)])
                            : 0;
                        const int32_t wa =
                            (int32_t)wd[kw + KW * kh + KW * KH * ic
                                        + KW * KH * C_in * oc];
                        const int32_t wb = (ic + 1 < C_in)
                            ? (int32_t)wd[kw + KW * kh + KW * KH * (ic + 1)
                                          + KW * KH * C_in * oc]
                            : 0;

                        /* VPMADDUBSW: multiply uint8 by int8, add the adjacent
                         * pair, and SATURATE the pair to int16.  A pair can
                         * reach 255*127*2 = 64770 against a limit of 32767, so
                         * this genuinely clips -- Microsoft ship a debug hook
                         * (CheckSaturationForVPMADDUBSW) precisely to catch it.
                         * Reproducing it is the whole point: without the clip
                         * this kernel is MORE accurate than ONNX Runtime and
                         * therefore disagrees with it. */
                        int32_t pair = xa * wa + xb * wb;
                        if (pair >  32767) pair =  32767;
                        if (pair < -32768) pair = -32768;
                        sum_pairs += pair;
                    }
                }
            }

            /* Zero points come out of the packed sums, exactly as
             * qlinearconv.cc:215 builds column_sums_ and the kernel folds in
             * RowSum: acc = sum(xq*wq) - x_zp*sum(wq) - w_zp*sum(xq) + bias.
             * w_zp is zero on every symmetric path, and MlasConvSymPackWSize
             * refuses the path otherwise, so only the x_zp term is present. */
            int32_t acc = sum_pairs + bias - x_zp * sum_w_oc;
            if (wzp != 0) {
                /* Not reachable on the symmetric path; kept so a non-zero
                 * weight zero point is still arithmetically correct. */
                int32_t sum_x = 0;
                for (int64_t kh = 0; kh < KH; kh++) {
                    const int64_t ih = oh * stride_h - pad_h + kh * dil_h;
                    for (int64_t kw = 0; kw < KW; kw++) {
                        const int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                        const int pad = ih < 0 || ih >= H_in || iw < 0 || iw >= W_in;
                        for (int64_t ic = 0; ic < C_in; ic++)
                            sum_x += pad ? x_zp :
                                (int32_t)xd[iw + W_in * ih + W_in * H_in * ic];
                    }
                }
                acc -= wzp * sum_x;
                acc += wzp * x_zp * (int32_t)(KW * KH * C_in);
            }

            /* rintf is round-half-to-even, which is what the spec asks for
             * ("it rounds to the nearest even").  roundf would send ties away
             * from zero and reintroduce the very off-by-one this exists to
             * remove.  Float, not double: see the note on `mult`. */
            /* Same diagnostic the shader has: write the accumulator instead of
             * the requantised value, so the two backends can be compared at
             * the step before rounding. */
            static int dbg_acc = -1;
            if (dbg_acc < 0) {
                const char *e = getenv("GGMLR_QCONV_DEBUG_ACC");
                dbg_acc = (e && *e) ? atoi(e) : 0;
            }
            if (dbg_acc) {
                float dv = (float)acc;
                if (dbg_acc == 2) dv = mult;
                if (dbg_acc == 3) dv = (float)oc;
                if (dbg_acc == 4) dv = (float)y_zp;
                if (dbg_acc == 5) dv = out_hi;
                if (dbg_acc == 6) dv = rintf((float)acc * mult) + (float)y_zp;
                od[ow + W_out * oh + W_out * H_out * oc] = dv;
                continue;
            }

            float v = rintf((float)acc * mult) + (float)y_zp;
            if (v < out_lo) v = out_lo;
            if (v > out_hi) v = out_hi;
            od[ow + W_out * oh + W_out * H_out * oc] = v;

            /* One element, both backends, printed identically so the two logs
             * can be diffed: the integer accumulator and the requantisation
             * are separate suspects, and only their bit patterns tell which
             * one moved. GGMLR_QCONV_ELEM=oc,oh,ow selects the element. */
            {
                static int want = -1, w_oc, w_oh, w_ow;
                if (want < 0) {
                    const char *e = getenv("GGMLR_QCONV_ELEM");
                    want = (e && sscanf(e, "%d,%d,%d", &w_oc, &w_oh, &w_ow) == 3);
                }
                if (want && oc == w_oc && oh == w_oh && ow == w_ow) {
                    uint32_t mb, pb;
                    const float prod = (float)acc * mult;
                    memcpy(&mb, &mult, 4);
                    memcpy(&pb, &prod, 4);
                    fprintf(stderr,
                        "[qelem] %s oc=%lld oh=%lld ow=%lld acc=%d wzp=%d "
                        "bias=%d sum_w=%d mult=%.9g(0x%08x) prod=%.9g(0x%08x) "
                        "v=%g\n",
                        dst->name, (long long)oc, (long long)oh, (long long)ow,
                        acc, wzp, bias, sum_w_oc, mult, mb, prod, pb, v);
                }
            }
        }
    }

    #undef QCONV_WZP
    #undef QCONV_BIAS
}
