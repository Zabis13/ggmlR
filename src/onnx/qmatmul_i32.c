/* qmatmul_i32.c — QLinearMatMul with an exact integer accumulator.
 * Copyright (c) 2026 ggmlR authors. MIT License.
 *
 * Same defect, same fix, one operator over: the default path dequantises A and
 * B to F32, multiplies, then requantises.  Every one of the K products carries
 * its own float error into the sum, so an output whose exact accumulator sits
 * within a float ULP of a quantisation boundary rounds to the neighbouring
 * code.  qconv_i32.c says the rest; this is the matmul shaped version of it.
 *
 * Measured on MaskRCNN-12-int8, the classifier head (2807, 1000x1024 @
 * 1024x81):
 *   f32 path      logits differ from ONNX Runtime in 25963 of 81000 elements,
 *                 every difference exactly one quantisation step (0.1422)
 *   effect        softmax score of box 45 lands at 0.0461 instead of 0.0516,
 *                 fails the Greater threshold, and NonZero returns 90 indices
 *                 where ONNX Runtime returns 91 -- one detection short, all
 *                 the way to the final 200-vs-204 output length
 *
 * The rule, straight from the operator spec:
 *   acc[i32] = sum_k (aq_k - a_zp) * (bq_k - b_zp)
 *   y        = round_half_even(acc * (a_scale * b_scale[col] / y_scale)) + y_zp
 * The sum is exact, so nothing rounds until the end, and b_scale may be per
 * output column rather than one constant.
 *
 * Scope: 2-D A (M x K) times 2-D B (K x N), which is what a quantised fully
 * connected head is.  Batched or broadcast matmul falls back to the f32 path
 * -- the caller checks before selecting this.
 */

#include "qmatmul_i32.h"
#ifdef GGML_USE_VULKAN
#include "../ggml-vulkan.h"    /* ggml_vk_qmatmul_i32_run: the GPU fast path */
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* On unless GGMLR_ONNX_QMATMUL_GPU=0 -- see the note in qmatmul_i32.h.
 * Cached: this runs per node per inference and getenv walks the environment. */
int qmatmul_i32_gpu_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("GGMLR_ONNX_GPU_QMATMUL");
        cached = (e && e[0] == '0' && e[1] == '\0') ? 0 : 1;
    }
    return cached;
}

static void qmatmul_i32_cpu_impl(struct ggml_tensor *dst,
                     const struct ggml_tensor *a,   /* dummy: shape only */
                     const struct ggml_tensor *b,   /* A, quantised, F32-stored */
                     const struct ggml_tensor *c,   /* B, quantised, F32-stored */
                     int ith, int nth, void *userdata) {
    (void)a;
    const qmatmul_i32_params_t *p = (const qmatmul_i32_params_t *)userdata;

    /* Every pointer is checked before it is followed: under segmented
     * execution a tensor that existed at build time may have no data now, and
     * reading it then is a bare segfault in a worker thread with no message,
     * because it never reaches GGML_ABORT. */
    if (!p || !b || !c || !dst || !b->data || !c->data || !dst->data) {
        if (ith == 0)
            fprintf(stderr, "[qmatmul_i32] missing tensor (p=%p a=%p b=%p dst=%p)"
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
            fprintf(stderr, "[qmatmul_i32] '%s': inputs are not on the host -- "
                            "this kernel is CPU-only\n", dst->name);
        return;
    }

    const float *ad = (const float *)b->data;
    const float *bd = (const float *)c->data;
    float       *od = (float *)dst->data;

    /* ggml ne[] is column-major: ne[0] is the fastest axis.  A is stored with
     * K fastest (one row of A contiguous), B with K fastest as well because
     * the builder hands this kernel B already transposed to [K, N] -- the
     * same layout ggml_mul_mat wants, and the one that makes both operands
     * walk their K axis contiguously here. */
    const int64_t K = b->ne[0];
    const int64_t M = b->ne[1];
    const int64_t N = dst->ne[0];

    /* Same stride assumption as qconv_i32.c, checked the same way: this kernel
     * walks arow[k] and bcol[k] as packed rows, and a view that says otherwise
     * reads neighbouring data silently. */
    {
        const size_t es = ggml_type_size(b->type);
        const size_t ec = ggml_type_size(c->type);
        if ((b->nb[0] != es || b->nb[1] != es * (size_t)b->ne[0] ||
             c->nb[0] != ec || c->nb[1] != ec * (size_t)c->ne[0]) && ith == 0)
            fprintf(stderr, "[qmatmul_i32] '%s': STRIDE MISMATCH -- results are "
                            "wrong. A nb=[%zu,%zu] expected [%zu,%zu]; "
                            "B nb=[%zu,%zu] expected [%zu,%zu]\n",
                    dst->name, b->nb[0], b->nb[1], es, es * (size_t)b->ne[0],
                    c->nb[0], c->nb[1], ec, ec * (size_t)c->ne[0]);
    }

    if (c->ne[0] != K) {
        if (ith == 0)
            fprintf(stderr, "[qmatmul_i32] '%s': K mismatch (A ne0=%lld, B ne0=%lld)"
                            " -- output left untouched\n",
                    dst->name, (long long)K, (long long)c->ne[0]);
        return;
    }

#ifdef GGML_USE_VULKAN
    /* Offer the whole matmul to the shader before splitting it across threads.
     *
     * Single-threaded only. The dispatch covers the ENTIRE output, so with
     * several workers each would submit its own identical dispatch and they
     * would race writing the same dst. The op is created with n_tasks = 1
     * (onnx_ops_quant.c:659).  That reasoning was wrong: n_tasks caps the work
     * items, nth is the size of the backend thread pool -- whatever n_threads
     * the model was loaded with, 12 by default -- so `nth == 1` was false on
     * every real inference and the GPU path never ran.  The warning below sits
     * inside this same block, so it never printed either: the shader was dead
     * code that cost nothing and did nothing.  Thread 0 now makes the single
     * dispatch and the rest return; see the note in qconv_i32.c.
     *
     * Declining is normal (no Vulkan backend, or a grid above the driver's
     * workgroup limit) and simply leaves the CPU loop below to do the work, so
     * this is a pure fast path: it cannot make a working model wrong by being
     * absent, only by being incorrect -- which the test suite is there to
     * catch, since a drifting requantisation changes detections rather than
     * merely perturbing numbers. */
    /* Set when this thread took the GPU branch and the dispatch turned it down:
     * thread 0 then owns all M rows, not a slice of them. */
    int gpu_attempted = 0;

    if (p->gpu_backend && qmatmul_i32_gpu_enabled()) {
        /* Everyone but thread 0 is done -- thread 0's dispatch covers the
         * whole output, and a second one would race it. */
        if (ith != 0) return;
        gpu_attempted = 1;
        int b_zp_any = 0;
        for (int i = 0; i < p->n_b_zp; i++)
            if (p->b_zp[i] != 0) { b_zp_any = 1; break; }

        if (ggml_vk_qmatmul_i32_run(
                p->gpu_backend, ad, bd, p->b_scale, p->b_zp, od,
                (unsigned)M, (unsigned)N, (unsigned)K,
                p->a_scale, p->y_scale, p->a_zp, p->y_zp,
                (unsigned)p->n_b_scale, (unsigned)p->n_b_zp,
                (unsigned)b_zp_any, p->out_lo, p->out_hi)) {
            return;
        }
        {
            static int warned = 0;
            if (!warned) {
                warned = 1;
                fprintf(stderr,
                    "[qmatmul_i32] '%s': the Vulkan dispatch declined (output "
                    "grid above the driver's workgroup limit) -- using the CPU "
                    "kernel.\n              Results are unaffected; only this "
                    "op runs on the host.  Further occurrences are not "
                    "reported.\n", dst->name);
            }
        }
    }
#endif

    /* Rows are split across threads; each output element is independent. */
    /* Unless the GPU branch was taken and declined: the other threads returned
     * at that branch, so thread 0 is alone here and takes every row. */
    const int64_t per   = gpu_attempted ? M : (M + nth - 1) / nth;
    const int64_t begin = gpu_attempted ? 0 : per * ith;
    const int64_t end   = begin + per < M ? begin + per : M;

    for (int64_t m = begin; m < end; m++) {
        const float *arow = ad + m * K;

        /* sum(aq) over this row: needed only when the weight zero point is
         * non-zero, and constant across the row's outputs either way. */
        int32_t sum_a = 0;
        {
            int b_any = 0;
            for (int i = 0; i < p->n_b_zp; i++) if (p->b_zp[i] != 0) { b_any = 1; break; }
            if (b_any)
                for (int64_t k = 0; k < K; k++) sum_a += (int32_t)arow[k];
        }

        for (int64_t n = 0; n < N; n++) {
            /* One multiplier per output column: b_scale may be per column.
             * Float, for the same reason qconv_i32.c keeps it in float: ONNX
             * Runtime requantises in float, and computing this in double
             * measurably moves AWAY from the reference. */
            const float mult = p->a_scale * p->b_scale[p->n_b_scale > 1 ? n : 0]
                             / p->y_scale;
            const int32_t bzp = p->b_zp[p->n_b_zp > 1 ? n : 0];
            const float *bcol = bd + n * K;

            /* Same VPMADDUBSW saturation as qconv_i32.c -- see the long note
             * there.  It matters more here, not less: K is 12544 for
             * MaskRCNN's fully connected head against 2304 for a 3x3
             * convolution, so there are 6272 adjacent pairs per output and a
             * pair reaches 189*127*2 = 48006 against the int16 limit of
             * 32767.  Several saturations per output is normal, which is why
             * the disagreement here showed up as deltas of 2 to 5 rather than
             * the single code seen in the convolutions. */
            int32_t sum_pairs = 0;
            for (int64_t k = 0; k < K; k += 2) {
                const int32_t a0 = (int32_t)arow[k];
                const int32_t b0 = (int32_t)bcol[k];
                const int32_t a1 = (k + 1 < K) ? (int32_t)arow[k + 1] : 0;
                const int32_t b1 = (k + 1 < K) ? (int32_t)bcol[k + 1] : 0;

                int32_t pair = a0 * b0 + a1 * b1;
                if (pair >  32767) pair =  32767;
                if (pair < -32768) pair = -32768;
                sum_pairs += pair;
            }

            /* Zero points come out of the packed sums, as MLAS folds RowSum
             * and ColumnSum in at the end rather than subtracting per term. */
            int32_t sum_b = 0;
            if (p->a_zp != 0)
                for (int64_t k = 0; k < K; k++) sum_b += (int32_t)bcol[k];

            int32_t acc = sum_pairs - bzp * sum_a - p->a_zp * sum_b
                        + p->a_zp * bzp * (int32_t)K;

            /* rintf is round-half-to-even, which is what the spec asks for
             * ("it rounds to the nearest even").  roundf would send ties away
             * from zero and reintroduce the very off-by-one this exists to
             * remove.  Float, not double: see the note on `mult`. */
            float v = rintf((float)acc * mult) + (float)p->y_zp;
            if (v < p->out_lo) v = p->out_lo;
            if (v > p->out_hi) v = p->out_hi;
            od[n + N * m] = v;
        }
    }
}

/* Timing wrapper, the qconv_i32.c one shaped for this operator
 * (GGMLR_QCONV_PROFILE=1 drives both, since the question they answer -- where
 * the host-side time in a quantised model goes -- spans the two).
 *
 * ggml_map_custom runs on the host, so these nodes are invisible to the Vulkan
 * perf logger, which on MaskRCNN accounts for 136 ms of a ~1066 ms run.
 *
 * A wrapper rather than timers in the body because that body returns early on
 * a missing tensor, and an inline stop would eventually be missed on one of
 * those paths.
 */
void qmatmul_i32_cpu(struct ggml_tensor *dst,
                     const struct ggml_tensor *a,
                     const struct ggml_tensor *b,
                     const struct ggml_tensor *c,
                     int ith, int nth, void *userdata) {
    static int  prof = -1;
    static long calls = 0;

    if (prof < 0) prof = (getenv("GGMLR_QCONV_PROFILE") != NULL);
    if (!prof) {
        qmatmul_i32_cpu_impl(dst, a, b, c, ith, nth, userdata);
        return;
    }

    const qmatmul_i32_params_t *p = (const qmatmul_i32_params_t *)userdata;

    const int64_t t0 = ggml_time_us();
    qmatmul_i32_cpu_impl(dst, a, b, c, ith, nth, userdata);
    const long us = (long)(ggml_time_us() - t0);

    if (ith == 0) {
        calls++;
        fprintf(stderr,
                "[qmatmul-prof] %3ld %-24s %8.2f ms  out(%lld,%lld,%lld,%lld)"
                "  gpu=%d\n",
                calls, dst && dst->name[0] ? dst->name : "(unnamed)",
                us / 1000.0,
                (long long)(dst ? dst->ne[0] : 0),
                (long long)(dst ? dst->ne[1] : 0),
                (long long)(dst ? dst->ne[2] : 0),
                (long long)(dst ? dst->ne[3] : 0),
                (p && p->gpu_backend && qmatmul_i32_gpu_enabled()) ? 1 : 0);
    }
}
