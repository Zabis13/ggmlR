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
#include "../ggml-impl.h"      /* ggml_get_op_params_* */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* AVX2 is compiled in only when configure was given --with-simd, which is what
 * puts -mavx2 in SIMD_CFLAGS; CRAN forbids that flag by default, so the scalar
 * loop below has to stay and stay correct. __AVX2__ is the compiler's own
 * answer to "may I emit these instructions", which is exactly the question. */
#if defined(__AVX2__)
#include <immintrin.h>
#define QMATMUL_I32_HAVE_AVX2 1
#endif

#if defined(QMATMUL_I32_HAVE_AVX2)
/* The pair-saturating dot product, on the instruction it was emulating.
 *
 * The scalar loop below this exists to reproduce ONNX Runtime's AVX2 kernel
 * bit for bit, saturation and all -- see the long note at the top of
 * qconv_i32.c. That makes this the unusual case where the vector form is the
 * DEFINITION and the scalar form the emulation of it: vpmaddubsw multiplies
 * uint8 by int8, adds adjacent pairs and saturates each pair to int16, which
 * is precisely the three steps the scalar code spells out.
 *
 * So the concern is not "is the vector version accurate enough" but "does it
 * pair the same elements". It does: the pairing is positional (k, k+1) in both,
 * and vpmaddubsw pairs adjacent bytes within each lane, which for a contiguous
 * load is the same adjacency.
 *
 * Returns the number of elements consumed, always even and a multiple of 32,
 * leaving any tail to the scalar loop. Writes the pair sums into *acc.
 *
 * The values live in floats but are quantised integers: A in [0,255] and B in
 * [-128,127] for the u8 x i8 form this instruction implements. Anything
 * outside that is not this kernel's case, so the caller checks first and this
 * is never reached for it -- a silent wrap here would be a wrong answer, not a
 * slow one.
 *
 * vpmaddubsw's int16 pair sums are widened to int32 before accumulating:
 * summing 6272 saturated pairs (K = 12544 on MaskRCNN's head) in int16 would
 * overflow the accumulator itself, which the scalar code never does because
 * its sum_pairs is int32. */
static int64_t qmatmul_i32_dot_avx2(const float *arow, const float *bcol,
                                    int64_t K, int32_t *acc) {
    const int64_t n32 = (K / 32) * 32;
    if (n32 == 0) return 0;

    __m256i sum = _mm256_setzero_si256();

    for (int64_t k = 0; k < n32; k += 32) {
        /* Convert 32 floats per operand to bytes. cvttps is truncation
         * toward zero, matching the scalar (int32_t) cast. */
        __m128i ab[4], bb[4];
        for (int j = 0; j < 4; j++) {
            const __m256i ai = _mm256_cvttps_epi32(_mm256_loadu_ps(arow + k + 8 * j));
            const __m256i bi = _mm256_cvttps_epi32(_mm256_loadu_ps(bcol + k + 8 * j));
            /* Pack 8x int32 -> 8x int16, keeping lane order by permuting the
             * two halves back after the in-lane pack. */
            const __m256i a16 = _mm256_permute4x64_epi64(
                _mm256_packs_epi32(ai, ai), 0xD8);
            const __m256i b16 = _mm256_permute4x64_epi64(
                _mm256_packs_epi32(bi, bi), 0xD8);
            ab[j] = _mm256_castsi256_si128(a16);
            bb[j] = _mm256_castsi256_si128(b16);
        }
        /* 16 int16 -> 16 bytes per register. A is unsigned, B is signed. */
        const __m256i a16lo = _mm256_set_m128i(ab[1], ab[0]);
        const __m256i a16hi = _mm256_set_m128i(ab[3], ab[2]);
        const __m256i b16lo = _mm256_set_m128i(bb[1], bb[0]);
        const __m256i b16hi = _mm256_set_m128i(bb[3], bb[2]);

        const __m256i au = _mm256_permute4x64_epi64(
            _mm256_packus_epi16(a16lo, a16hi), 0xD8);
        const __m256i bs = _mm256_permute4x64_epi64(
            _mm256_packs_epi16(b16lo, b16hi), 0xD8);

        /* The instruction this whole kernel is a mirror of. */
        const __m256i pairs = _mm256_maddubs_epi16(au, bs);

        /* Widen to int32 before accumulating: see the note above. */
        const __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(pairs));
        const __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(pairs, 1));
        sum = _mm256_add_epi32(sum, _mm256_add_epi32(lo, hi));
    }

    /* Horizontal sum of the 8 int32 lanes. Order does not matter: integer
     * addition is associative, so this cannot drift the way a float sum would. */
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum),
                              _mm256_extracti128_si256(sum, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    *acc += _mm_cvtsi128_si32(s);

    return n32;
}

/* Do the operands fit the u8 x i8 form vpmaddubsw implements?
 *
 * Checked on the data rather than assumed from the model's declared zero-point
 * dtype: QLinearMatMul allows int8 activations too, and this kernel is handed
 * floats that merely hold quantised values. One pass over the column, once per
 * column, against the M passes it saves. */
static int qmatmul_i32_range_ok(const float *v, int64_t n, float lo, float hi) {
    for (int64_t i = 0; i < n; i++)
        if (v[i] < lo || v[i] > hi) return 0;
    return 1;
}
#endif /* QMATMUL_I32_HAVE_AVX2 */

/* What the kernel reads from the node, gathered once: scalars from op_params
 * in the order ggml_qmatmul_i32() wrote them, tables from src[2] / src[3].
 * It used to be a qmatmul_i32_params_t passed as userdata -- 33 KB per node,
 * malloc'd at build, owned by the model and freed by count -- holding copies
 * of values the graph already carries. */
typedef struct {
    float        a_scale, y_scale;
    int32_t      a_zp, y_zp;
    float        out_lo, out_hi;
    int          b_zp_any;
    int64_t      n_b_scale, n_b_zp;
    const float *b_scale;
    const void  *b_zp;       /* NULL: every weight zero point is zero */
    int          b_zp_f32;   /* F32 (widened INT8 initializer) or I32 */
} qmatmul_i32_args_t;

/* A zero point is read at the type it actually carries: casting the pointer
 * instead would reinterpret a float's bit pattern as an integer -- not a
 * wrong number but a wild one. */
static inline int32_t qmatmul_i32_bzp(const qmatmul_i32_args_t *p, int64_t i) {
    if (!p->b_zp) return 0;
    return p->b_zp_f32 ? (int32_t)((const float   *)p->b_zp)[i]
                       :          ((const int32_t *)p->b_zp)[i];
}

static void qmatmul_i32_compute_impl(struct ggml_tensor *dst, int ith, int nth) {
    const struct ggml_tensor *b  = dst ? dst->src[0] : NULL;  /* A,  [K, M] */
    const struct ggml_tensor *c  = dst ? dst->src[1] : NULL;  /* Bt, [K, N] */
    const struct ggml_tensor *ts = dst ? dst->src[2] : NULL;  /* b_scale */
    const struct ggml_tensor *tz = dst ? dst->src[3] : NULL;  /* b_zp, may be NULL */

    /* Every pointer is checked before it is followed: under segmented
     * execution a tensor that existed at build time may have no data now, and
     * reading it then is a bare segfault in a worker thread with no message,
     * because it never reaches GGML_ABORT. */
    if (!b || !c || !ts || !dst || !b->data || !c->data || !ts->data ||
        !dst->data || (tz && !tz->data)) {
        if (ith == 0)
            fprintf(stderr, "[qmatmul_i32] missing tensor (a=%p b=%p bs=%p dst=%p)"
                            " -- output left untouched\n",
                    (const void *)b, (const void *)c, (const void *)ts,
                    (const void *)dst);
        return;
    }

    qmatmul_i32_args_t args;
    args.a_scale   = ggml_get_op_params_f32(dst, 0);
    args.y_scale   = ggml_get_op_params_f32(dst, 1);
    args.a_zp      = ggml_get_op_params_i32(dst, 2);
    args.y_zp      = ggml_get_op_params_i32(dst, 3);
    args.out_lo    = ggml_get_op_params_f32(dst, 4);
    args.out_hi    = ggml_get_op_params_f32(dst, 5);
    args.b_zp_any  = ggml_get_op_params_i32(dst, 6);
    args.n_b_scale = ts->ne[0];
    args.b_scale   = (const float *)ts->data;
    args.n_b_zp    = tz ? tz->ne[0] : 1;
    args.b_zp      = tz ? tz->data : NULL;
    args.b_zp_f32  = tz && tz->type == GGML_TYPE_F32;
    const qmatmul_i32_args_t *p = &args;

    /* The tables are read on the host too, like the operands. */
    if ((ts->buffer && !ggml_backend_buffer_is_host(ts->buffer)) ||
        (tz && tz->buffer && !ggml_backend_buffer_is_host(tz->buffer))) {
        if (ith == 0)
            fprintf(stderr, "[qmatmul_i32] '%s': tables are not on the host -- "
                            "this kernel is CPU-only\n", dst->name);
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

    /* Rows are split across threads; each output element is independent.
     * The GPU no longer comes through here: GGML_OP_QMATMUL_I32 runs on the
     * Vulkan backend as a node of its own, and this kernel only sees the
     * nodes the CPU backend was given. */
    const int64_t per   = (M + nth - 1) / nth;
    const int64_t begin = per * ith;
    const int64_t end   = begin + per < M ? begin + per : M;

    /* sum(bq) down each weight column, hoisted out of the row loop.
     *
     * This is a function of `n` alone -- it sums bcol = bd + n*K, which does
     * not depend on `m` -- but it used to be computed inside the `n` loop and
     * therefore recomputed for every (m, n) pair.  That put a second full pass
     * over the K-long column next to the one that does the actual multiply, so
     * whenever a_zp != 0 the kernel did twice the memory traffic it needed:
     * measured at 31.3% of a MaskRCNN int8 CPU run, where the fully connected
     * head has K = 12544.
     *
     * Each thread builds its own copy.  The threads split rows, not columns, so
     * every one of them walks all N columns and would otherwise need locking or
     * a barrier to share this; N is small next to M*N*K, so the duplicated
     * O(N*K) setup is paid once per thread against M*N*K saved.
     *
     * Only when a_zp != 0: with a zero point of zero the term drops out of acc
     * entirely and computing it is pure waste -- which is also why the original
     * guarded the inner loop the same way.
     *
     * If the allocation fails the loop below falls back to computing the sum
     * inline, exactly as before, so a failed malloc costs speed and not
     * correctness. */
    const int b_zp_any_cached = p->b_zp_any;

#if defined(QMATMUL_I32_HAVE_AVX2)
    /* Which columns of B fit the int8 form, decided once for the whole call.
     *
     * Done here and not inside the loops because the check is O(K) per column:
     * inside the (m, n) nest it would be re-run M times per column and cost
     * more than the vector path saves. One byte per column, so even N in the
     * thousands is nothing; a failed allocation just leaves the scalar path,
     * like every other optional buffer here. */
    unsigned char *b_ok = NULL;
    if (N > 0) {
        b_ok = (unsigned char *)malloc((size_t)N);
        if (b_ok)
            for (int64_t n = 0; n < N; n++)
                b_ok[n] = (unsigned char)qmatmul_i32_range_ok(bd + n * K, K,
                                                              -128.0f, 127.0f);
    }
#endif

    int32_t * sum_b_col = NULL;
    if (p->a_zp != 0 && N > 0) {
        sum_b_col = (int32_t *) malloc((size_t) N * sizeof(int32_t));
        if (sum_b_col) {
            for (int64_t n = 0; n < N; n++) {
                const float *bcol = bd + n * K;
                int32_t s = 0;
                for (int64_t k = 0; k < K; k++) s += (int32_t)bcol[k];
                sum_b_col[n] = s;
            }
        }
    }

    for (int64_t m = begin; m < end; m++) {
        const float *arow = ad + m * K;

        /* sum(aq) over this row: needed only when the weight zero point is
         * non-zero, and constant across the row's outputs either way.
         * The b_zp scan itself is loop-invariant and now sits above the row
         * loop -- it used to be re-run for every row. */
        int32_t sum_a = 0;
        if (b_zp_any_cached)
            for (int64_t k = 0; k < K; k++) sum_a += (int32_t)arow[k];

#if defined(QMATMUL_I32_HAVE_AVX2)
        /* Once per row, not once per (row, column): the A row is the same for
         * every output column. */
        const int a_range_ok = qmatmul_i32_range_ok(arow, K, 0.0f, 255.0f);
#endif

        for (int64_t n = 0; n < N; n++) {
            /* One multiplier per output column: b_scale may be per column.
             * Float, for the same reason qconv_i32.c keeps it in float: ONNX
             * Runtime requantises in float, and computing this in double
             * measurably moves AWAY from the reference. */
            const float mult = p->a_scale * p->b_scale[p->n_b_scale > 1 ? n : 0]
                             / p->y_scale;
            const int32_t bzp = qmatmul_i32_bzp(p, p->n_b_zp > 1 ? n : 0);
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
            int64_t k0 = 0;
#if defined(QMATMUL_I32_HAVE_AVX2)
            /* Vector prologue, scalar tail. Only when both operands are in the
             * u8 x i8 range the instruction implements -- a_range_ok is hoisted
             * out of the n loop, b_range_ok is per column. */
            if (a_range_ok && b_ok && b_ok[n])
                k0 = qmatmul_i32_dot_avx2(arow, bcol, K, &sum_pairs);
#endif
            for (int64_t k = k0; k < K; k += 2) {
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
             * and ColumnSum in at the end rather than subtracting per term.
             * Precomputed per column above; the inline branch is the fallback
             * for a failed allocation. */
            int32_t sum_b = 0;
            if (p->a_zp != 0) {
                if (sum_b_col) {
                    sum_b = sum_b_col[n];
                } else {
                    for (int64_t k = 0; k < K; k++) sum_b += (int32_t)bcol[k];
                }
            }

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

    free(sum_b_col);
#if defined(QMATMUL_I32_HAVE_AVX2)
    free(b_ok);
#endif
}

/* GGML_OP_QMATMUL_I32 on the host, with the GGMLR_QCONV_PROFILE timing line
 * (one variable drives both int8 kernels, since the question they answer --
 * where the host-side time in a quantised model goes -- spans the two).
 *
 * The line prints only for nodes the CPU backend runs: on a Vulkan model the
 * op is a GPU node and its time is in the Vulkan perf logger instead.
 *
 * A wrapper rather than timers in the body because that body returns early on
 * a missing tensor, and an inline stop would eventually be missed on one of
 * those paths. */
void qmatmul_i32_compute(struct ggml_tensor *dst, int ith, int nth) {
    static int  prof = -1;
    static long calls = 0;

    if (prof < 0) prof = (getenv("GGMLR_QCONV_PROFILE") != NULL);
    if (!prof) {
        qmatmul_i32_compute_impl(dst, ith, nth);
        return;
    }

    const int64_t t0 = ggml_time_us();
    qmatmul_i32_compute_impl(dst, ith, nth);
    const long us = (long)(ggml_time_us() - t0);

    if (ith == 0) {
        calls++;
        /* a_zp and K are here because they decide whether the hoisted
         * per-column sum_b does any work at all: the term drops out of acc
         * when a_zp == 0. n_b_scale says how many distinct requantisation
         * multipliers the node has -- the quantity a backend disagreement in
         * the multiplier would depend on. */
        const struct ggml_tensor *a  = dst ? dst->src[0] : NULL;
        const struct ggml_tensor *bs = dst ? dst->src[2] : NULL;
        fprintf(stderr,
                "[qmatmul-prof] %3ld %-24s %8.2f ms  out(%lld,%lld)"
                "  K=%lld a_zp=%d b_zp_any=%d n_b_scale=%lld\n",
                calls, dst && dst->name[0] ? dst->name : "(unnamed)",
                us / 1000.0,
                (long long)(dst ? dst->ne[0] : 0),
                (long long)(dst ? dst->ne[1] : 0),
                (long long)(a ? a->ne[0] : 0),
                dst ? ggml_get_op_params_i32(dst, 2) : 0,
                dst ? ggml_get_op_params_i32(dst, 6) : 0,
                (long long)(bs ? bs->ne[0] : 0));
    }
}
