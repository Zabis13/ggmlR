/* roi_align.c — RoiAlign CPU implementation */

#include "roi_align.h"
#include "../ggml-backend.h"   /* buffer_is_host: this kernel reads host memory */
#ifdef GGML_USE_VULKAN
#include "../ggml-vulkan.h"    /* ggml_vk_roi_align_run: the GPU fast path */
#endif
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>            /* getenv: the per-op GPU gate */

/* ON by default; GGMLR_ONNX_GPU_ROI_ALIGN=0 forces the CPU kernel.
 *
 * The default is on because the agreement is enforced by a test rather than by
 * a hand-run measurement: tests/testthat/test-onnx-roialign-vulkan.R runs the
 * same model on both devices and demands identical doubles across full
 * coverage, fractional bounds, degenerate ROIs, out-of-map taps, the adaptive
 * grid, max mode and batch selection.  A shader that drifts fails the suite
 * instead of quietly changing MaskRCNN's detections, which is the failure this
 * op is dangerous for -- see the header note.
 *
 * The variable survives as an escape hatch: a driver whose bilinear blend
 * disagrees can be worked around without a rebuild.
 *
 * Read once and cached: this sits in a kernel that runs per node per inference,
 * and getenv walks the environment on every call. */
int roi_align_gpu_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("GGMLR_ONNX_GPU_ROI_ALIGN");
        cached = (e && e[0] == '0' && e[1] == '\0') ? 0 : 1;
    }
    return cached;
}

/* Bilinear interpolation on feature map X[W, H, C, N] (ggml layout).
 * x, y are in spatial coordinates (float). c_idx = channel, n_idx = batch.
 *
 * `take_max` selects what the four corners are reduced with, and the two are
 * NOT variations on a theme -- they answer different questions:
 *
 *   avg mode (take_max = 0): w1*v1 + w2*v2 + w3*v3 + w4*v4, the interpolated
 *   value, which is what "bilinear" normally means.
 *
 *   max mode (take_max = 1): max(w1*v1, w2*v2, w3*v3, w4*v4) -- the largest of
 *   the four WEIGHTED TERMS taken separately. No interpolation happens at all.
 *
 * That is what ONNX Runtime does (roialign.cc, RoiAlignForward's max branch),
 * and it is not obviously intended: the weights sum to one, so each term is a
 * fraction of its corner and the maximum of them is systematically SMALLER
 * than the interpolated value -- measured here 3.15 where interpolation gives
 * 5.15. The ratio is not even constant (1.63 to 1.71 across one 2x2 output),
 * because it depends on how the weights happened to fall.
 *
 * It is reproduced rather than corrected for the same reason the quantised
 * kernels reproduce ORT's VPMADDUBSW saturation: agreement with the reference
 * is the contract, and being more principled than it is what makes numbers
 * differ. Found by ref_check_ops case roialign_max; no model in the suite uses
 * max mode, which is why it went unnoticed. */
static float bilinear_sample(const float *data,
                             int W, int H,
                             float x, float y,
                             int c_idx, int n_idx,
                             int C, int take_max) {
    /* Clamp to valid range */
    if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W)
        return 0.0f;

    y = fmaxf(y, 0.0f);
    x = fmaxf(x, 0.0f);

    int y_low = (int)y;
    int x_low = (int)x;
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    if (y_low >= H - 1) { y_low = y_high = H - 1; y = (float)y_low; }
    if (x_low >= W - 1) { x_low = x_high = W - 1; x = (float)x_low; }

    float ly = y - (float)y_low;
    float lx = x - (float)x_low;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    /* ggml layout: data[x + W * (y + H * (c + C * n))] */
    const float *base = data + (size_t)C * H * W * n_idx + (size_t)H * W * c_idx;

    float v1 = base[x_low  + W * y_low];
    float v2 = base[x_high + W * y_low];
    float v3 = base[x_low  + W * y_high];
    float v4 = base[x_high + W * y_high];

    const float t1 = hy * hx * v1, t2 = hy * lx * v2;
    const float t3 = ly * hx * v3, t4 = ly * lx * v4;

    if (take_max) {
        float m = t1 > t2 ? t1 : t2;
        if (t3 > m) m = t3;
        if (t4 > m) m = t4;
        return m;
    }
    return t1 + t2 + t3 + t4;
}

void roi_align_cpu(struct ggml_tensor *dst, int ith, int nth, void *userdata) {
    (void)ith; (void)nth;

    const roi_align_params_t *p = (const roi_align_params_t *)userdata;

    /* Refuse a missing operand instead of dereferencing it.
     *
     * A kernel runs long after the graph was built, and an input that failed
     * to map leaves its tensor unregistered rather than announcing itself
     * here: X arrived NULL because a Cast upstream declined, and reading
     * X->ne[0] faulted at address 0x10 -- the offset of ne[] within
     * ggml_tensor, which is what a null dereference looks like from the
     * outside.  The build-time failure is reported by name elsewhere; this
     * only has to avoid turning it into a crash. */
    const struct ggml_tensor *X        = dst ? dst->src[0] : NULL;
    const struct ggml_tensor *b        = dst ? dst->src[1] : NULL;
    const struct ggml_tensor *c_tensor = dst ? dst->src[2] : NULL;

    if (!p || !X || !b || !c_tensor || !dst) {
        fprintf(stderr, "[roi_align] missing tensor (params=%p X=%p rois=%p "
                        "batch=%p dst=%p) -- output left untouched\n",
                (const void *)p, (const void *)X,
                (const void *)b, (const void *)c_tensor, (const void *)dst);
        return;
    }
    if (!X->data || !b->data || !c_tensor->data || !dst->data) {
        fprintf(stderr, "[roi_align] tensor without data (X=%p rois=%p "
                        "batch=%p dst=%p) -- output left untouched\n",
                (const void *)X->data, (const void *)b->data,
                (const void *)c_tensor->data, (const void *)dst->data);
        return;
    }

    /* Everything below reads ->data as host memory, so every tensor has to
     * actually live on the host.
     *
     * A non-NULL ->data is not enough: on Vulkan a device tensor's ->data is
     * an offset into VRAM -- a small integer -- which passes the NULL check
     * above and then segfaults on the first read, in a worker thread, with no
     * message because it never reaches GGML_ABORT.  All three inputs are srcs
     * now, so the scheduler does bring them across; this stays as the guard
     * that turns a silent crash into a diagnosable refusal. */
    {
        const struct ggml_tensor *need_host[] = { X, b, c_tensor, dst };
        const char *names[] = { "X", "rois", "batch_indices", "dst" };
        for (int q = 0; q < 4; q++) {
            const struct ggml_tensor *t = need_host[q];
            if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
                fprintf(stderr,
                    "[roi_align] '%s': %s tensor is on a non-host backend (%s) "
                    "-- this kernel reads host memory only, output left "
                    "untouched.\n",
                    dst->name, names[q], ggml_backend_buffer_name(t->buffer));
                return;
            }
        }
    }

    /* X: ggml [W, H, C, N] */
    const int W_in  = (int)X->ne[0];
    const int H_in  = (int)X->ne[1];
    const int C     = (int)X->ne[2];

    /* b = rois: ggml [4, num_rois] */
    const int num_rois = (int)b->ne[1];

    const float *X_data    = (const float *)X->data;
    const float *roi_data  = (const float *)b->data;
    float       *out_data  = (float *)dst->data;

    /* batch_indices is INT64 in the operator, and the loader downcasts every
     * INT64 initializer to GGML_TYPE_I32 (onnx_ggml.c:251) -- so the tensor
     * that arrives here normally holds int32, not float.
     *
     * Reading it as float was silently wrong rather than obviously so: the
     * bit pattern of int32 1 read as a float is 1.4e-45, and (int) of that is
     * 0, so every ROI sampled batch 0 no matter which image it asked for. With
     * a batch of one -- which is every model in the suite, MaskRCNN included
     * -- batch 0 is the right answer by accident, which is why this survived.
     * The ref_check case roialign_batch is the first thing to use two images:
     * ggmlR returned image 0's numbers where ORT returned image 1's, exactly
     * -3x apart because that is how the case builds the second image.
     *
     * Read by the tensor's actual type instead of assuming one. */
    const int32_t *batch_i32 = NULL;
    const float   *batch_f32 = NULL;
    if (c_tensor->type == GGML_TYPE_I32) {
        batch_i32 = (const int32_t *)c_tensor->data;
    } else if (c_tensor->type == GGML_TYPE_F32) {
        batch_f32 = (const float *)c_tensor->data;
    } else {
        fprintf(stderr, "[roi_align] '%s': batch_indices has unsupported type "
                        "%s -- treating every ROI as batch 0\n",
                dst->name, ggml_type_name(c_tensor->type));
    }

    const int oh = p->output_height;
    const int ow = p->output_width;
    const float scale = p->spatial_scale;

    /* Offer the work to the Vulkan shader first, fall through to the loop
     * below when it declines.
     *
     * Declining is normal, not exceptional: the flag is off by default (see
     * roi_align_gpu_enabled), the backend is NULL for a CPU-loaded model, and
     * the dispatch itself returns false when the output grid would exceed the
     * driver's workgroup limit.  In every one of those cases the CPU kernel
     * below runs and the answer is the reference one, so this is a pure
     * fast-path: it can never turn a working model into a broken one by being
     * absent, only by being wrong -- which is what the strict max|d| = 0 gate
     * in ref_check exists to catch before the flag is ever turned on. */
#ifdef GGML_USE_VULKAN
    if (p->gpu_backend && roi_align_gpu_enabled()) {
        const int N_batch = (int)X->ne[3];
        /* The shader takes batch_indices as floats, because that is the one
         * form both storage types can be handed to it in; converting here
         * keeps the int32/float distinction from crossing into GLSL. */
        float *bi_f32 = (float *)malloc((size_t)num_rois * sizeof(float));
        int dispatched = 0;
        if (bi_f32) {
            for (int q = 0; q < num_rois; q++)
                bi_f32[q] = batch_i32 ? (float)batch_i32[q]
                          : batch_f32 ? batch_f32[q] : 0.0f;
            if (getenv("ONNX_TRACE_ROI_ALIGN"))
                fprintf(stderr, "[roi_align/gpu] '%s': X=[%d,%d,%d,%d] rois=%d "
                                "out=[%d,%d] sr=%d mode=%d scale=%g\n",
                        dst->name, W_in, H_in, C, N_batch, num_rois,
                        ow, oh, p->sampling_ratio, p->mode, (double)scale);
            dispatched = ggml_vk_roi_align_run(
                p->gpu_backend,
                X_data, roi_data, bi_f32, out_data,
                (unsigned)W_in, (unsigned)H_in, (unsigned)C, (unsigned)N_batch,
                (unsigned)num_rois, (unsigned)ow, (unsigned)oh,
                p->sampling_ratio, (unsigned)p->mode, scale);
            free(bi_f32);
        }
        if (dispatched) {
            return;
        }
        /* Declined -- the CPU loop below runs instead, so the answer is still
         * the reference one and this is not an error.  It is still said out
         * loud, because the GPU path is the default now and a silent fallback
         * is indistinguishable from a shader that ran: someone measuring GPU
         * time would otherwise be timing the CPU kernel without knowing.
         *
         * Once per process, not once per node per inference: MaskRCNN has four
         * RoiAlign nodes and would repeat this on every frame. */
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr,
                "[roi_align] '%s': the Vulkan dispatch declined (output grid "
                "above the driver's workgroup limit) -- using the CPU kernel.\n"
                "            Results are unaffected; only this op runs on the "
                "host.  Further occurrences are not reported.\n",
                dst->name);
        }
    }
#endif

    /* ONNX_TRACE_ROI_ALIGN=1 also summarises the ROIs themselves, not just the
     * shapes the GPU trace above prints.
     *
     * This exists to answer one question: does the sample-count cap added in
     * this file actually BIND on real input?  If it does, the ROIs are
     * malformed and the bug is upstream in the box decode -- the cap only
     * stops that from reaching the GPU as an unkillable dispatch.  One line
     * per call, not one per ROI: MaskRCNN sends 895 of them. */
    if (getenv("ONNX_TRACE_ROI_ALIGN")) {
        float max_rh = 0.0f, max_rw = 0.0f;
        int n_bad = 0, n_capped = 0;
        for (int q = 0; q < num_rois; q++) {
            const float rw = (roi_data[2 + 4 * q] - roi_data[0 + 4 * q]) * scale;
            const float rh = (roi_data[3 + 4 * q] - roi_data[1 + 4 * q]) * scale;
            /* !(x == x) catches NaN; the infinities fail the <= tests below. */
            if (!(rw == rw) || !(rh == rh)) { n_bad++; continue; }
            if (rw > max_rw) max_rw = rw;
            if (rh > max_rh) max_rh = rh;
            if (p->sampling_ratio <= 0) {
                if (!(rh / oh <= (float)ROI_ALIGN_MAX_SAMPLES) ||
                    !(rw / ow <= (float)ROI_ALIGN_MAX_SAMPLES))
                    n_capped++;
            }
        }
        fprintf(stderr, "[roi_align/cpu] '%s': %d rois, max %.1fx%.1f, "
                        "sr=%d, non-finite=%d, over-cap=%d%s\n",
                dst->name, num_rois, (double)max_rw, (double)max_rh,
                p->sampling_ratio, n_bad, n_capped,
                (n_bad || n_capped) ? "  <-- MALFORMED, check the box decode"
                                    : "");
    }

    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
        /* ROI coords: b layout is [4, num_rois], so roi_data[coord + 4*roi_idx] */
        float x1 = roi_data[0 + 4 * roi_idx] * scale;
        float y1 = roi_data[1 + 4 * roi_idx] * scale;
        float x2 = roi_data[2 + 4 * roi_idx] * scale;
        float y2 = roi_data[3 + 4 * roi_idx] * scale;

        int batch_idx = batch_i32 ? (int)batch_i32[roi_idx]
                      : batch_f32 ? (int)batch_f32[roi_idx] : 0;
        /* A ROI naming an image outside the batch would read past the feature
         * map; ORT rejects such a model, so clamping here is only a guard. */
        if (batch_idx < 0)          batch_idx = 0;
        if (batch_idx >= (int)X->ne[3]) batch_idx = (int)X->ne[3] - 1;

        /* A degenerate ROI is forced to 1x1, not to an epsilon.
         *
         * Clamping to 1e-6 keeps the box degenerate: every sample inside it
         * lands on the same pixel, and the bin size underflows.  ONNX Runtime
         * says it plainly -- "Force malformed ROIs to be 1x1" -- and uses 1.0,
         * which spreads the 7x7 grid over a real pixel instead.  MaskRCNN's
         * RPN emits 95 such ROIs of 895 (zero width or zero height), and every
         * channel of every one of them came out wrong. */
        float roi_h = y2 - y1;
        float roi_w = x2 - x1;
        if (roi_h < 1.0f) roi_h = 1.0f;
        if (roi_w < 1.0f) roi_w = 1.0f;

        float bin_h = roi_h / (float)oh;
        float bin_w = roi_w / (float)ow;

        int sr_h = p->sampling_ratio > 0 ? p->sampling_ratio : (int)ceilf(roi_h / oh);
        int sr_w = p->sampling_ratio > 0 ? p->sampling_ratio : (int)ceilf(roi_w / ow);

        /* ⚠️ The adaptive branch above is the ONLY loop bound in this op that
         * comes from BUFFER CONTENTS rather than from a dimension: roi_h is
         * whatever the RPN emitted, and nothing upstream bounds it.  A ROI a
         * few thousand pixels tall therefore asks for a few thousand taps per
         * axis -- millions per output element, over ow*oh*C*num_rois elements.
         *
         * On this kernel that is merely a very long, interruptible loop.  On
         * the Vulkan port it is a compute dispatch that cannot finish inside
         * the driver's watchdog: measured on MaskRCNN, three runs hung the
         * amdgpu compute ring (comp_1.1.x timeout), and the third took the
         * graphics ring with it -- MODE1 reset, "VRAM is lost", desktop gone.
         *
         * ORT bounds this implicitly: its ROIs come from a box decoder that
         * clips to the image, so the quotient stays small and the cap never
         * binds on well-formed input.  128 taps per axis is far above anything
         * a real feature map produces (MaskRCNN's largest legitimate value is
         * single-digit) and far below the point where the dispatch stalls.
         *
         * ⚠️ If this cap ever BINDS, the output is no longer bit-exact with
         * ORT -- it is a guard against malformed input, not a tuning knob.
         * Reaching it means the ROIs are wrong, and the bug is upstream in the
         * box decode, not here.  Keep it identical to roi_align.comp: the two
         * kernels are compared bit for bit and must clamp at the same point.
         *
         * A non-finite roi_h reaches (int)ceilf() as undefined behaviour, so
         * the guard is written to catch it: !(x <= CAP) is true for NaN, while
         * (x > CAP) is not. */
        if (!(sr_h <= ROI_ALIGN_MAX_SAMPLES)) sr_h = ROI_ALIGN_MAX_SAMPLES;
        if (!(sr_w <= ROI_ALIGN_MAX_SAMPLES)) sr_w = ROI_ALIGN_MAX_SAMPLES;

        /* At least one, as ONNX Runtime does: a zero grid would divide by
         * zero rather than simply contribute nothing. */
        int grid_count = sr_h * sr_w;
        if (grid_count < 1) grid_count = 1;
        float count = (float)grid_count;

        for (int c_idx = 0; c_idx < C; c_idx++) {
            for (int ph = 0; ph < oh; ph++) {
                for (int pw = 0; pw < ow; pw++) {
                    float val;
                    if (p->mode == 1) {
                        /* max mode */
                        val = -FLT_MAX;
                    } else {
                        val = 0.0f;
                    }

                    for (int iy = 0; iy < sr_h; iy++) {
                        float y = y1 + bin_h * ph + bin_h * ((float)iy + 0.5f) / (float)sr_h;
                        for (int ix = 0; ix < sr_w; ix++) {
                            float x = x1 + bin_w * pw + bin_w * ((float)ix + 0.5f) / (float)sr_w;
                            float sample = bilinear_sample(X_data, W_in, H_in,
                                                           x, y, c_idx, batch_idx, C,
                                                           p->mode == 1);
                            if (p->mode == 1) {
                                if (sample > val) val = sample;
                            } else {
                                val += sample;
                            }
                        }
                    }
                    if (p->mode == 0) val /= count;

                    /* dst layout: [ow, oh, C, num_rois] */
                    out_data[pw + ow * (ph + oh * (c_idx + C * roi_idx))] = val;
                }
            }
        }
    }
}
