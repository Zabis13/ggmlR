/* nms.c — NonMaxSuppression CPU implementation */

#include "nms.h"
#include "../ggml-backend.h"   /* buffer_is_host: this kernel reads host memory */
#ifdef GGML_USE_VULKAN
#include "../ggml-vulkan.h"    /* ggml_vk_nms_run: the GPU fast path */
#endif
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* On unless GGMLR_ONNX_GPU_NMS=0 -- see the note in nms.h.
 * Cached: this runs per node per inference and getenv walks the environment. */
int nms_gpu_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("GGMLR_ONNX_GPU_NMS");
        cached = (e && e[0] == '0' && e[1] == '\0') ? 0 : 1;
    }
    return cached;
}

/* Order a pair, the way ONNX Runtime's MaxMin does. */
static inline void nms_maxmin(float lhs, float rhs, float *mn, float *mx) {
    if (lhs >= rhs) { *mn = rhs; *mx = lhs; }
    else            { *mn = lhs; *mx = rhs; }
}

/* IoU, computed the way non_max_suppression_helper.h computes it.
 *
 * The shape of this is copied deliberately, operation for operation, because
 * the answer sits on a knife edge: MaskRCNN's RPN has a pair whose IoU is
 * 0.7000000477 against a threshold of 0.7 -- four parts in a hundred million
 * over the line.  Reassociating the arithmetic moves it to 0.6999989, which
 * is UNDER, and that one flip changes which box survives and cascades through
 * the whole detection list.
 *
 * Two details matter and neither is cosmetic:
 *   - each coordinate pair is ORDERED first (MaxMin), rather than assuming
 *     y1 < y2, and the areas are computed from the ordered bounds;
 *   - the overlap is rejected on the bounds themselves, before any area is
 *     formed, so a degenerate box never reaches the division. */
static float iou_corner(float y1_a, float x1_a, float y2_a, float x2_a,
                        float y1_b, float x1_b, float y2_b, float x2_b) {
    float x1_min, x1_max, x2_min, x2_max;
    float y1_min, y1_max, y2_min, y2_max;

    nms_maxmin(x1_a, x2_a, &x1_min, &x1_max);
    nms_maxmin(x1_b, x2_b, &x2_min, &x2_max);

    const float inter_x_min = x1_min > x2_min ? x1_min : x2_min;
    const float inter_x_max = x1_max < x2_max ? x1_max : x2_max;
    if (inter_x_max <= inter_x_min) return 0.0f;

    nms_maxmin(y1_a, y2_a, &y1_min, &y1_max);
    nms_maxmin(y1_b, y2_b, &y2_min, &y2_max);

    const float inter_y_min = y1_min > y2_min ? y1_min : y2_min;
    const float inter_y_max = y1_max < y2_max ? y1_max : y2_max;
    if (inter_y_max <= inter_y_min) return 0.0f;

    const float inter_area = (inter_x_max - inter_x_min) *
                             (inter_y_max - inter_y_min);
    if (inter_area <= 0.0f) return 0.0f;

    const float area_a = (x1_max - x1_min) * (y1_max - y1_min);
    const float area_b = (x2_max - x2_min) * (y2_max - y2_min);
    const float union_area = area_a + area_b - inter_area;

    if (area_a <= 0.0f || area_b <= 0.0f || union_area <= 0.0f) return 0.0f;

    return inter_area / union_area;
}

/* Does this score clear the threshold?
 *
 * One function, called by both the selection loop and the ONNX_TRACE_NMS
 * counter.  They used to test separately, and the copies drifted: the counter
 * kept a strict ">" while the loop was changed, so the diagnostic reported
 * 467 candidates while the loop was admitting 1000 -- and the number that a
 * person reads was the stale one.  A diagnostic that can disagree with the
 * code it describes is worse than none, because it is trusted.
 *
 * absent == threshold not supplied: ONNX applies no score filter at all, and
 * a score of exactly 0 (the bottom of a quantised range, and common) stays. */
static inline int nms_passes(float score, float thresh, int have_thresh) {
    return !have_thresh || score > thresh;
}

/* What did this kernel actually leave in dst?
 *
 * The per-node value trace reads dst LATER, through the scheduler, and on
 * Vulkan it reports sums that no selection could produce -- 111 elements all
 * equal to 1, or the denormal 0x00000001 -- while the inputs feeding this op
 * match the CPU bit for bit.  That is consistent with two very different
 * faults: the kernel computing the wrong answer, or the kernel computing the
 * right one into a buffer nobody reads back.
 *
 * Summing dst here, at the moment of writing, separates them.  If this number
 * matches the CPU reference and the node trace does not, the loss is in the
 * hand-off, not in the selection.  Called from every path that returns after
 * touching dst, so a path that quietly leaves dst alone is visible too. */
static void nms_dump_dst(const struct ggml_tensor *dst, const char *where,
                         int count) {
    if (!getenv("ONNX_TRACE_NMS_ENTER") || !dst || !dst->data) return;
    const float  *od = (const float *)dst->data;
    const int64_t n  = ggml_nelements(dst);
    double sum = 0.0;
    float  mn  = n ? od[0] : 0.0f, mx = mn;
    for (int64_t q = 0; q < n; q++) {
        sum += od[q];
        if (od[q] < mn) mn = od[q];
        if (od[q] > mx) mx = od[q];
    }
    fprintf(stderr, "[nms-wrote] '%s' via %s: n=%lld sum=%.6g min=%g max=%g "
                    "selected=%d\n",
            dst->name, where, (long long)n, sum, (double)mn, (double)mx, count);
}

/* Sort indices by score descending */
typedef struct { int idx; float score; } score_pair_t;

static int cmp_score_desc(const void *a, const void *b) {
    const score_pair_t *pa = (const score_pair_t *)a;
    const score_pair_t *pb = (const score_pair_t *)b;
    if (pa->score != pb->score)
        return (pb->score > pa->score) - (pb->score < pa->score);
    /* Equal scores break by index, lowest first.  qsort is not stable, so
     * without this the order among ties is whatever the implementation does,
     * and NMS is order-dependent: the box that comes first suppresses its
     * neighbours rather than the other way round.  MaskRCNN's surviving class
     * puts 95 candidates through here with the top five sharing one score to
     * the bit, and leaving the tie unordered dropped three detections that
     * ONNX Runtime keeps -- and dropped the highest-scoring ones at that. */
    return (pa->idx > pb->idx) - (pa->idx < pb->idx);
}

void nms_cpu(struct ggml_tensor *dst, int ith, int nth, void *userdata) {
    (void)ith; (void)nth;

    /* Diagnostic: did this kernel run at all?
     *
     * Every other print below sits behind a condition, so silence from this
     * function is ambiguous -- "never called" and "called, took a path that
     * happens not to print" look identical from outside.  On Vulkan the NMS
     * nodes produce output while ONNX_TRACE_NMS prints nothing at all, and
     * those two readings lead to opposite places: the kernel, or the segment
     * dispatch that should reach it.  This line is unconditional within the
     * function, so it separates them.
     *
     * The call counter makes the answer usable if the kernel IS reached: the
     * next question is always "from which call does it go wrong", and the
     * split index is not available here (a ggml_custom_4d callback is not told
     * which split it belongs to), so an ordinal is the closest stand-in. */
    if (getenv("ONNX_TRACE_NMS_ENTER")) {
        static int n_enter = 0;
        fprintf(stderr, "[nms-enter] #%d dst=%p name='%s'\n",
                ++n_enter, (const void *)dst, dst ? dst->name : "(null)");
    }

    const nms_params_t *p = (const nms_params_t *)userdata;

    /* All three inputs are srcs of dst, so the scheduler has already brought
     * them to the host by the time this runs.  They are still checked before
     * being followed: under segmented execution buffers are reset and
     * reallocated between build and run, so a tensor that existed at build
     * time may have no data now.  Reading it then is a null dereference in a
     * worker thread, which comes out as a bare segfault with no message and
     * no ring dump, because it never reaches GGML_ABORT. */
    const struct ggml_tensor *b        = dst ? dst->src[0] : NULL;
    const struct ggml_tensor *scores_s = dst ? dst->src[1] : NULL;
    const struct ggml_tensor *c_tensor = dst ? dst->src[2] : NULL;

    if (!p || !scores_s || !b || !c_tensor || !dst) {
        fprintf(stderr, "[nms] missing tensor (params=%p scores=%p boxes=%p "
                        "c=%p dst=%p) -- output left empty\n",
                (const void *)p, (const void *)scores_s,
                (const void *)b, (const void *)c_tensor, (const void *)dst);
        if (dst && dst->data) {
            float *od = (float *)dst->data;
            for (int64_t q = 0; q < ggml_nelements(dst); q++) od[q] = -1.0f;
        }
        return;
    }
    if (!b->data || !scores_s->data || !c_tensor->data || !dst->data) {
        fprintf(stderr, "[nms] tensor without data (boxes=%p scores=%p "
                        "params=%p dst=%p) -- output left empty\n",
                (const void *)b->data, (const void *)scores_s->data,
                (const void *)c_tensor->data, (const void *)dst->data);
        if (dst->data) {
            float *od = (float *)dst->data;
            for (int64_t q = 0; q < ggml_nelements(dst); q++) od[q] = -1.0f;
        }
        return;
    }
    /* Everything below reads ->data as host memory, so every tensor has to
     * actually live on the host.
     *
     * A non-NULL ->data is not enough.  On Vulkan a device tensor's ->data is
     * an offset into VRAM -- small integers like 0x3010 -- which passes the
     * NULL check above and then segfaults on the first read, in a worker
     * thread, with no message and no ring dump because it never reaches
     * GGML_ABORT.  That crash cost a session: the address in the report
     * (0x124c) looks like a corrupted pointer rather than what it is.
     *
     * The scheduler makes host copies of an op's srcs, and boxes, scores and
     * params are all srcs of dst now, so all three arrive on the host.  The
     * check stays as a guard: scores used to travel in userdata instead, and
     * on Vulkan it silently stayed in VRAM.  Refuse rather than read it. */
    {
        const struct ggml_tensor *need_host[] = { b, scores_s, c_tensor, dst };
        const char *names[] = { "boxes", "scores", "params", "dst" };
        for (int q = 0; q < 4; q++) {
            const struct ggml_tensor *t = need_host[q];
            if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
                fprintf(stderr,
                    "[nms] '%s': %s tensor is on a non-host backend (%s) -- "
                    "this kernel reads host memory only, output left empty.\n"
                    "      NonMaxSuppression needs its inputs on the CPU; run "
                    "this model with device=\"cpu\".\n",
                    dst->name, names[q],
                    ggml_backend_buffer_name(t->buffer));
                if (dst->data && dst->buffer &&
                    ggml_backend_buffer_is_host(dst->buffer)) {
                    float *od = (float *)dst->data;
                    for (int64_t z = 0; z < ggml_nelements(dst); z++) od[z] = -1.0f;
                }
                nms_dump_dst(dst, "nonhost-guard", -1);
                return;
            }
        }
    }

    const struct ggml_tensor *boxes_t  = b;
    const struct ggml_tensor *scores_t = scores_s;

    /* boxes: ggml [4, num_boxes, N] */
    const int num_boxes = (int)boxes_t->ne[1];
    const int N         = (int)boxes_t->ne[2];

    /* scores: ggml [num_boxes, num_classes, N] */
    const int num_classes = (int)scores_t->ne[1];

    /* c = params: [max_output_boxes, iou_threshold_bits, score_threshold_bits,
     *              have_score_threshold] */
    const float *params_data = (const float *)c_tensor->data;
    int   max_output = (int)params_data[0];
    float iou_thresh, score_thresh;
    memcpy(&iou_thresh,   &params_data[1], sizeof(float));
    memcpy(&score_thresh,  &params_data[2], sizeof(float));
    /* Older graphs built before slot 3 existed leave it unwritten; treating a
     * short params tensor as "threshold given" keeps their behaviour. */
    const int have_score_thresh =
        ggml_nelements(c_tensor) > 3 ? (params_data[3] != 0.0f) : 1;

    if (max_output <= 0) max_output = num_boxes;

    const float *box_data   = (const float *)boxes_t->data;
    const float *score_data = (const float *)scores_t->data;
    float       *out_data   = (float *)dst->data;

    int max_selected = (int)dst->ne[1];

    /* Initialize output to -1 */
    for (int i = 0; i < max_selected * 3; i++)
        out_data[i] = -1.0f;

    /* Temp arrays */
    score_pair_t *sorted = (score_pair_t *)malloc((size_t)num_boxes * sizeof(score_pair_t));
    /* Boxes kept so far for the current class.  A candidate is tested against
     * these and nothing else, which is what ONNX Runtime does; there is no
     * "suppressed" flag array any more, because a flag set by a box that the
     * per-class cap later dropped would outlive the box that set it. */
    int *selected_idx = (int *)malloc((size_t)num_boxes * sizeof(int));
    if (!sorted || !selected_idx) { free(sorted); free(selected_idx); return; }

    int total_selected = 0;

#ifdef GGML_USE_VULKAN
    /* Offer the per-class work to the shader.
     *
     * The shader answers only the parallel half: for each (batch, class) pair
     * it filters by score, sorts with the index tie-break, and runs that pair's
     * selection, writing the survivors into its own slice. This kernel still
     * does the serial half below -- walking the pairs in the operator's
     * batch-major, class-major order and stopping at the GLOBAL max_selected.
     * Those cannot move into the shader: workgroups finish in an arbitrary
     * order, and the global cap is a decision that spans them.
     *
     * Declining is normal (no backend, too many boxes for the shader's fixed
     * sort capacity, too many pairs for the driver) and simply leaves the CPU
     * loop to do everything, so this is a pure fast path. */
    if (p->gpu_backend && nms_gpu_enabled()) {
        const int cap = max_output < num_boxes ? max_output : num_boxes;
        const size_t n_pairs = (size_t)num_classes * N;
        int *sel_idx = (int *)malloc(n_pairs * (size_t)cap * sizeof(int));
        int *sel_cnt = (int *)malloc(n_pairs * sizeof(int));

        if (sel_idx && sel_cnt &&
            ggml_vk_nms_run(p->gpu_backend, box_data, score_data,
                            sel_idx, sel_cnt,
                            (unsigned)num_boxes, (unsigned)num_classes,
                            (unsigned)N, (unsigned)cap, max_output,
                            iou_thresh, score_thresh,
                            (unsigned)have_score_thresh,
                            (unsigned)p->center_point_box)) {
            /* Assemble in the operator's order, applying the global cap. This
             * is the same nesting as the CPU loop below, which is what makes
             * the two agree element for element. */
            for (int batch = 0; batch < N && total_selected < max_selected; batch++) {
                for (int cls = 0; cls < num_classes && total_selected < max_selected; cls++) {
                    const size_t pair = (size_t)batch * num_classes + cls;
                    const int    got  = sel_cnt[pair];
                    for (int s = 0; s < got && total_selected < max_selected; s++) {
                        out_data[0 + 3 * total_selected] = (float)batch;
                        out_data[1 + 3 * total_selected] = (float)cls;
                        out_data[2 + 3 * total_selected] =
                            (float)sel_idx[pair * (size_t)cap + s];
                        total_selected++;
                    }
                }
            }
            dst->op_params[NMS_COUNT_SLOT] = total_selected;
            nms_dump_dst(dst, "gpu", total_selected);
            free(sel_idx);
            free(sel_cnt);
            free(sorted);
            free(selected_idx);
            return;
        }

        free(sel_idx);
        free(sel_cnt);
        {
            static int warned = 0;
            if (!warned) {
                warned = 1;
                fprintf(stderr,
                    "[nms] '%s': the Vulkan dispatch declined (more boxes than "
                    "the shader's sort capacity, or too many classes for the "
                    "driver) -- using the CPU kernel.\n      Results are "
                    "unaffected; only this op runs on the host.  Further "
                    "occurrences are not reported.\n", dst->name);
            }
        }
    }
#endif

    /* One print that answers both questions at once.
     *
     * If the score range is degenerate -- all equal, all zero, or wildly out
     * of [0,1] -- the inputs never arrived intact and the fault is upstream,
     * in how this segment received its data.  If the range looks like real
     * scores and nothing is selected anyway, the fault is the comparison:
     * the threshold itself, or its dequantisation.
     *
     * Both numbers are needed together; either alone leaves the other
     * explanation open. */
    if (getenv("ONNX_TRACE_NMS")) {
        const int64_t n_sc = ggml_nelements(scores_t);
        const int64_t n_bx = ggml_nelements(boxes_t);
        /* An empty tensor would make the seed reads below go out of bounds. */
        float smin = n_sc ? score_data[0] : 0.0f;
        float smax = smin, ssum = 0.0f;
        int64_t n_above = 0, n_nan = 0;
        for (int64_t q = 0; q < n_sc; q++) {
            float v = score_data[q];
            if (v != v) { n_nan++; continue; }
            if (v < smin) smin = v;
            if (v > smax) smax = v;
            ssum += v;
            if (nms_passes(v, score_thresh, have_score_thresh)) n_above++;
        }
        float bmin = n_bx ? box_data[0] : 0.0f;
        float bmax = bmin;
        for (int64_t q = 0; q < n_bx; q++) {
            float v = box_data[q];
            if (v != v) continue;
            if (v < bmin) bmin = v;
            if (v > bmax) bmax = v;
        }
        fprintf(stderr,
            "[nms] '%s': boxes ne=[%lld,%lld,%lld] range [%g..%g] first %g,%g,%g,%g | "
            "scores ne=[%lld,%lld,%lld] n=%lld range [%g..%g] mean %g nan=%lld | "
            "thresh score=%g%s iou=%g max_out=%d -> %lld above threshold\n",
            dst->name,
            (long long)boxes_t->ne[0], (long long)boxes_t->ne[1], (long long)boxes_t->ne[2],
            (double)bmin, (double)bmax,
            (double)(n_bx > 0 ? box_data[0] : 0), (double)(n_bx > 1 ? box_data[1] : 0),
            (double)(n_bx > 2 ? box_data[2] : 0), (double)(n_bx > 3 ? box_data[3] : 0),
            (long long)scores_t->ne[0], (long long)scores_t->ne[1], (long long)scores_t->ne[2],
            (long long)n_sc, (double)smin, (double)smax,
            (double)(n_sc ? ssum / (float)(n_sc - n_nan) : 0.0f), (long long)n_nan,
            (double)score_thresh, have_score_thresh ? "" : "(absent)",
            (double)iou_thresh, max_output,
            (long long)n_above);
    }

    for (int batch = 0; batch < N && total_selected < max_selected; batch++) {
        const float *boxes_n = box_data + batch * 4 * num_boxes;

        for (int cls = 0; cls < num_classes && total_selected < max_selected; cls++) {
            const float *scores_nc = score_data + batch * num_classes * num_boxes + cls * num_boxes;

            /* Build sorted list by score */
            int n_candidates = 0;
            for (int i = 0; i < num_boxes; i++) {
                if (nms_passes(scores_nc[i], score_thresh, have_score_thresh)) {
                    sorted[n_candidates].idx = i;
                    sorted[n_candidates].score = scores_nc[i];
                    n_candidates++;
                }
            }
            qsort(sorted, (size_t)n_candidates, sizeof(score_pair_t), cmp_score_desc);

            /* A candidate is tested against the boxes ALREADY SELECTED, and
             * nothing is marked ahead of time.
             *
             * The kernel used to walk forward from each winner and flag every
             * overlapping candidate as suppressed.  That is the same thing
             * only while every winner survives to the end.  It is not the same
             * once max_output_boxes_per_class cuts the loop short: a candidate
             * flagged by a box that the cap later dropped stays flagged, and
             * can never be picked, though nothing that was actually selected
             * overlaps it.  ONNX Runtime's loop (non_max_suppression.cc) has
             * no such state -- it compares next_top_score against
             * selected_boxes_inside_class and nothing else.
             *
             * Measured on MaskRCNN node 1170, 510 selections: the forward
             * marking disagreed with ONNX Runtime on 49 of them, this form on
             * 3, and the residual 3 are boxes whose IoU sits within a float
             * ulp of the 0.7 threshold. */
            int selected_this_class = 0;
            int sel_count = 0;   /* indices into `selected_idx` for this class */

            for (int i = 0; i < n_candidates && selected_this_class < max_output
                                             && total_selected < max_selected; i++) {
                int idx_i = sorted[i].idx;

                float y1_i, x1_i, y2_i, x2_i;
                if (p->center_point_box == 1) {
                    float cx = boxes_n[0 + 4 * idx_i];
                    float cy = boxes_n[1 + 4 * idx_i];
                    float w  = boxes_n[2 + 4 * idx_i];
                    float h  = boxes_n[3 + 4 * idx_i];
                    y1_i = cy - h * 0.5f; x1_i = cx - w * 0.5f;
                    y2_i = cy + h * 0.5f; x2_i = cx + w * 0.5f;
                } else {
                    y1_i = boxes_n[0 + 4 * idx_i];
                    x1_i = boxes_n[1 + 4 * idx_i];
                    y2_i = boxes_n[2 + 4 * idx_i];
                    x2_i = boxes_n[3 + 4 * idx_i];
                }

                int keep = 1;
                for (int s = 0; s < sel_count; s++) {
                    int idx_j = selected_idx[s];

                    float y1_j, x1_j, y2_j, x2_j;
                    if (p->center_point_box == 1) {
                        float cx = boxes_n[0 + 4 * idx_j];
                        float cy = boxes_n[1 + 4 * idx_j];
                        float w  = boxes_n[2 + 4 * idx_j];
                        float h  = boxes_n[3 + 4 * idx_j];
                        y1_j = cy - h * 0.5f; x1_j = cx - w * 0.5f;
                        y2_j = cy + h * 0.5f; x2_j = cx + w * 0.5f;
                    } else {
                        y1_j = boxes_n[0 + 4 * idx_j];
                        x1_j = boxes_n[1 + 4 * idx_j];
                        y2_j = boxes_n[2 + 4 * idx_j];
                        x2_j = boxes_n[3 + 4 * idx_j];
                    }

                    float iou = iou_corner(y1_i, x1_i, y2_i, x2_i,
                                           y1_j, x1_j, y2_j, x2_j);
                    if (iou > iou_thresh) {
                        keep = 0;
                        if (getenv("ONNX_TRACE_NMS_SUPPRESS"))
                            fprintf(stderr, "[nmssup] '%s' cls=%d: %d suppresses %d (iou=%.6f > %.6f)\n",
                                    dst->name, cls, idx_j, idx_i, (double)iou, (double)iou_thresh);
                        break;
                    }
                }
                if (!keep) continue;

                selected_idx[sel_count++] = idx_i;
                selected_this_class++;

                if (total_selected < max_selected) {
                    /* dst layout: [3, max_selected], so out[coord + 3*sel] */
                    out_data[0 + 3 * total_selected] = (float)batch;
                    out_data[1 + 3 * total_selected] = (float)cls;
                    out_data[2 + 3 * total_selected] = (float)idx_i;
                    total_selected++;
                }
            }
        }
    }

    /* Store actual count in op_params for downstream */
    dst->op_params[NMS_COUNT_SLOT] = total_selected;
    nms_dump_dst(dst, "cpu", total_selected);

    free(sorted);
    free(selected_idx);
}
