/* nms.c — NonMaxSuppression CPU implementation */

#include "nms.h"
#include <string.h>
#include <stdlib.h>

static float iou_corner(float y1_a, float x1_a, float y2_a, float x2_a,
                        float y1_b, float x1_b, float y2_b, float x2_b) {
    float inter_y1 = y1_a > y1_b ? y1_a : y1_b;
    float inter_x1 = x1_a > x1_b ? x1_a : x1_b;
    float inter_y2 = y2_a < y2_b ? y2_a : y2_b;
    float inter_x2 = x2_a < x2_b ? x2_a : x2_b;

    float inter_h = inter_y2 - inter_y1;
    float inter_w = inter_x2 - inter_x1;
    if (inter_h <= 0.0f || inter_w <= 0.0f) return 0.0f;

    float inter_area = inter_h * inter_w;
    float area_a = (y2_a - y1_a) * (x2_a - x1_a);
    float area_b = (y2_b - y1_b) * (x2_b - x1_b);
    float union_area = area_a + area_b - inter_area;

    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

/* Sort indices by score descending */
typedef struct { int idx; float score; } score_pair_t;

static int cmp_score_desc(const void *a, const void *b) {
    float sa = ((const score_pair_t *)a)->score;
    float sb = ((const score_pair_t *)b)->score;
    return (sb > sa) - (sb < sa);
}

void nms_cpu(struct ggml_tensor *dst,
             const struct ggml_tensor *a,
             const struct ggml_tensor *b,
             const struct ggml_tensor *c_tensor,
             int ith, int nth, void *userdata) {
    (void)ith; (void)nth;

    (void)a; /* dummy — shape only */

    const nms_params_t *p = (const nms_params_t *)userdata;
    const struct ggml_tensor *boxes_t  = b;
    const struct ggml_tensor *scores_t = p->scores;

    /* boxes: ggml [4, num_boxes, N] */
    const int num_boxes = (int)boxes_t->ne[1];
    const int N         = (int)boxes_t->ne[2];

    /* scores: ggml [num_boxes, num_classes, N] */
    const int num_classes = (int)scores_t->ne[1];

    /* c = params: [max_output_boxes, iou_threshold_bits, score_threshold_bits] */
    const float *params_data = (const float *)c_tensor->data;
    int   max_output = (int)params_data[0];
    float iou_thresh, score_thresh;
    memcpy(&iou_thresh,   &params_data[1], sizeof(float));
    memcpy(&score_thresh,  &params_data[2], sizeof(float));

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
    int *suppressed = (int *)malloc((size_t)num_boxes * sizeof(int));
    if (!sorted || !suppressed) { free(sorted); free(suppressed); return; }

    int total_selected = 0;

    for (int batch = 0; batch < N && total_selected < max_selected; batch++) {
        const float *boxes_n = box_data + batch * 4 * num_boxes;

        for (int cls = 0; cls < num_classes && total_selected < max_selected; cls++) {
            const float *scores_nc = score_data + batch * num_classes * num_boxes + cls * num_boxes;

            /* Build sorted list by score */
            int n_candidates = 0;
            for (int i = 0; i < num_boxes; i++) {
                if (scores_nc[i] > score_thresh) {
                    sorted[n_candidates].idx = i;
                    sorted[n_candidates].score = scores_nc[i];
                    n_candidates++;
                }
            }
            qsort(sorted, (size_t)n_candidates, sizeof(score_pair_t), cmp_score_desc);

            memset(suppressed, 0, (size_t)num_boxes * sizeof(int));
            int selected_this_class = 0;

            for (int i = 0; i < n_candidates; i++) {
                int idx_i = sorted[i].idx;
                if (suppressed[idx_i]) continue;

                /* Output this box */
                if (total_selected < max_selected) {
                    /* dst layout: [3, max_selected], so out[coord + 3*sel] */
                    out_data[0 + 3 * total_selected] = (float)batch;
                    out_data[1 + 3 * total_selected] = (float)cls;
                    out_data[2 + 3 * total_selected] = (float)idx_i;
                    total_selected++;
                    selected_this_class++;
                }
                if (selected_this_class >= max_output) break;

                /* Suppress overlapping boxes */
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

                for (int j = i + 1; j < n_candidates; j++) {
                    int idx_j = sorted[j].idx;
                    if (suppressed[idx_j]) continue;

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
                    if (iou > iou_thresh)
                        suppressed[idx_j] = 1;
                }
            }
        }
    }

    /* Store actual count in op_params for downstream */
    dst->op_params[0] = total_selected;

    free(sorted);
    free(suppressed);
}
