/* nms.h — NonMaxSuppression custom op for ONNX
 *
 * ONNX NonMaxSuppression:
 *   Inputs:  boxes [N, num_boxes, 4]
 *            scores [N, num_classes, num_boxes]
 *            max_output_boxes_per_class (scalar)
 *            iou_threshold (scalar)
 *            score_threshold (scalar)
 *   Output:  selected_indices [num_selected, 3]
 *            each row = (batch_index, class_index, box_index)
 *
 *   center_point_box: 0 = corner format [y1,x1,y2,x2], 1 = center format [cx,cy,w,h]
 */

#ifndef NMS_H
#define NMS_H

#include "../ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int center_point_box;  /* 0 = corner, 1 = center */
    const struct ggml_tensor *scores; /* scores tensor (set at graph build time) */
} nms_params_t;

/* CPU callback for ggml_map_custom3.
 * dst: output [3, max_possible_selected] preallocated, filled with -1
 * a:   boxes [4, num_boxes, N] (ggml order)
 * b:   scores [num_boxes, num_classes, N] (ggml order)
 * c:   params tensor [3]: {max_output_boxes_per_class, iou_threshold_bits, score_threshold_bits}
 *      (thresholds stored as float bits in int via memcpy)
 *
 * Actual number of selected boxes stored in dst->op_params[0] after execution.
 */
void nms_cpu(struct ggml_tensor *dst,
             const struct ggml_tensor *a,
             const struct ggml_tensor *b,
             const struct ggml_tensor *c,
             int ith, int nth, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* NMS_H */
