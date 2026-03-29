/* roi_align.h — RoiAlign custom op for ONNX MaskRCNN etc.
 *
 * Implements ONNX RoiAlign (opset 10+):
 *   Input:  X [N,C,H,W], rois [num_rois,4], batch_indices [num_rois]
 *   Output: [num_rois, C, output_height, output_width]
 *
 * Each ROI is divided into output_height×output_width bins.
 * Each bin samples sampling_ratio² points via bilinear interpolation.
 * Mode: "avg" averages samples, "max" takes the max.
 */

#ifndef ROI_ALIGN_H
#define ROI_ALIGN_H

#include "../ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    output_height;    /* e.g. 7 */
    int    output_width;     /* e.g. 7 */
    int    sampling_ratio;   /* e.g. 2 (0 = adaptive) */
    float  spatial_scale;    /* e.g. 0.03125 */
    int    mode;             /* 0 = avg, 1 = max */
    const struct ggml_tensor *X; /* feature map tensor (set at graph build time) */
} roi_align_params_t;

/* CPU callback for ggml_map_custom3.
 * dst: output [output_width, output_height, C, num_rois] (ggml order)
 * a:   dummy (defines output shape, not used for data)
 * b:   rois [4, num_rois] (ggml order) — each roi = [x1, y1, x2, y2]
 * c:   batch_indices [num_rois] (F32 cast of int)
 * X feature map read from params->X
 */
void roi_align_cpu(struct ggml_tensor *dst,
                   const struct ggml_tensor *a,
                   const struct ggml_tensor *b,
                   const struct ggml_tensor *c,
                   int ith, int nth, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* ROI_ALIGN_H */
