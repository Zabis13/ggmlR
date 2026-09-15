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
#include "../ggml-backend.h"   /* ggml_backend_t: the GPU path's handle */

#ifdef __cplusplus
extern "C" {
#endif

/* Ceiling on the adaptive per-axis sample count (sampling_ratio <= 0).
 *
 * That branch derives its loop bound from ROI CONTENTS, which nothing upstream
 * bounds; without a cap a malformed ROI asks for millions of taps per output
 * element and the Vulkan dispatch hangs the compute ring hard enough to cost
 * the whole GPU (MODE1 reset, VRAM lost).  See the long note at the clamp site
 * in roi_align.c.
 *
 * ⚠️ Three places must agree on this number: roi_align.c, roi_align.comp (as a
 * literal -- GLSL cannot see this header), and the host gate in
 * ggml_vk_roi_align_run.  A binding cap means the ROIs are malformed and the
 * real bug is upstream in the box decode; it is not a performance knob. */
#define ROI_ALIGN_MAX_SAMPLES 128

typedef struct {
    int    output_height;    /* e.g. 7 */
    int    output_width;     /* e.g. 7 */
    int    sampling_ratio;   /* e.g. 2 (0 = adaptive) */
    float  spatial_scale;    /* e.g. 0.03125 */
    int    mode;             /* 0 = avg, 1 = max */

    /* The Vulkan backend, when the model was loaded on one, so the kernel can
     * offer its work to the GPU shader.  NULL means CPU-only and is also what
     * a CPU-loaded model leaves here.
     *
     * A backend handle is not a tensor: it is not graph state, does not move
     * when buffers are reallocated, and the scheduler has no business seeing
     * it -- so unlike X it legitimately travels in userdata. */
    ggml_backend_t gpu_backend;
} roi_align_params_t;

/* Is the GPU path allowed for RoiAlign?
 *
 * ON by default; GGMLR_ONNX_GPU_ROI_ALIGN=0 forces the CPU kernel.
 *
 * What licenses the default is an automated gate, not a judgement call:
 * tests/testthat/test-onnx-roialign-vulkan.R runs the same model on both
 * devices and demands IDENTICAL doubles -- expect_identical, no tolerance --
 * across full coverage, fractional bounds, degenerate ROIs, out-of-map taps,
 * the adaptive grid, max mode and batch selection.  The criterion is strict
 * because the risk is not a perturbed number: MaskRCNN's quantised scores sit
 * on thresholds, so one differing bit drops a detection.  If that suite ever
 * fails, the answer is to put this op back on the CPU, never to loosen the
 * comparison.
 *
 * The gate is per-op by design, so an op still under test cannot hold back one
 * that has passed, and the variable remains as an escape hatch for a driver
 * whose arithmetic disagrees.
 */
int roi_align_gpu_enabled(void);

/* CPU callback for ggml_custom_4d.
 *
 * Every input arrives as a real src of dst, NOT as a pointer remembered in
 * userdata.  X used to travel that way, with out->src[3] patched in after the
 * fact so the buffer stayed alive; neither reaches the scheduler, which walks
 * an op's srcs only up to its own arity when it plans splits.  On Vulkan the
 * feature map therefore stayed in VRAM and this host-only kernel could not
 * read it.  Same defect, same fix, as NonMaxSuppression -- see nms.h.
 *
 *   dst:    output [output_width, output_height, C, num_rois] (ggml order)
 *   src[0]: X    feature map [W, H, C, N] (ggml order)
 *   src[1]: rois [4, num_rois] (ggml order) -- each roi = [x1, y1, x2, y2]
 *   src[2]: batch_indices [num_rois] (F32 cast of int)
 */
void roi_align_cpu(struct ggml_tensor *dst, int ith, int nth, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* ROI_ALIGN_H */
