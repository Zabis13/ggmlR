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
#include "../ggml-backend.h"   /* ggml_backend_t: the GPU path's handle */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int center_point_box;  /* 0 = corner, 1 = center */

    /* The Vulkan backend when the model was loaded on one, else NULL, so the
     * kernel can offer its work to the shader.  A backend handle is not graph
     * state and does not move when buffers are reallocated, so unlike a tensor
     * it legitimately travels in userdata. */
    ggml_backend_t gpu_backend;
} nms_params_t;

/* Is the GPU path allowed for NonMaxSuppression?
 *
 * ON by default; GGMLR_ONNX_GPU_NMS=0 forces the CPU kernel.
 *
 * Per-op, like the other ONNX shaders, so an op still under test cannot hold
 * back one that has passed.  The shader only answers the parallel half of the
 * problem -- which boxes survive within each (batch, class) -- and this kernel
 * still assembles the output, because ONNX's ordering and its GLOBAL
 * max_selected cap are sequential decisions across classes. */
int nms_gpu_enabled(void);

/* Where nms_cpu reports how many boxes it kept.
 *
 * NOT op_params[0].  ggml_custom_4d stores its own
 * struct ggml_custom_op_params -- {fun, n_tasks, userdata}, 24 bytes --
 * at the start of the very same op_params array, so op_params[0..1] IS the
 * function pointer.  Writing the count there truncated nms_cpu's own address
 * to its high half plus the count: the first NMS node ran, corrupted the
 * pointer, and the second jumped to 0x....00000001 and died inside
 * the custom-op dispatcher with no name in the backtrace.
 *
 * Index 6 is the first int32 past that struct, and op_params holds 16. */
#define NMS_COUNT_SLOT 6

/* CPU callback for ggml_custom_4d.
 *
 * Every input arrives as a real src of dst, NOT as a pointer remembered in
 * userdata.  That is the whole reason this op is a GGML_OP_CUSTOM rather than
 * a map_custom3: the scheduler copies an op's srcs back to the host before
 * running a CPU-only op, and it can only do that for tensors it can see in
 * src[].  scores used to travel in userdata, so on Vulkan it stayed in VRAM
 * and the kernel refused to read it -- five NMS nodes in MaskRCNN produced
 * empty output, and the graph that got built around those empties ran into
 * GGML_SCHED_MAX_SPLIT_INPUTS.
 *
 *   dst:      output [3, max_possible_selected], filled with -1
 *   src[0]:   boxes  [4, num_boxes, N] (ggml order)
 *   src[1]:   scores [num_boxes, num_classes, N] (ggml order)
 *   src[2]:   params [4]: {max_output_boxes_per_class, iou_threshold,
 *             score_threshold, have_score_threshold}
 *
 * Actual number of selected boxes stored in dst->op_params[NMS_COUNT_SLOT].
 */
void nms_cpu(struct ggml_tensor *dst, int ith, int nth, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* NMS_H */
