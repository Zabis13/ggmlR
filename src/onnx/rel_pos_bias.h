/* rel_pos_bias.h — 2D Relative Position Bias (BoTNet-style)
 *
 * Fused custom op replacing the 60+ node pos_embed ONNX subgraph:
 *   x[B,H,W,C] × W_h[C,2H-1] + x_transposed × W_w[C,2W-1]
 *   → bias[B,H,W,H,W] (collapsed to 4D)
 *
 * Uses the Toeplitz/pad-flatten-slice trick to extract relative indices.
 */

#ifndef REL_POS_BIAS_H
#define REL_POS_BIAS_H

#include "../ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Shape of one detected pos_embed block, carried from the pre-pass to the
 * emission site.  No weights: both backends of the REL_POS_BIAS op read the
 * graph tensor built there by ggml_concat. */
typedef struct {
    int H;          /* spatial height */
    int W;          /* spatial width */
    int B;          /* number of heads (batch) */
    int C;          /* channel dim */
    int rel_h;      /* 2*H-1 */
    int rel_w;      /* 2*W-1 */
} rel_pos_bias_params_t;

/* rel_pos_bias_2d_cpu() was declared here, the ggml_map_custom3 callback from
 * when this was a custom op.  REL_POS_BIAS is a real ggml op now: the CPU side
 * is ggml_compute_forward_rel_pos_bias (ggml-cpu/ops-misc.cpp) and the GPU side
 * is vulkan-shaders/rel_pos_bias.comp.
 *
 * ⚠️Keeping the old kernel compiled alongside them cost a day: its formula was
 * the pre-ORT one, and when the live CPU kernel was corrected against ONNX
 * Runtime the shader was left matching this dead one instead -- BoTNet26t read
 * 1.55 off the reference on Vulkan while CPU read 1.43e-06.  One op, one
 * reference per backend. */

#ifdef __cplusplus
}
#endif

#endif /* REL_POS_BIAS_H */
