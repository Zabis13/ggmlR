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

/* Parameters stored in userdata (must have static lifetime during graph compute) */
typedef struct {
    int H;          /* spatial height */
    int W;          /* spatial width */
    int B;          /* number of heads (batch) */
    int C;          /* channel dim */
    int rel_h;      /* 2*H-1 */
    int rel_w;      /* 2*W-1 */
    /* CPU-side copies of W_h and W_w weights (needed when weights live on GPU) */
    float *w_cpu;   /* concat(W_h, W_w): [(rel_h+rel_w) * C] floats, col-major */
    int    w_cpu_stride; /* rel_h + rel_w */
} rel_pos_bias_params_t;

/* CPU callback for ggml_map_custom3.
 * dst: output [W, H, W, B*H] in ggml order = ONNX [B,H,H,W,W] collapsed
 * a:   dummy (same shape as dst, not used for data)
 * b:   x input [C, H*W, B] in ggml = ONNX [B, H*W, C]
 * c:   W_h and W_w concatenated [rel_h+rel_w, C] in ggml = ONNX [C, rel_h+rel_w]
 */
void rel_pos_bias_2d_cpu(struct ggml_tensor *dst,
                         const struct ggml_tensor *a,
                         const struct ggml_tensor *b,
                         const struct ggml_tensor *c,
                         int ith, int nth, void *userdata);

#ifdef __cplusplus
}
#endif

#endif /* REL_POS_BIAS_H */
