/* rel_pos_bias.c — CPU kernel for 2D Relative Position Bias (BoTNet)
 *
 * Replaces the 60+ node ONNX pos_embed subgraph with direct computation:
 *   bias[b,hq,wq,hk,wk] = sum_c(x[b,hq,wq,c] * W_h[c, hq-hk+H-1])
 *                        + sum_c(x[b,wq,hq,c] * W_w[c, wq-wk+W-1])
 */

#include "rel_pos_bias.h"
#include <stdlib.h>
#include <string.h>

void rel_pos_bias_2d_cpu(struct ggml_tensor *dst,
                         const struct ggml_tensor *a,
                         const struct ggml_tensor *b_tensor,
                         const struct ggml_tensor *c_tensor,
                         int ith, int nth, void *userdata)
{
    (void)a;
    (void)ith; (void)nth;

    const rel_pos_bias_params_t *p = (const rel_pos_bias_params_t *)userdata;
    const int H = p->H, W = p->W, B = p->B, C = p->C;
    const int rel_h = p->rel_h; /* 2*H-1 */
    const int rel_w = p->rel_w; /* 2*W-1 */
    const int HW = H * W;

    /* b_tensor = x: ggml ne[0]=C, ne[1]=H*W, ne[2]=B
     * x[b, h*W+w, c] = data[c + (h*W+w)*C + b*C*HW] */
    const float *x = (const float *)b_tensor->data;

    /* c_tensor = concat(W_h, W_w): weights may live on GPU buffer (->data invalid).
     * Use CPU-side copy from params if available, fall back to ->data for CPU runs. */
    const float *Wdata = p->w_cpu ? p->w_cpu : (const float *)c_tensor->data;
    const int Wstride = rel_h + rel_w;
    fprintf(stderr, "[rpb_dbg] H=%d W=%d B=%d C=%d "
            "b_ne=[%d,%d,%d] b_nb=[%zu,%zu,%zu] x=%p x[0]=%.4f x[1]=%.4f "
            "dst=%p dst_ne=[%d,%d,%d] Wdata=%p W[0]=%.4f W[1]=%.4f\n",
            H, W, B, C,
            (int)b_tensor->ne[0], (int)b_tensor->ne[1], (int)b_tensor->ne[2],
            b_tensor->nb[0], b_tensor->nb[1], b_tensor->nb[2],
            (void*)x, x ? x[0] : 0.f, x ? x[1] : 0.f,
            (void*)dst->data,
            (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2],
            (void*)Wdata, Wdata ? Wdata[0] : 0.f, Wdata ? Wdata[1] : 0.f);

    /* Output: dst ggml ne[0]=H*W, ne[1]=H*W, ne[2]=B
     * = ONNX [B, H*W, H*W]
     * dst[k_idx + q_idx*HW + b*HW*HW] where q=hq*W+wq, k=hk*W+wk */
    float *out = (float *)dst->data;
    memset(out, 0, (size_t)B * HW * HW * sizeof(float));

    /* Temp buffer for W-axis dot products: dot_w_cache[r] for r=0..2W-2 */
    float *dot_w_cache = (float *)malloc(rel_w * sizeof(float));
    if (!dot_w_cache) return;

    for (int b = 0; b < B; b++) {
        const float *xb = x + (size_t)b * C * HW;

        for (int hq = 0; hq < H; hq++) {
            for (int wq = 0; wq < W; wq++) {
                const int q_idx = hq * W + wq;

                /* x[b, hq, wq, :] — for H-axis matmul */
                const float *x_hw = xb + (size_t)(hq * W + wq) * C;

                /* x[b, wq, hq, :] — for W-axis matmul (transposed spatial) */
                const float *x_wh = xb + (size_t)(wq * H + hq) * C;

                /* Precompute all W-axis dot products for this (b, wq, hq) */
                for (int r = 0; r < rel_w; r++) {
                    float dot = 0.0f;
                    for (int ci = 0; ci < C; ci++)
                        dot += x_wh[ci] * Wdata[rel_h + r + ci * Wstride];
                    dot_w_cache[r] = dot;
                }

                for (int hk = 0; hk < H; hk++) {
                    const int r_h = hq - hk + H - 1;

                    /* H-axis dot product: x[b,hq,wq,:] · W_h[:, r_h] */
                    float dot_h = 0.0f;
                    for (int ci = 0; ci < C; ci++)
                        dot_h += x_hw[ci] * Wdata[r_h + ci * Wstride];

                    for (int wk = 0; wk < W; wk++) {
                        const int r_w = wq - wk + W - 1;
                        const int k_idx = hk * W + wk;

                        out[k_idx + (size_t)q_idx * HW + (size_t)b * HW * HW] =
                            dot_h + dot_w_cache[r_w];
                    }
                }
            }
        }
    }

    free(dot_w_cache);
}
