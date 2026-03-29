/* roi_align.c — RoiAlign CPU implementation */

#include "roi_align.h"
#include <math.h>
#include <float.h>

/* Bilinear interpolation on feature map X[W, H, C, N] (ggml layout).
 * x, y are in spatial coordinates (float). c_idx = channel, n_idx = batch. */
static float bilinear_sample(const float *data,
                             int W, int H,
                             float x, float y,
                             int c_idx, int n_idx,
                             int C) {
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

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

void roi_align_cpu(struct ggml_tensor *dst,
                   const struct ggml_tensor *a,
                   const struct ggml_tensor *b,
                   const struct ggml_tensor *c_tensor,
                   int ith, int nth, void *userdata) {
    (void)ith; (void)nth;
    (void)a; /* dummy — shape only */

    const roi_align_params_t *p = (const roi_align_params_t *)userdata;
    const struct ggml_tensor *X = p->X;

    /* X: ggml [W, H, C, N] */
    const int W_in  = (int)X->ne[0];
    const int H_in  = (int)X->ne[1];
    const int C     = (int)X->ne[2];

    /* b = rois: ggml [4, num_rois] */
    const int num_rois = (int)b->ne[1];

    const float *X_data    = (const float *)X->data;
    const float *roi_data  = (const float *)b->data;
    const float *batch_data = (const float *)c_tensor->data;
    float       *out_data  = (float *)dst->data;

    const int oh = p->output_height;
    const int ow = p->output_width;
    const float scale = p->spatial_scale;

    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
        /* ROI coords: b layout is [4, num_rois], so roi_data[coord + 4*roi_idx] */
        float x1 = roi_data[0 + 4 * roi_idx] * scale;
        float y1 = roi_data[1 + 4 * roi_idx] * scale;
        float x2 = roi_data[2 + 4 * roi_idx] * scale;
        float y2 = roi_data[3 + 4 * roi_idx] * scale;

        int batch_idx = (int)batch_data[roi_idx];

        float roi_h = y2 - y1;
        float roi_w = x2 - x1;
        if (roi_h < 1e-6f) roi_h = 1e-6f;
        if (roi_w < 1e-6f) roi_w = 1e-6f;

        float bin_h = roi_h / (float)oh;
        float bin_w = roi_w / (float)ow;

        int sr_h = p->sampling_ratio > 0 ? p->sampling_ratio : (int)ceilf(roi_h / oh);
        int sr_w = p->sampling_ratio > 0 ? p->sampling_ratio : (int)ceilf(roi_w / ow);

        float count = (float)(sr_h * sr_w);

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
                                                           x, y, c_idx, batch_idx, C);
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
