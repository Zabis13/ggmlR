/* qconv_i32.h — QLinearConv with an exact integer accumulator.
 *
 * See qconv_i32.c for why this exists and what it is scoped to.
 */

#ifndef QCONV_I32_H
#define QCONV_I32_H

#include "../ggml.h"
#include "../ggml-backend.h"   /* buffer_is_host: this kernel reads host memory */

#include <stdint.h>

#define QCONV_I32_MAX_CHANNELS 4096

/* Scalars and per-channel tables the kernel needs, copied by VALUE at graph
 * build time.
 *
 * Copied rather than referenced: a tensor pointer captured at build time can
 * dangle when the op actually runs, because segmented execution resets and
 * reallocates buffers in between.  w_scale and bias are per output channel in
 * real models, so they are arrays, not scalars. */
typedef struct {
    float   x_scale;
    float   y_scale;
    int32_t x_zp;
    int32_t y_zp;

    /* Weight scale AND zero point are both per output channel in real models:
     * MaskRCNN's node 482 carries 256 of each.  Reading only the first left
     * six outputs in channel 89 one code off -- an integer accumulator is
     * exact, so a wrong zero point is a wrong answer, not a rounding wobble. */
    int     n_w_scale;                        /* 1 = shared, else C_out */
    float   w_scale[QCONV_I32_MAX_CHANNELS];
    int     n_w_zp;                           /* 1 = shared, else C_out */
    int32_t w_zp[QCONV_I32_MAX_CHANNELS];
    int32_t bias_data[QCONV_I32_MAX_CHANNELS];
    const int32_t *bias;                      /* NULL when the conv has none */

    int     kw, kh;
    int     stride_w, stride_h;
    int     pad_w, pad_h;
    int     dil_w, dil_h;

    float   out_lo, out_hi;                   /* saturation, from the zp dtype */
} qconv_i32_params_t;

/* ggml_map_custom3 kernel: dst = requantise(conv_i32(b, c)).
 * `a` carries the output shape only; x is `b` and w is `c`. */
void qconv_i32_cpu(struct ggml_tensor *dst,
                   const struct ggml_tensor *a,
                   const struct ggml_tensor *b,
                   const struct ggml_tensor *c,
                   int ith, int nth, void *userdata);

#endif /* QCONV_I32_H */
