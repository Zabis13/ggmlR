/* qmatmul_i32.h — QLinearMatMul with an exact integer accumulator.
 *
 * See qmatmul_i32.c for why this exists and what it is scoped to.
 */

#ifndef QMATMUL_I32_H
#define QMATMUL_I32_H

#include "../ggml.h"
#include "../ggml-backend.h"   /* buffer_is_host: this kernel reads host memory */

#include <stdint.h>

/* Per-column tables are bounded the same way the conv path bounds its per
 * channel ones: a wider matmul falls back rather than overrun. */
#define QMATMUL_I32_MAX_COLS 4096

/* Scalars and per-column tables the kernel needs, copied by VALUE at graph
 * build time.
 *
 * Copied rather than referenced, for the same reason as qconv_i32_params_t:
 * a tensor pointer captured at build time can dangle when the op runs,
 * because segmented execution resets and reallocates buffers in between. */
typedef struct {
    float   a_scale;
    float   y_scale;
    int32_t a_zp;
    int32_t y_zp;

    /* b_scale and b_zp are per output COLUMN when the exporter quantised the
     * weight per channel, exactly as w_scale/w_zp are per output channel in
     * the conv path.  n_* == 1 means one shared value. */
    int     n_b_scale;
    float   b_scale[QMATMUL_I32_MAX_COLS];
    int     n_b_zp;
    int32_t b_zp[QMATMUL_I32_MAX_COLS];

    float   out_lo, out_hi;                   /* saturation, from the zp dtype */
} qmatmul_i32_params_t;

/* ggml_map_custom3 kernel: dst = requantise(a_i32 x b_i32).
 * `a` carries the output shape only; the A matrix is `b` and B is `c`. */
void qmatmul_i32_cpu(struct ggml_tensor *dst,
                     const struct ggml_tensor *a,
                     const struct ggml_tensor *b,
                     const struct ggml_tensor *c,
                     int ith, int nth, void *userdata);

#endif /* QMATMUL_I32_H */
