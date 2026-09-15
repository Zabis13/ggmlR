/* qconv_i32.h — QLinearConv with an exact integer accumulator.
 *
 * See qconv_i32.c for why this exists and what it is scoped to.
 */

#ifndef QCONV_I32_H
#define QCONV_I32_H

#include "../ggml.h"
#include "../ggml-backend.h"   /* buffer_is_host: this kernel reads host memory */

#include <stdint.h>

/* Upper bound on a per-output-channel table, used only to size the scratch
 * buffers the graph builder counts entries into. The kernel itself no longer
 * has a fixed limit: it reads the tables straight out of their tensors. */
#define QCONV_I32_MAX_CHANNELS 4096

/* GGML_OP_QCONV_I32 on the host: dst = requantise(qconv(src[0], src[1])).
 *
 * Scalars come from op_params and the per-output-channel tables from
 * src[2..4], so nothing is copied per node and nothing can dangle -- the
 * graph already owns every value this reads. See ggml_qconv_i32() in ggml.h
 * for the operand layout, and qconv_i32.c for why the arithmetic saturates. */
void qconv_i32_compute(struct ggml_tensor *dst, int ith, int nth);

#endif /* QCONV_I32_H */
