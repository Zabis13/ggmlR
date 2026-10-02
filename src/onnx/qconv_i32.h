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
 * for the operand layout, and qconv_i32.c for why the arithmetic saturates.
 *
 * wdata/wsize is the CPU plan's shared work buffer, sized by
 * qconv_i32_work_size(); barrier(barrier_ctx) synchronises all nth threads.
 * Every thread must make the call -- the kernel passes the barrier once, on
 * every path that gets that far, whether or not the thread has rows to do.
 * A buffer smaller than qconv_i32_work_size() selects the direct kernel. */
void qconv_i32_compute(struct ggml_tensor *dst, int ith, int nth,
                       void *wdata, size_t wsize,
                       void (*barrier)(void *), void *barrier_ctx);

/* Bytes of work buffer the im2col path needs for this node at n_threads.
 * The CPU plan calls it, so plan and kernel cannot disagree on the layout. */
size_t qconv_i32_work_size(const struct ggml_tensor *dst, int n_threads);

#endif /* QCONV_I32_H */
