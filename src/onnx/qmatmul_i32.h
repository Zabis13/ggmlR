/* qmatmul_i32.h — QLinearMatMul with an exact integer accumulator.
 *
 * See qmatmul_i32.c for why this exists and what it is scoped to.
 */

#ifndef QMATMUL_I32_H
#define QMATMUL_I32_H

#include "../ggml.h"
#include "../ggml-backend.h"   /* buffer_is_host: this kernel reads host memory */

#include <stdint.h>

/* Upper bound on a per-column table, used only to size the scratch buffers the
 * graph builder reads the initializers into. The kernel itself has no fixed
 * limit: it reads the tables straight out of their tensors. */
#define QMATMUL_I32_MAX_COLS 4096

/* GGML_OP_QMATMUL_I32 on the host: dst = requantise(src[0] x src[1]).
 *
 * Scalars come from op_params and the per-column tables from src[2..3], so
 * nothing is copied per node and nothing has to be freed per run -- the graph
 * owns every value this reads. See ggml_qmatmul_i32() in ggml.h for the operand
 * layout, and qconv_i32.c for why the arithmetic saturates.
 *
 * The Vulkan path is not here: on a Vulkan model the op is a GPU node
 * (ggml_vk_qmatmul_i32), and GGMLR_ONNX_GPU_QMATMUL=0 makes that backend
 * decline it so this kernel runs instead. */
void qmatmul_i32_compute(struct ggml_tensor *dst, int ith, int nth);

#endif /* QMATMUL_I32_H */
