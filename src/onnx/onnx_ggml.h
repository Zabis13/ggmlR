/* onnx_ggml.h — Map ONNX graph to ggml computation graph
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#ifndef ONNX_GGML_H
#define ONNX_GGML_H

#include "onnx_loader.h"
#include "../ggml.h"
#include "../ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── ONNX→ggml model context ───────────────────────────────────── */

typedef struct {
    onnx_model_t       *onnx;        /* parsed ONNX model (owns mmap) */
    struct ggml_context *ctx;         /* ggml context for tensors */
    struct ggml_cgraph  *graph;       /* computation graph */

    /* Scheduler with CPU fallback for unsupported Vulkan ops */
    ggml_backend_sched_t sched;       /* scheduler (owns buffer allocation) */
    ggml_backend_t       backend_gpu; /* Vulkan backend (NULL if CPU-only) */
    ggml_backend_t       backend_cpu; /* CPU backend (always present) */

    /* Name → ggml_tensor lookup for wiring nodes */
    struct ggml_tensor **tensor_map_vals;
    char              (*tensor_map_keys)[ONNX_MAX_NAME];
    int                 tensor_map_size;
    int                 tensor_map_cap;
} onnx_ggml_ctx_t;

/* Build ggml graph from parsed ONNX model.
 * device: "vulkan" or "cpu" (NULL defaults to vulkan if available, else cpu).
 * Returns NULL on failure. */
onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device);

/* Run inference. input_data/input_names: arrays of length n_inputs.
 * Each input_data[i] is a flat float array matching the model input shape.
 * Returns 0 on success. */
int onnx_ggml_run(onnx_ggml_ctx_t *ctx,
                  const char **input_names, const float **input_data,
                  int n_inputs);

/* Get output tensor by index. */
struct ggml_tensor *onnx_ggml_output(onnx_ggml_ctx_t *ctx, int index);

/* Free everything. */
void onnx_ggml_free(onnx_ggml_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_GGML_H */
