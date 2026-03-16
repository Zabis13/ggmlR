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
    int                *tensor_map_ndims;  /* original ONNX ndims (for >4D axis mapping) */
    int64_t           (*tensor_map_onnx_ne)[ONNX_MAX_DIMS]; /* full ONNX shape (up to 8D) */
    int                 tensor_map_size;
    int                 tensor_map_cap;

    /* Deferred data for Shape op outputs (filled after sched alloc) */
    struct ggml_tensor *shape_tensor_ptrs[64];
    int64_t             shape_tensors_ne[64][ONNX_MAX_DIMS + 1]; /* [0]=ndims, [1..]=dims */
    int                 n_shape_tensors;

    /* Deferred data for ConstantOfShape + scalar constants (filled after sched alloc) */
    struct ggml_tensor *const_fill_ptrs[256];
    float               const_fill_vals[256];
    int                 n_const_fills;

    /* Deferred data for EyeLike (identity matrix, filled after sched alloc) */
    struct ggml_tensor *eye_fill_ptrs[64];
    int                 eye_fill_rows[64]; /* ggml ne[1] */
    int                 eye_fill_cols[64]; /* ggml ne[0] */
    int                 eye_fill_k[64];    /* diagonal offset */
    int                 n_eye_fills;

    /* Deferred strided Slice (step != 1): copy src→dst with stride after alloc */
    struct ggml_tensor *slice_fill_src[64];
    struct ggml_tensor *slice_fill_dst[64];
    int64_t             slice_fill_starts[64][4];  /* per-ggml-dim start offsets */
    int64_t             slice_fill_steps[64][4];   /* per-ggml-dim step values */
    int64_t             slice_fill_out_ne[64][4];  /* output ne per ggml dim */
    int                 slice_fill_ndims[64];      /* onnx ndims */
    int                 n_slice_fills;

    /* Compile-time known values for shape tensors (Shape, Constant, Slice, Concat outputs).
     * Used by Reshape/Expand/etc. to determine target shape at graph build time. */
    char              (*cval_keys)[ONNX_MAX_NAME];
    int64_t           (*cval_data)[ONNX_MAX_DIMS]; /* values (not dims!) */
    int                *cval_lens;                  /* number of values */
    int                 cval_size;
    int                 cval_cap;

    int                 is_allocated;   /* 1 after first sched_alloc_and_load */
} onnx_ggml_ctx_t;

/* Build ggml graph from parsed ONNX model.
 * device: "vulkan" or "cpu" (NULL defaults to vulkan if available, else cpu).
 * Returns NULL on failure. */
onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device, int n_threads);

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
