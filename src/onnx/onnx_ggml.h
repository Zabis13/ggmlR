/* onnx_ggml.h — Map ONNX graph to ggml computation graph
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#ifndef ONNX_GGML_H
#define ONNX_GGML_H

#include "onnx_loader.h"
#include "rel_pos_bias.h"
#include "../ggml.h"
#include "../ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of deferred fill entries (Shape, Const, NonZero, etc.).
 * Transformer models with many attention heads can exceed 64 easily. */
#define ONNX_MAX_DEFERRED 512

/* ── ONNX→ggml model context ───────────────────────────────────── */

typedef struct {
    onnx_model_t       *onnx;        /* parsed ONNX model (owns mmap) */
    struct ggml_context *ctx;         /* ggml context for graph + compute tensors */
    struct ggml_context *ctx_weight;  /* ggml context for weight tensors (separate lifetime) */
    struct ggml_cgraph  *graph;       /* computation graph */

    /* Weight buffer — allocated once, never touched by sched */
    ggml_backend_buffer_t weight_buf; /* GPU (or CPU) buffer holding all weights */

    /* Host-visible pinned staging buffer for fast CPU→GPU input transfer.
     * Data is memcpy'd here, then ggml_backend_tensor_set detects pinned src
     * and does direct DMA (no intermediate staging copy). */
    ggml_backend_buffer_t pinned_buf; /* pinned staging buffer (NULL if unavailable) */
    void                 *pinned_ptr; /* mapped pointer into pinned_buf */
    size_t                pinned_size;/* allocated size in bytes */

    /* Scheduler with CPU fallback for unsupported Vulkan ops */
    ggml_backend_sched_t sched;       /* scheduler (owns compute buffer allocation) */
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
    struct ggml_tensor *shape_tensor_ptrs[ONNX_MAX_DEFERRED];
    int64_t             shape_tensors_ne[ONNX_MAX_DEFERRED][ONNX_MAX_DIMS + 1]; /* [0]=ndims, [1..]=dims */
    int                 n_shape_tensors;

    /* Deferred data for ConstantOfShape + scalar constants (filled after sched alloc) */
    struct ggml_tensor *const_fill_ptrs[ONNX_MAX_DEFERRED];
    float               const_fill_vals[ONNX_MAX_DEFERRED];
    int                 n_const_fills;

    /* Deferred data for EyeLike (identity matrix, filled after sched alloc) */
    struct ggml_tensor *eye_fill_ptrs[ONNX_MAX_DEFERRED];
    int                 eye_fill_rows[ONNX_MAX_DEFERRED]; /* ggml ne[1] */
    int                 eye_fill_cols[ONNX_MAX_DEFERRED]; /* ggml ne[0] */
    int                 eye_fill_k[ONNX_MAX_DEFERRED];    /* diagonal offset */
    int                 n_eye_fills;

    /* Deferred strided Slice (step != 1): copy src→dst with stride after alloc */
    struct ggml_tensor *slice_fill_src[ONNX_MAX_DEFERRED];
    struct ggml_tensor *slice_fill_dst[ONNX_MAX_DEFERRED];
    int64_t             slice_fill_starts[ONNX_MAX_DEFERRED][GGML_MAX_DIMS];  /* per-ggml-dim start offsets */
    int64_t             slice_fill_steps[ONNX_MAX_DEFERRED][GGML_MAX_DIMS];   /* per-ggml-dim step values */
    int64_t             slice_fill_out_ne[ONNX_MAX_DEFERRED][GGML_MAX_DIMS];  /* output ne per ggml dim */
    int                 slice_fill_ndims[ONNX_MAX_DEFERRED];      /* onnx ndims */
    int                 n_slice_fills;

    /* Deferred NonZero (filled after sched alloc: read input, write indices of non-zero elems) */
    struct ggml_tensor *nonzero_fill_src[ONNX_MAX_DEFERRED];   /* input tensor */
    struct ggml_tensor *nonzero_fill_dst[ONNX_MAX_DEFERRED];   /* output tensor [n_dims_input, nnz] in ggml layout */
    int                 nonzero_fill_ndims[ONNX_MAX_DEFERRED]; /* ONNX ndims of input */
    int                 n_nonzero_fills;

    /* Compile-time known values for shape tensors (Shape, Constant, Slice, Concat outputs).
     * Used by Reshape/Expand/etc. to determine target shape at graph build time. */
    char              (*cval_keys)[ONNX_MAX_NAME];
    int64_t           (*cval_data)[ONNX_MAX_DIMS]; /* values (not dims!) */
    int                *cval_lens;                  /* number of values */
    int                 cval_size;
    int                 cval_cap;

    int                 is_allocated;   /* 1 after first sched alloc + deferred fill */

    /* FP16 inference mode: 0 = F32 (default), 1 = F16 for large weights */
    int                 model_dtype;    /* GGML_TYPE_F32 or GGML_TYPE_F16 */

    /* RelPosBias2D fused op blocks (BoTNet pos_embed subgraphs) */
    #define ONNX_MAX_POS_EMBED 8
    struct {
        char   x_input_name[ONNX_MAX_NAME];   /* x tensor before first Reshape */
        char   wh_name[ONNX_MAX_NAME];         /* W_h initializer name */
        char   ww_name[ONNX_MAX_NAME];         /* W_w initializer name */
        char   output_name[ONNX_MAX_NAME];     /* final Reshape output name */
        int    first_node_idx;                  /* index of first node in block */
        int    last_node_idx;                   /* index of last node (final Reshape) */
        rel_pos_bias_params_t params;           /* H, W, B, C, rel_h, rel_w */
    } pos_embed_blocks[ONNX_MAX_POS_EMBED];
    int n_pos_embed_blocks;

    /* Storage for rel_pos_bias params (must outlive graph compute) */
    rel_pos_bias_params_t *pos_embed_params;    /* malloc'd array, freed in onnx_ggml_free */
} onnx_ggml_ctx_t;

/* Minimum number of elements for a weight tensor to be stored in FP16.
 * Smaller tensors (bias, scalars, BN params) stay F32 for numerical stability. */
#define ONNX_FP16_MIN_ELEMENTS 256

/* Build ggml graph from parsed ONNX model.
 * device: "vulkan" or "cpu" (NULL defaults to vulkan if available, else cpu).
 * Returns NULL on failure. */
/* model_dtype: GGML_TYPE_F32 (default) or GGML_TYPE_F16 for half-precision weights. */
onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device, int n_threads,
                                  enum ggml_type model_dtype);

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
