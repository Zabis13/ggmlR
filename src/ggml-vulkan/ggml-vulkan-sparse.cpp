// Sparse single-cell transforms — direct Vulkan dispatch (not a ggml-graph op).
//
// This file is #included into ggml-vulkan.cpp as one more part of that single
// translation unit (so all the `static` Vulkan helpers are visible). It exposes
// one extern "C" entry point, ggml_vk_sparse_lognorm_run(), which the R bridge
// calls.
//
// Rationale: Seurat's LogNormalize maps counts[i,j] -> log1p(counts/colSum*sf).
// Because log1p(0) = 0, the transform never creates new non-zeros, so it is a
// pure elementwise map over the dgCMatrix's stored values @x — no densification.
// That is the whole point of this path: on the full single-cell matrix the dense
// form is tens of GB, but @x is ~9% of that, so keeping it sparse both fits in
// memory and skips uploading 91% zeros across PCIe. NOTE: this is about removing
// the densify/OOM ceiling, not about beating the CPU on wall time — LogNormalize
// is memory-bound O(nnz), so parity with Seurat's sparse CPU path is the ceiling.

// One dispatch over the nnz stored values. vals is updated in place. The host
// (R) supplies the per-column factor (scale_factor / colSum, computed cheaply
// with Matrix::colSums) and the per-nnz column index (from the CSC @p pointers),
// so the shader needs no reduction and no binary search. Writes are race-free:
// each thread owns a distinct vals[k] slot (unlike UMAP's shared vertices), so
// no atomics or barriers are needed. The prototype comes from ggml-vulkan.h.
bool ggml_vk_sparse_lognorm_run(
        ggml_backend_t backend,
        float * vals,                  // nnz stored counts, in/out (@x of dgCMatrix)
        const float * factor,          // n_cols floats: scale_factor / colSum[col]
        const unsigned int * col_of_nnz, // nnz uints: 0-based column of each value
        unsigned int nnz, unsigned int n_cols) {

    if (!ggml_backend_is_vk(backend)) {
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (nnz == 0 || n_cols == 0) {
        return true;  // nothing to do (all-zero matrix)
    }

    const size_t vals_bytes   = (size_t)nnz    * sizeof(float);
    const size_t factor_bytes = (size_t)n_cols * sizeof(float);
    const size_t col_bytes    = (size_t)nnz    * sizeof(uint32_t);

    // --- allocate device-local SSBOs and upload the inputs ---
    vk_buffer d_vals   = ggml_vk_create_buffer_device(device, vals_bytes);
    vk_buffer d_factor = ggml_vk_create_buffer_device(device, factor_bytes);
    vk_buffer d_col    = ggml_vk_create_buffer_device(device, col_bytes);

    ggml_vk_buffer_write(d_vals,   0, vals,       vals_bytes);
    ggml_vk_buffer_write(d_factor, 0, factor,     factor_bytes);
    ggml_vk_buffer_write(d_col,    0, col_of_nnz, col_bytes);

    // Compile the pipeline and (on non-push drivers) allocate one descriptor set
    // for the single dispatch — same lazy-compile guard as the umap/pairwise paths.
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_sparse_lognorm, 1);

    // --- record one command buffer: a single 1D dispatch over nnz ---
    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    vk_op_sparse_lognorm_push_constants pc{};
    pc.nnz = nnz;

    const std::array<uint32_t, 3> elements = { nnz, 1, 1 };

    ggml_vk_dispatch_pipeline(
        ctx, subctx, ctx->device->pipeline_sparse_lognorm,
        { vk_subbuffer{ d_vals,   0, vals_bytes   },
          vk_subbuffer{ d_factor, 0, factor_bytes },
          vk_subbuffer{ d_col,    0, col_bytes    } },
        pc, elements);

    ggml_vk_ctx_end(subctx);

    // --- submit and wait ---
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    // --- read the transformed values back ---
    ggml_vk_buffer_read(d_vals, 0, vals, vals_bytes);

    // --- free the SSBOs ---
    ggml_vk_destroy_buffer(d_vals);
    ggml_vk_destroy_buffer(d_factor);
    ggml_vk_destroy_buffer(d_col);

    return true;
}
