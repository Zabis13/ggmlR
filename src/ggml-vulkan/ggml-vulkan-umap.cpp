// UMAP SGD layout optimisation — direct Vulkan dispatch (not a ggml-graph op).
//
// This file is #included into ggml-vulkan.cpp as one more part of that single
// translation unit (so all the `static` Vulkan helpers are visible). It exposes
// one extern "C" entry point, ggml_vk_umap_sgd_run(), which the R bridge calls.
//
// The whole optimisation runs in one command buffer: the three SSBOs (coords,
// edges, weights) are uploaded once, then the umap_sgd pipeline is dispatched
// once per epoch with a per-epoch learning rate and RNG seed pushed as
// constants, with a buffer barrier between epochs so each epoch sees the
// previous epoch's coordinate writes. Writes inside an epoch are Hogwild (the
// shader does not use atomics); the per-epoch barrier is the only ordering.

// One epoch per dispatch. coords is updated in place; the host decays alpha and
// advances the seed between epochs exactly as the CPU reference does. The
// prototype (with GGML_BACKEND_API / extern "C") comes from ggml-vulkan.h.
bool ggml_vk_umap_sgd_run(
        ggml_backend_t backend,
        float * coords,                // n*2 floats, in/out ([x,y] per vertex)
        const unsigned int * edges,    // ne*2 uints [from,to,...]
        const float * weights,         // ne floats (reserved; uploaded anyway)
        unsigned int n, unsigned int ne,
        unsigned int n_epochs, unsigned int n_neg,
        float a, float b, float alpha0, float gamma,
        unsigned int base_seed) {

    if (!ggml_backend_is_vk(backend)) {
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (n == 0 || ne == 0 || n_epochs == 0) {
        return true;  // nothing to do
    }

    const size_t coords_bytes  = (size_t)n  * 2 * sizeof(float);
    const size_t edges_bytes   = (size_t)ne * 2 * sizeof(uint32_t);
    const size_t weights_bytes = (size_t)ne * sizeof(float);

    // --- allocate device-local SSBOs and upload the inputs ---
    vk_buffer d_coords  = ggml_vk_create_buffer_device(device, coords_bytes);
    vk_buffer d_edges   = ggml_vk_create_buffer_device(device, edges_bytes);
    vk_buffer d_weights = ggml_vk_create_buffer_device(device, weights_bytes);

    ggml_vk_buffer_write(d_coords,  0, coords,  coords_bytes);
    ggml_vk_buffer_write(d_edges,   0, edges,   edges_bytes);
    ggml_vk_buffer_write(d_weights, 0, weights, weights_bytes);

    // Ensure the pipeline is compiled and (on non-push-descriptor drivers) that
    // enough descriptor sets are allocated. The graph path normally does this;
    // a direct dispatch must do it itself or bindPipeline hits a null handle
    // (segfault) and ctx->descriptor_sets[idx] runs off the end. One dispatch
    // per epoch -> request n_epochs sets. Reset the counters first, as the graph
    // does at the top of each compute pass.
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_umap_sgd, n_epochs);

    // --- record one command buffer: an epoch loop of dispatches ---
    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    const std::array<uint32_t, 3> elements = { ne, 1, 1 };

    for (uint32_t epoch = 0; epoch < n_epochs; ++epoch) {
        vk_op_umap_sgd_push_constants pc{};
        pc.n     = n;
        pc.ne    = ne;
        pc.n_neg = n_neg;
        pc.seed  = base_seed + epoch;                 // host advances seed/epoch
        pc.alpha = alpha0 * (1.0f - (float)epoch / (float)n_epochs);
        pc.a     = a;
        pc.b     = b;
        pc.gamma = gamma;

        ggml_vk_dispatch_pipeline(
            ctx, subctx, ctx->device->pipeline_umap_sgd,
            { vk_subbuffer{ d_coords,  0, coords_bytes  },
              vk_subbuffer{ d_edges,   0, edges_bytes   },
              vk_subbuffer{ d_weights, 0, weights_bytes } },
            pc, elements);

        // barrier so the next epoch observes this epoch's coordinate writes
        if (epoch + 1 < n_epochs) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    ggml_vk_ctx_end(subctx);

    // --- submit and wait ---
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    // --- read the optimised coordinates back ---
    ggml_vk_buffer_read(d_coords, 0, coords, coords_bytes);

    // --- free the SSBOs ---
    ggml_vk_destroy_buffer(d_coords);
    ggml_vk_destroy_buffer(d_edges);
    ggml_vk_destroy_buffer(d_weights);

    return true;
}

// Pairwise squared Euclidean distance matrix — direct Vulkan dispatch.
//
// The shader is tiled 32x32 with shared-memory staging: each thread still owns
// one output cell D2[i,j] = sum_d (x[i,d] - x[j,d])^2 accumulated in f32 (the
// honest path that avoids mul_mat's f16 accumulation), but the 32x32 workgroup
// co-loads each X value once per tile instead of once per column, and writes
// D2 symmetrically. X is uploaded once, D2 is computed in a single dispatch over
// the full n*n grid, then read back. The caller takes sqrt() for distance.
bool ggml_vk_pairwise_dist_run(
        ggml_backend_t backend,
        const float * x,               // n * dims floats, row-major
        float * d2,                    // n * n floats, row-major (out)
        unsigned int n, unsigned int dims) {

    if (!ggml_backend_is_vk(backend)) {
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (n == 0 || dims == 0) {
        return true;  // nothing to do
    }

    const size_t x_bytes  = (size_t)n * dims * sizeof(float);
    const size_t d2_bytes = (size_t)n * n    * sizeof(float);

    // --- allocate device-local SSBOs and upload X ---
    vk_buffer d_x  = ggml_vk_create_buffer_device(device, x_bytes);
    vk_buffer d_d2 = ggml_vk_create_buffer_device(device, d2_bytes);
    ggml_vk_buffer_write(d_x, 0, x, x_bytes);

    // Compile the pipeline and (on non-push drivers) allocate one descriptor set
    // for the single dispatch — same lazy-compile guard as the SGD path above.
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_pairwise_dist, 1);

    // --- record one command buffer: a single 2D dispatch ---
    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    vk_op_pairwise_dist_push_constants pc{};
    pc.n    = n;
    pc.dims = dims;

    // elements = total threads per axis; wg_denoms (32,32) come from the shader's
    // 32x32 tile, so the dispatch rounds n up to whole tiles on each axis.
    const std::array<uint32_t, 3> elements = { n, n, 1 };

    ggml_vk_dispatch_pipeline(
        ctx, subctx, ctx->device->pipeline_pairwise_dist,
        { vk_subbuffer{ d_x,  0, x_bytes  },
          vk_subbuffer{ d_d2, 0, d2_bytes } },
        pc, elements);

    ggml_vk_ctx_end(subctx);

    // --- submit and wait ---
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    // --- read the distance matrix back ---
    ggml_vk_buffer_read(d_d2, 0, d2, d2_bytes);

    // --- free the SSBOs ---
    ggml_vk_destroy_buffer(d_x);
    ggml_vk_destroy_buffer(d_d2);

    return true;
}
