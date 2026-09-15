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

    // ne (positive edges ~ n*k) can exceed maxComputeWorkGroupCount[0] (65535 on
    // NVIDIA; AMD is ~2^31) once ne > 65535*256 (~16.7M edges). Spread the edges
    // over a 2D grid: rows of row_stride edges stacked on Y, row_stride a multiple
    // of local_size_x (256) so the linear edge index e = gid.y*row_stride + gid.x
    // matches the 1D layout exactly — critical because e seeds the per-edge RNG,
    // which must stay bit-identical to the CPU reference (see sc_umap.R).
    const uint32_t local_x    = 256u;
    const uint32_t max_wg     = device->properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t wg_x_max   = std::min(max_wg, 65535u);
    const uint32_t row_stride = wg_x_max * local_x;               // edges per Y-row
    const uint32_t wg_x       = std::min(CEIL_DIV(ne, local_x), wg_x_max);
    const uint32_t wg_y       = CEIL_DIV(ne, row_stride);
    const std::array<uint32_t, 3> elements = { wg_x * local_x, wg_y, 1 };

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
        pc.epoch = epoch + 1;                         // shader's test is 1-based
        pc.row_stride = row_stride;

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

    // This path materialises the full n*n distance matrix, so it is memory-bound
    // (n*n*4 bytes: n=45k already needs 8 GB) long before the workgroup grid — one
    // tile-block per axis, ceil(n/32) — could reach maxComputeWorkGroupCount (65535
    // on NVIDIA at n~2.1M). Guard the axis limit anyway so an over-large n returns
    // false (the caller falls back / errors) instead of tripping the dispatch
    // GGML_ASSERT and killing the process silently. For large single-cell n use the
    // fused knn_tiled path, which never materialises n*n.
    const uint32_t max_wg = device->properties.limits.maxComputeWorkGroupCount[0];
    if (CEIL_DIV(n, 32u) > max_wg) {
        return false;
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

// Tiled fused k-NN — the honest-f32 GPU nearest-neighbour search that never
// materialises the n x n distance matrix (see vulkan-shaders/knn_tiled.comp).
// One workgroup per query row computes and selects that row's k nearest in a
// single dispatch; the only outputs are the n*k neighbour indices (0-based) and
// n*k Euclidean distances, sorted ascending per row. k must be <= the pipeline's
// top-k capacity (the K specialization constant, 32). The prototype (with
// GGML_BACKEND_API / extern "C") comes from ggml-vulkan.h.
bool ggml_vk_knn_tiled_run(
        ggml_backend_t backend,
        const float * x,               // n * dims floats, row-major
        unsigned int * knn_idx,        // n * k uints, row-major (out, 0-based rows)
        float * knn_dist,              // n * k floats, row-major (out, Euclidean)
        unsigned int n, unsigned int dims, unsigned int k) {

    if (!ggml_backend_is_vk(backend)) {
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (n == 0 || dims == 0 || k == 0) {
        return true;  // nothing to do
    }

    const size_t x_bytes   = (size_t)n * dims * sizeof(float);
    const size_t idx_bytes = (size_t)n * k    * sizeof(uint32_t);
    const size_t dst_bytes = (size_t)n * k    * sizeof(float);

    // --- allocate device-local SSBOs and upload X ---
    vk_buffer d_x   = ggml_vk_create_buffer_device(device, x_bytes);
    vk_buffer d_idx = ggml_vk_create_buffer_device(device, idx_bytes);
    vk_buffer d_dst = ggml_vk_create_buffer_device(device, dst_bytes);
    ggml_vk_buffer_write(d_x, 0, x, x_bytes);

    // Compile the pipeline and (on non-push drivers) allocate one descriptor set
    // for the single dispatch — same lazy-compile guard as the paths above.
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_knn_tiled, 1);

    // --- record one command buffer: a single 1D dispatch, one group per row ---
    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    vk_op_knn_tiled_push_constants pc{};
    pc.n    = n;
    pc.dims = dims;
    pc.k    = k;

    // wg_denoms for this pipeline are {1,1,1} (see create_pipeline), so elements
    // are workgroup counts directly — one workgroup per query row. A flat n on X
    // exceeds maxComputeWorkGroupCount[0] (65535 on NVIDIA; AMD is ~2^31, which is
    // why this only bit on the T4) once n > 65535. Stack the rows on a 2D grid:
    // X width = min(n, 65535), Y = ceil(n / X). The shader recovers the linear
    // row as gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x.
    const uint32_t max_wg = device->properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t wg_x   = std::min(n, std::min(max_wg, 65535u));
    const uint32_t wg_y   = CEIL_DIV(n, wg_x);
    const std::array<uint32_t, 3> elements = { wg_x, wg_y, 1 };

    ggml_vk_dispatch_pipeline(
        ctx, subctx, ctx->device->pipeline_knn_tiled,
        { vk_subbuffer{ d_x,   0, x_bytes   },
          vk_subbuffer{ d_idx, 0, idx_bytes },
          vk_subbuffer{ d_dst, 0, dst_bytes } },
        pc, elements);

    ggml_vk_ctx_end(subctx);

    // --- submit and wait ---
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    // --- read the neighbour indices and distances back ---
    ggml_vk_buffer_read(d_idx, 0, knn_idx,  idx_bytes);
    ggml_vk_buffer_read(d_dst, 0, knn_dist, dst_bytes);

    // --- free the SSBOs ---
    ggml_vk_destroy_buffer(d_x);
    ggml_vk_destroy_buffer(d_idx);
    ggml_vk_destroy_buffer(d_dst);

    return true;
}

// FP64 matmul (PoC): C[M,N] = A[M,K] * B[K,N] entirely in double, dispatched
// directly to matmul_f64.comp. Uploads/downloads double with NO float conversion
// (the whole point of the experiment). Returns false if the backend is not
// Vulkan or the device lacks fp64 (the pipeline was not created). The prototype
// (with GGML_BACKEND_API / extern "C") comes from ggml-vulkan.h.
bool ggml_vk_matmul_f64_run(
        ggml_backend_t backend,
        const double * a,              // M * K doubles, row-major
        const double * b,              // K * N doubles, row-major
        double * c,                    // M * N doubles, row-major (out)
        unsigned int M, unsigned int N, unsigned int K) {

    if (!ggml_backend_is_vk(backend)) {
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (M == 0 || N == 0 || K == 0) {
        return true;  // nothing to do
    }
    if (!device->pipeline_matmul_f64) {
        return false;  // device has no fp64 support -> pipeline never created
    }

    // The 16x16-tiled grid is ceil(N/16) x ceil(M/16) workgroups. Guard both axes
    // against the device limits (65535 per axis on NVIDIA, so N or M ~1.05M) so an
    // over-large matmul returns false instead of tripping the dispatch GGML_ASSERT
    // and killing the process silently.
    if (CEIL_DIV(N, 16u) > device->properties.limits.maxComputeWorkGroupCount[0] ||
        CEIL_DIV(M, 16u) > device->properties.limits.maxComputeWorkGroupCount[1]) {
        return false;
    }

    const size_t a_bytes = (size_t)M * K * sizeof(double);
    const size_t b_bytes = (size_t)K * N * sizeof(double);
    const size_t c_bytes = (size_t)M * N * sizeof(double);

    // --- allocate device-local SSBOs and upload A, B ---
    vk_buffer d_a = ggml_vk_create_buffer_device(device, a_bytes);
    vk_buffer d_b = ggml_vk_create_buffer_device(device, b_bytes);
    vk_buffer d_c = ggml_vk_create_buffer_device(device, c_bytes);
    ggml_vk_buffer_write(d_a, 0, a, a_bytes);
    ggml_vk_buffer_write(d_b, 0, b, b_bytes);

    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_f64, 1);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    vk_op_matmul_f64_push_constants pc{};
    pc.M = M;
    pc.N = N;
    pc.K = K;

    // wg_denoms (16,16) come from the shader's 16x16 tile, so the dispatch rounds
    // the N x M output up to whole tiles on each axis (x = columns, y = rows).
    const std::array<uint32_t, 3> elements = { N, M, 1 };

    ggml_vk_dispatch_pipeline(
        ctx, subctx, ctx->device->pipeline_matmul_f64,
        { vk_subbuffer{ d_a, 0, a_bytes },
          vk_subbuffer{ d_b, 0, b_bytes },
          vk_subbuffer{ d_c, 0, c_bytes } },
        pc, elements);

    ggml_vk_ctx_end(subctx);
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    ggml_vk_buffer_read(d_c, 0, c, c_bytes);

    ggml_vk_destroy_buffer(d_a);
    ggml_vk_destroy_buffer(d_b);
    ggml_vk_destroy_buffer(d_c);

    return true;
}

// RoiAlign (ONNX opset 10+) — direct Vulkan dispatch.
//
// One thread per output element over a flat 1-D grid; see
// vulkan-shaders/roi_align.comp for the sampling itself. The three inputs are
// uploaded, one dispatch covers the whole [ow, oh, C, num_rois] output, and the
// result is read straight back — there is no cross-thread state, so nothing
// needs staging or a second pass.
//
// ⚠️ Bit-exactness, not speed, is the acceptance criterion here. MaskRCNN-12-int8
// agrees with ONNX Runtime to the last bit on the CPU path, and this op is only
// allowed to take over once it reproduces that exactly; the shader is a
// line-for-line port of src/onnx/roi_align.c for that reason.
bool ggml_vk_roi_align_run(
        ggml_backend_t backend,
        const float * x, const float * rois, const float * batch_indices,
        float * dst,
        unsigned int W, unsigned int H, unsigned int C, unsigned int N,
        unsigned int num_rois, unsigned int ow, unsigned int oh,
        int sampling_ratio, unsigned int mode, float spatial_scale) {

    if (!ggml_backend_is_vk(backend)) {
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (W == 0 || H == 0 || C == 0 || N == 0 || num_rois == 0 || ow == 0 || oh == 0) {
        return true;  // nothing to do
    }

    const size_t total = (size_t)ow * oh * C * num_rois;

    // One workgroup per 256 output elements. NVIDIA caps a dispatch axis at
    // 65535 groups, and exceeding it aborts inside the dispatch rather than
    // failing here, so refuse and let the caller stay on the CPU kernel.
    const uint32_t max_wg = device->properties.limits.maxComputeWorkGroupCount[0];
    if (CEIL_DIV(total, (size_t)256) > (size_t)max_wg) {
        return false;
    }

    // ⚠️ The adaptive branch (sampling_ratio <= 0) takes its loop bound from
    // the ROI VALUES, not from any dimension, and nothing upstream bounds
    // them. An oversized or non-finite ROI makes the shader run millions of
    // taps per output element, which the driver's watchdog cannot wait out:
    // measured on MaskRCNN, that hung the amdgpu compute ring three times and
    // the third hang took the graphics ring with it (MODE1 reset, VRAM lost).
    //
    // The shader clamps too — this gate exists so the condition is DIAGNOSED
    // on the host rather than silently absorbed on the GPU. The values are
    // already here in `rois`, so checking costs one pass over 4*num_rois
    // floats. Falling back to the CPU kernel keeps the answer identical to
    // what that kernel would have produced (it applies the same clamp) while
    // leaving the GPU out of it entirely.
    //
    // Reaching this means the ROIs are malformed and the bug is upstream in
    // the box decode; the cap is a guard, not a tuning knob.
    //
    // The same number lives in ROI_ALIGN_MAX_SAMPLES (../onnx/roi_align.h) and
    // as a literal in roi_align.comp. Kept in step by hand, as the NMS gate
    // below does with MAX_CAND: this backend deliberately does not include the
    // ONNX headers.
    const int ROI_ALIGN_MAX_SAMPLES = 128;
    if (sampling_ratio <= 0) {
        const float cap = (float)ROI_ALIGN_MAX_SAMPLES;
        for (unsigned int i = 0; i < num_rois; i++) {
            const float rw = (rois[2 + 4 * i] - rois[0 + 4 * i]) * spatial_scale;
            const float rh = (rois[3 + 4 * i] - rois[1 + 4 * i]) * spatial_scale;
            // !(x <= cap) rather than (x > cap): true for NaN, which would
            // otherwise sail through and reach int(ceil()) as undefined.
            if (!(rh / (float)oh <= cap) || !(rw / (float)ow <= cap)) {
                GGML_LOG_INFO("%s: ROI %u is %.1fx%.1f, over the %d-sample cap "
                              "-- staying on the CPU kernel. Malformed ROIs "
                              "mean the box decode upstream is wrong.\n",
                              __func__, i, (double)rw, (double)rh,
                              ROI_ALIGN_MAX_SAMPLES);
                return false;
            }
        }
    }

    const size_t x_bytes    = (size_t)W * H * C * N * sizeof(float);
    const size_t rois_bytes = (size_t)4 * num_rois * sizeof(float);
    const size_t bi_bytes   = (size_t)num_rois * sizeof(float);
    const size_t dst_bytes  = total * sizeof(float);

    vk_buffer d_x    = ggml_vk_create_buffer_device(device, x_bytes);
    vk_buffer d_rois = ggml_vk_create_buffer_device(device, rois_bytes);
    vk_buffer d_bi   = ggml_vk_create_buffer_device(device, bi_bytes);
    vk_buffer d_dst  = ggml_vk_create_buffer_device(device, dst_bytes);

    ggml_vk_buffer_write(d_x,    0, x,             x_bytes);
    ggml_vk_buffer_write(d_rois, 0, rois,          rois_bytes);
    ggml_vk_buffer_write(d_bi,   0, batch_indices, bi_bytes);

    // Pipelines compile lazily: without this the bind below would run against
    // an empty handle and segfault. Requests one descriptor set for the one
    // dispatch (a no-op on push-descriptor drivers such as RADV).
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_roi_align, 1);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    vk_op_roi_align_push_constants pc{};
    pc.W              = W;
    pc.H              = H;
    pc.C              = C;
    pc.num_rois       = num_rois;
    pc.ow             = ow;
    pc.oh             = oh;
    pc.sampling_ratio = sampling_ratio;
    pc.mode           = mode;
    pc.spatial_scale  = spatial_scale;

    // elements = total threads; wg_denoms (256,1,1) come from the shader's
    // local_size, so the dispatch rounds up to whole workgroups and the
    // shader's own `gid >= total` test drops the tail.
    const std::array<uint32_t, 3> elements = { (uint32_t)total, 1, 1 };

    ggml_vk_dispatch_pipeline(
        ctx, subctx, ctx->device->pipeline_roi_align,
        { vk_subbuffer{ d_x,    0, x_bytes    },
          vk_subbuffer{ d_rois, 0, rois_bytes },
          vk_subbuffer{ d_bi,   0, bi_bytes   },
          vk_subbuffer{ d_dst,  0, dst_bytes  } },
        pc, elements);

    ggml_vk_ctx_end(subctx);
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    ggml_vk_buffer_read(d_dst, 0, dst, dst_bytes);

    ggml_vk_destroy_buffer(d_x);
    ggml_vk_destroy_buffer(d_rois);
    ggml_vk_destroy_buffer(d_bi);
    ggml_vk_destroy_buffer(d_dst);

    return true;
}

// Phase timing for the two direct-dispatch quant kernels (GGMLR_QI32_PROFILE=1).
//
// These dispatch outside the graph, so they create their own device buffers,
// upload every operand, compute, read the result back and destroy the buffers
// on EVERY call -- weights included, though those never change. That makes the
// transfer cost, not the arithmetic, the thing worth measuring: the question
// this answers is what share of a call is spent moving data rather than
// computing. Totals are printed at process exit, in one line per kernel.
//
// Deliberately not a sampling profiler: with a few hundred calls per inference
// the per-call clock cost is irrelevant next to a device buffer upload, and
// exact totals beat a statistical estimate for deciding whether caching the
// weight buffers is worth the complexity.
struct qi32_prof {
    double t_create = 0, t_upload = 0, t_dispatch = 0, t_read = 0;
    size_t bytes_up = 0, bytes_w = 0, calls = 0;
    /* Every path that returns before the timed section, so a zero call count
     * says WHICH guard turned the work away rather than only that it did. */
    size_t n_not_vk = 0, n_empty = 0, n_oversize = 0;
    const char * name;
    explicit qi32_prof(const char * n) : name(n) {}
    void report() {
        if (!calls) {
            if (n_not_vk || n_empty || n_oversize)
                fprintf(stderr, "[qi32] %-11s NO timed calls: not-vk=%zu empty=%zu "
                                "oversize=%zu\n", name, n_not_vk, n_empty, n_oversize);
            return;
        }
        const double tot = t_create + t_upload + t_dispatch + t_read;
        if (tot <= 0) return;
        fprintf(stderr,
            "[qi32] %-11s calls=%zu total=%.1f ms | create %.1f (%.0f%%) "
            "upload %.1f (%.0f%%) dispatch %.1f (%.0f%%) read %.1f (%.0f%%) "
            "| uploaded %.1f MB of which weights %.1f MB (%.0f%%)\n",
            name, calls, tot,
            t_create,   100.0 * t_create   / tot,
            t_upload,   100.0 * t_upload   / tot,
            t_dispatch, 100.0 * t_dispatch / tot,
            t_read,     100.0 * t_read     / tot,
            bytes_up / 1048576.0, bytes_w / 1048576.0,
            bytes_up ? 100.0 * (double)bytes_w / (double)bytes_up : 0.0);
    }
    /* Report every N calls rather than at process exit: R tears the session
     * down without running static destructors reliably, and a profile that
     * only prints on a clean exit is a profile that does not print. */
    void tick() {
        if ((calls % 64) == 0) report();
    }
};

static bool qi32_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("GGMLR_QI32_PROFILE");
        cached = (e && e[0] == '1' && e[1] == '\0') ? 1 : 0;
    }
    return cached != 0;
}

// Wall clock: these phases each block on the device, so CPU time would report
// the waiting as free. ggml_time_us rather than <chrono> because this file is
// #included into one translation unit with the rest of the Vulkan backend, and
// adding a header here adds it for all of them.
static double qi32_now_ms() {
    return (double)ggml_time_us() * 1e-3;
}

static qi32_prof g_qi32_matmul("qmatmul_i32");
static qi32_prof g_qi32_conv  ("qconv_i32");

#define QI32_TICK(var) const double var = prof ? qi32_now_ms() : 0.0
#define QI32_TOCK(var, field) do { if (prof) (field) += qi32_now_ms() - (var); } while (0)

// QLinearMatMul with an exact i32 accumulator (ONNX) — direct Vulkan dispatch.
//
// One thread per output element over a flat M*N grid; the K loop stays inside
// the thread because the int16 pair saturation it reproduces is order-dependent
// and cannot be split across a workgroup. See vulkan-shaders/qmatmul_i32.comp
// and src/onnx/qmatmul_i32.c — the shader is a line-for-line port of the latter,
// ORT's VPMADDUBSW precision loss included.
bool ggml_vk_qmatmul_i32_run(
        ggml_backend_t backend,
        const float * a, const float * b_mat,
        const float * b_scale, const int * b_zp,
        float * dst,
        unsigned int M, unsigned int N, unsigned int K,
        float a_scale, float y_scale, int a_zp, int y_zp,
        unsigned int n_b_scale, unsigned int n_b_zp, unsigned int b_zp_any,
        float out_lo, float out_hi) {

    if (!ggml_backend_is_vk(backend)) {
        if (qi32_profile_enabled()) g_qi32_matmul.n_not_vk++;
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (M == 0 || N == 0 || K == 0) {
        if (qi32_profile_enabled()) g_qi32_matmul.n_empty++;
        return true;  // nothing to do
    }

    const size_t total = (size_t)M * N;

    // A driver caps each dispatch axis (65535 on NVIDIA) and overshooting
    // aborts inside the dispatch rather than failing here, so refuse and let
    // the caller stay on the CPU kernel. The dispatch is 2D -- N across in
    // blocks of 16, M down in blocks of 16, matching the shader's block tile
    // -- so each axis is checked against its own limit, not a flat count.
    const uint32_t max_wg_x = device->properties.limits.maxComputeWorkGroupCount[0];
    const uint32_t max_wg_y = device->properties.limits.maxComputeWorkGroupCount[1];
    if (CEIL_DIV((size_t)N, (size_t)16) > (size_t)max_wg_x ||
        CEIL_DIV((size_t)M, (size_t)16) > (size_t)max_wg_y) {
        if (qi32_profile_enabled()) g_qi32_matmul.n_oversize++;
        return false;
    }

    // ⚠️ The workgroup test above bounds the number of THREADS, not the amount
    // of WORK. Each thread walks K from end to end -- up to three times, for
    // sum_a, sum_pairs and sum_b -- so the real cost is M*N*K, and that product
    // is unbounded no matter how the first test comes out.
    //
    // MaskRCNN's box head reaches this with M=1000, N=1024, K=12544: 4000
    // workgroups, comfortably inside the limit, but 1.28e10 multiply-adds and
    // roughly 100 GB of global traffic in ONE dispatch, because this kernel
    // gives every thread its own row of A and column of B and shares nothing.
    // That does not finish inside the driver's watchdog. Measured on
    // RX 9070 / RADV: the compute ring times out (comp_1.x.x, "signaled seq=N,
    // emitted seq=N+1"), and once enough of those pile up the graphics ring
    // goes with them -- MODE1 reset, "VRAM is lost", the session is gone. The
    // process sees only vk::DeviceLostError, naming nothing.
    //
    // ⚠️ This cap is a GUARD, NOT A VERDICT ON THE SHAPE, and the shader has
    // since been tiled so that the shape which used to hang no longer does.
    // Measured on RX 9070, MaskRCNN's 1.28e10 node:
    //
    //   naive (one thread per output)     hung the ring    (~96 GB of traffic)
    //   1D tile, A staged only            hung the ring    (~48 GB)
    //   2D tile, A and B staged (current)      284.74 ms   ( ~6 GB)
    //
    // versus 14117.71 ms for the same node on the CPU kernel -- a 50x gain,
    // and 16432 -> 1662 ms over the whole model. Falling back is therefore the
    // wrong answer whenever the shader can carry the work, which is why this
    // cap sits well above the shapes that occur rather than just above zero.
    //
    // 5e10 is ~4x the node that now runs in 285 ms, so the cap stands at
    // roughly a second of dispatch -- well inside the watchdog, while still
    // refusing a shape orders of magnitude past anything measured.
    // GGMLR_QI32_MAX_WORK overrides it without a rebuild; 0 lifts it
    // entirely, which is how the tiling was measured against the 1.28e10 node.
    static size_t QI32_MAX_WORK = (size_t)-1;
    if (QI32_MAX_WORK == (size_t)-1) {
        const char * e = getenv("GGMLR_QI32_MAX_WORK");
        QI32_MAX_WORK = e ? (size_t)atof(e) : (size_t)5e10;
    }
    if (QI32_MAX_WORK && total > QI32_MAX_WORK / (size_t)K) {  // M*N*K, no overflow
        if (qi32_profile_enabled()) g_qi32_matmul.n_oversize++;
        GGML_LOG_INFO("%s: M=%u N=%u K=%u is %.2g ops, over the %.2g cap -- "
                      "using the CPU kernel, which is far slower but cannot "
                      "trip the driver's watchdog. Raise GGMLR_QI32_MAX_WORK "
                      "if this shape is expected.\n",
                      __func__, M, N, K, (double)M * N * K,
                      (double)QI32_MAX_WORK);
        return false;
    }

    const size_t a_bytes  = (size_t)M * K * sizeof(float);
    const size_t b_bytes  = (size_t)N * K * sizeof(float);
    const size_t bs_bytes = (size_t)(n_b_scale > 1 ? n_b_scale : 1) * sizeof(float);
    const size_t bz_bytes = (size_t)(n_b_zp    > 1 ? n_b_zp    : 1) * sizeof(int32_t);
    const size_t d_bytes  = total * sizeof(float);

    const bool prof = qi32_profile_enabled();
    QI32_TICK(t0);
    vk_buffer d_a   = ggml_vk_create_buffer_device(device, a_bytes);
    vk_buffer d_b   = ggml_vk_create_buffer_device(device, b_bytes);
    vk_buffer d_bs  = ggml_vk_create_buffer_device(device, bs_bytes);
    vk_buffer d_bz  = ggml_vk_create_buffer_device(device, bz_bytes);
    vk_buffer d_dst = ggml_vk_create_buffer_device(device, d_bytes);
    QI32_TOCK(t0, g_qi32_matmul.t_create);

    QI32_TICK(t1);
    ggml_vk_buffer_write(d_a,  0, a,       a_bytes);
    ggml_vk_buffer_write(d_b,  0, b_mat,   b_bytes);
    ggml_vk_buffer_write(d_bs, 0, b_scale, bs_bytes);
    ggml_vk_buffer_write(d_bz, 0, b_zp,    bz_bytes);
    QI32_TOCK(t1, g_qi32_matmul.t_upload);
    if (prof) {
        /* B is the weight matrix here, with its scale and zero point. */
        g_qi32_matmul.bytes_w  += b_bytes + bs_bytes + bz_bytes;
        g_qi32_matmul.bytes_up += a_bytes + b_bytes + bs_bytes + bz_bytes;
        g_qi32_matmul.calls++;
        g_qi32_matmul.tick();
    }

    // Pipelines compile lazily; without this the bind runs against an empty
    // handle and segfaults.
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_qmatmul_i32, 1);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    vk_op_qmatmul_i32_push_constants pc{};
    pc.M         = M;
    pc.N         = N;
    pc.K         = K;
    pc.a_scale   = a_scale;
    pc.y_scale   = y_scale;
    pc.a_zp      = a_zp;
    pc.y_zp      = y_zp;
    pc.n_b_scale = n_b_scale;
    pc.n_b_zp    = n_b_zp;
    pc.b_zp_any  = b_zp_any;
    pc.out_lo    = out_lo;
    pc.out_hi    = out_hi;

    // 2D: N across, M down, matching the shader's 16x16 block tile and the
    // pipeline's {16,16,1} wg_denoms. A flat element count would hand the
    // shader one long row of blocks and every m would come out zero.
    const std::array<uint32_t, 3> elements = { N, M, 1 };

    ggml_vk_dispatch_pipeline(
        ctx, subctx, ctx->device->pipeline_qmatmul_i32,
        { vk_subbuffer{ d_a,   0, a_bytes  },
          vk_subbuffer{ d_b,   0, b_bytes  },
          vk_subbuffer{ d_bs,  0, bs_bytes },
          vk_subbuffer{ d_bz,  0, bz_bytes },
          vk_subbuffer{ d_dst, 0, d_bytes  } },
        pc, elements);

    QI32_TICK(t2);
    ggml_vk_ctx_end(subctx);
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);
    QI32_TOCK(t2, g_qi32_matmul.t_dispatch);

    QI32_TICK(t3);
    ggml_vk_buffer_read(d_dst, 0, dst, d_bytes);

    ggml_vk_destroy_buffer(d_a);
    ggml_vk_destroy_buffer(d_b);
    ggml_vk_destroy_buffer(d_bs);
    ggml_vk_destroy_buffer(d_bz);
    ggml_vk_destroy_buffer(d_dst);
    QI32_TOCK(t3, g_qi32_matmul.t_read);

    return true;
}


// NonMaxSuppression (ONNX) — direct Vulkan dispatch.
//
// One workgroup per (batch, class) pair; see vulkan-shaders/nms.comp for why
// the work splits that way (filter and sort are parallel, selection is not)
// and src/onnx/nms.c for the arithmetic it mirrors.
//
// The shader sorts candidates in shared memory of a fixed size, so a class with
// more candidates than that cannot be served: this returns false and the caller
// runs the CPU kernel for the whole node. num_boxes is the bound that matters
// because every box is a potential candidate when no score threshold is given.
bool ggml_vk_nms_run(
        ggml_backend_t backend,
        const float * boxes, const float * scores,
        int * sel_idx, int * sel_cnt,
        unsigned int num_boxes, unsigned int num_classes, unsigned int N,
        unsigned int max_per_pair, int max_output,
        float iou_thresh, float score_thresh, unsigned int have_score_thresh,
        unsigned int center_point_box) {

    if (!ggml_backend_is_vk(backend)) {
        return false;
    }
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_device& device = ctx->device;

    if (num_boxes == 0 || num_classes == 0 || N == 0 || max_per_pair == 0) {
        return true;  // nothing to do
    }

    // MAX_CAND in nms.comp. Kept in step by hand: the shader needs it as a
    // compile-time constant for its shared arrays, and exceeding it would drop
    // candidates silently rather than fail, so refuse here instead.
    const unsigned int MAX_CAND = 1024u;
    if (num_boxes > MAX_CAND) {
        return false;
    }

    const size_t n_pairs = (size_t)num_classes * N;

    const uint32_t max_wg = device->properties.limits.maxComputeWorkGroupCount[0];
    if (n_pairs > (size_t)max_wg) {
        return false;
    }

    const size_t boxes_bytes  = (size_t)4 * num_boxes * N * sizeof(float);
    const size_t scores_bytes = (size_t)num_boxes * num_classes * N * sizeof(float);
    const size_t idx_bytes    = n_pairs * max_per_pair * sizeof(int32_t);
    const size_t cnt_bytes    = n_pairs * sizeof(int32_t);

    vk_buffer d_boxes  = ggml_vk_create_buffer_device(device, boxes_bytes);
    vk_buffer d_scores = ggml_vk_create_buffer_device(device, scores_bytes);
    vk_buffer d_idx    = ggml_vk_create_buffer_device(device, idx_bytes);
    vk_buffer d_cnt    = ggml_vk_create_buffer_device(device, cnt_bytes);

    ggml_vk_buffer_write(d_boxes,  0, boxes,  boxes_bytes);
    ggml_vk_buffer_write(d_scores, 0, scores, scores_bytes);

    // Pipelines compile lazily; without this the bind runs against an empty
    // handle and segfaults.
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_nms, 1);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    ggml_vk_ctx_begin(device, subctx);

    vk_op_nms_push_constants pc{};
    pc.num_boxes         = num_boxes;
    pc.num_classes       = num_classes;
    pc.N                 = N;
    pc.max_per_pair      = max_per_pair;
    pc.max_output        = max_output;
    pc.iou_thresh        = iou_thresh;
    pc.score_thresh      = score_thresh;
    pc.have_score_thresh = have_score_thresh;
    pc.center_point_box  = center_point_box;

    // wg_denoms for this pipeline are {1,1,1}, so the element count IS the
    // workgroup count: one group per (batch, class) pair.
    const std::array<uint32_t, 3> elements = { (uint32_t)n_pairs, 1, 1 };

    ggml_vk_dispatch_pipeline(
        ctx, subctx, ctx->device->pipeline_nms,
        { vk_subbuffer{ d_boxes,  0, boxes_bytes  },
          vk_subbuffer{ d_scores, 0, scores_bytes },
          vk_subbuffer{ d_idx,    0, idx_bytes    },
          vk_subbuffer{ d_cnt,    0, cnt_bytes    } },
        pc, elements);

    ggml_vk_ctx_end(subctx);
    ggml_vk_submit(subctx, ctx->fence);
    ggml_vk_wait_for_fence(ctx);

    ggml_vk_buffer_read(d_idx, 0, sel_idx, idx_bytes);
    ggml_vk_buffer_read(d_cnt, 0, sel_cnt, cnt_bytes);

    ggml_vk_destroy_buffer(d_boxes);
    ggml_vk_destroy_buffer(d_scores);
    ggml_vk_destroy_buffer(d_idx);
    ggml_vk_destroy_buffer(d_cnt);

    return true;
}
