static void ggml_vk_flash_attn(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask, const ggml_tensor * sinks, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_flash_attn((" << q << ", name=" << q->name << ", type=" << q->type << ", ne0=" << q->ne[0] << ", ne1=" << q->ne[1] << ", ne2=" << q->ne[2] << ", ne3=" << q->ne[3] << ", nb0=" << q->nb[0] << ", nb1=" << q->nb[1] << ", nb2=" << q->nb[2] << ", nb3=" << q->nb[3];
    std::cerr << "), (" << k << ", name=" << k->name << ", type=" << k->type << ", ne0=" << k->ne[0] << ", ne1=" << k->ne[1] << ", ne2=" << k->ne[2] << ", ne3=" << k->ne[3] << ", nb0=" << k->nb[0] << ", nb1=" << k->nb[1] << ", nb2=" << k->nb[2] << ", nb3=" << k->nb[3];
    std::cerr << "), (" << v << ", name=" << v->name << ", type=" << v->type << ", ne0=" << v->ne[0] << ", ne1=" << v->ne[1] << ", ne2=" << v->ne[2] << ", ne3=" << v->ne[3] << ", nb0=" << v->nb[0] << ", nb1=" << v->nb[1] << ", nb2=" << v->nb[2] << ", nb3=" << v->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    if (sinks) {
        std::cerr << "), (" << sinks << ", name=" << sinks->name << ", type=" << sinks->type << ", ne0=" << sinks->ne[0] << ", ne1=" << sinks->ne[1] << ", ne2=" << sinks->ne[2] << ", ne3=" << sinks->ne[3] << ", nb0=" << sinks->nb[0] << ", nb1=" << sinks->nb[1] << ", nb2=" << sinks->nb[2] << ", nb3=" << sinks->nb[3];
    }
    std::cerr << "))");

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const uint32_t nem0 = mask ? mask->ne[0] : 0;
    const uint32_t nem1 = mask ? mask->ne[1] : 0;
    const uint32_t nem2 = mask ? mask->ne[2] : 0;
    const uint32_t nem3 = mask ? mask->ne[3] : 0;

    const uint32_t HSK = nek0;
    const uint32_t HSV = nev0;
    uint32_t N = neq1;
    const uint32_t KV = nek1;

    GGML_ASSERT(ne0 == HSV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

    GGML_ASSERT(neq0 == HSK);

    GGML_ASSERT(neq1 == N);

    GGML_ASSERT(nev1 == nek1);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    assert(dst->type == GGML_TYPE_F32);
    assert(q->type == GGML_TYPE_F32);
    uint32_t gqa_ratio = 1;
    uint32_t qk_ratio = neq2 / nek2;
    uint32_t workgroups_x = (uint32_t)neq1;
    uint32_t workgroups_y = (uint32_t)neq2;
    uint32_t workgroups_z = (uint32_t)neq3;

    const bool f32acc = !ctx->device->fp16 || dst->op_params[3] == GGML_PREC_F32 || k->type == GGML_TYPE_BF16;

    // dequant K/V once into an f16 scratch, reordered KV layout so FA can read without a stride
    auto is_dense_kv_cache = [](const ggml_tensor * t) {
        return t->nb[0] == ggml_type_size(t->type) &&
               t->nb[2] == ggml_row_size(t->type, t->ne[0]) &&
               t->nb[1] == t->nb[2] * t->ne[2] &&
               (t->ne[3] == 1 || t->nb[3] == t->nb[1] * t->ne[1]);
    };
    const bool k_quant = k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16 && k->type != GGML_TYPE_F32;
    const bool v_quant = v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_BF16 && v->type != GGML_TYPE_F32;
    const bool kv_raw = (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && k->type == v->type;
    // gated to AMD, except RDNA4, which regressed in benchmarks
    const bool kv_raw_strided = ctx->device->vendor_id == VK_VENDOR_ID_AMD &&
                                ctx->device->architecture != vk_device_architecture::AMD_RDNA4 &&
                                neq1 >= 256 &&
                                kv_raw &&
                                (k->nb[1] != (uint64_t)HSK * ggml_type_size(k->type) ||
                                 v->nb[1] != (uint64_t)HSV * ggml_type_size(v->type)) &&
                                (HSK % 8) == 0 && (HSV % 8) == 0;
    const size_t kv_scratch_ts = kv_raw_strided ? ggml_type_size(k->type) : sizeof(ggml_fp16_t);
    const bool use_dequant_kv = ((k_quant && v_quant) || kv_raw_strided) && neq1 >= 64 &&
                                is_dense_kv_cache(k) && is_dense_kv_cache(v) &&
                                (uint64_t)ggml_nelements(k) * kv_scratch_ts <= ctx->device->properties.limits.maxStorageBufferRange &&
                                (uint64_t)ggml_nelements(v) * kv_scratch_ts <= ctx->device->properties.limits.maxStorageBufferRange &&
                                (kv_raw_strided ||
                                 (ctx->device->pipeline_dequant_transpose[k->type] != nullptr &&
                                  ctx->device->pipeline_dequant_transpose[v->type] != nullptr)) &&
                                // coopmat2 path does not benefit from the f16 scratch
                                !ctx->device->coopmat2 &&
                                // Intel Xe1 regresses, see PR 25494
                                (ctx->device->vendor_id != VK_VENDOR_ID_INTEL ||
                                 (ctx->device->coopmat_support && ctx->device->architecture != vk_device_architecture::INTEL_XE1));
    const ggml_type k_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : k->type;
    const ggml_type v_type_eff = (use_dequant_kv && !kv_raw_strided) ? GGML_TYPE_F16 : v->type;

    // For scalar/coopmat1 FA, we can use the "large" size to accommodate qga.
    // For coopmat2 FA, we always use the small size (which is still pretty large for gqa).
    vk_fa_tuning_params tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, 512, KV, k_type_eff, v_type_eff, f32acc);
    const uint32_t max_gqa = std::min(tuning_params.block_rows, 32u);

    if (N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
        qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
        // grouped query attention - make the N dimension equal to gqa_ratio, reduce
        // workgroups proportionally in y dimension. The shader will detect gqa_ratio > 1
        // and change addressing calculations to index Q's dimension 2.
        gqa_ratio = qk_ratio;
        N = gqa_ratio;
        workgroups_y /= gqa_ratio;
    }

    tuning_params = get_fa_tuning_params(ctx->device, HSK, HSV, N, KV, k_type_eff, v_type_eff, f32acc);

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    // Sparse mask hint (op_params[4]): compact the <= n_kv_max finite positions and gather only those.
    const int32_t n_kv_max = mask ? ggml_get_op_params_i32(dst, 4) : 0;
    static const bool disable_sparse = getenv("GGML_VK_FA_SPARSE_DISABLE") != nullptr;
    const bool kv_f16 = k_type_eff == GGML_TYPE_F16 && v_type_eff == GGML_TYPE_F16;
    // cm2 dense is fast, so it needs a larger reduction to win.
    // With quantized K/V, sparse only breaks even around 16x (measured on RDNA3/RDNA4).
    const int64_t min_ratio = tuning_params.path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16);
    // coopmat2 vector decode requires 8B strides.
    auto sparse_gather_aligned = [](const ggml_tensor * t) {
        return (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) ||
               (t->nb[1] | t->nb[2] | t->nb[3]) % (4 * sizeof(ggml_fp16_t)) == 0;
    };
    const bool use_sparse = !disable_sparse && n_kv_max > 0 && mask &&
                            max_bias == 0.0f && logit_softcap == 0.0f &&
                            (tuning_params.path != FA_COOPMAT2 || (sparse_gather_aligned(k) && sparse_gather_aligned(v))) &&
                            nem0 == KV &&
                            (int64_t)KV >= std::max<int64_t>(4096, min_ratio * (int64_t)n_kv_max) &&
                            (gqa_ratio > 1 || (tuning_params.path == FA_SCALAR && N == 1));

    const uint32_t q_stride = (uint32_t)(nbq1 / ggml_type_size(q->type));
    uint32_t k_stride = (uint32_t)(nbk1 / ggml_type_size(k->type));
    uint32_t v_stride = (uint32_t)(nbv1 / ggml_type_size(v->type));

    // For F32, the shader treats it as a block of size 4 (for vec4 loads)
    if (k->type == GGML_TYPE_F32) {
        k_stride /= 4;
    }
    if (v->type == GGML_TYPE_F32) {
        v_stride /= 4;
    }

    uint32_t nbk2_eff = (uint32_t)nbk2, nbk3_eff = (uint32_t)nbk3;
    uint32_t nbv2_eff = (uint32_t)nbv2, nbv3_eff = (uint32_t)nbv3;
    if (use_dequant_kv) {
        k_stride = HSK;
        v_stride = HSV;
        nbk2_eff = (uint32_t)((uint64_t)HSK * KV * sizeof(ggml_fp16_t));
        nbk3_eff = (uint32_t)((uint64_t)HSK * KV * nek2 * sizeof(ggml_fp16_t));
        nbv2_eff = (uint32_t)((uint64_t)HSV * KV * sizeof(ggml_fp16_t));
        nbv3_eff = (uint32_t)((uint64_t)HSV * KV * nev2 * sizeof(ggml_fp16_t));
    }
    const uint32_t alignment = tuning_params.block_cols;
    bool aligned = (KV % alignment) == 0 &&
                   // the "aligned" shader variant will forcibly align strides, for performance
                   (q_stride & 7) == 0 && (k_stride & 7) == 0 && (v_stride & 7) == 0;

    // Need to use the coopmat2 variant that clamps loads when HSK/HSV aren't sufficiently aligned.
    if (((HSK | HSV) % 16) != 0 && tuning_params.path == FA_COOPMAT2) {
        aligned = false;
    }

    // Only use mask opt when the mask is fairly large. This hasn't been tuned extensively.
    bool use_mask_opt = mask && !use_sparse && nem1 >= 32 && nem0 * nem1 > 32768 && nem0 >= tuning_params.block_cols * 16
                        && (ctx->device->architecture != vk_device_architecture::AMD_GCN || HSK > 256 || HSV > 256);
    vk_fa_pipeline_state fa_pipeline_state = get_fa_pipeline_state(ctx->device, tuning_params, HSK, HSV, aligned, f32acc,
                                                                   mask != nullptr, use_mask_opt, logit_softcap != 0, use_sparse, k_type_eff, v_type_eff);

    vk_pipeline pipeline = nullptr;

    bool xe_fa_opt = false;
    bool fa_copy_qstate = false;
    bool xe_fa_supported_platform =
        (ctx->device.get()->architecture == INTEL_XE2 && ctx->device.get()->properties.deviceID != 0xFD80 && ctx->device.get()->properties.deviceID != 0xFD81) ||
        (ctx->device.get()->architecture == INTEL_XE1 && ctx->device.get()->coopmat_support && ctx->device.get()->uma);
    bool xe_fa_supported_usage = neq0 % 32 == 0 && nev0 % 16 == 0 && q->nb[1] > q->nb[2] && k->nb[1] > k->nb[2] && v->nb[1] > v->nb[2] && mask != nullptr;
    bool xe_fa_supported_dtype = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && (mask != nullptr && mask->type == GGML_TYPE_F16);
    std::pair<vk_pipeline, vk_pipeline> xe_fa_pipeline_dual_phases = { nullptr , nullptr };
    vk_pipeline xe_fa_pipeline = nullptr;
    size_t size_p = 0;
    size_t size_group_max = 0;

    {
        std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
        auto &pipelines = ctx->device->pipeline_flash_attn_f32_f16;
        auto it = pipelines.find(fa_pipeline_state);
        if (it != pipelines.end()) {
            pipeline = it->second;
        } else {
            pipelines[fa_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
        }
    }

    assert(pipeline);
    // Compile early to initialize wg_denoms.
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    uint32_t split_kv = KV;
    uint32_t split_k = 1;

    // Intel Alchemist prefers more workgroups
    const uint32_t shader_core_count_multiplier = (ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ctx->device->architecture != INTEL_XE2) ? 2 : 1;

    // Use a placeholder core count if one isn't available. split_k is a big help for perf.
    const uint32_t shader_core_count = ctx->device->shader_core_count ? ctx->device->shader_core_count * shader_core_count_multiplier : 16;

    const uint32_t Br = fa_pipeline_state.Br;
    const uint32_t Bc = fa_pipeline_state.Bc;

    GGML_ASSERT(Br == pipeline->wg_denoms[0]);
    const uint32_t Tr = CEIL_DIV(N, Br);

    // Try to use split_k when KV is large enough to be worth the overhead.
    // Sparse: split_kv carries n_kv_max, split_k partitions its blocks for occupancy.
    if (use_sparse) {
        split_kv = (uint32_t)n_kv_max;
        const uint32_t total_blocks = CEIL_DIV((uint32_t)n_kv_max, Bc);
        const uint32_t base_wgs = (gqa_ratio > 1 ? workgroups_x : Tr) * workgroups_y * workgroups_z;
        if (base_wgs < shader_core_count * 2) {
            split_k = shader_core_count * 2 / base_wgs;
        }
        split_k = std::max(1u, std::min(split_k, total_blocks));
        // Match the shader's per-split block count so no split is empty.
        const uint32_t per_blocks = CEIL_DIV(total_blocks, split_k);
        split_k = CEIL_DIV(total_blocks, per_blocks);
    } else if (gqa_ratio > 1 && workgroups_x <= Br) {
        split_k = shader_core_count * 2 / (workgroups_x * workgroups_y * workgroups_z);
    } else if (gqa_ratio <= 1) {
        uint32_t total_wgs_no_split = Tr * workgroups_y * workgroups_z;
        if (total_wgs_no_split < shader_core_count * 2) {
            split_k = shader_core_count * 2 / total_wgs_no_split;
        }
    }

    if (!use_sparse && split_k > 1) {
        // Try to evenly split KV into split_k chunks, but it needs to be a multiple
        // of "align", so recompute split_k based on that.
        split_kv = ROUNDUP_POW2(std::max(1u, KV / split_k), alignment);
        split_k = CEIL_DIV(KV, split_kv);
        xe_fa_opt = xe_fa_supported_platform && xe_fa_supported_usage && xe_fa_supported_dtype;
        if (xe_fa_opt) {
            const uint32_t split_p_size = 32;
            const size_t max_dim = (nek1 + split_p_size - 1) / split_p_size;
            const size_t p_dim = max_dim * split_p_size;
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            auto& pipelines = ctx->device->pipeline_xe_fa_decode_dual_phases;
            auto it = pipelines.find({ (uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1 });
            if (it != pipelines.end()) {
                xe_fa_pipeline_dual_phases = it->second;
            } else {
                pipelines[{(uint32_t)neq0, (uint32_t)nev0, qk_ratio, (uint32_t)neq1}] = xe_fa_pipeline_dual_phases = std::make_pair(std::make_shared<vk_pipeline_struct>(), std::make_shared<vk_pipeline_struct>());
            }
#endif
            if (xe_fa_pipeline_dual_phases.first == nullptr || xe_fa_pipeline_dual_phases.second == nullptr) {
                xe_fa_opt = false;
                fa_copy_qstate = false;
            } else {
                size_p = neq1 * neq2 * p_dim * neq3 * sizeof(ggml_fp16_t);
                size_group_max = neq1 * neq2 * max_dim * neq3 * sizeof(float);
                size_t temp_size = ggml_nelements(q) * sizeof(ggml_fp16_t) + size_p + size_group_max;
                fa_copy_qstate = true;
                if (ctx->prealloc_size_x < temp_size) {
                    ctx->prealloc_size_x = temp_size;
                    ggml_vk_preallocate_buffers(ctx, subctx);
                }

                if (ctx->prealloc_x_need_sync) {
                    ggml_vk_sync_buffers(ctx, subctx);
                }
            }
        }
    }

    if (xe_fa_opt == true) {
        use_mask_opt = false;
    }

    // Reserve space for split_k temporaries. For each split x batch, we need to store the O matrix (D x ne1)
    // and the per-row m and L values (ne1 rows). We store all the matrices first, followed by the rows.
    // For matrices, the order is (inner to outer) [HSV, ne1, k, ne2, ne3].
    // For L/M, the order is (inner to outer) [ne1, k, ne2, ne3].
    const uint64_t split_k_size = split_k > 1 ? (HSV * ne1 * sizeof(float) + ne1 * sizeof(float) * 2) * split_k * ne2 * ne3 : 0;
    if (split_k_size > ctx->device->properties.limits.maxStorageBufferRange) {
        GGML_ABORT("Requested preallocation size is too large");
    }
    if (ctx->prealloc_size_split_k < split_k_size) {
        ctx->prealloc_size_split_k = split_k_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }

    const uint32_t mask_opt_num_dwords = CEIL_DIV(nem0, 16 * Bc);
    const uint64_t mask_opt_size = sizeof(uint32_t) * mask_opt_num_dwords * CEIL_DIV(nem1, Br) * nem2 * nem3;

    vk_pipeline pipeline_fa_mask_opt = nullptr;
    if (use_mask_opt) {
        {
            std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
            auto &pipelines = ctx->device->pipeline_fa_mask_opt;
            auto it = pipelines.find({Br, Bc});
            if (it != pipelines.end()) {
                pipeline_fa_mask_opt = it->second;
            } else {
                pipelines[{Br, Bc}] = pipeline_fa_mask_opt = std::make_shared<vk_pipeline_struct>();
            }
        }
        assert(pipeline_fa_mask_opt);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline_fa_mask_opt, 1);

        if (ctx->prealloc_size_y < mask_opt_size) {
            ctx->prealloc_size_y = mask_opt_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    // Sparse index scratch reuses prealloc_y (mutually exclusive with mask opt).
    const uint64_t sparse_idx_size = use_sparse
        ? sizeof(int32_t) * (uint64_t)n_kv_max * nem1 * nem2 * nem3
        : 0;
    vk_pipeline sparse_compact_pipeline = ctx->device->fa_sparse_compact_use_subgroups
        ? ctx->device->pipeline_fa_sparse_compact_subgroup
        : ctx->device->pipeline_fa_sparse_compact;
    if (use_sparse) {
        ggml_pipeline_request_descriptor_sets(ctx, sparse_compact_pipeline, 1);
        if (ctx->prealloc_size_y < sparse_idx_size) {
            ctx->prealloc_size_y = sparse_idx_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
    }

    const uint32_t n_head_kv   = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer mask_buf = mask ? ggml_vk_tensor_subbuffer(ctx, mask) : q_buf;
    vk_subbuffer sinks_buf = sinks ? ggml_vk_tensor_subbuffer(ctx, sinks) : q_buf;
    vk_subbuffer mask_opt_buf = use_mask_opt ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    vk_subbuffer sparse_buf = use_sparse ? ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0) : q_buf;
    if (use_mask_opt || use_sparse) {
        // the mask opt bits and the sparse index list overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;
    }

    if (use_dequant_kv) {
        const uint64_t k_sz = (uint64_t)ggml_nelements(k) * kv_scratch_ts;
        const uint64_t v_sz = (uint64_t)ggml_nelements(v) * kv_scratch_ts;
        if (ctx->prealloc_size_x < k_sz + v_sz) {
            ctx->prealloc_size_x = k_sz + v_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        vk_subbuffer k_dst = vk_subbuffer{ ctx->prealloc_x, 0,    k_sz };
        vk_subbuffer v_dst = vk_subbuffer{ ctx->prealloc_x, k_sz, v_sz };
        const uint32_t k_nel = (uint32_t)ggml_nelements(k);
        const uint32_t v_nel = (uint32_t)ggml_nelements(v);
        if (kv_raw_strided) {
            vk_pipeline cp_k = ggml_vk_get_cpy_pipeline(ctx, k, nullptr, k->type);
            vk_pipeline cp_v = ggml_vk_get_cpy_pipeline(ctx, v, nullptr, v->type);
            ggml_pipeline_request_descriptor_sets(ctx, cp_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, cp_v, 1);
            if (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            auto make_pc = [](uint32_t hs, uint32_t nh, uint32_t kv, uint32_t nel) {
                vk_op_unary_push_constants pc{};
                pc.ne = nel;
                pc.ne00 = hs; pc.ne01 = nh;      pc.ne02 = kv; pc.ne03 = nel / (hs * nh * kv);
                pc.nb00 = 1;  pc.nb01 = hs;      pc.nb02 = hs * nh; pc.nb03 = hs * nh * kv;
                pc.ne10 = hs; pc.ne11 = nh;      pc.ne12 = kv; pc.ne13 = pc.ne03;
                pc.nb10 = 1;  pc.nb11 = hs * kv; pc.nb12 = hs; pc.nb13 = hs * kv * nh;
                init_pushconst_fastdiv(pc);
                return pc;
            };
            auto cpy_elems = [](uint32_t ne) -> std::array<uint32_t, 3> {
                if (ne > 262144) { return { 512, 512, CEIL_DIV(ne, 262144) }; }
                if (ne > 512)    { return { 512, CEIL_DIV(ne, 512), 1 }; }
                return { ne, 1, 1 };
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_k, { k_buf, k_dst },
                make_pc((uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, k_nel), cpy_elems(k_nel));
            ggml_vk_dispatch_pipeline(ctx, subctx, cp_v, { v_buf, v_dst },
                make_pc((uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, v_nel), cpy_elems(v_nel));
        } else {
            vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
            vk_pipeline tr_v = ctx->device->pipeline_dequant_transpose[v->type];
            ggml_pipeline_request_descriptor_sets(ctx, tr_k, 1);
            ggml_pipeline_request_descriptor_sets(ctx, tr_v, 1);
            if (ctx->prealloc_x_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            { const std::vector<uint32_t> pc = { (uint32_t)HSK, (uint32_t)nek2, (uint32_t)KV, 0, k_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_k, { k_buf, k_dst }, pc, { k_nel, 1, 1 }); }
            { const std::vector<uint32_t> pc = { (uint32_t)HSV, (uint32_t)nev2, (uint32_t)KV, 0, v_nel };
              ggml_vk_dispatch_pipeline(ctx, subctx, tr_v, { v_buf, v_dst }, pc, { v_nel, 1, 1 }); }
        }
        ggml_vk_sync_buffers(ctx, subctx);
        k_buf = k_dst;
        v_buf = v_dst;
    }

    uint32_t mask_n_head_log2 = ((sinks != nullptr) << 24) | n_head_log2;

    if (use_mask_opt)
    {
        const vk_op_flash_attn_mask_opt_push_constants opt_pc = {
            nem0,
            nem1,
            nem2,
            (uint32_t)(mask->nb[1] / sizeof(ggml_fp16_t)),
            (uint32_t)(mask->nb[2] / sizeof(ggml_fp16_t)),
            (uint32_t)(mask->nb[3] / sizeof(ggml_fp16_t)),
            mask_opt_num_dwords,
            mask_opt_num_dwords * CEIL_DIV(nem1, Br),
            mask_opt_num_dwords * CEIL_DIV(nem1, Br) * nem2,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline_fa_mask_opt,
                                  { mask_buf, mask_opt_buf }, opt_pc,
                                  { mask_opt_num_dwords, CEIL_DIV(nem1, Br), nem2 * nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    if (use_sparse)
    {
        const vk_op_flash_attn_sparse_compact_push_constants sc_pc = {
            KV,
            nem1,
            nem2,
            (uint32_t)(mask->nb[1] / sizeof(ggml_fp16_t)),
            (uint32_t)(mask->nb[2] / sizeof(ggml_fp16_t)),
            (uint32_t)(mask->nb[3] / sizeof(ggml_fp16_t)),
            (uint32_t)n_kv_max,
        };

        ggml_vk_dispatch_pipeline(ctx, subctx, sparse_compact_pipeline,
                                  { mask_buf, sparse_buf }, sc_pc,
                                  { nem1, nem2, nem3 });
        ggml_vk_sync_buffers(ctx, subctx);
    }

    const vk_flash_attn_push_constants pc = { N, KV,
                                              (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3,
                                              (uint32_t)neq2, (uint32_t)neq3,
                                              (uint32_t)nek2, (uint32_t)nek3,
                                              (uint32_t)nev2, (uint32_t)nev3,
                                              nem1, nem2, nem3,
                                              q_stride, (uint32_t)nbq2, (uint32_t)nbq3,
                                              k_stride, nbk2_eff, nbk3_eff,
                                              v_stride, nbv2_eff, nbv3_eff,
                                              scale, max_bias, logit_softcap,
                                              mask_n_head_log2, m0, m1,
                                              gqa_ratio, split_kv, split_k };

    if (xe_fa_opt && split_k > 1) {
        auto upper_power_of_2 = [&](uint32_t in) {
            GGML_ASSERT(in != 0);
            if (in <= 1) return 1u;
            uint32_t ret = in - 1;
            ret |= ret >> 1;
            ret |= ret >> 2;
            ret |= ret >> 4;
            ret |= ret >> 8;
            ret |= ret >> 16;
            return ret + 1;
        };
        auto to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, q->type);
        const uint32_t out_dim_per_wg = qk_ratio > 16 ? 8 : 16;
        size_t x_ne = ggml_nelements(q);
        size_t temp_buf_offset = 0;
        uint32_t head_stride_k = uint32_t(nbk2 / ggml_type_size(k->type));
        uint32_t head_stride_v = uint32_t(nbv2 / ggml_type_size(v->type));
        uint32_t batch_stride_q = uint32_t(nbq3 / ggml_type_size(q->type));
        uint32_t batch_stride_k = uint32_t(nbk3 / ggml_type_size(k->type));
        uint32_t batch_stride_v = uint32_t(nbv3 / ggml_type_size(v->type));
        uint32_t batch_stride_m = mask ? uint32_t(mask->nb[3] / ggml_type_size(mask->type)) : 0u;
        uint32_t batch_stride_o = uint32_t(nb3 / ggml_type_size(dst->type));
        vk_fa_xe_opt_push_constants pc_ph1 = { (uint32_t)nek1, (uint32_t)neq1, (uint32_t)neq2, (uint32_t)nek2, qk_ratio, 1, (sinks != nullptr) ? 1u : 0u, (uint32_t)k_stride, head_stride_k,
            batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_m, batch_stride_o, scale };
        vk_fa_xe_opt_push_constants pc_ph2 = pc_ph1;
        pc_ph2.nbkv_tok = v_stride;
        pc_ph2.nbkv_head = head_stride_v;
        vk_subbuffer q_temp_buf = fa_copy_qstate ? ggml_vk_subbuffer(ctx, ctx->prealloc_x, temp_buf_offset) : q_buf;
        temp_buf_offset += fa_copy_qstate ? x_ne * sizeof(ggml_fp16_t) : 0;
        vk_subbuffer p_temp_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, temp_buf_offset);
        temp_buf_offset += size_p;
        vk_subbuffer max_temp_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_x, temp_buf_offset);
        temp_buf_offset += size_group_max;
        uint32_t xe_native_sub_group_size = ctx->device.get()->architecture == INTEL_XE1 ? 8 : 16;
        uint32_t aligned_gqa_ratio = upper_power_of_2(qk_ratio);
        uint32_t out_per_wg_ph1 = std::min(256u / xe_native_sub_group_size, (uint32_t)neq1);
        uint32_t out_per_wg_ph2 = std::min(std::max(16u / aligned_gqa_ratio, 1u), (uint32_t)neq1);
        uint32_t ph1_wg = ((neq1 + out_per_wg_ph1 - 1) / out_per_wg_ph1) * nek2;
        uint32_t ph2_wg = ((neq1 + out_per_wg_ph2 - 1) / out_per_wg_ph2) * ne0 / out_dim_per_wg;
        if (fa_copy_qstate) {
            const std::vector<uint32_t> pc_cpy_fp16 =
            { (uint32_t)q->ne[0], (uint32_t)q->ne[1], (uint32_t)q->ne[2], (uint32_t)q->ne[3], (uint32_t)(x_ne) };
            ggml_vk_sync_buffers(ctx, subctx);
            ggml_pipeline_request_descriptor_sets(ctx, to_fp16_vk_0, 1);
            ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { q_buf, q_temp_buf }, pc_cpy_fp16, { (uint32_t)(x_ne), 1, 1 });
        }

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.first, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.first,
            { q_temp_buf, k_buf, mask_buf, p_temp_buf, max_temp_buf },
            pc_ph1, { (uint32_t)ph1_wg, (uint32_t)nek1, (uint32_t)neq3 });

        ggml_vk_sync_buffers(ctx, subctx);
        ggml_pipeline_request_descriptor_sets(ctx, xe_fa_pipeline_dual_phases.second, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, xe_fa_pipeline_dual_phases.second,
            { p_temp_buf, v_buf, max_temp_buf, sinks_buf, dst_buf },
            pc_ph2, { (uint32_t)ph2_wg, (uint32_t)nev2, (uint32_t)neq3 });

        ctx->prealloc_x_need_sync = true;
    } else if (split_k > 1) {
        ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_flash_attn_split_k_reduce, 1);

        if (ctx->prealloc_split_k_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        // We reuse workgroups_x to mean the number of splits, so we need to
        // cancel out the divide by wg_denoms[0].
        uint32_t dispatch_x;
        if (gqa_ratio > 1) {
            workgroups_x *= pipeline->wg_denoms[0];
            dispatch_x = split_k * workgroups_x;
        } else {
            dispatch_x = Tr * split_k * pipeline->wg_denoms[0];
        }

        vk_subbuffer split_k_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_split_k, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, mask_buf, sinks_buf, split_k_buf, mask_opt_buf, sparse_buf},
                                    pc, { dispatch_x, workgroups_y, workgroups_z });

        ggml_vk_sync_buffers(ctx, subctx);
        const vk_op_flash_attn_split_k_reduce_push_constants pc2 = { HSV, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)ne3, split_k, (sinks != nullptr) };
        ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_flash_attn_split_k_reduce,
                                    {split_k_buf, sinks_buf, dst_buf},
                                    pc2, { (uint32_t)ne1, HSV, (uint32_t)(ne2 * ne3) });
        ctx->prealloc_split_k_need_sync = true;
    } else {
        if (gqa_ratio > 1) {
            // When using gqa, we want one actual workgroup per batch, so cancel out wg_denoms
            workgroups_x *= pipeline->wg_denoms[0];
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                                    {q_buf, k_buf, v_buf, mask_buf, sinks_buf, dst_buf, mask_opt_buf, sparse_buf},
                                    pc, { workgroups_x, workgroups_y, workgroups_z });
    }

    if (use_dequant_kv) {
        ctx->prealloc_x_need_sync = true;
    }
    if (use_mask_opt || use_sparse) {
        ctx->prealloc_y_need_sync = true;
    }
}

static vk_conv_shapes ggml_vk_conv_select_shape(ggml_backend_vk_context * ctx, uint32_t K, uint32_t NPQ) {
    auto n_tiles = [&](vk_conv_shapes s) {
        return CEIL_DIV(K, vk_conv_block_sizes[s].K)
            * CEIL_DIV(NPQ, vk_conv_block_sizes[s].NPQ);
    };

    // We can't query number of shader cores on Intel, use 32 as a placeholder
    // so small convolutions will still choose a smaller tile.
    const uint32_t shader_core_count = ctx->device->shader_core_count > 0 ? ctx->device->shader_core_count : 32;

    // DIVERGENCE: upstream 0.27 disables 128x128 here on cm1 (no cm2) devices because its
    // cm1 conv shader can't fit that tile in shared memory. ggmlR's cm1 conv is its own
    // (fixed 64x64 tiles, always the CONV_SHAPE_128x128 slot) and does not go through this
    // selector, so the restriction would only cost the scalar F32 path its 128x128 tile.
    if (K > 64 && n_tiles(CONV_SHAPE_128x128) >= shader_core_count * 2) {
        return CONV_SHAPE_128x128;
    } else if (K <= 32 && n_tiles(CONV_SHAPE_32x256) >= shader_core_count * 2) {
        return CONV_SHAPE_32x256;
    } else if (K <= 64 && n_tiles(CONV_SHAPE_64x128) >= shader_core_count * 2) {
        return CONV_SHAPE_64x128;
    } else {
        return CONV_SHAPE_64x32;
    }
}

static vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * dst, ggml_op op) {
    switch (op) {
    case GGML_OP_GET_ROWS:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        if (src0->type == GGML_TYPE_I32) {
            // i32 src only supports i32 result
            GGML_ASSERT(dst->type == GGML_TYPE_I32);
            return ctx->device->pipeline_get_rows[src0->type];
        }
        if (dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_get_rows[src0->type];
        }
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_get_rows_f32[src0->type];
        }
        return nullptr;
    case GGML_OP_ACC:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_acc_f32;
        }
        return nullptr;
    case GGML_OP_SET:
        if (src0->type == src1->type && src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32)) {
            return ctx->device->pipeline_set_f32;
        }
        return nullptr;
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16)) {
            return nullptr;
        }
        switch (op) {
        case GGML_OP_ADD:
        {
            if (ctx->num_additional_fused_ops > 0) {
                if (ctx->do_add_rms_partials) {
                    return ctx->device->pipeline_multi_add_rms[ctx->num_additional_fused_ops];
                } else {
                    return ctx->device->pipeline_multi_add[ctx->num_additional_fused_ops];
                }
            }
            if (ctx->do_add_rms_partials) {
                auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_rms_norepeat : ctx->device->pipeline_add_rms;
                return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            } else {
                auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_add_norepeat : ctx->device->pipeline_add;
                return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
            }
        }
        case GGML_OP_SUB:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_sub_norepeat : ctx->device->pipeline_sub;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        case GGML_OP_MUL:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_mul_norepeat : ctx->device->pipeline_mul;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        case GGML_OP_DIV:
        {
            auto pipelines = ggml_are_same_shape(src0, src1) ? ctx->device->pipeline_div_norepeat : ctx->device->pipeline_div;
            return pipelines[src0->type == GGML_TYPE_F16][src1->type == GGML_TYPE_F16][dst->type == GGML_TYPE_F16];
        }
        default:
            break;
        }
        return nullptr;
    case GGML_OP_ADD_ID:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_add_id_f32;
        }
        return nullptr;
    case GGML_OP_CONCAT: {
        if (!ggml_vk_concat_supported(src0, src1, dst)) {
            return nullptr;
        }
        switch (ggml_vk_concat_unit_size(src0->type)) {
        case 1:
            return ctx->device->pipeline_concat_i8;
        case 2:
            return ctx->device->pipeline_concat_i16;
        case 4:
            return ctx->device->pipeline_concat_i32;
        case 8:
            return ctx->device->pipeline_concat_i64;
        default:
            return nullptr;
        }
    }
    case GGML_OP_UPSCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t mode = (ggml_get_op_params_i32(dst, 0) & (0xFF | GGML_SCALE_FLAG_ANTIALIAS));
            switch (mode) {
                case GGML_SCALE_MODE_NEAREST:
                    return ctx->device->pipeline_upscale_nearest_f32;
                case GGML_SCALE_MODE_BILINEAR:
                    return ctx->device->pipeline_upscale_bilinear_f32;
                case GGML_SCALE_MODE_BICUBIC:
                    return ctx->device->pipeline_upscale_bicubic_f32;
                case GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS:
                    return ctx->device->pipeline_upscale_bilinear_antialias_f32;
                default:
                    return nullptr;
            }
        }
        return nullptr;
    case GGML_OP_SCALE:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_scale[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_SQR:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_sqr[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_SQRT:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_sqrt[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_SIN:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_sin[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_COS:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_cos[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_LOG:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_log[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_TRI:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_tri[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_DIAG:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_diag[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_CLAMP:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_clamp[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_PAD:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pad_f32;
        }
        return nullptr;
    case GGML_OP_PAD_REFLECT_1D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pad_reflect_1d_f32;
        }
        return nullptr;
    case GGML_OP_ROLL:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_roll_f32;
        }
        return nullptr;
    case GGML_OP_REPEAT:
        if (ggml_type_size(src0->type) == sizeof(float) && ggml_type_size(dst->type) == sizeof(float)) {
            return ctx->device->pipeline_repeat_i32;
        }
        if (ggml_type_size(src0->type) == 2 && ggml_type_size(dst->type) == 2) {
            return ctx->device->pipeline_repeat_i16;
        }
        return nullptr;
    case GGML_OP_REPEAT_BACK:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_repeat_back_f32;
        }
        return nullptr;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        return ggml_vk_get_cpy_pipeline(ctx, src0, dst, dst->type);
    case GGML_OP_SET_ROWS:
        {
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                return nullptr;
            }
            const int src_idx = src0->type == GGML_TYPE_F16;
            if (src1->type == GGML_TYPE_I64) {
                return ctx->device->pipeline_set_rows_i64[src_idx][dst->type];
            } else if (src1->type == GGML_TYPE_I32) {
                return ctx->device->pipeline_set_rows_i32[src_idx][dst->type];
            }
            return nullptr;
        }
    case GGML_OP_SCATTER_ELEMENTS:
        {
            int32_t reduction;
            memcpy(&reduction, dst->op_params, sizeof(int32_t));
            return reduction == 1 ? ctx->device->pipeline_scatter_elements_add
                                  : ctx->device->pipeline_scatter_elements_none;
        }
    case GGML_OP_SILU_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_silu_back_f32;
        }
        return nullptr;
    case GGML_OP_GELU_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_gelu_back_f32;
        }
        return nullptr;
    case GGML_OP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_norm_f32;
        }
        return nullptr;
    case GGML_OP_GROUP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_group_norm_f32;
        }
        return nullptr;
    case GGML_OP_RMS_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (ctx->do_add_rms_partials) {
                return ctx->fused_rms_norm_mode == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_partials_f32 : ctx->device->pipeline_rms_norm_partials_f32;
            }
            return ctx->fused_rms_norm_mode == RMS_NORM_MUL ? ctx->device->pipeline_rms_norm_mul_f32 : ctx->device->pipeline_rms_norm_f32;
        }
        return nullptr;
    case GGML_OP_RMS_NORM_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rms_norm_back_f32;
        }
        return nullptr;
    // DIVERGENCE from upstream: LayerNorm backward, see ggml_norm_back().
    case GGML_OP_NORM_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_norm_back_f32;
        }
        return nullptr;
    case GGML_OP_L2_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_l2_norm_f32;
        }
        return nullptr;
    case GGML_OP_UNARY:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            return nullptr;
        }

        switch (ggml_get_unary_op(dst)) {
            case GGML_UNARY_OP_EXP:
                return ctx->device->pipeline_exp[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_EXPM1:
                return ctx->device->pipeline_expm1[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_ELU:
                return ctx->device->pipeline_elu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SILU:
                return ctx->device->pipeline_silu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU:
                return ctx->device->pipeline_gelu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU_ERF:
                return ctx->device->pipeline_gelu_erf[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_GELU_QUICK:
                return ctx->device->pipeline_gelu_quick[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_RELU:
                return ctx->device->pipeline_relu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_XIELU:
                return ctx->device->pipeline_xielu[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_NEG:
                return ctx->device->pipeline_neg[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_TANH:
                return ctx->device->pipeline_tanh[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SIGMOID:
                return ctx->device->pipeline_sigmoid[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_HARDSIGMOID:
                return ctx->device->pipeline_hardsigmoid[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_HARDSWISH:
                return ctx->device->pipeline_hardswish[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_ABS:
                return ctx->device->pipeline_abs[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SOFTPLUS:
                return ctx->device->pipeline_softplus[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_STEP:
                return ctx->device->pipeline_step[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_ROUND:
                return ctx->device->pipeline_round[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_ROUND_EVEN:
                return ctx->device->pipeline_round_even[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_CEIL:
                return ctx->device->pipeline_ceil[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_FLOOR:
                return ctx->device->pipeline_floor[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_TRUNC:
                return ctx->device->pipeline_trunc[dst->type == GGML_TYPE_F16];
            case GGML_UNARY_OP_SGN:
                return ctx->device->pipeline_sgn[dst->type == GGML_TYPE_F16];
            default:
                break;
        }
        return nullptr;
    case GGML_OP_GLU:
        if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
            (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) ||
            (src0->type != dst->type)) {
            return nullptr;
        }

        switch (ggml_get_glu_op(dst)) {
            case GGML_GLU_OP_GEGLU:
                return ctx->device->pipeline_geglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_REGLU:
                return ctx->device->pipeline_reglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_SWIGLU:
                return ctx->device->pipeline_swiglu[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_SWIGLU_OAI:
                return ctx->device->pipeline_swiglu_oai[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_GEGLU_ERF:
                return ctx->device->pipeline_geglu_erf[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_GEGLU_QUICK:
                return ctx->device->pipeline_geglu_quick[dst->type == GGML_TYPE_F16];
            case GGML_GLU_OP_SWIGLU_CLAMP:
                return ctx->device->pipeline_swiglu_clamp[dst->type == GGML_TYPE_F16];
            default:
                break;
        }
        return nullptr;
    case GGML_OP_DIAG_MASK_INF:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_diag_mask_inf_f32;
        }
        return nullptr;
    case GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
        GGML_ASSERT(!src2 || src2->type == GGML_TYPE_F32);

        if (ctx->num_additional_fused_ops) {
            uint32_t idx = (uint32_t)ceilf(log2f(float(dst->ne[0])));
            GGML_ASSERT(idx < num_topk_moe_pipelines);
            // use n_experts from push constant if it's not equal to the power of two spec constant
            bool use_push = dst->ne[0] != (1u << idx);
            return ctx->device->pipeline_topk_moe[idx][use_push];
        }

        if (src0->type == GGML_TYPE_F32 && (src1 == nullptr || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            return src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_wg512 : ctx->device->pipeline_soft_max_f32;
        }
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            return src0->ne[0] > 1024 ? ctx->device->pipeline_soft_max_f32_f16_wg512 : ctx->device->pipeline_soft_max_f32_f16;
        }
        if (src0->type == GGML_TYPE_F16 && (src1 == nullptr || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F16) {
            return src0->ne[0] >= 2048 ? ctx->device->pipeline_soft_max_f16_wg512 :
                   src0->ne[0] >= 256  ? ctx->device->pipeline_soft_max_f16_wg128 :
                                         ctx->device->pipeline_soft_max_f16;
        }
        return nullptr;
    case GGML_OP_SOFT_MAX_BACK:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_soft_max_back_f32;
        }
        return nullptr;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        {
            const ggml_tensor *rope = ctx->num_additional_fused_ops == 2 ? dst->src[0]->src[0] : dst;
            const int mode = ((const int32_t *) rope->op_params)[2];
            const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
            const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

            if (is_neox) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_neox_f32;
                }
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_neox_f32_f16;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_neox_f16;
                }
            } else if (is_mrope && !is_vision) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_multi_f32;
                }
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_multi_f32_f16;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_multi_f16;
                }
            } else if (is_vision) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_vision_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_vision_f16;
                }
            } else {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_norm_f32;
                }
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_norm_f32_f16;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_norm_f16;
                }
            }
            return nullptr;
        }
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_MEAN:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sum_rows[0];
        }
        if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_sum_rows[1];
        }
        return nullptr;
    case GGML_OP_CUMSUM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (src0->ne[0] <= 512) {
                return ctx->device->pipeline_cumsum_small_f32;
            } else {
                return ctx->device->pipeline_cumsum_f32;
            }
        }
        return nullptr;
    case GGML_OP_SOLVE_TRI:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {

            vk_solve_tri_pipeline_state solve_tri_pipeline_state(src0->ne[0], src1->ne[0]);

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                auto it = ctx->device->pipeline_solve_tri_f32.find(solve_tri_pipeline_state);
                if (it != ctx->device->pipeline_solve_tri_f32.end()) {
                    pipeline = it->second;
                } else {
                    ctx->device->pipeline_solve_tri_f32[solve_tri_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            return pipeline;
        }
        return nullptr;
    case GGML_OP_ARGMAX:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_argmax_f32;
        }
        return nullptr;
    case GGML_OP_COUNT_EQUAL:
        if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I64) {
            return ctx->device->pipeline_count_equal_i32;
        }
        return nullptr;
    case GGML_OP_IM2COL:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_im2col_f32;
        }
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_im2col_f32_f16;
        }
        return nullptr;
    case GGML_OP_IM2COL_3D:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_im2col_3d_f32;
        }
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_im2col_3d_f32_f16;
        }
        return nullptr;
    case GGML_OP_TIMESTEP_EMBEDDING:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_timestep_embedding_f32;
        }
        return nullptr;
    case GGML_OP_CONV_TRANSPOSE_1D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_conv_transpose_1d_f32;
        }
        return nullptr;
    case GGML_OP_COL2IM_1D:
        switch (src0->type) {
            case GGML_TYPE_F32:  return ctx->device->pipeline_col2im_1d_f32;
            case GGML_TYPE_F16:  return ctx->device->pipeline_col2im_1d_f16;
            case GGML_TYPE_BF16: return ctx->device->pipeline_col2im_1d_bf16;
            default:             return nullptr;
        }
    case GGML_OP_POOL_1D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pool1d_f32;
        }
        return nullptr;
    case GGML_OP_POOL_2D:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            // Wide windows use neighboring subgroup lanes to read neighboring inputs.
            if (ctx->device->pipeline_pool2d_subgroup_f32 && dst->op_params[1] >= int32_t(ctx->device->subgroup_size)) {
                return ctx->device->pipeline_pool2d_subgroup_f32;
            }
            return ctx->device->pipeline_pool2d_f32;
        }
        return nullptr;
    case GGML_OP_RWKV_WKV6:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rwkv_wkv6_f32;
        }
        return nullptr;
    case GGML_OP_RWKV_WKV7:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rwkv_wkv7_f32;
        }
        return nullptr;
    case GGML_OP_GATED_LINEAR_ATTN:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_gated_linear_attn_f32;
        }
        return nullptr;
    case GGML_OP_LIGHTNING_INDEXER:
        // only the k type selects a pipeline, the other types are fixed by ggml_lightning_indexer()
        if (ggml_vk_lightning_indexer_k_type_supported(src1->type)) {
            return ctx->device->pipeline_lightning_indexer_f32[src1->type];
        }
        return nullptr;
    case GGML_OP_GATED_DELTA_NET:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            const uint32_t S_v = dst->src[2]->ne[0];
            const uint32_t kda = (dst->src[3]->ne[0] == (int64_t)S_v) ? 1 : 0;
            uint32_t si;
            switch (S_v) {
                case 32:  si = 0; break;
                case 64:  si = 1; break;
                case 128: si = 2; break;
                default: return nullptr;
            }
            return ctx->device->pipeline_gated_delta_net[si][kda];
        }
        return nullptr;
    case GGML_OP_SSM_SCAN:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            const uint32_t d_state = src0->ne[0];
            if (d_state == 128) {
                return ctx->device->pipeline_ssm_scan_f32_d128;
            } else if (d_state == 256) {
                return ctx->device->pipeline_ssm_scan_f32_d256;
            }
        }
        return nullptr;
    case GGML_OP_SSM_CONV:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            switch (ctx->num_additional_fused_ops) {
                case 0:  return ctx->device->pipeline_ssm_conv_f32;
                case 1:  return ctx->device->pipeline_ssm_conv_silu_f32;
                case 2:  return ctx->device->pipeline_ssm_conv_bias_silu_f32;
                default: return nullptr;
            }
        }
        return nullptr;
    case GGML_OP_SSM_CONV_BACK:
        // ggmlR extension: backward of ssm_conv.
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_ssm_conv_back_f32;
        }
        return nullptr;
    case GGML_OP_SSM_SCAN_BACK:
        // ggmlR extension: backward of ssm_scan. One pipeline per supported
        // d_state, matching the forward; supports_op has already rejected any
        // other width and any device without float atomics.
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            const uint32_t d_state = (uint32_t)src0->ne[0];
            if (d_state == 128) {
                return ctx->device->pipeline_ssm_scan_back_f32_d128;
            } else if (d_state == 256) {
                return ctx->device->pipeline_ssm_scan_back_f32_d256;
            }
        }
        return nullptr;
    case GGML_OP_OUT_PROD:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_out_prod_f32;
        }
        return nullptr;
    case GGML_OP_GET_ROWS_BACK:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_get_rows_back_f32;
        }
        return nullptr;
    case GGML_OP_IM2COL_BACK:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_im2col_back_f32;
        }
        return nullptr;
    case GGML_OP_FLASH_ATTN_BACK:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_flash_attn_back_f32;
        }
        return nullptr;
    case GGML_OP_CROSS_ENTROPY_LOSS:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return src0->ne[0] > 1024 ? ctx->device->pipeline_cross_entropy_loss_f32_wg512 : ctx->device->pipeline_cross_entropy_loss_f32;
        }
        return nullptr;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_cross_entropy_loss_back_f32;
        }
        return nullptr;
    case GGML_OP_OPT_STEP_ADAMW:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_opt_step_adamw_f32;
        }
        return nullptr;
    case GGML_OP_OPT_STEP_SGD:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_opt_step_sgd_f32;
        }
        return nullptr;
    case GGML_OP_LEAKY_RELU:
        if (src0->type == dst->type &&
            (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16)) {
            return ctx->device->pipeline_leaky_relu[dst->type == GGML_TYPE_F16];
        }
        return nullptr;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            uint32_t K = dst->ne[2]; // Cout
            uint32_t NPQ = dst->ne[3] * dst->ne[1] * dst->ne[0]; // N * OH * OW
            vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, K, NPQ);

            bool transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;
            uint32_t KW = (uint32_t)src0->ne[0];
            uint32_t KH = (uint32_t)src0->ne[1];
            uint32_t s0 = (uint32_t)(ggml_get_op_params_i32(dst, 0));
            uint32_t s1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 1) : s0;
            uint32_t p0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 2) : 0;
            uint32_t p1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 3) : 0;
            uint32_t d0 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 4) : 1;
            uint32_t d1 = !transpose ? (uint32_t)ggml_get_op_params_i32(dst, 5) : 1;

            // tile-aligned shapes let the shader skip bounds checks
            const uint32_t Cin = (uint32_t)src1->ne[2];
            const uint32_t CRS = Cin * KW * KH;
            const uint32_t BS_K   = vk_conv_block_sizes[shape].K;
            const uint32_t BS_CRS = vk_conv_block_sizes[shape].CRS;
            const uint32_t BS_NPQ = vk_conv_block_sizes[shape].NPQ;
            const uint32_t aligned = ((K   % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv2d_pipeline_state conv2d_pipeline_state(s0, s1, p0, p1, d0, d1, KW, KH, aligned);

            std::map<vk_conv2d_pipeline_state, vk_pipeline> *pipelines = nullptr;
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // cm1 uses float16_t shared memory — only correct for F16 kernels.
            // For F32 kernels, cm1 silently downcasts inputs to F16 causing ~0.01 numerical error.
            const bool use_cm1 = ctx->device->coopmat_support && !ctx->device->coopmat2
                                  && src0->type == GGML_TYPE_F16;
#else
            const bool use_cm1 = false;
#endif
            // cm1 pipelines are compiled only into CONV_SHAPE_128x128 slot (fixed 64x64 tiles).
            const vk_conv_shapes cm1_shape = CONV_SHAPE_128x128;
            if (op == GGML_OP_CONV_2D) {
                if (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv2d_f32[shape];
                } else if (src0->type == GGML_TYPE_F16) {
                    pipelines = use_cm1 ? &ctx->device->pipeline_conv2d_f16_f32_cm1[cm1_shape]
                                        : &ctx->device->pipeline_conv2d_f16_f32[shape];
                }
            } else if (op == GGML_OP_CONV_TRANSPOSE_2D) {
                if (src0->type == GGML_TYPE_F32) {
                    pipelines = &ctx->device->pipeline_conv_transpose_2d_f32[shape];
                } else if (src0->type == GGML_TYPE_F16) {
                    pipelines = use_cm1 ? &ctx->device->pipeline_conv_transpose_2d_f16_f32_cm1[cm1_shape]
                                        : &ctx->device->pipeline_conv_transpose_2d_f16_f32[shape];
                }
            }

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                auto it = pipelines->find(conv2d_pipeline_state);
                if (it != pipelines->end()) {
                    pipeline = it->second;
                } else {
                    (*pipelines)[conv2d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            return pipeline;
        }
        return nullptr;
    case GGML_OP_CONV_2D_DW:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            if (ggml_is_contiguous(src1)) {
                return ctx->device->pipeline_conv2d_dw_whcn_f32;
            } else if (ggml_is_contiguous_channels(src1)) {
                return ctx->device->pipeline_conv2d_dw_cwhn_f32;
            }
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            if (ggml_is_contiguous(src1)) {
                return ctx->device->pipeline_conv2d_dw_whcn_f16_f32;
            } else if (ggml_is_contiguous_channels(src1)) {
                return ctx->device->pipeline_conv2d_dw_cwhn_f16_f32;
            }
        }
        return nullptr;
    case GGML_OP_CONV_3D:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            const uint32_t OC = (uint32_t)ggml_get_op_params_i32(dst, 11);
            const uint32_t IC = (uint32_t)ggml_get_op_params_i32(dst, 9);
            const uint32_t N  = (uint32_t)ggml_get_op_params_i32(dst, 10);
            const uint32_t NPQ = N * dst->ne[2] * dst->ne[1] * dst->ne[0];
            const vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, OC, NPQ);

            const uint32_t KW = (uint32_t)src0->ne[0];
            const uint32_t KH = (uint32_t)src0->ne[1];
            const uint32_t KD = (uint32_t)src0->ne[2];
            const uint32_t s0 = (uint32_t)ggml_get_op_params_i32(dst, 0);
            const uint32_t s1 = (uint32_t)ggml_get_op_params_i32(dst, 1);
            const uint32_t s2 = (uint32_t)ggml_get_op_params_i32(dst, 2);
            const uint32_t p0 = (uint32_t)ggml_get_op_params_i32(dst, 3);
            const uint32_t p1 = (uint32_t)ggml_get_op_params_i32(dst, 4);
            const uint32_t p2 = (uint32_t)ggml_get_op_params_i32(dst, 5);
            const uint32_t d0 = (uint32_t)ggml_get_op_params_i32(dst, 6);
            const uint32_t d1 = (uint32_t)ggml_get_op_params_i32(dst, 7);
            const uint32_t d2 = (uint32_t)ggml_get_op_params_i32(dst, 8);

            const uint32_t CRS = IC * KW * KH * KD;
            const uint32_t BS_K   = vk_conv_block_sizes[shape].K;
            const uint32_t BS_CRS = vk_conv_block_sizes[shape].CRS;
            const uint32_t BS_NPQ = vk_conv_block_sizes[shape].NPQ;
            const uint32_t aligned = ((OC  % BS_K   == 0) &&
                                      (CRS % BS_CRS == 0) &&
                                      (NPQ % BS_NPQ == 0)) ? 1u : 0u;

            vk_conv3d_pipeline_state conv3d_pipeline_state(s0, s1, s2, p0, p1, p2, d0, d1, d2, KW, KH, KD, aligned);

            std::map<vk_conv3d_pipeline_state, vk_pipeline> *pipelines = nullptr;
            if (src0->type == GGML_TYPE_F32) {
                pipelines = &ctx->device->pipeline_conv3d_f32[shape];
            } else if (src0->type == GGML_TYPE_F16) {
                pipelines = &ctx->device->pipeline_conv3d_f16_f32[shape];
            } else {
                return nullptr;
            }

            vk_pipeline pipeline = nullptr;

            {
                std::lock_guard<std::mutex> guard(ctx->device->compile_mutex);
                auto it = pipelines->find(conv3d_pipeline_state);
                if (it != pipelines->end()) {
                    pipeline = it->second;
                } else {
                    (*pipelines)[conv3d_pipeline_state] = pipeline = std::make_shared<vk_pipeline_struct>();
                }
            }

            return pipeline;
        }
        return nullptr;
    case GGML_OP_ADD1:
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_add1_f16_f16;
        }
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_add1_f16_f32;
        }
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_add1_f32_f32;
        }
        return nullptr;
    case GGML_OP_ARANGE:
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_arange_f32;
        }
        return nullptr;
    case GGML_OP_REL_POS_BIAS:
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rel_pos_bias_f32;
        }
        return nullptr;
    case GGML_OP_FILL:
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_fill_f32;
        }
        if (dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_fill_f16;
        }
        return nullptr;
    default:
        return nullptr;
    }

    GGML_UNUSED(src2);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_unary_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}



template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_sum_rows_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_pad_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_im2col_3d_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.misalign_offsets = (a_offset << 16) | d_offset;

    GGML_UNUSED(src0);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_binary_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t b_offset = get_misalign_bytes(ctx, src1) / ggml_type_size(src1->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    GGML_ASSERT(a_offset <= 0xFFFF);
    GGML_ASSERT(b_offset <= 0xFF);
    GGML_ASSERT(d_offset <= 0xFF);

    p.misalign_offsets = (a_offset << 16) | (b_offset << 8) | d_offset;

    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

template <> inline void init_pushconst_tensor_offsets(ggml_backend_vk_context * ctx, vk_op_upscale_push_constants &p, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst) {
    const uint32_t a_offset = get_misalign_bytes(ctx, src0) / ggml_type_size(src0->type);
    const uint32_t d_offset = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    p.a_offset = a_offset;
    p.d_offset = d_offset;

    GGML_UNUSED(src1);
    GGML_UNUSED(src2);
    GGML_UNUSED(src3);
}

