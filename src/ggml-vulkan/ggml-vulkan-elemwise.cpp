template<typename PC>
static void ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, const ggml_tensor * src3, ggml_tensor * dst, ggml_op op, PC&& pc, vk_pipeline pipeline_override = nullptr) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    if (src1 != nullptr) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    if (src2 != nullptr) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    if (src3 != nullptr) {
        std::cerr << "), (" << src3 << ", name=" << src3->name << ", type=" << src3->type << ", ne0=" << src3->ne[0] << ", ne1=" << src3->ne[1] << ", ne2=" << src3->ne[2] << ", ne3=" << src3->ne[3] << ", nb0=" << src3->nb[0] << ", nb1=" << src3->nb[1] << ", nb2=" << src3->nb[2] << ", nb3=" << src3->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3];
    std::cerr << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ROWS || op == GGML_OP_CPY || op == GGML_OP_CONCAT || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(dst->buffer != nullptr);
    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const bool use_src1 = src1 != nullptr;
    const uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const uint64_t ne13 = use_src1 ? src1->ne[3] : 0;

    const bool use_src2 = src2 != nullptr;
    const bool use_src3 = src3 != nullptr;

    init_pushconst_fastdiv(pc);

    vk_pipeline pipeline;
    if (pipeline_override) {
        pipeline = pipeline_override;
    } else {
        pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    }

    if (pipeline == nullptr) {
        // ggmlR: through the ggml log, not std::cerr -- cerr is a null stream in
        // this build (ggml-vulkan-device.cpp), so the op and types were lost and
        // the abort said only "fatal error".
        GGML_LOG_ERROR("ggml_vulkan: Error: Missing op: %s for %s%s%s to %s\n",
                       ggml_op_name(op), ggml_type_name(src0->type),
                       src1 != nullptr ? " and " : "",
                       src1 != nullptr ? ggml_type_name(src1->type) : "",
                       ggml_type_name(dst->type));
        GGML_ABORT("ggml_vulkan: missing pipeline for %s (%s%s%s -> %s)",
                   ggml_op_name(op), ggml_type_name(src0->type),
                   src1 != nullptr ? ", " : "",
                   src1 != nullptr ? ggml_type_name(src1->type) : "",
                   ggml_type_name(dst->type));
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0, true);
    vk_subbuffer src1_buf = use_src1 ? ggml_vk_tensor_subbuffer(ctx, src1, true) : vk_subbuffer{};
    vk_subbuffer src2_buf = use_src2 ? ggml_vk_tensor_subbuffer(ctx, src2, true) : vk_subbuffer{};
    vk_subbuffer src3_buf = use_src3 ? ggml_vk_tensor_subbuffer(ctx, src3, true) : vk_subbuffer{};
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, true);

    // Compute misalignment offset for descriptors and store it in in push constants.
    init_pushconst_tensor_offsets(ctx, pc, src0, src1, src2, src3, dst);

    std::array<uint32_t, 3> elements;

    switch (op) {
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM_BACK:
    case GGML_OP_NORM_BACK:   // DIVERGENCE from upstream: LayerNorm backward
    case GGML_OP_L2_NORM:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SOFT_MAX_BACK:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_CUMSUM:
    case GGML_OP_MEAN:
    case GGML_OP_ARGMAX:
        {
            const uint32_t nr = ggml_nrows(src0);
            if (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } else if (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } else {
                elements = { nr, 1, 1 };
            }
        } break;
    case GGML_OP_SOLVE_TRI:
        {
            uint32_t nr = (uint32_t)(ne02 * ne03 * src0->ne[4]);
            if (nr > 262144) {
                elements = { 512, 512, CEIL_DIV(nr, 262144) };
            } else if (nr > 512) {
                elements = { 512, CEIL_DIV(nr, 512), 1 };
            } else {
                elements = { nr, 1, 1 };
            }
        }
        break;
    case GGML_OP_RMS_NORM:
        if (ctx->do_add_rms_partials) {
            // Run one element per thread, 128 threads per workgroup
            elements = { (uint32_t)CEIL_DIV(ne00, 128), 1, 1 };
        } else {
            // z encodes dim3*dim4; shader splits samp_flat = samp4*ne03 + samp
            elements = { (uint32_t)ne01, (uint32_t)ne02, (uint32_t)(ne03 * src0->ne[4]) };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
        }
        break;

    case GGML_OP_SUM:
        // We use GGML_OP_SUM_ROWS with 1 row.
        elements = { 1, 1, 1 };
        break;
    case GGML_OP_GROUP_NORM:
        {
            const uint32_t num_groups = dst->op_params[0];
            elements = { num_groups * (uint32_t)src0->ne[3], 1, 1 };
        } break;
    case GGML_OP_DIAG_MASK_INF:
        elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
        break;
    case GGML_OP_ROPE:
    case GGML_OP_ROPE_BACK:
        {
            uint32_t nrows = (uint32_t)ggml_nrows(src0);
            uint32_t z = 1;
            if (nrows > ctx->device->properties.limits.maxComputeWorkGroupCount[0]) {
                z = CEIL_DIV(nrows, 32768);
                nrows = 32768;
            }
            elements = { nrows, (uint32_t)ne00, z };

        } break;
    case GGML_OP_GET_ROWS:
        elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12 * ne13) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
        break;
    case GGML_OP_ARGSORT:
        GGML_ASSERT(0);
        break;
    case GGML_OP_IM2COL:
        {
            const bool is_2D = dst->op_params[6] == 1;

            const uint32_t IC = src1->ne[is_2D ? 2 : 1];

            const uint32_t KH = is_2D ? src0->ne[1] : 1;
            const uint32_t KW =         src0->ne[0];

            const uint32_t OH = is_2D ? dst->ne[2] : 1;
            const uint32_t OW =         dst->ne[1];

            const uint32_t batch = src1->ne[is_2D ? 3 : 2];

            elements = { OW * KW * KH, OH, batch * IC };
            elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
            elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
        } break;
    case GGML_OP_IM2COL_3D:
        {
            const uint32_t IC = ((const uint32_t *)(dst->op_params))[9];

            const uint32_t N  = ne13 / IC;

            const uint32_t KD = ne02;
            const uint32_t KH = ne01;
            const uint32_t KW = ne00;

            const uint32_t OD = dst->ne[3] / N;
            const uint32_t OH = dst->ne[2];
            const uint32_t OW = dst->ne[1];

            const uint32_t IC_KD_KH_KW = IC*KD*KH*KW;
            const uint32_t N_OD_OH = N*OD*OH;

            elements = { IC_KD_KH_KW, OW, N_OD_OH };
            elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
        } break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        {
            const uint32_t dim = dst->op_params[0];
            uint32_t half_ceil = (dim + 1) / 2;
            elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
        } break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        {
            elements = {uint32_t(src0->ne[1]), 1, 1}; // parallelize in {Cout, 1, 1}
        } break;
    case GGML_OP_COL2IM_1D:
        {
            elements = { uint32_t(dst->ne[0]), uint32_t(dst->ne[1]), 1 };
        } break;
    case GGML_OP_POOL_1D:
        {
            const uint32_t N = dst->ne[3] * dst->ne[2];
            const uint32_t OC = dst->ne[1];
            const uint32_t OL = dst->ne[0];
            elements = { N * OC * OL, 1, 1};
        } break;
    case GGML_OP_POOL_2D:
        {
            // elements counts workgroups here (wg_denoms {1, 1, 1}); split over x and y
            // so that large outputs stay under maxComputeWorkGroupCount[0]. The x extent is
            // capped at 65535 (NVIDIA's limit) so every driver takes the same grid and the
            // y > 1 path is exercised everywhere, not only where the limit is small
            // (DIVERGENCE: 0.27 divides by the raw limit, which is UINT32_MAX on RADV).
            const uint32_t outputs = ggml_nelements(dst);
            const uint32_t outputs_per_workgroup = pipeline == ctx->device->pipeline_pool2d_subgroup_f32 ? 128 / ctx->device->subgroup_size : 512;
            const uint32_t workgroups = CEIL_DIV(outputs, outputs_per_workgroup);
            const uint32_t max_groups_x = std::min(ctx->device->properties.limits.maxComputeWorkGroupCount[0], 65535u);
            const uint32_t groups_y = CEIL_DIV(workgroups, max_groups_x);
            const uint32_t groups_x = CEIL_DIV(workgroups, groups_y);
            elements = { groups_x, groups_y, 1 };
        } break;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        if constexpr (std::is_same_v<PC, vk_op_conv2d_push_constants>) {
            const uint32_t NPQ = pc.N * pc.OH * pc.OW;
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            const bool use_cm1_dispatch = ctx->device->coopmat_support && !ctx->device->coopmat2
                                          && src0->type == GGML_TYPE_F16;
#else
            const bool use_cm1_dispatch = false;
#endif
            // cm1 uses fixed 64x64 tiles; scalar path uses shape-selected block sizes.
            const uint32_t npq_block_size = use_cm1_dispatch ? 64u
                                          : vk_conv_block_sizes[ggml_vk_conv_select_shape(ctx, pc.Cout, NPQ)].NPQ;
            const uint32_t NPQ_blocks = CEIL_DIV(NPQ, npq_block_size);

            elements = { pc.Cout, NPQ_blocks, 1 };
            if (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } else {
            GGML_ABORT("invalid push constant type for CONV_2D");
        }
        break;
    case GGML_OP_CONV_3D:
        if constexpr (std::is_same_v<PC, vk_op_conv3d_push_constants>) {
            const uint32_t NPQ = pc.N * pc.OD * pc.OH * pc.OW;
            const vk_conv_shapes shape = ggml_vk_conv_select_shape(ctx, pc.OC, NPQ);
            const uint32_t NPQ_blocks = CEIL_DIV(NPQ, vk_conv_block_sizes[shape].NPQ);

            elements = { pc.OC, NPQ_blocks, 1 };
            if (elements[1] > 512) {
                elements[2] = CEIL_DIV(elements[1], 512);
                elements[1] = 512;
            }
        } else {
            GGML_ABORT("invalid push constant type for CONV_3D");
        }
        break;
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_DIV:
    case GGML_OP_MUL:
    case GGML_OP_ADD1:
    case GGML_OP_ARANGE:
    case GGML_OP_FILL:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_SIN:
    case GGML_OP_COS:
    case GGML_OP_LOG:
    case GGML_OP_TRI:
    case GGML_OP_DIAG:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_PAD_REFLECT_1D:
    case GGML_OP_ROLL:
    case GGML_OP_REPEAT:
    case GGML_OP_REPEAT_BACK:
    case GGML_OP_CPY:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_UNARY:
    case GGML_OP_GLU:
    case GGML_OP_CONV_2D_DW:
    // ggmlR: silu_back.comp/gelu_back.comp index z*262144 + y*512 + x; the
    // default {ne, 1, 1} grid trips the x workgroup limit above 512*65535 elements.
    case GGML_OP_SILU_BACK:
    case GGML_OP_GELU_BACK:
        {
            uint32_t ne = ggml_nelements(dst);
            if (op == GGML_OP_CPY && ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                // Convert from number of logical elements to 2- or 4-byte units.
                ne /= ggml_blck_size(src0->type);
                if ((ggml_type_size(src0->type) % 4) == 0) {
                    ne *= ggml_type_size(src0->type) / 4;
                } else {
                    ne *= ggml_type_size(src0->type) / 2;
                }
            }
            if (op == GGML_OP_CONCAT && ggml_is_quantized(dst->type)) {
                ne = ne / ggml_blck_size(dst->type) * ggml_type_size(dst->type) / ggml_vk_concat_unit_size(dst->type);
            }
            // copy_to_quant has block size of 32, and each thread does QUANT_K elements.
            // Splitting into 512x512xZ wouldn't work well since each workgroup does 1024 elements.
            // So divide by block size here before splitting into 512x512 groups.
            if (op == GGML_OP_CPY && !ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
                ne = CEIL_DIV(ne, ggml_blck_size(dst->type));
            }
            if (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } else if (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } else {
                elements = { ne, 1, 1 };
            }

            if (pipeline == ctx->device->pipeline_cpy_transpose_02_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_02_16) {
                // 32x32 tiles over dims 0 and 2; dim1 and dim3 are the batch
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[2], 32);
                elements[2] = (uint32_t)(dst->ne[1]*dst->ne[3]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.maxComputeWorkGroupCount[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
            } else if (pipeline == ctx->device->pipeline_cpy_transpose_32 ||
                pipeline == ctx->device->pipeline_cpy_transpose_16) {
                // 32x32 tiles
                elements[0] = (uint32_t)CEIL_DIV(dst->ne[0], 32);
                elements[1] = (uint32_t)CEIL_DIV(dst->ne[1], 32);
                elements[2] = (uint32_t)(dst->ne[2]*dst->ne[3]*dst->ne[4]);
                elements[0] = std::min(elements[0], ctx->device->properties.limits.maxComputeWorkGroupCount[0]);
                elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
                elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);
            }
        } break;
    case GGML_OP_ADD_ID:
        {
            elements = { (uint32_t)ne01, (uint32_t)ne02, 1 };
        } break;
    case GGML_OP_SET_ROWS:
        {
            uint32_t ne = ggml_nelements(src0);
            if (ggml_is_quantized(dst->type)) {
                // quants run 32 threads each doing QUANT_K elements
                ne = CEIL_DIV(ne, 32 * ggml_blck_size(dst->type));
            } else {
                // scalar types do one element per thread, running 512 threads
                ne = CEIL_DIV(ne, 512);
            }
            if (ne > 262144) {
                elements = { 512, 512, CEIL_DIV(ne, 262144) };
            } else if (ne > 512) {
                elements = { 512, CEIL_DIV(ne, 512), 1 };
            } else {
                elements = { ne, 1, 1 };
            }
        }
        break;
    case GGML_OP_SCATTER_ELEMENTS:
        {
            /* src0=updates, total threads = row_size * n_idx, workgroup = 256 */
            uint32_t ne = (uint32_t)ggml_nelements(src0);
            ne = CEIL_DIV(ne, 256);
            if (ne > 262144) {
                elements = { 256, 256, CEIL_DIV(ne, 65536) };
            } else if (ne > 256) {
                elements = { 256, CEIL_DIV(ne, 256), 1 };
            } else {
                elements = { ne, 1, 1 };
            }
        }
        break;
    case GGML_OP_SSM_CONV:
        {
            const uint32_t nr  = src0->ne[1];
            const uint32_t n_t = dst->ne[1];
            const uint32_t n_s = dst->ne[2];
            elements = { nr, n_t, n_s };
        }
        break;
    case GGML_OP_OUT_PROD:
        {
            // dst extent; the pipeline's wg_denoms {32, 32, 1} turn it into one
            // workgroup per 32 x 32 dst tile, with dims 2 and 3 folded into z.
            // Folding keeps z within the workgroup-count limit that matters on
            // NVIDIA (65535 for x, far lower than AMD's), since dims 2/3 are
            // batch-like and small while ne0/ne1 are the wide ones.
            elements = { (uint32_t)dst->ne[0], (uint32_t)dst->ne[1],
                         (uint32_t)(dst->ne[2] * dst->ne[3]) };
        }
        break;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        {
            // One workgroup per row; the shader reduces each row in shared
            // memory, so the grid is the row count rather than the element one.
            elements = { (uint32_t)ggml_nrows(dst), 1, 1 };
        }
        break;
    default:
        elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
        break;
    }

    if (op == GGML_OP_ADD || op == GGML_OP_RMS_NORM) {
        vk_subbuffer a_buf = src0_buf;
        if (ctx->do_add_rms_partials) {
            a_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_add_rms_partials, ctx->prealloc_size_add_rms_partials_offset);
        }
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, src1_buf, dst_buf, a_buf }, pc, elements);
    } else if (op == GGML_OP_GLU) {
        // Empty src1 is possible in glu, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc, elements);
    } else if (op == GGML_OP_SOFT_MAX) {
        // Empty src1 and src2 is possible in soft_max, but the shader needs a buffer
        vk_subbuffer subbuf1 = use_src1 ? src1_buf : src0_buf;
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, subbuf2, dst_buf }, pc, elements);
    } else if (op == GGML_OP_ROPE || op == GGML_OP_ROPE_BACK) {
        // Empty src2 and src3 is possible in rope, but the shader needs a buffer
        vk_subbuffer subbuf2 = use_src2 ? src2_buf : src0_buf;
        vk_subbuffer subbuf3 = use_src3 ? src3_buf : src0_buf;
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, subbuf2, dst_buf, subbuf3 }, pc, elements);
    } else if (op == GGML_OP_IM2COL || op == GGML_OP_IM2COL_3D) {
        if (ctx->device->shader_int64 && ctx->device->buffer_device_address) {
            // buffer device address path doesn't use dst buffer
            dst_buf.size = 1;
        }
        // im2col uses only src1 and dst buffers
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src1_buf, dst_buf }, pc, elements);
    } else if (op == GGML_OP_COUNT_EQUAL) {
        // count_equal assumes that destination buffer is initialized with zeroes
        ggml_vk_buffer_memset_async(subctx, dst_buf.buffer, dst_buf.offset, 0, dst_buf.size);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } else if (op == GGML_OP_OPT_STEP_SGD) {
        // OPT_STEP_SGD works on src0, it does not need dst
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf }, pc, elements);
    } else if (use_src3) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, src3_buf, dst_buf }, pc, elements);
    } else if (use_src2) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, src2_buf, dst_buf }, pc, elements);
    } else if (use_src1) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
    } else {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, dst_buf }, pc, elements);
    }
}

static void ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GET_ROWS,
        vk_op_binary_push_constants_init(src0, src1, dst));
}

static void ggml_vk_acc(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    {
        vk_op_binary_push_constants p = vk_op_binary_push_constants_init(src0, src1, dst, 0, 0.0f, 0.0f, offset);
        // acc uses custom strides stored in op_params
        p.nb01 = (uint32_t)nb1; p.nb02 = (uint32_t)nb2;
        p.nb21 = (uint32_t)nb1; p.nb22 = (uint32_t)nb2;
        ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, dst->op, std::move(p));
    }
}

static void ggml_vk_multi_add(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor *first_node = cgraph->nodes[node_idx];
    const ggml_tensor *dst = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];

    // Make a list of all the tensors used by the op.
    // Last element of the list is the dest tensor.
    const ggml_tensor *tensors[MAX_PARAMETER_COUNT];
    uint32_t num_srcs = ctx->num_additional_fused_ops + 2;
    uint32_t num_tensors = num_srcs + 1;
    GGML_ASSERT(num_tensors + ctx->do_add_rms_partials <= MAX_PARAMETER_COUNT);

    tensors[0] = first_node->src[0];
    tensors[1] = first_node->src[1];
    for (int32_t i = 0; i < ctx->num_additional_fused_ops; ++i) {
        // check whether the previous result is src[0] or src[1]
        if (cgraph->nodes[node_idx + i] == cgraph->nodes[node_idx + i + 1]->src[0]) {
            tensors[i+2] = cgraph->nodes[node_idx + i + 1]->src[1];
        } else {
            tensors[i+2] = cgraph->nodes[node_idx + i + 1]->src[0];
        }
    }
    tensors[num_srcs] = dst;

    vk_op_multi_add_push_constants pc;
    pc.ne20 = (uint32_t)dst->ne[0];
    pc.ne21 = (uint32_t)dst->ne[1];
    pc.ne22 = (uint32_t)dst->ne[2];
    pc.ne23 = (uint32_t)(dst->ne[3] * dst->ne[4]); // collapse 5D → 4D

    for (uint32_t i = 0; i < num_tensors; ++i) {
        const ggml_tensor *t = tensors[i];
        pc.nb[i][0] = (uint32_t)t->nb[0] / sizeof(float);
        pc.nb[i][1] = (uint32_t)t->nb[1] / sizeof(float);
        pc.nb[i][2] = (uint32_t)t->nb[2] / sizeof(float);
        pc.nb[i][3] = (uint32_t)t->nb[3] / sizeof(float);
    }
    pc.rms_partials = ctx->do_add_rms_partials;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, tensors[0], tensors[1], nullptr, dst, dst->op);

    if (pipeline == nullptr) {
        std::cerr << "ggml_vulkan: Error: Missing multi_add";
        GGML_ABORT("fatal error");
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    ggml_backend_vk_buffer_context * buf_ctx[MAX_PARAMETER_COUNT];
    vk_buffer buf[MAX_PARAMETER_COUNT];
    size_t offset[MAX_PARAMETER_COUNT];
    bool uma[MAX_PARAMETER_COUNT];

    for (uint32_t i = 0; i < num_tensors; ++i) {
        buf_ctx[i] = (ggml_backend_vk_buffer_context *)tensors[i]->buffer->context;
        buf[i] = nullptr;
        offset[i] = 0;
        uma[i] = false;

        if (ctx->device->uma) {
            ggml_vk_host_get(ctx->device, tensors[i]->data, buf[i], offset[i]);
            uma[i] = buf[i] != nullptr;
        }
        if (!uma[i]) {
            buf[i] = buf_ctx[i]->dev_buffer;
            offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
        }
        GGML_ASSERT(buf[i] != nullptr);
    }
    // If any remaining descriptors are unused, just point them at src[0]
    for (uint32_t i = num_tensors; i < MAX_PARAMETER_COUNT; ++i) {
        buf[i] = buf[0];
        offset[i] = 0;
    }
    if (ctx->do_add_rms_partials) {
        buf[num_tensors] = ctx->prealloc_add_rms_partials;
        offset[num_tensors] = ctx->prealloc_size_add_rms_partials_offset;
    }

    std::array<uint32_t, 3> elements;

    uint32_t ne = ggml_nelements(dst);
    if (ne > 262144) {
        elements = { 512, 512, CEIL_DIV(ne, 262144) };
    } else if (ne > 512) {
        elements = { 512, CEIL_DIV(ne, 512), 1 };
    } else {
        elements = { ne, 1, 1 };
    }

    static_assert(MAX_PARAMETER_COUNT == 12);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {
            ggml_vk_subbuffer(ctx, buf[0], offset[0]),
            ggml_vk_subbuffer(ctx, buf[1], offset[1]),
            ggml_vk_subbuffer(ctx, buf[2], offset[2]),
            ggml_vk_subbuffer(ctx, buf[3], offset[3]),
            ggml_vk_subbuffer(ctx, buf[4], offset[4]),
            ggml_vk_subbuffer(ctx, buf[5], offset[5]),
            ggml_vk_subbuffer(ctx, buf[6], offset[6]),
            ggml_vk_subbuffer(ctx, buf[7], offset[7]),
            ggml_vk_subbuffer(ctx, buf[8], offset[8]),
            ggml_vk_subbuffer(ctx, buf[9], offset[9]),
            ggml_vk_subbuffer(ctx, buf[10], offset[10]),
            ggml_vk_subbuffer(ctx, buf[11], offset[11]),
        }, pc, elements);
}

static void ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_ADD,
        vk_op_binary_push_constants_init(src0, src1, dst, 0, 0.0f, 0.0f, ctx->do_add_rms_partials));
}

static void ggml_vk_sub(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SUB,
        vk_op_binary_push_constants_init(src0, src1, dst));
}

static void ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_MUL,
        vk_op_binary_push_constants_init(src0, src1, dst));
}

static int ggml_vk_unary_mul_op_index(ggml_unary_op op) {
    switch (op) {
        case GGML_UNARY_OP_GELU:     return 0;
        case GGML_UNARY_OP_SIGMOID:  return 1;
        case GGML_UNARY_OP_SILU:     return 2;
        case GGML_UNARY_OP_SOFTPLUS: return 3;
        default:                     return -1;
    }
}

static void ggml_vk_unary_mul(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * unary = cgraph->nodes[node_idx];
    ggml_tensor * mul = cgraph->nodes[node_idx + 1];

    // unary on src1 that tiles into src0
    const bool op_on_b = mul->src[1] == unary &&
                         !ggml_are_same_shape(unary->src[0], mul->src[0]) &&
                         ggml_can_repeat(unary, mul->src[0]);

    const ggml_tensor * src0 = op_on_b ? mul->src[0] : unary->src[0];
    const ggml_tensor * src1 = op_on_b ? unary->src[0] :
        ((mul->src[0] == unary) ? mul->src[1] : mul->src[0]);

    const bool f16 = src0->type == GGML_TYPE_F16;
    const bool norepeat = ggml_are_same_shape(src0, src1);
    const int oi = ggml_vk_unary_mul_op_index(ggml_get_unary_op(unary));
    if (oi < 0) {
        GGML_ABORT("fatal error");
    }
    vk_pipeline pipeline = ctx->device->pipeline_unary_mul[oi][f16][norepeat][op_on_b];

    // ggmlR: 5D push constants (ne*4/nb*4) via the shared init, as for MUL
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, mul, GGML_OP_UNARY,
        vk_op_binary_push_constants_init(src0, src1, mul, (uint32_t) ggml_nelements(op_on_b ? mul : src0)), pipeline);
}

static void ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_DIV,
        vk_op_binary_push_constants_init(src0, src1, dst));
}

static void ggml_vk_add_id(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t src2_type_size = ggml_type_size(src2->type);

    ggml_vk_op_f32<vk_op_add_id_push_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_ADD_ID, {
        (uint32_t)dst->ne[0],
        (uint32_t)dst->ne[1],
        (uint32_t)src0->nb[1] / src0_type_size,
        (uint32_t)src0->nb[2] / src0_type_size,
        (uint32_t)src1->nb[1] / src1_type_size,
        (uint32_t)src2->nb[1] / src2_type_size,
    });
}

static void ggml_vk_op_f32_wkv(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const vk_op_rwkv_wkv6_push_constants&& pc, int version) {
    GGML_ASSERT(version == 6 || version == 7);
    int num_srcs = version == 6 ? 6 : 7;

    for (int i = 0; i < num_srcs; i++) {
        GGML_ASSERT(!ggml_is_quantized(dst->src[i]->type));
    }

    GGML_ASSERT(dst->buffer != nullptr);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    for (int i = 0; i < num_srcs; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements = {
        (uint32_t)(pc.B * pc.H),
        1,
        1
    };

    if (version == 6) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
            pc, elements);
    } else if (version == 7) {
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
            pc, elements);
    } else {
        // shouldn't happen
        GGML_ASSERT(false);
    }
}

static void ggml_vk_rwkv_wkv6(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const size_t seq_length = dst->src[0]->ne[2];
    const size_t n_embed = dst->ne[0];
    const size_t n_heads = dst->src[0]->ne[1];
    const size_t n_seqs = dst->src[5]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_length,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        6
    );
}

static void ggml_vk_rwkv_wkv7(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const size_t seq_length = dst->src[0]->ne[2];
    const size_t n_embed = dst->ne[0];
    const size_t n_heads = dst->src[0]->ne[1];
    const size_t n_seqs = dst->src[6]->ne[1];

    ggml_vk_op_f32_wkv(
        ctx, subctx, dst,
        {
            (uint32_t)n_seqs,
            (uint32_t)seq_length,
            (uint32_t)n_embed,
            (uint32_t)n_heads,
        },
        7
    );
}

static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_beta  = dst->src[4];

    GGML_ASSERT(dst->buffer != nullptr);

    const uint32_t S_v      = (uint32_t)src_v->ne[0];
    const uint32_t H        = (uint32_t)src_v->ne[1];
    const uint32_t n_tokens = (uint32_t)src_v->ne[2];
    const uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    const uint32_t s_off = S_v * H * n_tokens * n_seqs;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[6] = {};
    for (int i = 0; i < 6; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    const uint32_t sq1 = (uint32_t)(src_q->nb[1] / sizeof(float));
    const uint32_t sq2 = (uint32_t)(src_q->nb[2] / sizeof(float));
    const uint32_t sq3 = (uint32_t)(src_q->nb[3] / sizeof(float));
    const uint32_t sv1 = (uint32_t)(src_v->nb[1] / sizeof(float));
    const uint32_t sv2 = (uint32_t)(src_v->nb[2] / sizeof(float));
    const uint32_t sv3 = (uint32_t)(src_v->nb[3] / sizeof(float));
    const uint32_t sb1 = (uint32_t)(src_beta->nb[1] / sizeof(float));
    const uint32_t sb2 = (uint32_t)(src_beta->nb[2] / sizeof(float));
    const uint32_t sb3 = (uint32_t)(src_beta->nb[3] / sizeof(float));

    const uint32_t neq1 = (uint32_t)src_q->ne[1];
    const uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    const float scale = 1.0f / sqrtf((float)S_v);
    const vk_op_gated_delta_net_push_constants pc = {
        H, n_tokens, n_seqs, s_off,
        sq1, sq2, sq3,
        sv1, sv2, sv3,
        sb1, sb2, sb3,
        neq1, rq3,
        scale,
        (uint32_t) ggml_get_op_params_i32(dst, 0) // K: state snapshot slots
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},
        pc, { H, n_seqs, S_v });
}

static void ggml_vk_ssm_scan(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const ggml_tensor * src3 = dst->src[3];
    const ggml_tensor * src4 = dst->src[4];
    const ggml_tensor * src5 = dst->src[5];

    GGML_ASSERT(dst->buffer != nullptr);

    const uint32_t head_dim = src0->ne[1];
    const uint32_t n_head = src1->ne[1];
    const uint32_t n_group = src4->ne[1];
    const uint32_t n_tok = src1->ne[2];
    const uint32_t n_seq = src1->ne[3];

    bool is_mamba2 = (src3->nb[1] == sizeof(float));
    GGML_ASSERT(is_mamba2);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    const vk_op_ssm_scan_push_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src3->nb[1],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        (uint32_t)s_off,
        n_head, head_dim, n_group, n_tok,
        n_seq, (uint32_t) ggml_get_op_params_i32(dst, 0)
    };

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[7] = {};
    for (int i = 0; i < 7 && dst->src[i] != nullptr; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    std::array<uint32_t, 3> elements;

    const uint32_t d_state = src0->ne[0];
    uint32_t num_subgroups = d_state / ctx->device->subgroup_size;
    const uint32_t num_workgroups_x = CEIL_DIV(n_head * head_dim, num_subgroups);
    const uint32_t num_workgroups_y = n_seq;
    elements = { num_workgroups_x, num_workgroups_y, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], src_buf[6], dst_buf},
        pc, elements);
}

static void ggml_vk_ssm_conv(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * conv = cgraph->nodes[node_idx];
    const ggml_tensor * src0 = conv->src[0];
    const ggml_tensor * src1 = conv->src[1];

    // Pick the destination tensor (last node in the fused chain) and the optional bias.
    // Fusion modes: 0 = ssm_conv, 1 = ssm_conv+silu, 2 = ssm_conv+add(bias)+silu.
    ggml_tensor * dst = conv;
    const ggml_tensor * bias = nullptr;
    if (ctx->num_additional_fused_ops == 1) {
        dst = cgraph->nodes[node_idx + 1]; // silu
    } else if (ctx->num_additional_fused_ops == 2) {
        ggml_tensor * add = cgraph->nodes[node_idx + 1];
        bias = (add->src[0] == conv) ? add->src[1] : add->src[0];
        dst = cgraph->nodes[node_idx + 2]; // silu
    }

    // The shader always declares 4 bindings; bind src0 as a dummy when bias isn't fused.
    const ggml_tensor * src2 = bias ? bias : src0;

    ggml_vk_op_f32<vk_op_ssm_conv_push_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_SSM_CONV, {
        (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
        (uint32_t)src1->nb[1],
        (uint32_t)dst->nb[0], (uint32_t)dst->nb[1], (uint32_t)dst->nb[2],
        (uint32_t)src1->ne[0],
        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)dst->ne[1],
        (uint32_t)dst->ne[2],
    });
}

// ggmlR extension: backward of GGML_OP_SSM_CONV.
//
// One invocation per channel; the shader loops over tokens and sequences
// itself, which is what keeps every write private to a channel and lets the
// kernel run without atomics. The output is the packed [d_sx | d_c] pair the
// builder allocates, so the split point goes in as an element offset.
static void ggml_vk_ssm_conv_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // sx   {ncs, nr, n_s}
    const ggml_tensor * src1 = dst->src[1];  // c    {nc, nr}
    const ggml_tensor * src2 = dst->src[2];  // grad {nr, n_t, n_s}

    GGML_ASSERT(dst->buffer != nullptr);

    const uint32_t nc  = (uint32_t)src1->ne[0];
    const uint32_t ncs = (uint32_t)src0->ne[0];
    const uint32_t nr  = (uint32_t)src0->ne[1];
    const uint32_t n_t = (uint32_t)src2->ne[1];
    const uint32_t n_s = (uint32_t)src2->ne[2];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const vk_op_ssm_conv_back_push_constants pc = {
        (uint32_t)src0->nb[1], (uint32_t)src0->nb[2],
        (uint32_t)src1->nb[1],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        nc, ncs, nr, n_t, n_s,
        (uint32_t)ggml_nelements(src0),
    };

    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf0 = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src_buf1 = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer src_buf2 = ggml_vk_tensor_subbuffer(ctx, src2);

    // Work items, not workgroups: ggml_vk_dispatch_pipeline divides by the
    // pipeline's workgroup size itself. Passing the already-divided count here
    // launches 1/256 of the channels and silently leaves the rest at zero.
    const std::array<uint32_t, 3> elements = { nr, 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf0, src_buf1, src_buf2, dst_buf},
        pc, elements);
}

// ggmlR extension: backward of GGML_OP_SSM_SCAN.
//
// One workgroup per (sequence, head, channel), d_state invocations wide, time
// walked sequentially inside the shader. The forward keeps no intermediate
// states, so the kernel replays the recurrence into a scratch buffer first --
// (nt+1) states per (head, channel), far too much for shared memory at Mamba
// sizes, hence prealloc_x rather than a shared array.
//
// The output buffer is zeroed first: four of the six gradients accumulate with
// atomicAdd, so whatever the allocator left behind would otherwise be added to.
static void ggml_vk_ssm_scan_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // s    {d_state, nr, nh, ns+}
    const ggml_tensor * src1 = dst->src[1];  // x    {nr, nh, nt, ns}
    const ggml_tensor * src2 = dst->src[2];  // dt   {nh, nt, ns}
    const ggml_tensor * src3 = dst->src[3];  // A    {1, nh}
    const ggml_tensor * src4 = dst->src[4];  // B    {d_state, ng, nt, ns}
    const ggml_tensor * src5 = dst->src[5];  // C    {d_state, ng, nt, ns}

    GGML_ASSERT(dst->buffer != nullptr);

    const uint32_t d_state = (uint32_t)src0->ne[0];
    const uint32_t nr      = (uint32_t)src0->ne[1];
    const uint32_t nh      = (uint32_t)src1->ne[1];
    const uint32_t ng      = (uint32_t)src4->ne[1];
    const uint32_t nt      = (uint32_t)src1->ne[2];
    const uint32_t ns      = (uint32_t)src1->ne[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    // Element offsets of the six gradients inside the packed output, in the
    // order ggml_ssm_scan takes its inputs. These must match the CPU kernel's
    // layout exactly -- ggml-graph.c views each part by the same arithmetic.
    const uint32_t n_s_el  = d_state * nr * nh * ns;
    const uint32_t n_x_el  = (uint32_t)ggml_nelements(src1);
    const uint32_t n_dt_el = (uint32_t)ggml_nelements(src2);
    const uint32_t n_A_el  = (uint32_t)ggml_nelements(src3);
    const uint32_t n_B_el  = (uint32_t)ggml_nelements(src4);

    const uint32_t d_s_off  = 0;
    const uint32_t d_x_off  = d_s_off  + n_s_el;
    const uint32_t d_dt_off = d_x_off  + n_x_el;
    const uint32_t d_A_off  = d_dt_off + n_dt_el;
    const uint32_t d_B_off  = d_A_off  + n_A_el;
    const uint32_t d_C_off  = d_B_off  + n_B_el;

    // The incoming gradient is outputs followed by final states, like the
    // forward result it belongs to.
    const uint32_t g_s_off = n_x_el;

    // Scratch: (nt + 1) states for every (sequence, head, channel).
    const size_t scratch_size =
        (size_t)ns * nh * nr * (nt + 1) * d_state * sizeof(float);

    if (ctx->prealloc_size_x < scratch_size) {
        ctx->prealloc_size_x = scratch_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    if (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    // Zero the destination before the atomics start accumulating into it.
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    ggml_vk_buffer_memset_async(subctx, dst_buf.buffer, dst_buf.offset, 0,
                                ggml_nbytes(dst));
    ggml_vk_sync_buffers(ctx, subctx);

    const vk_op_ssm_scan_back_push_constants pc = {
        (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)src2->nb[1], (uint32_t)src2->nb[2],
        (uint32_t)src4->nb[2], (uint32_t)src4->nb[3],
        (uint32_t)src5->nb[2], (uint32_t)src5->nb[3],
        nh, nr, ng, nt, ns,
        d_s_off, d_x_off, d_dt_off, d_A_off, d_B_off, d_C_off,
        g_s_off,
    };

    vk_subbuffer src_buf[8] = {};
    for (int i = 0; i < 8 && dst->src[i] != nullptr; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }
    vk_subbuffer scratch_buf = { ctx->prealloc_x, 0, VK_WHOLE_SIZE };

    // The grid is in work items: one workgroup per (head, channel) on x, and
    // the pipeline's workgroup is d_state wide, so x counts nh*nr*d_state.
    const std::array<uint32_t, 3> elements = { nh * nr * d_state, ns, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5],
         src_buf[6], src_buf[7], dst_buf, scratch_buf},
        pc, elements);

    ctx->prealloc_x_need_sync = true;
}

// ggmlR extension: GGML_OP_OUT_PROD on the GPU.
//
// Both gradients of ggml_mul_mat() are built from out_prod, so without this
// every training step of every dense layer fell back to the CPU. Strides are
// passed as byte counts; the shader divides by sizeof(float), which is what
// lets it handle a transposed src1 (the second mul_mat gradient passes
// ggml_transpose(grad)) and the dim-2/3 broadcast used by grouped-query
// attention.
static void ggml_vk_out_prod(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // Broadcast factors, matching the CPU kernel's dps2/dps3.
    const uint32_t dps2 = (uint32_t)(dst->ne[2] / src0->ne[2]);
    const uint32_t dps3 = (uint32_t)(dst->ne[3] / src0->ne[3]);

    ggml_vk_op_f32<vk_op_out_prod_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_OUT_PROD, {
        (uint32_t)dst->ne[0], (uint32_t)dst->ne[1],
        (uint32_t)dst->ne[2], (uint32_t)dst->ne[3],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->nb[1], (uint32_t)src0->nb[2], (uint32_t)src0->nb[3],
        (uint32_t)src1->nb[0], (uint32_t)src1->nb[1],
        (uint32_t)src1->nb[2], (uint32_t)src1->nb[3],
        (uint32_t)dst->nb[1], (uint32_t)dst->nb[2], (uint32_t)dst->nb[3],
        dps2, dps3,
    });
}

// ggmlR extension: GGML_OP_GET_ROWS_BACK on the GPU.
//
// src0 is the gradient of the gathered rows, src1 the token ids. The result is
// the gradient of the whole table, so it is zeroed first and then accumulated
// into with atomics -- a row nothing points at must come out zero, not stale.
static void ggml_vk_get_rows_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];   // d(gathered rows)
    const ggml_tensor * src1 = dst->src[1];   // ids

    GGML_ASSERT(dst->buffer != nullptr);

    const uint32_t nc = (uint32_t) src0->ne[0];
    const uint32_t nr = (uint32_t) ggml_nelements(src1);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, nullptr, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const vk_op_get_rows_back_push_constants pc = {
        nc,
        nr,
        (uint32_t)(dst->nb[1]  / sizeof(float)),
        (uint32_t)(src0->nb[1] / sizeof(float)),
        nr * nc,
    };

    vk_subbuffer src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);

    ggml_vk_buffer_memset_async(subctx, dst_buf.buffer, dst_buf.offset, 0, ggml_nbytes(dst));
    ggml_vk_sync_buffers(ctx, subctx);

    const std::array<uint32_t, 3> elements = { nr * nc, 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf, ids_buf, dst_buf}, pc, elements);
}

// ggmlR extension: GGML_OP_IM2COL_BACK on the GPU.
//
// src0 is the gradient of the im2col output, src1 the convolution kernel (used
// for its shape only). One invocation per output element -- see the shader for
// why that direction needs no atomics.
static void ggml_vk_im2col_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];   // d(im2col output)
    const ggml_tensor * src1 = dst->src[1];   // kernel

    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t p0 = ggml_get_op_params_i32(dst, 2);
    const int32_t p1 = ggml_get_op_params_i32(dst, 3);
    const int32_t d0 = ggml_get_op_params_i32(dst, 4);
    const int32_t d1 = ggml_get_op_params_i32(dst, 5);
    const bool is_2D = ggml_get_op_params_i32(dst, 6) == 1;

    // dst carries the image's shape: [IW, IH, IC, N] in 2D, [IW, IC, N] in 1D.
    const uint32_t N  = (uint32_t)(is_2D ? dst->ne[3] : dst->ne[2]);
    const uint32_t IC = (uint32_t)(is_2D ? dst->ne[2] : dst->ne[1]);
    const uint32_t IH = (uint32_t)(is_2D ? dst->ne[1] : 1);
    const uint32_t IW = (uint32_t) dst->ne[0];

    const uint32_t KH = (uint32_t)(is_2D ? src1->ne[1] : 1);
    const uint32_t KW = (uint32_t) src1->ne[0];

    const uint32_t OH = (uint32_t)(is_2D ? src0->ne[2] : 1);
    const uint32_t OW = (uint32_t) src0->ne[1];

    const uint32_t ne_total = N * IC * IH * IW;

    ggml_vk_op_f32<vk_op_im2col_back_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_IM2COL_BACK, {
        N, IC, IH, IW, OH, OW, KH, KW,
        s0, s1, p0, p1, d0, d1,
        is_2D ? 1u : 0u,
        ne_total,
    });
}

// ggmlR extension: backward of GGML_OP_FLASH_ATTN_EXT on the GPU.
//
// Produces dq | dk | dv packed into one buffer, the layout ggml-graph.c views
// each part out of. dk and dv accumulate with atomics because every query row
// of a kv head contributes to them, so the destination is zeroed first.
static void ggml_vk_flash_attn_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * d    = dst->src[4];

    GGML_ASSERT(dst->buffer != nullptr);

    const uint32_t DK = (uint32_t) q->ne[0];
    const uint32_t DV = (uint32_t) v->ne[0];
    const uint32_t N  = (uint32_t) q->ne[1];
    const uint32_t M  = (uint32_t) k->ne[1];
    const uint32_t H  = (uint32_t) q->ne[2];
    const uint32_t B  = (uint32_t) q->ne[3];

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    // Packed offsets, in elements, matching ggml_flash_attn_back()'s own
    // GGML_PAD arithmetic.
    const uint32_t pad_el = GGML_MEM_ALIGN / sizeof(float);
    auto pad = [pad_el](uint32_t n) { return ((n + pad_el - 1) / pad_el) * pad_el; };

    const uint32_t dq_off = 0;
    const uint32_t dk_off = dq_off + pad((uint32_t) ggml_nelements(q));
    const uint32_t dv_off = dk_off + pad((uint32_t) ggml_nelements(k));

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, v, dst, dst->op);

    // Profiling: GGMLR_FAB_PROFILE=1..4 swaps in a copy of the shader truncated
    // after that phase, so the cost of each phase is the difference between
    // consecutive runs. Unset (the normal case) uses the real pipeline. The
    // truncated ones produce WRONG gradients by design -- timing only.
    {
        const char * prof = getenv("GGMLR_FAB_PROFILE");
        if (prof != nullptr && prof[0] >= '1' && prof[0] <= '4') {
            pipeline = ctx->device->pipeline_flash_attn_back_f32_prof[prof[0] - '0'];
        }
    }
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    // Strides as element counts: the shader indexes float arrays directly.
    const uint32_t fs = sizeof(float);

    const vk_op_flash_attn_back_push_constants pc = {
        DK, DV, N, M, H, B,
        H / (uint32_t) k->ne[2], B / (uint32_t) k->ne[3],
        (uint32_t)(q->nb[1]/fs), (uint32_t)(q->nb[2]/fs), (uint32_t)(q->nb[3]/fs),
        (uint32_t)(k->nb[1]/fs), (uint32_t)(k->nb[2]/fs), (uint32_t)(k->nb[3]/fs),
        (uint32_t)(v->nb[1]/fs), (uint32_t)(v->nb[2]/fs), (uint32_t)(v->nb[3]/fs),
        (uint32_t)(d->nb[1]/fs), (uint32_t)(d->nb[2]/fs), (uint32_t)(d->nb[3]/fs),
        mask ? (uint32_t)(mask->nb[1]/sizeof(ggml_fp16_t)) : 0u,
        mask ? (uint32_t)(mask->nb[2]/sizeof(ggml_fp16_t)) : 0u,
        mask ? (uint32_t)(mask->nb[3]/sizeof(ggml_fp16_t)) : 0u,
        mask ? (uint32_t) mask->ne[2] : 1u,
        mask ? (uint32_t) mask->ne[3] : 1u,
        dq_off, dk_off, dv_off,
        scale,
        mask ? 1u : 0u,
    };

    vk_subbuffer q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, d);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    // The mask binding must be valid even when there is no mask; point it at q.
    vk_subbuffer m_buf = mask ? ggml_vk_tensor_subbuffer(ctx, mask) : q_buf;

    // Zero the destination: dk and dv are accumulated into it with atomicAdd,
    // and dq is written per row, so nothing may carry over from a previous run.
    ggml_vk_buffer_memset_async(subctx, dst_buf.buffer, dst_buf.offset, 0, ggml_nbytes(dst));
    ggml_vk_sync_buffers(ctx, subctx);

    // One workgroup per query row; the pipeline's workgroup is BLOCK_SIZE wide.
    const std::array<uint32_t, 3> elements = { N * H * B * 128u, 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {q_buf, k_buf, v_buf, m_buf, d_buf, dst_buf}, pc, elements);
}

// ggmlR extension: GGML_OP_CROSS_ENTROPY_LOSS_BACK on the GPU.
//
//     dst[row] = (softmax(logits[row]) - labels[row]) * grad[0] / nrows
//
// src[0] is the scalar loss gradient, src[1] the forward logits, src[2] the
// labels -- the order ggml_cross_entropy_loss_back() records them in.
static void ggml_vk_cross_entropy_loss_back(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0];
    const ggml_tensor * src0f = dst->src[1];
    const ggml_tensor * src1f = dst->src[2];

    const uint32_t nr = (uint32_t) ggml_nrows(src0f);

    ggml_vk_op_f32<vk_op_cross_entropy_loss_back_push_constants>(ctx, subctx, grad, src0f, src1f, nullptr, dst, GGML_OP_CROSS_ENTROPY_LOSS_BACK, {
        (uint32_t) src0f->ne[0],
        nr,
        1.0f / (float) nr,
    });
}

static void ggml_vk_op_f32_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const vk_op_push_constants&& pc) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * g = dst->src[1];
    const ggml_tensor * gm = dst->src[2];
    const ggml_tensor * gv = dst->src[3];
    const ggml_tensor * p = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(gm->type == GGML_TYPE_F32);
    GGML_ASSERT(gv->type == GGML_TYPE_F32);
    GGML_ASSERT(p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->buffer != nullptr);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(g));
    GGML_ASSERT(ggml_is_contiguous(gm));
    GGML_ASSERT(ggml_is_contiguous(gv));
    GGML_ASSERT(ggml_is_contiguous(p));
    GGML_ASSERT(ggml_are_same_shape(x, g));
    GGML_ASSERT(ggml_are_same_shape(x, gm));
    GGML_ASSERT(ggml_are_same_shape(x, gv));
    GGML_ASSERT(ggml_nelements(p) == 7);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, g, gm, gv, dst, GGML_OP_OPT_STEP_ADAMW);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer g_buf = ggml_vk_tensor_subbuffer(ctx, g);
    vk_subbuffer gm_buf = ggml_vk_tensor_subbuffer(ctx, gm);
    vk_subbuffer gv_buf = ggml_vk_tensor_subbuffer(ctx, gv);
    vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(x), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {x_buf, g_buf, gm_buf, gv_buf, p_buf},
        pc, elements);
}

static void ggml_vk_opt_step_adamw(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32_opt_step_adamw(
        ctx, subctx, dst,
        { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f }
    );
}

static void ggml_vk_opt_step_sgd(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    const size_t n = ggml_nelements(dst->src[0]);

    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_OPT_STEP_SGD, { (uint32_t)n, 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

static void ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    int * op_params = (int *)dst->op_params;

    const uint32_t unit_size = ggml_vk_concat_unit_size(dst->type);
    const uint32_t units_per_block = ggml_type_size(dst->type) / unit_size;
    const uint32_t block_size = ggml_blck_size(dst->type);
    const bool quantized = ggml_is_quantized(dst->type);

    // ggmlR: upstream fills the 4D layout by hand; ours goes through the 5D
    // initializer and then rescales dimension 0 and all strides to storage units.
    vk_op_concat_push_constants pc {vk_op_binary_push_constants_init(src0, src1, dst, 0, 0.0f, 0.0f, op_params[0])};

    // Address dimension 0 in packed storage units; higher strides may be noncontiguous.
    pc.ne00 = src0->ne[0] / block_size * units_per_block;
    pc.ne10 = src1->ne[0] / block_size * units_per_block;
    pc.ne20 =  dst->ne[0] / block_size * units_per_block;
    pc.ne   = pc.ne20 * (uint32_t)dst->ne[1] * (uint32_t)dst->ne[2] * (uint32_t)dst->ne[3] * (uint32_t)dst->ne[4];
    pc.nb00 = quantized ? 1 : src0->nb[0] / unit_size;
    pc.nb10 = quantized ? 1 : src1->nb[0] / unit_size;
    pc.nb20 = quantized ? 1 :  dst->nb[0] / unit_size;
    pc.nb01 = src0->nb[1] / unit_size; pc.nb02 = src0->nb[2] / unit_size; pc.nb03 = src0->nb[3] / unit_size; pc.nb04 = src0->nb[4] / unit_size;
    pc.nb11 = src1->nb[1] / unit_size; pc.nb12 = src1->nb[2] / unit_size; pc.nb13 = src1->nb[3] / unit_size; pc.nb14 = src1->nb[4] / unit_size;
    pc.nb21 =  dst->nb[1] / unit_size; pc.nb22 =  dst->nb[2] / unit_size; pc.nb23 =  dst->nb[3] / unit_size; pc.nb24 =  dst->nb[4] / unit_size;

    ggml_vk_op_f32<vk_op_concat_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONCAT, std::move(pc));
}

static void ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t mode = (uint32_t)ggml_get_op_params_i32(dst, 0);

    GGML_TENSOR_UNARY_OP_LOCALS

    float sf0 = (float)ne0 / ne00;
    float sf1 = (float)ne1 / ne01;
    float sf2 = (float)ne2 / ne02;
    float sf3 = (float)ne3 / ne03;
    float pixel_offset = 0.5f;

    if (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
        pixel_offset = 0.0f;
    }

    ggml_vk_op_f32<vk_op_upscale_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0, 0,
        (uint32_t)ne00, (uint32_t)ne01,
        (uint32_t)nb00 / src0_type_size, (uint32_t)nb01 / src0_type_size, (uint32_t)nb02 / src0_type_size, (uint32_t)nb03 / src0_type_size,
        (uint32_t)ne0, (uint32_t)ne1, (uint32_t)ne2, (uint32_t)(ne3 * dst->ne[4]),  // collapse dim4 into dim3
        sf0, sf1, sf2, sf3, pixel_offset
    });
}

static void ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SCALE, std::move(p));
}

static void ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SQR, vk_op_unary_push_constants_init(src0, dst));
}

static void ggml_vk_sqrt(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SQRT, vk_op_unary_push_constants_init(src0, dst));
}

static void ggml_vk_add1(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_ADD1,
        vk_op_binary_push_constants_init(src0, src1, dst));
}

static void ggml_vk_arange(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_arange(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");

    vk_op_push_constants pc = {
        (uint32_t)ggml_nelements(dst),
        1,
        ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 2),
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, dst, GGML_OP_ARANGE);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, false);

    std::array<uint32_t, 3> elements = { (uint32_t)ggml_nelements(dst), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

static void ggml_vk_fill(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_fill(dst=" << dst << ", ne=" << ggml_nelements(dst) << ")");
    const uint64_t n = ggml_nelements(dst);
    GGML_ASSERT(n > 0);

    vk_op_push_constants pc = {
        (uint32_t)n,
        1,
        ggml_get_op_params_f32(dst, 0),
        0.0f,
        0.0f, 0.0f,
    };

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, dst, GGML_OP_FILL);
    GGML_ASSERT(pipeline != nullptr);

    // Split the task distribution to 2D to avoid exceeding maxComputeWorkGroupCount
    const uint32_t total_wg = CEIL_DIV(n, pipeline->wg_denoms[0]);
    const uint32_t wg_x = std::min(total_wg, ctx->device->properties.limits.maxComputeWorkGroupCount[0]);
    const uint32_t wg_y = CEIL_DIV(total_wg, wg_x);
    GGML_ASSERT(wg_y <= ctx->device->properties.limits.maxComputeWorkGroupCount[1]);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst, false);

    std::array<uint32_t, 3> elements = { wg_x * pipeline->wg_denoms[0], wg_y * pipeline->wg_denoms[1], 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { dst_buf }, pc, elements);
}

static void ggml_vk_sin(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SIN, vk_op_unary_push_constants_init(src0, dst));
}

static void ggml_vk_cos(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_COS, vk_op_unary_push_constants_init(src0, dst));
}

static void ggml_vk_log(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_LOG, vk_op_unary_push_constants_init(src0, dst));
}

static void ggml_vk_tri(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_TRI, std::move(p));
}

static void ggml_vk_diag(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ggml_nelements(dst));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_DIAG, std::move(p));
}

static void ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = ggml_get_op_params_f32(dst, 0);
    p.param2 = ggml_get_op_params_f32(dst, 1);

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CLAMP, std::move(p));
}

static void ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_pad_push_constants p = vk_op_pad_push_constants_init(src0, dst);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_PAD, std::move(p));
}

static void ggml_vk_roll(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const int32_t s0 = ggml_get_op_params_i32(dst, 0);
    const int32_t s1 = ggml_get_op_params_i32(dst, 1);
    const int32_t s2 = ggml_get_op_params_i32(dst, 2);
    const int32_t s3 = ggml_get_op_params_i32(dst, 3);
    const uint32_t s01_packed = ((s0 + 0x8000) << 16) | (s1 + 0x8000);
    const uint32_t s23_packed = ((s2 + 0x8000) << 16) | (s3 + 0x8000);

    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    memcpy(&p.param1, &s01_packed, sizeof(float));
    memcpy(&p.param2, &s23_packed, sizeof(float));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_ROLL, std::move(p));
}

static void ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_REPEAT, std::move(p));
}

static void ggml_vk_repeat_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ggml_nelements(dst));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_REPEAT_BACK, std::move(p));
}

static void ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ne = (uint32_t)ggml_nelements(src0);
    if (ggml_is_quantized(src0->type) && ggml_is_quantized(dst->type)) {
        // Convert from number of logical elements to 2- or 4-byte units.
        ne /= ggml_blck_size(src0->type);
        if ((ggml_type_size(src0->type) % 4) == 0) {
            ne *= ggml_type_size(src0->type) / 4;
        } else {
            ne *= ggml_type_size(src0->type) / 2;
        }
    }

    const bool is_5d = (src0->ne[4] > 1 || dst->ne[4] > 1) && !ggml_is_contiguous(src0) && !ggml_is_contiguous(dst);
    if (is_5d && !ctx->device->supports_256_push_constants) {
        r_ggml_error("ggmlR: 5D tensor copy requires maxPushConstantsSize >= 256 bytes. "
                     "Please update your GPU driver (Mesa 25.0+ for AMD/Intel, 550+ for NVIDIA).");
    }
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ne);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CPY, std::move(p));
}

static void ggml_vk_set_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // Skip empty skip_rows operations. For most ops the empty check at the start
    // of ggml_vk_build_graph is sufficient, but set_rows can have a nonempty dst
    // with empty srcs.
    if (ggml_is_empty(src0) || ggml_is_empty(src1)) {
        return;
    }

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SET_ROWS,
        vk_op_binary_push_constants_init(src0, src1, dst));
}

static void ggml_vk_scatter_elements(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * data    = dst->src[0];
    const ggml_tensor * updates = dst->src[1];
    const ggml_tensor * indices = dst->src[2];

    /* TEMPORARY probe: does supports_op promise more than this function can
     * keep?  It validates only `reduction` and dst type, never src[0] -- yet
     * step 1 below copies device-to-device out of src[0], which requires it to
     * live in a Vulkan buffer.  Printed before the first subbuffer call so the
     * line survives a fault raised inside it. */
    if (getenv("GGMLR_TRACE_SCATTER")) {
        /* GGML_LOG_INFO, never fprintf(stderr, ...): r_ggml_compat.h redirects
         * `stderr` to a sentinel pointer AND `fprintf` to a wrapper that knows
         * about it.  In this translation unit only the first half applies, so a
         * direct fprintf hands the sentinel (0x1) to libc as a FILE* and
         * segfaults inside vfprintf -- measured, with the probe's own format
         * string in frame #0. */
        GGML_LOG_INFO(
            "[scatter] dst='%s' ne=[%lld,%lld,%lld,%lld] cont=%d\n"
            "          data='%s' buf=%p buft='%s' cont=%d type=%s ne=[%lld,%lld,%lld,%lld]\n"
            "          upd ='%s' buf=%p buft='%s' cont=%d type=%s\n"
            "          idx ='%s' buf=%p buft='%s' cont=%d type=%s\n",
            dst->name,
            (long long)dst->ne[0], (long long)dst->ne[1],
            (long long)dst->ne[2], (long long)dst->ne[3],
            ggml_is_contiguous(dst) ? 1 : 0,
            data->name, (void *)data->buffer,
            data->buffer ? ggml_backend_buft_name(data->buffer->buft) : "(null)",
            ggml_is_contiguous(data) ? 1 : 0, ggml_type_name(data->type),
            (long long)data->ne[0], (long long)data->ne[1],
            (long long)data->ne[2], (long long)data->ne[3],
            updates->name, (void *)updates->buffer,
            updates->buffer ? ggml_backend_buft_name(updates->buffer->buft) : "(null)",
            ggml_is_contiguous(updates) ? 1 : 0, ggml_type_name(updates->type),
            indices->name, (void *)indices->buffer,
            indices->buffer ? ggml_backend_buft_name(indices->buffer->buft) : "(null)",
            ggml_is_contiguous(indices) ? 1 : 0, ggml_type_name(indices->type));
    }

    /* Step 1: copy data → dst buffer */
    {
        vk_subbuffer src_sb = ggml_vk_tensor_subbuffer(ctx, data, true);
        vk_subbuffer dst_sb = ggml_vk_tensor_subbuffer(ctx, dst,  true);

        /* TEMPORARY probe, part 2: are the subbuffer offsets rounded down?
         *
         * ggml_vk_tensor_subbuffer(allow_misalign=true) rounds `offset` DOWN to
         * minStorageBufferOffsetAlignment and adds the same slack to `size`.
         * The copy below then pairs those rounded offsets with a RAW length
         * (ggml_nbytes), which is the one mismatch worth measuring: every other
         * copy_async caller in the tree pairs raw offsets with a raw length
         * (ggml-vulkan-graph.cpp:1193).  If raw != rounded here, the copy runs
         * off the end of its region by exactly that difference. */
        if (getenv("GGMLR_TRACE_SCATTER")) {
            const size_t src_raw = vk_tensor_offset(data) + data->view_offs;
            const size_t dst_raw = vk_tensor_offset(dst)  + dst->view_offs;
            GGML_LOG_INFO(
                "[scatter-off] align=%zu nbytes=%zu\n"
                "          src raw=%zu sb.off=%zu delta=%zu sb.size=%zu\n"
                "          dst raw=%zu sb.off=%zu delta=%zu sb.size=%zu\n",
                (size_t)ctx->device->properties.limits.minStorageBufferOffsetAlignment,
                ggml_nbytes(data),
                src_raw, src_sb.offset, src_raw - src_sb.offset, src_sb.size,
                dst_raw, dst_sb.offset, dst_raw - dst_sb.offset, dst_sb.size);
        }

        ggml_vk_buffer_copy_async(subctx, dst_sb.buffer, dst_sb.offset,
                                           src_sb.buffer, src_sb.offset,
                                           ggml_nbytes(data));
        /* Barrier: copy must finish before scatter shader reads/writes dst */
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            {},
            { vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite,
                                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite) },
            {}, {});
    }

    /* Step 2: run scatter shader — pass updates as src0, indices as src1 */
    {
        vk_op_binary_push_constants pc =
            vk_op_binary_push_constants_init(updates, indices, dst, 0, 0.0f, 0.0f,
                                             ((int32_t *)dst->op_params)[1]);
        if (getenv("GGMLR_TRACE_SCATTER")) {
            /* What the shader will actually see.  `total` is recomputed here
             * exactly as the shader computes it, because that product -- not
             * ggml_nelements(updates) -- is what bounds its loop, and a
             * mismatch between the two is invisible from either side alone. */
            GGML_LOG_INFO(
                "[scatter-pc] ne=%u ne00..03=%u,%u,%u,%u total=%u "
                "ne20..22=%u,%u,%u axis=%d misalign=0x%x\n",
                pc.ne, pc.ne00, pc.ne01, pc.ne02, pc.ne03,
                pc.ne00 * pc.ne01 * pc.ne02 * pc.ne03,
                pc.ne20, pc.ne21, pc.ne22, pc.param3, pc.misalign_offsets);
        }
        /* std::move: ggml_vk_op_f32 takes PC&&, so a named local has to be
         * handed over as an rvalue. */
        ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, updates, indices, nullptr, nullptr, dst,
                                                    GGML_OP_SCATTER_ELEMENTS, std::move(pc));
    }
}

static void ggml_vk_silu_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SILU_BACK, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

static void ggml_vk_gelu_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GELU_BACK, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

static void ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_NORM, std::move(p));
}

static void ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const int * int_op_params = (const int *)dst->op_params;
    const float * float_op_params = (const float *)dst->op_params;

    const uint32_t num_groups = int_op_params[0];
    const float eps = float_op_params[1];
    const uint32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);

    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_GROUP_NORM, { group_size, 0, eps, 0.0f, 0.0f, 0.0f });
}

static uint32_t ggml_vk_rms_num_partials(ggml_backend_vk_context * ctx, const ggml_tensor *node) {
    const uint32_t ne = (uint32_t)node->ne[0];
    const uint32_t denom = ctx->device->pipeline_add_rms[0][0][0]->wg_denoms[0];
    const uint32_t num_partials = CEIL_DIV(ne, denom);
    return num_partials;
}

static uint32_t ggml_vk_rms_partials_size(ggml_backend_vk_context * ctx, const ggml_tensor *node) {
    const uint32_t num_partials = ggml_vk_rms_num_partials(ctx, node);
    const uint32_t num_bytes = ROUNDUP_POW2(num_partials * sizeof(uint32_t), ctx->device->partials_binding_alignment);
    return num_bytes;
}

static vk_op_rope_push_constants ggml_vk_make_rope_constants(const ggml_tensor *dst, const ggml_tensor *src0, const bool has_ff, bool backprop, const uint32_t set_rows_stride) {
    const int n_dims        = ((const int32_t *) dst->op_params)[1];
    const int mode          = ((const int32_t *) dst->op_params)[2];
    const int n_offs        = ((const int32_t *) dst->op_params)[15];
    // const int n_ctx         = ((const int32_t *) dst->op_params)[3];
    const int n_ctx_orig    = ((const int32_t *) dst->op_params)[4];
    const float freq_base   = ((const float *)   dst->op_params)[5];
    const float freq_scale  = ((const float *)   dst->op_params)[6];
    const float ext_factor  = ((const float *)   dst->op_params)[7];
    const float attn_factor = ((const float *)   dst->op_params)[8];
    const float beta_fast   = ((const float *)   dst->op_params)[9];
    const float beta_slow   = ((const float *)   dst->op_params)[10];
    int sections[4] {};
    if (mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(sections, (const int32_t *) dst->op_params + 11, sizeof(int)*4);
    }

    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    uint32_t nb01 = src0->nb[1] / ggml_type_size(src0->type);
    uint32_t nb02 = src0->nb[2] / ggml_type_size(src0->type);
    uint32_t nb03 = src0->nb[3] / ggml_type_size(src0->type);

    uint32_t nb11 = dst->nb[1] / ggml_type_size(dst->type);
    uint32_t nb12 = dst->nb[2] / ggml_type_size(dst->type);
    uint32_t nb13 = dst->nb[3] / ggml_type_size(dst->type);

    vk_op_rope_push_constants rope {
        (uint32_t)mode, (uint32_t)ggml_nrows(src0), (uint32_t)n_dims, (uint32_t)n_offs, freq_scale,
        freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]}, theta_scale, has_ff,
        { sections[0], sections[1], sections[2], sections[3] }, is_imrope, backprop, set_rows_stride,

        (uint32_t)src0->ne[0],
        (uint32_t)src0->ne[1],
        (uint32_t)src0->ne[2],
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        0, 0, // a_offset, d_offset filled in by init_pushconst_tensor_offsets
    };

    return rope;
}

static void ggml_vk_rms_norm_finish(ggml_backend_vk_context * ctx, const ggml_tensor * src0) {
    if (ctx->do_add_rms_partials_offset_calculation) {
        ctx->prealloc_size_add_rms_partials_offset += ggml_vk_rms_partials_size(ctx, src0);
        ctx->do_add_rms_partials = false;
        ctx->do_add_rms_partials_offset_calculation = false;
    }
}

static void ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx, float * op_params) {
    ggml_tensor * rms = cgraph->nodes[node_idx];
    const ggml_tensor * src0 = rms->src[0];

    if (ctx->fused_rms_norm_mode == RMS_NORM_VIEW_SET_ROWS) {
        GGML_ASSERT(ctx->num_additional_fused_ops == 2);
        ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];
        const ggml_tensor * indices = set_rows->src[1];
        vk_op_binary_push_constants pc = ggml_vk_rms_norm_push_constants(src0, src0, set_rows, op_params[0], 0);
        init_pushconst_tensor_offsets(ctx, pc, src0, src0, nullptr, nullptr, set_rows);

        vk_pipeline pipeline = set_rows->type == GGML_TYPE_F16 ?
            ctx->device->pipeline_rms_norm_set_rows_f32_f16 : ctx->device->pipeline_rms_norm_set_rows_f32_f32;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_tensor_subbuffer(ctx, src0, true),
                ggml_vk_tensor_subbuffer(ctx, src0, true),
                ggml_vk_tensor_subbuffer(ctx, set_rows, true),
                ggml_vk_tensor_subbuffer(ctx, indices),
            }, pc, {
                (uint32_t)src0->ne[1],
                std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.maxComputeWorkGroupCount[1]),
                // ggmlR: z is flat over dim3*dim4 (5D); rms_norm.comp splits it back.
                std::min((uint32_t)(src0->ne[3] * src0->ne[4]), ctx->device->properties.limits.maxComputeWorkGroupCount[2]),
            });
        ggml_vk_rms_norm_finish(ctx, src0);
        return;
    }

    if (ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD || ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD_MUL) {
        ggml_tensor * mul = cgraph->nodes[node_idx + 1];
        ggml_tensor * add = cgraph->nodes[node_idx + 2];
        const ggml_tensor * weight = mul->src[0] == rms ? mul->src[1] : mul->src[0];
        const ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
        const bool do_post_multiply = ctx->fused_rms_norm_mode == RMS_NORM_MUL_ADD_MUL;
        GGML_ASSERT(ctx->num_additional_fused_ops == (do_post_multiply ? 3 : 2));
        ggml_tensor * dst = do_post_multiply ? cgraph->nodes[node_idx + 3] : add;
        const ggml_tensor * post_scale = do_post_multiply ?
            (dst->src[0] == add ? dst->src[1] : dst->src[0]) : src0;

        const uint32_t num_partials = ctx->do_add_rms_partials ? ggml_vk_rms_num_partials(ctx, dst) : 0;
        vk_op_binary_push_constants pc = ggml_vk_rms_norm_push_constants(src0, weight, dst, op_params[0], num_partials);
        init_pushconst_tensor_offsets(ctx, pc, src0, weight, residual, post_scale, dst);

        vk_pipeline pipeline;
        if (ctx->do_add_rms_partials) {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_add_mul_partials_f32 : ctx->device->pipeline_rms_norm_mul_add_partials_f32;
        } else {
            pipeline = do_post_multiply ?
                ctx->device->pipeline_rms_norm_mul_add_mul_f32 : ctx->device->pipeline_rms_norm_mul_add_f32;
        }
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        if (ctx->do_add_rms_partials) {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, true),
                    ggml_vk_tensor_subbuffer(ctx, weight, true),
                    ggml_vk_tensor_subbuffer(ctx, dst, true),
                    ggml_vk_subbuffer(ctx, ctx->prealloc_add_rms_partials, ctx->prealloc_size_add_rms_partials_offset),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, { (uint32_t)CEIL_DIV(src0->ne[0], 128), 1, 1 });
        } else {
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                {
                    ggml_vk_tensor_subbuffer(ctx, src0, true),
                    ggml_vk_tensor_subbuffer(ctx, weight, true),
                    ggml_vk_tensor_subbuffer(ctx, dst, true),
                    ggml_vk_tensor_subbuffer(ctx, residual),
                    ggml_vk_tensor_subbuffer(ctx, post_scale),
                }, pc, {
                    (uint32_t)src0->ne[1],
                    std::min((uint32_t)src0->ne[2], ctx->device->properties.limits.maxComputeWorkGroupCount[1]),
                    std::min((uint32_t)(src0->ne[3] * src0->ne[4]), ctx->device->properties.limits.maxComputeWorkGroupCount[2]),
                });
        }
        ggml_vk_rms_norm_finish(ctx, src0);
        return;
    }

    ggml_tensor * dst;
    const ggml_tensor * src1;

    if (ctx->fused_rms_norm_mode != RMS_NORM_COUNT) {
        ggml_tensor * mul = cgraph->nodes[node_idx + 1];
        dst = mul;
        src1 = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    } else {
        dst = rms;
        src1 = src0;
    }

    const uint32_t num_partials = ctx->do_add_rms_partials ? ggml_vk_rms_num_partials(ctx, dst) : 0;
    vk_op_binary_push_constants bin = ggml_vk_rms_norm_push_constants(src0, src1, dst, op_params[0], num_partials);

    if (ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE ||
        ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE_VIEW_SET_ROWS) {
        static constexpr uint32_t max_tensors = 7;
        const ggml_tensor *tensors[max_tensors] {};

        ggml_tensor *rms = cgraph->nodes[node_idx + 0];
        ggml_tensor *mul = cgraph->nodes[node_idx + 1];
        ggml_tensor *rope = cgraph->nodes[node_idx + 2];

        ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];

        bool do_set_rows = ctx->fused_rms_norm_mode == RMS_NORM_MUL_ROPE_VIEW_SET_ROWS;
        GGML_ASSERT(ctx->num_additional_fused_ops == (do_set_rows ? 4 : 2));

        tensors[0] = rms->src[0];
        tensors[1] = other_src;
        tensors[2] = mul;
        tensors[3] = rope->src[1]; // pos
        tensors[4] = rope->src[2]; // ff
        tensors[5] = cgraph->nodes[node_idx + ctx->num_additional_fused_ops]; // dst
        tensors[6] = do_set_rows ? tensors[5]->src[1] : nullptr;
        const uint32_t set_rows_stride = do_set_rows ? tensors[5]->nb[1] / ggml_type_size(tensors[5]->type) : 0;

        vk_op_rms_norm_mul_rope_push_constants pc;
        pc.bin = bin;
        pc.rope = ggml_vk_make_rope_constants(rope, rope->src[0], tensors[4] != nullptr, false, set_rows_stride);

        vk_pipeline pipeline = tensors[5]->type == GGML_TYPE_F16 ? ctx->device->pipeline_rms_norm_mul_rope_f32_f16 : ctx->device->pipeline_rms_norm_mul_rope_f32_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

        ggml_backend_vk_buffer_context * buf_ctx[max_tensors];
        vk_buffer buf[max_tensors];
        size_t offset[max_tensors];
        bool uma[max_tensors];

        for (uint32_t i = 0; i < max_tensors; ++i) {
            if (!tensors[i]) {
                // If any remaining descriptors are unused, just point them at src[0]
                buf[i] = buf[0];
                offset[i] = 0;
                continue;
            }
            buf_ctx[i] = (ggml_backend_vk_buffer_context *)tensors[i]->buffer->context;
            buf[i] = nullptr;
            offset[i] = 0;
            uma[i] = false;

            if (ctx->device->uma) {
                ggml_vk_host_get(ctx->device, tensors[i]->data, buf[i], offset[i]);
                uma[i] = buf[i] != nullptr;
            }
            if (!uma[i]) {
                buf[i] = buf_ctx[i]->dev_buffer;
                offset[i] = vk_tensor_offset(tensors[i]) + tensors[i]->view_offs;
            }
            GGML_ASSERT(buf[i] != nullptr);
        }

        // a_offset is unused (the fused path reads from shared memory), but the rope/set_rows dst can be misaligned.
        // Round the binding offset down to the storage buffer alignment; the in-element shift goes in pc.rope.d_offset.
        pc.rope.d_offset = get_misalign_bytes(ctx, tensors[5]) / ggml_type_size(tensors[5]->type);
        offset[5] &= ~(size_t(ctx->device->properties.limits.minStorageBufferOffsetAlignment) - 1);

        std::array<uint32_t, 3> elements;
        elements = { (uint32_t)rms->src[0]->ne[1], (uint32_t)rms->src[0]->ne[2], (uint32_t)(rms->src[0]->ne[3] * rms->src[0]->ne[4]) };
        elements[1] = std::min(elements[1], ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
        elements[2] = std::min(elements[2], ctx->device->properties.limits.maxComputeWorkGroupCount[2]);

        static_assert(max_tensors == 7);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            {
                ggml_vk_subbuffer(ctx, buf[0], offset[0]),
                ggml_vk_subbuffer(ctx, buf[1], offset[1]),
                ggml_vk_subbuffer(ctx, buf[2], offset[2]),
                ggml_vk_subbuffer(ctx, buf[3], offset[3]),
                ggml_vk_subbuffer(ctx, buf[4], offset[4]),
                ggml_vk_subbuffer(ctx, buf[5], offset[5]),
                ggml_vk_subbuffer(ctx, buf[6], offset[6]),
            }, pc, elements);
    } else {
        GGML_ASSERT(ctx->fused_rms_norm_mode == RMS_NORM_MUL || ctx->fused_rms_norm_mode == RMS_NORM_COUNT);
        ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_RMS_NORM, std::move(bin));
    }

    ggml_vk_rms_norm_finish(ctx, src0);
}

static void ggml_vk_rms_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_RMS_NORM_BACK, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

// DIVERGENCE from upstream: LayerNorm backward, see ggml_norm_back().
static void ggml_vk_norm_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_NORM_BACK, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f, 0.0f, 0.0f });
}

static void ggml_vk_l2_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const float * op_params = (const float *)dst->op_params;
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = op_params[0];
    ggml_vk_op_f32<vk_op_unary_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_L2_NORM, std::move(p));
}

static void ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UNARY, vk_op_unary_push_constants_init(src0, dst));
}

static void ggml_vk_xielu(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = op_params[1];
    p.param2 = op_params[2];
    p.param3 = op_params[3];
    p.param4 = op_params[4];
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_UNARY, std::move(p));
}

static void ggml_vk_glu(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const float * op_params_f = (const float *)dst->op_params;

    const bool swapped = (bool)dst->op_params[1];
    const bool split = src1 != nullptr;
    const float alpha = op_params_f[2];
    const float limit = op_params_f[3];

    if (!split) {
        GGML_ASSERT(src0->ne[0] / 2 == dst->ne[0]);
    } else {
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[0] == dst->ne[0]);
        GGML_ASSERT(src0->type == src1->type);
    }

    const uint32_t mode = split ? 2 : (swapped ? 1 : 0);
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = split ? ggml_type_size(src1->type) : src0_type_size;
    const uint32_t dst_type_size  = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_glu_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_GLU,
        {
            (uint32_t)ggml_nelements(dst),
            (uint32_t)src0->ne[0],
            (uint32_t)dst->ne[0],
            mode,
            alpha,
            limit,
            (uint32_t)(src0->nb[0] / src0_type_size),
            (uint32_t)(src0->nb[1] / src0_type_size),
            (uint32_t)(src0->nb[2] / src0_type_size),
            (uint32_t)(src0->nb[3] / src0_type_size),
            (uint32_t)((split ? src1->nb[0] : src0->nb[0]) / src1_type_size),
            (uint32_t)((split ? src1->nb[1] : src0->nb[1]) / src1_type_size),
            (uint32_t)((split ? src1->nb[2] : src0->nb[2]) / src1_type_size),
            (uint32_t)((split ? src1->nb[3] : src0->nb[3]) / src1_type_size),
            (uint32_t)(dst->nb[0] / dst_type_size),
            (uint32_t)(dst->nb[1] / dst_type_size),
            (uint32_t)(dst->nb[2] / dst_type_size),
            (uint32_t)(dst->nb[3] / dst_type_size),
            (uint32_t)dst->ne[1],
            (uint32_t)dst->ne[2],
            0,
            0, 0, 0, 0, 0, 0,
        });
}

static void ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_mask_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_DIAG_MASK_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

static void ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;

    float scale = op_params[0];
    float max_bias = op_params[1];

    const uint32_t ncols =   (uint32_t)src0->ne[0];
    const uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    const uint32_t nrows_y = (uint32_t)src0->ne[1];

    const uint32_t ne12 = src1 ? (uint32_t)(src1->ne[2]) : 0u;
    const uint32_t ne13 = src1 ? (uint32_t)(src1->ne[3] * src1->ne[4]) : 0u;  // collapse dim4
    const uint32_t nb11 = src1 ? (uint32_t)(src1->nb[1] / src1->nb[0]) : 0u;
    const uint32_t nb12 = src1 ? (uint32_t)(src1->nb[2] / src1->nb[0]) : 0u;
    const uint32_t nb13 = src1 ? (uint32_t)(src1->nb[3] / src1->nb[0]) : 0u;

    const uint32_t n_head_kv   = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    vk_op_soft_max_push_constants pc {
        ncols,
        src1 != nullptr ? nrows_y : (uint32_t)0,
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],
        ne12, ne13,
        nb11, nb12, nb13,
        scale, max_bias,
        m0, m1,
        n_head_log2,
        nrows_x,
        src2 != nullptr
    };

    if (ncols <= 16384) {
        ggml_vk_op_f32<vk_op_soft_max_push_constants>(ctx, subctx, src0, src1, src2, nullptr, dst, GGML_OP_SOFT_MAX, std::move(pc));
    } else {

        vk_subbuffer buf_a = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer buf_b = src1 ? ggml_vk_tensor_subbuffer(ctx, src1) : buf_a;
        vk_subbuffer buf_c = src2 ? ggml_vk_tensor_subbuffer(ctx, src2) : buf_a;
        vk_subbuffer buf_d = ggml_vk_tensor_subbuffer(ctx, dst);

        uint32_t elems_per_wg = 128 * 4;
        uint32_t num_wgs = CEIL_DIV(ncols, elems_per_wg);
        size_t tmp_size = num_wgs * nrows_x * sizeof(float);

        if (ctx->prealloc_size_x < tmp_size) {
            ctx->prealloc_size_x = tmp_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_size_y < tmp_size) {
            ctx->prealloc_size_y = tmp_size;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_x_need_sync || ctx->prealloc_y_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_subbuffer buf_x = { ctx->prealloc_x, 0, tmp_size };
        vk_subbuffer buf_y = { ctx->prealloc_y, 0, tmp_size };

        // the partial results overwrite a matmul input converted into prealloc_y
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;

        std::array<uint32_t, 3> elements = { num_wgs, nrows_x, 1 };

        vk_pipeline pipeline1 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large1_f32_f16 : ctx->device->pipeline_soft_max_large1_f32;
        vk_pipeline pipeline2 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large2_f32_f16 : ctx->device->pipeline_soft_max_large2_f32;
        vk_pipeline pipeline3 = src1 && src1->type == GGML_TYPE_F16 ? ctx->device->pipeline_soft_max_large3_f32_f16 : ctx->device->pipeline_soft_max_large3_f32;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline1, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline2, 1);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline3, 1);

        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline1, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline2, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);
        ggml_vk_sync_buffers(ctx, subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline3, { buf_a, buf_b, buf_c, buf_d, buf_x, buf_y }, pc, elements);

        ctx->prealloc_x_need_sync = true;
        ctx->prealloc_y_need_sync = true;
    }
}

static void ggml_vk_soft_max_back(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SOFT_MAX_BACK, { (uint32_t)src0->ne[0], (uint32_t)ggml_nrows(src0), op_params[0], op_params[1], 0.0f, 0.0f });
}

static void ggml_vk_topk_moe(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, int node_idx) {
    topk_moe_mode mode = ctx->fused_topk_moe_mode;
    const bool has_bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS || mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
    ggml_tensor * logits = cgraph->nodes[node_idx + 0]->src[0];
    ggml_tensor * bias = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[node_idx + 2]->src[1] :
                         mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[node_idx + 3]->src[1] :
                                                                   logits;
    ggml_tensor * weights = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];
    ggml_tensor * ids = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? cgraph->nodes[node_idx + 4] :
                        mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? cgraph->nodes[node_idx + 5] :
                        mode == TOPK_MOE_LATE_SOFTMAX             ? cgraph->nodes[node_idx + 1] :
                                                                   cgraph->nodes[node_idx + 3];

    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];
    const int n_expert_used = weights->ne[1];

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, nullptr, nullptr, nullptr, cgraph->nodes[node_idx], GGML_OP_SOFT_MAX);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer logits_buf = ggml_vk_tensor_subbuffer(ctx, logits);
    vk_subbuffer bias_buf = ggml_vk_tensor_subbuffer(ctx, bias);
    vk_subbuffer weights_buf = ggml_vk_tensor_subbuffer(ctx, weights);
    vk_subbuffer ids_buf = ggml_vk_tensor_subbuffer(ctx, ids);

    vk_op_topk_moe_push_constants pc {};
    pc.n_rows = n_rows;
    pc.n_experts_push = n_experts;
    pc.n_expert_used = n_expert_used;
    pc.clamp_min = -std::numeric_limits<float>::infinity();
    pc.clamp_max = std::numeric_limits<float>::infinity();
    if (mode == TOPK_MOE_EARLY_SOFTMAX_NORM) {
        ggml_tensor * clamp = cgraph->nodes[node_idx + 7];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    if (mode == TOPK_MOE_SIGMOID_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[node_idx + 8];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }
    if (mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        ggml_tensor * clamp = cgraph->nodes[node_idx + 9];
        GGML_ASSERT(clamp->op == GGML_OP_CLAMP);
        pc.clamp_min = ggml_get_op_params_f32(clamp, 0);
        pc.clamp_max = ggml_get_op_params_f32(clamp, 1);
    }

#define GATING_FUNC_SOFTMAX 0
#define GATING_FUNC_SIGMOID 1
#define GATING_FUNC_SOFTMAX_WEIGHT 2
#define GATING_FUNC_SQRT_SOFTPLUS 3

    pc.gating_func = mode == TOPK_MOE_SIGMOID_NORM_BIAS       ? GATING_FUNC_SIGMOID :
                     mode == TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS ? GATING_FUNC_SQRT_SOFTPLUS :
                     mode == TOPK_MOE_LATE_SOFTMAX             ? GATING_FUNC_SOFTMAX_WEIGHT :
                                                                 GATING_FUNC_SOFTMAX;
    pc.has_bias = has_bias;
    pc.with_norm = mode == TOPK_MOE_EARLY_SOFTMAX_NORM || has_bias;
    if (ctx->fused_topk_moe_scale) {
        GGML_ASSERT(weights->op == GGML_OP_SCALE);
        pc.output_scale = ggml_get_op_params_f32(weights, 0);
        pc.output_bias = ggml_get_op_params_f32(weights, 1);
    } else {
        pc.output_scale = 1.0f;
        pc.output_bias = 0.0f;
    }

    GGML_ASSERT(n_expert_used <= n_experts);

    const uint32_t rows_per_block = 4;
    std::array<uint32_t, 3> elements = { CEIL_DIV(n_rows, rows_per_block), 1, 1 };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {logits_buf, bias_buf, weights_buf, ids_buf}, pc, elements);
}

static void ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_cgraph * cgraph, int node_idx, bool backprop) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const ggml_tensor * src3 = nullptr;
    const int n_dims        = ((int32_t *) dst->op_params)[1];
    const int mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    const float freq_base   = ((float *)   dst->op_params)[5];
    const float beta_fast   = ((float *)   dst->op_params)[9];
    const float beta_slow   = ((float *)   dst->op_params)[10];
    int sections[4] {};
    if (mode & GGML_ROPE_TYPE_MROPE) {
        memcpy(sections, (int32_t *) dst->op_params + 11, sizeof(int)*4);
    }

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    uint32_t set_rows_stride = 0;
    // Fused rope + view + set_rows passes the set_rows destination stride in set_rows_stride
    // and overrides the dst and sets src3=row_indices
    if (ctx->num_additional_fused_ops > 0) {
        set_rows_stride = cgraph->nodes[node_idx + 2]->nb[1] / ggml_type_size(cgraph->nodes[node_idx + 2]->type);
        src3 = cgraph->nodes[node_idx + 2]->src[1];
        dst = cgraph->nodes[node_idx + 2];
    }

    ggml_vk_op_f32<vk_op_rope_push_constants>(ctx, subctx, src0, src1, src2, src3, dst, GGML_OP_ROPE,
        ggml_vk_make_rope_constants(cgraph->nodes[node_idx], src0, src2 != nullptr, backprop, set_rows_stride));
}

static void ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t * op_params = (const uint32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);

    uint32_t ncols_pad_log2 = (uint32_t)ceilf(log2f(float(ncols)));
    uint32_t ncolsp2 = 1 << ncols_pad_log2;

    vk_op_argsort_push_constants pc { ncols, ncolsp2, ncols_pad_log2, nrows, op_params[0], 0, 0, 0, 0, };

    // Pick the largest workgroup size <= ncolsp2
    uint32_t pipeline_idx = std::min(ncols_pad_log2, num_argsort_pipelines - 1);

    uint32_t max_wg_log2 = std::min(ctx->device->max_workgroup_size_log2, num_argsort_pipelines - 1);

    // Use the "small" argsort shader if the whole sort can be done by a single workgroup.
    bool use_small = ncols_pad_log2 <= max_wg_log2 &&
                     ctx->device->pipeline_argsort_f32[pipeline_idx] != nullptr;

    vk_pipeline pipeline = use_small ? ctx->device->pipeline_argsort_f32[pipeline_idx]
                                     : ctx->device->pipeline_argsort_large_f32[pipeline_idx];

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer subbuf1 = dst_buf;

    // Reserve space for ivec2 per element, with rows padded to a power of two
    if (!use_small) {
        const size_t x_sz = size_t{ncolsp2} * nrows * 2 * sizeof(int);

        if (ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        if (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        subbuf1 = { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    }

    std::array<uint32_t, 3> elements;

    elements[0] = ncolsp2;
    elements[1] = std::min((uint32_t)ggml_nrows(src0), ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
    elements[2] = 1;

    // First dispatch initializes tmp_idx and does the first N passes where
    // there is only communication between threads in the same workgroup.
    {
        vk_op_argsort_push_constants pc2 = pc;
        pc2.outer_start = 0;
        pc2.outer_end = std::min(ncols_pad_log2, max_wg_log2);
        pc2.inner_start = 0;
        pc2.inner_end = 100;
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
    }
    if (!use_small) {
        ggml_vk_sync_buffers(ctx, subctx);
        // Loop over outer/inner passes, synchronizing between each pass.
        for (uint32_t outer = max_wg_log2; outer < ncols_pad_log2; ++outer) {
            for (uint32_t inner = 0; inner < outer + 1; ++inner) {
                vk_op_argsort_push_constants pc2 = pc;
                pc2.outer_start = outer;
                pc2.outer_end = outer + 1;
                pc2.inner_start = inner;
                pc2.inner_end = inner + 1;
                // When the inner idx is large enough, there's only communication
                // within a workgroup. So the remaining inner iterations can all
                // run in the same dispatch.
                if (outer - inner < pipeline_idx) {
                    pc2.inner_end = 100;
                    inner = outer;
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx];
                } else {
                    // Smaller workgroup empirically seems to perform better
                    pipeline = ctx->device->pipeline_argsort_large_f32[pipeline_idx - 2];
                }
                ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
                ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, subbuf1, dst_buf }, pc2, elements);
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
        ctx->prealloc_x_need_sync = true;
    }
}

static void ggml_vk_topk(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t ncols = src0->ne[0];
    uint32_t nrows = ggml_nrows(src0);
    uint32_t k = dst->ne[0];

    // tournament path is faster where it fits; use radix-select only past its k limit
    const uint32_t k_min_pipeline = std::max((uint32_t) log2f(float(k)) + 1, ctx->device->subgroup_size_log2);
    if (k_min_pipeline >= num_topk_pipelines || ctx->device->pipeline_topk_f32[k_min_pipeline] == nullptr) {
        vk_pipeline pipeline = ctx->device->pipeline_topk_radix_f32;
        GGML_ASSERT(pipeline != nullptr);

        if (ctx->prealloc_x_need_sync) {
            ggml_vk_sync_buffers(ctx, subctx);
        }

        vk_op_topk_radix_push_constants pc { ncols, k, nrows, 0, 0, 0 };
        std::array<uint32_t, 3> elements {
            pipeline->wg_denoms[0],
            std::min(nrows, ctx->device->properties.limits.maxComputeWorkGroupCount[1]),
            1,
        };
        // the non-QSA path only uses bindings 0/1; bind valid buffers for the unused QSA slots
        vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
            { src0_buf, dst_buf, src0_buf, src0_buf, src0_buf }, pc, elements);
        return;
    }

    vk_op_topk_push_constants pc { ncols, ncols, ncols, k, nrows, 0, 0 };

    if (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    std::array<uint32_t, 3> elements;
    elements[1] = std::min(nrows, ctx->device->properties.limits.maxComputeWorkGroupCount[1]);
    elements[2] = 1;

    uint32_t num_elements = ncols;

    // Each iteration reduces a workgroup's worth of elements down to the K
    // largest elements. Repeat until we have the top K elements.
    // Need to do at least one iteration to write out the results.
    bool done_one_iter = false;
    uint32_t dbl_buf_index = 0;
    size_t dbl_buf_size;
    while (num_elements > k || !done_one_iter) {

        // Prefer going as small as num_topk_pipelines - 3 for perf reasons.
        // But if K is larger, then we need a larger workgroup
        uint32_t max_pipeline = num_topk_pipelines - 1;
        uint32_t preferred_pipeline = std::max(num_topk_pipelines - 3, (uint32_t)log2f(float(k)) + 2);
        max_pipeline = std::min(preferred_pipeline, max_pipeline);
        uint32_t min_pipeline = (uint32_t)log2f(float(k)) + 1;
        // require full subgroup
        min_pipeline = std::max(min_pipeline, ctx->device->subgroup_size_log2);

        uint32_t pipeline_idx = (uint32_t)ceilf(log2f(float(num_elements)));
        pipeline_idx = std::min(pipeline_idx, max_pipeline);
        pipeline_idx = std::max(pipeline_idx, min_pipeline);

        if (num_elements > (1u << pipeline_idx)) {
            // If we could finish on this loop iteration (i.e. a single workgroup)
            // then do so. It's better than the overhead of another pass.
            for (uint32_t i = pipeline_idx; i < num_topk_pipelines; ++i) {
                if (num_elements <= (1u << i)) {
                    pipeline_idx = i;
                    break;
                }
            }
        }

        vk_pipeline pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        // If the device doesn't support a pipeline this large, use smaller
        while (!pipeline) {
            pipeline_idx--;
            GGML_ASSERT(pipeline_idx >= min_pipeline);
            pipeline = ctx->device->pipeline_topk_f32[pipeline_idx];
        }

        vk_op_topk_push_constants pc2 = pc;
        pc2.ncols_input = num_elements;

        // Number of elements remaining after this pass
        uint32_t num_dst_elements = (num_elements / pipeline->wg_denoms[0]) * k + std::min(k, num_elements % pipeline->wg_denoms[0]);

        pc2.ncols_output = num_dst_elements;

        if (!done_one_iter) {
            // Reserve space for ivec2 per element, double buffered
            // K per workgroup per row
            dbl_buf_size = num_dst_elements * nrows * 2 * sizeof(int);
            dbl_buf_size = ROUNDUP_POW2(dbl_buf_size, ctx->device->properties.limits.minStorageBufferOffsetAlignment);
            const size_t x_sz = dbl_buf_size * 2;

            if (ctx->prealloc_size_x < x_sz) {
                ctx->prealloc_size_x = x_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
        }

        vk_subbuffer src_buf;
        vk_subbuffer dst_buf;

        if (num_elements == ncols) {
            pc2.first_pass = 1;
            src_buf = ggml_vk_tensor_subbuffer(ctx, src0);
        } else {
            src_buf = { ctx->prealloc_x, dbl_buf_index * dbl_buf_size, dbl_buf_size };
        }
        if (num_dst_elements == k) {
            pc2.last_pass = 1;
            dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
        } else {
            dst_buf = { ctx->prealloc_x, (dbl_buf_index ^ 1) * dbl_buf_size, dbl_buf_size };
        }

        elements[0] = num_elements;

        ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc2, elements);
        num_elements = num_dst_elements;
        dbl_buf_index ^= 1;
        if (num_elements > k) {
            ggml_vk_sync_buffers(ctx, subctx);
        }
        done_one_iter = true;
    }
    ctx->prealloc_x_need_sync = true;
}

static void ggml_vk_topk_qsa(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * get_rows = cgraph->nodes[node_idx + 0];
    const ggml_tensor * add      = cgraph->nodes[node_idx + ctx->num_additional_fused_ops - 1];
    ggml_tensor *       top_k    = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];

    const ggml_tensor * scores   = get_rows->src[0]; // [n_tps, n_blocks, n_stream]
    const ggml_tensor * cell_blk = get_rows->src[1]; // [n_kv, n_stream]

    // raw f16 mask: follow the reshape/cpy chain back to the materialized input
    const ggml_tensor * mask = add->src[1];
    while (mask->op == GGML_OP_RESHAPE || mask->op == GGML_OP_CPY) {
        mask = mask->src[0];
    }

    const uint32_t n_tps    = scores->ne[0];
    const uint32_t n_blocks = scores->ne[1];
    const uint32_t n_stream = scores->ne[2];
    const uint32_t n_kv     = cell_blk->ne[0];
    const uint32_t width    = top_k->ne[0];
    const uint32_t nrows    = n_tps * n_stream;

    vk_pipeline pipeline = ctx->device->pipeline_topk_radix_qsa;
    GGML_ASSERT(pipeline != nullptr);

    // scratch holds the gathered+masked input, materialized once and reused across passes
    const size_t scratch_size = size_t{ n_kv } * nrows * sizeof(float);
    if (ctx->prealloc_size_x < scratch_size) {
        ctx->prealloc_size_x = scratch_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    if (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_op_topk_radix_push_constants pc { n_kv, width, nrows, n_tps, n_blocks, n_stream };
    std::array<uint32_t, 3> elements {
        pipeline->wg_denoms[0],
        std::min(nrows, ctx->device->properties.limits.maxComputeWorkGroupCount[1]),
        1,
    };
    vk_subbuffer scratch_buf { ctx->prealloc_x, 0, ctx->prealloc_x->size };
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, scores), ggml_vk_tensor_subbuffer(ctx, top_k),
          ggml_vk_tensor_subbuffer(ctx, cell_blk), ggml_vk_tensor_subbuffer(ctx, mask),
          scratch_buf }, pc, elements);
    ctx->prealloc_x_need_sync = true;
}

static void ggml_vk_sum(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, ggml_nelements(src0));
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SUM, p);
}

static void ggml_vk_sum_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_SUM_ROWS, p);
}

static void ggml_vk_mean(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, src0->ne[0]);
    p.weight = 1.0f / (float)src0->ne[0];
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_MEAN, p);
}

static void ggml_vk_cumsum(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    vk_op_sum_rows_push_constants p = vk_op_sum_rows_push_constants_init(src0, dst, src0->ne[0]);
    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_CUMSUM, p);
}

static void ggml_vk_argmax(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_ARGMAX, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], 0.0f, 0.0f, 0.0f, 0.0f });
}

static void ggml_vk_count_equal(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_COUNT_EQUAL, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f, 0.0f, 0.0f });
}

static void ggml_vk_solve_tri(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_SOLVE_TRI,
        vk_op_binary_push_constants_init(src0, src1, dst));
}

static void ggml_vk_rel_pos_bias(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_rel_pos_bias(dst=" << dst << ")");

    const int H     = ggml_get_op_params_i32(dst, 0);
    const int W     = ggml_get_op_params_i32(dst, 1);
    const int B     = ggml_get_op_params_i32(dst, 2);
    const int C     = ggml_get_op_params_i32(dst, 3);
    const int rel_h = ggml_get_op_params_i32(dst, 4);
    const int rel_w = 2 * W - 1;
    const int HW    = H * W;

    vk_op_rel_pos_bias_push_constants pc = {
        (uint32_t)H, (uint32_t)W, (uint32_t)B, (uint32_t)C,
        (uint32_t)rel_h, (uint32_t)rel_w
    };

    vk_pipeline pipeline = ctx->device->pipeline_rel_pos_bias_f32;
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);

    /* dispatch: x=W, y=H, z=B*HW — each thread computes one output cell */
    std::array<uint32_t, 3> elements = { (uint32_t)W, (uint32_t)H, (uint32_t)(B * HW) };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, dst_buf }, pc, elements);
}

static void ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int32_t s0 = dst->op_params[0];
    const int32_t s1 = dst->op_params[1];
    const int32_t p0 = dst->op_params[2];
    const int32_t p1 = dst->op_params[3];
    const int32_t d0 = dst->op_params[4];
    const int32_t d1 = dst->op_params[5];

    const bool is_2D = dst->op_params[6] == 1;

    const uint32_t IC = src1->ne[is_2D ? 2 : 1];
    const uint32_t IH = is_2D ? src1->ne[1] : 1;
    const uint32_t IW =         src1->ne[0];

    const uint32_t KH = is_2D ? src0->ne[1] : 1;
    const uint32_t KW =         src0->ne[0];

    const uint32_t OH = is_2D ? dst->ne[2] : 1;
    const uint32_t OW =         dst->ne[1];

    const uint32_t offset_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const uint32_t batch_offset = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    const uint32_t batch = src1->ne[is_2D ? 3 : 2];

    const ggml_backend_vk_buffer_context * d_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    const vk_buffer d_buf = d_buf_ctx->dev_buffer;

    const vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

    ggml_vk_op_f32<vk_op_im2col_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_IM2COL, {
        dst_addr,
        batch_offset, offset_delta,
        IC, IW, IH, OW, OH, KW, KH,
        OH * batch,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1, batch * IC
    });
}

static void ggml_vk_im2col_3d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const int32_t IC = ((const int32_t *)(dst->op_params))[9];

    const int64_t N  = ne13 / IC;
    const int64_t ID = ne12;
    const int64_t IH = ne11;
    const int64_t IW = ne10;

    const int64_t KD = ne02;
    const int64_t KH = ne01;
    const int64_t KW = ne00;

    const int64_t OD = ne3 / N;
    const int64_t OH = ne2;
    const int64_t OW = ne1;

    const ggml_backend_vk_buffer_context * d_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    const vk_buffer d_buf = d_buf_ctx->dev_buffer;

    const vk::DeviceAddress dst_addr = d_buf->bda_addr + vk_tensor_offset(dst) + dst->view_offs;

    vk_op_im2col_3d_push_constants pc {};

    pc.dst_addr = dst_addr;
    pc.nb10 = nb10 / ggml_type_size(src1->type);
    pc.nb11 = nb11 / ggml_type_size(src1->type);
    pc.nb12 = nb12 / ggml_type_size(src1->type);
    pc.nb13 = nb13 / ggml_type_size(src1->type);
    pc.s0 = s0;
    pc.s1 = s1;
    pc.s2 = s2;
    pc.p0 = p0;
    pc.p1 = p1;
    pc.p2 = p2;
    pc.d0 = d0;
    pc.d1 = d1;
    pc.d2 = d2;
    pc.IW = IW;
    pc.IH = IH;
    pc.ID = ID;
    pc.IC = IC;
    pc.KW = KW;
    pc.OH = OH;
    pc.KD_KH_KW = KD*KH*KW;
    pc.KH_KW = KH*KW;
    pc.IC_KD_KH_KW = IC*KD*KH*KW;
    pc.N_OD_OH = N*OD*OH;
    pc.OD_OH = OD*OH;
    pc.OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    pc.OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    pc.OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;

    ggml_vk_op_f32<vk_op_im2col_3d_push_constants>(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_IM2COL_3D, std::move(pc));
}

static void ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t dim = dst->op_params[0];
    const uint32_t max_period = dst->op_params[1];
    const uint32_t nb1 = dst->nb[1] / ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

static void ggml_vk_conv_transpose_1d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // src0: (K, Cout, Cin, 1) -- kernel
    // src1: (L, Cin, 1, 1) -- input
    // dst: (*, Cout, 1, 1)

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb10 == sizeof(float));

    const int32_t s0 = dst->op_params[0];

    vk_op_conv_transpose_1d_push_constants p{};
    p.Cout = static_cast<uint32_t>(ne01);
    p.Cin = static_cast<uint32_t>(ne02);
    p.K = static_cast<uint32_t>(ne00);
    p.L = static_cast<uint32_t>(ne10);
    p.KL = static_cast<uint32_t>(ne0);
    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.s0 = static_cast<uint32_t>(s0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_TRANSPOSE_1D, std::move(p));
}

// ggml 0.27: ops added in the sync (bundle 7).
static void ggml_vk_pad_reflect_1d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t p0 = (uint32_t)dst->op_params[0];
    const uint32_t p1 = (uint32_t)dst->op_params[1];

    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst, ggml_nelements(dst));
    memcpy(&p.param1, &p0, sizeof(float));
    memcpy(&p.param2, &p1, sizeof(float));

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_PAD_REFLECT_1D, std::move(p));
}

static void ggml_vk_gated_linear_attn(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const size_t seq_length = dst->src[0]->ne[2];
    const size_t n_embed    = dst->ne[0];
    const size_t n_heads    = dst->src[0]->ne[1];
    const size_t n_seqs     = dst->src[4]->ne[1];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    GGML_ASSERT(dst->buffer != nullptr);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, dst->src[0], dst->src[1], dst->src[2], dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer dst_buf = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5] = {};
    for (int i = 0; i < 5; i++) {
        src_buf[i] = ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    const vk_op_gated_linear_attn_push_constants pc = {
        (uint32_t)n_seqs,
        (uint32_t)seq_length,
        (uint32_t)n_embed,
        (uint32_t)n_heads,
        scale,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], dst_buf},
        pc, { (uint32_t)(n_seqs * n_heads), 1, 1 });
}

static std::array<uint32_t, 3> ggml_vk_nrows_elements(uint32_t nr) {
    if (nr > 262144) {
        return { 512, 512, CEIL_DIV(nr, 262144) };
    }
    if (nr > 512) {
        return { 512, CEIL_DIV(nr, 512), 1 };
    }
    return { nr, 1, 1 };
}

static void ggml_vk_cross_entropy_loss(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));

    const uint32_t nclasses = (uint32_t)src0->ne[0];
    const uint32_t nrows    = (uint32_t)ggml_nrows(src0);

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, nullptr, dst, GGML_OP_CROSS_ENTROPY_LOSS);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_sum_rows[0], 1);

    vk_subbuffer src0_buf = ggml_vk_tensor_subbuffer(ctx, src0);
    vk_subbuffer src1_buf = ggml_vk_tensor_subbuffer(ctx, src1);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst, true);

    const vk_op_push_constants pc = { nclasses, nrows, 0.0f, 0.0f, 0.0f, 0.0f };

    const size_t tmp_size = (size_t)nrows * sizeof(float);
    if (ctx->prealloc_size_x < tmp_size) {
        ctx->prealloc_size_x = tmp_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    if (ctx->prealloc_x_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer tmp_buf = { ctx->prealloc_x, 0, tmp_size };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src0_buf, src1_buf, tmp_buf }, pc, ggml_vk_nrows_elements(nrows));
    ggml_vk_sync_buffers(ctx, subctx);

    vk_op_sum_rows_push_constants sp = {};
    sp.n_cols = nrows;
    sp.ne01 = 1;
    sp.ne02 = 1;
    sp.weight = 1.0f;
    init_pushconst_fastdiv(sp);
    sp.misalign_offsets = get_misalign_bytes(ctx, dst) / ggml_type_size(dst->type);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_sum_rows[0], { tmp_buf, dst_buf }, sp, { 1, 1, 1 });
    ctx->prealloc_x_need_sync = true;
}

static void ggml_vk_col2im_1d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    // src0: [K_OC, T_in] columns from matmul
    // dst:  [T_out, OC]

    const int32_t stride = dst->op_params[0];
    const int32_t oc     = dst->op_params[1];
    const int32_t p0     = dst->op_params[2];

    const uint32_t K_OC  = static_cast<uint32_t>(src0->ne[0]);
    const uint32_t T_in  = static_cast<uint32_t>(src0->ne[1]);
    const uint32_t T_out = static_cast<uint32_t>(dst->ne[0]);
    const uint32_t OC    = static_cast<uint32_t>(oc);
    const uint32_t K     = K_OC / OC;

    vk_op_col2im_1d_push_constants p{};
    p.T_out  = T_out;
    p.OC     = OC;
    p.K_OC   = K_OC;
    p.T_in   = T_in;
    p.K      = K;
    p.stride = stride;
    p.p0     = p0;

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_COL2IM_1D, std::move(p));
}

static void ggml_vk_snake_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * mul0 = cgraph->nodes[node_idx + 0];
    const ggml_tensor * sqr  = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mul1 = cgraph->nodes[node_idx + 3];
    ggml_tensor *       add  = cgraph->nodes[node_idx + 4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    vk_pipeline pipeline = nullptr;
    switch (x->type) {
        case GGML_TYPE_F32:  pipeline = ctx->device->pipeline_snake_f32;  break;
        case GGML_TYPE_F16:  pipeline = ctx->device->pipeline_snake_f16;  break;
        case GGML_TYPE_BF16: pipeline = ctx->device->pipeline_snake_bf16; break;
        default:             GGML_ABORT("unsupported type");
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer x_buf     = ggml_vk_tensor_subbuffer(ctx, x);
    vk_subbuffer a_buf     = ggml_vk_tensor_subbuffer(ctx, a);
    vk_subbuffer inv_b_buf = ggml_vk_tensor_subbuffer(ctx, inv_b);
    vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, add);

    vk_op_snake_push_constants pc{};
    pc.ne0 = static_cast<uint32_t>(x->ne[0]);
    pc.ne1 = static_cast<uint32_t>(x->ne[1]);

    std::array<uint32_t, 3> elements = { pc.ne0, pc.ne1, 1 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, a_buf, inv_b_buf, dst_buf }, pc, elements);
}

static void ggml_vk_pool_1d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    const int32_t k0 = dst->op_params[1];
    const int32_t s0 = dst->op_params[2];
    const int32_t p0 = dst->op_params[3];

    const uint32_t IL = src0->ne[0];

    const uint32_t N = dst->ne[3] * dst->ne[2];

    const uint32_t OC = dst->ne[1];
    const uint32_t OL = dst->ne[0];

    const uint32_t parallel_elements = N * OC * OL;

    ggml_vk_op_f32<vk_op_pool1d_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_POOL_1D, {
        IL, OL, OC,
        parallel_elements,
        op,
        k0, s0, p0,
    });
}

static void ggml_vk_conv_3d(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(nb00 == sizeof(float) || nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    vk_op_conv3d_push_constants p{};
    p.IC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 9));
    p.N  = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 10));
    p.OC = static_cast<uint32_t>(ggml_get_op_params_i32(dst, 11));
    GGML_ASSERT(src0->ne[3] == (int64_t)p.IC * p.OC);
    GGML_ASSERT(src1->ne[3] == (int64_t)p.IC * p.N);
    GGML_ASSERT(dst->ne[3] == (int64_t)p.OC * p.N);

    p.IW = static_cast<uint32_t>(ne10);
    p.IH = static_cast<uint32_t>(ne11);
    p.ID = static_cast<uint32_t>(ne12);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);
    p.OD = static_cast<uint32_t>(ne2);

    // the shader clamps src addresses to p.IC * p.N * p.IW * p.IH * p.ID - 1 in uint32, so the
    // total input element count must fit in a uint32.
    GGML_ASSERT((uint64_t)p.IC * p.N * p.IW * p.IH * p.ID <= 0xFFFFFFFFull);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_3D, std::move(p));
}

static void ggml_vk_pool_2d(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    uint32_t op = static_cast<uint32_t>(dst->op_params[0]);
    const int32_t k1 = dst->op_params[1];
    const int32_t k0 = dst->op_params[2];
    const int32_t s1 = dst->op_params[3];
    const int32_t s0 = dst->op_params[4];
    const int32_t p1 = dst->op_params[5];
    const int32_t p0 = dst->op_params[6];

    const uint32_t IH = src0->ne[1];
    const uint32_t IW = src0->ne[0];

    const uint32_t N = dst->ne[3];

    const uint32_t OC = dst->ne[2];
    const uint32_t OH = dst->ne[1];
    const uint32_t OW = dst->ne[0];

    const uint32_t parallel_elements = N * OC * OH * OW;

    ggml_vk_op_f32<vk_op_pool2d_push_constants>(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_POOL_2D, {
        IW, IH, OW, OH, OC,
        parallel_elements,
        op,
        k0, k1, s0, s1, p0, p1,
    });
}

static void ggml_vk_conv_2d(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * src0,
                            const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(nb00 == sizeof(float) || nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0 == sizeof(float));

    bool transpose = dst->op == GGML_OP_CONV_TRANSPOSE_2D;

    vk_op_conv2d_push_constants p{};
    p.Cout = static_cast<uint32_t>(!transpose ? ne03 : ne02);
    p.Cin  = static_cast<uint32_t>(!transpose ? ne02 : ne03);
    p.N    = static_cast<uint32_t>(ne13);
    GGML_ASSERT(p.Cout == ne2);
    GGML_ASSERT(p.Cin == ne12);

    p.W  = static_cast<uint32_t>(ne10);
    p.H  = static_cast<uint32_t>(ne11);
    p.OW = static_cast<uint32_t>(ne0);
    p.OH = static_cast<uint32_t>(ne1);

    p.nb01 = static_cast<uint32_t>(nb01 / nb00);
    p.nb02 = static_cast<uint32_t>(nb02 / nb00);
    p.nb03 = static_cast<uint32_t>(nb03 / nb00);

    p.nb11 = static_cast<uint32_t>(nb11 / nb10);
    p.nb12 = static_cast<uint32_t>(nb12 / nb10);
    p.nb13 = static_cast<uint32_t>(nb13 / nb10);

    p.nb1 = static_cast<uint32_t>(nb1 / nb0);
    p.nb2 = static_cast<uint32_t>(nb2 / nb0);
    p.nb3 = static_cast<uint32_t>(nb3 / nb0);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, dst->op, std::move(p));
}

static void ggml_vk_conv_2d_dw(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    vk_op_conv2d_dw_push_constants p{};
    p.ne = ggml_nelements(dst);
    p.channels = dst->ne[2];
    p.batches = dst->ne[3];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.src_w = src1->ne[0];
    p.src_h = src1->ne[1];
    p.knl_w = src0->ne[0];
    p.knl_h = src0->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(src0->ne[3] == p.channels);
    GGML_ASSERT(src1->ne[3] == p.batches);

    ggml_vk_op_f32(ctx, subctx, src0, src1, nullptr, nullptr, dst, GGML_OP_CONV_2D_DW, std::move(p));
}

// ggmlR extension: GGML_OP_QCONV_I32 (ONNX QLinearConv).
//
// Written out rather than routed through ggml_vk_op_f32 for two reasons: the op
// has FIVE sources where that helper carries four, and its shader takes its own
// push constant block rather than the shared one.
//
// Everything is read from where the scheduler already put it -- no buffer is
// created, filled or destroyed here, and there is no fence wait. An earlier
// direct-dispatch version of this shader did all of that per call, re-uploading
// the weights every time and blocking on the result; that is what running as a
// real op removes.
static void ggml_vk_qconv_i32(ggml_backend_vk_context * ctx, vk_context& subctx,
                              const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * src2, const ggml_tensor * src3,
                              const ggml_tensor * src4, const ggml_tensor * src5,
                              ggml_tensor * dst) {
    GGML_ASSERT(src0 != nullptr && src1 != nullptr && src2 != nullptr);
    GGML_ASSERT(src5 != nullptr);   // the precomputed multiplier is required
    GGML_ASSERT(dst->buffer != nullptr);

    vk_op_qconv_i32_push_constants p{};
    p.W_in      = (uint32_t)src0->ne[0];
    p.H_in      = (uint32_t)src0->ne[1];
    p.C_in      = (uint32_t)src0->ne[2];
    p.W_out     = (uint32_t)dst->ne[0];
    p.H_out     = (uint32_t)dst->ne[1];
    p.C_out     = (uint32_t)dst->ne[2];
    p.KW        = (uint32_t)src1->ne[0];
    p.KH        = (uint32_t)src1->ne[1];
    p.stride_w  = ggml_get_op_params_i32(dst, 0);
    p.stride_h  = ggml_get_op_params_i32(dst, 1);
    p.pad_w     = ggml_get_op_params_i32(dst, 2);
    p.pad_h     = ggml_get_op_params_i32(dst, 3);
    p.dil_w     = ggml_get_op_params_i32(dst, 4);
    p.dil_h     = ggml_get_op_params_i32(dst, 5);
    p.x_scale   = ggml_get_op_params_f32(dst, 6);
    p.y_scale   = ggml_get_op_params_f32(dst, 7);
    p.x_zp      = ggml_get_op_params_i32(dst, 8);
    p.y_zp      = ggml_get_op_params_i32(dst, 9);
    p.out_lo    = ggml_get_op_params_f32(dst, 10);
    p.out_hi    = ggml_get_op_params_f32(dst, 11);
    p.n_w_scale = (uint32_t)src2->ne[0];
    p.n_w_zp    = src3 ? (uint32_t)src3->ne[0] : 1u;
    p.has_bias  = src4 ? 1u : 0u;
    p.has_w_zp  = src3 ? 1u : 0u;
    p.N_batch   = (uint32_t)dst->ne[3];
    // The shader steps x and dst by one image using the shapes above, which is
    // only the right address when both are packed at that stride.
    GGML_ASSERT(src0->ne[3] == dst->ne[3]);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
    {
        static int dbg = -1;
        if (dbg < 0) {
            const char * e = getenv("GGMLR_QCONV_DEBUG_ACC");
            dbg = (e && *e) ? atoi(e) : 0;
        }
        p.debug_acc = (uint32_t)dbg;
    }

    // One thread per output element, the whole reduction inside it: the int16
    // pair saturation this reproduces is order-dependent, so a workgroup
    // cooperating on one output would pair different neighbours and diverge.
    //
    // ne[3] is part of the count, not a loop around the dispatch: every output
    // element is independent, so one flat grid over batch*C_out*H_out*W_out
    // keeps the threads busy when H_out*W_out is small and the batch is large --
    // which is exactly the mask-head shape (28x28 over 51 images).
    const uint32_t total = (uint32_t)(dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3]);
    const std::array<uint32_t, 3> elements = { total, 1, 1 };

    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_qconv_i32, 1);

    vk_subbuffer x_buf  = ggml_vk_tensor_subbuffer(ctx, src0, true);
    vk_subbuffer w_buf  = ggml_vk_tensor_subbuffer(ctx, src1, true);
    vk_subbuffer ws_buf = ggml_vk_tensor_subbuffer(ctx, src2, true);
    // Bindings 3 and 4 must be filled even when the tensor is absent: an
    // unwritten descriptor is not a legal source, and the shader gates every
    // read behind n_w_zp / has_bias rather than skipping the binding. w_scale
    // stands in because it is always present and always float.
    vk_subbuffer wz_buf = src3 ? ggml_vk_tensor_subbuffer(ctx, src3, true) : ws_buf;
    vk_subbuffer bi_buf = src4 ? ggml_vk_tensor_subbuffer(ctx, src4, true) : ws_buf;
    vk_subbuffer mt_buf = ggml_vk_tensor_subbuffer(ctx, src5, true);
    vk_subbuffer d_buf  = ggml_vk_tensor_subbuffer(ctx, dst, true);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_qconv_i32,
                              { x_buf, w_buf, ws_buf, wz_buf, bi_buf, d_buf, mt_buf },
                              p, elements);
}

// ggmlR extension: GGML_OP_QMATMUL_I32 (ONNX QLinearMatMul).
//
// Same shape as ggml_vk_qconv_i32 above and for the same reason: it reads
// every operand where the scheduler put it. The direct dispatch this replaces
// created five device buffers, uploaded A and B (~114 MB on MaskRCNN's box
// head), waited on a fence and read the result back on every call -- about
// 10% of the node's time, the other 90% being the shader, which is unchanged.
static void ggml_vk_qmatmul_i32(ggml_backend_vk_context * ctx, vk_context& subctx,
                                const ggml_tensor * src0, const ggml_tensor * src1,
                                const ggml_tensor * src2, const ggml_tensor * src3,
                                ggml_tensor * dst) {
    GGML_ASSERT(src0 != nullptr && src1 != nullptr && src2 != nullptr);
    GGML_ASSERT(dst->buffer != nullptr);
    // 2-D only (the builder asserts the same), and packed: the shader indexes
    // a[m*K + k] and bmat[n*K + k].
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                ggml_is_contiguous(dst));

    vk_op_qmatmul_i32_push_constants p{};
    p.M         = (uint32_t)src0->ne[1];
    p.N         = (uint32_t)src1->ne[1];
    p.K         = (uint32_t)src0->ne[0];
    p.a_scale   = ggml_get_op_params_f32(dst, 0);
    p.y_scale   = ggml_get_op_params_f32(dst, 1);
    p.a_zp      = ggml_get_op_params_i32(dst, 2);
    p.y_zp      = ggml_get_op_params_i32(dst, 3);
    p.out_lo    = ggml_get_op_params_f32(dst, 4);
    p.out_hi    = ggml_get_op_params_f32(dst, 5);
    p.b_zp_any  = (uint32_t)ggml_get_op_params_i32(dst, 6);
    p.n_b_scale = (uint32_t)src2->ne[0];
    p.n_b_zp    = src3 ? (uint32_t)src3->ne[0] : 1u;

    // 2D: N across, M down, matching the shader's 16x16 block tile and the
    // pipeline's {16,16,1} wg_denoms. A flat element count would hand the
    // shader one long row of blocks and every m would come out zero.
    const std::array<uint32_t, 3> elements = { p.N, p.M, 1 };

    ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_qmatmul_i32, 1);

    vk_subbuffer a_buf  = ggml_vk_tensor_subbuffer(ctx, src0, true);
    vk_subbuffer b_buf  = ggml_vk_tensor_subbuffer(ctx, src1, true);
    vk_subbuffer bs_buf = ggml_vk_tensor_subbuffer(ctx, src2, true);
    // Binding 3 must be filled even with no zero-point tensor: an unwritten
    // descriptor is not a legal source. b_scale stands in; the builder sets
    // b_zp_any = 0 in that case, and the shader reads b_zp only when it is set.
    vk_subbuffer bz_buf = src3 ? ggml_vk_tensor_subbuffer(ctx, src3, true) : bs_buf;
    vk_subbuffer d_buf  = ggml_vk_tensor_subbuffer(ctx, dst, true);

    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_qmatmul_i32,
                              { a_buf, b_buf, bs_buf, bz_buf, d_buf },
                              p, elements);
}

static void ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const float * op_params = (const float *)dst->op_params;
    vk_op_unary_push_constants p = vk_op_unary_push_constants_init(src0, dst);
    p.param1 = op_params[0];

    ggml_vk_op_f32(ctx, subctx, src0, nullptr, nullptr, nullptr, dst, GGML_OP_LEAKY_RELU, std::move(p));
}



// ggml 0.27: DeepSeek V4 hyper-connections and the DSA lightning indexer.

static uint32_t ggml_vk_nb_elem(const ggml_tensor * t, int i) {
    return (uint32_t)(t->nb[i] / ggml_type_size(t->type));
}

static void ggml_vk_dsv4_hc_comb(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * mixes, const ggml_tensor * scale, const ggml_tensor * base, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_comb(" << mixes << ", " << scale << ", " << base << ", " << dst << ")");

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_comb_f32;
    GGML_ASSERT(pipeline != nullptr);

    const uint32_t n_tokens = (uint32_t)mixes->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const vk_subbuffer mixes_buf = ggml_vk_tensor_subbuffer(ctx, mixes, true);
    const vk_subbuffer scale_buf = ggml_vk_tensor_subbuffer(ctx, scale, true);
    const vk_subbuffer base_buf  = ggml_vk_tensor_subbuffer(ctx, base,  true);
    const vk_subbuffer dst_buf   = ggml_vk_tensor_subbuffer(ctx, dst,   true);

    vk_op_dsv4_hc_comb_push_constants pc = {
        n_tokens,
        ggml_vk_nb_elem(mixes, 0), ggml_vk_nb_elem(mixes, 1),
        ggml_vk_nb_elem(scale, 0),
        ggml_vk_nb_elem(base,  0),
        ggml_vk_nb_elem(dst,   0), ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        0, 0, 0, 0,
        ggml_get_op_params_f32(dst, 0),
        (uint32_t)ggml_get_op_params_i32(dst, 1),
    };
    init_pushconst_tensor_offsets(ctx, pc, mixes, scale, base, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { mixes_buf, scale_buf, base_buf, dst_buf }, pc, { n_tokens, 1, 1 });
}

static void ggml_vk_dsv4_hc_pre(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * x, const ggml_tensor * weights, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_pre(" << x << ", " << weights << ", " << dst << ")");

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    vk_pipeline pipeline = gated ? ctx->device->pipeline_dsv4_hc_pre_gated_f32 : ctx->device->pipeline_dsv4_hc_pre_f32;
    GGML_ASSERT(pipeline != nullptr);

    const uint32_t n_embd   = (uint32_t)x->ne[0];
    const uint32_t n_tokens = (uint32_t)x->ne[2];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,       true);
    const vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, weights, true);
    const vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,     true);

    vk_op_dsv4_hc_pre_push_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1), ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(weights, 0), ggml_vk_nb_elem(weights, 1), ggml_vk_nb_elem(weights, 2),
        ggml_vk_nb_elem(dst, 0), ggml_vk_nb_elem(dst, 1),
        0, 0, 0,
        scale,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, weights, nullptr, nullptr, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, w_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

static void ggml_vk_dsv4_hc_post(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * x, const ggml_tensor * residual, const ggml_tensor * post, const ggml_tensor * comb, ggml_tensor * dst, const ggml_tensor * gate_scale_in = nullptr) {
    VK_LOG_DEBUG("ggml_vk_dsv4_hc_post(" << x << ", " << residual << ", " << post << ", " << comb << ", " << dst << ")");

    vk_pipeline pipeline = comb ? ctx->device->pipeline_dsv4_hc_post_f32 : ctx->device->pipeline_dsv4_hc_post_nocomb_f32;
    GGML_ASSERT(pipeline != nullptr);

    const uint32_t n_embd   = (uint32_t)x->ne[0];
    const uint32_t n_tokens = (uint32_t)x->ne[1];

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        true);
    const vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, true);
    // with a fused gate, post is scale(sigmoid(scale(p_src))) and the shader applies it to p_src
    const ggml_tensor * p_src = gate_scale_in ? gate_scale_in->src[0] : post;
    const vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    true);
    const vk_subbuffer c_buf = comb ? ggml_vk_tensor_subbuffer(ctx, comb, true) : x_buf;
    const vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst,      true);

    vk_op_dsv4_hc_post_push_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 0), ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 0), ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        comb ? ggml_vk_nb_elem(comb, 0) : 0, comb ? ggml_vk_nb_elem(comb, 1) : 0, comb ? ggml_vk_nb_elem(comb, 2) : 0,
        ggml_vk_nb_elem(dst,  0), ggml_vk_nb_elem(dst,  1), ggml_vk_nb_elem(dst,  2),
        0, 0, 0, 0, 0,
        gate_scale_in ? 1u : 0u,
        gate_scale_in ? ggml_get_op_params_f32(gate_scale_in, 0) : 1.0f,
        gate_scale_in ? ggml_get_op_params_f32(post, 0) : 1.0f,
    };
    init_pushconst_tensor_offsets(ctx, pc, x, residual, p_src, comb, dst);

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, c_buf, d_buf }, pc, { n_embd, n_tokens, 1 });
}

static void ggml_vk_lightning_indexer(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, q, k, w, dst, dst->op);
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const uint32_t n_kv      = k->ne[2];
    const uint32_t n_heads   = q->ne[1];
    const uint32_t n_tokens  = q->ne[2];
    const uint32_t n_streams = q->ne[3];
    const uint32_t n_masks   = m->ne[3];

    // one workgroup per tile of 64 keys and 8 tokens, see lightning_indexer.comp
    const uint32_t n_tiles_kv = CEIL_DIV(n_kv, 64);
    const uint32_t n_tiles_t  = CEIL_DIV(n_tokens, 8);

    // q, w and dst are f32 and m is f16, so their strides are passed in elements;
    // k may be quantized, so its strides stay in bytes
    const uint32_t q_nb1 = q->nb[1] / sizeof(float);
    const uint32_t q_nb2 = q->nb[2] / sizeof(float);
    const uint32_t q_nb3 = q->nb[3] / sizeof(float);
    const uint32_t k_nb2 = k->nb[2];
    const uint32_t k_nb3 = k->nb[3];
    const uint32_t w_nb1 = w->nb[1] / sizeof(float);
    const uint32_t w_nb3 = w->nb[3] / sizeof(float);
    const uint32_t m_nb1 = m->nb[1] / sizeof(ggml_fp16_t);
    const uint32_t m_nb3 = m->nb[3] / sizeof(ggml_fp16_t);
    const uint32_t d_nb1 = dst->nb[1] / sizeof(float);
    const uint32_t d_nb3 = dst->nb[3] / sizeof(float);

    const vk_op_lightning_indexer_push_constants pc = {
        n_kv, n_heads, n_tokens, n_masks,
        q_nb1, q_nb2, q_nb3,
        k_nb2, k_nb3,
        w_nb1, w_nb3,
        m_nb1, m_nb3,
        d_nb1, d_nb3,
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {ggml_vk_tensor_subbuffer(ctx, q), ggml_vk_tensor_subbuffer(ctx, k), ggml_vk_tensor_subbuffer(ctx, w), ggml_vk_tensor_subbuffer(ctx, m), ggml_vk_tensor_subbuffer(ctx, dst)},
        pc, {n_tiles_kv, n_tiles_t, n_streams});
}

