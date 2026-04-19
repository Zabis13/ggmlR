static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {

    if (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = true;
        ggml_vk_synchronize(ctx);
        ggml_vk_ctx_begin(ctx->device, subctx);
    }

    if (ctx->prealloc_x == nullptr || (ctx->prealloc_size_x > 0 && ctx->prealloc_x->size < ctx->prealloc_size_x)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_size_x << ")");
        // Resize buffer
        if (ctx->prealloc_x != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_x);
    }
    if (ctx->prealloc_y == nullptr || (ctx->prealloc_size_y > 0 && ctx->prealloc_y->size < ctx->prealloc_size_y)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_size_y << ")");
        // Resize buffer
        if (ctx->prealloc_y != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_y);
    }
    if (ctx->prealloc_split_k == nullptr || (ctx->prealloc_size_split_k > 0 && ctx->prealloc_split_k->size < ctx->prealloc_size_split_k)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_size_split_k << ")");
        // Resize buffer
        if (ctx->prealloc_split_k != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_split_k);
        }
        ctx->prealloc_split_k = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_split_k);
    }
    if (ctx->prealloc_add_rms_partials == nullptr || (ctx->prealloc_size_add_rms_partials > 0 && ctx->prealloc_add_rms_partials->size < ctx->prealloc_size_add_rms_partials)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(add_partials_size: " << ctx->prealloc_add_rms_partials << ")");
        // Resize buffer
        if (ctx->prealloc_add_rms_partials != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
        }
        ctx->prealloc_add_rms_partials = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_add_rms_partials);
    }
}

static void ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, int tensor_idx, bool almost_ready);

// Returns true if node has enqueued work into the queue, false otherwise
// If submit is true the current all operations queued so far are being submitted to Vulkan to overlap cmdlist creation and GPU execution.
static bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int node_idx, ggml_tensor *node_begin, int node_idx_begin, bool last_node, bool almost_ready, bool submit){
    ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_is_empty(node) || ggml_op_is_empty(node->op) || !node->buffer) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << node << ", " << ggml_op_name(node->op) << ")");
    ctx->semaphore_idx = 0;

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];
    ggml_tensor * src2 = node->src[2];
    ggml_tensor * src3 = node->src[3];

    if (node->op == GGML_OP_ADD) {
        int next_node_idx = node_idx + 1 + ctx->num_additional_fused_ops;
        if (next_node_idx < cgraph->n_nodes &&
            cgraph->nodes[next_node_idx]->op == GGML_OP_RMS_NORM &&
            cgraph->nodes[next_node_idx]->src[0] == cgraph->nodes[next_node_idx - 1] &&
            ggml_nrows(cgraph->nodes[next_node_idx]) == 1 &&
            ctx->device->add_rms_fusion) {
            uint32_t size = ggml_vk_rms_partials_size(ctx, cgraph->nodes[node_idx]);
            ctx->do_add_rms_partials_offset_calculation = true;
            if (ctx->prealloc_size_add_rms_partials_offset + size <= ctx->prealloc_size_add_rms_partials) {
                ctx->do_add_rms_partials = true;
            }
        }
    }

    vk_context compute_ctx;

    if (ctx->compute_ctx.expired()) {
        compute_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->compute_ctx = compute_ctx;
        ggml_vk_ctx_begin(ctx->device, compute_ctx);
    } else {
        compute_ctx = ctx->compute_ctx.lock();
    }

    {
        // This logic detects dependencies between modes in the graph and calls ggml_vk_sync_buffers
        // to synchronize them. This handles most "normal" synchronization when computing the graph, and when
        // there is no auxiliary memory use, it shouldn't be necessary to call ggml_vk_sync_buffers
        // outside of this logic. When a node uses one of the prealloc buffers for something like
        // dequantization or split_k, additional synchronization is needed between those passes.
        bool need_sync = false;

        // Check whether "node" requires synchronization. The node requires synchronization if it
        // overlaps in memory with another unsynchronized node and at least one of them is a write.
        // Destination nodes are checked against both the written/read lists. Source nodes are only
        // checked against the written list. Two nodes overlap in memory if they come from the same
        // buffer and the tensor or view ranges overlap.
        auto const &overlaps_unsynced = [&](const ggml_tensor *node, const std::vector<const ggml_tensor *> &unsynced_nodes) -> bool {
            if (unsynced_nodes.size() == 0) {
                return false;
            }
            auto n_base = vk_tensor_offset(node) + node->view_offs;
            auto n_size = ggml_nbytes(node);
            ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)node->buffer->context;
            vk_buffer a_buf = a_buf_ctx->dev_buffer;
            for (auto &other : unsynced_nodes) {
                ggml_backend_vk_buffer_context * o_buf_ctx = (ggml_backend_vk_buffer_context *)other->buffer->context;
                vk_buffer o_buf = o_buf_ctx->dev_buffer;
                if (a_buf == o_buf) {
                    auto o_base = vk_tensor_offset(other) + other->view_offs;
                    auto o_size = ggml_nbytes(other);

                    if ((o_base <= n_base && n_base < o_base + o_size) ||
                        (n_base <= o_base && o_base < n_base + n_size)) {
                        return true;
                    }
                }
            }
            return false;
        };

        // For all fused ops, check if the destination node or any of the source
        // nodes require synchronization.
        for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1 && !need_sync; ++i) {
            const ggml_tensor *cur_node = cgraph->nodes[node_idx + i];
            // If the node actually writes to memory, then check if it needs to sync
            if (ctx->fused_ops_write_mask & (1 << i)) {
                if (overlaps_unsynced(cur_node, ctx->unsynced_nodes_read) || overlaps_unsynced(cur_node, ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    break;
                }
            }
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (!cur_node->src[j]) {
                    continue;
                }
                if (overlaps_unsynced(cur_node->src[j], ctx->unsynced_nodes_written)) {
                    need_sync = true;
                    break;
                }
            }
        }

        if (need_sync) {
            if (vk_enable_sync_logger) {
                std::cerr <<  "sync" << std::endl;
            }
            ctx->unsynced_nodes_written.clear();
            ctx->unsynced_nodes_read.clear();
            ggml_vk_sync_buffers(ctx, compute_ctx);

            if (vk_perf_logger_enabled && vk_perf_logger_concurrent) {
                ctx->query_node_idx[ctx->query_idx] = node_idx;
                compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
            const ggml_tensor *cur_node = cgraph->nodes[node_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            if (ctx->fused_ops_write_mask & (1 << i)) {
                ctx->unsynced_nodes_written.push_back(cur_node);
            }
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (!cur_node->src[j]) {
                    continue;
                }
                ctx->unsynced_nodes_read.push_back(cur_node->src[j]);
            }
        }
    }
    if (vk_enable_sync_logger) {
        for (int i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
            auto *n = cgraph->nodes[node_idx + i];
            std::cerr << node_idx + i << " " << ggml_op_name(n->op) << " " <<  n->name;
            if (n->op == GGML_OP_GLU) {
                std::cerr << " " << ggml_glu_op_name(ggml_get_glu_op(n)) << " " << (n->src[1] ? "split" : "single") << " ";
            }
            if (n->op == GGML_OP_ROPE) {
                const int mode = ((const int32_t *) n->op_params)[2];
                std::cerr << " rope mode: " << mode;
            }
            std::cerr << std::endl;
        }
    }

    switch (node->op) {
    case GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_REPEAT_BACK:
        ggml_vk_repeat_back(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ACC:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_GET_ROWS:
        ggml_vk_get_rows(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ADD:
        if (ctx->num_additional_fused_ops) {
            ggml_vk_multi_add(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_add(ctx, compute_ctx, src0, src1, node);
        }
        break;
    case GGML_OP_SUB:
        ggml_vk_sub(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_MUL:
        ggml_vk_mul(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ADD_ID:
        ggml_vk_add_id(ctx, compute_ctx, src0, src1, src2, node);

        break;
    case GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ADD1:
        ggml_vk_add1(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ARANGE:
        ggml_vk_arange(ctx, compute_ctx, node);

        break;
    case GGML_OP_FILL:
        ggml_vk_fill(ctx, compute_ctx, node);

        break;
    case GGML_OP_SCALE:
        ggml_vk_scale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SQRT:
        ggml_vk_sqrt(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SIN:
        ggml_vk_sin(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_COS:
        ggml_vk_cos(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_LOG:
        ggml_vk_log(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_TRI:
        ggml_vk_tri(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_DIAG:
        ggml_vk_diag(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ROLL:
        ggml_vk_roll(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SET_ROWS:
        ggml_vk_set_rows(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_SCATTER_ELEMENTS:
        ggml_vk_scatter_elements(ctx, compute_ctx, node);

        break;
    case GGML_OP_SILU_BACK:
        ggml_vk_silu_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_GROUP_NORM:
        ggml_vk_group_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, cgraph, node_idx, (float *)node->op_params);
        break;
    case GGML_OP_RMS_NORM_BACK:
        ggml_vk_rms_norm_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_CEIL:
        case GGML_UNARY_OP_FLOOR:
        case GGML_UNARY_OP_TRUNC:
            ggml_vk_unary(ctx, compute_ctx, src0, node);
            break;
        case GGML_UNARY_OP_XIELU:
            ggml_vk_xielu(ctx, compute_ctx, src0, node);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_GLU:
        switch (ggml_get_glu_op(node)) {
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_REGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF:
        case GGML_GLU_OP_GEGLU_QUICK:
            ggml_vk_glu(ctx, compute_ctx, src0, src1, node);
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_DIAG_MASK_INF:
        ggml_vk_diag_mask_inf(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SOFT_MAX:
        if (ctx->num_additional_fused_ops) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_soft_max(ctx, compute_ctx, src0, src1, src2, node);
        }

        break;
    case GGML_OP_SOFT_MAX_BACK:
        ggml_vk_soft_max_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, cgraph, node_idx, false);

        break;
    case GGML_OP_ROPE_BACK:
        ggml_vk_rope(ctx, compute_ctx, cgraph, node_idx, true);

        break;
    case GGML_OP_ARGSORT:
        if (ctx->num_additional_fused_ops) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_argsort(ctx, compute_ctx, src0, node);
        }

        break;
    case GGML_OP_TOP_K:
        ggml_vk_topk(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SUM:
        ggml_vk_sum(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SUM_ROWS:
        ggml_vk_sum_rows(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CUMSUM:
        ggml_vk_cumsum(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_MEAN:
        ggml_vk_mean(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ARGMAX:
        ggml_vk_argmax(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_COUNT_EQUAL:
        ggml_vk_count_equal(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_SOLVE_TRI:
        ggml_vk_solve_tri(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_REL_POS_BIAS:
        ggml_vk_rel_pos_bias(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_CAST_NUMERIC:
        ggml_vk_cpy(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_IM2COL_3D:
        ggml_vk_im2col_3d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, cgraph, node_idx);

        break;
    case GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, cgraph, node_idx);

        break;

    case GGML_OP_FLASH_ATTN_EXT:
        ggml_vk_flash_attn(ctx, compute_ctx, src0, src1, src2, src3, node->src[4], node);

        break;

    case GGML_OP_RWKV_WKV6:
        ggml_vk_rwkv_wkv6(ctx, compute_ctx, node);

        break;

    case GGML_OP_RWKV_WKV7:
        ggml_vk_rwkv_wkv7(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_SCAN:
        ggml_vk_ssm_scan(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, node);

        break;

    case GGML_OP_OPT_STEP_ADAMW:
        ggml_vk_opt_step_adamw(ctx, compute_ctx, node);

        break;

    case GGML_OP_OPT_STEP_SGD:
        ggml_vk_opt_step_sgd(ctx, compute_ctx, src0, src1, src2, node);

        break;
    default:
        return false;
    }

    ctx->tensor_ctxs[node_idx] = compute_ctx;


    if (submit || last_node) {
        ggml_vk_ctx_end(compute_ctx);

        // TODO probably it'd be better to pass a exit_node flag to ggml_vk_compute_forward
        if (last_node) {
            compute_ctx->exit_tensor_idx = node_idx_begin;
        }
        else {
            compute_ctx->exit_tensor_idx = -1;
        }

        ctx->compute_ctx.reset();

        ggml_vk_compute_forward(ctx, cgraph, node_begin, node_idx_begin, almost_ready);
    }
    return true;
}

static void ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, int tensor_idx, bool almost_ready = false) {
    GGML_UNUSED(cgraph);
    GGML_UNUSED(tensor);

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();

    // Only run if ctx hasn't been submitted yet
    if (!subctx->seqs.empty()) {

        // Do staging buffer copies
        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        for (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        if (almost_ready && !ctx->almost_ready_fence_pending) {
            ggml_vk_submit(subctx, ctx->almost_ready_fence);
            ctx->almost_ready_fence_pending = true;
        } else {
            ggml_vk_submit(subctx, {});
        }
        ctx->submit_pending = true;

    }

    if (tensor_idx == subctx->exit_tensor_idx) {
        // Do staging buffer copies
        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        subctx->in_memcpys.clear();
        subctx->out_memcpys.clear();
        subctx->memsets.clear();
    }
}

// Clean up after graph processing is done
static void ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    ctx->prealloc_y_last_pipeline_used = {};

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);

    for (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    for (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
    ctx->pipeline_descriptor_set_requirements = 0;
    ctx->descriptor_set_idx = 0;
}

// Clean up on backend free
static void ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    // discard any unsubmitted command buffers
    ctx->transfer_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_buffer(ctx->prealloc_x);
    ggml_vk_destroy_buffer(ctx->prealloc_y);
    ggml_vk_destroy_buffer(ctx->prealloc_split_k);
    ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
    ggml_vk_destroy_buffer(ctx->sync_staging);

    ctx->prealloc_y_last_pipeline_used = nullptr;

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
    ctx->device->device.destroyFence(ctx->almost_ready_fence);

    for (auto& pool : ctx->descriptor_pools) {
        ctx->device->device.destroyDescriptorPool(pool);
    }
    ctx->descriptor_pools.clear();
    ctx->descriptor_sets.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    ctx->transfer_cmd_pool.destroy(ctx->device->device);
    if (vk_perf_logger_enabled) {
        ctx->perf_logger->print_timings(true);
    }
}

static int ggml_vk_get_device_count() {
    ggml_vk_instance_init();

    return vk_instance.device_indices.size();
}

static void ggml_vk_get_device_description(int device, char * description, size_t description_size) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_size, "%s", props.deviceName.data());
}

// backend interface

#define UNUSED GGML_UNUSED

// device backend

static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t buffer) {
    return buffer->buft->iface.get_name == ggml_backend_vk_buffer_type_name;
}

static void ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    ggml_vk_destroy_buffer(ctx->dev_buffer);
    delete ctx;
}

static void * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer) {
    return vk_ptr_base;

    UNUSED(buffer);
}

static enum ggml_status ggml_backend_vk_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << buffer << " (" << buffer->context << "), " << tensor << ")");
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_vk_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_buffer_memset(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, val32, size);
}

static void ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_write(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static void ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_read(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static bool ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;
        ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;

        vk_buffer src_buf = src_buf_ctx->dev_buffer;
        vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

        ggml_vk_buffer_copy(dst_buf, vk_tensor_offset(dst) + dst->view_offs, src_buf, vk_tensor_offset(src) + src->view_offs, ggml_nbytes(src));

        return true;
    }
    return false;

    UNUSED(buffer);
}

static void ggml_backend_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    ggml_vk_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
};

// vk buffer type
static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;

    vk_buffer dev_buffer = nullptr;
    try {
        dev_buffer = ggml_vk_create_buffer_device(ctx->device, size);
    } catch (const vk::SystemError& e) {
        return nullptr;
    }

    ggml_backend_vk_buffer_context * bufctx = new ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);

    return ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, bufctx, size);
}

static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->properties.limits.minStorageBufferOffsetAlignment;
}

static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->suballocation_block_size;
}

static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);

    UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_num << ")");

    vk_device dev = ggml_vk_get_device(dev_num);

    return &dev->buffer_type;
}

// host buffer type

static const char * ggml_backend_vk_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_VK_NAME "_Host";

    UNUSED(buft);
}

static const char * ggml_backend_vk_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_VK_NAME "_Host";

    UNUSED(buffer);
}

static void ggml_backend_vk_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(vk_instance.devices[0], buffer->context);
}

static ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    void * ptr = nullptr;
    try {
        ptr = ggml_vk_host_malloc(vk_instance.devices[0], size);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory (%s)\n", e.what());
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_vk_host_buffer_free_buffer;

    return buffer;

    UNUSED(buft);
}

static size_t ggml_backend_vk_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return vk_instance.devices[0]->properties.limits.minMemoryMapAlignment;

    UNUSED(buft);
}

static size_t ggml_backend_vk_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return vk_instance.devices[0]->suballocation_block_size;

    UNUSED(buft);
}

// Should be changed to return device-specific host buffer type
// but that probably requires changes in llama.cpp
ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_vk_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_vk_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_vk_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_vk_host_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_vk_host_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0),
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    return &ggml_backend_vk_buffer_type_host;
}


// backend

static const char * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return ctx->name.c_str();
}

static void ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    delete ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_vk_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return &ctx->device->buffer_type;
}

static void ggml_backend_vk_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    auto dst_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;

    bool ret = ggml_vk_buffer_write_async(transfer_ctx, buf, dst_offset, data, size);

    if (!ret) {
        ggml_vk_ensure_sync_staging_buffer(ctx, size);
        ggml_vk_sync_buffers(nullptr, transfer_ctx);

        vk::BufferCopy buffer_cpy;
        buffer_cpy.srcOffset = 0;
        buffer_cpy.dstOffset = dst_offset;
        buffer_cpy.size = size;

        transfer_ctx->s->buffer.copyBuffer(ctx->sync_staging->buffer, buf->buffer, { buffer_cpy });
        deferred_memcpy(ctx->sync_staging->ptr, data, size, &transfer_ctx->in_memcpys);
        ggml_vk_synchronize(ctx);
    }
}

static void ggml_backend_vk_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    auto src_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;
    bool ret = ggml_vk_buffer_read_async(transfer_ctx, buf, src_offset, data, size);

    // If that failed, copy synchronously through a staging buffer
    if (!ret) {
        ggml_vk_ensure_sync_staging_buffer(ctx, size);
        ggml_vk_sync_buffers(nullptr, transfer_ctx);

        vk::BufferCopy buffer_cpy;
        buffer_cpy.srcOffset = src_offset;
        buffer_cpy.dstOffset = 0;
        buffer_cpy.size = size;

        transfer_ctx->s->buffer.copyBuffer(buf->buffer, ctx->sync_staging->buffer, { buffer_cpy });
        deferred_memcpy(data, ctx->sync_staging->ptr, size, &transfer_ctx->out_memcpys);
        ggml_vk_synchronize(ctx);
    }
}

static bool ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    if ((dst->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || dst->buffer->buft == ggml_backend_vk_host_buffer_type()) && ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;
        ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;

        vk_context transfer_ctx;

        if (ctx->transfer_ctx.expired()) {
            // Initialize new transfer context
            transfer_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
            ctx->transfer_ctx = transfer_ctx;
            ggml_vk_ctx_begin(ctx->device, transfer_ctx);
        } else {
            transfer_ctx = ctx->transfer_ctx.lock();
        }

        vk_buffer src_buf = src_buf_ctx->dev_buffer;
        vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

        ggml_vk_buffer_copy_async(transfer_ctx, dst_buf, vk_tensor_offset(dst) + dst->view_offs, src_buf, vk_tensor_offset(src) + src->view_offs, ggml_nbytes(src));
        return true;
    }

    return false;
}

static void ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    bool do_transfer = !ctx->transfer_ctx.expired();

    vk_context transfer_ctx;
    if (do_transfer) {
        transfer_ctx = ctx->transfer_ctx.lock();

        ggml_vk_ctx_end(transfer_ctx);

        for (auto& cpy : transfer_ctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ggml_vk_submit(transfer_ctx, {});
        ctx->submit_pending = true;
    }

    if (ctx->submit_pending) {
        {
            std::lock_guard<std::mutex> guard(queue_mutex);
            ctx->device->compute_queue.queue.submit({}, ctx->fence);
        }
        ggml_vk_wait_for_fence(ctx);
        ctx->submit_pending = false;
    }

    if (do_transfer) {
        for (auto& cpy : transfer_ctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        ctx->transfer_ctx.reset();
    }
}

static void ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);
}

static bool ggml_vk_is_empty(ggml_tensor * node) {
    return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
}

static bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx + 1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        // rms_norm only supports f32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }
        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }
        // rms_norm shader assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    }
    auto const &mm_add_ok = [&](const ggml_tensor *mul, const ggml_tensor *add) {
        const ggml_tensor *bias = add->src[0] == mul ? add->src[1] : add->src[0];

        // mat-vec only
        if (ggml_nrows(mul) != 1) {
            return false;
        }
        // shaders assume the types match
        if (mul->type != bias->type) {
            return false;
        }
        // shaders reuse the D shape for bias
        if (!ggml_are_same_shape(mul, bias) ||
            !ggml_are_same_stride(mul, bias)) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }
        return true;
    };

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ADD) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];

        if (!mm_add_ok(mul, add)) {
            return false;
        }
        if (ops.size() == 3) {
            if (ops.begin()[2] != GGML_OP_ADD) {
                return false;
            }
            if (!mm_add_ok(add, cgraph->nodes[node_idx + 2])) {
                return false;
            }
        }
    }

    auto const &mmid_mul_ok = [&](const ggml_tensor *mmid, const ggml_tensor *mul) {
        const ggml_tensor *scale = mul->src[1];

        if (mmid != mul->src[0]) {
            return false;
        }
        // mat-vec only
        if (!ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
            return false;
        }
        // shaders assume the types match
        if (mmid->type != scale->type) {
            return false;
        }
        // shaders assume the bias is contiguous
        if (!ggml_is_contiguous(scale)) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, scale) != 0) {
            return false;
        }
        // shader only indexes by expert index
        if (scale->ne[0] != 1 ||
            scale->ne[1] != mul->ne[1] ||
            scale->ne[2] != 1 ||
            scale->ne[3] != 1) {
            return false;
        }
        return true;
    };

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_ADD_ID) {
        // additional constraints specific to this fusion
        const ggml_tensor *mul = cgraph->nodes[node_idx];
        const ggml_tensor *add = cgraph->nodes[node_idx + 1];
        const ggml_tensor *bias = add->src[1];

        if (mul != add->src[0]) {
            return false;
        }
        // mat-vec only
        if (!ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
            return false;
        }
        // shaders assume the types match
        if (mul->type != bias->type) {
            return false;
        }
        // shaders assume the bias is contiguous
        if (!ggml_is_contiguous(bias)) {
            return false;
        }
        // the ID tensor must be the same for mul_mat_id and add_id
        if (mul->src[2] != add->src[2]) {
            return false;
        }
        // unaligned bias isn't handled
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }

        if (ops.size() == 3) {
            if (ops.begin()[2] != GGML_OP_MUL) {
                return false;
            }
            const ggml_tensor *mul = cgraph->nodes[node_idx + 2];
            return mmid_mul_ok(add, mul);
        }
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT_ID && ops.begin()[1] == GGML_OP_MUL) {
        // additional constraints specific to this fusion
        const ggml_tensor *mmid = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx + 1];

        if (!mmid_mul_ok(mmid, mul)) {
            return false;
        }
    }

    return true;
}

static bool ggml_vk_can_fuse_topk_moe(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                      int node_idx, topk_moe_mode mode) {

    const ggml_tensor * softmax;
    const ggml_tensor * weights;
    const ggml_tensor * get_rows;
    const ggml_tensor * argsort;

    switch (mode) {
    case TOPK_MOE_EARLY_SOFTMAX_NORM:
        softmax = cgraph->nodes[node_idx + 0];
        weights = cgraph->nodes[node_idx + 9];
        get_rows = cgraph->nodes[node_idx + 4];
        argsort = cgraph->nodes[node_idx + 2];
        break;
    case TOPK_MOE_EARLY_SOFTMAX:
        softmax = cgraph->nodes[node_idx + 0];
        weights = cgraph->nodes[node_idx + 4];
        get_rows = cgraph->nodes[node_idx + 4];
        argsort = cgraph->nodes[node_idx + 2];
        break;
    case TOPK_MOE_LATE_SOFTMAX:
        softmax = cgraph->nodes[node_idx + 4];
        weights = cgraph->nodes[node_idx + 5];
        get_rows = cgraph->nodes[node_idx + 2];
        argsort = cgraph->nodes[node_idx + 0];
        break;
    default:
        return false;
    }

    ggml_tensor * probs = get_rows->src[0];
    if (probs->op != GGML_OP_RESHAPE) {
        return false;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    if (probs != selection_probs) {
        return false;
    }

    const float * op_params = (const float *)softmax->op_params;

    float scale = op_params[0];
    float max_bias = op_params[1];

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        return false;
    }

    if (scale != 1.0f || max_bias != 0.0f) {
        return false;
    }

    // don't fuse when masks or sinks are present
    if (softmax->src[1] || softmax->src[2]) {
        return false;
    }

    const int n_expert = softmax->ne[0];
    if (n_expert > (1 << (num_topk_moe_pipelines-1))) {
        return false;
    }

    if (!ctx->device->subgroup_arithmetic ||
        !ctx->device->subgroup_shuffle ||
        !ctx->device->subgroup_require_full_support ||
        ctx->device->disable_fusion) {
        return false;
    }

    return true;
}

static bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                           int node_idx) {
    GGML_UNUSED(ctx);
    const ggml_tensor *rope = cgraph->nodes[node_idx + 0];
    const ggml_tensor *view = cgraph->nodes[node_idx + 1];
    const ggml_tensor *set_rows = cgraph->nodes[node_idx + 2];

    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/mrope shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_MROPE) {
        return false;
    }

    return true;
}

// Check whether the tensors overlap in memory but are not equal.
// Fusions can potenitally overwrite src tensors in ways that are not prevented
// by ggml-alloc. If the fusion is entirely elementwise, then it's OK for them
// to overlap if they are exactly equal.
// XXX TODO this check is probably missing from several fusion optimizations.
static bool ggml_vk_tensors_overlap_but_not_equal(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)a->buffer->context;
    vk_buffer a_buf = a_buf_ctx->dev_buffer;
    ggml_backend_vk_buffer_context * b_buf_ctx = (ggml_backend_vk_buffer_context *)b->buffer->context;
    vk_buffer b_buf = b_buf_ctx->dev_buffer;
    if (a_buf == b_buf) {
        auto a_base = vk_tensor_offset(a) + a->view_offs;
        auto a_size = ggml_nbytes(a);
        auto b_base = vk_tensor_offset(b) + b->view_offs;
        auto b_size = ggml_nbytes(b);

        if (a_base == b_base && a_size == b_size) {
            return false;
        }

        if ((b_base <= a_base && a_base < b_base + b_size) ||
            (a_base <= b_base && b_base < a_base + a_size)) {
            return true;
        }
    }
    return false;
}

static bool ggml_vk_can_fuse_rms_norm_mul_rope(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                               int node_idx) {
    GGML_UNUSED(ctx);
    const ggml_tensor *rms = cgraph->nodes[node_idx + 0];
    const ggml_tensor *mul = cgraph->nodes[node_idx + 1];
    const ggml_tensor *rope = cgraph->nodes[node_idx + 2];

    const int mode = ((const int32_t *) rope->op_params)[2];

    // noncontig tensors aren't tested, and don't seem common in practice
    if (!ggml_is_contiguous(rms) ||
        !ggml_is_contiguous(mul) ||
        !ggml_is_contiguous(rope)) {
        return false;
    }

    // only norm/neox are handled in the shader
    if (mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_NORMAL) {
        return false;
    }

    // shared memory size for passing data from mul->rope
    if (mul->ne[0] > 1024) {
        return false;
    }

    // must not overwrite srcs in a way that's not elementwise
    ggml_tensor *other_src = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (ggml_vk_tensors_overlap_but_not_equal(rms->src[0], rope) ||
        ggml_vk_tensors_overlap_but_not_equal(other_src, rope)) {
        return false;
    }

    // conditions for pipeline creation
    if (!(ctx->device->float_controls_rte_fp16 &&
        sizeof(vk_op_rms_norm_mul_rope_push_constants) <= ctx->device->properties.limits.maxPushConstantsSize)) {
        return false;
    }

    return true;
}

static uint32_t ggml_vk_fuse_multi_add(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {

    const ggml_tensor *first_node = cgraph->nodes[node_idx];
    if (first_node->op != GGML_OP_ADD) {
        return 0;
    }

    if (!ctx->device->multi_add) {
        return 0;
    }

    int32_t num_adds = 1;
    while (node_idx + num_adds < cgraph->n_nodes &&
           cgraph->nodes[node_idx + num_adds]->op == GGML_OP_ADD &&
           num_adds < MAX_FUSED_ADDS) {
        num_adds++;
    }

    // The shader currently requires same shapes (but different strides are allowed),
    // everything f32, and no misalignment
    for (int32_t i = 0; i < num_adds; ++i) {
        const ggml_tensor *next_node = cgraph->nodes[node_idx + i];
        if (!ggml_are_same_shape(first_node, next_node->src[0]) ||
            !ggml_are_same_shape(first_node, next_node->src[1]) ||
            next_node->type != GGML_TYPE_F32 ||
            next_node->src[0]->type != GGML_TYPE_F32 ||
            next_node->src[1]->type != GGML_TYPE_F32 ||
            get_misalign_bytes(ctx, next_node) ||
            get_misalign_bytes(ctx, next_node->src[0]) ||
            get_misalign_bytes(ctx, next_node->src[1])) {
            num_adds = i;
        }
    }

    // Verify we can fuse these
    ggml_op adds[MAX_FUSED_ADDS];
    for (int32_t i = 0; i < num_adds; ++i) {
        adds[i] = GGML_OP_ADD;
    }

    // decrease num_adds if they can't all be fused
    while (num_adds > 1 && !ggml_can_fuse(cgraph, node_idx, adds, num_adds)) {
        num_adds--;
    }

    // a single add is not "fused", so just return zero
    if (num_adds == 1) {
        return 0;
    }
    return num_adds;
}

static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    if (vk_instance.debug_utils_support) {
        vk::DebugUtilsLabelEXT dul = {};
        dul.pLabelName = "ggml_backend_vk_graph_compute";
        dul.color = std::array<float,4>{1.0f, 1.0f, 1.0f, 1.0f};
        vk_instance.pfn_vkQueueBeginDebugUtilsLabelEXT(ctx->device->compute_queue.queue, reinterpret_cast<VkDebugUtilsLabelEXT*>(&dul));
    }

    ctx->prealloc_size_add_rms_partials_offset = 0;
    ctx->do_add_rms_partials = false;
    ctx->do_add_rms_partials_offset_calculation = false;

    int last_node = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    while (last_node > 0 && ggml_vk_is_empty(cgraph->nodes[last_node])) {
        last_node -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    bool first_node_in_batch = true; // true if next node will be first node in a batch
    int submit_node_idx = 0; // index to first node in a batch

    vk_context compute_ctx;
    if (vk_perf_logger_enabled) {
        // allocate/resize the query pool
        if (ctx->num_queries < cgraph->n_nodes + 1) {
            if (ctx->query_pool) {
                ctx->device->device.destroyQueryPool(ctx->query_pool);
            }
            vk::QueryPoolCreateInfo query_create_info;
            query_create_info.queryType = vk::QueryType::eTimestamp;
            query_create_info.queryCount = cgraph->n_nodes + 100;
            ctx->query_pool = ctx->device->device.createQueryPool(query_create_info);
            ctx->num_queries = query_create_info.queryCount;
            ctx->query_fusion_names.resize(ctx->num_queries);
            ctx->query_fusion_node_count.resize(ctx->num_queries);
            ctx->query_nodes.resize(ctx->num_queries);
            ctx->query_node_idx.resize(ctx->num_queries);
        }

        ctx->device->device.resetQueryPool(ctx->query_pool, 0, cgraph->n_nodes+1);
        std::fill(ctx->query_fusion_names.begin(), ctx->query_fusion_names.end(), nullptr);
        std::fill(ctx->query_fusion_node_count.begin(), ctx->query_fusion_node_count.end(), 0);
        std::fill(ctx->query_nodes.begin(), ctx->query_nodes.end(), nullptr);
        std::fill(ctx->query_node_idx.begin(), ctx->query_node_idx.end(), 0);

        GGML_ASSERT(ctx->compute_ctx.expired());
        compute_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->compute_ctx = compute_ctx;
        ggml_vk_ctx_begin(ctx->device, compute_ctx);
        ctx->query_idx = 0;
        compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
    }

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;

    if (ctx->prealloc_size_add_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, nullptr);
        if (ctx->compute_ctx.expired()) {
            compute_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
            ctx->compute_ctx = compute_ctx;
            ggml_vk_ctx_begin(ctx->device, compute_ctx);
        } else {
            compute_ctx = ctx->compute_ctx.lock();
        }
        // initialize partial sums to zero.
        ggml_vk_buffer_memset_async(compute_ctx, ctx->prealloc_add_rms_partials, 0, 0, ctx->prealloc_size_add_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of matmul work by looking at the weight matrix size, and submit every 100MB
    // (and scaled down based on model size, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without as much matmul.
    int nodes_per_submit = 100;
    int submitted_nodes = 0;
    int submit_count = 0;
    uint64_t mul_mat_bytes = 0;
    uint64_t total_mul_mat_bytes = 0;
    uint64_t mul_mat_bytes_per_submit = std::min(uint64_t(100*1000*1000), ctx->last_total_mul_mat_bytes / 40u);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (first_node_in_batch) {
            submit_node_idx = i;
        }

        if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT || cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
            auto bytes = ggml_nbytes(cgraph->nodes[i]->src[0]);
            mul_mat_bytes += bytes;
            total_mul_mat_bytes += bytes;
        }

        const char *fusion_string {};
        if (!ctx->device->disable_fusion) {
            uint32_t num_adds = ggml_vk_fuse_multi_add(ctx, cgraph, i);
            if (num_adds) {
                ctx->num_additional_fused_ops = num_adds - 1;
                fusion_string = "MULTI_ADD";
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD })) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
            } else if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { i + 4 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_mul_rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i + 2)) {
                ctx->num_additional_fused_ops = 4;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE })&&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "RMS_NORM_MUL_ROPE";
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "RMS_NORM_MUL";
            } else if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_norm_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->num_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->num_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_late_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->num_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 1;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
            }
        }
        ctx->fused_ops_write_mask |= 1 << ctx->num_additional_fused_ops;

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        bool almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        bool submit = (submitted_nodes >= nodes_per_submit) ||
                      (mul_mat_bytes_per_submit != 0 && mul_mat_bytes >= mul_mat_bytes_per_submit) ||
                      (i + ctx->num_additional_fused_ops >= last_node) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        bool enqueued = ggml_vk_build_graph(ctx, cgraph, i, cgraph->nodes[submit_node_idx], submit_node_idx, i + ctx->num_additional_fused_ops >= last_node, almost_ready, submit);

        if (vk_perf_logger_enabled && enqueued) {
            if (ctx->compute_ctx.expired()) {
                compute_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
                ctx->compute_ctx = compute_ctx;
                ggml_vk_ctx_begin(ctx->device, compute_ctx);
            } else {
                compute_ctx = ctx->compute_ctx.lock();
            }
            if (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_names[ctx->query_idx] = fusion_string;
                compute_ctx->s->buffer.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
            } else {
                // track a fusion string and number of fused ops for the current node_idx
                ctx->query_fusion_names[i] = fusion_string;
                ctx->query_fusion_node_count[i] = ctx->num_additional_fused_ops;
            }
        }

        if (enqueued) {
            ++submitted_nodes;

            if (first_node_in_batch) {
                first_node_in_batch = false;
            }
        }

        if (submit && enqueued) {
            first_node_in_batch = true;
            submitted_nodes = 0;
            mul_mat_bytes = 0;
            if (submit_count < 3) {
                mul_mat_bytes_per_submit *= 2;
            }
            submit_count++;
        }
        i += ctx->num_additional_fused_ops;
        ctx->num_additional_fused_ops = 0;
        ctx->fused_ops_write_mask = 0;
    }

    ctx->last_total_mul_mat_bytes = total_mul_mat_bytes;

    if (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.lock();
        ggml_vk_ctx_end(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, true, UINT64_MAX), "GGML_VULKAN_PERF waitForFences");
        ctx->device->device.resetFences({ ctx->device->fence });

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait), "get timestamp results");
        if (!vk_perf_logger_concurrent) {
            // Log each op separately
            for (int i = 1; i < ctx->query_idx; i++) {
                auto node = ctx->query_nodes[i];
                auto name = ctx->query_fusion_names[i];
                ctx->perf_logger->log_timing(node, name, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        } else {
            // Log each group of nodes
            int prev_node_idx = 0;
            for (int i = 1; i < ctx->query_idx; i++) {
                auto cur_node_idx = ctx->query_node_idx[i];
                std::vector<ggml_tensor *> nodes;
                std::vector<const char *> names;
                for (int node_idx = prev_node_idx; node_idx < cur_node_idx; ++node_idx) {
                    if (ggml_op_is_empty(cgraph->nodes[node_idx]->op)) {
                        continue;
                    }
                    nodes.push_back(cgraph->nodes[node_idx]);
                    names.push_back(ctx->query_fusion_names[node_idx]);
                    node_idx += ctx->query_fusion_node_count[node_idx];
                }
                prev_node_idx = cur_node_idx;
                ctx->perf_logger->log_timing(nodes, names, uint64_t((timestamps[i] - timestamps[i-1]) * ctx->device->properties.limits.timestampPeriod));
            }
        }
        ctx->perf_logger->print_timings();
    }

    if (!ctx->device->support_async) {
        ggml_vk_synchronize(ctx);
    }

    return GGML_STATUS_SUCCESS;

    UNUSED(backend);
}

// Sort the graph for improved parallelism.
static void ggml_vk_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    if (ctx->device->disable_graph_optimize) {
        return;
    }

    auto const &is_empty = [](ggml_tensor * node) -> bool {
        return node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    auto const &is_src_of = [](const ggml_tensor *dst, const ggml_tensor *src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor *dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor *src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    std::vector<ggml_tensor *> new_order;
    std::vector<bool> used(graph->n_nodes, false);
    std::set<ggml_tensor *> used_node_set;

    int first_unused = 0;
    while (first_unused < graph->n_nodes) {
        std::vector<int> current_set;

        // Check for fusion patterns and avoid reordering them
        auto const &match_pattern = [&](const std::initializer_list<ggml_op> &pattern, int start) -> bool {
            if (start + (int)pattern.size() <= graph->n_nodes) {
                bool is_pattern = true;
                for (size_t j = 0; j < pattern.size(); ++j) {
                    if (graph->nodes[start + j]->op != pattern.begin()[j] || used[start + j]) {
                        is_pattern = false;
                    }
                }
                return is_pattern;
            }
            return false;
        };

        auto const &keep_pattern = [&](const std::initializer_list<ggml_op> &pattern) -> bool {
            if (match_pattern(pattern, first_unused)) {
                for (size_t j = 0; j < pattern.size(); ++j) {
                    new_order.push_back(graph->nodes[first_unused + j]);
                    used_node_set.insert(graph->nodes[first_unused + j]);
                    used[first_unused + j] = true;
                }
                while (first_unused < graph->n_nodes && used[first_unused]) {
                    first_unused++;
                }
                return true;
            }
            return false;
        };

        if (keep_pattern(topk_moe_early_softmax_norm)) {
            continue;
        }
        if (keep_pattern(topk_moe_early_softmax)) {
            continue;
        }
        if (keep_pattern(topk_moe_late_softmax)) {
            continue;
        }

        // First, grab the next unused node.
        current_set.push_back(first_unused);

        // Loop through the next N nodes. Grab any that don't depend on other nodes that
        // haven't already been run. Nodes that have already been run have used[i] set
        // to true. Allow nodes that depend on the previous node if it's a fusion pattern
        // that we support (e.g. RMS_NORM + MUL).
        // This first pass only grabs "real" (non-view nodes). Second pass grabs view nodes.
        // The goal is to not interleave real and view nodes in a way that breaks fusion.
        const int NUM_TO_CHECK = 20;
        for (int j = first_unused+1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
            if (used[j]) {
                continue;
            }
            if (is_empty(graph->nodes[j])) {
                continue;
            }
            // Don't pull forward nodes from fusion patterns
            if (match_pattern(topk_moe_early_softmax_norm, j) ||
                match_pattern(topk_moe_early_softmax, j) ||
                match_pattern(topk_moe_late_softmax, j)) {
                continue;
            }
            bool ok = true;
            for (int c = first_unused; c < j; ++c) {
                if (!used[c] &&
                    is_src_of(graph->nodes[j], graph->nodes[c]) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ADD_ID) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_ADD && graph->nodes[j]->op == GGML_OP_ADD)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                current_set.push_back(j);

                int rope_idx = j;

                // When we've found RMS_NORM + MUL, try to find a ROPE that uses it
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_MUL &&
                    graph->nodes[j-1]->op == GGML_OP_RMS_NORM) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_ROPE &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // Check that other srcs are already valid
                            graph->nodes[k]->src[1]->op == GGML_OP_NONE &&
                            (graph->nodes[k]->src[2] == nullptr || graph->nodes[k]->src[2]->op == GGML_OP_NONE)) {
                            rope_idx = k;
                            current_set.push_back(rope_idx);
                            used[rope_idx] = true;
                            break;
                        }
                    }
                }
                // Look for ROPE + VIEW + SET_ROWS and make them consecutive
                if (graph->nodes[rope_idx]->op == GGML_OP_ROPE) {
                    int view_idx = -1;
                    int set_rows_idx = -1;
                    for (int k = rope_idx+1; k < std::min(rope_idx + 10, graph->n_nodes); ++k) {
                        if (view_idx == -1 &&
                            graph->nodes[k]->op == GGML_OP_VIEW &&
                            graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            continue;
                        }
                        if (view_idx != -1 &&
                            set_rows_idx == -1 &&
                            graph->nodes[k]->op == GGML_OP_SET_ROWS &&
                            graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            set_rows_idx = k;
                            break;
                        }
                    }
                    if (set_rows_idx != -1) {
                        current_set.push_back(view_idx);
                        current_set.push_back(set_rows_idx);
                        used[view_idx] = true;
                        used[set_rows_idx] = true;
                    }
                }
                // Look for MUL_MAT_ID + ADD_ID + MUL
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD_ID &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT_ID) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_MUL &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_node_set.find(graph->nodes[k]->src[1]) != used_node_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
                // Look for MUL_MAT + ADD + ADD
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_MUL_MAT) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_ADD &&
                            graph->nodes[k]->src[0] == graph->nodes[j] &&
                            // src1 must either be weights or already processed
                            (graph->nodes[k]->src[1]->op == GGML_OP_NONE || used_node_set.find(graph->nodes[k]->src[1]) != used_node_set.end())) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
            }
        }
        // Second pass grabs view nodes.
        // Skip this if it would break a fusion optimization (don't split up add->rms_norm or add->add).
        if (graph->nodes[current_set.back()]->op != GGML_OP_ADD) {
            for (int j = first_unused+1; j < std::min(first_unused + NUM_TO_CHECK, graph->n_nodes); ++j) {
                if (used[j]) {
                    continue;
                }
                if (!is_empty(graph->nodes[j])) {
                    continue;
                }
                bool ok = true;
                for (int c = first_unused; c < j; ++c) {
                    bool c_in_current_set = std::find(current_set.begin(), current_set.end(), c) != current_set.end();
                    // skip views whose srcs haven't been processed.
                    if (!used[c] &&
                        is_src_of(graph->nodes[j], graph->nodes[c]) &&
                        !c_in_current_set) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    current_set.push_back(j);
                }
            }
        }

        // Push the current set into new_order
        for (auto c : current_set) {
            new_order.push_back(graph->nodes[c]);
            used_node_set.insert(graph->nodes[c]);
            used[c] = true;
        }
        while (first_unused < graph->n_nodes && used[first_unused]) {
            first_unused++;
        }
    }
    // Replace the graph with the new order.
    for (int i = 0; i < graph->n_nodes; ++i) {
        graph->nodes[i] = new_order[i];
    }
}

static void ggml_backend_vk_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    // the backend interface doesn't have an explicit reset, so reset it here
    // before we record the command to set it
    ctx->device->device.resetEvent(vkev->event);
    ctx->device->device.resetFences({ vkev->fence });

    ggml_vk_set_event(transfer_ctx, vkev->event);

    ggml_vk_ctx_end(transfer_ctx);

    ggml_vk_submit(transfer_ctx, {vkev->fence});
    ctx->submit_pending = true;
    ctx->transfer_ctx.reset();
}

static void ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    ggml_vk_wait_events(transfer_ctx, {vkev->event});
    ggml_vk_ctx_end(transfer_ctx);
    ctx->transfer_ctx.reset();
}

// TODO: enable async and synchronize
static ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,  // ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .event_record            = */ ggml_backend_vk_event_record,
    /* .event_wait              = */ ggml_backend_vk_event_wait,
    /* .graph_optimize          = */ ggml_vk_graph_optimize,
};

static ggml_guid_t ggml_backend_vk_guid() {
    static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    return &guid;
}

ggml_backend_t ggml_backend_vk_init(size_t dev_num) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_num << ")");

    ggml_backend_vk_context * ctx = new ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_num);

    ggml_backend_t vk_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_vk_guid(),
        /* .iface   = */ ggml_backend_vk_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_vk_reg(), dev_num),
        /* .context = */ ctx,
    };

    if (!ctx->device->support_async) {
        vk_backend->iface.get_tensor_async = nullptr;
    }

    return vk_backend;
}

bool ggml_backend_is_vk(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

int ggml_backend_vk_get_device_count() {
    return ggml_vk_get_device_count();
}

void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size) {
    GGML_ASSERT(device < (int) vk_instance.device_indices.size());
    int dev_idx = vk_instance.device_indices[device];
    ggml_vk_get_device_description(dev_idx, description, description_size);
}

void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total) {
    GGML_ASSERT(device < (int) vk_instance.device_indices.size());
    GGML_ASSERT(device < (int) vk_instance.device_supports_membudget.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgetprops;
    vk::PhysicalDeviceMemoryProperties2 memprops = {};
    const bool membudget_supported = vk_instance.device_supports_membudget[device];
    const bool is_integrated_gpu = vkdev.getProperties().deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    if (membudget_supported) {
        memprops.pNext = &budgetprops;
    }
    vkdev.getMemoryProperties2(&memprops);

    *total = 0;
    *free = 0;

    for (uint32_t i = 0; i < memprops.memoryProperties.memoryHeapCount; ++i) {
        const vk::MemoryHeap & heap = memprops.memoryProperties.memoryHeaps[i];

        if (is_integrated_gpu || (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            *total += heap.size;

            if (membudget_supported && i < budgetprops.heapUsage.size()) {
                *free += budgetprops.heapBudget[i] - budgetprops.heapUsage[i];
            } else {
                *free += heap.size;
            }
        }
    }
}

void ggml_backend_vk_get_device_caps(int device_idx, bool * coopmat_support, bool * coopmat1_fa_support,
                                      bool * fp16, uint32_t * subgroup_size, bool * subgroup_no_shmem,
                                      uint32_t * subgroup_min_size, uint32_t * subgroup_max_size,
                                      uint32_t * wavefronts_per_simd, bool * bf16, bool * integer_dot_product,
                                      const char ** arch_name,
                                      uint32_t * coopmat_m, uint32_t * coopmat_n, uint32_t * coopmat_k,
                                      bool * supports_256_push_constants, uint32_t * max_push_constants_size) {
    ggml_vk_instance_init();
    GGML_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());
    vk_device dev = ggml_vk_get_device((size_t)device_idx);
    if (coopmat_support)    *coopmat_support    = dev->coopmat_support;
    if (coopmat1_fa_support)*coopmat1_fa_support = dev->coopmat1_fa_support;
    if (fp16)               *fp16               = dev->fp16;
    if (subgroup_size)      *subgroup_size       = dev->subgroup_size;
    // subgroup_no_shmem is active when subgroup_arithmetic is enabled and not AMD GCN
    if (subgroup_no_shmem)  *subgroup_no_shmem  = dev->subgroup_arithmetic &&
                                                   dev->architecture != vk_device_architecture::AMD_GCN;
    if (subgroup_min_size)    *subgroup_min_size    = dev->subgroup_min_size;
    if (subgroup_max_size)    *subgroup_max_size    = dev->subgroup_max_size;
    if (wavefronts_per_simd)  *wavefronts_per_simd  = dev->wavefronts_per_simd;
    if (bf16)                 *bf16                 = dev->bf16;
    if (integer_dot_product)  *integer_dot_product  = dev->integer_dot_product;
    if (arch_name) {
        switch (dev->architecture) {
            case vk_device_architecture::AMD_GCN:          *arch_name = "AMD_GCN";          break;
            case vk_device_architecture::AMD_RDNA1:        *arch_name = "AMD_RDNA1";        break;
            case vk_device_architecture::AMD_RDNA2:        *arch_name = "AMD_RDNA2";        break;
            case vk_device_architecture::AMD_RDNA3:        *arch_name = "AMD_RDNA3";        break;
            case vk_device_architecture::AMD_RDNA4:        *arch_name = "AMD_RDNA4";        break;
            case vk_device_architecture::INTEL_XE2:        *arch_name = "INTEL_XE2";        break;
            case vk_device_architecture::NVIDIA_PRE_TURING:*arch_name = "NVIDIA_PRE_TURING";break;
            default:                                        *arch_name = "OTHER";            break;
        }
    }
    if (coopmat_m) *coopmat_m = dev->coopmat_m;
    if (coopmat_n) *coopmat_n = dev->coopmat_n;
    if (coopmat_k) *coopmat_k = dev->coopmat_k;
    if (supports_256_push_constants) *supports_256_push_constants = dev->supports_256_push_constants;
    if (max_push_constants_size)     *max_push_constants_size     = dev->properties.limits.maxPushConstantsSize;
}

static vk::PhysicalDeviceType ggml_backend_vk_get_device_type(int device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    vk::PhysicalDeviceProperties2 props = {};
    device.getProperties2(&props);

    return props.properties.deviceType;
}

static std::string ggml_backend_vk_get_device_pci_id(int device_idx) {
    GGML_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());

    vk::PhysicalDevice device = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device_idx]];

    const std::vector<vk::ExtensionProperties> ext_props = device.enumerateDeviceExtensionProperties();

    bool ext_support = false;

    for (const auto& properties : ext_props) {
        if (strcmp("VK_EXT_pci_bus_info", properties.extensionName) == 0) {
            ext_support = true;
            break;
        }
    }

    if (!ext_support) {
        return "";
    }

    vk::PhysicalDeviceProperties2 props = {};
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info = {};

    props.pNext = &pci_bus_info;

    device.getProperties2(&props);

    const uint32_t pci_domain = pci_bus_info.pciDomain;
    const uint32_t pci_bus = pci_bus_info.pciBus;
    const uint32_t pci_device = pci_bus_info.pciDevice;
    const uint8_t pci_function = (uint8_t) pci_bus_info.pciFunction; // pci function is between 0 and 7, prevent printf overflow warning

    char pci_bus_id[16] = {};
    snprintf(pci_bus_id, sizeof(pci_bus_id), "%04x:%02x:%02x.%x", pci_domain, pci_bus, pci_device, pci_function);

    return std::string(pci_bus_id);
}

//////////////////////////

struct ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
};

static const char * ggml_backend_vk_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_vk_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_vk_device_get_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)device->context;
    ggml_backend_vk_get_device_memory(ctx->device, free, total);
}

static ggml_backend_buffer_type_t ggml_backend_vk_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ggml_backend_vk_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_vk_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return ggml_backend_vk_host_buffer_type();
}

static enum ggml_backend_dev_type ggml_backend_vk_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    return ctx->is_integrated_gpu ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_vk_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;

    props->name        = ggml_backend_vk_device_get_name(dev);
    props->description = ggml_backend_vk_device_get_description(dev);
    props->type        = ggml_backend_vk_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_vk_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ggml_backend_vk_init(ctx->device);
}

static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_TRUNC:
                    return ggml_is_contiguous(op->src[0]) &&
                           (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous(op->src[0]) &&
                           (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                const vk_device& device = ggml_vk_get_device(ctx->device);
                if (op->op == GGML_OP_MUL_MAT_ID) {
                    if (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        return false;
                    }
                }
                switch (src0_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        break;
                    default:
                        return false;
                }
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                if (!(ggml_vk_dim01_contiguous(op->src[0]) || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) ||
                    !(ggml_vk_dim01_contiguous(op->src[1]) || op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16)) {
                    return false;
                }
                if (op->src[0]->type == GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_F16) {
                    // We currently don't have a bf16 x f16 shader, or an fp16->bf16 copy shader.
                    // So don't support this combination for now.
                    return false;
                }

                return true;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                bool coopmat2 = device->coopmat2;
                uint32_t HSK = op->src[1]->ne[0];
                uint32_t HSV = op->src[2]->ne[0];
                if ((HSK % 8) != 0 || (HSV % 8) != 0) {
                    return false;
                }
                if (op->src[4] && op->src[4]->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->type != GGML_TYPE_F32) {
                    return false;
                }
                if (op->src[3] && op->src[3]->type != GGML_TYPE_F16) {
                    return false;
                }
                // It's straightforward to support different K/V dequant, but would
                // significantly increase the number of pipelines
                if (op->src[1]->type != op->src[2]->type) {
                    return false;
                }
                switch (op->src[1]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q8_0:
                    // supported in scalar and coopmat1/coopmat2 paths
                    break;
                case GGML_TYPE_Q4_K:
                    // supported in scalar and coopmat1 paths; most efficient when HSK is a multiple of 256
                    break;
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                // K dequants currently disabled because D dimension is rounded up to 256 and runs inefficiently
                //case GGML_TYPE_Q2_K:
                //case GGML_TYPE_Q3_K:
                //case GGML_TYPE_Q5_K:
                //case GGML_TYPE_Q6_K:
                //case GGML_TYPE_IQ1_S:
                //case GGML_TYPE_IQ1_M:
                //case GGML_TYPE_IQ2_XXS:
                //case GGML_TYPE_IQ2_XS:
                //case GGML_TYPE_IQ2_S:
                //case GGML_TYPE_IQ3_XXS:
                //case GGML_TYPE_IQ3_S:
                //case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ4_NL:
                    // currently supported only in coopmat2 path
                    if (!coopmat2) {
                        return false;
                    }
                    break;
                default:
                    return false;
                }
                if (!coopmat2 && !(device->subgroup_shuffle && device->subgroup_vote)) {
                    // scalar/coopmat1 FA uses subgroupShuffle/subgroupAll
                    return false;
                }
                return true;
            }
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_I32:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_SET_ROWS:
            {
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_SCATTER_ELEMENTS:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_CONT:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != nullptr ? op->src[1]->type : src0_type;

                if (src0_type == GGML_TYPE_F32) {
                    switch (src1_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        break;
                    }
                }
                if (src1_type == GGML_TYPE_F32) {
                    switch (src0_type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        break;
                    }
                }

                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }

                if (
                    (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) ||
                    (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32)
                ) {
                    return true;
                }

                // We can handle copying from a type to the same type if it's
                // either not quantized or is quantized and contiguous.
                // We use f16 or f32 shaders to do the copy,
                // so the type/block size must be a multiple of 4.
                if (src0_type == src1_type &&
                    (!ggml_is_quantized(src0_type) || (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op))) &&
                    (ggml_type_size(src0_type) % 2) == 0) {
                    return true;
                }
                return false;
            }
        case GGML_OP_REPEAT:
            return ggml_type_size(op->type) == sizeof(float) && ggml_type_size(op->src[0]->type) == sizeof(float);
        case GGML_OP_REPEAT_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->src[2]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32;
        case GGML_OP_SILU_BACK:
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LOG:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
        case GGML_OP_ARGSORT:
            {
                if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    return false;
                }
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                // pipeline_argsort_large_f32 requires vulkan memory model.
                if (device->vulkan_memory_model) {
                    return true;
                } else {
                    return op->ne[0] <= (1 << device->max_workgroup_size_log2);
                }
            }
        case GGML_OP_TOP_K:
            {
                if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    return false;
                }
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                // We could potentially support larger, using argsort to sort the
                // whole thing. Not clear if this is needed.
                uint32_t min_pipeline = (uint32_t)log2f(float(op->ne[0])) + 1;
                if (min_pipeline >= num_topk_pipelines ||
                    !device->pipeline_topk_f32[min_pipeline]) {
                    return false;
                }
            }
            return true;
        case GGML_OP_UPSCALE:
            if (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                if ((op->op_params[0] & 0xFF) != GGML_SCALE_MODE_BILINEAR) {
                    return false;
                }
            }
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ACC:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONCAT:
            return ggml_type_size(op->src[0]->type) == ggml_type_size(GGML_TYPE_F32);
        case GGML_OP_ADD1:
            return (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        case GGML_OP_ARANGE:
        case GGML_OP_FILL:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_SCALE:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
        case GGML_OP_ROLL:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_DIAG_MASK_INF:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SOFT_MAX:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && (!op->src[1] || (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16));
        case GGML_OP_SOFT_MAX_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_CUMSUM:
            {
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
                    return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
                }
                return false;
            }
        case GGML_OP_SOLVE_TRI:
            {
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                const vk_device& device = ggml_vk_get_device(ctx->device);

                if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                const uint32_t N = op->src[0]->ne[0];
                const uint32_t K = op->src[1]->ne[0];
                // K dimension limited to workgroup size
                if (K > 1u << device->max_workgroup_size_log2) {
                    return false;
                }
                const uint32_t batch_N = device->properties.limits.maxComputeSharedMemorySize / ((N + K) * sizeof(float));

                if (batch_N == 0) {
                    return false;
                }
                return true;
            }
        case GGML_OP_REL_POS_BIAS:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_CAST_NUMERIC:
            {
                const ggml_type st = op->src[0]->type;
                const ggml_type dt = op->type;
                if (!ggml_is_contiguous(op->src[0])) return false;
                return (st == GGML_TYPE_F32  && dt == GGML_TYPE_I32)
                    || (st == GGML_TYPE_I32  && dt == GGML_TYPE_F32)
                    || (st == GGML_TYPE_F32  && dt == GGML_TYPE_F16)
                    || (st == GGML_TYPE_F16  && dt == GGML_TYPE_F32)
                    || (st == GGML_TYPE_F32  && dt == GGML_TYPE_BF16)
                    || (st == GGML_TYPE_BF16 && dt == GGML_TYPE_F32)
                    || (st == GGML_TYPE_F32  && dt == GGML_TYPE_F32);
            }
        case GGML_OP_ARGMAX:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_COUNT_EQUAL:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_I32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_I32;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1])
                && op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_IM2COL_3D:
            return op->src[1]->type == GGML_TYPE_F32
                && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16)
                && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true; // all inputs are contiguous, see ggml.c
        case GGML_OP_SSM_SCAN:
            {
                for (int i = 0; i < 6; i++) {
                    if (op->src[i] && ggml_is_quantized(op->src[i]->type)) {
                        return false;
                    }
                }
                if (op->src[6] && op->src[6]->type != GGML_TYPE_I32) {
                    return false;
                }
                if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    return false;
                }

                const uint32_t d_state = op->src[0]->ne[0];
                const uint32_t head_dim = op->src[0]->ne[1];

                bool is_mamba2 = (op->src[3] && op->src[3]->nb[1] == sizeof(float));
                if (!is_mamba2) {
                    return false;
                }

                if ((d_state != 128 && d_state != 256) || head_dim % 16 != 0) {
                    return false;
                }

                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                const vk_device& device = ggml_vk_get_device(ctx->device);

                const uint32_t SPLIT_H = 16;

                size_t stateC_size = SPLIT_H * d_state * sizeof(float);

                if (stateC_size > device->properties.limits.maxComputeSharedMemorySize) {
                    return false;
                }

                return true;
            }
        case GGML_OP_SSM_CONV:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_1D:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                // Channel-contiguous format is not supported yet.
                return ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        default:
            return false;
    }

    UNUSED(dev);
}

static bool ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        return false;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_buffer_type_context * buft_ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return buft_ctx->device->idx == ctx->device;
}

static bool ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);
    return true;
}

static ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = new vk_event;
    if (!vkev) {
        return nullptr;
    }

    // The event/fence is expected to initially be in the signaled state.
    vkev->event = device->device.createEvent({});
    vkev->fence = device->device.createFence({vk::FenceCreateFlagBits::eSignaled});
    device->device.setEvent(vkev->event);

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

static void ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroyFence(vkev->fence);
    device->device.destroyEvent(vkev->event);
    delete vkev;
    delete event;
}

static void ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    VK_CHECK(device->device.waitForFences({ vkev->fence }, true, UINT64_MAX), "event_synchronize");
}

static const struct ggml_backend_device_i ggml_backend_vk_device_i = {
    /* .get_name             = */ ggml_backend_vk_device_get_name,
    /* .get_description      = */ ggml_backend_vk_device_get_description,
    /* .get_memory           = */ ggml_backend_vk_device_get_memory,
    /* .get_type             = */ ggml_backend_vk_device_get_type,
    /* .get_props            = */ ggml_backend_vk_device_get_props,
    /* .init_backend         = */ ggml_backend_vk_device_init,
    /* .get_buffer_type      = */ ggml_backend_vk_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_vk_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_vk_device_supports_op,
    /* .supports_buft        = */ ggml_backend_vk_device_supports_buft,
    /* .offload_op           = */ ggml_backend_vk_device_offload_op,
    /* .event_new            = */ ggml_backend_vk_device_event_new,
    /* .event_free           = */ ggml_backend_vk_device_event_free,
    /* .event_synchronize    = */ ggml_backend_vk_device_event_synchronize,
};

static const char * ggml_backend_vk_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return GGML_VK_NAME;
}

static size_t ggml_backend_vk_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return ggml_backend_vk_get_device_count();
}

static ggml_backend_dev_t ggml_backend_vk_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;

    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            for (int i = 0; i < ggml_backend_vk_get_device_count(); i++) {
                ggml_backend_vk_device_context * ctx = new ggml_backend_vk_device_context;
                char desc[256];
                ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                devices.push_back(new ggml_backend_device {
                    /* .iface   = */ ggml_backend_vk_device_i,
                    /* .reg     = */ reg,
                    /* .context = */ ctx,
                });
            }
            initialized = true;
        }
    }

    GGML_ASSERT(device < devices.size());
    return devices[device];
}

static const struct ggml_backend_reg_i ggml_backend_vk_reg_i = {
    /* .get_name         = */ ggml_backend_vk_reg_get_name,
    /* .get_device_count = */ ggml_backend_vk_reg_get_device_count,
    /* .get_device       = */ ggml_backend_vk_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_vk_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_vk_reg_i,
        /* .context     = */ nullptr,
    };
    try {
        ggml_vk_instance_init();
        return &reg;
    } catch (const vk::SystemError& e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: System error: " << e.what());
        return nullptr;
    } catch (const std::exception &e) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: " << e.what());
        return nullptr;
    } catch (...) {
        VK_LOG_DEBUG("ggml_backend_vk_reg() -> Error: unknown exception during Vulkan init");
        return nullptr;
    }
}

// Extension availability
static bool ggml_vk_instance_layer_settings_available() {
#ifdef GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    const std::string layer_name = "VK_LAYER_KHRONOS_validation";
    for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
        if (layer_name == layer.layerName.data()) {
            for (const auto& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                if (strcmp("VK_EXT_layer_settings", ext.extensionName.data()) == 0) {
                    return true;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::endl;
#endif
    return false;
}
static bool ggml_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
#endif
    return false;

    UNUSED(instance_extensions);
}

// Extension availability
static bool ggml_vk_instance_debug_utils_ext_available(
    const std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    for (const auto & properties : instance_extensions) {
        if (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            return true;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::endl;
    return false;

    UNUSED(instance_extensions);
}

static bool ggml_vk_device_is_supported(const vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    return vk11_features.storageBuffer16BitAccess;
}

static bool ggml_vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props, const vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    switch (props.vendorID) {
    case VK_VENDOR_ID_INTEL:
        // Only allowing Xe2 GPU at the moment since Xe2 GPU can gain significant performance boost,
        // while some older hardware (ex. Arc A770) has performance regressions
        return arch == vk_device_architecture::INTEL_XE2;
    case VK_VENDOR_ID_AMD:
        if (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            return arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        return true;
    default:
        return true;
    }
}

// checks


// GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)
// Disabled: we statically link backends in R package, not using dynamic loading
