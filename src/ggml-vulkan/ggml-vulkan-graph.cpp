#include "../r_dbg_filelog.h" /* opt-in trace via GGMLR_DBG_LOG env (no-op when unset) */
#ifdef GGML_VK_HARD_EXIT
#include <unistd.h>           /* _exit() for ggml_backend_vk_shutdown(hard=1, status) */
#endif

// Does this tensor sit in a buffer owned by the Vulkan backend?
//
// The only safe precondition for casting buffer->context to
// ggml_backend_vk_buffer_context. A graph handed to this backend may hold
// tensors that live on the CPU backend -- the scheduler puts an op it cannot
// run there and leaves its output where it produced it -- and for those the
// context is a foreign struct or null, so the cast yields a wild pointer.
//
// Same identity test as ggml_backend_buffer_is_vk() below, written separately
// because that one is defined further down this file and callers appear above
// it. ggml_backend_vk_buffer_type_name is forward-declared in
// ggml-vulkan-device.cpp, which precedes this file in the translation unit.
static inline bool vk_tensor_is_vk_buffer(const ggml_tensor * t) {
    return t != nullptr
        && t->buffer != nullptr
        && t->buffer->buft != nullptr
        && t->buffer->buft->iface.get_name == ggml_backend_vk_buffer_type_name
        && t->buffer->context != nullptr;
}

static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx, vk_context subctx) {

    if (subctx) {
        // Submit and wait for any pending work before reallocating the buffers
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, {});
        ctx->submit_pending = true;
        ggml_vk_synchronize(ctx);
        GGML_ASSERT(ctx->compute_ctx.expired());
        ggml_vk_ctx_begin(ctx->device, subctx);
        ctx->compute_ctx = subctx;
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
        ctx->prealloc_y_last_pipeline_used = nullptr;
        ctx->prealloc_y_last_tensor_used = nullptr;
        ctx->prealloc_y_last_k_padded = false;
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

static void ggml_vk_compute_forward(ggml_backend_vk_context* ctx, ggml_cgraph * cgraph, ggml_tensor* tensor, int tensor_idx, bool almost_ready = false);

// GGML_VK_TRACE_ORDER=1: diagnostics for graph_optimize reordering + barrier
// placement. Printed through GGML_LOG_INFO -- std::cerr is a null stream in
// this build (ggml-vulkan-device.cpp), so GGML_VK_SYNC_LOGGER prints nothing.
static bool ggml_vk_trace_order() {
    static const bool on = getenv("GGML_VK_TRACE_ORDER") != nullptr;
    return on;
}

// "[buf:base+size)" for a Vulkan-resident tensor, "-" otherwise.
static std::string ggml_vk_trace_range(const ggml_tensor * t) {
    if (t == nullptr || !vk_tensor_is_vk_buffer(t)) {
        return "-";
    }
    ggml_backend_vk_buffer_context * bc = (ggml_backend_vk_buffer_context *)t->buffer->context;
    char s[96];
    snprintf(s, sizeof(s), "[%p:%zu+%zu)", (void *)bc->dev_buffer.get(),
             (size_t)(vk_tensor_offset(t) + t->view_offs), ggml_nbytes(t));
    return s;
}

// After reordering: every source (and the view_src it aliases) that is itself
// a node of this graph must come earlier. Prints the order and each violation.
static void ggml_vk_trace_check_order(const ggml_cgraph * graph) {
    std::map<const ggml_tensor *, int> pos;
    for (int i = 0; i < graph->n_nodes; ++i) {
        pos[graph->nodes[i]] = i;
    }
    int bad = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * n = graph->nodes[i];
        GGML_LOG_INFO("[vk-order] %3d %-12s %s\n", i, ggml_op_name(n->op), n->name);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            for (const ggml_tensor * p = n->src[s]; p != nullptr; p = p->view_src) {
                auto it = pos.find(p);
                if (it != pos.end() && it->second >= i) {
                    GGML_LOG_INFO("[vk-order] VIOLATION: node %d '%s' reads '%s' produced at %d\n",
                                  i, n->name, p->name, it->second);
                    bad++;
                }
            }
        }
    }
    GGML_LOG_INFO("[vk-order] %d nodes, %d violation(s)\n", graph->n_nodes, bad);
}

// Returns true if node has enqueued work into the queue, false otherwise
// If submit is true the current all operations queued so far are being submitted to Vulkan to overlap cmdlist creation and GPU execution.
static bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int node_idx, ggml_tensor *node_begin, int node_idx_begin, bool last_node, bool almost_ready, bool submit){
    ggml_tensor * node = cgraph->nodes[node_idx];
    if (ggml_is_empty(node) || ggml_op_is_empty(node->op) || !node->buffer) {
        return false;
    }
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << node << ", " << ggml_op_name(node->op) << ")");
    r_dbg_logf("vk_node[%d] op=%s name='%s' ne=[%lld,%lld,%lld,%lld] type=%s",
               node_idx, ggml_op_name(node->op),
               node->name[0] ? node->name : "<unnamed>",
               (long long)node->ne[0], (long long)node->ne[1],
               (long long)node->ne[2], (long long)node->ne[3],
               ggml_type_name(node->type));
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

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

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
            // Only a VULKAN buffer may be cast to ggml_backend_vk_buffer_context.
            // Testing buffer != nullptr is not enough and was the bug here: this
            // graph holds tensors whose buffer belongs to the CPU backend (a
            // CPU-only op's output, e.g. the NMS custom op in an ONNX detector),
            // and for those ->context is either a different struct or null, so
            // reading ->dev_buffer through the cast lands on address 0. That is
            // the `address (nil)` crash on MaskRCNN-12-int8.
            //
            // A tensor on another backend also cannot overlap Vulkan device
            // memory, so skipping it is the correct answer, not just a safe one.
            //
            // The !node->buffer test at the top of ggml_vk_build_graph() does
            // not cover this. It guards cgraph->nodes[node_idx] alone, while
            // the callers below pass cur_node->src[j] (a source, never tested)
            // and, when ops are fused, nodes[node_idx + i] for i > 0 (also
            // never tested).
            if (!vk_tensor_is_vk_buffer(node)) {
                return false;
            }
            auto n_base = vk_tensor_offset(node) + node->view_offs;
            auto n_size = ggml_nbytes(node);
            ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)node->buffer->context;
            vk_buffer a_buf = a_buf_ctx->dev_buffer;
            for (auto &other : unsynced_nodes) {
                if (other == nullptr || !vk_tensor_is_vk_buffer(other)) {
                    continue;
                }
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

        if (ggml_vk_trace_order()) {
            const ggml_tensor * n = cgraph->nodes[node_idx];
            std::string srcs;
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (n->src[j]) {
                    srcs += " ";
                    srcs += ggml_vk_trace_range(n->src[j]);
                }
            }
            GGML_LOG_INFO("[vk-sync] %3d %-12s %-20s fused+%d dst %s src%s %s\n",
                          node_idx, ggml_op_name(n->op), n->name,
                          ctx->num_additional_fused_ops,
                          ggml_vk_trace_range(n).c_str(), srcs.c_str(),
                          need_sync ? "SYNC" : "");
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
                compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
            }
        }
        // Add all fused nodes to the unsynchronized lists.
        for (int32_t i = 0; i < ctx->num_additional_fused_ops + 1; ++i) {
            const ggml_tensor *cur_node = cgraph->nodes[node_idx + i];
            // Multiple outputs could be written, e.g. in topk_moe. Add them all to the list.
            if ((ctx->fused_ops_write_mask & (1 << i)) && vk_tensor_is_vk_buffer(cur_node)) {
                ctx->unsynced_nodes_written.push_back(cur_node);
            }
            for (uint32_t j = 0; j < GGML_MAX_SRC; ++j) {
                if (!cur_node->src[j]) {
                    continue;
                }
                // Keep non-Vulkan tensors out of the lists entirely. These lists
                // answer "does this overlap in Vulkan device memory", which a
                // tensor on another backend cannot; storing one only means a
                // later node reads it back through the vk context cast and dies.
                if (!vk_tensor_is_vk_buffer(cur_node->src[j])) {
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

    // closed explicitly below, and by the destructor on the early returns
    ggml_vk_debug_label dbg(compute_ctx, cgraph, node_idx, ctx->num_additional_fused_ops);

    switch (node->op) {
    case GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_REPEAT_BACK:
        ggml_vk_repeat_back(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_ACC:
    case GGML_OP_SET:
        ggml_vk_acc(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_GET_ROWS:
        if (ctx->fused_topk_qsa) {
            ggml_vk_topk_qsa(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_get_rows(ctx, compute_ctx, src0, src1, node);
        }

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
        if (ctx->num_additional_fused_ops) {
            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, node_idx);
        } else {
            ggml_vk_mul(ctx, compute_ctx, src0, src1, node);
        }

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
        if (ctx->fused_hc_post_gate) {
            ggml_tensor * hc_post = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];
            ggml_vk_dsv4_hc_post(ctx, compute_ctx, hc_post->src[0], hc_post->src[1], hc_post->src[2], hc_post->src[3], hc_post, node);
        } else {
            ggml_vk_scale(ctx, compute_ctx, src0, node);
        }

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
    case GGML_OP_PAD_REFLECT_1D:
        ggml_vk_pad_reflect_1d(ctx, compute_ctx, src0, node);

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
    case GGML_OP_GELU_BACK:
        ggml_vk_gelu_back(ctx, compute_ctx, src0, src1, node);

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
    // DIVERGENCE from upstream: LayerNorm backward, see ggml_norm_back().
    case GGML_OP_NORM_BACK:
        ggml_vk_norm_back(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_L2_NORM:
        ggml_vk_l2_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_UNARY:
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
            ggml_vk_topk_moe(ctx, compute_ctx, cgraph, node_idx);
            break;
        }
        if (ctx->num_additional_fused_ops) {
            ggml_vk_unary_mul(ctx, compute_ctx, cgraph, node_idx);
            break;
        }

        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_ELU:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_EXPM1:
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_HARDSIGMOID:
        case GGML_UNARY_OP_HARDSWISH:
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_STEP:
        case GGML_UNARY_OP_ROUND:
        case GGML_UNARY_OP_ROUND_EVEN:
        case GGML_UNARY_OP_CEIL:
        case GGML_UNARY_OP_FLOOR:
        case GGML_UNARY_OP_TRUNC:
        case GGML_UNARY_OP_SGN:
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
        case GGML_GLU_OP_SWIGLU_CLAMP:
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
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
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
        if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
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
    case GGML_OP_CROSS_ENTROPY_LOSS:
        ggml_vk_cross_entropy_loss(ctx, compute_ctx, node);

        break;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        ggml_vk_cross_entropy_loss_back(ctx, compute_ctx, node);

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
    case GGML_OP_COL2IM_1D:
        ggml_vk_col2im_1d(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_TRANSPOSE_1D:
        ggml_vk_conv_transpose_1d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_POOL_1D:
        ggml_vk_pool_1d(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_POOL_2D:
        ggml_vk_pool_2d(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CONV_2D:
    case GGML_OP_CONV_TRANSPOSE_2D:
        ggml_vk_conv_2d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_CONV_3D:
        ggml_vk_conv_3d(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_CONV_2D_DW:
        ggml_vk_conv_2d_dw(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_QCONV_I32:
        // Five sources, two of them optional -- read from node rather than the
        // src0/src1 locals this switch names.
        ggml_vk_qconv_i32(ctx, compute_ctx, node->src[0], node->src[1],
                          node->src[2], node->src[3], node->src[4],
                          node->src[5], node);

        break;
    case GGML_OP_QMATMUL_I32:
        // Four sources, the zero point optional -- read from node, as above.
        ggml_vk_qmatmul_i32(ctx, compute_ctx, node->src[0], node->src[1],
                            node->src[2], node->src[3], node);

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

    case GGML_OP_GATED_LINEAR_ATTN:
        ggml_vk_gated_linear_attn(ctx, compute_ctx, node);

        break;


    case GGML_OP_DSV4_HC_COMB:
        ggml_vk_dsv4_hc_comb(ctx, compute_ctx, src0, src1, src2, node);

        break;
    case GGML_OP_DSV4_HC_PRE:
        ggml_vk_dsv4_hc_pre(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_DSV4_HC_POST:
        ggml_vk_dsv4_hc_post(ctx, compute_ctx, src0, src1, src2, src3, node);

        break;
    case GGML_OP_LIGHTNING_INDEXER:
        ggml_vk_lightning_indexer(ctx, compute_ctx, node);

        break;
    case GGML_OP_GATED_DELTA_NET:
        ggml_vk_gated_delta_net(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_SCAN:
        ggml_vk_ssm_scan(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_CONV:
        ggml_vk_ssm_conv(ctx, compute_ctx, cgraph, node_idx);

        break;

    case GGML_OP_SSM_CONV_BACK:
        ggml_vk_ssm_conv_back(ctx, compute_ctx, node);

        break;

    case GGML_OP_SSM_SCAN_BACK:
        ggml_vk_ssm_scan_back(ctx, compute_ctx, node);

        break;

    case GGML_OP_OUT_PROD:
        ggml_vk_out_prod(ctx, compute_ctx, node);

        break;

    case GGML_OP_GET_ROWS_BACK:
        ggml_vk_get_rows_back(ctx, compute_ctx, node);

        break;

    case GGML_OP_IM2COL_BACK:
        ggml_vk_im2col_back(ctx, compute_ctx, node);

        break;

    case GGML_OP_FLASH_ATTN_BACK:
        ggml_vk_flash_attn_back(ctx, compute_ctx, node);

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

    // the submit path below can end the command buffer, so close the region first
    dbg.close();

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

static void ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, ggml_tensor * tensor, int tensor_idx, bool almost_ready) {
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

        if (ctx->device->serialize_submissions) {
            ggml_vk_submit(subctx, ctx->fence);
        } else if (almost_ready && !ctx->almost_ready_fence_pending) {
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
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = false;

    ctx->unsynced_nodes_written.clear();
    ctx->unsynced_nodes_read.clear();
    ctx->prealloc_x_need_sync = ctx->prealloc_y_need_sync = ctx->prealloc_split_k_need_sync = false;

    ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);
    if (ctx->device->async_use_transfer_queue) {
        ggml_vk_command_pool_cleanup(ctx->device, ctx->transfer_cmd_pool);
    }

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
    ctx->compute_ctx.reset();
    // wait for any pending command buffers to finish
    ggml_vk_synchronize(ctx);

    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_buffer(ctx->prealloc_x);
    ggml_vk_destroy_buffer(ctx->prealloc_y);
    ggml_vk_destroy_buffer(ctx->prealloc_split_k);
    ggml_vk_destroy_buffer(ctx->prealloc_add_rms_partials);
    ggml_vk_destroy_buffer(ctx->sync_staging);

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = false;

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
    ctx->descriptor_set_bindings.clear();

    ctx->compute_cmd_pool.destroy(ctx->device->device);
    if (ctx->device->async_use_transfer_queue) {
        ctx->device->device.destroySemaphore(ctx->transfer_semaphore.s);

        ctx->transfer_cmd_pool.destroy(ctx->device->device);
    }
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

// Industrial telemetry: live Vulkan device-buffer VRAM footprint (defined with
// the alloc path below; forward-declared here so free can decrement them).
static uint64_t g_vk_live_buffer_bytes;
static uint64_t g_vk_buffer_alloc_count;

// Forward declaration: defined near supports_op (further down) but called from
// graph_compute (above its definition) to summarize CPU-fallback ops.
static void ggml_vk_log_unsupported_op_summary(const char * phase);

static void ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    size_t freed = buffer->size;
    // dev_buffer is destroyed by ~ggml_backend_vk_buffer_context (called from
    // delete ctx). Calling ggml_vk_destroy_buffer here as well was a redundant
    // double-destroy of the same shared vk_buffer; matches upstream which only
    // does `delete ctx`.
    delete ctx;
    if (g_vk_live_buffer_bytes >= freed) {
        g_vk_live_buffer_bytes -= freed;
    } else {
        g_vk_live_buffer_bytes = 0;
    }
    if (g_vk_buffer_alloc_count > 0) {
        g_vk_buffer_alloc_count--;
    }
    GGML_LOG_DEBUG("ggml_vulkan: VRAM free %.2f MB | live=%.2f MB in %llu buffers\n",
                   freed / (1024.0 * 1024.0),
                   g_vk_live_buffer_bytes / (1024.0 * 1024.0),
                   (unsigned long long)g_vk_buffer_alloc_count);
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

    if (size == 0) {
        return;
    }

    uint32_t val32 = (uint32_t)value * 0x01010101;
    ggml_vk_buffer_memset(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, val32, size);
}

static unsigned long long g_vk_set_tensor_calls = 0;
static unsigned long long g_vk_set_tensor_bytes = 0;

static void ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    g_vk_set_tensor_calls++;
    g_vk_set_tensor_bytes += size;
    r_dbg_logf("vk_set_tensor: #%llu name='%s' type=%s size=%zu off=%zu cum=%llu",
               g_vk_set_tensor_calls,
               tensor->name[0] ? tensor->name : "<unnamed>",
               ggml_type_name(tensor->type), size, offset, g_vk_set_tensor_bytes);

    if (size == 0) {
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_write(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static void ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    if (size == 0) {
        return;
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_read(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, size);
}

static void ggml_backend_vk_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor_2d(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_data << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    vk_buffer buf = buf_ctx->dev_buffer;

    if (size == 0) {
        return;
    }

    ggml_vk_buffer_write_2d(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, stride_data, stride_tensor, size, n_copies);
}

static void ggml_backend_vk_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                                 size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor_2d(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ", " <<
                 n_copies << ", " << stride_tensor << ", " << stride_data << ")");
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    if (size == 0) {
        return;
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    ggml_vk_buffer_read_2d(buf, vk_tensor_offset(tensor) + tensor->view_offs + offset, data, stride_tensor, stride_data, size, n_copies);
}

static bool ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_nbytes(src) == 0) {
        return true;
    }

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
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
};

// vk buffer type
static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

// g_vk_live_buffer_bytes / g_vk_buffer_alloc_count are forward-declared above
// (near free_buffer). Updated on alloc/free to track live VRAM footprint.

static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;

    vk_buffer dev_buffer = nullptr;
    try {
        dev_buffer = ggml_vk_create_buffer_device(ctx->device, size);
    } catch (const vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: VRAM alloc FAILED for %.2f MB (live=%.2f MB in %llu buffers): %s\n",
                      size / (1024.0 * 1024.0),
                      g_vk_live_buffer_bytes / (1024.0 * 1024.0),
                      (unsigned long long)g_vk_buffer_alloc_count,
                      e.what());
        return nullptr;
    } catch (const std::exception& e) {
        GGML_LOG_WARN("ggml_vulkan: VRAM alloc FAILED for %.2f MB (live=%.2f MB in %llu buffers): %s\n",
                      size / (1024.0 * 1024.0),
                      g_vk_live_buffer_bytes / (1024.0 * 1024.0),
                      (unsigned long long)g_vk_buffer_alloc_count,
                      e.what());
        return nullptr;
    }
    g_vk_live_buffer_bytes += size;
    g_vk_buffer_alloc_count++;
    GGML_LOG_DEBUG("ggml_vulkan: VRAM alloc %.2f MB | live=%.2f MB in %llu buffers\n",
                   size / (1024.0 * 1024.0),
                   g_vk_live_buffer_bytes / (1024.0 * 1024.0),
                   (unsigned long long)g_vk_buffer_alloc_count);

    ggml_backend_vk_buffer_context * bufctx = nullptr;
    try {
        bufctx = new ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);
    } catch (const std::exception& e) {
        return nullptr;
    }

    ggml_backend_buffer_t buf = ggml_backend_buffer_init(buft, &ggml_backend_vk_buffer_interface, bufctx, size);
    return buf;
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

static void ggml_backend_vk_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(buffer->context);
}

static ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    void * ptr = ggml_vk_host_malloc(size);
    if (ptr == nullptr) {
        GGML_LOG_WARN("ggml_vulkan: Failed to allocate pinned memory\n");
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
            /* .get_name            = */ ggml_backend_vk_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_vk_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ nullptr,
            /* .get_alignment       = */ ggml_backend_vk_host_buffer_type_get_alignment,
            /* .get_max_size        = */ ggml_backend_vk_host_buffer_type_get_max_size,
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
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

static void ggml_backend_vk_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context cpy_ctx;

    if (ctx->device->async_use_transfer_queue) {
        cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
    } else {
        cpy_ctx = ggml_vk_get_compute_ctx(ctx);
    }

    vk_buffer buf = buf_ctx->dev_buffer;

    auto dst_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;

    bool ret = ggml_vk_buffer_write_2d_async(cpy_ctx, buf, dst_offset, data, stride_data, stride_tensor, size, n_copies);

    if (!ret) {
        const size_t staging_size = size * n_copies;
        ggml_vk_ensure_sync_staging_buffer(ctx, staging_size);
        ggml_vk_sync_buffers(nullptr, cpy_ctx);

        std::vector<vk::BufferCopy> slices(1);
        if (size == stride_tensor) {
            slices[0].srcOffset = 0;
            slices[0].dstOffset = dst_offset;
            slices[0].size = staging_size;
        } else {
            slices.resize(n_copies);
            for (size_t i = 0; i < n_copies; i++) {
                slices[i].srcOffset = i * size;
                slices[i].dstOffset = dst_offset + i * stride_tensor;
                slices[i].size = size;
            }
        }

        cpy_ctx->s->buffer->buf.copyBuffer(ctx->sync_staging->buffer, buf->buffer, slices);

        if (size == stride_data) {
            deferred_memcpy(ctx->sync_staging->ptr, data, staging_size, &cpy_ctx->in_memcpys);
        } else {
            for (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)ctx->sync_staging->ptr + i * size, (const uint8_t *)data + i * stride_data, size, &cpy_ctx->in_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

static void ggml_backend_vk_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset,
                                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_2d_async(" << size << ", " << n_copies << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    if (size == 0) {
        return;
    }

    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *)tensor->buffer->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    vk_buffer buf = buf_ctx->dev_buffer;

    auto src_offset = vk_tensor_offset(tensor) + tensor->view_offs + offset;
    bool ret = ggml_vk_buffer_read_2d_async(compute_ctx, buf, src_offset, data, stride_tensor, stride_data, size, n_copies);

    if (!ret) {
        const size_t staging_size = size * n_copies;
        ggml_vk_ensure_sync_staging_buffer(ctx, staging_size);
        ggml_vk_sync_buffers(nullptr, compute_ctx);

        std::vector<vk::BufferCopy> slices(1);
        if (size == stride_tensor) {
            slices[0].srcOffset = src_offset;
            slices[0].dstOffset = 0;
            slices[0].size = staging_size;
        } else {
            slices.resize(n_copies);
            for (size_t i = 0; i < n_copies; i++) {
                slices[i].srcOffset = src_offset + i * stride_tensor;
                slices[i].dstOffset = i * size;
                slices[i].size = size;
            }
        }

        compute_ctx->s->buffer->buf.copyBuffer(buf->buffer, ctx->sync_staging->buffer, slices);

        if (size == stride_data) {
            deferred_memcpy(data, ctx->sync_staging->ptr, staging_size, &compute_ctx->out_memcpys);
        } else {
            for (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)data + i * stride_data, (const uint8_t *)ctx->sync_staging->ptr + i * size, size, &compute_ctx->out_memcpys);
            }
        }
        ggml_vk_synchronize(ctx);
    }
}

static void ggml_backend_vk_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_set_tensor_2d_async(backend, tensor, data, offset, size, 1, size, size);
}

static void ggml_backend_vk_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_get_tensor_2d_async(backend, tensor, data, offset, size, 1, size, size);
}


static bool ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async(" << src << " -> " << dst << ", size=" << ggml_nbytes(src) << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend_dst->context;

    // Skip zero-size tensors
    if (ggml_nbytes(src) == 0) {
        return true;
    }

    if (dst->buffer->buft != ggml_backend_vk_get_default_buffer_type(backend_dst)) {
        return false;
    }

    ggml_backend_vk_buffer_context * dst_buf_ctx = (ggml_backend_vk_buffer_context *)dst->buffer->context;
    vk_buffer dst_buf = dst_buf_ctx->dev_buffer;

    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_backend_vk_buffer_context * src_buf_ctx = (ggml_backend_vk_buffer_context *)src->buffer->context;

        // Async copy only works within the same device
        if (src_buf_ctx->dev_buffer->device != dst_buf->device) {
            return false;
        }

        vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

        ggml_vk_buffer_copy_async(compute_ctx, dst_buf, vk_tensor_offset(dst) + dst->view_offs,
                                   src_buf_ctx->dev_buffer, vk_tensor_offset(src) + src->view_offs,
                                   ggml_nbytes(src));
        return true;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        vk_buffer pinned_buf = nullptr;
        size_t pinned_offset = 0;
        ggml_vk_host_get(ctx->device, src->data, pinned_buf, pinned_offset);
        if (pinned_buf == nullptr) {
            return false;
        }

        // If the backend is idle, use a CPU copy to avoid GPU synchronization overhead.
        static constexpr size_t max_cpu_copy_size = 128 * 1024;
        const bool src_backend_synchronous = backend_src->iface.synchronize == nullptr;
        const bool transfer_idle = !ctx->device->async_use_transfer_queue ||
                                   ctx->transfer_semaphore_last_submitted == ctx->transfer_semaphore.value;
        const bool backend_idle = ctx->compute_ctx.expired() && ctx->transfer_ctx.expired() &&
                                  !ctx->submit_pending && !ctx->almost_ready_fence_pending && transfer_idle;
        const bool dst_host_coherent =
            (dst_buf->memory_property_flags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)) ==
            (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        if ((backend_src == backend_dst || src_backend_synchronous) && backend_idle && dst_host_coherent && ggml_nbytes(src) <= max_cpu_copy_size) {
            ggml_vk_buffer_write(dst_buf, vk_tensor_offset(dst) + dst->view_offs, src->data, ggml_nbytes(src));
            return true;
        }

        vk_context cpy_ctx;
        if (ctx->device->async_use_transfer_queue) {
            cpy_ctx = ggml_vk_get_transfer_ctx(ctx);
        } else {
            cpy_ctx = ggml_vk_get_compute_ctx(ctx);
        }

        return ggml_vk_buffer_write_async(cpy_ctx, dst_buf,
                                          vk_tensor_offset(dst) + dst->view_offs,
                                          src->data, ggml_nbytes(src));
    }

    return false;
}

static void ggml_vk_synchronize(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_synchronize()");

    bool do_transfer = !ctx->compute_ctx.expired();

    if (ggml_vk_submit_transfer_ctx(ctx)) {
        ctx->submit_pending = true;
    }

    vk_context compute_ctx;
    vk_command_buffer* cmd_buf = nullptr;
    if (do_transfer) {
        compute_ctx = ctx->compute_ctx.lock();
        if (compute_ctx->s) {
            cmd_buf = compute_ctx->s->buffer;
        }

        ggml_vk_ctx_end(compute_ctx);

        for (auto& cpy : compute_ctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        if (ctx->device->serialize_submissions) {
            ggml_vk_submit(compute_ctx, ctx->fence);
            VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "synchronize waitForFences", ctx->device);
            ctx->device->device.resetFences({ ctx->fence });
        } else {
            ggml_vk_submit(compute_ctx, {});
        }
        ctx->submit_pending = true;
    }

    if (ctx->submit_pending) {
        if (ctx->device->serialize_submissions) {
            ctx->submit_pending = false;
        } else if (ctx->device->async_use_transfer_queue && ctx->transfer_semaphore_last_submitted < ctx->transfer_semaphore.value) {
            vk::TimelineSemaphoreSubmitInfo tl_info{
                1, &ctx->transfer_semaphore.value,
                0, nullptr,
            };
            vk::PipelineStageFlags stage = ctx->device->transfer_queue->stage_flags;
            vk::SubmitInfo si{
                1, &ctx->transfer_semaphore.s, &stage,
                0, nullptr,
                0, nullptr,
            };
            si.setPNext(&tl_info);
            ctx->device->compute_queue->handle->submit({ si }, ctx->fence);
            ctx->transfer_semaphore_last_submitted = ctx->transfer_semaphore.value;
        } else {
            ctx->device->compute_queue->handle->submit({}, ctx->fence);
        }
        if (!ctx->device->serialize_submissions) {
            ggml_vk_wait_for_fence(ctx);
        }
        ctx->submit_pending = false;
        if (cmd_buf) {
            cmd_buf->in_use = false;
            cmd_buf->buf.reset();
        }
    }

    if (do_transfer) {
        for (auto& cpy : compute_ctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        ctx->compute_ctx.reset();
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

static bool ggml_vk_can_fuse_unary_mul(const struct ggml_cgraph * cgraph, int unary_idx, int mul_idx) {
    const ggml_tensor * unary = cgraph->nodes[unary_idx];
    const ggml_tensor * mul = cgraph->nodes[mul_idx];

    if (ggml_vk_unary_mul_op_index(ggml_get_unary_op(unary)) < 0) {
        return false;
    }
    if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
        return false;
    }
    if (unary->type != mul->type) {
        return false;
    }
    if (mul->src[0] != unary && mul->src[1] != unary) {
        return false;
    }
    const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
    if (other == nullptr || other->type != unary->type) {
        return false;
    }
    if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0])) {
        return false;
    }
    // fastmod needs src to tile into dst
    if (mul->src[0] == unary) {
        return ggml_can_repeat(other, unary);
    }
    return ggml_can_repeat(unary, mul->src[0]);
}

static bool ggml_vk_can_fuse_unary_mul_pair(const struct ggml_cgraph * cgraph, int node_idx) {
    const enum ggml_op ops[]    = { GGML_OP_UNARY, GGML_OP_MUL };
    const int           outputs[] = { node_idx + 1 };
    return ggml_can_fuse_subgraph(cgraph, node_idx, 2, ops, outputs, 1) &&
           ggml_vk_can_fuse_unary_mul(cgraph, node_idx, node_idx + 1);
}

static bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL) {
        return ggml_vk_can_fuse_unary_mul_pair(cgraph, node_idx);
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3 || ops.size() == 4) &&
        ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
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

        if (ops.size() >= 3 && ops.begin()[2] == GGML_OP_ADD) {
            const ggml_tensor *add = cgraph->nodes[node_idx + 2];
            const ggml_tensor *residual = add->src[0] == mul ? add->src[1] : add->src[0];
            if (add->src[0] != mul && add->src[1] != mul) {
                return false;
            }
            if (residual->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
                !ggml_are_same_shape(add, residual) || !ggml_is_contiguous(residual) ||
                !ggml_is_contiguous(add) || get_misalign_bytes(ctx, residual) != 0) {
                return false;
            }

            const ggml_tensor *dst = add;
            if (ops.size() == 4) {
                if (ops.begin()[3] != GGML_OP_MUL) {
                    return false;
                }

                const ggml_tensor *post_mul = cgraph->nodes[node_idx + 3];
                const ggml_tensor *scale = post_mul->src[0] == add ? post_mul->src[1] : post_mul->src[0];
                if (post_mul->src[0] != add && post_mul->src[1] != add) {
                    return false;
                }
                // The shader reads data_e[0], so the final multiply must use a scalar.
                if (scale->type != GGML_TYPE_F32 || post_mul->type != GGML_TYPE_F32 ||
                    ggml_nelements(scale) != 1 || !ggml_is_contiguous(post_mul) ||
                    get_misalign_bytes(ctx, scale) != 0) {
                    return false;
                }
                dst = post_mul;
            }

            if (get_misalign_bytes(ctx, dst) != 0) {
                return false;
            }
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
    case TOPK_MOE_SIGMOID_NORM_BIAS:
        softmax = cgraph->nodes[node_idx + 0]; // really sigmoid
        weights = cgraph->nodes[node_idx + 10];
        get_rows = cgraph->nodes[node_idx + 5];
        argsort = cgraph->nodes[node_idx + 3];
        if (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SIGMOID) {
            return false;
        }
        // bias is expected to be 1D
        if (ggml_nrows(cgraph->nodes[node_idx + 2]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[node_idx + 2]->src[1])) {
            return false;
        }
        // sigmoid fusion seems to generate infinities on moltenvk
        if (ctx->device->driver_id == vk::DriverId::eMoltenvk) {
            return false;
        }
        break;
    case TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS:
        softmax = cgraph->nodes[node_idx + 0]; // really softplus
        weights = cgraph->nodes[node_idx + 11];
        get_rows = cgraph->nodes[node_idx + 6];
        argsort = cgraph->nodes[node_idx + 4];
        if (ggml_get_unary_op(softmax) != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
        // bias is expected to be 1D
        if (ggml_nrows(cgraph->nodes[node_idx + 3]->src[1]) != 1 ||
            !ggml_is_contiguous(cgraph->nodes[node_idx + 3]->src[1])) {
            return false;
        }
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

    if (probs != selection_probs &&
        mode != TOPK_MOE_SIGMOID_NORM_BIAS &&
        mode != TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS) {
        return false;
    }

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        return false;
    }

    if (softmax->op == GGML_OP_SOFT_MAX) {
        const float * op_params = (const float *)softmax->op_params;

        float scale = op_params[0];
        float max_bias = op_params[1];

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
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

static bool ggml_vk_match_ops(const struct ggml_cgraph * cgraph, int node_idx,
                              const std::initializer_list<ggml_op> & ops) {
    if (node_idx + (int) ops.size() > cgraph->n_nodes) {
        return false;
    }
    for (size_t j = 0; j < ops.size(); ++j) {
        const ggml_tensor * node = cgraph->nodes[node_idx + j];
        if (node->op != ops.begin()[j] ||
            (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
            (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            return false;
        }
    }
    return true;
}

static bool ggml_vk_can_fuse_topk_qsa(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    if (ctx->device->disable_fusion || !ctx->device->pipeline_topk_radix_qsa) {
        return false;
    }

    const int n_ops = topk_qsa_pattern.size();
    if (!ggml_vk_match_ops(cgraph, node_idx, topk_qsa_pattern) ||
        !ggml_check_edges(cgraph, node_idx, topk_qsa_edges)) {
        return false;
    }

    // elided nodes must be single-use (cpy counts its own src[1] self-reference)
    for (int j = 0; j < n_ops - 1; ++j) {
        const ggml_tensor * node = cgraph->nodes[node_idx + j];
        const int32_t want = node->op == GGML_OP_CPY ? 2 : 1;
        if (ggml_node_get_use_count(cgraph, node_idx + j) != want) {
            return false;
        }
    }

    const ggml_tensor * get_rows = cgraph->nodes[node_idx + 0];
    const ggml_tensor * add      = cgraph->nodes[node_idx + n_ops - 2];
    const ggml_tensor * top_k    = cgraph->nodes[node_idx + n_ops - 1];

    const ggml_tensor * scores   = get_rows->src[0]; // [n_tps, n_blocks, n_stream]
    const ggml_tensor * cell_blk = get_rows->src[1]; // [n_kv, n_stream]
    const ggml_tensor * expanded = add->src[0];      // [n_kv, n_tps, n_stream]

    // raw mask: follow the reshape/cpy chain back to the materialized f16 input
    const ggml_tensor * mask = add->src[1];
    while (mask && (mask->op == GGML_OP_RESHAPE || mask->op == GGML_OP_CPY)) {
        mask = mask->src[0];
    }
    if (!mask || mask->type != GGML_TYPE_F16) {
        return false;
    }

    if (scores->type != GGML_TYPE_F32 || cell_blk->type != GGML_TYPE_I32 || top_k->type != GGML_TYPE_I32) {
        return false;
    }
    if (!ggml_is_contiguous(scores) || !ggml_is_contiguous(cell_blk) || !ggml_is_contiguous(mask) ||
        !ggml_is_contiguous(expanded) || !ggml_is_contiguous(top_k)) {
        return false;
    }

    const int64_t n_tps    = scores->ne[0];
    const int64_t n_blocks = scores->ne[1];
    const int64_t n_stream = scores->ne[2];
    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t width    = top_k->ne[0];

    // ggmlR DIVERGENCE (5D): ne4 is not encoded by the shader either.
    if (scores->ne[4] != 1 || top_k->ne[4] != 1) {
        return false;
    }

    // pin the indexer layout the shader's addressing assumes
    if (scores->ne[3] != 1 || cell_blk->ne[1] != n_stream || ggml_nrows(cell_blk) != n_stream ||
        ggml_nelements(mask) != n_kv * n_tps * n_stream ||
        expanded->ne[0] != n_kv || expanded->ne[1] != n_tps || expanded->ne[2] != n_stream ||
        top_k->ne[1] != n_tps || top_k->ne[2] != n_stream || top_k->ne[3] != 1 ||
        n_blocks <= 0 || n_kv <= 0 || width <= 0 || width > n_kv) {
        return false;
    }

    // only worth it in the radix regime; small k uses the faster tournament unfused
    const uint32_t k_min_pipeline = std::max((uint32_t) log2f(float(width)) + 1, ctx->device->subgroup_size_log2);
    if (k_min_pipeline < num_topk_pipelines && ctx->device->pipeline_topk_f32[k_min_pipeline]) {
        return false;
    }
    return true;
}

static bool ggml_vk_can_fuse_rope_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                           int node_idx) {
    const ggml_tensor *rope = cgraph->nodes[node_idx + 0];
    const ggml_tensor *view = cgraph->nodes[node_idx + 1];
    const ggml_tensor *set_rows = cgraph->nodes[node_idx + 2];

    // The set_rows epilogue uses one index per ne2 slice and does not encode ne3.
    // ggmlR DIVERGENCE (5D): ne4 is not encoded either; rope itself is 4D.
    if (rope->src[0]->ne[3] != 1 || rope->src[0]->ne[4] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    // The shader reads each aligned I64 index as a uvec2 and uses its low 32 bits.
    if (set_rows->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(set_rows->src[1]) ||
        set_rows->nb[0] != ggml_type_size(set_rows->type) || get_misalign_bytes(ctx, set_rows->src[1]) != 0) {
        return false;
    }

    // SET_ROWS consumes one flattened [ne0*ne1] row for each ne2 slice.
    if (!ggml_is_contiguous(view) ||
        view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(set_rows->src[1]) != rope->ne[2]) {
        return false;
    }

    // Only norm/neox/mrope/imrope shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_vk_can_fuse_rms_norm_set_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                               int node_idx) {
    const ggml_tensor * rms = cgraph->nodes[node_idx];
    const ggml_tensor * view = cgraph->nodes[node_idx + 1];
    const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

    // The RMS kernel reads F32 and writes directly to the F32 or F16 SET_ROWS destination.
    if (rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) ||
        set_rows->src[1]->type != GGML_TYPE_I64 || !ggml_is_contiguous(set_rows->src[1]) ||
        set_rows->nb[0] != ggml_type_size(set_rows->type) || get_misalign_bytes(ctx, set_rows->src[1]) != 0) {
        return false;
    }
    // As with the ROPE epilogue, each ne2 slice supplies one flattened row and ne3 is not encoded.
    // ggmlR DIVERGENCE (5D): rms_norm.comp loops z over ne3*ne4 but the set_rows d_offset
    // ignores samp/samp4, so ne4 > 1 would overwrite the same rows.
    if (rms->ne[3] != 1 || rms->ne[4] != 1 || !ggml_is_contiguous(rms->src[0]) || !ggml_is_contiguous(view)) {
        return false;
    }
    if (view->ne[0] != rms->ne[0] * rms->ne[1] || view->ne[1] != rms->ne[2] ||
        view->ne[2] != 1 || view->ne[3] != 1 ||
        ggml_nelements(set_rows->src[1]) != rms->ne[2]) {
        return false;
    }

    return true;
}

static bool ggml_vk_can_fuse_hc_post_gate(const struct ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * scale_in  = cgraph->nodes[node_idx];
    const ggml_tensor * sigmoid   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * scale_out = cgraph->nodes[node_idx + 2];

    // the shader folds scale -> sigmoid -> scale; a bias on either scale is not handled
    return ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
           ggml_get_op_params_f32(scale_in, 1) == 0.0f &&
           ggml_get_op_params_f32(scale_out, 1) == 0.0f &&
           scale_in->src[0]->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(scale_in->src[0], scale_out);
}

// ggmlR DIVERGENCE: a fused shader writes the last node's output while it still
// reads the pattern's external inputs. Upstream relies on the allocator deps that
// graph_optimize registers (add_pattern_alloc_deps), which only the scheduler
// honors; a graph allocated by gallocr directly gets none. So the fusion is legal
// only if, in the memory actually allocated, the output overlaps no external input.
// Call after allocation (build_graph): an unallocated tensor counts as overlapping.
static bool ggml_vk_fused_output_overlaps_inputs(const struct ggml_cgraph * cgraph, int node_idx, int n_nodes) {
    const ggml_tensor * out = cgraph->nodes[node_idx + n_nodes - 1];
    if (out->data == nullptr) {
        return true;
    }
    const char * out_lo = (const char *) out->data;
    const char * out_hi = out_lo + ggml_nbytes(out);

    for (int j = 0; j < n_nodes; ++j) {
        const ggml_tensor * node = cgraph->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src == nullptr) {
                continue;
            }
            bool internal = false;
            for (int k = 0; k < j; ++k) {
                internal = internal || src == cgraph->nodes[node_idx + k];
            }
            if (internal) {
                continue;
            }
            if (src->data == nullptr) {
                return true;
            }
            if (src->buffer != out->buffer) {
                continue;
            }
            const char * src_lo = (const char *) src->data;
            const char * src_hi = src_lo + ggml_nbytes(src);
            if (src_lo < out_hi && out_lo < src_hi) {
                return true;
            }
        }
    }
    return false;
}

static bool ggml_vk_can_fuse_ssm_conv(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph,
                                      int node_idx, int num_extra) {
    const ggml_tensor * conv = cgraph->nodes[node_idx];
    if (conv->op != GGML_OP_SSM_CONV) {
        return false;
    }

    const ggml_tensor * silu = nullptr;
    const ggml_tensor * bias = nullptr;

    if (num_extra == 1) {
        if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
            return false;
        }
        silu = cgraph->nodes[node_idx + 1];
    } else if (num_extra == 2) {
        if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY })) {
            return false;
        }
        const ggml_tensor * add = cgraph->nodes[node_idx + 1];
        silu = cgraph->nodes[node_idx + 2];
        bias = (add->src[0] == conv) ? add->src[1] : add->src[0];

        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        // bias must be channel-wise (one element per channel of the conv output)
        if (ggml_nelements(bias) != conv->ne[0] || bias->ne[0] != conv->ne[0]) {
            return false;
        }
        if (add->type != GGML_TYPE_F32) {
            return false;
        }
        // The shader doesn't apply per-tensor offsets, so reject misaligned bias.
        if (get_misalign_bytes(ctx, bias) != 0) {
            return false;
        }
    } else {
        return false;
    }

    if (ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU) {
        return false;
    }
    if (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
        return false;
    }
    // The shader writes to the fused dst using its own strides, but the push constants don't
    // carry a per-tensor offset, so the binding must be naturally aligned.
    if (get_misalign_bytes(ctx, silu) != 0) {
        return false;
    }
    return true;
}

// Number of HC_POST_GATE fusions dispatched. Read from R (R_ggml_vk_hc_post_gate_count)
// so a test can tell the fusion happened, not only that the result is right.
static uint64_t ggml_vk_hc_post_gate_fusions = 0;
extern "C" uint64_t ggml_vk_hc_post_gate_fusion_count(void) { return ggml_vk_hc_post_gate_fusions; }

// Likewise for SSM_CONV(+ADD)+SILU (R_ggml_vk_ssm_conv_fusion_count).
static uint64_t ggml_vk_ssm_conv_fusions = 0;
extern "C" uint64_t ggml_vk_ssm_conv_fusion_count(void) { return ggml_vk_ssm_conv_fusions; }

// Likewise for UNARY+MUL (gelu/sigmoid/silu/softplus_mul, R_ggml_vk_unary_mul_fusion_count).
static uint64_t ggml_vk_unary_mul_fusions = 0;
extern "C" uint64_t ggml_vk_unary_mul_fusion_count(void) { return ggml_vk_unary_mul_fusions; }

// Test hook for ggml_vk_fused_output_overlaps_inputs (R_ggml_vk_test_fused_overlap):
// moves the pattern's output onto an input, calls the predicate, restores the output.
// The tensors only need data pointers, so a plain allocated ggml context will do.
//   mode 0: as allocated   1: output = first scale's input   2: output starts halfway
//   into that input   3: output unallocated   4: as 1, but in another buffer
//   5: output = hc_post's residual (src[1])
extern "C" int ggml_vk_test_fused_output_overlaps(struct ggml_cgraph * cgraph, int node_idx, int n_nodes, int mode) {
    ggml_tensor * out = cgraph->nodes[node_idx + n_nodes - 1];
    const ggml_tensor * first = cgraph->nodes[node_idx]->src[0];
    void * data = out->data;
    ggml_backend_buffer_t buffer = out->buffer;
    switch (mode) {
        case 1: out->data = first->data; break;
        case 2: out->data = (char *) first->data + ggml_nbytes(first) / 2; break;
        case 3: out->data = nullptr; break;
        case 4: out->data = first->data; out->buffer = (ggml_backend_buffer_t) &buffer; break;
        case 5: out->data = out->src[1]->data; break;
        default: break;
    }
    const bool overlaps = ggml_vk_fused_output_overlaps_inputs(cgraph, node_idx, n_nodes);
    out->data = data;
    out->buffer = buffer;
    return overlaps;
}

static bool ggml_vk_can_fuse_snake(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_can_fuse(cgraph, node_idx, snake_pattern)) {
        return false;
    }

    const ggml_tensor * mul0     = cgraph->nodes[node_idx + 0];
    const ggml_tensor * sin_node = cgraph->nodes[node_idx + 1];
    const ggml_tensor * sqr      = cgraph->nodes[node_idx + 2];
    const ggml_tensor * mul1     = cgraph->nodes[node_idx + 3];
    const ggml_tensor * add      = cgraph->nodes[node_idx + 4];

    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    const ggml_tensor * inv_b    = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    if (x_in_add != x) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_BF16) {
        return false;
    }
    // Shader bindings: data_a is A_TYPE so it follows x's precision, while
    // data_b and data_c are hardcoded float, so the broadcast operands must
    // be F32 regardless of x's type.
    if (a->type     != GGML_TYPE_F32) return false;
    if (inv_b->type != GGML_TYPE_F32) return false;
    // Chain intermediates and output share x's precision (single A_TYPE / D_TYPE pipeline).
    if (mul0->type     != x->type) return false;
    if (sin_node->type != x->type) return false;
    if (sqr->type      != x->type) return false;
    if (mul1->type     != x->type) return false;
    if (add->type      != x->type) return false;
    if (!ggml_are_same_shape(a, inv_b)) {
        return false;
    }
    if (a->ne[0] != 1 || a->ne[1] != x->ne[1]) {
        return false;
    }
    // Dispatch is 2D over (ne0, ne1), so x and add must be 2D and a / inv_b
    // must collapse to [1, C, 1, 1]. Higher dims are not handled by the shader.
    if (x->ne[2]     != 1 || x->ne[3]     != 1) return false;
    if (add->ne[2]   != 1 || add->ne[3]   != 1) return false;
    if (a->ne[2]     != 1 || a->ne[3]     != 1) return false;
    if (inv_b->ne[2] != 1 || inv_b->ne[3] != 1) return false;
    // ggmlR DIVERGENCE (5D): ne4 is not encoded by the shader either.
    if (x->ne[4] != 1 || add->ne[4] != 1 || a->ne[4] != 1 || inv_b->ne[4] != 1) return false;
    // ggmlR DIVERGENCE: the dispatch puts ne1 on grid y unclamped (NVIDIA limit 65535).
    if ((uint64_t) x->ne[1] > ctx->device->properties.limits.maxComputeWorkGroupCount[1]) return false;
    // Shader uses idx = i0 + i1 * ne0 and reads data_b[i1] / data_c[i1],
    // so every operand must be contiguous.
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(add) ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(inv_b)) {
        return false;
    }
    return true;
}

// Check whether the tensors overlap in memory but are not equal.
static int32_t find_first_set(uint32_t x) {
    int32_t ret = 0;
    if (!x) {
        return -1;
    }
    while (!(x & 1)) {
        x >>= 1;
        ret++;
    }
    return ret;
}

// Fusions can potenitally overwrite src tensors in ways that are not prevented
// by ggml-alloc. If the fusion is entirely elementwise, then it's OK for them
// to overlap if they are exactly equal.
static bool ggml_vk_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b, bool elementwise) {
    ggml_backend_vk_buffer_context * a_buf_ctx = (ggml_backend_vk_buffer_context *)a->buffer->context;
    vk_buffer a_buf = a_buf_ctx->dev_buffer;
    ggml_backend_vk_buffer_context * b_buf_ctx = (ggml_backend_vk_buffer_context *)b->buffer->context;
    vk_buffer b_buf = b_buf_ctx->dev_buffer;
    if (a_buf == b_buf) {
        auto a_base = vk_tensor_offset(a) + a->view_offs;
        auto a_size = ggml_nbytes(a);
        auto b_base = vk_tensor_offset(b) + b->view_offs;
        auto b_size = ggml_nbytes(b);

        if (elementwise && a_base == b_base && a_size == b_size) {
            return false;
        }

        if ((b_base <= a_base && a_base < b_base + b_size) ||
            (a_base <= b_base && b_base < a_base + a_size)) {
            return true;
        }
    }
    return false;
}

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

    // ggmlR DIVERGENCE (5D): the rope epilogue is called without samp4,
    // so ne4 > 1 would overwrite the same rows; rope itself is 4D.
    if (rms->ne[4] != 1) {
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
    if (sizeof(vk_op_rms_norm_mul_rope_push_constants) > ctx->device->properties.limits.maxPushConstantsSize) {
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
    r_dbg_logf("vk_graph_compute: ENTER n_nodes=%d (set_tensor so far=%llu, %.1f MB)",
               cgraph->n_nodes, g_vk_set_tensor_calls, g_vk_set_tensor_bytes / (1024.0 * 1024.0));
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    ctx->device->diag_cgraph = nullptr;
    ctx->device->diag_prev_start = -1;
    ctx->device->diag_prev_end = -1;

    if (vk_instance.debug_utils_support) {
        ctx->device->debug_cmdbuf_idx = 0;
    }

    // queue scope, so it encloses every submit this evaluation makes.
    // closed when the function returns
    ggml_vk_debug_label queue_dbg(ctx->device->compute_queue->handle.get(), "ggml_backend_vk_graph_compute");

    ctx->prealloc_size_add_rms_partials_offset = 0;
    ctx->do_add_rms_partials = false;
    ctx->do_add_rms_partials_offset_calculation = false;

    int last_node = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    while (last_node > 0 && (ggml_vk_is_empty(cgraph->nodes[last_node]) || ((cgraph->nodes[last_node]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0))) {
        last_node -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    bool first_node_in_batch = true; // true if next node will be first node in a batch
    int submit_node_idx = 0; // index to first node in a batch

    ggml_vk_submit_transfer_ctx(ctx);

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
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        ctx->query_idx = 0;
        compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;
    ctx->prealloc_y_last_k_padded = false;

    if (ctx->prealloc_size_add_rms_partials) {
        ggml_vk_preallocate_buffers(ctx, nullptr);
        compute_ctx = ggml_vk_get_compute_ctx(ctx);
        // initialize partial sums to zero.
        ggml_vk_buffer_memset_async(compute_ctx, ctx->prealloc_add_rms_partials, 0, 0, ctx->prealloc_size_add_rms_partials);
        ggml_vk_sync_buffers(ctx, compute_ctx);
    }

    // Submit after enough work has accumulated, to overlap CPU cmdbuffer generation with GPU execution.
    // Estimate the amount of compute work using flops, and submit every 200 GFLOP
    // (and scaled down based on total graph flops, so smaller models submit earlier).
    // Also submit at least every 100 nodes, in case there are workloads without heavy compute.
    uint32_t submitted_nodes = 0;
    uint32_t submit_count = 0;
    uint64_t batch_flops = 0;
    uint64_t total_flops = 0;
    uint64_t flops_cap = 200'000'000'000ULL;

    // On weaker AMD GPUs larger submissions can hit a driver timeout, submit more often to avoid this
    if (ctx->device->vendor_id == VK_VENDOR_ID_AMD && ctx->device->shader_core_count > 0) {
        if (ctx->device->architecture == AMD_GCN && ctx->device->shader_core_count < 32) {
            flops_cap = 500'000'000ULL * ctx->device->shader_core_count;
        } else if (ctx->device->architecture != AMD_GCN && ctx->device->shader_core_count < 24) {
            flops_cap = 2'000'000'000ULL * ctx->device->shader_core_count;
        }
    }
    uint64_t flops_per_submit = std::min(flops_cap, ctx->last_total_flops / 40u);

    auto const submit_after = [&](int start, int end) {
        if (ctx->device->serialize_submissions) {
            try {
                auto res = ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX);
                if (res != vk::Result::eSuccess) {
                    GGML_LOG_ERROR("ggml_vulkan: waitForFences error during serialized submission\n");
                    throw vk::SystemError(vk::make_error_code(res), "ggml_vulkan: waitForFences during serialized submission");
                }
            } catch (vk::DeviceLostError &) {
                ggml_vk_print_device_fault_info(ctx->device);
                GGML_LOG_ERROR("ggml_vulkan: device lost on %s waiting for submission (nodes %d to %d):\n",
                        ctx->device->name.c_str(), start, end);
                ggml_vk_print_node_list(cgraph, start, end);
                throw;
            }
            ctx->device->device.resetFences({ ctx->fence });
            ctx->submit_pending = false;
            ctx->device->diag_cgraph = cgraph;
            ctx->device->diag_prev_start = start;
            ctx->device->diag_prev_end = end;
        }
        first_node_in_batch = true;
        submitted_nodes = 0;
        batch_flops = 0;
        if (submit_count < 3) {
            flops_per_submit *= 2;
        }
        submit_count++;
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (first_node_in_batch) {
            submit_node_idx = i;
        }

        {
            auto node_flops = ggml_vk_get_node_flops(cgraph->nodes[i]);
            total_flops += node_flops;

            // Flush the current batch before recording a node that would push it over the flop threshold
            if (flops_per_submit != 0 && submitted_nodes > 0 && batch_flops + node_flops >= flops_per_submit) {
                vk_context flush_ctx = ggml_vk_get_compute_ctx(ctx);
                ggml_vk_ctx_end(flush_ctx);
                flush_ctx->exit_tensor_idx = -1;
                ctx->compute_ctx.reset();
                ggml_vk_compute_forward(ctx, cgraph, cgraph->nodes[submit_node_idx], submit_node_idx, false);
                submit_after(submit_node_idx, i - 1);
                submit_node_idx = i;
            }

            batch_flops += node_flops;
        }

        // op_srcs_fused_elementwise indicates whether an op's srcs all contribute to
        // the fused result in an elementwise-way. This affects whether the memory for
        // the src is allowed to overlap the memory for the destination.
        // The array is sized to handle the largest fusion (asserted later).
        bool op_srcs_fused_elementwise[13];

        ctx->fused_topk_moe_mode = TOPK_MOE_COUNT;
        ctx->fused_topk_moe_scale = false;
        ctx->fused_topk_qsa = false;
        ctx->fused_hc_post_gate = false;
        ctx->fused_rms_norm_mode = RMS_NORM_COUNT;
        const char *fusion_string {};
        if (!ctx->device->disable_fusion) {
            uint32_t num_adds = ggml_vk_fuse_multi_add(ctx, cgraph, i);
            if (num_adds) {
                ctx->num_additional_fused_ops = num_adds - 1;
                fusion_string = "MULTI_ADD";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, true);
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_ADD })) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ADD_ADD";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ADD";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "MUL_MAT_ID_ADD_ID_MUL";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_ADD_ID";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "MUL_MAT_ID_MUL";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_rope_view_set_rows_pattern, { i + 4 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_mul_rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i + 2)) {
                ctx->num_additional_fused_ops = 4;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ROPE_VIEW_SET_ROWS;
                fusion_string = "RMS_NORM_MUL_ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = false;
                op_srcs_fused_elementwise[2] = false;
                op_srcs_fused_elementwise[3] = false;
                op_srcs_fused_elementwise[4] = false;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) &&
                       ggml_vk_can_fuse_rms_norm_mul_rope(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ROPE;
                fusion_string = "RMS_NORM_MUL_ROPE";
                // rope is approximately elementwise - whole rows are done by a single workgroup and it's row-wise
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_add_mul_pattern)) {
                ctx->num_additional_fused_ops = 3;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ADD_MUL;
                fusion_string = "RMS_NORM_MUL_ADD_MUL";
                std::fill_n(op_srcs_fused_elementwise, 4, true);
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, rms_norm_mul_add_pattern)) {
                ctx->num_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL_ADD;
                fusion_string = "RMS_NORM_MUL_ADD";
                std::fill_n(op_srcs_fused_elementwise, 3, true);
            } else if (ggml_can_fuse_subgraph(cgraph, i, rms_norm_view_set_rows_pattern, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rms_norm_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rms_norm_set_rows(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 2;
                ctx->fused_rms_norm_mode = RMS_NORM_VIEW_SET_ROWS;
                fusion_string = "RMS_NORM_VIEW_SET_ROWS";
                std::fill_n(op_srcs_fused_elementwise, 3, false);
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 1;
                ctx->fused_rms_norm_mode = RMS_NORM_MUL;
                fusion_string = "RMS_NORM_MUL";
                // rms_norm is not elementwise, but whole rows must be consumed and the scale factor computed before
                // they are overwritten, and one workgroup per row. So close enough.
                op_srcs_fused_elementwise[0] = true;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_vk_can_fuse(ctx, cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL })) {
                ctx->num_additional_fused_ops = 1;
                switch (ggml_get_unary_op(cgraph->nodes[i])) {
                    case GGML_UNARY_OP_GELU:     fusion_string = "GELU_MUL";     break;
                    case GGML_UNARY_OP_SIGMOID:  fusion_string = "SIGMOID_MUL";  break;
                    case GGML_UNARY_OP_SILU:     fusion_string = "SILU_MUL";     break;
                    default:                     fusion_string = "SOFTPLUS_MUL"; break;
                }
                op_srcs_fused_elementwise[0] = true;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_can_fuse_subgraph(cgraph, i, hc_post_gate_pattern, { i + 3 }) &&
                       ggml_check_edges(cgraph, i, hc_post_gate_edges) &&
                       ggml_vk_can_fuse_hc_post_gate(cgraph, i) &&
                       !ggml_vk_fused_output_overlaps_inputs(cgraph, i, (int) hc_post_gate_pattern.size())) {
                ctx->num_additional_fused_ops = hc_post_gate_pattern.size() - 1;
                ctx->fused_hc_post_gate = true;
                fusion_string = "HC_POST_GATE";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 2) &&
                       !ggml_vk_fused_output_overlaps_inputs(cgraph, i, 3)) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "SSM_CONV_BIAS_SILU";
                // ssm_conv reads multiple input tokens per output, so it's not elementwise w.r.t. its srcs.
                // The downstream add and silu are elementwise on the conv output.
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
                op_srcs_fused_elementwise[2] = true;
            } else if (ggml_vk_can_fuse_ssm_conv(ctx, cgraph, i, 1) &&
                       !ggml_vk_fused_output_overlaps_inputs(cgraph, i, 2)) {
                ctx->num_additional_fused_ops = 1;
                fusion_string = "SSM_CONV_SILU";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = true;
            } else if (ggml_can_fuse_subgraph(cgraph, i, rope_view_set_rows_pattern, { i + 2 }) &&
                       ggml_check_edges(cgraph, i, rope_view_set_rows_edges) &&
                       ggml_vk_can_fuse_rope_set_rows(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 2;
                fusion_string = "ROPE_VIEW_SET_ROWS";
                op_srcs_fused_elementwise[0] = false;
                op_srcs_fused_elementwise[1] = false;
                op_srcs_fused_elementwise[2] = false;
            } else if (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = 4;
                fusion_string = "SNAKE";
                // elementwise=true: snake.comp is safe under exact aliasing because each
                // thread reads data_x[idx] into a register before writing data_d[idx]
                // with a data dependency on that register. The overlap check still
                // rejects partial overlaps (different base or size).
                std::fill_n(op_srcs_fused_elementwise, 5, true);
            } else if (ggml_vk_can_fuse_topk_qsa(ctx, cgraph, i)) {
                ctx->num_additional_fused_ops = topk_qsa_pattern.size() - 1;
                ctx->fused_topk_qsa = true;
                fusion_string = "TOPK_QSA";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax_norm, { i + 3, i + 9 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_norm_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX_NORM)) {
                ctx->num_additional_fused_ops = topk_moe_early_softmax_norm.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX_NORM;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX_NORM";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sigmoid_norm_bias, { i + 4, i + 10 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_sigmoid_norm_bias_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SIGMOID_NORM_BIAS)) {
                ctx->num_additional_fused_ops = topk_moe_sigmoid_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 4;
                ctx->fused_topk_moe_mode = TOPK_MOE_SIGMOID_NORM_BIAS;
                fusion_string = "TOPK_MOE_SIGMOID_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_sqrt_softplus_norm_bias, { i + 5, i + 11 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_sqrt_softplus_norm_bias_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS)) {
                ctx->num_additional_fused_ops = topk_moe_sqrt_softplus_norm_bias.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 5;
                ctx->fused_topk_moe_mode = TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS;
                fusion_string = "TOPK_MOE_SQRT_SOFTPLUS_NORM_BIAS";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_early_softmax, { i + 3, i + 4 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_early_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_EARLY_SOFTMAX)) {
                ctx->num_additional_fused_ops = topk_moe_early_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 3;
                ctx->fused_topk_moe_mode = TOPK_MOE_EARLY_SOFTMAX;
                fusion_string = "TOPK_MOE_EARLY_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            } else if (ggml_can_fuse_subgraph(cgraph, i, topk_moe_late_softmax, { i + 1, i + 5 }) &&
                       ggml_check_edges(cgraph, i, topk_moe_late_softmax_edges) &&
                       ggml_vk_can_fuse_topk_moe(ctx, cgraph, i, TOPK_MOE_LATE_SOFTMAX)) {
                ctx->num_additional_fused_ops = topk_moe_late_softmax.size() - 1;
                // view of argsort writes to memory
                ctx->fused_ops_write_mask |= 1 << 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_LATE_SOFTMAX;
                fusion_string = "TOPK_MOE_LATE_SOFTMAX";
                std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            }
            if (ctx->fused_topk_moe_mode != TOPK_MOE_COUNT) {
                // Look for an additional scale op to fuse - occurs in deepseek2 and nemotron3 nano.
                if (ggml_can_fuse_subgraph(cgraph, i + ctx->num_additional_fused_ops - 1, { GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE }, { i + ctx->num_additional_fused_ops + 1 }) ||
                    ggml_can_fuse_subgraph(cgraph, i + ctx->num_additional_fused_ops, { GGML_OP_GET_ROWS, GGML_OP_SCALE }, { i + ctx->num_additional_fused_ops + 1 })) {
                    ctx->fused_topk_moe_scale = true;
                    ctx->num_additional_fused_ops++;
                    op_srcs_fused_elementwise[ctx->num_additional_fused_ops] = false;
                }
            }
        }
        GGML_ASSERT(ctx->num_additional_fused_ops < (int)(sizeof(op_srcs_fused_elementwise) / sizeof(op_srcs_fused_elementwise[0])));
        ctx->fused_ops_write_mask |= 1 << ctx->num_additional_fused_ops;

        // Check whether fusion would overwrite src operands while they're still in use.
        // If so, disable fusion.
        if (ctx->num_additional_fused_ops) {
            // There are up to two output nodes - topk_moe has two.
            uint32_t bits = ctx->fused_ops_write_mask & ~(1 << ctx->num_additional_fused_ops);
            ggml_tensor *output_nodes[2] {};
            output_nodes[0] = cgraph->nodes[i + ctx->num_additional_fused_ops];
            if (bits) {
                int output_idx = find_first_set(bits);
                GGML_ASSERT(bits == (1u << output_idx));
                output_nodes[1] = cgraph->nodes[i + output_idx];
            }

            bool need_disable = false;

            for (int j = 0; j < 2; ++j) {
                ggml_tensor *dst = output_nodes[j];
                if (!dst) {
                    continue;
                }
                // Loop over all srcs of all nodes in the fusion. If the src overlaps
                // the destination and the src is not an intermediate node that's being
                // elided, then disable fusion.
                for (int k = 0; k <= ctx->num_additional_fused_ops; ++k) {
                    for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                        ggml_tensor *src = cgraph->nodes[i + k]->src[s];
                        if (!src || src->op == GGML_OP_NONE) {
                            continue;
                        }
                        if (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {
                            bool found = false;
                            for (int n = 0; n < k; ++n) {
                                if (cgraph->nodes[i + n] == src) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                need_disable = true;
                            }
                        }
                    }
                }
            }
            if (need_disable) {
                ctx->num_additional_fused_ops = 0;
                ctx->fused_ops_write_mask = 1;
                ctx->fused_topk_moe_mode = TOPK_MOE_COUNT;
                ctx->fused_topk_moe_scale = false;
                ctx->fused_topk_qsa = false;
                ctx->fused_hc_post_gate = false;
                ctx->fused_rms_norm_mode = RMS_NORM_COUNT;
                fusion_string = nullptr;
            }
        }

        // Signal the almost_ready fence when the graph is mostly complete (< 20% remaining)
        bool almost_ready = (cgraph->n_nodes - i) < cgraph->n_nodes / 5;
        bool submit = (submitted_nodes >= ctx->device->max_nodes_per_submit) ||
                      (flops_per_submit != 0 && batch_flops >= flops_per_submit) ||
                      (i + ctx->num_additional_fused_ops >= last_node) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        // counted here, after need_disable may have dropped the fusion
        if (ctx->fused_hc_post_gate) {
            ggml_vk_hc_post_gate_fusions++;
        }
        if (ctx->num_additional_fused_ops > 0 && cgraph->nodes[i]->op == GGML_OP_SSM_CONV) {
            ggml_vk_ssm_conv_fusions++;
        }
        if (ctx->num_additional_fused_ops == 1 && cgraph->nodes[i]->op == GGML_OP_UNARY &&
            ctx->fused_topk_moe_mode == TOPK_MOE_COUNT) {
            ggml_vk_unary_mul_fusions++;
        }
        bool enqueued = ggml_vk_build_graph(ctx, cgraph, i, cgraph->nodes[submit_node_idx], submit_node_idx, i + ctx->num_additional_fused_ops >= last_node, almost_ready, submit);

        if (vk_perf_logger_enabled && enqueued) {
            compute_ctx = ggml_vk_get_compute_ctx(ctx);
            if (!vk_perf_logger_concurrent) {
                // track a single node/fusion for the current query
                ctx->query_nodes[ctx->query_idx] = cgraph->nodes[i];
                ctx->query_fusion_names[ctx->query_idx] = fusion_string;
                compute_ctx->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, ctx->query_pool, ctx->query_idx++);
                ggml_vk_sync_buffers(ctx, compute_ctx);
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
            submit_after(submit_node_idx, i + (int)ctx->num_additional_fused_ops);
        }
        i += ctx->num_additional_fused_ops;
        ctx->num_additional_fused_ops = 0;
        ctx->fused_ops_write_mask = 0;
    }

    ctx->last_total_flops = total_flops;

    if (vk_perf_logger_enabled) {
        // End the command buffer and submit/wait
        GGML_ASSERT(!ctx->compute_ctx.expired());
        compute_ctx = ctx->compute_ctx.lock();
        ggml_vk_ctx_end(compute_ctx);

        ggml_vk_submit(compute_ctx, ctx->device->fence);
        VK_CHECK(ctx->device->device.waitForFences({ ctx->device->fence }, true, UINT64_MAX), "GGML_VULKAN_PERF waitForFences", ctx->device);
        ctx->device->device.resetFences({ ctx->device->fence });
        ctx->compute_ctx.reset();

        // Get the results and pass them to the logger
        std::vector<uint64_t> timestamps(cgraph->n_nodes + 1);
        VK_CHECK(ctx->device->device.getQueryPoolResults(ctx->query_pool, 0, ctx->query_idx, (cgraph->n_nodes + 1)*sizeof(uint64_t), timestamps.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait), "get timestamp results", ctx->device);
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

    // Industrial telemetry: report any ops that the scheduler ran on CPU because
    // the Vulkan backend rejected them (silent perf cost via GPU<->CPU copies).
    ggml_vk_log_unsupported_op_summary("graph compute");

    if (getenv("GGML_VKG_SUBMITS")) {
        r_ggml_fprintf(stderr, "[VKG_SUBMITS] n_nodes=%d submit_count=%d support_async=%d\n",
                       cgraph->n_nodes, submit_count, (int) ctx->device->support_async);
    }

    r_dbg_logf("vk_graph_compute: DONE n_nodes=%d", cgraph->n_nodes);
    return GGML_STATUS_SUCCESS;

    UNUSED(backend);
}

// Sort the graph for improved parallelism.
static void ggml_vk_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph, struct ggml_backend_graph_optimize_params * params)
{
    VK_LOG_DEBUG("ggml_vk_graph_optimize(" << graph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    // ggmlR DIVERGENCE: allocator deps for the SSM_CONV(+ADD)+SILU fusion, which 0.27
    // registers none for. The fused shader reads several input tokens per output
    // while it writes the SILU output, so the conv's inputs (and the bias) must stay
    // alive until that node. build_graph still checks the actual layout
    // (ggml_vk_fused_output_overlaps_inputs); the deps only make the fusion possible
    // under the scheduler.
    if (params != nullptr && params->add_alloc_dep != nullptr) {
        auto const &is_silu_of = [](const ggml_tensor * node, const ggml_tensor * src) -> bool {
            return node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SILU && node->src[0] == src;
        };
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * conv = graph->nodes[i];
            if (conv->op != GGML_OP_SSM_CONV) {
                continue;
            }
            ggml_tensor * last = nullptr;
            ggml_tensor * bias = nullptr;
            for (int j = i + 1; j < graph->n_nodes && last == nullptr; ++j) {
                ggml_tensor * node = graph->nodes[j];
                if (is_silu_of(node, conv)) {
                    last = node;
                } else if (node->op == GGML_OP_ADD && (node->src[0] == conv || node->src[1] == conv)) {
                    for (int k = j + 1; k < graph->n_nodes; ++k) {
                        if (is_silu_of(graph->nodes[k], node)) {
                            last = graph->nodes[k];
                            bias = node->src[0] == conv ? node->src[1] : node->src[0];
                            break;
                        }
                    }
                    break;
                }
            }
            if (last == nullptr) {
                continue;
            }
            params->add_alloc_dep(params->user_data, conv->src[0], last);
            params->add_alloc_dep(params->user_data, conv->src[1], last);
            if (bias != nullptr) {
                params->add_alloc_dep(params->user_data, bias, last);
            }
        }
    }

    if (ctx->device->disable_graph_optimize) {
        return;
    }

    auto const &is_empty =[](const ggml_tensor * node) -> bool {
        return node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    auto const &is_src_of = [&is_empty](const ggml_tensor *dst, const ggml_tensor *src) -> bool {
        auto const &base = [](const ggml_tensor * tensor) {
            return tensor->view_src ? tensor->view_src : tensor;
        };
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
            if (is_empty(dst) || is_empty(src)) {
                continue;
            }
            // A source view of dst may read storage written through a different view by src.
            if (dst->src[s] && base(dst->src[s]) == base(src)) {
                return true;
            }
            // Moving dst forward may overwrite storage still read through a view by src.
            if (src->src[s] && base(dst) == base(src->src[s])) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        if (base(dst) == base(src)) {
            return true;
        }
        return false;
    };

    std::vector<ggml_tensor *> new_order;
    std::vector<bool> used(graph->n_nodes, false);
    std::set<ggml_tensor *> used_node_set;

    int first_unused = 0;

    // scheduled or zero-compute nodes in [lo, hi)
    auto const &empty_or_scheduled_between = [&](int lo, int hi) -> bool {
        for (int v = lo; v < hi; ++v) {
            if (!used[v] && !is_empty(graph->nodes[v])) {
                return false;
            }
        }
        return true;
    };
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

        auto const &add_pattern_alloc_deps = [&](const std::initializer_list<ggml_op> &pattern, int last_node) {
            // Keep external inputs alive through the fused output.
            std::set<ggml_tensor *> seen;
            for (size_t j = 0; j < pattern.size(); ++j) {
                ggml_tensor * node = graph->nodes[first_unused + j];
                for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
                    ggml_tensor * src = node->src[s];
                    if (src && seen.insert(src).second) {
                        params->add_alloc_dep(params->user_data, src, graph->nodes[last_node]);
                    }
                }
                seen.insert(node);
            }
        };

        auto const &keep_topk_moe_pattern = [&](const std::initializer_list<ggml_op> &pattern) -> bool {
            if (!match_pattern(pattern, first_unused)) {
                return false;
            }

            int last_node = first_unused + (int) pattern.size() - 1;
            // Some TOPK_MOE variants fuse a trailing scale.
            if (last_node + 1 < graph->n_nodes && graph->nodes[last_node + 1]->op == GGML_OP_SCALE) {
                last_node++;
            }

            add_pattern_alloc_deps(pattern, last_node);

            return keep_pattern(pattern);
        };

        if (keep_topk_moe_pattern(topk_moe_early_softmax_norm)) {
            continue;
        }
        if (keep_topk_moe_pattern(topk_moe_sigmoid_norm_bias)) {
            continue;
        }
        if (keep_topk_moe_pattern(topk_moe_sqrt_softplus_norm_bias)) {
            continue;
        }
        if (keep_topk_moe_pattern(topk_moe_early_softmax)) {
            continue;
        }
        if (keep_topk_moe_pattern(topk_moe_late_softmax)) {
            continue;
        }
        if (keep_pattern(snake_pattern)) {
            continue;
        }
        if (keep_pattern(topk_qsa_pattern)) {
            continue;
        }

        if (keep_pattern(rms_norm_mul_add_mul_pattern)) {
            continue;
        }
        if (keep_pattern(rms_norm_mul_add_pattern)) {
            continue;
        }
        if (keep_pattern(rms_norm_mul_rope_view_set_rows_pattern)) {
            continue;
        }
        if (keep_pattern(rms_norm_view_set_rows_pattern)) {
            continue;
        }
        if (keep_pattern(rope_view_set_rows_pattern)) {
            continue;
        }
        if (match_pattern(hc_post_gate_pattern, first_unused)) {
            add_pattern_alloc_deps(hc_post_gate_pattern, first_unused + (int) hc_post_gate_pattern.size() - 1);
            keep_pattern(hc_post_gate_pattern);
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
            auto const &in_qsa_pattern = [&](int n) -> bool {
                for (int o = 0; o < (int) topk_qsa_pattern.size(); ++o) {
                    if (n - o >= 0 && match_pattern(topk_qsa_pattern, n - o)) {
                        return true;
                    }
                }
                return false;
            };
            if (match_pattern(topk_moe_early_softmax_norm, j) ||
                match_pattern(topk_moe_sigmoid_norm_bias, j) ||
                match_pattern(topk_moe_sqrt_softplus_norm_bias, j) ||
                match_pattern(topk_moe_early_softmax, j) ||
                match_pattern(topk_moe_late_softmax, j) ||
                match_pattern(snake_pattern, j) ||
                in_qsa_pattern(j) ||
                match_pattern(rms_norm_mul_add_mul_pattern, j) ||
                match_pattern(rms_norm_mul_add_pattern, j) ||
                match_pattern(rms_norm_mul_rope_view_set_rows_pattern, j) ||
                match_pattern(rms_norm_view_set_rows_pattern, j) ||
                match_pattern(rope_view_set_rows_pattern, j) ||
                match_pattern(hc_post_gate_pattern, j)) {
                continue;
            }
            bool ok = true;
            for (int c = first_unused; c < j; ++c) {
                if (!used[c] &&
                    is_src_of(graph->nodes[j], graph->nodes[c]) &&
                    !(c == current_set.back() && graph->nodes[c]->op == GGML_OP_RMS_NORM && graph->nodes[j]->op == GGML_OP_MUL && empty_or_scheduled_between(c+1, j)) &&
                    !(c == current_set.back() && graph->nodes[c]->op == GGML_OP_UNARY && graph->nodes[j]->op == GGML_OP_MUL && empty_or_scheduled_between(c+1, j)) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_ADD_ID) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_MUL_MAT_ID && graph->nodes[j]->op == GGML_OP_MUL) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_ADD && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_ADD) &&
                    !(j == c+1 && c == current_set.back() && graph->nodes[c]->op == GGML_OP_SSM_CONV && graph->nodes[j]->op == GGML_OP_UNARY)) {
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
                // Look for ROPE/RMS_NORM + VIEW + SET_ROWS and make them consecutive
                if (graph->nodes[rope_idx]->op == GGML_OP_ROPE || graph->nodes[rope_idx]->op == GGML_OP_RMS_NORM) {
                    int view_idx = -1;
                    int set_rows_idx = -1;
                    for (int k = rope_idx + 1; k < std::min(rope_idx + 15, graph->n_nodes); ++k) {
                        if (used[k]) {
                            continue;
                        }
                        if (view_idx == -1 && graph->nodes[k]->op == GGML_OP_VIEW && graph->nodes[k]->src[0] == graph->nodes[rope_idx]) {
                            view_idx = k;
                            continue;
                        }
                        if (view_idx != -1 && graph->nodes[k]->op == GGML_OP_SET_ROWS && graph->nodes[k]->src[0] == graph->nodes[view_idx]) {
                            set_rows_idx = k;
                            break;
                        }
                    }
                    if (set_rows_idx != -1) {
                        const int node_idxs[] = { rope_idx, view_idx, set_rows_idx };
                        const ggml_op ops[] = { graph->nodes[rope_idx]->op, GGML_OP_VIEW, GGML_OP_SET_ROWS };
                        bool can_pull = ggml_can_fuse_subgraph_ext(graph, node_idxs, 3, ops, &set_rows_idx, 1);

                        for (int c = rope_idx + 1; can_pull && c < set_rows_idx; ++c) {
                            if (!used[c] && c != view_idx && !is_empty(graph->nodes[c]) &&
                                is_src_of(graph->nodes[set_rows_idx], graph->nodes[c])) {
                                can_pull = false;
                            }
                        }

                        if (can_pull) {
                            current_set.push_back(view_idx);
                            current_set.push_back(set_rows_idx);
                            used[view_idx] = true;
                            used[set_rows_idx] = true;
                        }
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
                // SSM_CONV + ADD + UNARY: pull the consuming UNARY forward
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_ADD &&
                    graph->nodes[j-1]->op == GGML_OP_SSM_CONV) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        if (graph->nodes[k]->op == GGML_OP_UNARY &&
                            graph->nodes[k]->src[0] == graph->nodes[j]) {
                            current_set.push_back(k);
                            used[k] = true;
                            break;
                        }
                    }
                }
                // UNARY + MUL: pull the consuming MUL forward
                if (j > 0 &&
                    graph->nodes[j]->op == GGML_OP_UNARY) {
                    for (int k = j + 1; k < std::min(j + 15, graph->n_nodes); ++k) {
                        ggml_tensor * mul = graph->nodes[k];
                        if (mul->op != GGML_OP_MUL || (mul->src[0] != graph->nodes[j] && mul->src[1] != graph->nodes[j])) {
                            continue;
                        }
                        ggml_tensor * other = (mul->src[0] == graph->nodes[j]) ? mul->src[1] : mul->src[0];
                        // the other src must either be weights or already processed
                        if (!(other->op == GGML_OP_NONE || used_node_set.find(other) != used_node_set.end())) {
                            continue;
                        }
                        if (!ggml_vk_can_fuse_unary_mul(graph, j, k)) {
                            continue;
                        }
                        current_set.push_back(k);
                        used[k] = true;
                        break;
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
    if (ggml_vk_trace_order()) {
        ggml_vk_trace_check_order(graph);
    }
}

static void ggml_backend_vk_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_record(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    ggml_vk_submit_transfer_ctx(ctx);

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);
    auto* cmd_buf = compute_ctx->s->buffer; // retrieve pointer before it gets reset

    if (vkev->has_event) {
        // Move existing event into submitted
        vkev->events_submitted.push_back(vkev->event);
    }

    // Grab the next event and record it, create one if necessary
    if (vkev->events_free.empty()) {
        vkev->event = ctx->device->device.createEvent({});
    } else {
        vkev->event = vkev->events_free.back();
        vkev->events_free.pop_back();
    }

    vkev->has_event = true;

    ggml_vk_set_event(compute_ctx, vkev->event);

    vkev->tl_semaphore.value++;
    compute_ctx->s->signal_semaphores.push_back(vkev->tl_semaphore);
    ggml_vk_ctx_end(compute_ctx);

    ggml_vk_submit(compute_ctx, {});
    ctx->submit_pending = true;
    vkev->cmd_buffer = cmd_buf;
    vkev->cmd_buffer_use_counter = cmd_buf->use_counter;
    ctx->compute_ctx.reset();
}

static void ggml_backend_vk_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_event_wait(backend=" << backend << ", event=" << event << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    vk_event *vkev = (vk_event *)event->context;

    vk_context compute_ctx = ggml_vk_get_compute_ctx(ctx);

    if (vkev->has_event) {
        // Wait for latest event
        ggml_vk_wait_events(compute_ctx, { vkev->event });

        if (ctx->device->async_use_transfer_queue) {
            vk_context transfer_ctx = ggml_vk_get_transfer_ctx(ctx);
            transfer_ctx->s->wait_semaphores.push_back(vkev->tl_semaphore);
        }
    }
}

// TODO: enable async and synchronize
static ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .set_tensor_async        = */ ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_vk_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_vk_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_vk_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_vk_cpy_tensor_async,
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

// ggmlR Tensor Parallelism (P2P), not upstream ggml.
// Enumerate Vulkan device groups (VK_KHR_device_group / LDA) and probe whether
// the driver reports direct peer memory access between the physical devices in
// each group. A group with >1 device AND peer Copy/Generic features is the
// prerequisite for NVLink-routed (or PCIe-P2P) transfers via a device-group
// logical device — as opposed to opaque-fd between independent VkDevices, which
// the driver may route through host. Writes a human-readable report into
// `report` (truncated to report_size) and returns the number of device groups.
int ggml_backend_vk_get_device_groups(char * report, size_t report_size) {
    // Ensure the Vulkan instance (and dynamic dispatcher) is initialized before
    // touching vk_instance.instance — otherwise enumeratePhysicalDeviceGroups()
    // dereferences a null handle. Mirrors the other entry points here.
    ggml_vk_instance_init();

    size_t off = 0;
    auto emit = [&](const char * fmt, ...) {
        if (!report || off + 1 >= report_size) return;
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(report + off, report_size - off, fmt, ap);
        va_end(ap);
        if (n > 0) off += (size_t) n;
    };

    std::vector<vk::PhysicalDeviceGroupProperties> groups;
    try {
        groups = vk_instance.instance.enumeratePhysicalDeviceGroups();
    } catch (const vk::SystemError& e) {
        emit("device-group enumeration failed: %s\n", e.what());
        return 0;
    }

    emit("Vulkan device groups: %zu\n", groups.size());
    for (size_t g = 0; g < groups.size(); g++) {
        const auto & grp = groups[g];
        emit("  group %zu: %u device(s), subsetAllocation=%d\n",
             g, grp.physicalDeviceCount, (int) grp.subsetAllocation);
        for (uint32_t d = 0; d < grp.physicalDeviceCount; d++) {
            vk::PhysicalDeviceProperties props = grp.physicalDevices[d].getProperties();
            emit("    dev[%u]: %s\n", d, props.deviceName.data());
        }

        // A single-device group cannot do peer transfer; skip the probe.
        if (grp.physicalDeviceCount < 2) {
            emit("    peer: n/a (single-device group — no NVLink/P2P path)\n");
            continue;
        }

        // Build a temporary logical device over the group to query peer memory
        // features. Heap 0 is used as a representative device-local heap.
        vk::DeviceGroupDeviceCreateInfo grp_ci;
        grp_ci.physicalDeviceCount = grp.physicalDeviceCount;
        grp_ci.pPhysicalDevices    = grp.physicalDevices.data();

        // Minimal queue on device 0 of the group.
        float prio = 1.0f;
        vk::DeviceQueueCreateInfo qci({}, 0, 1, &prio);
        vk::DeviceCreateInfo dci({}, 1, &qci);
        dci.setPNext(&grp_ci);

        vk::Device tmp_dev;
        try {
            tmp_dev = grp.physicalDevices[0].createDevice(dci);
        } catch (const vk::SystemError& e) {
            emit("    peer: device-group logical device creation failed: %s\n", e.what());
            continue;
        }

        // With the dynamic dispatcher, resolve device-level function pointers
        // (getGroupPeerMemoryFeatures) against this temporary device before use.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(tmp_dev);

        bool any_peer = false;
        for (uint32_t a = 0; a < grp.physicalDeviceCount; a++) {
            for (uint32_t b = 0; b < grp.physicalDeviceCount; b++) {
                if (a == b) continue;
                vk::PeerMemoryFeatureFlags feat =
                    tmp_dev.getGroupPeerMemoryFeatures(0 /*heap*/, a, b);
                bool copy_src = (bool) (feat & vk::PeerMemoryFeatureFlagBits::eCopySrc);
                bool copy_dst = (bool) (feat & vk::PeerMemoryFeatureFlagBits::eCopyDst);
                bool gen_src  = (bool) (feat & vk::PeerMemoryFeatureFlagBits::eGenericSrc);
                bool gen_dst  = (bool) (feat & vk::PeerMemoryFeatureFlagBits::eGenericDst);
                if (copy_src || copy_dst || gen_src || gen_dst) any_peer = true;
                emit("    peer %u->%u: CopySrc=%d CopyDst=%d GenericSrc=%d GenericDst=%d\n",
                     a, b, copy_src, copy_dst, gen_src, gen_dst);
            }
        }
        emit("    => direct peer access: %s\n",
             any_peer ? "YES (device-group P2P path available)"
                      : "NO (would fall back to host staging)");
        tmp_dev.destroy();
        // Re-point the dynamic dispatcher back to the instance level: the
        // device-level pointers we just loaded belong to the now-destroyed
        // temporary device and must not be used by later Vulkan calls.
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vk_instance.instance);
    }

    return (int) groups.size();
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
                // Driver may report heapBudget > heap.size so we add a safeguard here
                // See https://github.com/ggml-org/llama.cpp/issues/29277
                const vk::DeviceSize budget = std::min(budgetprops.heapBudget[i], heap.size);
                *free += budget > budgetprops.heapUsage[i] ? budget - budgetprops.heapUsage[i] : 0;
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
                                      bool * supports_256_push_constants, uint32_t * max_push_constants_size,
                                      bool * subgroup_shuffle, bool * subgroup_vote) {
    ggml_vk_instance_init();
    GGML_ASSERT(device_idx >= 0 && device_idx < (int) vk_instance.device_indices.size());
    vk_device dev = ggml_vk_get_device((size_t)device_idx);
    if (coopmat_support)    *coopmat_support    = dev->coopmat_support;
    if (coopmat1_fa_support)*coopmat1_fa_support = dev->coopmat1_fa_support;
    if (fp16)               *fp16               = dev->fp16;
    if (subgroup_size)      *subgroup_size       = dev->subgroup_size;
    // Flash Attention scalar/coopmat1 path needs both of these (see FA gate in
    // supports_op); exposed so callers can tell why FA falls back to CPU.
    if (subgroup_shuffle)   *subgroup_shuffle   = dev->subgroup_shuffle;
    if (subgroup_vote)      *subgroup_vote      = dev->subgroup_vote;
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
    int op_offload_min_batch_size;  // ops with batch < this stay on CPU (avoids per-step weight copy for GET_ROWS etc.)
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
        /* .mmap_support          = */ !ctx->is_integrated_gpu,
    };
}

static ggml_backend_t ggml_backend_vk_device_init(ggml_backend_dev_t dev, const char * params) {
    UNUSED(params);
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    return ggml_backend_vk_init(ctx->device);
}

static bool ggml_backend_vk_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op) {
    {
        // ggml 0.27: reject any tensor larger than the device can bind, so the op
        // falls back to CPU instead of failing or computing garbage on the GPU.
        ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
        const vk_device& device = ggml_vk_get_device(ctx->device);

        const bool uses_bda = (op->op == GGML_OP_IM2COL || op->op == GGML_OP_IM2COL_3D) &&
                              device->shader_int64 && device->buffer_device_address;

        // ggmlR: always log the fallback, once per (op, tensor, size) -- supports_op
        // runs for every node on every schedule, so an unthrottled message would
        // flood the log with the same line.
        auto const & tensor_size_supported = [&](const ggml_tensor * t) {
            const size_t tensor_size = ggml_nbytes(t);
            const char * limit_name = nullptr;
            uint64_t limit = 0;
            if (tensor_size > device->max_buffer_size) {
                limit_name = "max buffer size";
                limit = device->max_buffer_size;
            } else if (!uses_bda && !device->shader_64b_indexing &&
                       tensor_size > device->properties.limits.maxStorageBufferRange) {
                // For im2col shaders using BDA, and with shader64BitIndexing,
                // the maxStorageBufferRange limit doesn't apply.
                limit_name = "maxStorageBufferRange";
                limit = device->properties.limits.maxStorageBufferRange;
            }
            if (limit_name == nullptr) {
                return true;
            }
            static std::mutex log_mutex;
            static std::set<std::string> logged;
            const std::string key = std::string(ggml_op_name(op->op)) + "|" + t->name + "|" + std::to_string(tensor_size);
            std::lock_guard<std::mutex> guard(log_mutex);
            if (logged.insert(key).second) {
                GGML_LOG_WARN("ggml_vulkan: %s '%s' not run on GPU: tensor '%s' is %.1f MiB, over the device %s of %.1f MiB; falling back to CPU\n",
                              ggml_op_name(op->op), op->name, t->name,
                              tensor_size / 1048576.0, limit_name, limit / 1048576.0);
            }
            return false;
        };
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && !tensor_size_supported(op->src[i])) {
                return false;
            }
        }
        if (!tensor_size_supported(op)) {
            return false;
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
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
                case GGML_UNARY_OP_ROUND_EVEN:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_SGN:
                    return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
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
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    // Match upstream: no full-contiguity requirement (the shader
                    // takes per-row strides). Requiring ggml_is_contiguous forced
                    // GLU to CPU for permuted views, splitting the graph.
                    // glu_main.glsl is 4D: it folds dim 4 into dim 3 through
                    // nb[3], which is only right when dim 4 directly follows
                    // dim 3 in memory (always true upstream, GGML_MAX_DIMS = 4).
                    {
                        auto dim4_folds = [](const ggml_tensor * t) {
                            return !t || t->ne[4] == 1 || t->nb[4] == t->nb[3] * t->ne[3];
                        };
                        if (!dim4_folds(op->src[0]) || !dim4_folds(op->src[1]) || !dim4_folds(op)) {
                            return false;
                        }
                    }
                    return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           (op->src[0]->type == op->type) &&
                           (!op->src[1] || op->src[1]->type == op->src[0]->type);
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
                    if (!ggml_vk_use_mul_mat_vec_id(op)) {
                        const ggml_tensor * ids = op->src[2];
                        // Shaders store row IDs as uint16_t slot and token indices.
                        if (ids->ne[0] > int64_t(UINT16_MAX) + 1 || ids->ne[1] > int64_t(UINT16_MAX) + 1) {
                            return false;
                        }
                    }
                    if (!device->mul_mat_id_s[src0_type] && !device->mul_mat_id_m[src0_type] && !device->mul_mat_id_l[src0_type]) {
                        // If there's not enough shared memory for row_ids and the result tile, fallback to CPU
                        return false;
                    }
                    if (ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                        return false;
                    }
                }
                switch (src0_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
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
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_TQ1_0:
                    case GGML_TYPE_TQ2_0:
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
                if (op->src[1]->type == GGML_TYPE_BF16 && op->src[0]->type != GGML_TYPE_BF16 && op->op != GGML_OP_MUL_MAT) {
                    // BF16 in src1 (widened to F32/F16) is only wired up for MUL_MAT
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
                // ggml 0.27: K and V types are independent spec constants, so
                // mixed K/V is fine on every path.
                // ggmlR: Q4_K K/V (not upstream) -- scalar and coopmat1 paths only.
                auto fa_kv_ok = [coopmat2](ggml_type t) {
                    switch (t) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    case GGML_TYPE_Q4_K:
                        return !coopmat2;
                    default:
                        return false;
                    }
                };
                if (!fa_kv_ok(op->src[1]->type) || !fa_kv_ok(op->src[2]->type)) {
                    return false;
                }
                if ((op->src[1]->type == GGML_TYPE_BF16) != (op->src[2]->type == GGML_TYPE_BF16)) {
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
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
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
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_TQ1_0:
                    case GGML_TYPE_TQ2_0:
                    case GGML_TYPE_I32:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_SET_ROWS:
            {
                if ((op->src[0]->type != GGML_TYPE_F32 && op->src[0]->type != GGML_TYPE_F16) ||
                    (op->src[1]->type != GGML_TYPE_I32 && op->src[1]->type != GGML_TYPE_I64)) {
                    return false;
                }
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
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
            {
                // ggmlR: the add reduction accumulates with atomicAdd on floats,
                // which needs VK_EXT_shader_atomic_float. The plain scatter does
                // not, so only the add mode is gated -- refusing both would send
                // ordinary scatters to the CPU for no reason.
                int32_t reduction;
                memcpy(&reduction, op->op_params, sizeof(int32_t));
                if (reduction == 1) {
                    ggml_backend_vk_device_context * vk_ctx =
                        (ggml_backend_vk_device_context *)dev->context;
                    const vk_device& vk_dev = ggml_vk_get_device(vk_ctx->device);
                    if (!vk_dev->atomic_float_add) {
                        return false;
                    }
                }
                return op->type == GGML_TYPE_F32;
            }
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
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
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
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
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
            return ggml_type_size(op->type) == ggml_type_size(op->src[0]->type) &&
                  (ggml_type_size(op->type) == sizeof(float) || ggml_type_size(op->type) == 2);
        case GGML_OP_REPEAT_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ROPE:
            // ggml 0.27 guard + ggmlR DIVERGENCE: rope_funcs.glsl splits a row
            // into i1/i2/i3 only, so with ne[4] > 1 i3 runs past ne03.
            return ggml_is_contiguous_rows(op) && ggml_is_contiguous_rows(op->src[0]) &&
                   op->src[0]->ne[4] == 1;
        case GGML_OP_ROPE_BACK:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NORM:
            // ggml 0.27: norm.comp indexes through generic_unary_head, so rows
            // may sit at any stride. Upstream accepts any layout; ggmlR keeps
            // contiguous rows, which is what the CPU kernel handles (it asserts
            // nb[0] == sizeof(float)), so a fallback never meets a layout the
            // CPU cannot compute.
        case GGML_OP_L2_NORM:
            // Match upstream: L2_NORM only needs contiguous *rows*, not a fully
            // contiguous tensor. Requiring full contiguity (as this case used to,
            // sharing the NORM/GROUP_NORM branch) rejected the per-head Q/K L2 norm
            // in Qwen3.5-style models (src is a permuted view), forcing it to CPU
            // and shattering the graph into ~1 split per layer on GPUs without
            // Flash Attention (e.g. Tesla P100).
            return ggml_is_contiguous_rows(op->src[0]) &&
                   op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
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
        // DIVERGENCE from upstream: GELU backward, see ggml_gelu_back().
        case GGML_OP_GELU_BACK:
        case GGML_OP_RMS_NORM_BACK:
        // DIVERGENCE from upstream: LayerNorm backward, see ggml_norm_back().
        case GGML_OP_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->type == op->src[0]->type;
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
                    return op->ne[0] <= (1 << std::min(device->max_workgroup_size_log2, num_argsort_pipelines - 1));
                }
            }
        case GGML_OP_TOP_K:
            {
                if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0])) {
                    return false;
                }
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                // large k falls back to radix-select
                const uint32_t min_pipeline =
                    std::max((uint32_t) log2f(float(op->ne[0])) + 1, device->subgroup_size_log2);
                if (min_pipeline < num_topk_pipelines && device->pipeline_topk_f32[min_pipeline]) {
                    return true;
                }
                return device->pipeline_topk_radix_f32 != nullptr;
            }
        case GGML_OP_UPSCALE:
            if (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                if ((op->op_params[0] & 0xFF) != GGML_SCALE_MODE_BILINEAR) {
                    return false;
                }
            }
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ACC:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_SET:
            return op->src[0]->type == op->src[1]->type && op->src[0]->type == op->type &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_I32);
        case GGML_OP_CONCAT: {
            return ggml_vk_concat_supported(op->src[0], op->src[1], op);
        }
        case GGML_OP_ADD1:
            return (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32)
                || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16);
        case GGML_OP_ARANGE:
            return op->type == GGML_TYPE_F32;
        case GGML_OP_FILL:
            return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16;
        case GGML_OP_SCALE:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
            // ggmlR DIVERGENCE (5D): pad_reflect_1d.comp indexes only ne0..ne3.
            return op->ne[4] == 1 && op->src[0]->ne[4] == 1 && op->src[0]->type == GGML_TYPE_F32;
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
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                if (op->type != GGML_TYPE_F32) {
                    return false;
                }
                for (uint32_t i = 0; i < GGML_MAX_SRC; ++i) {
                    if (op->src[i] && op->src[i]->type != GGML_TYPE_F32) {
                        return false;
                    }
                }
                // hc is hardcoded to 4 in the shaders. ggml only constrains it
                // to 4 for COMB, so PRE/POST have to be checked here.
                if (op->op == GGML_OP_DSV4_HC_PRE && op->src[0]->ne[1] != 4) {
                    return false;
                }
                if (op->op == GGML_OP_DSV4_HC_POST && op->src[1]->ne[1] != 4) {
                    return false;
                }
                if (op->op == GGML_OP_DSV4_HC_COMB) {
                    return device->pipeline_dsv4_hc_comb_f32 != nullptr;
                }
                return true;
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
        case GGML_OP_CROSS_ENTROPY_LOSS:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32
                && ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32
                && ggml_are_same_shape(op->src[0], op->src[1])
                && ggml_is_contiguous(op) && ggml_is_scalar(op) && op->type == GGML_TYPE_F32;
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
        case GGML_OP_POOL_1D:
            // ggmlR DIVERGENCE (5D): the pool1d dispatch folds only ne2*ne3 into N.
            return op->ne[4] == 1 && op->src[0]->ne[4] == 1 &&
                   ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true; // all inputs are contiguous, see ggml.c
        case GGML_OP_GATED_LINEAR_ATTN:
            // the shader block size is hardcoded to head_size 64
            // ggmlR DIVERGENCE (5D): gla.comp is 4D.
            return op->ne[4] == 1 && op->src[0]->ne[4] == 1 &&
                   op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[0]->ne[0] == 64;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
                auto device = ggml_vk_get_device(ctx->device);
                const ggml_tensor * q = op->src[0];
                const ggml_tensor * k = op->src[1];
                const ggml_tensor * w = op->src[2];
                const ggml_tensor * m = op->src[3];

                // the q/w/m types and the shape relationships between q, k, w, m and dst
                // are already asserted in ggml_lightning_indexer()
                if (!ggml_vk_lightning_indexer_k_type_supported(k->type) || !device->fp16) {
                    return false;
                }

                // the shader block size is hardcoded to head size 128
                if (q->ne[0] != 128) {
                    return false;
                }

                // the shader indexes the buffers by element stride, and is dispatched
                // without allow_misalign
                for (const ggml_tensor * t : {q, k, w, m, op}) {
                    if (t->nb[0] != ggml_type_size(t->type) ||
                        (vk_tensor_offset(t) + t->view_offs) % device->properties.limits.minStorageBufferOffsetAlignment != 0) {
                        return false;
                    }
                    // the strides get scaled down from bytes, so the division must be exact
                    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                        if (t->nb[i] % ggml_type_size(t->type) != 0) {
                            return false;
                        }
                    }
                }
                return true;
            }
        case GGML_OP_GATED_DELTA_NET:
            {
                const uint32_t S_v = op->src[2]->ne[0];
                if (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
                    return false;
                }
                for (int i = 0; i < 6; i++) {
                    if (op->src[i] == nullptr || op->src[i]->type != GGML_TYPE_F32) {
                        return false;
                    }
                }
                return op->type == GGML_TYPE_F32;
            }
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

                // 0.11/0.27 shader: one d_state row in shared memory, subgroup reductions
                size_t shmem_size = d_state * sizeof(float);

                if (shmem_size > device->properties.limits.maxComputeSharedMemorySize) {
                    return false;
                }

                if (!device->subgroup_basic) {
                    return false;
                }

                return true;
            }
        case GGML_OP_SSM_CONV:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SSM_CONV_BACK:
            // ggmlR extension: backward of ssm_conv. The shader keeps the
            // kernel-sized d_c accumulator in registers, so d_conv is capped at
            // the MAX_D_CONV the shader declares (Mamba uses 4); anything wider
            // falls back to the CPU rather than reading past the array.
            if (op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
                return false;
            }
            if (op->src[1]->ne[0] > 8) {
                return false;
            }
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float) &&
                   op->src[1]->nb[0] == sizeof(float) &&
                   op->src[2]->nb[0] == sizeof(float);
        case GGML_OP_SSM_SCAN_BACK:
            {
                // ggmlR extension: backward of ssm_scan.
                for (int i = 0; i < 8; i++) {
                    if (op->src[i] == nullptr) {
                        return false;
                    }
                }
                if (op->src[6]->type != GGML_TYPE_I32) {
                    return false;
                }
                if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                // Mamba-2 only, like the forward shader: a per-state decay would
                // need a different reduction over d_state.
                const bool is_mamba2 = (op->src[3]->nb[1] == sizeof(float));
                if (!is_mamba2) {
                    return false;
                }

                const uint32_t d_state = (uint32_t)op->src[0]->ne[0];
                if (d_state != 128 && d_state != 256) {
                    return false;
                }

                ggml_backend_vk_device_context * vk_ctx =
                    (ggml_backend_vk_device_context *)dev->context;
                const vk_device& vk_dev = ggml_vk_get_device(vk_ctx->device);

                // Four of the six gradients are shared across workgroups and
                // accumulate with atomicAdd.
                if (!vk_dev->atomic_float_add) {
                    return false;
                }

                // The replay scratch is (nt+1) states per (sequence, head,
                // channel), which grows linearly in tokens and can reach
                // hundreds of megabytes at long sequence lengths. Cap it at a
                // quarter of device-local memory and let the CPU take anything
                // larger: a fallback costs time, an allocation failure costs the
                // whole run. TODO: a quarter is a judgement call, not a measured
                // optimum -- revisit once there is a sequence length that
                // actually presses against it.
                const uint32_t nr = (uint32_t)op->src[0]->ne[1];
                const uint32_t nh = (uint32_t)op->src[1]->ne[1];
                const uint32_t nt = (uint32_t)op->src[1]->ne[2];
                const uint32_t ns = (uint32_t)op->src[1]->ne[3];

                const size_t scratch_size =
                    (size_t)ns * nh * nr * (nt + 1) * d_state * sizeof(float);

                size_t mem_free = 0, mem_total = 0;
                ggml_backend_vk_get_device_memory(vk_ctx->device, &mem_free, &mem_total);
                if (mem_total > 0 && scratch_size > mem_total / 4) {
                    return false;
                }

                return true;
            }
        case GGML_OP_OUT_PROD:
            // ggmlR: dst and both sources must be F32, and the innermost stride
            // of src0 and dst must be the element size -- the same constraints
            // the CPU kernel asserts. A transposed src1 IS supported (the
            // second mul_mat gradient needs it), so nb10 is deliberately not
            // constrained here.
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float) &&
                   op->nb[0] == sizeof(float);
        case GGML_OP_GET_ROWS_BACK:
            {
                // ggmlR: backward of embedding lookup. Repeated tokens scatter
                // into the same row, so this needs atomic float add.
                ggml_backend_vk_device_context * vk_ctx =
                    (ggml_backend_vk_device_context *)dev->context;
                const vk_device& vk_dev = ggml_vk_get_device(vk_ctx->device);
                if (!vk_dev->atomic_float_add) {
                    return false;
                }
                return op->type == GGML_TYPE_F32 &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]);
            }
        case GGML_OP_IM2COL_BACK:
            // ggmlR: backward of convolution. The CPU kernel requires both
            // sources to be F32 and so does this; an F16 kernel is rejected
            // earlier, when the backward graph is built (see ggml-graph.c).
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op) &&
                   ggml_is_contiguous(op->src[0]);
        case GGML_OP_FLASH_ATTN_BACK:
            {
                // ggmlR: backward of flash attention (upstream has neither a
                // working op nor a shader). Without this the whole backward
                // left the GPU, splitting a training graph once per attention
                // block.
                ggml_backend_vk_device_context * vk_ctx =
                    (ggml_backend_vk_device_context *)dev->context;
                const vk_device& vk_dev = ggml_vk_get_device(vk_ctx->device);

                // dk and dv are shared by every query row of a kv head, so
                // they accumulate with atomicAdd.
                if (!vk_dev->atomic_float_add) {
                    return false;
                }

                // The shader caches the softmax row and dP in shared memory,
                // sized by the pipeline's MAX_M specialisation constant. Longer
                // kv runs go to the CPU rather than overrun it.
                if (op->src[1]->ne[1] > 1024) {
                    return false;
                }

                // f32 only, and contiguous: the shader indexes with element
                // strides and assumes rows are packed.
                if (op->type != GGML_TYPE_F32 ||
                    op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32 ||
                    op->src[2]->type != GGML_TYPE_F32 ||
                    op->src[4] == nullptr || op->src[4]->type != GGML_TYPE_F32) {
                    return false;
                }
                if (!ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1]) ||
                    !ggml_is_contiguous(op->src[2]) || !ggml_is_contiguous(op->src[4])) {
                    return false;
                }

                // The mask, when present, is f16 and contiguous.
                if (op->src[3] != nullptr &&
                    (op->src[3]->type != GGML_TYPE_F16 || !ggml_is_contiguous(op->src[3]))) {
                    return false;
                }

                return true;
            }
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            // ggmlR: the shader indexes rows as row*nc, so everything must be
            // contiguous -- the same restriction the CPU kernel asserts.
            return op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op) &&
                   ggml_is_contiguous(op->src[1]) &&
                   ggml_is_contiguous(op->src[2]) &&
                   // ggml 0.27: scalar loss gradient, logits/labels/result of one shape
                   ggml_is_scalar(op->src[0]) &&
                   ggml_are_same_shape(op->src[1], op->src[2]) &&
                   ggml_are_same_shape(op->src[1], op);
        case GGML_OP_CONV_TRANSPOSE_1D:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_COL2IM_1D:
            // ggmlR DIVERGENCE (5D): col2im_1d.comp is 4D.
            return op->ne[4] == 1 && op->src[0]->ne[4] == 1 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16) &&
                   op->type == op->src[0]->type &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op);
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                const bool transpose = op->op == GGML_OP_CONV_TRANSPOSE_2D;
                const int64_t cout = !transpose ? op->src[0]->ne[3] : op->src[0]->ne[2];
                const int64_t cin  = !transpose ? op->src[0]->ne[2] : op->src[0]->ne[3];

                // Channel-contiguous format is not supported yet.
                return ((op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    (op->src[0]->nb[0] == sizeof(float) || op->src[0]->nb[0] == sizeof(ggml_fp16_t) ) &&
                    op->src[1]->type == GGML_TYPE_F32 &&
                    op->type == GGML_TYPE_F32 &&
                    cout == op->ne[2] &&
                    cin == op->src[1]->ne[2] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op));
            }
        // ggmlR extension: ONNX QLinearConv. The shader is a line-for-line
        // port of the host kernel, int16 pair saturation included, so the two
        // backends agree bit for bit -- see vulkan-shaders/qconv_i32.comp.
        case GGML_OP_QCONV_I32:
            {
                // Says WHICH condition declined the op. Several can, and a bare
                // false leaves the node on the CPU with nothing to read: the op
                // simply appears under CPU-only in a profile, exactly as it
                // would if the backend had never heard of it.
                //
                // This is the only place the reason is visible, and it is a
                // blind spot worth keeping lit: the same shape of bug -- a GPU
                // path silently declining and no way to see why -- has cost
                // this op two debugging sessions already.
                //
                // Gated on ONNX_TRACE_NODES, the flag the rest of the ONNX
                // layer already uses, rather than a private one. The value is
                // read with getenv instead of calling onnx_trace_nodes(),
                // because the Vulkan backend deliberately does not link
                // against the ONNX layer; an environment variable is part of
                // the program's external interface, so sharing it is not the
                // same as sharing code.
                static int trace = -1;
                if (trace < 0) {
                    const char * e = getenv("ONNX_TRACE_NODES");
                    trace = (e && *e && *e != '0') ? 1 : 0;
                }
                // GGML_LOG_INFO, the same channel the rest of this backend
                // logs through (the device-caps line, the unsupported-op
                // summary below). A plain fprintf does NOT work here:
                // r_ggml_compat.h rewrites it to REvprintf, which routes to
                // R's console, and a line emitted from the scheduler's call
                // into supports_op never reached the terminal at all.
                #define QC_NO(reason) do {                                     \
                    if (trace) GGML_LOG_INFO(                                  \
                        "[qconv-supports] '%s': NO (%s) types x=%d w=%d ws=%d " \
                        "wz=%d bi=%d dst=%d\n", op->name, reason,              \
                        (int)op->src[0]->type, (int)op->src[1]->type,          \
                        op->src[2] ? (int)op->src[2]->type : -1,               \
                        op->src[3] ? (int)op->src[3]->type : -1,               \
                        op->src[4] ? (int)op->src[4]->type : -1,               \
                        (int)op->type);                                        \
                    return false;                                              \
                } while (0)

                // The kernel indexes x, w and dst as packed arrays, so a view
                // with any other stride would read neighbouring data and
                // return plausible numbers rather than fail.
                if (!ggml_is_contiguous(op))          QC_NO("dst not contiguous");
                if (!ggml_is_contiguous(op->src[0]))  QC_NO("x not contiguous");
                if (!ggml_is_contiguous(op->src[1]))  QC_NO("w not contiguous");
                if (op->type != GGML_TYPE_F32)          QC_NO("dst not f32");
                if (op->src[0]->type != GGML_TYPE_F32)  QC_NO("x not f32");
                if (op->src[1]->type != GGML_TYPE_F32)  QC_NO("w not f32");

                // Each table is bound at the type it actually arrives with,
                // which is not the same for all three: a scale is float, a
                // zero point is INT8 in the file and comes out F32 because the
                // loader widens it, and a bias is INT32 by the operator spec
                // and stays integer. Requiring one type for all of them
                // declined every convolution in the model over its bias.
                if (!op->src[2])                         QC_NO("no w_scale");
                if (op->src[2]->type != GGML_TYPE_F32)   QC_NO("w_scale not f32");
                if (op->src[3] && op->src[3]->type != GGML_TYPE_F32)
                                                         QC_NO("w_zp not f32");
                if (op->src[4] && op->src[4]->type != GGML_TYPE_I32)
                                                         QC_NO("bias not i32");
                if (!op->src[5])                         QC_NO("no mult table");
                if (op->src[5]->type != GGML_TYPE_F32)   QC_NO("mult not f32");

                // One thread per output element, so the grid is the whole
                // output. A dispatch axis over the driver's limit does not
                // fail -- it aborts inside the dispatch, which on NVIDIA
                // (65535) kills the process with nothing printed. Decline and
                // let the host kernel take it instead.
                {
                    ggml_backend_vk_device_context * dev_ctx =
                        (ggml_backend_vk_device_context *)dev->context;
                    vk_device vkdev = ggml_vk_get_device(dev_ctx->device);
                    // ne[3] included: ggml_vk_qconv_i32 dispatches one flat
                    // grid over batch*C_out*H_out*W_out, so a guard without
                    // the batch undercounted by that factor -- 51x on
                    // MaskRCNN's mask head, which is the shape most likely
                    // to approach the limit.
                    const uint64_t total  = (uint64_t)op->ne[0] * (uint64_t)op->ne[1] *
                                            (uint64_t)op->ne[2] * (uint64_t)op->ne[3];
                    const uint64_t groups = (total + 255u) / 256u;   // local_size_x
                    if (groups > (uint64_t)vkdev->properties.limits.maxComputeWorkGroupCount[0]) {
                        QC_NO("output grid above the workgroup limit");
                    }
                }
                #undef QC_NO
                // Accepted nodes are reported too: "which of the 63 went to the
                // GPU" is the other half of the same question, and a trace that
                // only prints refusals cannot distinguish an op that was
                // accepted from one that was never asked about.
                if (trace) {
                    GGML_LOG_INFO("[qconv-supports] '%s': YES\n", op->name);
                }
                return true;
            }
        // ggmlR extension: ONNX QLinearMatMul, see vulkan-shaders/qmatmul_i32.comp.
        // Declining is always safe: the host kernel computes the same bits.
        case GGML_OP_QMATMUL_I32:
            {
                static int trace = -1;
                if (trace < 0) {
                    const char * e = getenv("ONNX_TRACE_NODES");
                    trace = (e && *e && *e != '0') ? 1 : 0;
                }
                // Same reasoning and channel as QC_NO above.
                #define QM_NO(reason) do {                                     \
                    if (trace) GGML_LOG_INFO(                                  \
                        "[qmatmul-supports] '%s': NO (%s)\n", op->name, reason); \
                    return false;                                              \
                } while (0)

                // The escape hatch the direct dispatch had, kept: =0 sends every
                // QLinearMatMul to the host kernel, for a driver whose integer
                // arithmetic disagrees.
                {
                    static int off = -1;
                    if (off < 0) {
                        const char * e = getenv("GGMLR_ONNX_GPU_QMATMUL");
                        off = (e && e[0] == '0' && e[1] == '\0') ? 1 : 0;
                    }
                    if (off) QM_NO("GGMLR_ONNX_GPU_QMATMUL=0");
                }

                const ggml_tensor * a  = op->src[0];
                const ggml_tensor * b  = op->src[1];
                const ggml_tensor * bs = op->src[2];
                const ggml_tensor * bz = op->src[3];
                if (!a || !b || !bs)                     QM_NO("missing source");
                if (op->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 ||
                    b->type != GGML_TYPE_F32)            QM_NO("not f32");
                if (bs->type != GGML_TYPE_F32)           QM_NO("b_scale not f32");
                // The shader binds the zero point as float; an I32 table would
                // be read as bit patterns. The host kernel takes either.
                if (bz && bz->type != GGML_TYPE_F32)     QM_NO("b_zp not f32");
                if (!ggml_is_contiguous(op) || !ggml_is_contiguous(a) ||
                    !ggml_is_contiguous(b))              QM_NO("not contiguous");
                // The batch axes, checked rather than assumed: the shader has
                // no stride for them, so anything but 1 would be computed as
                // the first slice only. Decided here from the start rather
                // than copied from QCONV_I32, whose guard left ne[3] out.
                if (a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
                    op->ne[2] != 1 || op->ne[3] != 1)    QM_NO("batched");

                const uint64_t M = (uint64_t)a->ne[1];
                const uint64_t N = (uint64_t)b->ne[1];
                const uint64_t K = (uint64_t)a->ne[0];
                {
                    ggml_backend_vk_device_context * dev_ctx =
                        (ggml_backend_vk_device_context *)dev->context;
                    vk_device vkdev = ggml_vk_get_device(dev_ctx->device);
                    // 2D grid of 16x16 blocks: N across, M down. Each axis
                    // against its own limit -- overshooting aborts inside the
                    // dispatch on NVIDIA (65535) with nothing printed.
                    if ((N + 15u) / 16u > (uint64_t)vkdev->properties.limits.maxComputeWorkGroupCount[0] ||
                        (M + 15u) / 16u > (uint64_t)vkdev->properties.limits.maxComputeWorkGroupCount[1])
                        QM_NO("output grid above the workgroup limit");
                }

                // ⚠️ The workgroup test bounds the number of THREADS, not the
                // WORK. Each thread walks K end to end, so one dispatch costs
                // M*N*K and that is unbounded whatever the grid. Untiled, the
                // box head of MaskRCNN (M=1000, N=1024, K=12544, ~96 GB of
                // traffic) did not finish inside the driver's watchdog on
                // RX 9070 / RADV: the compute ring timed out and took the
                // session with it (MODE1 reset, VRAM lost, only
                // vk::DeviceLostError reaching the process). The 2D tile
                // (A and B staged, ~6 GB) brought it to ~285 ms.
                //
                // A guard, not a verdict on a shape: 5e10 is ~4x that node,
                // roughly a second of dispatch, well inside the watchdog.
                // GGMLR_QI32_MAX_WORK overrides it (0 lifts it); a shape over
                // the cap runs on the host kernel, slower but unable to hang
                // the device.
                {
                    static double max_work = -1.0;
                    if (max_work < 0.0) {
                        const char * e = getenv("GGMLR_QI32_MAX_WORK");
                        max_work = e ? atof(e) : 5e10;
                    }
                    if (max_work > 0.0 && (double)M * (double)N * (double)K > max_work)
                        QM_NO("M*N*K above GGMLR_QI32_MAX_WORK");
                }
                #undef QM_NO
                if (trace) {
                    GGML_LOG_INFO("[qmatmul-supports] '%s': YES\n", op->name);
                }
                return true;
            }
        case GGML_OP_CONV_3D:
            // ggmlR DIVERGENCE (5D): IC*N is packed into ne3, nothing may live in ne4.
            if (op->ne[4] != 1 || op->src[0]->ne[4] != 1 || op->src[1]->ne[4] != 1) {
                return false;
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op);
        default:
            return false;
    }

    UNUSED(dev);
}

// Industrial telemetry: ops the Vulkan backend cannot run fall back to CPU,
// which forces GPU<->CPU copies around each such node and silently slows graphs.
// supports_op is on a hot path (called per node at graph build), so we never log
// per-call — instead we tally rejected op types and expose them on demand via
// ggml_vk_log_unsupported_op_summary() (called after graph build / compute).
static std::map<ggml_op, uint64_t> g_vk_unsupported_op_counts;
static uint64_t                    g_vk_unsupported_op_total = 0;

static bool ggml_backend_vk_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    bool ok = ggml_backend_vk_device_supports_op_impl(dev, op);
    if (!ok) {
        g_vk_unsupported_op_counts[op->op]++;
        g_vk_unsupported_op_total++;
    }
    return ok;
}

// Emit a one-line summary of ops that were rejected by the Vulkan backend (and
// therefore ran on CPU), then reset the tally. No-op when everything ran on GPU.
static void ggml_vk_log_unsupported_op_summary(const char * phase) {
    if (g_vk_unsupported_op_total == 0) {
        return;
    }
    // Gated behind GGMLR_LOG_DEVICE (same switch as the device-caps line) so it
    // stays available for production diagnostics but does not flood test output,
    // where training graphs routinely fall back ops like OUT_PROD /
    // CROSS_ENTROPY_LOSS_BACK to CPU. Still reset the tally either way.
    const char * log_dev = getenv("GGMLR_LOG_DEVICE");
    if (log_dev != nullptr && log_dev[0] != '\0' && log_dev[0] != '0') {
        std::string detail;
        for (const auto & kv : g_vk_unsupported_op_counts) {
            if (!detail.empty()) {
                detail += ", ";
            }
            detail += std::string(ggml_op_name(kv.first)) + "=" + std::to_string(kv.second);
        }
        GGML_LOG_INFO("ggml_vulkan: %llu op(s) not supported on GPU during %s, ran on CPU (per-type: %s)\n",
                      (unsigned long long)g_vk_unsupported_op_total,
                      phase ? phase : "graph",
                      detail.c_str());
    }
    g_vk_unsupported_op_counts.clear();
    g_vk_unsupported_op_total = 0;
}

static bool ggml_backend_vk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        return false;
    }

    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    ggml_backend_vk_buffer_type_context * buft_ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return buft_ctx->device->idx == ctx->device;
}

// Estimate the effective batch dimension of an op for offload decisions.
// Matches upstream llama.cpp/ggml-vulkan: GET_ROWS reports 0 so that decode
// (single-token) doesn't force the tok_embd matrix to be marshalled to the
// GPU on every step, MUL_MAT uses ne[1], MUL_MAT_ID/ROPE/ROPE_BACK use ne[2].
static int64_t ggml_vk_get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_vk_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_vk_device_context * dev_ctx = (ggml_backend_vk_device_context *)dev->context;

    return ggml_vk_get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_vk_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = new vk_event;
    if (!vkev) {
        return nullptr;
    }

    // No events initially, they get created on demand
    vkev->has_event = false;

    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vkev->tl_semaphore = { device->device.createSemaphore(ci), 0 };

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ vkev,
    };
}

static void ggml_backend_vk_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);

    vk_event *vkev = (vk_event *)event->context;

    device->device.destroySemaphore(vkev->tl_semaphore.s);
    for (auto& event : vkev->events_free) {
        device->device.destroyEvent(event);
    }
    for (auto& event : vkev->events_submitted) {
        device->device.destroyEvent(event);
    }
    if (vkev->has_event) {
        device->device.destroyEvent(vkev->event);
    }
    delete vkev;
    delete event;
}

static void ggml_backend_vk_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    VK_LOG_DEBUG("ggml_backend_vk_device_event_synchronize(backend=" << dev << ", event=" << event << ")");
    ggml_backend_vk_device_context * ctx = (ggml_backend_vk_device_context *)dev->context;
    auto device = ggml_vk_get_device(ctx->device);
    vk_event *vkev = (vk_event *)event->context;

    // Only do something if the event has actually been used
    if (vkev->has_event) {
        vk::Semaphore sem = vkev->tl_semaphore.s;
        uint64_t val = vkev->tl_semaphore.value;
        vk::SemaphoreWaitInfo swi{vk::SemaphoreWaitFlags{}, sem, val};
        VK_CHECK(device->device.waitSemaphores(swi, UINT64_MAX), "event_synchronize", device);

        // Reset and move submitted events
        for (auto& event : vkev->events_submitted) {
            device->device.resetEvent(event);
        }
        vkev->events_free.insert(vkev->events_free.end(), vkev->events_submitted.begin(), vkev->events_submitted.end());
        vkev->events_submitted.clear();

        // Finished using current command buffer so we flag for reuse
        if (vkev->cmd_buffer) {
            // Only flag for reuse if it hasn't been reused already
            if (vkev->cmd_buffer_use_counter == vkev->cmd_buffer->use_counter) {
                vkev->cmd_buffer->in_use = false;
                vkev->cmd_buffer->buf.reset();
            }
            vkev->cmd_buffer = nullptr;
        }
    }
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
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
            for (int i = 0; i < ggml_backend_vk_get_device_count(); i++) {
                ggml_backend_vk_device_context * ctx = new ggml_backend_vk_device_context;
                char desc[256];
                ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
                ctx->device = i;
                ctx->name = GGML_VK_NAME + std::to_string(i);
                ctx->description = desc;
                ctx->is_integrated_gpu = ggml_backend_vk_get_device_type(i) == vk::PhysicalDeviceType::eIntegratedGpu;
                ctx->pci_bus_id = ggml_backend_vk_get_device_pci_id(i);
                ctx->op_offload_min_batch_size = min_batch_size;
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

// ggmlR, not upstream: explicit Vulkan teardown, called from R's .onUnload while
// the package DLL — and crucially the Vulkan loader / Mesa ICD .so files — are
// still mapped. The static `vk_instance` (instance + shared_ptr devices[]) is
// otherwise destroyed at process exit in an order the C runtime does not
// coordinate with the loader's own static destructors, so the loader can be
// unmapped before our device destructors run their loader calls
// (destroyFence/destroyDescriptorSetLayout/device.destroy), giving a flaky
// segfault in unmapped memory after results are already returned. Draining and
// releasing the instance's device refs here, while the loader is still mapped,
// gases idle driver threads early and shrinks that race.
//
// We release the instance's DEVICE references but leave the instance handle
// itself alive (see the note at the end): late R XPtr finalizers may still need
// a valid instance to run vkFreeMemory against.
//
// Safe w.r.t. downstream (llamaR/sd2R): `vk_device` is a shared_ptr, so
// resetting the instance's reference only actually destroys a device once every
// live backend that copied it has been freed — never prematurely. waitIdle first
// drains any in-flight driver work so the loader threads are quiescent.
extern "C" void ggml_backend_vk_shutdown(int hard, int status) {
    r_tp_tracef("vk_shutdown: enter (initialized=%d hard=%d status=%d)",
                (int) vk_instance_initialized, hard, status);
    if (!vk_instance_initialized) {
        r_tp_tracef("vk_shutdown: already down, no-op");
#ifdef GGML_VK_HARD_EXIT
        if (hard) {
            // Still honour the hard request: skip the process teardown that races
            // the Vulkan loader's static destructors even when we own no devices.
            r_tp_tracef("vk_shutdown: hard exit(%d) (nothing to tear down)", status);
            fflush(NULL);
            _exit(status);
        }
#endif
        return;   // idempotent: guards double-shutdown and use-after-shutdown init
    }
    vk_instance_initialized = false;

    int idx = 0;
    for (auto & dev : vk_instance.devices) {
        if (dev) {
            // use_count > 1 means a live backend (e.g. from split_mul_mat/pp_forward,
            // or held by llamaR/sd2R) still references this device — resetting the
            // instance's ref will NOT destroy it now; ~vk_device_struct runs later,
            // when that backend is freed, possibly after we destroy the instance.
            r_tp_tracef("vk_shutdown: dev[%d] use_count=%ld", idx, (long) dev.use_count());
        }
        if (dev && dev->device) {
            try { dev->device.waitIdle(); } catch (...) {}
        }
        try { dev.reset(); } catch (...) {}
        idx++;
    }
    // Deliberately DO NOT destroy vk_instance.instance here.
    //
    // A backend created inside pp_forward/split_mul_mat is wrapped in an R XPtr
    // whose finalizer (r_ggml_vk_backend_finalizer, onexit=TRUE) may not have run
    // yet when this explicit shutdown fires. That finalizer -> ggml_backend_free ->
    // last shared_ptr ref -> ~vk_device_struct -> device.destroy()/vkFreeMemory,
    // which needs a LIVE instance. If we destroy the instance now, that late
    // vkFreeMemory hits a dead instance => "[Vulkan Loader] vkFreeMemory: Invalid
    // device" -> Aborted. So we drain+release the instance's own device refs (kills
    // idle driver threads) but leave the instance handle valid; the OS reclaims it
    // at process exit after every late device finalizer has run against it.
    r_tp_tracef("vk_shutdown: devices reset, instance left LIVE for late finalizers");

#ifdef GGML_VK_HARD_EXIT
    if (hard) {
        // The f1ba0 segfault is a flaky race between NVIDIA/Mesa driver worker
        // threads still winding down and the dynamic loader unmapping the Vulkan
        // ICD .so files during the C runtime's atexit/static-destruction phase.
        // No R exit hook can win that race (R unmaps the loader before any hook).
        // Since all results have already been produced and printed by this point,
        // the only reliable fix is to leave via _exit(), which terminates the
        // process immediately WITHOUT running atexit handlers, C++ static
        // destructors, or unmapping shared objects — so there is no loader
        // static-destruction phase for the driver threads to fault against.
        // Callers opt in (hard=TRUE) as the last statement of a script/example.
        // `status` is the process exit code — pass non-zero from an error path so
        // a failed run does not masquerade as success.
        r_tp_tracef("vk_shutdown: hard exit(%d) — skipping atexit/loader teardown", status);
        fflush(NULL);
        _exit(status);
    }
#else
    // CRAN builds omit the _exit() path entirely (Repository Policy forbids a
    // package terminating the R session). ggml_vulkan_shutdown() warns the caller
    // that hard=TRUE degraded to the soft teardown above; see r_interface_vulkan.c.
    (void) hard;
    (void) status;
#endif
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
        // Only allowing Xe2/Xe3 GPU and integrated Xe GPUs at the moment since older hardware (ex. Arc A770) has performance regressions.
        return (arch == vk_device_architecture::INTEL_XE2) ||
            (arch == vk_device_architecture::INTEL_XE1 && props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu && driver_props.driverID == vk::DriverId::eIntelProprietaryWindows);
    case VK_VENDOR_ID_AMD:
        if (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            return arch == vk_device_architecture::AMD_RDNA3 || arch == vk_device_architecture::AMD_RDNA4;
        }
        return true;
    case VK_VENDOR_ID_QUALCOMM:
        // Only allow Adreno GPUs with hardware matrix cores (Gen 6+).
        return arch == vk_device_architecture::QUALCOMM_ADRENO;
    default:
        return true;
    }
}

// checks


// GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)
// Disabled: we statically link backends in R package, not using dynamic loading
