/* onnx_ops_special.c — special/compute-intensive ops: NonZero, RoiAlign, NMS
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ops_internal.h"

/* Returns 1 = handled, 0 = not this group's op, -1 = error */
int map_node_special(onnx_ggml_ctx_t *c, const onnx_node_t *n,
                     struct ggml_tensor *a, struct ggml_tensor *b,
                     struct ggml_tensor **out_p, int *out_nd_p)
{
    const char *op = n->op_type;
    struct ggml_tensor *out = NULL;
    int out_nd = -1;

    /* ── NonZero ────────────────────────────────────────────────── */
    if (strcmp(op, "NonZero") == 0) {
        if (!a) return -1;
        /* NonZero returns indices of non-zero elements.
         * Output shape: ONNX [n_dims_input, nnz], ggml [nnz, n_dims_input].
         * For transformer position embeddings: input is ConstantOfShape(value=1),
         * so nnz = total_elements and result = arange(0, N).
         *
         * At graph-build time we need to know nnz to allocate output tensor.
         * If input is from ConstantOfShape with non-zero fill, nnz = total_elements.
         * Otherwise, defer to runtime fill. */
        int input_ndims = tmap_get_ndims(c, n->inputs[0]);
        if (input_ndims < 1) input_ndims = (int)ggml_n_dims(a);

        int64_t total_elems = (int64_t)ggml_nelements(a);

        /* How many elements are actually non-zero cannot be known while the
         * graph is being built -- it depends on the input's VALUES.  Two cases:
         *
         *  - Segmented execution has already run the segment that computes
         *    this input, measured the count and recorded it; use that.
         *  - Otherwise (first mapping, or no segmentation) fall back to
         *    assuming every element is non-zero.  That is right for the common
         *    ConstantOfShape(value=1) case and wrong for a genuine mask, which
         *    is precisely why the segmented path exists. */
        int64_t nnz = onnx_resolved_size(c, n->outputs[0]);
        if (nnz < 0) {
            nnz = total_elems;
        } else if (onnx_trace_nodes()) {
            fprintf(stderr, "[NonZero] %s: using measured nnz=%lld (guess was %lld)\n",
                    n->outputs[0], (long long)nnz, (long long)total_elems);
        }
        /* ggml has no zero-length dimension, so an empty result still has to
         * be a one-element tensor.  That element is never filled -- there is
         * nothing to fill it with -- and reads back as 0, which downstream is
         * an ordinary row index.  Remember the emptiness instead and let the
         * consumers below drop the branch; see tmap_mark_empty(). */
        const int logically_empty = (nnz == 0);
        if (nnz < 1) nnz = 1;

        /* Output: ONNX [input_ndims, nnz] → ggml [nnz, input_ndims] */
        {
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            out = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, nnz, (int64_t)input_ndims);
        }
        if (out) {
            ggml_set_input(out);
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, 2); /* ONNX 2D output */
            if (logically_empty) tmap_mark_empty(c, n->outputs[0]);

            /* Register for deferred fill after sched alloc */
            if (c->n_nonzero_fills < ONNX_MAX_DEFERRED) {
                c->nonzero_fill_src[c->n_nonzero_fills] = a;
                c->nonzero_fill_dst[c->n_nonzero_fills] = out;
                c->nonzero_fill_ndims[c->n_nonzero_fills] = input_ndims;
                c->n_nonzero_fills++;
            }

            /* cval propagation for 1D all-nonzero case: [0, 1, 2, ..., N-1] */
            if (input_ndims == 1 && nnz <= ONNX_MAX_DIMS) {
                int64_t cvals[ONNX_MAX_DIMS];
                for (int j = 0; j < (int)nnz; j++)
                    cvals[j] = (int64_t)j;
                cval_put(c, n->outputs[0], cvals, (int)nnz);
            }
        }
        return 1; /* already registered */
    }

    /* ── RoiAlign ───────────────────────────────────────────────── */
    else if (strcmp(op, "RoiAlign") == 0) {
        /* Inputs: X [N,C,H,W], rois [num_rois,4], batch_indices [num_rois] */
        struct ggml_tensor *X    = get_input(c, n, 0);
        struct ggml_tensor *rois = get_input(c, n, 1);
        struct ggml_tensor *bi   = get_input(c, n, 2);
        if (!X || !rois) return -1;

        int oh = (int)onnx_attr_int(n, "output_height", 1);
        int ow = (int)onnx_attr_int(n, "output_width", 1);
        int sr = (int)onnx_attr_int(n, "sampling_ratio", 0);
        float ss = onnx_attr_float(n, "spatial_scale", 1.0f);
        char mode_str[16] = "avg";
        onnx_attr_str(n, "mode", mode_str, sizeof(mode_str));
        int mode = (strcmp(mode_str, "max") == 0) ? 1 : 0;

        /* X is ggml [W, H, C, N] */
        int C_feat = (int)X->ne[2];
        int num_rois_val = (int)rois->ne[1]; /* rois ggml [4, num_rois] */

        /* Allocate params (must outlive graph).
         *
         * The same lifetime trap as nms_params, and it had already bitten:
         * `p` is handed to ggml_map_custom3 as userdata and dereferenced when
         * the graph runs, so it must not move.  Growing an array of structs
         * with realloc moves the block, leaving every earlier RoiAlign node
         * pointing into freed memory -- MaskRCNN has four of them, and the
         * first one's kernel read p->X as NULL and refused.
         *
         * One allocation per entry: only the array OF POINTERS moves. */
        if (c->n_roi_aligns >= c->roi_align_params_cap) {
            int newcap = c->roi_align_params_cap ? c->roi_align_params_cap * 2 : 8;
            roi_align_params_t **np = (roi_align_params_t **)realloc(
                c->roi_align_params, (size_t)newcap * sizeof(roi_align_params_t *));
            if (!np) return -1;
            c->roi_align_params = np;
            c->roi_align_params_cap = newcap;
        }
        roi_align_params_t *p = (roi_align_params_t *)malloc(sizeof(roi_align_params_t));
        if (!p) return -1;
        c->roi_align_params[c->n_roi_aligns] = p;
        p->output_height  = oh;
        p->output_width   = ow;
        p->sampling_ratio = sr;
        p->spatial_scale  = ss;
        p->mode           = mode;

        /* Output: ggml [ow, oh, C, num_rois] */
        struct ggml_tensor *dummy = ggml_new_tensor_4d(c->ctx, GGML_TYPE_F32,
                                                        ow, oh, C_feat, num_rois_val);
        if (!bi) {
            /* Create zero batch_indices if missing */
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            bi = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, num_rois_val);
            ggml_set_input(bi);
            /* Will be zero-filled by default */
        }

        p->X = X; /* callback reads feature map from params */
        out = ggml_map_custom3(c->ctx, dummy, rois, bi,
                               roi_align_cpu, 1, p);
        /* Add X as dependency so scheduler keeps its buffer alive */
        out->src[3] = X;
        c->n_roi_aligns++;
        out_nd = 4;
    }

    /* ── NonMaxSuppression ─────────────────────────────────────── */
    else if (strcmp(op, "NonMaxSuppression") == 0) {
        /* Inputs: boxes [N,num_boxes,4], scores [N,num_classes,num_boxes],
         *         max_output_boxes_per_class (scalar), iou_threshold (scalar),
         *         score_threshold (scalar) */
        struct ggml_tensor *boxes  = get_input(c, n, 0);
        struct ggml_tensor *scores = get_input(c, n, 1);
        if (!boxes || !scores) return -1;

        int cpb = (int)onnx_attr_int(n, "center_point_box", 0);

        /* Get scalar params from raw initializer data.
         *
         * score_threshold is OPTIONAL, and "absent" is not the same as "zero".
         * ONNX applies no score filter at all when the input is missing, so a
         * candidate scoring exactly 0 stays in; with the input present and set
         * to 0 the comparison is strict and that same candidate goes out.
         * Collapsing both onto 0.0f cost MaskRCNN 217 of its 511 RPN proposals
         * on the first pyramid level alone -- quantised scores bottom out at
         * exactly 0, and every one of those was being dropped. */
        int max_boxes_val = 0;
        float iou_thresh_val = 0.0f;
        float score_thresh_val = 0.0f;
        int have_score_thresh = 0;

        /* max_output_boxes_per_class (INT64 scalar) */
        if (n->n_inputs > 2 && n->inputs[2][0]) {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[2]);
            if (mi && mi->raw_data && mi->raw_size >= 8 &&
                mi->data_type == ONNX_DTYPE_INT64) {
                int64_t v; memcpy(&v, mi->raw_data, sizeof(int64_t));
                max_boxes_val = (int)v;
            } else {
                int64_t cv[1] = {0};
                if (cval_get(c, n->inputs[2], cv, 1))
                    max_boxes_val = (int)cv[0];
            }
        }
        /* iou_threshold (FLOAT scalar) */
        if (n->n_inputs > 3 && n->inputs[3][0]) {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[3]);
            if (mi && mi->raw_data && mi->raw_size >= 4 &&
                mi->data_type == ONNX_DTYPE_FLOAT) {
                memcpy(&iou_thresh_val, mi->raw_data, sizeof(float));
            }
        }
        /* score_threshold (FLOAT scalar, optional) */
        if (n->n_inputs > 4 && n->inputs[4][0]) {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[4]);
            if (mi && mi->raw_data && mi->raw_size >= 4 &&
                mi->data_type == ONNX_DTYPE_FLOAT) {
                memcpy(&score_thresh_val, mi->raw_data, sizeof(float));
                have_score_thresh = 1;
            }
        }

        /* Allocate NMS params.
         *
         * The address of `p` is handed to ggml_map_custom3 as the kernel's
         * userdata and is dereferenced when the graph runs, long after this
         * function has returned -- so it has to stay put.  This used to be an
         * array of structs grown one element at a time with realloc, which
         * moves the block: every userdata pointer handed to an earlier NMS
         * node was left pointing into freed memory, and the kernel read it
         * anyway (the null checks in nms_cpu cannot see a dangling pointer).
         * MaskRCNN has 84 NMS nodes and died in nms_cpu with 'invalid
         * permissions'; it survived under ONNX_TRACE_LIVE only because the
         * extra reads changed the allocator's behaviour enough to leave the
         * freed block mapped.
         *
         * Each entry now gets its own allocation, so only the array OF
         * POINTERS moves when it grows, and the entries themselves never do. */
        if (c->n_nms_ops >= c->nms_params_cap) {
            int newcap = c->nms_params_cap ? c->nms_params_cap * 2 : 16;
            nms_params_t **np = (nms_params_t **)realloc(
                c->nms_params, (size_t)newcap * sizeof(nms_params_t *));
            if (!np) return -1;
            c->nms_params = np;
            c->nms_params_cap = newcap;
        }
        nms_params_t *p = (nms_params_t *)malloc(sizeof(nms_params_t));
        if (!p) return -1;
        c->nms_params[c->n_nms_ops] = p;
        p->center_point_box = cpb;
        p->scores = scores;

        /* boxes ggml [4, num_boxes, N], scores ggml [num_boxes, num_classes, N] */
        int num_boxes_val = (int)boxes->ne[1];
        int N_batch = (int)boxes->ne[2];
        int num_classes_val = (int)scores->ne[1];

        /* Max possible output: N * num_classes * max_boxes (or num_boxes if max=0) */
        int mb = max_boxes_val > 0 ? max_boxes_val : num_boxes_val;
        int max_selected = N_batch * num_classes_val * mb;
        if (max_selected > num_boxes_val * N_batch)
            max_selected = num_boxes_val * N_batch;

        /* ONNX says the output is [num_selected, 3] -- the COUNT, not the
         * capacity.  The count is data-dependent, so the first build has to
         * guess it at the upper bound above, and this node is a segment's cut
         * op precisely so the guess can be replaced: once the segment has run,
         * the kernel has written how many boxes it kept and
         * resolve_segment_sizes recorded it.  Take the measurement when it
         * exists, exactly as TopK does with its own.
         *
         * Without this the surplus slots stay at the -1 the kernel fills them
         * with to mean "empty", and the Gather that reads the selected indices
         * hands those -1s straight to get_rows: MaskRCNN kept 38 boxes out of
         * 147 and the remaining 109 slots aborted the process on an index of
         * -1.  Clamping the index would only hide it -- -1 here is not ONNX's
         * "last element", it is our own padding. */
        {
            int64_t measured = onnx_resolved_size(c, n->outputs[0]);
            if (measured > 0 && (int)measured < max_selected) {
                if (onnx_trace_nodes())
                    fprintf(stderr, "[NMS] %s: capacity %d -> measured %lld "
                                    "selected\n",
                            n->outputs[0], max_selected, (long long)measured);
                max_selected = (int)measured;
            }
        }

        /* Create params tensor [4] to pass scalar params at runtime.  Slot 3
         * carries whether score_threshold was given at all, which the kernel
         * cannot infer from the value: 0 is both a legal threshold and the
         * placeholder for "no threshold". */
        struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
        struct ggml_tensor *params_t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4);
        ggml_set_input(params_t);
        ggml_set_name(params_t, "nms_params");

        /* Register deferred fill for params tensor */
        if (c->n_nms_deferred < ONNX_MAX_DEFERRED) {
            c->nms_param_tensors[c->n_nms_deferred] = params_t;
            c->nms_max_boxes[c->n_nms_deferred] = max_boxes_val;
            c->nms_iou_thresh[c->n_nms_deferred] = iou_thresh_val;
            c->nms_score_thresh[c->n_nms_deferred] = score_thresh_val;
            c->nms_have_score_thresh[c->n_nms_deferred] = have_score_thresh;
            c->n_nms_deferred++;
        }

        /* Output: ggml [3, max_selected] */
        struct ggml_tensor *nms_dummy = ggml_new_tensor_2d(c->ctx, GGML_TYPE_F32,
                                                            3, max_selected);
        out = ggml_map_custom3(c->ctx, nms_dummy, boxes, params_t,
                               nms_cpu, 1, p);
        /* Add boxes and scores as dependencies */
        out->src[3] = boxes;
        out->src[4] = scores;
        c->n_nms_ops++;
        out_nd = 2;
    }
    else {
        return 0; /* not this group */
    }

    *out_p    = out;
    *out_nd_p = out_nd;
    return 1;
}
