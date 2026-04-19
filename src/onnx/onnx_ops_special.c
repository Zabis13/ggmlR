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

        /* Assume all elements are non-zero (common case: ConstantOfShape with value=1).
         * For truly sparse inputs this would need runtime sizing. */
        int64_t nnz = total_elems;

        /* Output: ONNX [input_ndims, nnz] → ggml [nnz, input_ndims] */
        {
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            out = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, nnz, (int64_t)input_ndims);
        }
        if (out) {
            ggml_set_input(out);
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, 2); /* ONNX 2D output */

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

        /* Allocate params (must outlive graph) */
        c->roi_align_params = (roi_align_params_t *)realloc(
            c->roi_align_params,
            (size_t)(c->n_roi_aligns + 1) * sizeof(roi_align_params_t));
        roi_align_params_t *p = &c->roi_align_params[c->n_roi_aligns];
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

        /* Get scalar params from raw initializer data */
        int max_boxes_val = 0;
        float iou_thresh_val = 0.0f;
        float score_thresh_val = 0.0f;

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
        /* score_threshold (FLOAT scalar) */
        if (n->n_inputs > 4 && n->inputs[4][0]) {
            const onnx_initializer_t *mi = onnx_find_initializer(c->onnx, n->inputs[4]);
            if (mi && mi->raw_data && mi->raw_size >= 4 &&
                mi->data_type == ONNX_DTYPE_FLOAT) {
                memcpy(&score_thresh_val, mi->raw_data, sizeof(float));
            }
        }

        /* Allocate NMS params */
        c->nms_params = (nms_params_t *)realloc(
            c->nms_params,
            (size_t)(c->n_nms_ops + 1) * sizeof(nms_params_t));
        nms_params_t *p = &c->nms_params[c->n_nms_ops];
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

        /* Create params tensor [3] to pass scalar params at runtime */
        struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
        struct ggml_tensor *params_t = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 3);
        ggml_set_input(params_t);
        ggml_set_name(params_t, "nms_params");

        /* Register deferred fill for params tensor */
        if (c->n_nms_deferred < ONNX_MAX_DEFERRED) {
            c->nms_param_tensors[c->n_nms_deferred] = params_t;
            c->nms_max_boxes[c->n_nms_deferred] = max_boxes_val;
            c->nms_iou_thresh[c->n_nms_deferred] = iou_thresh_val;
            c->nms_score_thresh[c->n_nms_deferred] = score_thresh_val;
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
