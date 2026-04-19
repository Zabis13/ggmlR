/* onnx_ops_tensor.c — tensor manipulation ops: Reshape, Transpose, Flatten,
 * Unsqueeze, Squeeze, Concat, Gather, ScatterElements, Slice, Split,
 * Resize, Expand, and misc (Identity, Constant, Cast, Shape, Pow, Erf,
 * Sin, Cos, Tile, Where, Equal, EyeLike, ConstantOfShape, Pad)
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_ops_internal.h"

/* Returns 1 = handled, 0 = not this group's op, -1 = error */
int map_node_tensor(onnx_ggml_ctx_t *c, const onnx_node_t *n,
                    struct ggml_tensor *a, struct ggml_tensor *b,
                    struct ggml_tensor **out_p, int *out_nd_p)
{
    const char *op = n->op_type;
    struct ggml_tensor *out = NULL;
    int out_nd = -1;

    /* ── Shape ops ──────────────────────────────────────────────── */
    if (strcmp(op, "Reshape") == 0) {
        if (!a || !b) return -1;
        /* b is the shape tensor — from initializer, Constant, or computed shape */
        int64_t shape[ONNX_MAX_DIMS];
        int ndims = 0;

        /* Try 1: static initializer or Constant node */
        const onnx_initializer_t *shape_init = onnx_find_initializer(c->onnx, n->inputs[1]);
        if (!shape_init)
            shape_init = find_constant_tensor(c->onnx, n->inputs[1]);
        if (shape_init) {
            if (shape_init->raw_data && shape_init->data_type == ONNX_DTYPE_INT64) {
                ndims = (int)(shape_init->raw_size / sizeof(int64_t));
                if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
                memcpy(shape, shape_init->raw_data, ndims * sizeof(int64_t));
            } else if (shape_init->decoded_data) {
                ndims = (int)(shape_init->decoded_size / sizeof(int64_t));
                if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
                memcpy(shape, shape_init->decoded_data, ndims * sizeof(int64_t));
            }
        }
        /* Try 2: compile-time value map (for dynamic Shape→Slice→Concat chains) */
        if (ndims == 0) {
            ndims = cval_get(c, n->inputs[1], shape, ONNX_MAX_DIMS);
        }
        if (ndims == 0) {
            fprintf(stderr, "[onnx] Reshape: cannot resolve shape tensor '%s'\n", n->inputs[1]);
            return -1;
        }

        /* Resolve -1 and 0 dims (in ONNX order).
         * shape[d]==0 means keep original ONNX dim d.
         * We can get ONNX dim d from a->ne[ndims_a-1-d] (reversed). */
        int64_t total = ggml_nelements(a);
        int64_t product = 1;
        int neg_idx = -1;
        int ndims_a = ggml_n_dims(a);
        for (int d = 0; d < ndims; d++) {
            if (shape[d] == 0) {
                /* Map ONNX dim d → ggml ne index (reversed) */
                int ggml_d = ndims_a - 1 - d;
                shape[d] = (ggml_d >= 0 && ggml_d < ndims_a) ? a->ne[ggml_d] : 1;
            }
            if (shape[d] == -1) neg_idx = d;
            else product *= shape[d];
        }
        if (neg_idx >= 0 && product > 0)
            shape[neg_idx] = total / product;

        /* Collapse >5D ONNX shape into 5D by merging leading ONNX dims. */
        int orig_ndims = ndims; /* save for tmap_put_nd */
        int64_t orig_shape[ONNX_MAX_DIMS];
        memcpy(orig_shape, shape, orig_ndims * sizeof(int64_t));
        if (ndims > GGML_MAX_DIMS) {
            int64_t merged = 1;
            for (int d = 0; d < ndims - (GGML_MAX_DIMS - 1); d++)
                merged *= shape[d];
            int64_t tmp[GGML_MAX_DIMS];
            tmp[0] = merged;
            for (int d = 1; d < GGML_MAX_DIMS; d++)
                tmp[d] = shape[ndims - (GGML_MAX_DIMS - 1) + d - 1];
            memcpy(shape, tmp, sizeof(tmp));
            ndims = GGML_MAX_DIMS;
        }

        /* Reverse ONNX shape → ggml ne order for reshape */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < ndims; d++)
            ne[d] = shape[ndims - 1 - d];

        /* Verify element count match */
        {
            int64_t total_out = ne_product(ne, ndims);
            if (total_out != total) {
                return -1;
            }
        }
        out = onnx_reshape_nd(c->ctx, a, ne, ndims);
        /* Register with ONNX shape for correct axis mapping downstream.
         * If >5D was collapsed, use collapsed ndims (not orig) so downstream
         * ops don't see dimensions beyond what ggml can represent. */
        int reg_ndims = (orig_ndims > GGML_MAX_DIMS) ? ndims : orig_ndims;
        int64_t *reg_shape = (orig_ndims > GGML_MAX_DIMS) ? shape : orig_shape;
        if (out) {
            for (int i = 0; i < n->n_outputs; i++) {
                if (n->outputs[i][0] != '\0') {
                    ggml_set_name(out, n->outputs[i]);
                    tmap_put_shape(c, n->outputs[i], out, reg_shape, reg_ndims);
                }
            }
            /* Propagate cval through Reshape (flat order unchanged) */
            {
                int64_t cv[ONNX_MAX_DIMS];
                int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
                if (ncv > 0)
                    cval_put(c, n->outputs[0], cv, ncv);
            }
            return 1;
        }
    }
    else if (strcmp(op, "Transpose") == 0) {
        if (!a) return -1;
        int64_t perm[ONNX_MAX_DIMS];
        int n_perm = onnx_attr_ints(n, "perm", perm, ONNX_MAX_DIMS);
        int nd = tmap_get_ndims(c, n->inputs[0]);
        if (nd <= 0) nd = ggml_n_dims(a);
        (void)0;
        /* Check for identity perm first */
        int is_identity = 0;
        if (n_perm > 0) {
            is_identity = 1;
            for (int i = 0; i < n_perm; i++) {
                if (perm[i] != i) { is_identity = 0; break; }
            }
        }
        /* If perm has more dims than tensor (Reshape collapsed leading dims),
         * remap perm to match actual tensor ndims. */
        if (n_perm > nd && nd >= 2) {
            int n_merged = n_perm - nd; /* number of leading ONNX dims collapsed into dim 0 */
            /* Build collapsed perm: remap dim references and skip merged output dims */
            int64_t cperm[ONNX_MAX_DIMS];
            int cp = 0;
            for (int i = 0; i < n_perm; i++) {
                int64_t src = perm[i];
                /* Remap source: dims 0..n_merged → 0, dims n_merged+k → k+1... no:
                 * dims 0..n_merged all map to collapsed dim 0,
                 * dim n_merged+k maps to dim k (since leading merged into 0, rest shift) */
                int64_t mapped_src;
                if (src <= n_merged) mapped_src = 0;
                else mapped_src = src - n_merged;
                /* Skip output positions that refer to within-merged reshuffling
                 * (output dims 0..n_merged that all point to merged source) */
                if (i <= n_merged && mapped_src == 0 && cp > 0 && cperm[cp-1] == 0)
                    continue; /* duplicate merged dim in output — skip */
                if (cp < ONNX_MAX_DIMS) cperm[cp++] = mapped_src;
            }
            /* If we got exactly nd entries, use collapsed perm */
            if (cp == nd) {
                memcpy(perm, cperm, nd * sizeof(int64_t));
                n_perm = nd;
            }
        }
        if (is_identity) {
            out = a;
        } else if (nd >= 5 && n_perm >= 5) {
            /* 5D Transpose via squeeze-batch → 4D permute → restore 5D.
             * Find a unit ONNX *input* dim to squeeze out temporarily. */
            int batch_in = -1; /* ONNX input dim index with size==1 */
            for (int i = 0; i < nd && i < 5; i++) {
                int ggml_d = nd - 1 - i;
                if (ggml_d >= 0 && ggml_d < GGML_MAX_DIMS && a->ne[ggml_d] == 1) {
                    batch_in = i; break;
                }
            }
            if (batch_in < 0) {
                fprintf(stderr, "[onnx] Transpose 5D: no unit dim to squeeze\n");
                batch_in = 0;
            }
            /* Find where batch_in lands in output: perm[batch_out]==batch_in */
            int batch_out = 0;
            for (int i = 0; i < n_perm; i++) {
                if (perm[i] == batch_in) { batch_out = i; break; }
            }

            /* Build 4D perm: skip batch_out in output, renumber refs past batch_in */
            int64_t perm4[4];
            int p4 = 0;
            for (int i = 0; i < n_perm; i++) {
                if (i == batch_out) continue; /* skip the output dim that receives batch */
                int64_t v = perm[i];
                if (v > batch_in) v--;  /* renumber: input dims shift down */
                else if (v == batch_in) v = 0; /* shouldn't happen (filtered above) */
                perm4[p4++] = v;
            }

            /* Build 4D ONNX input shape (squeeze out batch_in dim) */
            int64_t onnx_in4[4];
            int s4 = 0;
            for (int i = 0; i < nd && i < 5; i++) {
                if (i == batch_in) continue;
                onnx_in4[s4++] = a->ne[nd - 1 - i];
            }

            /* Reshape to 4D ggml (reverse onnx_in4) */
            int64_t ne4[4];
            for (int d = 0; d < 4; d++) ne4[d] = onnx_in4[3 - d];
            struct ggml_tensor *a4 = ggml_reshape_4d(c->ctx, a, ne4[0], ne4[1], ne4[2], ne4[3]);

            /* Compute 4D ggml permute axes from 4D ONNX perm */
            int ax[4] = {0, 1, 2, 3};
            for (int i = 0; i < 4; i++) {
                int ggml_dst = 3 - i;
                int ggml_src = 3 - (int)perm4[i];
                ax[ggml_src] = ggml_dst;
            }

            struct ggml_tensor *permuted;
            if (ax[0] == 0 && ax[1] == 1 && ax[2] == 2 && ax[3] == 3)
                permuted = a4;
            else
                permuted = ggml_cont(c->ctx, ggml_permute(c->ctx, a4, ax[0], ax[1], ax[2], ax[3]));

            /* Restore 5D: build output ONNX shape, insert batch=1 at batch_out */
            int64_t onnx_out4[4];
            for (int i = 0; i < 4; i++) onnx_out4[i] = onnx_in4[perm4[i]];

            int64_t onnx_out5[5];
            s4 = 0;
            for (int i = 0; i < 5; i++) {
                if (i == batch_out) onnx_out5[i] = 1;
                else onnx_out5[i] = onnx_out4[s4++];
            }

            /* Reverse to ggml ne order */
            int64_t ne5[5];
            for (int d = 0; d < 5; d++) ne5[d] = onnx_out5[4 - d];
            /* Use onnx_reshape_nd to squeeze trailing 1s (ggml max 4D ops) */
            out = onnx_reshape_nd(c->ctx, permuted, ne5, 5);
            out_nd = 5;
        } else if (n_perm == 0 || nd == 2) {
            /* Default: reverse all dims → standard transpose */
            out = ggml_cont(c->ctx, ggml_transpose(c->ctx, a));
        } else {
            /* 4D (or less) Transpose via ggml_permute.
             * Convert ONNX perm to ggml permute axes.
             * ONNX perm[i]=j means output ONNX dim i ← input ONNX dim j.
             * ggml ax[k] means output ggml dim k ← input ggml dim ax[k].
             * ONNX dim i → ggml dim (reversed). */
            int onnx_nd = n_perm > nd ? n_perm : nd;
            if (n->n_inputs > 0) {
                int in_nd = tmap_get_ndims(c, n->inputs[0]);
                if (in_nd > onnx_nd) onnx_nd = in_nd;
            }

            int ax[4] = {0, 1, 2, 3};
            for (int i = 0; i < n_perm; i++) {
                int ggml_dst = onnx_nd - 1 - i;
                int ggml_src = onnx_nd - 1 - (int)perm[i];
                if (ggml_dst < 0) ggml_dst = 0;
                if (ggml_dst > 3) ggml_dst = 3;
                if (ggml_src < 0) ggml_src = 0;
                if (ggml_src > 3) ggml_src = 3;
                ax[ggml_src] = ggml_dst;
            }
            if (ax[0] == 0 && ax[1] == 1 && ax[2] == 2 && ax[3] == 3) {
                out = a;
            } else {
                out = ggml_cont(c->ctx, ggml_permute(c->ctx, a, ax[0], ax[1], ax[2], ax[3]));
            }
        }
        /* Propagate cval through Transpose (permute flat elements).
         * cval stores values in ggml flat order (col-major: dim0 fastest).
         * Transpose swaps dims, so we need to remap element indices. */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0 && nd <= 4) {
                /* Build effective perm — default (n_perm==0) reverses all dims */
                int64_t eff_perm[4];
                int eff_n = nd;
                if (n_perm > 0) {
                    for (int d = 0; d < nd; d++)
                        eff_perm[d] = (d < n_perm) ? perm[d] : d;
                } else {
                    for (int d = 0; d < nd; d++)
                        eff_perm[d] = nd - 1 - d;
                }

                /* Input/output shapes in ONNX order */
                int64_t shape_in[4] = {1,1,1,1};
                for (int d = 0; d < nd; d++) shape_in[d] = a->ne[nd-1-d];
                int64_t shape_out[4] = {1,1,1,1};
                for (int d = 0; d < eff_n; d++)
                    shape_out[d] = shape_in[eff_perm[d]];

                /* Output ggml ne (reversed from ONNX) */
                int64_t out_ne0[4] = {1,1,1,1};
                for (int d = 0; d < nd; d++)
                    out_ne0[d] = shape_out[nd-1-d];

                int64_t result[ONNX_MAX_DIMS];
                int ncv_out = 1;
                for (int d = 0; d < nd; d++) ncv_out *= (int)shape_out[d];
                if (ncv_out == ncv && ncv_out <= ONNX_MAX_DIMS) {
                    /* Iterate output in ggml flat order (dim0 fastest) */
                    for (int oi = 0; oi < ncv_out; oi++) {
                        /* Decompose oi into ggml output indices [g0,g1,g2,g3] */
                        int gi[4] = {0};
                        int tmp = oi;
                        for (int d = 0; d < nd; d++) {
                            gi[d] = tmp % (int)out_ne0[d];
                            tmp /= (int)out_ne0[d];
                        }
                        /* Convert ggml output → ONNX output indices */
                        int onnx_out[4] = {0};
                        for (int d = 0; d < nd; d++)
                            onnx_out[d] = gi[nd-1-d];
                        /* Map ONNX output → ONNX input via inverse perm */
                        int onnx_in[4] = {0};
                        for (int d = 0; d < eff_n; d++)
                            onnx_in[(int)eff_perm[d]] = onnx_out[d];
                        /* Convert ONNX input → ggml input indices */
                        int gi_in[4] = {0};
                        for (int d = 0; d < nd; d++)
                            gi_in[d] = onnx_in[nd-1-d];
                        /* Compute flat ggml input index */
                        int ii = 0;
                        for (int d = nd-1; d >= 0; d--)
                            ii = ii * (int)a->ne[d] + gi_in[d];
                        result[oi] = cv[ii];
                    }
                    cval_put(c, n->outputs[0], result, ncv_out);
                }
            }
        }
        out_nd = nd;  /* Transpose preserves ndims */
    }
    else if (strcmp(op, "Flatten") == 0) {
        if (!a) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 1);
        int nd_a = tmap_get_ndims(c, n->inputs[0]);
        if (nd_a <= 0) nd_a = ggml_n_dims(a);
        if (axis < 0) axis += nd_a;

        /* ONNX Flatten: output shape = [product(dims[:axis]), product(dims[axis:])] */
        if (axis == 0) {
            /* Special case: [1, total] */
            int64_t total = ggml_nelements(a);
            out = ggml_reshape_2d(c->ctx, a, total, 1);
        } else if (axis >= nd_a) {
            /* All dims go to first part: [total, 1] */
            int64_t total = ggml_nelements(a);
            out = ggml_reshape_2d(c->ctx, a, 1, total);
        } else {
            /* Compute product of ONNX dims [axis:] → ggml dims [0..nd_a-1-axis]
             * and ONNX dims [:axis] → ggml dims [nd_a-axis..nd_a-1] */
            int64_t inner = 1, outer = 1;
            for (int d = 0; d < nd_a; d++) {
                int onnx_d = nd_a - 1 - d;
                if (onnx_d >= axis) inner *= a->ne[d];
                else outer *= a->ne[d];
            }
            /* ggml [inner, outer] = ONNX [outer, inner] */
            out = ggml_reshape_2d(c->ctx, a, inner, outer);
        }
        /* Propagate cval through Flatten (flat order unchanged in ggml) */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0)
                cval_put(c, n->outputs[0], cv, ncv);
        }
        /* Register with ONNX ndims=2 */
        if (out) {
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, 2);
            return 1;
        }
    }
    else if (strcmp(op, "Unsqueeze") == 0) {
        if (!a) return -1;
        /* Get axes from attribute (opset < 13) or second input (opset >= 13) */
        int64_t axes[ONNX_MAX_DIMS];
        int n_axes = onnx_attr_ints(n, "axes", axes, ONNX_MAX_DIMS);
        if (n_axes == 0 && n->n_inputs > 1) {
            const onnx_initializer_t *axes_init = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (axes_init && axes_init->raw_data && axes_init->data_type == ONNX_DTYPE_INT64) {
                n_axes = (int)(axes_init->raw_size / sizeof(int64_t));
                if (n_axes > ONNX_MAX_DIMS) n_axes = ONNX_MAX_DIMS;
                memcpy(axes, axes_init->raw_data, n_axes * sizeof(int64_t));
            }
            /* Fallback: try cval (from Constant nodes) */
            if (n_axes == 0) {
                n_axes = cval_get(c, n->inputs[1], axes, ONNX_MAX_DIMS);
            }
        }

        /* Use stored ONNX ndims (ggml_n_dims drops trailing 1s) */
        int nd_in = tmap_get_ndims(c, n->inputs[0]);
        if (nd_in <= 0) {
            nd_in = GGML_MAX_DIMS;
            while (nd_in > 1 && a->ne[nd_in - 1] == 1) nd_in--;
        }
        /* Check if ONNX input had more dims (trailing 1s) by looking at
         * total expected output ndims vs axes. If max axis >= nd_in + n_axes,
         * we need more input dims. */
        int nd_out = nd_in + n_axes;
        if (nd_out > GGML_MAX_DIMS) nd_out = GGML_MAX_DIMS;

        /* Normalize negative axes (relative to output ndims) */
        for (int i = 0; i < n_axes; i++)
            if (axes[i] < 0) axes[i] += nd_out;

        /* Build input ONNX dims from ggml ne (reversed) */
        int64_t onnx_in[ONNX_MAX_DIMS];
        for (int i = 0; i < nd_in; i++)
            onnx_in[i] = a->ne[nd_in - 1 - i];

        /* Build output ONNX shape by inserting 1s at axes positions */
        int64_t onnx_out[ONNX_MAX_DIMS];
        int in_idx = 0;
        for (int o = 0; o < nd_out; o++) {
            int is_new = 0;
            for (int j = 0; j < n_axes; j++)
                if (axes[j] == o) { is_new = 1; break; }
            if (is_new)
                onnx_out[o] = 1;
            else
                onnx_out[o] = (in_idx < nd_in) ? onnx_in[in_idx++] : 1;
        }

        /* Reverse to ggml ne order */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < nd_out && d < GGML_MAX_DIMS; d++)
            ne[d] = onnx_out[nd_out - 1 - d];

        out = onnx_reshape_nd(c->ctx, a, ne, nd_out);
        /* Preserve ONNX ndims only for real data tensors.
         * For scalar-like tensors (all dims==1), let squeeze determine ndims
         * so shape-tensor chains (Unsqueeze→Concat→Reshape) work correctly. */
        {
            int has_nonunit = 0;
            for (int d = 0; d < nd_out; d++)
                if (ne[d] > 1) { has_nonunit = 1; break; }
            if (has_nonunit) out_nd = nd_out;
        }

        /* cval propagation: Unsqueeze preserves values */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0)
                cval_put(c, n->outputs[0], cv, ncv);
        }
    }
    else if (strcmp(op, "Squeeze") == 0) {
        if (!a) return -1;
        /* Get axes from attribute (opset < 13) or second input (opset >= 13) */
        int64_t axes[ONNX_MAX_DIMS];
        int n_axes = onnx_attr_ints(n, "axes", axes, ONNX_MAX_DIMS);
        if (n_axes == 0 && n->n_inputs > 1) {
            const onnx_initializer_t *axes_init = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (axes_init && axes_init->raw_data && axes_init->data_type == ONNX_DTYPE_INT64) {
                n_axes = (int)(axes_init->raw_size / sizeof(int64_t));
                if (n_axes > ONNX_MAX_DIMS) n_axes = ONNX_MAX_DIMS;
                memcpy(axes, axes_init->raw_data, n_axes * sizeof(int64_t));
            }
            /* Fallback: try cval (from Constant nodes) */
            if (n_axes == 0) {
                n_axes = cval_get(c, n->inputs[1], axes, ONNX_MAX_DIMS);
            }
        }

        /* Use stored ONNX ndims when available, fallback to 4 */
        int nd_in = tmap_get_ndims(c, n->inputs[0]);
        if (nd_in <= 0) {
            nd_in = GGML_MAX_DIMS;
            while (nd_in > 1 && a->ne[nd_in - 1] == 1) nd_in--;
        }
        if (nd_in > GGML_MAX_DIMS) nd_in = GGML_MAX_DIMS;

        /* Normalize negative axes */
        for (int i = 0; i < n_axes; i++)
            if (axes[i] < 0) axes[i] += nd_in;

        /* Build ONNX dims from ggml ne (reversed, using nd_in) */
        int64_t onnx_in[ONNX_MAX_DIMS];
        for (int i = 0; i < ONNX_MAX_DIMS; i++) onnx_in[i] = 1;
        for (int i = 0; i < nd_in; i++)
            onnx_in[i] = a->ne[nd_in - 1 - i];

        int64_t onnx_out[ONNX_MAX_DIMS];
        int nd_out = 0;
        for (int i = 0; i < nd_in; i++) {
            int squeeze = 0;
            if (n_axes == 0) {
                squeeze = (onnx_in[i] == 1);
            } else {
                for (int j = 0; j < n_axes; j++)
                    if (axes[j] == i) { squeeze = 1; break; }
            }
            if (!squeeze)
                onnx_out[nd_out++] = onnx_in[i];
        }
        if (nd_out == 0) { nd_out = 1; onnx_out[0] = 1; }

        /* Reverse to ggml ne order */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < nd_out && d < GGML_MAX_DIMS; d++)
            ne[d] = onnx_out[nd_out - 1 - d];

        out = onnx_reshape_nd(c->ctx, a, ne, nd_out);
        {
            int has_nonunit = 0;
            for (int d = 0; d < nd_out; d++)
                if (ne[d] > 1) { has_nonunit = 1; break; }
            if (has_nonunit) out_nd = nd_out;
        }
    }

    /* ── Concat ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Concat") == 0) {
        if (!a) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        /* Determine effective ONNX ndims for axis mapping.
         * Use tmap stored ndims (from value_info or previous ops),
         * falling back to ggml_n_dims. */
        int eff_nd = ggml_n_dims(a);
        for (int i = 0; i < n->n_inputs; i++) {
            int stored = tmap_get_ndims(c, n->inputs[i]);
            if (stored > eff_nd) eff_nd = stored;
            struct ggml_tensor *ti = get_input(c, n, i);
            if (ti) {
                int ndi = ggml_n_dims(ti);
                if (ndi > eff_nd) eff_nd = ndi;
            }
        }
        if (eff_nd < 1) eff_nd = 1;
        int nd = eff_nd;
        int onnx_axis = (int)axis;
        if (onnx_axis < 0) onnx_axis = eff_nd + onnx_axis;
        int dim = eff_nd - 1 - onnx_axis;
        if (dim < 0) dim = 0;
        if (dim > GGML_MAX_DIMS - 1) dim = GGML_MAX_DIMS - 1;
        /* Concat supports N inputs — chain pairwise */
        out = a;
        for (int i = 1; i < n->n_inputs; i++) {
            struct ggml_tensor *inp = get_input(c, n, i);
            if (!inp) continue;
            /* ggml_concat requires matching types — cast if needed */
            if (out->type != inp->type) {
                if (inp->type == GGML_TYPE_I32 && out->type == GGML_TYPE_F32)
                    inp = ggml_cast(c->ctx, inp, GGML_TYPE_F32);
                else if (out->type == GGML_TYPE_I32 && inp->type == GGML_TYPE_F32) {
                    out = ggml_cast(c->ctx, out, GGML_TYPE_F32);
                }
            }
            out = ggml_concat(c->ctx, out, inp, dim);
        }
        out_nd = nd; /* preserve ONNX ndims for correct axis mapping downstream */

        /* Propagate compile-time values for 1D Concat (shape tensor concatenation) */
        if (onnx_axis == 0 && nd == 1) {
            int64_t merged[ONNX_MAX_DIMS];
            int total = 0;
            int all_known = 1;
            for (int i = 0; i < n->n_inputs; i++) {
                int64_t part[ONNX_MAX_DIMS];
                int np = cval_get(c, n->inputs[i], part, ONNX_MAX_DIMS);
                if (np == 0) { all_known = 0; break; }
                for (int j = 0; j < np && total < ONNX_MAX_DIMS; j++)
                    merged[total++] = part[j];
            }
            if (all_known && total > 0)
                cval_put(c, n->outputs[0], merged, total);
        }
    }

    /* ── Gather ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Gather") == 0) {
        if (!a || !b) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        int b_nd = tmap_get_ndims(c, n->inputs[1]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);
        if (b_nd <= 0) b_nd = (int)ggml_n_dims(b);

        /* Case 1: scalar/shape indexing — data is small 1D (shape tensor,
         * constants) and both have cval → create scalar constant.
         * This avoids ggml_get_rows which is designed for embedding lookup. */
        int64_t cv_data[ONNX_MAX_DIMS], cv_idx[ONNX_MAX_DIMS];
        int ncv_data = cval_get(c, n->inputs[0], cv_data, ONNX_MAX_DIMS);
        int ncv_idx  = cval_get(c, n->inputs[1], cv_idx, ONNX_MAX_DIMS);
        if (ncv_data > 0 && ncv_idx > 0) {
            /* Both compile-time known: resolve at graph-build time */
            int64_t result[ONNX_MAX_DIMS];
            int nr = 0;
            for (int j = 0; j < ncv_idx; j++) {
                int64_t idx = cv_idx[j];
                if (idx < 0) idx += ncv_data;
                if (idx >= 0 && idx < ncv_data && nr < ONNX_MAX_DIMS)
                    result[nr++] = cv_data[idx];
            }
            /* Create scalar/small constant tensor in ctx_weight */
            struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
            if (nr == 0) nr = 1;
            out = ggml_new_tensor_1d(wctx, a->type, nr);
            if (out) {
                ggml_set_input(out);
                ggml_set_name(out, n->outputs[0]);
                tmap_put_nd(c, n->outputs[0], out, nr > 1 ? 1 : 1);
                cval_put(c, n->outputs[0], result, nr);
                /* Register for deferred fill (const_fill can't handle multi-value,
                 * so stash as shape tensor) */
                if (c->n_shape_tensors < ONNX_MAX_DEFERRED) {
                    c->shape_tensors_ne[c->n_shape_tensors][0] = nr;
                    for (int j = 0; j < nr; j++)
                        c->shape_tensors_ne[c->n_shape_tensors][j + 1] = result[j];
                    c->shape_tensor_ptrs[c->n_shape_tensors] = out;
                    c->n_shape_tensors++;
                }
            }
            return 1;
        }

        /* Case 2: embedding lookup — ggml_get_rows */
        if (b->type != GGML_TYPE_I32) {
            b = ggml_cast(c->ctx, b, GGML_TYPE_I32);
        }

        /* Normalize negative axis */
        if (axis < 0) axis += a_nd;

        if (axis == 0 && a_nd > 2) {
            /* Gather axis=0 on rank>2: ggml_get_rows only handles 2D.
             * ONNX axis=0 selects along the first ONNX dim = last ggml dim.
             * Flatten all dims except the last into row_size, gather, reshape back. */
            int g_nd = ggml_n_dims(a);
            if (g_nd < 2) g_nd = 2;
            int64_t row_size = 1;
            for (int d = 0; d < g_nd - 1; d++)
                row_size *= a->ne[d];
            struct ggml_tensor *a2d = ggml_reshape_2d(c->ctx, a, row_size, a->ne[g_nd - 1]);
            struct ggml_tensor *gathered = ggml_get_rows(c->ctx, a2d, b);
            /* Reshape back: replace the gathered dim with n_indices */
            int64_t n_idx = (int64_t)ggml_nelements(b);
            int64_t back_ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
            for (int d = 0; d < g_nd - 1; d++)
                back_ne[d] = a->ne[d];
            back_ne[g_nd - 1] = n_idx;
            out = onnx_reshape_nd(c->ctx, gathered, back_ne, g_nd);
        } else {
            out = ggml_get_rows(c->ctx, a, b);
        }

        /* Register output with correct ONNX ndims.
         * ONNX Gather(data, indices, axis=0):
         * output_shape = indices_shape + data_shape[1:]
         * E.g. data [V,D] (2D) + indices [B,S] (2D) → [B,S,D] (3D). */
        if (out) {
            int out_nd = b_nd + (a_nd > 1 ? a_nd - 1 : 0);
            if (out_nd < 1) out_nd = 1;
            if (out_nd > 4) out_nd = 4;
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, out_nd);
        }
        return 1; /* already registered */
    }

    /* ── ScatterElements ────────────────────────────────────────── */
    else if (strcmp(op, "ScatterElements") == 0) {
        if (!a || !b) return -1;
        /* ONNX ScatterElements(data, indices, updates, axis=0, reduction)
         * data:    [D0, D1, ...] — base tensor
         * indices: [I0, I1, ...] — index tensor (same shape as updates)
         * updates: [I0, I1, ...] — values to scatter
         * Input mapping: a=data, b=indices, c_upd=updates (input[2]) */
        struct ggml_tensor *c_upd = NULL;
        if (n->n_inputs > 2 && n->inputs[2][0] != '\0')
            c_upd = tmap_get(c, n->inputs[2]);
        if (!c_upd) { fprintf(stderr, "[onnx] ScatterElements: missing updates\n"); return -1; }

        int64_t axis = onnx_attr_int(n, "axis", 0);
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);
        if (axis < 0) axis += a_nd;

        /* ONNX axis → ggml dim (reversed) */
        int ggml_axis = a_nd - 1 - (int)axis;
        if (ggml_axis < 0) ggml_axis = 0;

        /* Determine reduction: 0=none, 1=add */
        int reduction = 0;
        {
            char red_str[64] = {0};
            int red_len = onnx_attr_str(n, "reduction", red_str, sizeof(red_str));
            if (red_len > 0) {
                if (strcmp(red_str, "add") == 0) reduction = 1;
                else if (strcmp(red_str, "none") == 0) reduction = 0;
                else {
                    fprintf(stderr, "[onnx] ScatterElements: reduction='%s' not supported\n", red_str);
                    return -1;
                }
            }
        }

        /* Cast indices to I32 if needed */
        if (b->type != GGML_TYPE_I32) {
            b = ggml_cast(c->ctx, b, GGML_TYPE_I32);
        }
        /* Cast updates to F32 if needed */
        if (c_upd->type != GGML_TYPE_F32) {
            c_upd = ggml_cast(c->ctx, c_upd, GGML_TYPE_F32);
        }
        /* Cast data to F32 if needed */
        if (a->type != GGML_TYPE_F32) {
            a = ggml_cast(c->ctx, a, GGML_TYPE_F32);
        }

        out = ggml_scatter_elements(c->ctx, a, c_upd, b, reduction, ggml_axis);
        if (out) {
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, a_nd);
        }
        return 1;
    }

    /* ── Slice ──────────────────────────────────────────────────── */
    else if (strcmp(op, "Slice") == 0) {
        if (!a) return -1;
        /* Inputs: data, starts, ends, [axes], [steps] — all from initializers */
        int64_t starts[GGML_MAX_DIMS] = {0}, ends[GGML_MAX_DIMS] = {0};
        int64_t axes_arr[GGML_MAX_DIMS] = {0, 1, 2, 3, 4}, steps[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        int n_slices = 0;
        int has_axes = 0;

        /* Helper macro: read int64 values from initializer, Constant, or cval */
        #define READ_SLICE_INPUT(idx, dst, cnt) do { \
            if (n->n_inputs > (idx) && n->inputs[idx][0] != '\0') { \
                const onnx_initializer_t *_si = onnx_find_initializer(c->onnx, n->inputs[idx]); \
                if (!_si) _si = find_constant_tensor(c->onnx, n->inputs[idx]); \
                if (_si && _si->raw_data && _si->data_type == ONNX_DTYPE_INT64) { \
                    int _n = (int)(_si->raw_size / sizeof(int64_t)); \
                    if (_n > GGML_MAX_DIMS) _n = GGML_MAX_DIMS; \
                    memcpy(dst, _si->raw_data, _n * sizeof(int64_t)); \
                    cnt = _n; \
                } else { \
                    int _n = cval_get(c, n->inputs[idx], dst, GGML_MAX_DIMS); \
                    if (_n > 0) cnt = _n; \
                } \
            } \
        } while(0)

        /* Read starts (input 1) */
        READ_SLICE_INPUT(1, starts, n_slices);
        /* Read ends (input 2) */
        { int _d_ends = 0; READ_SLICE_INPUT(2, ends, _d_ends); (void)_d_ends; }
        /* Read axes (input 3, optional) */
        if (n->n_inputs > 3 && n->inputs[3][0] != '\0') {
            int _d_axes = 0;
            READ_SLICE_INPUT(3, axes_arr, _d_axes);
            (void)_d_axes;
            has_axes = 1;
        }
        /* Read steps (input 4, optional) */
        if (n->n_inputs > 4 && n->inputs[4][0] != '\0') {
            int _dummy_steps = 0;
            READ_SLICE_INPUT(4, steps, _dummy_steps);
            (void)_dummy_steps;
        }
        #undef READ_SLICE_INPUT

        /* Determine ONNX ndims — prefer stored ndims over ggml_n_dims
         * (ggml_n_dims ignores trailing 1-dims, e.g. [24,128,4,1] → 3 not 4) */
        int nd_onnx = tmap_get_ndims(c, n->inputs[0]);
        if (nd_onnx <= 0) nd_onnx = ggml_n_dims(a);
        if (nd_onnx < 1) nd_onnx = 1;

        /* Convert ONNX axes to ggml dims and compute view params.
         * ONNX dim d → ggml dim (nd_onnx-1-d).
         * We build offset, output ne[], and normalized starts/steps in ggml order. */
        int64_t out_ne[GGML_MAX_DIMS], offsets[GGML_MAX_DIMS] = {0};
        int64_t norm_starts[GGML_MAX_DIMS] = {0}, norm_steps[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < GGML_MAX_DIMS; d++)
            out_ne[d] = a->ne[d];

        for (int i = 0; i < n_slices; i++) {
            int onnx_ax = has_axes ? (int)axes_arr[i] : i;
            if (onnx_ax < 0) onnx_ax += nd_onnx;
            int ggml_d = nd_onnx - 1 - onnx_ax;
            if (ggml_d < 0 || ggml_d > GGML_MAX_DIMS - 1) continue;

            int64_t dim_size = a->ne[ggml_d];
            int64_t s = starts[i], e = ends[i], st = steps[i];
            if (st == 0) st = 1; /* safety */

            /* Normalize negative indices */
            if (s < 0) s += dim_size;
            if (e < 0) e += dim_size;

            /* Clamp per ONNX spec (different for positive vs negative step) */
            if (st > 0) {
                if (s < 0) s = 0;
                if (s > dim_size) s = dim_size;
                if (e < 0) e = 0;
                if (e > dim_size) e = dim_size;
            } else {
                /* step < 0: clamp to [-1, dim_size-1] */
                if (s < -1) s = -1;
                if (s > dim_size - 1) s = dim_size - 1;
                if (e < -1) e = -1;
                if (e > dim_size - 1) e = dim_size - 1;
            }

            int64_t len;
            if (st > 0) {
                len = (e - s + st - 1) / st; /* ceil((e-s)/st) */
            } else {
                /* step < 0: len = ceil((s - e) / |st|) — e.g. s=2,e=-1,st=-1 → len=3 */
                len = (s - e + (-st) - 1) / (-st);
            }
            if (len < 0) len = 0;

            offsets[ggml_d] = (st > 0) ? s : 0; /* for step>0 view; step<0 uses deferred */
            norm_starts[ggml_d] = s;
            norm_steps[ggml_d] = st;
            out_ne[ggml_d] = len;
        }

        /* Only step=1 supported via ggml_view — check */
        int all_step1 = 1;
        for (int i = 0; i < n_slices; i++)
            if (steps[i] != 1) { all_step1 = 0; break; }

        if (all_step1) {
            size_t offset_bytes = 0;
            for (int d = 0; d < GGML_MAX_DIMS; d++)
                offset_bytes += offsets[d] * a->nb[d];
            /* Use ONNX ndims for view dimension (not ggml_n_dims which drops trailing 1s) */
            int nd = nd_onnx;
            if (nd > GGML_MAX_DIMS) nd = GGML_MAX_DIMS;

            switch (nd) {
                case 1:
                    out = ggml_view_1d(c->ctx, a, out_ne[0], offset_bytes);
                    break;
                case 2:
                    out = ggml_view_2d(c->ctx, a, out_ne[0], out_ne[1],
                                       a->nb[1], offset_bytes);
                    break;
                case 3:
                    out = ggml_view_3d(c->ctx, a, out_ne[0], out_ne[1], out_ne[2],
                                       a->nb[1], a->nb[2], offset_bytes);
                    break;
                case 4:
                    out = ggml_view_4d(c->ctx, a, out_ne[0], out_ne[1],
                                       out_ne[2], out_ne[3],
                                       a->nb[1], a->nb[2], a->nb[3],
                                       offset_bytes);
                    break;
                default:
                    out = ggml_view_5d(c->ctx, a, out_ne[0], out_ne[1],
                                       out_ne[2], out_ne[3], out_ne[4],
                                       a->nb[1], a->nb[2], a->nb[3], a->nb[4],
                                       offset_bytes);
                    break;
            }
            /* Make contiguous so downstream ops work correctly */
            out = ggml_cont(c->ctx, out);
        } else {
            /* step != 1: deferred strided copy after alloc */
            if (c->n_slice_fills >= ONNX_MAX_DEFERRED) {
                fprintf(stderr, "onnx_ggml: too many strided Slice ops (max 64)\n");
                return -1;
            }
            /* Create output tensor with correct shape */
            int nd = nd_onnx;
            if (nd > GGML_MAX_DIMS) nd = GGML_MAX_DIMS;
            out = onnx_new_tensor_nd(c->ctx, a->type, out_ne, nd);
            ggml_set_input(out);

            int sf = c->n_slice_fills;
            c->slice_fill_src[sf] = a;
            c->slice_fill_dst[sf] = out;
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                c->slice_fill_starts[sf][d] = norm_starts[d];
                c->slice_fill_steps[sf][d]  = norm_steps[d];
                c->slice_fill_out_ne[sf][d] = out_ne[d];
            }
            c->slice_fill_ndims[sf] = nd_onnx;
            c->n_slice_fills++;
        }

        /* Propagate compile-time values through Slice (for shape tensors).
         * Supports strided/reverse slicing and multi-dim tensors. */
        {
            int64_t src_vals[ONNX_MAX_DIMS];
            int nv = cval_get(c, n->inputs[0], src_vals, ONNX_MAX_DIMS);
            if (nv > 0 && n_slices > 0) {
                /* Total output elements */
                int64_t total_out = ne_product(out_ne, GGML_MAX_DIMS);

                if (total_out > 0 && total_out <= ONNX_MAX_DIMS) {
                    int64_t result[ONNX_MAX_DIMS];
                    /* Compute strides for source and output in ggml order */
                    int64_t src_stride[GGML_MAX_DIMS], out_stride[GGML_MAX_DIMS];
                    src_stride[0] = 1; out_stride[0] = 1;
                    for (int d = 1; d < GGML_MAX_DIMS; d++) {
                        src_stride[d] = src_stride[d-1] * a->ne[d-1];
                        out_stride[d] = out_stride[d-1] * out_ne[d-1];
                    }
                    /* Iterate all output elements via flat index */
                    for (int64_t di = 0; di < total_out; di++) {
                        int64_t si = 0;
                        int64_t rem = di;
                        for (int d = GGML_MAX_DIMS - 1; d >= 0; d--) {
                            int64_t coord = rem / out_stride[d];
                            rem -= coord * out_stride[d];
                            si += (norm_starts[d] + coord * norm_steps[d]) * src_stride[d];
                        }
                        if ((size_t)si < (size_t)nv)
                            result[di] = src_vals[si];
                    }
                    cval_put(c, n->outputs[0], result, (int)total_out);
                }
            }
        }
    }

    /* ── Split ──────────────────────────────────────────────────── */
    else if (strcmp(op, "Split") == 0) {
        if (!a) return -1;
        int64_t axis = onnx_attr_int(n, "axis", 0);
        /* Use original ONNX ndims for >4D axis mapping */
        int nd = ggml_n_dims(a);
        if (n->n_inputs > 0) {
            int in_nd = tmap_get_ndims(c, n->inputs[0]);
            if (in_nd > nd) nd = in_nd;
        }
        if (nd < 1) nd = 1;
        if (axis < 0) axis += nd;
        int ggml_d = nd - 1 - (int)axis;
        if (ggml_d < 0) ggml_d = 0;
        if (ggml_d > GGML_MAX_DIMS - 1) ggml_d = GGML_MAX_DIMS - 1;

        int64_t dim_size = a->ne[ggml_d];
        int n_out = n->n_outputs;

        /* Get split sizes: from input 1 (opset 13+) or attribute */
        int64_t splits[ONNX_MAX_OUTPUTS];
        int n_splits = onnx_attr_ints(n, "split", splits, ONNX_MAX_OUTPUTS);

        if (n_splits == 0 && n->n_inputs > 1 && n->inputs[1][0] != '\0') {
            const onnx_initializer_t *si = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (si && si->raw_data && si->data_type == ONNX_DTYPE_INT64) {
                n_splits = (int)(si->raw_size / sizeof(int64_t));
                if (n_splits > ONNX_MAX_OUTPUTS) n_splits = ONNX_MAX_OUTPUTS;
                memcpy(splits, si->raw_data, n_splits * sizeof(int64_t));
            }
        }

        /* Default: equal split */
        if (n_splits == 0 && n_out > 0) {
            int64_t chunk = dim_size / n_out;
            for (int i = 0; i < n_out; i++)
                splits[i] = chunk;
            splits[n_out - 1] = dim_size - chunk * (n_out - 1);
            n_splits = n_out;
        }

        /* Create view for each split output */
        int64_t offset = 0;
        for (int i = 0; i < n_splits && i < n_out; i++) {
            int64_t out_ne[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; d++)
                out_ne[d] = a->ne[d];
            out_ne[ggml_d] = splits[i];

            size_t offset_bytes = offset * a->nb[ggml_d];
            int vnd = GGML_MAX_DIMS;
            while (vnd > 1 && out_ne[vnd-1] == 1) vnd--;

            struct ggml_tensor *view;
            switch (vnd) {
                case 1:
                    view = ggml_view_1d(c->ctx, a, out_ne[0], offset_bytes);
                    break;
                case 2:
                    view = ggml_view_2d(c->ctx, a, out_ne[0], out_ne[1],
                                        a->nb[1], offset_bytes);
                    break;
                case 3:
                    view = ggml_view_3d(c->ctx, a, out_ne[0], out_ne[1], out_ne[2],
                                        a->nb[1], a->nb[2], offset_bytes);
                    break;
                case 4:
                    view = ggml_view_4d(c->ctx, a, out_ne[0], out_ne[1],
                                        out_ne[2], out_ne[3],
                                        a->nb[1], a->nb[2], a->nb[3],
                                        offset_bytes);
                    break;
                default:
                    view = ggml_view_5d(c->ctx, a, out_ne[0], out_ne[1],
                                        out_ne[2], out_ne[3], out_ne[4],
                                        a->nb[1], a->nb[2], a->nb[3], a->nb[4],
                                        offset_bytes);
                    break;
            }

            if (n->outputs[i][0] != '\0') {
                /* Last split output: dup to prevent gallocr buffer aliasing */
                struct ggml_tensor *out_tensor;
                if (i == n_splits - 1) {
                    out_tensor = ggml_dup(c->ctx, view);
                } else {
                    out_tensor = view;
                }
                ggml_set_name(out_tensor, n->outputs[i]);
                tmap_put_nd(c, n->outputs[i], out_tensor, nd);
            }
            offset += splits[i];
        }
        return 1; /* outputs already registered */
    }

    /* ── Resize / Upsample ──────────────────────────────────────── */
    else if (strcmp(op, "Resize") == 0 || strcmp(op, "Upsample") == 0) {
        if (!a) return -1;

        /* mode attribute */
        char mode_str[32] = "nearest";
        onnx_attr_str(n, "mode", mode_str, sizeof(mode_str));
        enum ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST;
        if (strcmp(mode_str, "linear") == 0 || strcmp(mode_str, "bilinear") == 0)
            mode = GGML_SCALE_MODE_BILINEAR;

        /* Target sizes: from "sizes" input (input 3) or "scales" input (input 2).
         * Resize: inputs = [X, roi, scales, sizes]
         * Upsample: inputs = [X, scales] */
        int64_t target_ne[4];
        for (int d = 0; d < 4; d++)
            target_ne[d] = a->ne[d];

        int got_target = 0;

        /* Try sizes input (Resize input 3) */
        if (n->n_inputs > 3 && n->inputs[3][0] != '\0') {
            const onnx_initializer_t *si = onnx_find_initializer(c->onnx, n->inputs[3]);
            if (si && si->raw_data && si->data_type == ONNX_DTYPE_INT64) {
                int nsz = (int)(si->raw_size / sizeof(int64_t));
                int64_t sizes[4];
                if (nsz > 4) nsz = 4;
                memcpy(sizes, si->raw_data, nsz * sizeof(int64_t));
                /* sizes are in ONNX order → reverse to ggml */
                for (int d = 0; d < nsz && d < 4; d++)
                    target_ne[d] = sizes[nsz - 1 - d];
                got_target = 1;
            }
        }

        /* Try scales input (Resize input 2, or Upsample input 1) */
        if (!got_target) {
            int scales_idx = (strcmp(op, "Upsample") == 0) ? 1 : 2;
            if (n->n_inputs > scales_idx && n->inputs[scales_idx][0] != '\0') {
                const onnx_initializer_t *sci = onnx_find_initializer(c->onnx, n->inputs[scales_idx]);
                if (sci && sci->raw_data && sci->data_type == ONNX_DTYPE_FLOAT) {
                    int nsc = (int)(sci->raw_size / sizeof(float));
                    float scales[4];
                    if (nsc > 4) nsc = 4;
                    memcpy(scales, sci->raw_data, nsc * sizeof(float));
                    /* scales in ONNX order → reverse to ggml */
                    for (int d = 0; d < nsc && d < 4; d++) {
                        int ggml_d = nsc - 1 - d;
                        target_ne[ggml_d] = (int64_t)(a->ne[ggml_d] * scales[d]);
                        if (target_ne[ggml_d] < 1) target_ne[ggml_d] = 1;
                    }
                    got_target = 1;
                }
            }
        }

        if (!got_target) {
            /* Try cval (compile-time value from Shape→Concat chain) for sizes */
            if (n->n_inputs > 3 && n->inputs[3][0] != '\0') {
                int64_t cv[ONNX_MAX_DIMS];
                int ncv = cval_get(c, n->inputs[3], cv, ONNX_MAX_DIMS);
                if (ncv > 0) {
                    /* cv is in ONNX order → reverse to ggml */
                    for (int d = 0; d < ncv && d < 4; d++)
                        target_ne[d] = cv[ncv - 1 - d];
                    got_target = 1;
                }
            }
        }
        if (!got_target) {
            /* Try cval for scales */
            int scales_idx = (strcmp(op, "Upsample") == 0) ? 1 : 2;
            if (n->n_inputs > scales_idx && n->inputs[scales_idx][0] != '\0') {
                int64_t cv[ONNX_MAX_DIMS];
                int ncv = cval_get(c, n->inputs[scales_idx], cv, ONNX_MAX_DIMS);
                if (ncv > 0) {
                    for (int d = 0; d < ncv && d < 4; d++) {
                        float s;
                        int32_t bits = (int32_t)cv[d];
                        memcpy(&s, &bits, sizeof(float));
                        int ggml_d = ncv - 1 - d;
                        target_ne[ggml_d] = (int64_t)(a->ne[ggml_d] * s);
                        if (target_ne[ggml_d] < 1) target_ne[ggml_d] = 1;
                    }
                    got_target = 1;
                }
            }
        }
        out = ggml_interpolate(c->ctx, a,
                               target_ne[0], target_ne[1],
                               target_ne[2], target_ne[3],
                               (uint32_t)mode);
        if (out->type != GGML_TYPE_F32)
            out = ggml_cast_numeric(c->ctx, out, GGML_TYPE_F32);
    }

    /* ── Expand ─────────────────────────────────────────────────── */
    else if (strcmp(op, "Expand") == 0) {
        if (!a || !b) return -1;
        /* b is the target shape tensor — read from initializer, Constant, or cval */
        int64_t shape[ONNX_MAX_DIMS];
        int ndims = 0;

        const onnx_initializer_t *shape_init = onnx_find_initializer(c->onnx, n->inputs[1]);
        if (!shape_init)
            shape_init = find_constant_tensor(c->onnx, n->inputs[1]);

        if (shape_init && shape_init->raw_data && shape_init->data_type == ONNX_DTYPE_INT64) {
            ndims = (int)(shape_init->raw_size / sizeof(int64_t));
            if (ndims > ONNX_MAX_DIMS) ndims = ONNX_MAX_DIMS;
            memcpy(shape, shape_init->raw_data, ndims * sizeof(int64_t));
        }

        /* Fallback: try compile-time value map */
        if (ndims == 0) {
            ndims = cval_get(c, n->inputs[1], shape, ONNX_MAX_DIMS);
        }
        if (ndims == 0) return -1;

        /* Resolve -1 (keep dim) using full ONNX shape of input tensor. */
        {
            int64_t a_onnx[ONNX_MAX_DIMS];
            int a_nd = tmap_get_shape(c, n->inputs[0], a_onnx, ONNX_MAX_DIMS);
            /* Align from the right: shape[ndims-1-k] ← a_onnx[a_nd-1-k] */
            for (int d = 0; d < ndims; d++) {
                if (shape[d] == -1) {
                    int a_d = d - (ndims - a_nd); /* right-aligned index into a_onnx */
                    if (a_d >= 0 && a_d < a_nd)
                        shape[d] = a_onnx[a_d];
                    else
                        shape[d] = 1;
                }
            }
        }
        /* Save resolved full ONNX shape before collapse */
        int orig_expand_ndims = ndims;
        int64_t orig_expand_shape[ONNX_MAX_DIMS];
        memcpy(orig_expand_shape, shape, ndims * sizeof(int64_t));

        /* Collapse >5D ONNX shape into 5D by merging leading ONNX dims. */
        if (ndims > GGML_MAX_DIMS) {
            int64_t merged = 1;
            for (int d = 0; d < ndims - (GGML_MAX_DIMS - 1); d++)
                merged *= shape[d];
            int64_t tmp[GGML_MAX_DIMS];
            tmp[0] = merged;
            for (int d = 1; d < GGML_MAX_DIMS; d++)
                tmp[d] = shape[ndims - (GGML_MAX_DIMS - 1) + d - 1];
            memcpy(shape, tmp, sizeof(tmp));
            ndims = GGML_MAX_DIMS;
        }

        /* Numpy-style broadcast: if rank(a) < rank(target), left-pad a
         * with 1s so ranks match, then apply broadcast rules. */
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) {
            a_nd = (int)ggml_n_dims(a);
        }
        if (a_nd < ndims && ndims <= GGML_MAX_DIMS) {
            int64_t a_ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
            for (int d = 0; d < a_nd; d++)
                a_ne[d] = a->ne[d];
            a = onnx_reshape_nd(c->ctx, a, a_ne, ndims);
        }

        /* Reverse ONNX shape → ggml ne order */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < ndims; d++)
            ne[d] = shape[ndims - 1 - d];

        /* Expand semantics: broadcast a into target shape.
         * For each dim: if shape[d]==1 → use a->ne[d],
         * if a->ne[d]==1 → use shape[d], otherwise must match. */
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (ne[d] == 1) ne[d] = a->ne[d];
        }

        struct ggml_tensor *target = onnx_new_tensor_nd(c->ctx, a->type, ne, ndims);
        if (ggml_are_same_shape(a, target)) {
            out = ggml_dup(c->ctx, a);  /* same shape: copy, not alias */
        } else {
            if (!ggml_can_repeat(a, target)) {
                fprintf(stderr, "[Expand] repeat FAIL: '%s' a.ne=[%lld,%lld,%lld,%lld] target=[%lld,%lld,%lld,%lld] shape_input='%s' ndims=%d\n",
                        n->outputs[0],
                        (long long)a->ne[0],(long long)a->ne[1],(long long)a->ne[2],(long long)a->ne[3],
                        (long long)ne[0],(long long)ne[1],(long long)ne[2],(long long)ne[3],
                        n->inputs[1], ndims);
                fprintf(stderr, "  ONNX shape=[");
                for (int d = 0; d < ndims; d++) fprintf(stderr, "%lld%s", (long long)shape[d], d<ndims-1?",":"");
                fprintf(stderr, "] a_input='%s' a_ndims=%d\n", n->inputs[0], tmap_get_ndims(c, n->inputs[0]));
            }
            out = ggml_repeat(c->ctx, a, target);
        }
        /* Register with full resolved ONNX shape (before collapse) */
        if (out) {
            ggml_set_name(out, n->outputs[0]);
            tmap_put_shape(c, n->outputs[0], out, orig_expand_shape, orig_expand_ndims);
            return 1;
        }
    }

    else {
        return 0; /* not this group */
    }

    *out_p    = out;
    *out_nd_p = out_nd;
    return 1;
}
