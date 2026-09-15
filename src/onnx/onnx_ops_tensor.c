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
        /* Prefer the stored ONNX ndims: ggml_n_dims() drops trailing 1-dims, so
         * an input like [24,128,4,1] reports 3 and the ONNX-dim -> ggml-ne
         * mapping below would be off by one for every shape[d]==0 entry. */
        int ndims_a = tmap_get_ndims(c, n->inputs[0]);
        if (ndims_a <= 0) ndims_a = ggml_n_dims(a);
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

        if (onnx_trace_nodes()) {
            fprintf(stderr, "[Reshape] %s: in='%s' nelem=%lld shape_src='%s' resolved=[",
                    n->outputs[0], n->inputs[0], (long long)total, n->inputs[1]);
            for (int d = 0; d < ndims; d++)
                fprintf(stderr, "%lld%s", (long long)shape[d], d < ndims - 1 ? "," : "");
            fprintf(stderr, "]\n");
        }

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
        /* Register the shape the MODEL asked for, even past 5D.  The tmap
         * field holds up to ONNX_MAX_DIMS, and a ggml reshape is a view over
         * the same bytes -- nothing moves -- so the physical tensor being
         * collapsed to 5D does not make the logical rank a lie.
         *
         * Keeping it matters for the op that comes next.  Collapsing here
         * always merges the LEADING dims, which is a guess about what the
         * following Transpose will do with them: Swin window partition
         * reshapes to 6D purely to express a permutation, and merging its
         * batch into H/M is right only while the batch is 1.  Transpose can
         * merge the pair that ITS perm leaves adjacent -- but only if the
         * real shape is still here to read. */
        int reg_ndims = orig_ndims;
        int64_t *reg_shape = orig_shape;
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
        if (onnx_trace_nodes()) {
            fprintf(stderr, "[Transpose/in] %s: nd=%d n_perm=%d perm=[", n->outputs[0], nd, n_perm);
            for (int i = 0; i < n_perm; i++)
                fprintf(stderr, "%lld%s", (long long)perm[i], i < n_perm-1 ? "," : "");
            fprintf(stderr, "] a.ne=[%lld,%lld,%lld,%lld]\n",
                    (long long)a->ne[0], (long long)a->ne[1],
                    (long long)a->ne[2], (long long)a->ne[3]);
        }

        /* Check for identity perm first */
        int is_identity = 0;
        int merged_rank = 0;   /* set when axes were merged below */
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
                /* is_identity was decided from the pre-collapse perm, so it is
                 * stale now: collapsing can turn a non-identity permutation
                 * into the identity, and acting on the old answer would then
                 * shuffle axes the model wanted left alone. */
                is_identity = 1;
                for (int i = 0; i < n_perm; i++)
                    if (perm[i] != i) { is_identity = 0; break; }
            }
        }
        /* Rank above what ggml can hold: merge axes the permutation keeps
         * together.  If perm maps two ONNX axes to adjacent output positions
         * in the same order, no reshape can tell them apart and no permute
         * separates them -- so they are one axis as far as this op is
         * concerned, and merging them drops the rank without changing a byte.
         *
         * Swin's window partition is the case this exists for: it reshapes
         * [B,H,W,C] to 6D [B,H/M,M,W/M,M,C] purely to express one swap of the
         * middle axes, then reshapes straight back.  Merging the trailing
         * (M,C) pair -- untouched by that perm -- leaves 5D, which ggml holds.
         * Verified against a hand-computed 6D permutation before it was
         * written, for a merge at the head as well as at the tail.
         *
         * Only the perm decides what may merge.  Reshape, which produced this
         * tensor, always merges the LEADING axes because it has no way to know
         * what comes next; that is right only when the leading pair happens to
         * be the one the perm leaves alone.
         *
         * Entry is gated on the rank ggml cannot hold at all (> GGML_MAX_DIMS)
         * while the loop reduces to 4.  The asymmetry is deliberate: rank 5 is
         * representable and the 5-D branch below handles it, so merging there
         * would rewrite permutations that already work -- xcit's [48,4,3,784]
         * with perm 2,0,3,4,1 came out as [784,192,1,3] the one time this was
         * entered at rank 5.  Once a merge IS needed, though, stopping at 5
         * lands in that same branch, which requires a unit axis to squeeze and
         * silently mis-permutes without one; 4 is the rank that always works. */
        if (nd > GGML_MAX_DIMS && n_perm == nd) {
            int pos[ONNX_MAX_DIMS];
            for (int i = 0; i < nd; i++) pos[i] = -1;
            for (int i = 0; i < n_perm; i++)
                if (perm[i] >= 0 && perm[i] < nd) pos[perm[i]] = i;

            int64_t shape[ONNX_MAX_DIMS];
            int have_shape = tmap_get_shape(c, n->inputs[0], shape, ONNX_MAX_DIMS);
            if (have_shape < nd) {
                /* Without the real shape there is nothing to merge FROM: the
                 * tensor in hand was already collapsed by Reshape along its
                 * leading axes, and reinterpreting it needs to know what the
                 * model's axes actually were. */
                fprintf(stderr, "[onnx] Transpose %s: rank %d but only %d stored dims\n",
                        n->outputs[0], nd, have_shape);
                return -1;
            }

            while (nd > 4) {
                /* find an input axis k whose successor k+1 stays right after
                 * it in the output */
                int k = -1;
                for (int i = 0; i + 1 < nd; i++) {
                    if (pos[i] >= 0 && pos[i + 1] == pos[i] + 1) { k = i; break; }
                }
                if (k < 0) break;   /* nothing mergeable -- leave it to fail loudly */

                if (have_shape >= nd) {
                    shape[k] *= shape[k + 1];
                    for (int i = k + 1; i + 1 < nd; i++) shape[i] = shape[i + 1];
                }
                /* drop input axis k+1 and output position pos[k]+1 */
                int drop_out = pos[k] + 1;
                int64_t np[ONNX_MAX_DIMS];
                int c2 = 0;
                for (int i = 0; i < n_perm; i++) {
                    if (i == drop_out) continue;
                    int64_t v = perm[i];
                    if (v > k) v--;             /* input axis k+1 is gone */
                    np[c2++] = v;
                }
                memcpy(perm, np, (size_t)c2 * sizeof(int64_t));
                n_perm = c2;
                nd--;
                for (int i = 0; i < nd; i++) pos[i] = -1;
                for (int i = 0; i < n_perm; i++)
                    if (perm[i] >= 0 && perm[i] < nd) pos[perm[i]] = i;
            }

            /* Merging down to 4 is preferred, but 5 is still representable:
             * if the permutation offers no further mergeable pair, hand the
             * node to the 5-D branch rather than refuse it.  Refusing at 5 is
             * what dropped super-resolution's Transpose -- a rank-6 node that
             * merges once and then has nothing left adjacent, which the 5-D
             * branch had been handling correctly all along. */
            if (nd > GGML_MAX_DIMS) {
                fprintf(stderr, "[onnx] Transpose %s: rank %d with no mergeable "
                                "axis pair -- cannot be brought within %d axes\n",
                        n->outputs[0], nd, GGML_MAX_DIMS);
                return -1;
            }
            /* The tensor itself never changed; only the description of it. */
            {
                int64_t ne_m[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
                for (int d = 0; d < nd && d < GGML_MAX_DIMS; d++)
                    ne_m[d] = shape[nd - 1 - d];
                a = onnx_reshape_nd(c->ctx, a, ne_m, nd);
                merged_rank = 1;
            }
            if (onnx_trace_nodes()) {
                fprintf(stderr, "[Transpose/merge] %s: rank -> %d perm=[", n->outputs[0], nd);
                for (int i = 0; i < n_perm; i++)
                    fprintf(stderr, "%lld%s", (long long)perm[i], i < n_perm-1 ? "," : "");
                fprintf(stderr, "]\n");
            }
            /* recheck identity on the merged permutation */
            is_identity = 1;
            for (int i = 0; i < n_perm; i++)
                if (perm[i] != i) { is_identity = 0; break; }
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
            /* No perm at all means "reverse every axis", which for a rank-2
             * tensor is a plain transpose.
             *
             * A rank-2 tensor WITH a perm is only a transpose when that perm
             * actually swaps -- [1,0].  The identity [0,1] is caught earlier,
             * but a perm longer than the tensor's rank reaches here too (the
             * collapse above leaves n_perm > nd when it cannot remap), and
             * transposing on the strength of nd==2 alone then reverses axes
             * the model never asked to move. */
            int swaps = (n_perm == 0);
            if (n_perm >= 2 && perm[0] == 1 && perm[1] == 0) swaps = 1;
            if (onnx_trace_nodes())
                fprintf(stderr, "[Transpose] %s: nd=%d n_perm=%d perm=[%lld,%lld] -> %s\n",
                        n->outputs[0], nd, n_perm,
                        n_perm > 0 ? (long long)perm[0] : -1,
                        n_perm > 1 ? (long long)perm[1] : -1,
                        swaps ? "transpose" : "pass-through");
            out = swaps ? ggml_cont(c->ctx, ggml_transpose(c->ctx, a)) : a;
        } else {
            /* 4D (or less) Transpose via ggml_permute.
             * Convert ONNX perm to ggml permute axes.
             * ONNX perm[i]=j means output ONNX dim i ← input ONNX dim j.
             * ggml ax[k] means output ggml dim k ← input ggml dim ax[k].
             * ONNX dim i → ggml dim (reversed). */
            int onnx_nd = n_perm > nd ? n_perm : nd;
            /* After a merge the stored rank describes the shape the
             * MODEL wrote, not the one being permuted here -- reading it
             * back would undo the merge. */
            if (n->n_inputs > 0 && !merged_rank) {
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

        if (onnx_trace_nodes()) {
            fprintf(stderr, "[Unsqueeze] %s: in_nd=%d n_axes=%d axes=[", n->outputs[0], nd_in, n_axes);
            for (int i = 0; i < n_axes; i++)
                fprintf(stderr, "%lld%s", (long long)axes[i], i < n_axes-1 ? "," : "");
            fprintf(stderr, "] nd_out=%d onnx_out=[", nd_out);
            for (int d = 0; d < nd_out; d++)
                fprintf(stderr, "%lld%s", (long long)onnx_out[d], d < nd_out-1 ? "," : "");
            fprintf(stderr, "] ne=[%lld,%lld,%lld,%lld]\n",
                    (long long)ne[0],(long long)ne[1],(long long)ne[2],(long long)ne[3]);
        }
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

        if (onnx_trace_nodes()) {
            fprintf(stderr, "[Squeeze] %s: nd_in=%d onnx_in=[", n->outputs[0], nd_in);
            for (int i = 0; i < nd_in; i++)
                fprintf(stderr, "%lld%s", (long long)onnx_in[i], i < nd_in-1 ? "," : "");
            fprintf(stderr, "] n_axes=%d axes=[", n_axes);
            for (int j = 0; j < n_axes; j++)
                fprintf(stderr, "%lld%s", (long long)axes[j], j < n_axes-1 ? "," : "");
            fprintf(stderr, "] -> nd_out=%d onnx_out=[", nd_out);
            for (int i = 0; i < nd_out; i++)
                fprintf(stderr, "%lld%s", (long long)onnx_out[i], i < nd_out-1 ? "," : "");
            fprintf(stderr, "] a.ne=[%lld,%lld,%lld,%lld]\n",
                    (long long)a->ne[0], (long long)a->ne[1],
                    (long long)a->ne[2], (long long)a->ne[3]);
        }
        if (nd_out == 0) { nd_out = 1; onnx_out[0] = 1; }

        /* Reverse to ggml ne order */
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int d = 0; d < nd_out && d < GGML_MAX_DIMS; d++)
            ne[d] = onnx_out[nd_out - 1 - d];

        out = onnx_reshape_nd(c->ctx, a, ne, nd_out);
        /* nd_out is what Squeeze actually computed, so report it -- including
         * when every remaining dim is 1.
         *
         * Withholding it in that case (as `if (has_nonunit)` did) left out_nd
         * at -1, and the generic path then inherited the INPUT's rank: a
         * Squeeze of [1,1] down to [1] still reported rank 2.  MaskRCNN's
         * five FPN levels each feed that result through Slice into a Gather
         * index, where a rank above 1 selects the "pass the index straight
         * through" branch -- so no gather happened, each level kept its full
         * length (9408, 2352, 588, 147, 48), and the Concat that joins them
         * refused shapes that no longer matched.
         *
         * An all-ones shape is exactly where the rank cannot be recovered
         * later by looking at the tensor, since ggml_n_dims collapses it to
         * 1 -- which is why it has to be recorded here. */
        out_nd = nd_out;
        /* cval propagation.  Squeeze drops unit axes and touches no value, so
         * whatever is known about the contents survives unchanged -- and a
         * shape value passing through here is the common case, not the
         * exception: whisper's decoder builds its position range as
         * Shape -> Slice -> Squeeze -> Range, and Range needs a build-time
         * limit.  Without this the chain went dark at the last step and the
         * positional embedding was dropped from the graph entirely. */
        {
            int64_t cv[ONNX_MAX_DIMS];
            int ncv = cval_get(c, n->inputs[0], cv, ONNX_MAX_DIMS);
            if (ncv > 0)
                cval_put(c, n->outputs[0], cv, ncv);
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
        /* Concat supports N inputs — chain pairwise.
         *
         * Logically empty inputs are left out of the chain entirely, so the
         * result really is shorter rather than carrying a row that ONNX says
         * does not exist.  Skipping them only at read time would keep the
         * output its full length with unselected rows still inside it, which
         * is exactly the bug this is here to fix: MaskRCNN concatenates one
         * branch per class, and the 79 classes where nothing passed the score
         * threshold each contributed a phantom box built from the unfilled
         * element of an empty NonZero. */
        int first = -1;
        for (int i = 0; i < n->n_inputs; i++) {
            if (n->inputs[i][0] == '\0') continue;
            if (tmap_is_empty(c, n->inputs[i])) continue;
            if (!get_input(c, n, i)) continue;
            first = i;
            break;
        }
        if (first < 0) {
            /* Every branch was empty.  There is no shorter tensor to build --
             * ggml cannot hold one -- so keep input 0's single row and mark the
             * result, leaving the decision to whoever consumes it. */
            out = a;
            if (onnx_trace_nodes())
                fprintf(stderr, "[Concat] %s: all %d input(s) empty -> marking output empty\n",
                        n->outputs[0], n->n_inputs);
            for (int i = 0; i < n->n_outputs; i++)
                if (n->outputs[i][0] != '\0') {
                    tmap_put_nd(c, n->outputs[i], out, nd);
                    tmap_mark_empty(c, n->outputs[i]);
                }
            return 1; /* already registered */
        }
        out = get_input(c, n, first);
        for (int i = first + 1; i < n->n_inputs; i++) {
            struct ggml_tensor *inp = get_input(c, n, i);
            if (!inp) continue;
            if (tmap_is_empty(c, n->inputs[i])) {
                if (onnx_trace_nodes())
                    fprintf(stderr, "[Concat] %s: skipping empty input '%s'\n",
                            n->outputs[0], n->inputs[i]);
                continue;
            }
            /* ggml_concat requires matching types — cast if needed */
            /* cast_numeric throughout: these convert between integer and
             * float, where ggml_cast would hand over the bit pattern. */
            if (out->type != inp->type) {
                if (inp->type == GGML_TYPE_I32 && out->type == GGML_TYPE_F32)
                    inp = ggml_cast_numeric(c->ctx, inp, GGML_TYPE_F32);
                else if (out->type == GGML_TYPE_I32 && inp->type == GGML_TYPE_F32) {
                    out = ggml_cast_numeric(c->ctx, out, GGML_TYPE_F32);
                }
            }
            /* ggml_concat asserts on every axis but `dim`, and its abort names
             * no tensor -- so print the two shapes here first.  Reading them
             * off the model is what turns "assert failed somewhere in a 284
             * node graph" into a specific mismatched axis. */
            if (onnx_trace_nodes()) {
                fprintf(stderr, "[Concat] %s dim=%d  a='%s' ne=[%lld,%lld,%lld,%lld,%lld]"
                                "  b='%s' ne=[%lld,%lld,%lld,%lld,%lld]\n",
                        n->outputs[0], dim,
                        ggml_get_name(out),
                        (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2],
                        (long long)out->ne[3], (long long)out->ne[4],
                        ggml_get_name(inp),
                        (long long)inp->ne[0], (long long)inp->ne[1], (long long)inp->ne[2],
                        (long long)inp->ne[3], (long long)inp->ne[4]);
            }
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (d == dim) continue;
                if (out->ne[d] != inp->ne[d]) {
                    fprintf(stderr, "[onnx] Concat %s: axis %d differs (%lld vs %lld) "
                                    "concatenating '%s' and '%s' along dim %d\n",
                            n->outputs[0], d, (long long)out->ne[d], (long long)inp->ne[d],
                            ggml_get_name(out), ggml_get_name(inp), dim);
                    return -1;
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


        /* True ONNX rank of the index.  tmap coerces a rank-0 initializer to
         * 1, so by the time b_nd is computed a scalar index and a length-1
         * vector look identical -- yet ONNX gives them different output
         * ranks (a scalar index drops the axis, a length-1 vector keeps it).
         * The honest rank is still in the initializer, so read it there.
         * LIMITATION: this only works for a constant index; when the index is
         * computed in the graph the rank is not recorded anywhere and b_nd
         * stays the approximation, exactly as before this was introduced. */
        int idx_rank = -1;
        {
            const onnx_initializer_t *ii = onnx_find_initializer(c->onnx, n->inputs[1]);
            if (!ii) ii = find_constant_tensor(c->onnx, n->inputs[1]);
            if (ii) idx_rank = ii->n_dims;
        }
        if (onnx_trace_nodes()) {
            int has_axis = (onnx_node_find_attr(n, "axis") != NULL);
            fprintf(stderr, "[Gather] %s: data=%s a.ne=[%lld,%lld,%lld,%lld] a_nd=%d idx=%s b_nd=%d axis=%lld(attr=%d) idx_rank=%d\n",
                    n->outputs[0], n->inputs[0],
                    (long long)a->ne[0], (long long)a->ne[1],
                    (long long)a->ne[2], (long long)a->ne[3],
                    a_nd, n->inputs[1], b_nd, (long long)axis, has_axis, idx_rank);
        }

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

            if (onnx_trace_nodes()) {
                fprintf(stderr, "[Gather/cval] %s: data_vals=[", n->outputs[0]);
                for (int j = 0; j < ncv_data; j++)
                    fprintf(stderr, "%lld%s", (long long)cv_data[j], j < ncv_data - 1 ? "," : "");
                fprintf(stderr, "] idx=[");
                for (int j = 0; j < ncv_idx; j++)
                    fprintf(stderr, "%lld%s", (long long)cv_idx[j], j < ncv_idx - 1 ? "," : "");
                fprintf(stderr, "] -> [");
                for (int j = 0; j < nr; j++)
                    fprintf(stderr, "%lld%s", (long long)result[j], j < nr - 1 ? "," : "");
                fprintf(stderr, "]\n");
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
            /* cast_numeric, not cast: ggml_cast reinterprets the bits, so an
             * index arrives as whatever the float's bit pattern means as an
             * integer.  Small indices land close enough to plausible values to
             * hide it -- MaskRCNN's 0, 62, 63 came out as 1, 63, 64, which
             * reads as an off-by-one and sent this hunt after a phantom
             * 1-based/0-based mismatch -- until index 80 ran off the end of an
             * 80-row table.  TopK already avoids this; Gather did not. */
            b = ggml_cast_numeric(c->ctx, b, GGML_TYPE_I32);
        }

        /* Normalize negative axis */
        if (axis < 0) axis += a_nd;

        /* ggml_get_rows(a[n_embd,ne1,ne2,ne3], b[n_rows,ne2,ne3])
         *   -> [n_embd, n_rows, ne2, ne3]
         * gathers along ggml axis 1, keeps axis 0 whole and treats axes 2/3
         * as a batch shared with b, so the ONNX axis being gathered has to
         * sit at ggml axis 1 before the call.
         *
         * ONNX axis k <-> ggml axis (a_nd - 1 - k), which relies on a_nd
         * being the true ONNX rank -- see the rank inheritance in
         * map_node()'s generic path.
         *
         * LIMITATION: an `indices` tensor of rank > 1 is passed straight
         * through.  ONNX splices the index dimensions in at position `axis`
         * while ggml_get_rows appends them as a batch; the two agree only
         * for a scalar or 1-D index. */
        /* Number of ggml axes actually carrying data.  ggml_n_dims() stops at
         * the last axis with ne != 1 and so misses a populated ne[4] sitting
         * behind a unit ne[3] -- as in cait's [48,576,6,1,3] QKV tensor -- so
         * the axes are counted here over all GGML_MAX_DIMS. */
        int g_nd = 1;
        for (int d = 0; d < GGML_MAX_DIMS; d++)
            if (a->ne[d] != 1) g_nd = d + 1;
        if (g_nd < 2) g_nd = 2;

        /* Gathering a unit ONNX axis with a scalar index selects the only
         * slice there is, so the data passes through untouched.  This has to
         * be caught before the clamp below, which would otherwise move the
         * gather onto a neighbouring populated axis and cut the tensor down to
         * one element of it.
         *
         * MaskRCNN does exactly this: the decoded boxes are [1,147,4] and a
         * Gather with axis=0 drops the batch axis, leaving all 147 boxes.  The
         * clamp turned it into a gather over the 147 instead, so a single box
         * reached NMS -- which then had nothing to suppress, and the detection
         * counts came out wrong with no error anywhere. */
        {
            int64_t oshape[ONNX_MAX_DIMS];
            int ond = tmap_get_shape(c, n->inputs[0], oshape, ONNX_MAX_DIMS);
            if (ond == a_nd && axis >= 0 && axis < ond && oshape[axis] == 1 &&
                idx_rank == 0) {
                if (onnx_trace_nodes())
                    fprintf(stderr, "[Gather] %s: ONNX axis %lld is unit; "
                                    "scalar gather is a passthrough\n",
                            n->outputs[0], (long long)axis);
                /* Materialise rather than forward the view.  The gather this
                 * replaces went through get_rows, which always produced a
                 * contiguous result, so every consumer downstream was written
                 * against that guarantee -- MaskRCNN's next Reshape asserts on
                 * it directly (ggml_reshape_1d), and a permuted view arriving
                 * there aborts the process instead of returning an error.
                 *
                 * Always a fresh node, never `a` itself: the output is named
                 * below, and renaming the input in place would make every
                 * other consumer of it -- and every trace line -- refer to a
                 * tensor by the wrong name. */
                out = ggml_cont(c->ctx, a);
                ggml_set_name(out, n->outputs[0]);
                tmap_put_nd(c, n->outputs[0], out, a_nd > 1 ? a_nd - 1 : 1);
                return 1;
            }
        }

        /* The ONNX rank can exceed the ggml one, because ONNX dims of size 1
         * need no ggml axis.  Those extra dims are leading, so clamping keeps
         * ONNX axis 0 on the highest populated ggml axis, where it belongs. */
        int g_ax = a_nd - 1 - (int)axis;
        if (g_ax > g_nd - 1) g_ax = g_nd - 1;

        if (g_ax < 0 || b_nd > 1) {
            /* Outside the mapping above -- gather as before. */
            out = ggml_get_rows(c->ctx, a, b);
        } else if (g_ax == 1) {
            /* Already in place; ggml_get_rows handles the rest itself. */
            out = ggml_get_rows(c->ctx, a, b);
        } else if (g_ax >= g_nd - 1 && g_nd > 2) {
            /* Highest ggml axis of a rank>2 tensor: folding axes 0..g_nd-2
             * into one leaves the gathered axis at position 1. */
            int64_t row_size = 1;
            for (int d = 0; d < g_nd - 1; d++)
                row_size *= a->ne[d];
            struct ggml_tensor *a2d =
                ggml_reshape_2d(c->ctx, a, row_size, a->ne[g_nd - 1]);
            struct ggml_tensor *gathered = ggml_get_rows(c->ctx, a2d, b);
            int64_t n_idx = (int64_t)ggml_nelements(b);
            int64_t back_ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
            for (int d = 0; d < g_nd - 1; d++)
                back_ne[d] = a->ne[d];
            back_ne[g_nd - 1] = n_idx;
            out = onnx_reshape_nd(c->ctx, gathered, back_ne, g_nd);
        } else if (g_nd > 4) {
            /* ggml_permute addresses 4 axes only, so a 5-D tensor whose
             * gathered axis is neither axis 1 nor the highest one is left to
             * the historical behaviour. */
            out = ggml_get_rows(c->ctx, a, b);
        } else {
            /* Any other axis (notably g_ax == 0 on a 2-D tensor): swap it
             * with axis 1, gather, swap back.  ggml_permute takes
             * DESTINATION slots -- the argument at index i says where source
             * axis i ends up -- and a two-axis swap is its own inverse. */
            int ax[4] = {0, 1, 2, 3};
            ax[g_ax] = 1;
            ax[1]    = g_ax;

            /* ⚠️ EXPERIMENT, not a finished change: break the view between
             * `a` and the permute, unconditionally, to find out whether that
             * is what the Vulkan fault needs.
             *
             * ggml_permute returns a VIEW of `a`, so the two share storage and
             * the scheduler cannot place them on different backends.  When `a`
             * is an op Vulkan cannot run (NonMaxSuppression is GGML_OP_CUSTOM)
             * and the permute is one it can, the pre-assign in
             * sched_alloc_and_fill_on pins the view to the GPU and drags the
             * CPU-only tensor along with it -- and moving only the source
             * instead was already measured to fault the device.
             *
             * A cont here gives the permute its own storage, so the source is
             * free to live on the host.  Doing it for EVERY gather on this path
             * (496 permutes, of which 9 sit on a CPU-only tensor) is wasteful
             * and is not the shape of the eventual fix; it is the cheapest way
             * to learn whether breaking the view helps at all, before paying
             * for a narrow version that has to plumb the backend handle into
             * this file. */
            struct ggml_tensor *a_src =
                getenv("ONNX_EXPERIMENT_CONT_BEFORE_PERMUTE")
                    ? ggml_cont(c->ctx, a) : a;
            struct ggml_tensor *ap =
                ggml_cont(c->ctx,
                          ggml_permute(c->ctx, a_src, ax[0], ax[1], ax[2], ax[3]));
            struct ggml_tensor *gathered = ggml_get_rows(c->ctx, ap, b);

            /* A scalar index (ONNX rank 0) drops the gathered axis, so the
             * result is one rank lower than `a` and the swap has nothing to
             * undo: get_rows already leaves the kept axis at ggml 0, which is
             * where a rank-(a_nd-1) tensor wants it.  Permuting back here
             * would move it to ggml 1 and produce [1,N] for what ONNX calls
             * [N] -- a shape that disagrees with the rank recorded below and
             * loses every element but the first when the shape is rebuilt
             * from ne[] at that rank.
             *
             * A length-1 vector index (rank 1) is NOT the same case: it keeps
             * the axis, the output stays at rank a_nd, and the swap back is
             * required.  The two are indistinguishable by `b` alone -- hence
             * idx_rank, read from the initializer above.  When the index is
             * computed in the graph (idx_rank < 0) the rank is unknown, so
             * the historical swap-back stands. */
            int drop_axis = (idx_rank == 0);
            if (drop_axis) {
                out = gathered;
            } else {
                out = ggml_cont(c->ctx,
                                ggml_permute(c->ctx, gathered,
                                             ax[0], ax[1], ax[2], ax[3]));
            }

            if (onnx_trace_nodes())
                fprintf(stderr,
                        "[Gather/swap] %s: g_ax=%d g_nd=%d a_nd=%d idx_rank=%d "
                        "permuted=[%lld,%lld,%lld,%lld] gathered=[%lld,%lld,%lld,%lld] "
                        "-> %s out=[%lld,%lld,%lld,%lld]\n",
                        n->outputs[0], g_ax, g_nd, a_nd, idx_rank,
                        (long long)ap->ne[0], (long long)ap->ne[1],
                        (long long)ap->ne[2], (long long)ap->ne[3],
                        (long long)gathered->ne[0], (long long)gathered->ne[1],
                        (long long)gathered->ne[2], (long long)gathered->ne[3],
                        drop_axis ? "no_swap_back(scalar_idx)" : "swap_back",
                        (long long)out->ne[0], (long long)out->ne[1],
                        (long long)out->ne[2], (long long)out->ne[3]);
        }

        /* Register output with correct ONNX ndims.
         * ONNX Gather(data, indices, axis=0):
         * output_shape = indices_shape + data_shape[1:]
         * E.g. data [V,D] (2D) + indices [B,S] (2D) → [B,S,D] (3D). */
        if (out) {
            /* ONNX: output rank = rank(indices) + rank(data) - 1.  Use the
             * index's true rank where it is known -- a scalar index (rank 0)
             * drops the gathered axis, while a length-1 vector keeps it, and
             * b_nd reports 1 for both. */
            int idx_nd = (idx_rank >= 0) ? idx_rank : b_nd;
            int out_nd = idx_nd + (a_nd > 1 ? a_nd - 1 : 0);
            if (out_nd < 1) out_nd = 1;
            if (out_nd > 4) out_nd = 4;
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, out_nd);
        }
        return 1; /* already registered */
    }

    /* ── TopK ───────────────────────────────────────────────────── */
    /* TopK(X, K) -> Values, Indices, both taken along `axis` (default -1).
     * ggml sorts along ne[0] only, so any other axis is rotated there and
     * back.  ggml_argsort_top_k returns sorted indices, matching ONNX's
     * default sorted=1, and the values are then read back with a per-row
     * get_rows gather (see the offset formula below). */
    else if (strcmp(op, "TopK") == 0) {
        if (!a) return -1;

        int64_t axis = onnx_attr_int(n, "axis", -1);
        int a_nd = tmap_get_ndims(c, n->inputs[0]);
        if (a_nd <= 0) a_nd = (int)ggml_n_dims(a);
        if (axis < 0) axis += a_nd;
        /* TopK runs along ne[0], so an axis elsewhere is rotated there and
         * back -- the same swap Softmax and CumSum use, and self-inverse for
         * the same reason.  MaskRCNN's final TopK picks over axis 0 of a
         * rank-3 tensor, and refusing it left the whole detection branch
         * unbuilt: the graph came out with zero nodes and every output stayed
         * at its uninitialised value. */
        int ggml_d = a_nd - 1 - (int)axis;
        if (ggml_d < 0) ggml_d = 0;
        if (ggml_d > 3) {
            fprintf(stderr, "[onnx] TopK %s: axis %d of rank %d maps to ggml "
                            "dim %d, past what permute reaches\n",
                    n->outputs[0], (int)axis, a_nd, ggml_d);
            return -1;
        }
        int tk_ax[4] = {0, 1, 2, 3};
        if (ggml_d > 0) { tk_ax[0] = ggml_d; tk_ax[ggml_d] = 0; }
        struct ggml_tensor *src = a;
        if (ggml_d > 0)
            src = ggml_cont(c->ctx,
                      ggml_permute(c->ctx, a, tk_ax[0], tk_ax[1], tk_ax[2], tk_ax[3]));
        int64_t largest = onnx_attr_int(n, "largest", 1);
        if (!largest) {
            fprintf(stderr, "[onnx] TopK: largest=0 is not supported\n");
            return -1;
        }

        /* K comes as a 1-element tensor input; it has to be known while the
         * graph is built, since it sets the output shape. */
        int64_t kv[ONNX_MAX_DIMS];
        int nk = (n->n_inputs > 1) ? cval_get(c, n->inputs[1], kv, ONNX_MAX_DIMS) : 0;
        if (nk < 1) {
            const onnx_initializer_t *ki = (n->n_inputs > 1)
                ? onnx_find_initializer(c->onnx, n->inputs[1]) : NULL;
            if (!ki) ki = (n->n_inputs > 1)
                ? find_constant_tensor(c->onnx, n->inputs[1]) : NULL;
            if (ki && ki->raw_data && ki->data_type == ONNX_DTYPE_INT64) {
                int64_t kk; memcpy(&kk, ki->raw_data, sizeof(int64_t));
                kv[0] = kk; nk = 1;
            }
        }
        if (nk < 1) {
            fprintf(stderr, "[onnx] TopK: K is not known at build time\n");
            return -1;
        }
        int k = (int)kv[0];

        /* The measurement is an upper BOUND on k, never its value.
         *
         * K is the op's second input, and ONNX lets it be computed rather than
         * stored: MaskRCNN builds it as ReduceMin(Concat(limit, n_candidates))
         * -- "at most `limit` proposals, fewer if fewer arrived" -- which the
         * constant folder resolves to a real number by the time we get here.
         * onnx_resolved_size, by contrast, measures the axis being ranked once
         * the previous segment has run: that is how many candidates EXIST, not
         * how many the model asked to keep.
         *
         * Taking the measurement as k outright (as this did) throws the limit
         * away and turns TopK into a full sort.  MaskRCNN's two RPN nodes then
         * handed NMS 9408 and 2352 proposals instead of 1000 each, and the
         * model returned 80 detections where onnxruntime returns 51.
         *
         * The clamp is still needed the other way: k may exceed the candidate
         * count, and ggml_argsort_top_k cannot return more than exists. */
        if (k <= 0) {
            /* Not the "K is larger than the axis" case the clamp below covers:
             * the spec requires K positive, so this is a K that was resolved
             * to something TopK has no meaning for.  Refusing beats ranking a
             * nonsense number of elements and reporting success. */
            fprintf(stderr, "[onnx] TopK %s: K resolved to %d; the spec requires "
                            "a positive value\n", n->outputs[0], k);
            return -1;
        }
        {
            int64_t measured = onnx_resolved_size(c, n->outputs[0]);
            if (measured > 0) {
                int km = (int)measured;
                if (k > km) {
                    /* The model asked for more than exists.  A valid model does
                     * not, so say so -- but keep going at the axis size, which
                     * is what every element there can supply. */
                    fprintf(stderr, "[onnx] TopK %s: K=%d exceeds the %d elements "
                                    "on the ranked axis; clamped\n",
                            n->outputs[0], k, km);
                    k = km;
                }
                if (onnx_trace_nodes())
                    fprintf(stderr, "[TopK] node=%s: k_graph=%d, axis_size=%d, "
                                    "k_used=%d\n",
                            n->outputs[0], (int)kv[0], km, k);
            }
        }
        /* k < 1 is refused above, so only the upper bound is left -- and this
         * one still matters when nothing was measured, since argsort_top_k
         * cannot return more elements than the axis holds. */
        if (k > (int)src->ne[0]) k = (int)src->ne[0];

        struct ggml_tensor *idx = ggml_argsort_top_k(c->ctx, src, k);
        idx = ggml_cont(c->ctx, idx);   /* argsort_top_k returns a view */

        /* Values: a per-row gather.  argsort_top_k ranks each row of ne[0]
         * independently and its indices are row-local, while get_rows copies
         * whole rows of its data argument and cannot pick an element inside
         * one.  Treating every scalar of src as a row of length 1 turns the
         * element pick into a row pick, which get_rows does express:
         *
         *   flat_idx(j,r) = idx(j,r) + r*n0,  0 <= j < k, 0 <= r < nrows
         *
         * where r is the linear index of [i1,i2,i3] in ggml memory order.
         *
         * This is one path for every shape, not a single-row special case and
         * a multi-row one: at nrows == 1 the offset is zero, flat_idx equals
         * idx, and the reshape returns the same [k,1,1,1] the old single-row
         * branch did.  The two-branch version had TopK's semantics written
         * twice, and every later fix -- negative axis, k=1, contiguity after
         * permute, index dtype -- would have had to land in both. */
        const int64_t n0    = src->ne[0];
        const int64_t nrows = ggml_nrows(src);

        /* src is contiguous here (cont() above, or untouched when ggml_d == 0),
         * so ne[0]-sized blocks of its data are exactly the independent rows
         * the indices address, and idx carries src's trailing dims, so its
         * rows enumerate in the same order.  Both are the assumptions the
         * offset formula rests on; state them rather than infer them. */
        if (!ggml_is_contiguous(src) ||
            idx->ne[1] != src->ne[1] || idx->ne[2] != src->ne[2] ||
            idx->ne[3] != src->ne[3]) {
            fprintf(stderr, "[onnx] TopK %s: gather precondition failed "
                            "(contig=%d src ne=[%lld,%lld,%lld,%lld] "
                            "idx ne=[%lld,%lld,%lld,%lld])\n",
                    n->outputs[0], (int)ggml_is_contiguous(src),
                    (long long)src->ne[0], (long long)src->ne[1],
                    (long long)src->ne[2], (long long)src->ne[3],
                    (long long)idx->ne[0], (long long)idx->ne[1],
                    (long long)idx->ne[2], (long long)idx->ne[3]);
            return -1;
        }

        /* The offsets travel through F32 because arange and the binary ops are
         * F32-only, and cast_numeric truncates -- so every offset has to be an
         * exactly representable integer.  Past 2^24 it is not, and the gather
         * would quietly read a neighbouring element instead of failing. */
        if (n0 * nrows >= (1LL << 24)) {
            fprintf(stderr, "[onnx] TopK %s: %lld elements exceeds the 2^24 "
                            "index range an F32 offset can carry exactly\n",
                    n->outputs[0], (long long)(n0 * nrows));
            return -1;
        }

        struct ggml_tensor *idx_fv = ggml_cast_numeric(c->ctx, idx, GGML_TYPE_F32);
        struct ggml_tensor *flat  = idx_fv;
        if (nrows > 1) {
            /* r*n0, shaped [1,r1,r2,r3] so it broadcasts along the k axis. */
            struct ggml_tensor *off =
                ggml_scale(c->ctx, ggml_arange(c->ctx, 0.0f, (float)nrows, 1.0f),
                           (float)n0);
            off = ggml_reshape_4d(c->ctx, off, 1, src->ne[1], src->ne[2], src->ne[3]);
            off = ggml_repeat(c->ctx, off, idx_fv);
            flat = ggml_add(c->ctx, idx_fv, off);
        }
        struct ggml_tensor *flat_i = ggml_reshape_1d(c->ctx,
            ggml_cont(c->ctx, ggml_cast_numeric(c->ctx, flat, GGML_TYPE_I32)),
            (int64_t)k * nrows);

        struct ggml_tensor *a2 =
            ggml_reshape_2d(c->ctx, ggml_cont(c->ctx, src), 1, n0 * nrows);
        struct ggml_tensor *g = ggml_get_rows(c->ctx, a2, flat_i);
        struct ggml_tensor *vals = ggml_reshape_4d(c->ctx, ggml_cont(c->ctx, g),
            k, src->ne[1], src->ne[2], src->ne[3]);

        /* Back to the model's axis order.  The swap is its own inverse, so
         * the same permutation serves both ways. */
        if (ggml_d > 0) {
            vals = ggml_cont(c->ctx,
                       ggml_permute(c->ctx, vals, tk_ax[0], tk_ax[1], tk_ax[2], tk_ax[3]));
            idx  = ggml_cont(c->ctx,
                       ggml_permute(c->ctx, idx,  tk_ax[0], tk_ax[1], tk_ax[2], tk_ax[3]));
        }

        if (n->n_outputs > 0 && n->outputs[0][0] != '\0') {
            ggml_set_name(vals, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], vals, a_nd);
        }
        if (n->n_outputs > 1 && n->outputs[1][0] != '\0') {
            /* cast_numeric, not cast: the latter reinterprets the bits, so an
             * index of 3 would come out as 4.2e-45 rather than 3.0. */
            struct ggml_tensor *idx_f = ggml_cast_numeric(c->ctx, idx, GGML_TYPE_F32);
            ggml_set_name(idx_f, n->outputs[1]);
            tmap_put_nd(c, n->outputs[1], idx_f, a_nd);
        }
        return 1; /* outputs registered here */
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

        /* Cast indices to I32 if needed -- by value, not by bit pattern. */
        if (b->type != GGML_TYPE_I32) {
            b = ggml_cast_numeric(c->ctx, b, GGML_TYPE_I32);
        }
        /* Cast updates to F32 if needed */
        if (c_upd->type != GGML_TYPE_F32) {
            c_upd = ggml_cast_numeric(c->ctx, c_upd, GGML_TYPE_F32);
        }
        /* Cast data to F32 if needed */
        if (a->type != GGML_TYPE_F32) {
            a = ggml_cast_numeric(c->ctx, a, GGML_TYPE_F32);
        }

        out = ggml_scatter_elements(c->ctx, a, c_upd, b, reduction, ggml_axis);
        if (out) {
            ggml_set_name(out, n->outputs[0]);
            tmap_put_nd(c, n->outputs[0], out, a_nd);
        }
        return 1;
    }



    /* ── GatherND ───────────────────────────────────────────────── */
    else if (strcmp(op, "GatherND") == 0) {
        if (!a || !b) return -1;
        /* GatherND(data, indices): each row of `indices` is a q-tuple naming
         * one slice of `data`, and the slices are stacked in the order the
         * tuples appear.
         *
         * The mirror image of ScatterND above, and built the same way: fold
         * the q addressed axes of data into one and the slice becomes a row,
         * which is exactly what ggml_get_rows takes.  The flat row index is
         * computed in the graph -- sum over k of idx[..., k] * stride_k --
         * so indices produced at runtime work as well as constant ones.
         *
         * batch_dims is refused rather than approximated: it changes which
         * axes are addressed and which are carried, and no model here uses it.
         */
        int64_t batch_dims = onnx_attr_int(n, "batch_dims", 0);
        if (batch_dims != 0) {
            fprintf(stderr, "[onnx] GatherND %s: batch_dims=%lld not supported\n",
                    n->outputs[0], (long long)batch_dims);
            return -1;
        }

        int r_nd = tmap_get_ndims(c, n->inputs[0]);
        int i_nd = tmap_get_ndims(c, n->inputs[1]);
        if (r_nd <= 0) r_nd = (int)ggml_n_dims(a);
        if (i_nd <= 0) i_nd = (int)ggml_n_dims(b);
        int64_t q = b->ne[0];            /* last ONNX axis of indices */
        if (q < 1 || q > r_nd) {
            fprintf(stderr, "[onnx] GatherND %s: index width %lld against rank %d\n",
                    n->outputs[0], (long long)q, r_nd);
            return -1;
        }

        int64_t row_len = 1;
        for (int d = 0; d < r_nd - (int)q; d++) row_len *= a->ne[d];
        int64_t total = ggml_nelements(a);
        int64_t n_rows = row_len > 0 ? total / row_len : 0;
        if (row_len <= 0 || n_rows <= 0) {
            fprintf(stderr, "[onnx] GatherND %s: empty data\n", n->outputs[0]);
            return -1;
        }

        /* Stride of each addressed axis, counted in rows: ONNX index component
         * k walks ONNX axis k, which is ggml axis r_nd-1-k. */
        int64_t stride_of_k[GGML_MAX_DIMS];
        for (int k = 0; k < (int)q; k++) {
            int ggml_d = r_nd - 1 - k;
            int64_t s = 1;
            for (int d = r_nd - (int)q; d < ggml_d; d++) s *= a->ne[d];
            stride_of_k[k] = s;
        }

        int64_t n_idx = ggml_nelements(b) / q;

        struct ggml_tensor *idx = b;
        if (idx->type != GGML_TYPE_F32) idx = ggml_cast_numeric(c->ctx, idx, GGML_TYPE_F32);
        struct ggml_tensor *idx2 = ggml_reshape_2d(c->ctx, idx, q, n_idx);

        struct ggml_tensor *flat = NULL;
        for (int k = 0; k < (int)q; k++) {
            struct ggml_tensor *row = ggml_cont(c->ctx,
                ggml_view_2d(c->ctx, idx2, 1, n_idx, idx2->nb[1],
                             (size_t)k * idx2->nb[0]));
            struct ggml_tensor *term = ggml_scale(c->ctx, row, (float)stride_of_k[k]);
            flat = flat ? ggml_add(c->ctx, flat, term) : term;
        }
        if (!flat) return -1;
        /* get_rows wants a 1-D I32 index list. */
        flat = ggml_cast_numeric(c->ctx,
                   ggml_reshape_1d(c->ctx, ggml_cont(c->ctx, flat), n_idx),
                   GGML_TYPE_I32);

        struct ggml_tensor *data2 = ggml_reshape_2d(c->ctx, a, row_len, n_rows);
        if (data2->type != GGML_TYPE_F32 && data2->type != GGML_TYPE_I32)
            data2 = ggml_cast_numeric(c->ctx, data2, GGML_TYPE_F32);

        struct ggml_tensor *g = ggml_get_rows(c->ctx, data2, flat);   /* [row_len, n_idx] */
        if (!g) return -1;

        /* Output shape: indices.shape[:-1] ++ data.shape[q:], which in ggml
         * order is the carried data axes first, then the index axes. */
        int64_t out_ne[GGML_MAX_DIMS] = {1, 1, 1, 1, 1};
        int od = 0;
        for (int d = 0; d < r_nd - (int)q && od < GGML_MAX_DIMS; d++)
            out_ne[od++] = a->ne[d];
        int64_t idx_shape[ONNX_MAX_DIMS];
        int have_idx = tmap_get_shape(c, n->inputs[1], idx_shape, ONNX_MAX_DIMS);
        if (have_idx >= i_nd && i_nd >= 1) {
            /* index axes, ONNX order minus the trailing q, reversed for ggml */
            for (int d = i_nd - 2; d >= 0 && od < GGML_MAX_DIMS; d--)
                out_ne[od++] = idx_shape[d];
        } else if (od < GGML_MAX_DIMS) {
            out_ne[od++] = n_idx;
        }
        int out_rank = od > 0 ? od : 1;
        out = onnx_reshape_nd(c->ctx, g, out_ne, out_rank);
        out_nd = (r_nd - (int)q) + (i_nd - 1);
        if (out_nd < 1) out_nd = 1;
    }
    /* ── ScatterND ──────────────────────────────────────────────── */
    else if (strcmp(op, "ScatterND") == 0) {
        if (!a || !b) return -1;
        /* ScatterND(data, indices, updates): each row of `indices` is a
         * q-tuple naming one slice of `data`, and the matching slice of
         * `updates` is written there.
         *
         * With the leading q axes of data folded into one, that slice IS a
         * row, and the op becomes ggml_scatter_elements over a 2-D view --
         * no new kernel, no new shader.  What has to be built is the flat
         * row index: sum over k of idx[..., k] * stride_k.  It is computed
         * in the GRAPH, not on the host, because Swin's masks derive their
         * indices from Mod on runtime shapes; a build-time reading of them
         * would be the same defect that made strided Slice publish
         * uninitialised memory.
         *
         * Layouts, in ggml order (ONNX reversed):
         *   data    ne = [d_{r-1}, ..., d_0]      rank r
         *   indices ne = [q, m_{p-2}, ..., m_0]   last ONNX axis is q, so it
         *                                          is ggml axis 0
         *   updates ne = [d_{r-1}, ..., d_q, m...]
         */
        struct ggml_tensor *upd = NULL;
        if (n->n_inputs > 2 && n->inputs[2][0] != '\0')
            upd = tmap_get(c, n->inputs[2]);
        if (!upd) {
            fprintf(stderr, "[onnx] ScatterND %s: missing updates\n", n->outputs[0]);
            return -1;
        }

        {
            char red[64] = {0};
            int rl = onnx_attr_str(n, "reduction", red, sizeof(red));
            if (rl > 0 && strcmp(red, "none") != 0) {
                /* add/mul/max/min exist in the spec; scatter_elements offers
                 * add only, and the rest have no caller here.  Refuse by name
                 * rather than silently overwrite. */
                fprintf(stderr, "[onnx] ScatterND %s: reduction='%s' not supported\n",
                        n->outputs[0], red);
                return -1;
            }
        }

        int r_nd = tmap_get_ndims(c, n->inputs[0]);
        int i_nd = tmap_get_ndims(c, n->inputs[1]);
        if (r_nd <= 0) r_nd = (int)ggml_n_dims(a);
        if (i_nd <= 0) i_nd = (int)ggml_n_dims(b);
        int64_t q = b->ne[0];            /* last ONNX axis of indices */
        if (q < 1 || q > r_nd) {
            fprintf(stderr, "[onnx] ScatterND %s: index width %lld against rank %d\n",
                    n->outputs[0], (long long)q, r_nd);
            return -1;
        }

        /* Row length = product of the data axes NOT addressed by an index,
         * i.e. ggml axes 0 .. r_nd-1-q.  n_rows is the rest. */
        int64_t row_len = 1;
        for (int d = 0; d < r_nd - (int)q; d++) row_len *= a->ne[d];
        int64_t total = ggml_nelements(a);
        int64_t n_rows = row_len > 0 ? total / row_len : 0;
        if (row_len <= 0 || n_rows <= 0) {
            fprintf(stderr, "[onnx] ScatterND %s: empty data\n", n->outputs[0]);
            return -1;
        }

        /* Strides of the addressed axes, in rows.  ONNX index component k
         * counts along ONNX axis k, which is ggml axis r_nd-1-k; the stride
         * of that axis measured in rows is the product of the row-counts
         * below it. */
        int64_t stride_of_k[GGML_MAX_DIMS];
        for (int k = 0; k < (int)q; k++) {
            int ggml_d = r_nd - 1 - k;
            int64_t s = 1;
            for (int d = r_nd - (int)q; d < ggml_d; d++) s *= a->ne[d];
            stride_of_k[k] = s;
        }

        /* Number of index tuples = everything in `indices` except the q axis. */
        int64_t n_upd = ggml_nelements(b) / q;

        struct ggml_tensor *idx = b;
        if (idx->type != GGML_TYPE_F32) idx = ggml_cast_numeric(c->ctx, idx, GGML_TYPE_F32);
        struct ggml_tensor *idx2 = ggml_reshape_2d(c->ctx, idx, q, n_upd);

        /* flat = sum_k idx2[k, :] * stride_k, built from views of one row
         * each so the arithmetic stays inside the graph. */
        struct ggml_tensor *flat = NULL;
        for (int k = 0; k < (int)q; k++) {
            struct ggml_tensor *row = ggml_view_2d(c->ctx, idx2, 1, n_upd,
                                                   idx2->nb[1],
                                                   (size_t)k * idx2->nb[0]);
            row = ggml_cont(c->ctx, row);
            struct ggml_tensor *term = ggml_scale(c->ctx, row, (float)stride_of_k[k]);
            flat = flat ? ggml_add(c->ctx, flat, term) : term;
        }
        if (!flat) return -1;
        /* scatter_elements wants the index at every element of the row it
         * writes, so the one flat index per update is repeated across the
         * row: [1, n_upd] -> [row_len, n_upd]. */
        struct ggml_tensor *idx_rep = flat;
        if (row_len > 1)
            idx_rep = ggml_repeat_4d(c->ctx, flat, row_len, n_upd, 1, 1);
        idx_rep = ggml_cast_numeric(c->ctx, ggml_cont(c->ctx, idx_rep), GGML_TYPE_I32);

        struct ggml_tensor *data2 = ggml_reshape_2d(c->ctx, a, row_len, n_rows);
        struct ggml_tensor *upd2  = ggml_reshape_2d(c->ctx, upd, row_len, n_upd);
        if (data2->type != GGML_TYPE_F32)
            data2 = ggml_cast_numeric(c->ctx, data2, GGML_TYPE_F32);
        if (upd2->type != GGML_TYPE_F32)
            upd2 = ggml_cast_numeric(c->ctx, upd2, GGML_TYPE_F32);

        struct ggml_tensor *sc = ggml_scatter_elements(c->ctx, data2, upd2,
                                                       idx_rep, 0 /* none */,
                                                       1 /* axis: the row axis */);
        if (!sc) return -1;
        out = onnx_reshape_nd(c->ctx, sc, a->ne,
                              r_nd > GGML_MAX_DIMS ? GGML_MAX_DIMS : r_nd);
        out_nd = r_nd;
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


        /* A strided slice along ONE axis, where the step divides that axis
         * evenly and the start picks one element of each group, is expressible
         * as ggml graph ops -- no deferred host copy needed.
         *
         * That matters for correctness, not just speed: the deferred path
         * (below) reads its source with ggml_backend_tensor_get at ALLOCATION
         * time, before inputs are uploaded and before the graph is computed.
         * When the source is a computed tensor -- as in xcit's positional
         * embedding, where the sin/cos halves are sliced out of a Div result --
         * it copies uninitialized memory, and the model gets NaN and 1e19
         * garbage that looks like an arithmetic disagreement three hundred
         * nodes later.
         *
         * Two shapes of the same idea, by which ggml dim is sliced:
         *   dim > 0: ggml_view takes an explicit nb for that dim, so a step of
         *            k is just nb[d]*k with ne[d] = len.
         *   dim = 0: nb[0] must stay the element size, so instead reshape the
         *            axis  [.., n, ..] -> [.., k, n/k, ..], which puts each
         *            group of k in the new leading dim, and view element
         *            `start` of it.
         * Both were verified against ggml before this was written. */
        int fast_axis = -1;      /* the single ggml dim being strided */
        if (!all_step1) {
            int n_strided = 0, ok = 1;
            for (int i = 0; i < n_slices && ok; i++) {
                int onnx_ax = has_axes ? (int)axes_arr[i] : i;
                if (onnx_ax < 0) onnx_ax += nd_onnx;
                int ggml_d = nd_onnx - 1 - onnx_ax;
                if (ggml_d < 0 || ggml_d > GGML_MAX_DIMS - 1) continue;
                if (steps[i] == 1) {
                    /* a step-1 axis is only free if it slices nothing away */
                    if (out_ne[ggml_d] != a->ne[ggml_d]) ok = 0;
                    continue;
                }
                int64_t st  = steps[i];
                int64_t dim = a->ne[ggml_d];
                int64_t s   = norm_starts[ggml_d];
                if (st <= 0 || dim <= 0 || dim % st != 0 ||
                    s < 0 || s >= st || out_ne[ggml_d] != dim / st) { ok = 0; break; }
                n_strided++;
                fast_axis = ggml_d;
            }
            if (!ok || n_strided != 1) fast_axis = -1;
        }

        if (fast_axis >= 0) {
            int64_t d   = fast_axis;
            int64_t st  = norm_steps[d];
            int64_t s   = norm_starts[d];
            int64_t len = out_ne[d];
            struct ggml_tensor *v;

            if (d > 0) {
                /* Step along a non-leading dim: widen that dim's stride. */
                int64_t ne_v[GGML_MAX_DIMS];
                for (int k = 0; k < GGML_MAX_DIMS; k++) ne_v[k] = a->ne[k];
                ne_v[d] = len;
                size_t off = (size_t)s * a->nb[d];
                v = ggml_view_4d(c->ctx, a, ne_v[0], ne_v[1], ne_v[2], ne_v[3],
                                 d == 1 ? a->nb[1] * (size_t)st : a->nb[1],
                                 d == 2 ? a->nb[2] * (size_t)st : a->nb[2],
                                 d == 3 ? a->nb[3] * (size_t)st : a->nb[3],
                                 off);
            } else {
                /* Step along ne[0]: split it into (st, len) and take row s. */
                struct ggml_tensor *r = ggml_reshape_4d(c->ctx, a, st, len,
                                                        a->ne[1],
                                                        a->ne[2] * a->ne[3]);
                v = ggml_view_4d(c->ctx, r, 1, len, r->ne[2], r->ne[3],
                                 r->nb[1], r->nb[2], r->nb[3],
                                 (size_t)s * r->nb[0]);
            }
            out = ggml_cont(c->ctx, v);
            /* Back to the slice's own shape: the view above carries the
             * split/strided layout, not the shape the model expects. */
            out = onnx_reshape_nd(c->ctx, out, out_ne,
                                  nd_onnx > GGML_MAX_DIMS ? GGML_MAX_DIMS : nd_onnx);
            out_nd = nd_onnx;
            if (onnx_trace_nodes())
                fprintf(stderr, "[Slice/strided-view] %s: dim=%d start=%lld step=%lld len=%lld\n",
                        n->outputs[0], (int)d, (long long)s,
                        (long long)st, (long long)len);
        } else if (all_step1) {
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
            if (onnx_trace_nodes()) {
                /* The strided path prints its start; this one printed nothing,
                 * so a slice that took the wrong element of a per-class block
                 * looked identical in the trace to one that took the right
                 * one -- same output shape either way. */
                fprintf(stderr, "[Slice/view] %s: off=%zu bytes (elem %zu) ne=[%lld,%lld,%lld,%lld]"
                                " offsets=[%lld,%lld,%lld,%lld]\n",
                        n->outputs[0], offset_bytes,
                        a->nb[0] ? offset_bytes / a->nb[0] : (size_t)0,
                        (long long)out_ne[0], (long long)out_ne[1],
                        (long long)out_ne[2], (long long)out_ne[3],
                        (long long)offsets[0], (long long)offsets[1],
                        (long long)offsets[2], (long long)offsets[3]);
            }
        } else {
            /* Deferred host copy: correct only while the source already holds
             * its final data when the copy runs -- an initializer, a constant,
             * or a model input (set_model_inputs calls the fill after upload).
             * A source produced BY the graph does not, and the copy then
             * publishes uninitialized memory as if it were data.  Say so here
             * rather than let the wrong numbers surface hundreds of nodes
             * later: that silence cost a long bisection on xcit once.
             *
             * The fix for such a slice is the graph path above; when its
             * arithmetic does not apply (step not dividing the axis, several
             * strided axes, start >= step, negative step), the slice needs to
             * run as its own graph segment, not another special case here. */
            /* Only when the slice actually produces something.  An empty
             * result -- ends = INT64_MIN with a positive step, which the spec
             * and ONNX Runtime both resolve to zero elements -- copies nothing
             * and so cannot publish garbage; warning about it sends the reader
             * after a defect that is not there.  MaskRCNN has twenty such
             * slices and every one of them was reported. */
            if (a->op != GGML_OP_NONE && ne_product(out_ne, GGML_MAX_DIMS) > 0)
                fprintf(stderr, "[onnx] strided Slice %s: source %s is computed by the "
                                "graph, so the deferred copy reads it before it holds "
                                "anything -- output will be garbage\n",
                        n->outputs[0], n->inputs[0]);

            /* step != 1: deferred strided copy after alloc */
            if (c->n_slice_fills >= ONNX_MAX_DEFERRED) {
                fprintf(stderr, "onnx_ggml: too many strided Slice ops (max 64)\n");
                return -1;
            }
            /* Create output tensor with correct shape */
            int nd = nd_onnx;
            if (nd > GGML_MAX_DIMS) nd = GGML_MAX_DIMS;
            /* ctx_weight, not ctx: this tensor is filled by
             * fill_strided_slices() after every allocation, so the pointer in
             * slice_fill_dst[] has to stay valid for as long as that list
             * does. c->ctx is the segment's own pool, released once the segment
             * has computed, and a registration pointing into it would be read
             * after the free. Every other deferred list already targets
             * ctx_weight or ctx_host for this reason. */
            {
                struct ggml_context *wctx = c->ctx_weight ? c->ctx_weight : c->ctx;
                out = onnx_new_tensor_nd(wctx, a->type, out_ne, nd);
            }
            ggml_set_input(out);

            /* Every other deferred list checks this bound before appending;
             * this one did not, so a model with more strided slices than the
             * array holds wrote past its end -- into whatever field of the
             * context struct follows.  That corruption surfaces later and
             * elsewhere, as a failure inside malloc. */
            if (c->n_slice_fills >= ONNX_MAX_DEFERRED) {
                fprintf(stderr, "[onnx] too many strided slices (>%d) -- '%s' skipped\n",
                        ONNX_MAX_DEFERRED, n->outputs[0]);
            } else {
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
                    /* Zeroed: an output element whose source index falls
                     * outside the known values is skipped below, and would
                     * otherwise be published as whatever the stack held. */
                    int64_t result[ONNX_MAX_DIMS] = {0};
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
                            if (out_stride[d] <= 0) continue;  /* empty slice */
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
            if (!si) si = find_constant_tensor(c->onnx, n->inputs[3]);
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
                /* Exporters emit these as a Constant node just as often as an
                 * initializer -- yolov8n's two upsamples do -- and reading
                 * only the latter left Resize a no-op: the tensor came out
                 * the same size it went in, and the Concat downstream failed
                 * on axes that had never been resized. */
                if (!sci) sci = find_constant_tensor(c->onnx, n->inputs[scales_idx]);
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
                    /* The cval map holds numbers, not bit patterns: a float
                     * initializer is stored with (int64_t)value, so a scale of
                     * 2.0 arrives here as 2.  Reinterpreting the int64 as float
                     * bits instead read 2 as a denormal near 2e-45, multiplied
                     * every extent to zero, and the clamp below turned that into
                     * 1 -- which is how all nine of MaskRCNN's FPN upsamples
                     * collapsed a [7,7,256] feature map to a single element and
                     * then broadcast that scalar back over the level it was
                     * meant to be added to. */
                    for (int d = 0; d < ncv && d < 4; d++) {
                        int ggml_d = ncv - 1 - d;
                        target_ne[ggml_d] = a->ne[ggml_d] * cv[d];
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

        if (onnx_trace_nodes()) {
            fprintf(stderr, "[Expand] %s: in='%s' a.ne=[%lld,%lld,%lld,%lld] shape_src='%s' resolved=[",
                    n->outputs[0], n->inputs[0],
                    (long long)a->ne[0], (long long)a->ne[1],
                    (long long)a->ne[2], (long long)a->ne[3], n->inputs[1]);
            for (int d = 0; d < orig_expand_ndims; d++)
                fprintf(stderr, "%lld%s", (long long)orig_expand_shape[d],
                        d < orig_expand_ndims - 1 ? "," : "");
            fprintf(stderr, "]\n");
        }

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
