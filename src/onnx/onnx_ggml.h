/* onnx_ggml.h — Map ONNX graph to ggml computation graph
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#ifndef ONNX_GGML_H
#define ONNX_GGML_H

#include "onnx_loader.h"
#include "rel_pos_bias.h"
#include "roi_align.h"
#include "nms.h"
#include "qconv_i32.h"
#include "qmatmul_i32.h"
#include "../ggml.h"
#include "../ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of deferred fill entries (Shape, Const, NonZero, etc.).
 * Transformer models with many attention heads can exceed 64 easily. */
#define ONNX_MAX_DEFERRED 512

/* Constant nodes get their own, much larger limit.  Their count scales with
 * the size of the model rather than with the number of ops of one kind:
 * roberta alone emits over 512, every axis, shape and index in the graph
 * being one.  Overflowing this is not a soft failure -- a Constant left out
 * of the fill list keeps whatever its buffer held, and a Gather index read
 * from it lands outside the table it addresses.
 *
 * Kept separate from ONNX_MAX_DEFERRED because the arrays that constant uses
 * are two pointers each, while the ones sharing that limit carry up to three
 * int64[GGML_MAX_DIMS] rows apiece. */
#define ONNX_MAX_CONSTANTS 8192

/* Nodes whose src[] can be snapshotted around one segment's allocation.
 *
 * Sized against the graphs themselves rather than guessed: the largest
 * segment MaskRCNN-12-int8 builds holds 1216 nodes (ONNX_TRACE_SGRAPH=1),
 * and only nodes of the segment currently being built are ever recorded, so
 * this bounds one segment, not the model. Overflow is reported and the
 * snapshot abandoned rather than truncated -- a half-restored graph would
 * silently keep some scheduler copies and lose others. */
#define ONNX_MAX_SRC_SNAPSHOT 4096

/* ── Data-dependent shape segmentation ─────────────────────────────
 * Some ONNX ops have an output shape that depends on the VALUES of their
 * input, not just its shape -- NonZero is the canonical case (the count of
 * non-zero elements is unknowable until the input is computed).  ggml graphs
 * are built with static shapes, so such a shape cannot be known while the
 * graph is being built.
 *
 * The graph is therefore cut into segments at these ops: each segment is
 * built and executed in turn, and the real shape is read back from the
 * finished segment before the next one is built.  Segments are numbered from
 * 0; a model with no data-dependent op has n_segments == 0 and takes the
 * original single-pass path unchanged.
 *
 * Cut points are grouped into WAVES: data-dependent ops whose inputs do not
 * depend on any other not-yet-executed data-dependent op belong to the same
 * wave and are resolved by one execution.  MaskRCNN's twelve NonZero nodes
 * form six waves (the five FPN levels plus the detection head), so it needs
 * six segments rather than twelve. */
#define ONNX_MAX_SEGMENTS 32

/* Backend buffers held for segment-added weights and boundary copies.
 *
 * Deliberately far larger than ONNX_MAX_SEGMENTS: both producers run once per
 * segment per RUN, so the count grows with how many times the model is asked
 * for a prediction, not with the size of the graph.  Running out here aborts
 * an inference that would otherwise have succeeded, which is why the headroom
 * is generous; the entries themselves are one pointer each. */
#define ONNX_MAX_WEIGHT_BUFS 4096

/* Ops whose output shape depends on input values.  Adding an op here makes
 * the pre-pass cut the graph at it; it does NOT implement the op itself. */
#define ONNX_SHAPE_DEPENDENT_OPS { "NonZero", "NonMaxSuppression", "TopK" }

typedef struct {
    /* Node range [first_node, last_node] of onnx->nodes[] built in this
     * segment.  Ranges are contiguous and cover every node in order. */
    int first_node;
    int last_node;

    /* Indices into onnx->nodes[] of the data-dependent ops resolved at the
     * END of this segment (one wave).  Empty for the final segment, which
     * has no cut point after it. */
    int cut_nodes[ONNX_MAX_DEFERRED];
    int n_cut_nodes;
} onnx_segment_t;

/* ── ONNX→ggml model context ───────────────────────────────────── */

typedef struct {
    onnx_model_t       *onnx;        /* parsed ONNX model (owns mmap) */
    struct ggml_context *ctx;         /* ggml context for graph + compute tensors */
    struct ggml_context *ctx_weight;  /* ggml context for weight tensors (separate lifetime) */
    struct ggml_cgraph  *graph;       /* computation graph */

    /* Weight buffer — allocated once, never touched by sched */
    ggml_backend_buffer_t weight_buf; /* GPU (or CPU) buffer holding all weights */

    /* Extra weight buffers from later ggml_backend_alloc_ctx_tensors() calls.
     *
     * ctx_weight keeps growing during segmented execution: every segment can
     * add Constant/scalar/NonZero/NMS tensors to it.  alloc_ctx_tensors skips
     * tensors that already have data and returns a NEW buffer covering only
     * the ones added since the last call (NULL when nothing was added), so
     * each call's buffer has to be kept to be freed later -- keeping only the
     * latest would leak every earlier one. weight_buf above is the first. */
    /* Sized well past the segment count, because this fills up per RUN, not
     * per segment: onnx_ggml_run walks the segments again on every call and
     * allocates afresh each time, so a model with 24 segments would exhaust a
     * 32-slot budget on its second inference.  The buffers cannot simply be
     * freed between runs -- they hold the weights and constants those
     * segments' tensors point at, and freeing them is the very defect fixed
     * for orphan inputs above. */
    ggml_backend_buffer_t extra_weight_bufs[ONNX_MAX_WEIGHT_BUFS];
    int                   n_extra_weight_bufs;

    /* Small parameter tensors that only a CPU-only kernel ever reads, kept in
     * their own context so they can be allocated on the HOST while ctx_weight
     * goes to the GPU.
     *
     * ggml_set_input() is not enough to keep such a tensor on the host. The
     * scheduler does force GGML_TENSOR_FLAG_INPUT tensors to the CPU backend
     * (ggml-backend.cpp:918), but the check above it wins: a tensor that
     * already sits in a buffer cannot be moved at all, and everything in
     * ctx_weight is pre-allocated on the weight backend.
     *
     * The cost of getting this wrong is not bandwidth -- these are 16 bytes --
     * it is SPLIT INPUT SLOTS. Every tensor a CPU split has to fetch from the
     * GPU occupies one of GGML_SCHED_MAX_SPLIT_INPUTS, and MaskRCNN runs ~42
     * NonMaxSuppression nodes in one split, each pulling boxes, scores and its
     * own nms_params across. Measured: 128 inputs, all of them under 256 bytes,
     * 0.00 MB in total -- the graph aborted on slot exhaustion while moving
     * essentially no data. Hosting the params alone takes that split from 128
     * to 86. */
    struct ggml_context  *ctx_host;
    ggml_backend_buffer_t host_buf;

    /* Host-visible pinned staging buffer for fast CPU→GPU input transfer.
     * Data is memcpy'd here, then ggml_backend_tensor_set detects pinned src
     * and does direct DMA (no intermediate staging copy). */
    ggml_backend_buffer_t pinned_buf; /* pinned staging buffer (NULL if unavailable) */
    void                 *pinned_ptr; /* mapped pointer into pinned_buf */
    size_t                pinned_size;/* allocated size in bytes */

    /* Scheduler with CPU fallback for unsupported Vulkan ops */
    ggml_backend_sched_t sched;       /* scheduler (owns compute buffer allocation) */
    ggml_backend_t       backend_gpu; /* Vulkan backend (NULL if CPU-only) */
    ggml_backend_t       backend_cpu; /* CPU backend (always present) */

    /* Name → ggml_tensor lookup for wiring nodes */
    struct ggml_tensor **tensor_map_vals;
    char              (*tensor_map_keys)[ONNX_MAX_NAME];
    int                *tensor_map_ndims;  /* original ONNX ndims (for >4D axis mapping) */
    int64_t           (*tensor_map_onnx_ne)[ONNX_MAX_DIMS]; /* full ONNX shape (up to 8D) */
    /* Logically empty: the ONNX semantics say this tensor has zero rows, but
     * ggml cannot represent ne=0, so it is carried as a one-element tensor.
     * Set by NonZero when the measured count is zero; without it the unused
     * element reads back as index 0, which is a perfectly valid row number and
     * therefore becomes a detection nothing selected.  A separate flag rather
     * than a sentinel index: -1 already means "last element" to ONNX Gather. */
    unsigned char      *tensor_map_empty;
    int                 tensor_map_size;
    int                 tensor_map_cap;

    /* Deferred data for Shape op outputs (filled after sched alloc) */
    struct ggml_tensor *shape_tensor_ptrs[ONNX_MAX_DEFERRED];
    int64_t             shape_tensors_ne[ONNX_MAX_DEFERRED][ONNX_MAX_DIMS + 1]; /* [0]=ndims, [1..]=dims */
    int                 n_shape_tensors;

    /* Deferred data for ConstantOfShape + scalar constants (filled after sched alloc) */
    struct ggml_tensor *const_fill_ptrs[ONNX_MAX_DEFERRED];
    float               const_fill_vals[ONNX_MAX_DEFERRED];
    int                 n_const_fills;

    /* Deferred payload for Constant nodes whose data lives in the node's
     * "value" attribute rather than in graph.initializer.
     *
     * load_weights() walks onnx->initializers[], and the loader fills that
     * array only from GP_INITIALIZER -- a Constant node's tensor is never in
     * it.  Such a tensor therefore got a buffer and no contents: an index read
     * out of it was whatever the allocation happened to hold, which is how a
     * Gather index came back as 25711 against a 128-row table.
     *
     * Filled from fill_deferred_tensors() rather than from load_weights() so
     * that segmented execution reloads them too: every segment reallocates the
     * scheduler buffers, and a constant living in one of them is blank again
     * afterwards. */
    struct ggml_tensor *cinit_fill_ptrs[ONNX_MAX_CONSTANTS];
    const onnx_initializer_t *cinit_fill_srcs[ONNX_MAX_CONSTANTS];
    int                 n_cinit_fills;

    /* Deferred data for EyeLike (identity matrix, filled after sched alloc) */
    struct ggml_tensor *eye_fill_ptrs[ONNX_MAX_DEFERRED];
    int                 eye_fill_rows[ONNX_MAX_DEFERRED]; /* ggml ne[1] */
    int                 eye_fill_cols[ONNX_MAX_DEFERRED]; /* ggml ne[0] */
    int                 eye_fill_k[ONNX_MAX_DEFERRED];    /* diagonal offset */
    int                 n_eye_fills;

    /* Deferred strided Slice (step != 1): copy src→dst with stride after alloc */
    struct ggml_tensor *slice_fill_src[ONNX_MAX_DEFERRED];
    struct ggml_tensor *slice_fill_dst[ONNX_MAX_DEFERRED];
    int64_t             slice_fill_starts[ONNX_MAX_DEFERRED][GGML_MAX_DIMS];  /* per-ggml-dim start offsets */
    int64_t             slice_fill_steps[ONNX_MAX_DEFERRED][GGML_MAX_DIMS];   /* per-ggml-dim step values */
    int64_t             slice_fill_out_ne[ONNX_MAX_DEFERRED][GGML_MAX_DIMS];  /* output ne per ggml dim */
    int                 slice_fill_ndims[ONNX_MAX_DEFERRED];      /* onnx ndims */
    int                 n_slice_fills;

    /* Deferred NonZero (filled after sched alloc: read input, write indices of non-zero elems) */
    struct ggml_tensor *nonzero_fill_src[ONNX_MAX_DEFERRED];   /* input tensor */
    struct ggml_tensor *nonzero_fill_dst[ONNX_MAX_DEFERRED];   /* output tensor [n_dims_input, nnz] in ggml layout */
    int                 nonzero_fill_ndims[ONNX_MAX_DEFERRED]; /* ONNX ndims of input */
    int                 n_nonzero_fills;

    /* Values carried across a cut op's re-map.
     *
     * A cut op is rebuilt at its measured size once its own segment has run,
     * and that replaces the tensor holding the result. Deferred-fill ops do
     * not care: fill_deferred_tensors() rewrites them after every allocation.
     * NonMaxSuppression does -- only its PARAMS are deferred, the output is a
     * computed ggml_custom_4d node that nothing evaluates a second time -- so
     * its values are saved before the re-map and written back once the
     * rebuilt tensor has memory.
     *
     * ⚠️ GGML_OP_CUSTOM only. Carrying TopK outputs too (they reach tmap
     * through a RESHAPE, so `op != GGML_OP_NONE` catches them) writes stale
     * values over a deferred fill and corrupts the run. */
    struct ggml_tensor *cut_carry_dst[ONNX_MAX_DEFERRED];
    void               *cut_carry_buf[ONNX_MAX_DEFERRED];
    size_t              cut_carry_bytes[ONNX_MAX_DEFERRED];
    int                 n_cut_carry;

    /* Original src[] of every node in a segment's graph, taken before the
     * scheduler is allowed to rewrite them.
     *
     * ggml_backend_sched_split_graph() replaces a src that lives on another
     * backend with a copy of its own ("Vulkan0#<name>#0"), writing it straight
     * into the node: `node->src[j] = tensor_id_copy(...)` in ggml-backend.cpp.
     * A segmented model reuses the SAME tensor objects for the next segment,
     * so that rewrite is still there when the next graph is built -- and the
     * copy has no view_src, so ggml_build_forward_expand() stops at it instead
     * of walking through to the op that produces the data.
     *
     * Measured on MaskRCNN-12-int8 (Vulkan): segment 17's '1170 (permuted)
     * (cont)' had src0 = "Vulkan0#1170 (permuted)#0", the NMS node that feeds
     * it was therefore absent from the graph, the kernel ran 2 times instead
     * of 33, and the cont read a buffer that had been handed to something
     * else. On CPU there is no second backend, no copy is made, and segment 17
     * holds the NMS node exactly as segment 16 does.
     *
     * Restoring the saved pointers before the next graph is built puts the
     * dependency walk back on the real producer. */
    struct ggml_tensor *src_snap_node[ONNX_MAX_SRC_SNAPSHOT];
    struct ggml_tensor *src_snap_src[ONNX_MAX_SRC_SNAPSHOT][GGML_MAX_SRC];
    int                 n_src_snap;

    /* Compile-time known values for shape tensors (Shape, Constant, Slice, Concat outputs).
     * Used by Reshape/Expand/etc. to determine target shape at graph build time. */
    char              (*cval_keys)[ONNX_MAX_NAME];
    int64_t           (*cval_data)[ONNX_MAX_DIMS]; /* values (not dims!) */
    int                *cval_lens;                  /* number of values */
    int                 cval_size;
    int                 cval_cap;

    int                 is_allocated;   /* 1 after first sched alloc + deferred fill */

    /* Segmentation for data-dependent shapes.  n_segments == 0 means the
     * model has no such op and takes the original single-pass path -- the
     * fast path that all 14 working models use, left untouched. */
    onnx_segment_t      segments[ONNX_MAX_SEGMENTS];
    int                 n_segments;

    /* Segmented execution state (only used when n_segments > 1).
     *
     * ctx_boundary holds a copy of every tensor that has to outlive its own
     * segment: a segment's ggml context and scheduler buffers are torn down
     * once it has run, so anything a later segment reads must first be copied
     * somewhere the teardown does not touch.  This mirrors what ctx_weight
     * already does for initializers -- tensors allocated there have ->buffer
     * set, so the scheduler skips them and never aliases over their data. */
    struct ggml_context  *ctx_boundary;
    ggml_backend_buffer_t boundary_buf;

    /* Real sizes recovered from a finished segment, keyed by the ONNX name of
     * the op that produced them.
     *
     * A data-dependent op is built with a guess (NonZero assumes every element
     * is non-zero) because its true size cannot be known until its input has
     * been computed.  Once the segment holding it has run, the input IS known,
     * so the count is measured and recorded here; the next segment's mapping
     * reads it back instead of guessing again. */
    #define ONNX_MAX_RESOLVED 512
    char   resolved_names[ONNX_MAX_RESOLVED][ONNX_MAX_NAME];
    int64_t resolved_sizes[ONNX_MAX_RESOLVED];
    int    n_resolved;

    /* One scheduler per segment, so each segment keeps its own allocation.
     *
     * A single shared scheduler cannot do this: ggml_gallocr_needs_realloc
     * compares a graph against the last one allocated, segments differ in node
     * count, so every segment transition re-reserves the arena and frees the
     * buffers the previous segment's tensors point into -- losing them within
     * one run, not only between runs.
     *
     * Giving each segment its own scheduler makes the allocation survive until
     * that segment has computed.  The cost is one arena per segment. */
    ggml_backend_sched_t seg_scheds[ONNX_MAX_SEGMENTS];
    /* Segment 0's graph, which the build produced and the segment loop used
     * to overwrite: ctx->graph is reassigned for each later segment, so after
     * one run the pointer named the LAST segment and a second run computed
     * that instead of the first.  Saved here so the next run can put it back. */
    struct ggml_cgraph *seg0_graph;

    /* The context a segment's graph and intermediates are built in.
     *
     * c->ctx points here while a segment is being mapped, so every op builder
     * -- which all allocate from c->ctx -- puts this segment's tensors in a
     * pool that can be thrown away once the segment has computed. Without it
     * the mapping's tensors accumulate for the life of the model: a ggml
     * context frees nothing until it is destroyed, and re-mapping every
     * segment on every run added ~195 constants and ~193 shape tensors per
     * inference, which exhausted the pool within a handful of predictions.
     *
     * Only tensors nothing outlives the segment may live here. Weights,
     * constants and the outputs of cut ops go to ctx_weight or ctx_host;
     * whatever a later segment reads is copied into ctx_boundary first. */
    struct ggml_context *ctx_seg;
    struct ggml_context *ctx_prev;   /* the segment before it, still open */
    struct ggml_context *ctx_main;   /* c->ctx outside a segment */
    size_t               seg_ctx_size;  /* bytes per segment pool */

    /* Copies queued by one segment: the destination tensors have to be given
     * memory before any data can be written into them, so the pairs are
     * collected first and filled once the buffer exists. */
    #define ONNX_MAX_BOUNDARY 1024
    struct {
        struct ggml_tensor *src;
        struct ggml_tensor *dst;
    } boundary_pending[ONNX_MAX_BOUNDARY];
    int                 n_boundary_pending;

    /* Index of the segment being built, so map_node and the op handlers can
     * tell which pass they are in.  -1 outside segmented execution. */
    int                 cur_segment;

    /* FP16 inference mode: 0 = F32 (default), 1 = F16 for large weights */
    int                 model_dtype;    /* GGML_TYPE_F32 or GGML_TYPE_F16 */

    /* RelPosBias2D fused op blocks (BoTNet pos_embed subgraphs) */
    #define ONNX_MAX_POS_EMBED 8
    struct {
        char   x_input_name[ONNX_MAX_NAME];   /* x tensor before first Reshape */
        char   wh_name[ONNX_MAX_NAME];         /* W_h initializer name */
        char   ww_name[ONNX_MAX_NAME];         /* W_w initializer name */
        char   output_name[ONNX_MAX_NAME];     /* final Reshape output name */
        int    first_node_idx;                  /* index of first node in block */
        int    last_node_idx;                   /* index of last node (final Reshape) */
        rel_pos_bias_params_t params;           /* H, W, B, C, rel_h, rel_w */
        struct ggml_tensor *x_cpu_tensor;       /* ggml_cont copy of x — pinned to CPU */
    } pos_embed_blocks[ONNX_MAX_POS_EMBED];
    int n_pos_embed_blocks;

    /* Storage for rel_pos_bias params (must outlive graph compute) */
    rel_pos_bias_params_t *pos_embed_params;    /* malloc'd array, freed in onnx_ggml_free */

    /* RoiAlign custom op params (must outlive graph compute) */
    roi_align_params_t **roi_align_params;      /* malloc'd array of malloc'd entries */
    int                 n_roi_aligns;
    int                 roi_align_params_cap;

    /* NMS custom op params (must outlive graph compute).
     *
     * An array of pointers, not of structs: each entry's address is handed to
     * ggml_map_custom3 as the kernel's userdata and has to stay valid for the
     * life of the graph.  Growing an array of structs with realloc moves it,
     * which leaves every userdata pointer already given out dangling -- see
     * the note in onnx_ops_special.c. */
    nms_params_t      **nms_params;             /* malloc'd array of malloc'd entries */
    int                 n_nms_ops;
    int                 nms_params_cap;

    /* Userdata for the exact-integer QLinearConv path, same ownership rule as
     * nms_params above: individually malloc'd, pointers kept so the context
     * can free them, never realloc'd as a block (that would dangle every
     * pointer already handed to an op). */
    void              **qconv_params;
    int                 n_qconv_ops;
    int                 qconv_params_cap;

    /* Per-output-channel requantisation multipliers for GGML_OP_QCONV_I32,
     * one tensor per node, filled after allocation.
     *
     * The values are computed here, on the host, once -- deliberately, not in
     * the kernels. The C kernel and the Vulkan shader evaluating
     * x_scale*w_scale[oc]/y_scale independently landed one ulp apart on 976 of
     * 4096 channels, which moved the rounded output a whole quantisation code
     * on 584 elements of one node and cost nine of ten detections by the end
     * of the graph.
     *
     * Refilled from fill_deferred_tensors() like every other deferred payload,
     * because segmented execution reallocates the scheduler's buffers and a
     * tensor living in one of them is blank again afterwards. */
    struct ggml_tensor *qconv_mult_tensors[ONNX_MAX_DEFERRED];
    float              *qconv_mult_values[ONNX_MAX_DEFERRED];
    int                 qconv_mult_n[ONNX_MAX_DEFERRED];
    int                 n_qconv_mult;

    /* Deferred NMS output sizing (filled after sched alloc) */
    struct ggml_tensor *nms_param_tensors[ONNX_MAX_DEFERRED]; /* param tensors to fill */
    int                 nms_max_boxes[ONNX_MAX_DEFERRED];
    float               nms_iou_thresh[ONNX_MAX_DEFERRED];
    float               nms_score_thresh[ONNX_MAX_DEFERRED];
    int                 nms_have_score_thresh[ONNX_MAX_DEFERRED]; /* input present? */
    int                 n_nms_deferred;

    /* First node map_node declined, and its op.  A declined node leaves its
     * output unregistered, so everything downstream of it goes unbuilt too --
     * the later failures are consequences and the first one is the cause. */
    char                first_failed_node[ONNX_MAX_NAME];
    char                first_failed_op[64];

    /* Orphan-input CPU buffers allocated in sched_alloc_and_fill (one per
     * unbuffered real input). Freed and reset on each re-alloc and at ctx free. */
    ggml_backend_buffer_t orphan_input_bufs[ONNX_MAX_DEFERRED];
    /* The tensor each of those buffers backs.  Freeing a buffer leaves its
     * tensor pointing into memory the allocator has taken back, and the next
     * write through that tensor corrupts whatever now owns it; ggml also
     * asserts buffer == NULL when handing a tensor a new one.  Both are
     * avoided by clearing the tensor at the same moment the buffer goes. */
    struct ggml_tensor   *orphan_input_tensors[ONNX_MAX_DEFERRED];
    int                   n_orphan_input_bufs;
} onnx_ggml_ctx_t;

/* Minimum number of elements for a weight tensor to be stored in FP16.
 * Smaller tensors (bias, scalars, BN params) stay F32 for numerical stability. */
#define ONNX_FP16_MIN_ELEMENTS 256

/* Build ggml graph from parsed ONNX model.
 * device: "vulkan" or "cpu" (NULL defaults to vulkan if available, else cpu).
 * Returns NULL on failure. */
/* model_dtype: GGML_TYPE_F32 (default) or GGML_TYPE_F16 for half-precision weights. */
onnx_ggml_ctx_t *onnx_ggml_build(onnx_model_t *onnx, const char *device, int n_threads,
                                  enum ggml_type model_dtype);

/* Run inference. input_data/input_names: arrays of length n_inputs.
 * Each input_data[i] is a flat float array matching the model input shape.
 * Returns 0 on success. */
/* Run the model.
 *
 * input_lens gives the element count of each array in input_data.  It is not
 * optional: segmented execution writes the inputs once per segment, and a
 * tensor can be rebuilt at a different size in between, so the tensor's own
 * size is not a safe bound for how much of the caller's array may be read. */
int onnx_ggml_run(onnx_ggml_ctx_t *ctx,
                  const char **input_names, const float **input_data,
                  const int64_t *input_lens, int n_inputs);

/* Get output tensor by index. */
struct ggml_tensor *onnx_ggml_output(onnx_ggml_ctx_t *ctx, int index);

/* Free everything. */
void onnx_ggml_free(onnx_ggml_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_GGML_H */
