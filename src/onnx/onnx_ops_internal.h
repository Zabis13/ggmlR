/* onnx_ops_internal.h — shared internals for onnx_ops_*.c split files
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 *
 * NOT a public API. Included only by onnx_ggml.c and onnx_ops_*.c.
 * Declares helper functions that live in onnx_ggml.c and are used
 * by the split op files.
 */

#ifndef ONNX_OPS_INTERNAL_H
#define ONNX_OPS_INTERNAL_H

#include "onnx_ggml.h"
#include "../ggml.h"
#include "../ggml-alloc.h"
#include "../ggml-backend.h"
#include "../ggml-cpu.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#ifdef GGML_USE_VULKAN
#include "../ggml-vulkan.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Helpers declared in onnx_ggml.c, used by ops files ─────────── */

/* RANK RULE, learned four times the hard way:
 *
 *   ONNX rank comes from tmap_get_ndims(), never from ggml_n_dims().
 *
 * ggml_n_dims() reports the physical shape, and it collapses trailing unit
 * axes unconditionally: an ONNX [9408,1] arrives as ne=[9408,1,1,1] and
 * answers 1, not 2.  Anything that converts an ONNX axis to a ggml one --
 * the usual `ggml_dim = rank - 1 - axis` -- is then off by one, and the error
 * is invisible, because the shape it produces is perfectly plausible.
 *
 * Four operators were wrong this way, each found only after the model built
 * a graph that ran and returned the wrong answer:
 *   TopK           measured 1 candidate where there were 9408
 *   Squeeze        withheld the rank when every remaining dim was 1
 *   Gather         selected along an axis of length 1 -- silently returning
 *                  the whole tensor on 36 nodes, aborting on 5
 *   QLinearConcat  read the rank off the shape and concatenated wrongly
 *
 * Use ggml_n_dims() only as a fallback when the map has nothing (returns
 * <= 0), and only where the tensor was built here rather than named by the
 * model.  For an ONNX axis attribute, the rank it is expressed against is
 * the DECLARED one, which is what the map carries. */
struct ggml_tensor *tmap_get(onnx_ggml_ctx_t *c, const char *name);
int                 tmap_get_ndims(onnx_ggml_ctx_t *c, const char *name);
void                tmap_put(onnx_ggml_ctx_t *c, const char *name, struct ggml_tensor *t);
void                tmap_put_nd(onnx_ggml_ctx_t *c, const char *name,
                                struct ggml_tensor *t, int onnx_ndims);
void                tmap_put_shape(onnx_ggml_ctx_t *c, const char *name,
                                   struct ggml_tensor *t,
                                   const int64_t *onnx_shape, int onnx_ndims);
int                 tmap_get_shape(onnx_ggml_ctx_t *c, const char *name,
                                   int64_t *shape, int max_dims);
/* Logically empty tensors: ONNX says zero rows, ggml carries one.  An operator
 * that is a pure pass-through, view or index of a single input inherits the
 * mark from that input; Concat drops marked inputs and marks its own output
 * only when every input was marked.  Anything not on that list does not
 * inherit -- it has to decide what empty means for itself. */
void                tmap_mark_empty(onnx_ggml_ctx_t *c, const char *name);
int                 tmap_is_empty(onnx_ggml_ctx_t *c, const char *name);

void                cval_put(onnx_ggml_ctx_t *c, const char *name,
                             const int64_t *vals, int n);
int                 cval_get(onnx_ggml_ctx_t *c, const char *name,
                             int64_t *out, int max_n);

struct ggml_tensor *get_input(onnx_ggml_ctx_t *c, const onnx_node_t *n, int idx);

struct ggml_tensor *onnx_reshape_nd(struct ggml_context *ctx,
                                    struct ggml_tensor *a,
                                    const int64_t *ne, int ndims);
struct ggml_tensor *onnx_new_tensor_nd(struct ggml_context *ctx,
                                       enum ggml_type type,
                                       const int64_t *ne, int ndims);
int64_t             ne_product(const int64_t *ne, int ndims);
int                 onnx_squeeze_ndims(const int64_t *ne, int ndims);

struct ggml_tensor *make_scalar(onnx_ggml_ctx_t *c, float val);

/* Widen integer operands to F32 before an elementwise binary op: ggml's
 * binary kernels are F32/F16 only and abort on i32.  Returns 1 if it
 * converted anything. */
int                 onnx_binary_promote(struct ggml_context *ctx,
                                        struct ggml_tensor **pa,
                                        struct ggml_tensor **pb);

void                onnx_broadcast_prepare(struct ggml_context *ctx,
                                           struct ggml_tensor **pa,
                                           struct ggml_tensor **pb);

const onnx_initializer_t *find_constant_tensor(const onnx_model_t *m,
                                                const char *name);

enum ggml_type onnx_dtype_to_ggml(int32_t dt);

/* Does the loader actually convert this dtype, or would its bytes be copied
 * at the wrong width?  onnx_dtype_to_ggml has to return some ggml type and
 * answers F32 for anything it does not know, so it cannot be used to tell.
 * Check this before trusting a tensor's payload. */
int            onnx_dtype_supported(int32_t dt);

/* Upload an initializer's payload into a tensor, converting ONNX dtypes ggml
 * does not have (INT64 indices, doubles) on the way.  Shared by load_weights()
 * and by the deferred fill for Constant nodes, whose data sits in a node
 * attribute and so never reaches onnx->initializers[]. */
int onnx_upload_initializer(struct ggml_tensor *t,
                            const onnx_initializer_t *init);
size_t         onnx_dtype_size(int32_t dt);

/* g_current_node — set in map_node() before each call, used for diagnostics */
extern const onnx_node_t *g_current_node;

/* onnx_trace_nodes() — nonzero when ONNX_TRACE_NODES=1; gates graph tracing */
int onnx_trace_nodes(void);

/* onnx_trace_ring() — nonzero when ONNX_TRACE_RING=1; keeps the last few graph
 * nodes with their edge values so they can be printed when a run dies. */
int onnx_trace_ring(void);

/* onnx_trace_vals() — nonzero when ONNX_TRACE_VALS=1; prints every node's
 * output as it is computed (first three values), for diffing a whole run
 * against a reference.  Thousands of lines on a real model: redirect it. */
int onnx_trace_vals(void);
/* onnx_trace_sum() — nonzero when ONNX_TRACE_SUM=1; adds a whole-tensor
 * checksum (n/sum/min/max) to each [val] line, for comparing two backends
 * on more than the three values that fit on the line. */
int onnx_trace_sum(void);

/* Print that ring.  Registered as r_ggml_abort_hook, since a ggml assertion
 * reaches R through Rf_error and never returns here. */
void onnx_ring_dump(void);
int onnx_use_segments(void);

/* Measured output size of a data-dependent op, or -1 when it has not been
 * measured yet (the first time the op is mapped, or outside segmented
 * execution).  Ops fall back to their build-time guess when it is -1. */
int64_t onnx_resolved_size(onnx_ggml_ctx_t *c, const char *name);

/* Report an unsupported op once per model load (reset by the build entry). */
void onnx_warn_unsupported_op(const char *op);
void onnx_reset_unsupported_warnings(void);

/* ── Op group dispatcher functions ──────────────────────────────── */
/* Each returns: 1 = handled, 0 = not this group's op, -1 = error */

int map_node_basic  (onnx_ggml_ctx_t *c, const onnx_node_t *n,
                     struct ggml_tensor *a, struct ggml_tensor *b,
                     struct ggml_tensor **out, int *out_nd);

int map_node_tensor (onnx_ggml_ctx_t *c, const onnx_node_t *n,
                     struct ggml_tensor *a, struct ggml_tensor *b,
                     struct ggml_tensor **out, int *out_nd);

int map_node_nn     (onnx_ggml_ctx_t *c, const onnx_node_t *n,
                     struct ggml_tensor *a, struct ggml_tensor *b,
                     struct ggml_tensor **out, int *out_nd);

int map_node_quant  (onnx_ggml_ctx_t *c, const onnx_node_t *n,
                     struct ggml_tensor *a, struct ggml_tensor *b,
                     struct ggml_tensor **out, int *out_nd);

int map_node_special(onnx_ggml_ctx_t *c, const onnx_node_t *n,
                     struct ggml_tensor *a, struct ggml_tensor *b,
                     struct ggml_tensor **out, int *out_nd);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_OPS_INTERNAL_H */
