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

void                onnx_broadcast_prepare(struct ggml_context *ctx,
                                           struct ggml_tensor **pa,
                                           struct ggml_tensor **pb);

const onnx_initializer_t *find_constant_tensor(const onnx_model_t *m,
                                                const char *name);

enum ggml_type onnx_dtype_to_ggml(int32_t dt);
size_t         onnx_dtype_size(int32_t dt);

/* g_current_node — set in map_node() before each call, used for diagnostics */
extern const onnx_node_t *g_current_node;

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
