/* onnx_loader.h — ONNX model loader for ggmlR
 * Parses .onnx protobuf files and builds ggml computation graphs.
 * Zero-copy weight loading via mmap.
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#ifndef ONNX_LOADER_H
#define ONNX_LOADER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Protobuf wire types ────────────────────────────────────────── */
#define PB_WIRE_VARINT  0
#define PB_WIRE_64BIT   1
#define PB_WIRE_LEN     2
#define PB_WIRE_32BIT   5

/* ── ONNX TensorProto.DataType ──────────────────────────────────── */
#define ONNX_DTYPE_UNDEFINED  0
#define ONNX_DTYPE_FLOAT      1
#define ONNX_DTYPE_UINT8      2
#define ONNX_DTYPE_INT8       3
#define ONNX_DTYPE_UINT16     4
#define ONNX_DTYPE_INT16      5
#define ONNX_DTYPE_INT32      6
#define ONNX_DTYPE_INT64      7
#define ONNX_DTYPE_STRING     8
#define ONNX_DTYPE_BOOL       9
#define ONNX_DTYPE_FLOAT16   10
#define ONNX_DTYPE_DOUBLE    11
#define ONNX_DTYPE_UINT32    12
#define ONNX_DTYPE_UINT64    13
#define ONNX_DTYPE_BFLOAT16  16

/* ── ONNX AttributeProto.AttributeType ──────────────────────────── */
#define ONNX_ATTR_UNDEFINED  0
#define ONNX_ATTR_FLOAT      1
#define ONNX_ATTR_INT        2
#define ONNX_ATTR_STRING     3
#define ONNX_ATTR_TENSOR     4
#define ONNX_ATTR_GRAPH      5
#define ONNX_ATTR_FLOATS     6
#define ONNX_ATTR_INTS       7
#define ONNX_ATTR_STRINGS    8
#define ONNX_ATTR_TENSORS    9
#define ONNX_ATTR_GRAPHS    10

/* ── Limits ─────────────────────────────────────────────────────── */
#define ONNX_MAX_DIMS        8
#define ONNX_MAX_NAME      256
#define ONNX_MAX_INPUTS     16
#define ONNX_MAX_OUTPUTS     8
#define ONNX_MAX_ATTRS      32

/* ── Parsed structures ──────────────────────────────────────────── */

/* Forward declaration for use in onnx_attr_t */
typedef struct onnx_initializer onnx_initializer_t;

typedef struct {
    char     name[ONNX_MAX_NAME];
    int32_t  type;       /* ONNX_ATTR_* */
    float    f;          /* ONNX_ATTR_FLOAT */
    int64_t  i;          /* ONNX_ATTR_INT */
    /* For ONNX_ATTR_STRING: ptr+len into mmap'd buffer */
    const uint8_t *s_data;
    size_t         s_len;
    /* For ONNX_ATTR_INTS: owned array (repeated or packed) */
    int64_t       *ints;
    int             n_ints;
    /* For ONNX_ATTR_FLOATS */
    const float   *floats;
    int             n_floats;
    /* For ONNX_ATTR_TENSOR (Constant op value) */
    onnx_initializer_t *tensor;  /* owned, NULL if not set */
} onnx_attr_t;

typedef struct {
    char     name[ONNX_MAX_NAME];
    char     op_type[128];
    char     domain[128];
    /* Input/output tensor names */
    char     inputs[ONNX_MAX_INPUTS][ONNX_MAX_NAME];
    int      n_inputs;
    char     outputs[ONNX_MAX_OUTPUTS][ONNX_MAX_NAME];
    int      n_outputs;
    /* Attributes */
    onnx_attr_t attrs[ONNX_MAX_ATTRS];
    int         n_attrs;
} onnx_node_t;

struct onnx_initializer {
    char     name[ONNX_MAX_NAME];
    int32_t  data_type;      /* ONNX_DTYPE_* */
    int64_t  dims[ONNX_MAX_DIMS];
    int      n_dims;
    /* Zero-copy pointer into mmap'd file (raw_data) */
    const uint8_t *raw_data;
    size_t         raw_size;
    /* If data is in float_data/int32_data/int64_data fields (not raw_data),
       we decode into a malloc'd buffer */
    void    *decoded_data;
    size_t   decoded_size;
};

typedef struct {
    char     name[ONNX_MAX_NAME];
    int32_t  elem_type;      /* ONNX_DTYPE_* */
    int64_t  dims[ONNX_MAX_DIMS];
    int      n_dims;
} onnx_value_info_t;

typedef struct {
    /* Model metadata */
    int64_t  ir_version;
    int64_t  opset_version;
    char     producer_name[ONNX_MAX_NAME];
    char     graph_name[ONNX_MAX_NAME];

    /* Graph inputs/outputs */
    onnx_value_info_t *inputs;
    int                n_inputs;
    onnx_value_info_t *outputs;
    int                n_outputs;

    /* Nodes (ops) */
    onnx_node_t       *nodes;
    int                n_nodes;

    /* Initializers (weights) — zero-copy into mmap'd file */
    onnx_initializer_t *initializers;
    int                 n_initializers;

    /* mmap state */
    uint8_t  *mmap_data;
    size_t    mmap_size;
    int       mmap_fd;
} onnx_model_t;


/* ── API ────────────────────────────────────────────────────────── */

/* Load and parse an .onnx file. Returns NULL on failure.
 * The file is mmap'd; weights point directly into it. */
onnx_model_t *onnx_load(const char *path);

/* Free all resources (including munmap). */
void onnx_free(onnx_model_t *model);

/* Find an initializer by name. Returns NULL if not found. */
const onnx_initializer_t *onnx_find_initializer(const onnx_model_t *model,
                                                  const char *name);

/* Find an attribute by name in a node. Returns NULL if not found. */
const onnx_attr_t *onnx_node_find_attr(const onnx_node_t *node,
                                        const char *name);

/* Get attribute as int with default value. */
int64_t onnx_attr_int(const onnx_node_t *node, const char *name, int64_t def);

/* Get attribute as float with default value. */
float onnx_attr_float(const onnx_node_t *node, const char *name, float def);

/* Get attribute as string. Returns length, writes to out[]. */
int onnx_attr_str(const onnx_node_t *node, const char *name,
                  char *out, int max_len);

/* Get attribute as int array. Returns count, writes to out[]. */
int onnx_attr_ints(const onnx_node_t *node, const char *name,
                   int64_t *out, int max_count);

#ifdef __cplusplus
}
#endif

#endif /* ONNX_LOADER_H */
