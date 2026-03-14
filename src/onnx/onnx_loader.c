/* onnx_loader.c — ONNX protobuf parser + ggml graph builder
 *
 * Direct protobuf wire format parsing (no protobuf library dependency).
 * Zero-copy weight loading via mmap.
 *
 * Copyright (c) 2026 ggmlR authors. MIT License.
 */

#include "onnx_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ── Protobuf wire format reader ────────────────────────────────── */

typedef struct {
    const uint8_t *data;
    const uint8_t *end;
    const uint8_t *cur;
} pb_reader_t;

static void pb_init(pb_reader_t *r, const uint8_t *data, size_t len) {
    r->data = data;
    r->cur  = data;
    r->end  = data + len;
}

static int pb_eof(const pb_reader_t *r) {
    return r->cur >= r->end;
}

static uint64_t pb_read_varint(pb_reader_t *r) {
    uint64_t val = 0;
    int shift = 0;
    while (r->cur < r->end) {
        uint8_t b = *r->cur++;
        val |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) return val;
        shift += 7;
        if (shift >= 64) break;
    }
    return val;
}

static uint32_t pb_read_fixed32(pb_reader_t *r) {
    if (r->cur + 4 > r->end) { r->cur = r->end; return 0; }
    uint32_t v;
    memcpy(&v, r->cur, 4);
    r->cur += 4;
    return v;
}

static uint64_t pb_read_fixed64(pb_reader_t *r) {
    if (r->cur + 8 > r->end) { r->cur = r->end; return 0; }
    uint64_t v;
    memcpy(&v, r->cur, 8);
    r->cur += 8;
    return v;
}

/* Read a tag, return field_number and wire_type. Returns 0 on EOF. */
static uint32_t pb_read_tag(pb_reader_t *r, int *wire_type) {
    if (pb_eof(r)) return 0;
    uint64_t t = pb_read_varint(r);
    *wire_type = (int)(t & 0x07);
    return (uint32_t)(t >> 3);
}

/* Get length-delimited field as sub-reader. */
static int pb_read_submsg(pb_reader_t *r, pb_reader_t *sub) {
    uint64_t len = pb_read_varint(r);
    if (r->cur + len > r->end) { r->cur = r->end; return -1; }
    sub->data = r->cur;
    sub->cur  = r->cur;
    sub->end  = r->cur + len;
    r->cur += len;
    return 0;
}

/* Skip a field based on wire type. */
static void pb_skip(pb_reader_t *r, int wire_type) {
    switch (wire_type) {
        case PB_WIRE_VARINT: pb_read_varint(r); break;
        case PB_WIRE_64BIT:  r->cur += 8; break;
        case PB_WIRE_32BIT:  r->cur += 5; break; /* group end (deprecated) */
        case PB_WIRE_LEN: {
            uint64_t len = pb_read_varint(r);
            r->cur += len;
            break;
        }
        default: r->cur = r->end; break;
    }
    if (r->cur > r->end) r->cur = r->end;
}

/* Copy a length-delimited string into a fixed-size buffer. */
static void pb_read_string(pb_reader_t *r, char *buf, size_t bufsize) {
    uint64_t len = pb_read_varint(r);
    if (r->cur + len > r->end) { r->cur = r->end; return; }
    size_t copy = len < bufsize - 1 ? len : bufsize - 1;
    memcpy(buf, r->cur, copy);
    buf[copy] = '\0';
    r->cur += len;
}

/* ── Dynamic array helpers ──────────────────────────────────────── */

#define DA_GROW(arr, count, cap, type) do { \
    if ((count) >= (cap)) { \
        (cap) = (cap) ? (cap) * 2 : 16; \
        (arr) = (type *)realloc((arr), (cap) * sizeof(type)); \
    } \
} while(0)

/* ── mmap / munmap cross-platform ───────────────────────────────── */

static uint8_t *onnx_mmap_file(const char *path, size_t *out_size, int *out_fd) {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(hFile, &fsize)) { CloseHandle(hFile); return NULL; }
    *out_size = (size_t)fsize.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return NULL; }
    uint8_t *data = (uint8_t *)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    *out_fd = -1;
    return data;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }
    *out_size = (size_t)st.st_size;
    uint8_t *data = (uint8_t *)mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { close(fd); return NULL; }
    *out_fd = fd;
    return data;
#endif
}

static void onnx_munmap(uint8_t *data, size_t size, int fd) {
#ifdef _WIN32
    (void)fd;
    if (data) UnmapViewOfFile(data);
#else
    if (data && data != MAP_FAILED) munmap(data, size);
    if (fd >= 0) close(fd);
#endif
}

/* ── Parse ONNX TensorProto ─────────────────────────────────────── */

/* ONNX TensorProto field numbers */
#define TP_DIMS        1   /* repeated int64, packed */
#define TP_DATA_TYPE   2   /* int32 */
#define TP_FLOAT_DATA  4   /* repeated float, packed */
#define TP_INT32_DATA  5   /* repeated int32, packed */
#define TP_STRING_DATA 6   /* repeated bytes */
#define TP_INT64_DATA  7   /* repeated int64, packed */
#define TP_NAME        8   /* string */
#define TP_RAW_DATA   13   /* bytes */
#define TP_DOUBLE_DATA 10  /* repeated double, packed */
#define TP_UINT64_DATA 11  /* repeated uint64, packed */
#define TP_EXTERNAL    14  /* repeated ExternalData */

static int parse_tensor_proto(pb_reader_t *r, onnx_initializer_t *t) {
    memset(t, 0, sizeof(*t));
    int wire;
    while (!pb_eof(r)) {
        uint32_t field = pb_read_tag(r, &wire);
        switch (field) {
            case TP_DIMS: {
                if (wire == PB_WIRE_LEN) {
                    /* packed repeated int64 */
                    pb_reader_t sub;
                    pb_read_submsg(r, &sub);
                    while (!pb_eof(&sub) && t->n_dims < ONNX_MAX_DIMS) {
                        t->dims[t->n_dims++] = (int64_t)pb_read_varint(&sub);
                    }
                } else {
                    /* non-packed single varint */
                    if (t->n_dims < ONNX_MAX_DIMS)
                        t->dims[t->n_dims++] = (int64_t)pb_read_varint(r);
                    else
                        pb_read_varint(r);
                }
                break;
            }
            case TP_DATA_TYPE:
                t->data_type = (int32_t)pb_read_varint(r);
                break;
            case TP_NAME:
                pb_read_string(r, t->name, ONNX_MAX_NAME);
                break;
            case TP_RAW_DATA: {
                uint64_t len = pb_read_varint(r);
                if (r->cur + len <= r->end) {
                    t->raw_data = r->cur;
                    t->raw_size = (size_t)len;
                }
                r->cur += len;
                break;
            }
            case TP_FLOAT_DATA: {
                if (wire == PB_WIRE_LEN) {
                    uint64_t len = pb_read_varint(r);
                    if (r->cur + len <= r->end) {
                        t->decoded_data = malloc(len);
                        if (t->decoded_data) {
                            memcpy(t->decoded_data, r->cur, len);
                            t->decoded_size = len;
                        }
                    }
                    r->cur += len;
                } else if (wire == PB_WIRE_32BIT) {
                    /* non-packed repeated float */
                    uint32_t bits = pb_read_fixed32(r);
                    size_t count = t->decoded_data ? t->decoded_size / sizeof(float) : 0;
                    float *buf = (float *)realloc(t->decoded_data,
                                                   (count + 1) * sizeof(float));
                    memcpy(&buf[count], &bits, sizeof(float));
                    t->decoded_data = buf;
                    t->decoded_size = (count + 1) * sizeof(float);
                } else {
                    pb_skip(r, wire);
                }
                break;
            }
            case TP_INT32_DATA: {
                if (wire == PB_WIRE_LEN) {
                    /* packed varint int32 — decode into buffer */
                    pb_reader_t sub;
                    pb_read_submsg(r, &sub);
                    size_t cap = 64;
                    size_t count = 0;
                    int32_t *buf = (int32_t *)malloc(cap * sizeof(int32_t));
                    while (!pb_eof(&sub)) {
                        if (count >= cap) {
                            cap *= 2;
                            buf = (int32_t *)realloc(buf, cap * sizeof(int32_t));
                        }
                        buf[count++] = (int32_t)pb_read_varint(&sub);
                    }
                    t->decoded_data = buf;
                    t->decoded_size = count * sizeof(int32_t);
                } else if (wire == PB_WIRE_VARINT) {
                    /* non-packed repeated int32 */
                    int32_t val = (int32_t)pb_read_varint(r);
                    size_t count = t->decoded_data ? t->decoded_size / sizeof(int32_t) : 0;
                    int32_t *buf = (int32_t *)realloc(t->decoded_data,
                                                       (count + 1) * sizeof(int32_t));
                    buf[count] = val;
                    t->decoded_data = buf;
                    t->decoded_size = (count + 1) * sizeof(int32_t);
                } else {
                    pb_skip(r, wire);
                }
                break;
            }
            case TP_INT64_DATA: {
                if (wire == PB_WIRE_LEN) {
                    /* packed repeated int64 */
                    pb_reader_t sub;
                    pb_read_submsg(r, &sub);
                    size_t cap = 64;
                    size_t count = 0;
                    int64_t *buf = (int64_t *)malloc(cap * sizeof(int64_t));
                    while (!pb_eof(&sub)) {
                        if (count >= cap) {
                            cap *= 2;
                            buf = (int64_t *)realloc(buf, cap * sizeof(int64_t));
                        }
                        buf[count++] = (int64_t)pb_read_varint(&sub);
                    }
                    t->decoded_data = buf;
                    t->decoded_size = count * sizeof(int64_t);
                } else if (wire == PB_WIRE_VARINT) {
                    /* non-packed repeated int64 — one element at a time */
                    int64_t val = (int64_t)pb_read_varint(r);
                    size_t count = t->decoded_data ? t->decoded_size / sizeof(int64_t) : 0;
                    int64_t *buf = (int64_t *)realloc(t->decoded_data,
                                                       (count + 1) * sizeof(int64_t));
                    buf[count] = val;
                    t->decoded_data = buf;
                    t->decoded_size = (count + 1) * sizeof(int64_t);
                } else {
                    pb_skip(r, wire);
                }
                break;
            }
            default:
                pb_skip(r, wire);
                break;
        }
    }
    return 0;
}

/* ── Parse ONNX AttributeProto ──────────────────────────────────── */

/* AttributeProto field numbers */
#define AP_NAME    1   /* string */
#define AP_F       2   /* float */
#define AP_I       3   /* int64 */
#define AP_S       4   /* bytes (UTF-8 string) */
#define AP_T       5   /* TensorProto */
#define AP_FLOATS  7   /* repeated float */
#define AP_INTS    8   /* repeated int64 */
#define AP_TYPE   20   /* int32 (AttributeType) */

static int parse_attr(pb_reader_t *r, onnx_attr_t *a) {
    memset(a, 0, sizeof(*a));
    int wire;
    while (!pb_eof(r)) {
        uint32_t field = pb_read_tag(r, &wire);
        switch (field) {
            case AP_NAME:
                pb_read_string(r, a->name, ONNX_MAX_NAME);
                break;
            case AP_TYPE:
                a->type = (int32_t)pb_read_varint(r);
                break;
            case AP_F: {
                uint32_t bits = pb_read_fixed32(r);
                memcpy(&a->f, &bits, sizeof(float));
                break;
            }
            case AP_I:
                a->i = (int64_t)pb_read_varint(r);
                break;
            case AP_S: {
                uint64_t len = pb_read_varint(r);
                if (r->cur + len <= r->end) {
                    a->s_data = r->cur;
                    a->s_len  = (size_t)len;
                }
                r->cur += len;
                break;
            }
            case AP_INTS: {
                if (wire == PB_WIRE_LEN) {
                    /* packed int64 */
                    pb_reader_t sub;
                    pb_read_submsg(r, &sub);
                    size_t cap = 16;
                    size_t count = 0;
                    int64_t *buf = (int64_t *)malloc(cap * sizeof(int64_t));
                    while (!pb_eof(&sub)) {
                        if (count >= cap) {
                            cap *= 2;
                            buf = (int64_t *)realloc(buf, cap * sizeof(int64_t));
                        }
                        buf[count++] = (int64_t)pb_read_varint(&sub);
                    }
                    a->ints   = buf;
                    a->n_ints = (int)count;
                } else if (wire == PB_WIRE_VARINT) {
                    /* repeated (non-packed) int64 — single element */
                    int64_t val = (int64_t)pb_read_varint(r);
                    size_t count = (size_t)a->n_ints;
                    size_t cap = count + 1;
                    int64_t *buf = (int64_t *)realloc(a->ints, cap * sizeof(int64_t));
                    buf[count] = val;
                    a->ints   = buf;
                    a->n_ints = (int)cap;
                } else {
                    pb_skip(r, wire);
                }
                break;
            }
            case AP_FLOATS: {
                if (wire == PB_WIRE_LEN) {
                    uint64_t len = pb_read_varint(r);
                    if (r->cur + len <= r->end) {
                        a->floats   = (const float *)r->cur;
                        a->n_floats = (int)(len / sizeof(float));
                    }
                    r->cur += len;
                } else {
                    pb_skip(r, wire);
                }
                break;
            }
            default:
                pb_skip(r, wire);
                break;
        }
    }
    return 0;
}

/* ── Parse ONNX NodeProto ───────────────────────────────────────── */

/* NodeProto field numbers */
#define NP_INPUT    1   /* repeated string */
#define NP_OUTPUT   2   /* repeated string */
#define NP_NAME     3   /* string */
#define NP_OP_TYPE  4   /* string */
#define NP_DOMAIN   7   /* string */
#define NP_ATTR     5   /* repeated AttributeProto */

static int parse_node(pb_reader_t *r, onnx_node_t *n) {
    memset(n, 0, sizeof(*n));
    int wire;
    while (!pb_eof(r)) {
        uint32_t field = pb_read_tag(r, &wire);
        switch (field) {
            case NP_INPUT:
                if (n->n_inputs < ONNX_MAX_INPUTS)
                    pb_read_string(r, n->inputs[n->n_inputs++], ONNX_MAX_NAME);
                else
                    pb_skip(r, wire);
                break;
            case NP_OUTPUT:
                if (n->n_outputs < ONNX_MAX_OUTPUTS)
                    pb_read_string(r, n->outputs[n->n_outputs++], ONNX_MAX_NAME);
                else
                    pb_skip(r, wire);
                break;
            case NP_NAME:
                pb_read_string(r, n->name, ONNX_MAX_NAME);
                break;
            case NP_OP_TYPE:
                pb_read_string(r, n->op_type, sizeof(n->op_type));
                break;
            case NP_DOMAIN:
                pb_read_string(r, n->domain, sizeof(n->domain));
                break;
            case NP_ATTR: {
                if (n->n_attrs < ONNX_MAX_ATTRS) {
                    pb_reader_t sub;
                    pb_read_submsg(r, &sub);
                    parse_attr(&sub, &n->attrs[n->n_attrs++]);
                } else {
                    pb_skip(r, wire);
                }
                break;
            }
            default:
                pb_skip(r, wire);
                break;
        }
    }
    return 0;
}

/* ── Parse ONNX ValueInfoProto ──────────────────────────────────── */

/* ValueInfoProto field numbers */
#define VI_NAME  1  /* string */
#define VI_TYPE  2  /* TypeProto */

/* TypeProto field numbers */
#define TP_TYPE_TENSOR  1  /* TypeProto.Tensor */

/* TypeProto.Tensor field numbers */
#define TT_ELEM_TYPE  1  /* int32 */
#define TT_SHAPE      2  /* TensorShapeProto */

/* TensorShapeProto field numbers */
#define TS_DIM  1  /* repeated Dimension */

/* Dimension field numbers */
#define DIM_VALUE  1  /* int64 */
#define DIM_PARAM  2  /* string */

static int parse_value_info(pb_reader_t *r, onnx_value_info_t *vi) {
    memset(vi, 0, sizeof(*vi));
    int wire;
    while (!pb_eof(r)) {
        uint32_t field = pb_read_tag(r, &wire);
        switch (field) {
            case VI_NAME:
                pb_read_string(r, vi->name, ONNX_MAX_NAME);
                break;
            case VI_TYPE: {
                /* TypeProto */
                pb_reader_t type_r;
                pb_read_submsg(r, &type_r);
                int tw;
                while (!pb_eof(&type_r)) {
                    uint32_t tf = pb_read_tag(&type_r, &tw);
                    if (tf == TP_TYPE_TENSOR && tw == PB_WIRE_LEN) {
                        /* TypeProto.Tensor */
                        pb_reader_t tensor_r;
                        pb_read_submsg(&type_r, &tensor_r);
                        int ttw;
                        while (!pb_eof(&tensor_r)) {
                            uint32_t ttf = pb_read_tag(&tensor_r, &ttw);
                            if (ttf == TT_ELEM_TYPE) {
                                vi->elem_type = (int32_t)pb_read_varint(&tensor_r);
                            } else if (ttf == TT_SHAPE && ttw == PB_WIRE_LEN) {
                                /* TensorShapeProto */
                                pb_reader_t shape_r;
                                pb_read_submsg(&tensor_r, &shape_r);
                                int sw;
                                while (!pb_eof(&shape_r)) {
                                    uint32_t sf = pb_read_tag(&shape_r, &sw);
                                    if (sf == TS_DIM && sw == PB_WIRE_LEN) {
                                        pb_reader_t dim_r;
                                        pb_read_submsg(&shape_r, &dim_r);
                                        int dw;
                                        while (!pb_eof(&dim_r)) {
                                            uint32_t df = pb_read_tag(&dim_r, &dw);
                                            if (df == DIM_VALUE) {
                                                if (vi->n_dims < ONNX_MAX_DIMS)
                                                    vi->dims[vi->n_dims++] = (int64_t)pb_read_varint(&dim_r);
                                                else
                                                    pb_read_varint(&dim_r);
                                            } else if (df == DIM_PARAM) {
                                                /* symbolic dim like "batch" — store as -1 */
                                                pb_skip(&dim_r, dw);
                                                if (vi->n_dims < ONNX_MAX_DIMS)
                                                    vi->dims[vi->n_dims++] = -1;
                                            } else {
                                                pb_skip(&dim_r, dw);
                                            }
                                        }
                                    } else {
                                        pb_skip(&shape_r, sw);
                                    }
                                }
                            } else {
                                pb_skip(&tensor_r, ttw);
                            }
                        }
                    } else {
                        pb_skip(&type_r, tw);
                    }
                }
                break;
            }
            default:
                pb_skip(r, wire);
                break;
        }
    }
    return 0;
}

/* ── Parse ONNX GraphProto ──────────────────────────────────────── */

/* GraphProto field numbers */
#define GP_NODE         1   /* repeated NodeProto */
#define GP_NAME         2   /* string */
#define GP_INITIALIZER  5   /* repeated TensorProto */
#define GP_INPUT       11   /* repeated ValueInfoProto */
#define GP_OUTPUT      12   /* repeated ValueInfoProto */

static int parse_graph(pb_reader_t *r, onnx_model_t *m) {
    int wire;
    int nodes_cap = 0, init_cap = 0, inp_cap = 0, out_cap = 0;

    while (!pb_eof(r)) {
        uint32_t field = pb_read_tag(r, &wire);
        switch (field) {
            case GP_NAME:
                pb_read_string(r, m->graph_name, ONNX_MAX_NAME);
                break;
            case GP_NODE: {
                pb_reader_t sub;
                pb_read_submsg(r, &sub);
                DA_GROW(m->nodes, m->n_nodes, nodes_cap, onnx_node_t);
                parse_node(&sub, &m->nodes[m->n_nodes++]);
                break;
            }
            case GP_INITIALIZER: {
                pb_reader_t sub;
                pb_read_submsg(r, &sub);
                DA_GROW(m->initializers, m->n_initializers, init_cap, onnx_initializer_t);
                parse_tensor_proto(&sub, &m->initializers[m->n_initializers++]);
                break;
            }
            case GP_INPUT: {
                pb_reader_t sub;
                pb_read_submsg(r, &sub);
                DA_GROW(m->inputs, m->n_inputs, inp_cap, onnx_value_info_t);
                parse_value_info(&sub, &m->inputs[m->n_inputs++]);
                break;
            }
            case GP_OUTPUT: {
                pb_reader_t sub;
                pb_read_submsg(r, &sub);
                DA_GROW(m->outputs, m->n_outputs, out_cap, onnx_value_info_t);
                parse_value_info(&sub, &m->outputs[m->n_outputs++]);
                break;
            }
            default:
                pb_skip(r, wire);
                break;
        }
    }
    return 0;
}

/* ── Parse ONNX ModelProto (top level) ──────────────────────────── */

/* ModelProto field numbers */
#define MP_IR_VERSION    1   /* int64 */
#define MP_OPSET        8   /* repeated OperatorSetIdProto */
#define MP_PRODUCER     2   /* string */
#define MP_DOMAIN       4   /* string */
#define MP_MODEL_VER    5   /* int64 */
#define MP_GRAPH        7   /* GraphProto */

/* OperatorSetIdProto field numbers */
#define OS_DOMAIN  1  /* string */
#define OS_VERSION 2  /* int64 */

onnx_model_t *onnx_load(const char *path) {
    onnx_model_t *m = (onnx_model_t *)calloc(1, sizeof(onnx_model_t));
    if (!m) return NULL;

    m->mmap_fd = -1;
    m->mmap_data = onnx_mmap_file(path, &m->mmap_size, &m->mmap_fd);
    if (!m->mmap_data) {
        free(m);
        return NULL;
    }

    pb_reader_t r;
    pb_init(&r, m->mmap_data, m->mmap_size);

    int wire;
    while (!pb_eof(&r)) {
        uint32_t field = pb_read_tag(&r, &wire);
        switch (field) {
            case MP_IR_VERSION:
                m->ir_version = (int64_t)pb_read_varint(&r);
                break;
            case MP_PRODUCER:
                pb_read_string(&r, m->producer_name, ONNX_MAX_NAME);
                break;
            case MP_GRAPH: {
                pb_reader_t sub;
                pb_read_submsg(&r, &sub);
                parse_graph(&sub, m);
                break;
            }
            case MP_OPSET: {
                /* Parse opset — we only care about default domain version */
                pb_reader_t sub;
                pb_read_submsg(&r, &sub);
                char domain[256] = {0};
                int64_t version = 0;
                int ow;
                while (!pb_eof(&sub)) {
                    uint32_t of = pb_read_tag(&sub, &ow);
                    if (of == OS_DOMAIN)
                        pb_read_string(&sub, domain, sizeof(domain));
                    else if (of == OS_VERSION)
                        version = (int64_t)pb_read_varint(&sub);
                    else
                        pb_skip(&sub, ow);
                }
                /* Default domain (empty string) */
                if (domain[0] == '\0' && version > m->opset_version)
                    m->opset_version = version;
                break;
            }
            default:
                pb_skip(&r, wire);
                break;
        }
    }

    return m;
}

void onnx_free(onnx_model_t *m) {
    if (!m) return;

    /* Free decoded tensor data (non-raw_data initializers) */
    for (int i = 0; i < m->n_initializers; i++) {
        if (m->initializers[i].decoded_data)
            free(m->initializers[i].decoded_data);
    }

    /* Free attribute ints arrays (allocated during parsing) */
    for (int i = 0; i < m->n_nodes; i++) {
        for (int j = 0; j < m->nodes[i].n_attrs; j++) {
            if (m->nodes[i].attrs[j].ints)
                free((void *)m->nodes[i].attrs[j].ints);
        }
    }

    free(m->nodes);
    free(m->initializers);
    free(m->inputs);
    free(m->outputs);

    onnx_munmap(m->mmap_data, m->mmap_size, m->mmap_fd);
    free(m);
}

const onnx_initializer_t *onnx_find_initializer(const onnx_model_t *model,
                                                  const char *name) {
    for (int i = 0; i < model->n_initializers; i++) {
        if (strcmp(model->initializers[i].name, name) == 0)
            return &model->initializers[i];
    }
    return NULL;
}

const onnx_attr_t *onnx_node_find_attr(const onnx_node_t *node,
                                        const char *name) {
    for (int i = 0; i < node->n_attrs; i++) {
        if (strcmp(node->attrs[i].name, name) == 0)
            return &node->attrs[i];
    }
    return NULL;
}

int64_t onnx_attr_int(const onnx_node_t *node, const char *name, int64_t def) {
    const onnx_attr_t *a = onnx_node_find_attr(node, name);
    return (a && a->type == ONNX_ATTR_INT) ? a->i : def;
}

float onnx_attr_float(const onnx_node_t *node, const char *name, float def) {
    const onnx_attr_t *a = onnx_node_find_attr(node, name);
    return (a && a->type == ONNX_ATTR_FLOAT) ? a->f : def;
}

int onnx_attr_str(const onnx_node_t *node, const char *name,
                   char *out, int max_len) {
    const onnx_attr_t *a = onnx_node_find_attr(node, name);
    if (!a || a->type != ONNX_ATTR_STRING || !a->s_data || a->s_len == 0) {
        if (max_len > 0) out[0] = '\0';
        return 0;
    }
    int len = (int)a->s_len;
    if (len >= max_len) len = max_len - 1;
    memcpy(out, a->s_data, len);
    out[len] = '\0';
    return len;
}

int onnx_attr_ints(const onnx_node_t *node, const char *name,
                   int64_t *out, int max_count) {
    const onnx_attr_t *a = onnx_node_find_attr(node, name);
    if (!a || a->type != ONNX_ATTR_INTS || !a->ints) return 0;
    int n = a->n_ints < max_count ? a->n_ints : max_count;
    memcpy(out, a->ints, n * sizeof(int64_t));
    return n;
}
