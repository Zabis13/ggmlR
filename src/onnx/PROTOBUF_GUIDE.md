# ONNX↔ggml Protobuf & Dimension Mapping Guide

## Dimension Mapping & Collapsing

### ONNX vs ggml dimension order

ONNX uses **row-major** (outermost-first): `[batch, channels, height, width]`
ggml uses **column-major** (innermost-first): `ne[0]=width, ne[1]=height, ne[2]=channels, ne[3]=batch`

Mapping formula (for ndims ≤ 4):
```
ggml_dim = ndims - 1 - onnx_axis
```

Example (4D): ONNX `[N,C,H,W]` → ggml `ne[3]=N, ne[2]=C, ne[1]=H, ne[0]=W`

### >4D collapse to 4D

ggml supports max 4 dimensions. ONNX tensors with ndims > 4 are collapsed by **merging leading ONNX dims** into ggml dim 3:

5D ONNX `[A, B, C, D, E]` → 4D ggml `ne[0]=E, ne[1]=D, ne[2]=C, ne[3]=A*B`

6D ONNX `[A, B, C, D, E, F]` → 4D ggml `ne[0]=F, ne[1]=E, ne[2]=D, ne[3]=A*B*C`

General rule: last 3 ONNX dims map to ggml dims 0,1,2. All remaining leading ONNX dims are multiplied together into ggml dim 3.

### Axis mapping for collapsed >4D tensors

We track the **original ONNX ndims** via `tensor_map_ndims` in `onnx_ggml_ctx_t`.

For an op (Concat, Split, etc.) with ONNX axis on a tensor that was originally N-dimensional:

```
ggml_dim = orig_ndims - 1 - onnx_axis
clamped to [0, 3]
```

Axes that fall in the merged region (ggml_dim > 3) are clamped to dim 3.

**Example: 5D tensor, Concat axis=4**
```
orig_ndims = 5
onnx_axis = 4  (last dim)
ggml_dim = 5 - 1 - 4 = 0   → concat along ggml ne[0]  ✓
```

**Example: 5D tensor, Concat axis=3**
```
orig_ndims = 5
onnx_axis = 3  (second-to-last dim)
ggml_dim = 5 - 1 - 3 = 1   → concat along ggml ne[1]  ✓
```

**Example: 5D tensor, Concat axis=0**
```
orig_ndims = 5
onnx_axis = 0  (first dim — in merged region)
ggml_dim = 5 - 1 - 0 = 4   → clamped to 3  ✓  (merges into batch dim)
```

### Negative axis handling

```c
if (onnx_axis < 0) onnx_axis = eff_ndims + onnx_axis;
```
Then apply the same `eff_ndims - 1 - onnx_axis` formula.

## Protobuf Wire Format

### Wire types
| ID | Name | Size |
|----|------|------|
| 0 | Varint | variable |
| 1 | Fixed64 | 8 bytes |
| 2 | Length-delimited | variable |
| 5 | Fixed32 | 4 bytes |

### TensorProto fields (onnx.proto3)
| Field | Name | Wire |
|-------|------|------|
| 1 | dims | varint/packed |
| 2 | data_type | varint |
| 4 | float_data | packed float |
| 5 | int32_data | packed varint |
| 6 | string_data | bytes |
| 7 | int64_data | packed varint |
| 8 | name | string |
| 9 | **raw_data** | bytes |
| 10 | double_data | packed double |
| 11 | uint64_data | packed uint64 |
| 12 | doc_string | string |
| 13 | external_data | repeated msg |
| 14 | data_location | varint |

### ONNX data types
| Value | Type | Bytes |
|-------|------|-------|
| 1 | FLOAT | 4 |
| 2 | UINT8 | 1 |
| 3 | INT8 | 1 |
| 5 | INT16 | 2 |
| 6 | INT32 | 4 |
| 7 | INT64 | 8 |
| 10 | FLOAT16 | 2 |
| 11 | DOUBLE | 8 |
| 16 | BFLOAT16 | 2 |

### GraphProto fields
| Field | Name |
|-------|------|
| 1 | node (repeated NodeProto) |
| 2 | name |
| 5 | initializer (repeated TensorProto) |
| 11 | input (repeated ValueInfoProto) |
| 12 | output (repeated ValueInfoProto) |

### NodeProto fields
| Field | Name |
|-------|------|
| 1 | input (repeated string) |
| 2 | output (repeated string) |
| 3 | name |
| 4 | op_type |
| 5 | attribute (repeated AttributeProto) |
| 7 | domain |

### AttributeProto fields
| Field | Name | Type |
|-------|------|------|
| 1 | name | string |
| 2 | f | float (fixed32) |
| 3 | i | int (varint) |
| 4 | s | bytes |
| 5 | t | TensorProto |
| 6 | g | GraphProto |
| 7 | floats | repeated float |
| 8 | ints | repeated int |
| 20 | type | varint (1=FLOAT,2=INT,3=STRING,4=TENSOR,6=FLOATS,7=INTS) |
