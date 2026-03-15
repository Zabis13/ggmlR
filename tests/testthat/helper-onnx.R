# onnx_test_helpers.R — R-based minimal ONNX protobuf serializer
#
# Internal functions for generating test .onnx files without Python.
# Not exported. Used only in tests/testthat/test-onnx.R.

# ── Protobuf wire format primitives ──────────────────────────────

# Encode unsigned varint
.pb_varint <- function(value) {
  value <- as.numeric(value)
  bytes <- raw(0)
  repeat {
    b <- as.integer(value %% 128)
    value <- floor(value / 128)
    if (value > 0) b <- bitwOr(b, 0x80L)
    bytes <- c(bytes, as.raw(b))
    if (value == 0) break
  }
  bytes
}

# Encode tag (field_number, wire_type)
.pb_tag <- function(field, wire_type) {
  .pb_varint(field * 8 + wire_type)
}

# Length-delimited field (wire type 2)
.pb_bytes <- function(field, data) {
  data <- as.raw(data)
  c(.pb_tag(field, 2L), .pb_varint(length(data)), data)
}

# Varint field (wire type 0)
.pb_varint_field <- function(field, value) {
  c(.pb_tag(field, 0L), .pb_varint(value))
}

# Fixed32 field (wire type 5) — for floats
.pb_fixed32 <- function(field, value) {
  c(.pb_tag(field, 5L), writeBin(as.double(value), raw(), size = 4))
}

# Encode string as bytes
.pb_string <- function(field, s) {
  .pb_bytes(field, charToRaw(s))
}

# Encode float as 4 bytes (little-endian)
.float_bytes <- function(x) {
  writeBin(as.double(x), raw(), size = 4)
}

# Encode int64 as 8 bytes (little-endian)
.int64_bytes <- function(x) {
  writeBin(as.integer(x), raw(), size = 8, endian = "little")
}

# ── ONNX protobuf message builders ──────────────────────────────

# TensorShapeProto.Dimension (field 1 = dim_value as varint)
.onnx_dim <- function(value) {
  .pb_varint_field(1L, value)
}

# TensorShapeProto (field 1 = dim, repeated)
.onnx_shape <- function(dims) {
  out <- raw(0)
  for (d in dims) {
    out <- c(out, .pb_bytes(1L, .onnx_dim(d)))
  }
  out
}

# TypeProto.Tensor (field 1 = elem_type, field 2 = shape)
.onnx_tensor_type <- function(elem_type, dims) {
  out <- .pb_varint_field(1L, elem_type)
  if (length(dims) > 0) {
    out <- c(out, .pb_bytes(2L, .onnx_shape(dims)))
  }
  out
}

# TypeProto (field 1 = tensor_type)
.onnx_type_proto <- function(elem_type, dims) {
  .pb_bytes(1L, .onnx_tensor_type(elem_type, dims))
}

# ValueInfoProto (field 1 = name, field 2 = type)
.onnx_value_info <- function(name, elem_type = 1L, dims = integer(0)) {
  out <- .pb_string(1L, name)
  out <- c(out, .pb_bytes(2L, .onnx_type_proto(elem_type, dims)))
  out
}

# TensorProto (field 1 = dims repeated, field 2 = data_type,
#              field 8 = name, field 9 = raw_data)
.onnx_tensor <- function(name, dims, data_type = 1L, raw_data = raw(0)) {
  out <- raw(0)
  # dims (field 1, varint, packed would be better but repeated varint works)
  for (d in dims) {
    out <- c(out, .pb_varint_field(1L, d))
  }
  out <- c(out, .pb_varint_field(2L, data_type))
  out <- c(out, .pb_string(8L, name))
  if (length(raw_data) > 0) {
    out <- c(out, .pb_bytes(9L, raw_data))
  }
  out
}

# AttributeProto
.onnx_attr_int <- function(name, value) {
  out <- .pb_string(1L, name)   # name
  out <- c(out, .pb_varint_field(3L, value))  # i
  out <- c(out, .pb_varint_field(20L, 2L))     # type = INT
  out
}

.onnx_attr_float <- function(name, value) {
  out <- .pb_string(1L, name)    # name
  out <- c(out, .pb_fixed32(2L, value))  # f (field 2 in AttributeProto)
  out <- c(out, .pb_varint_field(20L, 1L))      # type = FLOAT
  out
}

.onnx_attr_ints <- function(name, values) {
  out <- .pb_string(1L, name)   # name
  for (v in values) {
    out <- c(out, .pb_varint_field(8L, v))  # ints (repeated)
  }
  out <- c(out, .pb_varint_field(20L, 7L))     # type = INTS
  out
}

# AttributeProto with TensorProto value (field 5 = t, type 20 = 4 TENSOR)
.onnx_attr_tensor <- function(name, dims, data_type = 1L, raw_data = raw(0)) {
  out <- .pb_string(1L, name)   # name
  tensor <- .onnx_tensor("", dims, data_type, raw_data)
  out <- c(out, .pb_bytes(5L, tensor))  # t (field 5)
  out <- c(out, .pb_varint_field(20L, 4L))  # type = TENSOR
  out
}

# NodeProto (field 1 = input repeated, 2 = output repeated,
#            3 = name, 4 = op_type, 5 = attribute repeated)
.onnx_node <- function(op_type, inputs, outputs, name = "", attrs = list()) {
  out <- raw(0)
  for (inp in inputs)  out <- c(out, .pb_string(1L, inp))
  for (outp in outputs) out <- c(out, .pb_string(2L, outp))
  if (nzchar(name)) out <- c(out, .pb_string(3L, name))
  out <- c(out, .pb_string(4L, op_type))
  for (a in attrs) {
    out <- c(out, .pb_bytes(5L, a))
  }
  out
}

# GraphProto (field 1 = node repeated, 2 = name,
#             5 = initializer repeated, 11 = input repeated,
#             12 = output repeated)
.onnx_graph <- function(name, nodes, inputs, outputs, initializers = list()) {
  out <- raw(0)
  for (n in nodes)       out <- c(out, .pb_bytes(1L, n))
  out <- c(out, .pb_string(2L, name))
  for (init in initializers) out <- c(out, .pb_bytes(5L, init))
  for (inp in inputs)    out <- c(out, .pb_bytes(11L, inp))
  for (outp in outputs)  out <- c(out, .pb_bytes(12L, outp))
  out
}

# OperatorSetIdProto (field 2 = version)
.onnx_opset <- function(version = 13L) {
  .pb_varint_field(2L, version)
}

# ModelProto (field 1 = ir_version, field 7 = graph, field 8 = opset_import)
.onnx_model <- function(graph, ir_version = 7L, opset_version = 13L) {
  out <- .pb_varint_field(1L, ir_version)
  out <- c(out, .pb_bytes(8L, .onnx_opset(opset_version)))
  out <- c(out, .pb_bytes(7L, graph))
  out
}

# ── High-level helpers for common test models ────────────────────

# Create a simple unary op model: input → op → output
# Returns path to temporary .onnx file
.onnx_make_unary <- function(op_type, input_dims = c(1L, 4L),
                              elem_type = 1L, attrs = list()) {
  inp <- .onnx_value_info("X", elem_type, input_dims)
  outp <- .onnx_value_info("Y", elem_type, input_dims)
  node <- .onnx_node(op_type, "X", "Y", attrs = attrs)
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

# Create a simple binary op model: (A, B) → op → Y
.onnx_make_binary <- function(op_type, dims_a = c(1L, 4L),
                               dims_b = dims_a, elem_type = 1L,
                               attrs = list()) {
  inp_a <- .onnx_value_info("A", elem_type, dims_a)
  inp_b <- .onnx_value_info("B", elem_type, dims_b)
  outp  <- .onnx_value_info("Y", elem_type, dims_a)
  node  <- .onnx_node(op_type, c("A", "B"), "Y", attrs = attrs)
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

# Create MatMul model: A[M,K] @ B[K,N] → Y[M,N]
.onnx_make_matmul <- function(M = 2L, K = 3L, N = 4L) {
  inp_a <- .onnx_value_info("A", 1L, c(M, K))
  inp_b <- .onnx_value_info("B", 1L, c(K, N))
  outp  <- .onnx_value_info("Y", 1L, c(M, N))
  node  <- .onnx_node("MatMul", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

# Create Gemm model with bias: A[M,K] @ B[K,N] + C[N] → Y[M,N]
.onnx_make_gemm <- function(M = 2L, K = 3L, N = 4L,
                             transA = 0L, transB = 0L,
                             weight_data = NULL, bias_data = NULL) {
  inp_a <- .onnx_value_info("A", 1L, c(M, K))

  inits <- list()
  graph_inputs <- list(inp_a)

  # Weight B as initializer
  if (is.null(weight_data)) {
    weight_data <- rep(1.0, K * N)
  }
  b_raw <- unlist(lapply(weight_data, .float_bytes))
  b_tensor <- .onnx_tensor("B", c(K, N), 1L, b_raw)
  b_vi <- .onnx_value_info("B", 1L, c(K, N))
  inits <- c(inits, list(b_tensor))
  graph_inputs <- c(graph_inputs, list(b_vi))

  # Bias C as initializer
  if (!is.null(bias_data)) {
    c_raw <- unlist(lapply(bias_data, .float_bytes))
    c_tensor <- .onnx_tensor("C", N, 1L, c_raw)
    c_vi <- .onnx_value_info("C", 1L, N)
    inits <- c(inits, list(c_tensor))
    graph_inputs <- c(graph_inputs, list(c_vi))
  }

  outp <- .onnx_value_info("Y", 1L, c(M, N))

  attrs <- list(
    .onnx_attr_int("transA", transA),
    .onnx_attr_int("transB", transB)
  )

  node_inputs <- if (!is.null(bias_data)) c("A", "B", "C") else c("A", "B")
  node <- .onnx_node("Gemm", node_inputs, "Y", attrs = attrs)
  graph <- .onnx_graph("test", list(node), graph_inputs, list(outp), inits)
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

# Create model with initializer weights: input → op(input, weight) → output
# For testing ops that need pre-loaded weights
.onnx_make_with_weight <- function(op_type, input_dims, weight_dims,
                                    weight_data, output_dims = input_dims,
                                    elem_type = 1L, attrs = list()) {
  inp <- .onnx_value_info("X", elem_type, input_dims)

  # Weight as initializer
  w_raw <- unlist(lapply(weight_data, .float_bytes))
  w_tensor <- .onnx_tensor("W", weight_dims, elem_type, w_raw)
  w_vi <- .onnx_value_info("W", elem_type, weight_dims)

  outp <- .onnx_value_info("Y", elem_type, output_dims)
  node <- .onnx_node(op_type, c("X", "W"), "Y", attrs = attrs)

  graph <- .onnx_graph("test", list(node),
                        list(inp, w_vi), list(outp),
                        list(w_tensor))
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

# Create a chain model: X → op1 → tmp → op2 → Y
.onnx_make_chain <- function(op1, op2, dims = c(1L, 4L),
                              attrs1 = list(), attrs2 = list()) {
  inp  <- .onnx_value_info("X", 1L, dims)
  outp <- .onnx_value_info("Y", 1L, dims)
  n1 <- .onnx_node(op1, "X", "tmp", attrs = attrs1)
  n2 <- .onnx_node(op2, "tmp", "Y", attrs = attrs2)
  graph <- .onnx_graph("test", list(n1, n2), list(inp), list(outp))
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

# Create Reshape model: X[input_dims] → Reshape → Y[output_dims]
.onnx_make_reshape <- function(input_dims, output_dims) {
  inp <- .onnx_value_info("X", 1L, input_dims)

  # Shape tensor as initializer (int64)
  shape_raw <- raw(0)
  for (d in output_dims) {
    shape_raw <- c(shape_raw, writeBin(as.integer(d), raw(), size = 8,
                                        endian = "little"))
  }
  shape_tensor <- .onnx_tensor("shape", length(output_dims), 7L, shape_raw)
  shape_vi <- .onnx_value_info("shape", 7L, length(output_dims))

  # Resolve -1 and 0 for output value_info (protobuf can't encode negative varints)
  resolved <- output_dims
  total_in <- prod(input_dims)
  neg_idx <- which(resolved == -1L)
  zero_idx <- which(resolved == 0L)
  if (length(zero_idx) > 0) {
    for (i in zero_idx) resolved[i] <- input_dims[i]
  }
  if (length(neg_idx) == 1) {
    known <- prod(resolved[-neg_idx])
    resolved[neg_idx] <- total_in / known
  }
  outp <- .onnx_value_info("Y", 1L, resolved)
  node <- .onnx_node("Reshape", c("X", "shape"), "Y")

  graph <- .onnx_graph("test", list(node),
                        list(inp, shape_vi), list(outp),
                        list(shape_tensor))
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

# Create LayerNorm model: X → LayerNormalization(scale, bias) → Y
.onnx_make_layer_norm <- function(dims = c(1L, 4L), eps = 1e-5) {
  n <- dims[length(dims)]  # normalize over last dim
  inp <- .onnx_value_info("X", 1L, dims)

  # Scale = ones
  scale_raw <- unlist(lapply(rep(1.0, n), .float_bytes))
  scale_t <- .onnx_tensor("scale", n, 1L, scale_raw)
  scale_vi <- .onnx_value_info("scale", 1L, n)

  # Bias = zeros
  bias_raw <- unlist(lapply(rep(0.0, n), .float_bytes))
  bias_t <- .onnx_tensor("bias", n, 1L, bias_raw)
  bias_vi <- .onnx_value_info("bias", 1L, n)

  outp <- .onnx_value_info("Y", 1L, dims)
  attrs <- list(.onnx_attr_float("epsilon", eps))
  node <- .onnx_node("LayerNormalization", c("X", "scale", "bias"), "Y",
                      attrs = attrs)

  graph <- .onnx_graph("test", list(node),
                        list(inp, scale_vi, bias_vi), list(outp),
                        list(scale_t, bias_t))
  model <- .onnx_model(graph)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}
