# Chain tests: ConstantOfShape with INT64 value attribute
# This is the root cause of roberta-9 NaN and xcit_tiny dim=0.
#
# ConstantOfShape "value" attr can be TensorProto with data_type=7 (INT64).
# Currently the code reads raw_data as float → gets garbage (1.4e-45 instead of 1).
# This breaks:
#   - Attention mask generation (all -10000 → NaN in softmax)
#   - Position ID generation (NonZero on zeros → empty → dim=0)
#   - Expand/Tile repeat counts (0 repeats → zero-size tensors)

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Bug repro: ConstantOfShape(INT64 value=1) → NonZero → arange ──

test_that("chain constantofshape-int64: INT64 value=1 → NonZero → arange (roberta position_ids)", {
  # This reproduces the roberta-9 position_ids bug:
  # Shape([1,4]) → Gather(idx=1) → 4
  # Sub(4, 0) → 4
  # ConstantOfShape([4], value=INT64(1)) → [1,1,1,1] (should be all-ones)
  # NonZero → [0,1,2,3] (arange)
  # The bug: ConstantOfShape fills 1.4e-45 instead of 1.0 → NonZero returns empty

  inp <- .onnx_value_info("X", 1L, c(1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  # Shape constant: [4] as INT64 (simulating Shape output)
  shape_raw <- .int64_bytes(4L)
  shape_t  <- .onnx_tensor("sh", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("sh", 7L, c(1L))

  # ConstantOfShape with INT64 value=1
  # In ONNX protobuf, the "value" attribute is a TensorProto
  # with data_type=7 (INT64) and raw_data = int64(1)
  cos_node <- .onnx_node("ConstantOfShape", "sh", "ones",
    attrs = list(.onnx_attr_tensor("value", c(), 7L, .int64_bytes(1L))))
  # NonZero: returns indices of non-zero elements → [0,1,2,3] if all ones
  nz_node <- .onnx_node("NonZero", "ones", "nz")
  # Transpose to get [4,1] → squeeze → [4]
  tr_node <- .onnx_node("Transpose", "nz", "nzt",
    attrs = list(.onnx_attr_ints("perm", c(1L, 0L))))
  sq_node <- .onnx_node("Squeeze", "nzt", "ids")
  # Cast to F32 for output
  cast_node <- .onnx_node("Cast", "ids", "idsf",
    attrs = list(.onnx_attr_int("to", 1L)))
  # Unsqueeze to [1,4]
  usq_raw <- .int64_bytes(0L)
  usq_t  <- .onnx_tensor("usq_ax", c(1L), 7L, usq_raw)
  usq_vi <- .onnx_value_info("usq_ax", 7L, c(1L))
  usq_node <- .onnx_node("Unsqueeze", c("idsf", "usq_ax"), "Y")

  graph <- .onnx_graph("test",
    list(cos_node, nz_node, tr_node, sq_node, cast_node, usq_node),
    list(inp, shape_vi, usq_vi), list(outp),
    list(shape_t, usq_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = rep(0, 4)))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  # Should be arange: [0, 1, 2, 3]
  expect_equal(r, c(0, 1, 2, 3), tolerance = 1e-3)
})

# ── Bug repro: ConstantOfShape(INT64 value=1) → mask → Softmax ──

test_that("chain constantofshape-int64: INT64 ones mask → Sub → Mul → Add → Softmax (roberta mask)", {
  # Reproduces: ConstantOfShape(value=INT64(1)) → Cast → Sub(1, x) → Mul(-10000)
  # If fill is wrong (≈0 instead of 1), Sub gives ≈1, Mul gives -10000 → all masked → NaN

  inp <- .onnx_value_info("X", 1L, c(1L, 4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  # "ones" mask from ConstantOfShape
  shape_raw <- .int64_bytes(4L)
  shape_t  <- .onnx_tensor("sh", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("sh", 7L, c(1L))

  one_raw <- .float_bytes(1.0)
  one_t  <- .onnx_tensor("one", c(1L), 1L, one_raw)
  one_vi <- .onnx_value_info("one", 1L, c(1L))

  neg_raw <- .float_bytes(-10000.0)
  neg_t  <- .onnx_tensor("neg", c(1L), 1L, neg_raw)
  neg_vi <- .onnx_value_info("neg", 1L, c(1L))

  # ConstantOfShape([4], value=INT64(1)) → [1,1,1,1] (all valid tokens)
  cos_node <- .onnx_node("ConstantOfShape", "sh", "mask_i64",
    attrs = list(.onnx_attr_tensor("value", c(), 7L, .int64_bytes(1L))))
  # Cast to F32
  cast_node <- .onnx_node("Cast", "mask_i64", "mask",
    attrs = list(.onnx_attr_int("to", 1L)))
  # Sub(1, mask) → 0 for valid, 1 for padding
  sub_node <- .onnx_node("Sub", c("one", "mask"), "invmask")
  # Mul(-10000) → 0 for valid, -10000 for padding
  mul_node <- .onnx_node("Mul", c("invmask", "neg"), "attn_mask")
  # Add to scores (X) + mask
  add_node <- .onnx_node("Add", c("X", "attn_mask"), "masked")
  # Softmax
  sm_node <- .onnx_node("Softmax", "masked", "Y",
    attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
    list(cos_node, cast_node, sub_node, mul_node, add_node, sm_node),
    list(inp, shape_vi, one_vi, neg_vi),
    list(outp),
    list(shape_t, one_t, neg_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1.0, 2.0, 3.0, 4.0)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_true(all(is.finite(r)))  # No NaN!
  expect_true(all(r > 0))
  expect_equal(sum(r), 1.0, tolerance = 1e-3)  # Valid softmax
})

# ── Gather on 4D tensor (cait pattern) ──────────────────────

test_that("chain constantofshape-int64: Gather axis=0 on 4D [3,6,4,2] (cait QKV split)", {
  # Simulates CaiT: qkv tensor [3, 6, 576, 48] in ONNX (= [48,576,6,3] in ggml)
  # Gather(axis=0, idx=0) selects Q → [6, 576, 48] in ONNX
  # ggml_get_rows fails because it expects ne[2]==b.ne[1]

  # Smaller version: [3, 2, 4, 2] ONNX → ggml [2, 4, 2, 3]
  inp <- .onnx_value_info("X", 1L, c(3L, 2L, 4L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 4L, 2L))

  # Index = 0 (select first slice = Q)
  idx_raw <- .int64_bytes(0L)
  idx_t  <- .onnx_tensor("idx", c(), 7L, idx_raw)  # scalar
  idx_vi <- .onnx_value_info("idx", 7L, c())

  gather_node <- .onnx_node("Gather", c("X", "idx"), "q",
    attrs = list(.onnx_attr_int("axis", 0L)))
  relu_node <- .onnx_node("Relu", "q", "Y")

  graph <- .onnx_graph("test", list(gather_node, relu_node),
    list(inp, idx_vi), list(outp), list(idx_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # 3*2*4*2 = 48 elements
  x <- seq(1, 48) / 10
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  # Should select first 1/3 of data: elements 1..16 (first 2*4*2 = 16)
  expect_equal(length(r), 16)
  expect_true(all(r >= 0))
  expect_true(all(is.finite(r)))
})

# ── Boundary: ConstantOfShape with INT64 value=0 ───────────

test_that("chain constantofshape-int64: INT64 value=0 (zeros tensor, boundary)", {
  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))

  shape_raw <- .int64_bytes(4L)
  shape_t  <- .onnx_tensor("sh", c(1L), 7L, shape_raw)
  shape_vi <- .onnx_value_info("sh", 7L, c(1L))

  cos_node <- .onnx_node("ConstantOfShape", "sh", "zeros",
    attrs = list(.onnx_attr_tensor("value", c(), 7L, .int64_bytes(0L))))
  # Add X + zeros = X
  add_node <- .onnx_node("Add", c("X", "zeros"), "Y")

  graph <- .onnx_graph("test", list(cos_node, add_node),
    list(inp, shape_vi), list(outp), list(shape_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(r, x, tolerance = 1e-3)
})
