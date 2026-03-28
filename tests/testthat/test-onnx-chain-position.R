# Chain tests: Positional encoding patterns (RoBERTa-style)
# Shape → Add → ConstantOfShape → NonZero → Gather
#
# Tests cval propagation through shape ops — the pattern that
# caused real bugs in RoBERTa model loading.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}


# ── Minimal (3 ops): Shape → ConstantOfShape → NonZero ───────

test_that("chain position: Shape→ConstantOfShape→NonZero (minimal cval)", {
  # Input: X[4] (1D tensor of length 4)
  # Shape(X) → [4] (shape tensor: cval=[4])
  # ConstantOfShape([4], value=1.0) → [4] filled with 1.0
  # NonZero → [1, 4] (all indices since all elements are 1)
  # Output should be [0, 1, 2, 3] reshaped as [1, 4]

  inp <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 4L))

  # ConstantOfShape value attribute: scalar 1.0
  val_tensor_raw <- .float_bytes(1.0)
  cos_attr <- .onnx_attr_tensor("value", c(1L), 1L, val_tensor_raw)

  shape_node <- .onnx_node("Shape", "X", "shape_out")
  cos_node   <- .onnx_node("ConstantOfShape", "shape_out", "cos_out",
                            attrs = list(cos_attr))
  nz_node    <- .onnx_node("NonZero", "cos_out", "Y")

  graph <- .onnx_graph("test",
                        list(shape_node, cos_node, nz_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(10, 20, 30, 40)  # values don't matter, only shape
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)
  expect_equal(r, c(0, 1, 2, 3), tolerance = 1e-4)
})


# ── Real (5 ops): Shape → Add → ConstantOfShape → NonZero → Gather ──

test_that("chain position: Shape→Add→ConstantOfShape→NonZero→Gather (RoBERTa)", {
  # Simulates RoBERTa position embedding construction:
  # Shape(X) → [seq_len]  (cval = [seq_len])
  # Add([seq_len], offset=2) → [seq_len + 2]  (RoBERTa offset)
  # ConstantOfShape([seq_len + 2]) → ones tensor of length seq_len+2
  # NonZero → [0, 1, 2, ..., seq_len+1] (position indices)
  # Gather(pos_embed_table, indices) → position embeddings

  seq_len <- 3L
  inp <- .onnx_value_info("X", 1L, c(seq_len))

  # Position embedding table: [5, 2] (seq_len+2=5 positions, dim=2)
  pos_data <- c(0.1, 0.2,   # pos 0
                0.3, 0.4,   # pos 1
                0.5, 0.6,   # pos 2
                0.7, 0.8,   # pos 3
                0.9, 1.0)   # pos 4
  pos_raw <- unlist(lapply(pos_data, .float_bytes))
  pos_t  <- .onnx_tensor("PT", c(5L, 2L), 1L, pos_raw)
  pos_vi <- .onnx_value_info("PT", 1L, c(5L, 2L))

  outp <- .onnx_value_info("Y", 1L, c(5L, 2L))

  # Offset constant: scalar 2 (as int64 → float for Add)
  offset_raw <- .float_bytes(2.0)
  offset_t  <- .onnx_tensor("offset", c(1L), 1L, offset_raw)
  offset_vi <- .onnx_value_info("offset", 1L, c(1L))

  # ConstantOfShape value = 1.0
  val_raw <- .float_bytes(1.0)
  cos_attr <- .onnx_attr_tensor("value", c(1L), 1L, val_raw)

  shape_node <- .onnx_node("Shape", "X", "s1")
  add_node   <- .onnx_node("Add", c("s1", "offset"), "s2")
  cos_node   <- .onnx_node("ConstantOfShape", "s2", "ones",
                            attrs = list(cos_attr))
  nz_node    <- .onnx_node("NonZero", "ones", "indices")
  gather_node <- .onnx_node("Gather", c("PT", "indices"), "Y",
                             attrs = list(.onnx_attr_int("axis", 0L)))

  graph <- .onnx_graph("test",
                        list(shape_node, add_node, cos_node, nz_node, gather_node),
                        list(inp, pos_vi, offset_vi),
                        list(outp),
                        list(pos_t, offset_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(99, 99, 99)  # values irrelevant — only shape matters
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)

  # Should gather all 5 positions [0..4] from table
  expect_equal(length(r), 10)
  expect_equal(r, pos_data, tolerance = 1e-4)
})


# ── Boundary: scalar shape (seq_len=1) ───────────────────────

test_that("chain position: seq_len=1 scalar shape (boundary)", {
  # Shape(X[1]) → [1]
  # ConstantOfShape([1]) → [1.0]
  # NonZero → [0]

  inp <- .onnx_value_info("X", 1L, c(1L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 1L))

  val_raw <- .float_bytes(1.0)
  cos_attr <- .onnx_attr_tensor("value", c(1L), 1L, val_raw)

  shape_node <- .onnx_node("Shape", "X", "s1")
  cos_node   <- .onnx_node("ConstantOfShape", "s1", "ones",
                            attrs = list(cos_attr))
  nz_node    <- .onnx_node("NonZero", "ones", "Y")

  graph <- .onnx_graph("test",
                        list(shape_node, cos_node, nz_node),
                        list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(42)))
  r <- as.numeric(result)
  expect_equal(length(r), 1)
  expect_equal(r, 0, tolerance = 1e-4)
})
