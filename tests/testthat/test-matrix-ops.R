# Tests for Matrix Operations

# ============================================================================
# Outer Product
# ============================================================================

test_that("ggml_out_prod computes outer product", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1, 2, 3))
  ggml_set_f32(b, c(1, 2, 3, 4))

  result <- ggml_out_prod(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # Outer product: a * b^T
  # Result shape: 3x4 = 12 elements
  expect_length(output, 12)

  # Column-major: first column = a * b[1], etc.
  expected <- as.vector(outer(c(1, 2, 3), c(1, 2, 3, 4)))
  expect_equal(output, expected, tolerance = 1e-5)
})

test_that("ggml_out_prod with single elements", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(a, 3)
  ggml_set_f32(b, 4)

  result <- ggml_out_prod(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, 12, tolerance = 1e-5)
})

# ============================================================================
# Diagonal Matrix
# ============================================================================

test_that("ggml_diag creates diagonal matrix", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
  ggml_set_f32(a, c(1, 2, 3))

  result <- ggml_diag(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # Result should be 3x3 diagonal matrix
  expect_length(output, 9)

  # Check diagonal elements (column-major)
  expected <- c(1, 0, 0, 0, 2, 0, 0, 0, 3)
  expect_equal(output, expected, tolerance = 1e-5)
})

test_that("ggml_diag with zeros", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2)
  ggml_set_f32(a, c(0, 0))

  result <- ggml_diag(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, rep(0, 4), tolerance = 1e-5)
})

# ============================================================================
# Concatenation
# ============================================================================

test_that("ggml_concat along dimension 0", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3)
  ggml_set_f32(a, as.numeric(1:6))
  ggml_set_f32(b, as.numeric(7:12))

  result <- ggml_concat(ctx, a, b, dim = 0)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  shape <- ggml_tensor_shape(result)
  expect_equal(shape[1], 4)  # 2 + 2
  expect_equal(shape[2], 3)
})

test_that("ggml_concat along dimension 1", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
  ggml_set_f32(a, rnorm(8))
  ggml_set_f32(b, rnorm(12))

  result <- ggml_concat(ctx, a, b, dim = 1)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  shape <- ggml_tensor_shape(result)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 5)  # 2 + 3
})

# ============================================================================
# Get Rows (Embedding Lookup)
# ============================================================================

test_that("ggml_get_rows extracts rows by index", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Embedding table: 4 tokens, 3-dim embeddings
  embeddings <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
  # Row 0: [1,2,3], Row 1: [4,5,6], Row 2: [7,8,9], Row 3: [10,11,12]
  ggml_set_f32(embeddings, c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

  # Indices to look up
  indices <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2)
  ggml_set_i32(indices, c(0L, 2L))  # Get rows 0 and 2

  result <- ggml_get_rows(ctx, embeddings, indices)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # Should get rows 0 and 2: [1,2,3] and [7,8,9]
  expect_length(output, 6)
  expect_equal(output, c(1, 2, 3, 7, 8, 9), tolerance = 1e-5)
})

test_that("ggml_get_rows with single index", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  embeddings <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)
  ggml_set_f32(embeddings, as.numeric(1:20))

  indices <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1)
  ggml_set_i32(indices, 3L)

  result <- ggml_get_rows(ctx, embeddings, indices)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 4)
  # Row 3 (0-indexed): [13, 14, 15, 16]
  expect_equal(output, c(13, 14, 15, 16), tolerance = 1e-5)
})

test_that("ggml_get_rows with repeated indices", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  embeddings <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3)
  ggml_set_f32(embeddings, c(1, 2, 3, 4, 5, 6))

  indices <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)
  ggml_set_i32(indices, c(0L, 0L, 1L))  # Row 0 twice, row 1 once

  result <- ggml_get_rows(ctx, embeddings, indices)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 6)
  # [1,2], [1,2], [3,4]
  expect_equal(output, c(1, 2, 1, 2, 3, 4), tolerance = 1e-5)
})

# ============================================================================
# Transpose
# ============================================================================

test_that("ggml_transpose transposes 2D matrix", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # 3x2 matrix
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
  ggml_set_f32(a, as.numeric(1:6))

  result <- ggml_transpose(ctx, a)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  shape <- ggml_tensor_shape(result)
  expect_equal(shape[1], 2)
  expect_equal(shape[2], 3)
})

test_that("double transpose returns original shape", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 5)

  t1 <- ggml_transpose(ctx, a)
  t2 <- ggml_transpose(ctx, t1)

  shape <- ggml_tensor_shape(t2)
  expect_equal(shape[1], 4)
  expect_equal(shape[2], 5)
})

# ============================================================================
# Matrix Multiplication
# ============================================================================

test_that("ggml_mul_mat computes matrix multiplication", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # In GGML: ggml_mul_mat(a, b) where a[k,n] and b[k,m] -> result[n,m]
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)  # k=4, n=3
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)  # k=4, m=2
  ggml_set_f32(a, rep(1, 12))
  ggml_set_f32(b, rep(1, 8))

  result <- ggml_mul_mat(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  # Result shape: [3, 2] = 6 elements
  # Each element = dot product of 4 ones = 4
  expect_length(output, 6)
  expect_equal(output, rep(4, 6), tolerance = 1e-5)
})

test_that("ggml_mul_mat with identity-like matrix", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create matrices where result is predictable
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2)  # 2x2
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3)  # 2x3
  # a = [[1,0], [0,1]] (identity-ish for this operation)
  ggml_set_f32(a, c(1, 0, 0, 1))
  ggml_set_f32(b, c(1, 2, 3, 4, 5, 6))

  result <- ggml_mul_mat(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_length(output, 6)
})

# ============================================================================
# Copy Operation
# ============================================================================

test_that("ggml_cpy copies tensor data", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5))

  result <- ggml_cpy(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, result)
  ggml_graph_compute(ctx, graph)

  output <- ggml_get_f32(result)
  expect_equal(output, c(1, 2, 3, 4, 5), tolerance = 1e-5)
})
