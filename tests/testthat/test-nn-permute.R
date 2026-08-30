# Tests for ggml_layer_permute (sequential + functional API)
#
# The permutation tests deliberately use NON-SYMMETRIC permutations on 3-D
# inputs. A plain two-axis swap is self-inverse, so it passes whether the
# implementation treats `dims` as source axes (aperm order, what this layer
# promises) or as destination positions (what the underlying ggml_permute()
# takes). Only a 3-cycle distinguishes the two conventions.

# ============================================================================
# Argument validation
# ============================================================================

test_that("ggml_layer_permute validates dims", {
  m <- ggml_model_sequential()

  expect_error(ggml_layer_permute(m, c(1, 3)), "permutation")
  expect_error(ggml_layer_permute(m, c(1, 1)), "permutation")
  expect_error(ggml_layer_permute(m, c(1, NA)), "free of NA")
  expect_error(ggml_layer_permute(m, "a"), "numeric")
})

test_that("ggml_layer_permute records config", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(8, input_shape = c(4, 5)) |>
    ggml_layer_permute(c(2, 1))

  perm <- model$layers[[2]]
  expect_equal(perm$type, "permute")
  expect_equal(perm$config$dims, c(2L, 1L))
  expect_equal(length(perm$weights), 0)
})

# ============================================================================
# Shape inference
# ============================================================================

test_that("permute reorders the inferred shape in aperm order", {
  # dims[i] names the SOURCE axis landing at position i, so c(3,1,2) applied
  # to [4,5,6] gives [6,4,5] -- not [5,6,4], which is what the destination
  # reading would produce.
  model <- ggml_model_sequential() |>
    ggml_layer_permute(c(3, 1, 2), input_shape = c(4, 5, 6))
  model <- ggmlR:::nn_infer_shapes(model)

  expect_equal(model$layers[[1]]$output_shape, c(6L, 4L, 5L))
})

test_that("permute rejects a dims/input rank mismatch", {
  model <- ggml_model_sequential() |>
    ggml_layer_permute(c(2, 1), input_shape = c(4, 5, 6))

  expect_error(ggmlR:::nn_infer_shapes(model), "must agree")
})

# ============================================================================
# Values -- the axes actually move
# ============================================================================

test_that("permute moves data, not just the shape (non-symmetric, 3-D)", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # [2,3,4] with a batch of 1; a linear ramp makes every element distinguishable.
  x <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2L, 3L, 4L, 1L)
  ggml_set_f32(x, as.numeric(seq_len(2 * 3 * 4)))
  ggml_set_input(x)

  out <- ggmlR:::nn_build_permute_op(ctx, x, c(3L, 1L, 2L), c(2L, 3L, 4L))
  ggml_set_output(out)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)

  # aperm() is the reference: same convention, same expected result.
  want <- aperm(array(seq_len(24), dim = c(2, 3, 4)), c(3, 1, 2))

  expect_equal(ggml_tensor_shape(out)[1:3], c(4L, 2L, 3L))
  expect_equal(ggml_get_f32(out), as.numeric(want), tolerance = 1e-5)
})

test_that("permute output is contiguous", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  x <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2L, 3L, 4L, 1L)
  out <- ggmlR:::nn_build_permute_op(ctx, x, c(2L, 3L, 1L), c(2L, 3L, 4L))

  expect_true(ggml_is_contiguous(out))
})

# ============================================================================
# Interaction with a sequence node
# ============================================================================

test_that("an embedding needs no permute to feed a sequence layer", {
  # This is deliberately the opposite of what an earlier version of this file
  # asserted. The embedding build emits [dim, seq_len, N], which is already the
  # layout attention and GRU/LSTM want; the shape function used to report it as
  # c(dim, seq_len), inverting the package's R-to-ggml rule. Inserting a permute
  # to "fix" that made the R shape pass validation while transposing the actual
  # tensor -- the graph then failed in ggml_can_mul_mat.
  seq_len_ <- 6L
  dim_     <- 8L

  inp <- ggml_input(shape = seq_len_, dtype = "int32")
  emb <- inp |> ggml_layer_embedding(vocab_size = 20L, dim = dim_)

  emb_shape <- ggmlR:::nn_functional_output_shape(emb, list(seq_len_))
  expect_equal(emb_shape, c(seq_len_, dim_))

  # Same convention as a declared sequence input, which is what makes the two
  # interchangeable in front of attention.
  seq_in <- ggml_input(shape = c(seq_len_, dim_))
  expect_equal(ggmlR:::nn_functional_output_shape(seq_in, list(NULL)),
               c(seq_len_, dim_))
})

# ============================================================================
# Functional API
# ============================================================================

test_that("permute works as a functional node", {
  inp <- ggml_input(shape = c(4, 5, 6))
  out <- inp |> ggml_layer_permute(c(3, 1, 2))

  expect_s3_class(out, "ggml_tensor_node")
  expect_equal(out$node_type, "permute")
  expect_equal(ggmlR:::nn_functional_output_shape(out, list(c(4L, 5L, 6L))),
               c(6L, 4L, 5L))
})
