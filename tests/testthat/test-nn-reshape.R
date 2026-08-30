# Tests for ggml_layer_reshape (sequential + functional API)
#
# The element-count checks matter more than they look: ggml_reshape_*()
# enforces the count with a GGML_ASSERT, which aborts the process instead of
# raising an R error. If the R-side validation regresses, a bad shape takes the
# whole session down rather than failing a test, so these expect_error() cases
# are the guard rail.

# ============================================================================
# Argument validation (construction time)
# ============================================================================

test_that("ggml_layer_reshape validates shape", {
  m <- ggml_model_sequential()

  expect_error(ggml_layer_reshape(m, c(-1, -1, 4)), "at most one -1")
  expect_error(ggml_layer_reshape(m, c(0, 4)), "must be positive")
  expect_error(ggml_layer_reshape(m, c(-2, 4)), "must be positive")
  expect_error(ggml_layer_reshape(m, c(4, NA)), "free of NA")
  expect_error(ggml_layer_reshape(m, "x"), "numeric")
  expect_error(ggml_layer_reshape(m, numeric(0)), "non-empty")
})

test_that("ggml_layer_reshape records config", {
  model <- ggml_model_sequential() |>
    ggml_layer_dense(512, input_shape = 128) |>
    ggml_layer_reshape(c(-1, 32))

  resh <- model$layers[[2]]
  expect_equal(resh$type, "reshape")
  expect_equal(resh$config$shape, c(-1L, 32L))
  expect_equal(length(resh$weights), 0)
})

# ============================================================================
# Shape inference and the inferred axis
# ============================================================================

test_that("reshape infers a -1 axis from the element count", {
  model <- ggml_model_sequential() |>
    ggml_layer_reshape(c(-1, 32), input_shape = c(4, 8, 16))
  model <- ggmlR:::nn_infer_shapes(model)

  expect_equal(model$layers[[1]]$output_shape, c(16L, 32L))
})

test_that("reshape accepts a fully explicit shape", {
  model <- ggml_model_sequential() |>
    ggml_layer_reshape(c(16, 32), input_shape = c(4, 8, 16))
  model <- ggmlR:::nn_infer_shapes(model)

  expect_equal(model$layers[[1]]$output_shape, c(16L, 32L))
})

test_that("reshape rejects a shape that changes the element count", {
  model <- ggml_model_sequential() |>
    ggml_layer_reshape(c(16, 33), input_shape = c(4, 8, 16))

  expect_error(ggmlR:::nn_infer_shapes(model), "preserve the element count")
})

test_that("reshape rejects a -1 axis that does not divide evenly", {
  model <- ggml_model_sequential() |>
    ggml_layer_reshape(c(-1, 7), input_shape = c(4, 8, 16))

  expect_error(ggmlR:::nn_infer_shapes(model), "not divisible")
})

# ============================================================================
# Values -- data keeps its memory order
# ============================================================================

test_that("reshape preserves the data in order", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # [2,3,4] sample, batch of 1 -> reshape the sample to [6,4].
  x <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2L, 3L, 4L, 1L)
  ggml_set_f32(x, as.numeric(seq_len(24)))
  ggml_set_input(x)

  out <- ggmlR:::nn_build_reshape_op(ctx, x, c(6L, 4L), c(2L, 3L, 4L))
  ggml_set_output(out)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)

  # A reshape only relabels the axes, so the flat order is untouched.
  expect_equal(ggml_tensor_shape(out)[1:3], c(6L, 4L, 1L))
  expect_equal(ggml_get_f32(out), as.numeric(seq_len(24)), tolerance = 1e-5)
})

test_that("reshape keeps the batch axis out of the target shape", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # Batch of 5, sample [2,3,4]; the batch must survive as the trailing axis.
  x <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2L, 3L, 4L, 5L)
  out <- ggmlR:::nn_build_reshape_op(ctx, x, c(6L, 4L), c(2L, 3L, 4L))

  expect_equal(ggml_tensor_shape(out)[1:3], c(6L, 4L, 5L))
  expect_equal(ggml_nelements(out), ggml_nelements(x))
})

test_that("reshape accepts a non-contiguous input", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  # A bare permute is a non-contiguous view; ggml_reshape_*() would assert on
  # it, so the layer has to insert a cont of its own.
  x <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2L, 3L, 4L, 1L)
  ggml_set_f32(x, as.numeric(seq_len(24)))
  ggml_set_input(x)

  permuted <- ggml_permute(ctx, x, 1L, 0L, 2L, 3L)   # no cont: [3,2,4,1] view
  expect_false(ggml_is_contiguous(permuted))

  out <- ggmlR:::nn_build_reshape_op(ctx, permuted, c(24L), c(3L, 2L, 4L))
  ggml_set_output(out)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  ggml_backend_cpu_set_n_threads(backend, 2L)

  graph <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)

  # cont materialises the permuted order, so the result is the transposed
  # ramp, not the original one.
  want <- as.numeric(aperm(array(seq_len(24), dim = c(2, 3, 4)), c(2, 1, 3)))
  expect_equal(ggml_get_f32(out), want, tolerance = 1e-5)
})

# ============================================================================
# Functional API
# ============================================================================

test_that("reshape works as a functional node", {
  inp <- ggml_input(shape = c(4, 8, 16))
  out <- inp |> ggml_layer_reshape(c(-1, 32))

  expect_s3_class(out, "ggml_tensor_node")
  expect_equal(out$node_type, "reshape")
  expect_equal(ggmlR:::nn_functional_output_shape(out, list(c(4L, 8L, 16L))),
               c(16L, 32L))
})

test_that("reshape composes with permute in the functional API", {
  # permute reorders the axes, reshape then folds them: the two are different
  # operations and both are needed.
  inp <- ggml_input(shape = c(4, 8, 16))
  prm <- inp |> ggml_layer_permute(c(3, 1, 2))
  out <- prm |> ggml_layer_reshape(c(-1, 8))

  prm_shape <- ggmlR:::nn_functional_output_shape(prm, list(c(4L, 8L, 16L)))
  expect_equal(prm_shape, c(16L, 4L, 8L))
  expect_equal(ggmlR:::nn_functional_output_shape(out, list(prm_shape)),
               c(64L, 8L))
})
