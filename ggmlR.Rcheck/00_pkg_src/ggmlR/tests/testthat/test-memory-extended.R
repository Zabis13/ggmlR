# Tests for extended memory management functions

test_that("set_no_alloc and get_no_alloc work correctly", {
  ctx <- ggml_init(4 * 1024 * 1024)

  # Default should be FALSE
  expect_false(ggml_get_no_alloc(ctx))

  # Set to TRUE
  ggml_set_no_alloc(ctx, TRUE)
  expect_true(ggml_get_no_alloc(ctx))

  # Set back to FALSE
  ggml_set_no_alloc(ctx, FALSE)
  expect_false(ggml_get_no_alloc(ctx))

  ggml_free(ctx)
})

test_that("get_max_tensor_size returns reasonable value", {
  ctx <- ggml_init(4 * 1024 * 1024)

  max_size <- ggml_get_max_tensor_size(ctx)
  expect_type(max_size, "double")

  # Max tensor size should be less than or equal to context size
  # (0 means not calculated yet, which is also valid)
  expect_gte(max_size, 0)

  ggml_free(ctx)
})

test_that("print_objects runs without error", {
  ctx <- ggml_init(4 * 1024 * 1024)

  # Create some tensors
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 5)

  # Should not throw error
  expect_silent(ggml_print_objects(ctx))

  ggml_free(ctx)
})

test_that("print_mem_status shows correct info", {
  ctx <- ggml_init(4 * 1024 * 1024)

  # Create a tensor to use some memory
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)

  # Capture output
  result <- ggml_print_mem_status(ctx)

  expect_type(result, "list")
  expect_true("total" %in% names(result))
  expect_true("used" %in% names(result))
  expect_true("free" %in% names(result))

  expect_gt(result$used, 0)
  expect_equal(result$total, result$used + result$free)

  ggml_free(ctx)
})
