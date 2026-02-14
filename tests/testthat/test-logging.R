# Tests for logging and debugging functions

test_that("logging functions exist and work", {
  # Check that R logging can be enabled/disabled
  expect_silent(ggml_log_set_r())
  expect_true(ggml_log_is_r_enabled())

  expect_silent(ggml_log_set_default())
  expect_false(ggml_log_is_r_enabled())

  # Restore R logging (suppresses debug messages in subsequent tests)
  ggml_log_set_r()
})

test_that("abort callback functions exist and work", {
  # Check that R abort handler can be enabled/disabled
  expect_silent(ggml_set_abort_callback_r())
  expect_true(ggml_abort_is_r_enabled())

  expect_silent(ggml_set_abort_callback_default())
  expect_false(ggml_abort_is_r_enabled())

  # Restore R abort handler for subsequent tests
  ggml_set_abort_callback_r()
})

test_that("op_params functions work", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  # Create a tensor
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)

  # Get raw op_params
  params <- ggml_get_op_params(a)
  expect_type(params, "raw")
  expect_length(params, 64)  # GGML_MAX_OP_PARAMS

  # Set/get integer parameter
  ggml_set_op_params_i32(a, 0, 42L)
  expect_equal(ggml_get_op_params_i32(a, 0), 42L)

  ggml_set_op_params_i32(a, 1, -100L)
  expect_equal(ggml_get_op_params_i32(a, 1), -100L)

  # Set/get float parameter
  ggml_set_op_params_f32(a, 2, 3.14)
  result <- ggml_get_op_params_f32(a, 2)
  expect_equal(result, 3.14, tolerance = 1e-5)

  # Set raw params
  new_params <- as.raw(c(1, 2, 3, 4, rep(0, 60)))
  ggml_set_op_params(a, new_params)
  retrieved <- ggml_get_op_params(a)
  expect_equal(retrieved[1:4], as.raw(c(1, 2, 3, 4)))
})

test_that("op_params index bounds are checked", {
  ctx <- ggml_init(1024 * 1024)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)

  # These should error with out-of-range indices
  expect_error(ggml_get_op_params_i32(a, -1))
  expect_error(ggml_get_op_params_i32(a, 100))
  expect_error(ggml_set_op_params_i32(a, -1, 0))
})
