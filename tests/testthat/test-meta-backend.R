# Meta backend smoke tests (Stage 5).
# Meta backend wraps N "simple" devices; with 1 device it is degenerate
# (no-op); with N >= 2 it splits/mirrors tensors based on a callback.
# These tests exercise the R bridge end-to-end without requiring a GPU.

test_that("ggml_backend_meta_device validates inputs", {
  cpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  skip_if(is.null(cpu_dev), "No CPU device available")

  # devs must be a list
  expect_error(ggml_backend_meta_device("not-a-list", function(ti, n) NULL))
  # split_fn must be a function
  expect_error(ggml_backend_meta_device(list(cpu_dev), "not-a-function"))
  # empty devs list
  expect_error(ggml_backend_meta_device(list(), function(ti, n) NULL))
})

test_that("meta backend (1 device, MIRRORED) round-trips tensor data", {
  cpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  skip_if(is.null(cpu_dev), "No CPU device available")

  split_fn <- function(tensor_info, n_devs) {
    # MIRRORED axis = 10
    list(axis = 10L, ne = rep(0L, n_devs), n_segments = 1L)
  }

  meta_dev <- ggml_backend_meta_device(list(cpu_dev), split_fn)
  expect_true(!is.null(meta_dev))

  meta_backend <- ggml_backend_dev_init(meta_dev)
  skip_if(is.null(meta_backend), "Could not init meta backend")

  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)

  buffer <- ggml_backend_alloc_ctx_tensors(ctx, meta_backend)
  skip_if(is.null(buffer), "Could not allocate meta buffer")

  data_in <- as.numeric(1:10)
  ggml_backend_tensor_set_data(tensor, data_in)
  data_out <- ggml_backend_tensor_get_data(tensor)
  expect_equal(data_out, data_in)

  ggml_backend_buffer_free(buffer)
  ggml_free(ctx)
  ggml_backend_free(meta_backend)
})

test_that("meta backend (2 devices, MIRRORED) round-trips tensor data", {
  cpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  skip_if(is.null(cpu_dev), "No CPU device available")

  # Two refs to the same CPU device — meta backend treats them as 2 devs and
  # mirrored MIRRORED stores a full copy on each. This exercises the multi-buffer
  # path without requiring a second physical device.
  split_fn <- function(tensor_info, n_devs) {
    list(axis = 10L, ne = rep(0L, n_devs), n_segments = 1L)
  }

  meta_dev <- ggml_backend_meta_device(list(cpu_dev, cpu_dev), split_fn)
  expect_true(!is.null(meta_dev))

  meta_backend <- ggml_backend_dev_init(meta_dev)
  skip_if(is.null(meta_backend), "Could not init meta backend")

  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)

  buffer <- ggml_backend_alloc_ctx_tensors(ctx, meta_backend)
  skip_if(is.null(buffer), "Could not allocate meta buffer (2 devs)")

  data_in <- as.numeric(c(-3, -1, 0, 1, 2, 3, 4, 5))
  ggml_backend_tensor_set_data(tensor, data_in)
  data_out <- ggml_backend_tensor_get_data(tensor)
  expect_equal(data_out, data_in)

  ggml_backend_buffer_free(buffer)
  ggml_free(ctx)
  ggml_backend_free(meta_backend)
})

test_that("meta backend split_fn receives tensor_info with expected fields", {
  cpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  skip_if(is.null(cpu_dev), "No CPU device available")

  observed <- new.env(parent = emptyenv())
  observed$calls <- 0L
  observed$last_info <- NULL
  observed$last_n_devs <- NA_integer_

  split_fn <- function(tensor_info, n_devs) {
    observed$calls <- observed$calls + 1L
    observed$last_info <- tensor_info
    observed$last_n_devs <- as.integer(n_devs)
    list(axis = 10L, ne = rep(0L, n_devs), n_segments = 1L)
  }

  meta_dev <- ggml_backend_meta_device(list(cpu_dev, cpu_dev), split_fn,
                                       env = environment())
  meta_backend <- ggml_backend_dev_init(meta_dev)
  skip_if(is.null(meta_backend), "Could not init meta backend")

  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  t1 <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, meta_backend)
  skip_if(is.null(buffer), "Could not allocate meta buffer")

  expect_gt(observed$calls, 0L)
  expect_equal(observed$last_n_devs, 2L)
  expect_true(is.list(observed$last_info))
  expect_true(all(c("name", "type", "ne", "op", "flags") %in% names(observed$last_info)))
  expect_type(observed$last_info$ne, "double")

  ggml_backend_buffer_free(buffer)
  ggml_free(ctx)
  ggml_backend_free(meta_backend)
})

test_that("meta backend tolerates a misbehaving split_fn (sticky fallback)", {
  cpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  skip_if(is.null(cpu_dev), "No CPU device available")

  # split_fn always errors — meta backend must fall back to MIRRORED and stop
  # calling the callback after the first failure (sticky error flag).
  split_fn <- function(tensor_info, n_devs) {
    stop("intentional split_fn failure")
  }

  meta_dev <- ggml_backend_meta_device(list(cpu_dev, cpu_dev), split_fn)
  meta_backend <- ggml_backend_dev_init(meta_dev)
  skip_if(is.null(meta_backend), "Could not init meta backend")

  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6)
  buffer <- ggml_backend_alloc_ctx_tensors(ctx, meta_backend)
  skip_if(is.null(buffer), "Could not allocate meta buffer despite failing split_fn")

  data_in <- as.numeric(seq_len(6))
  ggml_backend_tensor_set_data(tensor, data_in)
  data_out <- ggml_backend_tensor_get_data(tensor)
  expect_equal(data_out, data_in)

  ggml_backend_buffer_free(buffer)
  ggml_free(ctx)
  ggml_backend_free(meta_backend)
})
