# Tests for extended backend functions

test_that("device type constants work", {
  expect_type(ggml_backend_device_type_cpu(), "integer")
  expect_type(ggml_backend_device_type_gpu(), "integer")
  expect_type(ggml_backend_device_type_igpu(), "integer")
  expect_type(ggml_backend_device_type_accel(), "integer")

  # CPU type should be 0
  expect_equal(ggml_backend_device_type_cpu(), 0L)
})

test_that("buffer usage constants work", {
  expect_type(ggml_backend_buffer_usage_any(), "integer")
  expect_type(ggml_backend_buffer_usage_weights(), "integer")
  expect_type(ggml_backend_buffer_usage_compute(), "integer")
})

test_that("device enumeration works", {
  count <- ggml_backend_dev_count()
  expect_type(count, "double")
  expect_gte(count, 1)  # At least CPU device

  # Get first device
  dev <- ggml_backend_dev_get(0)
  expect_false(is.null(dev))

  # Get by name
  dev_cpu <- ggml_backend_dev_by_name("CPU")
  expect_false(is.null(dev_cpu))

  # Get by type (CPU)
  dev_cpu2 <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
  expect_false(is.null(dev_cpu2))
})

test_that("device properties work", {
  dev <- ggml_backend_dev_get(0)
  skip_if(is.null(dev), "No device available")

  # Device name
  name <- ggml_backend_dev_name(dev)
  expect_type(name, "character")
  expect_true(nchar(name) > 0)

  # Device description
  desc <- ggml_backend_dev_description(dev)
  expect_type(desc, "character")

  # Device memory
  mem <- ggml_backend_dev_memory(dev)
  expect_type(mem, "double")
  expect_length(mem, 2)
  expect_true("free" %in% names(mem))
  expect_true("total" %in% names(mem))

  # Device type
  dtype <- ggml_backend_dev_type(dev)
  expect_type(dtype, "integer")

  # Device properties
  props <- ggml_backend_dev_get_props(dev)
  expect_type(props, "list")
  expect_true("name" %in% names(props))
  expect_true("description" %in% names(props))
})

test_that("backend registry works", {
  count <- ggml_backend_reg_count()
  expect_type(count, "double")
  expect_gte(count, 1)  # At least CPU registry

  # Get first registry
  reg <- ggml_backend_reg_get(0)
  expect_false(is.null(reg))

  # Get registry name
  name <- ggml_backend_reg_name(reg)
  expect_type(name, "character")

  # Get device count in registry
  dev_count <- ggml_backend_reg_dev_count(reg)
  expect_type(dev_count, "double")
  expect_gte(dev_count, 1)

  # Get device from registry
  dev <- ggml_backend_reg_dev_get(reg, 0)
  expect_false(is.null(dev))

  # Get registry by name
  reg_cpu <- ggml_backend_reg_by_name("CPU")
  expect_false(is.null(reg_cpu))
})

test_that("backend initialization works", {
  # Init by name
  backend_cpu <- ggml_backend_init_by_name("CPU")
  expect_false(is.null(backend_cpu))

  # Get backend name
  name <- ggml_backend_name(backend_cpu)
  expect_type(name, "character")
  expect_equal(name, "CPU")

  # Get device from backend
  dev <- ggml_backend_get_device(backend_cpu)
  expect_false(is.null(dev))

  # Synchronize
  ggml_backend_synchronize(backend_cpu)

  # Free backend
  ggml_backend_free(backend_cpu)
})

test_that("init by type works", {
  backend <- ggml_backend_init_by_type(ggml_backend_device_type_cpu())
  expect_false(is.null(backend))
  ggml_backend_free(backend)
})

test_that("init best works", {
  backend <- ggml_backend_init_best()
  expect_false(is.null(backend))
  ggml_backend_free(backend)
})

test_that("device init works", {
  dev <- ggml_backend_dev_get(0)
  skip_if(is.null(dev), "No device available")

  backend <- ggml_backend_dev_init(dev)
  expect_false(is.null(backend))
  ggml_backend_free(backend)
})

test_that("buffer management works", {
  backend <- ggml_backend_cpu_init()
  skip_if(is.null(backend), "Could not initialize CPU backend")

  # Create context with no_alloc = TRUE for backend allocation
  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)

  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  skip_if(is.null(buffer), "Could not allocate buffer")

  # Buffer name
  name <- ggml_backend_buffer_name(buffer)
  expect_type(name, "character")

  # Buffer size
  size <- ggml_backend_buffer_get_size(buffer)
  expect_type(size, "double")
  expect_gt(size, 0)

  # Is host memory
  is_host <- ggml_backend_buffer_is_host(buffer)
  expect_type(is_host, "logical")
  expect_true(is_host)  # CPU buffer should be host memory

  # Buffer usage
  usage <- ggml_backend_buffer_get_usage(buffer)
  expect_type(usage, "integer")

  # Set usage
  ggml_backend_buffer_set_usage(buffer, ggml_backend_buffer_usage_weights())
  usage2 <- ggml_backend_buffer_get_usage(buffer)
  expect_equal(usage2, ggml_backend_buffer_usage_weights())

  # Clear buffer
  ggml_backend_buffer_clear(buffer, 0L)

  # Reset buffer
  ggml_backend_buffer_reset(buffer)

  # Free resources
  ggml_backend_buffer_free(buffer)
  ggml_free(ctx)
  ggml_backend_free(backend)
})

test_that("tensor data operations work", {
  backend <- ggml_backend_cpu_init()
  skip_if(is.null(backend), "Could not initialize CPU backend")

  # Create context with no_alloc = TRUE for backend allocation
  ctx <- ggml_init(16 * 1024, no_alloc = TRUE)
  tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)

  buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  skip_if(is.null(buffer), "Could not allocate buffer")

  # Set data first
  data_in <- as.numeric(1:10)
  ggml_backend_tensor_set_data(tensor, data_in)

  # Get data - default offset=0, n_elements=NULL (all)
  data_out <- ggml_backend_tensor_get_data(tensor)
  expect_equal(data_out, data_in)

  # Free resources
  ggml_backend_buffer_free(buffer)
  ggml_free(ctx)
  ggml_backend_free(backend)
})

test_that("device supports operation check works", {
  dev <- ggml_backend_dev_get(0)
  skip_if(is.null(dev), "No device available")

  ctx <- ggml_init(16 * 1024)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
  c <- ggml_add(ctx, a, b)

  # Check support (should be TRUE for CPU and basic ops)
  supports <- ggml_backend_dev_supports_op(dev, c)
  expect_type(supports, "logical")

  ggml_free(ctx)
})

test_that("load all backends works", {
  # Should not error
  expect_no_error(ggml_backend_load_all())
})

test_that("event functions exist", {
  # Just verify functions exist - events may not work for CPU
  expect_true(is.function(ggml_backend_event_new))
  expect_true(is.function(ggml_backend_event_free))
  expect_true(is.function(ggml_backend_event_record))
  expect_true(is.function(ggml_backend_event_synchronize))
  expect_true(is.function(ggml_backend_event_wait))
})

test_that("graph plan functions exist", {
  expect_true(is.function(ggml_backend_graph_plan_create))
  expect_true(is.function(ggml_backend_graph_plan_free))
  expect_true(is.function(ggml_backend_graph_plan_compute))
})

test_that("async functions exist", {
  expect_true(is.function(ggml_backend_tensor_set_async))
  expect_true(is.function(ggml_backend_tensor_get_async))
  expect_true(is.function(ggml_backend_tensor_copy_async))
})
