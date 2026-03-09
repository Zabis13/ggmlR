test_that("scheduler basic functionality works", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")

  device_count <- ggml_vulkan_device_count()
  skip_if(device_count == 0, "No Vulkan devices found")

  # Single GPU scheduler
  gpu1 <- ggml_vulkan_init(0)
  expect_true(!is.null(gpu1))

  sched <- ggml_backend_sched_new(list(gpu1), parallel = TRUE)
  expect_true(!is.null(sched))

  # Check number of backends (GPU + auto-added CPU)
  n_backends <- ggml_backend_sched_get_n_backends(sched)
  expect_equal(n_backends, 2)  # 1 GPU + 1 CPU (auto-added)

  # Get backend
  backend <- ggml_backend_sched_get_backend(sched, 0)
  expect_true(!is.null(backend))

  # Cleanup
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu1)
})

test_that("multi-GPU scheduler works", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")

  device_count <- ggml_vulkan_device_count()
  skip_if(device_count < 2, "Need at least 2 GPUs for multi-GPU test")

  # Create two GPU backends
  gpu1 <- ggml_vulkan_init(0)
  gpu2 <- ggml_vulkan_init(1)

  expect_true(!is.null(gpu1))
  expect_true(!is.null(gpu2))

  # Create scheduler with both GPUs
  sched <- ggml_backend_sched_new(list(gpu1, gpu2), parallel = TRUE)
  expect_true(!is.null(sched))

  # Check we have 3 backends (2 GPUs + 1 CPU auto-added)
  n_backends <- ggml_backend_sched_get_n_backends(sched)
  expect_equal(n_backends, 3)

  # Cleanup
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu1)
  ggml_vulkan_free(gpu2)
})

test_that("scheduler can compute simple graphs", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")

  device_count <- ggml_vulkan_device_count()
  skip_if(device_count == 0, "No Vulkan devices found")

  # Create GPU backend
  gpu <- ggml_vulkan_init(0)

  # Create context with no_alloc = TRUE (important for scheduler!)
  ctx <- ggml_init_auto(64 * 1024 * 1024, no_alloc = TRUE)

  # Create simple computation: c = a + b
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  c <- ggml_add(ctx, a, b)

  # Build computation graph
  graph <- ggml_build_forward_expand(ctx, c)

  # Create scheduler
  sched <- ggml_backend_sched_new(list(gpu), parallel = TRUE)

  # Allocate graph
  alloc_success <- ggml_backend_sched_alloc_graph(sched, graph)
  expect_true(alloc_success)

  # Set input data AFTER allocation
  data_a <- rnorm(1000)
  data_b <- rnorm(1000)
  ggml_backend_tensor_set_data(a, data_a)
  ggml_backend_tensor_set_data(b, data_b)

  # Compute graph using scheduler
  status <- ggml_backend_sched_graph_compute(sched, graph)
  expect_equal(status, 0)  # 0 = success

  # Get results using backend API
  result <- ggml_backend_tensor_get_data(c)

  # Verify computation
  expected <- data_a + data_b
  expect_equal(result, expected, tolerance = 1e-5)

  # Check statistics
  n_splits <- ggml_backend_sched_get_n_splits(sched)
  expect_true(n_splits >= 0)

  # Cleanup
  ggml_free(ctx)
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu)
})

test_that("multi-GPU computation distributes work", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")

  device_count <- ggml_vulkan_device_count()
  skip_if(device_count < 2, "Need at least 2 GPUs")

  # Create two GPU backends
  gpu1 <- ggml_vulkan_init(0)
  gpu2 <- ggml_vulkan_init(1)

  # Create context with no_alloc = TRUE
  ctx <- ggml_init_auto(256 * 1024 * 1024, no_alloc = TRUE)

  # Create larger tensors to encourage splitting
  n <- 100000
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)

  # Build graph: c = a + b, d = c * 2
  c <- ggml_add(ctx, a, b)
  two <- ggml_new_f32(ctx, 2.0)
  d <- ggml_mul(ctx, c, two)
  graph <- ggml_build_forward_expand(ctx, d)

  # Create multi-GPU scheduler
  sched <- ggml_backend_sched_new(list(gpu1, gpu2), parallel = TRUE)

  # Allocate graph
  alloc_success <- ggml_backend_sched_alloc_graph(sched, graph)
  expect_true(alloc_success)

  # Set input data AFTER allocation
  data_a <- rnorm(n)
  data_b <- rnorm(n)
  ggml_backend_tensor_set_data(a, data_a)
  ggml_backend_tensor_set_data(b, data_b)
  ggml_backend_tensor_set_data(two, 2.0)

  # Compute
  status <- ggml_backend_sched_graph_compute(sched, graph)
  expect_equal(status, 0)

  # Get results
  result <- ggml_backend_tensor_get_data(d)

  # Verify computation
  expected <- (data_a + data_b) * 2.0
  expect_equal(result, expected, tolerance = 1e-5)

  # Check that work was distributed (splits > 0 suggests multi-backend usage)
  n_splits <- ggml_backend_sched_get_n_splits(sched)
  cat("\nMulti-GPU splits:", n_splits, "\n")

  n_copies <- ggml_backend_sched_get_n_copies(sched)
  cat("Tensor copies:", n_copies, "\n")

  # Cleanup
  ggml_free(ctx)
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu1)
  ggml_vulkan_free(gpu2)
})

test_that("scheduler reset works", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")

  device_count <- ggml_vulkan_device_count()
  skip_if(device_count == 0, "No Vulkan devices found")

  gpu <- ggml_vulkan_init(0)
  sched <- ggml_backend_sched_new(list(gpu), parallel = TRUE)

  # Reset should not crash
  expect_silent(ggml_backend_sched_reset(sched))

  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu)
})

test_that("scheduler tensor backend assignment works", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")

  device_count <- ggml_vulkan_device_count()
  skip_if(device_count == 0, "No Vulkan devices found")

  gpu <- ggml_vulkan_init(0)
  sched <- ggml_backend_sched_new(list(gpu), parallel = TRUE)

  # Create a simple computation
  ctx <- ggml_init_auto(64 * 1024 * 1024, no_alloc = TRUE)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
  c <- ggml_add(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, c)

  # Reserve memory first
  ggml_backend_sched_reserve(sched, graph)

  # Get GPU backend from scheduler
  backend_gpu <- ggml_backend_sched_get_backend(sched, 0)

  # Set tensor backend manually (before allocation)
  expect_silent(ggml_backend_sched_set_tensor_backend(sched, a, backend_gpu))

  # Get tensor backend
  assigned_backend <- ggml_backend_sched_get_tensor_backend(sched, a)
  expect_true(!is.null(assigned_backend))

  # Cleanup
  ggml_free(ctx)
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu)
})

test_that("async compute and synchronize work", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")

  device_count <- ggml_vulkan_device_count()
  skip_if(device_count == 0, "No Vulkan devices found")

  gpu <- ggml_vulkan_init(0)
  ctx <- ggml_init_auto(64 * 1024 * 1024, no_alloc = TRUE)

  # Create simple computation
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000)
  c <- ggml_add(ctx, a, b)
  graph <- ggml_build_forward_expand(ctx, c)

  # Create scheduler
  sched <- ggml_backend_sched_new(list(gpu), parallel = TRUE)

  # Allocate graph
  alloc_success <- ggml_backend_sched_alloc_graph(sched, graph)
  expect_true(alloc_success)

  # Set input data AFTER allocation
  data_a <- rnorm(1000)
  data_b <- rnorm(1000)
  ggml_backend_tensor_set_data(a, data_a)
  ggml_backend_tensor_set_data(b, data_b)

  # Compute asynchronously
  status <- ggml_backend_sched_graph_compute_async(sched, graph)
  expect_equal(status, 0)

  # Synchronize
  expect_silent(ggml_backend_sched_synchronize(sched))

  # Get results
  result <- ggml_backend_tensor_get_data(c)
  expected <- data_a + data_b
  expect_equal(result, expected, tolerance = 1e-5)

  # Cleanup
  ggml_free(ctx)
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu)
})
