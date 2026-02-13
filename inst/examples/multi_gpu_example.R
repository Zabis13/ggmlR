# Multi-GPU Example with Backend Scheduler
# This example demonstrates how to use multiple GPUs with ggmlR

library(ggmlR)

# Check if Vulkan is available
if (!ggml_vulkan_available()) {
  stop("Vulkan is not available. Install with: R CMD INSTALL --configure-args='--with-vulkan' .")
}

# Check available devices
ggml_vulkan_status()
n_devices <- ggml_vulkan_device_count()

if (n_devices == 0) {
  stop("No Vulkan devices found")
}

cat("\n=== Multi-GPU Backend Scheduler Example ===\n\n")

# Example 1: Single GPU using scheduler
cat("Example 1: Single GPU computation\n")
cat("----------------------------------\n")

gpu1 <- ggml_vulkan_init(0)
sched <- ggml_backend_sched_new(list(gpu1), parallel = TRUE)

cat("Scheduler created with", ggml_backend_sched_get_n_backends(sched), "backend(s)\n\n")

# Create a simple computation
ctx <- ggml_init(64 * 1024 * 1024)  # 64MB

n <- 10000
a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)

data_a <- rnorm(n)
data_b <- rnorm(n)
ggml_set_f32(a, data_a)
ggml_set_f32(b, data_b)

# c = a + b
c <- ggml_add(ctx, a, b)
graph <- ggml_build_forward_expand(ctx, c)

# Reserve memory and compute
ggml_backend_sched_reserve(sched, graph)

t1 <- Sys.time()
ggml_backend_sched_graph_compute(sched, graph)
t2 <- Sys.time()

result <- ggml_get_f32(c)
cat("Computation time:", format(difftime(t2, t1, units = "secs")), "\n")
cat("Splits:", ggml_backend_sched_get_n_splits(sched), "\n")
cat("Copies:", ggml_backend_sched_get_n_copies(sched), "\n")

# Verify result
expected <- data_a + data_b
max_error <- max(abs(result - expected))
cat("Max error:", max_error, "\n\n")

# Cleanup
ggml_free(ctx)
ggml_backend_sched_free(sched)
ggml_vulkan_free(gpu1)


# Example 2: Multi-GPU if available
if (n_devices >= 2) {
  cat("\nExample 2: Multi-GPU computation\n")
  cat("----------------------------------\n")

  # Create two GPU backends
  gpu1 <- ggml_vulkan_init(0)
  gpu2 <- ggml_vulkan_init(1)

  cat("GPU 1:", ggml_vulkan_backend_name(gpu1), "\n")
  cat("GPU 2:", ggml_vulkan_backend_name(gpu2), "\n\n")

  # Create scheduler with both GPUs
  sched <- ggml_backend_sched_new(list(gpu1, gpu2), parallel = TRUE)
  cat("Multi-GPU scheduler created with", ggml_backend_sched_get_n_backends(sched), "backends\n\n")

  # Create larger computation to benefit from multiple GPUs
  ctx <- ggml_init(256 * 1024 * 1024)  # 256MB

  n <- 1000000  # 1 million elements
  cat("Creating tensors with", n, "elements each\n")

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
  b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)

  data_a <- rnorm(n)
  data_b <- rnorm(n)

  cat("Setting data...\n")
  ggml_set_f32(a, data_a)
  ggml_set_f32(b, data_b)

  # Build computation graph: d = (a + b) * (a - b)
  cat("Building computation graph...\n")
  c <- ggml_add(ctx, a, b)
  e <- ggml_sub(ctx, a, b)
  d <- ggml_mul(ctx, c, e)
  graph <- ggml_build_forward_expand(ctx, d)

  # Reserve memory
  cat("Reserving memory...\n")
  ggml_backend_sched_reserve(sched, graph)

  # Compute using both GPUs
  cat("Computing on multiple GPUs...\n")
  t1 <- Sys.time()
  status <- ggml_backend_sched_graph_compute(sched, graph)
  t2 <- Sys.time()

  cat("Status:", status, "(0 = success)\n")
  cat("Computation time:", format(difftime(t2, t1, units = "secs")), "\n")
  cat("Graph splits:", ggml_backend_sched_get_n_splits(sched), "\n")
  cat("Tensor copies:", ggml_backend_sched_get_n_copies(sched), "\n")

  # Get and verify result
  result <- ggml_get_f32(d)
  expected <- (data_a + data_b) * (data_a - data_b)
  max_error <- max(abs(result - expected))
  cat("Max error:", max_error, "\n\n")

  # Cleanup
  ggml_free(ctx)
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu1)
  ggml_vulkan_free(gpu2)

} else {
  cat("\nSkipping multi-GPU example (only", n_devices, "GPU(s) available)\n")
}


# Example 3: Matrix multiplication with multi-GPU
if (n_devices >= 2) {
  cat("\nExample 3: Multi-GPU Matrix Multiplication\n")
  cat("-------------------------------------------\n")

  gpu1 <- ggml_vulkan_init(0)
  gpu2 <- ggml_vulkan_init(1)
  sched <- ggml_backend_sched_new(list(gpu1, gpu2), parallel = TRUE)

  ctx <- ggml_init(512 * 1024 * 1024)  # 512MB

  # Create large matrices
  m <- 2048
  n <- 2048
  k <- 2048

  cat(sprintf("Matrix A: %dx%d\n", m, k))
  cat(sprintf("Matrix B: %dx%d\n", k, n))
  cat(sprintf("Matrix C: %dx%d\n", m, n))

  A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m)
  B <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, k)

  cat("Initializing matrices...\n")
  ggml_set_f32(A, rnorm(m * k, sd = 0.01))
  ggml_set_f32(B, rnorm(k * n, sd = 0.01))

  # Matrix multiplication: C = A * B
  cat("Building matrix multiplication graph...\n")
  C <- ggml_mul_mat(ctx, A, B)
  graph <- ggml_build_forward_expand(ctx, C)

  # Reserve and compute
  cat("Reserving memory...\n")
  ggml_backend_sched_reserve(sched, graph)

  cat("Computing matrix multiplication on", ggml_backend_sched_get_n_backends(sched), "GPUs...\n")
  t1 <- Sys.time()
  status <- ggml_backend_sched_graph_compute(sched, graph)
  t2 <- Sys.time()

  elapsed <- as.numeric(difftime(t2, t1, units = "secs"))
  cat("Status:", status, "\n")
  cat("Time:", sprintf("%.3f", elapsed), "seconds\n")

  # Calculate GFLOPS
  flops <- 2.0 * m * n * k  # 2 operations (mul + add) per element
  gflops <- (flops / elapsed) / 1e9
  cat("Performance:", sprintf("%.2f", gflops), "GFLOPS\n")

  cat("Splits:", ggml_backend_sched_get_n_splits(sched), "\n")
  cat("Copies:", ggml_backend_sched_get_n_copies(sched), "\n\n")

  # Cleanup
  ggml_free(ctx)
  ggml_backend_sched_free(sched)
  ggml_vulkan_free(gpu1)
  ggml_vulkan_free(gpu2)
}

cat("\n=== All examples completed ===\n")
