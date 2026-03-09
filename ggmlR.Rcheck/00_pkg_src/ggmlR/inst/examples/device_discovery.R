# Device Discovery and Backend Management Example
# Demonstrates the extended backend API for multi-device inference

library(ggmlR)

cat("=== GGML Device Discovery ===\n\n")

# Load all available backends
cat("Loading all backends...\n")
ggml_backend_load_all()

# ============================================================================
# Part 1: Device Enumeration
# ============================================================================
cat("\n--- Device Enumeration ---\n")

n_devices <- ggml_backend_dev_count()
cat("Total devices found:", n_devices, "\n\n")

if (n_devices == 0) {
  stop("No devices found!")
}

# List all devices with properties
for (i in seq_len(n_devices)) {
  dev <- ggml_backend_dev_get(i - 1)  # 0-based index

  name <- ggml_backend_dev_name(dev)
  desc <- ggml_backend_dev_description(dev)
  dtype <- ggml_backend_dev_type(dev)
  mem <- ggml_backend_dev_memory(dev)

  type_name <- switch(as.character(dtype),
    "0" = "CPU",
    "1" = "GPU",
    "2" = "Integrated GPU",
    "3" = "Accelerator",
    "Unknown"
  )

  cat(sprintf("Device %d: %s\n", i, name))
  cat(sprintf("  Type: %s\n", type_name))
  cat(sprintf("  Description: %s\n", desc))
  cat(sprintf("  Memory: %.2f GB free / %.2f GB total\n",
              mem["free"] / 1e9, mem["total"] / 1e9))
  cat("\n")
}

# ============================================================================
# Part 2: Backend Registry
# ============================================================================
cat("--- Backend Registry ---\n")

n_regs <- ggml_backend_reg_count()
cat("Registered backends:", n_regs, "\n\n")

for (i in seq_len(n_regs)) {
  reg <- ggml_backend_reg_get(i - 1)
  reg_name <- ggml_backend_reg_name(reg)
  dev_count <- ggml_backend_reg_dev_count(reg)

  cat(sprintf("Registry %d: %s (%d device(s))\n", i, reg_name, dev_count))
}
cat("\n")

# ============================================================================
# Part 3: Find Device by Type
# ============================================================================
cat("--- Find by Type ---\n")

# Find CPU device
cpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_cpu())
if (!is.null(cpu_dev)) {
  cat("CPU device found:", ggml_backend_dev_name(cpu_dev), "\n")
}

# Find GPU device
gpu_dev <- ggml_backend_dev_by_type(ggml_backend_device_type_gpu())
if (!is.null(gpu_dev)) {
  cat("GPU device found:", ggml_backend_dev_name(gpu_dev), "\n")

  # Get full properties
  props <- ggml_backend_dev_get_props(gpu_dev)
  cat("  Capabilities:\n")
  cat("    Async:", props$caps["async"], "\n")
  cat("    Host buffer:", props$caps["host_buffer"], "\n")
  cat("    Events:", props$caps["events"], "\n")
} else {
  cat("No discrete GPU found\n")
}
cat("\n")

# ============================================================================
# Part 4: Initialize Best Backend
# ============================================================================
cat("--- Initialize Best Backend ---\n")

backend <- ggml_backend_init_best()
if (!is.null(backend)) {
  dev <- ggml_backend_get_device(backend)
  name <- ggml_backend_dev_name(dev)
  cat("Best backend initialized:", name, "\n")
  ggml_backend_free(backend)
}
cat("\n")

# ============================================================================
# Part 5: Simple Computation with Device
# ============================================================================
cat("--- Simple Computation ---\n")

# Initialize backend by name
backend <- ggml_backend_init_by_name("CPU")
if (is.null(backend)) {
  stop("Failed to initialize CPU backend")
}

# Create context
ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)

# Create tensors
a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
c <- ggml_add(ctx, a, b)

# Allocate on backend
buffer <- ggml_backend_alloc_ctx_tensors(ctx, backend)

# Set data
ggml_backend_tensor_set_data(a, as.numeric(1:100))
ggml_backend_tensor_set_data(b, as.numeric(101:200))

# Build and compute graph
graph <- ggml_build_forward_expand(ctx, c)
status <- ggml_backend_graph_compute(backend, graph)

cat("Computation status:", status, "(0 = success)\n")

# Get result
result <- ggml_backend_tensor_get_data(c)
cat("Result[1:5]:", paste(result[1:5], collapse=", "), "\n")
cat("Expected[1:5]: 102, 104, 106, 108, 110\n")

# Check buffer properties
cat("\nBuffer info:\n")
cat("  Name:", ggml_backend_buffer_name(buffer), "\n")
cat("  Size:", ggml_backend_buffer_get_size(buffer), "bytes\n")
cat("  Is host:", ggml_backend_buffer_is_host(buffer), "\n")

# Cleanup
ggml_backend_buffer_free(buffer)
ggml_free(ctx)
ggml_backend_free(backend)

cat("\n=== Device Discovery Complete ===\n")
