# Multi-GPU Support in ggmlR

ggmlR supports distributing computation across multiple GPUs using the backend scheduler API.

## Overview

The backend scheduler automatically distributes computation graphs across multiple backends (GPUs or CPU), enabling:

- **Parallel execution** on multiple GPUs
- **Automatic work distribution** across available devices
- **Transparent memory management** with inter-GPU transfers
- **Mixed GPU/CPU computation** for optimal performance

## Basic Usage

### 1. Check Available Devices

```r
library(ggmlR)

# Check Vulkan availability and devices
ggml_vulkan_status()
```

Output:
```
Vulkan: AVAILABLE
  Devices: 2
  [0] Tesla T4
      Memory: 16.08 GB free / 16.36 GB total
  [1] Tesla T4
      Memory: 16.08 GB free / 16.36 GB total
```

### 2. Create Multi-GPU Scheduler

```r
# Initialize GPU backends
gpu1 <- ggml_vulkan_init(0)
gpu2 <- ggml_vulkan_init(1)

# Create scheduler with both GPUs
sched <- ggml_backend_sched_new(
  backends = list(gpu1, gpu2),
  parallel = TRUE,
  graph_size = 2048
)
```

### 3. Compute on Multiple GPUs

```r
# Create computation context
ctx <- ggml_init(256 * 1024 * 1024)

# Create tensors
n <- 1000000
a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)
b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n)

# Set data
ggml_set_f32(a, rnorm(n))
ggml_set_f32(b, rnorm(n))

# Build computation graph
c <- ggml_add(ctx, a, b)
d <- ggml_mul(ctx, c, a)
graph <- ggml_build_forward_expand(ctx, d)

# Reserve memory based on graph requirements
ggml_backend_sched_reserve(sched, graph)

# Compute using both GPUs
status <- ggml_backend_sched_graph_compute(sched, graph)

# Get results
result <- ggml_get_f32(d)

# Check work distribution
cat("Graph splits:", ggml_backend_sched_get_n_splits(sched), "\n")
cat("Tensor copies:", ggml_backend_sched_get_n_copies(sched), "\n")
```

### 4. Cleanup

```r
ggml_free(ctx)
ggml_backend_sched_free(sched)
ggml_vulkan_free(gpu1)
ggml_vulkan_free(gpu2)
```

## Advanced Features

### Manual Tensor Assignment

You can manually control which GPU handles specific tensors:

```r
# Assign tensor 'a' to GPU 1
ggml_backend_sched_set_tensor_backend(sched, a, gpu1)

# Assign tensor 'b' to GPU 2
ggml_backend_sched_set_tensor_backend(sched, b, gpu2)

# Check assignment
backend <- ggml_backend_sched_get_tensor_backend(sched, a)
```

### Asynchronous Computation

```r
# Start computation asynchronously
status <- ggml_backend_sched_graph_compute_async(sched, graph)

# Do other work...

# Wait for completion
ggml_backend_sched_synchronize(sched)

# Get results
result <- ggml_get_f32(d)
```

### Resetting Scheduler

When computing multiple graphs, reset the scheduler between computations:

```r
# Compute first graph
ggml_backend_sched_graph_compute(sched, graph1)

# Reset allocations
ggml_backend_sched_reset(sched)

# Compute second graph
ggml_backend_sched_reserve(sched, graph2)
ggml_backend_sched_graph_compute(sched, graph2)
```

## Performance Considerations

### When to Use Multi-GPU

Multi-GPU is beneficial for:

- **Large tensor operations** (>100MB per tensor)
- **Complex computation graphs** with many operations
- **Matrix multiplications** on large matrices (>1024x1024)
- **Parallel independent operations**

Multi-GPU may have overhead for:

- Small tensors (<10MB)
- Simple operations (single add/mul)
- Sequential dependencies in the graph

### Monitoring Performance

Check graph splitting to understand work distribution:

```r
n_splits <- ggml_backend_sched_get_n_splits(sched)
n_copies <- ggml_backend_sched_get_n_copies(sched)

cat("Graph was split into", n_splits, "parts\n")
cat("Required", n_copies, "tensor copies between GPUs\n")
```

- **More splits** = better distribution across GPUs
- **Fewer copies** = less inter-GPU communication overhead

### Benchmark Example

```r
# Single GPU
gpu1 <- ggml_vulkan_init(0)
sched_single <- ggml_backend_sched_new(list(gpu1))

t1 <- Sys.time()
ggml_backend_sched_graph_compute(sched_single, graph)
t2 <- Sys.time()
time_single <- difftime(t2, t1, units = "secs")

# Multi-GPU
gpu1 <- ggml_vulkan_init(0)
gpu2 <- ggml_vulkan_init(1)
sched_multi <- ggml_backend_sched_new(list(gpu1, gpu2))

t1 <- Sys.time()
ggml_backend_sched_graph_compute(sched_multi, graph)
t2 <- Sys.time()
time_multi <- difftime(t2, t1, units = "secs")

cat("Speedup:", time_single / time_multi, "x\n")
```

## API Reference

### Core Functions

- `ggml_backend_sched_new(backends, parallel, graph_size)` - Create scheduler
- `ggml_backend_sched_free(sched)` - Free scheduler
- `ggml_backend_sched_reserve(sched, graph)` - Reserve memory for graph
- `ggml_backend_sched_graph_compute(sched, graph)` - Compute graph
- `ggml_backend_sched_reset(sched)` - Reset allocations

### Information Functions

- `ggml_backend_sched_get_n_backends(sched)` - Get number of backends
- `ggml_backend_sched_get_backend(sched, index)` - Get specific backend
- `ggml_backend_sched_get_n_splits(sched)` - Get number of graph splits
- `ggml_backend_sched_get_n_copies(sched)` - Get number of tensor copies

### Tensor Assignment

- `ggml_backend_sched_set_tensor_backend(sched, tensor, backend)` - Assign tensor to backend
- `ggml_backend_sched_get_tensor_backend(sched, tensor)` - Get tensor's backend

### Async Operations

- `ggml_backend_sched_graph_compute_async(sched, graph)` - Compute asynchronously
- `ggml_backend_sched_synchronize(sched)` - Wait for async operations

## Examples

See `examples/multi_gpu_example.R` for complete working examples including:

1. Single GPU computation
2. Multi-GPU vector operations
3. Multi-GPU matrix multiplication with performance metrics

## Troubleshooting

### "No Vulkan devices found"

Make sure:
- Vulkan drivers are installed
- Package was compiled with `--with-vulkan` flag
- GPUs are visible to the system

### Poor multi-GPU performance

Possible causes:
- Tensors too small (high overhead-to-computation ratio)
- Sequential graph structure prevents parallelism
- Too many inter-GPU copies

Try:
- Increasing problem size
- Manually assigning tensors to specific GPUs
- Checking `n_splits` and `n_copies` metrics

### Memory errors

If you get out-of-memory errors:
- Reduce tensor sizes
- Use fewer backends
- Check available memory with `ggml_vulkan_device_memory()`

## Technical Details

The scheduler uses the GGML backend scheduler (`ggml_backend_sched_*`) which:

1. Analyzes the computation graph
2. Splits it into subgraphs based on backend capabilities
3. Assigns subgraphs to backends (GPUs)
4. Manages inter-backend tensor transfers
5. Executes subgraphs in parallel when possible
6. Synchronizes results

Priority is given to backends listed first in the `backends` list.
