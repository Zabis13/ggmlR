# ggmlR - CPU/GPU Tensor Operations for R

[![R-hub check on the R Consortium cluster](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml/badge.svg)](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml)

R bindings for the GGML tensor library, optimized for CPU and GPU computations. This package provides low-level tensor operations for machine learning, particularly useful for LLM inference and other deep learning tasks on CPU.

## Features

- ✅ Efficient CPU tensor operations
- ✅ Support for multiple data types (F32, F16, quantized formats)
- ✅ Common neural network operations (matmul, activations, normalization)
- ✅ Computation graph building and execution
- ✅ Memory-efficient design
- ✅ No external dependencies (all C/C++ code included)

## Installation
```r
# From source
install.packages("ggmlR")

# with vulkan for GPU
install.packages("ggmlR", configure.args = "--with-vulkan")
```

## Quick Start

### Basic Tensor Operations
```r
library(ggmlR)

# Initialize context
ctx <- ggml_init(16 * 1024 * 1024)  # 16MB

# Create tensors
a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)

# Set data
ggml_set_f32(a, rnorm(10))
ggml_set_f32(b, rnorm(10))

# Perform operations
c <- ggml_add(ctx, a, b)

# Compute
graph <- ggml_build_forward_expand(ctx, c)
ggml_graph_compute(ctx, graph)

# Get results
result <- ggml_get_f32(c)

# Cleanup
ggml_free(ctx)
```

### Matrix Multiplication
```r
ctx <- ggml_init(16 * 1024 * 1024)

# Create matrices
A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 200)  # 100x200
B <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 200, 50)   # 200x50

# Initialize with random data
ggml_set_f32(A, rnorm(100 * 200))
ggml_set_f32(B, rnorm(200 * 50))

# Matrix multiplication: C = A * B (100x50)
C <- ggml_mul_mat(ctx, A, B)

# Compute
graph <- ggml_build_forward_expand(ctx, C)
ggml_graph_compute(ctx, graph)

result <- ggml_get_f32(C)

ggml_free(ctx)
```

### Neural Network Layer
```r
ctx <- ggml_init(128 * 1024 * 1024)

# Input
input <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128)
ggml_set_f32(input, rnorm(128))

# Weights and bias
W <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 256)
b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256)

ggml_set_f32(W, rnorm(128 * 256, sd = 0.01))
ggml_set_f32(b, rep(0, 256))

# Forward: GELU(W * input + b)
h <- ggml_mul_mat(ctx, W, input)
h <- ggml_add(ctx, h, b)
output <- ggml_gelu(ctx, h)

# Compute
graph <- ggml_build_forward_expand(ctx, output)
ggml_graph_compute(ctx, graph)

result <- ggml_get_f32(output)

ggml_free(ctx)
```

## Supported Operations

### Tensor Creation
- `ggml_new_tensor_1d()` - 1D tensor (vector)
- `ggml_new_tensor_2d()` - 2D tensor (matrix)
- `ggml_new_tensor_3d()` - 3D tensor
- `ggml_new_tensor_4d()` - 4D tensor

### Data Types
- `GGML_TYPE_F32` - 32-bit float
- `GGML_TYPE_F16` - 16-bit float
- `GGML_TYPE_Q4_0`, `GGML_TYPE_Q4_1` - 4-bit quantized
- `GGML_TYPE_Q5_0`, `GGML_TYPE_Q5_1` - 5-bit quantized
- `GGML_TYPE_Q8_0`, `GGML_TYPE_Q8_1` - 8-bit quantized

### Operations
- `ggml_mul_mat()` - Matrix multiplication
- `ggml_add()` - Element-wise addition
- `ggml_mul()` - Element-wise multiplication
- `ggml_relu()` - ReLU activation
- `ggml_gelu()` - GELU activation
- `ggml_silu()` - SiLU/Swish activation
- `ggml_norm()` - Layer normalization
- `ggml_rms_norm()` - RMS normalization

## Use Cases

### LLM Inference
This package is designed for running language model inference on CPU or GPU:
- Load quantized model weights
- Build transformer layers
- Run token-by-token generation
- Efficient memory usage with quantization

### Stable Diffusion
Can be used for diffusion model inference on CPU or GPU:
- U-Net architecture building blocks
- Attention mechanisms
- Residual connections
- Normalization layers

## Performance

Optimized for x86-64 CPUs with:
- SIMD vectorization
- Multi-threading support
- Efficient memory layout
- Cache-friendly operations

## Future Plans
- Additional operations (softmax, attention, etc.)
- Model loading utilities
- Pre-built model examples

## System Requirements

- C++17 compiler
- x86-64 CPU (ARM support planned)
- R >= 4.0.0

## License

MIT License

## Citation

If you use this package in your research, please cite:
```
@software{ggmlR,
  author = {Yuri Baramykov},
  title = {ggmlR: CPU/GPU Tensor Operations for R},
  year = {2026},
  url = {https://github.com/Zabis13/ggmlR}
}
```

## See Also

- GGML library: https://github.com/ggerganov/ggml
- llama.cpp: https://github.com/ggerganov/llama.cpp


## Acknowledgements

- Thanks to the R community for the inspiration.
- Code structure and documentation were partially refined with the assistance of generative AI tools.
