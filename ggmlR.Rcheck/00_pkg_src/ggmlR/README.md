# ggmlR - CPU/GPU Tensor Operations for R

[![R-hub check on the R Consortium cluster](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml/badge.svg)](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml)

R bindings for the [GGML](https://github.com/ggml-org/ggml) tensor library with CPU and optional Vulkan GPU acceleration. Provides low-level tensor operations and a high-level Keras-like API for building, training, and deploying neural networks in R. Serves as the backend engine for [llamaR](https://github.com/Zabis13/llamaR) (LLM inference) and [sdR](https://github.com/Zabis13/sdR) (Stable Diffusion image generation).

## Features

- **CPU + GPU**: Efficient CPU operations with optional Vulkan GPU acceleration (NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno)
- **Keras-like Sequential API**: Dense, Conv1D, Conv2D, MaxPooling2D, BatchNorm, Flatten layers with automatic shape inference
- **Training**: AdamW and SGD optimizers, MSE and cross-entropy losses, training history with plots
- **Quantization**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16 formats for memory-efficient inference
- **LLM & Diffusion backend**: Powers llamaR and sdR packages with GPU support
- **No external dependencies**: All C/C++ code included in the package

## Vulkan GPU Acceleration

ggmlR includes built-in Vulkan GPU support. When enabled, the Vulkan backend accelerates 90%+ of ML/DL operations directly on GPU.

### GPU-Accelerated Operations

| Category | Operations |
|----------|-----------|
| Matrix math | `mul_mat` with cooperative matrix extensions, quantized GEMM (Q4_0, Q4_1, Q8_0, F16) |
| Element-wise | `add`, `mul`, `div`, `sqr`, `sqrt`, `clamp` |
| Activations | `gelu`, `silu`, `relu`, `soft_max` |
| Normalization | `norm`, `rms_norm`, `group_norm` |
| Attention | `rope`, `rope_neox`, `flash_attn` |
| Pooling | `pool_2d` |
| Convolutions | `conv_1d`, `conv_2d` |
| Reshaping | `transpose`, `permute`, `view`, `concat` |

### Architectural Advantages

- **Cross-platform**: Works on NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno GPUs
- **Batch submissions**: Multiple operations in a single command buffer to reduce overhead
- **Cooperative matrix extensions**: Hardware-accelerated matrix ops on modern GPUs
- **Pipeline optimization**: Pre-compiled shaders, Split-K matmul for large matrices
- **5x-20x speedup** over CPU for typical ML workloads

### Checking GPU Availability

```r
library(ggmlR)
ggml_vulkan_available()  # TRUE if Vulkan GPU is detected
ggml_vulkan_status()     # Detailed GPU information
```

## Installation

```r
install.packages("ggmlR")
```

Vulkan GPU support is auto-detected at build time. If `libvulkan-dev` and `glslc` are installed, Vulkan is enabled automatically. Otherwise, the package builds with CPU only.

To force Vulkan on (error if deps missing) or off:
```r
install.packages("ggmlR", configure.args = "--with-vulkan")    # require Vulkan
install.packages("ggmlR", configure.args = "--without-vulkan")  # force CPU only
```

### Ubuntu / Debian

```bash
# Install Vulkan build dependencies (optional, for GPU support)
sudo apt install libvulkan-dev glslc

# Install from R — Vulkan will be detected automatically
install.packages("ggmlR")
```

For CPU-only builds, no additional system packages are required.

### Windows

```bash
# 1. Install Rtools (provides C++17 compiler)
#    Download from: https://cran.r-project.org/bin/windows/Rtools/

# 2. (Optional) Install Vulkan SDK for GPU support
#    Download from: https://vulkan.lunarg.com/sdk/home
#    The installer sets the VULKAN_SDK environment variable automatically.

# 3. Install from R
install.packages("ggmlR")
```

If the `VULKAN_SDK` environment variable is set and `glslc.exe` is found, Vulkan GPU support is enabled automatically. Otherwise, the package builds with CPU only.

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

### Keras-like Sequential Model (MNIST Example)
```r
library(ggmlR)

# Build model
model <- ggml_model_sequential(input_shape = c(28, 28, 1)) |>
  ggml_layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  ggml_layer_max_pooling_2d(pool_size = 2) |>
  ggml_layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  ggml_layer_max_pooling_2d(pool_size = 2) |>
  ggml_layer_flatten() |>
  ggml_layer_dense(units = 128, activation = "relu") |>
  ggml_layer_dense(units = 10, activation = "softmax")

# Compile
model <- ggml_compile(model, optimizer = "adamw", loss = "cross_entropy")

# Train
model <- ggml_fit(model, x_train, y_train,
                  epochs = 10, batch_size = 32,
                  validation_data = list(x_test, y_test))

# Plot training history
plot(model$history)

# Evaluate
ggml_evaluate(model, x_test, y_test)

# Predict
classes <- ggml_predict_classes(model, x_new)
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

### Core Operations
- `ggml_mul_mat()` - Matrix multiplication
- `ggml_add()`, `ggml_mul()`, `ggml_div()` - Element-wise arithmetic
- `ggml_sqr()`, `ggml_sqrt()`, `ggml_abs()` - Math functions
- `ggml_clamp()`, `ggml_floor()`, `ggml_ceil()`, `ggml_round()` - Rounding

### Activations
- `ggml_relu()`, `ggml_gelu()`, `ggml_silu()` - Standard activations
- `ggml_geglu_split()`, `ggml_swiglu_split()`, `ggml_reglu_split()` - GLU variants
- `ggml_soft_max()` - Softmax

### Neural Network Layers
- `ggml_norm()`, `ggml_rms_norm()`, `ggml_group_norm()` - Normalization
- `ggml_conv_1d()`, `ggml_conv_2d()` - Convolutions
- `ggml_pool_2d()` - Pooling
- `ggml_rope()` - Rotary position embeddings

### Sequential Model API
- `ggml_model_sequential()` - Create model
- `ggml_layer_dense()` - Fully connected layer
- `ggml_layer_conv_1d()`, `ggml_layer_conv_2d()` - Convolution layers
- `ggml_layer_max_pooling_2d()` - Pooling layer
- `ggml_layer_batch_norm()` - Batch normalization
- `ggml_layer_flatten()` - Flatten layer
- `ggml_compile()`, `ggml_fit()`, `ggml_evaluate()`, `ggml_predict()` - Training workflow

## Use Cases

### LLM Inference
Backend engine for [llamaR](https://github.com/Zabis13/llamaR):
- Quantized model loading and inference
- Transformer layers with GPU acceleration
- Token-by-token generation
- Memory-efficient with quantization

### Stable Diffusion
Backend engine for [sdR](https://github.com/Zabis13/sdR):
- U-Net, attention, normalization on GPU
- Timestep embeddings for diffusion models
- Residual connections

### Training Neural Networks
Train models directly in R:
- Keras-like API for quick prototyping
- AdamW/SGD optimizers with gradient accumulation
- Training history, evaluation, prediction

## Performance

### CPU
- SIMD vectorization (AVX2, AVX-512, ARM NEON)
- OpenMP multi-threading
- Cache-friendly memory layout

### GPU (Vulkan)
- 5x-20x speedup over CPU for ML workloads
- Batch command submissions
- Cooperative matrix extensions
- Pre-compiled shader pipelines

## System Requirements

- C++17 compiler
- R >= 4.1.0
- **Optional**: `libvulkan-dev` and `glslc` for GPU acceleration (auto-detected at build time)
- Supported platforms: Linux, macOS, Windows (x86-64, ARM64)

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

- GGML library: https://github.com/ggml-org/ggml
- llamaR (LLM inference): https://github.com/Zabis13/llamaR
- sdR (Stable Diffusion): https://github.com/Zabis13/sdR

## Acknowledgements

- Thanks to the R community for the inspiration.
- Code structure and documentation were partially refined with the assistance of generative AI tools.
