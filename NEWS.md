# ggmlR 0.5.2

## New Features (LLM and Stable Diffusion support)

* Backend engine for LLM inference (`llamaR`) and Stable Diffusion image generation (`sdR`) with Vulkan GPU acceleration

## Bug Fixes

* Fixed duplicate symbol linker error on macOS ARM64 (x86 guards for 5 repack functions in `arch/x86/repack.cpp`)
* Fixed UBSan "applying non-zero offset to null pointer" in `ggml.c:ggml_graph_nbytes()` — upstream ggml uses NULL pointer arithmetic for size calculation; patched to use `uintptr_t` arithmetic (CRAN m1-san)

## New Features (Stable Diffusion support)

* Added `ggml_timestep_embedding()` — sinusoidal timestep embeddings for diffusion models
* Added N-D indexed tensor access:
  - `ggml_set_f32_nd()` / `ggml_get_f32_nd()` — set/get float by [i0,i1,i2,i3] index
  - `ggml_set_i32_nd()` / `ggml_get_i32_nd()` — set/get int32 by index
  - Backend-aware: auto-detects CPU data vs backend buffer
* Added tensor utilities:
  - `ggml_tensor_nb()` — get tensor byte strides (nb0..nb3)
  - `ggml_tensor_num()` — count tensors in context
  - `ggml_tensor_copy()` — direct memcpy between same-size tensors
  - `ggml_tensor_set_f32_scalar()` — fill all elements with a single value
  - `ggml_get_first_tensor()` / `ggml_get_next_tensor()` — iterate tensors in context
  - `ggml_backend_tensor_get_f32_first()` — read first f32 from backend/CPU tensor
  - `ggml_backend_tensor_get_and_sync()` — read raw bytes with backend synchronization

# ggmlR 0.5.1

## New Features

* Export static library `libggml.a` for linking by dependent packages (llamaR)
* Added `gguf.cpp` for GGUF file format support
* Headers exported via `inst/include/` for `LinkingTo`

## CRAN Submission Fixes

* Expanded acronyms in DESCRIPTION: 'AdamW' (Adam with Weight decay),
  'SGD' (Stochastic Gradient Descent), 'MSE' (Mean Squared Error),
  GPU (Graphics Processing Unit)
* Added all contributors and copyright holders to Authors@R:
  - Georgi Gerganov (GGML library author)
  - Jeffrey Quesnelle and Bowen Peng (ops.cpp contributors)
  - Mozilla Foundation (llamafile/sgemm.cpp)
* Replaced `\dontrun{}` with `\donttest{}` in all examples
* Added `\value` documentation to all exported functions and constants

## Documentation

* Added vignette: Vulkan GPU Backend (`vignette("vulkan-backend")`)
* Added vignette: Multi-GPU Inference (`vignette("multi-gpu")`)
* Added vignette: Working with Quantized Models (`vignette("quantization")`)

## Internal

* `r_ggml_io.o` moved to GGML_OBJECTS for proper symbol export
* Static library excluded from source tarball via `.Rbuildignore`

# ggmlR 0.5.0

## Major Features

* Added full optimization/training API (39 new functions)
  - `ggml_opt_init()`, `ggml_opt_free()`, `ggml_opt_reset()` — optimizer lifecycle
  - `ggml_opt_fit()` — high-level training loop
  - `ggml_opt_epoch()` — single epoch with R callback support
  - `ggml_opt_eval()`, `ggml_opt_alloc()` — model evaluation
  - `ggml_opt_prepare_alloc()` — non-static graph support
  - `ggml_opt_grad_acc()` — gradient accumulator access

* Dataset management
  - `ggml_opt_dataset_init()`, `ggml_opt_dataset_free()`
  - `ggml_opt_dataset_data()`, `ggml_opt_dataset_labels()`
  - `ggml_opt_dataset_shuffle()`, `ggml_opt_dataset_get_batch()`

* Training results
  - `ggml_opt_result_init()`, `ggml_opt_result_free()`, `ggml_opt_result_reset()`
  - `ggml_opt_result_ndata()`, `ggml_opt_result_loss()`, `ggml_opt_result_accuracy()`
  - `ggml_opt_result_pred()` — get predictions as integer vector

* Loss functions: MSE, cross-entropy, mean, sum
* Optimizers: AdamW, SGD

## R Callback Support

* `ggml_opt_epoch()` now supports custom R callback functions
* Callbacks receive: train flag, batch index, max batches, start time, result pointer
* Built-in progress bar callback available via `callback_train = TRUE`

## Extended Backend API (~50 new functions)

* Device management
  - `ggml_backend_dev_count()`, `ggml_backend_dev_get()`, `ggml_backend_dev_by_name()`
  - `ggml_backend_dev_by_type()` — find devices by type (CPU, GPU, etc.)
  - `ggml_backend_dev_supports_op()`, `ggml_backend_dev_supports_buft()`
  - `ggml_backend_dev_memory()`, `ggml_backend_dev_description()`, `ggml_backend_dev_get_props()`
  - `ggml_backend_dev_init()` — initialize backend from device

* Device type constants: `ggml_backend_device_type_cpu/gpu/igpu/accel()`
* Buffer usage constants: `ggml_backend_buffer_usage_any/weights/compute()`

* Backend registry
  - `ggml_backend_reg_count()`, `ggml_backend_reg_get()`, `ggml_backend_reg_by_name()`
  - `ggml_backend_load()`, `ggml_backend_unload()`, `ggml_backend_load_all()`

* Events for synchronization
  - `ggml_backend_event_new()`, `ggml_backend_event_free()`
  - `ggml_backend_event_record()`, `ggml_backend_event_synchronize()`, `ggml_backend_event_wait()`

* Async operations
  - `ggml_backend_tensor_set_async()`, `ggml_backend_tensor_get_async()`
  - `ggml_backend_tensor_copy_async()`

* Graph planning
  - `ggml_backend_graph_plan_create()`, `ggml_backend_graph_plan_free()`
  - `ggml_backend_graph_plan_compute()`

* Buffer management
  - `ggml_backend_buffer_clear()`, `ggml_backend_buffer_set_usage()`
  - `ggml_backend_buffer_get_usage()`, `ggml_backend_buffer_reset()`, `ggml_backend_buffer_is_host()`

* Direct backend initialization
  - `ggml_backend_init_by_name()`, `ggml_backend_init_by_type()`, `ggml_backend_init_best()`
  - `ggml_backend_synchronize()`, `ggml_backend_get_device()`

## Testing

* Added 67 tests for optimization functions
* Added 60 tests for extended backend functions
* R CMD check: 0 errors, 0 warnings

# ggmlR 0.4.1

* Fixed spelling notes for CRAN submission
* Updated documentation

# ggmlR 0.4.0

* Added multi-GPU backend scheduler API (14 new functions)
* Added Vulkan GPU backend support (10 new functions)
* Fixed integer overflow for large tensors (>2 GB)
* Improved OpenMP handling for mixed C/C++ code

# ggmlR 0.2.0

* Initial CRAN submission
* R bindings for 'GGML' tensor library
* Core tensor operations: creation, arithmetic, reshaping
* Neural network operations: attention, convolutions, normalization
* Activation functions: GELU, SiLU, ReLU, and variants
* Quantization support (Q4_0, Q4_1, Q8_0)
* OpenMP parallelization for CPU backend
* Computation graph API for building and executing models
