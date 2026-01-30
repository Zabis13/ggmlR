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
