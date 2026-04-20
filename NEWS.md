# ggmlR 0.7.5

* **Vulkan 1.4 Support**: integrated push constants raised to 256 bytes, targeting 5D tensor operations — enables larger parameter blocks in compute shaders without staging buffers.
* **Architecture Update**: refactored core file structure for improved project organization.

# ggmlR 0.7.4

* **ONNX Conv**: replaced `ggml_conv_2d` (IM2COL+GEMM) with `ggml_conv_2d_direct` (`GGML_OP_CONV_2D`) in `onnx_ggml.c` — SuperResolution GPU time 344 ms → 5 ms (~70×).
* **Vulkan softmax**: `wg512` pipeline threshold lowered from `>1024` to `>=512` — improves attention softmax at seq_len 512–1024.
* **New examples**: `benchmark_ops.R` (36-op CPU/GPU micro-benchmark), `profile_onnx_superres_gpu.R` (GPU profiler for SuperResolution).

# ggmlR 0.7.3

## Vulkan: subgroup-shuffle mmq for Q4_K / Q5_K / Q6_K (wavefront-64 devices)

* **`USE_SUBGROUP_NO_SHMEM` path added to `mul_mmq.comp`** — on wavefront-64 devices (RDNA4, subgroup_size=64) the `block_a` weight tile is loaded directly into registers via `subgroupShuffle` / `subgroupBroadcast`, eliminating the shared-memory round-trip in `block_a_to_shmem → block_a_to_registers`. Measured on RX 9070: Flux 768×768 sampling 22.38s → 20.80s (~7% end-to-end; sampling is not pure matmul so the gain on isolated Q4_K GEMM is higher).
* **New device capability field `subgroup_no_shmem`** — `ggml_vulkan_device_caps()` now returns this flag (logical), indicating whether the shuffle mmq path is active.
* **`GL_EXT_shader_subgroup_extended_types_float16`** added to `mul_mmq.comp` under `#ifdef USE_SUBGROUP_NO_SHMEM && FLOAT16` — required for `subgroupShuffle` on `float16_t` components of `f16vec2`.
* **`ggml_vulkan_device_caps()` extended** — `wavefronts_per_simd` and `arch` fields added; all 14 fields now documented.
* **New pipeline `pipeline_dequant_mul_mat_mat_q8_1_no_shmem`** — registered in device struct; selected at dispatch when `subgroup_size == 64` and src0 is Q4_K / Q5_K / Q6_K; falls back to standard mmq pipeline gracefully when not compiled.
* **`GGML_TYPE_Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` exported** — these constants were defined in `tensors.R` but missing from NAMESPACE; `roxygen2::roxygenise()` now includes them.
* **`inst/examples/vulkan_caps.R` extended** — new section shows `USE_SUBGROUP_NO_SHMEM: ACTIVE/INACTIVE` with explanation of conditions.
* **Tests** — `tests/testthat/test-vulkan.R` adds smoke tests for Q4_K / Q5_K / Q6_K quantized matmul via Vulkan (no NaN/Inf, correct shape); `test-vulkan-caps.R` asserts `integer_dot_product=TRUE` on RDNA4.

# ggmlR 0.7.2

## Vulkan: RDNA4 (RX 9000) cooperative matrix support

* **AMD RDNA4 (GFX12xx) detected correctly** — `get_device_architecture()` now identifies RDNA4 by `wavefrontsPerSimd == 16` (distinct from RDNA3's 8 and RDNA1's 20). Previously GFX1201 fell through to `AMD_RDNA3` due to identical subgroup size range (min=32, max=64).
* **`VK_AMD_shader_core_properties` queried at device init** — `wavefronts_per_simd` is now stored in `vk_device_struct` and read once during `ggml_vk_get_device()`, not just inside `get_device_architecture()`.
* **`SHADERGEN_DEFINES` propagated to C++ compiler** — `configure` now appends `SHADERGEN_DEFINES` (which includes `-DGGML_VULKAN_COOPMAT_GLSLC_SUPPORT`) to `VULKAN_CPPFLAGS`. Previously these defines were only passed to `vulkan-shaders-gen`, so all `#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)` blocks in `ggml-vulkan.cpp` were dead code at runtime.
* **`ggml_backend_vk_get_device_caps()` extended** — now returns `subgroup_min_size`, `subgroup_max_size`, `wavefronts_per_simd`, and `arch` (string) in addition to the original 5 fields. R function `ggml_vulkan_device_caps()` exposes all 9 fields.
* **Result on RX 9070 (RADV GFX1201):** `coopmat_support=YES`, `coopmat1_fa_support=YES` — KHR cooperative matrix GEMM and flash-attention paths now active.

## Vulkan: Q4_K flash attention (FA_SCALAR + FA_COOPMAT1)

* **Q4_K in flash attention** — `GGML_OP_FLASH_ATTN_EXT` now accepts `K`/`V` tensors in `Q4_K` format on Vulkan. Previously Q4_K fell back to CPU; now it runs fully on GPU via both the scalar and cooperative-matrix (KHR) paths.
* `dequantize4_q4k()` added to `flash_attn_base.glsl` — decodes 4 consecutive Q4_K elements from a `block_q4_K_packed16` block: reconstructs the 6-bit scale and min for the sub-block, reads two consecutive `uint16` from `qs[]`, and extracts four nibbles. Works for both K and V bindings.
* `flash_attn.comp` (FA_SCALAR) and `flash_attn_cm1.comp` (FA_COOPMAT1) now compiled with `DATA_A_Q4_K` / `BLOCK_SIZE=QUANT_K_Q4_K=256`. Four SPIR-V variants generated: f32acc and f16acc for each path.
* `vulkan-shaders-gen.cpp` — `q4_k` added to the FA scalar and coopmat1 generation conditions.
* `ggml-vulkan.cpp` — `CREATE_FA(GGML_TYPE_Q4_K, ...)` added for FA_SCALAR and FA_COOPMAT1; `GGML_TYPE_Q4_K` added to the supported-types switch in `ggml_backend_vk_device_supports_op`.
* Note: most efficient when head dimension (`HSK`) is a multiple of 256 (e.g. DeepSeek-V2/V3 MLA). For HSK=128 (Llama, Mistral) the shader is functionally correct but pads the inner loop to 256.

# ggmlR 0.7.1

## tidymodels / parsnip integration

* **`"ggml"` engine for `parsnip::mlp()`** — registers a `"ggml"` engine for both classification and regression modes. After `library(ggmlR)` (with `parsnip` installed), use:
  ```r
  mlp(hidden_units = 64, epochs = 100) |>
    set_engine("ggml", batch_size = 32, backend = "auto") |>
    set_mode("classification")
  ```
  Engine arguments: `batch_size`, `backend`, `verbose`, `validation_split`, `optimizer`, `callbacks`. All `mlp()` parameters (`hidden_units`, `epochs`, `dropout`, `activation`, `learn_rate`) are mapped through.
* **`backend = "gpu"` in parsnip** — `"gpu"` is now correctly translated to `"vulkan"` inside `ggmlr_parsnip_fit_classif()` and `ggmlr_parsnip_fit_regr()`. Previously the string was passed through and caused an unknown backend error.
* **`learn_rate` callback** — the `learn_rate` argument from `mlp()` is applied via an internal `on_epoch_begin` callback that sets the optimizer learning rate at the start of epoch 1. Works for both `"adam"` and `"sgd"` optimizers.
* **New `Suggests`:** `parsnip`, `tibble`, `rlang`, `dials`.
* **New example:** `inst/examples/tidymodels_integration.R` — CPU vs GPU comparison for iris classification and mtcars regression using the parsnip engine.

## mlr3 integration

* **`LearnerClassifGGML` / `LearnerRegrGGML` always defined** — R6 class definitions are now unconditional (no longer wrapped in `if (requireNamespace("mlr3"))`). This ensures the classes are always present in the ggmlR namespace, so `ggmlR:::.register_mlr3()` can be called reliably from vignettes and tests regardless of package load order.
* **Registration robustness** — `.onLoad()` no longer uses `mlr3misc::register_namespace_callback()` (which had a bug in v0.21.0 causing `R CMD check` warning `namespace can be unloaded cleanly`). Registration now uses `isNamespaceLoaded()` + `setHook()` directly, covering both "mlr3 already loaded" and "mlr3 loads after ggmlR" scenarios.
* **`mlr3misc` removed from `Suggests`** — no longer needed.
* **New example:** `inst/examples/mlr3_integration.R` — CPU vs GPU comparison for iris classification and mtcars regression, plus 3-fold CV.

## Bug fixes

* `marshal_model.*` / `unmarshal_model.*` S3 methods no longer appear in `NAMESPACE` as `S3method(mlr3::marshal_model, ...)` — this caused `Error: namespace 'marshal_model' not found` on package load. Methods are now registered exclusively via `registerS3method()` in `.onLoad()`.

## Tests

* `test-parsnip.R` — new tests: `learn_rate` applied without error; `backend="gpu"` accepted and converted to `"vulkan"` (skipped when Vulkan unavailable).
* `test-mlr3-learner.R` — explicit `ggmlR:::.register_mlr3()` call at top of file for reliable registration in `R CMD check` test process.

# ggmlR 0.7.0

## Vignettes: prebuilt HTML via Rcpp::asis

* Seven vignettes (Autograd Engine, Data Parallel Training, Embedding ggmlR, GPU Vulkan Backend, Keras-like API, ONNX Import, Quantization) are now shipped as prebuilt HTML using the `Rcpp::asis` vignette engine. No rendering on CRAN runners.
* Removed `rmarkdown` from Suggests (no longer needed).

## Test suite

* Suppressed spurious stdout/stderr output from tests: `ggml_graph_print()` output captured in `test-graph-utils.R`; C-level broadcast warnings captured in ONNX broadcast and resize-broadcast tests.

# ggmlR 0.6.9

## GGUF file reader

* **`gguf_load(path)`** — opens a GGUF file (v2/v3) and reads all metadata and tensor descriptors. Returns an S3 object of class `"gguf"`.
* **`gguf_metadata(x)`** — returns all key-value metadata pairs as a named list (architecture, tokenizer config, quantization info, etc.).
* **`gguf_tensor_names(x)`** — lists all tensor names in the file.
* **`gguf_tensor_info(x, name)`** — returns shape, type, and size in bytes for a single tensor.
* **`gguf_tensor_data(x, name)`** — dequantizes (if needed) and returns tensor weights as an R numeric array with correct dimensions.
* **`gguf_free(x)`** — explicitly frees GGUF context (also called by GC).
* Supports all ggml quantization types (F32, F16, Q4_0, Q8_0, K-quants, etc.) with automatic dequantization to F32.
* `print.gguf()` method shows file version, tensor count, and metadata count.

## Vulkan backend: revert to Vulkan 1.2 + Push Descriptors

* **Vulkan API version capped at 1.2** (was 1.3). Requesting a Vulkan 1.3 instance implicitly enables Synchronization2 (core in 1.3), which causes significant performance degradation on RADV (Mesa) drivers — particularly on newer AMD hardware (RX 9070 and similar). Capping at 1.2 avoids the implicit promotion while retaining all functionality.
* **Push Descriptors** (`VK_KHR_push_descriptor`): unchanged — when the extension is available and `maxPushDescriptors >= 12`, descriptor sets are pushed directly into the command buffer via `pushDescriptorSetKHR()`, eliminating descriptor pool overhead. Falls back to the traditional descriptor pool path on hardware without the extension.

## Keras-compatible API

* **`fit()`** now accepts a `callbacks` parameter for sequential models (passed through to `ggml_fit_sequential()`).

## Test suite

* New test files: `test-gguf.R`, `test-graph-utils.R`, `test-inplace-ops.R`, `test-keras-api.R`, `test-misc-ops.R`, `test-model-ops.R`, `test-print-methods.R`, `test-tensor-utils.R`, `test-threading.R`, `test-autograd-missing.R`, `test-nn-functional-missing.R`, `test-quants-missing.R`.

# ggmlR 0.6.8

## Bug fixes

* Fixed ABI mismatch between `src/` and `inst/include/` headers: `configure` and `configure.win` now automatically sync all public headers from `src/` to `inst/include/` at install time. Previously, changes to `GGML_MAX_DIMS` (4→5) and other structs in `src/ggml.h` were not propagated to the exported headers, causing segfaults in downstream packages (e.g. sd2R).
* Added `tests/testthat/test-headers-sync.R` to verify that `inst/include/` headers remain in sync with `src/` headers and that `GGML_MAX_DIMS` is consistent.

# ggmlR 0.6.7

## ggml engine: native 5D tensor support

* **`ggml_view_5d()`** — new API function for creating 5D views with explicit strides, extending the existing 1D–4D view family. Uses the existing `ggml_view_impl()` internally.
* **`ggml_repeat_5d()`** — new API function for tiling tensors up to 5D. CPU kernels (`ggml_compute_forward_repeat_f32`, `ggml_compute_forward_repeat_f16`) updated with a 5th loop dimension. Vulkan dispatch collapses dim3×dim4 into push constants transparently (no shader changes needed — push constants remain at 128 bytes).
* ONNX tensor pipeline upgraded from hardcoded 4D to 5D throughout `onnx_ggml.c` (~20 sites):
  - Initializers, inputs, Constant, ConstantOfShape: `ne[GGML_MAX_DIMS]` arrays, switch with `case 5: new_tensor_5d`.
  - Broadcast (`onnx_broadcast_align`): all reshape/new_tensor calls use dimension-aware helpers.
  - Softmax: reshape-back via generic `onnx_reshape_nd()`.
  - Reshape op: collapse threshold raised from >4D to >5D.
  - Slice: 5D view/offset support, generic stride-based cval propagation and deferred fill.
  - Split: 5D view support.
  - Expand: 5D broadcast with rank promotion.
  - Tile: uses `ggml_repeat_5d()`.
  - Gather axis=0: generic reshape-back for any rank.
  - `tmap_put_nd()` and `slice_fill` arrays updated to `GGML_MAX_DIMS`.
* New internal helpers: `onnx_reshape_nd()`, `onnx_new_tensor_nd()`, `ne_product()` — eliminate switch/case duplication.
* Resize/Interpolate remains 4D (spatial op, 5D not relevant). Transpose/Permute remains 4D (`ggml_permute` API limitation).

## ONNX: ConstantOfShape INT64/INT32/DOUBLE value fix

* **roberta-9 model now loads and runs** (was producing NaN in softmax). Root cause: `ConstantOfShape` read the `value` TensorProto attribute as float regardless of `data_type`. When `data_type=7` (INT64), the 8-byte int64 was reinterpreted as a 4-byte float, producing garbage values (~1.4e-45 instead of 1). This broke attention mask generation (fill=0 instead of 1) and position ID generation (NonZero on zeros = empty).
* Fix: `ConstantOfShape` now checks `data_type` and correctly handles INT64, INT32, DOUBLE, and FLOAT value attributes.

## ONNX: Gather axis=0 on rank>2 tensors

* **Gather on 4D tensors** no longer asserts. Previous code always used `ggml_get_rows` which only supports 2D data. For axis=0 on rank>2 (e.g. CaiT QKV split on `[48,576,6,3]`), the tensor is now reshaped to 2D, gathered, and reshaped back.

## ONNX: ScatterElements op (GPU + CPU)

* New `GGML_OP_SCATTER_ELEMENTS` added to the ggml engine with both CPU kernel and Vulkan compute shader.
* **Vulkan shader** (`scatter_elements.comp`): two variants compiled at install time — `scatter_elements_none` (overwrite) and `scatter_elements_add` (atomicAdd via `GL_EXT_shader_atomic_float`). Data is copied to output via `vkCmdCopyBuffer` with a pipeline barrier before the scatter dispatch.
* **CPU kernel**: single-threaded scatter with memcpy (overwrite) or element-wise addition (reduce=add).
* ONNX mapper: `ScatterElements` op with `axis=0` and `reduction="none"/"add"` attributes. Indices cast to I32, updates/data cast to F32 automatically.
* This unblocks sageconv (GNN message passing with scatter-add).

## Model count

* **12/15** ONNX Model Zoo models now pass (was 11/15). New: roberta-9.
* Remaining failures: sageconv (ScatterElements shape mismatch needs further work), cait_xs24_384 (reshape size mismatch), MaskRCNN-12-int8 (spatial broadcast mismatch), xcit_tiny (broadcast dim mismatch).

# ggmlR 0.6.6

## ONNX: BoTNet RelPosBias2D fused custom op

* **botnet26t_256 model now loads and runs** (was failing on 5D Transpose in pos_embed subgraph). Three pos_embed subgraphs (~60-80 ONNX nodes each) are detected via pre-pass scanner and replaced with a single fused `ggml_map_custom3` op. The CPU kernel computes 2D relative position bias directly: `bias[b,hq,wq,hk,wk] = dot(x, W_h) + dot(x_transposed, W_w)`.
* Pre-pass scanner: `detect_pos_embed_blocks()` identifies contiguous node ranges with `/pos_embed/` in output names, extracts W_h/W_w initializer shapes to determine H, W, C, validates F32 data type.
* Model count: **13/15** ONNX Model Zoo models now pass (was 12/15).

## ONNX: pinned staging buffer for GPU input transfer

* When Vulkan GPU is available, a host-visible pinned memory buffer is allocated at model load time for ONNX input data. In `onnx_ggml_run()`, input data is copied into pinned memory before `ggml_backend_tensor_set()` — the Vulkan driver detects the pinned source pointer and performs direct DMA transfer to VRAM, bypassing the internal staging copy.
* Fallback: if `ggml_backend_vk_host_buffer_type()` returns NULL or buffer is too small, the standard staging path is used transparently.

## Bug fixes

* `onnx_device_info()`: added NULL guards for `ctx->graph` and `n_nodes == 0` edge cases that caused segfault when called on models before first inference run.

# ggmlR 0.6.5

## Bug fixes

* **`ggml_predict()` with stochastic dropout**: `nn_build_graph()` now receives `training = FALSE` during inference, so stochastic Bernoulli dropout is disabled at predict time. Previously, `stochastic = TRUE` dropout layers applied random masks during inference, degrading accuracy.
* **`ggml_fit()` return value**: the return value of `ggml_fit()` must be assigned back to `model` to obtain trained weights (`model <- ggml_fit(...)`). This is now clarified in all examples and documentation. Using `history <- ggml_fit(...)` without reassigning `model` leaves the model with untrained weights.
* **`ggml_evaluate()` return value**: now includes `n_samples` in addition to `loss` and `accuracy`. Metrics are computed on all samples without truncation (via `ggml_predict()` internally).

## Examples

* `inst/examples/titanic_classification.R` — new end-to-end binary classification example on the Titanic dataset. Demonstrates feature engineering (Title, FamilySize, IsAlone), stratified train/val split, one-hot encoding, dropout regularization, and manual validation metrics (accuracy, precision, recall, F1, confusion matrix). Achieves ~82% val accuracy.

## ONNX inference: dedicated weight buffer architecture

* **Zero-overhead repeated inference**: weights are loaded to GPU (or CPU) once via a dedicated `weight_buf` and never re-transferred between runs. Previous architecture reloaded all weights before every `onnx_run()` call — eliminated entirely.
* Separate `ctx_weight` / `ctx` contexts: weight tensors live in a permanent GPU buffer that the scheduler never aliases; compute tensors are managed by `ggml_backend_sched` independently.
* GPU speedups from eliminated weight reload (vs 0.6.3):
  - SuperResolution: 354 ms → 7 ms (48x)
  - BERT: 100 ms → 15 ms (7x)
  - Inception V3 Op18: 106 ms → 14 ms (7x)
  - Inception V3: 24 ms → 14 ms (1.7x)
  - EmotionFerPlus: 4.7 ms → 1.7 ms (2.8x)
  - BAT-ResNeXt: 14 ms → 9 ms (1.6x)
* `onnx_device_info()` — scheduler diagnostic: number of splits, GPU/CPU op counts, CPU-only op list.
* GPT-NeoX model now loads and runs successfully (was failing on shape propagation).
* Benchmark script (`inst/examples/benchmark_onnx.R`): proper VRAM cleanup between models via `rm()` + `gc()`.

# ggmlR 0.6.3

## ONNX model import

* `onnx_load(path, device, input_shapes)` — load an ONNX model file, build a ggml computation graph, and allocate tensors on Vulkan GPU or CPU. Weights are loaded via memory-mapped file (zero-copy where possible).
* `onnx_run(model, inputs)` — run inference on a loaded ONNX model with named input data.
* `onnx_inputs(model)` — list expected input tensor names and shapes.
* `onnx_summary(model)` — return model metadata (IR version, opset, producer, ops used).
* `print.onnx_model()` — formatted summary of a loaded ONNX model.
* Built-in zero-dependency protobuf parser: no external libraries or Python required.
* `input_shapes` parameter for models with dynamic dimensions: specify fixed shapes at load time (e.g. `input_shapes = list(image = c(1L, 3L, 224L, 224L))`).
* 40+ supported ONNX ops: Add, Sub, Mul, Div, MatMul, Gemm, Conv (1D/2D), ConvTranspose (1D/2D), Relu, Sigmoid, Tanh, GELU, SiLU, LeakyRelu, Elu, Softmax, MaxPool, AveragePool, GlobalAveragePool, BatchNormalization, LayerNormalization, GroupNormalization, RMSNormalization, Reshape, Transpose, Concat, Flatten, Squeeze, Unsqueeze, Gather, Pad, Clip, Cast, Constant, ConstantOfShape, Shape, Expand, Slice, Split, Where, Erf, Pow, Sqrt, Exp, Log, Abs, Neg, Floor, Ceil, ReduceMean, ReduceSum, Resize/Upsample, Identity, Dropout.
* `auto_pad` attribute (SAME_UPPER, SAME_LOWER) supported for Conv and pooling ops.
* Numpy-style broadcast for binary ops (Add/Sub/Mul/Div): handles mismatched ranks and dimensions, with left-align, right-align, and greedy dim-matching strategies.
* Scalar Constant tensors (0-dimensional TensorProto) correctly handled.

## Tested real-world ONNX models (13/15 from ONNX Model Zoo)

* mnist-8 — OK (12 nodes)
* squeezenet1.0-8 — OK (66 nodes: Conv, Relu, MaxPool, Concat, Dropout, GlobalAveragePool, Softmax)
* adv_inception_v3 Opset 17/18 — OK (215 nodes)
* super-resolution-10 — OK with `input_shapes` (Conv, Reshape, Transpose)
* bert Opset 17 — OK (533 nodes: MatMul, Add, LayerNorm, GELU/Erf, Softmax, Shape, Gather, Cast, Where, ConstantOfShape)
* emotion-ferplus-8 — OK (52 nodes: Conv, Relu, MaxPool, Reshape, Gemm, Constant)
* sageconv Opset 16 — OK (24 nodes: MatMul, Add, Mul, Sigmoid, ReduceSum)
* roberta-sequence-classification-9 — OK with `input_shapes` (1180 nodes)
* bat_resnext26ts Opset 18 — OK (570 nodes: Conv, BatchNorm, SiLU, Concat, Expand, Split)
* gptneox Opset 18 — OK with `input_shapes` (482 nodes: MatMul, LayerNorm, GELU, Softmax)
* xcit_tiny — OK (436 nodes: MatMul, LayerNorm, Softmax, Concat, Transpose)
* MaskRCNN-12-int8 — OK (937 nodes: QLinearConv, DequantizeLinear, Resize, Concat, Reshape)
* botnet26t_256 (Opset 16) — OK (RelPosBias2D fused custom op, 3 pos_embed blocks replaced)
* Remaining failures: cait_xs24_384 (batched matmul 3D+).

# ggmlR 0.6.2
* Fixed Windows cleanup script that removed `inst/lib/libggml.a`, breaking static linking from dependent packages (e.g. llamaR).


# ggmlR 0.6.1

* `dp_train(make_model, data, loss_fn, forward_fn, target_fn, n_gpu, n_iter, lr, max_norm, verbose)` — data-parallel training across multiple replicas. Weights are broadcast from replica 0 before the first step; gradients are averaged across replicas each iteration; weights are re-broadcast after each optimizer update. Returns `list(params, loss_history, model)`.
* `ag_mul` and `ag_sub` now support CPU broadcast: `[d×s] * [1×s]` and `[d×s] * [d×1]` shapes work correctly with proper gradient reduction.
* `ag_softmax_cross_entropy_loss` accepts integer target vectors (0-based class indices) and converts them to one-hot automatically.
* `ggml_sum_rows` f16 on Vulkan: F16→F16 dispatch now supported natively (no CPU fallback).

# ggmlR 0.6.0

## Dynamic autograd engine (PyTorch-style training)

* `ag_tensor()` / `ag_param()` — environment-backed tensors with reference semantics; in-place optimizer updates visible to all references.
* `with_grad_tape({ ... })` — enables the global gradient tape for the enclosed forward pass.
* `backward(loss)` — reverse-mode automatic differentiation; returns a gradient environment keyed by tensor id.
* Differentiable ops: `ag_matmul`, `ag_add` (with bias broadcast), `ag_sub`, `ag_mul`, `ag_scale`.
* Activations: `ag_relu`, `ag_sigmoid`, `ag_tanh`, `ag_softmax`.
* Reduction / math ops: `ag_sum`, `ag_mean`, `ag_log`, `ag_exp`, `ag_pow`, `ag_clamp`.
* Shape ops: `ag_reshape`, `ag_transpose`.
* Loss functions: `ag_mse_loss`, `ag_cross_entropy_loss`, `ag_softmax_cross_entropy_loss` (numerically-stable fused).
* `optimizer_sgd()` — SGD with optional momentum.
* `optimizer_adam()` — Adam with bias-corrected moment estimates.
* `ag_linear()` — Glorot-initialised dense layer (closure-based, returns `$forward`, `$params()`).
* `ag_gradcheck()` — central finite-difference gradient checker (like `torch.autograd.gradcheck`).

## Layer objects (environment-based, train/eval modes)

* `ag_sequential(...)` — ordered layer container; collects all parameters for the optimizer.
* `ag_dropout(rate)` — inverted dropout; identity in eval mode.
* `ag_batch_norm(num_features)` — batch normalisation with running statistics and learnable γ/β.
* `ag_embedding(vocab_size, dim)` — token lookup with scatter-add backward.
* `ag_train(model)` / `ag_eval(model)` — switch all sub-layers between train and eval mode.

## Training utilities

* `ag_dataloader(x, y, batch_size, shuffle, col_major)` — mini-batch iterator with shuffle and `$epoch()` helper.
* `lr_scheduler_step(optimizer, step_size, gamma)` — step-decay learning rate.
* `lr_scheduler_cosine(optimizer, T_max, lr_min, restart)` — cosine-annealing (with optional SGDR warm restarts).
* `clip_grad_norm(params, grads, max_norm)` — clips all gradients by global L2 norm in-place.

# ggmlR 0.5.9

* `ggml_layer_lstm()` — LSTM recurrent layer (unrolled BPTT).
* `ggml_layer_gru()` — GRU recurrent layer (unrolled BPTT).
* `ggml_layer_global_max_pooling_2d()` — reduces `[H,W,C]` to `[C]` via max pooling.
* `ggml_layer_global_average_pooling_2d()` — reduces `[H,W,C]` to `[C]` via average pooling.
* `ggml_save_model()` — saves full model (architecture + weights) to RDS file.
* `ggml_load_model()` — restores a model saved with `ggml_save_model()`.
* `ggml_dense()`, `ggml_conv_2d()`, `ggml_conv_1d()`, `ggml_batch_norm()`, `ggml_embedding()`, `ggml_lstm()`, `ggml_gru()` — layer object constructors returning a reusable `ggml_layer` object.
* `ggml_apply(tensor, layer)` — applies a `ggml_layer` object to a tensor node; shared weights by object identity.

# ggmlR 0.5.7

* `ggml_layer_dropout()` — dropout with deterministic or stochastic (per-epoch Bernoulli mask) mode.
* `ggml_layer_embedding()` — token embedding lookup for integer inputs.
* `ggml_input()` gains `dtype` argument (`"float32"` or `"int32"`).
* Multi-output support in `ggml_model()` and `ggml_predict()`.

# ggmlR 0.5.6

* `ggml_input()` — declare a symbolic input tensor node (Functional API).
* `ggml_model()` — assemble a `ggml_functional_model` from input/output nodes.
* `ggml_layer_add()` — element-wise addition of tensor nodes (residual connections).
* `ggml_layer_concatenate()` — concatenate tensor nodes along an axis.
* All `ggml_layer_*()` functions now accept a `ggml_tensor_node` as first argument (Functional API mode).
* `ggml_compile()`, `ggml_fit()`, `ggml_evaluate()`, `ggml_predict()` are now S3 generics with methods for `ggml_functional_model`.

# ggmlR 0.5.5

* `ggml_fit_opt()` — low-level optimizer loop with callbacks and learning-rate control.
* `ggml_callback_early_stopping()` — stops training when a metric stagnates.
* `ggml_schedule_step_decay()` — step learning-rate decay.
* `ggml_schedule_cosine_decay()` — cosine learning-rate annealing.
* `ggml_schedule_reduce_on_plateau()` — reduces LR when metric stops improving.
* `ggml_opt_init_for_fit()`, `ggml_opt_set_lr()`, `ggml_opt_get_lr()` — learning-rate control without recreating the optimizer context.

# ggmlR 0.5.4

* Vulkan GPU backend support on Windows via `configure.win`.
* Vulkan auto-detected at build time on Linux and Windows.

# ggmlR 0.5.3

* `ggml_layer_conv_1d()` — 1D convolution layer.
* `ggml_layer_batch_norm()` — batch normalization layer.
* `ggml_predict_classes()` — argmax wrapper returning 1-based class indices.
* `summary.ggml_sequential_model()` — detailed model summary with parameter counts.
* `ggml_fit()` now returns `model$history` (class `ggml_history`) with `print` and `plot` methods.
* Sequential API: `ggml_model_sequential()`, `ggml_layer_dense()`, `ggml_layer_conv_2d()`, `ggml_layer_max_pooling_2d()`, `ggml_layer_flatten()`, `ggml_compile()`, `ggml_fit()`, `ggml_evaluate()`, `ggml_predict()`, `ggml_save_weights()`, `ggml_load_weights()`.
* Vulkan GPU backend covering 90%+ of ML operations.

# ggmlR 0.5.2

* `ggml_timestep_embedding()` — sinusoidal timestep embeddings.
* N-D tensor access: `ggml_set_f32_nd()`, `ggml_get_f32_nd()`, `ggml_set_i32_nd()`, `ggml_get_i32_nd()`.
* Tensor utilities: `ggml_tensor_nb()`, `ggml_tensor_num()`, `ggml_tensor_copy()`, `ggml_tensor_set_f32_scalar()`, `ggml_get_first_tensor()`, `ggml_get_next_tensor()`.

# ggmlR 0.5.1

* Static library `libggml.a` exported for linking by dependent packages.
* `gguf.cpp` added for GGUF file format support.
* Headers exported via `inst/include/` for `LinkingTo`.

# ggmlR 0.5.0

* Full optimization/training API: `ggml_opt_init()`, `ggml_opt_free()`, `ggml_opt_fit()`, `ggml_opt_epoch()`, `ggml_opt_eval()`.
* Dataset management: `ggml_opt_dataset_init()`, `ggml_opt_dataset_data()`, `ggml_opt_dataset_labels()`, `ggml_opt_dataset_shuffle()`.
* Training results: `ggml_opt_result_init()`, `ggml_opt_result_loss()`, `ggml_opt_result_accuracy()`, `ggml_opt_result_pred()`.
* Extended backend API: device management, registry, async operations, graph planning, buffer management (~50 new functions).
* Loss functions: MSE, cross-entropy. Optimizers: AdamW, SGD.

# ggmlR 0.4.0

* Multi-GPU backend scheduler API.
* Vulkan GPU backend support.

# ggmlR 0.2.0

* Initial release: R bindings for GGML tensor library.
* Core tensor operations, neural network ops, activation functions, quantization (Q4_0, Q4_1, Q8_0), OpenMP parallelization, computation graph API.
