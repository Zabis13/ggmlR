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

## Tested real-world ONNX models (9/15 from ONNX Model Zoo)

* mnist-8 — OK (12 nodes)
* squeezenet1.0-8 — OK (66 nodes: Conv, Relu, MaxPool, Concat, Dropout, GlobalAveragePool, Softmax)
* adv_inception_v3 Opset 17/18 — OK (215 nodes)
* super-resolution-10 — OK with `input_shapes` (Conv, Reshape, Transpose)
* bert Opset 17 — OK (533 nodes: MatMul, Add, LayerNorm, GELU/Erf, Softmax, Shape, Gather, Cast, Where, ConstantOfShape)
* emotion-ferplus-8 — OK (52 nodes: Conv, Relu, MaxPool, Reshape, Gemm, Constant)
* sageconv Opset 16 — OK (24 nodes: MatMul, Add, Mul, Sigmoid, ReduceSum)
* roberta-sequence-classification-9 — OK with `input_shapes` (1180 nodes)
* Remaining failures: bat_resnext26ts (MatMul 3D broadcast), botnet26t_256 (MatMul dims), cait_xs24 (Concat mismatch), gptneox (shape propagation), MaskRCNN (quantized ops), xcit_tiny (Concat mismatch).

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
