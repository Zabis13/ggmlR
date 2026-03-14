## TODO



### llamaR Integration
- [x] Export static library `libggml.a`
- [x] Export headers via `inst/include/`
- [x] Add `gguf.cpp` for GGUF support
- [ ] Sync ggml version with llama.cpp (add `ggml_build_forward_select`)



### Custom Operations
- [ ] `ggml_custom()` / `ggml_custom_inplace()` — требуют C расширение



### Концепция API

Один ядро-тип модели (`ggml_model`) с backend-полем (cpu, vulkan, sched).
Над ним: Functional API для создания, ONNX/GGUF загрузчики для импорта,
keras-совместимый compile/fit/predict, deployment-helpers.

---

#### 1. Functional API (ядро ggmlR)

Декларативное описание графа: `ggml_input()` → `ggml_layer_*()` → `ggml_model(inputs, outputs)`.

```r
x <- ggml_input(shape = c(1, 3, 224, 224), name = "image")
y <- x |>
  ggml_layer_conv_2d(filters = 64, kernel_size = 3, activation = "silu") |>
  ggml_layer_max_pool(pool_size = 2) |>
  ggml_layer_dense(units = 1000, activation = "softmax")
model <- ggml_model(inputs = x, outputs = y, name = "resnet_tiny")
```

- [x] Shared layers (`name=`) для Siamese/multi-branch
- [ ] Multi-input: `ggml_model(inputs = list(x1, x2), outputs = y)`
- [x] Сохранение/загрузка архитектуры + весов
- [ ] Загрузка pre-trained весов из .gguf
- [ ] `ggml_layer_concatenate()` с backward pass (требует патча ggml C)
- [ ] Dropout маска per-batch (требует C-расширения)
- [ ] Custom layer API за 5 строк R-кода:
  ```r
  layer_my_act <- ggml_layer_custom(
    name = "my_act",
    forward = function(x) x * ggml_tanh(ggml_log(1 + ggml_exp(x)))
  )
  ```

---

#### 2. Training API (keras-совместимый)

`compile()/fit()/predict()` идентичны keras3, но поверх ggml-backend.

- [ ] `compile(model, optimizer, loss, metrics = NULL)`
- [ ] `fit(model, x, y, batch_size, epochs, validation_data, callbacks)`
- [ ] `predict(model, x, batch_size = NULL)` — единый вход для ggmlR и ONNX моделей
- [ ] Проверить совместимость с keras3 (compile/fit должны быть идентичны)
- [ ] `ggml_fit()` с generator/iterator вместо полной матрицы в памяти
- [ ] Data augmentation (flip, rotate, crop)

##### Callbacks & Monitoring
- [ ] Cost tracker — реальные затраты GPU/CPU в реальном времени
- [ ] Auto-quantize — автоматическое уменьшение модели при ухудшении метрики

---

#### 3. ONNX и внешние модели

ONNX-модели — ещё один источник графов для общего API (`ggml_model`).

```r
onnx <- onnx_load("unet.onnx", input_shapes = list(sample = c(1, 4, 64, 64)))
pred <- predict(onnx, x)
```

- [x] `onnx_load(path, input_shapes = NULL, backend = c("auto", "cpu", "vulkan"))`
- [x] Dynamic dims без `input_shapes` → понятная ошибка
- [ ] Результат совместим с `predict()` / `ggml_model` API

---

#### 4. Deployment API

Все принимают один и тот же объект модели (Functional API, ONNX, GGUF).

- [ ] `ggml_export_gguf(model, path, quantization = "q4_0")`
- [ ] `ggml_plumber_api(model, path = "api.R")` — генерация Plumber-сервиса с `/predict`
- [ ] `ggml_vetiver_model(model, model_name, ...)` — обёртка для vetiver, единый S3-метод
- [ ] Примеры квантизированных моделей (Q4_0, Q8_0)

---

### Оптимизация
- [ ] Профилирование scheduler overhead
- [ ] Минимизация копий между GPU
- [ ] **Vulkan profiling API на R уровне** — экспортировать `vk_perf_logger` из `ggml-vulkan.cpp` в R через `.Call()`, чтобы видеть breakdown по операциям (мс на каждый op/fusion). Нужно для диагностики bottleneck'ов в sd2R sampling loop (552s на Flux). В ggml уже есть `vk_perf_logger_enabled` и timestamp queries — нужен R-интерфейс для включения/чтения результатов.

---

### Autograd / GPU (текущая работа)



#### Новые фичи (0.6.1)
- [x] `dp_train()` — data-parallel training: N реплик, синхронизация весов, усреднение градиентов

#### Следующие фичи
- [ ] Gradient checkpointing — экономия памяти при глубоких сетях
- [ ] Flash Attention — эффективный attention через ggml_flash_attn_ext
- [ ] Optimizer states в f32 при f16 весах (true mixed precision training)

---

### ONNX ops — MVP subset для трансформеров и SD

#### 1. Базовые тензорные операции
- [x] Add, Sub, Mul, Div
- [x] MatMul (пофикшен ggml_cont после transpose)
- [x] Reshape, View (через Reshape+Identity)
- [x] Transpose (пофикшен ggml_cont после permute)
- [x] Concat
- [x] Gather (через ggml_get_rows)
- [x] Identity
- [x] Slice (через ggml_view, step=1)
- [x] Unsqueeze, Squeeze (реальная логика dims, opset < 13 атрибуты + opset 13+ inputs)
- [x] Split (равные части + explicit sizes, multi-output)
- [x] Expand (через ggml_repeat)

#### 2. Нормализация и активации
- [x] LayerNormalization (1D работает, 2D+ нужен broadcast scale/bias)
- [x] RMSNormalization
- [x] GroupNormalization
- [x] BatchNormalization
- [x] GELU, Relu, SiLU/Swish, Sigmoid, Tanh, Softmax
- [x] LeakyRelu, Elu (CPU + Vulkan OK)

#### 3. Attention / линейные блоки
- [x] MatMul, Softmax, Reshape, Transpose, Mul — всё реализовано
- [x] Gemm (transA/transB/alpha/beta + bias)

> ONNX-экспортеры обычно не делают FlashAttention op, а оставляют классический QK^T → Softmax → V — маппится на имеющиеся ggml-ops.

#### 4. Convolution / UNet (для SD)
- [x] Conv (1D/2D, с padding/stride/dilation + bias)
- [x] BatchNormalization
- [x] GroupNormalization
- [x] ConvTranspose (1D: stride/pad/dilation, 2D: stride + pad cropping)

#### 5. Pooling / Upsampling
- [x] MaxPool, AveragePool, GlobalAveragePool
- [x] Upsample / Resize (nearest/bilinear через ggml_interpolate, scales + sizes)

#### 6. Прочие реализованные ops
- [x] Sqrt, Exp, Log, Abs, Neg, Floor, Ceil, Clip
- [x] ReduceMean, ReduceSum
- [x] Flatten, Dropout (pass-through)
- [x] Constant (TensorProto value + value_float/value_int скаляры)
- [x] Shape (deferred int32 fill после sched alloc)
- [x] Cast (pass-through, keep f32)
- [x] Erf (tanh approximation для GELU exact)
- [x] Where (condition * X + (1-cond) * Y)
- [x] ConstantOfShape (deferred fill)
- [x] Pow (exp(b * log(a)))
- [x] Pad (ggml_pad)

#### 7. Random / Scheduler
> Для SD денойзинг лупа (scheduler, генерация латентов) обычно вне ONNX, в хост-коде. ONNX-UNet получает готовые латенты и timestep. Оставить в R-логике (как в sd2R), ONNX-граф — чистый UNet/VAE/text encoder.

#### Тесты
- [x] 50 тестов в `tests/testthat/test-onnx.R` (R protobuf сериализатор, без Python)
- [x] CPU: все ops проходят
- [x] Vulkan: 24/24 ops проходят

#### Реальные модели (ONNX Model Zoo) — 9/15 OK
- [x] mnist-8 — OK (12 nodes)
- [x] squeezenet1.0-8 — OK (66 nodes: Conv, Relu, MaxPool, Concat, Dropout, GlobalAveragePool, Softmax)
- [x] adv_inception_v3 (Opset 17, 18) — OK (215 nodes)
- [x] super-resolution-10 — OK с input_shapes (Conv, Reshape+Constant, Transpose)
- [x] bert (Opset 17) — OK (533 nodes: MatMul, LayerNorm, GELU/Erf, Softmax, Shape, Gather, Cast, Where, ConstantOfShape)
- [x] emotion-ferplus-8 — OK (52 nodes: Conv, Relu, MaxPool, Gemm, Constant)
- [x] sageconv (Opset 16) — OK (24 nodes: MatMul, Add, Mul, Sigmoid, ReduceSum)
- [x] roberta-9 — OK с input_shapes (1180 nodes)
- [ ] bat_resnext26ts — MatMul 3D broadcast (ne[2] mismatch)
- [ ] botnet26t_256 — MatMul dims (ggml_can_mul_mat)
- [ ] cait_xs24_384 — Concat dim mismatch
- [ ] gptneox — shape propagation через Reshape
- [ ] MaskRCNN-12-int8 — quantized ops (QuantizeLinear, QLinearConv и т.д.)
- [ ] xcit_tiny — Concat dim mismatch

#### Баги / ограничения
- [x] **ELU на Vulkan** — добавлен шейдер `elu.comp` + регистрация в ggml-vulkan.cpp (6 точек). Тесты CPU и Vulkan проходят.
- [x] **ONNX runtime с scheduler** — переделан на `ggml_backend_sched` (Vulkan + CPU fallback). Unsupported Vulkan ops автоматически выполняются на CPU.
- [x] **LayerNorm broadcast** — 2D+ input с 1D scale/bias: transpose → norm → scale/bias → transpose back
- [x] **Softmax axis** — фикс для opset < 13 (default axis=1), reshape для non-ne[0] axis
- [x] **Reshape + Constant** — Reshape теперь ищет shape в Constant nodes (не только в initializers)
- [x] **Scalar Constant** — TensorProto с n_dims=0 (скаляры) теперь корректно создают 1-элементный тензор
- [x] **Broadcast** — numpy-style broadcast для Add/Sub/Mul/Div: left-align, right-align, greedy dim-matching

#### MVP + трансформеры — готов ✓
Все базовые + трансформерные ops реализованы. 9/15 моделей из ONNX Zoo работают.

#### Следующий этап — расширенная совместимость
- [ ] MatMul broadcast для 3D+ тензоров (batched matmul)
- [ ] Concat dim mismatch — проверка axis mapping для 3D+
- [ ] Quantized ops (QuantizeLinear, DequantizeLinear, QLinearConv)
- [ ] NonZero, Equal, Less, Greater — логические ops
