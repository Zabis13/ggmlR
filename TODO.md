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
- [x] Dropout `stochastic=TRUE` отключается при predict (`training=FALSE` в `nn_build_graph`)
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
- [x] Dedicated weight buffer (`ctx_weight` + `weight_buf`) — веса на GPU один раз, sched не трогает
- [x] Убран `reload_static_data()` — zero-overhead repeated inference
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
- [x] Conv grouped (group>1: split+conv+concat, depthwise: ggml_conv_2d_dw)
- [x] BatchNormalization
- [x] GroupNormalization
- [x] ConvTranspose (1D: stride/pad/dilation, 2D: stride + pad cropping)

#### 5. Pooling / Upsampling
- [x] MaxPool, AveragePool, GlobalAveragePool (+ ceil_mode, asymmetric padding)
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

#### Реальные модели (ONNX Model Zoo) — 13/15 OK
- [x] mnist-8 — OK (12 nodes)
- [x] squeezenet1.0-8 — OK (66 nodes: Conv, Relu, MaxPool, Concat, Dropout, GlobalAveragePool, Softmax)
- [x] adv_inception_v3 (Opset 17, 18) — OK (215 nodes)
- [x] super-resolution-10 — OK с input_shapes (Conv, Reshape+Constant, Transpose)
- [x] bert (Opset 17) — OK (533 nodes: MatMul, LayerNorm, GELU/Erf, Softmax, Shape, Gather, Cast, Where, ConstantOfShape)
- [x] emotion-ferplus-8 — OK (52 nodes: Conv, Relu, MaxPool, Gemm, Constant)
- [x] bat_resnext26ts — OK (grouped conv, AdaptiveMaxPool baked as fixed kernel, input 256x256)
- [x] gptneox (Opset 18) — OK с input_shapes (482 nodes: MatMul, LayerNorm, GELU, Softmax)
- [x] botnet26t_256 — RelPosBias2D fused custom op (pre-pass scanner + ggml_map_custom3)
- [x] roberta-9 — OK (ConstantOfShape INT64 fix: attention mask + position IDs теперь корректны)
- [ ] sageconv (Opset 16) — ScatterElements shape mismatch (op реализован, маппинг indices нуждается в доработке)
- [ ] xcit_tiny — broadcast dim mismatch a=28, b=32
- [ ] cait_xs24_384 — reshape_2d element count mismatch (Gather 4D op добавлен, но upstream shape проблема)
- [ ] MaskRCNN-12-int8 — spatial broadcast mismatch 14×14 + 7×7
- [ ] roberta dynamic — динамические shapes без input_shapes (требует shape inference)

#### Баги / ограничения
- [x] **ELU на Vulkan** — добавлен шейдер `elu.comp` + регистрация в ggml-vulkan.cpp (6 точек). Тесты CPU и Vulkan проходят.
- [x] **ONNX runtime с scheduler** — переделан на `ggml_backend_sched` (Vulkan + CPU fallback). Unsupported Vulkan ops автоматически выполняются на CPU.
- [x] **LayerNorm broadcast** — 2D+ input с 1D scale/bias: transpose → norm → scale/bias → transpose back
- [x] **Softmax axis** — фикс для opset < 13 (default axis=1), reshape для non-ne[0] axis
- [x] **Reshape + Constant** — Reshape теперь ищет shape в Constant nodes (не только в initializers)
- [x] **Scalar Constant** — TensorProto с n_dims=0 (скаляры) теперь корректно создают 1-элементный тензор
- [x] **Broadcast** — numpy-style broadcast для Add/Sub/Mul/Div: left-align, right-align, greedy dim-matching

#### MVP + трансформеры — готов ✓
Все базовые + трансформерные ops реализованы. 12/15 моделей из ONNX Zoo работают.

#### Исправлено в 0.6.7
- [x] **Native 5D tensor support** — `ggml_view_5d()`, `ggml_repeat_5d()` добавлены в ggml API. CPU repeat kernels (f32/f16) обновлены на 5D. Vulkan repeat: dim3×dim4 collapse в dispatch (push constants 128 байт, шейдеры без изменений).
- [x] **ONNX pipeline 4D→5D** — ~20 мест в onnx_ggml.c обновлены: initializers, inputs, Constant, ConstantOfShape, broadcast, Softmax, Reshape, Slice, Split, Expand, Tile, Gather. Helpers: `onnx_reshape_nd()`, `onnx_new_tensor_nd()`, `ne_product()`. slice_fill arrays в onnx_ggml.h обновлены на `[GGML_MAX_DIMS]`.
- [x] Buffer overflow в deferred arrays — shape_tensors_ne[64] переполнялся, введён ONNX_MAX_DEFERRED=512
- [x] Transpose: инвертированная перестановка осей — ggml_permute ожидает src→dst, код строил dst→src
- [x] Generic path ndims — введена out_nd переменная, handler выставляет ndims авторитетно (Transpose, MatMul)
- [x] ConstantOfShape ndims — tmap_put_nd с правильным рангом
- [x] Auto-cast I32→F32 в бинарных операторах (Add, Sub, Mul, Div)
- [x] cval propagation для Add, Sub, Div, Unsqueeze, Gather(scalar), Constant(scalar), initializers(int64)
- [x] Cast F32↔I32 через ggml_cast (был pass-through)
- [x] Gather auto-cast indices F32→I32
- [x] Shape op: сохранение batch dim=1 через max(tmap_ndims, ggml_n_dims)
- [x] NonZero deferred fill после sched alloc
- [x] ConstantOfShape cval fallback (Shape→Gather→Add→Div цепочки)
- [x] Expand rank promotion (left-pad с 1s)
- [x] Squeeze: использует tmap_get_ndims вместо hardcoded 4
- [x] Gather на shape-тензорах: scalar element access вместо ggml_get_rows
- [x] **ConstantOfShape INT64/INT32/DOUBLE** — value attribute теперь читается с учётом data_type (было: всегда float → мусор для INT64). Закрыло roberta-9 NaN.
- [x] **Gather axis=0 на rank>2** — reshape 2D → get_rows → reshape back (было: assert fail на 4D)
- [x] **ScatterElements** — новый GGML_OP_SCATTER_ELEMENTS: CPU kernel + Vulkan шейдер (atomicAdd для reduction=add)
- **РЕГРЕССИЯ**: Transpose fix сломал MaskRCNN и xcit_tiny (нужна проверка)

#### Исправлено в 0.6.6
- [x] botnet26t_256 — RelPosBias2D fused custom op (pre-pass scanner + emit ggml_map_custom3)
- [x] Pinned staging buffer для GPU input transfer (ggml_backend_vk_host_buffer_type)
- [x] onnx_device_info() NULL guards (segfault fix)

#### Исправлено в 0.6.5
- [x] `ggml_predict()` с `stochastic=TRUE` dropout — `training=FALSE` теперь передаётся в `nn_build_graph`, маска не применяется при инференсе
- [x] `ggml_evaluate()` возвращает `n_samples` + считает метрики на всех сэмплах без truncation
- [x] Новый пример `titanic_classification.R` — бинарная классификация, ~82% val accuracy

#### Исправлено в 0.6.x
- [x] Conv grouped (group>1: split+conv+concat, depthwise: ggml_conv_2d_dw)
- [x] MaxPool ceil_mode + asymmetric padding
- [x] Concat axis mapping (GGML_MAX_DIMS для multi-dim, 1 для 1D shape tensors)
- [x] Compile-time value propagation (cval) для Shape→Slice→Concat→Reshape chains
- [x] TP_RAW_DATA protobuf field fix (field 9, не 13)
- [x] detectCores() NA handling (Kaggle/cloud)

#### Следующий этап — расширенная совместимость
- [ ] **sageconv ScatterElements** — op реализован, но indices shape mismatch (2D indices нужен 1D extract)
- [ ] **cait_xs24_384** — reshape_2d element count mismatch (upstream shape propagation)
- [ ] **MaskRCNN-12-int8** — spatial broadcast mismatch 14×14 + 7×7 (Resize upstream?)
- [ ] **xcit_tiny** — broadcast dim mismatch a=28, b=32
- [ ] MatMul broadcast для 3D+ тензоров (batched matmul)
- [ ] NonZero, Equal, Less, Greater — логические ops
- [x] botnet26t_256 — RelPosBias2D fused custom op (pre-pass + ggml_map_custom3, CPU kernel)
- [ ] botnet26t_256 — Vulkan шейдер для RelPosBias2D (будущая оптимизация)

#### ONNX ops — следующая волна

##### 1. Indexing / логика выбора
- [ ] ScatterND — изменение значений по многомерным индексам
- [ ] GatherND — выборка по многомерным индексам (детекторы объектов)
- [ ] NonMaxSuppression (NMS) — убирает дублирующиеся рамки (YOLO, SSD)
- [ ] TopK — выбор крупнейших элементов (классификация, генерация текста)

##### 2. Активации
- [ ] HardSigmoid / HardSwish — мобильные сети (MobileNetV3)
- [ ] PRelu — ReLU с обучаемым коэффициентом
- [ ] GridSample — трансформация изображений, GAN, ворпинг

##### 3. Последовательности (Recurrent/Time)
- [ ] LSTM / GRU — обработка последовательностей (аудио)
- [ ] NonZero — индексы ненулевых элементов (маскирование)
- [ ] Range — генерация последовательности (позиционное кодирование)

##### 4. Математика и редукция
- [ ] Einsum — универсальные тензорные вычисления (замена MatMul/Transpose комбинаций)
- [ ] ReduceMax / ReduceMin — экстремумы по осям
- [ ] Mod — остаток от деления
- [ ] BitShift / BitwiseAnd/Or/Xor — оптимизированные модели

##### 5. Случайные числа (Stochastic)
- [ ] RandomNormal / RandomUniform — генерация шума в графе (Stable Diffusion)

##### 6. Контроль потока (Control Flow)
- [ ] If — ветвление графа
- [ ] Loop — циклы (авторегрессионная генерация токенов)

#### ONNX цепочки — интеграционные тесты

Тесты на типовые end-to-end цепочки ops (генерируем минимальный .onnx в R, прогоняем через `onnx_run`).

- [ ] **Классификация** (ResNet/EfficientNet): Conv → BatchNorm → Relu → GlobalAveragePool → Flatten → MatMul → Softmax — все ops готовы
- [ ] **Трансформер / BERT**: Gather(embeddings) → Add(pos_emb) → LayerNorm → MatMul(Q/K/V) → Softmax → MatMul → Add(residual) — все ops готовы
- [ ] **Детекция объектов** (YOLO/SSD): Conv → Sigmoid → Reshape → TopK → GatherND → NonMaxSuppression → Squeeze
- [ ] **Сегментация** (MaskRCNN): RoiAlign → Conv → Upsample → Sigmoid → NonZero → ScatterND
- [ ] **Позиционное кодирование** (GPT/RoBERTa): Shape → Slice → Sub → Range → Unsqueeze → Expand → Gather(pos_table)
- [ ] **SuperResolution** (ESRGAN/EDSR): Conv → LeakyRelu → Add(residual) → Conv → PixelShuffle(Reshape+Transpose) → Clip — все ops готовы
- [ ] **Авторегрессионная генерация** (GPT-NeoX): MatMul(logits) → Div(temperature) → Softmax → TopK → Gather → Concat(kv_cache)
- [ ] **Голосовые модели** (Whisper/TTS): LSTM/GRU → Transpose → Reshape → MatMul → LogSoftmax → ArgMax
- [ ] **GAN / SD UNet**: RandomNormal → Conv → GroupNorm → SiLU → MatMul(attention) → Add(skip) → ConvTranspose
- [ ] **Граф-нейросети** (SAGEConv): Gather(node_features) → ScatterElements → MatMul → Add(bias) → Relu → Concat → LayerNorm — все ops готовы
