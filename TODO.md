## TODO



### llamaR Integration
- [x] Export static library `libggml.a`
- [x] Export headers via `inst/include/`
- [x] Add `gguf.cpp` for GGUF support
- [ ] Sync ggml version with llama.cpp (add `ggml_build_forward_select`)



### Custom Operations
- [ ] `ggml_custom()` / `ggml_custom_inplace()` — требуют C расширение



### Концепция API

Один ядро-тип модели (`ggml_model`) с backend-полем (cpu, vulkan).
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
- [x] Загрузка pre-trained весов из .gguf (`gguf_load`, `gguf_tensor_data`, `gguf_metadata`)
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
- [x] Vulkan 1.3 Synchronization2 — `pipelineBarrier2`/`setEvent2`/`waitEvents2` вместо legacy API
- [x] Push Descriptors (`VK_KHR_push_descriptor`) — убран descriptor pool overhead, runtime guard + fallback
- [ ] Профилирование scheduler overhead
- [ ] Минимизация копий между GPU
- [ ] **KHR coopmat path в conv2d_mm.comp (Вариант A)** — ускорение VAE decode на RDNA4/Ampere+
  - Цель: VAE decode 12.9s → ~4-6s на RX 9070 (сейчас F32 FMA, нет coopmat)
  - Устройство: wave64, M=16 N=16 K=16, subgroup_size=64
  - **Шаг 1 — шейдер** `conv2d_mm.comp`:
    - `#ifdef COOPMAT` ветка (GL_KHR_cooperative_matrix, subgroup scope)
    - Workgroup = 4 subgroup'а по 64 треда = 256 тредов (как в mul_mm.comp)
    - Каждый subgroup вычисляет блок 32×32 результата из 4 coopmat 16×16
    - Shared memory: F16 staging тайл BS_K × BS_CRS + BS_CRS × BS_NPQ (как сейчас)
    - Все subgroup'ы читают общий CRS-тайл → максимальный reuse
    - Accumulator: F32 (coopmat<float, subgroup, 16, 16, Accumulator>)
    - A/B матрицы: F16 (загружаем F32→F16 в shared, потом coopMatLoad)
    - Новые spec constants: `CM_TM`, `CM_TN`, `CM_TK` (дефолт 16×16×16)
  - **Шаг 2 — генерация** `vulkan-shaders-gen.cpp`:
    - Под `#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)` добавить `_cm1` варианты
    - `string_to_spv(name + "_cm1", "conv2d_mm.comp", defines, true, true, false)`
  - **Шаг 3 — pipeline** `ggml-vulkan.cpp`:
    - `pipeline_conv2d_f32_cm1[CONV_SHAPE_COUNT]` и `_f16_f32_cm1`
    - В блоке создания pipeline: `if (device->coopmat_support) { CREATE_CONVS(_cm1) } else if (coopmat2) ...`
    - Spec constants для CM path: WG_SIZE=256, BS_K=64, BS_NPQ=64, BS_CRS=16
  - **Шаг 4 — тест**: `test-flash-attn-q4k.R` как шаблон; benchmark в `test_vae_f16.R`
- [ ] **Vulkan profiling API на R уровне** — экспортировать `vk_perf_logger` из `ggml-vulkan.cpp` в R через `.Call()`, чтобы видеть breakdown по операциям (мс на каждый op/fusion). Нужно для диагностики bottleneck'ов в sd2R sampling loop (552s на Flux). В ggml уже есть `vk_perf_logger_enabled` и timestamp queries — нужен R-интерфейс для включения/чтения результатов.
- [x] **Subgroup shuffle деквантование — Вариант 2 (Q4_K/Q5_K/Q6_K)** — `block_a_to_registers_shuffle()` в `mul_mmq_funcs.glsl`, `USE_SUBGROUP_NO_SHMEM` путь в `mul_mmq.comp`, новый pipeline `pipeline_dequant_mul_mat_mat_q8_1_no_shmem` в `ggml-vulkan.cpp`. Активен на wavefront-64 (RDNA4, subgroup_size=64). Выпущен в 0.7.3.
- [ ] **Subgroup shuffle деквантование — Вариант 3 (все типы)** — расширение варианта 2 (Q4_K/Q5_K/Q6_K) на стандартные quants (Q4_0, Q5_0, Q8_0). Основная сложность: BLOCK_SIZE=256 > gl_SubgroupSize=64 → нужен multi-pass shuffle (4 прохода × 64) или partial shuffle + shmem для остатка.
  - `mul_mmq_funcs.glsl`: `block_a_to_registers_shuffle()` для Q4_0/Q5_0/Q8_0; `block_b_to_registers_shuffle()` (B-сторона тоже через shmem)
  - `mul_mmq.comp`: расширить `#ifdef USE_SUBGROUP_NO_SHMEM` на все типы; убрать `barrier()` в shuffle-пути
  - `ggml-vulkan.cpp`: `CREATE_MMQ` для `pipeline_dequant_mul_mat_mat_q8_1[Q4_0/Q5_0/Q8_0]` с `_no_shmem` суффиксом; дублировать `.f32acc` вариант
  - `vulkan-shaders-gen.cpp` / `CMakeLists.txt`: зарегистрировать новые `_no_shmem` шейдеры
  - Тест: `test-vulkan.R` — сравнение CPU vs GPU output для Q4_0/Q8_0 matmul
- [ ] **Async Compute (Vulkan dual-queue)** — перекрытие transfer и compute через второй VkQueue. Цель: убрать pipeline "пузыри" когда compute блоки ждут планировщика команд. Сейчас все команды идут в один VkQueue.
  - `ggml-vulkan.cpp`: создать второй VkQueue (`compute_queue` + `transfer_queue`) при инициализации устройства, если `queueFamilyProperties` позволяет
  - Добавить `VkSemaphore` для синхронизации между transfer и compute очередями
  - Переделать `ggml_vk_submit()`: DMA upload → `transfer_queue`, compute dispatch → `compute_queue`, барьер через семафор
  - Протестировать на RX 9070 (RDNA4 имеет выделенный DMA engine)
  - Риск: большая правка в `ggml-vulkan.cpp`, возможны регрессии в планировщике (`ggml_backend_sched`)

---

### Autograd / GPU (текущая работа)



#### Новые фичи (0.6.1)
- [x] `dp_train()` — data-parallel training: N реплик, синхронизация весов, усреднение градиентов

#### Следующие фичи
- [ ] Gradient checkpointing — экономия памяти при глубоких сетях
- [x] Flash Attention Q4_K — `GGML_OP_FLASH_ATTN_EXT` с Q4_K K/V на Vulkan (FA_SCALAR + FA_COOPMAT1) — v0.7.2
- [ ] Flash Attention autograd — эффективный attention через ggml_flash_attn_ext в ag_* API
- [ ] Optimizer states в f32 при f16 весах (true mixed precision training)

---

### ONNX ops — MVP subset для трансформеров и SD



#### 3. Attention / линейные блоки
- [x] MatMul, Softmax, Reshape, Transpose, Mul — всё реализовано
- [x] Gemm (transA/transB/alpha/beta + bias)

> ONNX-экспортеры обычно не делают FlashAttention op, а оставляют классический QK^T → Softmax → V — маппится на имеющиеся ggml-ops.



#### 7. Random / Scheduler
> Для SD денойзинг лупа (scheduler, генерация латентов) обычно вне ONNX, в хост-коде. ONNX-UNet получает готовые латенты и timestep. Оставить в R-логике (как в sd2R), ONNX-граф — чистый UNet/VAE/text encoder.

#### Тесты
- [x] 50 тестов в `tests/testthat/test-onnx.R` (R protobuf сериализатор, без Python)
- [x] CPU: все ops проходят
- [x] Vulkan: 24/24 ops проходят

#### Реальные модели (ONNX Model Zoo) — 13/15 OK
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

---

### mlr3 integration

#### v1 (в работе)
- [x] `LearnerClassifGGML` / `LearnerRegrGGML` — R6-классы, sequential + functional API через `model_fn`
- [x] `ggml_default_mlp()` — дефолтный builder (classif + regr)
- [x] Marshal через контейнер `ggmlR_marshaled` (format/version/sha256/payload) — sequential/functional only
- [x] S3 методы `marshal_model` / `unmarshal_model` + ленивая регистрация в `.onLoad`
- [x] Callbacks как `p_uty` (sequential-fit honour'ит; functional-fit молча игнорирует)
- [x] `properties = "weights"` для classif — `task$weights_learner` → `sample_weight` в `ggml_fit`
- [ ] `properties = "weights"` для regr — **заблокировано**: `ggml_fit_sequential` для mse делает
  `y <- y * sample_weight`, что эквивалентно `(w*y - pred)²`, а не `w*(y - pred)²`. Сначала надо
  исправить семантику weighted mse в самом `ggml_fit_sequential` (scale residual, не target),
  потом пробросить в learner.
- [x] Регистрация в `mlr_learners` (через `.onLoad`) + `DESCRIPTION` Suggests (mlr3 >= 0.21, paradox, R6, checkmate, digest, mlr3pipelines) — v0.7.1
- [x] Обновлены README (секция mlr3 + parsnip Integration), NEWS (0.7.1), TODO — v0.7.1
- [x] `tests/testthat/test-mlr3-learner.R` smoke-тесты train/predict/marshal/CV — v0.7.1
- [x] `tests/testthat/test-parsnip.R` — тесты classif/regr/prob/learn_rate/backend=gpu — v0.7.1
- [x] parsnip `"ggml"` engine для `mlp()` classif + regr — v0.7.1
- [x] `backend="gpu"` → `"vulkan"` в parsnip fit функциях — v0.7.1
- [x] Vignette `mlr3-integration.Rmd` — принудительная регистрация в setup chunk — v0.7.1
- [x] R6 классы всегда определяются (убрана `if (requireNamespace)` обёртка) — v0.7.1
- [x] Убран `mlr3misc` из Suggests, регистрация через `setHook` напрямую — v0.7.1

#### v2 — autograd support (отложено)
Причина отсрочки: autograd API (`ag_*`, `with_grad_tape`, `backward`, `ag_sequential`)
использует другой training loop и не поддерживает `ggml_save_model()`. Параметры
модели — external pointers на C-уровневые ggml тензоры, `saveRDS` даёт мусор.

Нужно для полной поддержки в mlr3:

- [ ] **autograd training tradepath в learner** — диспетчер в `.train()`:
  `inherits(model, "ag_sequential") → autograd loop (optimizer_adam + grad tape + backward)`.
  Дефолтный `training_fn`, переопределяемый пользователем. Новые параметры:
  `learning_rate`, `max_grad_norm`. Даёт `learning_rate` в tuning (недоступен для
  sequential API, т.к. `ggml_compile` его не принимает).
- [ ] **Marshal для autograd — вариант M2 (weights + model_fn)**: контейнер хранит
  `list(weights = named_list_of_numeric_vectors, pars)`, архитектура восстанавливается
  вызовом `self$model_fn(task, ...)` заново, веса переливаются по именам из
  `$parameters()`. Работает потому что `model_fn` сериализуется как часть learner'а.
  Требует: (a) обход дерева слоёв с `ggml_backend_tensor_get_data()`, (b) обратная
  заливка через `$data <- ...`, (c) проверка совпадения имён параметров.
- [ ] **`ag_save_model()` / `ag_load_model()` в ggmlR** — опциональная альтернатива M2,
  если хочется marshal без обязательного `model_fn`. Гораздо больше работы: нужна
  сериализация произвольного environment-based module с замыканиями. Оценка 200-400
  строк + тесты. Не блокер для mlr3 integration (M2 уже покрывает use case).
- [ ] **`dp_train()` интеграция** — опционально, multi-GPU training как fast-path для
  autograd learner'а когда `n_gpu > 1`. Нужно проработать: `dp_train` возвращает
  `replicas[[1L]]`, learner хранит одну модель, predict идёт через forward на этой
  модели. Профит — бесплатный grad clipping + weight sync.
- [ ] **`class_weight` / `sample_weight` + `properties = "weights"`** — sequential
  `ggml_fit` это поддерживает, learner'у надо пробросить `task$weights` (если
  `"weights"` в properties, mlr3 автоматически кладёт их в task).
- [ ] Документация: явное сравнение sequential vs autograd tradepath, когда какой
  выбирать, ограничения marshal для autograd.
