## TODO



### llamaR Integration
- [x] Export static library `libggml.a`
- [x] Export headers via `inst/include/`
- [x] Add `gguf.cpp` for GGUF support
- [ ] Sync ggml version with llama.cpp (add `ggml_build_forward_select`)



### Custom Operations
- [ ] `ggml_custom()` / `ggml_custom_inplace()` — требуют C расширение








### Архитектура (Functional API)





#### Блок 3 — Расширения
- [x] Shared layers — повторное использование одного слоя (по имени `name=`) для Siamese/multi-branch топологий; `ggml_set_param` вызывается один раз на shared веса
- [ ] Несколько входов (multi-input `ggml_model`)
- [x] Сохранение/загрузка архитектуры (не только весов)
- [ ] Загрузка pre-trained весов из .gguf
- [ ] `ggml_layer_concatenate()` с backward pass (требует патча ggml C)
- [ ] Dropout маска per-batch (требует C-расширения)

### Данные
- [ ] Data augmentation (flip, rotate, crop)
- [ ] `ggml_fit()` с generator/iterator вместо полной матрицы в памяти

### Оптимизация
- [ ] Профилирование scheduler overhead
- [ ] Минимизация копий между GPU
- [ ] **Vulkan profiling API на R уровне** — экспортировать `vk_perf_logger` из `ggml-vulkan.cpp` в R через `.Call()`, чтобы видеть breakdown по операциям (мс на каждый op/fusion). Нужно для диагностики bottleneck'ов в sd2R sampling loop (552s на Flux). В ggml уже есть `vk_perf_logger_enabled` и timestamp queries — нужен R-интерфейс для включения/чтения результатов.

### Интеграция
- [ ] Примеры квантизированных моделей (Q4_0, Q8_0)

### Roadmap (новые фичи)

#### Training API
- [ ] Custom layer API за 5 строк R-кода (без C++)
- [ ] Проверить совместимость с keras3 (compile/fit должны быть идентичны)

#### Callbacks & Monitoring
- [ ] Cost tracker — реальные затраты GPU/CPU в реальном времени во время обучения
- [ ] Auto-quantize — автоматическое уменьшение модели в 4 раза при ухудшении метрики

#### Deployment
- [ ] Экспорт в GGUF + генерация Plumber API (2 строки кода)
- [ ] Vetiver integration через единый S3-метод

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
- [ ] Slice (отдельный от Gather)
- [ ] Unsqueeze, Squeeze (заглушки через ggml_cont, нужна реальная логика)
- [ ] Split
- [ ] Expand / Broadcast (можно частично эмулировать через shape-логику)

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
- [ ] ConvTranspose (для апскейла в декодерах UNet/VAE)

#### 5. Pooling / Upsampling
- [x] MaxPool, AveragePool, GlobalAveragePool
- [ ] Upsample / Resize (nearest/linear интерполяция)

#### 6. Прочие реализованные ops
- [x] Sqrt, Exp, Log, Abs, Neg, Floor, Ceil, Clip
- [x] ReduceMean, ReduceSum
- [x] Flatten, Dropout (pass-through)
- [ ] Constant (TODO — создание тензора из атрибутов)

#### 7. Random / Scheduler
> Для SD денойзинг лупа (scheduler, генерация латентов) обычно вне ONNX, в хост-коде. ONNX-UNet получает готовые латенты и timestep. Оставить в R-логике (как в sd2R), ONNX-граф — чистый UNet/VAE/text encoder.

#### Тесты
- [x] 34 теста в `tests/testthat/test-onnx.R` (R protobuf сериализатор, без Python)
- [x] CPU: все ops проходят
- [x] Vulkan: 24/24 ops проходят

#### Баги / ограничения
- [x] **ELU на Vulkan** — добавлен шейдер `elu.comp` + регистрация в ggml-vulkan.cpp (6 точек). Тесты CPU и Vulkan проходят.
- [x] **ONNX runtime с scheduler** — переделан на `ggml_backend_sched` (Vulkan + CPU fallback). Unsupported Vulkan ops автоматически выполняются на CPU.
- [x] **LayerNorm broadcast** — 2D+ input с 1D scale/bias: transpose → norm → scale/bias → transpose back

#### Осталось реализовать (MVP)
1. ConvTranspose, Slice, Split, Expand, Resize/Upsample, Constant
2. Squeeze/Unsqueeze с реальной логикой dims
