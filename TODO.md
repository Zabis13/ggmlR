## TODO

### CRAN Submission
- [x] Расшифровать аббревиатуры в DESCRIPTION (MSE, AdamW, SGD, GPU)
- [x] Добавить авторов и copyright holders в Authors@R
- [x] Заменить `\dontrun{}` на `\donttest{}` в примерах
- [x] Добавить `\value` во все .Rd файлы
- [x] Исправить некорректные примеры (неопределённые переменные)
- [x] Удалить `.gitkeep` из inst/lib/

### llamaR Integration
- [x] Export static library `libggml.a`
- [x] Export headers via `inst/include/`
- [x] Add `gguf.cpp` for GGUF support
- [ ] Sync ggml version with llama.cpp (add `ggml_build_forward_select`)

### sdR (Stable Diffusion port)
- [x] `ggml_timestep_embedding()` — базовая ggml операция для SD
- [x] N-D indexed tensor access (`ggml_set_f32_nd`, `ggml_get_f32_nd`, etc.)
- [x] Tensor utilities (`ggml_tensor_nb`, `ggml_tensor_copy`, `ggml_tensor_num`, etc.)
- [x] Backend sync utilities (`ggml_backend_tensor_get_and_sync`, `ggml_backend_tensor_get_f32_first`)
- [x] Context tensor iteration (`ggml_get_first_tensor`, `ggml_get_next_tensor`)

### Custom Operations
- [ ] `ggml_custom()` / `ggml_custom_inplace()` — требуют C расширение

### Документация
- [x] Виньетка: Vulkan backend tutorial
- [x] Виньетка: Multi-GPU inference
- [x] Виньетка: Working with Quantized Models

### Sequential API
- [x] Fix `ggml_evaluate()` — веса не передавались из обученной модели (random init вместо trained weights)
- [x] Тест `ggml_evaluate()` — проверка что accuracy после evaluate совпадает с train
- [x] `ggml_predict()` — предсказание без меток
- [x] Сохранение/загрузка весов модели (save/load)
- [x] `validation_data = list(x_val, y_val)` в `ggml_fit()` — как в Keras
- [x] `class_weight` и `sample_weight` в `ggml_fit()` и `ggml_evaluate()`

### Новые слои
- [x] `ggml_layer_batch_norm()` — batch normalization через ggml_norm + gamma/beta
- [x] `ggml_layer_conv_1d()` — 1D свёртка через ggml_conv_1d (shape inference работает; training требует fix backend_sched для im2col F16)
- [x] `ggml_layer_embedding()` — для NLP задач, I32 вход, ggml_get_rows lookup, Uniform(-0.05, 0.05) init
- [x] `ggml_layer_dropout()` — детерминированный режим (`stochastic=FALSE`) и inverted dropout (`stochastic=TRUE`) с Bernoulli-маской на epoch; маска через ggml_mul (backward корректен)
- [x] `ggml_layer_global_max_pooling_2d()` / `ggml_layer_global_average_pooling_2d()`
- [x] `ggml_layer_lstm()` / `ggml_layer_gru()` — рекуррентные слои

### API удобства
- [x] `summary(model)` — детальный вывод (trainable/non-trainable params, memory)
- [x] `ggml_predict_classes()` — argmax от predict, возвращает 1-based integer вектор
- [x] History объект из `ggml_fit()` — loss/accuracy по эпохам, `plot(history)`, `print(history)`
- [x] Callbacks: early stopping, learning rate scheduler
  - [x] C: `R_ggml_opt_init_for_fit()` + `r_opt_get_constant_lr` — обновление LR без пересоздания контекста, momentum AdamW сохраняется
  - [x] C: `R_ggml_opt_set_lr()` / `R_ggml_opt_get_lr()` — смена LR между эпохами через userdata-структуру
  - [x] C: регистрация новых функций в `r_interface.c`
  - [x] R: `R/callbacks.R` — `ggml_callback_early_stopping()`, расписания LR
  - [x] R: готовые расписания — `ggml_schedule_step_decay()`, `ggml_schedule_cosine_decay()`, `ggml_schedule_reduce_on_plateau()`
  - [x] R: `ggml_fit()` — цикл эпох в R, параметр `callbacks = list(on_epoch_begin=..., on_epoch_end=...)`, возвращает `data.frame`
  - [x] Тесты: `tests/testthat/test-callbacks.R` — 67 тестов
- [x] `ggml_get_layer(model, index=)` / `ggml_get_layer(model, name=)` — доступ к слою по индексу или имени
- [x] `ggml_pop_layer(model)` — удаление последнего слоя
- [x] Именованные слои — автогенерация (`dense_1`, `conv_2d_1`, ...) и кастомные имена (`name=`)
- [x] `ggml_freeze_weights(model, from, to)` / `ggml_unfreeze_weights()` — заморозка весов, `layer$trainable`

### Архитектура (Functional API)

#### Блок 1 — Основа (реализован)
- [x] `ggml_input()` — создание входного узла графа
- [x] `ggml_model(inputs, outputs)` — сборка модели из узлов
- [x] Нелинейная топология — skip connections, residual blocks
- [x] `ggml_layer_add()` — поэлементное сложение тензоров (residual/skip connections)
- [x] `ggml_layer_concatenate()` — конкатенация вдоль оси (только forward pass; ggml CONCAT не имеет backward)
- [x] Двойная диспетчеризация в `ggml_layer_*()` — принимают и `ggml_sequential_model`, и `ggml_tensor_node`
- [x] S3-диспетчеризация `ggml_compile/fit/evaluate/predict` по классу модели
- [x] Топологическая сортировка DAG-графа (`nn_topo_sort`)
- [x] Ленивое построение графа при fit/evaluate/predict (`nn_build_functional_graph`)
- [x] 51 тест в `tests/testthat/test-nn-functional.R`

#### Блок 2 — Базовые слои (реализован)
- [x] `ggml_layer_dropout(stochastic=FALSE)` — детерминированный expected-value dropout
- [x] `ggml_layer_dropout(stochastic=TRUE)` — inverted dropout с Bernoulli-маской на epoch
- [x] `ggml_layer_embedding(vocab_size, dim)` — token embedding lookup (I32 вход)
- [x] `ggml_input(dtype="int32")` — поддержка целочисленных входов
- [x] Multi-output: `ggml_model(outputs=list(...))`, `ggml_predict()` возвращает list матриц
- [x] Восстановление весов между fit/predict/evaluate (через `model$node_weights`)
- [x] 90 тестов в `test-nn-functional.R`

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

#### Сделано
- [x] GPU dispatch для всех `ag_*` операций (matmul, add, sub, mul, scale, relu, sigmoid, tanh, softmax, log, exp, clamp, sum/mean all/dim=1/2, pow, transpose, reshape)
- [x] Mixed precision: `ag_dtype()` / `ag_default_dtype()`, `GGML_TYPE_BF16`, dtype в `ag_tensor` / `ag_param`
- [x] BF16 → F16 fallback для Vulkan (`.ag_compute_dtype()`)
- [x] C-уровень: F16/BF16 в `R_ggml_backend_tensor_set/get` (`r_interface_graph.c`)
- [x] Новые Vulkan f16 шейдеры: `scale_f16`, `sqr_f16`, `sqrt_f16`, `soft_max_f16`
- [x] `ag_multihead_attention` — self/cross attention, causal mask, dropout, train/eval (30 тестов)
- [x] Пример `transformer_encoder_demo.R` — Transformer Encoder block + tiny LM

#### Баги (исправлено в 0.6.1)
- [x] **`ag_mul` CPU broadcast** — `[d,s] * [1,s]` и `[d,s] * [d,1]` — `rep`-индексация + корректный reduce в backward
- [x] **`ag_sub` CPU broadcast** — аналогичное исправление
- [x] **`ggml_sum_rows` f16 на Vulkan** — добавлена F16→F16 ветка в `ggml-vulkan.cpp` (`pipeline_sum_rows[1]`)
- [x] **NaN в обучении при f16** — целочисленные targets → one-hot внутри `ag_softmax_cross_entropy_loss`; `ag_record` защищён от нетензорных входов
- [x] **`dp_train` утечка device state** — сохранение/восстановление через `on.exit()`

#### Новые фичи (0.6.1)
- [x] `dp_train()` — data-parallel training: N реплик, синхронизация весов, усреднение градиентов

#### Следующие фичи
- [ ] Gradient checkpointing — экономия памяти при глубоких сетях
- [ ] Flash Attention — эффективный attention через ggml_flash_attn_ext
- [ ] Optimizer states в f32 при f16 весах (true mixed precision training)
