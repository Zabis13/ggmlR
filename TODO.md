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

### Новые слои
- [x] `ggml_layer_batch_norm()` — batch normalization через ggml_norm + gamma/beta
- [x] `ggml_layer_conv_1d()` — 1D свёртка через ggml_conv_1d (shape inference работает; training требует fix backend_sched для im2col F16)
- [ ] `ggml_layer_embedding()` — для NLP задач (требует I32 input в fit/predict)
- [ ] Dropout layer (с отключением на inference) — требует C-расширения

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

### Архитектура
- [ ] Functional API — модели с ветвлениями, skip connections
- [ ] `ggml_layer_add()` / `ggml_layer_concat()` — merge слои
- [ ] Загрузка pre-trained весов из .gguf

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


