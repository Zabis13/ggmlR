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

### Оптимизация
- [ ] Профилирование scheduler overhead
- [ ] Минимизация копий между GPU
- [ ] Автоматический выбор backend по размеру задачи

### Интеграция
- [ ] Интеграция с huggingface transformers
- [ ] Примеры квантизированных моделей (Q4_0, Q8_0)
