## TODO

### llamaR Integration
- [x] Export static library `libggml.a`
- [x] Export headers via `inst/include/`
- [x] Add `gguf.cpp` for GGUF support
- [ ] Sync ggml version with llama.cpp (add `ggml_build_forward_select`)

### Custom Operations
- [ ] `ggml_custom()` / `ggml_custom_inplace()` — требуют C расширение

### Документация
- [ ] Виньетка: Vulkan backend tutorial
- [ ] Виньетка: Multi-GPU inference
- [ ] Примеры квантизированных моделей

### Оптимизация
- [ ] Профилирование scheduler overhead
- [ ] Минимизация копий между GPU
- [ ] Автоматический выбор backend по размеру задачи

### Интеграция
- [ ] Интеграция с huggingface transformers
- [ ] Примеры квантизированных моделей (Q4_0, Q8_0)
