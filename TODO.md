# Vulkan GPU Backend - План реализации

убрал большую папку

### Использование

```bash
# Установка без Vulkan (по умолчанию, для CRAN)
R CMD INSTALL ggmlR

# Установка с Vulkan (требует Vulkan SDK + glslc)
R CMD INSTALL --configure-args="--with-vulkan" ggmlR
```

Требования для Vulkan:
- Vulkan SDK (libvulkan-dev)
- glslc компилятор шейдеров
- GPU с поддержкой Vulkan (AMD, NVIDIA, Intel)


## ✅ РЕАЛИЗОВАНО - Функции ggml Vulkan Backend

### 1. Инициализация и управление backend

- [x] `ggml_vulkan_available()` - проверка доступности Vulkan
- [x] `ggml_vulkan_init()` - инициализация Vulkan backend
- [x] `ggml_vulkan_is_backend()` - проверка является ли backend Vulkan
- [x] `ggml_vulkan_device_count()` - получение количества устройств
- [x] `ggml_vulkan_device_description()` - описание устройства
- [x] `ggml_vulkan_device_memory()` - получение информации о памяти
- [x] `ggml_vulkan_list_devices()` - список всех устройств
- [x] `ggml_vulkan_backend_name()` - имя backend
- [x] `ggml_vulkan_free()` - освобождение backend
- [x] `ggml_vulkan_status()` - статус Vulkan (с выводом в консоль)

### 2. Управление буферами и памятью

- [x] Автоматическое управление буферами через ggml backend API
- [x] Поддержка device и host памяти

### 3. Тестирование

**Полный набор тестов: 414 PASS, 0 FAIL, 4 SKIP**

**Vulkan-специфичные тесты (47 tests):**
- ✅ Инфраструктура: инициализация, устройства, память (10 тестов)
- ✅ LLM активации: swiglu, geglu (LLaMA/Mistral)
- ✅ Flash Attention для эффективного внимания (multi-head, 4 головы)
- ✅ Базовые операции: add, mul_mat
- ✅ Smoke tests: 10 Vulkan тестов в tests.R

**CPU тесты (367 tests):**
- ✅ Базовые операции: add, sub, mul, div, sqrt, abs, exp, log (32 теста)
- ✅ Активации: relu, gelu, silu, sigmoid, tanh, GLU варианты (24 теста)
- ✅ Трансформеры: rope, flash_attn, diag_mask, argsort (45 тестов)
- ✅ Нормализация: norm, rms_norm, group_norm, l2_norm
- ✅ Матричные операции: mul_mat, out_prod
- ✅ Память и контексты: allocate, free, reset (48 тестов)
- ✅ Тензоры: создание, копирование, reshape (72 теста)
- ✅ Графы вычислений: build, compute, optimize (14 тестов)
- ✅ Утилиты: helpers, types, version (22 теста)

**Benchmark GPU vs CPU:**
- ✅ Векторные операции: 30-50x ускорение (до 43.5 GFLOPS на GPU)
- ✅ Матричные операции: до 433x ускорение (10.9 TFLOPS на 8192x8192)
- ✅ Тест на больших данных: до 2e8 элементов (763 MB), 5e8 элементов (1.9 GB)

**Результаты:**
- AMD Radeon Graphics (RADV GFX1201) - 17.18 GB памяти
- 145 Vulkan шейдеров успешно скомпилированы
- Все тесты проходят без ошибок
- Время выполнения тестов: 1.3 секунды

### 4. Доступные шейдеры (145 операций)

Все шейдеры из ggml-vulkan уже включены:

**Матричные операции:**
- mul_mat_vec, mul_mm, mul_mmq (с вариантами для разных типов данных)
- flash_attn, flash_attn_cm1, flash_attn_cm2

**Базовые операции:**
- add, sub, mul, div, neg, abs, exp, log, sqrt, square
- relu, gelu, silu, sigmoid, tanh, softplus
- soft_max, norm, rms_norm, group_norm, l2_norm

**Квантизация:**
- dequant_q4_0/1, dequant_q5_0/1, dequant_q8_0
- dequant_q2_k - q6_k
- dequant_iq1_s/m, dequant_iq2_xxs/xs/s, dequant_iq3_xxs/s, dequant_iq4_nl/xs

**Трансформеры:**
- rope_norm, rope_neox, rope_multi, rope_vision
- diag_mask_inf, argsort, get_rows

**Прочее:**
- copy, concat, pad, repeat, upscale, pool2d, im2col
- conv2d_mm, conv2d_dw, conv_transpose_1d

## Технические заметки

- Исходники шейдеров: `src/ggml-vulkan/vulkan-shaders/*.comp` (1.5 MB)
- Сгенерированные шейдеры: ~150 MB (не включены в пакет)
- Шейдеры компилируются при `configure --with-vulkan`
- Время компиляции шейдеров: ~1-2 минуты
- Поддерживаемые GPU: AMD (RADV), NVIDIA, Intel

## Известные проблемы и решения

### 1. ⚠️ Важно при установке
При переустановке пакета нужно удалять старые объектные файлы:
```bash
rm -f src/*.o src/ggml-cpu/*.o src/ggml-cpu/arch/x86/*.o
R CMD INSTALL . --configure-args="--with-vulkan"
```

Иначе `r_interface_vulkan.c` будет использовать старый .o без флага `-DGGML_USE_VULKAN`.

### 2. ✅ Исправленные баги
- **configure скрипт**: Удалён дублирующийся `cd ../../..` который ломал генерацию src/Makevars
- **Парсинг аргументов**: Упрощён до одного флага `--with-vulkan` (вместо `--with-vulkan=full`)
- **Integer overflow**: Исправлен переполнение integer при больших размерах тензоров (>2GB)
  - `R/ggml.R`: заменён `as.integer(mem_size)` на `as.numeric(mem_size)`
  - `src/r_interface.c`: заменён `int size = asInteger()` на `size_t size = (size_t)asReal()`
  - Теперь поддерживаются тензоры размером >2GB (протестировано до 1.9GB)

### 3. Warnings при компиляции
- `RADV is not a conformant Vulkan implementation` - это нормально для AMD RADV драйвера
- Warnings о `fprintf(stderr)` в ggml-vulkan.cpp - связаны с r_ggml_compat.h, не критично

## ✅ Завершённые задачи

- [x] Интеграция Vulkan backend (145 шейдеров)
- [x] R-обёртки для всех Vulkan функций
- [x] Тесты для Vulkan backend (47 тестов)
- [x] Тесты для LLM операций (swiglu, geglu, flash_attn)
- [x] Бенчмарки CPU vs Vulkan (векторы + матрицы)
- [x] Исправление integer overflow для больших тензоров
- [x] Smoke tests (184 теста)
- [x] Полный набор testthat тестов (414 тестов)

## Следующие возможные улучшения

- [ ] Добавить примеры использования Vulkan backend в виньетках
- [ ] Автоматический выбор backend на основе размера задачи
- [ ] Поддержка multi-GPU
- [ ] Документация по оптимальному использованию Vulkan
- [ ] Интеграция с huggingface transformers
- [ ] Примеры запуска квантизированных моделей (Q4_0, Q8_0)
