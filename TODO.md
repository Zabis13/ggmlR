
## Статус реализации GGML API (v0.5.1+)

**Экспортировано функций: 397**

---

## ✅ РЕАЛИЗОВАНО

### Ядро GGML
- Управление контекстом: `ggml_init`, `ggml_free`, `ggml_reset`
- Создание тензоров: `ggml_new_tensor_1d/2d/3d/4d`, `ggml_dup_tensor`
- Доступ к данным: `ggml_set_f32`, `ggml_get_f32`, `ggml_set_i32`, `ggml_get_i32`
- Информация о тензорах: `ggml_nelements`, `ggml_nbytes`, `ggml_tensor_shape`, `ggml_tensor_type`

### Операции (150+ функций)
- **Арифметика**: add, sub, mul, div, scale, clamp (+ inplace варианты)
- **Математика**: sqr, sqrt, exp, log, abs, neg, sin, cos, ceil, floor, round
- **Активации**: relu, gelu, silu, sigmoid, tanh, elu, softplus, hardsigmoid, hardswish, leaky_relu
- **GLU**: glu, reglu, geglu, swiglu (+ split варианты)
- **Нормализация**: norm, rms_norm, l2_norm, group_norm (+ inplace варианты)
- **Softmax**: soft_max, soft_max_ext (+ inplace и back варианты)
- **Редукция**: sum, sum_rows, mean, argmax
- **Матричные**: mul_mat, mul_mat_id, out_prod, transpose
- **Reshape/View**: reshape_1d/2d/3d/4d, view_1d/2d/3d/4d, permute, cont
- **CNN**: conv_1d, conv_2d, conv_transpose_1d, pool_1d, pool_2d, im2col
- **Attention**: flash_attn_ext, flash_attn_back, diag_mask_inf/zero
- **RoPE**: rope, rope_ext, rope_multi (+ inplace и back варианты)

### Backend System (60+ функций)
- **CPU Backend**: `ggml_backend_cpu_init`, `ggml_backend_cpu_set_n_threads`
- **Device Management**: `ggml_backend_dev_count/get/by_name/by_type`
- **Device Properties**: `ggml_backend_dev_name/description/memory/type/get_props`
- **Registry**: `ggml_backend_reg_count/get/by_name/dev_count/dev_get`
- **Buffer Management**: `ggml_backend_buffer_*` (free, get_size, name, clear, usage, is_host)
- **Events**: `ggml_backend_event_new/free/record/synchronize/wait`
- **Graph Plans**: `ggml_backend_graph_plan_create/free/compute`
- **Async Operations**: `ggml_backend_tensor_set/get/copy_async`
- **Scheduler**: `ggml_backend_sched_*` (new, free, reserve, alloc_graph, graph_compute, synchronize)
- **Init Helpers**: `ggml_backend_init_by_name/by_type/init_best`, `ggml_backend_load/load_all`

### Vulkan Backend (10 функций)
- `ggml_vulkan_available`, `ggml_vulkan_device_count/description/memory`
- `ggml_vulkan_init`, `ggml_vulkan_free`, `ggml_vulkan_list_devices`

### Optimizer System (39 функций)
- **Dataset**: `ggml_opt_dataset_init/free/ndata/data/labels/shuffle/get_batch`
- **Context**: `ggml_opt_init/free/reset/alloc/static_graphs`
- **Training**: `ggml_opt_fit`, `ggml_opt_epoch`, `ggml_opt_eval`, `ggml_opt_grad_acc`
- **Tensors**: `ggml_opt_inputs/outputs/labels/loss/pred/ncorrect`
- **Results**: `ggml_opt_result_init/free/reset/ndata/loss/accuracy/pred`
- **Constants**: loss types (mean, sum, cross_entropy, mse), optimizer types (adamw, sgd)

### CPU Feature Detection (28 функций)
- **x86**: sse3, ssse3, avx, avx2, avx_vnni, bmi2, f16c, fma, avx512, avx512_vbmi/vnni/bf16, amx_int8
- **ARM**: neon, arm_fma, fp16_va, dotprod, matmul_int8, sve, sme + sve_cnt
- **Other**: riscv_v + rvv_vlen, vsx, vxe, wasm_simd, llamafile
- **Helper**: `ggml_cpu_features()` — все фичи как named list

### Tensor Layout/Contiguity (9 функций)
- `ggml_is_contiguous_0/1/2`, `ggml_is_contiguous_channels/rows`
- `ggml_is_contiguously_allocated`, `ggml_are_same_stride`
- `ggml_can_repeat`, `ggml_count_equal`

### Type System (10 функций)
- `ggml_type_name`, `ggml_type_size`, `ggml_type_sizef`, `ggml_blck_size`
- `ggml_is_quantized`, `ggml_ftype_to_ggml_type`
- `ggml_op_name`, `ggml_op_symbol`, `ggml_op_desc`, `ggml_get_unary_op`

### Quantization (4 функции)
- `ggml_quantize_init`, `ggml_quantize_free`, `ggml_quantize_requires_imatrix`
- `ggml_quantize_chunk`

### Graph Operations
- `ggml_build_forward_expand`, `ggml_graph_compute`, `ggml_graph_compute_with_ctx`
- `ggml_graph_n_nodes`, `ggml_graph_node`, `ggml_graph_get_tensor`
- `ggml_graph_print`, `ggml_graph_reset`, `ggml_graph_dump_dot`, `ggml_graph_overhead`

### Memory Allocators
- `ggml_gallocr_new`, `ggml_gallocr_free`, `ggml_gallocr_reserve`
- `ggml_gallocr_alloc_graph`, `ggml_gallocr_get_buffer_size`
- `ggml_backend_alloc_ctx_tensors`

---

## 🔴 НЕ РЕАЛИЗОВАНО (Критичные)

- [ ] `ggml_backend_graph_compute_async()` — async graph compute
- [ ] `ggml_backend_multi_buffer_*()` — multi-buffer операции
- [ ] `ggml_backend_register()` — dynamic backend registration

---

## 🟡 НЕ РЕАЛИЗОВАНО (Средний приоритет)

### Advanced RoPE (1 функция)
- [ ] `ggml_rope_multi_back()` — backward для multi-head RoPE
- ⚠️ `ggml_rope_custom*()` — deprecated, использовать rope_ext

### Graph Introspection (8 функций)
- [ ] `ggml_build_backward_expand()` — для обучения
- [ ] `ggml_graph_add_node()` / `ggml_graph_clear()` / `ggml_graph_cpy()` / `ggml_graph_dup()`
- [ ] `ggml_graph_get_grad()` / `ggml_graph_get_grad_acc()`
- [ ] `ggml_graph_view()`, `ggml_cgraph_eval_order()`
- [ ] `ggml_op_can_inplace()`, `ggml_cplan()`

### Advanced Attention/Loss (6 функций)
- [ ] `ggml_cross_entropy_loss()` / `ggml_cross_entropy_loss_back()`
- [ ] `ggml_cumsum()`
- [ ] `ggml_flash_attn_ext_add_sinks()`
- [ ] `ggml_flash_attn_ext_get_prec()` / `ggml_flash_attn_ext_set_prec()`

---

## 🟢 НЕ РЕАЛИЗОВАНО (Низкий приоритет)

### Низкоуровневая квантизация (100+ функций)
Row-level операции для типов: q4_0, q5_0, q8_0, q2_K-q8_K, iq2_xxs/xs/s, iq3_xxs/s, iq4_nl/xs, tq1_0, tq2_0, mxfp4.

⚠️ Высокоуровневый `ggml_quantize_chunk()` уже реализован.

### Custom Operations (5 функций)
⚠️ Требуют C callback (сложно в R)
- [ ] `ggml_custom()` / `ggml_custom_inplace()`
- [ ] `ggml_set_op_params*()`

### Logging & Debugging (2 функции)
⚠️ Требуют C callback
- [ ] `ggml_log_set()`, `ggml_set_abort_callback()`

### Internal Functions (не экспортируются)
- `ggml_are_same_layout()` — inline в ggml-impl.h
- `ggml_can_fuse*()`, `ggml_check_edges()` — требуют cgraph internals

---

## Use Cases Status

| Use Case | Статус | Комментарий |
|----------|--------|-------------|
| Inference на CPU | ✅ Полная | Backend, scheduler, все операции |
| Inference на GPU (Vulkan) | ✅ Базовая | Device discovery, compute |
| Multi-GPU | ✅ Базовая | Scheduler, device management |
| Обучение/Fine-tuning | ✅ Полная | ggml_opt_* (39 функций) |
| Экономия памяти | ✅ Полная | 28+ inplace операций |
| Квантизация | ✅ Базовая | quantize_chunk, type system |
| Диагностика | ✅ Полная | CPU features, tensor layout, type info |
| Custom операции | ❌ | Требуют C callbacks |

---

## Следующие шаги

### Документация
- [ ] Виньетка: Vulkan backend tutorial
- [ ] Виньетка: Multi-GPU inference
- [ ] Примеры квантизированных моделей

### Функциональность
- [ ] `ggml_cross_entropy_loss()` — для обучения классификаторов
- [ ] `ggml_build_backward_expand()` — автоматическое построение backward graph

### Оптимизация
- [ ] Профилирование scheduler overhead
- [ ] Минимизация копий между GPU
