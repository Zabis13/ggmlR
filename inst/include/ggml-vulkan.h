#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);
GGML_BACKEND_API void ggml_backend_vk_get_device_caps(int device, bool * coopmat_support, bool * coopmat1_fa_support, bool * fp16, uint32_t * subgroup_size, bool * subgroup_no_shmem, uint32_t * subgroup_min_size, uint32_t * subgroup_max_size, uint32_t * wavefronts_per_simd, bool * bf16, bool * integer_dot_product, const char ** arch_name, uint32_t * coopmat_m, uint32_t * coopmat_n, uint32_t * coopmat_k, bool * supports_256_push_constants, uint32_t * max_push_constants_size);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
