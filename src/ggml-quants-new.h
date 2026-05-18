#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header: Q1_0 and NVFP4 quantization (new in ggml-0.11.0)

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
GGML_API void quantize_row_q1_0_ref (const float * GGML_RESTRICT x, block_q1_0  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_q1_0 (const block_q1_0  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Row-wise quantize (with optional imatrix; both currently ignore it)
GGML_API size_t quantize_q1_0 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_nvfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

#ifdef __cplusplus
}
#endif
