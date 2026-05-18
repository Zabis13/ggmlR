// Q1_0 and NVFP4 quantization (new types added in ggml-0.11.0)
// Kept in a separate translation unit so we don't disturb ggml-quants-legacy.c.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants-new.h"
#include "ggml-quants.h"        // brings in GGML_RESTRICT etc.
#include "ggml-impl.h"          // ggml_ue4m3_to_fp32, ggml_fp32_to_ue4m3
#include "ggml-quants-helpers.h" // best_index_mxfp4
#include <assert.h>
#include <math.h>
#include <string.h>

// ===== Q1_0 =====

void quantize_row_q1_0_ref(const float * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK1_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float sum_abs = 0.0f;
        for (int j = 0; j < qk; j++) {
            sum_abs += fabsf(x[i*qk + j]);
        }
        const float d = sum_abs / qk;

        y[i].d = GGML_FP32_TO_FP16(d);

        // Clear all bits first
        for (int j = 0; j < qk / 8; ++j) {
            y[i].qs[j] = 0;
        }

        // Just store sign of each weight directly (no normalization)
        for (int j = 0; j < qk; ++j) {
            const int bit_index = j;
            const int byte_index = bit_index / 8;
            const int bit_offset = bit_index % 8;

            if (x[i*qk + j] >= 0.0f) {
                y[i].qs[byte_index] |= (1 << bit_offset);
            }
        }
    }
}

void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK1_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float neg_d = -d;

        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 8;
            const int bit_offset = j % 8;
            const uint8_t bit = (x[i].qs[byte_index] >> bit_offset) & 1;
            y[i*qk + j] = bit ? d : neg_d;
        }
    }
}

size_t quantize_q1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    if (!quant_weights) {
        quantize_row_q1_0_ref(src, dst, (int64_t)nrow*n_per_row);
        return nrow * ggml_row_size(GGML_TYPE_Q1_0, n_per_row);
    }
    size_t row_size = ggml_row_size(GGML_TYPE_Q1_0, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_q1_0_ref(src, (block_q1_0*)qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

// ===== NVFP4 =====

void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float * xb = x + i*qk + s*qk_sub;

            float amax = 0.0f;
            for (int j = 0; j < qk_sub; j++) {
                if (amax < fabsf(xb[j])) {
                    amax = fabsf(xb[j]);
                }
            }

            // UE4M3 scale: amax / 6.0 maps the max E2M1 value (6.0) to amax
            const uint8_t ue = ggml_fp32_to_ue4m3(amax / 6.0f);
            y[i].d[s] = ue;
            const float d = ggml_ue4m3_to_fp32(ue);

            for (int j = 0; j < qk_sub/2; ++j) {
                const uint8_t x0 = best_index_mxfp4(xb[0        + j], d);
                const uint8_t x1 = best_index_mxfp4(xb[qk_sub/2 + j], d);

                y[i].qs[s*(qk_sub/2) + j] = x0 | (x1 << 4);
            }
        }
    }
}

void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float d = ggml_ue4m3_to_fp32(x[i].d[s]);
            float * yb = y + i*qk + s*qk_sub;

            for (int j = 0; j < qk_sub/2; ++j) {
                const int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
                const int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];

                yb[j + 0       ] = v0*d;
                yb[j + qk_sub/2] = v1*d;
            }
        }
    }
}

size_t quantize_nvfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    quantize_row_nvfp4_ref(src, dst, (int64_t)nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_NVFP4, n_per_row);
}
