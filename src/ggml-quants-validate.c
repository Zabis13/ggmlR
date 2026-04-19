#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include <math.h>
#include <float.h>
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu.h"

#define UNUSED GGML_UNUSED

// =============================== data validation

static bool validate_float(float f, size_t i) {
    if (isinf(f)) {
        fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i);
        return false;
    }

    if (isnan(f)) {
        fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i);
        return false;
    }

    return true;
}

static bool isinf_fp16(ggml_fp16_t f) {
    return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) == 0;
}

static bool isnan_fp16(ggml_fp16_t f) {
    return (f & 0x7c00) == 0x7c00 && (f & 0x03ff) != 0;
}

static bool validate_fp16(ggml_fp16_t f, size_t i) {
    if (isinf_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found inf value at block %zu\n", i);
        return false;
    }

    if (isnan_fp16(f)) {
        fprintf(stderr, "ggml_validate_row_data: found nan value at block %zu\n", i);
        return false;
    }

    return true;
}

static bool validate_e_e8m0(uint8_t e, size_t i) {
    if (e == 0xff) {
        fprintf(stderr, "ggml_validate_row_data: found invalid e value %d at block %zu\n", e, i);
        return false;
    }

    return true;
}

#define VALIDATE_ROW_DATA_D_F16_IMPL(type, data, nb) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_fp16(q[i].d, i)) { \
            return false; \
        } \
    }

#define VALIDATE_ROW_DATA_DM_F16_IMPL(type, data, nb, d, m) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_fp16(q[i].d, i) || !validate_fp16(q[i].m, i)) { \
            return false; \
        } \
    }

#define VALIDATE_ROW_DATA_E_E8M0_IMPL(type, data, nb) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        if (!validate_e_e8m0(q[i].e, i)) { \
            return false; \
        } \
    }

#define VALIDATE_ROW_DATA_DVEC_F16_IMPL(type, data, nb, nr) \
    const type * q = (const type *) (data); \
    for (size_t i = 0; i < (nb); ++i) { \
        for (size_t j = 0; j < (nr); ++j) { \
            if (!validate_fp16(q[i].d[j], i)) { \
                return false; \
            } \
        } \
    }

bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid type %d\n", __func__, type);
        return false;
    }

    if (nbytes % ggml_type_size(type) != 0) {
        fprintf(stderr, "%s: invalid size %zu for type %s (type size = %zu)\n", __func__, nbytes, ggml_type_name(type), ggml_type_size(type));
        return false;
    }

    const size_t nb = nbytes/ggml_type_size(type);

    switch (type) {
        case GGML_TYPE_BF16:
            {
                int nans = 0;
                int infs = 0;
                const unsigned short * f = (const unsigned short *) data;
                for (size_t i = 0; i < nb; ++i) {
                    nans += (f[i] & 0x7fff) > 0x7f80;
                    infs += (f[i] & 0x7fff) == 0x7f80;
                }
                if (nans) {
                    fprintf(stderr, "%s: found %d NaNs in row of %zu BF16 values\n", __func__, nans, nb);
                    return false;
                }
                if (infs) {
                    fprintf(stderr, "%s: found %d infinities in row of %zu BF16 values\n", __func__, infs, nb);
                    return false;
                }
            } break;
        case GGML_TYPE_F16:
            {
                const ggml_fp16_t * f = (const ggml_fp16_t *) data;
                size_t i = 0;
#if defined(__AVX2__)
                for (; i + 15 < nb; i += 16) {
                    __m256i v = _mm256_loadu_si256((const __m256i *)(f + i));
                    __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi16(0x7c00));
                    __m256i cmp = _mm256_cmpeq_epi16(vexp, _mm256_set1_epi16(0x7c00));
                    int mask = _mm256_movemask_epi8(cmp);
                    if (mask) {
                        for (size_t j = 0; j < 16; ++j) {
                            if (!validate_fp16(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#elif defined(__ARM_NEON)
                for (; i + 7 < nb; i += 8) {
                    uint16x8_t v = vld1q_u16(f + i);
                    uint16x8_t vexp = vandq_u16(v, vdupq_n_u16(0x7c00));
                    uint16x8_t cmp = vceqq_u16(vexp, vdupq_n_u16(0x7c00));
                    uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(cmp, 4)), 0);
                    if (mask) {
                        for (size_t j = 0; j < 8; ++j) {
                            if (!validate_fp16(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#endif
                for (; i < nb; ++i) {
                    if (!validate_fp16(f[i], i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_F32:
            {
                const float * f = (const float *) data;
                size_t i = 0;
#if defined(__AVX2__)
                for (; i + 7 < nb; i += 8) {
                    __m256i v = _mm256_loadu_si256((const __m256i *)(f + i));
                    __m256i vexp = _mm256_and_si256(v, _mm256_set1_epi32(0x7f800000));
                    __m256i cmp = _mm256_cmpeq_epi32(vexp, _mm256_set1_epi32(0x7f800000));
                    int mask = _mm256_movemask_epi8(cmp);
                    if (mask) {
                        for (size_t j = 0; j < 8; ++j) {
                            if (!validate_float(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#elif defined(__ARM_NEON)
                for (; i + 3 < nb; i += 4) {
                    uint32x4_t v = vld1q_u32((const uint32_t *)f + i);
                    uint32x4_t vexp = vandq_u32(v, vdupq_n_u32(0x7f800000));
                    uint32x4_t cmp = vceqq_u32(vexp, vdupq_n_u32(0x7f800000));
                    uint64_t mask = vget_lane_u64(vreinterpret_u64_u16(vshrn_n_u32(cmp, 8)), 0);
                    if (mask) {
                        for (size_t j = 0; j < 4; ++j) {
                            if (!validate_float(f[i + j], i + j)) {
                                return false;
                            }
                        }
                        GGML_UNREACHABLE();
                    }
                }
#endif
                for (; i < nb; ++i) {
                    if (!validate_float(f[i], i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_F64:
            {
                const double * f = (const double *) data;
                for (size_t i = 0; i < nb; ++i) {
                    if (!validate_float(f[i], i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_Q4_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q4_0, data, nb);
            } break;
        case GGML_TYPE_Q4_1:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_1, data, nb, d, m);
            } break;
        case GGML_TYPE_Q5_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q5_0, data, nb);
            } break;
        case GGML_TYPE_Q5_1:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_1, data, nb, d, m);
            } break;
        case GGML_TYPE_Q8_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q8_0, data, nb);
            } break;
        case GGML_TYPE_MXFP4:
            {
                VALIDATE_ROW_DATA_E_E8M0_IMPL(block_mxfp4, data, nb);
            } break;
        case GGML_TYPE_Q2_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q2_K, data, nb, d, dmin);
            } break;
        case GGML_TYPE_Q3_K:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q3_K, data, nb);
            } break;
        case GGML_TYPE_Q4_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q4_K, data, nb, d, dmin);
            } break;
        case GGML_TYPE_Q5_K:
            {
                VALIDATE_ROW_DATA_DM_F16_IMPL(block_q5_K, data, nb, d, dmin);
            } break;
        case GGML_TYPE_Q6_K:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_q6_K, data, nb);
            } break;
        case GGML_TYPE_Q8_K:
            {
                const block_q8_K * q = (const block_q8_K *) data;
                for (size_t i = 0; i < nb; ++i) {
                    if (!validate_float(q[i].d, i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_TQ1_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq1_0, data, nb);
            } break;
        case GGML_TYPE_TQ2_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_0, data, nb);
            } break;
        case GGML_TYPE_IQ1_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq1_s, data, nb);
            } break;
        case GGML_TYPE_IQ1_M:
            {
                const block_iq1_m * q = (const block_iq1_m *) data;
                for (size_t i = 0; i < nb; ++i) {
                    iq1m_scale_t scale;
                    const uint16_t * sc = (const uint16_t *)q[i].scales;
                    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                    if (!validate_fp16(scale.f16, i)) {
                        return false;
                    }
                }
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xxs, data, nb);
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_xs, data, nb);
            } break;
        case GGML_TYPE_IQ2_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq2_s, data, nb);
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_xxs, data, nb);
            } break;

        case GGML_TYPE_IQ3_S:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq3_s, data, nb);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_xs, data, nb);
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_iq4_nl, data, nb);
            } break;

        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
            // nothing to validate
            break;
        default:
            {
                fprintf(stderr, "%s: invalid type %d\n", __func__, type);
                return false;
            }
    }

    return true;
}
