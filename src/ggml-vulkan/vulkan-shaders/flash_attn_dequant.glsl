// Asymmetric K/V flash attention: aliased SSBO views of bindings 1 (K) and 2 (V)
// covering every supported FA element type, plus an uber dequantize4() that
// switches on FaTypeK / FaTypeV. After spec-constant specialization the driver
// folds away every path except the one matching the K/V type for this pipeline.
//
// Included by flash_attn.comp and flash_attn_cm1.comp. Not included by
// flash_attn_cm2.comp, which has its own buffer_reference-based decode path.
//
// We use macros (rather than per-quant decode functions taking a struct) on
// purpose: the FA shaders don't enable GL_EXT_shader_explicit_arithmetic_types_float16
// when FLOAT16 isn't defined, which makes float16-containing struct values
// illegal to return from / pass to functions. Macros expand inline where the
// float16 stays in storage and is converted to FLOAT_TYPE at use.

// F32 is fed as a vec4 "block" (4 floats), matching what dequant_funcs_cm2.glsl
// does for F32 in the cm2 shader. FaBlockBytesK/V == 16 for F32.
layout (binding = 1) readonly buffer K_PACKED_F32  { vec4 data[]; }                k_packed_f32;
layout (binding = 2) readonly buffer V_PACKED_F32  { vec4 data[]; }                v_packed_f32;

layout (binding = 1) readonly buffer K_PACKED_Q4_0 { block_q4_0_packed16 data[]; } k_packed_q4_0;
layout (binding = 2) readonly buffer V_PACKED_Q4_0 { block_q4_0_packed16 data[]; } v_packed_q4_0;
layout (binding = 1) readonly buffer K_PACKED_Q4_1 { block_q4_1_packed16 data[]; } k_packed_q4_1;
layout (binding = 2) readonly buffer V_PACKED_Q4_1 { block_q4_1_packed16 data[]; } v_packed_q4_1;
layout (binding = 1) readonly buffer K_PACKED_Q5_0 { block_q5_0_packed16 data[]; } k_packed_q5_0;
layout (binding = 2) readonly buffer V_PACKED_Q5_0 { block_q5_0_packed16 data[]; } v_packed_q5_0;
layout (binding = 1) readonly buffer K_PACKED_Q5_1 { block_q5_1_packed16 data[]; } k_packed_q5_1;
layout (binding = 2) readonly buffer V_PACKED_Q5_1 { block_q5_1_packed16 data[]; } v_packed_q5_1;
layout (binding = 1) readonly buffer K_PACKED_Q8_0 { block_q8_0_packed16 data[]; } k_packed_q8_0;
layout (binding = 2) readonly buffer V_PACKED_Q8_0 { block_q8_0_packed16 data[]; } v_packed_q8_0;
layout (binding = 1) readonly buffer K_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } k_packed_iq4_nl;
layout (binding = 2) readonly buffer V_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } v_packed_iq4_nl;
// ggmlR: Q4_K K/V (not upstream).
layout (binding = 1) readonly buffer K_PACKED_Q4_K { block_q4_K_packed16 data[]; } k_packed_q4_k;
layout (binding = 2) readonly buffer V_PACKED_Q4_K { block_q4_K_packed16 data[]; } v_packed_q4_k;

layout (binding = 1) readonly buffer K_PACKED_BF16 { u16vec4 data[]; } k_packed_bf16;
layout (binding = 2) readonly buffer V_PACKED_BF16 { u16vec4 data[]; } v_packed_bf16;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

// Per-quant decode bodies are expanded once for the K view set and once for
// the V view set. The macros take the buffer name as a parameter.
#define FA_DEQUANT4_F32(BUF) \
    return FLOAT_TYPEV4(BUF.data[a_offset + ib]);

#define FA_DEQUANT4_Q4_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));                  \
}

#define FA_DEQUANT4_Q4_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * nibbles                                        \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q5_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = uint(BUF.data[a_offset + ib].qh[0])                                                 \
            | (uint(BUF.data[a_offset + ib].qh[1]) << 16);                                        \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));            \
}

#define FA_DEQUANT4_Q5_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = BUF.data[a_offset + ib].qh;                                                         \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb)                                 \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q8_0(BUF) {                                                                   \
    const i8vec2 v0 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2    ])).xy;               \
    const i8vec2 v1 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2 + 1])).xy;               \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);          \
}

#define FA_DEQUANT4_IQ4_NL(BUF) {                                                                 \
    const uint shift = (iqs & 0x10) >> 2;                                                         \
    const uint qs_i = (iqs & 0xC) >> 1;                                                           \
    const uint qsw = uint(BUF.data[a_offset + ib].qs[qs_i])                                       \
                   | (uint(BUF.data[a_offset + ib].qs[qs_i + 1u]) << 16);                         \
    const FLOAT_TYPE d = FLOAT_TYPE(BUF.data[a_offset + ib].d);                                   \
    const u8vec4 q = unpack8((qsw >> shift) & 0x0F0F0F0Fu);                                       \
    return d * FLOAT_TYPEV4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],                               \
                            kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);                              \
}

#define FA_DEQUANT4_BF16(BUF) \
    return FLOAT_TYPEV4(bf16_to_fp32(uvec4(BUF.data[(a_offset + ib) / 4])));

// ggmlR: Q4_K (not upstream). Same bit layout as dequant_funcs.glsl DATA_A_Q4_K:
// 8 sub-blocks of 32 with 6-bit (scale, min) pairs; each 64-element group shares
// 32 qs bytes, lower nibbles for the first 32 elements and upper for the next 32.
// The 4 elements at iqs (a multiple of 4) share one sub-block, so one (d, m) pair.
#define FA_Q4K_SC_U8(BUF, i) ((uint(BUF.data[a_offset + ib].scales[(i) / 2u]) >> (((i) % 2u) * 8u)) & 0xFFu)
#define FA_DEQUANT4_Q4_K(BUF) {                                                                   \
    const uint is       = iqs / 32u;                                                              \
    const uint shift    = (is % 2u) * 4u;                                                         \
    const uint scidx0   = (is < 4u) ? is    : (is + 4u);                                          \
    const uint scidx1   = (is < 4u) ? is    : (is - 4u);                                          \
    const uint scmask1  = (is < 4u) ? 0x30u : 0xC0u;                                              \
    const uint scshift1 = (is < 4u) ? 0u    : 2u;                                                 \
    const uint mbidx0   = is + 4u;                                                                \
    const uint mbidx1   = (is < 4u) ? (is + 4u) : is;                                             \
    const uint mbmask0  = (is < 4u) ? 0x0Fu : 0xF0u;                                              \
    const uint mbshift0 = (is < 4u) ? 0u    : 4u;                                                 \
    const uint mbmask1  = (is < 4u) ? 0x30u : 0xC0u;                                              \
    const uint mbshift1 = (is < 4u) ? 0u    : 2u;                                                 \
    const uint sc   = (FA_Q4K_SC_U8(BUF, scidx0) & 0xFu) |                                        \
                      ((FA_Q4K_SC_U8(BUF, scidx1) & scmask1) >> scshift1);                        \
    const uint mval = ((FA_Q4K_SC_U8(BUF, mbidx0) & mbmask0) >> mbshift0) |                       \
                      ((FA_Q4K_SC_U8(BUF, mbidx1) & mbmask1) >> mbshift1);                        \
    const vec2 dm = vec2(BUF.data[a_offset + ib].dm);                                             \
    const FLOAT_TYPE d = FLOAT_TYPE(dm.x * float(sc));                                            \
    const FLOAT_TYPE m = FLOAT_TYPE(-dm.y * float(mval));                                         \
    const uint base_u8 = (is / 2u) * 32u + iqs % 32u;                                             \
    const uint w0 = uint(BUF.data[a_offset + ib].qs[base_u8 / 2u]);                               \
    const uint w1 = uint(BUF.data[a_offset + ib].qs[base_u8 / 2u + 1u]);                          \
    return FLOAT_TYPEV4(fma(d, FLOAT_TYPE( w0        >> shift & 0xFu), m),                        \
                        fma(d, FLOAT_TYPE((w0 >> 8u) >> shift & 0xFu), m),                        \
                        fma(d, FLOAT_TYPE( w1        >> shift & 0xFu), m),                        \
                        fma(d, FLOAT_TYPE((w1 >> 8u) >> shift & 0xFu), m));                       \
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        switch (FaTypeK) {
            case GGML_TYPE_F32:  FA_DEQUANT4_F32 (k_packed_f32)
            case GGML_TYPE_Q4_0: FA_DEQUANT4_Q4_0(k_packed_q4_0)
            case GGML_TYPE_Q4_1: FA_DEQUANT4_Q4_1(k_packed_q4_1)
            case GGML_TYPE_Q5_0: FA_DEQUANT4_Q5_0(k_packed_q5_0)
            case GGML_TYPE_Q5_1: FA_DEQUANT4_Q5_1(k_packed_q5_1)
            case GGML_TYPE_Q8_0: FA_DEQUANT4_Q8_0(k_packed_q8_0)
            case GGML_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(k_packed_iq4_nl)
            case GGML_TYPE_BF16: FA_DEQUANT4_BF16(k_packed_bf16)
            case GGML_TYPE_Q4_K: FA_DEQUANT4_Q4_K(k_packed_q4_k)
        }
    } else {
        switch (FaTypeV) {
            case GGML_TYPE_F32:  FA_DEQUANT4_F32 (v_packed_f32)
            case GGML_TYPE_Q4_0: FA_DEQUANT4_Q4_0(v_packed_q4_0)
            case GGML_TYPE_Q4_1: FA_DEQUANT4_Q4_1(v_packed_q4_1)
            case GGML_TYPE_Q5_0: FA_DEQUANT4_Q5_0(v_packed_q5_0)
            case GGML_TYPE_Q5_1: FA_DEQUANT4_Q5_1(v_packed_q5_1)
            case GGML_TYPE_Q8_0: FA_DEQUANT4_Q8_0(v_packed_q8_0)
            case GGML_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(v_packed_iq4_nl)
            case GGML_TYPE_BF16: FA_DEQUANT4_BF16(v_packed_bf16)
            case GGML_TYPE_Q4_K: FA_DEQUANT4_Q4_K(v_packed_q4_k)
        }
    }
    return FLOAT_TYPEV4(0);
}
