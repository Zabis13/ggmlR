
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id = 0) const uint32_t WorkGroupSize = 128;
layout (constant_id = 1) const uint32_t Br = 1;
layout (constant_id = 2) const uint32_t Bc = 32;
layout (constant_id = 3) const uint32_t HSK = 32;
layout (constant_id = 4) const uint32_t HSV = 32;
layout (constant_id = 5) const uint32_t Clamp = 0;
layout (constant_id = 6) const uint32_t D_split = 16;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define MASK_ENABLE_BIT (1<<16)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1
#if defined(DATA_A_F32)
layout (binding = 1) readonly buffer K_PACKED {vec4 k_data_packed[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED {vec4 v_data_packed[];} v_packed;
#elif defined(A_TYPE_PACKED16)
layout (binding = 1) readonly buffer K_PACKED16 {A_TYPE_PACKED16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {A_TYPE_PACKED16 v_data_packed16[];} v_packed;
#endif

#if defined(DATA_A_F32)
#undef BLOCK_SIZE
#define BLOCK_SIZE 4
#define BLOCK_BYTE_SIZE 16

vec4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // iqs is currently always zero in the flash attention shaders
    if (binding_idx == BINDING_IDX_K) {
        return k_packed.k_data_packed[a_offset + ib];
    } else {
        return v_packed.v_data_packed[a_offset + ib];
    }
}
#endif

#if defined(DATA_A_Q4_0)
#define BLOCK_BYTE_SIZE 18

vec4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return float(k_packed.k_data_packed16[a_offset + ib].d) * (vec4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - 8.0f);
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return float(v_packed.v_data_packed16[a_offset + ib].d) * (vec4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF) - 8.0f);
    }
}
#endif

#if defined(DATA_A_Q8_0)
#define BLOCK_BYTE_SIZE 34
vec4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        const i8vec2 v0 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return float(k_packed.k_data_packed16[a_offset + ib].d) * vec4(v0.x, v0.y, v1.x, v1.y);
    } else {
        const i8vec2 v0 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return float(v_packed.v_data_packed16[a_offset + ib].d) * vec4(v0.x, v0.y, v1.x, v1.y);
    }
}
#endif

#if defined(DATA_A_Q4_K)
#define BLOCK_BYTE_SIZE 144

// Dequantize 4 consecutive elements from a Q4_K block.
// iqs: element index within block (multiple of 4, range [0, 252]).
//
// Q4_K layout (256 elements, qs[128 uint8]):
//   8 sub-blocks of 32 elements with 8 6-bit (sc, min) pairs in scales[12 uint8].
//   The key: each 64-element group shares 32 uint8 bytes of qs[].
//     Elements group*64+0  .. group*64+31  → lower nibbles of qs[group*32 .. group*32+31]
//     Elements group*64+32 .. group*64+63  → upper nibbles of qs[group*32 .. group*32+31]
//   Sub-block index: is = iqs/32 (0..7), maps to (group=is/2, half=is%2).
//   4 consecutive elements at iqs all share the same group and half (since iqs is mult of 4).
//   Their uint8 byte indices: group*32 + (iqs%32), +1, +2, +3
//   Nibble shift: half==0 → 0, half==1 → 4
vec4 dequantize4_q4k(A_TYPE_PACKED16 blk, uint iqs) {
    const uint is    = iqs / 32u;        // sub-block index 0..7
    const uint group = is / 2u;          // 64-element group 0..3
    const uint nibhalf = is % 2u;        // 0 = lower nibbles, 1 = upper nibbles
    const uint shift = nibhalf * 4u;     // 0 or 4

    // Position within the 32-byte qs region for this group
    const uint pos_in_group = iqs % 32u;  // 0..28 (multiple of 4)

    // --- Reconstruct 6-bit scale and 6-bit min for sub-block 'is' ---
    // (same bit-field layout as dequant_funcs.glsl DATA_A_Q4_K)
    // In packed16 view: scales_u8[i] = (blk.scales[i/2] >> ((i%2)*8)) & 0xFF
    const uint scidx0  = (is < 4u) ? is       : (is + 4u);
    const uint scidx1  = (is < 4u) ? is       : (is - 4u);
    const uint scmask1 = (is < 4u) ? 0x30u    : 0xC0u;
    const uint scshift1= (is < 4u) ? 0u       : 2u;
    const uint mbidx0  = is + 4u;
    const uint mbidx1  = (is < 4u) ? (is + 4u): is;
    const uint mbmask0 = (is < 4u) ? 0x0Fu    : 0xF0u;
    const uint mbshift0= (is < 4u) ? 0u       : 4u;
    const uint mbmask1 = (is < 4u) ? 0x30u    : 0xC0u;
    const uint mbshift1= (is < 4u) ? 0u       : 2u;

    #define SC_U8(i) ((uint(blk.scales[(i)/2u]) >> (((i) % 2u) * 8u)) & 0xFFu)
    const uint sc   = (SC_U8(scidx0) & 0xFu) | ((SC_U8(scidx1) & scmask1) >> scshift1);
    const uint mval = ((SC_U8(mbidx0) & mbmask0) >> mbshift0) | ((SC_U8(mbidx1) & mbmask1) >> mbshift1);
    #undef SC_U8

    const vec2 dm = vec2(blk.dm);
    const float d = dm.x * float(sc);
    const float m = -dm.y * float(mval);

    // --- Decode 4 nibble values from qs ---
    // uint8 indices of the 4 bytes: base, base+1, base+2, base+3
    // where base = group*32 + pos_in_half
    // In packed16 view: blk.qs[u16_idx] holds qs_u8[2*u16_idx] in bits 0..7
    //                                    and qs_u8[2*u16_idx+1] in bits 8..15
    const uint base_u8 = group * 32u + pos_in_group;
    // base_u8 is always even (group*32 is mult of 32, pos_in_group is mult of 4)
    // so blk.qs[base_u8/2] holds bytes base_u8 and base_u8+1
    //    blk.qs[base_u8/2+1] holds bytes base_u8+2 and base_u8+3
    const uint w0 = uint(blk.qs[base_u8 / 2u]);
    const uint w1 = uint(blk.qs[base_u8 / 2u + 1u]);

    const float e0 = float((w0        >> shift) & 0xFu);   // byte base_u8+0
    const float e1 = float((w0 >> 8u  >> shift) & 0xFu);   // byte base_u8+1
    const float e2 = float((w1        >> shift) & 0xFu);   // byte base_u8+2
    const float e3 = float((w1 >> 8u  >> shift) & 0xFu);   // byte base_u8+3

    return vec4(fma(d, e0, m), fma(d, e1, m), fma(d, e2, m), fma(d, e3, m));
}

vec4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        return dequantize4_q4k(k_packed.k_data_packed16[a_offset + ib], iqs);
    } else {
        return dequantize4_q4k(v_packed.v_data_packed16[a_offset + ib], iqs);
    }
}
#endif

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    i = gl_WorkGroupID.x;
    split_k_index = 0;

    if (p.k_num > 1) {
        i = 0;
        split_k_index = gl_WorkGroupID.x;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}
