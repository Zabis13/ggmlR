#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

layout (push_constant) uniform parameter
{
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint ne04;
    uint nb00; uint nb01; uint nb02; uint nb03; uint nb04;
    uint ne10; uint ne11; uint ne12; uint ne13; uint ne14;
    uint nb10; uint nb11; uint nb12; uint nb13; uint nb14;
    uint misalign_offsets;
    float param1; float param2;

    uint ne0_0123mp; uint ne0_0123L;
    uint ne0_012mp;  uint ne0_012L;
    uint ne0_01mp;   uint ne0_01L;
    uint ne0_0mp;    uint ne0_0L;
    uint ne1_0123mp; uint ne1_0123L;
    uint ne1_012mp;  uint ne1_012L;
    uint ne1_01mp;   uint ne1_01L;
    uint ne1_0mp;    uint ne1_0L;
} p;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif

layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

uint get_idx() {
    return gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
}

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_doffset() { return p.misalign_offsets & 0xFFFF; }

// see init_fastdiv_values in ggml-vulkan.cpp
uint fastdiv(uint n, uint mp, uint L) {
    uint msbs, lsbs;
    // msbs = mulhi(n, mp)
    umulExtended(n, mp, msbs, lsbs);
    return (msbs + n) >> L;
}

uint src0_idx(uint idx) {
    const uint i04 = fastdiv(idx, p.ne0_0123mp, p.ne0_0123L);
    const uint rem4 = idx - i04 * p.ne03*p.ne02*p.ne01*p.ne00;
    const uint i03 = fastdiv(rem4, p.ne0_012mp, p.ne0_012L);
    const uint rem3 = rem4 - i03 * p.ne02*p.ne01*p.ne00;
    const uint i02 = fastdiv(rem3, p.ne0_01mp, p.ne0_01L);
    const uint rem2 = rem3 - i02*p.ne01*p.ne00;
    const uint i01 = fastdiv(rem2, p.ne0_0mp, p.ne0_0L);
    const uint i00 = rem2 - i01*p.ne00;
    return i04*p.nb04 + i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + i00*p.nb00;
}

uint dst_idx(uint idx) {
    const uint i14 = fastdiv(idx, p.ne1_0123mp, p.ne1_0123L);
    const uint rem4 = idx - i14 * p.ne13*p.ne12*p.ne11*p.ne10;
    const uint i13 = fastdiv(rem4, p.ne1_012mp, p.ne1_012L);
    const uint rem3 = rem4 - i13 * p.ne12*p.ne11*p.ne10;
    const uint i12 = fastdiv(rem3, p.ne1_01mp, p.ne1_01L);
    const uint rem2 = rem3 - i12*p.ne11*p.ne10;
    const uint i11 = fastdiv(rem2, p.ne1_0mp, p.ne1_0L);
    const uint i10 = rem2 - i11*p.ne10;
    return i14*p.nb14 + i13*p.nb13 + i12*p.nb12 + i11*p.nb11 + i10*p.nb10;
}

uint src0_idx_quant(uint idx, uint qk) {
    const uint i04 = fastdiv(idx, p.ne0_0123mp, p.ne0_0123L);
    const uint rem4 = idx - i04 * p.ne03*p.ne02*p.ne01*p.ne00;
    const uint i03 = fastdiv(rem4, p.ne0_012mp, p.ne0_012L);
    const uint rem3 = rem4 - i03 * p.ne02*p.ne01*p.ne00;
    const uint i02 = fastdiv(rem3, p.ne0_01mp, p.ne0_01L);
    const uint rem2 = rem3 - i02*p.ne01*p.ne00;
    const uint i01 = fastdiv(rem2, p.ne0_0mp, p.ne0_0L);
    const uint i00 = rem2 - i01*p.ne00;
    return i04*p.nb04 + i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + (i00/qk)*p.nb00;
}

uint dst_idx_quant(uint idx, uint qk) {
    const uint i14 = fastdiv(idx, p.ne1_0123mp, p.ne1_0123L);
    const uint rem4 = idx - i14 * p.ne13*p.ne12*p.ne11*p.ne10;
    const uint i13 = fastdiv(rem4, p.ne1_012mp, p.ne1_012L);
    const uint rem3 = rem4 - i13 * p.ne12*p.ne11*p.ne10;
    const uint i12 = fastdiv(rem3, p.ne1_01mp, p.ne1_01L);
    const uint rem2 = rem3 - i12*p.ne11*p.ne10;
    const uint i11 = fastdiv(rem2, p.ne1_0mp, p.ne1_0L);
    const uint i10 = rem2 - i11*p.ne10;
    return i14*p.nb14 + i13*p.nb13 + i12*p.nb12 + i11*p.nb11 + (i10/qk)*p.nb10;
}
