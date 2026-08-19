#include "ops.h"
#include <vector>

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "binary-ops.h"
#include "ggml.h"
#include "unary-ops.h"
#include "vec.h"

#include <cfloat>
#include <algorithm>
#include <cmath>
#include <functional>

// ggml_compute_forward_rwkv_wkv6

static void ggml_compute_forward_rwkv_wkv6_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t HEADS = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[5]->ne[1];
    const int64_t head_size = C / HEADS;

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    const int ith = params->ith;
    const int nth = params->nth;

    // DIVERGENCE from upstream: the zeroing and its barrier were placed AFTER
    // the `ith >= HEADS` early return below, so with more threads than heads
    // the surplus threads left without reaching ggml_barrier() while the rest
    // waited on them forever -- a hard deadlock, not a slowdown. The barrier is
    // collective, so every thread must reach it; the early return now happens
    // after it. Reproduced with HEADS=2 on 4 threads.
    if (ith == 0) {
        memset(dst_data, 0, T * C * sizeof(float));
    }
    ggml_barrier(params->threadpool);

    if (ith >= HEADS) {
        return;
    }

    // DIVERGENCE from upstream: the per-sequence length is derived as
    // T / n_seqs with an integer division that upstream never checks. An
    // indivisible pair (e.g. 3 tokens over 2 sequences) walks off the state
    // buffer and segfaults, and T < n_seqs divides by zero. Assert instead of
    // corrupting memory.
    GGML_ASSERT(n_seqs > 0 && "rwkv_wkv6: state must hold at least one sequence");
    GGML_ASSERT(T % n_seqs == 0 &&
        "rwkv_wkv6: n_tokens must be a multiple of n_seqs");

    // DIVERGENCE from upstream: the original range was
    //     h_start = HEADS*ith/nth, h_end = min(HEADS*(ith+1)/nth, HEADS)
    // which only covers every head when nth <= HEADS. With more threads than
    // heads it hands thread 0 the empty range [0,0) while every other thread
    // leaves through the `ith >= HEADS` check above, so NO thread computes
    // anything and the output stays zero -- silently wrong, neither a crash nor
    // a hang. Reproduced with HEADS=1 on 2 threads. A block split covers the
    // heads for any thread count.
    const int hpt = (HEADS + nth - 1) / nth;
    const int h_start = hpt * ith;
    const int h_end = (h_start + hpt < HEADS) ? h_start + hpt : HEADS;

    float * k =          (float *) dst->src[0]->data;
    float * v =          (float *) dst->src[1]->data;
    float * r =          (float *) dst->src[2]->data;
    float * time_faaaa = (float *) dst->src[3]->data;
    float * time_decay = (float *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;


    #if defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #elif defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #elif defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        int wkv_vector_size;
        #if defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #else
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        const int64_t vec_count = head_size / wkv_vector_size;

        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float r_val = r[t_h_i_offset];
                    float time_faaaa_val = time_faaaa[h_i_offset];
                    float time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    for (int64_t j = 0; j < vec_count; j++) {
                        size_t base_j = j * wkv_vector_size;
                        size_t t_h_j_offset = t_h_offset + base_j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X temp_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, time_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, r_vec);
                        GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, time_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    for (int64_t j = vec_count * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #else
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float r_val = r[t_h_i_offset];
                    float time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    float time_decay_val = time_decay[t_h_i_offset];

                    for (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


// ggml_compute_forward_rwkv_wkv6_back
//
// ggmlR extension: backward pass for ggml_rwkv_wkv6, absent upstream.
//
// Forward, per head and per state cell (i,j):
//     kv        = k[i]*v[j]
//     y[j]     += r[i]*(kv*tf[i] + S[i,j])
//     S[i,j]    = S[i,j]*td[i] + kv
//
// Walking back, each cell feeds the loss twice: through y at this token, and
// through the state carried into the next one. Both terms are summed into the
// gradient that flows to the previous token -- dropping either still trains,
// just towards the wrong optimum.
//
// The forward keeps only the final state, so this replays the recurrence into a
// scratch buffer and then walks it backwards. Gradients land in one packed
// output, in forward source order: [d_k | d_v | d_r | d_tf | d_td | d_state].
//
// Single-threaded on purpose: d_tf is shared by every token of a head and d_v
// is accumulated across the i loop, so splitting over heads alone would race.
// The early return sits AFTER the barrier -- the deadlock fixed in this same
// file came from having it before.
static void ggml_compute_forward_rwkv_wkv6_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src_k  = dst->src[0];
    const ggml_tensor * src_v  = dst->src[1];
    const ggml_tensor * src_r  = dst->src[2];
    const ggml_tensor * src_tf = dst->src[3];
    const ggml_tensor * src_td = dst->src[4];
    const ggml_tensor * src_s  = dst->src[5];
    const ggml_tensor * src_g  = dst->src[6];

    const int64_t S        = src_k->ne[0];
    const int64_t H        = src_k->ne[1];
    const int64_t T        = src_k->ne[2];
    const int64_t n_seqs   = src_s->ne[1];
    const int64_t C        = S * H;
    const int64_t seq_len  = T / n_seqs;

    GGML_ASSERT(n_seqs > 0 && "rwkv_wkv6_back: state must hold at least one sequence");
    GGML_ASSERT(T % n_seqs == 0 &&
        "rwkv_wkv6_back: n_tokens must be a multiple of n_seqs");

    const int ith = params->ith;

    if (ith == 0) {
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    if (ith != 0) {
        return;
    }

    const float * k  = (const float *) src_k->data;
    const float * v  = (const float *) src_v->data;
    const float * r  = (const float *) src_r->data;
    const float * tf = (const float *) src_tf->data;
    const float * td = (const float *) src_td->data;
    const float * s0 = (const float *) src_s->data;

    // The forward result is [S*H, T + S*n_seqs]: outputs, then the final state.
    const float * g_y = (const float *) src_g->data;
    const float * g_s = g_y + C * T;

    float * d_k  = (float *) dst->data;
    float * d_v  = d_k  + ggml_nelements(src_k);
    float * d_r  = d_v  + ggml_nelements(src_v);
    float * d_tf = d_r  + ggml_nelements(src_r);
    float * d_td = d_tf + ggml_nelements(src_tf);
    float * d_s  = d_td + ggml_nelements(src_td);

    // Scratch for the replayed states of one head: seq_len+1 snapshots.
    std::vector<float> St((size_t)(seq_len + 1) * S * S);
    std::vector<float> gs((size_t) S * S);

    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t h = 0; h < H; ++h) {
            const size_t sb = (size_t) seq * S * S * H + (size_t) h * S * S;

            // --- replay forward, keeping every state ---
            for (int64_t ij = 0; ij < S * S; ++ij) {
                St[ij] = s0[sb + ij];
            }
            for (int64_t t = 0; t < seq_len; ++t) {
                const int64_t tt = seq * seq_len + t;
                const float * prev = &St[(size_t) t * S * S];
                float * cur = &St[(size_t)(t + 1) * S * S];
                for (int64_t i = 0; i < S; ++i) {
                    const float kv_i = k[tt*C + h*S + i];
                    const float tdv  = td[tt*C + h*S + i];
                    for (int64_t j = 0; j < S; ++j) {
                        cur[i*S + j] = prev[i*S + j]*tdv + kv_i*v[tt*C + h*S + j];
                    }
                }
            }

            // --- walk back ---
            for (int64_t ij = 0; ij < S * S; ++ij) {
                gs[ij] = g_s[(size_t) seq * S * C + (size_t) h * S * S + ij];
            }

            for (int64_t t = seq_len - 1; t >= 0; --t) {
                const int64_t tt = seq * seq_len + t;
                const float * prev = &St[(size_t) t * S * S];

                for (int64_t i = 0; i < S; ++i) {
                    const float k_i  = k[tt*C + h*S + i];
                    const float r_i  = r[tt*C + h*S + i];
                    const float tf_i = tf[h*S + i];
                    const float td_i = td[tt*C + h*S + i];

                    float gk_i = 0.0f, gr_i = 0.0f, gtf_i = 0.0f, gtd_i = 0.0f;

                    for (int64_t j = 0; j < S; ++j) {
                        const float v_j  = v[tt*C + h*S + j];
                        const float kv   = k_i * v_j;
                        const float pv   = prev[i*S + j];
                        const float gy_j = g_y[tt*C + h*S + j];

                        // through y
                        gr_i  += gy_j * (kv*tf_i + pv);
                        gtf_i += gy_j * r_i * kv;
                        float gkv   = gy_j * r_i * tf_i;
                        float gprev = gy_j * r_i;

                        // through the carried state
                        const float gsn = gs[i*S + j];
                        gtd_i += gsn * pv;
                        gkv   += gsn;
                        gprev += gsn * td_i;

                        gk_i += gkv * v_j;
                        d_v[tt*C + h*S + j] += gkv * k_i;

                        gs[i*S + j] = gprev;
                    }

                    d_k [tt*C + h*S + i] += gk_i;
                    d_r [tt*C + h*S + i] += gr_i;
                    d_tf[h*S + i]        += gtf_i;
                    d_td[tt*C + h*S + i] += gtd_i;
                }
            }

            // What remains in gs is the gradient of the initial state.
            for (int64_t ij = 0; ij < S * S; ++ij) {
                d_s[sb + ij] = gs[ij];
            }
        }
    }
}

void ggml_compute_forward_rwkv_wkv6_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_compute_forward_rwkv_wkv6(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

static void ggml_compute_forward_gla_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t HEADS = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[4]->ne[1];
    const int64_t head_size = C / HEADS;
    const float scale = ggml_get_op_params_f32(dst, 0);

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    const int ith = params->ith;
    const int nth = params->nth;

    // DIVERGENCE from upstream: same two defects as ggml_compute_forward_rwkv_wkv6_f32
    // above -- the barrier sat after the `ith >= HEADS` early return (deadlock
    // whenever there are more threads than heads), and T / n_seqs was an
    // unchecked integer division (out-of-bounds state access when the two do
    // not divide). See the comments there.
    if (ith == 0) {
        memset(dst_data, 0, T * C * sizeof(float));
    }
    ggml_barrier(params->threadpool);

    if (ith >= HEADS) {
        return;
    }

    GGML_ASSERT(n_seqs > 0 && "gated_linear_attn: state must hold at least one sequence");
    GGML_ASSERT(T % n_seqs == 0 &&
        "gated_linear_attn: n_tokens must be a multiple of n_seqs");

    // DIVERGENCE from upstream: the original range was
    //     h_start = HEADS*ith/nth, h_end = min(HEADS*(ith+1)/nth, HEADS)
    // which only covers every head when nth <= HEADS. With more threads than
    // heads it hands thread 0 the empty range [0,0) while every other thread
    // leaves through the `ith >= HEADS` check above, so NO thread computes
    // anything and the output stays zero -- silently wrong, neither a crash nor
    // a hang. Reproduced with HEADS=1 on 2 threads. A block split covers the
    // heads for any thread count.
    const int hpt = (HEADS + nth - 1) / nth;
    const int h_start = hpt * ith;
    const int h_end = (h_start + hpt < HEADS) ? h_start + hpt : HEADS;

    float * k = (float *) dst->src[0]->data;
    float * v = (float *) dst->src[1]->data;
    float * q = (float *) dst->src[2]->data;
    float * g = (float *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;


    #if defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #elif defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #elif defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        int gla_vector_size;
        #if defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #else
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        const int64_t vec_count = head_size / gla_vector_size;

        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[4]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float q_val = q[t_h_i_offset] * scale;
                    float g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    for (int64_t j = 0; j < vec_count; j++) {
                        size_t base_j = j * gla_vector_size;
                        size_t t_h_j_offset = t_h_offset + base_j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X temp_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, q_vec);
                        GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], temp_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    for (int64_t j = vec_count * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #else
        for (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[4]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float k_val = k[t_h_i_offset];
                    float q_val = q[t_h_i_offset] * scale;
                    float g_val = g[t_h_i_offset];

                    for (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        float v_val = v[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        float temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


// ggml_compute_forward_gla_back
//
// ggmlR extension: backward pass for ggml_gated_linear_attn, absent upstream.
//
// Forward, per head and channel i:
//     S[i,j] = S[i,j]*g[i] + k[i]*v[j]
//     y[j]  += S[i,j]*(q[i]*scale)
//
// Note the order: the state is updated BEFORE it is read into y, so y sees the
// new state, not the previous one -- unlike wkv6, where y reads the carried
// state. Getting that backwards shifts every gradient by one token.
//
// Packed output, in forward source order: [d_k | d_v | d_q | d_g | d_state].
static void ggml_compute_forward_gla_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src_k = dst->src[0];
    const ggml_tensor * src_v = dst->src[1];
    const ggml_tensor * src_q = dst->src[2];
    const ggml_tensor * src_g = dst->src[3];
    const ggml_tensor * src_s = dst->src[4];
    const ggml_tensor * src_d = dst->src[5];

    const float scale = ggml_get_op_params_f32(dst, 0);

    const int64_t S       = src_k->ne[0];
    const int64_t H       = src_k->ne[1];
    const int64_t T       = src_k->ne[2];
    const int64_t n_seqs  = src_s->ne[1];
    const int64_t C       = S * H;
    const int64_t seq_len = T / n_seqs;

    GGML_ASSERT(n_seqs > 0 && "gated_linear_attn_back: state must hold at least one sequence");
    GGML_ASSERT(T % n_seqs == 0 &&
        "gated_linear_attn_back: n_tokens must be a multiple of n_seqs");

    const int ith = params->ith;
    if (ith == 0) {
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);
    if (ith != 0) {
        return;
    }

    const float * k  = (const float *) src_k->data;
    const float * v  = (const float *) src_v->data;
    const float * q  = (const float *) src_q->data;
    const float * g  = (const float *) src_g->data;
    const float * s0 = (const float *) src_s->data;

    const float * g_y = (const float *) src_d->data;
    const float * g_s = g_y + C * T;

    float * d_k = (float *) dst->data;
    float * d_v = d_k + ggml_nelements(src_k);
    float * d_q = d_v + ggml_nelements(src_v);
    float * d_g = d_q + ggml_nelements(src_q);
    float * d_s = d_g + ggml_nelements(src_g);

    std::vector<float> St((size_t)(seq_len + 1) * S * S);
    std::vector<float> gs((size_t) S * S);

    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t h = 0; h < H; ++h) {
            const size_t sb = (size_t) seq * S * S * H + (size_t) h * S * S;

            for (int64_t ij = 0; ij < S * S; ++ij) {
                St[ij] = s0[sb + ij];
            }
            for (int64_t t = 0; t < seq_len; ++t) {
                const int64_t tt = seq * seq_len + t;
                const float * prev = &St[(size_t) t * S * S];
                float * cur = &St[(size_t)(t + 1) * S * S];
                for (int64_t i = 0; i < S; ++i) {
                    const float g_i = g[tt*C + h*S + i];
                    const float k_i = k[tt*C + h*S + i];
                    for (int64_t j = 0; j < S; ++j) {
                        cur[i*S + j] = prev[i*S + j]*g_i + k_i*v[tt*C + h*S + j];
                    }
                }
            }

            for (int64_t ij = 0; ij < S * S; ++ij) {
                gs[ij] = g_s[(size_t) seq * S * C + (size_t) h * S * S + ij];
            }

            for (int64_t t = seq_len - 1; t >= 0; --t) {
                const int64_t tt = seq * seq_len + t;
                const float * prev = &St[(size_t) t * S * S];
                const float * cur  = &St[(size_t)(t + 1) * S * S];

                for (int64_t i = 0; i < S; ++i) {
                    const float g_i = g[tt*C + h*S + i];
                    const float k_i = k[tt*C + h*S + i];
                    const float qv  = q[tt*C + h*S + i] * scale;
                    float gg_i = 0.0f, gk_i = 0.0f;

                    for (int64_t j = 0; j < S; ++j) {
                        const float gy_j = g_y[tt*C + h*S + j];

                        // y reads the NEW state, so d_q uses cur, not prev.
                        d_q[tt*C + h*S + i] += gy_j * cur[i*S + j] * scale;

                        const float g_cur = gs[i*S + j] + gy_j * qv;

                        gg_i += g_cur * prev[i*S + j];
                        gk_i += g_cur * v[tt*C + h*S + j];
                        d_v[tt*C + h*S + j] += g_cur * k_i;

                        gs[i*S + j] = g_cur * g_i;
                    }

                    d_g[tt*C + h*S + i] += gg_i;
                    d_k[tt*C + h*S + i] += gk_i;
                }
            }

            for (int64_t ij = 0; ij < S * S; ++ij) {
                d_s[sb + ij] = gs[ij];
            }
        }
    }
}

void ggml_compute_forward_gla_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_compute_forward_gla(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_solve_tri_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    const struct ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t k = ne10;   // number of RHS columns
    const int64_t n = ne11;   // A is n×n
    const int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    const float * A = (const float *) src0->data;  // [n, n, B1, B2]
    const float * B = (const float *) src1->data;  // [n, k, B1, B2]
          float * X = (      float *) dst->data;   // [n, k, B1, B2]

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*k);
        const int64_t i02 = (ir - i03*ne02*k)/k;
        const int64_t i01 = (ir - i03*ne02*k - i02*k);

        const float * A_batch = A + i02 * nb02 / sizeof(float) + i03 * nb03 / sizeof(float);
        const float * B_batch = B + i02 * nb12 / sizeof(float) + i03 * nb13 / sizeof(float);

        float * X_batch = X + i02 * nb2 / sizeof(float) + i03 * nb3 / sizeof(float);

        for (int64_t i00 = 0; i00 < n; ++i00) {
            float sum = 0.0f;
            for (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            const float diag = A_batch[i00 * n + i00];
            assert(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

void ggml_compute_forward_solve_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net

static void ggml_compute_forward_gated_delta_net_one_chunk(
    const ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    GGML_ASSERT(src_g->ne[0] == 1 || src_g->ne[0] == S_v);
    GGML_ASSERT(src_beta->ne[0] == 1);

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(int64_t, neg, src_g, ne);
    GGML_TENSOR_LOCALS(size_t,  nbg, src_g, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const bool kda = (neg0 == S_v);

    // scratch layout per thread: [delta(S_v)]
    const int64_t scratch_per_thread = S_v;
    const int ith = params->ith;

    float * delta = (float *)params->wdata + ith * scratch_per_thread + CACHE_LINE_SIZE_F32;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs floats
    // new_states:  S_v * S_v * H * n_seqs floats
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float * attn_out_base  = (float *)dst->data;
    float * state_out_base = (float *)dst->data + attn_score_elems;

    const float * state_in_base = (const float *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    const int64_t rq3 = nev3 / neq3;
    const int64_t rk3 = nev3 / nek3;

    const float scale = 1.0f / sqrtf((float) S_v);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H; // head_index
        const int64_t iv3 = ir / H; // sequence

        const int64_t iq1 = iv1 % neq1;
        const int64_t ik1 = iv1 % nek1;

        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        float * s_out = state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into output buffer and operate in-place
        const float * s_in = state_in_base + (iv3 * H + iv1) * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * sizeof(float));

        // attn output pointer for first token of this (head, seq)
        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        for (int64_t t = 0; t < n_tokens; t++) {
            const float * q_d = (const float *)((const char *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            const float * k_d = (const float *)((const char *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            const float * v_d = (const float *)((const char *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            const float beta_val = *(const float *)((const char *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            const float * g_d    =  (const float *)((const char *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            if (kda) {
                // precompute exp(g) into delta scratch (reused below)
                for (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                for (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } else {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            for (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            for (int64_t j = 0; j < S_v; ++j) {
                float sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token
        }
    }
}


static void ggml_compute_forward_gated_delta_net_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    const bool disable_chunking = ggml_is_numa();

    int nth = params->nth;
    int ith = params->ith;

    // 4x chunks per thread
    int nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    if (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    if (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    const int64_t dr = (nr + nchunk - 1) / nchunk;

    int current_chunk = ith;

    while (current_chunk < nchunk) {
        const int64_t ir0 = dr * current_chunk;
        const int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

void ggml_compute_forward_gated_delta_net(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv7

static void ggml_compute_forward_rwkv_wkv7_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const int64_t T = dst->src[1]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t HEADS = dst->src[1]->ne[1];
    const int64_t n_seqs = dst->src[6]->ne[1];
    const int64_t head_size = C / HEADS;

    float * dst_data = (float *) dst->data;
    float * state = ((float *) dst->data) + C * T;

    const int ith = params->ith;
    const int nth = params->nth;

    // DIVERGENCE from upstream: wkv6 and gla zero dst before computing, but
    // this kernel never did -- and its SIMD path needs that, because the
    // leftover loop after GGML_F32_VEC_REDUCE accumulates with `+=` into what
    // is otherwise uninitialised memory. With more threads than heads the heads
    // nobody computes also keep whatever was in the buffer. Both show up as
    // NaN/Inf in the output on a perfectly ordinary input (reproduced with
    // HEADS=1 on 2 threads, finite inputs, while 1 thread gave correct values).
    // The barrier must be reached by every thread, so the zeroing goes before
    // the early return below.
    if (ith == 0) {
        memset(dst_data, 0, T * C * sizeof(float));
    }
    ggml_barrier(params->threadpool);

    if (ith >= HEADS) {
        return;
    }

    // DIVERGENCE from upstream: T / n_seqs is an unchecked integer division, so
    // an indivisible pair walks off the state buffer, and T < n_seqs divides by
    // zero.
    GGML_ASSERT(n_seqs > 0 && "rwkv_wkv7: state must hold at least one sequence");
    GGML_ASSERT(T % n_seqs == 0 &&
        "rwkv_wkv7: n_tokens must be a multiple of n_seqs");

    // DIVERGENCE from upstream: the original range was
    //     h_start = HEADS*ith/nth, h_end = min(HEADS*(ith+1)/nth, HEADS)
    // which only covers every head when nth <= HEADS. With more threads than
    // heads it hands thread 0 the empty range [0,0) while every other thread
    // leaves through the `ith >= HEADS` check above, so NO thread computes
    // anything and the output stays zero -- silently wrong, neither a crash nor
    // a hang. Reproduced with HEADS=1 on 2 threads. A block split covers the
    // heads for any thread count.
    const int hpt = (HEADS + nth - 1) / nth;
    const int h_start = hpt * ith;
    const int h_end = (h_start + hpt < HEADS) ? h_start + hpt : HEADS;

    float * r = (float *) dst->src[0]->data;
    float * w = (float *) dst->src[1]->data;
    float * k = (float *) dst->src[2]->data;
    float * v = (float *) dst->src[3]->data;
    float * a = (float *) dst->src[4]->data;
    float * b = (float *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #if defined(GGML_SIMD)
        #if defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            for (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                float * state_cur = state + state_offset;
                float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

                for (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    for (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        float v_val = v[t_h_i_offset];

                        float sa = 0, result = 0;
                        for (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        for (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            float r_val = r[t_h_j_offset];
                            float w_val = w[t_h_j_offset];
                            float k_val = k[t_h_j_offset];
                            float b_val = b[t_h_j_offset];
                            float kv_val = v_val * k_val;
                            float prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #else
            // DIVERGENCE from upstream: the vector loops below step by
            // GGML_F32_STEP (32 floats under AVX) and are entered whenever
            // head_size > 0, with no remainder handling -- so any head_size not
            // a multiple of GGML_F32_STEP reads past the end of a, r, w, k, b
            // and the state. The garbage that comes back turns the whole output
            // into NaN. Real RWKV models use head_size 64, which is why upstream
            // never hits it; anything smaller does, every time.
            //
            // Run the scalar arithmetic instead when the vector width does not
            // divide head_size. Correctness first: it is the same computation,
            // just without the vector loads that would run off the end.
            if (head_size % GGML_F32_STEP != 0) {
                for (int64_t t = 0; t < T; t++) {
                    int64_t t_offset = t * t_stride;
                    int64_t state_offset = head_size * C * (t / (T / n_seqs));
                    float * state_cur = state + state_offset;
                    float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

                    for (int64_t h = h_start; h < h_end; h++) {
                        int64_t h_offset = h * h_stride;
                        int64_t t_h_offset = t_offset + h_offset;
                        int64_t h_2d_offset = h * h_stride_2d;

                        for (int64_t i = 0; i < head_size; i++) {
                            int64_t t_h_i_offset = t_h_offset + i;
                            int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                            float v_val = v[t_h_i_offset];

                            float sa = 0, result = 0;
                            for (int64_t j = 0; j < head_size; j++) {
                                sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                            }

                            for (int64_t j = 0; j < head_size; j++) {
                                int64_t t_h_j_offset = t_h_offset + j;
                                int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                                float r_val = r[t_h_j_offset];
                                float w_val = w[t_h_j_offset];
                                float k_val = k[t_h_j_offset];
                                float b_val = b[t_h_j_offset];
                                float kv_val = v_val * k_val;
                                float prev_state_val = state_prev[h_2d_i_j_offset];
                                state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                                result += state_cur[h_2d_i_j_offset] * r_val;
                            }
                            dst_data[t_h_i_offset] = result;
                        }
                    }
                }
                return;
            }
            for (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                float * state_cur = state + state_offset;
                float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

                for (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    for (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        float sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            for (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                for (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_LOAD(&a[t_h_offset + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_offset + j + kk * GGML_F32_EPR]);
                                    sum[kk] = GGML_F32_VEC_FMA(sum[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, sum);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC result_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        for (; j < head_size; j += GGML_F32_STEP) {
                            for (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_offset = t_h_offset + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_offset = h_2d_i_offset + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_LOAD(&r[t_h_j_offset]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_LOAD(&w[t_h_j_offset]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_LOAD(&k[t_h_j_offset]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_LOAD(&b[t_h_j_offset]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_j_offset]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_offset], state_vec);

                                result_vec[kk] = GGML_F32_VEC_FMA(result_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_data[t_h_i_offset], result_vec);

                        // There shouldn't be left-overs though.
                        for (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            float r_val = r[t_h_j_offset];
                            float w_val = w[t_h_j_offset];
                            float k_val = k[t_h_j_offset];
                            float b_val = b[t_h_j_offset];
                            float kv_val = v[t_h_i_offset] * k_val;

                            float prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #else
        for (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            float * state_cur = state + state_offset;
            float * state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

            for (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                for (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    float v_val = v[t_h_i_offset];

                    float sa = 0, result = 0;
                    for (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    for (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        float r_val = r[t_h_j_offset];
                        float w_val = w[t_h_j_offset];
                        float k_val = k[t_h_j_offset];
                        float b_val = b[t_h_j_offset];
                        float kv_val = v_val * k_val;
                        float prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
}


// ggml_compute_forward_rwkv_wkv7_back
//
// ggmlR extension: backward pass for ggml_rwkv_wkv7, absent upstream.
//
// Forward, per head and channel i (state row i):
//     sa[i]   = sum_j a[j]*S[i,j]
//     S[i,j]  = S[i,j]*w[j] + v[i]*k[j] + sa[i]*b[j]
//     y[i]    = sum_j S[i,j]*r[j]
//
// Harder than wkv6 because sa couples the whole row: the gradient reaching
// S[i,j] arrives by two routes -- directly through w, and again through sa,
// which every j of that row contributed to. Both are accumulated before the
// row's gradient is carried to the previous token.
//
// Packed output, in forward source order:
//     [d_r | d_w | d_k | d_v | d_a | d_b | d_state]
//
// Single-threaded and with the early return after the barrier, for the same
// reasons as the wkv6 backward above.
static void ggml_compute_forward_rwkv_wkv7_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src_r = dst->src[0];
    const ggml_tensor * src_w = dst->src[1];
    const ggml_tensor * src_k = dst->src[2];
    const ggml_tensor * src_v = dst->src[3];
    const ggml_tensor * src_a = dst->src[4];
    const ggml_tensor * src_b = dst->src[5];
    const ggml_tensor * src_s = dst->src[6];
    const ggml_tensor * src_g = dst->src[7];

    const int64_t S       = src_k->ne[0];
    const int64_t H       = src_k->ne[1];
    const int64_t T       = src_k->ne[2];
    const int64_t n_seqs  = src_s->ne[1];
    const int64_t C       = S * H;
    const int64_t seq_len = T / n_seqs;

    GGML_ASSERT(n_seqs > 0 && "rwkv_wkv7_back: state must hold at least one sequence");
    GGML_ASSERT(T % n_seqs == 0 &&
        "rwkv_wkv7_back: n_tokens must be a multiple of n_seqs");

    const int ith = params->ith;
    if (ith == 0) {
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);
    if (ith != 0) {
        return;
    }

    const float * r  = (const float *) src_r->data;
    const float * w  = (const float *) src_w->data;
    const float * k  = (const float *) src_k->data;
    const float * v  = (const float *) src_v->data;
    const float * a  = (const float *) src_a->data;
    const float * b  = (const float *) src_b->data;
    const float * s0 = (const float *) src_s->data;

    const float * g_y = (const float *) src_g->data;
    const float * g_s = g_y + C * T;

    float * d_r = (float *) dst->data;
    float * d_w = d_r + ggml_nelements(src_r);
    float * d_k = d_w + ggml_nelements(src_w);
    float * d_v = d_k + ggml_nelements(src_k);
    float * d_a = d_v + ggml_nelements(src_v);
    float * d_b = d_a + ggml_nelements(src_a);
    float * d_s = d_b + ggml_nelements(src_b);

    std::vector<float> St((size_t)(seq_len + 1) * S * S);
    std::vector<float> SA((size_t) seq_len * S);
    std::vector<float> gs((size_t) S * S);

    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t h = 0; h < H; ++h) {
            const size_t sb = (size_t) seq * S * S * H + (size_t) h * S * S;

            // --- replay forward, keeping states and the sa of each token ---
            for (int64_t ij = 0; ij < S * S; ++ij) {
                St[ij] = s0[sb + ij];
            }
            for (int64_t t = 0; t < seq_len; ++t) {
                const int64_t tt = seq * seq_len + t;
                const float * prev = &St[(size_t) t * S * S];
                float * cur = &St[(size_t)(t + 1) * S * S];
                for (int64_t i = 0; i < S; ++i) {
                    float sa = 0.0f;
                    for (int64_t j = 0; j < S; ++j) {
                        sa += a[tt*C + h*S + j] * prev[i*S + j];
                    }
                    SA[(size_t) t * S + i] = sa;
                    for (int64_t j = 0; j < S; ++j) {
                        cur[i*S + j] = prev[i*S + j]*w[tt*C + h*S + j]
                                     + v[tt*C + h*S + i]*k[tt*C + h*S + j]
                                     + sa*b[tt*C + h*S + j];
                    }
                }
            }

            // --- walk back ---
            for (int64_t ij = 0; ij < S * S; ++ij) {
                gs[ij] = g_s[(size_t) seq * S * C + (size_t) h * S * S + ij];
            }

            for (int64_t t = seq_len - 1; t >= 0; --t) {
                const int64_t tt = seq * seq_len + t;
                const float * prev = &St[(size_t) t * S * S];
                const float * cur  = &St[(size_t)(t + 1) * S * S];

                for (int64_t i = 0; i < S; ++i) {
                    const float gy_i = g_y[tt*C + h*S + i];
                    const float sa_i = SA[(size_t) t * S + i];
                    float gsa = 0.0f;
                    float gv_i = 0.0f;

                    for (int64_t j = 0; j < S; ++j) {
                        const float r_j = r[tt*C + h*S + j];

                        d_r[tt*C + h*S + j] += gy_i * cur[i*S + j];

                        // Through y at this token, and through the carry.
                        const float g_cur = gs[i*S + j] + gy_i * r_j;

                        d_w[tt*C + h*S + j] += g_cur * prev[i*S + j];
                        d_k[tt*C + h*S + j] += g_cur * v[tt*C + h*S + i];
                        gv_i                += g_cur * k[tt*C + h*S + j];
                        d_b[tt*C + h*S + j] += g_cur * sa_i;
                        gsa                 += g_cur * b[tt*C + h*S + j];

                        gs[i*S + j] = g_cur * w[tt*C + h*S + j];
                    }

                    d_v[tt*C + h*S + i] += gv_i;

                    // sa[i] = sum_j a[j]*prev[i,j] -- feeds both a and the
                    // previous state, on top of the direct path above.
                    for (int64_t j = 0; j < S; ++j) {
                        d_a[tt*C + h*S + j] += gsa * prev[i*S + j];
                        gs[i*S + j]         += gsa * a[tt*C + h*S + j];
                    }
                }
            }

            for (int64_t ij = 0; ij < S * S; ++ij) {
                d_s[sb + ij] = gs[ij];
            }
        }
    }
}

void ggml_compute_forward_rwkv_wkv7_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_compute_forward_rwkv_wkv7(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

