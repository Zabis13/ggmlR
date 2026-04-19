#include "ops.h"

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

// ggml_compute_forward_map_custom1

void ggml_compute_forward_map_custom1(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * a = dst->src[0];

    struct ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

void ggml_compute_forward_map_custom2(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];

    struct ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

void ggml_compute_forward_map_custom3(
        const ggml_compute_params * params,
              ggml_tensor * dst) {

    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * c = dst->src[2];

    struct ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

void ggml_compute_forward_custom(
    const struct ggml_compute_params * params,
          struct ggml_tensor * dst) {

    struct ggml_custom_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

static void ggml_compute_forward_cross_entropy_loss_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nrows(src0);

    const int ith = params->ith;
    const int nth = params->nth;

    float * sums =  (float *) params->wdata;
    float * st   = ((float *) params->wdata) + nth + ith*nc;
    float sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= sizeof(float) * (nth + nth * nc));

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t i1 = ir0; i1 < ir1; ++i1) {
        const float * s0 = (const float *)((const char *) src0->data + i1*src0->nb[1]);
        const float * s1 = (const float *)((const char *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif

        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        const ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        assert(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        float sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            assert(!isnan(st[i]));
            assert(!isinf(st[i]));
        }
#endif
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    if (ith == 0) {
        float * dp = (float *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (float) nr;
    }
}

void ggml_compute_forward_cross_entropy_loss(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

static void ggml_compute_forward_cross_entropy_loss_back_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    const ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    const ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    const int64_t nc = src0f->ne[0];
    const int64_t nr = ggml_nrows(src0f);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    const float d_by_nr = ((const float *) grad->data)[0] / (float) nr;

    for (int64_t i1 = ir0; i1 < ir1; i1++) {
        float       * ds0 = (float       *)((char       *) dst->data   + i1*dst->nb[1]);
        const float * s0  = (const float *)((const char *) src0f->data + i1*src0f->nb[1]);
        const float * s1  = (const float *)((const char *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif

        // soft_max
        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        const ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        assert(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        for (int64_t i = 0; i < nc; ++i) {
            assert(!isnan(ds0[i]));
            assert(!isinf(ds0[i]));
        }
#endif
    }
}

void ggml_compute_forward_cross_entropy_loss_back(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_opt_step_adamw_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0         = dst->src[0];
    const ggml_tensor * src0_grad    = dst->src[1];
    const ggml_tensor * src0_grad_m  = dst->src[2];
    const ggml_tensor * src0_grad_v  = dst->src[3];
    const ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    const float alpha  = adamw_params_ptr[0];
    const float beta1  = adamw_params_ptr[1];
    const float beta2  = adamw_params_ptr[2];
    const float eps    = adamw_params_ptr[3];
    const float wd     = adamw_params_ptr[4];
    const float beta1h = adamw_params_ptr[5];
    const float beta2h = adamw_params_ptr[6];
    const float keep   = 1.f - alpha * wd;
    for (int ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        float       * w = (float       *) ((char       *) src0->data        + offset); // weight
        const float * g = (const float *) ((const char *) src0_grad->data   + offset); // grad
        float       * m = (float       *) ((char       *) src0_grad_m->data + offset);
        float       * v = (float       *) ((char       *) src0_grad_v->data + offset);

        for (int i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            const float mh =       m[i00]*beta1h;
            const float vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

void ggml_compute_forward_opt_step_adamw(
        const ggml_compute_params * params,
        ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

static void ggml_compute_forward_opt_step_sgd_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0       = dst->src[0];
    const ggml_tensor * src0_grad  = dst->src[1];
    const ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    const float * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    const float   alpha            = sgd_params_ptr[0];
    const float   keep             = 1.f - alpha * sgd_params_ptr[1];

    for (int ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        const size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        float *       w = (float *) ((char *) src0->data + offset);                   // weight
        const float * g = (const float *) ((const char *) src0_grad->data + offset);  // grad

        for (int i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

void ggml_compute_forward_opt_step_sgd(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            break;
        default:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

// ggml_compute_forward_rel_pos_bias

void ggml_compute_forward_rel_pos_bias(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) return;

    const struct ggml_tensor * x    = dst->src[0];
    const struct ggml_tensor * wcat = dst->src[1];

    const int H     = ggml_get_op_params_i32(dst, 0);
    const int W     = ggml_get_op_params_i32(dst, 1);
    const int B     = ggml_get_op_params_i32(dst, 2);
    const int C     = ggml_get_op_params_i32(dst, 3);
    const int rel_h = ggml_get_op_params_i32(dst, 4);
    const int rel_w = (2 * W - 1);
    const int HW    = H * W;
    const int Wstride = rel_h + rel_w;

    const float * xdata = (const float *)x->data;
    const float * Wdata = (const float *)wcat->data;
    float       * out   = (float *)dst->data;

    memset(out, 0, (size_t)B * HW * HW * sizeof(float));

    for (int b = 0; b < B; b++) {
        const float * xb = xdata + (size_t)b * C * HW;
        for (int hq = 0; hq < H; hq++) {
            for (int wq = 0; wq < W; wq++) {
                const int q_idx = hq * W + wq;
                const float * x_hw = xb + (size_t)q_idx * C;
                const float * x_wh = xb + (size_t)(wq * H + hq) * C;

                for (int hk = 0; hk < H; hk++) {
                    const int r_h = hq - hk + H - 1;
                    float dot_h = 0.0f;
                    for (int ci = 0; ci < C; ci++)
                        dot_h += x_hw[ci] * Wdata[r_h + ci * Wstride];

                    for (int wk = 0; wk < W; wk++) {
                        const int r_w = wq - wk + W - 1;
                        float dot_w = 0.0f;
                        for (int ci = 0; ci < C; ci++)
                            dot_w += x_wh[ci] * Wdata[rel_h + r_w + ci * Wstride];

                        const int k_idx = hk * W + wk;
                        out[k_idx + (size_t)q_idx * HW + (size_t)b * HW * HW] = dot_h + dot_w;
                    }
                }
            }
        }
    }
}

// ggml_compute_forward_cast_numeric
// Numeric type conversion: truncate/round, not bitwise reinterpret.
// Supported pairs: F32<->I32, F32<->I16, F32<->I8, F16<->F32, BF16<->F32.
// Falls back to ggml_compute_forward_dup for other pairs.
void ggml_compute_forward_cast_numeric(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int ith = params->ith;
    const int nth = params->nth;
    const int64_t n = ggml_nelements(src);
    const int64_t per_thread = (n + nth - 1) / nth;
    const int64_t i0 = ith * per_thread;
    const int64_t i1 = i0 + per_thread < n ? i0 + per_thread : n;

    if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
        const float   * s = (const float *)   src->data;
              int32_t * d = (      int32_t *) dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = (int32_t)s[i];
    } else if (src->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32) {
        const int32_t * s = (const int32_t *) src->data;
              float   * d = (      float *)   dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = (float)s[i];
    } else if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I16) {
        const float   * s = (const float *)   src->data;
              int16_t * d = (      int16_t *) dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = (int16_t)s[i];
    } else if (src->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_F32) {
        const int16_t * s = (const int16_t *) src->data;
              float   * d = (      float *)   dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = (float)s[i];
    } else if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I8) {
        const float * s = (const float *) src->data;
              int8_t* d = (      int8_t*) dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = (int8_t)s[i];
    } else if (src->type == GGML_TYPE_I8 && dst->type == GGML_TYPE_F32) {
        const int8_t* s = (const int8_t*) src->data;
              float * d = (      float *) dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = (float)s[i];
    } else if (src->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) src->data;
              float       * d = (      float *)        dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = GGML_FP16_TO_FP32(s[i]);
    } else if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        const float       * s = (const float *)        src->data;
              ggml_fp16_t * d = (      ggml_fp16_t *)  dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = GGML_FP32_TO_FP16(s[i]);
    } else if (src->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        const ggml_bf16_t * s = (const ggml_bf16_t *) src->data;
              float       * d = (      float *)        dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = GGML_BF16_TO_FP32(s[i]);
    } else if (src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        const float       * s = (const float *)        src->data;
              ggml_bf16_t * d = (      ggml_bf16_t *)  dst->data;
        for (int64_t i = i0; i < i1; i++) d[i] = GGML_FP32_TO_BF16(s[i]);
    } else {
        GGML_ABORT("ggml_cast_numeric: unsupported type pair %s -> %s",
                   ggml_type_name(src->type), ggml_type_name(dst->type));
    }
}
