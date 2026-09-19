/*
 * Differential backend testing: run the same op on the CPU and on an accelerated
 * backend and compare.
 *
 * This is a reduced, R-facing version of upstream's tests/test-backend-ops.cpp.
 * The comparison core (uniform init, NMSE) follows upstream so the numbers mean
 * the same thing; the op list covers what ggmlR actually builds rather than
 * every op ggml supports.
 *
 * Two things here are deliberately NOT upstream behaviour:
 *
 *  - The RNG is seeded per test case, not from random_device. A backend test
 *    that reports a different set of failures on every run is not usable as a
 *    regression test.
 *
 *  - Every case is executed REPEAT_RUNS times against the same allocated graph
 *    and the last run is the one compared. Upstream computes each graph once, so
 *    it cannot see faults that only appear on re-execution -- and those are
 *    exactly what has bitten this package: a predict loop where the first batch
 *    was exact and later ones read stale buffers, and an optimizer step whose
 *    first iteration matched bit-for-bit and then drifted.
 */

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#define MAX_NODES 256
#define REPEAT_RUNS 3

// ---------------------------------------------------------------------------
// Comparison core
// ---------------------------------------------------------------------------

// Deterministic RNG so a failure is reproducible.
static unsigned long long rng_state = 88172645463325252ULL;

static void rng_seed(unsigned long long s) {
    rng_state = s ? s : 88172645463325252ULL;
}

static float rng_uniform(float lo, float hi) {
    // xorshift64
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    const double u = (double)(rng_state >> 11) / (double)(1ULL << 53);
    return (float)(lo + u * (hi - lo));
}

static void init_tensor_uniform(struct ggml_tensor * t, float lo, float hi) {
    const int64_t n = ggml_nelements(t);
    float * data = (float *) malloc((size_t) n * sizeof(float));
    if (!data) return;
    for (int64_t i = 0; i < n; i++) {
        data[i] = rng_uniform(lo, hi);
    }
    ggml_backend_tensor_set(t, data, 0, (size_t) n * sizeof(float));
    free(data);
}

// Normalized mean squared error, as upstream defines it.
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) a[i] - (double) b[i];
        mse_a_b += d * d;
        mse_a_0 += (double) a[i] * (double) a[i];
    }
    if (mse_a_0 == 0.0) {
        return mse_a_b == 0.0 ? 0.0 : 1.0;
    }
    return mse_a_b / mse_a_0;
}

// A case builds its graph into ctx and returns the tensor to compare.
typedef struct ggml_tensor * (*build_fn)(struct ggml_context * ctx);

typedef struct {
    const char * name;
    build_fn     build;
} test_case;

// Result of running one case on one backend.
typedef struct {
    int      ok;          // graph built, allocated and computed
    int      n;           // element count
    float *  data;        // owned
    char     note[128];
} run_result;

static void run_result_free(run_result * r) {
    free(r->data);
    r->data = NULL;
}

// Build, allocate and compute a case on one backend, repeating the compute so
// that re-execution faults surface.
static run_result run_case(const test_case * tc, ggml_backend_t backend, unsigned long long seed) {
    run_result res;
    memset(&res, 0, sizeof(res));

    struct ggml_init_params params;
    params.mem_size   = ggml_tensor_overhead() * MAX_NODES + ggml_graph_overhead();
    params.mem_buffer = NULL;
    params.no_alloc   = true;

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        snprintf(res.note, sizeof(res.note), "ggml_init failed");
        return res;
    }

    struct ggml_tensor * out = tc->build(ctx);
    if (!out) {
        snprintf(res.note, sizeof(res.note), "build returned NULL");
        ggml_free(ctx);
        return res;
    }
    ggml_set_output(out);

    // An op the backend does not implement is not a wrong answer -- report it as
    // unsupported rather than letting the scheduler abort the R session.
    if (!ggml_backend_supports_op(backend, out)) {
        snprintf(res.note, sizeof(res.note), "unsupported by backend");
        ggml_free(ctx);
        return res;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        snprintf(res.note, sizeof(res.note), "alloc_ctx_tensors failed");
        ggml_free(ctx);
        return res;
    }

    // Fill every leaf with the same values on every backend.
    //
    // I32 leaves are row indices (ggml_get_rows), and they must land inside the
    // table being indexed or the op asserts. The table is the other source of
    // the node that consumes them, so look it up rather than guessing a range.
    rng_seed(seed);
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
         t = ggml_get_next_tensor(ctx, t)) {
        if (t->op != GGML_OP_NONE || t->view_src != NULL) {
            continue;
        }
        if (t->type == GGML_TYPE_F32) {
            init_tensor_uniform(t, -1.0f, 1.0f);
        } else if (t->type == GGML_TYPE_F16) {
            // Fill via an F32 staging buffer converted with ggml's own helper, so
            // both backends receive bit-identical F16 input.
            const int64_t n = ggml_nelements(t);
            float * f32 = (float *) malloc((size_t) n * sizeof(float));
            ggml_fp16_t * f16 = (ggml_fp16_t *) malloc((size_t) n * sizeof(ggml_fp16_t));
            if (f32 && f16) {
                for (int64_t i = 0; i < n; i++) {
                    f32[i] = rng_uniform(-1.0f, 1.0f);
                }
                ggml_fp32_to_fp16_row(f32, f16, n);
                ggml_backend_tensor_set(t, f16, 0, (size_t) n * sizeof(ggml_fp16_t));
            }
            free(f32);
            free(f16);
        } else if (t->type == GGML_TYPE_I32) {
            // Index tensors must land inside whatever they index or the op
            // asserts, and the bound differs per consumer: get_rows indexes the
            // ROWS of its table (ne[1]), while mul_mat_id indexes the EXPERT
            // dimension of its stack of matrices (ne[2]). Look the consumer up
            // rather than guessing -- an unrecognised index tensor falls back to
            // all-zeros below, which for mul_mat_id would route every token to
            // expert 0 and quietly stop testing the routing at all.
            int64_t n_rows = 0;
            for (struct ggml_tensor * u = ggml_get_first_tensor(ctx); u != NULL;
                 u = ggml_get_next_tensor(ctx, u)) {
                if (u->op == GGML_OP_GET_ROWS && u->src[1] == t && u->src[0] != NULL) {
                    n_rows = u->src[0]->ne[1];
                    break;
                }
                if (u->op == GGML_OP_MUL_MAT_ID && u->src[2] == t && u->src[0] != NULL) {
                    n_rows = u->src[0]->ne[2];
                    break;
                }
            }
            if (n_rows <= 0) {
                n_rows = 1;   // not an index tensor we recognise; keep it in range
            }
            const int64_t n = ggml_nelements(t);
            int32_t * idx = (int32_t *) malloc((size_t) n * sizeof(int32_t));
            if (idx) {
                // Stride by a value coprime with the bound where possible, so
                // consecutive slots do not all select the same expert and the
                // selection actually varies across the tensor.
                for (int64_t i = 0; i < n; i++) {
                    idx[i] = (int32_t) ((i * 7 + i / 3) % n_rows);
                }
                ggml_backend_tensor_set(t, idx, 0, (size_t) n * sizeof(int32_t));
                free(idx);
            }
        }
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    for (int rep = 0; rep < REPEAT_RUNS; rep++) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            snprintf(res.note, sizeof(res.note), "compute failed on run %d", rep + 1);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return res;
        }
    }

    // Read back according to the output's ACTUAL type: a case like cpy_f16 ends
    // in an F16 tensor, and asking for n*sizeof(float) bytes there reads past the
    // end of the buffer. Everything is converted to F32 for comparison.
    const int64_t n = ggml_nelements(out);
    res.data = (float *) malloc((size_t) n * sizeof(float));
    if (!res.data) {
        snprintf(res.note, sizeof(res.note), "out of memory");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }

    if (out->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(out, res.data, 0, (size_t) n * sizeof(float));
    } else if (out->type == GGML_TYPE_F16) {
        ggml_fp16_t * tmp = (ggml_fp16_t *) malloc((size_t) n * sizeof(ggml_fp16_t));
        if (!tmp) {
            snprintf(res.note, sizeof(res.note), "out of memory");
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return res;
        }
        ggml_backend_tensor_get(out, tmp, 0, (size_t) n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp, res.data, n);
        free(tmp);
    } else if (out->type == GGML_TYPE_I32) {
        int32_t * tmp = (int32_t *) malloc((size_t) n * sizeof(int32_t));
        if (!tmp) {
            snprintf(res.note, sizeof(res.note), "out of memory");
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return res;
        }
        ggml_backend_tensor_get(out, tmp, 0, (size_t) n * sizeof(int32_t));
        for (int64_t i = 0; i < n; i++) {
            res.data[i] = (float) tmp[i];
        }
        free(tmp);
    } else {
        snprintf(res.note, sizeof(res.note), "unhandled output type %s",
                 ggml_type_name(out->type));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }

    res.n  = (int) n;
    res.ok = 1;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// ---------------------------------------------------------------------------
// Cases: the ops ggmlR builds
// ---------------------------------------------------------------------------

static struct ggml_tensor * bo_mul_mat(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 12);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    return ggml_mul_mat(ctx, a, b);
}

static struct ggml_tensor * bo_mul_mat_batched(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 12, 4);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 8, 4);
    return ggml_mul_mat(ctx, a, b);
}

// F16 weights against F32 activations: the shape real inference takes, and a
// different Vulkan pipeline from the all-F32 one above. Leaves of both types are
// seeded identically on every backend, so the comparison stays exact.
static struct ggml_tensor * bo_mul_mat_f16(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 24);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
    return ggml_mul_mat(ctx, a, b);
}

// The 0213-permuted attention path: ggml_vk_mul_mat_vec_p021_f16_f32, selected
// only when src0 is F16, BOTH operands are permuted, dst->ne[1] == 1, both
// ne[3] == 1, and the strides are ordered nb[0] <= nb[2] <= nb[1] <= nb[3].
// ggml_permute(ctx, x, 0, 2, 1, 3) produces exactly that ordering -- axis 1 and
// axis 2 swap positions, which is what the "0213" in the kernel name refers to.
//
// Note the shapes are chosen so that dst->ne[1] == 1 AFTER the permute: dst
// takes ne[1] from src1, so it is the permuted b whose ne[1] must be 1, which
// makes the pre-permute b->ne[2] the constrained one. Getting this wrong does
// not fail loudly -- the dispatcher just falls through to the generic path and
// the case silently stops testing p021 at all.
// The kernel reads ne02/ne12 from the tensors as handed to it, i.e. after the
// permute: 8/8 here, so this exercises the gqa_ratio==1 pipeline.
static struct ggml_tensor * bo_mul_mat_vec_p021(struct ggml_context * ctx) {
    // head dim 32, 8 keys, 2 heads -> permuted to [32,2,8,1] x [32,1,8,1].
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 8, 2, 1);
    struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 8, 1, 1);
    return ggml_mul_mat(ctx, ggml_permute(ctx, a, 0, 2, 1, 3),
                             ggml_permute(ctx, b, 0, 2, 1, 3));
}

// The non-contiguous-but-not-permuted vector path:
// ggml_vk_mul_mat_vec_nc_f16_f32, selected when src0 is F16 and NOT contiguous
// while neither operand is permuted or transposed, with dst->ne[1] == 1.
// A view with a row gap satisfies that: it is strided, so !ggml_is_contiguous,
// yet the axis order is untouched, so !ggml_is_permuted.
static struct ggml_tensor * bo_mul_mat_vec_nc(struct ggml_context * ctx) {
    struct ggml_tensor * wide = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 40, 12, 2);
    // First 32 of each 40-element row; dim 0 stays packed, rows stay 40 apart.
    struct ggml_tensor * a = ggml_view_3d(ctx, wide, 32, 12, 2,
                                          wide->nb[1], wide->nb[2], 0);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
    return ggml_mul_mat(ctx, a, b);
}

// Large enough to leave the vector path and reach the tiled matmul kernels,
// where the m/n thresholds in ggml_vk_guess_matmul_pipeline select the large
// tile. The small cases above never get there.
//
// K is a multiple of every pipeline alignment, so this takes the ALIGNED
// variant (mmp->a_l): the fast path with no remainder handling.
static struct ggml_tensor * bo_mul_mat_large(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 512);
    return ggml_mul_mat(ctx, a, b);
}

// The same tiled kernels on a shape that divides evenly nowhere. K=509 is prime,
// so ne10 != kpad and ggml_vk_guess_matmul_pipeline returns the UNALIGNED
// variant (mmp->l rather than mmp->a_l) -- a different shader, reached only by
// a K that is not a multiple of the pipeline alignment. Every other mul_mat
// case here uses K in {16, 32, 512}, so without this one the unaligned tile
// kernels and their remainder handling are never executed at all. M and N are
// likewise off every tile boundary, so the partial edge tiles are covered too.
static struct ggml_tensor * bo_mul_mat_unaligned(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 509, 511);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 509, 513);
    return ggml_mul_mat(ctx, a, b);
}

// Strided src[1]: a column-slice view of a wider matrix, so nb[1] is bigger
// than the row it addresses and the kernel reads with a gap between rows.
// This is the shape real graphs produce (sliced activations, KV-cache views)
// and it exercises the strided read path rather than the packed one.
//
// Note what may NOT be done here: mul_mat requires BOTH operands to be
// contiguous along dim 0 -- ggml-cpu-backend.c asserts
// nb00/nb10 == ggml_type_size ("we don't support permuted src0 or src1").
// Handing it a transposed operand aborts the process from inside the kernel,
// and ggml_backend_supports_op does not catch it beforehand. A transposed
// operand is materialised with ggml_cont() in real graphs, which is a
// different op and no longer a mul_mat test.
static struct ggml_tensor * bo_mul_mat_noncont(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 24);
    struct ggml_tensor * wide = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 40, 8);
    // First 16 of each 40-element row: dim 0 stays packed, rows stay 40 apart.
    struct ggml_tensor * b = ggml_view_2d(ctx, wide, 16, 8, wide->nb[1], 0);
    return ggml_mul_mat(ctx, a, b);
}

// Broadcast: one weight matrix applied across a batch of activations, i.e. a
// with ne[2]==1 against b with ne[2]==4. This is the dimension-matching rule
// that ONNX models have already tripped over.
static struct ggml_tensor * bo_mul_mat_broadcast(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 12, 1);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 8, 4);
    return ggml_mul_mat(ctx, a, b);
}

// MoE routing: a stack of expert matrices, a per-token expert selection, and
// one row of activations per token. Constraints, all asserted by the builder:
// as is 3d with ne[3]==1, b is 3d, ids is 2d I32, ids->ne[1] == b->ne[2] (one
// expert list per b row), as->ne[0] == b->ne[0], and ids->ne[0] % b->ne[1] == 0.
// This reaches ggml_vk_mul_mat_id_q_f16 -- one of the four functions whose
// operand repack was wrong until mul_mat_noncont exposed it, and the only one
// of them with no coverage here at all.
static struct ggml_tensor * bo_mul_mat_id(struct ggml_context * ctx) {
    // 4 experts of [16 x 12], 6 tokens, 2 experts chosen per token.
    struct ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 12, 4);
    struct ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 2, 6);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 6);
    return ggml_mul_mat_id(ctx, as, b, ids);
}

// The same op down the vector path: ne[1]==1 on b selects
// ggml_vk_mul_mat_vec_id_q_f16 rather than the matrix kernel, which is a
// separate dispatcher branch and the fourth function the repack bug touched.
static struct ggml_tensor * bo_mul_mat_id_vec(struct ggml_context * ctx) {
    struct ggml_tensor * as  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 12, 4);
    struct ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 1, 6);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 6);
    return ggml_mul_mat_id(ctx, as, b, ids);
}

static struct ggml_tensor * bo_add(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 5, 3);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 5, 3);
    return ggml_add(ctx, a, b);
}

static struct ggml_tensor * bo_add_broadcast(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 8, 4);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 1, 1);
    return ggml_add(ctx, a, ggml_repeat(ctx, b, a));
}

static struct ggml_tensor * bo_mul(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 8, 4);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 1, 1);
    return ggml_mul(ctx, a, ggml_repeat(ctx, b, a));
}

static struct ggml_tensor * bo_relu(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
    return ggml_relu(ctx, a);
}

static struct ggml_tensor * bo_soft_max(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    return ggml_soft_max(ctx, a);
}

static struct ggml_tensor * bo_rms_norm(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 8);
    return ggml_rms_norm(ctx, a, 1e-5f);
}

// The permute/cont round trip conv_1d relies on.
static struct ggml_tensor * bo_permute_cont(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 8, 4);
    struct ggml_tensor * p = ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3));
    return ggml_cont(ctx, ggml_permute(ctx, p, 1, 0, 2, 3));
}

static struct ggml_tensor * bo_transpose_cont(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    return ggml_cont(ctx, ggml_transpose(ctx, a));
}

// im2col in the 1-D form nn_build_conv_1d() uses, F32 throughout.
static struct ggml_tensor * bo_im2col_1d(struct ggml_context * ctx) {
    struct ggml_tensor * kern = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 7);
    struct ggml_tensor * data = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 4, 8);
    return ggml_im2col(ctx, kern, data, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
}

static struct ggml_tensor * bo_conv_2d(struct ggml_context * ctx) {
    struct ggml_tensor * kern = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 4);
    struct ggml_tensor * data = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6, 6, 1, 8);
    return ggml_conv_2d(ctx, kern, data, 1, 1, 0, 0, 1, 1);
}

// The batch_norm inference branch: per-channel centre and scale on [C, L, N].
static struct ggml_tensor * bo_batch_norm_infer(struct ggml_context * ctx) {
    struct ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 8, 4);
    struct ggml_tensor * m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    struct ggml_tensor * v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    struct ggml_tensor * m3 = ggml_reshape_3d(ctx, m, 7, 1, 1);
    struct ggml_tensor * v3 = ggml_reshape_3d(ctx, v, 7, 1, 1);
    struct ggml_tensor * c  = ggml_sub(ctx, x, ggml_cont(ctx, ggml_repeat(ctx, m3, x)));
    // v is uniform in [-1, 1]; square it so the sqrt argument stays positive.
    struct ggml_tensor * d  = ggml_sqrt(ctx, ggml_scale_bias(ctx, ggml_sqr(ctx, v3), 1.0f, 1e-5f));
    return ggml_div(ctx, c, ggml_cont(ctx, ggml_repeat(ctx, d, x)));
}

// The training branch: fold the reduced axes into ne[0] and take the mean.
static struct ggml_tensor * bo_batch_norm_train(struct ggml_context * ctx) {
    struct ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 4, 5);
    struct ggml_tensor * r = ggml_reshape_3d(ctx, x, 6, 4, 5);
    struct ggml_tensor * p = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
    struct ggml_tensor * w = ggml_reshape_2d(ctx, p, 30, 4);
    struct ggml_tensor * mu = ggml_mean(ctx, w);
    struct ggml_tensor * c  = ggml_cont(ctx, ggml_sub(ctx, w, ggml_cont(ctx, ggml_repeat(ctx, mu, w))));
    struct ggml_tensor * var = ggml_mean(ctx, ggml_sqr(ctx, c));
    struct ggml_tensor * den = ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0f, 1e-5f));
    return ggml_div(ctx, c, ggml_cont(ctx, ggml_repeat(ctx, den, c)));
}

static struct ggml_tensor * bo_get_rows(struct ggml_context * ctx) {
    struct ggml_tensor * emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 32);
    struct ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8);
    return ggml_get_rows(ctx, emb, idx);
}

static struct ggml_tensor * bo_pool_2d(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 3, 4);
    return ggml_pool_2d(ctx, a, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
}

static struct ggml_tensor * bo_sum_rows(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 8);
    return ggml_sum_rows(ctx, a);
}

static struct ggml_tensor * bo_mean(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 24, 8, 3);
    return ggml_mean(ctx, a);
}

static struct ggml_tensor * bo_cont_after_view(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 16);
    struct ggml_tensor * v = ggml_view_2d(ctx, a, 16, 16, a->nb[1], 0);
    return ggml_cont(ctx, v);
}

// ---------------------------------------------------------------------------
// Cases: the rest of what the Vulkan backend implements
// ---------------------------------------------------------------------------

// Elementwise unary. One shape for all of them; the point is the shader, not the
// geometry. Inputs are uniform in [-1, 1], so ops with a restricted domain get a
// shifted or squared argument.
#define BO_UNARY(name, expr)                                              \
    static struct ggml_tensor * bo_##name(struct ggml_context * ctx) {    \
        struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6); \
        return expr;                                                      \
    }

BO_UNARY(abs,          ggml_abs(ctx, a))
BO_UNARY(neg,          ggml_neg(ctx, a))
BO_UNARY(sgn,          ggml_sgn(ctx, a))
BO_UNARY(step,         ggml_step(ctx, a))
BO_UNARY(elu,          ggml_elu(ctx, a))
BO_UNARY(gelu,         ggml_gelu(ctx, a))
BO_UNARY(gelu_erf,     ggml_gelu_erf(ctx, a))
BO_UNARY(gelu_quick,   ggml_gelu_quick(ctx, a))
BO_UNARY(silu,         ggml_silu(ctx, a))
BO_UNARY(sigmoid,      ggml_sigmoid(ctx, a))
BO_UNARY(tanh,         ggml_tanh(ctx, a))
BO_UNARY(hardsigmoid,  ggml_hardsigmoid(ctx, a))
BO_UNARY(hardswish,    ggml_hardswish(ctx, a))
BO_UNARY(exp,          ggml_exp(ctx, a))
BO_UNARY(sin,          ggml_sin(ctx, a))
BO_UNARY(cos,          ggml_cos(ctx, a))
BO_UNARY(sqr,          ggml_sqr(ctx, a))
BO_UNARY(floor,        ggml_floor(ctx, a))
BO_UNARY(ceil,         ggml_ceil(ctx, a))
BO_UNARY(round,        ggml_round(ctx, a))
BO_UNARY(trunc,        ggml_trunc(ctx, a))
BO_UNARY(leaky_relu,   ggml_leaky_relu(ctx, a, 0.1f, false))
BO_UNARY(clamp,        ggml_clamp(ctx, a, -0.5f, 0.5f))
BO_UNARY(scale,        ggml_scale(ctx, a, 2.5f))
BO_UNARY(scale_bias,   ggml_scale_bias(ctx, a, 2.5f, 0.5f))
// Positive domain: square first so log/sqrt stay defined.
BO_UNARY(log,          ggml_log(ctx, ggml_scale_bias(ctx, ggml_sqr(ctx, a), 1.0f, 0.5f)))
BO_UNARY(sqrt,         ggml_sqrt(ctx, ggml_scale_bias(ctx, ggml_sqr(ctx, a), 1.0f, 0.5f)))

#undef BO_UNARY

// Binary and structural.
static struct ggml_tensor * bo_sub(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    return ggml_sub(ctx, a, b);
}

static struct ggml_tensor * bo_div(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    // Keep the divisor away from zero.
    return ggml_div(ctx, a, ggml_scale_bias(ctx, ggml_sqr(ctx, b), 1.0f, 0.5f));
}

static struct ggml_tensor * bo_add1(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    struct ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    return ggml_add1(ctx, a, s);
}

static struct ggml_tensor * bo_concat(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 5, 3);
    struct ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 5, 3);
    return ggml_concat(ctx, a, b, 2);
}

static struct ggml_tensor * bo_repeat(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 6);
    return ggml_repeat(ctx, a, b);
}

static struct ggml_tensor * bo_repeat_back(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    return ggml_repeat_back(ctx, a, b);
}

static struct ggml_tensor * bo_pad(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 6, 5, 2, 1);
    return ggml_pad(ctx, a, 2, 1, 0, 0);
}

static struct ggml_tensor * bo_upscale(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 5, 4, 2, 1);
    return ggml_upscale(ctx, a, 2, GGML_SCALE_MODE_NEAREST);
}

static struct ggml_tensor * bo_roll(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 5, 2, 1);
    return ggml_roll(ctx, a, 2, 1, 0, 0);
}

static struct ggml_tensor * bo_acc(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 3);
    return ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], 0);
}

// diag_mask_inf writes -inf into the masked half, and NMSE over infinities is
// NaN regardless of whether the backends agree. Feed the masked tensor through
// soft_max, which is how it is used in practice and which maps -inf to 0.
static struct ggml_tensor * bo_diag_mask_inf(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    return ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, a, 2));
}

static struct ggml_tensor * bo_norm(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 6);
    return ggml_norm(ctx, a, 1e-5f);
}

static struct ggml_tensor * bo_group_norm(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 4, 2);
    return ggml_group_norm(ctx, a, 2, 1e-5f);
}

static struct ggml_tensor * bo_l2_norm(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 6);
    return ggml_l2_norm(ctx, a, 1e-5f);
}

static struct ggml_tensor * bo_rms_norm_back(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 6);
    return ggml_rms_norm_back(ctx, a, b, 1e-5f);
}

static struct ggml_tensor * bo_soft_max_back(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 6);
    return ggml_soft_max_ext_back(ctx, a, ggml_soft_max(ctx, b), 1.0f, 0.0f);
}

static struct ggml_tensor * bo_silu_back(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    return ggml_silu_back(ctx, a, b);
}

static struct ggml_tensor * bo_sum(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    return ggml_sum(ctx, a);
}

static struct ggml_tensor * bo_cumsum(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    return ggml_cumsum(ctx, a);
}

static struct ggml_tensor * bo_conv_2d_dw(struct ggml_context * ctx) {
    struct ggml_tensor * kern = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 4);
    struct ggml_tensor * data = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 4, 2);
    return ggml_conv_2d_dw_direct(ctx, kern, data, 1, 1, 1, 1, 1, 1);
}

static struct ggml_tensor * bo_conv_transpose_1d(struct ggml_context * ctx) {
    struct ggml_tensor * kern = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 4, 2);
    struct ggml_tensor * data = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 2, 1);
    return ggml_conv_transpose_1d(ctx, kern, data, 1, 0, 1);
}

static struct ggml_tensor * bo_im2col_2d(struct ggml_context * ctx) {
    struct ggml_tensor * kern = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 2, 4);
    struct ggml_tensor * data = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 2, 2);
    return ggml_im2col(ctx, kern, data, 1, 1, 0, 0, 1, 1, true, GGML_TYPE_F32);
}

static struct ggml_tensor * bo_pool_2d_avg(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 3, 2);
    return ggml_pool_2d(ctx, a, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
}

static struct ggml_tensor * bo_timestep_embedding(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    return ggml_timestep_embedding(ctx, a, 16, 10000);
}

static struct ggml_tensor * bo_cpy_f16(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 24, 6);
    return ggml_cpy(ctx, a, b);
}

static struct ggml_tensor * bo_dup(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 24, 6);
    return ggml_dup(ctx, a);
}

static struct ggml_tensor * bo_set(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 6);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 12, 3);
    return ggml_set(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], 0);
}

static struct ggml_tensor * bo_swiglu(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 6);
    return ggml_swiglu(ctx, a);
}

static struct ggml_tensor * bo_geglu(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 6);
    return ggml_geglu(ctx, a);
}

static struct ggml_tensor * bo_reglu(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 6);
    return ggml_reglu(ctx, a);
}

static struct ggml_tensor * bo_rope(struct ggml_context * ctx) {
    struct ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 4, 2);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    return ggml_rope(ctx, a, pos, 16, 0);
}

static struct ggml_tensor * bo_flash_attn(struct ggml_context * ctx) {
    // head dim 32, 8 keys, 4 queries, 1 head
    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 4, 1, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 8, 1, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 8, 1, 1);
    return ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f / 8.0f, 0.0f, 0.0f);
}

// 2D relative position bias (BoTNet pos_embed), in three shapes.
//
// This op was uncovered here while its Vulkan shader disagreed with the CPU
// kernel by 1.55 on BoTNet26t, and the shape of that bug dictates the shapes
// of these cases:
//
//  - H != W is mandatory. The shader had W_h and W_w swapped against the H and
//    W offsets of the packed weight table, and on a square input both halves
//    have the same length, so the swap stays inside the buffer and the case
//    passes while computing the wrong thing. rel_pos_bias_rect is the one that
//    actually pins the axis/weight pairing down.
//  - The relative index is k-q, and a wrong sign mirrors the bias about the
//    diagonal. Asymmetric H and W make that visible too.
//  - rel_pos_bias_botnet carries the real failing geometry (H=W=16, B=4 heads,
//    C=64), which is also the only case here big enough to span more than one
//    workgroup on the {8,8,4} dispatch.
//
// wcat packs W_h then W_w along axis 0, so its ne[0] is (2H-1) + (2W-1).
static struct ggml_tensor * bo_rel_pos_bias(struct ggml_context * ctx) {
    const int H = 4, W = 4, B = 2, C = 8;
    struct ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, H * W, B);
    struct ggml_tensor * wcat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   (2 * H - 1) + (2 * W - 1), C);
    return ggml_rel_pos_bias(ctx, x, wcat, H, W);
}

static struct ggml_tensor * bo_rel_pos_bias_rect(struct ggml_context * ctx) {
    const int H = 3, W = 5, B = 2, C = 6;
    struct ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, H * W, B);
    struct ggml_tensor * wcat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   (2 * H - 1) + (2 * W - 1), C);
    return ggml_rel_pos_bias(ctx, x, wcat, H, W);
}

static struct ggml_tensor * bo_rel_pos_bias_botnet(struct ggml_context * ctx) {
    const int H = 16, W = 16, B = 4, C = 64;
    struct ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, H * W, B);
    struct ggml_tensor * wcat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   (2 * H - 1) + (2 * W - 1), C);
    return ggml_rel_pos_bias(ctx, x, wcat, H, W);
}

static const test_case g_cases[] = {
    // --- ops ggmlR's layers build ---
    { "mul_mat",            bo_mul_mat            },
    { "mul_mat_batched",    bo_mul_mat_batched    },
    { "mul_mat_f16",        bo_mul_mat_f16        },
    { "mul_mat_large",      bo_mul_mat_large      },
    { "mul_mat_unaligned",  bo_mul_mat_unaligned  },
    { "mul_mat_noncont",    bo_mul_mat_noncont    },
    { "mul_mat_id",         bo_mul_mat_id         },
    { "mul_mat_id_vec",     bo_mul_mat_id_vec     },
    { "mul_mat_vec_p021",   bo_mul_mat_vec_p021   },
    { "mul_mat_vec_nc",     bo_mul_mat_vec_nc     },
    { "mul_mat_broadcast",  bo_mul_mat_broadcast  },
    { "add",                bo_add                },
    { "add_broadcast",      bo_add_broadcast      },
    { "mul_broadcast",      bo_mul                },
    { "relu",               bo_relu               },
    { "soft_max",           bo_soft_max           },
    { "rms_norm",           bo_rms_norm           },
    { "permute_cont",       bo_permute_cont       },
    { "transpose_cont",     bo_transpose_cont     },
    { "im2col_1d",          bo_im2col_1d          },
    { "conv_2d",            bo_conv_2d            },
    { "batch_norm_infer",   bo_batch_norm_infer   },
    { "batch_norm_train",   bo_batch_norm_train   },
    { "get_rows",           bo_get_rows           },
    { "pool_2d",            bo_pool_2d            },
    { "sum_rows",           bo_sum_rows           },
    { "mean",               bo_mean               },
    { "cont_after_view",    bo_cont_after_view    },

    // --- elementwise unary ---
    { "abs",                bo_abs                },
    { "neg",                bo_neg                },
    { "sgn",                bo_sgn                },
    { "step",               bo_step               },
    { "elu",                bo_elu                },
    { "gelu",               bo_gelu               },
    { "gelu_erf",           bo_gelu_erf           },
    { "gelu_quick",         bo_gelu_quick         },
    { "silu",               bo_silu               },
    { "sigmoid",            bo_sigmoid            },
    { "tanh",               bo_tanh               },
    { "hardsigmoid",        bo_hardsigmoid        },
    { "hardswish",          bo_hardswish          },
    { "exp",                bo_exp                },
    { "sin",                bo_sin                },
    { "cos",                bo_cos                },
    { "sqr",                bo_sqr                },
    { "sqrt",               bo_sqrt               },
    { "log",                bo_log                },
    { "floor",              bo_floor              },
    { "ceil",               bo_ceil               },
    { "round",              bo_round              },
    { "trunc",              bo_trunc              },
    { "leaky_relu",         bo_leaky_relu         },
    { "clamp",              bo_clamp              },
    { "scale",              bo_scale              },
    { "scale_bias",         bo_scale_bias         },

    // --- binary / structural ---
    { "sub",                bo_sub                },
    { "div",                bo_div                },
    { "add1",               bo_add1               },
    { "concat",             bo_concat             },
    { "repeat",             bo_repeat             },
    { "repeat_back",        bo_repeat_back        },
    { "pad",                bo_pad                },
    { "upscale",            bo_upscale            },
    { "roll",               bo_roll               },
    { "acc",                bo_acc                },
    { "set",                bo_set                },
    { "dup",                bo_dup                },
    { "cpy_f16",            bo_cpy_f16            },
    { "diag_mask_inf",      bo_diag_mask_inf      },

    // --- normalization ---
    { "norm",               bo_norm               },
    { "group_norm",         bo_group_norm         },
    { "l2_norm",            bo_l2_norm            },

    // --- reductions ---
    { "sum",                bo_sum                },
    { "cumsum",             bo_cumsum             },

    // --- backward kernels ---
    { "rms_norm_back",      bo_rms_norm_back      },
    { "soft_max_back",      bo_soft_max_back      },
    { "silu_back",          bo_silu_back          },

    // --- convolution family ---
    { "conv_2d_dw",         bo_conv_2d_dw         },
    { "conv_transpose_1d",  bo_conv_transpose_1d  },
    { "im2col_2d",          bo_im2col_2d          },
    { "pool_2d_avg",        bo_pool_2d_avg        },

    // --- attention / transformer ---
    { "rope",               bo_rope               },
    { "flash_attn",         bo_flash_attn         },
    { "swiglu",             bo_swiglu             },
    { "geglu",              bo_geglu              },
    { "reglu",              bo_reglu              },
    { "timestep_embedding", bo_timestep_embedding },
    { "rel_pos_bias",       bo_rel_pos_bias       },
    { "rel_pos_bias_rect",  bo_rel_pos_bias_rect  },
    { "rel_pos_bias_botnet", bo_rel_pos_bias_botnet },
};

static const int g_n_cases = (int) (sizeof(g_cases) / sizeof(g_cases[0]));

// ---------------------------------------------------------------------------
// Stateful case: the AdamW optimizer step
// ---------------------------------------------------------------------------
//
// This one cannot go through run_case(). ggml_opt_step_adamw() updates the
// weights and both moment buffers IN PLACE, so what it computes on step N
// depends on every step before it, and the bias-correction terms it is handed
// change each step. A single execution compares almost nothing: the first step
// of Adam matches bit-for-bit on both backends here, and the divergence only
// appears from the second one onwards.
//
// So run ADAMW_STEPS steps against the same graph, feeding fresh parameters each
// time exactly as ggml-opt.cpp does, and compare the weight trajectory.

#define ADAMW_STEPS 6
#define ADAMW_N     64

typedef struct {
    int    ok;
    double w_after[ADAMW_STEPS];  // checksum of the weights after each step
    char   note[128];
} adamw_result;

static adamw_result run_adamw(ggml_backend_t backend, unsigned long long seed) {
    adamw_result res;
    memset(&res, 0, sizeof(res));

    struct ggml_init_params params;
    params.mem_size   = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    params.mem_buffer = NULL;
    params.no_alloc   = true;

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        snprintf(res.note, sizeof(res.note), "ggml_init failed");
        return res;
    }

    struct ggml_tensor * w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ADAMW_N);
    ggml_set_param(w);   // ggml_opt_step_adamw asserts the weight carries FLAG_PARAM
    struct ggml_tensor * grad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ADAMW_N);
    struct ggml_tensor * m    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ADAMW_N);
    struct ggml_tensor * v    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ADAMW_N);
    struct ggml_tensor * pars = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);

    struct ggml_tensor * step = ggml_opt_step_adamw(ctx, w, grad, m, v, pars);

    if (!ggml_backend_supports_op(backend, step)) {
        snprintf(res.note, sizeof(res.note), "unsupported by backend");
        ggml_free(ctx);
        return res;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        snprintf(res.note, sizeof(res.note), "alloc_ctx_tensors failed");
        ggml_free(ctx);
        return res;
    }

    // Same starting point on every backend. Moments start at zero, as ggml-opt
    // initializes them.
    rng_seed(seed);
    init_tensor_uniform(w, -1.0f, 1.0f);
    {
        float * zeros = (float *) calloc(ADAMW_N, sizeof(float));
        if (zeros) {
            ggml_backend_tensor_set(m, zeros, 0, ADAMW_N * sizeof(float));
            ggml_backend_tensor_set(v, zeros, 0, ADAMW_N * sizeof(float));
            free(zeros);
        }
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, step);

    const float alpha = 0.001f, beta1 = 0.9f, beta2 = 0.999f;
    const float eps = 1e-8f, wd = 0.0f;

    float * gbuf = (float *) malloc(ADAMW_N * sizeof(float));
    float * wbuf = (float *) malloc(ADAMW_N * sizeof(float));
    if (!gbuf || !wbuf) {
        snprintf(res.note, sizeof(res.note), "out of memory");
        free(gbuf); free(wbuf);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }

    for (int it = 1; it <= ADAMW_STEPS; it++) {
        // A different gradient each step, identical across backends.
        rng_seed(seed * 31u + (unsigned long long) it);
        for (int i = 0; i < ADAMW_N; i++) {
            gbuf[i] = rng_uniform(-0.5f, 0.5f);
        }
        ggml_backend_tensor_set(grad, gbuf, 0, ADAMW_N * sizeof(float));

        // Bias correction depends on the step number -- this is the part that
        // differs from SGD, whose parameters are constant.
        float p[7];
        p[0] = alpha;
        p[1] = beta1;
        p[2] = beta2;
        p[3] = eps;
        p[4] = wd;
        p[5] = 1.0f / (1.0f - powf(beta1, (float) it));
        p[6] = 1.0f / (1.0f - powf(beta2, (float) it));
        ggml_backend_tensor_set(pars, p, 0, sizeof(p));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            snprintf(res.note, sizeof(res.note), "compute failed on step %d", it);
            free(gbuf); free(wbuf);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return res;
        }

        ggml_backend_tensor_get(w, wbuf, 0, ADAMW_N * sizeof(float));
        double sum = 0.0;
        for (int i = 0; i < ADAMW_N; i++) {
            sum += (double) wbuf[i];
        }
        res.w_after[it - 1] = sum;
    }

    res.ok = 1;
    free(gbuf); free(wbuf);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// ---------------------------------------------------------------------------
// R entry point
// ---------------------------------------------------------------------------

// Compare the AdamW weight trajectory step by step.
extern "C" SEXP R_ggml_test_adamw_steps(SEXP backend_ptr) {
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(backend_ptr);
    if (backend == NULL) {
        Rf_error("Invalid backend pointer");
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == NULL) {
        Rf_error("Failed to init CPU backend");
    }

    const unsigned long long seed = 20260728ULL;
    adamw_result a = run_adamw(cpu, seed);
    adamw_result b = run_adamw(backend, seed);
    ggml_backend_free(cpu);

    if (!a.ok || !b.ok) {
        SEXP note = PROTECT(Rf_mkString(!a.ok ? a.note : b.note));
        SEXP out  = PROTECT(Rf_allocVector(VECSXP, 1));
        SET_VECTOR_ELT(out, 0, note);
        SEXP nm = PROTECT(Rf_allocVector(STRSXP, 1));
        SET_STRING_ELT(nm, 0, Rf_mkChar("note"));
        Rf_setAttrib(out, R_NamesSymbol, nm);
        UNPROTECT(3);
        return out;
    }

    SEXP step  = PROTECT(Rf_allocVector(INTSXP,  ADAMW_STEPS));
    SEXP w_cpu = PROTECT(Rf_allocVector(REALSXP, ADAMW_STEPS));
    SEXP w_be  = PROTECT(Rf_allocVector(REALSXP, ADAMW_STEPS));
    SEXP diff  = PROTECT(Rf_allocVector(REALSXP, ADAMW_STEPS));

    for (int i = 0; i < ADAMW_STEPS; i++) {
        INTEGER(step)[i]  = i + 1;
        REAL(w_cpu)[i]    = a.w_after[i];
        REAL(w_be)[i]     = b.w_after[i];
        REAL(diff)[i]     = fabs(a.w_after[i] - b.w_after[i]);
    }

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(out, 0, step);
    SET_VECTOR_ELT(out, 1, w_cpu);
    SET_VECTOR_ELT(out, 2, w_be);
    SET_VECTOR_ELT(out, 3, diff);

    SEXP nm = PROTECT(Rf_allocVector(STRSXP, 4));
    SET_STRING_ELT(nm, 0, Rf_mkChar("step"));
    SET_STRING_ELT(nm, 1, Rf_mkChar("w_cpu"));
    SET_STRING_ELT(nm, 2, Rf_mkChar("w_backend"));
    SET_STRING_ELT(nm, 3, Rf_mkChar("abs_diff"));
    Rf_setAttrib(out, R_NamesSymbol, nm);

    UNPROTECT(6);
    return out;
}

extern "C" SEXP R_ggml_test_backend_ops(SEXP backend_ptr, SEXP filter) {
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(backend_ptr);
    if (backend == NULL) {
        Rf_error("Invalid backend pointer");
    }

    const char * pat = NULL;
    if (filter != R_NilValue && Rf_length(filter) > 0) {
        pat = CHAR(STRING_ELT(filter, 0));
        if (pat[0] == '\0') pat = NULL;
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == NULL) {
        Rf_error("Failed to init CPU backend");
    }

    int selected[sizeof(g_cases) / sizeof(g_cases[0])];
    int n_sel = 0;
    for (int i = 0; i < g_n_cases; i++) {
        if (pat == NULL || strstr(g_cases[i].name, pat) != NULL) {
            selected[n_sel++] = i;
        }
    }

    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_sel));
    SEXP errs  = PROTECT(Rf_allocVector(REALSXP, n_sel));
    SEXP notes = PROTECT(Rf_allocVector(STRSXP, n_sel));

    for (int k = 0; k < n_sel; k++) {
        const test_case * tc = &g_cases[selected[k]];
        // Same seed for both backends: the inputs must be identical.
        const unsigned long long seed = 1234567ULL + (unsigned long long) selected[k] * 7919ULL;

        run_result a = run_case(tc, cpu, seed);
        run_result b = run_case(tc, backend, seed);

        SET_STRING_ELT(names, k, Rf_mkChar(tc->name));

        if (!a.ok || !b.ok) {
            REAL(errs)[k] = NA_REAL;
            SET_STRING_ELT(notes, k, Rf_mkChar(!a.ok ? a.note : b.note));
        } else if (a.n != b.n) {
            REAL(errs)[k] = NA_REAL;
            SET_STRING_ELT(notes, k, Rf_mkChar("element count differs"));
        } else {
            REAL(errs)[k] = nmse(a.data, b.data, (size_t) a.n);
            SET_STRING_ELT(notes, k, Rf_mkChar(""));
        }

        run_result_free(&a);
        run_result_free(&b);
    }

    SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
    SET_VECTOR_ELT(out, 0, names);
    SET_VECTOR_ELT(out, 1, errs);
    SET_VECTOR_ELT(out, 2, notes);

    SEXP out_names = PROTECT(Rf_allocVector(STRSXP, 3));
    SET_STRING_ELT(out_names, 0, Rf_mkChar("op"));
    SET_STRING_ELT(out_names, 1, Rf_mkChar("nmse"));
    SET_STRING_ELT(out_names, 2, Rf_mkChar("note"));
    Rf_setAttrib(out, R_NamesSymbol, out_names);

    ggml_backend_free(cpu);
    UNPROTECT(5);
    return out;
}
