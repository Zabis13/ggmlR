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
// ggmlR Tensor Parallelism (P2P), not upstream: enumerate device groups (LDA) and
// probe peer memory access; writes a report and returns the number of groups.
GGML_BACKEND_API int  ggml_backend_vk_get_device_groups(char * report, size_t report_size);
// ggmlR Tensor Parallelism (P2P), not upstream: pure row-split math for the split
// buffer type, exposed for unit-testing slice boundaries without a GPU.
// Given a row count, a per-device weight vector (may be NULL for an even split)
// and n_devices, fills row_low[i]/row_high[i] (each length n_devices) with the
// [low, high) row range owned by device i. Returns 0 on success, -1 on bad args.
GGML_BACKEND_API int  ggml_backend_vk_split_row_ranges(int64_t nrows, const float * weights,
                                                       int n_devices,
                                                       int64_t * row_low, int64_t * row_high);
// ggmlR Tensor Parallelism (P2P), not upstream: opaque-fd P2P self-test. Exports
// an fd on src_dev, imports it on dst_dev, copies `bytes` and verifies the data;
// when src_dev != dst_dev also times `iters` device->device copies and reports the
// achieved bandwidth (GB/s) in *out_gbps. Returns 0 on success, <0 on failure.
// A rate above ~16 GB/s (PCIe 3.0 x16) is empirical evidence a faster link (e.g.
// NVLink) carried the bytes — the route is inferred, not queried from Vulkan.
// `transport`: 0 = host-staging (default, portable), 1 = opaque-fd, 2 = device-group.
GGML_BACKEND_API int  ggml_backend_vk_p2p_selftest(int src_dev, int dst_dev,
                                                   size_t bytes, int iters, int transport,
                                                   double * out_gbps,
                                                   char * report, size_t report_size);
// ggmlR Tensor Parallelism (P2P), not upstream: Stage E3 tensor-parallel mul_mat.
// Computes Y = W * X with W ([K cols, N rows]) row-split across n_devices devices
// and X ([K cols, M rows]) broadcast. Flat f32 column-major buffers:
//   w: N*K (w[n*K+k]), x: M*K (x[m*K+k]), y: M*N out (y[m*N+n]).
// `weights` (may be NULL for an even split) is the per-device row weighting.
// `transport`: 0 = host-staging (default), 1 = opaque-fd, 2 = device-group.
// Returns 0 on success, <0 on failure; `report` (optional) gets a short summary.
GGML_BACKEND_API int  ggml_backend_vk_split_mul_mat(const float * w, const float * x, float * y,
                                                    int64_t N, int64_t K, int64_t M,
                                                    const float * weights, int n_devices,
                                                    const int * device_ids, int transport,
                                                    char * report, size_t report_size);
// ggmlR Tensor Parallelism (P2P), not upstream: Stage E4 split buffer type.
// Creates (or fetches from cache) a Vulkan tensor-split buffer type that row-
// splits weights across n_devices devices. `tensor_split` is a per-device weight
// vector of length n_devices (NULL for an even split); `main_device` holds non-
// split fallbacks; `transport` selects the gather transport (0=host-staging).
// Returns NULL on bad arguments. The buffer_type is cached — do not free it.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_split_buffer_type(
        int main_device, const float * tensor_split, int n_devices,
        const int * device_ids, int transport);
// ggmlR Tensor Parallelism (P2P), not upstream: Stage E7 pipeline handoff.
// Copies an activation tensor `src` (on one device's Vulkan buffer) into the next
// pipeline stage's input tensor `dst` (on another device's buffer) via host
// staging. `src` and `dst` must be Vulkan-backed and have equal ggml_nbytes.
// Returns 0 on success, <0 on a shape/buffer mismatch.
GGML_BACKEND_API int ggml_backend_vk_stage_handoff(const struct ggml_tensor * src,
                                                   struct ggml_tensor * dst);
GGML_BACKEND_API void ggml_backend_vk_get_device_caps(int device, bool * coopmat_support, bool * coopmat1_fa_support, bool * fp16, uint32_t * subgroup_size, bool * subgroup_no_shmem, uint32_t * subgroup_min_size, uint32_t * subgroup_max_size, uint32_t * wavefronts_per_simd, bool * bf16, bool * integer_dot_product, const char ** arch_name, uint32_t * coopmat_m, uint32_t * coopmat_n, uint32_t * coopmat_k, bool * supports_256_push_constants, uint32_t * max_push_constants_size, bool * subgroup_shuffle, bool * subgroup_vote);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// ggmlR, not upstream: explicitly release the Vulkan devices while the loader/ICD
// .so files are still mapped. Idempotent. If hard != 0, calls _exit(status) after
// teardown to skip the atexit/static-destruction phase entirely — the only
// reliable way to avoid the flaky exit-time loader-race segfault (f1ba0), since
// no R exit hook runs before R unmaps the loader. Use hard=1 as the last
// statement of a script/example, after all results are produced. `status` is the
// process exit code passed to _exit(); pass non-zero from an error path so a
// failed run does not exit 0. `status` is ignored when hard == 0.
GGML_BACKEND_API void ggml_backend_vk_shutdown(int hard, int status);

// UMAP SGD layout optimisation, dispatched directly (not via the ggml graph).
// coords is n*2 floats updated in place. Returns false if backend is not Vulkan.
GGML_BACKEND_API bool ggml_vk_umap_sgd_run(
    ggml_backend_t backend,
    float * coords, const unsigned int * edges, const float * weights,
    unsigned int n, unsigned int ne,
    unsigned int n_epochs, unsigned int n_neg,
    float a, float b, float alpha0, float gamma,
    unsigned int base_seed);

// Pairwise squared Euclidean distance matrix, dispatched directly. X is n rows
// of `dims` floats (row-major); d2 receives n*n floats (row-major, D2[i*n+j]).
// The caller takes sqrt() where it wants Euclidean distance. Returns false if
// the backend is not Vulkan.
GGML_BACKEND_API bool ggml_vk_pairwise_dist_run(
    ggml_backend_t backend,
    const float * x, float * d2,
    unsigned int n, unsigned int dims);

// RoiAlign (ONNX opset 10+), dispatched directly. Samples num_rois regions out
// of the feature map x (ggml layout [W, H, C, N]) into dst (ggml layout
// [ow, oh, C, num_rois]); rois is [4, num_rois] as x1, y1, x2, y2 and
// batch_indices is [num_rois] F32 indices into N. sampling_ratio 0 means
// adaptive, mode 0 = avg / 1 = max. Returns false if the backend is not Vulkan.
//
// This is a bit-exact port of the CPU kernel in src/onnx/roi_align.c and must
// stay one: MaskRCNN-12-int8 matches ONNX Runtime exactly, and that agreement
// depends on reproducing ORT's arithmetic rather than a cleaner equivalent.
GGML_BACKEND_API bool ggml_vk_roi_align_run(
    ggml_backend_t backend,
    const float * x, const float * rois, const float * batch_indices,
    float * dst,
    unsigned int W, unsigned int H, unsigned int C, unsigned int N,
    unsigned int num_rois, unsigned int ow, unsigned int oh,
    int sampling_ratio, unsigned int mode, float spatial_scale);

// QLinearMatMul with an exact i32 accumulator (ONNX), dispatched directly.
// A is [K, M] and b_mat is [K, N], both quantised values held as f32, K fastest
// in each; dst receives [N, M]. b_scale / b_zp are per output column when
// n_b_scale / n_b_zp exceed 1, otherwise one shared value. b_zp_any tells the
// shader whether any zero point is non-zero, so it can skip a row sum.
// Returns false if the backend is not Vulkan.
//
// This is a bit-exact port of src/onnx/qmatmul_i32.c, including ORT's
// VPMADDUBSW int16 pair saturation -- reproducing that LOSS of precision is
// what makes the numbers match; exact arithmetic here disagrees.
GGML_BACKEND_API bool ggml_vk_qmatmul_i32_run(
    ggml_backend_t backend,
    const float * a, const float * b_mat,
    const float * b_scale, const int * b_zp,
    float * dst,
    unsigned int M, unsigned int N, unsigned int K,
    float a_scale, float y_scale, int a_zp, int y_zp,
    unsigned int n_b_scale, unsigned int n_b_zp, unsigned int b_zp_any,
    float out_lo, float out_hi);

// NonMaxSuppression (ONNX), dispatched directly. boxes is ggml [4, num_boxes,
// N] and scores ggml [num_boxes, num_classes, N]. One workgroup per
// (batch, class) pair filters, sorts and selects that pair independently and
// writes its survivors into its own slice: sel_idx is
// [max_per_pair, num_classes * N] and sel_cnt holds how many each pair kept,
// both indexed by the flat pair (n * num_classes + cls).
//
// The caller assembles the operator's [3, max_selected] output from those in
// batch-major, class-major order and applies the GLOBAL max_selected cap. That
// bookkeeping is deliberately not in the shader: workgroups finish in an
// arbitrary order, and the global cap is a sequential decision across them.
//
// max_per_pair caps one pair's survivors; a class with more candidates than the
// shader's fixed sort capacity is refused (returns false) so the caller can
// fall back. Also returns false if the backend is not Vulkan.
GGML_BACKEND_API bool ggml_vk_nms_run(
    ggml_backend_t backend,
    const float * boxes, const float * scores,
    int * sel_idx, int * sel_cnt,
    unsigned int num_boxes, unsigned int num_classes, unsigned int N,
    unsigned int max_per_pair, int max_output,
    float iou_thresh, float score_thresh, unsigned int have_score_thresh,
    unsigned int center_point_box);

// Tiled fused k-NN, dispatched directly. For each of n rows of x (dims each),
// finds the k nearest other rows in honest f32 without materialising the n x n
// distance matrix. Outputs knn_idx (n*k 0-based row indices) and knn_dist (n*k
// Euclidean distances), sorted ascending per row. k must be <= the pipeline's
// top-k capacity (32). Returns false if the backend is not Vulkan.
GGML_BACKEND_API bool ggml_vk_knn_tiled_run(
    ggml_backend_t backend,
    const float * x, unsigned int * knn_idx, float * knn_dist,
    unsigned int n, unsigned int dims, unsigned int k);

// FP64 matmul (PoC, benchmark only): C[M,N] = A[M,K] * B[K,N] entirely in
// double, dispatched directly (no float conversion). Returns false if the
// backend is not Vulkan or the device lacks fp64 support.
GGML_BACKEND_API bool ggml_vk_matmul_f64_run(
    ggml_backend_t backend,
    const double * a, const double * b, double * c,
    unsigned int M, unsigned int N, unsigned int K);

// Sparse LogNormalize over a dgCMatrix's stored non-zeros, dispatched directly.
// vals is nnz floats updated in place; factor is scale_factor/colSum per column
// (n_cols floats); col_of_nnz is the 0-based column of each stored value (nnz
// uints). Returns false if the backend is not Vulkan.
GGML_BACKEND_API bool ggml_vk_sparse_lognorm_run(
    ggml_backend_t backend,
    float * vals, const float * factor, const unsigned int * col_of_nnz,
    unsigned int nnz, unsigned int n_cols);

#ifdef  __cplusplus
}
#endif
