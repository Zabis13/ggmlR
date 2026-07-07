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
GGML_BACKEND_API void ggml_backend_vk_get_device_caps(int device, bool * coopmat_support, bool * coopmat1_fa_support, bool * fp16, uint32_t * subgroup_size, bool * subgroup_no_shmem, uint32_t * subgroup_min_size, uint32_t * subgroup_max_size, uint32_t * wavefronts_per_simd, bool * bf16, bool * integer_dot_product, const char ** arch_name, uint32_t * coopmat_m, uint32_t * coopmat_n, uint32_t * coopmat_k, bool * supports_256_push_constants, uint32_t * max_push_constants_size);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// ggmlR, not upstream: explicitly release the Vulkan devices while the loader/ICD
// .so files are still mapped. Idempotent. If hard != 0, calls _exit(0) after
// teardown to skip the atexit/static-destruction phase entirely — the only
// reliable way to avoid the flaky exit-time loader-race segfault (f1ba0), since
// no R exit hook runs before R unmaps the loader. Use hard=1 as the last
// statement of a script/example, after all results are produced.
GGML_BACKEND_API void ggml_backend_vk_shutdown(int hard);

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

#ifdef  __cplusplus
}
#endif
