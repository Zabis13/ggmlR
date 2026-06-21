// Vulkan backend R interface
// Only compiled when GGML_USE_VULKAN is defined

#include <math.h>
#include <R.h>
#include <Rinternals.h>
#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

// ============================================================================
// Vulkan Backend Functions
// ============================================================================

// Check if Vulkan support is compiled in
SEXP R_ggml_vulkan_is_available(void) {
#ifdef GGML_USE_VULKAN
    return ScalarLogical(TRUE);
#else
    return ScalarLogical(FALSE);
#endif
}

// Get number of Vulkan devices
SEXP R_ggml_vulkan_device_count(void) {
#ifdef GGML_USE_VULKAN
    int count = ggml_backend_vk_get_device_count();
    return ScalarInteger(count);
#else
    return ScalarInteger(0);
#endif
}

// Get device description
SEXP R_ggml_vulkan_device_description(SEXP device_idx) {
#ifdef GGML_USE_VULKAN
    int device = asInteger(device_idx);
    int count = ggml_backend_vk_get_device_count();

    if (device < 0 || device >= count) {
        error("Invalid device index: %d (available: 0-%d)", device, count - 1);
    }

    char description[256];
    ggml_backend_vk_get_device_description(device, description, sizeof(description));

    return mkString(description);
#else
    error("Vulkan support not compiled. Reinstall with --configure-args=\"--with-vulkan\"");
    return R_NilValue;
#endif
}

// Get device memory info
SEXP R_ggml_vulkan_device_memory(SEXP device_idx) {
#ifdef GGML_USE_VULKAN
    int device = asInteger(device_idx);
    int count = ggml_backend_vk_get_device_count();

    if (device < 0 || device >= count) {
        error("Invalid device index: %d (available: 0-%d)", device, count - 1);
    }

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_vk_get_device_memory(device, &free_mem, &total_mem);

    // Return named list with free and total memory
    SEXP result = PROTECT(allocVector(VECSXP, 2));
    SEXP names = PROTECT(allocVector(STRSXP, 2));

    SET_VECTOR_ELT(result, 0, ScalarReal((double)free_mem));
    SET_VECTOR_ELT(result, 1, ScalarReal((double)total_mem));

    SET_STRING_ELT(names, 0, mkChar("free"));
    SET_STRING_ELT(names, 1, mkChar("total"));
    setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
#else
    error("Vulkan support not compiled. Reinstall with --configure-args=\"--with-vulkan\"");
    return R_NilValue;
#endif
}

#ifdef GGML_USE_VULKAN
// Finalizer: free the Vulkan backend when its external pointer is GC'd.
// Cleared on manual R_ggml_backend_free, so this never double-frees.
static void r_ggml_vk_backend_finalizer(SEXP ptr) {
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(ptr);
    if (backend != NULL) {
        ggml_backend_free(backend);
        R_ClearExternalPtr(ptr);
    }
}
#endif

// Initialize Vulkan backend
SEXP R_ggml_vulkan_init(SEXP device_idx) {
#ifdef GGML_USE_VULKAN
    int device = asInteger(device_idx);
    int count = ggml_backend_vk_get_device_count();

    if (count == 0) {
        error("No Vulkan devices found");
    }

    if (device < 0 || device >= count) {
        error("Invalid device index: %d (available: 0-%d)", device, count - 1);
    }

    ggml_backend_t backend = ggml_backend_vk_init((size_t)device);

    if (backend == NULL) {
        error("Failed to initialize Vulkan backend for device %d", device);
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(backend, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, r_ggml_vk_backend_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
#else
    error("Vulkan support not compiled. Reinstall with --configure-args=\"--with-vulkan\"");
    return R_NilValue;
#endif
}

// Free Vulkan backend
SEXP R_ggml_vulkan_free(SEXP backend_ptr) {
#ifdef GGML_USE_VULKAN
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

    if (backend != NULL) {
        ggml_backend_free(backend);
        R_ClearExternalPtr(backend_ptr);
    }

    return R_NilValue;
#else
    error("Vulkan support not compiled");
    return R_NilValue;
#endif
}

// Check if backend is Vulkan
SEXP R_ggml_vulkan_is_backend(SEXP backend_ptr) {
#ifdef GGML_USE_VULKAN
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

    if (backend == NULL) {
        return ScalarLogical(FALSE);
    }

    return ScalarLogical(ggml_backend_is_vk(backend));
#else
    return ScalarLogical(FALSE);
#endif
}

// Get backend name
SEXP R_ggml_vulkan_backend_name(SEXP backend_ptr) {
#ifdef GGML_USE_VULKAN
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }

    const char* name = ggml_backend_name(backend);
    return mkString(name);
#else
    error("Vulkan support not compiled");
    return R_NilValue;
#endif
}

// List all Vulkan devices (convenience function)
SEXP R_ggml_vulkan_list_devices(void) {
#ifdef GGML_USE_VULKAN
    int count = ggml_backend_vk_get_device_count();

    if (count == 0) {
        return allocVector(VECSXP, 0);
    }

    SEXP result = PROTECT(allocVector(VECSXP, count));
    SEXP names = PROTECT(allocVector(STRSXP, count));

    for (int i = 0; i < count; i++) {
        // Create device info list
        SEXP device_info = PROTECT(allocVector(VECSXP, 4));
        SEXP device_names = PROTECT(allocVector(STRSXP, 4));

        // Device index
        SET_VECTOR_ELT(device_info, 0, ScalarInteger(i));
        SET_STRING_ELT(device_names, 0, mkChar("index"));

        // Device description
        char description[256];
        ggml_backend_vk_get_device_description(i, description, sizeof(description));
        SET_VECTOR_ELT(device_info, 1, mkString(description));
        SET_STRING_ELT(device_names, 1, mkChar("name"));

        // Memory info
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_vk_get_device_memory(i, &free_mem, &total_mem);
        SET_VECTOR_ELT(device_info, 2, ScalarReal((double)free_mem));
        SET_STRING_ELT(device_names, 2, mkChar("free_memory"));
        SET_VECTOR_ELT(device_info, 3, ScalarReal((double)total_mem));
        SET_STRING_ELT(device_names, 3, mkChar("total_memory"));

        setAttrib(device_info, R_NamesSymbol, device_names);
        SET_VECTOR_ELT(result, i, device_info);

        // Name for outer list
        char idx_str[32];
        snprintf(idx_str, sizeof(idx_str), "device_%d", i);
        SET_STRING_ELT(names, i, mkChar(idx_str));

        UNPROTECT(2);
    }

    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
#else
    return allocVector(VECSXP, 0);
#endif
}

SEXP R_ggml_vulkan_device_caps(SEXP device_idx) {
#ifdef GGML_USE_VULKAN
    bool coopmat_support = false, coopmat1_fa_support = false, fp16 = false, subgroup_no_shmem = false;
    bool bf16 = false, integer_dot_product = false, supports_256_push_constants = false;
    uint32_t subgroup_size = 0, subgroup_min_size = 0, subgroup_max_size = 0, wavefronts_per_simd = 0;
    uint32_t coopmat_m = 0, coopmat_n = 0, coopmat_k = 0, max_push_constants_size = 0;
    const char *arch_name = "OTHER";
    ggml_backend_vk_get_device_caps(asInteger(device_idx), &coopmat_support, &coopmat1_fa_support,
                                     &fp16, &subgroup_size, &subgroup_no_shmem,
                                     &subgroup_min_size, &subgroup_max_size,
                                     &wavefronts_per_simd, &bf16, &integer_dot_product, &arch_name,
                                     &coopmat_m, &coopmat_n, &coopmat_k,
                                     &supports_256_push_constants, &max_push_constants_size);

    const char *nms[] = {"coopmat_support", "coopmat1_fa_support", "fp16", "bf16",
                         "integer_dot_product",
                         "subgroup_size", "subgroup_min_size", "subgroup_max_size",
                         "subgroup_no_shmem", "wavefronts_per_simd", "arch",
                         "coopmat_m", "coopmat_n", "coopmat_k",
                         "supports_256_push_constants", "max_push_constants_size"};
    SEXP result = PROTECT(allocVector(VECSXP, 16));
    SEXP names  = PROTECT(allocVector(STRSXP, 16));
    SET_VECTOR_ELT(result, 0,  ScalarLogical(coopmat_support));
    SET_VECTOR_ELT(result, 1,  ScalarLogical(coopmat1_fa_support));
    SET_VECTOR_ELT(result, 2,  ScalarLogical(fp16));
    SET_VECTOR_ELT(result, 3,  ScalarLogical(bf16));
    SET_VECTOR_ELT(result, 4,  ScalarLogical(integer_dot_product));
    SET_VECTOR_ELT(result, 5,  ScalarInteger((int)subgroup_size));
    SET_VECTOR_ELT(result, 6,  ScalarInteger((int)subgroup_min_size));
    SET_VECTOR_ELT(result, 7,  ScalarInteger((int)subgroup_max_size));
    SET_VECTOR_ELT(result, 8,  ScalarLogical(subgroup_no_shmem));
    SET_VECTOR_ELT(result, 9,  ScalarInteger((int)wavefronts_per_simd));
    SET_VECTOR_ELT(result, 10, mkString(arch_name));
    SET_VECTOR_ELT(result, 11, ScalarInteger((int)coopmat_m));
    SET_VECTOR_ELT(result, 12, ScalarInteger((int)coopmat_n));
    SET_VECTOR_ELT(result, 13, ScalarInteger((int)coopmat_k));
    SET_VECTOR_ELT(result, 14, ScalarLogical(supports_256_push_constants));
    SET_VECTOR_ELT(result, 15, ScalarInteger((int)max_push_constants_size));
    for (int i = 0; i < 16; i++) SET_STRING_ELT(names, i, mkChar(nms[i]));
    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
#else
    error("Vulkan support not compiled");
    return R_NilValue;
#endif
}

// ============================================================================
// UMAP SGD layout optimisation (direct Vulkan dispatch)
// ============================================================================

// Args:
//   backend_ptr : Vulkan backend external pointer
//   coords      : numeric, length n*2, [x0,y0,x1,y1,...] (modified -> returned)
//   edges       : integer, length ne*2, [from0,to0,...], 0-based vertex indices
//   weights     : numeric, length ne
//   n, ne, n_epochs, n_neg : integer scalars
//   a, b, alpha0, gamma    : numeric scalars
//   base_seed              : integer scalar
// Returns a numeric vector of length n*2 (the optimised coordinates).
SEXP R_ggml_umap_sgd(SEXP backend_ptr, SEXP coords, SEXP edges, SEXP weights,
                     SEXP n_, SEXP ne_, SEXP n_epochs_, SEXP n_neg_,
                     SEXP a_, SEXP b_, SEXP alpha0_, SEXP gamma_, SEXP seed_) {
#ifdef GGML_USE_VULKAN
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    if (backend == NULL || !ggml_backend_is_vk(backend)) {
        error("R_ggml_umap_sgd: backend is not a valid Vulkan backend");
    }

    unsigned int n        = (unsigned int)asInteger(n_);
    unsigned int ne       = (unsigned int)asInteger(ne_);
    unsigned int n_epochs = (unsigned int)asInteger(n_epochs_);
    unsigned int n_neg    = (unsigned int)asInteger(n_neg_);
    unsigned int seed     = (unsigned int)asInteger(seed_);
    float a      = (float)asReal(a_);
    float b      = (float)asReal(b_);
    float alpha0 = (float)asReal(alpha0_);
    float gamma  = (float)asReal(gamma_);

    if ((R_xlen_t)XLENGTH(coords) != (R_xlen_t)n * 2)
        error("R_ggml_umap_sgd: coords length != n*2");
    if ((R_xlen_t)XLENGTH(edges) != (R_xlen_t)ne * 2)
        error("R_ggml_umap_sgd: edges length != ne*2");

    // R doubles/ints -> float/uint32 working buffers
    float *c = (float*)R_alloc((size_t)n * 2, sizeof(float));
    unsigned int *e = (unsigned int*)R_alloc((size_t)ne * 2, sizeof(unsigned int));
    float *w = (float*)R_alloc((size_t)ne, sizeof(float));

    double *cd = REAL(coords);
    for (R_xlen_t i = 0; i < (R_xlen_t)n * 2; i++) c[i] = (float)cd[i];
    int *ei = INTEGER(edges);
    for (R_xlen_t i = 0; i < (R_xlen_t)ne * 2; i++) e[i] = (unsigned int)ei[i];
    double *wd = REAL(weights);
    for (R_xlen_t i = 0; i < (R_xlen_t)ne; i++) w[i] = (float)wd[i];

    bool ok = ggml_vk_umap_sgd_run(backend, c, e, w, n, ne, n_epochs, n_neg,
                                   a, b, alpha0, gamma, seed);
    if (!ok) error("R_ggml_umap_sgd: GPU dispatch failed");

    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)n * 2));
    double *od = REAL(out);
    for (R_xlen_t i = 0; i < (R_xlen_t)n * 2; i++) od[i] = (double)c[i];
    UNPROTECT(1);
    return out;
#else
    error("Vulkan support not compiled");
    return R_NilValue;
#endif
}

// ============================================================================
// Pairwise squared-distance matrix (direct Vulkan dispatch, f32-accurate)
// ============================================================================

// Args:
//   backend_ptr : Vulkan backend external pointer
//   x_          : numeric, length n*dims, row-major [row0_d0,row0_d1,...]
//   n_, dims_   : integer scalars
// Returns a numeric vector of length n*n: the Euclidean distance matrix in
// (symmetric) order D[i*n+j]. The shader returns squared distances; we take the
// sqrt here, in the same pass that converts float->double, so the R side needs
// no further work. The matrix is symmetric, so row-major and column-major
// orderings coincide and the caller can read it as either.
SEXP R_ggml_dist_f32(SEXP backend_ptr, SEXP x_, SEXP n_, SEXP dims_) {
#ifdef GGML_USE_VULKAN
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    if (backend == NULL || !ggml_backend_is_vk(backend)) {
        error("R_ggml_dist_f32: backend is not a valid Vulkan backend");
    }

    unsigned int n    = (unsigned int)asInteger(n_);
    unsigned int dims = (unsigned int)asInteger(dims_);

    if ((R_xlen_t)XLENGTH(x_) != (R_xlen_t)n * dims)
        error("R_ggml_dist_f32: x length != n*dims");

    // R doubles -> float working buffers
    float *x  = (float*)R_alloc((size_t)n * dims, sizeof(float));
    float *d2 = (float*)R_alloc((size_t)n * n,    sizeof(float));

    double *xd = REAL(x_);
    for (R_xlen_t i = 0; i < (R_xlen_t)n * dims; i++) x[i] = (float)xd[i];

    bool ok = ggml_vk_pairwise_dist_run(backend, x, d2, n, dims);
    if (!ok) error("R_ggml_dist_f32: GPU dispatch failed");

    // float->double + sqrt(max(0, .)) in one pass. The clamp guards tiny
    // negative round-off from f32 accumulation; doing it here saves the R side a
    // full extra sweep over n*n elements (matrix/sqrt/clamp), which dominated the
    // wrapper cost at scale.
    SEXP out = PROTECT(allocVector(REALSXP, (R_xlen_t)n * n));
    double *od = REAL(out);
    for (R_xlen_t i = 0; i < (R_xlen_t)n * n; i++) {
        float v = d2[i];
        od[i] = (v > 0.0f) ? sqrt((double)v) : 0.0;
    }
    UNPROTECT(1);
    return out;
#else
    error("Vulkan support not compiled");
    return R_NilValue;
#endif
}
