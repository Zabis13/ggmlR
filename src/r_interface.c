#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "ggml.h"
#include "ggml-cpu.h"

// ============================================================================
// Context Management
// ============================================================================

SEXP R_ggml_init(SEXP mem_size) {
    int size = asInteger(mem_size);
    
    struct ggml_init_params params = {
        .mem_size = size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    
    if (ctx == NULL) {
        error("Failed to initialize GGML context");
    }
    
    SEXP ptr = PROTECT(R_MakeExternalPtr(ctx, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_free(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    
    if (ctx != NULL) {
        ggml_free(ctx);
        R_ClearExternalPtr(ctx_ptr);
    }
    
    return R_NilValue;
}

SEXP R_ggml_reset(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);

    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    // Reset context - clears all allocations in the memory pool
    // Memory can be reused without recreating context
    ggml_reset(ctx);

    return R_NilValue;
}

// ============================================================================
// Time Functions
// ============================================================================

SEXP R_ggml_time_init(void) {
    ggml_time_init();
    return R_NilValue;
}

SEXP R_ggml_time_ms(void) {
    int64_t ms = ggml_time_ms();
    return ScalarReal((double) ms);
}

SEXP R_ggml_time_us(void) {
    int64_t us = ggml_time_us();
    return ScalarReal((double) us);
}

SEXP R_ggml_cycles(void) {
    int64_t cycles = ggml_cycles();
    return ScalarReal((double) cycles);
}

SEXP R_ggml_cycles_per_ms(void) {
    int64_t cpm = ggml_cycles_per_ms();
    return ScalarReal((double) cpm);
}

// ============================================================================
// Memory Information
// ============================================================================

SEXP R_ggml_tensor_overhead(void) {
    size_t overhead = ggml_tensor_overhead();
    return ScalarReal((double) overhead);
}

SEXP R_ggml_get_mem_size(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }
    
    size_t mem_size = ggml_get_mem_size(ctx);
    return ScalarReal((double) mem_size);
}

SEXP R_ggml_used_mem(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }
    
    size_t used = ggml_used_mem(ctx);
    return ScalarReal((double) used);
}

// ============================================================================
// Tensor Operations
// ============================================================================

SEXP R_ggml_new_tensor_1d(SEXP ctx_ptr, SEXP type, SEXP ne0) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }
    
    enum ggml_type dtype = (enum ggml_type) asInteger(type);
    int64_t n0 = (int64_t) asReal(ne0);
    
    // Проверка доступной памяти
    size_t type_size = ggml_type_size(dtype);
    size_t needed = n0 * type_size + ggml_tensor_overhead() + 256;
    size_t available = ggml_get_mem_size(ctx) - ggml_used_mem(ctx);
    
    if (needed > available) {
        error("Not enough memory in context: need %zu MB, available %zu MB",
              needed / (1024*1024), available / (1024*1024));
    }
    
    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx, dtype, n0);
    
    if (tensor == NULL) {
        error("Failed to create tensor");
    }
    
    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

SEXP R_ggml_new_tensor_2d(SEXP ctx_ptr, SEXP type, SEXP ne0, SEXP ne1) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }
    
    enum ggml_type dtype = (enum ggml_type) asInteger(type);
    int64_t n0 = (int64_t) asReal(ne0);
    int64_t n1 = (int64_t) asReal(ne1);
    
    // Проверка доступной памяти
    size_t type_size = ggml_type_size(dtype);
    size_t needed = n0 * n1 * type_size + ggml_tensor_overhead() + 256;
    size_t available = ggml_get_mem_size(ctx) - ggml_used_mem(ctx);
    
    if (needed > available) {
        error("Not enough memory in context: need %zu MB, available %zu MB",
              needed / (1024*1024), available / (1024*1024));
    }
    
    struct ggml_tensor * tensor = ggml_new_tensor_2d(ctx, dtype, n0, n1);
    
    if (tensor == NULL) {
        error("Failed to create tensor");
    }
    
    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

// ============================================================================
// Tensor Data Access
// ============================================================================

SEXP R_ggml_set_f32(SEXP tensor_ptr, SEXP data) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    
    if (tensor->type != GGML_TYPE_F32) {
        error("Tensor type must be F32");
    }
    
    int n = length(data);
    double *r_data = REAL(data);
    float *tensor_data = (float *) tensor->data;
    
    int64_t n_elements = ggml_nelements(tensor);
    if (n != n_elements) {
        error("Data length (%d) does not match tensor size (%lld)", n, (long long)n_elements);
    }
    
    for (int i = 0; i < n; i++) {
        tensor_data[i] = (float) r_data[i];
    }
    
    return R_NilValue;
}

SEXP R_ggml_get_f32(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    
    if (tensor->type != GGML_TYPE_F32) {
        error("Tensor type must be F32");
    }
    
    int64_t n_elements = ggml_nelements(tensor);
    SEXP result = PROTECT(allocVector(REALSXP, n_elements));
    
    float *tensor_data = (float *) tensor->data;
    double *r_data = REAL(result);
    
    for (int64_t i = 0; i < n_elements; i++) {
        r_data[i] = (double) tensor_data[i];
    }
    
    UNPROTECT(1);
    return result;
}

SEXP R_ggml_get_i32(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    if (tensor->type != GGML_TYPE_I32) {
        error("Tensor type must be I32 (got type %d)", tensor->type);
    }

    int64_t n_elements = ggml_nelements(tensor);
    SEXP result = PROTECT(allocVector(INTSXP, n_elements));

    int32_t *tensor_data = (int32_t *) tensor->data;
    int *r_data = INTEGER(result);

    for (int64_t i = 0; i < n_elements; i++) {
        r_data[i] = tensor_data[i];
    }

    UNPROTECT(1);
    return result;
}

SEXP R_ggml_set_i32(SEXP tensor_ptr, SEXP data) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    if (tensor->type != GGML_TYPE_I32) {
        error("Tensor type must be I32 (got type %d)", tensor->type);
    }

    int n = length(data);
    int *r_data = INTEGER(data);

    int64_t n_elements = ggml_nelements(tensor);
    if (n != n_elements) {
        error("Data length (%d) does not match tensor size (%lld)", n, (long long)n_elements);
    }

    for (int i = 0; i < n; i++) {
        ggml_set_i32_1d(tensor, i, (int32_t) r_data[i]);
    }

    return R_NilValue;
}

// ============================================================================
// Direct CPU Operations
// ============================================================================

SEXP R_ggml_cpu_add(SEXP a_ptr, SEXP b_ptr) {
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (a == NULL || b == NULL) {
        error("Invalid tensor pointer");
    }

    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        error("Both tensors must be F32");
    }

    int64_t n = ggml_nelements(a);
    if (n != ggml_nelements(b)) {
        error("Tensors must have same number of elements");
    }

    SEXP result = PROTECT(allocVector(REALSXP, n));

    float *a_data = (float *) a->data;
    float *b_data = (float *) b->data;
    double *r_data = REAL(result);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t i = 0; i < n; i++) {
        r_data[i] = (double)(a_data[i] + b_data[i]);
    }

    UNPROTECT(1);
    return result;
}

SEXP R_ggml_cpu_mul(SEXP a_ptr, SEXP b_ptr) {
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (a == NULL || b == NULL) {
        error("Invalid tensor pointer");
    }

    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        error("Both tensors must be F32");
    }

    int64_t n = ggml_nelements(a);
    if (n != ggml_nelements(b)) {
        error("Tensors must have same number of elements");
    }

    SEXP result = PROTECT(allocVector(REALSXP, n));

    float *a_data = (float *) a->data;
    float *b_data = (float *) b->data;
    double *r_data = REAL(result);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int64_t i = 0; i < n; i++) {
        r_data[i] = (double)(a_data[i] * b_data[i]);
    }

    UNPROTECT(1);
    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

SEXP R_ggml_version(void) {
    return mkString("0.9.5");
}

SEXP R_ggml_test(void) {
    Rprintf("GGML library loaded successfully!\n");
    Rprintf("GGML version: %s\n", "0.9.5");
    Rprintf("Tensor overhead: %zu bytes\n", ggml_tensor_overhead());
    return ScalarLogical(1);
}

SEXP R_ggml_nelements(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    
    int64_t n = ggml_nelements(tensor);
    return ScalarReal((double) n);
}

SEXP R_ggml_nbytes(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    
    size_t n = ggml_nbytes(tensor);
    return ScalarReal((double) n);
}

// ============================================================================
// Function Registration
// ============================================================================

// Forward declarations for graph operations (defined in r_interface_graph.c)
SEXP R_ggml_add(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_sub(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_mul(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_div(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_mul_mat(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_dup(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_add1(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_sgn(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_step(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_build_forward_expand(SEXP ctx_ptr, SEXP tensor_ptr);
SEXP R_ggml_graph_compute(SEXP ctx_ptr, SEXP graph_ptr);
SEXP R_ggml_graph_n_nodes(SEXP graph_ptr);
SEXP R_ggml_graph_print(SEXP graph_ptr);
SEXP R_ggml_graph_reset(SEXP graph_ptr);
SEXP R_ggml_graph_node(SEXP graph_ptr, SEXP i);
SEXP R_ggml_graph_overhead(void);
SEXP R_ggml_graph_get_tensor(SEXP graph_ptr, SEXP name);
SEXP R_ggml_relu(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_gelu(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_silu(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_tanh(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP eps);
SEXP R_ggml_rms_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP eps);
SEXP R_ggml_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP eps);
SEXP R_ggml_rms_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP eps);
SEXP R_ggml_soft_max(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_soft_max_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_soft_max_ext(SEXP ctx_ptr, SEXP a_ptr, SEXP mask_ptr, SEXP scale, SEXP max_bias);
SEXP R_ggml_transpose(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_sum(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_sum_rows(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_mean(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_argmax(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_repeat(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_sigmoid(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_gelu_quick(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_elu(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_leaky_relu(SEXP ctx_ptr, SEXP a_ptr, SEXP negative_slope, SEXP inplace);
SEXP R_ggml_hardswish(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_hardsigmoid(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_softplus(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_gelu_erf(SEXP ctx_ptr, SEXP a_ptr);
// View/Reshape
SEXP R_ggml_view_tensor(SEXP ctx_ptr, SEXP src_ptr);
SEXP R_ggml_reshape_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0);
SEXP R_ggml_reshape_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1);
SEXP R_ggml_reshape_3d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1, SEXP ne2);
SEXP R_ggml_reshape_4d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1, SEXP ne2, SEXP ne3);
SEXP R_ggml_permute(SEXP ctx_ptr, SEXP a_ptr, SEXP axis0, SEXP axis1, SEXP axis2, SEXP axis3);
SEXP R_ggml_cont(SEXP ctx_ptr, SEXP a_ptr);
// Tensor info
SEXP R_ggml_n_dims(SEXP tensor_ptr);
SEXP R_ggml_is_contiguous(SEXP tensor_ptr);
SEXP R_ggml_is_transposed(SEXP tensor_ptr);
SEXP R_ggml_is_permuted(SEXP tensor_ptr);
SEXP R_ggml_tensor_shape(SEXP tensor_ptr);
SEXP R_ggml_tensor_type(SEXP tensor_ptr);
// Mathematical operations
SEXP R_ggml_sqr(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_sqrt(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_log(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_exp(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_abs(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_neg(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_sin(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_cos(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_scale(SEXP ctx_ptr, SEXP a_ptr, SEXP s);
SEXP R_ggml_clamp(SEXP ctx_ptr, SEXP a_ptr, SEXP min_val, SEXP max_val);
SEXP R_ggml_floor(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_ceil(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_round(SEXP ctx_ptr, SEXP a_ptr);
// GLU (Gated Linear Unit) operations
SEXP R_ggml_glu(SEXP ctx_ptr, SEXP a_ptr, SEXP op, SEXP swapped);
SEXP R_ggml_reglu(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_geglu(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_swiglu(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_geglu_quick(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_glu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP op);
SEXP R_ggml_reglu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_geglu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_swiglu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);

// Row operations
SEXP R_ggml_get_rows(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);

// Diagonal masking
SEXP R_ggml_diag_mask_inf(SEXP ctx_ptr, SEXP a_ptr, SEXP n_past);
SEXP R_ggml_diag_mask_inf_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP n_past);
SEXP R_ggml_diag_mask_zero(SEXP ctx_ptr, SEXP a_ptr, SEXP n_past);

// RoPE (Rotary Position Embedding)
SEXP R_ggml_rope(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP n_dims, SEXP mode);
SEXP R_ggml_rope_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP n_dims, SEXP mode);
SEXP R_ggml_rope_ext(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                     SEXP n_dims, SEXP mode, SEXP n_ctx_orig,
                     SEXP freq_base, SEXP freq_scale, SEXP ext_factor,
                     SEXP attn_factor, SEXP beta_fast, SEXP beta_slow);

// Flash Attention
SEXP R_ggml_flash_attn_ext(SEXP ctx_ptr, SEXP q_ptr, SEXP k_ptr, SEXP v_ptr,
                           SEXP mask_ptr, SEXP scale, SEXP max_bias, SEXP logit_softcap);

// Graph compute with context
SEXP R_ggml_graph_compute_with_ctx(SEXP ctx_ptr, SEXP graph_ptr, SEXP n_threads);

// Graph dump to DOT
SEXP R_ggml_graph_dump_dot(SEXP graph_ptr, SEXP leafs_ptr, SEXP filename);

// Backend tensor access
SEXP R_ggml_backend_tensor_set(SEXP tensor_ptr, SEXP data, SEXP offset);
SEXP R_ggml_backend_tensor_get(SEXP tensor_ptr, SEXP offset, SEXP size);
SEXP R_ggml_backend_alloc_ctx_tensors(SEXP ctx_ptr, SEXP backend_ptr);

// Graph allocator (gallocr)
SEXP R_ggml_gallocr_new(void);
SEXP R_ggml_gallocr_new_buft(SEXP buft_ptr);
SEXP R_ggml_gallocr_free(SEXP galloc_ptr);
SEXP R_ggml_gallocr_reserve(SEXP galloc_ptr, SEXP graph_ptr);
SEXP R_ggml_gallocr_alloc_graph(SEXP galloc_ptr, SEXP graph_ptr);
SEXP R_ggml_gallocr_get_buffer_size(SEXP galloc_ptr, SEXP buffer_id);

// Backend buffer operations
SEXP R_ggml_backend_buffer_free(SEXP buffer_ptr);
SEXP R_ggml_backend_buffer_get_size(SEXP buffer_ptr);
SEXP R_ggml_backend_buffer_name(SEXP buffer_ptr);

// ============================================================================
// Context Management - Extended
// ============================================================================

SEXP R_ggml_set_no_alloc(SEXP ctx_ptr, SEXP no_alloc) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    bool val = asLogical(no_alloc);
    ggml_set_no_alloc(ctx, val);

    return R_NilValue;
}

SEXP R_ggml_get_no_alloc(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    bool val = ggml_get_no_alloc(ctx);
    return ScalarLogical(val);
}

SEXP R_ggml_get_max_tensor_size(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    size_t size = ggml_get_max_tensor_size(ctx);
    return ScalarReal((double) size);
}

SEXP R_ggml_print_objects(SEXP ctx_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    ggml_print_objects(ctx);
    return R_NilValue;
}

// ============================================================================
// Tensor Creation - Extended (3D, 4D, dup)
// ============================================================================

SEXP R_ggml_new_tensor_3d(SEXP ctx_ptr, SEXP type, SEXP ne0, SEXP ne1, SEXP ne2) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    enum ggml_type dtype = (enum ggml_type) asInteger(type);
    int64_t n0 = (int64_t) asReal(ne0);
    int64_t n1 = (int64_t) asReal(ne1);
    int64_t n2 = (int64_t) asReal(ne2);

    struct ggml_tensor * tensor = ggml_new_tensor_3d(ctx, dtype, n0, n1, n2);

    if (tensor == NULL) {
        error("Failed to create 3D tensor");
    }

    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

SEXP R_ggml_new_tensor_4d(SEXP ctx_ptr, SEXP type, SEXP ne0, SEXP ne1, SEXP ne2, SEXP ne3) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    enum ggml_type dtype = (enum ggml_type) asInteger(type);
    int64_t n0 = (int64_t) asReal(ne0);
    int64_t n1 = (int64_t) asReal(ne1);
    int64_t n2 = (int64_t) asReal(ne2);
    int64_t n3 = (int64_t) asReal(ne3);

    struct ggml_tensor * tensor = ggml_new_tensor_4d(ctx, dtype, n0, n1, n2, n3);

    if (tensor == NULL) {
        error("Failed to create 4D tensor");
    }

    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

SEXP R_ggml_dup_tensor(SEXP ctx_ptr, SEXP tensor_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);

    if (ctx == NULL || tensor == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);

    if (dup == NULL) {
        error("Failed to duplicate tensor");
    }

    return R_MakeExternalPtr(dup, R_NilValue, R_NilValue);
}

// Generic tensor creation with arbitrary dimensions
SEXP R_ggml_new_tensor(SEXP ctx_ptr, SEXP type, SEXP n_dims_sexp, SEXP ne_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    enum ggml_type dtype = (enum ggml_type) asInteger(type);
    int n_dims = asInteger(n_dims_sexp);

    if (n_dims < 1 || n_dims > 4) {
        error("n_dims must be between 1 and 4");
    }

    int64_t ne[4] = {1, 1, 1, 1};
    double *ne_r = REAL(ne_sexp);
    for (int i = 0; i < n_dims; i++) {
        ne[i] = (int64_t) ne_r[i];
    }

    struct ggml_tensor * tensor = ggml_new_tensor(ctx, dtype, n_dims, ne);

    if (tensor == NULL) {
        error("Failed to create tensor");
    }

    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

// Set all tensor elements to zero
SEXP R_ggml_set_zero(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    ggml_set_zero(tensor);
    return R_NilValue;
}

// ============================================================================
// Threading Control
// ============================================================================

#ifdef _OPENMP
#include <omp.h>
#endif

// Defined in r_interface_graph.c
extern void ggmlR_set_n_threads(int n);
extern int ggmlR_get_n_threads(void);

SEXP R_ggml_set_n_threads(SEXP n_threads) {
    int threads = asInteger(n_threads);

    if (threads < 1) {
        error("Number of threads must be at least 1");
    }

    // Set for OpenMP (general parallelism)
    #ifdef _OPENMP
    omp_set_num_threads(threads);
    #endif

    // Set for GGML backend (graph computation)
    ggmlR_set_n_threads(threads);

    return ScalarInteger(threads);
}

SEXP R_ggml_get_n_threads(void) {
    #ifdef _OPENMP
    int threads = omp_get_max_threads();
    #else
    int threads = 1;
    #endif
    
    return ScalarInteger(threads);
}

static const R_CallMethodDef CallEntries[] = {
    // Context management
    {"R_ggml_init",    (DL_FUNC) &R_ggml_init,    1},
    {"R_ggml_free",    (DL_FUNC) &R_ggml_free,    1},
    {"R_ggml_reset",   (DL_FUNC) &R_ggml_reset,   1},
    {"R_ggml_set_no_alloc",      (DL_FUNC) &R_ggml_set_no_alloc,      2},
    {"R_ggml_get_no_alloc",      (DL_FUNC) &R_ggml_get_no_alloc,      1},
    {"R_ggml_get_max_tensor_size", (DL_FUNC) &R_ggml_get_max_tensor_size, 1},
    {"R_ggml_print_objects",     (DL_FUNC) &R_ggml_print_objects,     1},

    // Time functions
    {"R_ggml_time_init",     (DL_FUNC) &R_ggml_time_init,     0},
    {"R_ggml_time_ms",       (DL_FUNC) &R_ggml_time_ms,       0},
    {"R_ggml_time_us",       (DL_FUNC) &R_ggml_time_us,       0},
    {"R_ggml_cycles",        (DL_FUNC) &R_ggml_cycles,        0},
    {"R_ggml_cycles_per_ms", (DL_FUNC) &R_ggml_cycles_per_ms, 0},

    // Memory info
    {"R_ggml_tensor_overhead", (DL_FUNC) &R_ggml_tensor_overhead, 0},
    {"R_ggml_get_mem_size",    (DL_FUNC) &R_ggml_get_mem_size,    1},
    {"R_ggml_used_mem",        (DL_FUNC) &R_ggml_used_mem,        1},

    // Tensor creation
    {"R_ggml_new_tensor",    (DL_FUNC) &R_ggml_new_tensor,    4},
    {"R_ggml_new_tensor_1d", (DL_FUNC) &R_ggml_new_tensor_1d, 3},
    {"R_ggml_new_tensor_2d", (DL_FUNC) &R_ggml_new_tensor_2d, 4},
    {"R_ggml_new_tensor_3d", (DL_FUNC) &R_ggml_new_tensor_3d, 5},
    {"R_ggml_new_tensor_4d", (DL_FUNC) &R_ggml_new_tensor_4d, 6},
    {"R_ggml_dup_tensor",    (DL_FUNC) &R_ggml_dup_tensor,    2},
    {"R_ggml_set_zero",      (DL_FUNC) &R_ggml_set_zero,      1},

    // Data access
    {"R_ggml_set_f32", (DL_FUNC) &R_ggml_set_f32, 2},
    {"R_ggml_get_f32", (DL_FUNC) &R_ggml_get_f32, 1},
    {"R_ggml_set_i32", (DL_FUNC) &R_ggml_set_i32, 2},
    {"R_ggml_get_i32", (DL_FUNC) &R_ggml_get_i32, 1},

    // CPU operations (direct)
    {"R_ggml_cpu_add", (DL_FUNC) &R_ggml_cpu_add, 2},
    {"R_ggml_cpu_mul", (DL_FUNC) &R_ggml_cpu_mul, 2},

    // Graph-based operations
    {"R_ggml_add",     (DL_FUNC) &R_ggml_add,     3},
    {"R_ggml_sub",     (DL_FUNC) &R_ggml_sub,     3},
    {"R_ggml_mul",     (DL_FUNC) &R_ggml_mul,     3},
    {"R_ggml_div",     (DL_FUNC) &R_ggml_div,     3},
    {"R_ggml_mul_mat", (DL_FUNC) &R_ggml_mul_mat, 3},
    {"R_ggml_dup",     (DL_FUNC) &R_ggml_dup,     2},
    {"R_ggml_add1",    (DL_FUNC) &R_ggml_add1,    3},
    {"R_ggml_sgn",     (DL_FUNC) &R_ggml_sgn,     2},
    {"R_ggml_step",    (DL_FUNC) &R_ggml_step,    2},

    // Graph building and execution
    {"R_ggml_build_forward_expand", (DL_FUNC) &R_ggml_build_forward_expand, 2},
    {"R_ggml_graph_compute",        (DL_FUNC) &R_ggml_graph_compute,        2},
    {"R_ggml_graph_n_nodes",        (DL_FUNC) &R_ggml_graph_n_nodes,        1},
    {"R_ggml_graph_print",          (DL_FUNC) &R_ggml_graph_print,          1},
    {"R_ggml_graph_reset",          (DL_FUNC) &R_ggml_graph_reset,          1},
    {"R_ggml_graph_node",           (DL_FUNC) &R_ggml_graph_node,           2},
    {"R_ggml_graph_overhead",       (DL_FUNC) &R_ggml_graph_overhead,       0},
    {"R_ggml_graph_get_tensor",     (DL_FUNC) &R_ggml_graph_get_tensor,     2},

    // Activation functions
    {"R_ggml_relu",     (DL_FUNC) &R_ggml_relu,     2},
    {"R_ggml_gelu",     (DL_FUNC) &R_ggml_gelu,     2},
    {"R_ggml_silu",     (DL_FUNC) &R_ggml_silu,     2},
    {"R_ggml_tanh",     (DL_FUNC) &R_ggml_tanh,     2},

    // Normalization
    {"R_ggml_norm",           (DL_FUNC) &R_ggml_norm,           3},
    {"R_ggml_rms_norm",       (DL_FUNC) &R_ggml_rms_norm,       3},
    {"R_ggml_norm_inplace",   (DL_FUNC) &R_ggml_norm_inplace,   3},
    {"R_ggml_rms_norm_inplace", (DL_FUNC) &R_ggml_rms_norm_inplace, 3},

    // Softmax
    {"R_ggml_soft_max",         (DL_FUNC) &R_ggml_soft_max,         2},
    {"R_ggml_soft_max_inplace", (DL_FUNC) &R_ggml_soft_max_inplace, 2},
    {"R_ggml_soft_max_ext",     (DL_FUNC) &R_ggml_soft_max_ext,     5},

    // Basic operations - extended
    {"R_ggml_transpose",  (DL_FUNC) &R_ggml_transpose,  2},
    {"R_ggml_sum",        (DL_FUNC) &R_ggml_sum,        2},
    {"R_ggml_sum_rows",   (DL_FUNC) &R_ggml_sum_rows,   2},
    {"R_ggml_mean",       (DL_FUNC) &R_ggml_mean,       2},
    {"R_ggml_argmax",     (DL_FUNC) &R_ggml_argmax,     2},
    {"R_ggml_repeat",     (DL_FUNC) &R_ggml_repeat,     3},

    // Additional activations
    {"R_ggml_sigmoid",    (DL_FUNC) &R_ggml_sigmoid,    2},
    {"R_ggml_gelu_quick", (DL_FUNC) &R_ggml_gelu_quick, 2},
    {"R_ggml_elu",        (DL_FUNC) &R_ggml_elu,        2},
    {"R_ggml_leaky_relu", (DL_FUNC) &R_ggml_leaky_relu, 4},
    {"R_ggml_hardswish",  (DL_FUNC) &R_ggml_hardswish,  2},
    {"R_ggml_hardsigmoid",(DL_FUNC) &R_ggml_hardsigmoid,2},
    {"R_ggml_softplus",   (DL_FUNC) &R_ggml_softplus,   2},
    {"R_ggml_gelu_erf",   (DL_FUNC) &R_ggml_gelu_erf,   2},

    // View/Reshape operations
    {"R_ggml_view_tensor", (DL_FUNC) &R_ggml_view_tensor, 2},
    {"R_ggml_reshape_1d",  (DL_FUNC) &R_ggml_reshape_1d,  3},
    {"R_ggml_reshape_2d",  (DL_FUNC) &R_ggml_reshape_2d,  4},
    {"R_ggml_reshape_3d",  (DL_FUNC) &R_ggml_reshape_3d,  5},
    {"R_ggml_reshape_4d",  (DL_FUNC) &R_ggml_reshape_4d,  6},
    {"R_ggml_permute",     (DL_FUNC) &R_ggml_permute,     6},
    {"R_ggml_cont",        (DL_FUNC) &R_ggml_cont,        2},

    // Tensor info
    {"R_ggml_n_dims",        (DL_FUNC) &R_ggml_n_dims,        1},
    {"R_ggml_is_contiguous", (DL_FUNC) &R_ggml_is_contiguous, 1},
    {"R_ggml_is_transposed", (DL_FUNC) &R_ggml_is_transposed, 1},
    {"R_ggml_is_permuted",   (DL_FUNC) &R_ggml_is_permuted,   1},
    {"R_ggml_tensor_shape",  (DL_FUNC) &R_ggml_tensor_shape,  1},
    {"R_ggml_tensor_type",   (DL_FUNC) &R_ggml_tensor_type,   1},

    // Utility
    {"R_ggml_version",   (DL_FUNC) &R_ggml_version,   0},
    {"R_ggml_test",      (DL_FUNC) &R_ggml_test,      0},
    {"R_ggml_nelements", (DL_FUNC) &R_ggml_nelements, 1},
    {"R_ggml_nbytes",    (DL_FUNC) &R_ggml_nbytes,    1},

    // Threading
    {"R_ggml_set_n_threads", (DL_FUNC) &R_ggml_set_n_threads, 1},
    {"R_ggml_get_n_threads", (DL_FUNC) &R_ggml_get_n_threads, 0},

    // Mathematical operations
    {"R_ggml_sqr",   (DL_FUNC) &R_ggml_sqr,   2},
    {"R_ggml_sqrt",  (DL_FUNC) &R_ggml_sqrt,  2},
    {"R_ggml_log",   (DL_FUNC) &R_ggml_log,   2},
    {"R_ggml_exp",   (DL_FUNC) &R_ggml_exp,   2},
    {"R_ggml_abs",   (DL_FUNC) &R_ggml_abs,   2},
    {"R_ggml_neg",   (DL_FUNC) &R_ggml_neg,   2},
    {"R_ggml_sin",   (DL_FUNC) &R_ggml_sin,   2},
    {"R_ggml_cos",   (DL_FUNC) &R_ggml_cos,   2},
    {"R_ggml_scale", (DL_FUNC) &R_ggml_scale, 3},
    {"R_ggml_clamp", (DL_FUNC) &R_ggml_clamp, 4},
    {"R_ggml_floor", (DL_FUNC) &R_ggml_floor, 2},
    {"R_ggml_ceil",  (DL_FUNC) &R_ggml_ceil,  2},
    {"R_ggml_round", (DL_FUNC) &R_ggml_round, 2},

    // GLU (Gated Linear Unit) operations
    {"R_ggml_glu",          (DL_FUNC) &R_ggml_glu,          4},
    {"R_ggml_reglu",        (DL_FUNC) &R_ggml_reglu,        2},
    {"R_ggml_geglu",        (DL_FUNC) &R_ggml_geglu,        2},
    {"R_ggml_swiglu",       (DL_FUNC) &R_ggml_swiglu,       2},
    {"R_ggml_geglu_quick",  (DL_FUNC) &R_ggml_geglu_quick,  2},
    {"R_ggml_glu_split",    (DL_FUNC) &R_ggml_glu_split,    4},
    {"R_ggml_reglu_split",  (DL_FUNC) &R_ggml_reglu_split,  3},
    {"R_ggml_geglu_split",  (DL_FUNC) &R_ggml_geglu_split,  3},
    {"R_ggml_swiglu_split", (DL_FUNC) &R_ggml_swiglu_split, 3},

    // Row operations
    {"R_ggml_get_rows",     (DL_FUNC) &R_ggml_get_rows,     3},

    // Diagonal masking (for causal attention)
    {"R_ggml_diag_mask_inf",         (DL_FUNC) &R_ggml_diag_mask_inf,         3},
    {"R_ggml_diag_mask_inf_inplace", (DL_FUNC) &R_ggml_diag_mask_inf_inplace, 3},
    {"R_ggml_diag_mask_zero",        (DL_FUNC) &R_ggml_diag_mask_zero,        3},

    // RoPE (Rotary Position Embedding)
    {"R_ggml_rope",         (DL_FUNC) &R_ggml_rope,         5},
    {"R_ggml_rope_inplace", (DL_FUNC) &R_ggml_rope_inplace, 5},
    {"R_ggml_rope_ext",     (DL_FUNC) &R_ggml_rope_ext,     13},

    // Flash Attention
    {"R_ggml_flash_attn_ext", (DL_FUNC) &R_ggml_flash_attn_ext, 8},

    // Graph compute with context
    {"R_ggml_graph_compute_with_ctx", (DL_FUNC) &R_ggml_graph_compute_with_ctx, 3},

    // Graph dump to DOT
    {"R_ggml_graph_dump_dot", (DL_FUNC) &R_ggml_graph_dump_dot, 3},

    // Backend tensor access
    {"R_ggml_backend_tensor_set",      (DL_FUNC) &R_ggml_backend_tensor_set,      3},
    {"R_ggml_backend_tensor_get",      (DL_FUNC) &R_ggml_backend_tensor_get,      3},
    {"R_ggml_backend_alloc_ctx_tensors", (DL_FUNC) &R_ggml_backend_alloc_ctx_tensors, 2},

    // Graph allocator (gallocr)
    {"R_ggml_gallocr_new",             (DL_FUNC) &R_ggml_gallocr_new,             0},
    {"R_ggml_gallocr_new_buft",        (DL_FUNC) &R_ggml_gallocr_new_buft,        1},
    {"R_ggml_gallocr_free",            (DL_FUNC) &R_ggml_gallocr_free,            1},
    {"R_ggml_gallocr_reserve",         (DL_FUNC) &R_ggml_gallocr_reserve,         2},
    {"R_ggml_gallocr_alloc_graph",     (DL_FUNC) &R_ggml_gallocr_alloc_graph,     2},
    {"R_ggml_gallocr_get_buffer_size", (DL_FUNC) &R_ggml_gallocr_get_buffer_size, 2},

    // Backend buffer operations
    {"R_ggml_backend_buffer_free",     (DL_FUNC) &R_ggml_backend_buffer_free,     1},
    {"R_ggml_backend_buffer_get_size", (DL_FUNC) &R_ggml_backend_buffer_get_size, 1},
    {"R_ggml_backend_buffer_name",     (DL_FUNC) &R_ggml_backend_buffer_name,     1},

    {NULL, NULL, 0}
};

void R_init_ggmlR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
