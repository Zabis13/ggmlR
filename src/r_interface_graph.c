#include <R.h>
#include <Rinternals.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#ifdef _OPENMP
#undef match  // R defines 'match' macro that conflicts with OpenMP pragma
#include <omp.h>
#endif

// ============================================================================
// Graph-based Operations - These create computation nodes
// ============================================================================

SEXP R_ggml_add(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    
    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    // Create computation node (does NOT execute yet)
    struct ggml_tensor * result = ggml_add(ctx, a, b);
    
    if (result == NULL) {
        error("Failed to create add operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sub(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    
    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_sub(ctx, a, b);
    
    if (result == NULL) {
        error("Failed to create sub operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_mul(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    
    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_mul(ctx, a, b);
    
    if (result == NULL) {
        error("Failed to create mul operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_div(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    
    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_div(ctx, a, b);
    
    if (result == NULL) {
        error("Failed to create div operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_mul_mat(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    // Matrix multiplication: result = a * b
    struct ggml_tensor * result = ggml_mul_mat(ctx, a, b);

    if (result == NULL) {
        error("Failed to create mul_mat operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ggml_dup - Copy tensor (graph operation)
SEXP R_ggml_dup(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_dup(ctx, a);

    if (result == NULL) {
        error("Failed to create dup operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ggml_add1 - Add scalar (1-element tensor) to tensor
SEXP R_ggml_add1(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_add1(ctx, a, b);

    if (result == NULL) {
        error("Failed to create add1 operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ggml_sgn - Sign function: sgn(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0
SEXP R_ggml_sgn(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_sgn(ctx, a);

    if (result == NULL) {
        error("Failed to create sgn operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ggml_step - Step function: step(x) = 0 if x <= 0, 1 if x > 0
SEXP R_ggml_step(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_step(ctx, a);

    if (result == NULL) {
        error("Failed to create step operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Graph Building and Execution
// ============================================================================

SEXP R_ggml_build_forward_expand(SEXP ctx_ptr, SEXP tensor_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    
    if (ctx == NULL || tensor == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    // Create computation graph
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    
    if (graph == NULL) {
        error("Failed to create computation graph");
    }
    
    // Build forward pass by expanding from the output tensor
    ggml_build_forward_expand(graph, tensor);
    
    return R_MakeExternalPtr(graph, R_NilValue, R_NilValue);
}

// Global thread count for backend (default: use all available via OpenMP)
static int g_n_threads = 0;

void ggmlR_set_n_threads(int n) {
    g_n_threads = n;
}

int ggmlR_get_n_threads(void) {
    if (g_n_threads <= 0) {
        #ifdef _OPENMP
        return omp_get_max_threads();
        #else
        return 1;
        #endif
    }
    return g_n_threads;
}

SEXP R_ggml_graph_compute(SEXP ctx_ptr, SEXP graph_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (ctx == NULL || graph == NULL) {
        error("Invalid pointer (context or graph is NULL)");
    }

    // Create CPU backend for computation
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == NULL) {
        error("Failed to initialize CPU backend");
    }

    // Set number of threads for the backend
    int n_threads = ggmlR_get_n_threads();
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    // Compute the graph
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    // Free backend
    ggml_backend_free(backend);

    if (status != GGML_STATUS_SUCCESS) {
        error("Graph computation failed with status: %d", status);
    }

    return R_NilValue;
}

// ============================================================================
// Graph Information Functions
// ============================================================================

SEXP R_ggml_graph_n_nodes(SEXP graph_ptr) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    
    if (graph == NULL) {
        error("Invalid graph pointer");
    }
    
    int n_nodes = ggml_graph_n_nodes(graph);
    return ScalarInteger(n_nodes);
}

SEXP R_ggml_graph_print(SEXP graph_ptr) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    ggml_graph_print(graph);
    return R_NilValue;
}

SEXP R_ggml_graph_reset(SEXP graph_ptr) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    ggml_graph_reset(graph);
    return R_NilValue;
}

SEXP R_ggml_graph_node(SEXP graph_ptr, SEXP i) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    int index = asInteger(i);
    struct ggml_tensor * node = ggml_graph_node(graph, index);

    if (node == NULL) {
        error("Invalid node index");
    }

    return R_MakeExternalPtr(node, R_NilValue, R_NilValue);
}

SEXP R_ggml_graph_overhead(void) {
    size_t overhead = ggml_graph_overhead();
    return ScalarReal((double) overhead);
}

SEXP R_ggml_graph_get_tensor(SEXP graph_ptr, SEXP name) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    const char * tensor_name = CHAR(STRING_ELT(name, 0));
    struct ggml_tensor * tensor = ggml_graph_get_tensor(graph, tensor_name);

    if (tensor == NULL) {
        return R_NilValue;  // Return NULL if not found
    }

    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

// ============================================================================
// Activation Functions
// ============================================================================

SEXP R_ggml_relu(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    
    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_relu(ctx, a);
    
    if (result == NULL) {
        error("Failed to create relu operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_gelu(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    
    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_gelu(ctx, a);
    
    if (result == NULL) {
        error("Failed to create gelu operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_silu(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    
    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_silu(ctx, a);
    
    if (result == NULL) {
        error("Failed to create silu operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_tanh(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    
    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_tanh(ctx, a);
    
    if (result == NULL) {
        error("Failed to create tanh operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Normalization Functions  
// ============================================================================

SEXP R_ggml_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float epsilon = (float) asReal(eps);
    
    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }
    
    struct ggml_tensor * result = ggml_norm(ctx, a, epsilon);
    
    if (result == NULL) {
        error("Failed to create norm operation");
    }
    
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_rms_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_rms_norm(ctx, a, epsilon);

    if (result == NULL) {
        error("Failed to create rms_norm operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_norm_inplace(ctx, a, epsilon);

    if (result == NULL) {
        error("Failed to create norm_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_rms_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_rms_norm_inplace(ctx, a, epsilon);

    if (result == NULL) {
        error("Failed to create rms_norm_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Group Normalization - used in stable-diffusion
SEXP R_ggml_group_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP n_groups_sexp, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    int n_groups = asInteger(n_groups_sexp);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_group_norm(ctx, a, n_groups, epsilon);

    if (result == NULL) {
        error("Failed to create group_norm operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_group_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP n_groups_sexp, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    int n_groups = asInteger(n_groups_sexp);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_group_norm_inplace(ctx, a, n_groups, epsilon);

    if (result == NULL) {
        error("Failed to create group_norm_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// L2 Normalization - used in RWKV v7
SEXP R_ggml_l2_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_l2_norm(ctx, a, epsilon);

    if (result == NULL) {
        error("Failed to create l2_norm operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_l2_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_l2_norm_inplace(ctx, a, epsilon);

    if (result == NULL) {
        error("Failed to create l2_norm_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// RMS Norm backward - for training
SEXP R_ggml_rms_norm_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP eps) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    float epsilon = (float) asReal(eps);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_rms_norm_back(ctx, a, b, epsilon);

    if (result == NULL) {
        error("Failed to create rms_norm_back operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Softmax
// ============================================================================

SEXP R_ggml_soft_max(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_soft_max(ctx, a);

    if (result == NULL) {
        error("Failed to create softmax operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_soft_max_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_soft_max_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create soft_max_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Extended softmax: fused soft_max(a*scale + mask*(ALiBi slope))
// mask: optional attention mask (F16 or F32), NULL for no mask
// scale: scaling factor (usually 1/sqrt(head_dim))
// max_bias: maximum ALiBi bias, 0.0 for no ALiBi
SEXP R_ggml_soft_max_ext(SEXP ctx_ptr, SEXP a_ptr, SEXP mask_ptr,
                          SEXP scale_sexp, SEXP max_bias_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * mask = (mask_ptr == R_NilValue) ? NULL :
                                (struct ggml_tensor *) R_ExternalPtrAddr(mask_ptr);
    float scale = (float) asReal(scale_sexp);
    float max_bias = (float) asReal(max_bias_sexp);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);

    if (result == NULL) {
        error("Failed to create soft_max_ext operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Extended softmax inplace (returns view of a)
SEXP R_ggml_soft_max_ext_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP mask_ptr,
                                  SEXP scale_sexp, SEXP max_bias_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * mask = (mask_ptr == R_NilValue) ? NULL :
                                (struct ggml_tensor *) R_ExternalPtrAddr(mask_ptr);
    float scale = (float) asReal(scale_sexp);
    float max_bias = (float) asReal(max_bias_sexp);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_soft_max_ext_inplace(ctx, a, mask, scale, max_bias);

    if (result == NULL) {
        error("Failed to create soft_max_ext_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Basic Operations - Extended
// ============================================================================

SEXP R_ggml_transpose(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_transpose(ctx, a);

    if (result == NULL) {
        error("Failed to create transpose operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sum(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_sum(ctx, a);

    if (result == NULL) {
        error("Failed to create sum operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sum_rows(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_sum_rows(ctx, a);

    if (result == NULL) {
        error("Failed to create sum_rows operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_mean(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_mean(ctx, a);

    if (result == NULL) {
        error("Failed to create mean operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_argmax(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_argmax(ctx, a);

    if (result == NULL) {
        error("Failed to create argmax operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_repeat(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_repeat(ctx, a, b);

    if (result == NULL) {
        error("Failed to create repeat operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Additional Activations
// ============================================================================

SEXP R_ggml_sigmoid(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_sigmoid(ctx, a);

    if (result == NULL) {
        error("Failed to create sigmoid operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_gelu_quick(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_gelu_quick(ctx, a);

    if (result == NULL) {
        error("Failed to create gelu_quick operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_elu(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_elu(ctx, a);

    if (result == NULL) {
        error("Failed to create elu operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_leaky_relu(SEXP ctx_ptr, SEXP a_ptr, SEXP negative_slope_sexp, SEXP inplace_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float negative_slope = (float) asReal(negative_slope_sexp);
    bool inplace = asLogical(inplace_sexp);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_leaky_relu(ctx, a, negative_slope, inplace);

    if (result == NULL) {
        error("Failed to create leaky_relu operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_hardswish(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_hardswish(ctx, a);

    if (result == NULL) {
        error("Failed to create hardswish operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_hardsigmoid(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_hardsigmoid(ctx, a);

    if (result == NULL) {
        error("Failed to create hardsigmoid operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_softplus(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_softplus(ctx, a);

    if (result == NULL) {
        error("Failed to create softplus operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_gelu_erf(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_gelu_erf(ctx, a);

    if (result == NULL) {
        error("Failed to create gelu_erf operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// View/Reshape Operations
// ============================================================================

SEXP R_ggml_view_tensor(SEXP ctx_ptr, SEXP src_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * src = (struct ggml_tensor *) R_ExternalPtrAddr(src_ptr);

    if (ctx == NULL || src == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, src);

    if (result == NULL) {
        error("Failed to create view tensor");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_reshape_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t n0 = (int64_t) asReal(ne0);
    struct ggml_tensor * result = ggml_reshape_1d(ctx, a, n0);

    if (result == NULL) {
        error("Failed to reshape to 1D");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_reshape_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t n0 = (int64_t) asReal(ne0);
    int64_t n1 = (int64_t) asReal(ne1);
    struct ggml_tensor * result = ggml_reshape_2d(ctx, a, n0, n1);

    if (result == NULL) {
        error("Failed to reshape to 2D");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_reshape_3d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1, SEXP ne2) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t n0 = (int64_t) asReal(ne0);
    int64_t n1 = (int64_t) asReal(ne1);
    int64_t n2 = (int64_t) asReal(ne2);
    struct ggml_tensor * result = ggml_reshape_3d(ctx, a, n0, n1, n2);

    if (result == NULL) {
        error("Failed to reshape to 3D");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_reshape_4d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1, SEXP ne2, SEXP ne3) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t n0 = (int64_t) asReal(ne0);
    int64_t n1 = (int64_t) asReal(ne1);
    int64_t n2 = (int64_t) asReal(ne2);
    int64_t n3 = (int64_t) asReal(ne3);
    struct ggml_tensor * result = ggml_reshape_4d(ctx, a, n0, n1, n2, n3);

    if (result == NULL) {
        error("Failed to reshape to 4D");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_permute(SEXP ctx_ptr, SEXP a_ptr, SEXP axis0, SEXP axis1, SEXP axis2, SEXP axis3) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int ax0 = asInteger(axis0);
    int ax1 = asInteger(axis1);
    int ax2 = asInteger(axis2);
    int ax3 = asInteger(axis3);

    struct ggml_tensor * result = ggml_permute(ctx, a, ax0, ax1, ax2, ax3);

    if (result == NULL) {
        error("Failed to permute tensor");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_cont(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_cont(ctx, a);

    if (result == NULL) {
        error("Failed to make contiguous");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Tensor Info Functions
// ============================================================================

SEXP R_ggml_n_dims(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    int n = ggml_n_dims(tensor);
    return ScalarInteger(n);
}

SEXP R_ggml_is_contiguous(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    bool result = ggml_is_contiguous(tensor);
    return ScalarLogical(result);
}

SEXP R_ggml_is_transposed(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    bool result = ggml_is_transposed(tensor);
    return ScalarLogical(result);
}

SEXP R_ggml_is_permuted(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    bool result = ggml_is_permuted(tensor);
    return ScalarLogical(result);
}

SEXP R_ggml_tensor_shape(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    SEXP result = PROTECT(allocVector(REALSXP, 4));
    double * data = REAL(result);
    data[0] = (double) tensor->ne[0];
    data[1] = (double) tensor->ne[1];
    data[2] = (double) tensor->ne[2];
    data[3] = (double) tensor->ne[3];

    UNPROTECT(1);
    return result;
}

SEXP R_ggml_tensor_type(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    return ScalarInteger((int) tensor->type);
}

// ============================================================================
// Mathematical Operations
// ============================================================================

SEXP R_ggml_sqr(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_sqr(ctx, a);

    if (result == NULL) {
        error("Failed to create sqr operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sqrt(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_sqrt(ctx, a);

    if (result == NULL) {
        error("Failed to create sqrt operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_log(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_log(ctx, a);

    if (result == NULL) {
        error("Failed to create log operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_exp(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_exp(ctx, a);

    if (result == NULL) {
        error("Failed to create exp operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_abs(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_abs(ctx, a);

    if (result == NULL) {
        error("Failed to create abs operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_neg(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_neg(ctx, a);

    if (result == NULL) {
        error("Failed to create neg operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sin(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_sin(ctx, a);

    if (result == NULL) {
        error("Failed to create sin operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_cos(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_cos(ctx, a);

    if (result == NULL) {
        error("Failed to create cos operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_scale(SEXP ctx_ptr, SEXP a_ptr, SEXP s) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float scale = (float) asReal(s);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_scale(ctx, a, scale);

    if (result == NULL) {
        error("Failed to create scale operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_clamp(SEXP ctx_ptr, SEXP a_ptr, SEXP min_val, SEXP max_val) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    float minv = (float) asReal(min_val);
    float maxv = (float) asReal(max_val);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_clamp(ctx, a, minv, maxv);

    if (result == NULL) {
        error("Failed to create clamp operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_floor(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_floor(ctx, a);

    if (result == NULL) {
        error("Failed to create floor operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_ceil(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_ceil(ctx, a);

    if (result == NULL) {
        error("Failed to create ceil operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_round(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_round(ctx, a);

    if (result == NULL) {
        error("Failed to create round operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// GLU (Gated Linear Unit) Operations
// ============================================================================

SEXP R_ggml_glu(SEXP ctx_ptr, SEXP a_ptr, SEXP op_sexp, SEXP swapped_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    enum ggml_glu_op op = (enum ggml_glu_op) asInteger(op_sexp);
    bool swapped = asLogical(swapped_sexp);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_glu(ctx, a, op, swapped);

    if (result == NULL) {
        error("Failed to create GLU operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_reglu(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_reglu(ctx, a);

    if (result == NULL) {
        error("Failed to create ReGLU operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_geglu(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_geglu(ctx, a);

    if (result == NULL) {
        error("Failed to create GeGLU operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_swiglu(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_swiglu(ctx, a);

    if (result == NULL) {
        error("Failed to create SwiGLU operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_geglu_quick(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_geglu_quick(ctx, a);

    if (result == NULL) {
        error("Failed to create GeGLU quick operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Split variants - separate gate and input tensors

SEXP R_ggml_glu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP op_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    enum ggml_glu_op op = (enum ggml_glu_op) asInteger(op_sexp);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_glu_split(ctx, a, b, op);

    if (result == NULL) {
        error("Failed to create GLU split operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_reglu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_reglu_split(ctx, a, b);

    if (result == NULL) {
        error("Failed to create ReGLU split operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_geglu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_geglu_split(ctx, a, b);

    if (result == NULL) {
        error("Failed to create GeGLU split operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_swiglu_split(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_swiglu_split(ctx, a, b);

    if (result == NULL) {
        error("Failed to create SwiGLU split operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Row Operations
// ============================================================================

// Get rows from tensor by indices
// a: data tensor [n_embd, n_rows, ...]
// b: indices tensor (int32) [n_indices]
// Returns: [n_embd, n_indices, ...]
SEXP R_ggml_get_rows(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_get_rows(ctx, a, b);

    if (result == NULL) {
        error("Failed to create get_rows operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Diagonal Masking Operations (for causal attention)
// ============================================================================

// Set elements above the diagonal to -INF
// n_past: number of past tokens (shifts the diagonal)
SEXP R_ggml_diag_mask_inf(SEXP ctx_ptr, SEXP a_ptr, SEXP n_past_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    int n_past = asInteger(n_past_sexp);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_diag_mask_inf(ctx, a, n_past);

    if (result == NULL) {
        error("Failed to create diag_mask_inf operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// In-place version - returns view(a)
SEXP R_ggml_diag_mask_inf_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP n_past_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    int n_past = asInteger(n_past_sexp);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_diag_mask_inf_inplace(ctx, a, n_past);

    if (result == NULL) {
        error("Failed to create diag_mask_inf_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Set elements above the diagonal to 0
SEXP R_ggml_diag_mask_zero(SEXP ctx_ptr, SEXP a_ptr, SEXP n_past_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    int n_past = asInteger(n_past_sexp);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_diag_mask_zero(ctx, a, n_past);

    if (result == NULL) {
        error("Failed to create diag_mask_zero operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// RoPE - Rotary Position Embedding
// ============================================================================

// Basic RoPE
// a: input tensor [n_embd, n_head, n_tokens, batch]
// b: position tensor (int32) [n_tokens]
// n_dims: number of dimensions to rotate (usually n_embd / n_head)
// mode: RoPE mode (0 = normal, 1 = neox style, etc.)
SEXP R_ggml_rope(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP n_dims_sexp, SEXP mode_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    int n_dims = asInteger(n_dims_sexp);
    int mode = asInteger(mode_sexp);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_rope(ctx, a, b, n_dims, mode);

    if (result == NULL) {
        error("Failed to create rope operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// In-place RoPE
SEXP R_ggml_rope_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP n_dims_sexp, SEXP mode_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    int n_dims = asInteger(n_dims_sexp);
    int mode = asInteger(mode_sexp);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_rope_inplace(ctx, a, b, n_dims, mode);

    if (result == NULL) {
        error("Failed to create rope_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Extended RoPE with frequency scaling (for context extension)
// c: optional frequency factors tensor (can be NULL)
// freq_base: base frequency (default 10000)
// freq_scale: frequency scale factor (1.0 = no scaling)
// ext_factor: extension factor for YaRN
// attn_factor: attention scale factor
// beta_fast, beta_slow: YaRN parameters
SEXP R_ggml_rope_ext(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                     SEXP n_dims_sexp, SEXP mode_sexp, SEXP n_ctx_orig_sexp,
                     SEXP freq_base_sexp, SEXP freq_scale_sexp,
                     SEXP ext_factor_sexp, SEXP attn_factor_sexp,
                     SEXP beta_fast_sexp, SEXP beta_slow_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = (c_ptr == R_NilValue) ? NULL :
                             (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);

    int n_dims = asInteger(n_dims_sexp);
    int mode = asInteger(mode_sexp);
    int n_ctx_orig = asInteger(n_ctx_orig_sexp);
    float freq_base = (float) asReal(freq_base_sexp);
    float freq_scale = (float) asReal(freq_scale_sexp);
    float ext_factor = (float) asReal(ext_factor_sexp);
    float attn_factor = (float) asReal(attn_factor_sexp);
    float beta_fast = (float) asReal(beta_fast_sexp);
    float beta_slow = (float) asReal(beta_slow_sexp);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_rope_ext(ctx, a, b, c, n_dims, mode, n_ctx_orig,
                                                 freq_base, freq_scale, ext_factor,
                                                 attn_factor, beta_fast, beta_slow);

    if (result == NULL) {
        error("Failed to create rope_ext operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Extended RoPE inplace (returns view of a)
SEXP R_ggml_rope_ext_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                              SEXP n_dims_sexp, SEXP mode_sexp, SEXP n_ctx_orig_sexp,
                              SEXP freq_base_sexp, SEXP freq_scale_sexp,
                              SEXP ext_factor_sexp, SEXP attn_factor_sexp,
                              SEXP beta_fast_sexp, SEXP beta_slow_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = (c_ptr == R_NilValue) ? NULL :
                             (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);

    int n_dims = asInteger(n_dims_sexp);
    int mode = asInteger(mode_sexp);
    int n_ctx_orig = asInteger(n_ctx_orig_sexp);
    float freq_base = (float) asReal(freq_base_sexp);
    float freq_scale = (float) asReal(freq_scale_sexp);
    float ext_factor = (float) asReal(ext_factor_sexp);
    float attn_factor = (float) asReal(attn_factor_sexp);
    float beta_fast = (float) asReal(beta_fast_sexp);
    float beta_slow = (float) asReal(beta_slow_sexp);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_rope_ext_inplace(ctx, a, b, c, n_dims, mode, n_ctx_orig,
                                                         freq_base, freq_scale, ext_factor,
                                                         attn_factor, beta_fast, beta_slow);

    if (result == NULL) {
        error("Failed to create rope_ext_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Multi-rope (MRoPE) for vision models
SEXP R_ggml_rope_multi(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                        SEXP n_dims_sexp, SEXP sections_sexp, SEXP mode_sexp,
                        SEXP n_ctx_orig_sexp, SEXP freq_base_sexp, SEXP freq_scale_sexp,
                        SEXP ext_factor_sexp, SEXP attn_factor_sexp,
                        SEXP beta_fast_sexp, SEXP beta_slow_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = (c_ptr == R_NilValue) ? NULL :
                             (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);

    int n_dims = asInteger(n_dims_sexp);
    int mode = asInteger(mode_sexp);
    int n_ctx_orig = asInteger(n_ctx_orig_sexp);
    float freq_base = (float) asReal(freq_base_sexp);
    float freq_scale = (float) asReal(freq_scale_sexp);
    float ext_factor = (float) asReal(ext_factor_sexp);
    float attn_factor = (float) asReal(attn_factor_sexp);
    float beta_fast = (float) asReal(beta_fast_sexp);
    float beta_slow = (float) asReal(beta_slow_sexp);

    // Parse sections array (must be length 4)
    int sections[4] = {0, 0, 0, 0};
    if (sections_sexp != R_NilValue && LENGTH(sections_sexp) >= 4) {
        int * sec_ptr = INTEGER(sections_sexp);
        for (int i = 0; i < 4; i++) {
            sections[i] = sec_ptr[i];
        }
    }

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_rope_multi(ctx, a, b, c, n_dims, sections, mode,
                                                   n_ctx_orig, freq_base, freq_scale,
                                                   ext_factor, attn_factor, beta_fast, beta_slow);

    if (result == NULL) {
        error("Failed to create rope_multi operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Multi-rope inplace
SEXP R_ggml_rope_multi_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                                SEXP n_dims_sexp, SEXP sections_sexp, SEXP mode_sexp,
                                SEXP n_ctx_orig_sexp, SEXP freq_base_sexp, SEXP freq_scale_sexp,
                                SEXP ext_factor_sexp, SEXP attn_factor_sexp,
                                SEXP beta_fast_sexp, SEXP beta_slow_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = (c_ptr == R_NilValue) ? NULL :
                             (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);

    int n_dims = asInteger(n_dims_sexp);
    int mode = asInteger(mode_sexp);
    int n_ctx_orig = asInteger(n_ctx_orig_sexp);
    float freq_base = (float) asReal(freq_base_sexp);
    float freq_scale = (float) asReal(freq_scale_sexp);
    float ext_factor = (float) asReal(ext_factor_sexp);
    float attn_factor = (float) asReal(attn_factor_sexp);
    float beta_fast = (float) asReal(beta_fast_sexp);
    float beta_slow = (float) asReal(beta_slow_sexp);

    int sections[4] = {0, 0, 0, 0};
    if (sections_sexp != R_NilValue && LENGTH(sections_sexp) >= 4) {
        int * sec_ptr = INTEGER(sections_sexp);
        for (int i = 0; i < 4; i++) {
            sections[i] = sec_ptr[i];
        }
    }

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_rope_multi_inplace(ctx, a, b, c, n_dims, sections, mode,
                                                           n_ctx_orig, freq_base, freq_scale,
                                                           ext_factor, attn_factor, beta_fast, beta_slow);

    if (result == NULL) {
        error("Failed to create rope_multi_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Graph Compute with Context
// ============================================================================

// Compute graph using context-based allocation (legacy method)
// Uses ggml_graph_compute() with ggml_cplan
SEXP R_ggml_graph_compute_with_ctx(SEXP ctx_ptr, SEXP graph_ptr, SEXP n_threads_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (ctx == NULL || graph == NULL) {
        error("Invalid pointer (context or graph is NULL)");
    }

    int n_threads = asInteger(n_threads_sexp);
    if (n_threads <= 0) {
        n_threads = ggmlR_get_n_threads();
    }

    // Create computation plan (threadpool = NULL uses internal threads)
    struct ggml_cplan cplan = ggml_graph_plan(graph, n_threads, NULL);

    // Allocate work buffer if needed
    if (cplan.work_size > 0) {
        cplan.work_data = (uint8_t *) malloc(cplan.work_size);
        if (cplan.work_data == NULL) {
            error("Failed to allocate work buffer (%zu bytes)", cplan.work_size);
        }
    }

    // Compute the graph
    enum ggml_status status = ggml_graph_compute(graph, &cplan);

    // Free work buffer
    if (cplan.work_data != NULL) {
        free(cplan.work_data);
    }

    if (status != GGML_STATUS_SUCCESS) {
        error("Graph computation failed with status: %d", status);
    }

    return R_NilValue;
}

// ============================================================================
// Graph Dump to DOT format
// ============================================================================

// Export graph to DOT format for visualization
SEXP R_ggml_graph_dump_dot(SEXP graph_ptr, SEXP leafs_ptr, SEXP filename_sexp) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    struct ggml_cgraph * leafs = (leafs_ptr == R_NilValue) ? NULL :
                                  (struct ggml_cgraph *) R_ExternalPtrAddr(leafs_ptr);

    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    const char * filename = CHAR(STRING_ELT(filename_sexp, 0));

    ggml_graph_dump_dot(graph, leafs, filename);

    return R_NilValue;
}

// ============================================================================
// Backend Tensor Data Access
// ============================================================================

// Set tensor data from R vector (works with any backend)
SEXP R_ggml_backend_tensor_set(SEXP tensor_ptr, SEXP data_sexp, SEXP offset_sexp) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);

    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    size_t offset = (size_t) asReal(offset_sexp);

    // Determine the data type and size
    if (tensor->type == GGML_TYPE_F32) {
        int n = length(data_sexp);
        size_t size = n * sizeof(float);

        // Convert R doubles to float
        float * buffer = (float *) malloc(size);
        if (buffer == NULL) {
            error("Failed to allocate buffer");
        }

        double * r_data = REAL(data_sexp);
        for (int i = 0; i < n; i++) {
            buffer[i] = (float) r_data[i];
        }

        ggml_backend_tensor_set(tensor, buffer, offset, size);
        free(buffer);
    } else if (tensor->type == GGML_TYPE_I32) {
        int n = length(data_sexp);
        size_t size = n * sizeof(int32_t);

        int * r_data = INTEGER(data_sexp);
        ggml_backend_tensor_set(tensor, r_data, offset, size);
    } else {
        error("Unsupported tensor type for ggml_backend_tensor_set");
    }

    return R_NilValue;
}

// Get tensor data to R vector (works with any backend)
SEXP R_ggml_backend_tensor_get(SEXP tensor_ptr, SEXP offset_sexp, SEXP size_sexp) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);

    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    size_t offset = (size_t) asReal(offset_sexp);
    int64_t n_elements = (size_sexp == R_NilValue) ?
                         ggml_nelements(tensor) : (int64_t) asReal(size_sexp);

    if (tensor->type == GGML_TYPE_F32) {
        size_t size = n_elements * sizeof(float);

        float * buffer = (float *) malloc(size);
        if (buffer == NULL) {
            error("Failed to allocate buffer");
        }

        ggml_backend_tensor_get(tensor, buffer, offset, size);

        SEXP result = PROTECT(allocVector(REALSXP, n_elements));
        double * r_data = REAL(result);
        for (int64_t i = 0; i < n_elements; i++) {
            r_data[i] = (double) buffer[i];
        }

        free(buffer);
        UNPROTECT(1);
        return result;
    } else if (tensor->type == GGML_TYPE_I32) {
        size_t size = n_elements * sizeof(int32_t);

        SEXP result = PROTECT(allocVector(INTSXP, n_elements));
        ggml_backend_tensor_get(tensor, INTEGER(result), offset, size);

        UNPROTECT(1);
        return result;
    } else {
        error("Unsupported tensor type for ggml_backend_tensor_get");
        return R_NilValue;
    }
}

// ============================================================================
// Backend Context Tensor Allocation
// ============================================================================

// Allocate all tensors in a context using a backend
SEXP R_ggml_backend_alloc_ctx_tensors(SEXP ctx_ptr, SEXP backend_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(backend_ptr);

    if (ctx == NULL || backend == NULL) {
        error("Invalid context or backend pointer");
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    if (buffer == NULL) {
        error("Failed to allocate context tensors");
    }

    return R_MakeExternalPtr(buffer, R_NilValue, R_NilValue);
}

// ============================================================================
// Graph Allocator (gallocr)
// ============================================================================

// Create a new graph allocator with CPU buffer type
SEXP R_ggml_gallocr_new(void) {
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);

    if (galloc == NULL) {
        error("Failed to create graph allocator");
    }

    return R_MakeExternalPtr(galloc, R_NilValue, R_NilValue);
}

// Create graph allocator with specific buffer type
SEXP R_ggml_gallocr_new_buft(SEXP buft_ptr) {
    ggml_backend_buffer_type_t buft = (buft_ptr == R_NilValue) ?
        ggml_backend_cpu_buffer_type() :
        (ggml_backend_buffer_type_t) R_ExternalPtrAddr(buft_ptr);

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);

    if (galloc == NULL) {
        error("Failed to create graph allocator");
    }

    return R_MakeExternalPtr(galloc, R_NilValue, R_NilValue);
}

// Free graph allocator
SEXP R_ggml_gallocr_free(SEXP galloc_ptr) {
    ggml_gallocr_t galloc = (ggml_gallocr_t) R_ExternalPtrAddr(galloc_ptr);

    if (galloc != NULL) {
        ggml_gallocr_free(galloc);
        R_ClearExternalPtr(galloc_ptr);
    }

    return R_NilValue;
}

// Reserve memory for a graph (optional, for pre-allocation)
SEXP R_ggml_gallocr_reserve(SEXP galloc_ptr, SEXP graph_ptr) {
    ggml_gallocr_t galloc = (ggml_gallocr_t) R_ExternalPtrAddr(galloc_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (galloc == NULL || graph == NULL) {
        error("Invalid pointer");
    }

    bool success = ggml_gallocr_reserve(galloc, graph);

    return ScalarLogical(success);
}

// Allocate memory for a graph
SEXP R_ggml_gallocr_alloc_graph(SEXP galloc_ptr, SEXP graph_ptr) {
    ggml_gallocr_t galloc = (ggml_gallocr_t) R_ExternalPtrAddr(galloc_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (galloc == NULL || graph == NULL) {
        error("Invalid pointer");
    }

    bool success = ggml_gallocr_alloc_graph(galloc, graph);

    return ScalarLogical(success);
}

// Get buffer size used by the allocator
SEXP R_ggml_gallocr_get_buffer_size(SEXP galloc_ptr, SEXP buffer_id_sexp) {
    ggml_gallocr_t galloc = (ggml_gallocr_t) R_ExternalPtrAddr(galloc_ptr);

    if (galloc == NULL) {
        error("Invalid galloc pointer");
    }

    int buffer_id = asInteger(buffer_id_sexp);
    size_t size = ggml_gallocr_get_buffer_size(galloc, buffer_id);

    return ScalarReal((double) size);
}

// ============================================================================
// Backend Buffer Operations
// ============================================================================

// Free a backend buffer
SEXP R_ggml_backend_buffer_free(SEXP buffer_ptr) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t) R_ExternalPtrAddr(buffer_ptr);

    if (buffer != NULL) {
        ggml_backend_buffer_free(buffer);
        R_ClearExternalPtr(buffer_ptr);
    }

    return R_NilValue;
}

// Get buffer size
SEXP R_ggml_backend_buffer_get_size(SEXP buffer_ptr) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t) R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    size_t size = ggml_backend_buffer_get_size(buffer);
    return ScalarReal((double) size);
}

// Get buffer name
SEXP R_ggml_backend_buffer_name(SEXP buffer_ptr) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t) R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    const char * name = ggml_backend_buffer_name(buffer);
    return mkString(name);
}

// ============================================================================
// Scalar Tensor Creation
// ============================================================================

// Create a scalar i32 tensor
SEXP R_ggml_new_i32(SEXP ctx_ptr, SEXP value_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    int32_t value = (int32_t) asInteger(value_sexp);
    struct ggml_tensor * tensor = ggml_new_i32(ctx, value);

    if (tensor == NULL) {
        error("Failed to create i32 scalar tensor");
    }

    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

// Create a scalar f32 tensor
SEXP R_ggml_new_f32(SEXP ctx_ptr, SEXP value_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    if (ctx == NULL) {
        error("Invalid context pointer");
    }

    float value = (float) asReal(value_sexp);
    struct ggml_tensor * tensor = ggml_new_f32(ctx, value);

    if (tensor == NULL) {
        error("Failed to create f32 scalar tensor");
    }

    return R_MakeExternalPtr(tensor, R_NilValue, R_NilValue);
}

// ============================================================================
// View Operations with Offset
// ============================================================================

// 1D view with byte offset
SEXP R_ggml_view_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0_sexp, SEXP offset_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t ne0 = (int64_t) asReal(ne0_sexp);
    size_t offset = (size_t) asReal(offset_sexp);

    struct ggml_tensor * result = ggml_view_1d(ctx, a, ne0, offset);

    if (result == NULL) {
        error("Failed to create view_1d");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 2D view with byte offset
SEXP R_ggml_view_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0_sexp, SEXP ne1_sexp,
                    SEXP nb1_sexp, SEXP offset_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t ne0 = (int64_t) asReal(ne0_sexp);
    int64_t ne1 = (int64_t) asReal(ne1_sexp);
    size_t nb1 = (size_t) asReal(nb1_sexp);
    size_t offset = (size_t) asReal(offset_sexp);

    struct ggml_tensor * result = ggml_view_2d(ctx, a, ne0, ne1, nb1, offset);

    if (result == NULL) {
        error("Failed to create view_2d");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 3D view with byte offset
SEXP R_ggml_view_3d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0_sexp, SEXP ne1_sexp, SEXP ne2_sexp,
                    SEXP nb1_sexp, SEXP nb2_sexp, SEXP offset_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t ne0 = (int64_t) asReal(ne0_sexp);
    int64_t ne1 = (int64_t) asReal(ne1_sexp);
    int64_t ne2 = (int64_t) asReal(ne2_sexp);
    size_t nb1 = (size_t) asReal(nb1_sexp);
    size_t nb2 = (size_t) asReal(nb2_sexp);
    size_t offset = (size_t) asReal(offset_sexp);

    struct ggml_tensor * result = ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset);

    if (result == NULL) {
        error("Failed to create view_3d");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 4D view with byte offset
SEXP R_ggml_view_4d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0_sexp, SEXP ne1_sexp,
                    SEXP ne2_sexp, SEXP ne3_sexp,
                    SEXP nb1_sexp, SEXP nb2_sexp, SEXP nb3_sexp, SEXP offset_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int64_t ne0 = (int64_t) asReal(ne0_sexp);
    int64_t ne1 = (int64_t) asReal(ne1_sexp);
    int64_t ne2 = (int64_t) asReal(ne2_sexp);
    int64_t ne3 = (int64_t) asReal(ne3_sexp);
    size_t nb1 = (size_t) asReal(nb1_sexp);
    size_t nb2 = (size_t) asReal(nb2_sexp);
    size_t nb3 = (size_t) asReal(nb3_sexp);
    size_t offset = (size_t) asReal(offset_sexp);

    struct ggml_tensor * result = ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);

    if (result == NULL) {
        error("Failed to create view_4d");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Copy Operation
// ============================================================================

// Copy tensor a to tensor b (with type conversion if needed)
SEXP R_ggml_cpy(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_cpy(ctx, a, b);

    if (result == NULL) {
        error("Failed to create cpy operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Set Operation
// ============================================================================

// Set a[offset:] = b (copy b into a starting at offset)
SEXP R_ggml_set(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                SEXP nb1_sexp, SEXP nb2_sexp, SEXP nb3_sexp, SEXP offset_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    size_t nb1 = (size_t) asReal(nb1_sexp);
    size_t nb2 = (size_t) asReal(nb2_sexp);
    size_t nb3 = (size_t) asReal(nb3_sexp);
    size_t offset = (size_t) asReal(offset_sexp);

    struct ggml_tensor * result = ggml_set(ctx, a, b, nb1, nb2, nb3, offset);

    if (result == NULL) {
        error("Failed to create set operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 1D set - simpler interface
SEXP R_ggml_set_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP offset_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    size_t offset = (size_t) asReal(offset_sexp);
    struct ggml_tensor * result = ggml_set_1d(ctx, a, b, offset);

    if (result == NULL) {
        error("Failed to create set_1d operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 2D set
SEXP R_ggml_set_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP nb1_sexp, SEXP offset_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    size_t nb1 = (size_t) asReal(nb1_sexp);
    size_t offset = (size_t) asReal(offset_sexp);
    struct ggml_tensor * result = ggml_set_2d(ctx, a, b, nb1, offset);

    if (result == NULL) {
        error("Failed to create set_2d operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Matrix Operations
// ============================================================================

// Outer product: C = a * b^T
SEXP R_ggml_out_prod(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_out_prod(ctx, a, b);

    if (result == NULL) {
        error("Failed to create out_prod operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Diagonal matrix from vector
SEXP R_ggml_diag(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_diag(ctx, a);

    if (result == NULL) {
        error("Failed to create diag operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Backward Pass Operations (for training)
// ============================================================================

// SiLU backward
SEXP R_ggml_silu_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_silu_back(ctx, a, b);

    if (result == NULL) {
        error("Failed to create silu_back operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Get rows backward
SEXP R_ggml_get_rows_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);

    if (ctx == NULL || a == NULL || b == NULL || c == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_get_rows_back(ctx, a, b, c);

    if (result == NULL) {
        error("Failed to create get_rows_back operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Softmax backward (extended version)
SEXP R_ggml_soft_max_ext_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                               SEXP scale_sexp, SEXP max_bias_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    float scale = (float) asReal(scale_sexp);
    float max_bias = (float) asReal(max_bias_sexp);

    struct ggml_tensor * result = ggml_soft_max_ext_back(ctx, a, b, scale, max_bias);

    if (result == NULL) {
        error("Failed to create soft_max_ext_back operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Softmax backward inplace (extended version)
SEXP R_ggml_soft_max_ext_back_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                                       SEXP scale_sexp, SEXP max_bias_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    float scale = (float) asReal(scale_sexp);
    float max_bias = (float) asReal(max_bias_sexp);

    struct ggml_tensor * result = ggml_soft_max_ext_back_inplace(ctx, a, b, scale, max_bias);

    if (result == NULL) {
        error("Failed to create soft_max_ext_back_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// RoPE backward (extended version)
SEXP R_ggml_rope_ext_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                          SEXP n_dims_sexp, SEXP mode_sexp, SEXP n_ctx_orig_sexp,
                          SEXP freq_base_sexp, SEXP freq_scale_sexp,
                          SEXP ext_factor_sexp, SEXP attn_factor_sexp,
                          SEXP beta_fast_sexp, SEXP beta_slow_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = (c_ptr == R_NilValue) ? NULL :
                             (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);

    int n_dims = asInteger(n_dims_sexp);
    int mode = asInteger(mode_sexp);
    int n_ctx_orig = asInteger(n_ctx_orig_sexp);
    float freq_base = (float) asReal(freq_base_sexp);
    float freq_scale = (float) asReal(freq_scale_sexp);
    float ext_factor = (float) asReal(ext_factor_sexp);
    float attn_factor = (float) asReal(attn_factor_sexp);
    float beta_fast = (float) asReal(beta_fast_sexp);
    float beta_slow = (float) asReal(beta_slow_sexp);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_rope_ext_back(ctx, a, b, c, n_dims, mode, n_ctx_orig,
                                                       freq_base, freq_scale, ext_factor,
                                                       attn_factor, beta_fast, beta_slow);

    if (result == NULL) {
        error("Failed to create rope_ext_back operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Concatenation
// ============================================================================

// Concatenate tensors along specified dimension
SEXP R_ggml_concat(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP dim_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    int dim = asInteger(dim_sexp);
    struct ggml_tensor * result = ggml_concat(ctx, a, b, dim);

    if (result == NULL) {
        error("Failed to create concat operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Mixture of Experts (MoE) Operations
// ============================================================================

// Indirect matrix multiplication for MoE
// as: stacked expert weight matrices [n_embd, n_ff, n_experts]
// b: input tensor
// ids: expert selection indices
SEXP R_ggml_mul_mat_id(SEXP ctx_ptr, SEXP as_ptr, SEXP b_ptr, SEXP ids_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * as = (struct ggml_tensor *) R_ExternalPtrAddr(as_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * ids = (struct ggml_tensor *) R_ExternalPtrAddr(ids_ptr);

    if (ctx == NULL || as == NULL || b == NULL || ids == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_mul_mat_id(ctx, as, b, ids);

    if (result == NULL) {
        error("Failed to create mul_mat_id operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Flash Attention
// ============================================================================

// Flash Attention with KV cache support
// q: query tensor  [n_embd, n_head, n_tokens, batch]
// k: key tensor    [n_embd, n_head_kv, n_kv, batch]
// v: value tensor  [n_embd, n_head_kv, n_kv, batch]
// mask: attention mask (optional, can be NULL)
// scale: attention scale (usually 1/sqrt(head_dim))
// max_bias: maximum ALiBi bias (0 = disabled)
// logit_softcap: softcap for logits (0 = disabled)
SEXP R_ggml_flash_attn_ext(SEXP ctx_ptr, SEXP q_ptr, SEXP k_ptr, SEXP v_ptr,
                           SEXP mask_ptr, SEXP scale_sexp, SEXP max_bias_sexp,
                           SEXP logit_softcap_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * q = (struct ggml_tensor *) R_ExternalPtrAddr(q_ptr);
    struct ggml_tensor * k = (struct ggml_tensor *) R_ExternalPtrAddr(k_ptr);
    struct ggml_tensor * v = (struct ggml_tensor *) R_ExternalPtrAddr(v_ptr);
    struct ggml_tensor * mask = (mask_ptr == R_NilValue) ? NULL :
                                (struct ggml_tensor *) R_ExternalPtrAddr(mask_ptr);
    float scale = (float) asReal(scale_sexp);
    float max_bias = (float) asReal(max_bias_sexp);
    float logit_softcap = (float) asReal(logit_softcap_sexp);

    if (ctx == NULL || q == NULL || k == NULL || v == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_flash_attn_ext(ctx, q, k, v, mask,
                                                       scale, max_bias, logit_softcap);

    if (result == NULL) {
        error("Failed to create flash_attn_ext operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Flash Attention backward pass (for training)
// q, k, v: same as forward pass
// d: gradient from upstream (same shape as output)
// masked: whether causal mask was used
SEXP R_ggml_flash_attn_back(SEXP ctx_ptr, SEXP q_ptr, SEXP k_ptr, SEXP v_ptr,
                            SEXP d_ptr, SEXP masked_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * q = (struct ggml_tensor *) R_ExternalPtrAddr(q_ptr);
    struct ggml_tensor * k = (struct ggml_tensor *) R_ExternalPtrAddr(k_ptr);
    struct ggml_tensor * v = (struct ggml_tensor *) R_ExternalPtrAddr(v_ptr);
    struct ggml_tensor * d = (struct ggml_tensor *) R_ExternalPtrAddr(d_ptr);
    bool masked = asLogical(masked_sexp);

    if (ctx == NULL || q == NULL || k == NULL || v == NULL || d == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_flash_attn_back(ctx, q, k, v, d, masked);

    if (result == NULL) {
        error("Failed to create flash_attn_back operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Sequence/Token Operations
// ============================================================================

// Pad tensor with zeros
// p0, p1, p2, p3: padding amounts for each dimension (right side only)
SEXP R_ggml_pad(SEXP ctx_ptr, SEXP a_ptr, SEXP p0_sexp, SEXP p1_sexp,
                SEXP p2_sexp, SEXP p3_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int p0 = asInteger(p0_sexp);
    int p1 = asInteger(p1_sexp);
    int p2 = asInteger(p2_sexp);
    int p3 = asInteger(p3_sexp);

    struct ggml_tensor * result = ggml_pad(ctx, a, p0, p1, p2, p3);

    if (result == NULL) {
        error("Failed to create pad operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Argsort - returns indices that would sort the tensor
// order: 0 = ascending (GGML_SORT_ORDER_ASC), 1 = descending (GGML_SORT_ORDER_DESC)
SEXP R_ggml_argsort(SEXP ctx_ptr, SEXP a_ptr, SEXP order_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int order_int = asInteger(order_sexp);
    enum ggml_sort_order order = (order_int == 0) ? GGML_SORT_ORDER_ASC : GGML_SORT_ORDER_DESC;

    struct ggml_tensor * result = ggml_argsort(ctx, a, order);

    if (result == NULL) {
        error("Failed to create argsort operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Top-K - returns top k elements per row
// Note: the resulting indices are in no particular order
SEXP R_ggml_top_k(SEXP ctx_ptr, SEXP a_ptr, SEXP k_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int k = asInteger(k_sexp);

    struct ggml_tensor * result = ggml_top_k(ctx, a, k);

    if (result == NULL) {
        error("Failed to create top_k operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Sequence Operations - Additional
// ============================================================================

// Repeat backward - sums repetitions back to original shape
SEXP R_ggml_repeat_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    struct ggml_tensor * result = ggml_repeat_back(ctx, a, b);

    if (result == NULL) {
        error("Failed to create repeat_back operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Upscale - multiply ne0 and ne1 by scale factor
// mode: 0 = nearest, 1 = bilinear, 2 = bicubic
SEXP R_ggml_upscale(SEXP ctx_ptr, SEXP a_ptr, SEXP scale_factor_sexp, SEXP mode_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int scale_factor = asInteger(scale_factor_sexp);
    int mode_int = asInteger(mode_sexp);
    enum ggml_scale_mode mode = (enum ggml_scale_mode) mode_int;

    struct ggml_tensor * result = ggml_upscale(ctx, a, scale_factor, mode);

    if (result == NULL) {
        error("Failed to create upscale operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Utility Functions - Additional
// ============================================================================

// Get type size in bytes
SEXP R_ggml_type_size(SEXP type_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);
    size_t size = ggml_type_size(type);
    return ScalarReal((double) size);
}

// Get element size for tensor
SEXP R_ggml_element_size(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    size_t size = ggml_element_size(tensor);
    return ScalarReal((double) size);
}

// Get number of rows
SEXP R_ggml_nrows(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    int64_t nrows = ggml_nrows(tensor);
    return ScalarReal((double) nrows);
}

// Compare tensor shapes
SEXP R_ggml_are_same_shape(SEXP a_ptr, SEXP b_ptr) {
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (a == NULL || b == NULL) {
        error("Invalid tensor pointer");
    }

    bool same = ggml_are_same_shape(a, b);
    return ScalarLogical(same);
}

// Set tensor name
SEXP R_ggml_set_name(SEXP tensor_ptr, SEXP name_sexp) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    const char * name = CHAR(STRING_ELT(name_sexp, 0));
    ggml_set_name(tensor, name);

    return tensor_ptr;  // Return the tensor for chaining
}

// Get tensor name
SEXP R_ggml_get_name(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    const char * name = ggml_get_name(tensor);
    if (name == NULL || name[0] == '\0') {
        return R_NilValue;
    }
    return mkString(name);
}

// ============================================================================
// Backend Functions - Direct Access
// ============================================================================

// Initialize CPU backend
SEXP R_ggml_backend_cpu_init(void) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == NULL) {
        error("Failed to initialize CPU backend");
    }
    return R_MakeExternalPtr(backend, R_NilValue, R_NilValue);
}

// Free backend
SEXP R_ggml_backend_free(SEXP backend_ptr) {
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(backend_ptr);
    if (backend != NULL) {
        ggml_backend_free(backend);
        R_ClearExternalPtr(backend_ptr);
    }
    return R_NilValue;
}

// Set number of threads for CPU backend
SEXP R_ggml_backend_cpu_set_n_threads(SEXP backend_ptr, SEXP n_threads_sexp) {
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(backend_ptr);
    if (backend == NULL) {
        error("Invalid backend pointer");
    }

    int n_threads = asInteger(n_threads_sexp);
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    return R_NilValue;
}

// Compute graph with backend
SEXP R_ggml_backend_graph_compute(SEXP backend_ptr, SEXP graph_ptr) {
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(backend_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);

    if (backend == NULL || graph == NULL) {
        error("Invalid pointer");
    }

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);

    return ScalarInteger((int) status);
}

// Get backend name
SEXP R_ggml_backend_name(SEXP backend_ptr) {
    ggml_backend_t backend = (ggml_backend_t) R_ExternalPtrAddr(backend_ptr);
    if (backend == NULL) {
        error("Invalid backend pointer");
    }

    const char * name = ggml_backend_name(backend);
    return mkString(name);
}

// ============================================================================
// CNN Operations
// ============================================================================

// 1D Convolution
SEXP R_ggml_conv_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                    SEXP s0_sexp, SEXP p0_sexp, SEXP d0_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    int s0 = asInteger(s0_sexp);
    int p0 = asInteger(p0_sexp);
    int d0 = asInteger(d0_sexp);

    struct ggml_tensor * result = ggml_conv_1d(ctx, a, b, s0, p0, d0);

    if (result == NULL) {
        error("Failed to create conv_1d operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 2D Convolution
SEXP R_ggml_conv_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                    SEXP s0_sexp, SEXP s1_sexp,
                    SEXP p0_sexp, SEXP p1_sexp,
                    SEXP d0_sexp, SEXP d1_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    int s0 = asInteger(s0_sexp);
    int s1 = asInteger(s1_sexp);
    int p0 = asInteger(p0_sexp);
    int p1 = asInteger(p1_sexp);
    int d0 = asInteger(d0_sexp);
    int d1 = asInteger(d1_sexp);

    struct ggml_tensor * result = ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1);

    if (result == NULL) {
        error("Failed to create conv_2d operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Transposed 1D Convolution
SEXP R_ggml_conv_transpose_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                               SEXP s0_sexp, SEXP p0_sexp, SEXP d0_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    int s0 = asInteger(s0_sexp);
    int p0 = asInteger(p0_sexp);
    int d0 = asInteger(d0_sexp);

    struct ggml_tensor * result = ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0);

    if (result == NULL) {
        error("Failed to create conv_transpose_1d operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 1D Pooling
// op: 0 = max, 1 = avg
SEXP R_ggml_pool_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP op_sexp,
                    SEXP k0_sexp, SEXP s0_sexp, SEXP p0_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int op_int = asInteger(op_sexp);
    enum ggml_op_pool op = (op_int == 0) ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;
    int k0 = asInteger(k0_sexp);
    int s0 = asInteger(s0_sexp);
    int p0 = asInteger(p0_sexp);

    struct ggml_tensor * result = ggml_pool_1d(ctx, a, op, k0, s0, p0);

    if (result == NULL) {
        error("Failed to create pool_1d operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// 2D Pooling
// op: 0 = max, 1 = avg
SEXP R_ggml_pool_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP op_sexp,
                    SEXP k0_sexp, SEXP k1_sexp,
                    SEXP s0_sexp, SEXP s1_sexp,
                    SEXP p0_sexp, SEXP p1_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }

    int op_int = asInteger(op_sexp);
    enum ggml_op_pool op = (op_int == 0) ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;
    int k0 = asInteger(k0_sexp);
    int k1 = asInteger(k1_sexp);
    int s0 = asInteger(s0_sexp);
    int s1 = asInteger(s1_sexp);
    float p0 = (float) asReal(p0_sexp);
    float p1 = (float) asReal(p1_sexp);

    struct ggml_tensor * result = ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1);

    if (result == NULL) {
        error("Failed to create pool_2d operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Image to Column (for efficient convolution)
SEXP R_ggml_im2col(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                   SEXP s0_sexp, SEXP s1_sexp,
                   SEXP p0_sexp, SEXP p1_sexp,
                   SEXP d0_sexp, SEXP d1_sexp,
                   SEXP is_2D_sexp, SEXP dst_type_sexp) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }

    int s0 = asInteger(s0_sexp);
    int s1 = asInteger(s1_sexp);
    int p0 = asInteger(p0_sexp);
    int p1 = asInteger(p1_sexp);
    int d0 = asInteger(d0_sexp);
    int d1 = asInteger(d1_sexp);
    bool is_2D = asLogical(is_2D_sexp);
    enum ggml_type dst_type = (enum ggml_type) asInteger(dst_type_sexp);

    struct ggml_tensor * result = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D, dst_type);

    if (result == NULL) {
        error("Failed to create im2col operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Quantization Functions
// ============================================================================

// Initialize quantization tables for a type
SEXP R_ggml_quantize_init(SEXP type_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);
    ggml_quantize_init(type);
    return R_NilValue;
}

// Free quantization resources
SEXP R_ggml_quantize_free(void) {
    ggml_quantize_free();
    return R_NilValue;
}

// ============================================================================
// In-place Operations (Memory-efficient, 2-3x memory savings)
// ============================================================================

// Binary inplace operations (ctx, a, b) -> view(a)

SEXP R_ggml_add_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_add_inplace(ctx, a, b);

    if (result == NULL) {
        error("Failed to create add_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sub_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_sub_inplace(ctx, a, b);

    if (result == NULL) {
        error("Failed to create sub_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_mul_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_mul_inplace(ctx, a, b);

    if (result == NULL) {
        error("Failed to create mul_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_div_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);

    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_div_inplace(ctx, a, b);

    if (result == NULL) {
        error("Failed to create div_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Unary inplace math operations (ctx, a) -> view(a)

SEXP R_ggml_sqr_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_sqr_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create sqr_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sqrt_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_sqrt_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create sqrt_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_exp_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_exp_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create exp_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_log_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_log_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create log_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_abs_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_abs_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create abs_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_neg_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_neg_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create neg_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_ceil_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_ceil_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create ceil_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_floor_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_floor_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create floor_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_round_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_round_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create round_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Activation inplace operations (ctx, a) -> view(a)

SEXP R_ggml_relu_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_relu_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create relu_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_gelu_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_gelu_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create gelu_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_silu_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_silu_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create silu_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_sigmoid_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_sigmoid_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create sigmoid_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_tanh_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_tanh_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create tanh_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_softplus_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_softplus_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create softplus_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

SEXP R_ggml_elu_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_elu_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create elu_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Scale inplace (ctx, a, s) -> view(a)

SEXP R_ggml_scale_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP s) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    float scale = (float) asReal(s);
    struct ggml_tensor * result = ggml_scale_inplace(ctx, a, scale);

    if (result == NULL) {
        error("Failed to create scale_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Dup inplace (ctx, a) -> view(a)

SEXP R_ggml_dup_inplace(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);

    if (ctx == NULL || a == NULL) {
        error("Invalid pointer (context or tensor is NULL)");
    }

    struct ggml_tensor * result = ggml_dup_inplace(ctx, a);

    if (result == NULL) {
        error("Failed to create dup_inplace operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Check if quantization type requires importance matrix
SEXP R_ggml_quantize_requires_imatrix(SEXP type_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);
    bool requires = ggml_quantize_requires_imatrix(type);
    return ScalarLogical(requires);
}

// Quantize a chunk of data
// Returns the number of bytes written to dst
SEXP R_ggml_quantize_chunk(SEXP type_sexp, SEXP src_sexp, SEXP nrows_sexp, SEXP n_per_row_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);

    // Get source data from R numeric vector
    int n = length(src_sexp);
    double * r_src = REAL(src_sexp);

    // Convert to float
    float * src = (float *) R_alloc(n, sizeof(float));
    for (int i = 0; i < n; i++) {
        src[i] = (float) r_src[i];
    }

    int64_t nrows = (int64_t) asReal(nrows_sexp);
    int64_t n_per_row = (int64_t) asReal(n_per_row_sexp);

    // Calculate output size
    size_t type_size = ggml_type_size(type);
    int64_t blck_size = ggml_blck_size(type);
    size_t dst_size = (n_per_row / blck_size) * type_size * nrows;

    // Allocate destination buffer
    void * dst = R_alloc(dst_size, 1);

    // Perform quantization
    size_t result = ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, NULL);

    // Create raw vector with quantized data
    SEXP out = PROTECT(allocVector(RAWSXP, result));
    memcpy(RAW(out), dst, result);

    UNPROTECT(1);
    return out;
}

// ============================================================================
// Type System Functions
// ============================================================================

// Get type name as string
SEXP R_ggml_type_name(SEXP type_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);
    const char * name = ggml_type_name(type);
    return mkString(name ? name : "unknown");
}

// Get type size as float (for quantized types)
SEXP R_ggml_type_sizef(SEXP type_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);
    double sizef = (double)ggml_row_size(type, 1);
    return ScalarReal(sizef);
}

// Get block size for type
SEXP R_ggml_blck_size(SEXP type_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);
    int64_t blck = ggml_blck_size(type);
    return ScalarReal((double)blck);
}

// Check if type is quantized
SEXP R_ggml_is_quantized(SEXP type_sexp) {
    enum ggml_type type = (enum ggml_type) asInteger(type_sexp);
    bool quantized = ggml_is_quantized(type);
    return ScalarLogical(quantized);
}

// Convert ftype to ggml_type
SEXP R_ggml_ftype_to_ggml_type(SEXP ftype_sexp) {
    enum ggml_ftype ftype = (enum ggml_ftype) asInteger(ftype_sexp);
    enum ggml_type type = ggml_ftype_to_ggml_type(ftype);
    return ScalarInteger((int)type);
}

// ============================================================================
// Operation Info Functions
// ============================================================================

// Get operation name
SEXP R_ggml_op_name(SEXP op_sexp) {
    enum ggml_op op = (enum ggml_op) asInteger(op_sexp);
    const char * name = ggml_op_name(op);
    return mkString(name ? name : "unknown");
}

// Get operation symbol
SEXP R_ggml_op_symbol(SEXP op_sexp) {
    enum ggml_op op = (enum ggml_op) asInteger(op_sexp);
    const char * symbol = ggml_op_symbol(op);
    return mkString(symbol ? symbol : "?");
}

// Get unary operation name
SEXP R_ggml_unary_op_name(SEXP op_sexp) {
    enum ggml_unary_op op = (enum ggml_unary_op) asInteger(op_sexp);
    const char * name = ggml_unary_op_name(op);
    return mkString(name ? name : "unknown");
}

// Get operation description from tensor
SEXP R_ggml_op_desc(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    const char * desc = ggml_op_desc(tensor);
    return mkString(desc ? desc : "");
}

// Get unary op from tensor
SEXP R_ggml_get_unary_op(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    enum ggml_unary_op op = ggml_get_unary_op(tensor);
    return ScalarInteger((int)op);
}

// ============================================================================
// CPU Feature Detection Functions
// ============================================================================

// x86 SIMD features
SEXP R_ggml_cpu_has_sse3(void) {
    return ScalarLogical(ggml_cpu_has_sse3());
}

SEXP R_ggml_cpu_has_ssse3(void) {
    return ScalarLogical(ggml_cpu_has_ssse3());
}

SEXP R_ggml_cpu_has_avx(void) {
    return ScalarLogical(ggml_cpu_has_avx());
}

SEXP R_ggml_cpu_has_avx_vnni(void) {
    return ScalarLogical(ggml_cpu_has_avx_vnni());
}

SEXP R_ggml_cpu_has_avx2(void) {
    return ScalarLogical(ggml_cpu_has_avx2());
}

SEXP R_ggml_cpu_has_bmi2(void) {
    return ScalarLogical(ggml_cpu_has_bmi2());
}

SEXP R_ggml_cpu_has_f16c(void) {
    return ScalarLogical(ggml_cpu_has_f16c());
}

SEXP R_ggml_cpu_has_fma(void) {
    return ScalarLogical(ggml_cpu_has_fma());
}

SEXP R_ggml_cpu_has_avx512(void) {
    return ScalarLogical(ggml_cpu_has_avx512());
}

SEXP R_ggml_cpu_has_avx512_vbmi(void) {
    return ScalarLogical(ggml_cpu_has_avx512_vbmi());
}

SEXP R_ggml_cpu_has_avx512_vnni(void) {
    return ScalarLogical(ggml_cpu_has_avx512_vnni());
}

SEXP R_ggml_cpu_has_avx512_bf16(void) {
    return ScalarLogical(ggml_cpu_has_avx512_bf16());
}

SEXP R_ggml_cpu_has_amx_int8(void) {
    return ScalarLogical(ggml_cpu_has_amx_int8());
}

// ARM SIMD features
SEXP R_ggml_cpu_has_neon(void) {
    return ScalarLogical(ggml_cpu_has_neon());
}

SEXP R_ggml_cpu_has_arm_fma(void) {
    return ScalarLogical(ggml_cpu_has_arm_fma());
}

SEXP R_ggml_cpu_has_fp16_va(void) {
    return ScalarLogical(ggml_cpu_has_fp16_va());
}

SEXP R_ggml_cpu_has_dotprod(void) {
    return ScalarLogical(ggml_cpu_has_dotprod());
}

SEXP R_ggml_cpu_has_matmul_int8(void) {
    return ScalarLogical(ggml_cpu_has_matmul_int8());
}

SEXP R_ggml_cpu_has_sve(void) {
    return ScalarLogical(ggml_cpu_has_sve());
}

SEXP R_ggml_cpu_get_sve_cnt(void) {
    return ScalarInteger(ggml_cpu_get_sve_cnt());
}

SEXP R_ggml_cpu_has_sme(void) {
    return ScalarLogical(ggml_cpu_has_sme());
}

// Other architectures
SEXP R_ggml_cpu_has_riscv_v(void) {
    return ScalarLogical(ggml_cpu_has_riscv_v());
}

SEXP R_ggml_cpu_get_rvv_vlen(void) {
    return ScalarInteger(ggml_cpu_get_rvv_vlen());
}

SEXP R_ggml_cpu_has_vsx(void) {
    return ScalarLogical(ggml_cpu_has_vsx());
}

SEXP R_ggml_cpu_has_vxe(void) {
    return ScalarLogical(ggml_cpu_has_vxe());
}

SEXP R_ggml_cpu_has_wasm_simd(void) {
    return ScalarLogical(ggml_cpu_has_wasm_simd());
}

SEXP R_ggml_cpu_has_llamafile(void) {
    return ScalarLogical(ggml_cpu_has_llamafile());
}

// ============================================================================
// Tensor Layout/Contiguity Functions
// ============================================================================

// Check contiguity at different dimensions
SEXP R_ggml_is_contiguous_0(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_is_contiguous_0(tensor));
}

SEXP R_ggml_is_contiguous_1(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_is_contiguous_1(tensor));
}

SEXP R_ggml_is_contiguous_2(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_is_contiguous_2(tensor));
}

// Check if tensor is contiguously allocated (data pointer is valid)
SEXP R_ggml_is_contiguously_allocated(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_is_contiguously_allocated(tensor));
}

// Check channel-wise contiguity (for CNN operations)
SEXP R_ggml_is_contiguous_channels(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_is_contiguous_channels(tensor));
}

// Check row-wise contiguity (for matrix operations)
SEXP R_ggml_is_contiguous_rows(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_is_contiguous_rows(tensor));
}

// Compare tensor strides
SEXP R_ggml_are_same_stride(SEXP a_ptr, SEXP b_ptr) {
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    if (a == NULL || b == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_are_same_stride(a, b));
}

// Check if tensor a can be repeated to match tensor b
SEXP R_ggml_can_repeat(SEXP a_ptr, SEXP b_ptr) {
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    if (a == NULL || b == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarLogical(ggml_can_repeat(a, b));
}

// Count equal elements between two tensors
SEXP R_ggml_count_equal(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }
    struct ggml_tensor * result = ggml_count_equal(ctx, a, b);
    if (result == NULL) {
        error("Failed to create count_equal operation");
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// RoPE multi backward
SEXP R_ggml_rope_multi_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                            SEXP n_dims, SEXP sections, SEXP mode, SEXP n_ctx_orig,
                            SEXP freq_base, SEXP freq_scale, SEXP ext_factor,
                            SEXP attn_factor, SEXP beta_fast, SEXP beta_slow) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = c_ptr == R_NilValue ? NULL : (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);
    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }
    int sect[4] = {0, 0, 0, 0};
    if (sections != R_NilValue && length(sections) == 4) {
        int * s = INTEGER(sections);
        sect[0] = s[0]; sect[1] = s[1]; sect[2] = s[2]; sect[3] = s[3];
    }
    struct ggml_tensor * result = ggml_rope_multi_back(ctx, a, b, c,
        asInteger(n_dims), sect, asInteger(mode), asInteger(n_ctx_orig),
        (float)asReal(freq_base), (float)asReal(freq_scale), (float)asReal(ext_factor),
        (float)asReal(attn_factor), (float)asReal(beta_fast), (float)asReal(beta_slow));
    if (result == NULL) {
        error("Failed to create rope_multi_back operation");
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Graph Construction & Introspection Functions
// ============================================================================

// Build backward graph for training
SEXP R_ggml_build_backward_expand(SEXP ctx_ptr, SEXP graph_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    if (ctx == NULL || graph == NULL) {
        error("Invalid pointer");
    }
    ggml_build_backward_expand(ctx, graph, NULL);
    return R_NilValue;
}

// Add node to graph
SEXP R_ggml_graph_add_node(SEXP graph_ptr, SEXP tensor_ptr) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (graph == NULL || tensor == NULL) {
        error("Invalid pointer");
    }
    ggml_graph_add_node(graph, tensor);
    return R_NilValue;
}

// Clear graph
SEXP R_ggml_graph_clear(SEXP graph_ptr) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    if (graph == NULL) {
        error("Invalid graph pointer");
    }
    ggml_graph_clear(graph);
    return R_NilValue;
}

// Copy graph
SEXP R_ggml_graph_cpy(SEXP src_ptr, SEXP dst_ptr) {
    struct ggml_cgraph * src = (struct ggml_cgraph *) R_ExternalPtrAddr(src_ptr);
    struct ggml_cgraph * dst = (struct ggml_cgraph *) R_ExternalPtrAddr(dst_ptr);
    if (src == NULL || dst == NULL) {
        error("Invalid graph pointer");
    }
    ggml_graph_cpy(src, dst);
    return R_NilValue;
}

// Duplicate graph
SEXP R_ggml_graph_dup(SEXP ctx_ptr, SEXP graph_ptr, SEXP force_grads) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    if (ctx == NULL || graph == NULL) {
        error("Invalid pointer");
    }
    struct ggml_cgraph * result = ggml_graph_dup(ctx, graph, asLogical(force_grads));
    if (result == NULL) {
        error("Failed to duplicate graph");
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Get gradient tensor from graph
SEXP R_ggml_graph_get_grad(SEXP graph_ptr, SEXP node_ptr) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    struct ggml_tensor * node = (struct ggml_tensor *) R_ExternalPtrAddr(node_ptr);
    if (graph == NULL || node == NULL) {
        error("Invalid pointer");
    }
    struct ggml_tensor * result = ggml_graph_get_grad(graph, node);
    if (result == NULL) {
        return R_NilValue;
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Get gradient accumulator from graph
SEXP R_ggml_graph_get_grad_acc(SEXP graph_ptr, SEXP node_ptr) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    struct ggml_tensor * node = (struct ggml_tensor *) R_ExternalPtrAddr(node_ptr);
    if (graph == NULL || node == NULL) {
        error("Invalid pointer");
    }
    struct ggml_tensor * result = ggml_graph_get_grad_acc(graph, node);
    if (result == NULL) {
        return R_NilValue;
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Advanced Attention/Loss Functions
// ============================================================================

// Cross entropy loss
SEXP R_ggml_cross_entropy_loss(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    if (ctx == NULL || a == NULL || b == NULL) {
        error("Invalid pointer");
    }
    struct ggml_tensor * result = ggml_cross_entropy_loss(ctx, a, b);
    if (result == NULL) {
        error("Failed to create cross_entropy_loss operation");
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Cross entropy loss backward
SEXP R_ggml_cross_entropy_loss_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    struct ggml_tensor * c = (struct ggml_tensor *) R_ExternalPtrAddr(c_ptr);
    if (ctx == NULL || a == NULL || b == NULL || c == NULL) {
        error("Invalid pointer");
    }
    struct ggml_tensor * result = ggml_cross_entropy_loss_back(ctx, a, b, c);
    if (result == NULL) {
        error("Failed to create cross_entropy_loss_back operation");
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Cumulative sum
SEXP R_ggml_cumsum(SEXP ctx_ptr, SEXP a_ptr) {
    struct ggml_context * ctx = (struct ggml_context *) R_ExternalPtrAddr(ctx_ptr);
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    if (ctx == NULL || a == NULL) {
        error("Invalid pointer");
    }
    struct ggml_tensor * result = ggml_cumsum(ctx, a);
    if (result == NULL) {
        error("Failed to create cumsum operation");
    }
    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// Flash attention set precision
SEXP R_ggml_flash_attn_ext_set_prec(SEXP tensor_ptr, SEXP prec) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    ggml_flash_attn_ext_set_prec(tensor, (enum ggml_prec) asInteger(prec));
    return R_NilValue;
}

// Flash attention get precision
SEXP R_ggml_flash_attn_ext_get_prec(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    return ScalarInteger((int) ggml_flash_attn_ext_get_prec(tensor));
}

// Flash attention add sinks
SEXP R_ggml_flash_attn_ext_add_sinks(SEXP tensor_ptr, SEXP sinks_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    struct ggml_tensor * sinks = (struct ggml_tensor *) R_ExternalPtrAddr(sinks_ptr);
    if (tensor == NULL || sinks == NULL) {
        error("Invalid tensor pointer");
    }
    ggml_flash_attn_ext_add_sinks(tensor, sinks);
    return R_NilValue;
}

// Soft max add sinks
SEXP R_ggml_soft_max_add_sinks(SEXP tensor_ptr, SEXP sinks_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    struct ggml_tensor * sinks = (struct ggml_tensor *) R_ExternalPtrAddr(sinks_ptr);
    if (tensor == NULL || sinks == NULL) {
        error("Invalid tensor pointer");
    }
    ggml_soft_max_add_sinks(tensor, sinks);
    return R_NilValue;
}

// ============================================================================
// Graph Introspection Functions
// ============================================================================

// Create a view of a subgraph (nodes from i0 to i1)
// Note: ggml_graph_view returns struct by value, we allocate and copy
SEXP R_ggml_graph_view(SEXP graph_ptr, SEXP i0, SEXP i1) {
    struct ggml_cgraph * graph = (struct ggml_cgraph *) R_ExternalPtrAddr(graph_ptr);
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    int start = asInteger(i0);
    int end = asInteger(i1);

    // ggml_graph_view returns struct by value
    struct ggml_cgraph view = ggml_graph_view(graph, start, end);

    // Allocate memory for the view copy
    struct ggml_cgraph * view_copy = (struct ggml_cgraph *) malloc(sizeof(struct ggml_cgraph));
    if (view_copy == NULL) {
        error("Failed to allocate memory for graph view");
    }
    *view_copy = view;

    SEXP ptr = PROTECT(R_MakeExternalPtr(view_copy, R_NilValue, R_NilValue));
    // Note: caller is responsible for freeing this memory
    UNPROTECT(1);
    return ptr;
}

// Check if operation can be done in-place
SEXP R_ggml_op_can_inplace(SEXP op_sexp) {
    enum ggml_op op = (enum ggml_op) asInteger(op_sexp);
    return ScalarLogical(ggml_op_can_inplace(op));
}

// Check if two tensors have the same layout (type, shape, strides)
// Note: ggml_are_same_layout is static inline in ggml-impl.h, we reimplement it
SEXP R_ggml_are_same_layout(SEXP a_ptr, SEXP b_ptr) {
    struct ggml_tensor * a = (struct ggml_tensor *) R_ExternalPtrAddr(a_ptr);
    struct ggml_tensor * b = (struct ggml_tensor *) R_ExternalPtrAddr(b_ptr);
    if (a == NULL || b == NULL) {
        error("Invalid tensor pointer");
    }

    // Check type
    if (a->type != b->type) {
        return ScalarLogical(FALSE);
    }

    // Check dimensions and strides
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return ScalarLogical(FALSE);
        }
        if (a->nb[i] != b->nb[i]) {
            return ScalarLogical(FALSE);
        }
    }

    return ScalarLogical(TRUE);
}

// ============================================================================
// Logging & Debugging Functions
// ============================================================================

// Static flag to track if R logging is enabled
static int r_log_enabled = 0;

// R-compatible log callback that redirects to R's message system
static void r_ggml_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (text == NULL) return;

    switch (level) {
        case GGML_LOG_LEVEL_DEBUG:
        case GGML_LOG_LEVEL_INFO:
        case GGML_LOG_LEVEL_CONT:
            Rprintf("%s", text);
            break;
        case GGML_LOG_LEVEL_WARN:
            REprintf("Warning: %s", text);
            break;
        case GGML_LOG_LEVEL_ERROR:
            REprintf("Error: %s", text);
            break;
        default:
            Rprintf("%s", text);
            break;
    }
}

// Enable R-compatible logging
SEXP R_ggml_log_set_r(void) {
    ggml_log_set(r_ggml_log_callback, NULL);
    r_log_enabled = 1;
    return R_NilValue;
}

// Disable logging (set to NULL - stderr)
SEXP R_ggml_log_set_default(void) {
    ggml_log_set(NULL, NULL);
    r_log_enabled = 0;
    return R_NilValue;
}

// Check if R logging is enabled
SEXP R_ggml_log_is_r_enabled(void) {
    return ScalarLogical(r_log_enabled);
}

// Static flag for R abort handler
static int r_abort_enabled = 0;

// R-compatible abort callback
static void r_ggml_abort_callback(const char * error_message) {
    if (error_message != NULL) {
        Rf_error("GGML abort: %s", error_message);
    } else {
        Rf_error("GGML abort: unknown error");
    }
}

// Enable R-compatible abort handling
SEXP R_ggml_set_abort_callback_r(void) {
    ggml_set_abort_callback(r_ggml_abort_callback);
    r_abort_enabled = 1;
    return R_NilValue;
}

// Disable custom abort (restore default)
SEXP R_ggml_set_abort_callback_default(void) {
    ggml_set_abort_callback(NULL);
    r_abort_enabled = 0;
    return R_NilValue;
}

// Check if R abort handler is enabled
SEXP R_ggml_abort_is_r_enabled(void) {
    return ScalarLogical(r_abort_enabled);
}

// ============================================================================
// Tensor Op Params Functions
// ============================================================================

// Get op_params as raw bytes
SEXP R_ggml_get_op_params(SEXP tensor_ptr) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    size_t size = GGML_MAX_OP_PARAMS;
    SEXP result = PROTECT(allocVector(RAWSXP, size));
    memcpy(RAW(result), tensor->op_params, size);
    UNPROTECT(1);
    return result;
}

// Set op_params from raw bytes
SEXP R_ggml_set_op_params(SEXP tensor_ptr, SEXP params) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    if (TYPEOF(params) != RAWSXP) {
        error("params must be a raw vector");
    }

    size_t len = LENGTH(params);
    if (len > GGML_MAX_OP_PARAMS) {
        error("params too large (max %d bytes)", GGML_MAX_OP_PARAMS);
    }

    memset(tensor->op_params, 0, GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, RAW(params), len);
    return R_NilValue;
}

// Get single int32 from op_params at index
SEXP R_ggml_get_op_params_i32(SEXP tensor_ptr, SEXP index) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    int idx = asInteger(index);
    int max_idx = GGML_MAX_OP_PARAMS / sizeof(int32_t);
    if (idx < 0 || idx >= max_idx) {
        error("Index out of range (0-%d)", max_idx - 1);
    }

    int32_t * params = (int32_t *)tensor->op_params;
    return ScalarInteger(params[idx]);
}

// Set single int32 in op_params at index
SEXP R_ggml_set_op_params_i32(SEXP tensor_ptr, SEXP index, SEXP value) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    int idx = asInteger(index);
    int max_idx = GGML_MAX_OP_PARAMS / sizeof(int32_t);
    if (idx < 0 || idx >= max_idx) {
        error("Index out of range (0-%d)", max_idx - 1);
    }

    int32_t * params = (int32_t *)tensor->op_params;
    params[idx] = asInteger(value);
    return R_NilValue;
}

// Get single float from op_params at index
SEXP R_ggml_get_op_params_f32(SEXP tensor_ptr, SEXP index) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    int idx = asInteger(index);
    int max_idx = GGML_MAX_OP_PARAMS / sizeof(float);
    if (idx < 0 || idx >= max_idx) {
        error("Index out of range (0-%d)", max_idx - 1);
    }

    float * params = (float *)tensor->op_params;
    return ScalarReal((double)params[idx]);
}

// Set single float in op_params at index
SEXP R_ggml_set_op_params_f32(SEXP tensor_ptr, SEXP index, SEXP value) {
    struct ggml_tensor * tensor = (struct ggml_tensor *) R_ExternalPtrAddr(tensor_ptr);
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    int idx = asInteger(index);
    int max_idx = GGML_MAX_OP_PARAMS / sizeof(float);
    if (idx < 0 || idx >= max_idx) {
        error("Index out of range (0-%d)", max_idx - 1);
    }

    float * params = (float *)tensor->op_params;
    params[idx] = (float)asReal(value);
    return R_NilValue;
}
