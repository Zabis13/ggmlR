#include <R.h>
#include <Rinternals.h>
#include <stdlib.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#ifdef _OPENMP
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
