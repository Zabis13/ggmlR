// Backend scheduler R interface
// Multi-GPU support through backend scheduler

#include <R.h>
#include <Rinternals.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// ============================================================================
// Backend Scheduler Functions
// ============================================================================

// Create a new backend scheduler
// backends: list of backend pointers
// parallel: whether to run backends in parallel
// graph_size: expected maximum graph size (default: 2048)
// add_cpu: whether to automatically add CPU backend (default: TRUE)
SEXP R_ggml_backend_sched_new(SEXP backends_list, SEXP parallel, SEXP graph_size) {
    if (!isNewList(backends_list)) {
        error("backends must be a list of backend pointers");
    }

    int n_user_backends = length(backends_list);
    if (n_user_backends == 0) {
        error("At least one backend is required");
    }

    bool is_parallel = asLogical(parallel);
    size_t max_graph_size = (size_t)asReal(graph_size);

    // GGML scheduler requires last backend to be CPU
    // Allocate space for user backends + CPU backend
    int n_backends = n_user_backends + 1;
    ggml_backend_t * backends = (ggml_backend_t *)malloc(n_backends * sizeof(ggml_backend_t));
    if (backends == NULL) {
        error("Failed to allocate memory for backends array");
    }

    // Extract backend pointers from list
    for (int i = 0; i < n_user_backends; i++) {
        SEXP backend_ptr = VECTOR_ELT(backends_list, i);
        backends[i] = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

        if (backends[i] == NULL) {
            free(backends);
            error("Invalid backend pointer at index %d", i);
        }
    }

    // Add CPU backend as the last backend (required by GGML)
    backends[n_backends - 1] = ggml_backend_cpu_init();
    if (backends[n_backends - 1] == NULL) {
        free(backends);
        error("Failed to initialize CPU backend");
    }

    // Create scheduler
    // bufts = NULL means use default buffer types for each backend
    // op_offload = true enables offloading operations to backends
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends,
        NULL,           // buffer types (NULL = auto)
        n_backends,
        max_graph_size,
        is_parallel,    // parallel execution
        true            // op_offload
    );

    free(backends);

    if (sched == NULL) {
        error("Failed to create backend scheduler");
    }

    // Create external pointer with finalizer
    SEXP ptr = PROTECT(R_MakeExternalPtr(sched, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Free backend scheduler
SEXP R_ggml_backend_sched_free(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched != NULL) {
        ggml_backend_sched_free(sched);
        R_ClearExternalPtr(sched_ptr);
    }

    return R_NilValue;
}

// Reserve memory for scheduler based on a measure graph
SEXP R_ggml_backend_sched_reserve(SEXP sched_ptr, SEXP graph_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    bool success = ggml_backend_sched_reserve(sched, graph);
    return ScalarLogical(success);
}

// Get number of backends in scheduler
SEXP R_ggml_backend_sched_get_n_backends(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    int n = ggml_backend_sched_get_n_backends(sched);
    return ScalarInteger(n);
}

// Get backend at specific index from scheduler
SEXP R_ggml_backend_sched_get_backend(SEXP sched_ptr, SEXP index) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    int i = asInteger(index);
    int n = ggml_backend_sched_get_n_backends(sched);

    if (i < 0 || i >= n) {
        error("Backend index %d out of range (0-%d)", i, n - 1);
    }

    ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);

    SEXP ptr = PROTECT(R_MakeExternalPtr(backend, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get number of splits in last computed graph
SEXP R_ggml_backend_sched_get_n_splits(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    int n = ggml_backend_sched_get_n_splits(sched);
    return ScalarInteger(n);
}

// Get number of copies in last computed graph
SEXP R_ggml_backend_sched_get_n_copies(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    int n = ggml_backend_sched_get_n_copies(sched);
    return ScalarInteger(n);
}

// Set which backend a tensor should use
SEXP R_ggml_backend_sched_set_tensor_backend(SEXP sched_ptr, SEXP tensor_ptr, SEXP backend_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_tensor * tensor = (struct ggml_tensor *)R_ExternalPtrAddr(tensor_ptr);
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }
    if (backend == NULL) {
        error("Invalid backend pointer");
    }

    ggml_backend_sched_set_tensor_backend(sched, tensor, backend);
    return R_NilValue;
}

// Get which backend a tensor is assigned to
SEXP R_ggml_backend_sched_get_tensor_backend(SEXP sched_ptr, SEXP tensor_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_tensor * tensor = (struct ggml_tensor *)R_ExternalPtrAddr(tensor_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);

    if (backend == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(backend, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Allocate graph on scheduler
SEXP R_ggml_backend_sched_alloc_graph(SEXP sched_ptr, SEXP graph_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    bool success = ggml_backend_sched_alloc_graph(sched, graph);
    return ScalarLogical(success);
}

// Compute graph using scheduler (distributes work across backends)
SEXP R_ggml_backend_sched_graph_compute(SEXP sched_ptr, SEXP graph_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);

    return ScalarInteger((int)status);
}

// Compute graph asynchronously
SEXP R_ggml_backend_sched_graph_compute_async(SEXP sched_ptr, SEXP graph_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    enum ggml_status status = ggml_backend_sched_graph_compute_async(sched, graph);

    return ScalarInteger((int)status);
}

// Synchronize scheduler (wait for async operations)
SEXP R_ggml_backend_sched_synchronize(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    ggml_backend_sched_synchronize(sched);
    return R_NilValue;
}

// Reset scheduler (deallocates all tensors)
SEXP R_ggml_backend_sched_reset(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    ggml_backend_sched_reset(sched);
    return R_NilValue;
}
