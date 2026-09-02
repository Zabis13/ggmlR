// Backend scheduler R interface
// Multi-GPU support through backend scheduler

#include <R.h>
#include <Rinternals.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "r_ptr_check.h"
#include "r_sched_threads.h"
#include <stdlib.h>  // malloc/free; previously arrived transitively

// r_interface_custom.c -- CPU-only guard for GGML_OP_CUSTOM nodes.
extern const char * ggmlR_custom_check_sched(struct ggml_cgraph * graph,
                                             ggml_backend_sched_t sched);

// ============================================================================
// Backend Scheduler Functions
// ============================================================================

// Create a new backend scheduler
// backends: list of backend pointers
// parallel: whether to run backends in parallel
// graph_size: expected maximum graph size (default: 2048)
// add_cpu: whether to automatically add CPU backend (default: TRUE)
// Finalizer: free the scheduler (and the CPU backend it owns, stored in the
// protected tag) when the external pointer is GC'd. Cleared on manual
// R_ggml_backend_sched_free, so this never double-frees.
static void r_ggml_sched_finalizer(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t) R_ExternalPtrAddr(sched_ptr);
    if (sched != NULL) {
        ggml_backend_sched_free(sched);
        R_ClearExternalPtr(sched_ptr);

        SEXP cpu_ptr = R_ExternalPtrProtected(sched_ptr);
        if (cpu_ptr != R_NilValue) {
            ggml_backend_t cpu_backend = (ggml_backend_t) R_ExternalPtrAddr(cpu_ptr);
            if (cpu_backend != NULL) {
                ggml_backend_free(cpu_backend);
                R_ClearExternalPtr(cpu_ptr);
            }
        }
    }
}

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
        if (TYPEOF(backend_ptr) != EXTPTRSXP) {
            free(backends);
            error("backends[[%d]] is not an external pointer", i + 1);
        }
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
    ggml_backend_cpu_set_n_threads(backends[n_backends - 1], ggmlR_get_n_threads());

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

    // Store CPU backend pointer to free it later
    ggml_backend_t cpu_backend = backends[n_backends - 1];

    free(backends);

    if (sched == NULL) {
        // Free CPU backend if scheduler creation failed
        ggml_backend_free(cpu_backend);
        error("Failed to create backend scheduler");
    }

    // External pointer layout:
    //   prot = cpu_ptr   — the CPU backend we created here (freed in finalizer)
    //   tag  = backends_list — the user backends, kept reachable so their own
    //                          finalizers cannot fire before the scheduler's
    //                          (sched_free dereferences these backends)
    SEXP cpu_ptr = PROTECT(R_MakeExternalPtr(cpu_backend, R_NilValue, R_NilValue));
    SEXP ptr = PROTECT(R_MakeExternalPtr(sched, backends_list, cpu_ptr));
    R_RegisterCFinalizerEx(ptr, r_ggml_sched_finalizer, TRUE);
    UNPROTECT(2);
    return ptr;
}

// Free backend scheduler
SEXP R_ggml_backend_sched_free(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t) r_ptr_freeable(sched_ptr, "scheduler");

    if (sched != NULL) {
        ggml_backend_sched_free(sched);
        R_ClearExternalPtr(sched_ptr);

        // Free the CPU backend that was created in R_ggml_backend_sched_new
        SEXP cpu_ptr = R_ExternalPtrProtected(sched_ptr);
        if (cpu_ptr != R_NilValue) {
            ggml_backend_t cpu_backend = (ggml_backend_t)R_ExternalPtrAddr(cpu_ptr);
            if (cpu_backend != NULL) {
                ggml_backend_free(cpu_backend);
                R_ClearExternalPtr(cpu_ptr);
            }
        }
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
    ggml_backend_sched_t sched = (ggml_backend_sched_t) r_ptr_required(sched_ptr, "scheduler");
    struct ggml_tensor * tensor = (struct ggml_tensor *) r_ptr_required(tensor_ptr, "tensor");
    ggml_backend_t backend = (ggml_backend_t) r_ptr_required(backend_ptr, "backend");

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

    // The upstream lookup reads the backend index out of tensor->extra and
    // indexes sched->backends with it, returning NULL only for a clean -1. A
    // tensor the scheduler has not assigned -- asking after ggml_backend_sched_reserve()
    // but before a compute, or about a tensor from another graph -- can carry a
    // stale index instead, and the pointer that comes back is then garbage.
    // Dereferencing it (ggml_backend_name(), say) segfaults the R session, which
    // an exported, documented function must never do. Only hand back a pointer
    // that is genuinely one of this scheduler's backends.
    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    int known = 0;
    for (int i = 0; i < n_backends; i++) {
        if (ggml_backend_sched_get_backend(sched, i) == backend) {
            known = 1;
            break;
        }
    }
    if (!known) {
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
// Per-node tracing for debugging backend divergence.
//
// Reading intermediate tensors after ggml_backend_sched_graph_compute() returns
// is unreliable: the scheduler re-aliases intermediate buffers once the graph is
// done, so the values read back may belong to a different node. The eval
// callback fires right after each node is computed, while its buffer is still
// valid, which is the only way to compare two backends node by node.
//
// Enabled per call via R_ggml_backend_sched_trace(); prints
//   name | op | ne[] | sum | min | max
// to stderr for every node, so two runs can be diffed by NAME rather than by
// node index (the graphs on two backends need not enumerate nodes alike).
static bool r_ggml_trace_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (ask) {
        return true;   // observe every node
    }
    if (t == NULL || t->type != GGML_TYPE_F32) {
        return true;
    }

    const int64_t n = ggml_nelements(t);
    float * buf = (float *) malloc((size_t) n * sizeof(float));
    if (buf == NULL) {
        return true;
    }
    ggml_backend_tensor_get(t, buf, 0, (size_t) n * sizeof(float));

    double sum = 0.0;
    float mn = buf[0], mx = buf[0];
    for (int64_t i = 0; i < n; i++) {
        sum += buf[i];
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
    }
    free(buf);

    REprintf("[trace] %-34s %-12s ne=[%lld,%lld,%lld,%lld] sum=%.6f min=%.6f max=%.6f\n",
            t->name[0] ? t->name : "<unnamed>", ggml_op_name(t->op),
            (long long) t->ne[0], (long long) t->ne[1],
            (long long) t->ne[2], (long long) t->ne[3],
            sum, (double) mn, (double) mx);
    return true;
}

SEXP R_ggml_backend_sched_trace(SEXP sched_ptr, SEXP enable) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (asLogical(enable)) {
        ggml_backend_sched_set_eval_callback(sched, r_ggml_trace_eval_cb, NULL);
    } else {
        ggml_backend_sched_set_eval_callback(sched, NULL, NULL);
    }
    return R_NilValue;
}

SEXP R_ggml_backend_sched_graph_compute(SEXP sched_ptr, SEXP graph_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    const char * bad = ggmlR_custom_check_sched(graph, sched);
    if (bad != NULL) {
        error("Custom op node '%s' cannot run: ggml_custom() kernels are "
              "CPU-only and this scheduler has no CPU backend to fall back to. "
              "Include a CPU backend when creating the scheduler.", bad);
    }

    r_sched_sync_cpu_threads(sched);
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

    const char * bad = ggmlR_custom_check_sched(graph, sched);
    if (bad != NULL) {
        error("Custom op node '%s' cannot run: ggml_custom() kernels are "
              "CPU-only and this scheduler has no CPU backend to fall back to. "
              "Include a CPU backend when creating the scheduler.", bad);
    }

    r_sched_sync_cpu_threads(sched);
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
