// Re-apply the current ggmlR thread setting to a scheduler's CPU backends.
//
// A CPU backend is given ggmlR_get_n_threads() when it is created, so this only
// matters once ggml_set_n_threads() is called afterwards. Two paths need it,
// which is why it lives in a header rather than in either .c file:
//
//   - R_ggml_backend_sched_graph_compute() and its async twin, which sync right
//     before handing a graph to the scheduler;
//   - the training entry points (R_ggml_opt_fit, R_ggml_opt_fit_multi) and the
//     optimizer inits, since ggml_opt_eval() calls
//     ggml_backend_sched_graph_compute() straight from C and therefore never
//     passes through the wrappers above.
//
// An R-side epoch loop has no single "before the loop" moment, so it re-syncs
// per epoch through R_ggml_sched_sync_threads (see .ggml_sched_sync_threads).

#ifndef R_SCHED_THREADS_H
#define R_SCHED_THREADS_H

#include "ggml-backend.h"
#include "ggml-cpu.h"

extern int ggmlR_get_n_threads(void);

static inline void r_sched_sync_cpu_threads(ggml_backend_sched_t sched) {
    int n_threads = ggmlR_get_n_threads();
    int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; i++) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_is_cpu(b)) {
            ggml_backend_cpu_set_n_threads(b, n_threads);
        }
    }
}

#endif // R_SCHED_THREADS_H
