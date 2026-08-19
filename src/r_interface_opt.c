// Optimization R interface
// Training and fine-tuning support through ggml-opt

#include <R.h>
#include <Rinternals.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opt.h"
#include "r_ptr_check.h"
#include "r_sched_threads.h"
#include <stdlib.h>  // malloc/free
#include <string.h>  // memcpy
#include <assert.h>

// ============================================================================
// Loss Type Constants
// ============================================================================

SEXP R_ggml_opt_loss_type_mean(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_MEAN);
}

SEXP R_ggml_opt_loss_type_sum(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_SUM);
}

SEXP R_ggml_opt_loss_type_cross_entropy(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
}

SEXP R_ggml_opt_loss_type_mse(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
}

SEXP R_ggml_opt_loss_type_weighted_mse(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_WEIGHTED_MEAN_SQUARED_ERROR);
}

SEXP R_ggml_opt_loss_type_mae(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_MEAN_ABSOLUTE_ERROR);
}

SEXP R_ggml_opt_loss_type_huber(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_HUBER);
}

SEXP R_ggml_opt_loss_type_binary_cross_entropy(void) {
    return ScalarInteger(GGML_OPT_LOSS_TYPE_BINARY_CROSS_ENTROPY);
}

// ============================================================================
// Optimizer Type Constants
// ============================================================================

SEXP R_ggml_opt_optimizer_type_adamw(void) {
    return ScalarInteger(GGML_OPT_OPTIMIZER_TYPE_ADAMW);
}

SEXP R_ggml_opt_optimizer_type_sgd(void) {
    return ScalarInteger(GGML_OPT_OPTIMIZER_TYPE_SGD);
}

// ============================================================================
// Dataset Functions
// ============================================================================

// Create a new dataset
SEXP R_ggml_opt_dataset_init(SEXP type_data, SEXP type_label,
                              SEXP ne_datapoint, SEXP ne_label,
                              SEXP ndata, SEXP ndata_shard) {
    enum ggml_type t_data = (enum ggml_type)asInteger(type_data);
    enum ggml_type t_label = (enum ggml_type)asInteger(type_label);
    int64_t n_dp = (int64_t)asReal(ne_datapoint);
    int64_t n_lb = (int64_t)asReal(ne_label);
    int64_t n_data = (int64_t)asReal(ndata);
    int64_t n_shard = (int64_t)asReal(ndata_shard);

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(t_data, t_label, n_dp, n_lb, n_data, n_shard);

    if (dataset == NULL) {
        error("Failed to create dataset");
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(dataset, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Free dataset
SEXP R_ggml_opt_dataset_free(SEXP dataset_ptr) {
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t) r_ptr_freeable(dataset_ptr, "dataset");

    if (dataset != NULL) {
        ggml_opt_dataset_free(dataset);
        R_ClearExternalPtr(dataset_ptr);
    }

    return R_NilValue;
}

// Get number of data points
SEXP R_ggml_opt_dataset_ndata(SEXP dataset_ptr) {
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }

    int64_t n = ggml_opt_dataset_ndata(dataset);
    return ScalarReal((double)n);
}

// Get data tensor
SEXP R_ggml_opt_dataset_data(SEXP dataset_ptr) {
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }

    struct ggml_tensor * data = ggml_opt_dataset_data(dataset);

    SEXP ptr = PROTECT(R_MakeExternalPtr(data, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get labels tensor
SEXP R_ggml_opt_dataset_labels(SEXP dataset_ptr) {
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }

    struct ggml_tensor * labels = ggml_opt_dataset_labels(dataset);

    if (labels == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(labels, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get (lazily allocating) per-datapoint weights tensor [1, ndata]
SEXP R_ggml_opt_dataset_weights(SEXP dataset_ptr) {
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }

    struct ggml_tensor * weights = ggml_opt_dataset_weights(dataset);

    SEXP ptr = PROTECT(R_MakeExternalPtr(weights, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Shuffle dataset
SEXP R_ggml_opt_dataset_shuffle(SEXP opt_ctx_ptr, SEXP dataset_ptr, SEXP idata) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }
    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }

    int64_t i = (int64_t)asReal(idata);
    ggml_opt_dataset_shuffle(opt_ctx, dataset, i);

    return R_NilValue;
}

// Get batch from dataset
SEXP R_ggml_opt_dataset_get_batch(SEXP dataset_ptr, SEXP data_batch_ptr,
                                   SEXP labels_batch_ptr, SEXP ibatch) {
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t) r_ptr_required(dataset_ptr, "dataset");
    struct ggml_tensor * data_batch = (struct ggml_tensor *) r_ptr_required(data_batch_ptr, "data_batch tensor");
    struct ggml_tensor * labels_batch =
        (struct ggml_tensor *) r_ptr_or_null(labels_batch_ptr, "labels_batch");

    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }
    if (data_batch == NULL) {
        error("Invalid data_batch tensor pointer");
    }

    int64_t i = (int64_t)asReal(ibatch);
    ggml_opt_dataset_get_batch(dataset, data_batch, labels_batch, i);

    return R_NilValue;
}

// Get one output head's slice of the batch labels (multi-loss).
// data_batch may be NULL to copy only the labels: the caller copies the inputs
// once, with the first head, and the remaining heads pull just their slice.
// labels_off is 0-based, in elements, within a concatenated label row.
SEXP R_ggml_opt_dataset_get_batch_head(SEXP dataset_ptr, SEXP data_batch_ptr,
                                       SEXP labels_batch_ptr, SEXP labels_off,
                                       SEXP ibatch) {
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t) r_ptr_required(dataset_ptr, "dataset");
    struct ggml_tensor * data_batch =
        (struct ggml_tensor *) r_ptr_or_null(data_batch_ptr, "data_batch");
    struct ggml_tensor * labels_batch =
        (struct ggml_tensor *) r_ptr_required(labels_batch_ptr, "labels_batch tensor");

    const double off_d = asReal(labels_off);
    if (!R_FINITE(off_d) || off_d < 0) {
        error("labels_off must be a non-negative number");
    }

    ggml_opt_dataset_get_batch_head(dataset, data_batch, labels_batch,
                                    (int64_t) off_d, (int64_t) asReal(ibatch));
    return R_NilValue;
}

// ============================================================================
// Optimizer Context Functions
// ============================================================================

// Get default optimizer params
SEXP R_ggml_opt_default_params(SEXP sched_ptr, SEXP loss_type) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    enum ggml_opt_loss_type lt = (enum ggml_opt_loss_type)asInteger(loss_type);
    struct ggml_opt_params params = ggml_opt_default_params(sched, lt);

    // Return as a list
    SEXP result = PROTECT(allocVector(VECSXP, 4));
    SEXP names = PROTECT(allocVector(STRSXP, 4));

    SET_STRING_ELT(names, 0, mkChar("loss_type"));
    SET_STRING_ELT(names, 1, mkChar("build_type"));
    SET_STRING_ELT(names, 2, mkChar("opt_period"));
    SET_STRING_ELT(names, 3, mkChar("optimizer"));

    SET_VECTOR_ELT(result, 0, ScalarInteger(params.loss_type));
    SET_VECTOR_ELT(result, 1, ScalarInteger(params.build_type));
    SET_VECTOR_ELT(result, 2, ScalarInteger(params.opt_period));
    SET_VECTOR_ELT(result, 3, ScalarInteger(params.optimizer));

    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

// Initialize optimizer context
SEXP R_ggml_opt_init(SEXP sched_ptr, SEXP loss_type, SEXP optimizer_type, SEXP opt_period,
                     SEXP ctx_compute_ptr, SEXP inputs_ptr, SEXP outputs_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    enum ggml_opt_loss_type lt = (enum ggml_opt_loss_type)asInteger(loss_type);
    struct ggml_opt_params params = ggml_opt_default_params(sched, lt);

    params.optimizer = (enum ggml_opt_optimizer_type)asInteger(optimizer_type);
    params.opt_period = asInteger(opt_period);

    // Optional: set ctx_compute, inputs, outputs for static graph mode
    if (ctx_compute_ptr != R_NilValue) {
        params.ctx_compute = (struct ggml_context *)R_ExternalPtrAddr(ctx_compute_ptr);
    }
    if (inputs_ptr != R_NilValue) {
        params.inputs = (struct ggml_tensor *)R_ExternalPtrAddr(inputs_ptr);
    }
    if (outputs_ptr != R_NilValue) {
        params.outputs = (struct ggml_tensor *)R_ExternalPtrAddr(outputs_ptr);
    }

    r_sched_sync_cpu_threads(sched);

    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    if (opt_ctx == NULL) {
        error("Failed to initialize optimizer context");
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(opt_ctx, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Free optimizer context
SEXP R_ggml_opt_free(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t) r_ptr_freeable(opt_ctx_ptr, "optimizer context");

    if (opt_ctx != NULL) {
        ggml_opt_free(opt_ctx);
        R_ClearExternalPtr(opt_ctx_ptr);
    }

    return R_NilValue;
}

// Reset optimizer context
SEXP R_ggml_opt_reset(SEXP opt_ctx_ptr, SEXP optimizer) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    bool reset_optimizer = asLogical(optimizer);
    ggml_opt_reset(opt_ctx, reset_optimizer);

    return R_NilValue;
}

// Check if using static graphs
SEXP R_ggml_opt_static_graphs(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    bool is_static = ggml_opt_static_graphs(opt_ctx);
    return ScalarLogical(is_static);
}

// Get inputs tensor
SEXP R_ggml_opt_inputs(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);

    if (inputs == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(inputs, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get outputs tensor
SEXP R_ggml_opt_outputs(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    struct ggml_tensor * outputs = ggml_opt_outputs(opt_ctx);

    if (outputs == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(outputs, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get labels tensor
SEXP R_ggml_opt_labels(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);

    if (labels == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(labels, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get loss tensor
SEXP R_ggml_opt_loss(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    struct ggml_tensor * loss = ggml_opt_loss(opt_ctx);

    if (loss == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(loss, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get predictions tensor
SEXP R_ggml_opt_pred(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    struct ggml_tensor * pred = ggml_opt_pred(opt_ctx);

    if (pred == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(pred, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get number of correct predictions tensor
SEXP R_ggml_opt_ncorrect(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    struct ggml_tensor * ncorrect = ggml_opt_ncorrect(opt_ctx);

    if (ncorrect == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(ncorrect, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}


// ---- per-head accessors (multi-loss) ----
//
// These take a 1-based head index from R and pass a 0-based one to C. The
// arity-1 accessors above are kept as-is (head 0) so existing R callers and
// downstream packages keep working unchanged.

// Number of output heads the optimizer context was built with
SEXP R_ggml_opt_n_loss(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t) r_ptr_required(opt_ctx_ptr, "optimizer context");
    return ScalarInteger((int) ggml_opt_n_loss(opt_ctx));
}

// Shared body of the per-head accessors: validates the index against the
// context's head count and wraps the returned tensor (NULL -> R_NilValue).
static SEXP r_opt_head_tensor(SEXP opt_ctx_ptr, SEXP ihead,
                              struct ggml_tensor * (*getter)(ggml_opt_context_t, int64_t),
                              const char * what) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t) r_ptr_required(opt_ctx_ptr, "optimizer context");

    const int i = asInteger(ihead);
    const int n = (int) ggml_opt_n_loss(opt_ctx);
    // Checked here rather than in C: the C helper asserts, which would abort R.
    if (i == NA_INTEGER || i < 1 || i > n) {
        error("head index %d out of range for %s: model has %d output head%s",
              i == NA_INTEGER ? 0 : i, what, n, n == 1 ? "" : "s");
    }

    struct ggml_tensor * t = getter(opt_ctx, (int64_t) (i - 1));
    if (t == NULL) {
        return R_NilValue;
    }
    SEXP ptr = PROTECT(R_MakeExternalPtr(t, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_opt_outputs_i(SEXP opt_ctx_ptr, SEXP ihead) {
    return r_opt_head_tensor(opt_ctx_ptr, ihead, ggml_opt_outputs_i, "outputs");
}

SEXP R_ggml_opt_labels_i(SEXP opt_ctx_ptr, SEXP ihead) {
    return r_opt_head_tensor(opt_ctx_ptr, ihead, ggml_opt_labels_i, "labels");
}

SEXP R_ggml_opt_loss_weights_i(SEXP opt_ctx_ptr, SEXP ihead) {
    return r_opt_head_tensor(opt_ctx_ptr, ihead, ggml_opt_loss_weights_i, "loss weights");
}

SEXP R_ggml_opt_pred_i(SEXP opt_ctx_ptr, SEXP ihead) {
    return r_opt_head_tensor(opt_ctx_ptr, ihead, ggml_opt_pred_i, "predictions");
}

SEXP R_ggml_opt_ncorrect_i(SEXP opt_ctx_ptr, SEXP ihead) {
    return r_opt_head_tensor(opt_ctx_ptr, ihead, ggml_opt_ncorrect_i, "ncorrect");
}

// Per-head loss scalar BEFORE weighting/reduction (for the training history).
// R_ggml_opt_loss returns the weighted total that is actually optimized.
SEXP R_ggml_opt_loss_i(SEXP opt_ctx_ptr, SEXP ihead) {
    return r_opt_head_tensor(opt_ctx_ptr, ihead, ggml_opt_loss_i, "loss");
}

// Get optimizer type
SEXP R_ggml_opt_context_optimizer_type(SEXP opt_ctx_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    enum ggml_opt_optimizer_type opt_type = ggml_opt_context_optimizer_type(opt_ctx);
    return ScalarInteger((int)opt_type);
}

// Get optimizer name
SEXP R_ggml_opt_optimizer_name(SEXP optimizer_type) {
    enum ggml_opt_optimizer_type opt_type = (enum ggml_opt_optimizer_type)asInteger(optimizer_type);
    const char * name = ggml_opt_optimizer_name(opt_type);
    return mkString(name);
}

// ============================================================================
// Result Functions
// ============================================================================

// Initialize result
SEXP R_ggml_opt_result_init(void) {
    ggml_opt_result_t result = ggml_opt_result_init();

    if (result == NULL) {
        error("Failed to initialize result");
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(result, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Free result
SEXP R_ggml_opt_result_free(SEXP result_ptr) {
    ggml_opt_result_t result = (ggml_opt_result_t) r_ptr_freeable(result_ptr, "result");

    if (result != NULL) {
        ggml_opt_result_free(result);
        R_ClearExternalPtr(result_ptr);
    }

    return R_NilValue;
}

// Reset result
SEXP R_ggml_opt_result_reset(SEXP result_ptr) {
    ggml_opt_result_t result = (ggml_opt_result_t)R_ExternalPtrAddr(result_ptr);

    if (result == NULL) {
        error("Invalid result pointer");
    }

    ggml_opt_result_reset(result);
    return R_NilValue;
}

// Get number of data points from result
SEXP R_ggml_opt_result_ndata(SEXP result_ptr) {
    ggml_opt_result_t result = (ggml_opt_result_t)R_ExternalPtrAddr(result_ptr);

    if (result == NULL) {
        error("Invalid result pointer");
    }

    int64_t ndata;
    ggml_opt_result_ndata(result, &ndata);
    return ScalarReal((double)ndata);
}

// Get loss from result
SEXP R_ggml_opt_result_loss(SEXP result_ptr) {
    ggml_opt_result_t result = (ggml_opt_result_t)R_ExternalPtrAddr(result_ptr);

    if (result == NULL) {
        error("Invalid result pointer");
    }

    double loss, unc;
    ggml_opt_result_loss(result, &loss, &unc);

    SEXP r = PROTECT(allocVector(REALSXP, 2));
    REAL(r)[0] = loss;
    REAL(r)[1] = unc;

    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("loss"));
    SET_STRING_ELT(names, 1, mkChar("uncertainty"));
    setAttrib(r, R_NamesSymbol, names);

    UNPROTECT(2);
    return r;
}

// Get accuracy from result
SEXP R_ggml_opt_result_accuracy(SEXP result_ptr) {
    ggml_opt_result_t result = (ggml_opt_result_t)R_ExternalPtrAddr(result_ptr);

    if (result == NULL) {
        error("Invalid result pointer");
    }

    double accuracy, unc;
    ggml_opt_result_accuracy(result, &accuracy, &unc);

    SEXP r = PROTECT(allocVector(REALSXP, 2));
    REAL(r)[0] = accuracy;
    REAL(r)[1] = unc;

    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("accuracy"));
    SET_STRING_ELT(names, 1, mkChar("uncertainty"));
    setAttrib(r, R_NamesSymbol, names);

    UNPROTECT(2);
    return r;
}

// ---- per-head result accessors (multi-loss) ----
//
// ggml_opt_result_loss()/_accuracy() report the weighted total the optimizer
// actually minimizes; these report one head on its own, for the training
// history. Head indices are 1-based on the R side, 0-based in C.

// Number of heads accumulated in a result. Zero before the first epoch, so the
// caller must treat a fresh result as "no per-head data yet".
SEXP R_ggml_opt_result_n_loss(SEXP result_ptr) {
    ggml_opt_result_t result = (ggml_opt_result_t) r_ptr_required(result_ptr, "result");
    return ScalarInteger((int) ggml_opt_result_n_loss(result));
}

// Shared body: validates the 1-based head index against the result's head count
// and returns c(<value>, uncertainty). The C helpers assert on a bad index,
// which would abort R, so the range check happens here.
static SEXP r_opt_result_head_value(SEXP result_ptr, SEXP ihead,
                                    void (*getter)(ggml_opt_result_t, int64_t, double *, double *),
                                    const char * what) {
    ggml_opt_result_t result = (ggml_opt_result_t) r_ptr_required(result_ptr, "result");

    const int i = asInteger(ihead);
    const int n = (int) ggml_opt_result_n_loss(result);
    if (i == NA_INTEGER || i < 1 || i > n) {
        error("head index %d out of range for %s: result holds %d output head%s",
              i == NA_INTEGER ? 0 : i, what, n, n == 1 ? "" : "s");
    }

    double value = NA_REAL, unc = NA_REAL;
    getter(result, (int64_t) (i - 1), &value, &unc);

    SEXP r = PROTECT(allocVector(REALSXP, 2));
    // A head whose loss is not cross-entropy has no accuracy; ggml reports that
    // as NaN, which R conventions want as NA.
    REAL(r)[0] = (value != value) ? NA_REAL : value;
    REAL(r)[1] = (unc   != unc)   ? NA_REAL : unc;

    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar(what));
    SET_STRING_ELT(names, 1, mkChar("uncertainty"));
    setAttrib(r, R_NamesSymbol, names);

    UNPROTECT(2);
    return r;
}

SEXP R_ggml_opt_result_loss_i(SEXP result_ptr, SEXP ihead) {
    return r_opt_result_head_value(result_ptr, ihead, ggml_opt_result_loss_i, "loss");
}

SEXP R_ggml_opt_result_accuracy_i(SEXP result_ptr, SEXP ihead) {
    return r_opt_result_head_value(result_ptr, ihead, ggml_opt_result_accuracy_i, "accuracy");
}

// ============================================================================
// Computation Functions
// ============================================================================

// Allocate graph for evaluation
SEXP R_ggml_opt_alloc(SEXP opt_ctx_ptr, SEXP backward) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    bool do_backward = asLogical(backward);
    ggml_opt_alloc(opt_ctx, do_backward);

    return R_NilValue;
}

// Evaluate (forward pass, optionally backward pass)
SEXP R_ggml_opt_eval(SEXP opt_ctx_ptr, SEXP result_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);
    ggml_opt_result_t result = NULL;

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }

    if (result_ptr != R_NilValue) {
        result = (ggml_opt_result_t)R_ExternalPtrAddr(result_ptr);
    }

    ggml_opt_eval(opt_ctx, result);

    return R_NilValue;
}

// ============================================================================
// High-Level Training Function
// ============================================================================

// Helper: get batch size from last dimension of tensor
static int64_t r_ggml_opt_batch_size(const struct ggml_tensor * t) {
    return t->ne[ggml_n_dims(t) - 1];
}

// Fit model to dataset, returning history (loss/accuracy per epoch)
SEXP R_ggml_opt_fit(SEXP sched_ptr, SEXP ctx_compute_ptr,
                    SEXP inputs_ptr, SEXP outputs_ptr,
                    SEXP dataset_ptr, SEXP loss_type, SEXP optimizer_type,
                    SEXP nepoch, SEXP nbatch_logical, SEXP val_split, SEXP silent) {

    ggml_backend_sched_t sched = (ggml_backend_sched_t) r_ptr_required(sched_ptr, "scheduler");
    struct ggml_context * ctx_compute = (struct ggml_context *) r_ptr_required(ctx_compute_ptr, "compute context");
    struct ggml_tensor * inputs = (struct ggml_tensor *) r_ptr_required(inputs_ptr, "inputs tensor");
    struct ggml_tensor * outputs = (struct ggml_tensor *) r_ptr_required(outputs_ptr, "outputs tensor");
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t) r_ptr_required(dataset_ptr, "dataset");

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }
    if (ctx_compute == NULL) {
        error("Invalid compute context pointer");
    }
    if (inputs == NULL) {
        error("Invalid inputs tensor pointer");
    }
    if (outputs == NULL) {
        error("Invalid outputs tensor pointer");
    }
    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }

    enum ggml_opt_loss_type lt = (enum ggml_opt_loss_type)asInteger(loss_type);
    enum ggml_opt_optimizer_type ot = (enum ggml_opt_optimizer_type)asInteger(optimizer_type);
    int64_t n_epoch = (int64_t)asReal(nepoch);
    int64_t n_batch = (int64_t)asReal(nbatch_logical);
    float v_split = (float)asReal(val_split);
    bool is_silent = asLogical(silent);

    // Compute parameters (mirroring ggml_opt_fit logic from ggml-opt.cpp)
    const int64_t ndata = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = r_ggml_opt_batch_size(inputs);
    const int64_t opt_period = n_batch / nbatch_physical;
    const int64_t nbatches_logical = ndata / n_batch;
    const int64_t ibatch_split = (int64_t)(((1.0f - v_split) * nbatches_logical)) * opt_period;
    const int64_t idata_split = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    struct ggml_opt_params params = ggml_opt_default_params(sched, lt);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = ggml_opt_get_default_optimizer_params;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = ot;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    if (n_batch < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1);
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = is_silent ? NULL : ggml_opt_epoch_callback_progress_bar;

    r_sched_sync_cpu_threads(sched);

    // Allocate history arrays
    double * hist_train_loss = (double *)R_alloc(n_epoch, sizeof(double));
    double * hist_train_acc  = (double *)R_alloc(n_epoch, sizeof(double));
    double * hist_val_loss   = (double *)R_alloc(n_epoch, sizeof(double));
    double * hist_val_acc    = (double *)R_alloc(n_epoch, sizeof(double));

    for (; epoch <= n_epoch; ++epoch) {
        if (n_batch < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!is_silent) {
            Rprintf("Epoch %d/%d:\n", (int)epoch, (int)n_epoch);
        }

        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val,
                       idata_split, epoch_callback, epoch_callback);

        if (!is_silent) {
            Rprintf("\n");
        }

        // Collect metrics
        int idx = (int)(epoch - 1);

        ggml_opt_result_loss(result_train, &hist_train_loss[idx], NULL);
        ggml_opt_result_accuracy(result_train, &hist_train_acc[idx], NULL);

        if (v_split > 0.0f) {
            ggml_opt_result_loss(result_val, &hist_val_loss[idx], NULL);
            ggml_opt_result_accuracy(result_val, &hist_val_acc[idx], NULL);
        } else {
            hist_val_loss[idx] = NA_REAL;
            hist_val_acc[idx]  = NA_REAL;
        }
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);

    // Build R list with history
    SEXP r_train_loss = PROTECT(allocVector(REALSXP, n_epoch));
    SEXP r_train_acc  = PROTECT(allocVector(REALSXP, n_epoch));
    SEXP r_val_loss   = PROTECT(allocVector(REALSXP, n_epoch));
    SEXP r_val_acc    = PROTECT(allocVector(REALSXP, n_epoch));

    memcpy(REAL(r_train_loss), hist_train_loss, n_epoch * sizeof(double));
    memcpy(REAL(r_train_acc),  hist_train_acc,  n_epoch * sizeof(double));
    memcpy(REAL(r_val_loss),   hist_val_loss,   n_epoch * sizeof(double));
    memcpy(REAL(r_val_acc),    hist_val_acc,    n_epoch * sizeof(double));

    SEXP result = PROTECT(allocVector(VECSXP, 4));
    SEXP names = PROTECT(allocVector(STRSXP, 4));

    SET_VECTOR_ELT(result, 0, r_train_loss);
    SET_VECTOR_ELT(result, 1, r_train_acc);
    SET_VECTOR_ELT(result, 2, r_val_loss);
    SET_VECTOR_ELT(result, 3, r_val_acc);

    SET_STRING_ELT(names, 0, mkChar("train_loss"));
    SET_STRING_ELT(names, 1, mkChar("train_accuracy"));
    SET_STRING_ELT(names, 2, mkChar("val_loss"));
    SET_STRING_ELT(names, 3, mkChar("val_accuracy"));

    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(6);
    return result;
}

// ---- multi-head fit ----

// Per-head losses and accuracies now come from ggml_opt_result itself
// (ggml_opt_result_loss_i / _accuracy_i), so no epoch-callback accumulator and
// no file-static state are needed here.
#define R_OPT_MAX_HEADS 32

// Fit a multi-output model. outputs_list/loss_types/loss_weights/labels_offs
// are parallel, one entry per output head. Returns the usual history plus a
// "head_loss" matrix [n_epoch x n_head] of the unweighted per-head losses, so
// the caller can see which head is not learning.
SEXP R_ggml_opt_fit_multi(SEXP sched_ptr, SEXP ctx_compute_ptr,
                          SEXP inputs_ptr, SEXP outputs_list,
                          SEXP dataset_ptr, SEXP loss_types, SEXP loss_weights,
                          SEXP labels_offs, SEXP optimizer_type,
                          SEXP nepoch, SEXP nbatch_logical, SEXP val_split, SEXP silent) {

    ggml_backend_sched_t sched = (ggml_backend_sched_t) r_ptr_required(sched_ptr, "scheduler");
    struct ggml_context * ctx_compute = (struct ggml_context *) r_ptr_required(ctx_compute_ptr, "compute context");
    struct ggml_tensor * inputs = (struct ggml_tensor *) r_ptr_required(inputs_ptr, "inputs tensor");
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t) r_ptr_required(dataset_ptr, "dataset");

    if (TYPEOF(outputs_list) != VECSXP) {
        error("outputs must be a list of tensor pointers");
    }
    const int n_head = LENGTH(outputs_list);
    if (n_head < 1) {
        error("outputs must contain at least one output head");
    }
    if (n_head > R_OPT_MAX_HEADS) {
        error("too many output heads: %d (max %d)", n_head, R_OPT_MAX_HEADS);
    }
    if (LENGTH(loss_types) != n_head) {
        error("loss must have one entry per output head (%d given, %d expected)",
              LENGTH(loss_types), n_head);
    }
    if (LENGTH(loss_weights) != n_head) {
        error("loss_weights must have one entry per output head (%d given, %d expected)",
              LENGTH(loss_weights), n_head);
    }
    if (LENGTH(labels_offs) != n_head) {
        error("labels_offs must have one entry per output head (%d given, %d expected)",
              LENGTH(labels_offs), n_head);
    }

    struct ggml_tensor * outs[R_OPT_MAX_HEADS];
    enum ggml_opt_loss_type lts[R_OPT_MAX_HEADS];
    float                   lws[R_OPT_MAX_HEADS];
    int64_t                 offs[R_OPT_MAX_HEADS];

    for (int i = 0; i < n_head; ++i) {
        outs[i] = (struct ggml_tensor *) r_ptr_required(VECTOR_ELT(outputs_list, i), "output tensor");
        lts[i]  = (enum ggml_opt_loss_type) INTEGER(loss_types)[i];
        lws[i]  = (float) REAL(loss_weights)[i];
        offs[i] = (int64_t) REAL(labels_offs)[i];
    }

    enum ggml_opt_optimizer_type ot = (enum ggml_opt_optimizer_type) asInteger(optimizer_type);
    int64_t n_epoch = (int64_t) asReal(nepoch);
    int64_t n_batch = (int64_t) asReal(nbatch_logical);
    float   v_split = (float)   asReal(val_split);
    bool    is_silent = asLogical(silent);

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = r_ggml_opt_batch_size(inputs);
    if (nbatch_physical <= 0 || n_batch % nbatch_physical != 0) {
        error("nbatch_logical (%g) must be a multiple of the physical batch size (%g)",
              (double) n_batch, (double) nbatch_physical);
    }
    const int64_t opt_period       = n_batch / nbatch_physical;
    const int64_t nbatches_logical = ndata / n_batch;
    const int64_t ibatch_split     = (int64_t)(((1.0f - v_split) * nbatches_logical)) * opt_period;
    const int64_t idata_split      = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    struct ggml_opt_params params = ggml_opt_default_params_multi(sched, n_head, lts, lws);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs_multi   = outs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = ggml_opt_get_default_optimizer_params;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = ot;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    ggml_opt_set_labels_offs(opt_ctx, n_head, offs);

    if (n_batch < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1);
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    r_sched_sync_cpu_threads(sched);

    double * hist_train_loss = (double *) R_alloc(n_epoch, sizeof(double));
    double * hist_train_acc  = (double *) R_alloc(n_epoch, sizeof(double));
    double * hist_val_loss   = (double *) R_alloc(n_epoch, sizeof(double));
    double * hist_val_acc    = (double *) R_alloc(n_epoch, sizeof(double));
    // column-major [n_epoch x n_head], matching R's matrix layout
    double * hist_head_loss  = (double *) R_alloc(n_epoch * n_head, sizeof(double));
    double * hist_head_acc   = (double *) R_alloc(n_epoch * n_head, sizeof(double));

    for (; epoch <= n_epoch; ++epoch) {
        if (n_batch < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!is_silent) {
            Rprintf("Epoch %d/%d:\n", (int) epoch, (int) n_epoch);
        }

        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val,
                       idata_split, NULL, NULL);

        if (!is_silent) {
            Rprintf("\n");
        }

        const int idx = (int) (epoch - 1);

        ggml_opt_result_loss(result_train, &hist_train_loss[idx], NULL);
        ggml_opt_result_accuracy(result_train, &hist_train_acc[idx], NULL);

        // Per-head training loss and accuracy, straight from the result.
        for (int i = 0; i < n_head; ++i) {
            double lv = NA_REAL, av = NA_REAL;
            if (i < (int) ggml_opt_result_n_loss(result_train)) {
                ggml_opt_result_loss_i(result_train, (int64_t) i, &lv, NULL);
                ggml_opt_result_accuracy_i(result_train, (int64_t) i, &av, NULL);
            }
            hist_head_loss[idx + (size_t) i * n_epoch] = lv;
            // NaN (non-CE head) becomes NA on the R side.
            hist_head_acc[idx + (size_t) i * n_epoch] = (av != av) ? NA_REAL : av;
        }

        if (v_split > 0.0f) {
            ggml_opt_result_loss(result_val, &hist_val_loss[idx], NULL);
            ggml_opt_result_accuracy(result_val, &hist_val_acc[idx], NULL);
        } else {
            hist_val_loss[idx] = NA_REAL;
            hist_val_acc[idx]  = NA_REAL;
        }
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);

    SEXP r_train_loss = PROTECT(allocVector(REALSXP, n_epoch));
    SEXP r_train_acc  = PROTECT(allocVector(REALSXP, n_epoch));
    SEXP r_val_loss   = PROTECT(allocVector(REALSXP, n_epoch));
    SEXP r_val_acc    = PROTECT(allocVector(REALSXP, n_epoch));
    SEXP r_head_loss  = PROTECT(allocMatrix(REALSXP, n_epoch, n_head));
    SEXP r_head_acc   = PROTECT(allocMatrix(REALSXP, n_epoch, n_head));

    memcpy(REAL(r_train_loss), hist_train_loss, n_epoch * sizeof(double));
    memcpy(REAL(r_train_acc),  hist_train_acc,  n_epoch * sizeof(double));
    memcpy(REAL(r_val_loss),   hist_val_loss,   n_epoch * sizeof(double));
    memcpy(REAL(r_val_acc),    hist_val_acc,    n_epoch * sizeof(double));
    memcpy(REAL(r_head_loss),  hist_head_loss,  (size_t) n_epoch * n_head * sizeof(double));
    memcpy(REAL(r_head_acc),   hist_head_acc,   (size_t) n_epoch * n_head * sizeof(double));

    SEXP result = PROTECT(allocVector(VECSXP, 6));
    SEXP names  = PROTECT(allocVector(STRSXP, 6));

    SET_VECTOR_ELT(result, 0, r_train_loss);
    SET_VECTOR_ELT(result, 1, r_train_acc);
    SET_VECTOR_ELT(result, 2, r_val_loss);
    SET_VECTOR_ELT(result, 3, r_val_acc);
    SET_VECTOR_ELT(result, 4, r_head_loss);
    SET_VECTOR_ELT(result, 5, r_head_acc);

    SET_STRING_ELT(names, 0, mkChar("train_loss"));
    SET_STRING_ELT(names, 1, mkChar("train_accuracy"));
    SET_STRING_ELT(names, 2, mkChar("val_loss"));
    SET_STRING_ELT(names, 3, mkChar("val_accuracy"));
    SET_STRING_ELT(names, 4, mkChar("head_loss"));
    SET_STRING_ELT(names, 5, mkChar("head_accuracy"));

    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(8);
    return result;
}

// ============================================================================
// Additional Functions
// ============================================================================

// Get gradient accumulator for a node
SEXP R_ggml_opt_grad_acc(SEXP opt_ctx_ptr, SEXP node_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);
    struct ggml_tensor * node = (struct ggml_tensor *)R_ExternalPtrAddr(node_ptr);

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }
    if (node == NULL) {
        error("Invalid node tensor pointer");
    }

    struct ggml_tensor * grad_acc = ggml_opt_grad_acc(opt_ctx, node);

    if (grad_acc == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(grad_acc, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// Get predictions from result (returns integer vector of length ndata)
SEXP R_ggml_opt_result_pred(SEXP result_ptr) {
    ggml_opt_result_t result = (ggml_opt_result_t)R_ExternalPtrAddr(result_ptr);

    if (result == NULL) {
        error("Invalid result pointer");
    }

    // First get ndata
    int64_t ndata;
    ggml_opt_result_ndata(result, &ndata);

    if (ndata <= 0) {
        return allocVector(INTSXP, 0);
    }

    // Allocate R integer vector
    SEXP r_pred = PROTECT(allocVector(INTSXP, (R_xlen_t)ndata));

    // Get predictions
    ggml_opt_result_pred(result, INTEGER(r_pred));

    UNPROTECT(1);
    return r_pred;
}

// Prepare allocation for non-static graphs
SEXP R_ggml_opt_prepare_alloc(SEXP opt_ctx_ptr, SEXP ctx_compute_ptr,
                               SEXP graph_ptr, SEXP inputs_ptr, SEXP outputs_ptr) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t) r_ptr_required(opt_ctx_ptr, "optimizer context");
    struct ggml_context * ctx_compute = (struct ggml_context *) r_ptr_required(ctx_compute_ptr, "compute context");
    struct ggml_cgraph * gf = (struct ggml_cgraph *) r_ptr_required(graph_ptr, "graph");
    struct ggml_tensor * inputs = (struct ggml_tensor *) r_ptr_required(inputs_ptr, "inputs tensor");
    struct ggml_tensor * outputs = (struct ggml_tensor *) r_ptr_required(outputs_ptr, "outputs tensor");

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }
    if (ctx_compute == NULL) {
        error("Invalid compute context pointer");
    }
    if (gf == NULL) {
        error("Invalid graph pointer");
    }
    if (inputs == NULL) {
        error("Invalid inputs tensor pointer");
    }
    if (outputs == NULL) {
        error("Invalid outputs tensor pointer");
    }

    ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, inputs, outputs);

    return R_NilValue;
}

// ============================================================================
// R Callback Support for ggml_opt_epoch
// ============================================================================

// Global storage for R callback functions (protected from GC)
// Initialized to NULL, will be set to R_NilValue when needed
static SEXP g_callback_train = NULL;
static SEXP g_callback_eval = NULL;

// C wrapper that calls R callback function
static void r_callback_wrapper(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {

    // Select the appropriate R callback
    SEXP callback = train ? g_callback_train : g_callback_eval;

    if (callback == NULL || callback == R_NilValue || !Rf_isFunction(callback)) {
        return;
    }

    // Build the call: callback(train, ibatch, ibatch_max, t_start_us, result)
    // We pass simplified arguments that are easy to use in R
    SEXP call = PROTECT(Rf_allocVector(LANGSXP, 6));
    SETCAR(call, callback);

    SEXP args = CDR(call);
    SETCAR(args, Rf_ScalarLogical(train));

    args = CDR(args);
    SETCAR(args, Rf_ScalarReal((double)ibatch));

    args = CDR(args);
    SETCAR(args, Rf_ScalarReal((double)ibatch_max));

    args = CDR(args);
    SETCAR(args, Rf_ScalarReal((double)t_start_us));

    args = CDR(args);
    // Pass result as external pointer so user can query loss/accuracy
    SEXP result_ptr = PROTECT(R_MakeExternalPtr(result, R_NilValue, R_NilValue));
    SETCAR(args, result_ptr);

    // Evaluate the call in the global environment
    // Use R_tryEval to catch R errors without crashing
    int error_occurred = 0;
    R_tryEval(call, R_GlobalEnv, &error_occurred);

    if (error_occurred) {
        Rf_warning("Error in R callback function");
    }

    UNPROTECT(2);
}

// Run one epoch with R callback support
SEXP R_ggml_opt_epoch(SEXP opt_ctx_ptr, SEXP dataset_ptr,
                       SEXP result_train_ptr, SEXP result_eval_ptr,
                       SEXP idata_split, SEXP callback_train, SEXP callback_eval) {
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t) r_ptr_required(opt_ctx_ptr, "optimizer context");
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t) r_ptr_required(dataset_ptr, "dataset");

    ggml_opt_result_t result_train = NULL;
    ggml_opt_result_t result_eval = NULL;

    if (opt_ctx == NULL) {
        error("Invalid optimizer context pointer");
    }
    if (dataset == NULL) {
        error("Invalid dataset pointer");
    }

    if (result_train_ptr != R_NilValue) {
        result_train = (ggml_opt_result_t)R_ExternalPtrAddr(result_train_ptr);
    }
    if (result_eval_ptr != R_NilValue) {
        result_eval = (ggml_opt_result_t)R_ExternalPtrAddr(result_eval_ptr);
    }

    int64_t split = (int64_t)asReal(idata_split);

    // Determine callbacks
    ggml_opt_epoch_callback cb_train_func = NULL;
    ggml_opt_epoch_callback cb_eval_func = NULL;

    // Store R callbacks in global variables (protected)
    // Check if callback_train is a function or special value
    if (Rf_isFunction(callback_train)) {
        R_PreserveObject(callback_train);
        g_callback_train = callback_train;
        cb_train_func = r_callback_wrapper;
    } else if (Rf_isLogical(callback_train) && asLogical(callback_train)) {
        // TRUE means use built-in progress bar
        cb_train_func = ggml_opt_epoch_callback_progress_bar;
    } else {
        g_callback_train = NULL;
    }

    if (Rf_isFunction(callback_eval)) {
        R_PreserveObject(callback_eval);
        g_callback_eval = callback_eval;
        cb_eval_func = r_callback_wrapper;
    } else if (Rf_isLogical(callback_eval) && asLogical(callback_eval)) {
        // TRUE means use built-in progress bar
        cb_eval_func = ggml_opt_epoch_callback_progress_bar;
    } else {
        g_callback_eval = NULL;
    }

    // Run the epoch
    ggml_opt_epoch(opt_ctx, dataset, result_train, result_eval, split, cb_train_func, cb_eval_func);

    // Release R callbacks
    if (g_callback_train != NULL && Rf_isFunction(g_callback_train)) {
        R_ReleaseObject(g_callback_train);
        g_callback_train = NULL;
    }
    if (g_callback_eval != NULL && Rf_isFunction(g_callback_eval)) {
        R_ReleaseObject(g_callback_eval);
        g_callback_eval = NULL;
    }

    return R_NilValue;
}

// ============================================================================
// LR-controllable optimizer context for R-side epoch loop
// ============================================================================

// Userdata for R-controlled LR: holds current optimizer params
// (updated by R between epochs via R_ggml_opt_set_lr)
typedef struct {
    struct ggml_opt_optimizer_params params;
} r_opt_lr_userdata;

// C callback: simply returns the stored params (R updates them between epochs)
static struct ggml_opt_optimizer_params r_opt_get_constant_lr(void * userdata) {
    r_opt_lr_userdata * ud = (r_opt_lr_userdata *)userdata;
    return ud->params;
}

// Finalizer: free userdata when external pointer is GC'd
static void r_opt_lr_userdata_finalizer(SEXP ptr) {
    void * ud = R_ExternalPtrAddr(ptr);
    if (ud != NULL) {
        free(ud);
        R_ClearExternalPtr(ptr);
    }
}

// Initialize optimizer context for R-side epoch loop.
// Returns a list: list(opt_ctx=<ptr>, lr_ud=<ptr>) so R can call R_ggml_opt_set_lr.
// Default LR is taken from ggml_opt_get_default_optimizer_params.
SEXP R_ggml_opt_init_for_fit(SEXP sched_ptr, SEXP loss_type, SEXP optimizer_type,
                              SEXP opt_period, SEXP ctx_compute_ptr,
                              SEXP inputs_ptr, SEXP outputs_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    if (sched == NULL) error("Invalid scheduler pointer");

    enum ggml_opt_loss_type lt = (enum ggml_opt_loss_type)asInteger(loss_type);
    struct ggml_opt_params params = ggml_opt_default_params(sched, lt);
    params.optimizer = (enum ggml_opt_optimizer_type)asInteger(optimizer_type);
    params.opt_period = asInteger(opt_period);

    if (ctx_compute_ptr != R_NilValue)
        params.ctx_compute = (struct ggml_context *)R_ExternalPtrAddr(ctx_compute_ptr);
    if (inputs_ptr != R_NilValue)
        params.inputs = (struct ggml_tensor *)R_ExternalPtrAddr(inputs_ptr);
    if (outputs_ptr != R_NilValue)
        params.outputs = (struct ggml_tensor *)R_ExternalPtrAddr(outputs_ptr);

    // Allocate userdata with default LR from ggml defaults
    r_opt_lr_userdata * ud = (r_opt_lr_userdata *)malloc(sizeof(r_opt_lr_userdata));
    if (ud == NULL) error("Failed to allocate LR userdata");
    // Get default params (epoch=1 just to get defaults)
    int64_t dummy_epoch = 1;
    ud->params = ggml_opt_get_default_optimizer_params(&dummy_epoch);

    params.get_opt_pars    = r_opt_get_constant_lr;
    params.get_opt_pars_ud = ud;

    // Same as R_ggml_opt_init and the multi-head variant: apply the current
    // thread setting at init. The R-side epoch loop re-syncs per epoch on top
    // of this (see .ggml_sched_sync_threads).
    r_sched_sync_cpu_threads(sched);

    ggml_opt_context_t opt_ctx = ggml_opt_init(params);
    if (opt_ctx == NULL) { free(ud); error("Failed to initialize optimizer context"); }

    SEXP opt_ptr = PROTECT(R_MakeExternalPtr(opt_ctx, R_NilValue, R_NilValue));
    SEXP ud_ptr  = PROTECT(R_MakeExternalPtr(ud, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ud_ptr, r_opt_lr_userdata_finalizer, TRUE);

    SEXP result = PROTECT(allocVector(VECSXP, 2));
    SEXP names  = PROTECT(allocVector(STRSXP, 2));
    SET_VECTOR_ELT(result, 0, opt_ptr);
    SET_VECTOR_ELT(result, 1, ud_ptr);
    SET_STRING_ELT(names, 0, mkChar("opt_ctx"));
    SET_STRING_ELT(names, 1, mkChar("lr_ud"));
    setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(4);
    return result;
}

// Multi-output counterpart of R_ggml_opt_init_for_fit: initialize an optimizer
// context with several output heads for an R-side epoch loop.
//
// Same contract as the single-head version -- the returned lr_ud drives
// r_opt_get_constant_lr, so callbacks can change the learning rate between
// epochs -- plus the per-head parameters R_ggml_opt_fit_multi sets internally:
// loss types, head weights and the offset of each head inside a label row.
SEXP R_ggml_opt_init_for_fit_multi(SEXP sched_ptr, SEXP loss_types, SEXP loss_weights,
                                   SEXP labels_offs, SEXP optimizer_type, SEXP opt_period,
                                   SEXP ctx_compute_ptr, SEXP inputs_ptr, SEXP outputs_list) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t) r_ptr_required(sched_ptr, "scheduler");

    if (TYPEOF(outputs_list) != VECSXP) {
        error("outputs must be a list of tensor pointers");
    }
    const int n_head = LENGTH(outputs_list);
    if (n_head < 1) {
        error("outputs must contain at least one output head");
    }
    if (n_head > R_OPT_MAX_HEADS) {
        error("too many output heads: %d (max %d)", n_head, R_OPT_MAX_HEADS);
    }
    if (LENGTH(loss_types) != n_head) {
        error("loss must have one entry per output head (%d given, %d expected)",
              LENGTH(loss_types), n_head);
    }
    if (LENGTH(loss_weights) != n_head) {
        error("loss_weights must have one entry per output head (%d given, %d expected)",
              LENGTH(loss_weights), n_head);
    }
    if (LENGTH(labels_offs) != n_head) {
        error("labels_offs must have one entry per output head (%d given, %d expected)",
              LENGTH(labels_offs), n_head);
    }

    struct ggml_tensor * outs[R_OPT_MAX_HEADS];
    enum ggml_opt_loss_type lts[R_OPT_MAX_HEADS];
    float                   lws[R_OPT_MAX_HEADS];
    int64_t                 offs[R_OPT_MAX_HEADS];

    for (int i = 0; i < n_head; ++i) {
        outs[i] = (struct ggml_tensor *) r_ptr_required(VECTOR_ELT(outputs_list, i), "output tensor");
        lts[i]  = (enum ggml_opt_loss_type) INTEGER(loss_types)[i];
        lws[i]  = (float) REAL(loss_weights)[i];
        offs[i] = (int64_t) REAL(labels_offs)[i];
    }

    struct ggml_opt_params params = ggml_opt_default_params_multi(sched, n_head, lts, lws);
    params.optimizer  = (enum ggml_opt_optimizer_type) asInteger(optimizer_type);
    params.opt_period = asInteger(opt_period);

    if (ctx_compute_ptr != R_NilValue)
        params.ctx_compute = (struct ggml_context *) r_ptr_required(ctx_compute_ptr, "compute context");
    if (inputs_ptr != R_NilValue)
        params.inputs = (struct ggml_tensor *) r_ptr_required(inputs_ptr, "inputs tensor");
    params.outputs_multi = outs;

    // Same LR userdata as the single-head path, so ggml_opt_set_lr() works here.
    r_opt_lr_userdata * ud = (r_opt_lr_userdata *) malloc(sizeof(r_opt_lr_userdata));
    if (ud == NULL) error("Failed to allocate LR userdata");
    int64_t dummy_epoch = 1;
    ud->params = ggml_opt_get_default_optimizer_params(&dummy_epoch);

    params.get_opt_pars    = r_opt_get_constant_lr;
    params.get_opt_pars_ud = ud;

    ggml_opt_context_t opt_ctx = ggml_opt_init(params);
    if (opt_ctx == NULL) { free(ud); error("Failed to initialize optimizer context"); }

    // Where each head's labels sit in a dataset label row. ggml_opt_fit_multi()
    // does this itself; driving ggml_opt_epoch from R means doing it here.
    ggml_opt_set_labels_offs(opt_ctx, n_head, offs);

    // The CPU backend's thread count is set at init in the single-head path via
    // ggml_opt_fit; an R-side loop never goes through it, so sync it here.
    r_sched_sync_cpu_threads(sched);

    SEXP opt_ptr = PROTECT(R_MakeExternalPtr(opt_ctx, R_NilValue, R_NilValue));
    SEXP ud_ptr  = PROTECT(R_MakeExternalPtr(ud, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ud_ptr, r_opt_lr_userdata_finalizer, TRUE);

    SEXP result = PROTECT(allocVector(VECSXP, 2));
    SEXP names  = PROTECT(allocVector(STRSXP, 2));
    SET_VECTOR_ELT(result, 0, opt_ptr);
    SET_VECTOR_ELT(result, 1, ud_ptr);
    SET_STRING_ELT(names, 0, mkChar("opt_ctx"));
    SET_STRING_ELT(names, 1, mkChar("lr_ud"));
    setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(4);
    return result;
}

// Update learning rate in the userdata (called between epochs from R).
// adamw_lr: new AdamW LR (NA to keep current)
// sgd_lr:   new SGD LR (NA to keep current)
SEXP R_ggml_opt_set_lr(SEXP ud_ptr, SEXP adamw_lr, SEXP sgd_lr) {
    r_opt_lr_userdata * ud = (r_opt_lr_userdata *)R_ExternalPtrAddr(ud_ptr);
    if (ud == NULL) error("Invalid LR userdata pointer");

    if (!ISNA(asReal(adamw_lr)))
        ud->params.adamw.alpha = (float)asReal(adamw_lr);
    if (!ISNA(asReal(sgd_lr)))
        ud->params.sgd.alpha = (float)asReal(sgd_lr);

    return R_NilValue;
}

// Get current LR from userdata
SEXP R_ggml_opt_get_lr(SEXP ud_ptr) {
    r_opt_lr_userdata * ud = (r_opt_lr_userdata *)R_ExternalPtrAddr(ud_ptr);
    if (ud == NULL) error("Invalid LR userdata pointer");

    SEXP result = PROTECT(allocVector(REALSXP, 2));
    SEXP names  = PROTECT(allocVector(STRSXP, 2));
    REAL(result)[0] = (double)ud->params.adamw.alpha;
    REAL(result)[1] = (double)ud->params.sgd.alpha;
    SET_STRING_ELT(names, 0, mkChar("adamw"));
    SET_STRING_ELT(names, 1, mkChar("sgd"));
    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(2);
    return result;
}

// Re-apply the current ggmlR thread setting to the scheduler's CPU backends.
//
// Every ggml_backend_cpu_init() already sets the thread count when the backend
// is created, so this only matters when ggml_set_n_threads() is called later:
// the single C entry points (R_ggml_opt_fit, R_ggml_opt_fit_multi) sync once
// before their loop, but an R-side epoch loop has no such moment. Calling this
// per epoch means a thread count changed mid-training is picked up rather than
// silently ignored.
SEXP R_ggml_sched_sync_threads(SEXP sched_ptr) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t) r_ptr_required(sched_ptr, "scheduler");
    r_sched_sync_cpu_threads(sched);
    return R_NilValue;
}
