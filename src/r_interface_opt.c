// Optimization R interface
// Training and fine-tuning support through ggml-opt

#include <R.h>
#include <Rinternals.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

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
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

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
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);
    struct ggml_tensor * data_batch = (struct ggml_tensor *)R_ExternalPtrAddr(data_batch_ptr);
    struct ggml_tensor * labels_batch = NULL;

    if (labels_batch_ptr != R_NilValue) {
        labels_batch = (struct ggml_tensor *)R_ExternalPtrAddr(labels_batch_ptr);
    }

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
SEXP R_ggml_opt_init(SEXP sched_ptr, SEXP loss_type, SEXP optimizer_type, SEXP opt_period) {
    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);

    if (sched == NULL) {
        error("Invalid scheduler pointer");
    }

    enum ggml_opt_loss_type lt = (enum ggml_opt_loss_type)asInteger(loss_type);
    struct ggml_opt_params params = ggml_opt_default_params(sched, lt);

    params.optimizer = (enum ggml_opt_optimizer_type)asInteger(optimizer_type);
    params.opt_period = asInteger(opt_period);

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
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);

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
    ggml_opt_result_t result = (ggml_opt_result_t)R_ExternalPtrAddr(result_ptr);

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

// Fit model to dataset
SEXP R_ggml_opt_fit(SEXP sched_ptr, SEXP ctx_compute_ptr,
                    SEXP inputs_ptr, SEXP outputs_ptr,
                    SEXP dataset_ptr, SEXP loss_type, SEXP optimizer_type,
                    SEXP nepoch, SEXP nbatch_logical, SEXP val_split, SEXP silent) {

    ggml_backend_sched_t sched = (ggml_backend_sched_t)R_ExternalPtrAddr(sched_ptr);
    struct ggml_context * ctx_compute = (struct ggml_context *)R_ExternalPtrAddr(ctx_compute_ptr);
    struct ggml_tensor * inputs = (struct ggml_tensor *)R_ExternalPtrAddr(inputs_ptr);
    struct ggml_tensor * outputs = (struct ggml_tensor *)R_ExternalPtrAddr(outputs_ptr);
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

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

    ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset, lt, ot,
                 ggml_opt_get_default_optimizer_params, n_epoch, n_batch, v_split, is_silent);

    return R_NilValue;
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
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);
    struct ggml_context * ctx_compute = (struct ggml_context *)R_ExternalPtrAddr(ctx_compute_ptr);
    struct ggml_cgraph * gf = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);
    struct ggml_tensor * inputs = (struct ggml_tensor *)R_ExternalPtrAddr(inputs_ptr);
    struct ggml_tensor * outputs = (struct ggml_tensor *)R_ExternalPtrAddr(outputs_ptr);

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
    ggml_opt_context_t opt_ctx = (ggml_opt_context_t)R_ExternalPtrAddr(opt_ctx_ptr);
    ggml_opt_dataset_t dataset = (ggml_opt_dataset_t)R_ExternalPtrAddr(dataset_ptr);

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
