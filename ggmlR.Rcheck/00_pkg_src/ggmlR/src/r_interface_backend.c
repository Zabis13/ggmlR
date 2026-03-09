// Extended backend R interface
// Device management, registry, events, graph planning, buffer management

#include <R.h>
#include <Rinternals.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

// ============================================================================
// Device Type Constants
// ============================================================================

SEXP R_ggml_backend_device_type_cpu(void) {
    return ScalarInteger(GGML_BACKEND_DEVICE_TYPE_CPU);
}

SEXP R_ggml_backend_device_type_gpu(void) {
    return ScalarInteger(GGML_BACKEND_DEVICE_TYPE_GPU);
}

SEXP R_ggml_backend_device_type_igpu(void) {
    return ScalarInteger(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

SEXP R_ggml_backend_device_type_accel(void) {
    return ScalarInteger(GGML_BACKEND_DEVICE_TYPE_ACCEL);
}

// ============================================================================
// Buffer Usage Constants
// ============================================================================

SEXP R_ggml_backend_buffer_usage_any(void) {
    return ScalarInteger(GGML_BACKEND_BUFFER_USAGE_ANY);
}

SEXP R_ggml_backend_buffer_usage_weights(void) {
    return ScalarInteger(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
}

SEXP R_ggml_backend_buffer_usage_compute(void) {
    return ScalarInteger(GGML_BACKEND_BUFFER_USAGE_COMPUTE);
}

// ============================================================================
// Device Enumeration
// ============================================================================

SEXP R_ggml_backend_dev_count(void) {
    size_t count = ggml_backend_dev_count();
    return ScalarReal((double)count);
}

SEXP R_ggml_backend_dev_get(SEXP index) {
    size_t idx = (size_t)asReal(index);
    ggml_backend_dev_t device = ggml_backend_dev_get(idx);

    if (device == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(device, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_dev_by_name(SEXP name) {
    const char * dev_name = CHAR(STRING_ELT(name, 0));
    ggml_backend_dev_t device = ggml_backend_dev_by_name(dev_name);

    if (device == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(device, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_dev_by_type(SEXP type) {
    enum ggml_backend_dev_type dev_type = (enum ggml_backend_dev_type)asInteger(type);
    ggml_backend_dev_t device = ggml_backend_dev_by_type(dev_type);

    if (device == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(device, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// ============================================================================
// Device Properties
// ============================================================================

SEXP R_ggml_backend_dev_name(SEXP device_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    const char * name = ggml_backend_dev_name(device);
    return mkString(name ? name : "");
}

SEXP R_ggml_backend_dev_description(SEXP device_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    const char * desc = ggml_backend_dev_description(device);
    return mkString(desc ? desc : "");
}

SEXP R_ggml_backend_dev_memory(SEXP device_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    size_t free_mem = 0, total_mem = 0;
    ggml_backend_dev_memory(device, &free_mem, &total_mem);

    SEXP result = PROTECT(allocVector(REALSXP, 2));
    REAL(result)[0] = (double)free_mem;
    REAL(result)[1] = (double)total_mem;

    SEXP names = PROTECT(allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, mkChar("free"));
    SET_STRING_ELT(names, 1, mkChar("total"));
    setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(2);
    return result;
}

SEXP R_ggml_backend_dev_type(SEXP device_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(device);
    return ScalarInteger((int)dev_type);
}

SEXP R_ggml_backend_dev_get_props(SEXP device_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    struct ggml_backend_dev_props props;
    ggml_backend_dev_get_props(device, &props);

    // Return as a list
    SEXP result = PROTECT(allocVector(VECSXP, 7));
    SEXP names = PROTECT(allocVector(STRSXP, 7));

    SET_STRING_ELT(names, 0, mkChar("name"));
    SET_STRING_ELT(names, 1, mkChar("description"));
    SET_STRING_ELT(names, 2, mkChar("memory_free"));
    SET_STRING_ELT(names, 3, mkChar("memory_total"));
    SET_STRING_ELT(names, 4, mkChar("type"));
    SET_STRING_ELT(names, 5, mkChar("device_id"));
    SET_STRING_ELT(names, 6, mkChar("caps"));

    SET_VECTOR_ELT(result, 0, mkString(props.name ? props.name : ""));
    SET_VECTOR_ELT(result, 1, mkString(props.description ? props.description : ""));
    SET_VECTOR_ELT(result, 2, ScalarReal((double)props.memory_free));
    SET_VECTOR_ELT(result, 3, ScalarReal((double)props.memory_total));
    SET_VECTOR_ELT(result, 4, ScalarInteger((int)props.type));
    SET_VECTOR_ELT(result, 5, mkString(props.device_id ? props.device_id : ""));

    // Capabilities as a named logical vector
    SEXP caps = PROTECT(allocVector(LGLSXP, 4));
    SEXP caps_names = PROTECT(allocVector(STRSXP, 4));
    LOGICAL(caps)[0] = props.caps.async;
    LOGICAL(caps)[1] = props.caps.host_buffer;
    LOGICAL(caps)[2] = props.caps.buffer_from_host_ptr;
    LOGICAL(caps)[3] = props.caps.events;
    SET_STRING_ELT(caps_names, 0, mkChar("async"));
    SET_STRING_ELT(caps_names, 1, mkChar("host_buffer"));
    SET_STRING_ELT(caps_names, 2, mkChar("buffer_from_host_ptr"));
    SET_STRING_ELT(caps_names, 3, mkChar("events"));
    setAttrib(caps, R_NamesSymbol, caps_names);
    SET_VECTOR_ELT(result, 6, caps);

    setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
}

SEXP R_ggml_backend_dev_supports_op(SEXP device_ptr, SEXP op_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);
    struct ggml_tensor * op = (struct ggml_tensor *)R_ExternalPtrAddr(op_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }
    if (op == NULL) {
        error("Invalid tensor pointer");
    }

    bool supports = ggml_backend_dev_supports_op(device, op);
    return ScalarLogical(supports);
}

SEXP R_ggml_backend_dev_supports_buft(SEXP device_ptr, SEXP buft_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);
    ggml_backend_buffer_type_t buft = (ggml_backend_buffer_type_t)R_ExternalPtrAddr(buft_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }
    if (buft == NULL) {
        error("Invalid buffer type pointer");
    }

    bool supports = ggml_backend_dev_supports_buft(device, buft);
    return ScalarLogical(supports);
}

SEXP R_ggml_backend_dev_offload_op(SEXP device_ptr, SEXP op_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);
    struct ggml_tensor * op = (struct ggml_tensor *)R_ExternalPtrAddr(op_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }
    if (op == NULL) {
        error("Invalid tensor pointer");
    }

    bool offload = ggml_backend_dev_offload_op(device, op);
    return ScalarLogical(offload);
}

SEXP R_ggml_backend_dev_init(SEXP device_ptr, SEXP params) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    const char * params_str = NULL;
    if (params != R_NilValue && TYPEOF(params) == STRSXP) {
        params_str = CHAR(STRING_ELT(params, 0));
    }

    ggml_backend_t backend = ggml_backend_dev_init(device, params_str);

    if (backend == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(backend, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// ============================================================================
// Backend Registry
// ============================================================================

SEXP R_ggml_backend_reg_count(void) {
    size_t count = ggml_backend_reg_count();
    return ScalarReal((double)count);
}

SEXP R_ggml_backend_reg_get(SEXP index) {
    size_t idx = (size_t)asReal(index);
    ggml_backend_reg_t reg = ggml_backend_reg_get(idx);

    if (reg == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(reg, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_reg_by_name(SEXP name) {
    const char * reg_name = CHAR(STRING_ELT(name, 0));
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(reg_name);

    if (reg == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(reg, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_reg_name(SEXP reg_ptr) {
    ggml_backend_reg_t reg = (ggml_backend_reg_t)R_ExternalPtrAddr(reg_ptr);

    if (reg == NULL) {
        error("Invalid registry pointer");
    }

    const char * name = ggml_backend_reg_name(reg);
    return mkString(name ? name : "");
}

SEXP R_ggml_backend_reg_dev_count(SEXP reg_ptr) {
    ggml_backend_reg_t reg = (ggml_backend_reg_t)R_ExternalPtrAddr(reg_ptr);

    if (reg == NULL) {
        error("Invalid registry pointer");
    }

    size_t count = ggml_backend_reg_dev_count(reg);
    return ScalarReal((double)count);
}

SEXP R_ggml_backend_reg_dev_get(SEXP reg_ptr, SEXP index) {
    ggml_backend_reg_t reg = (ggml_backend_reg_t)R_ExternalPtrAddr(reg_ptr);

    if (reg == NULL) {
        error("Invalid registry pointer");
    }

    size_t idx = (size_t)asReal(index);
    ggml_backend_dev_t device = ggml_backend_reg_dev_get(reg, idx);

    if (device == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(device, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_load(SEXP path) {
    const char * lib_path = CHAR(STRING_ELT(path, 0));
    ggml_backend_reg_t reg = ggml_backend_load(lib_path);

    if (reg == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(reg, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_unload(SEXP reg_ptr) {
    ggml_backend_reg_t reg = (ggml_backend_reg_t)R_ExternalPtrAddr(reg_ptr);

    if (reg != NULL) {
        ggml_backend_unload(reg);
    }

    return R_NilValue;
}

SEXP R_ggml_backend_load_all(void) {
    ggml_backend_load_all();
    return R_NilValue;
}

// ============================================================================
// Events
// ============================================================================

SEXP R_ggml_backend_event_new(SEXP device_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    ggml_backend_event_t event = ggml_backend_event_new(device);

    if (event == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(event, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_event_free(SEXP event_ptr) {
    ggml_backend_event_t event = (ggml_backend_event_t)R_ExternalPtrAddr(event_ptr);

    if (event != NULL) {
        ggml_backend_event_free(event);
        R_ClearExternalPtr(event_ptr);
    }

    return R_NilValue;
}

SEXP R_ggml_backend_event_record(SEXP event_ptr, SEXP backend_ptr) {
    ggml_backend_event_t event = (ggml_backend_event_t)R_ExternalPtrAddr(event_ptr);
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

    if (event == NULL) {
        error("Invalid event pointer");
    }
    if (backend == NULL) {
        error("Invalid backend pointer");
    }

    ggml_backend_event_record(event, backend);
    return R_NilValue;
}

SEXP R_ggml_backend_event_synchronize(SEXP event_ptr) {
    ggml_backend_event_t event = (ggml_backend_event_t)R_ExternalPtrAddr(event_ptr);

    if (event == NULL) {
        error("Invalid event pointer");
    }

    ggml_backend_event_synchronize(event);
    return R_NilValue;
}

SEXP R_ggml_backend_event_wait(SEXP backend_ptr, SEXP event_ptr) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    ggml_backend_event_t event = (ggml_backend_event_t)R_ExternalPtrAddr(event_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }
    if (event == NULL) {
        error("Invalid event pointer");
    }

    ggml_backend_event_wait(backend, event);
    return R_NilValue;
}

// ============================================================================
// Graph Planning
// ============================================================================

SEXP R_ggml_backend_graph_plan_create(SEXP backend_ptr, SEXP graph_ptr) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    ggml_backend_graph_plan_t plan = ggml_backend_graph_plan_create(backend, graph);

    if (plan == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(plan, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_graph_plan_free(SEXP backend_ptr, SEXP plan_ptr) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    ggml_backend_graph_plan_t plan = (ggml_backend_graph_plan_t)R_ExternalPtrAddr(plan_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }
    if (plan != NULL) {
        ggml_backend_graph_plan_free(backend, plan);
        R_ClearExternalPtr(plan_ptr);
    }

    return R_NilValue;
}

SEXP R_ggml_backend_graph_plan_compute(SEXP backend_ptr, SEXP plan_ptr) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    ggml_backend_graph_plan_t plan = (ggml_backend_graph_plan_t)R_ExternalPtrAddr(plan_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }
    if (plan == NULL) {
        error("Invalid plan pointer");
    }

    enum ggml_status status = ggml_backend_graph_plan_compute(backend, plan);
    return ScalarInteger((int)status);
}

// ============================================================================
// Async Tensor Operations
// ============================================================================

SEXP R_ggml_backend_tensor_set_async(SEXP backend_ptr, SEXP tensor_ptr, SEXP data, SEXP offset, SEXP size) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    struct ggml_tensor * tensor = (struct ggml_tensor *)R_ExternalPtrAddr(tensor_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    size_t off = (size_t)asReal(offset);
    size_t sz = (size_t)asReal(size);

    // Convert R data to appropriate format
    void * data_ptr = NULL;
    if (TYPEOF(data) == REALSXP) {
        data_ptr = REAL(data);
    } else if (TYPEOF(data) == INTSXP) {
        data_ptr = INTEGER(data);
    } else {
        error("Data must be numeric or integer vector");
    }

    ggml_backend_tensor_set_async(backend, tensor, data_ptr, off, sz);
    return R_NilValue;
}

SEXP R_ggml_backend_tensor_get_async(SEXP backend_ptr, SEXP tensor_ptr, SEXP offset, SEXP size) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    struct ggml_tensor * tensor = (struct ggml_tensor *)R_ExternalPtrAddr(tensor_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }
    if (tensor == NULL) {
        error("Invalid tensor pointer");
    }

    size_t off = (size_t)asReal(offset);
    size_t sz = (size_t)asReal(size);

    // Allocate R vector to receive data
    size_t n_floats = sz / sizeof(float);
    SEXP result = PROTECT(allocVector(REALSXP, n_floats));

    ggml_backend_tensor_get_async(backend, tensor, REAL(result), off, sz);

    UNPROTECT(1);
    return result;
}

SEXP R_ggml_backend_tensor_copy_async(SEXP backend_src_ptr, SEXP backend_dst_ptr, SEXP src_ptr, SEXP dst_ptr) {
    ggml_backend_t backend_src = (ggml_backend_t)R_ExternalPtrAddr(backend_src_ptr);
    ggml_backend_t backend_dst = (ggml_backend_t)R_ExternalPtrAddr(backend_dst_ptr);
    struct ggml_tensor * src = (struct ggml_tensor *)R_ExternalPtrAddr(src_ptr);
    struct ggml_tensor * dst = (struct ggml_tensor *)R_ExternalPtrAddr(dst_ptr);

    if (backend_src == NULL || backend_dst == NULL) {
        error("Invalid backend pointer");
    }
    if (src == NULL || dst == NULL) {
        error("Invalid tensor pointer");
    }

    ggml_backend_tensor_copy_async(backend_src, backend_dst, src, dst);
    return R_NilValue;
}

// ============================================================================
// Buffer Management
// ============================================================================

SEXP R_ggml_backend_buffer_clear(SEXP buffer_ptr, SEXP value) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    uint8_t val = (uint8_t)asInteger(value);
    ggml_backend_buffer_clear(buffer, val);
    return R_NilValue;
}

SEXP R_ggml_backend_buffer_set_usage(SEXP buffer_ptr, SEXP usage) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    enum ggml_backend_buffer_usage buf_usage = (enum ggml_backend_buffer_usage)asInteger(usage);
    ggml_backend_buffer_set_usage(buffer, buf_usage);
    return R_NilValue;
}

SEXP R_ggml_backend_buffer_get_usage(SEXP buffer_ptr) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    enum ggml_backend_buffer_usage usage = ggml_backend_buffer_get_usage(buffer);
    return ScalarInteger((int)usage);
}

SEXP R_ggml_backend_buffer_reset(SEXP buffer_ptr) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    ggml_backend_buffer_reset(buffer);
    return R_NilValue;
}

SEXP R_ggml_backend_buffer_is_host(SEXP buffer_ptr) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    bool is_host = ggml_backend_buffer_is_host(buffer);
    return ScalarLogical(is_host);
}

// ============================================================================
// Direct Backend Initialization
// ============================================================================

SEXP R_ggml_backend_init_by_name(SEXP name, SEXP params) {
    const char * backend_name = CHAR(STRING_ELT(name, 0));
    const char * params_str = NULL;

    if (params != R_NilValue && TYPEOF(params) == STRSXP) {
        params_str = CHAR(STRING_ELT(params, 0));
    }

    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, params_str);

    if (backend == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(backend, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_init_by_type(SEXP type, SEXP params) {
    enum ggml_backend_dev_type dev_type = (enum ggml_backend_dev_type)asInteger(type);
    const char * params_str = NULL;

    if (params != R_NilValue && TYPEOF(params) == STRSXP) {
        params_str = CHAR(STRING_ELT(params, 0));
    }

    ggml_backend_t backend = ggml_backend_init_by_type(dev_type, params_str);

    if (backend == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(backend, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_init_best(void) {
    ggml_backend_t backend = ggml_backend_init_best();

    if (backend == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(backend, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_synchronize(SEXP backend_ptr) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }

    ggml_backend_synchronize(backend);
    return R_NilValue;
}

SEXP R_ggml_backend_get_device(SEXP backend_ptr) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);

    if (device == NULL) {
        return R_NilValue;
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(device, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

// ============================================================================
// Async Graph Compute
// ============================================================================

SEXP R_ggml_backend_graph_compute_async(SEXP backend_ptr, SEXP graph_ptr) {
    ggml_backend_t backend = (ggml_backend_t)R_ExternalPtrAddr(backend_ptr);
    struct ggml_cgraph * graph = (struct ggml_cgraph *)R_ExternalPtrAddr(graph_ptr);

    if (backend == NULL) {
        error("Invalid backend pointer");
    }
    if (graph == NULL) {
        error("Invalid graph pointer");
    }

    enum ggml_status status = ggml_backend_graph_compute_async(backend, graph);
    return ScalarInteger((int)status);
}

// ============================================================================
// Multi-buffer Operations
// ============================================================================

SEXP R_ggml_backend_multi_buffer_alloc_buffer(SEXP buffers_list) {
    if (TYPEOF(buffers_list) != VECSXP) {
        error("buffers must be a list of buffer pointers");
    }

    size_t n_buffers = (size_t)LENGTH(buffers_list);
    if (n_buffers == 0) {
        error("buffers list cannot be empty");
    }

    ggml_backend_buffer_t * buffers = (ggml_backend_buffer_t *)R_alloc(n_buffers, sizeof(ggml_backend_buffer_t));

    for (size_t i = 0; i < n_buffers; i++) {
        SEXP buf_ptr = VECTOR_ELT(buffers_list, i);
        buffers[i] = (ggml_backend_buffer_t)R_ExternalPtrAddr(buf_ptr);
        if (buffers[i] == NULL) {
            error("Invalid buffer pointer at index %zu", i);
        }
    }

    ggml_backend_buffer_t multi_buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);

    if (multi_buffer == NULL) {
        error("Failed to allocate multi-buffer");
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(multi_buffer, R_NilValue, R_NilValue));
    UNPROTECT(1);
    return ptr;
}

SEXP R_ggml_backend_buffer_is_multi_buffer(SEXP buffer_ptr) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    return ScalarLogical(ggml_backend_buffer_is_multi_buffer(buffer));
}

SEXP R_ggml_backend_multi_buffer_set_usage(SEXP buffer_ptr, SEXP usage) {
    ggml_backend_buffer_t buffer = (ggml_backend_buffer_t)R_ExternalPtrAddr(buffer_ptr);

    if (buffer == NULL) {
        error("Invalid buffer pointer");
    }

    enum ggml_backend_buffer_usage buf_usage = (enum ggml_backend_buffer_usage)asInteger(usage);
    ggml_backend_multi_buffer_set_usage(buffer, buf_usage);
    return R_NilValue;
}

// ============================================================================
// Backend Registration
// ============================================================================

SEXP R_ggml_backend_register(SEXP reg_ptr) {
    ggml_backend_reg_t reg = (ggml_backend_reg_t)R_ExternalPtrAddr(reg_ptr);

    if (reg == NULL) {
        error("Invalid registry pointer");
    }

    ggml_backend_register(reg);
    return R_NilValue;
}

SEXP R_ggml_backend_device_register(SEXP device_ptr) {
    ggml_backend_dev_t device = (ggml_backend_dev_t)R_ExternalPtrAddr(device_ptr);

    if (device == NULL) {
        error("Invalid device pointer");
    }

    ggml_backend_device_register(device);
    return R_NilValue;
}
