#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "ggml.h"
#include "ggml-cpu.h"

// Vulkan functions (defined in r_interface_vulkan.c)
extern SEXP R_ggml_vulkan_is_available(void);
extern SEXP R_ggml_vulkan_device_count(void);
extern SEXP R_ggml_vulkan_device_description(SEXP device_idx);
extern SEXP R_ggml_vulkan_device_memory(SEXP device_idx);
extern SEXP R_ggml_vulkan_init(SEXP device_idx);
extern SEXP R_ggml_vulkan_free(SEXP backend_ptr);
extern SEXP R_ggml_vulkan_is_backend(SEXP backend_ptr);
extern SEXP R_ggml_vulkan_backend_name(SEXP backend_ptr);
extern SEXP R_ggml_vulkan_list_devices(void);

// Backend scheduler functions (defined in r_interface_scheduler.c)
extern SEXP R_ggml_backend_sched_new(SEXP backends_list, SEXP parallel, SEXP graph_size);
extern SEXP R_ggml_backend_sched_free(SEXP sched_ptr);
extern SEXP R_ggml_backend_sched_reserve(SEXP sched_ptr, SEXP graph_ptr);
extern SEXP R_ggml_backend_sched_get_n_backends(SEXP sched_ptr);
extern SEXP R_ggml_backend_sched_get_backend(SEXP sched_ptr, SEXP index);
extern SEXP R_ggml_backend_sched_get_n_splits(SEXP sched_ptr);
extern SEXP R_ggml_backend_sched_get_n_copies(SEXP sched_ptr);
extern SEXP R_ggml_backend_sched_set_tensor_backend(SEXP sched_ptr, SEXP tensor_ptr, SEXP backend_ptr);
extern SEXP R_ggml_backend_sched_get_tensor_backend(SEXP sched_ptr, SEXP tensor_ptr);
extern SEXP R_ggml_backend_sched_alloc_graph(SEXP sched_ptr, SEXP graph_ptr);
extern SEXP R_ggml_backend_sched_graph_compute(SEXP sched_ptr, SEXP graph_ptr);
extern SEXP R_ggml_backend_sched_graph_compute_async(SEXP sched_ptr, SEXP graph_ptr);
extern SEXP R_ggml_backend_sched_synchronize(SEXP sched_ptr);
extern SEXP R_ggml_backend_sched_reset(SEXP sched_ptr);

// Optimization functions (defined in r_interface_opt.c)
extern SEXP R_ggml_opt_loss_type_mean(void);
extern SEXP R_ggml_opt_loss_type_sum(void);
extern SEXP R_ggml_opt_loss_type_cross_entropy(void);
extern SEXP R_ggml_opt_loss_type_mse(void);
extern SEXP R_ggml_opt_optimizer_type_adamw(void);
extern SEXP R_ggml_opt_optimizer_type_sgd(void);
extern SEXP R_ggml_opt_dataset_init(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_opt_dataset_free(SEXP);
extern SEXP R_ggml_opt_dataset_ndata(SEXP);
extern SEXP R_ggml_opt_dataset_data(SEXP);
extern SEXP R_ggml_opt_dataset_labels(SEXP);
extern SEXP R_ggml_opt_dataset_shuffle(SEXP, SEXP, SEXP);
extern SEXP R_ggml_opt_dataset_get_batch(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_opt_default_params(SEXP, SEXP);
extern SEXP R_ggml_opt_init(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_opt_free(SEXP);
extern SEXP R_ggml_opt_reset(SEXP, SEXP);
extern SEXP R_ggml_opt_static_graphs(SEXP);
extern SEXP R_ggml_opt_inputs(SEXP);
extern SEXP R_ggml_opt_outputs(SEXP);
extern SEXP R_ggml_opt_labels(SEXP);
extern SEXP R_ggml_opt_loss(SEXP);
extern SEXP R_ggml_opt_pred(SEXP);
extern SEXP R_ggml_opt_ncorrect(SEXP);
extern SEXP R_ggml_opt_context_optimizer_type(SEXP);
extern SEXP R_ggml_opt_optimizer_name(SEXP);
extern SEXP R_ggml_opt_result_init(void);
extern SEXP R_ggml_opt_result_free(SEXP);
extern SEXP R_ggml_opt_result_reset(SEXP);
extern SEXP R_ggml_opt_result_ndata(SEXP);
extern SEXP R_ggml_opt_result_loss(SEXP);
extern SEXP R_ggml_opt_result_accuracy(SEXP);
extern SEXP R_ggml_opt_alloc(SEXP, SEXP);
extern SEXP R_ggml_opt_eval(SEXP, SEXP);
extern SEXP R_ggml_opt_fit(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_opt_grad_acc(SEXP, SEXP);
extern SEXP R_ggml_opt_result_pred(SEXP);
extern SEXP R_ggml_opt_prepare_alloc(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_opt_epoch(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

// Extended backend functions (defined in r_interface_backend.c)
// Device type constants
extern SEXP R_ggml_backend_device_type_cpu(void);
extern SEXP R_ggml_backend_device_type_gpu(void);
extern SEXP R_ggml_backend_device_type_igpu(void);
extern SEXP R_ggml_backend_device_type_accel(void);
// Buffer usage constants
extern SEXP R_ggml_backend_buffer_usage_any(void);
extern SEXP R_ggml_backend_buffer_usage_weights(void);
extern SEXP R_ggml_backend_buffer_usage_compute(void);
// Device enumeration
extern SEXP R_ggml_backend_dev_count(void);
extern SEXP R_ggml_backend_dev_get(SEXP);
extern SEXP R_ggml_backend_dev_by_name(SEXP);
extern SEXP R_ggml_backend_dev_by_type(SEXP);
// Device properties
extern SEXP R_ggml_backend_dev_name(SEXP);
extern SEXP R_ggml_backend_dev_description(SEXP);
extern SEXP R_ggml_backend_dev_memory(SEXP);
extern SEXP R_ggml_backend_dev_type(SEXP);
extern SEXP R_ggml_backend_dev_get_props(SEXP);
extern SEXP R_ggml_backend_dev_supports_op(SEXP, SEXP);
extern SEXP R_ggml_backend_dev_supports_buft(SEXP, SEXP);
extern SEXP R_ggml_backend_dev_offload_op(SEXP, SEXP);
extern SEXP R_ggml_backend_dev_init(SEXP, SEXP);
// Backend registry
extern SEXP R_ggml_backend_reg_count(void);
extern SEXP R_ggml_backend_reg_get(SEXP);
extern SEXP R_ggml_backend_reg_by_name(SEXP);
extern SEXP R_ggml_backend_reg_name(SEXP);
extern SEXP R_ggml_backend_reg_dev_count(SEXP);
extern SEXP R_ggml_backend_reg_dev_get(SEXP, SEXP);
extern SEXP R_ggml_backend_load(SEXP);
extern SEXP R_ggml_backend_unload(SEXP);
extern SEXP R_ggml_backend_load_all(void);
// Events
extern SEXP R_ggml_backend_event_new(SEXP);
extern SEXP R_ggml_backend_event_free(SEXP);
extern SEXP R_ggml_backend_event_record(SEXP, SEXP);
extern SEXP R_ggml_backend_event_synchronize(SEXP);
extern SEXP R_ggml_backend_event_wait(SEXP, SEXP);
// Graph planning
extern SEXP R_ggml_backend_graph_plan_create(SEXP, SEXP);
extern SEXP R_ggml_backend_graph_plan_free(SEXP, SEXP);
extern SEXP R_ggml_backend_graph_plan_compute(SEXP, SEXP);
// Async operations
extern SEXP R_ggml_backend_tensor_set_async(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_backend_tensor_get_async(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_backend_tensor_copy_async(SEXP, SEXP, SEXP, SEXP);
// Buffer management
extern SEXP R_ggml_backend_buffer_clear(SEXP, SEXP);
extern SEXP R_ggml_backend_buffer_set_usage(SEXP, SEXP);
extern SEXP R_ggml_backend_buffer_get_usage(SEXP);
extern SEXP R_ggml_backend_buffer_reset(SEXP);
extern SEXP R_ggml_backend_buffer_is_host(SEXP);
// Direct backend init
extern SEXP R_ggml_backend_init_by_name(SEXP, SEXP);
extern SEXP R_ggml_backend_init_by_type(SEXP, SEXP);
extern SEXP R_ggml_backend_init_best(void);
extern SEXP R_ggml_backend_synchronize(SEXP);
extern SEXP R_ggml_backend_get_device(SEXP);

// CPU Feature Detection functions (defined in r_interface_graph.c)
// x86 SIMD
extern SEXP R_ggml_cpu_has_sse3(void);
extern SEXP R_ggml_cpu_has_ssse3(void);
extern SEXP R_ggml_cpu_has_avx(void);
extern SEXP R_ggml_cpu_has_avx_vnni(void);
extern SEXP R_ggml_cpu_has_avx2(void);
extern SEXP R_ggml_cpu_has_bmi2(void);
extern SEXP R_ggml_cpu_has_f16c(void);
extern SEXP R_ggml_cpu_has_fma(void);
extern SEXP R_ggml_cpu_has_avx512(void);
extern SEXP R_ggml_cpu_has_avx512_vbmi(void);
extern SEXP R_ggml_cpu_has_avx512_vnni(void);
extern SEXP R_ggml_cpu_has_avx512_bf16(void);
extern SEXP R_ggml_cpu_has_amx_int8(void);
// ARM SIMD
extern SEXP R_ggml_cpu_has_neon(void);
extern SEXP R_ggml_cpu_has_arm_fma(void);
extern SEXP R_ggml_cpu_has_fp16_va(void);
extern SEXP R_ggml_cpu_has_dotprod(void);
extern SEXP R_ggml_cpu_has_matmul_int8(void);
extern SEXP R_ggml_cpu_has_sve(void);
extern SEXP R_ggml_cpu_get_sve_cnt(void);
extern SEXP R_ggml_cpu_has_sme(void);
// Other architectures
extern SEXP R_ggml_cpu_has_riscv_v(void);
extern SEXP R_ggml_cpu_get_rvv_vlen(void);
extern SEXP R_ggml_cpu_has_vsx(void);
extern SEXP R_ggml_cpu_has_vxe(void);
extern SEXP R_ggml_cpu_has_wasm_simd(void);
extern SEXP R_ggml_cpu_has_llamafile(void);

// Tensor Layout/Contiguity functions (defined in r_interface_graph.c)
extern SEXP R_ggml_is_contiguous_0(SEXP);
extern SEXP R_ggml_is_contiguous_1(SEXP);
extern SEXP R_ggml_is_contiguous_2(SEXP);
extern SEXP R_ggml_is_contiguously_allocated(SEXP);
extern SEXP R_ggml_is_contiguous_channels(SEXP);
extern SEXP R_ggml_is_contiguous_rows(SEXP);
extern SEXP R_ggml_are_same_stride(SEXP, SEXP);
extern SEXP R_ggml_can_repeat(SEXP, SEXP);
extern SEXP R_ggml_count_equal(SEXP, SEXP, SEXP);

// Advanced RoPE functions (defined in r_interface_graph.c)
extern SEXP R_ggml_rope_multi_back(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

// Graph Construction & Introspection functions (defined in r_interface_graph.c)
extern SEXP R_ggml_build_backward_expand(SEXP, SEXP);
extern SEXP R_ggml_graph_add_node(SEXP, SEXP);
extern SEXP R_ggml_graph_clear(SEXP);
extern SEXP R_ggml_graph_cpy(SEXP, SEXP);
extern SEXP R_ggml_graph_dup(SEXP, SEXP, SEXP);
extern SEXP R_ggml_graph_get_grad(SEXP, SEXP);
extern SEXP R_ggml_graph_get_grad_acc(SEXP, SEXP);
extern SEXP R_ggml_graph_view(SEXP, SEXP, SEXP);
extern SEXP R_ggml_op_can_inplace(SEXP);
extern SEXP R_ggml_are_same_layout(SEXP, SEXP);

// Backend async/multi-buffer functions (defined in r_interface_backend.c)
extern SEXP R_ggml_backend_graph_compute_async(SEXP, SEXP);
extern SEXP R_ggml_backend_multi_buffer_alloc_buffer(SEXP);
extern SEXP R_ggml_backend_buffer_is_multi_buffer(SEXP);
extern SEXP R_ggml_backend_multi_buffer_set_usage(SEXP, SEXP);
extern SEXP R_ggml_backend_register(SEXP);
extern SEXP R_ggml_backend_device_register(SEXP);

// Advanced Attention/Loss functions (defined in r_interface_graph.c)
extern SEXP R_ggml_cross_entropy_loss(SEXP, SEXP, SEXP);
extern SEXP R_ggml_cross_entropy_loss_back(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_cumsum(SEXP, SEXP);
extern SEXP R_ggml_flash_attn_ext_set_prec(SEXP, SEXP);
extern SEXP R_ggml_flash_attn_ext_get_prec(SEXP);
extern SEXP R_ggml_flash_attn_ext_add_sinks(SEXP, SEXP);
extern SEXP R_ggml_soft_max_add_sinks(SEXP, SEXP);

// Logging & debugging functions (defined in r_interface_graph.c)
extern SEXP R_ggml_log_set_r(void);
extern SEXP R_ggml_log_set_default(void);
extern SEXP R_ggml_log_is_r_enabled(void);
extern SEXP R_ggml_set_abort_callback_r(void);
extern SEXP R_ggml_set_abort_callback_default(void);
extern SEXP R_ggml_abort_is_r_enabled(void);

// Op params functions (defined in r_interface_graph.c)
extern SEXP R_ggml_get_op_params(SEXP);
extern SEXP R_ggml_set_op_params(SEXP, SEXP);
extern SEXP R_ggml_get_op_params_i32(SEXP, SEXP);
extern SEXP R_ggml_set_op_params_i32(SEXP, SEXP, SEXP);
extern SEXP R_ggml_get_op_params_f32(SEXP, SEXP);
extern SEXP R_ggml_set_op_params_f32(SEXP, SEXP, SEXP);

// Timestep embedding (defined in r_interface_graph.c)
extern SEXP R_ggml_timestep_embedding(SEXP, SEXP, SEXP, SEXP);

// CPU-side tensor data access (defined in r_interface_graph.c)
extern SEXP R_ggml_set_f32_nd(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_get_f32_nd(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_get_i32_nd(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_set_i32_nd(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_tensor_nb(SEXP);
extern SEXP R_ggml_backend_tensor_get_and_sync(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_ggml_backend_tensor_get_f32(SEXP);
extern SEXP R_ggml_tensor_num(SEXP);
extern SEXP R_ggml_tensor_data_ptr(SEXP);
extern SEXP R_ggml_tensor_copy(SEXP, SEXP);
extern SEXP R_ggml_tensor_set_f32_scalar(SEXP, SEXP);
extern SEXP R_ggml_get_first_tensor(SEXP);
extern SEXP R_ggml_get_next_tensor(SEXP, SEXP);

// Low-level quantization functions (defined in r_interface_quants.c)
// Dequantize row functions
extern SEXP R_dequantize_row_q4_0(SEXP, SEXP);
extern SEXP R_dequantize_row_q4_1(SEXP, SEXP);
extern SEXP R_dequantize_row_q5_0(SEXP, SEXP);
extern SEXP R_dequantize_row_q5_1(SEXP, SEXP);
extern SEXP R_dequantize_row_q8_0(SEXP, SEXP);
extern SEXP R_dequantize_row_q2_K(SEXP, SEXP);
extern SEXP R_dequantize_row_q3_K(SEXP, SEXP);
extern SEXP R_dequantize_row_q4_K(SEXP, SEXP);
extern SEXP R_dequantize_row_q5_K(SEXP, SEXP);
extern SEXP R_dequantize_row_q6_K(SEXP, SEXP);
extern SEXP R_dequantize_row_q8_K(SEXP, SEXP);
extern SEXP R_dequantize_row_tq1_0(SEXP, SEXP);
extern SEXP R_dequantize_row_tq2_0(SEXP, SEXP);
extern SEXP R_dequantize_row_iq2_xxs(SEXP, SEXP);
extern SEXP R_dequantize_row_iq2_xs(SEXP, SEXP);
extern SEXP R_dequantize_row_iq2_s(SEXP, SEXP);
extern SEXP R_dequantize_row_iq3_xxs(SEXP, SEXP);
extern SEXP R_dequantize_row_iq3_s(SEXP, SEXP);
extern SEXP R_dequantize_row_iq4_nl(SEXP, SEXP);
extern SEXP R_dequantize_row_iq4_xs(SEXP, SEXP);
extern SEXP R_dequantize_row_iq1_s(SEXP, SEXP);
extern SEXP R_dequantize_row_iq1_m(SEXP, SEXP);
extern SEXP R_dequantize_row_mxfp4(SEXP, SEXP);
// Quantize functions (with imatrix)
extern SEXP R_quantize_q4_0(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q4_1(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q5_0(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q5_1(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q8_0(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q2_K(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q3_K(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q4_K(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q5_K(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_q6_K(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_tq1_0(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_tq2_0(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq2_xxs(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq2_xs(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq2_s(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq3_xxs(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq3_s(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq1_s(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq1_m(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq4_nl(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_iq4_xs(SEXP, SEXP, SEXP, SEXP);
extern SEXP R_quantize_mxfp4(SEXP, SEXP, SEXP, SEXP);
// Quantize row ref functions
extern SEXP R_quantize_row_q4_0_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q4_1_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q5_0_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q5_1_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q8_0_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q8_1_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q2_K_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q3_K_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q4_K_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q5_K_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q6_K_ref(SEXP, SEXP);
extern SEXP R_quantize_row_q8_K_ref(SEXP, SEXP);
extern SEXP R_quantize_row_tq1_0_ref(SEXP, SEXP);
extern SEXP R_quantize_row_tq2_0_ref(SEXP, SEXP);
extern SEXP R_quantize_row_iq3_xxs_ref(SEXP, SEXP);
extern SEXP R_quantize_row_iq4_nl_ref(SEXP, SEXP);
extern SEXP R_quantize_row_iq4_xs_ref(SEXP, SEXP);
extern SEXP R_quantize_row_iq3_s_ref(SEXP, SEXP);
extern SEXP R_quantize_row_iq2_s_ref(SEXP, SEXP);
extern SEXP R_quantize_row_mxfp4_ref(SEXP, SEXP);
// IQ init/free
extern SEXP R_iq2xs_init_impl(SEXP);
extern SEXP R_iq2xs_free_impl(SEXP);
extern SEXP R_iq3xs_init_impl(SEXP);
extern SEXP R_iq3xs_free_impl(SEXP);
// Quantization block info
extern SEXP R_ggml_quant_block_info(SEXP);

// ============================================================================
// Context Management
// ============================================================================

SEXP R_ggml_init(SEXP mem_size, SEXP no_alloc) {
    size_t size = (size_t) asReal(mem_size);
    int no_alloc_flag = asLogical(no_alloc);

    struct ggml_init_params params = {
        .mem_size = size,
        .mem_buffer = NULL,
        .no_alloc = no_alloc_flag,
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
SEXP R_ggml_group_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP n_groups, SEXP eps);
SEXP R_ggml_group_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP n_groups, SEXP eps);
SEXP R_ggml_l2_norm(SEXP ctx_ptr, SEXP a_ptr, SEXP eps);
SEXP R_ggml_l2_norm_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP eps);
SEXP R_ggml_rms_norm_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP eps);
SEXP R_ggml_soft_max(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_soft_max_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_soft_max_ext(SEXP ctx_ptr, SEXP a_ptr, SEXP mask_ptr, SEXP scale, SEXP max_bias);
SEXP R_ggml_soft_max_ext_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP mask_ptr, SEXP scale, SEXP max_bias);
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
// In-place operations (memory-efficient)
SEXP R_ggml_add_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_sub_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_mul_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_div_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_sqr_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_sqrt_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_exp_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_log_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_abs_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_neg_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_ceil_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_floor_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_round_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_relu_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_gelu_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_silu_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_sigmoid_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_tanh_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_softplus_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_elu_inplace(SEXP ctx_ptr, SEXP a_ptr);
SEXP R_ggml_scale_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP s);
SEXP R_ggml_dup_inplace(SEXP ctx_ptr, SEXP a_ptr);
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
SEXP R_ggml_rope_ext_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                              SEXP n_dims, SEXP mode, SEXP n_ctx_orig,
                              SEXP freq_base, SEXP freq_scale, SEXP ext_factor,
                              SEXP attn_factor, SEXP beta_fast, SEXP beta_slow);
SEXP R_ggml_rope_multi(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                        SEXP n_dims, SEXP sections, SEXP mode,
                        SEXP n_ctx_orig, SEXP freq_base, SEXP freq_scale,
                        SEXP ext_factor, SEXP attn_factor,
                        SEXP beta_fast, SEXP beta_slow);
SEXP R_ggml_rope_multi_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                                SEXP n_dims, SEXP sections, SEXP mode,
                                SEXP n_ctx_orig, SEXP freq_base, SEXP freq_scale,
                                SEXP ext_factor, SEXP attn_factor,
                                SEXP beta_fast, SEXP beta_slow);

// Flash Attention
SEXP R_ggml_flash_attn_ext(SEXP ctx_ptr, SEXP q_ptr, SEXP k_ptr, SEXP v_ptr,
                           SEXP mask_ptr, SEXP scale, SEXP max_bias, SEXP logit_softcap);
SEXP R_ggml_flash_attn_back(SEXP ctx_ptr, SEXP q_ptr, SEXP k_ptr, SEXP v_ptr,
                            SEXP d_ptr, SEXP masked);

// Mixture of Experts
SEXP R_ggml_mul_mat_id(SEXP ctx_ptr, SEXP as_ptr, SEXP b_ptr, SEXP ids_ptr);

// Scalar tensor creation
SEXP R_ggml_new_i32(SEXP ctx_ptr, SEXP value);
SEXP R_ggml_new_f32(SEXP ctx_ptr, SEXP value);

// View operations with offset
SEXP R_ggml_view_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP offset);
SEXP R_ggml_view_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1, SEXP nb1, SEXP offset);
SEXP R_ggml_view_3d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1, SEXP ne2,
                    SEXP nb1, SEXP nb2, SEXP offset);
SEXP R_ggml_view_4d(SEXP ctx_ptr, SEXP a_ptr, SEXP ne0, SEXP ne1, SEXP ne2, SEXP ne3,
                    SEXP nb1, SEXP nb2, SEXP nb3, SEXP offset);

// Copy and Set operations
SEXP R_ggml_cpy(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_set(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP nb1, SEXP nb2, SEXP nb3, SEXP offset);
SEXP R_ggml_set_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP offset);
SEXP R_ggml_set_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP nb1, SEXP offset);

// Matrix operations
SEXP R_ggml_out_prod(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_diag(SEXP ctx_ptr, SEXP a_ptr);

// Backward pass operations
SEXP R_ggml_silu_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_get_rows_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr);
SEXP R_ggml_soft_max_ext_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                               SEXP scale, SEXP max_bias);
SEXP R_ggml_soft_max_ext_back_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                                       SEXP scale, SEXP max_bias);
SEXP R_ggml_rope_ext_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP c_ptr,
                          SEXP n_dims, SEXP mode, SEXP n_ctx_orig,
                          SEXP freq_base, SEXP freq_scale, SEXP ext_factor,
                          SEXP attn_factor, SEXP beta_fast, SEXP beta_slow);

// Concatenation
SEXP R_ggml_concat(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr, SEXP dim);

// Sequence/Token operations
SEXP R_ggml_pad(SEXP ctx_ptr, SEXP a_ptr, SEXP p0, SEXP p1, SEXP p2, SEXP p3);
SEXP R_ggml_argsort(SEXP ctx_ptr, SEXP a_ptr, SEXP order);
SEXP R_ggml_top_k(SEXP ctx_ptr, SEXP a_ptr, SEXP k);
SEXP R_ggml_repeat_back(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_upscale(SEXP ctx_ptr, SEXP a_ptr, SEXP scale_factor, SEXP mode);

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

// Utility functions - additional
SEXP R_ggml_type_size(SEXP type);
SEXP R_ggml_element_size(SEXP tensor_ptr);
SEXP R_ggml_nrows(SEXP tensor_ptr);
SEXP R_ggml_are_same_shape(SEXP a_ptr, SEXP b_ptr);
SEXP R_ggml_set_name(SEXP tensor_ptr, SEXP name);
SEXP R_ggml_get_name(SEXP tensor_ptr);

// Type system functions
SEXP R_ggml_type_name(SEXP type);
SEXP R_ggml_type_sizef(SEXP type);
SEXP R_ggml_blck_size(SEXP type);
SEXP R_ggml_is_quantized(SEXP type);
SEXP R_ggml_ftype_to_ggml_type(SEXP ftype);

// Operation info functions
SEXP R_ggml_op_name(SEXP op);
SEXP R_ggml_op_symbol(SEXP op);
SEXP R_ggml_unary_op_name(SEXP op);
SEXP R_ggml_op_desc(SEXP tensor_ptr);
SEXP R_ggml_get_unary_op(SEXP tensor_ptr);

// Backend functions - direct access
SEXP R_ggml_backend_cpu_init(void);
SEXP R_ggml_backend_free(SEXP backend_ptr);
SEXP R_ggml_backend_cpu_set_n_threads(SEXP backend_ptr, SEXP n_threads);
SEXP R_ggml_backend_graph_compute(SEXP backend_ptr, SEXP graph_ptr);
SEXP R_ggml_backend_name(SEXP backend_ptr);

// CNN operations
SEXP R_ggml_conv_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                    SEXP s0, SEXP p0, SEXP d0);
SEXP R_ggml_conv_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                    SEXP s0, SEXP s1, SEXP p0, SEXP p1, SEXP d0, SEXP d1);
SEXP R_ggml_conv_transpose_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                               SEXP s0, SEXP p0, SEXP d0);
SEXP R_ggml_pool_1d(SEXP ctx_ptr, SEXP a_ptr, SEXP op,
                    SEXP k0, SEXP s0, SEXP p0);
SEXP R_ggml_pool_2d(SEXP ctx_ptr, SEXP a_ptr, SEXP op,
                    SEXP k0, SEXP k1, SEXP s0, SEXP s1, SEXP p0, SEXP p1);
SEXP R_ggml_im2col(SEXP ctx_ptr, SEXP a_ptr, SEXP b_ptr,
                   SEXP s0, SEXP s1, SEXP p0, SEXP p1,
                   SEXP d0, SEXP d1, SEXP is_2D, SEXP dst_type);

// Quantization functions
SEXP R_ggml_quantize_init(SEXP type);
SEXP R_ggml_quantize_free(void);
SEXP R_ggml_quantize_requires_imatrix(SEXP type);
SEXP R_ggml_quantize_chunk(SEXP type, SEXP src, SEXP nrows, SEXP n_per_row);

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
#undef match  // R defines 'match' macro that conflicts with OpenMP pragma
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
    {"R_ggml_init",    (DL_FUNC) &R_ggml_init,    2},
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
    {"R_ggml_group_norm",     (DL_FUNC) &R_ggml_group_norm,     4},
    {"R_ggml_group_norm_inplace", (DL_FUNC) &R_ggml_group_norm_inplace, 4},
    {"R_ggml_l2_norm",        (DL_FUNC) &R_ggml_l2_norm,        3},
    {"R_ggml_l2_norm_inplace",(DL_FUNC) &R_ggml_l2_norm_inplace,3},
    {"R_ggml_rms_norm_back",  (DL_FUNC) &R_ggml_rms_norm_back,  4},

    // Softmax
    {"R_ggml_soft_max",             (DL_FUNC) &R_ggml_soft_max,             2},
    {"R_ggml_soft_max_inplace",     (DL_FUNC) &R_ggml_soft_max_inplace,     2},
    {"R_ggml_soft_max_ext",         (DL_FUNC) &R_ggml_soft_max_ext,         5},
    {"R_ggml_soft_max_ext_inplace", (DL_FUNC) &R_ggml_soft_max_ext_inplace, 5},

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

    // In-place operations (memory-efficient, 2-3x savings)
    {"R_ggml_add_inplace",      (DL_FUNC) &R_ggml_add_inplace,      3},
    {"R_ggml_sub_inplace",      (DL_FUNC) &R_ggml_sub_inplace,      3},
    {"R_ggml_mul_inplace",      (DL_FUNC) &R_ggml_mul_inplace,      3},
    {"R_ggml_div_inplace",      (DL_FUNC) &R_ggml_div_inplace,      3},
    {"R_ggml_sqr_inplace",      (DL_FUNC) &R_ggml_sqr_inplace,      2},
    {"R_ggml_sqrt_inplace",     (DL_FUNC) &R_ggml_sqrt_inplace,     2},
    {"R_ggml_exp_inplace",      (DL_FUNC) &R_ggml_exp_inplace,      2},
    {"R_ggml_log_inplace",      (DL_FUNC) &R_ggml_log_inplace,      2},
    {"R_ggml_abs_inplace",      (DL_FUNC) &R_ggml_abs_inplace,      2},
    {"R_ggml_neg_inplace",      (DL_FUNC) &R_ggml_neg_inplace,      2},
    {"R_ggml_ceil_inplace",     (DL_FUNC) &R_ggml_ceil_inplace,     2},
    {"R_ggml_floor_inplace",    (DL_FUNC) &R_ggml_floor_inplace,    2},
    {"R_ggml_round_inplace",    (DL_FUNC) &R_ggml_round_inplace,    2},
    {"R_ggml_relu_inplace",     (DL_FUNC) &R_ggml_relu_inplace,     2},
    {"R_ggml_gelu_inplace",     (DL_FUNC) &R_ggml_gelu_inplace,     2},
    {"R_ggml_silu_inplace",     (DL_FUNC) &R_ggml_silu_inplace,     2},
    {"R_ggml_sigmoid_inplace",  (DL_FUNC) &R_ggml_sigmoid_inplace,  2},
    {"R_ggml_tanh_inplace",     (DL_FUNC) &R_ggml_tanh_inplace,     2},
    {"R_ggml_softplus_inplace", (DL_FUNC) &R_ggml_softplus_inplace, 2},
    {"R_ggml_elu_inplace",      (DL_FUNC) &R_ggml_elu_inplace,      2},
    {"R_ggml_scale_inplace",    (DL_FUNC) &R_ggml_scale_inplace,    3},
    {"R_ggml_dup_inplace",      (DL_FUNC) &R_ggml_dup_inplace,      2},

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
    {"R_ggml_rope",               (DL_FUNC) &R_ggml_rope,               5},
    {"R_ggml_rope_inplace",       (DL_FUNC) &R_ggml_rope_inplace,       5},
    {"R_ggml_rope_ext",           (DL_FUNC) &R_ggml_rope_ext,           13},
    {"R_ggml_rope_ext_inplace",   (DL_FUNC) &R_ggml_rope_ext_inplace,   13},
    {"R_ggml_rope_multi",         (DL_FUNC) &R_ggml_rope_multi,         14},
    {"R_ggml_rope_multi_inplace", (DL_FUNC) &R_ggml_rope_multi_inplace, 14},

    // Flash Attention
    {"R_ggml_flash_attn_ext",  (DL_FUNC) &R_ggml_flash_attn_ext,  8},
    {"R_ggml_flash_attn_back", (DL_FUNC) &R_ggml_flash_attn_back, 6},

    // Mixture of Experts
    {"R_ggml_mul_mat_id",     (DL_FUNC) &R_ggml_mul_mat_id,     4},

    // Scalar tensor creation
    {"R_ggml_new_i32",        (DL_FUNC) &R_ggml_new_i32,        2},
    {"R_ggml_new_f32",        (DL_FUNC) &R_ggml_new_f32,        2},

    // View operations with offset
    {"R_ggml_view_1d",        (DL_FUNC) &R_ggml_view_1d,        4},
    {"R_ggml_view_2d",        (DL_FUNC) &R_ggml_view_2d,        6},
    {"R_ggml_view_3d",        (DL_FUNC) &R_ggml_view_3d,        8},
    {"R_ggml_view_4d",        (DL_FUNC) &R_ggml_view_4d,        10},

    // Copy and Set operations
    {"R_ggml_cpy",            (DL_FUNC) &R_ggml_cpy,            3},
    {"R_ggml_set",            (DL_FUNC) &R_ggml_set,            7},
    {"R_ggml_set_1d",         (DL_FUNC) &R_ggml_set_1d,         4},
    {"R_ggml_set_2d",         (DL_FUNC) &R_ggml_set_2d,         5},

    // Matrix operations
    {"R_ggml_out_prod",       (DL_FUNC) &R_ggml_out_prod,       3},
    {"R_ggml_diag",           (DL_FUNC) &R_ggml_diag,           2},

    // Backward pass operations
    {"R_ggml_silu_back",                (DL_FUNC) &R_ggml_silu_back,                3},
    {"R_ggml_get_rows_back",            (DL_FUNC) &R_ggml_get_rows_back,            4},
    {"R_ggml_soft_max_ext_back",        (DL_FUNC) &R_ggml_soft_max_ext_back,        5},
    {"R_ggml_soft_max_ext_back_inplace",(DL_FUNC) &R_ggml_soft_max_ext_back_inplace,5},
    {"R_ggml_rope_ext_back",            (DL_FUNC) &R_ggml_rope_ext_back,            13},

    // Concatenation
    {"R_ggml_concat",         (DL_FUNC) &R_ggml_concat,         4},

    // Sequence/Token operations
    {"R_ggml_pad",            (DL_FUNC) &R_ggml_pad,            6},
    {"R_ggml_argsort",        (DL_FUNC) &R_ggml_argsort,        3},
    {"R_ggml_top_k",          (DL_FUNC) &R_ggml_top_k,          3},
    {"R_ggml_repeat_back",    (DL_FUNC) &R_ggml_repeat_back,    3},
    {"R_ggml_upscale",        (DL_FUNC) &R_ggml_upscale,        4},

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

    // Utility functions - additional
    {"R_ggml_type_size",          (DL_FUNC) &R_ggml_type_size,          1},
    {"R_ggml_element_size",       (DL_FUNC) &R_ggml_element_size,       1},
    {"R_ggml_nrows",              (DL_FUNC) &R_ggml_nrows,              1},
    {"R_ggml_are_same_shape",     (DL_FUNC) &R_ggml_are_same_shape,     2},
    {"R_ggml_set_name",           (DL_FUNC) &R_ggml_set_name,           2},
    {"R_ggml_get_name",           (DL_FUNC) &R_ggml_get_name,           1},

    // Type system functions
    {"R_ggml_type_name",          (DL_FUNC) &R_ggml_type_name,          1},
    {"R_ggml_type_sizef",         (DL_FUNC) &R_ggml_type_sizef,         1},
    {"R_ggml_blck_size",          (DL_FUNC) &R_ggml_blck_size,          1},
    {"R_ggml_is_quantized",       (DL_FUNC) &R_ggml_is_quantized,       1},
    {"R_ggml_ftype_to_ggml_type", (DL_FUNC) &R_ggml_ftype_to_ggml_type, 1},

    // Operation info functions
    {"R_ggml_op_name",            (DL_FUNC) &R_ggml_op_name,            1},
    {"R_ggml_op_symbol",          (DL_FUNC) &R_ggml_op_symbol,          1},
    {"R_ggml_unary_op_name",      (DL_FUNC) &R_ggml_unary_op_name,      1},
    {"R_ggml_op_desc",            (DL_FUNC) &R_ggml_op_desc,            1},
    {"R_ggml_get_unary_op",       (DL_FUNC) &R_ggml_get_unary_op,       1},

    // Backend functions - direct access
    {"R_ggml_backend_cpu_init",          (DL_FUNC) &R_ggml_backend_cpu_init,          0},
    {"R_ggml_backend_free",              (DL_FUNC) &R_ggml_backend_free,              1},
    {"R_ggml_backend_cpu_set_n_threads", (DL_FUNC) &R_ggml_backend_cpu_set_n_threads, 2},
    {"R_ggml_backend_graph_compute",     (DL_FUNC) &R_ggml_backend_graph_compute,     2},
    {"R_ggml_backend_name",              (DL_FUNC) &R_ggml_backend_name,              1},

    // CNN operations
    {"R_ggml_conv_1d",           (DL_FUNC) &R_ggml_conv_1d,           6},
    {"R_ggml_conv_2d",           (DL_FUNC) &R_ggml_conv_2d,           9},
    {"R_ggml_conv_transpose_1d", (DL_FUNC) &R_ggml_conv_transpose_1d, 6},
    {"R_ggml_pool_1d",           (DL_FUNC) &R_ggml_pool_1d,           6},
    {"R_ggml_pool_2d",           (DL_FUNC) &R_ggml_pool_2d,           9},
    {"R_ggml_im2col",            (DL_FUNC) &R_ggml_im2col,            11},

    // Quantization functions
    {"R_ggml_quantize_init",             (DL_FUNC) &R_ggml_quantize_init,             1},
    {"R_ggml_quantize_free",             (DL_FUNC) &R_ggml_quantize_free,             0},
    {"R_ggml_quantize_requires_imatrix", (DL_FUNC) &R_ggml_quantize_requires_imatrix, 1},
    {"R_ggml_quantize_chunk",            (DL_FUNC) &R_ggml_quantize_chunk,            4},

    // Vulkan backend functions
    {"R_ggml_vulkan_is_available",      (DL_FUNC) &R_ggml_vulkan_is_available,      0},
    {"R_ggml_vulkan_device_count",      (DL_FUNC) &R_ggml_vulkan_device_count,      0},
    {"R_ggml_vulkan_device_description",(DL_FUNC) &R_ggml_vulkan_device_description,1},
    {"R_ggml_vulkan_device_memory",     (DL_FUNC) &R_ggml_vulkan_device_memory,     1},
    {"R_ggml_vulkan_init",              (DL_FUNC) &R_ggml_vulkan_init,              1},
    {"R_ggml_vulkan_free",              (DL_FUNC) &R_ggml_vulkan_free,              1},
    {"R_ggml_vulkan_is_backend",        (DL_FUNC) &R_ggml_vulkan_is_backend,        1},
    {"R_ggml_vulkan_backend_name",      (DL_FUNC) &R_ggml_vulkan_backend_name,      1},
    {"R_ggml_vulkan_list_devices",      (DL_FUNC) &R_ggml_vulkan_list_devices,      0},

    // Backend scheduler functions
    {"R_ggml_backend_sched_new",                (DL_FUNC) &R_ggml_backend_sched_new,                3},
    {"R_ggml_backend_sched_free",               (DL_FUNC) &R_ggml_backend_sched_free,               1},
    {"R_ggml_backend_sched_reserve",            (DL_FUNC) &R_ggml_backend_sched_reserve,            2},
    {"R_ggml_backend_sched_get_n_backends",     (DL_FUNC) &R_ggml_backend_sched_get_n_backends,     1},
    {"R_ggml_backend_sched_get_backend",        (DL_FUNC) &R_ggml_backend_sched_get_backend,        2},
    {"R_ggml_backend_sched_get_n_splits",       (DL_FUNC) &R_ggml_backend_sched_get_n_splits,       1},
    {"R_ggml_backend_sched_get_n_copies",       (DL_FUNC) &R_ggml_backend_sched_get_n_copies,       1},
    {"R_ggml_backend_sched_set_tensor_backend", (DL_FUNC) &R_ggml_backend_sched_set_tensor_backend, 3},
    {"R_ggml_backend_sched_get_tensor_backend", (DL_FUNC) &R_ggml_backend_sched_get_tensor_backend, 2},
    {"R_ggml_backend_sched_alloc_graph",        (DL_FUNC) &R_ggml_backend_sched_alloc_graph,        2},
    {"R_ggml_backend_sched_graph_compute",      (DL_FUNC) &R_ggml_backend_sched_graph_compute,      2},
    {"R_ggml_backend_sched_graph_compute_async",(DL_FUNC) &R_ggml_backend_sched_graph_compute_async,2},
    {"R_ggml_backend_sched_synchronize",        (DL_FUNC) &R_ggml_backend_sched_synchronize,        1},
    {"R_ggml_backend_sched_reset",              (DL_FUNC) &R_ggml_backend_sched_reset,              1},

    // Optimization functions (r_interface_opt.c)
    {"R_ggml_opt_loss_type_mean",               (DL_FUNC) &R_ggml_opt_loss_type_mean,               0},
    {"R_ggml_opt_loss_type_sum",                (DL_FUNC) &R_ggml_opt_loss_type_sum,                0},
    {"R_ggml_opt_loss_type_cross_entropy",      (DL_FUNC) &R_ggml_opt_loss_type_cross_entropy,      0},
    {"R_ggml_opt_loss_type_mse",                (DL_FUNC) &R_ggml_opt_loss_type_mse,                0},
    {"R_ggml_opt_optimizer_type_adamw",         (DL_FUNC) &R_ggml_opt_optimizer_type_adamw,         0},
    {"R_ggml_opt_optimizer_type_sgd",           (DL_FUNC) &R_ggml_opt_optimizer_type_sgd,           0},
    {"R_ggml_opt_dataset_init",                 (DL_FUNC) &R_ggml_opt_dataset_init,                 6},
    {"R_ggml_opt_dataset_free",                 (DL_FUNC) &R_ggml_opt_dataset_free,                 1},
    {"R_ggml_opt_dataset_ndata",                (DL_FUNC) &R_ggml_opt_dataset_ndata,                1},
    {"R_ggml_opt_dataset_data",                 (DL_FUNC) &R_ggml_opt_dataset_data,                 1},
    {"R_ggml_opt_dataset_labels",               (DL_FUNC) &R_ggml_opt_dataset_labels,               1},
    {"R_ggml_opt_dataset_shuffle",              (DL_FUNC) &R_ggml_opt_dataset_shuffle,              3},
    {"R_ggml_opt_dataset_get_batch",            (DL_FUNC) &R_ggml_opt_dataset_get_batch,            4},
    {"R_ggml_opt_default_params",               (DL_FUNC) &R_ggml_opt_default_params,               2},
    {"R_ggml_opt_init",                         (DL_FUNC) &R_ggml_opt_init,                         4},
    {"R_ggml_opt_free",                         (DL_FUNC) &R_ggml_opt_free,                         1},
    {"R_ggml_opt_reset",                        (DL_FUNC) &R_ggml_opt_reset,                        2},
    {"R_ggml_opt_static_graphs",                (DL_FUNC) &R_ggml_opt_static_graphs,                1},
    {"R_ggml_opt_inputs",                       (DL_FUNC) &R_ggml_opt_inputs,                       1},
    {"R_ggml_opt_outputs",                      (DL_FUNC) &R_ggml_opt_outputs,                      1},
    {"R_ggml_opt_labels",                       (DL_FUNC) &R_ggml_opt_labels,                       1},
    {"R_ggml_opt_loss",                         (DL_FUNC) &R_ggml_opt_loss,                         1},
    {"R_ggml_opt_pred",                         (DL_FUNC) &R_ggml_opt_pred,                         1},
    {"R_ggml_opt_ncorrect",                     (DL_FUNC) &R_ggml_opt_ncorrect,                     1},
    {"R_ggml_opt_context_optimizer_type",       (DL_FUNC) &R_ggml_opt_context_optimizer_type,       1},
    {"R_ggml_opt_optimizer_name",               (DL_FUNC) &R_ggml_opt_optimizer_name,               1},
    {"R_ggml_opt_result_init",                  (DL_FUNC) &R_ggml_opt_result_init,                  0},
    {"R_ggml_opt_result_free",                  (DL_FUNC) &R_ggml_opt_result_free,                  1},
    {"R_ggml_opt_result_reset",                 (DL_FUNC) &R_ggml_opt_result_reset,                 1},
    {"R_ggml_opt_result_ndata",                 (DL_FUNC) &R_ggml_opt_result_ndata,                 1},
    {"R_ggml_opt_result_loss",                  (DL_FUNC) &R_ggml_opt_result_loss,                  1},
    {"R_ggml_opt_result_accuracy",              (DL_FUNC) &R_ggml_opt_result_accuracy,              1},
    {"R_ggml_opt_alloc",                        (DL_FUNC) &R_ggml_opt_alloc,                        2},
    {"R_ggml_opt_eval",                         (DL_FUNC) &R_ggml_opt_eval,                         2},
    {"R_ggml_opt_fit",                          (DL_FUNC) &R_ggml_opt_fit,                          11},
    {"R_ggml_opt_grad_acc",                      (DL_FUNC) &R_ggml_opt_grad_acc,                      2},
    {"R_ggml_opt_result_pred",                   (DL_FUNC) &R_ggml_opt_result_pred,                   1},
    {"R_ggml_opt_prepare_alloc",                 (DL_FUNC) &R_ggml_opt_prepare_alloc,                 5},
    {"R_ggml_opt_epoch",                         (DL_FUNC) &R_ggml_opt_epoch,                         7},

    // Extended backend functions
    {"R_ggml_backend_device_type_cpu",          (DL_FUNC) &R_ggml_backend_device_type_cpu,           0},
    {"R_ggml_backend_device_type_gpu",          (DL_FUNC) &R_ggml_backend_device_type_gpu,           0},
    {"R_ggml_backend_device_type_igpu",         (DL_FUNC) &R_ggml_backend_device_type_igpu,          0},
    {"R_ggml_backend_device_type_accel",        (DL_FUNC) &R_ggml_backend_device_type_accel,         0},
    {"R_ggml_backend_buffer_usage_any",         (DL_FUNC) &R_ggml_backend_buffer_usage_any,          0},
    {"R_ggml_backend_buffer_usage_weights",     (DL_FUNC) &R_ggml_backend_buffer_usage_weights,      0},
    {"R_ggml_backend_buffer_usage_compute",     (DL_FUNC) &R_ggml_backend_buffer_usage_compute,      0},
    {"R_ggml_backend_dev_count",                (DL_FUNC) &R_ggml_backend_dev_count,                 0},
    {"R_ggml_backend_dev_get",                  (DL_FUNC) &R_ggml_backend_dev_get,                   1},
    {"R_ggml_backend_dev_by_name",              (DL_FUNC) &R_ggml_backend_dev_by_name,               1},
    {"R_ggml_backend_dev_by_type",              (DL_FUNC) &R_ggml_backend_dev_by_type,               1},
    {"R_ggml_backend_dev_name",                 (DL_FUNC) &R_ggml_backend_dev_name,                  1},
    {"R_ggml_backend_dev_description",          (DL_FUNC) &R_ggml_backend_dev_description,           1},
    {"R_ggml_backend_dev_memory",               (DL_FUNC) &R_ggml_backend_dev_memory,                1},
    {"R_ggml_backend_dev_type",                 (DL_FUNC) &R_ggml_backend_dev_type,                  1},
    {"R_ggml_backend_dev_get_props",            (DL_FUNC) &R_ggml_backend_dev_get_props,             1},
    {"R_ggml_backend_dev_supports_op",          (DL_FUNC) &R_ggml_backend_dev_supports_op,           2},
    {"R_ggml_backend_dev_supports_buft",        (DL_FUNC) &R_ggml_backend_dev_supports_buft,         2},
    {"R_ggml_backend_dev_offload_op",           (DL_FUNC) &R_ggml_backend_dev_offload_op,            2},
    {"R_ggml_backend_dev_init",                 (DL_FUNC) &R_ggml_backend_dev_init,                  2},
    {"R_ggml_backend_reg_count",                (DL_FUNC) &R_ggml_backend_reg_count,                 0},
    {"R_ggml_backend_reg_get",                  (DL_FUNC) &R_ggml_backend_reg_get,                   1},
    {"R_ggml_backend_reg_by_name",              (DL_FUNC) &R_ggml_backend_reg_by_name,               1},
    {"R_ggml_backend_reg_name",                 (DL_FUNC) &R_ggml_backend_reg_name,                  1},
    {"R_ggml_backend_reg_dev_count",            (DL_FUNC) &R_ggml_backend_reg_dev_count,             1},
    {"R_ggml_backend_reg_dev_get",              (DL_FUNC) &R_ggml_backend_reg_dev_get,               2},
    {"R_ggml_backend_load",                     (DL_FUNC) &R_ggml_backend_load,                      1},
    {"R_ggml_backend_unload",                   (DL_FUNC) &R_ggml_backend_unload,                    1},
    {"R_ggml_backend_load_all",                 (DL_FUNC) &R_ggml_backend_load_all,                  0},
    {"R_ggml_backend_event_new",                (DL_FUNC) &R_ggml_backend_event_new,                 1},
    {"R_ggml_backend_event_free",               (DL_FUNC) &R_ggml_backend_event_free,                1},
    {"R_ggml_backend_event_record",             (DL_FUNC) &R_ggml_backend_event_record,              2},
    {"R_ggml_backend_event_synchronize",        (DL_FUNC) &R_ggml_backend_event_synchronize,         1},
    {"R_ggml_backend_event_wait",               (DL_FUNC) &R_ggml_backend_event_wait,                2},
    {"R_ggml_backend_graph_plan_create",        (DL_FUNC) &R_ggml_backend_graph_plan_create,         2},
    {"R_ggml_backend_graph_plan_free",          (DL_FUNC) &R_ggml_backend_graph_plan_free,           2},
    {"R_ggml_backend_graph_plan_compute",       (DL_FUNC) &R_ggml_backend_graph_plan_compute,        2},
    {"R_ggml_backend_tensor_set_async",         (DL_FUNC) &R_ggml_backend_tensor_set_async,          5},
    {"R_ggml_backend_tensor_get_async",         (DL_FUNC) &R_ggml_backend_tensor_get_async,          4},
    {"R_ggml_backend_tensor_copy_async",        (DL_FUNC) &R_ggml_backend_tensor_copy_async,         4},
    {"R_ggml_backend_buffer_clear",             (DL_FUNC) &R_ggml_backend_buffer_clear,              2},
    {"R_ggml_backend_buffer_set_usage",         (DL_FUNC) &R_ggml_backend_buffer_set_usage,          2},
    {"R_ggml_backend_buffer_get_usage",         (DL_FUNC) &R_ggml_backend_buffer_get_usage,          1},
    {"R_ggml_backend_buffer_reset",             (DL_FUNC) &R_ggml_backend_buffer_reset,              1},
    {"R_ggml_backend_buffer_is_host",           (DL_FUNC) &R_ggml_backend_buffer_is_host,            1},
    {"R_ggml_backend_init_by_name",             (DL_FUNC) &R_ggml_backend_init_by_name,              2},
    {"R_ggml_backend_init_by_type",             (DL_FUNC) &R_ggml_backend_init_by_type,              2},
    {"R_ggml_backend_init_best",                (DL_FUNC) &R_ggml_backend_init_best,                 0},
    {"R_ggml_backend_synchronize",              (DL_FUNC) &R_ggml_backend_synchronize,               1},
    {"R_ggml_backend_get_device",               (DL_FUNC) &R_ggml_backend_get_device,                1},

    // CPU Feature Detection (x86)
    {"R_ggml_cpu_has_sse3",                     (DL_FUNC) &R_ggml_cpu_has_sse3,                      0},
    {"R_ggml_cpu_has_ssse3",                    (DL_FUNC) &R_ggml_cpu_has_ssse3,                     0},
    {"R_ggml_cpu_has_avx",                      (DL_FUNC) &R_ggml_cpu_has_avx,                       0},
    {"R_ggml_cpu_has_avx_vnni",                 (DL_FUNC) &R_ggml_cpu_has_avx_vnni,                  0},
    {"R_ggml_cpu_has_avx2",                     (DL_FUNC) &R_ggml_cpu_has_avx2,                      0},
    {"R_ggml_cpu_has_bmi2",                     (DL_FUNC) &R_ggml_cpu_has_bmi2,                      0},
    {"R_ggml_cpu_has_f16c",                     (DL_FUNC) &R_ggml_cpu_has_f16c,                      0},
    {"R_ggml_cpu_has_fma",                      (DL_FUNC) &R_ggml_cpu_has_fma,                       0},
    {"R_ggml_cpu_has_avx512",                   (DL_FUNC) &R_ggml_cpu_has_avx512,                    0},
    {"R_ggml_cpu_has_avx512_vbmi",              (DL_FUNC) &R_ggml_cpu_has_avx512_vbmi,               0},
    {"R_ggml_cpu_has_avx512_vnni",              (DL_FUNC) &R_ggml_cpu_has_avx512_vnni,               0},
    {"R_ggml_cpu_has_avx512_bf16",              (DL_FUNC) &R_ggml_cpu_has_avx512_bf16,               0},
    {"R_ggml_cpu_has_amx_int8",                 (DL_FUNC) &R_ggml_cpu_has_amx_int8,                  0},
    // CPU Feature Detection (ARM)
    {"R_ggml_cpu_has_neon",                     (DL_FUNC) &R_ggml_cpu_has_neon,                      0},
    {"R_ggml_cpu_has_arm_fma",                  (DL_FUNC) &R_ggml_cpu_has_arm_fma,                   0},
    {"R_ggml_cpu_has_fp16_va",                  (DL_FUNC) &R_ggml_cpu_has_fp16_va,                   0},
    {"R_ggml_cpu_has_dotprod",                  (DL_FUNC) &R_ggml_cpu_has_dotprod,                   0},
    {"R_ggml_cpu_has_matmul_int8",              (DL_FUNC) &R_ggml_cpu_has_matmul_int8,               0},
    {"R_ggml_cpu_has_sve",                      (DL_FUNC) &R_ggml_cpu_has_sve,                       0},
    {"R_ggml_cpu_get_sve_cnt",                  (DL_FUNC) &R_ggml_cpu_get_sve_cnt,                   0},
    {"R_ggml_cpu_has_sme",                      (DL_FUNC) &R_ggml_cpu_has_sme,                       0},
    // CPU Feature Detection (Other)
    {"R_ggml_cpu_has_riscv_v",                  (DL_FUNC) &R_ggml_cpu_has_riscv_v,                   0},
    {"R_ggml_cpu_get_rvv_vlen",                 (DL_FUNC) &R_ggml_cpu_get_rvv_vlen,                  0},
    {"R_ggml_cpu_has_vsx",                      (DL_FUNC) &R_ggml_cpu_has_vsx,                       0},
    {"R_ggml_cpu_has_vxe",                      (DL_FUNC) &R_ggml_cpu_has_vxe,                       0},
    {"R_ggml_cpu_has_wasm_simd",                (DL_FUNC) &R_ggml_cpu_has_wasm_simd,                 0},
    {"R_ggml_cpu_has_llamafile",                (DL_FUNC) &R_ggml_cpu_has_llamafile,                 0},
    // Tensor Layout/Contiguity
    {"R_ggml_is_contiguous_0",                  (DL_FUNC) &R_ggml_is_contiguous_0,                   1},
    {"R_ggml_is_contiguous_1",                  (DL_FUNC) &R_ggml_is_contiguous_1,                   1},
    {"R_ggml_is_contiguous_2",                  (DL_FUNC) &R_ggml_is_contiguous_2,                   1},
    {"R_ggml_is_contiguously_allocated",        (DL_FUNC) &R_ggml_is_contiguously_allocated,         1},
    {"R_ggml_is_contiguous_channels",           (DL_FUNC) &R_ggml_is_contiguous_channels,            1},
    {"R_ggml_is_contiguous_rows",               (DL_FUNC) &R_ggml_is_contiguous_rows,                1},
    {"R_ggml_are_same_stride",                  (DL_FUNC) &R_ggml_are_same_stride,                   2},
    {"R_ggml_can_repeat",                       (DL_FUNC) &R_ggml_can_repeat,                        2},
    {"R_ggml_count_equal",                      (DL_FUNC) &R_ggml_count_equal,                       3},

    // Advanced RoPE
    {"R_ggml_rope_multi_back",                  (DL_FUNC) &R_ggml_rope_multi_back,                  14},

    // Graph Construction & Introspection
    {"R_ggml_build_backward_expand",            (DL_FUNC) &R_ggml_build_backward_expand,             2},
    {"R_ggml_graph_add_node",                   (DL_FUNC) &R_ggml_graph_add_node,                    2},
    {"R_ggml_graph_clear",                      (DL_FUNC) &R_ggml_graph_clear,                       1},
    {"R_ggml_graph_cpy",                        (DL_FUNC) &R_ggml_graph_cpy,                         2},
    {"R_ggml_graph_dup",                        (DL_FUNC) &R_ggml_graph_dup,                         3},
    {"R_ggml_graph_get_grad",                   (DL_FUNC) &R_ggml_graph_get_grad,                    2},
    {"R_ggml_graph_get_grad_acc",               (DL_FUNC) &R_ggml_graph_get_grad_acc,                2},
    {"R_ggml_graph_view",                       (DL_FUNC) &R_ggml_graph_view,                        3},
    {"R_ggml_op_can_inplace",                   (DL_FUNC) &R_ggml_op_can_inplace,                    1},
    {"R_ggml_are_same_layout",                  (DL_FUNC) &R_ggml_are_same_layout,                   2},

    // Backend async/multi-buffer
    {"R_ggml_backend_graph_compute_async",      (DL_FUNC) &R_ggml_backend_graph_compute_async,       2},
    {"R_ggml_backend_multi_buffer_alloc_buffer",(DL_FUNC) &R_ggml_backend_multi_buffer_alloc_buffer, 1},
    {"R_ggml_backend_buffer_is_multi_buffer",   (DL_FUNC) &R_ggml_backend_buffer_is_multi_buffer,    1},
    {"R_ggml_backend_multi_buffer_set_usage",   (DL_FUNC) &R_ggml_backend_multi_buffer_set_usage,    2},
    {"R_ggml_backend_register",                 (DL_FUNC) &R_ggml_backend_register,                  1},
    {"R_ggml_backend_device_register",          (DL_FUNC) &R_ggml_backend_device_register,           1},

    // Advanced Attention/Loss
    {"R_ggml_cross_entropy_loss",               (DL_FUNC) &R_ggml_cross_entropy_loss,                3},
    {"R_ggml_cross_entropy_loss_back",          (DL_FUNC) &R_ggml_cross_entropy_loss_back,           4},
    {"R_ggml_cumsum",                           (DL_FUNC) &R_ggml_cumsum,                            2},
    {"R_ggml_flash_attn_ext_set_prec",          (DL_FUNC) &R_ggml_flash_attn_ext_set_prec,           2},
    {"R_ggml_flash_attn_ext_get_prec",          (DL_FUNC) &R_ggml_flash_attn_ext_get_prec,           1},
    {"R_ggml_flash_attn_ext_add_sinks",         (DL_FUNC) &R_ggml_flash_attn_ext_add_sinks,          2},
    {"R_ggml_soft_max_add_sinks",               (DL_FUNC) &R_ggml_soft_max_add_sinks,                2},

    // Logging & debugging
    {"R_ggml_log_set_r",                        (DL_FUNC) &R_ggml_log_set_r,                         0},
    {"R_ggml_log_set_default",                  (DL_FUNC) &R_ggml_log_set_default,                   0},
    {"R_ggml_log_is_r_enabled",                 (DL_FUNC) &R_ggml_log_is_r_enabled,                  0},
    {"R_ggml_set_abort_callback_r",             (DL_FUNC) &R_ggml_set_abort_callback_r,              0},
    {"R_ggml_set_abort_callback_default",       (DL_FUNC) &R_ggml_set_abort_callback_default,        0},
    {"R_ggml_abort_is_r_enabled",               (DL_FUNC) &R_ggml_abort_is_r_enabled,                0},

    // Op params
    {"R_ggml_get_op_params",                    (DL_FUNC) &R_ggml_get_op_params,                     1},
    {"R_ggml_set_op_params",                    (DL_FUNC) &R_ggml_set_op_params,                     2},
    {"R_ggml_get_op_params_i32",                (DL_FUNC) &R_ggml_get_op_params_i32,                 2},
    {"R_ggml_set_op_params_i32",                (DL_FUNC) &R_ggml_set_op_params_i32,                 3},
    {"R_ggml_get_op_params_f32",                (DL_FUNC) &R_ggml_get_op_params_f32,                 2},
    {"R_ggml_set_op_params_f32",                (DL_FUNC) &R_ggml_set_op_params_f32,                 3},

    // Low-level quantization - dequantize row
    {"R_dequantize_row_q4_0",                   (DL_FUNC) &R_dequantize_row_q4_0,                    2},
    {"R_dequantize_row_q4_1",                   (DL_FUNC) &R_dequantize_row_q4_1,                    2},
    {"R_dequantize_row_q5_0",                   (DL_FUNC) &R_dequantize_row_q5_0,                    2},
    {"R_dequantize_row_q5_1",                   (DL_FUNC) &R_dequantize_row_q5_1,                    2},
    {"R_dequantize_row_q8_0",                   (DL_FUNC) &R_dequantize_row_q8_0,                    2},
    {"R_dequantize_row_q2_K",                   (DL_FUNC) &R_dequantize_row_q2_K,                    2},
    {"R_dequantize_row_q3_K",                   (DL_FUNC) &R_dequantize_row_q3_K,                    2},
    {"R_dequantize_row_q4_K",                   (DL_FUNC) &R_dequantize_row_q4_K,                    2},
    {"R_dequantize_row_q5_K",                   (DL_FUNC) &R_dequantize_row_q5_K,                    2},
    {"R_dequantize_row_q6_K",                   (DL_FUNC) &R_dequantize_row_q6_K,                    2},
    {"R_dequantize_row_q8_K",                   (DL_FUNC) &R_dequantize_row_q8_K,                    2},
    {"R_dequantize_row_tq1_0",                  (DL_FUNC) &R_dequantize_row_tq1_0,                   2},
    {"R_dequantize_row_tq2_0",                  (DL_FUNC) &R_dequantize_row_tq2_0,                   2},
    {"R_dequantize_row_iq2_xxs",                (DL_FUNC) &R_dequantize_row_iq2_xxs,                 2},
    {"R_dequantize_row_iq2_xs",                 (DL_FUNC) &R_dequantize_row_iq2_xs,                  2},
    {"R_dequantize_row_iq2_s",                  (DL_FUNC) &R_dequantize_row_iq2_s,                   2},
    {"R_dequantize_row_iq3_xxs",                (DL_FUNC) &R_dequantize_row_iq3_xxs,                 2},
    {"R_dequantize_row_iq3_s",                  (DL_FUNC) &R_dequantize_row_iq3_s,                   2},
    {"R_dequantize_row_iq4_nl",                 (DL_FUNC) &R_dequantize_row_iq4_nl,                  2},
    {"R_dequantize_row_iq4_xs",                 (DL_FUNC) &R_dequantize_row_iq4_xs,                  2},
    {"R_dequantize_row_iq1_s",                  (DL_FUNC) &R_dequantize_row_iq1_s,                   2},
    {"R_dequantize_row_iq1_m",                  (DL_FUNC) &R_dequantize_row_iq1_m,                   2},
    {"R_dequantize_row_mxfp4",                  (DL_FUNC) &R_dequantize_row_mxfp4,                   2},

    // Low-level quantization - quantize (with imatrix)
    {"R_quantize_q4_0",                         (DL_FUNC) &R_quantize_q4_0,                          4},
    {"R_quantize_q4_1",                         (DL_FUNC) &R_quantize_q4_1,                          4},
    {"R_quantize_q5_0",                         (DL_FUNC) &R_quantize_q5_0,                          4},
    {"R_quantize_q5_1",                         (DL_FUNC) &R_quantize_q5_1,                          4},
    {"R_quantize_q8_0",                         (DL_FUNC) &R_quantize_q8_0,                          4},
    {"R_quantize_q2_K",                         (DL_FUNC) &R_quantize_q2_K,                          4},
    {"R_quantize_q3_K",                         (DL_FUNC) &R_quantize_q3_K,                          4},
    {"R_quantize_q4_K",                         (DL_FUNC) &R_quantize_q4_K,                          4},
    {"R_quantize_q5_K",                         (DL_FUNC) &R_quantize_q5_K,                          4},
    {"R_quantize_q6_K",                         (DL_FUNC) &R_quantize_q6_K,                          4},
    {"R_quantize_tq1_0",                        (DL_FUNC) &R_quantize_tq1_0,                         4},
    {"R_quantize_tq2_0",                        (DL_FUNC) &R_quantize_tq2_0,                         4},
    {"R_quantize_iq2_xxs",                      (DL_FUNC) &R_quantize_iq2_xxs,                       4},
    {"R_quantize_iq2_xs",                       (DL_FUNC) &R_quantize_iq2_xs,                        4},
    {"R_quantize_iq2_s",                        (DL_FUNC) &R_quantize_iq2_s,                         4},
    {"R_quantize_iq3_xxs",                      (DL_FUNC) &R_quantize_iq3_xxs,                       4},
    {"R_quantize_iq3_s",                        (DL_FUNC) &R_quantize_iq3_s,                         4},
    {"R_quantize_iq1_s",                        (DL_FUNC) &R_quantize_iq1_s,                         4},
    {"R_quantize_iq1_m",                        (DL_FUNC) &R_quantize_iq1_m,                         4},
    {"R_quantize_iq4_nl",                       (DL_FUNC) &R_quantize_iq4_nl,                        4},
    {"R_quantize_iq4_xs",                       (DL_FUNC) &R_quantize_iq4_xs,                        4},
    {"R_quantize_mxfp4",                        (DL_FUNC) &R_quantize_mxfp4,                         4},

    // Low-level quantization - quantize row ref
    {"R_quantize_row_q4_0_ref",                 (DL_FUNC) &R_quantize_row_q4_0_ref,                  2},
    {"R_quantize_row_q4_1_ref",                 (DL_FUNC) &R_quantize_row_q4_1_ref,                  2},
    {"R_quantize_row_q5_0_ref",                 (DL_FUNC) &R_quantize_row_q5_0_ref,                  2},
    {"R_quantize_row_q5_1_ref",                 (DL_FUNC) &R_quantize_row_q5_1_ref,                  2},
    {"R_quantize_row_q8_0_ref",                 (DL_FUNC) &R_quantize_row_q8_0_ref,                  2},
    {"R_quantize_row_q8_1_ref",                 (DL_FUNC) &R_quantize_row_q8_1_ref,                  2},
    {"R_quantize_row_q2_K_ref",                 (DL_FUNC) &R_quantize_row_q2_K_ref,                  2},
    {"R_quantize_row_q3_K_ref",                 (DL_FUNC) &R_quantize_row_q3_K_ref,                  2},
    {"R_quantize_row_q4_K_ref",                 (DL_FUNC) &R_quantize_row_q4_K_ref,                  2},
    {"R_quantize_row_q5_K_ref",                 (DL_FUNC) &R_quantize_row_q5_K_ref,                  2},
    {"R_quantize_row_q6_K_ref",                 (DL_FUNC) &R_quantize_row_q6_K_ref,                  2},
    {"R_quantize_row_q8_K_ref",                 (DL_FUNC) &R_quantize_row_q8_K_ref,                  2},
    {"R_quantize_row_tq1_0_ref",                (DL_FUNC) &R_quantize_row_tq1_0_ref,                 2},
    {"R_quantize_row_tq2_0_ref",                (DL_FUNC) &R_quantize_row_tq2_0_ref,                 2},
    {"R_quantize_row_iq3_xxs_ref",              (DL_FUNC) &R_quantize_row_iq3_xxs_ref,               2},
    {"R_quantize_row_iq4_nl_ref",               (DL_FUNC) &R_quantize_row_iq4_nl_ref,                2},
    {"R_quantize_row_iq4_xs_ref",               (DL_FUNC) &R_quantize_row_iq4_xs_ref,                2},
    {"R_quantize_row_iq3_s_ref",                (DL_FUNC) &R_quantize_row_iq3_s_ref,                 2},
    {"R_quantize_row_iq2_s_ref",                (DL_FUNC) &R_quantize_row_iq2_s_ref,                 2},
    {"R_quantize_row_mxfp4_ref",                (DL_FUNC) &R_quantize_row_mxfp4_ref,                 2},

    // IQ init/free
    {"R_iq2xs_init_impl",                       (DL_FUNC) &R_iq2xs_init_impl,                        1},
    {"R_iq2xs_free_impl",                       (DL_FUNC) &R_iq2xs_free_impl,                        1},
    {"R_iq3xs_init_impl",                       (DL_FUNC) &R_iq3xs_init_impl,                        1},
    {"R_iq3xs_free_impl",                       (DL_FUNC) &R_iq3xs_free_impl,                        1},

    // Quantization info
    {"R_ggml_quant_block_info",                 (DL_FUNC) &R_ggml_quant_block_info,                  1},

    // Timestep embedding
    {"R_ggml_timestep_embedding",               (DL_FUNC) &R_ggml_timestep_embedding,                4},

    // CPU-side tensor data access
    {"R_ggml_set_f32_nd",                       (DL_FUNC) &R_ggml_set_f32_nd,                        6},
    {"R_ggml_get_f32_nd",                       (DL_FUNC) &R_ggml_get_f32_nd,                        5},
    {"R_ggml_get_i32_nd",                       (DL_FUNC) &R_ggml_get_i32_nd,                        5},
    {"R_ggml_set_i32_nd",                       (DL_FUNC) &R_ggml_set_i32_nd,                        6},
    {"R_ggml_tensor_nb",                        (DL_FUNC) &R_ggml_tensor_nb,                         1},
    {"R_ggml_backend_tensor_get_and_sync",      (DL_FUNC) &R_ggml_backend_tensor_get_and_sync,       4},
    {"R_ggml_backend_tensor_get_f32",           (DL_FUNC) &R_ggml_backend_tensor_get_f32,            1},
    {"R_ggml_tensor_num",                       (DL_FUNC) &R_ggml_tensor_num,                        1},
    {"R_ggml_tensor_data_ptr",                  (DL_FUNC) &R_ggml_tensor_data_ptr,                   1},
    {"R_ggml_tensor_copy",                      (DL_FUNC) &R_ggml_tensor_copy,                       2},
    {"R_ggml_tensor_set_f32_scalar",            (DL_FUNC) &R_ggml_tensor_set_f32_scalar,             2},
    {"R_ggml_get_first_tensor",                 (DL_FUNC) &R_ggml_get_first_tensor,                  1},
    {"R_ggml_get_next_tensor",                  (DL_FUNC) &R_ggml_get_next_tensor,                   2},

    {NULL, NULL, 0}
};

void R_init_ggmlR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
