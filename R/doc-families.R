# Overview pages for large function families.
#
# These replace roxygen @family tags: with @family, every member page carries
# a link to every other member, so the manual grows quadratically with the
# family size. Members link here once via @seealso instead.

#' Backend functions
#'
#' Low-level access to the ggml backend API: devices, registries, buffer
#' types, buffers, events, graph plans and asynchronous transfers.
#'
#' @section Devices:
#' \code{\link{ggml_backend_dev_count}}, \code{\link{ggml_backend_dev_get}},
#' \code{\link{ggml_backend_dev_by_name}}, \code{\link{ggml_backend_dev_by_type}},
#' \code{\link{ggml_backend_dev_name}}, \code{\link{ggml_backend_dev_description}},
#' \code{\link{ggml_backend_dev_memory}}, \code{\link{ggml_backend_dev_type}},
#' \code{\link{ggml_backend_dev_get_props}}, \code{\link{ggml_backend_dev_supports_op}},
#' \code{\link{ggml_backend_dev_supports_buft}}, \code{\link{ggml_backend_dev_offload_op}},
#' \code{\link{ggml_backend_dev_buffer_type}}, \code{\link{ggml_backend_dev_host_buffer_type}},
#' \code{\link{ggml_backend_dev_init}}, \code{\link{ggml_backend_get_device}},
#' \code{\link{ggml_backend_meta_device}}, \code{\link{ggml_backend_device_register}}.
#'
#' Device type constants: \code{\link{ggml_backend_device_type_cpu}},
#' \code{\link{ggml_backend_device_type_gpu}}, \code{\link{ggml_backend_device_type_igpu}},
#' \code{\link{ggml_backend_device_type_accel}}.
#'
#' @section Backend initialization and execution:
#' \code{\link{ggml_backend_init_by_name}}, \code{\link{ggml_backend_init_by_type}},
#' \code{\link{ggml_backend_init_best}}, \code{\link{ggml_backend_synchronize}},
#' \code{\link{ggml_backend_graph_compute_async}},
#' \code{\link{ggml_backend_graph_plan_create}}, \code{\link{ggml_backend_graph_plan_free}},
#' \code{\link{ggml_backend_graph_plan_compute}}.
#'
#' @section Registry and dynamic loading:
#' \code{\link{ggml_backend_reg_count}}, \code{\link{ggml_backend_reg_get}},
#' \code{\link{ggml_backend_reg_by_name}}, \code{\link{ggml_backend_reg_name}},
#' \code{\link{ggml_backend_reg_dev_count}}, \code{\link{ggml_backend_reg_dev_get}},
#' \code{\link{ggml_backend_register}}, \code{\link{ggml_backend_load}},
#' \code{\link{ggml_backend_unload}}, \code{\link{ggml_backend_load_all}}.
#'
#' @section Buffer types and buffers:
#' \code{\link{ggml_backend_buft_name}}, \code{\link{ggml_backend_buft_get_alignment}},
#' \code{\link{ggml_backend_buft_get_max_size}}, \code{\link{ggml_backend_buft_is_host}},
#' \code{\link{ggml_backend_buffer_clear}}, \code{\link{ggml_backend_buffer_set_usage}},
#' \code{\link{ggml_backend_buffer_get_usage}}, \code{\link{ggml_backend_buffer_reset}},
#' \code{\link{ggml_backend_buffer_is_host}},
#' \code{\link{ggml_backend_multi_buffer_alloc_buffer}},
#' \code{\link{ggml_backend_buffer_is_multi_buffer}},
#' \code{\link{ggml_backend_multi_buffer_set_usage}}.
#'
#' Buffer usage constants: \code{\link{ggml_backend_buffer_usage_any}},
#' \code{\link{ggml_backend_buffer_usage_weights}},
#' \code{\link{ggml_backend_buffer_usage_compute}}.
#'
#' @section Events and asynchronous transfers:
#' \code{\link{ggml_backend_event_new}}, \code{\link{ggml_backend_event_free}},
#' \code{\link{ggml_backend_event_record}}, \code{\link{ggml_backend_event_synchronize}},
#' \code{\link{ggml_backend_event_wait}}, \code{\link{ggml_backend_tensor_set_async}},
#' \code{\link{ggml_backend_tensor_get_async}}, \code{\link{ggml_backend_tensor_copy_async}}.
#'
#' @name ggmlR-backend
#' @keywords internal
NULL

#' Optimization (ggml-opt) functions
#'
#' Training through the ggml-opt API: datasets, optimizer contexts, results
#' and fit loops.
#'
#' @section High-level training:
#' \code{\link{ggml_fit_opt}}, \code{\link{ggml_fit_opt_multi}},
#' \code{\link{ggml_fit_opt_gen}}, \code{\link{ggml_opt_fit}},
#' \code{\link{ggml_opt_fit_multi}}, \code{\link{ggml_opt_epoch}},
#' \code{\link{ggml_opt_eval}}, \code{\link{ggml_trainer}},
#' \code{\link{ggml_snapshot_free}}.
#'
#' @section Constants, datasets and results:
#' \code{\link{ggml_opt_loss_type}} (loss and optimizer type constants),
#' \code{\link{ggml_opt_dataset}} (dataset functions),
#' \code{\link{ggml_opt_result}} (result functions).
#'
#' @section Optimizer context:
#' \code{\link{ggml_opt_default_params}}, \code{\link{ggml_opt_init}},
#' \code{\link{ggml_opt_init_for_fit}}, \code{\link{ggml_opt_init_for_fit_multi}},
#' \code{\link{ggml_opt_free}}, \code{\link{ggml_opt_reset}},
#' \code{\link{ggml_opt_static_graphs}}, \code{\link{ggml_opt_prepare_alloc}},
#' \code{\link{ggml_opt_alloc}}, \code{\link{ggml_opt_grad_acc}},
#' \code{\link{ggml_opt_set_lr}}, \code{\link{ggml_opt_get_lr}},
#' \code{\link{ggml_opt_context_optimizer_type}}, \code{\link{ggml_opt_optimizer_name}}.
#'
#' @section Graph tensors:
#' \code{\link{ggml_opt_inputs}}, \code{\link{ggml_opt_outputs}},
#' \code{\link{ggml_opt_outputs_i}}, \code{\link{ggml_opt_labels}},
#' \code{\link{ggml_opt_loss}}, \code{\link{ggml_opt_loss_i}},
#' \code{\link{ggml_opt_n_loss}}, \code{\link{ggml_opt_pred}},
#' \code{\link{ggml_opt_ncorrect}}.
#'
#' @name ggmlR-optimization
#' @keywords internal
NULL

#' Quantization functions
#'
#' Low-level quantization and dequantization of raw data for all ggml quant
#' types.
#'
#' \describe{
#'   \item{\code{\link{quantize_data}}}{Quantize rows with optional importance
#'     matrix (\code{quantize_q4_0()}, \code{quantize_q2_K()}, ...).}
#'   \item{\code{\link{quantize_row_ref}}}{Reference row quantizers
#'     (\code{quantize_row_q4_0_ref()}, ...).}
#'   \item{\code{\link{dequantize_row}}}{Dequantize rows
#'     (\code{dequantize_row_q4_0()}, ...).}
#'   \item{IQ lookup tables}{\code{\link{iq2xs_init_impl}},
#'     \code{\link{iq2xs_free_impl}}, \code{\link{iq3xs_init_impl}},
#'     \code{\link{iq3xs_free_impl}}.}
#'   \item{Block layout}{\code{\link{ggml_quant_block_info}}.}
#' }
#'
#' @name ggmlR-quantization
#' @keywords internal
NULL
