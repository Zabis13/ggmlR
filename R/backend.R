# Extended backend functions
# Device management, registry, events, graph planning, buffer management

# ============================================================================
# Device Type Constants
# ============================================================================

#' Device type: CPU
#' @return Integer constant for CPU device type
#' @export
#' @family backend
ggml_backend_device_type_cpu <- function() {
 .Call("R_ggml_backend_device_type_cpu")
}

#' Device type: GPU
#' @return Integer constant for GPU device type
#' @export
#' @family backend
ggml_backend_device_type_gpu <- function() {
 .Call("R_ggml_backend_device_type_gpu")
}

#' Device type: Integrated GPU
#' @return Integer constant for integrated GPU device type
#' @export
#' @family backend
ggml_backend_device_type_igpu <- function() {
 .Call("R_ggml_backend_device_type_igpu")
}

#' Device type: Accelerator
#' @return Integer constant for accelerator device type (e.g. BLAS, AMX)
#' @export
#' @family backend
ggml_backend_device_type_accel <- function() {
 .Call("R_ggml_backend_device_type_accel")
}

# ============================================================================
# Buffer Usage Constants
# ============================================================================

#' Buffer usage: Any
#' @return Integer constant for any buffer usage
#' @export
#' @family backend
ggml_backend_buffer_usage_any <- function() {
 .Call("R_ggml_backend_buffer_usage_any")
}

#' Buffer usage: Weights
#' @return Integer constant for weights buffer usage
#' @export
#' @family backend
ggml_backend_buffer_usage_weights <- function() {
 .Call("R_ggml_backend_buffer_usage_weights")
}

#' Buffer usage: Compute
#' @return Integer constant for compute buffer usage
#' @export
#' @family backend
ggml_backend_buffer_usage_compute <- function() {
 .Call("R_ggml_backend_buffer_usage_compute")
}

# ============================================================================
# Device Enumeration
# ============================================================================

#' Get number of available devices
#' @return Number of devices
#' @export
#' @family backend
ggml_backend_dev_count <- function() {
 .Call("R_ggml_backend_dev_count")
}

#' Get device by index
#' @param index Device index (0-based)
#' @return External pointer to device, or NULL if not found
#' @export
#' @family backend
ggml_backend_dev_get <- function(index) {
 .Call("R_ggml_backend_dev_get", as.numeric(index))
}

#' Get device by name
#' @param name Device name
#' @return External pointer to device, or NULL if not found
#' @export
#' @family backend
ggml_backend_dev_by_name <- function(name) {
 .Call("R_ggml_backend_dev_by_name", as.character(name))
}

#' Get device by type
#' @param type Device type (use ggml_backend_device_type_* functions)
#' @return External pointer to first device of given type, or NULL if not found
#' @export
#' @family backend
ggml_backend_dev_by_type <- function(type) {
 .Call("R_ggml_backend_dev_by_type", as.integer(type))
}

# ============================================================================
# Device Properties
# ============================================================================

#' Get device name
#' @param device External pointer to device
#' @return Device name
#' @export
#' @family backend
ggml_backend_dev_name <- function(device) {
 .Call("R_ggml_backend_dev_name", device)
}

#' Get device description
#' @param device External pointer to device
#' @return Device description
#' @export
#' @family backend
ggml_backend_dev_description <- function(device) {
 .Call("R_ggml_backend_dev_description", device)
}

#' Get device memory
#' @param device External pointer to device
#' @return Named numeric vector with 'free' and 'total' memory in bytes
#' @export
#' @family backend
ggml_backend_dev_memory <- function(device) {
 .Call("R_ggml_backend_dev_memory", device)
}

#' Get device type
#' @param device External pointer to device
#' @return Device type constant
#' @export
#' @family backend
ggml_backend_dev_type <- function(device) {
 .Call("R_ggml_backend_dev_type", device)
}

#' Get device properties
#' @param device External pointer to device
#' @return List with name, description, memory_free, memory_total, type, device_id, caps
#' @export
#' @family backend
ggml_backend_dev_get_props <- function(device) {
 .Call("R_ggml_backend_dev_get_props", device)
}

#' Check if device supports operation
#' @param device External pointer to device
#' @param op External pointer to tensor/operation
#' @return Logical indicating support
#' @export
#' @family backend
ggml_backend_dev_supports_op <- function(device, op) {
 .Call("R_ggml_backend_dev_supports_op", device, op)
}

#' Check if device supports buffer type
#' @param device External pointer to device
#' @param buft External pointer to buffer type
#' @return Logical indicating support
#' @export
#' @family backend
ggml_backend_dev_supports_buft <- function(device, buft) {
 .Call("R_ggml_backend_dev_supports_buft", device, buft)
}

#' Check if device should offload operation
#' @param device External pointer to device
#' @param op External pointer to tensor/operation
#' @return Logical indicating if operation should be offloaded
#' @export
#' @family backend
ggml_backend_dev_offload_op <- function(device, op) {
 .Call("R_ggml_backend_dev_offload_op", device, op)
}

#' Initialize backend from device
#' @param device External pointer to device
#' @param params Optional parameters string
#' @return External pointer to backend, or NULL on failure
#' @export
#' @family backend
ggml_backend_dev_init <- function(device, params = NULL) {
 .Call("R_ggml_backend_dev_init", device, params)
}

# ============================================================================
# Backend Registry
# ============================================================================

#' Get number of registered backends
#' @return Number of registered backends
#' @export
#' @family backend
ggml_backend_reg_count <- function() {
 .Call("R_ggml_backend_reg_count")
}

#' Get backend registry by index
#' @param index Registry index (0-based)
#' @return External pointer to registry, or NULL if not found
#' @export
#' @family backend
ggml_backend_reg_get <- function(index) {
 .Call("R_ggml_backend_reg_get", as.numeric(index))
}

#' Get backend registry by name
#' @param name Registry name
#' @return External pointer to registry, or NULL if not found
#' @export
#' @family backend
ggml_backend_reg_by_name <- function(name) {
 .Call("R_ggml_backend_reg_by_name", as.character(name))
}

#' Get registry name
#' @param reg External pointer to registry
#' @return Registry name
#' @export
#' @family backend
ggml_backend_reg_name <- function(reg) {
 .Call("R_ggml_backend_reg_name", reg)
}

#' Get number of devices in registry
#' @param reg External pointer to registry
#' @return Number of devices
#' @export
#' @family backend
ggml_backend_reg_dev_count <- function(reg) {
 .Call("R_ggml_backend_reg_dev_count", reg)
}

#' Get device from registry
#' @param reg External pointer to registry
#' @param index Device index (0-based)
#' @return External pointer to device
#' @export
#' @family backend
ggml_backend_reg_dev_get <- function(reg, index) {
 .Call("R_ggml_backend_reg_dev_get", reg, as.numeric(index))
}

#' Load backend from dynamic library
#' @param path Path to dynamic library
#' @return External pointer to registry, or NULL on failure
#' @export
#' @family backend
ggml_backend_load <- function(path) {
 .Call("R_ggml_backend_load", as.character(path))
}

#' Unload backend
#' @param reg External pointer to registry
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_unload <- function(reg) {
 invisible(.Call("R_ggml_backend_unload", reg))
}

#' Load all available backends
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_load_all <- function() {
 invisible(.Call("R_ggml_backend_load_all"))
}

# ============================================================================
# Events
# ============================================================================

#' Create new event
#' @param device External pointer to device
#' @return External pointer to event, or NULL on failure
#' @export
#' @family backend
ggml_backend_event_new <- function(device) {
 .Call("R_ggml_backend_event_new", device)
}

#' Free event
#' @param event External pointer to event
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_event_free <- function(event) {
 invisible(.Call("R_ggml_backend_event_free", event))
}

#' Record event
#' @param event External pointer to event
#' @param backend External pointer to backend
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_event_record <- function(event, backend) {
 invisible(.Call("R_ggml_backend_event_record", event, backend))
}

#' Synchronize event
#' @param event External pointer to event
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_event_synchronize <- function(event) {
 invisible(.Call("R_ggml_backend_event_synchronize", event))
}

#' Wait for event
#' @param backend External pointer to backend
#' @param event External pointer to event
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_event_wait <- function(backend, event) {
 invisible(.Call("R_ggml_backend_event_wait", backend, event))
}

# ============================================================================
# Graph Planning
# ============================================================================

#' Create graph execution plan
#' @param backend External pointer to backend
#' @param graph External pointer to computation graph
#' @return External pointer to plan, or NULL on failure
#' @export
#' @family backend
ggml_backend_graph_plan_create <- function(backend, graph) {
 .Call("R_ggml_backend_graph_plan_create", backend, graph)
}

#' Free graph execution plan
#' @param backend External pointer to backend
#' @param plan External pointer to plan
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_graph_plan_free <- function(backend, plan) {
 invisible(.Call("R_ggml_backend_graph_plan_free", backend, plan))
}

#' Execute graph plan
#' @param backend External pointer to backend
#' @param plan External pointer to plan
#' @return Status code (0 = success)
#' @export
#' @family backend
ggml_backend_graph_plan_compute <- function(backend, plan) {
 .Call("R_ggml_backend_graph_plan_compute", backend, plan)
}

# ============================================================================
# Async Operations
# ============================================================================
#' Set tensor data asynchronously
#' @param backend External pointer to backend
#' @param tensor External pointer to tensor
#' @param data Numeric or integer vector
#' @param offset Byte offset (default 0)
#' @param size Number of bytes to copy
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_tensor_set_async <- function(backend, tensor, data, offset = 0, size = NULL) {
 if (is.null(size)) {
   size <- length(data) * ifelse(is.integer(data), 4L, 8L)
 }
 invisible(.Call("R_ggml_backend_tensor_set_async", backend, tensor, data,
                 as.numeric(offset), as.numeric(size)))
}

#' Get tensor data asynchronously
#' @param backend External pointer to backend
#' @param tensor External pointer to tensor
#' @param offset Byte offset (default 0)
#' @param size Number of bytes to read
#' @return Numeric vector with data
#' @export
#' @family backend
ggml_backend_tensor_get_async <- function(backend, tensor, offset = 0, size) {
 .Call("R_ggml_backend_tensor_get_async", backend, tensor,
       as.numeric(offset), as.numeric(size))
}

#' Copy tensor asynchronously between backends
#' @param backend_src Source backend
#' @param backend_dst Destination backend
#' @param src Source tensor
#' @param dst Destination tensor
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_tensor_copy_async <- function(backend_src, backend_dst, src, dst) {
 invisible(.Call("R_ggml_backend_tensor_copy_async", backend_src, backend_dst, src, dst))
}

# ============================================================================
# Buffer Management
# ============================================================================

#' Clear buffer memory
#' @param buffer External pointer to buffer
#' @param value Byte value to fill with (default 0)
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_buffer_clear <- function(buffer, value = 0L) {
 invisible(.Call("R_ggml_backend_buffer_clear", buffer, as.integer(value)))
}

#' Set buffer usage hint
#' @param buffer External pointer to buffer
#' @param usage Usage constant (use ggml_backend_buffer_usage_* functions)
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_buffer_set_usage <- function(buffer, usage) {
 invisible(.Call("R_ggml_backend_buffer_set_usage", buffer, as.integer(usage)))
}

#' Get buffer usage
#' @param buffer External pointer to buffer
#' @return Usage constant
#' @export
#' @family backend
ggml_backend_buffer_get_usage <- function(buffer) {
 .Call("R_ggml_backend_buffer_get_usage", buffer)
}

#' Reset buffer
#' @param buffer External pointer to buffer
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_buffer_reset <- function(buffer) {
 invisible(.Call("R_ggml_backend_buffer_reset", buffer))
}

#' Check if buffer is host memory
#' @param buffer External pointer to buffer
#' @return Logical indicating if buffer is in host memory
#' @export
#' @family backend
ggml_backend_buffer_is_host <- function(buffer) {
 .Call("R_ggml_backend_buffer_is_host", buffer)
}

# ============================================================================
# Direct Backend Initialization
# ============================================================================

#' Initialize backend by name
#' @param name Backend name (e.g. "CPU", "Vulkan")
#' @param params Optional parameters string
#' @return External pointer to backend, or NULL on failure
#' @export
#' @family backend
ggml_backend_init_by_name <- function(name, params = NULL) {
 .Call("R_ggml_backend_init_by_name", as.character(name), params)
}

#' Initialize backend by type
#' @param type Device type constant
#' @param params Optional parameters string
#' @return External pointer to backend, or NULL on failure
#' @export
#' @family backend
ggml_backend_init_by_type <- function(type, params = NULL) {
 .Call("R_ggml_backend_init_by_type", as.integer(type), params)
}

#' Initialize best available backend
#' @return External pointer to backend (GPU if available, otherwise CPU)
#' @export
#' @family backend
ggml_backend_init_best <- function() {
 .Call("R_ggml_backend_init_best")
}

#' Synchronize backend
#' @param backend External pointer to backend
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_synchronize <- function(backend) {
 invisible(.Call("R_ggml_backend_synchronize", backend))
}

#' Get device from backend
#' @param backend External pointer to backend
#' @return External pointer to device
#' @export
#' @family backend
ggml_backend_get_device <- function(backend) {
 .Call("R_ggml_backend_get_device", backend)
}

# ============================================================================
# Async Graph Compute
# ============================================================================

#' Compute graph asynchronously
#'
#' Starts graph computation without blocking. Use ggml_backend_synchronize()
#' to wait for completion.
#'
#' @param backend External pointer to backend
#' @param graph External pointer to computation graph
#' @return Integer status code (0 = success)
#' @export
#' @family backend
#' @examples
#' \dontrun{
#' status <- ggml_backend_graph_compute_async(backend, graph)
#' # Do other work while computation runs...
#' ggml_backend_synchronize(backend)  # Wait for completion
#' }
ggml_backend_graph_compute_async <- function(backend, graph) {
  .Call("R_ggml_backend_graph_compute_async", backend, graph)
}

# ============================================================================
# Multi-buffer Operations
# ============================================================================

#' Allocate multi-buffer
#'
#' Creates a buffer that combines multiple backend buffers into one.
#' Useful for managing memory across different backends.
#'
#' @param buffers List of backend buffer external pointers
#' @return External pointer to multi-buffer
#' @export
#' @family backend
#' @examples
#' \dontrun{
#' buf1 <- ggml_backend_alloc_buffer(backend, 1024)
#' buf2 <- ggml_backend_alloc_buffer(backend, 2048)
#' multi <- ggml_backend_multi_buffer_alloc_buffer(list(buf1, buf2))
#' }
ggml_backend_multi_buffer_alloc_buffer <- function(buffers) {
  if (!is.list(buffers)) {
    buffers <- list(buffers)
  }
  .Call("R_ggml_backend_multi_buffer_alloc_buffer", buffers)
}

#' Check if buffer is a multi-buffer
#'
#' @param buffer External pointer to buffer
#' @return Logical indicating if buffer is a multi-buffer
#' @export
#' @family backend
ggml_backend_buffer_is_multi_buffer <- function(buffer) {
  .Call("R_ggml_backend_buffer_is_multi_buffer", buffer)
}

#' Set usage for all buffers in a multi-buffer
#'
#' @param buffer External pointer to multi-buffer
#' @param usage Usage constant (from ggml_backend_buffer_usage_*)
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_multi_buffer_set_usage <- function(buffer, usage) {
  invisible(.Call("R_ggml_backend_multi_buffer_set_usage", buffer,
                  as.integer(usage)))
}

# ============================================================================
# Backend Registration
# ============================================================================

#' Register a backend
#'
#' Dynamically registers a new backend in the global registry.
#' This is an advanced function for custom backend development.
#'
#' @param reg External pointer to backend registry
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_register <- function(reg) {
  invisible(.Call("R_ggml_backend_register", reg))
}

#' Register a device
#'
#' Dynamically registers a new device in the global registry.
#' This is an advanced function for custom backend development.
#'
#' @param device External pointer to device
#' @return NULL invisibly
#' @export
#' @family backend
ggml_backend_device_register <- function(device) {
  invisible(.Call("R_ggml_backend_device_register", device))
}
