# Extended backend functions
# Device management, registry, events, graph planning, buffer management

# ============================================================================
# Device Type Constants
# ============================================================================

#' Backend Device Type Constants
#' @return Integer constant for CPU device type
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_device_type_cpu <- function() {
 .Call("R_ggml_backend_device_type_cpu")
}

#' @rdname ggml_backend_device_type_cpu
#' @details \code{ggml_backend_device_type_gpu()}: 
#' Returns: Integer constant for GPU device type
#' @export
ggml_backend_device_type_gpu <- function() {
 .Call("R_ggml_backend_device_type_gpu")
}

#' @rdname ggml_backend_device_type_cpu
#' @details \code{ggml_backend_device_type_igpu()}: 
#' Returns: Integer constant for integrated GPU device type
#' @export
ggml_backend_device_type_igpu <- function() {
 .Call("R_ggml_backend_device_type_igpu")
}

#' @rdname ggml_backend_device_type_cpu
#' @details \code{ggml_backend_device_type_accel()}: 
#' Returns: Integer constant for accelerator device type (e.g. BLAS, AMX)
#' @export
ggml_backend_device_type_accel <- function() {
 .Call("R_ggml_backend_device_type_accel")
}

# ============================================================================
# Buffer Usage Constants
# ============================================================================

#' Backend Buffer Usage Constants
#' @return Integer constant for any buffer usage
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_buffer_usage_any <- function() {
 .Call("R_ggml_backend_buffer_usage_any")
}

#' @rdname ggml_backend_buffer_usage_any
#' @details \code{ggml_backend_buffer_usage_weights()}: 
#' Returns: Integer constant for weights buffer usage
#' @export
ggml_backend_buffer_usage_weights <- function() {
 .Call("R_ggml_backend_buffer_usage_weights")
}

#' @rdname ggml_backend_buffer_usage_any
#' @details \code{ggml_backend_buffer_usage_compute()}: 
#' Returns: Integer constant for compute buffer usage
#' @export
ggml_backend_buffer_usage_compute <- function() {
 .Call("R_ggml_backend_buffer_usage_compute")
}

# ============================================================================
# Device Enumeration
# ============================================================================

#' Backend Device Lookup
#' @return Number of devices
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_dev_count <- function() {
 .Call("R_ggml_backend_dev_count")
}

#' @rdname ggml_backend_dev_count
#' @details \code{ggml_backend_dev_get()}: 
#' Returns: External pointer to device, or NULL if not found
#' @param index Device index (0-based)
#' @export
ggml_backend_dev_get <- function(index) {
 .Call("R_ggml_backend_dev_get", as.numeric(index))
}

#' @rdname ggml_backend_dev_count
#' @details \code{ggml_backend_dev_by_name()}: 
#' Returns: External pointer to device, or NULL if not found
#' @param name Device name
#' @export
ggml_backend_dev_by_name <- function(name) {
 .Call("R_ggml_backend_dev_by_name", as.character(name))
}

#' @rdname ggml_backend_dev_count
#' @details \code{ggml_backend_dev_by_type()}: 
#' Returns: External pointer to first device of given type, or NULL if not found
#' @param type Device type (use ggml_backend_device_type_* functions)
#' @export
ggml_backend_dev_by_type <- function(type) {
 .Call("R_ggml_backend_dev_by_type", as.integer(type))
}

# ============================================================================
# Device Properties
# ============================================================================

#' Backend Device Properties
#' @param device External pointer to device
#' @return Device name
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_dev_name <- function(device) {
 .Call("R_ggml_backend_dev_name", device)
}

#' @rdname ggml_backend_dev_name
#' @details \code{ggml_backend_dev_description()}: 
#' Returns: Device description
#' @export
ggml_backend_dev_description <- function(device) {
 .Call("R_ggml_backend_dev_description", device)
}

#' @rdname ggml_backend_dev_name
#' @details \code{ggml_backend_dev_memory()}: 
#' Returns: Named numeric vector with 'free' and 'total' memory in bytes
#' @export
ggml_backend_dev_memory <- function(device) {
 .Call("R_ggml_backend_dev_memory", device)
}

#' @rdname ggml_backend_dev_name
#' @details \code{ggml_backend_dev_type()}: 
#' Returns: Device type constant
#' @export
ggml_backend_dev_type <- function(device) {
 .Call("R_ggml_backend_dev_type", device)
}

#' @rdname ggml_backend_dev_name
#' @details \code{ggml_backend_dev_get_props()}: 
#' Returns: List with name, description, memory_free, memory_total, type, device_id, caps
#' @export
ggml_backend_dev_get_props <- function(device) {
 .Call("R_ggml_backend_dev_get_props", device)
}

#' Backend Device Capability Queries
#' @param device External pointer to device
#' @param op External pointer to tensor/operation
#' @return Logical indicating support
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_dev_supports_op <- function(device, op) {
 .Call("R_ggml_backend_dev_supports_op", device, op)
}

#' @rdname ggml_backend_dev_supports_op
#' @details \code{ggml_backend_dev_supports_buft()}: 
#' Returns: Logical indicating support
#' @param buft External pointer to buffer type
#' @export
ggml_backend_dev_supports_buft <- function(device, buft) {
 .Call("R_ggml_backend_dev_supports_buft", device, buft)
}

#' @rdname ggml_backend_dev_supports_op
#' @details \code{ggml_backend_dev_offload_op()}: 
#' Returns: Logical indicating if operation should be offloaded
#' @export
ggml_backend_dev_offload_op <- function(device, op) {
 .Call("R_ggml_backend_dev_offload_op", device, op)
}

#' Backend Device Buffer Types
#'
#' Returns the buffer type a device allocates from by default. This is the
#' buffer type the device itself reports as supported, so it is the natural
#' argument to \code{\link{ggml_backend_dev_supports_buft}}.
#'
#' The buffer type is owned by the backend and lives for as long as the backend
#' does; the returned external pointer has no finalizer and must not be freed.
#'
#' @param device External pointer to device
#' @return External pointer to the buffer type, or NULL if the device has none
#' @export
#' @seealso \link{ggmlR-backend}
#' @examples
#' \donttest{
#' dev <- ggml_backend_dev_get(0)
#' buft <- ggml_backend_dev_buffer_type(dev)
#' ggml_backend_buft_name(buft)
#' ggml_backend_dev_supports_buft(dev, buft)  # TRUE: a device supports its own
#' }
ggml_backend_dev_buffer_type <- function(device) {
 .Call("R_ggml_backend_dev_buffer_type", device)
}

#' @rdname ggml_backend_dev_buffer_type
#' @details \code{ggml_backend_dev_host_buffer_type()}: Returns the buffer type used for pinned host memory, which allows faster
#' host-device transfers. Many devices — including the CPU backend — have no
#' such buffer type and return NULL.
#'
#' As with \code{\link{ggml_backend_dev_buffer_type}}, the returned pointer is
#' owned by the backend and must not be freed.
#' Returns: External pointer to the host buffer type, or NULL if unavailable
#' @export
ggml_backend_dev_host_buffer_type <- function(device) {
 .Call("R_ggml_backend_dev_host_buffer_type", device)
}

#' Backend Buffer Type Information
#' @param buft External pointer to buffer type
#' @return Character name of the buffer type
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_buft_name <- function(buft) {
 .Call("R_ggml_backend_buft_name", buft)
}

#' @rdname ggml_backend_buft_name
#' @details \code{ggml_backend_buft_get_alignment()}: 
#' Returns: Numeric alignment in bytes
#' @export
ggml_backend_buft_get_alignment <- function(buft) {
 .Call("R_ggml_backend_buft_get_alignment", buft)
}

#' @rdname ggml_backend_buft_name
#' @details \code{ggml_backend_buft_get_max_size()}: 
#' Returns: Numeric maximum single-allocation size in bytes
#' @export
ggml_backend_buft_get_max_size <- function(buft) {
 .Call("R_ggml_backend_buft_get_max_size", buft)
}

#' @rdname ggml_backend_buft_name
#' @details \code{ggml_backend_buft_is_host()}: 
#' Returns: Logical, TRUE if buffers of this type live in host memory
#' @export
ggml_backend_buft_is_host <- function(buft) {
 .Call("R_ggml_backend_buft_is_host", buft)
}

#' @rdname ggml_backend_init_by_name
#' @details \code{ggml_backend_dev_init()}: 
#' Returns: External pointer to backend, or NULL on failure
#' @param device External pointer to device
#' @export
ggml_backend_dev_init <- function(device, params = NULL) {
 .Call("R_ggml_backend_dev_init", device, params)
}

# ============================================================================
# Backend Registry
# ============================================================================

#' Backend Registry
#' @return Number of registered backends
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_reg_count <- function() {
 .Call("R_ggml_backend_reg_count")
}

#' @rdname ggml_backend_reg_count
#' @details \code{ggml_backend_reg_get()}: 
#' Returns: External pointer to registry, or NULL if not found
#' @param index Registry index (0-based)
#' @export
ggml_backend_reg_get <- function(index) {
 .Call("R_ggml_backend_reg_get", as.numeric(index))
}

#' @rdname ggml_backend_reg_count
#' @details \code{ggml_backend_reg_by_name()}: 
#' Returns: External pointer to registry, or NULL if not found
#' @param name Registry name
#' @export
ggml_backend_reg_by_name <- function(name) {
 .Call("R_ggml_backend_reg_by_name", as.character(name))
}

#' @rdname ggml_backend_reg_count
#' @details \code{ggml_backend_reg_name()}: 
#' Returns: Registry name
#' @param reg External pointer to registry
#' @export
ggml_backend_reg_name <- function(reg) {
 .Call("R_ggml_backend_reg_name", reg)
}

#' @rdname ggml_backend_reg_count
#' @details \code{ggml_backend_reg_dev_count()}: 
#' Returns: Number of devices
#' @export
ggml_backend_reg_dev_count <- function(reg) {
 .Call("R_ggml_backend_reg_dev_count", reg)
}

#' @rdname ggml_backend_reg_count
#' @details \code{ggml_backend_reg_dev_get()}: 
#' Returns: External pointer to device
#' @export
ggml_backend_reg_dev_get <- function(reg, index) {
 .Call("R_ggml_backend_reg_dev_get", reg, as.numeric(index))
}

#' Dynamic Backend Loading
#' @param path Path to dynamic library
#' @return External pointer to registry, or NULL on failure
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_load <- function(path) {
 .Call("R_ggml_backend_load", as.character(path))
}

#' @rdname ggml_backend_load
#' @details \code{ggml_backend_unload()}: 
#' Returns: NULL invisibly
#' @param reg External pointer to registry
#' @export
ggml_backend_unload <- function(reg) {
 invisible(.Call("R_ggml_backend_unload", reg))
}

#' @rdname ggml_backend_load
#' @details \code{ggml_backend_load_all()}: 
#' Returns: NULL invisibly
#' @export
ggml_backend_load_all <- function() {
 invisible(.Call("R_ggml_backend_load_all"))
}

# ============================================================================
# Events
# ============================================================================

#' Backend Events
#' @param device External pointer to device
#' @return External pointer to event, or NULL on failure
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_event_new <- function(device) {
 .Call("R_ggml_backend_event_new", device)
}

#' @rdname ggml_backend_event_new
#' @details \code{ggml_backend_event_free()}: 
#' Returns: NULL invisibly
#' @param event External pointer to event
#' @export
ggml_backend_event_free <- function(event) {
 invisible(.Call("R_ggml_backend_event_free", event))
}

#' @rdname ggml_backend_event_new
#' @details \code{ggml_backend_event_record()}: 
#' Returns: NULL invisibly
#' @param backend External pointer to backend
#' @export
ggml_backend_event_record <- function(event, backend) {
 invisible(.Call("R_ggml_backend_event_record", event, backend))
}

#' @rdname ggml_backend_event_new
#' @details \code{ggml_backend_event_synchronize()}: 
#' Returns: NULL invisibly
#' @export
ggml_backend_event_synchronize <- function(event) {
 invisible(.Call("R_ggml_backend_event_synchronize", event))
}

#' @rdname ggml_backend_event_new
#' @details \code{ggml_backend_event_wait()}: 
#' Returns: NULL invisibly
#' @export
ggml_backend_event_wait <- function(backend, event) {
 invisible(.Call("R_ggml_backend_event_wait", backend, event))
}

# ============================================================================
# Graph Planning
# ============================================================================

#' Backend Graph Plans
#' @param backend External pointer to backend
#' @param graph External pointer to computation graph
#' @return External pointer to plan, or NULL on failure
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_graph_plan_create <- function(backend, graph) {
 .Call("R_ggml_backend_graph_plan_create", backend, graph)
}

#' @rdname ggml_backend_graph_plan_create
#' @details \code{ggml_backend_graph_plan_free()}: 
#' Returns: NULL invisibly
#' @param plan External pointer to plan
#' @export
ggml_backend_graph_plan_free <- function(backend, plan) {
 invisible(.Call("R_ggml_backend_graph_plan_free", backend, plan))
}

#' @rdname ggml_backend_graph_plan_create
#' @details \code{ggml_backend_graph_plan_compute()}: 
#' Returns: Status code (0 = success)
#' @export
ggml_backend_graph_plan_compute <- function(backend, plan) {
 .Call("R_ggml_backend_graph_plan_compute", backend, plan)
}

# ============================================================================
# Async Operations
# ============================================================================
#' Asynchronous Tensor Transfers
#' @param backend External pointer to backend
#' @param tensor External pointer to tensor
#' @param data Numeric or integer vector
#' @param offset Byte offset (default 0)
#' @param size Number of bytes to copy
#' @return NULL invisibly
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_tensor_set_async <- function(backend, tensor, data, offset = 0, size = NULL) {
 if (is.null(size)) {
   size <- length(data) * ifelse(is.integer(data), 4L, 8L)
 }
 invisible(.Call("R_ggml_backend_tensor_set_async", backend, tensor, data,
                 as.numeric(offset), as.numeric(size)))
}

#' @rdname ggml_backend_tensor_set_async
#' @details \code{ggml_backend_tensor_get_async()}: 
#' Returns: Double vector of \code{ceiling(size / 8)} elements carrying the raw bytes (reinterpret with \code{writeBin}/\code{readBin}); bytes past \code{size} are zero. Filled only after \code{ggml_backend_synchronize()}.
#' @export
ggml_backend_tensor_get_async <- function(backend, tensor, offset = 0, size) {
 .Call("R_ggml_backend_tensor_get_async", backend, tensor,
       as.numeric(offset), as.numeric(size))
}

#' @rdname ggml_backend_tensor_set_async
#' @details \code{ggml_backend_tensor_copy_async()}: 
#' Returns: NULL invisibly
#' @param backend_src Source backend
#' @param backend_dst Destination backend
#' @param src Source tensor
#' @param dst Destination tensor
#' @export
ggml_backend_tensor_copy_async <- function(backend_src, backend_dst, src, dst) {
 invisible(.Call("R_ggml_backend_tensor_copy_async", backend_src, backend_dst, src, dst))
}

# ============================================================================
# Buffer Management
# ============================================================================

#' Backend Buffer Operations
#' @param buffer External pointer to buffer
#' @param value Byte value to fill with (default 0)
#' @return NULL invisibly
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_buffer_clear <- function(buffer, value = 0L) {
 invisible(.Call("R_ggml_backend_buffer_clear", buffer, as.integer(value)))
}

#' @rdname ggml_backend_buffer_clear
#' @details \code{ggml_backend_buffer_set_usage()}: 
#' Returns: NULL invisibly
#' @param usage Usage constant (use ggml_backend_buffer_usage_* functions)
#' @export
ggml_backend_buffer_set_usage <- function(buffer, usage) {
 invisible(.Call("R_ggml_backend_buffer_set_usage", buffer, as.integer(usage)))
}

#' @rdname ggml_backend_buffer_clear
#' @details \code{ggml_backend_buffer_get_usage()}: 
#' Returns: Usage constant
#' @export
ggml_backend_buffer_get_usage <- function(buffer) {
 .Call("R_ggml_backend_buffer_get_usage", buffer)
}

#' @rdname ggml_backend_buffer_clear
#' @details \code{ggml_backend_buffer_reset()}: 
#' Returns: NULL invisibly
#' @export
ggml_backend_buffer_reset <- function(buffer) {
 invisible(.Call("R_ggml_backend_buffer_reset", buffer))
}

#' @rdname ggml_backend_buffer_clear
#' @details \code{ggml_backend_buffer_is_host()}: 
#' Returns: Logical indicating if buffer is in host memory
#' @export
ggml_backend_buffer_is_host <- function(buffer) {
 .Call("R_ggml_backend_buffer_is_host", buffer)
}

# ============================================================================
# Direct Backend Initialization
# ============================================================================

#' Backend Initialization
#' @param name Backend name (e.g. "CPU", "Vulkan")
#' @param params Optional parameters string
#' @return External pointer to backend, or NULL on failure
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_init_by_name <- function(name, params = NULL) {
 .Call("R_ggml_backend_init_by_name", as.character(name), params)
}

#' @rdname ggml_backend_init_by_name
#' @details \code{ggml_backend_init_by_type()}: 
#' Returns: External pointer to backend, or NULL on failure
#' @param type Device type constant
#' @export
ggml_backend_init_by_type <- function(type, params = NULL) {
 .Call("R_ggml_backend_init_by_type", as.integer(type), params)
}

#' @rdname ggml_backend_init_by_name
#' @details \code{ggml_backend_init_best()}: 
#' Returns: External pointer to backend (GPU if available, otherwise CPU)
#' @export
ggml_backend_init_best <- function() {
 .Call("R_ggml_backend_init_best")
}

#' Synchronize backend
#' @param backend External pointer to backend
#' @return NULL invisibly
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_synchronize <- function(backend) {
 invisible(.Call("R_ggml_backend_synchronize", backend))
}

#' Get device from backend
#' @param backend External pointer to backend
#' @return External pointer to device
#' @export
#' @seealso \link{ggmlR-backend}
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
#' @seealso \link{ggmlR-backend}
#' @examples
#' \donttest{
#' cpu <- ggml_backend_cpu_init()
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' b <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, b)
#' ggml_set_f32(a, rnorm(100))
#' # Start async computation
#' status <- ggml_backend_graph_compute_async(cpu, graph)
#' # Do other work while computation runs...
#' ggml_backend_synchronize(cpu)
#' ggml_backend_free(cpu)
#' ggml_free(ctx)
#' }
ggml_backend_graph_compute_async <- function(backend, graph) {
  .Call("R_ggml_backend_graph_compute_async", backend, graph)
}

# ============================================================================
# Multi-buffer Operations
# ============================================================================

#' Backend Multi-buffers
#'
#' Creates a buffer that combines multiple backend buffers into one.
#' Useful for managing memory across different backends.
#'
#' @param buffers List of backend buffer external pointers
#' @return External pointer to multi-buffer
#' @export
#' @seealso \link{ggmlR-backend}
#' @examples
#' \donttest{
#' cpu <- ggml_backend_cpu_init()
#' ctx1 <- ggml_init(1024, no_alloc = TRUE)
#' ctx2 <- ggml_init(2048, no_alloc = TRUE)
#' a <- ggml_new_tensor_1d(ctx1, GGML_TYPE_F32, 10)
#' b <- ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, 20)
#' buf1 <- ggml_backend_alloc_ctx_tensors(ctx1, cpu)
#' buf2 <- ggml_backend_alloc_ctx_tensors(ctx2, cpu)
#' multi <- ggml_backend_multi_buffer_alloc_buffer(list(buf1, buf2))
#' ggml_backend_buffer_free(multi)
#' ggml_backend_free(cpu)
#' ggml_free(ctx1)
#' ggml_free(ctx2)
#' }
ggml_backend_multi_buffer_alloc_buffer <- function(buffers) {
  if (!is.list(buffers)) {
    buffers <- list(buffers)
  }
  .Call("R_ggml_backend_multi_buffer_alloc_buffer", buffers)
}

#' @rdname ggml_backend_multi_buffer_alloc_buffer
#' @details \code{ggml_backend_buffer_is_multi_buffer()}: 
#' Returns: Logical indicating if buffer is a multi-buffer
#' @param buffer External pointer to buffer
#' @export
ggml_backend_buffer_is_multi_buffer <- function(buffer) {
  .Call("R_ggml_backend_buffer_is_multi_buffer", buffer)
}

#' @rdname ggml_backend_multi_buffer_alloc_buffer
#' @details \code{ggml_backend_multi_buffer_set_usage()}: 
#' Returns: NULL invisibly
#' @param usage Usage constant (from ggml_backend_buffer_usage_*)
#' @export
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
#' @seealso \link{ggmlR-backend}
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
#' @seealso \link{ggmlR-backend}
ggml_backend_device_register <- function(device) {
  invisible(.Call("R_ggml_backend_device_register", device))
}

# ============================================================================
# Meta backend (tensor parallelism across multiple devices) — experimental
# ============================================================================

#' Create a Meta Backend Device
#'
#' Creates a "meta" device that wraps multiple "simple" backend devices for
#' tensor parallelism. Each tensor is split across the wrapped devices
#' according to the result of \code{split_fn}, which is called by ggml when
#' weight buffers are allocated.
#'
#' The split function is invoked with two arguments:
#' \describe{
#'   \item{tensor_info}{a named list with fields \code{name} (character),
#'     \code{type} (integer ggml_type enum), \code{ne} (numeric vector of
#'     dimensions), \code{op} (integer op enum), \code{flags} (integer).}
#'   \item{n_devs}{the number of simple devices wrapped by the meta backend.}
#' }
#'
#' It must return a named list with:
#' \describe{
#'   \item{axis}{integer; one of 0..3 to split along a tensor axis,
#'     10 for \code{MIRRORED} (full copy on each device),
#'     11 for \code{PARTIAL} (each device has a partial sum), or
#'     98/99 for \code{NONE}/\code{UNKNOWN}.}
#'   \item{ne}{integer or numeric vector of length \code{n_segments * n_devs}
#'     giving the per-segment, per-device slice size along the split axis.}
#'   \item{n_segments}{integer; usually 1, larger for fused tensors like QKV.}
#' }
#'
#' If \code{split_fn} errors or returns an unparseable result, the meta
#' backend silently falls back to \code{MIRRORED} for that tensor and stops
#' calling the callback (sticky error). This is intentional: a misbehaving
#' callback would otherwise spray errors for every tensor in the model.
#'
#' Note: with a single device this is a degenerate (no-op) configuration —
#' useful for testing but provides no parallelism benefit. The feature is
#' \strong{experimental} and the API may change.
#'
#' @param devs A list of \code{ggml_backend_dev_t} external pointers.
#' @param split_fn A function \code{function(tensor_info, n_devs)} returning
#'   the split state as described above.
#' @param env An environment in which to evaluate \code{split_fn};
#'   defaults to the function's enclosing environment.
#' @return External pointer to the meta \code{ggml_backend_dev_t}.
#' @export
#' @seealso \link{ggmlR-backend}
ggml_backend_meta_device <- function(devs, split_fn, env = environment(split_fn)) {
  stopifnot(is.list(devs), length(devs) >= 1L)
  stopifnot(is.function(split_fn))
  if (is.null(env)) env <- globalenv()

  # The C trampoline treats an errored split_fn as the signal to fall back to
  # MIRRORED (sticky after the first failure). Letting the error propagate
  # works, but a session-level options(error = recover/browser) would hijack
  # it and drop into a debugger prompt mid-callback (devtools::test() sets
  # error = recover). Catch it here and hand the C side an explicit sentinel
  # of class "ggmlR_meta_split_error" instead; the trampoline recognises it
  # and applies the exact same sticky MIRRORED fallback. options() is left
  # untouched so this is CRAN-clean.
  user_split_fn <- split_fn
  split_fn <- function(tensor_info, n_devs) {
    tryCatch(
      user_split_fn(tensor_info, n_devs),
      error = function(e) {
        structure(
          list(message = conditionMessage(e)),
          class = "ggmlR_meta_split_error"
        )
      }
    )
  }

  .Call("R_ggml_backend_meta_device", devs, split_fn, env)
}
