# Device management for ag_* autograd engine
#
# Phase 1: forward pass can run on ggml backend (CPU or Vulkan GPU),
# backward remains R-level (uses .ag_data() to pull values back to CPU).
#
# Design:
#   - .ag_device_state holds singleton backend + a persistent context
#   - ag_param always keeps $data (R matrix) as source-of-truth
#   - $ptr is a handle to backend-allocated tensor memory valid for the
#     current ctx lifetime
#   - with_grad_tape() resets the ctx before each tape so ptrs are fresh
#   - Per-operation eager execution: build single-node graph, compute, read
#
# Allocation strategy:
#   Each call to .ag_alloc_buf() creates / grows the backend buffer as needed.
#   Tensors are allocated via ggml_backend_alloc_ctx_tensors(ctx, backend)
#   called ONCE per with_grad_tape() in .ag_reset_ggml_ctx().
#   New tensors created during ops (intermediate results) get their own
#   small fresh ctx so they don't interfere with the parameter ctx.

# ============================================================================
# Device state singleton
# ============================================================================

.ag_device_state <- new.env(parent = emptyenv())
.ag_device_state$device  <- "cpu"   # "cpu" | "gpu"
.ag_device_state$backend <- NULL    # ggml backend (ext ptr)
.ag_device_state$ctx     <- NULL    # ggml context for params (ext ptr)
.ag_device_state$buffer  <- NULL    # backend buffer for param ctx

# ============================================================================
# Public API
# ============================================================================

#' Set the default compute device for ag_* operations
#'
#' Switches all subsequent \code{ag_tensor} / \code{ag_param} operations to run
#' on the specified device.  Calling \code{ag_device("gpu")} initialises the
#' best available ggml backend (Vulkan, Metal, CUDA, or CPU fallback) the first
#' time it is called.
#'
#' @param device \code{"cpu"} (default) or \code{"gpu"}
#' @return Invisibly the previous device string
#' @export
ag_device <- function(device) {
  device <- match.arg(device, c("cpu", "gpu"))
  prev   <- .ag_device_state$device

  if (device == "gpu" && is.null(.ag_device_state$backend)) {
    .ag_init_gpu_backend()
  }

  .ag_device_state$device <- device
  invisible(prev)
}

#' Return the current default compute device
#'
#' @return \code{"cpu"} or \code{"gpu"}
#' @export
ag_default_device <- function() {
  .ag_device_state$device
}

#' Move a tensor to the specified device
#'
#' Copies an \code{ag_tensor} to the target device, returning a new tensor.
#' The original tensor is not modified.
#'
#' @param tensor An \code{ag_tensor}
#' @param device \code{"cpu"} or \code{"gpu"}
#' @return A new \code{ag_tensor} on the target device (or the original if
#'   already on the target device)
#' @export
ag_to_device <- function(tensor, device) {
  stopifnot(is_ag_tensor(tensor))
  device <- match.arg(device, c("cpu", "gpu"))

  if (device == tensor$device) return(tensor)

  # Pull CPU data from wherever it lives
  data <- .ag_data(tensor)

  out <- ag_tensor(data, device = device)
  out$requires_grad <- tensor$requires_grad
  out
}

# ============================================================================
# Internal helpers
# ============================================================================

# Check whether a tensor lives on GPU
.ag_on_gpu <- function(t) {
  is_ag_tensor(t) && isTRUE(t$device == "gpu")
}

# Initialise the best available GPU backend (called once)
.ag_init_gpu_backend <- function() {
  ggml_backend_load_all()
  backend <- ggml_backend_init_best()
  if (is.null(backend))
    stop("No ggml backend available. Install Vulkan drivers or use device='cpu'.")
  .ag_device_state$backend <- backend
}

# Free old param ctx/buffer and create a fresh ggml context for this tape.
# Called at the start of with_grad_tape() when device="gpu".
.ag_reset_ggml_ctx <- function(size_mb = 128L) {
  if (!is.null(.ag_device_state$ctx)) {
    tryCatch(ggml_free(.ag_device_state$ctx), error = function(e) NULL)
    .ag_device_state$ctx <- NULL
  }
  if (!is.null(.ag_device_state$buffer)) {
    tryCatch(ggml_backend_buffer_free(.ag_device_state$buffer),
             error = function(e) NULL)
    .ag_device_state$buffer <- NULL
  }

  mem_size <- as.integer(size_mb) * 1024L * 1024L
  ctx <- ggml_init(mem_size, no_alloc = TRUE)
  .ag_device_state$ctx <- ctx
}

# Execute a ggml graph for a single result node and return its data as a matrix.
# op_fn(ctx, ptrs) builds the ggml node; inputs is a list of numeric matrices.
.ag_run_op <- function(op_fn, inputs, out_shape, mem_mb = 32L) {
  backend  <- .ag_device_state$backend
  mem_size <- as.integer(mem_mb) * 1024L * 1024L
  ctx      <- ggml_init(mem_size, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  # Create input tensor stubs (no memory yet)
  ptrs <- lapply(inputs, function(m) {
    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrow(m), ncol(m))
  })

  # Build the op node
  node <- op_fn(ctx, ptrs)

  # Allocate all tensors (inputs + node) on the backend in one shot
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)

  # Upload input data
  for (i in seq_along(inputs)) {
    ggml_backend_tensor_set_data(ptrs[[i]], as.numeric(inputs[[i]]))
  }

  # Build graph and compute
  graph <- ggml_build_forward_expand(ctx, node)
  ggml_backend_graph_compute(backend, graph)

  # Download result
  raw <- ggml_backend_tensor_get_data(node)
  matrix(raw, out_shape[1L], out_shape[2L])
}

# Upload an R matrix to a ggml tensor in the global param context.
# The global ctx must already exist (set up by .ag_reset_ggml_ctx).
.ag_r_to_gpu <- function(data) {
  ctx <- .ag_device_state$ctx
  if (is.null(ctx))
    stop("GPU context not initialised. Call ag_device('gpu') and use with_grad_tape().")
  if (is.vector(data) && !is.list(data)) data <- matrix(data, ncol = 1L)
  nr  <- nrow(data)
  nc  <- ncol(data)
  ptr <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nr, nc)
  # Allocate / grow the buffer for the param context
  buf <- ggml_backend_alloc_ctx_tensors(ctx, .ag_device_state$backend)
  if (!is.null(buf)) .ag_device_state$buffer <- buf
  ggml_backend_tensor_set_data(ptr, as.numeric(data))
  ptr
}

# Download data from a ggml tensor pointer to an R matrix.
.ag_gpu_to_r <- function(tensor) {
  ptr   <- tensor$ptr
  shape <- tensor$shape   # [nr, nc] stored at creation time
  raw   <- ggml_backend_tensor_get_data(ptr)
  matrix(raw, nrow = shape[1L], ncol = shape[2L])
}

# Retrieve data from an ag_tensor as an R matrix regardless of device.
# Returns plain numeric/matrix input unchanged.
.ag_data <- function(t) {
  if (!is_ag_tensor(t)) return(t)
  if (isTRUE(t$device == "gpu")) {
    # For GPU tensors that have a ptr, read from backend
    if (!is.null(t$ptr)) return(.ag_gpu_to_r(t))
    # For GPU params that only have $data (ptr freed), return $data
    return(t$data)
  }
  t$data
}
