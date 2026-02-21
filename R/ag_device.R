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
.ag_device_state$dtype   <- "f32"   # "f32" | "f16" | "bf16"
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

#' Set the default floating-point precision for ag_* GPU operations
#'
#' Controls the dtype used when uploading tensors to the ggml backend.
#' \code{"bf16"} halves memory usage vs \code{"f32"} with minimal accuracy loss.
#' Backward pass always uses f32 R matrices regardless of this setting.
#'
#' @param dtype \code{"f32"} (default), \code{"f16"}, or \code{"bf16"}
#' @return Invisibly the previous dtype string
#' @export
ag_dtype <- function(dtype) {
  dtype <- match.arg(dtype, c("f32", "f16", "bf16"))
  prev  <- .ag_device_state$dtype
  .ag_device_state$dtype <- dtype
  invisible(prev)
}

#' Return the current default dtype for GPU operations
#'
#' @return \code{"f32"}, \code{"f16"}, or \code{"bf16"}
#' @export
ag_default_dtype <- function() {
  .ag_device_state$dtype
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

# Map dtype string to GGML_TYPE_* constant
.ag_dtype_to_ggml <- function(dtype) {
  switch(dtype,
    "f32"  = GGML_TYPE_F32,
    "f16"  = GGML_TYPE_F16,
    "bf16" = GGML_TYPE_BF16,
    stop("Unknown dtype: ", dtype, ". Use 'f32', 'f16', or 'bf16'.")
  )
}

# Return the dtype actually used for compute on the current backend.
# Vulkan does not support BF16 — fall back to F16.
.ag_compute_dtype <- function(dtype = .ag_device_state$dtype) {
  if (dtype != "bf16") return(dtype)
  backend <- .ag_device_state$backend
  if (is.null(backend)) return(dtype)
  name <- tryCatch(ggml_backend_name(backend), error = function(e) "")
  if (grepl("^Vulkan", name, ignore.case = TRUE)) "f16" else dtype
}

# Execute a ggml graph for a single result node and return its data as a matrix.
# op_fn(ctx, ptrs) builds the ggml node; inputs is a list of numeric matrices.
# dtype controls the precision of input tensors ("f32", "f16", "bf16").
.ag_run_op <- function(op_fn, inputs, out_shape, mem_mb = 32L,
                       dtype = .ag_device_state$dtype) {
  backend   <- .ag_device_state$backend
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype(dtype))
  mem_size  <- as.integer(mem_mb) * 1024L * 1024L
  ctx       <- ggml_init(mem_size, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  # Create input tensor stubs (no memory yet)
  ptrs <- lapply(inputs, function(m) {
    ggml_new_tensor_2d(ctx, ggml_type, nrow(m), ncol(m))
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

  # Download result (always returns f32 doubles)
  raw <- ggml_backend_tensor_get_data(node)
  matrix(raw, out_shape[1L], out_shape[2L])
}

# ============================================================================
# Per-op GPU helpers (call .ag_run_op with the appropriate ggml function)
# ============================================================================

# A[m,k] %*% B[k,n]  ->  [m,n]
# ggml_mul_mat(ctx, src0[K,M], src1[K,N]) = [M,N]
# So: src0 = t(A) stored as [k,m], src1 = B [k,n]
.ag_gpu_matmul <- function(a_data, b_data) {
  nr_a <- nrow(a_data); nc_a <- ncol(a_data)   # m, k
  nr_b <- nrow(b_data); nc_b <- ncol(b_data)   # k, n
  at_data <- t(a_data)                          # [k, m]
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_mul_mat(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(at_data, b_data),
    out_shape = c(nr_a, nc_b)
  )
}

# ggml_add supports broadcasting: b[m,1] broadcasts to a[m,n], b[1,n] broadcasts to a[m,n]
.ag_gpu_add <- function(a_data, b_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_add(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(a_data, b_data),
    out_shape = dim(a_data)
  )
}

.ag_gpu_sub <- function(a_data, b_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_sub(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(a_data, b_data),
    out_shape = dim(a_data)
  )
}

.ag_gpu_mul <- function(a_data, b_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_mul(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(a_data, b_data),
    out_shape = dim(a_data)
  )
}

.ag_gpu_scale <- function(x_data, scalar) {
  s <- as.double(scalar)
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_scale(ctx, ptrs[[1L]], s),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

.ag_gpu_relu <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_relu(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

.ag_gpu_sigmoid <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_sigmoid(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

.ag_gpu_tanh <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_tanh(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

# ggml_soft_max applies softmax along ne0 = rows in R = each column sums to 1
.ag_gpu_softmax <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_soft_max(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

.ag_gpu_log <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_log(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

.ag_gpu_exp <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_exp(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

.ag_gpu_clamp <- function(x_data, lo, hi) {
  lo <- as.double(lo); hi <- as.double(hi)
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_clamp(ctx, ptrs[[1L]], lo, hi),
    inputs   = list(x_data),
    out_shape = dim(x_data)
  )
}

# ggml_sum returns a 1-element tensor; we wrap it in [1,1]
.ag_gpu_sum_all <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_sum(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = c(1L, 1L)
  )
}

.ag_gpu_mean_all <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_mean(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = c(1L, 1L)
  )
}

# ag_sum(dim=2) = colSums: ggml_sum_rows(a[m,n]) -> [1,n]
# ggml-vulkan pipeline selection requires src0->type == F32 (ggml-vulkan.cpp:8792).
# f16 shader exists but is unreachable via current C++ dispatch — CPU fallback required.
.ag_gpu_sum_cols <- function(x_data) {
  if (.ag_device_state$dtype != "f32") {
    return(matrix(colSums(x_data), 1L, ncol(x_data)))
  }
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_sum_rows(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = c(1L, ncol(x_data))
  )
}

# ag_sum(dim=1) = rowSums: CPU fallback (Vulkan transpose+sum_rows not supported).
.ag_gpu_sum_rows <- function(x_data) {
  matrix(rowSums(x_data), nrow = nrow(x_data), ncol = 1L)
}

# ag_mean(dim=2) = colMeans = colSums / nrow
# Same f16 restriction as sum_cols — CPU fallback for non-f32.
.ag_gpu_mean_cols <- function(x_data) {
  nr <- nrow(x_data)
  if (.ag_device_state$dtype != "f32") {
    return(matrix(colMeans(x_data), 1L, ncol(x_data)))
  }
  .ag_run_op(
    op_fn    = function(ctx, ptrs) {
      ggml_scale(ctx, ggml_sum_rows(ctx, ptrs[[1L]]), 1.0 / nr)
    },
    inputs   = list(x_data),
    out_shape = c(1L, ncol(x_data))
  )
}

# ag_mean(dim=1) = rowMeans: CPU fallback.
.ag_gpu_mean_rows <- function(x_data) {
  matrix(rowMeans(x_data), nrow = nrow(x_data), ncol = 1L)
}

# ag_pow(x, p) = x^p
# Special cases: p=2 -> ggml_sqr, p=0.5 -> ggml_sqrt, general -> exp(p*log(x))
.ag_gpu_pow <- function(x_data, p) {
  if (p == 2) {
    .ag_run_op(
      op_fn    = function(ctx, ptrs) ggml_sqr(ctx, ptrs[[1L]]),
      inputs   = list(x_data),
      out_shape = dim(x_data)
    )
  } else if (p == 0.5) {
    .ag_run_op(
      op_fn    = function(ctx, ptrs) ggml_sqrt(ctx, ptrs[[1L]]),
      inputs   = list(x_data),
      out_shape = dim(x_data)
    )
  } else {
    s <- as.double(p)
    .ag_run_op(
      op_fn    = function(ctx, ptrs)
                   ggml_exp(ctx, ggml_scale(ctx, ggml_log(ctx, ptrs[[1L]]), s)),
      inputs   = list(x_data),
      out_shape = dim(x_data)
    )
  }
}

# ggml_transpose returns a view; ggml_cont makes it contiguous.
# Result shape: [ncol(x), nrow(x)]
.ag_gpu_transpose <- function(x_data) {
  out_shape <- c(ncol(x_data), nrow(x_data))
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_cont(ctx, ggml_transpose(ctx, ptrs[[1L]])),
    inputs   = list(x_data),
    out_shape = out_shape
  )
}

# Reshape: ggml_reshape_2d + ggml_cont
.ag_gpu_reshape <- function(x_data, new_nrow, new_ncol) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs)
                 ggml_cont(ctx, ggml_reshape_2d(ctx, ptrs[[1L]],
                                                as.integer(new_nrow),
                                                as.integer(new_ncol))),
    inputs   = list(x_data),
    out_shape = c(new_nrow, new_ncol)
  )
}

# ============================================================================
# Upload an R matrix to a ggml tensor in the global param context.
# The global ctx must already exist (set up by .ag_reset_ggml_ctx).
.ag_r_to_gpu <- function(data, dtype = .ag_device_state$dtype) {
  ctx <- .ag_device_state$ctx
  if (is.null(ctx))
    stop("GPU context not initialised. Call ag_device('gpu') and use with_grad_tape().")
  if (is.vector(data) && !is.list(data)) data <- matrix(data, ncol = 1L)
  nr        <- nrow(data)
  nc        <- ncol(data)
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype(dtype))
  ptr       <- ggml_new_tensor_2d(ctx, ggml_type, nr, nc)
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
