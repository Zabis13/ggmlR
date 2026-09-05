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
.ag_device_state$ctx     <- NULL    # current ggml context for resident tensors
.ag_device_state$buffer  <- NULL    # last buffer allocated (legacy slot)

# Residency bookkeeping.
#
# contexts / buffers are LISTS, not single slots. Three facts from the vendored
# ggml shape this:
#
#   * ggml_backend_alloc_ctx_tensors() allocates only the tensors of a context
#     that have no data yet and returns a NEW buffer covering exactly those, or
#     NULL when there was nothing left to do (ggml-alloc.c:1147-1152, 1213).
#     Keeping a single $buffer slot drops the reference to every earlier buffer,
#     which leaks device memory.
#   * A context that runs out of descriptor space does not report failure:
#     ggml_new_object() returns NULL and ggml_new_tensor_impl() turns that into
#     GGML_ASSERT(obj_new) (ggml-context.c:514,584) — an abort, not an error. So
#     overflow is predicted before a tensor is created, and the allocator rolls
#     over into a fresh context. Earlier contexts stay alive, so tensors already
#     handed out remain valid.
#   * Vulkan caps the NUMBER of allocations a device grants
#     (maxMemoryAllocationCount, commonly 4096) and the ggml Vulkan backend does
#     not track it — it only guards allocation SIZE
#     (ggml-vulkan-shaders.cpp:1994). Allocating per tensor would therefore hit
#     VK_ERROR_TOO_MANY_OBJECTS on a long tape, so tensors accumulate in the
#     current context and are allocated in one batch per context.
#
# ctx_gen is a generation counter. Resident tensors record the generation they
# were allocated under; .ag_residency_reset() frees everything and bumps it, so
# a pointer left behind on a longer-lived ag_tensor is recognised as stale
# instead of being read as freed memory.
.ag_device_state$contexts  <- list()  # all live contexts, oldest first
.ag_device_state$buffers   <- list()  # all live backend buffers
.ag_device_state$ctx_gen   <- 0L      # bumped on every reset
.ag_device_state$ctx_mb    <- 128L    # size of each context, in MB
.ag_device_state$mem_limit <- Inf     # tape budget in bytes (Inf = unlimited)

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

  # Switching to the CPU releases the GPU backend, rather than only recording
  # the choice.
  #
  # Leaving it in place made the device state leak across a session: anything
  # reading $backend directly -- gpu_linalg, sc_umap, ag_flash_attention --
  # would keep computing on Vulkan after the caller asked for the CPU. Under
  # test_dir() that meant a file running after any GPU test silently ran on the
  # GPU too, and the only visible symptom was f16-level disagreement with a
  # double-precision reference. Nothing errors; the numbers are just quietly
  # from a different device than the one requested.
  #
  # The residency reset is what makes this safe: contexts and buffers allocated
  # from the GPU backend outlive it otherwise, and freeing the backend under
  # them is a use-after-free. .ag_residency_reset() frees both and bumps
  # ctx_gen, so any ag_tensor still holding a $ptr is detected as stale rather
  # than read back as garbage.
  #
  # The CPU path itself needs no backend at all -- ag_* computes 2D ops in R --
  # so dropping it costs nothing until the next GPU op re-creates it.
  if (device == "cpu" && !is.null(.ag_device_state$backend)) {
    .ag_residency_reset()
    tryCatch(ggml_backend_free(.ag_device_state$backend),
             error = function(e) NULL)
    .ag_device_state$backend <- NULL
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

# ---------------------------------------------------------------------------
# Residency: contexts, buffers, generation, memory ledger
# ---------------------------------------------------------------------------

# Smallest context worth creating, in MB. The growth tests use it to force a
# rollover cheaply.
.ag_min_ctx_mb <- function() 1L

# How many more tensor descriptors fit in `ctx` before it overflows.
#
# With no_alloc = TRUE a tensor costs only its descriptor: ggml_new_tensor_impl
# leaves obj_alloc_size at 0 for non-view tensors in a no_alloc context
# (ggml-context.c:576-579), so the DATA size does not come out of the context.
# The cost per tensor is ggml_tensor_overhead(), and ggml_new_object() wants one
# further GGML_OBJECT_SIZE of slack on top of each request
# (ggml-context.c:514) — hence the extra slot held back here.
.ag_ctx_capacity <- function(ctx) {
  if (is.null(ctx)) return(0L)
  per   <- as.double(ggml_tensor_overhead())
  total <- as.double(ggml_get_mem_size(ctx))
  used  <- as.double(ggml_used_mem(ctx))
  free  <- total - used - per
  if (!is.finite(free) || free <= 0) return(0L)
  as.integer(free %/% per)
}

# Allocate every tensor in `ctx` that still lacks memory, in ONE buffer, and
# retain that buffer so it can be freed later.
#
# Cheap to repeat: tensors that already have data are skipped (ggml-alloc.c:1185)
# and the call returns NULL when there is nothing left to allocate. The binding
# has to tell that case apart from a real allocation failure, since ggml-alloc
# returns NULL for both (r_interface_graph.c, R_ggml_backend_alloc_ctx_tensors).
.ag_ctx_flush <- function(ctx = .ag_device_state$ctx) {
  if (is.null(ctx)) return(invisible(NULL))
  if (is.null(.ag_device_state$backend))
    stop("ggmlR: no compute backend is initialised. ag_device(\"cpu\") only ",
         "records the choice; call .ag_ensure_backend() (or ag_device(\"gpu\")) ",
         "before running device ops.", call. = FALSE)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, .ag_device_state$backend)
  if (is.null(buf)) return(invisible(NULL))
  .ag_device_state$buffers <- c(.ag_device_state$buffers, list(buf))
  .ag_device_state$buffer  <- buf        # legacy slot: last buffer allocated
  invisible(buf)
}

# Free every context and buffer, then start a new generation.
#
# Bumping ctx_gen is what makes stale pointers detectable: an ag_tensor that
# outlives this call still holds a $ptr into memory that has just been freed.
.ag_residency_reset <- function(size_mb = NULL) {
  # Rescue anything that must outlive the buffers before they go.
  #
  # A resident $grad (component 3) is a handle into a buffer freed just below.
  # The tensor holding it is an ordinary R object that survives the reset, so
  # without this its gradient would become a pointer into released memory --
  # caught loudly by the generation check, but only at the next read, far from
  # here. Tensor VALUES need no such rescue: .ag_data() keeps a host copy or
  # can refuse, whereas a gradient has no fallback and no second source.
  .ag_materialise_pending_grads()

  for (buf in .ag_device_state$buffers) {
    tryCatch(ggml_backend_buffer_free(buf), error = function(e) NULL)
  }
  for (ctx in .ag_device_state$contexts) {
    tryCatch(ggml_free(ctx), error = function(e) NULL)
  }
  .ag_device_state$buffers  <- list()
  .ag_device_state$contexts <- list()
  .ag_device_state$ctx      <- NULL
  .ag_device_state$buffer   <- NULL
  .ag_device_state$ctx_gen  <- .ag_device_state$ctx_gen + 1L
  if (!is.null(size_mb)) .ag_device_state$ctx_mb <- as.integer(size_mb)
  invisible(.ag_device_state$ctx_gen)
}

# Backwards-compatible name: with_grad_tape() calls this at the start of a tape.
.ag_reset_ggml_ctx <- function(size_mb = 128L) {
  .ag_residency_reset(size_mb = size_mb)
  .ag_ctx_ensure()
}

# Make sure a context with room for `n` more tensors is current.
#
# The context being retired is flushed on the way out: it may still hold
# tensors that were never backed by memory, and nothing will return to it once
# a new context is current. A rollover does NOT free the old context — tensors
# already handed out stay valid — so ctx_gen is left untouched. Only a reset
# invalidates pointers.
.ag_ctx_ensure <- function(n = 1L) {
  ctx <- .ag_device_state$ctx
  if (!is.null(ctx) && .ag_ctx_capacity(ctx) >= n) return(ctx)

  if (!is.null(ctx)) .ag_ctx_flush(ctx)

  # Size the new context so the request fits even when it exceeds the default.
  per     <- as.double(ggml_tensor_overhead())
  need_mb <- ceiling((per * (as.double(n) + 1)) / (1024 * 1024))
  mb      <- max(as.double(.ag_device_state$ctx_mb), need_mb,
                 as.double(.ag_min_ctx_mb()))
  ctx     <- ggml_init(mb * 1024 * 1024, no_alloc = TRUE)
  if (is.null(ctx)) stop("ggmlR: failed to create a ggml context for the tape.")

  .ag_device_state$contexts <- c(.ag_device_state$contexts, list(ctx))
  .ag_device_state$ctx      <- ctx
  ctx
}

# Current tape memory usage.
#
# ctx_bytes/ctx_used cover descriptors (host side); buffer_bytes is the device
# memory actually backing resident tensors.
.ag_tape_mem <- function() {
  ctx_bytes <- sum(vapply(.ag_device_state$contexts,
                          function(c) as.double(ggml_get_mem_size(c)), numeric(1)),
                   0)
  ctx_used  <- sum(vapply(.ag_device_state$contexts,
                          function(c) as.double(ggml_used_mem(c)), numeric(1)),
                   0)
  buf_bytes <- sum(vapply(.ag_device_state$buffers,
                          function(b) as.double(ggml_backend_buffer_get_size(b)),
                          numeric(1)),
                   0)
  list(ctx_bytes    = ctx_bytes,
       ctx_used     = ctx_used,
       buffer_bytes = buf_bytes,
       n_contexts   = length(.ag_device_state$contexts),
       n_buffers    = length(.ag_device_state$buffers))
}

# Get or set the tape memory budget, in bytes. Returns the previous value.
#
# A resident tape has no obvious ceiling: every intermediate stays on the device
# until the tape is reset. Running into the driver's own limit surfaces deep
# inside the backend, so the ledger refuses first, naming the tape.
.ag_tape_mem_limit <- function(bytes = NULL) {
  old <- .ag_device_state$mem_limit
  if (!is.null(bytes)) .ag_device_state$mem_limit <- as.double(bytes)
  invisible(old)
}

# Refuse an allocation that would push the tape past its budget.
.ag_check_mem_budget <- function(extra_bytes) {
  limit <- .ag_device_state$mem_limit
  if (!is.finite(limit)) return(invisible(TRUE))
  used <- .ag_tape_mem()$buffer_bytes
  if (used + extra_bytes > limit) {
    stop(sprintf(
      paste0("ggmlR: autograd tape memory budget exceeded (%.1f MB used + ",
             "%.1f MB requested > %.1f MB limit). Reset the tape, shorten it, ",
             "or raise the limit with .ag_tape_mem_limit()."),
      used / 1024^2, extra_bytes / 1024^2, limit / 1024^2), call. = FALSE)
  }
  invisible(TRUE)
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
  if (grepl("^Vulkan", name, ignore.case = TRUE)) {
    # Industrial telemetry: surface the silent precision downgrade. Logged once
    # per session to avoid flooding the per-op hot path.
    if (!isTRUE(.ag_device_state$bf16_fallback_warned)) {
      message("ggmlR: requested dtype 'bf16' is not supported on the Vulkan backend; ",
              "falling back to 'f16' for compute.")
      .ag_device_state$bf16_fallback_warned <- TRUE
    }
    "f16"
  } else {
    dtype
  }
}

# Size of the throwaway context that holds one op's graph.
#
# ggml_new_graph() is hardcoded to GGML_DEFAULT_GRAPH_SIZE = 8192 nodes and the
# cgraph is allocated inside the context, so the context must be able to hold
# ggml_graph_overhead() no matter how small the op is. Asking ggml for the
# figure keeps this correct if the default or the struct layout changes.
.ag_graph_ctx_bytes <- function() {
  as.double(ggml_graph_overhead()) + 64 * 1024   # + slack for object headers
}

# Execute a ggml graph for a single result node and return its data as a matrix.
# op_fn(ctx, ptrs) builds the ggml node; inputs is a list of numeric matrices.
# dtype controls the precision of input tensors ("f32", "f16", "bf16").
#
# Tensors live in the persistent residency context (.ag_ctx_ensure), not in a
# context built and torn down per call: creating a context, allocating a buffer
# and freeing both on every single op is pure overhead, and the buffer churn is
# what the Vulkan allocation-count cap punishes on a long tape.
#
# The GRAPH is the exception and still gets a throwaway context of its own.
# ggml_new_graph() allocates the cgraph inside the context it is given, sized
# GGML_DEFAULT_GRAPH_SIZE = 8192 nodes -> ~330 KB per call (nodes + leafs +
# hash set, ggml-graph.c:1341). Putting that in the persistent context would
# fill it within a handful of ops, so the graph context is created small, used
# once and freed here. Tensors are unaffected: they were allocated out of the
# residency context and stay valid after this call returns.
#' @param inputs List of operands, each either an R matrix or an \code{ag_handle}
#'   naming a tensor already resident in the context. A handle is used in place
#'   rather than uploaded -- that skip is the whole point of the type.
#' @param resident When TRUE, return an \code{ag_handle} for the result instead
#'   of downloading it. The caller then owns the decision of when the numbers
#'   come back, which is what lets a chain of ops cost one download rather than
#'   one per operation.
#' @noRd
.ag_run_op <- function(op_fn, inputs, out_shape, mem_mb = 32L,
                       dtype = .ag_device_state$dtype, node_hook = NULL,
                       resident = FALSE) {
  backend   <- .ag_device_state$backend
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype(dtype))

  # Per-stage timing, off by default: one field read when it is (R/ag_fwd_profile.R).
  fprof <- isTRUE(.ag_fwd$prof)
  facc  <- NULL
  ftk   <- if (fprof) Sys.time() else NULL
  fstage <- function(name) {
    if (!fprof) return(invisible(NULL))
    now <- Sys.time()
    facc[[name]] <<- as.numeric(difftime(now, ftk, units = "secs")) * 1000
    ftk <<- now
    invisible(NULL)
  }

  # A handle from a dead generation would be a pointer into freed memory. Fail
  # here, naming the problem, rather than letting ggml read whatever is there.
  for (i in seq_along(inputs)) {
    if (.ag_is_handle(inputs[[i]]) && !.ag_handle_live(inputs[[i]]))
      stop("ggmlR: operand ", i, " is a device handle from generation ",
           inputs[[i]]$gen %||% NA, ", but the tape has been reset since ",
           "(generation ", .ag_device_state$ctx_gen, ").", call. = FALSE)
  }

  # Budget check before anything is created: past this point a failure would
  # leave half-built tensors sitting in the shared persistent context.
  #
  # Handles are already allocated, so only the uploads and the result are new
  # memory -- counting a handle again would refuse work that fits.
  tsize <- as.double(ggml_type_size(ggml_type))
  new_elems <- sum(vapply(inputs, function(m) {
    if (.ag_is_handle(m)) 0 else as.double(nrow(m)) * as.double(ncol(m))
  }, numeric(1)))
  .ag_check_mem_budget((new_elems + prod(as.double(out_shape))) * tsize)

  # Reserve descriptor room for the inputs plus the op's own nodes. An op may
  # build more than one node (e.g. reshape + cont), so leave slack: overflowing
  # a context aborts R inside ggml_new_tensor_impl rather than returning.
  ctx <- .ag_ctx_ensure(length(inputs) + 4L)
  fstage("ctx")

  # A handle contributes its existing pointer; a matrix gets a fresh tensor
  # that is filled below.
  ptrs <- lapply(inputs, function(m) {
    if (.ag_is_handle(m)) m$ptr
    else ggml_new_tensor_2d(ctx, ggml_type, nrow(m), ncol(m))
  })
  fstage("create")

  # Build the op node
  node <- op_fn(ctx, ptrs)

  # Optional post-build node tweak (e.g. forcing f32 accumulation precision on a
  # mul_mat node). Runs before allocation/compute so it affects the kernel pick.
  if (!is.null(node_hook)) node_hook(node)

  # Allocate everything this op just added (inputs + nodes) in one buffer. The
  # flush is a no-op for tensors that already have memory, so earlier residents
  # of the context are not touched.
  .ag_ctx_flush(ctx)
  fstage("flush")

  # Upload input data -- but only for operands that are not already there. This
  # skip is the point of the handle type: a weight reused across a chain is
  # sent once instead of once per operation, which measured at 10-20% of a
  # backward pass on its own.
  for (i in seq_along(inputs)) {
    if (.ag_is_handle(inputs[[i]])) next
    ggml_backend_tensor_set_data(ptrs[[i]], as.numeric(inputs[[i]]))
  }
  fstage("upload")

  # Graph-only context: freed on exit, unlike the tensors above.
  #
  # Safe because ownership runs one way. ggml_new_graph_custom() puts the
  # cgraph -- nodes[], leafs[] and the hash set -- inside THIS context's
  # mem_buffer (ggml-graph.c:1366), and those arrays hold POINTERS to tensors
  # that live in the residency context, with their data in a backend buffer
  # owned by .ag_device_state$buffers. ggml_free() releases only ctx->mem_buffer
  # (ggml-context.c:443-449), so dropping the graph cannot reach the tensors.
  # The reverse would be a bug: a graph outliving .ag_residency_reset() would
  # keep dangling pointers to freed tensors. It cannot happen here -- `graph` is
  # local and this context dies with the call -- and $ctx_gen would NOT catch
  # it, since that guards ag_tensors against a freed tensor context, not graphs.
  ctx_graph <- ggml_init(.ag_graph_ctx_bytes(), no_alloc = TRUE)
  if (is.null(ctx_graph))
    stop("ggmlR: failed to create a ggml context for the op graph.")
  on.exit(ggml_free(ctx_graph), add = TRUE)

  graph <- ggml_build_forward_expand(ctx_graph, node)
  fstage("graph")
  ggml_backend_graph_compute(backend, graph)
  fstage("compute")

  # Resident: hand back a name for the result and let the caller decide when
  # (or whether) it comes off the device.
  #
  # Safe with respect to the graph context freed on exit above: ownership runs
  # one way. `node` was allocated from the residency context and its data lives
  # in a backend buffer that .ag_device_state owns, so dropping the graph
  # cannot reach it -- the same argument the comment above makes for tensors.
  if (isTRUE(resident)) {
    if (fprof) .ag_fwd_prof_record(facc)
    return(.ag_handle(node, out_shape))
  }

  # Download result (always returns f32 doubles)
  raw <- ggml_backend_tensor_get_data(node)
  out <- matrix(raw, out_shape[1L], out_shape[2L])
  if (fprof) {
    fstage("download")
    .ag_fwd_prof_record(facc)
  }
  out
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

  # ggml_mul_mat(a, b) needs the shared dimension in ne[0] of BOTH operands
  # (ggml.h:1462-1464), and an R matrix [m,k] lands as ne0=m, ne1=k -- so `a`
  # has to be transposed for k to reach ne[0].
  #
  # The transpose stays in R. Doing it in the graph instead (ggml_transpose +
  # ggml_cont) was tried and measured: identical time (42.0 vs 42.3 ms on a
  # chain of 8 matmuls at 512x512, i.e. noise) and identical accuracy. ggml_cont
  # copies on the device exactly as t() copies on the host, so nothing is saved
  # -- only two extra graph nodes and their descriptor space. Do not "optimise"
  # this again without a measurement: scratchpad probe_transpose.R compares the
  # two directly.
  at_data <- t(a_data)                          # [k, m]
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_mul_mat(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(at_data, b_data),
    out_shape = c(nr_a, nc_b)
  )
}

# A %*% B with f32 accumulation forced on the matmul node. The Vulkan backend
# accumulates mul_mat in f16 by default (~2.7e-4 relative error), which is fine
# for neural-net layers but corrupts precision-sensitive downstream maths — e.g.
# a Gram matrix whose ||x_i||^2 + ||x_j||^2 - 2 G[i,j] distances feed kNN, where
# the f16 noise reorders nearest neighbours. GGML_PREC_F32 selects the f32 kernel.
GGML_PREC_F32 <- 10L
.ag_gpu_matmul_f32 <- function(a_data, b_data) {
  nr_a <- nrow(a_data); nc_b <- ncol(b_data)
  # Transpose in R -- see .ag_gpu_matmul above for why the graph version was
  # measured and rejected.
  at_data <- t(a_data)
  .ag_run_op(
    op_fn     = function(ctx, ptrs) ggml_mul_mat(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs    = list(at_data, b_data),
    out_shape = c(nr_a, nc_b),
    node_hook = function(node)
      .Call("R_ggml_mul_mat_set_prec", node, GGML_PREC_F32, PACKAGE = "ggmlR")
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
# Vulkan supports f32 (pipeline[0]) and f16 (pipeline[1]).
.ag_gpu_sum_cols <- function(x_data) {
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
# ggml_sum_rows supports f32 and f16; ggml_scale also supports both.
.ag_gpu_mean_cols <- function(x_data) {
  nr <- nrow(x_data)
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
  if (is.null(.ag_device_state$backend))
    stop("GPU backend not initialised. Call ag_device('gpu') first.")
  if (is.vector(data) && !is.list(data)) data <- matrix(data, ncol = 1L)
  nr        <- nrow(data)
  nc        <- ncol(data)
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype(dtype))

  # Check the budget first: past this point a failure would leave a half-built
  # tensor sitting in the context.
  .ag_check_mem_budget(as.double(nr) * as.double(nc) *
                       as.double(ggml_type_size(ggml_type)))

  # Roll over to a fresh context if this descriptor would not fit. Creating the
  # tensor first would abort R rather than return an error.
  ctx <- .ag_ctx_ensure(1L)
  ptr <- ggml_new_tensor_2d(ctx, ggml_type, nr, nc)

  # This tensor has to be readable when the call returns, so its memory must
  # exist now. The flush covers every unallocated tensor of the context at once,
  # so tensors created back-to-back before any upload share a single buffer.
  .ag_ctx_flush(ctx)

  ggml_backend_tensor_set_data(ptr, as.numeric(data))
  ptr
}

# Create resident tensors for several matrices under ONE allocation.
#
# The single-tensor path has to allocate on every call, because it must return
# something readable. Callers that upload a group of tensors (a layer's
# parameters, a tape's inputs) should use this instead: descriptors are created
# first, allocated in one batch, and only then filled — one buffer for the whole
# group rather than one per tensor.
.ag_r_to_gpu_batch <- function(mats, dtype = .ag_device_state$dtype) {
  if (is.null(.ag_device_state$backend))
    stop("GPU backend not initialised. Call ag_device('gpu') first.")
  if (!length(mats)) return(list())

  mats <- lapply(mats, function(m) {
    if (is.vector(m) && !is.list(m)) matrix(m, ncol = 1L) else m
  })
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype(dtype))
  tsize     <- as.double(ggml_type_size(ggml_type))
  .ag_check_mem_budget(sum(vapply(mats, function(m)
    as.double(nrow(m)) * as.double(ncol(m)) * tsize, numeric(1))))

  # All descriptors must land in one context for a single flush to cover them.
  ctx  <- .ag_ctx_ensure(length(mats))
  ptrs <- lapply(mats, function(m)
    ggml_new_tensor_2d(ctx, ggml_type, nrow(m), ncol(m)))

  .ag_ctx_flush(ctx)

  for (i in seq_along(mats)) {
    ggml_backend_tensor_set_data(ptrs[[i]], as.numeric(mats[[i]]))
  }
  ptrs
}

# Download data from a ggml tensor pointer to an R matrix.
.ag_gpu_to_r <- function(tensor) {
  ptr   <- tensor$ptr
  shape <- tensor$shape   # [nr, nc] stored at creation time
  raw   <- ggml_backend_tensor_get_data(ptr)
  matrix(raw, nrow = shape[1L], ncol = shape[2L])
}

# ---------------------------------------------------------------------------
# Value access. The three functions below are the whole supported surface for
# reading and writing an ag_tensor's value; see inst/docs/ag_data_contract.md.
# Reading $data directly gets NULL for a resident tensor, and assigning to it
# leaves the device copy stale -- both fail silently, which is why the contract
# exists.
# ---------------------------------------------------------------------------

# Read: return the value as an R matrix, materialising from the device if that
# is where it lives. Plain numeric/matrix input passes through unchanged.
#
# Read-only. The result is a copy for a resident tensor, so writing to it
# changes nothing -- use .ag_data_mut() + .ag_data_set() to modify a value.
.ag_data <- function(t) {
  if (!is_ag_tensor(t)) return(t)
  if (isTRUE(t$device == "gpu")) {
    # A pointer is readable only while its context is alive. Contexts are freed
    # by .ag_residency_reset(), which bumps ctx_gen; an ag_tensor is an ordinary
    # R object and can easily outlive that, leaving a pointer into freed memory.
    # Reading it would return plausible-looking garbage rather than fail, so the
    # generation is checked first and the retained R matrix used instead.
    if (!is.null(t$ptr) && .ag_ptr_is_live(t)) {
      # Materialise once per generation: a download is a device->host copy, and
      # the tape reads the same tensor repeatedly (17 967 reads from ag_matmul
      # alone across the test suite). The cache is dropped whenever the value
      # or the pointer changes, so it cannot go stale behind the value.
      if (!is.null(t$data) && identical(t$data_gen, t$ctx_gen)) return(t$data)
      val         <- .ag_gpu_to_r(t)
      t$data      <- val
      t$data_gen  <- t$ctx_gen
      return(val)
    }
    if (!is.null(t$data)) return(t$data)
    if (!is.null(t$ptr))
      stop("ggmlR: this tensor's GPU buffer was freed by a tape reset (stale ",
           "pointer, generation ", t$ctx_gen %||% NA, " < ",
           .ag_device_state$ctx_gen, ") and it has no CPU copy to fall back on.",
           call. = FALSE)
    return(NULL)
  }
  t$data
}

# Read for modification: materialise and hand back a writable copy.
#
# Separate from .ag_data() so that read-modify-write shows up as such in the
# code. The value is not observed until .ag_data_set() is called -- mutating
# the returned matrix alone changes nothing for a resident tensor.
.ag_data_mut <- function(t) {
  if (!is_ag_tensor(t)) return(t)
  val <- .ag_data(t)
  if (is.null(val))
    stop("ggmlR: cannot modify a tensor whose value is unavailable ",
         "(no CPU copy and no live device pointer).", call. = FALSE)
  val
}

# Write: install a new value.
#
# The only supported way to change a tensor's value. Any device residency is
# dropped, so the next read re-uploads rather than returning the old buffer:
# keeping the pointer would leave the device holding the previous value with
# nothing to signal the disagreement.
.ag_data_set <- function(t, value) {
  if (!is_ag_tensor(t))
    stop("ggmlR: .ag_data_set() expects an ag_tensor.", call. = FALSE)
  if (is.vector(value) && !is.list(value)) value <- matrix(value, ncol = 1L)
  t$data     <- value
  t$data_gen <- NULL
  if (!is.null(t$ptr)) {
    t$ptr     <- NULL
    t$ctx_gen <- NULL
    t$shape   <- NULL
  }
  invisible(t)
}

# TRUE when a resident tensor's pointer still belongs to the current generation.
# Tensors created before generations were tracked carry no ctx_gen; they count
# as stale, since nothing proves their context survived.
.ag_ptr_is_live <- function(t) {
  identical(t$ctx_gen, .ag_device_state$ctx_gen)
}

# ---------------------------------------------------------------------------
# Component 2 of the resident contract: a tensor whose value starts on the
# device.
#
# .ag_data() has always been able to READ from a device pointer; what was
# missing is a way to CREATE a tensor that has one and no host copy. Without
# this, an operation could return a handle (component 1) but the moment it
# became an ag_tensor the value had to be downloaded -- the round trip the
# redesign exists to remove.
#
# The three fields set here are exactly the ones .ag_data() consults, so a
# tensor built this way is indistinguishable from one that became resident
# later. $data stays NULL, which the contract defines as "not materialised",
# never as "empty" (rule 4).
# ---------------------------------------------------------------------------

# Install a device handle as a tensor's value, dropping any host copy.
#
# The host copy MUST go: keeping it would leave two values with no way to tell
# which is current, which is the failure the contract exists to prevent. The
# next .ag_data() re-materialises from the device and caches under $data_gen.
.ag_data_set_handle <- function(t, h) {
  if (!is_ag_tensor(t))
    stop("ggmlR: .ag_data_set_handle() expects an ag_tensor.", call. = FALSE)
  if (!.ag_is_handle(h))
    stop("ggmlR: .ag_data_set_handle() expects an ag_handle.", call. = FALSE)
  if (!.ag_handle_live(h))
    stop("ggmlR: refusing to install a device handle from generation ",
         h$gen %||% NA, " (current is ", .ag_device_state$ctx_gen, ").",
         call. = FALSE)
  t$ptr      <- h$ptr
  t$shape    <- h$shape
  t$ctx_gen  <- h$gen
  t$data     <- NULL     # not materialised -- see rule 4
  t$data_gen <- NULL
  invisible(t)
}

# Build an ag_tensor directly from a device handle: the constructor for a value
# that never touched the host.
#
# Used where an op produced a resident result and the caller wants a tensor
# rather than a handle. The device is "gpu" by construction -- a handle cannot
# exist otherwise.
.ag_tensor_from_handle <- function(h, dtype = .ag_device_state$dtype) {
  t <- ag_tensor(matrix(numeric(0), 0L, 0L), device = "gpu", dtype = dtype)
  .ag_data_set_handle(t, h)
  t
}

# ---------------------------------------------------------------------------
# Rescuing resident gradients across a tape reset.
#
# A $grad holding a device handle is the one piece of state with no host
# fallback: a tensor's value can always be re-read or re-uploaded, but a
# gradient exists only in the buffer the backward pass wrote it to. When
# with_grad_tape() resets the contexts at the start of the next pass, that
# buffer goes -- so the handles have to be turned into matrices first.
#
# The register is a list of the tensors currently holding one. It is a plain
# list of environments, so registering costs nothing and a tensor that is
# garbage-collected simply never comes up again (its entry keeps it alive until
# the next reset, which is one pass at most).
# ---------------------------------------------------------------------------

.ag_device_state$pending_grads <- list()

# Note that `t` now holds a resident gradient, so the next reset materialises it.
#
# Keyed by tensor id: a training loop that never calls zero_grad() would
# otherwise append the same tensors every pass, and the reset would materialise
# each of them as many times as it was registered.
.ag_register_pending_grad <- function(t) {
  key <- as.character(t$id)
  .ag_device_state$pending_grads[[key]] <- t
  invisible(NULL)
}

# Drop the register without materialising anything.
#
# Called by zero_grad(): the gradients have been consumed by the optimizer and
# are about to be discarded, so rescuing them at the next reset would be a
# download of numbers nobody will read. Measured at 14.5 ms of a 78 ms step on
# a 4-layer 1024-wide model -- rescuing four gradients at ~9 ms each, all of
# them already used.
.ag_forget_pending_grads <- function() {
  .ag_device_state$pending_grads <- list()
  invisible(NULL)
}

# Turn every registered resident gradient into an R matrix, then forget them.
#
# Called from .ag_residency_reset() before the buffers are freed. Reading is
# wrapped: a handle whose generation has already moved on cannot be rescued,
# and losing one gradient must not stop the reset itself.
.ag_materialise_pending_grads <- function() {
  pend <- .ag_device_state$pending_grads
  if (length(pend) == 0L) return(invisible(NULL))
  for (t in pend) {
    g <- t$grad
    if (!.ag_is_handle(g)) next
    t$grad <- tryCatch(.ag_handle_to_r(g), error = function(e) NULL)
  }
  .ag_device_state$pending_grads <- list()
  invisible(NULL)
}

# The handle naming a tensor's value, or NULL when it has none live.
#
# The inverse of .ag_data_set_handle: lets an operation pass a resident operand
# straight through to .ag_run_op instead of materialising it. Returns NULL
# rather than erroring for a host-side tensor, so callers can write
#   op(.ag_handle_of(x) %||% .ag_data(x), ...)
# and get residency where it exists without a special case where it does not.
.ag_handle_of <- function(t) {
  if (!is_ag_tensor(t)) return(NULL)
  if (is.null(t$ptr) || !.ag_ptr_is_live(t)) return(NULL)
  .ag_handle(t$ptr, t$shape)
}
