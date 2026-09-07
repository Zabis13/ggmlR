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
#
# ---------------------------------------------------------------------------
# Two lifetimes, not one.
#
# Residency is split into two pools, because the tensors that need to stay on
# the device have two different lifetimes and one pool cannot serve both:
#
#   "pass"       graph nodes, activations, intermediates. Dead once the step
#                is over, and freed wholesale at the next with_grad_tape().
#   "persistent" weights, optimizer moments, gradient accumulators. Must
#                survive the tape reset, or they are re-uploaded every step.
#
# Why this had to change. with_grad_tape() starts each tape with
# .ag_reset_ggml_ctx(), which freed EVERY context and buffer. So a weight left
# on the device was guaranteed to be destroyed once per training step -- weight
# residency and per-pass freeing were mutually exclusive, and measurement put
# the cost of that at 74-98% of a forward pass spent re-uploading weights
# (inst/scripts/proto_ag_weight_cache.R). Keeping the pass pool freeable while
# the persistent pool survives is what makes resident weights possible at all.
#
# Each pool carries its own generation counter, so a stale pointer is still
# caught: freeing the pass pool cannot make a persistent handle look stale, and
# vice versa. A handle records which pool it came from and is checked against
# that pool's counter (see .ag_handle in R/ag_handle.R).
#
# The pass pool keeps the original field names. Every existing caller writes
# and reads $contexts / $buffers / $ctx / $ctx_gen and means the per-pass pool,
# so leaving those names alone keeps this change additive: nothing that does
# not ask for "persistent" behaves differently than before.
.ag_device_state$contexts  <- list()  # pass pool: live contexts, oldest first
.ag_device_state$buffers   <- list()  # pass pool: live backend buffers
.ag_device_state$ctx_gen   <- 0L      # pass pool: bumped on every pass reset
.ag_device_state$ctx_mb    <- 128L    # size of each context, in MB
.ag_device_state$mem_limit <- Inf     # tape budget in bytes (Inf = unlimited)

# Persistent pool. Same three slots, an independent lifetime.
.ag_device_state$p_contexts <- list()
.ag_device_state$p_buffers  <- list()
.ag_device_state$p_ctx      <- NULL
.ag_device_state$p_ctx_gen  <- 0L

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

# ---------------------------------------------------------------------------
# Pool addressing.
#
# The two pools differ only in which state slots they use, so the allocator is
# written once and told which pool to work in. Field names are computed rather
# than branched on, which keeps every code path below identical for both.
# ---------------------------------------------------------------------------

.ag_scopes <- c("pass", "persistent")

# Slot names for a pool: contexts, buffers, current context, generation.
.ag_pool_slots <- function(scope = "pass") {
  scope <- match.arg(scope, .ag_scopes)
  if (scope == "pass")
    list(ctxs = "contexts", bufs = "buffers", cur = "ctx", gen = "ctx_gen")
  else
    list(ctxs = "p_contexts", bufs = "p_buffers", cur = "p_ctx", gen = "p_ctx_gen")
}

# Current generation of a pool. Handles are validated against this.
.ag_scope_gen <- function(scope = "pass") {
  .ag_device_state[[.ag_pool_slots(scope)$gen]]
}

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
.ag_ctx_flush <- function(ctx = .ag_device_state$ctx, scope = "pass") {
  if (is.null(ctx)) return(invisible(NULL))
  if (is.null(.ag_device_state$backend))
    stop("ggmlR: no compute backend is initialised. ag_device(\"cpu\") only ",
         "records the choice; call .ag_ensure_backend() (or ag_device(\"gpu\")) ",
         "before running device ops.", call. = FALSE)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, .ag_device_state$backend)
  if (is.null(buf)) return(invisible(NULL))
  slots <- .ag_pool_slots(scope)
  .ag_device_state[[slots$bufs]] <- c(.ag_device_state[[slots$bufs]], list(buf))
  # Legacy slot: last buffer allocated. Only the pass pool maintains it, since
  # that is what every existing reader means by "the current buffer".
  if (scope == "pass") .ag_device_state$buffer <- buf
  invisible(buf)
}

# Free every context and buffer, then start a new generation.
#
# Bumping ctx_gen is what makes stale pointers detectable: an ag_tensor that
# outlives this call still holds a $ptr into memory that has just been freed.
# `scope` says which pool to free:
#   "all"        both pools (the default -- a bare call means "free everything",
#                which is what every existing caller has always meant)
#   "pass"       graph nodes and activations only; resident weights survive
#   "persistent" weights and optimizer state only
#
# The default is deliberately the old behaviour. Making "pass" the default
# would silently turn every existing bare call into a partial reset, leaking
# the persistent pool wherever a caller wanted a clean slate -- and those call
# sites would have to be audited one by one to tell which meaning was intended.
# Only with_grad_tape() asks for a narrower scope, and it does so explicitly.
.ag_residency_reset <- function(size_mb = NULL, scope = "all") {
  scope <- match.arg(scope, c("all", .ag_scopes))
  pools <- if (scope == "all") .ag_scopes else scope

  # Rescue anything that must outlive the buffers before they go.
  #
  # A resident $grad (component 3) is a handle into a buffer freed just below.
  # The tensor holding it is an ordinary R object that survives the reset, so
  # without this its gradient would become a pointer into released memory --
  # caught loudly by the generation check, but only at the next read, far from
  # here. Tensor VALUES need no such rescue: .ag_data() keeps a host copy or
  # can refuse, whereas a gradient has no fallback and no second source.
  #
  # This runs for a pass reset too. Gradients are pass-pool tensors -- they are
  # produced by the backward graph, not carried between steps -- so freeing the
  # pass pool is exactly the event they need rescuing from.
  # ⚠️ ORDER. The deferred queue is settled BEFORE the two rescues below, not
  # after, and not by discarding it.
  #
  # Both rescues read values off the device (.ag_gpu_to_r), and a value being
  # rescued may be exactly a node this queue has not computed yet -- a resident
  # weight written by a deferred op, or a gradient whose snapshot is one. Freeing
  # the queue first would rescue whatever the buffer happened to hold: not an
  # error, not a stale-pointer trap, just wrong numbers written into $data as
  # though they were authoritative. That is the failure this contract exists to
  # prevent, so the queue is DRAINED here rather than dropped.
  #
  # Draining is also why .ag_gpu_to_r's own drain is not enough on its own: it
  # would fire mid-rescue, after the loop below had already freed a pool.
  #
  # It costs one compute of a graph whose results may go unread. That is the
  # price of the rescue being correct.
  #
  # With fusion the queue reaching here is no longer unusual: backward() does
  # NOT drain on the graph path, so a step whose gradients were never read
  # arrives at the next tape with the whole forward and backward still queued.
  # Draining is then exactly right -- .ag_materialise_pending_grads() below is
  # about to read those gradients, and they do not exist until this runs.
  if ("pass" %in% pools && .ag_defer_len()) {
    tryCatch(.ag_defer_drain(), error = function(e) NULL)
  }
  .ag_defer_discard()

  .ag_materialise_pending_grads()

  for (sc in pools) {
    # Resident VALUES need the same rescue, and for a stronger reason: a weight
    # with $data NULL exists nowhere else, so freeing its buffer without this
    # would destroy a trained model rather than merely cost a recomputation.
    # Done per pool, so a pass reset leaves persistent weights on the device.
    .ag_materialise_resident_values(sc)
    slots <- .ag_pool_slots(sc)
    for (buf in .ag_device_state[[slots$bufs]]) {
      tryCatch(ggml_backend_buffer_free(buf), error = function(e) NULL)
    }
    for (ctx in .ag_device_state[[slots$ctxs]]) {
      tryCatch(ggml_free(ctx), error = function(e) NULL)
    }
    .ag_device_state[[slots$bufs]] <- list()
    .ag_device_state[[slots$ctxs]] <- list()
    .ag_device_state[[slots$cur]]  <- NULL
    # Each pool counts its own generations, so freeing one cannot make the
    # other's live handles look stale.
    .ag_device_state[[slots$gen]]  <- .ag_device_state[[slots$gen]] + 1L
    if (sc == "pass") .ag_device_state$buffer <- NULL
  }

  if (!is.null(size_mb)) .ag_device_state$ctx_mb <- as.integer(size_mb)
  invisible(.ag_device_state$ctx_gen)
}

# Backwards-compatible name: with_grad_tape() calls this at the start of a tape,
# with scope = "pass" so resident weights survive the step boundary.
.ag_reset_ggml_ctx <- function(size_mb = 128L, scope = "all") {
  .ag_residency_reset(size_mb = size_mb, scope = scope)
  .ag_ctx_ensure()
}

# Make sure a context with room for `n` more tensors is current.
#
# The context being retired is flushed on the way out: it may still hold
# tensors that were never backed by memory, and nothing will return to it once
# a new context is current. A rollover does NOT free the old context — tensors
# already handed out stay valid — so ctx_gen is left untouched. Only a reset
# invalidates pointers.
.ag_ctx_ensure <- function(n = 1L, scope = "pass") {
  slots <- .ag_pool_slots(scope)
  ctx   <- .ag_device_state[[slots$cur]]
  if (!is.null(ctx) && .ag_ctx_capacity(ctx) >= n) return(ctx)

  if (!is.null(ctx)) .ag_ctx_flush(ctx, scope = scope)

  # Size the new context so the request fits even when it exceeds the default.
  per     <- as.double(ggml_tensor_overhead())
  need_mb <- ceiling((per * (as.double(n) + 1)) / (1024 * 1024))
  mb      <- max(as.double(.ag_device_state$ctx_mb), need_mb,
                 as.double(.ag_min_ctx_mb()))
  ctx     <- ggml_init(mb * 1024 * 1024, no_alloc = TRUE)
  if (is.null(ctx)) stop("ggmlR: failed to create a ggml context for the tape.")

  .ag_device_state[[slots$ctxs]] <- c(.ag_device_state[[slots$ctxs]], list(ctx))
  .ag_device_state[[slots$cur]]  <- ctx
  ctx
}

# Current tape memory usage.
#
# ctx_bytes/ctx_used cover descriptors (host side); buffer_bytes is the device
# memory actually backing resident tensors.
#
# The unprefixed fields keep their old meaning -- the pass pool -- so existing
# readers (ag_tape_memory(), the growth tests) see exactly what they saw before.
# The persistent pool is reported alongside under p_*, and total_buffer_bytes
# is the sum, which is what the VRAM budget has to look at: device memory held
# by resident weights is just as real as memory held by activations.
.ag_pool_mem <- function(scope) {
  slots <- .ag_pool_slots(scope)
  ctxs  <- .ag_device_state[[slots$ctxs]]
  bufs  <- .ag_device_state[[slots$bufs]]
  list(
    ctx_bytes = sum(vapply(ctxs,
                           function(c) as.double(ggml_get_mem_size(c)), numeric(1)), 0),
    ctx_used  = sum(vapply(ctxs,
                           function(c) as.double(ggml_used_mem(c)), numeric(1)), 0),
    buf_bytes = sum(vapply(bufs,
                           function(b) as.double(ggml_backend_buffer_get_size(b)),
                           numeric(1)), 0),
    n_ctx     = length(ctxs),
    n_buf     = length(bufs))
}

.ag_tape_mem <- function() {
  a <- .ag_pool_mem("pass")
  p <- .ag_pool_mem("persistent")
  list(ctx_bytes    = a$ctx_bytes,
       ctx_used     = a$ctx_used,
       buffer_bytes = a$buf_bytes,
       n_contexts   = a$n_ctx,
       n_buffers    = a$n_buf,
       p_ctx_bytes    = p$ctx_bytes,
       p_ctx_used     = p$ctx_used,
       p_buffer_bytes = p$buf_bytes,
       p_n_contexts   = p$n_ctx,
       p_n_buffers    = p$n_buf,
       total_buffer_bytes = a$buf_bytes + p$buf_bytes)
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
  # Both pools count: the limit guards device memory, and a resident weight
  # occupies it exactly as an activation does.
  used <- .ag_tape_mem()$total_buffer_bytes
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
#' @param scope Residency pool to allocate from: \code{"pass"} (default, freed
#'   at the next tape reset) or \code{"persistent"} (survives the reset, for
#'   weights and optimizer state).
#' @param out Optional \code{ag_handle} (or raw tensor pointer) to receive the
#'   result in place, instead of allocating a new tensor. The handle is returned
#'   unchanged, so repeated updates of the same tensor -- an optimizer stepping
#'   a weight -- neither grow the pool nor invalidate handles.
#' @noRd
.ag_run_op <- function(op_fn, inputs, out_shape, mem_mb = 32L,
                       dtype = .ag_device_state$dtype, node_hook = NULL,
                       resident = FALSE, scope = "pass", out = NULL) {
  scope <- match.arg(scope, .ag_scopes)
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
      stop("ggmlR: operand ", i, " is a device handle from the ",
           .ag_handle_scope(inputs[[i]]), " pool, generation ",
           inputs[[i]]$gen %||% NA, ", but that pool has been reset since ",
           "(generation ", .ag_scope_gen(.ag_handle_scope(inputs[[i]])), ").",
           call. = FALSE)
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
  # `scope` picks the pool this op's tensors are allocated from. The default,
  # "pass", is freed at the next tape reset -- right for activations and graph
  # nodes, which is every op on the training path. An op asked to produce a
  # value that must outlive the step (a weight update, an optimizer moment)
  # allocates from the persistent pool instead, and its result then survives
  # with_grad_tape()'s reset.
  ctx <- .ag_ctx_ensure(length(inputs) + 4L, scope = scope)
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

  # Deferred: the node is built, so this op's work as far as R is concerned is
  # done. Queue it and hand back a pending handle; the flush, the uploads, the
  # graph and the compute all happen once, at the barrier (R/ag_defer.R).
  #
  # Returning here skips the four stages below, which is the entire point: on a
  # depth-16 chain they run 32 times instead of once. What it does NOT skip is
  # the upload -- those operands still have to reach the device, they just do it
  # in one pass instead of thirty-two. That is why the measured gain is modest
  # on real shapes and why this is groundwork, not an optimisation.
  #
  # ⚠️ resident = FALSE IS HONOURED, and deferral gives way to it.
  #
  # The first version deferred regardless and returned a handle either way, on
  # the assumption that callers read results through .ag_as_matrix, which drains
  # first. That assumption is false: .ag_gpu_mse_parts asks for the loss with
  # resident = FALSE and does `as.numeric(s) / n` on the result directly, so it
  # got a list where it wanted a number ("'list' object cannot be coerced to
  # type 'double'"). Four tests caught it, which is the good case -- the same
  # mistake against a caller doing arithmetic on a partly-numeric structure
  # would have computed something instead of erroring.
  #
  # So the rule is the one the data contract already states: what a function
  # returns is decided by what the caller asked for, not by an optimisation
  # underneath it. A caller wanting a value gets a value; the queue is drained
  # below to produce it, which costs a compute exactly where the old path had
  # one anyway.
  if (isTRUE(resident) && .ag_defer_ok(scope, out)) {
    ups <- list()
    for (i in seq_along(inputs)) {
      if (.ag_is_handle(inputs[[i]])) next
      ups[[length(ups) + 1L]] <- list(ptr = ptrs[[i]],
                                      val = as.numeric(inputs[[i]]))
    }
    .ag_defer_push(node, ups)
    if (fprof) .ag_fwd_prof_record(facc)
    return(.ag_handle(node, out_shape, scope = scope, pending = TRUE))
  }

  # Not deferring -- because the caller wants a value (resident = FALSE), or
  # because this op is one deferral refuses (out=, the persistent pool). Either
  # way the queue has to go first: this op's operands may be pending handles,
  # and the compute below would read buffers that ggml_backend_alloc_ctx_tensors
  # has given memory to but nothing has filled. Not a crash -- plausible
  # garbage, which is worse.
  if (.ag_defer_len()) .ag_defer_drain()

  # In-place: land the result in a tensor the caller already owns.
  #
  # Why this is not "return a handle and let the caller keep it". The optimizer
  # updates the same weight every step, and a fresh result tensor per step would
  # grow the persistent pool without bound -- that pool cannot be reset while
  # the weights in it are live, so growth there is a leak with no collector.
  # Copying into the existing tensor keeps the pool a fixed size and, as a side
  # effect, keeps every handle to that weight valid.
  #
  # This MUST be built before the flush below. ggml_cpy returns a view of the
  # destination, and a view is still a tensor the context has to allocate; built
  # after the flush it would have no memory, and the graph would then compute
  # nothing at all -- silently, since an unallocated node is not an error. The
  # symptom is a destination that never changes, which reads like a wrong
  # kernel rather than a missing allocation.
  #
  # It is built in `ctx`, NOT in ctx_graph. That context is freed when this call
  # returns and may hold only the graph's own arrays; a tensor from it would
  # dangle. `ctx` is the residency context `node` itself came from, so the copy
  # shares its lifetime.
  root <- node
  if (!is.null(out)) {
    if (!identical(as.integer(out_shape), as.integer(.ag_dim(out))))
      stop("ggmlR: .ag_run_op(out=) shape mismatch: op yields ",
           paste(out_shape, collapse = "x"), ", destination is ",
           paste(.ag_dim(out), collapse = "x"), ".", call. = FALSE)
    root <- ggml_cpy(ctx, node, if (.ag_is_handle(out)) out$ptr else out)
  }

  # Allocate everything this op just added (inputs + nodes) in one buffer. The
  # flush is a no-op for tensors that already have memory, so earlier residents
  # of the context are not touched.
  .ag_ctx_flush(ctx, scope = scope)
  fstage("flush")

  # Upload input data -- but only for operands that are not already there. This
  # skip is the point of the handle type: a weight reused across a chain is
  # sent once instead of once per operation, which measured at 10-20% of a
  # backward pass on its own.
  for (i in seq_along(inputs)) {
    if (.ag_is_handle(inputs[[i]])) next
    .ag_xfer_up(ptrs[[i]], as.numeric(inputs[[i]]), "run_op operands")
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

  graph <- ggml_build_forward_expand(ctx_graph, root)
  fstage("graph")
  ggml_backend_graph_compute(backend, graph)
  fstage("compute")

  # The destination already holds the result; handing back the caller's own
  # handle keeps the pointer (and its generation) the one they started with.
  if (!is.null(out)) {
    if (fprof) .ag_fwd_prof_record(facc)
    return(out)
  }

  # Resident: hand back a name for the result and let the caller decide when
  # (or whether) it comes off the device.
  #
  # Safe with respect to the graph context freed on exit above: ownership runs
  # one way. `node` was allocated from the residency context and its data lives
  # in a backend buffer that .ag_device_state owns, so dropping the graph
  # cannot reach it -- the same argument the comment above makes for tensors.
  if (isTRUE(resident)) {
    if (fprof) .ag_fwd_prof_record(facc)
    return(.ag_handle(node, out_shape, scope = scope))
  }

  # Download result (always returns f32 doubles)
  raw <- .ag_xfer_down(node, "run_op result")
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
  # Shapes come from the accessors, which read a handle's recorded shape without
  # materialising it -- nrow() on a handle would be NULL.
  nr_a <- .ag_nrow(a_data); nc_a <- .ag_ncol(a_data)   # m, k
  nr_b <- .ag_nrow(b_data); nc_b <- .ag_ncol(b_data)   # k, n

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
  # A resident operand cannot be transposed on the host -- t() would have to
  # download it first, which is the round trip residency exists to remove. So
  # the transpose moves into the graph exactly when the operand is already on
  # the device, and stays in R otherwise.
  #
  # The note above (measured: 42.0 vs 42.3 ms, identical accuracy) is what makes
  # this safe: the two forms cost the same, so the choice can follow where the
  # data already is rather than which is faster.
  if (.ag_is_handle(a_data)) {
    return(.ag_run_op(
      op_fn = function(ctx, ptrs)
                ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, ptrs[[1L]])),
                             ptrs[[2L]]),
      inputs    = list(a_data, b_data),
      out_shape = c(nr_a, nc_b),
      resident  = .ag_is_handle(b_data) || .ag_is_handle(a_data)))
  }

  at_data <- t(a_data)                          # [k, m]
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_mul_mat(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(at_data, b_data),
    out_shape = c(nr_a, nc_b),
    resident = .ag_is_handle(b_data)
  )
}

# A %*% B with f32 accumulation forced on the matmul node. The Vulkan backend
# accumulates mul_mat in f16 by default (~2.7e-4 relative error), which is fine
# for neural-net layers but corrupts precision-sensitive downstream maths — e.g.
# a Gram matrix whose ||x_i||^2 + ||x_j||^2 - 2 G[i,j] distances feed kNN, where
# the f16 noise reorders nearest neighbours. GGML_PREC_F32 selects the f32 kernel.
GGML_PREC_F32 <- 10L
.ag_gpu_matmul_f32 <- function(a_data, b_data) {
  nr_a <- .ag_nrow(a_data); nc_b <- .ag_ncol(b_data)
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
  # The result takes a's shape, which is what makes the native broadcast work:
  # b may be [m,1] or [1,n] against a's [m,n], and ggml_add repeats it.
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_add(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(a_data, b_data),
    out_shape = .ag_dim(a_data),
    resident  = .ag_is_handle(a_data) || .ag_is_handle(b_data)
  )
}

.ag_gpu_sub <- function(a_data, b_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_sub(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(a_data, b_data),
    out_shape = .ag_dim(a_data)
  )
}

.ag_gpu_mul <- function(a_data, b_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_mul(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs   = list(a_data, b_data),
    out_shape = .ag_dim(a_data)
  )
}

.ag_gpu_scale <- function(x_data, scalar) {
  s <- as.double(scalar)
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_scale(ctx, ptrs[[1L]], s),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data)
  )
}

.ag_gpu_relu <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_relu(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data),
    resident  = .ag_is_handle(x_data)
  )
}

.ag_gpu_sigmoid <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_sigmoid(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data),
    resident  = .ag_is_handle(x_data)
  )
}

.ag_gpu_tanh <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_tanh(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data),
    resident  = .ag_is_handle(x_data)
  )
}

# ggml_soft_max applies softmax along ne0 = rows in R = each column sums to 1
.ag_gpu_softmax <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_soft_max(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data),
    resident  = .ag_is_handle(x_data)
  )
}

.ag_gpu_log <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_log(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data),
    resident  = .ag_is_handle(x_data)
  )
}

.ag_gpu_exp <- function(x_data) {
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_exp(ctx, ptrs[[1L]]),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data)
  )
}

.ag_gpu_clamp <- function(x_data, lo, hi) {
  lo <- as.double(lo); hi <- as.double(hi)
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_clamp(ctx, ptrs[[1L]], lo, hi),
    inputs   = list(x_data),
    out_shape = .ag_dim(x_data)
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
    out_shape = c(1L, .ag_ncol(x_data))
  )
}

# ag_sum(dim=1) = rowSums: CPU fallback (Vulkan transpose+sum_rows not supported).
.ag_gpu_sum_rows <- function(x_data) {
  matrix(rowSums(x_data), nrow = nrow(x_data), ncol = 1L)
}

# ag_mean(dim=2) = colMeans = colSums / nrow
# ggml_sum_rows supports f32 and f16; ggml_scale also supports both.
.ag_gpu_mean_cols <- function(x_data) {
  nr <- .ag_nrow(x_data)
  .ag_run_op(
    op_fn    = function(ctx, ptrs) {
      ggml_scale(ctx, ggml_sum_rows(ctx, ptrs[[1L]]), 1.0 / nr)
    },
    inputs   = list(x_data),
    out_shape = c(1L, .ag_ncol(x_data))
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
      out_shape = .ag_dim(x_data),
      resident  = .ag_is_handle(x_data)
    )
  } else if (p == 0.5) {
    .ag_run_op(
      op_fn    = function(ctx, ptrs) ggml_sqrt(ctx, ptrs[[1L]]),
      inputs   = list(x_data),
      out_shape = .ag_dim(x_data),
      resident  = .ag_is_handle(x_data)
    )
  } else {
    s <- as.double(p)
    .ag_run_op(
      op_fn    = function(ctx, ptrs)
                   ggml_exp(ctx, ggml_scale(ctx, ggml_log(ctx, ptrs[[1L]]), s)),
      inputs   = list(x_data),
      out_shape = .ag_dim(x_data),
      resident  = .ag_is_handle(x_data)
    )
  }
}

# ggml_transpose returns a view; ggml_cont makes it contiguous.
# Result shape: [ncol(x), nrow(x)]
.ag_gpu_transpose <- function(x_data) {
  out_shape <- c(.ag_ncol(x_data), .ag_nrow(x_data))
  .ag_run_op(
    op_fn    = function(ctx, ptrs) ggml_cont(ctx, ggml_transpose(ctx, ptrs[[1L]])),
    inputs   = list(x_data),
    out_shape = out_shape,
    resident  = .ag_is_handle(x_data)
  )
}

# ---------------------------------------------------------------------------
# The Adam step, run on the device.
#
# What this replaces. The host step reads the gradient, both moments and the
# weight as R matrices, does six lines of arithmetic, and writes the weight
# back -- four downloads and one upload per parameter per step, measured at
# 4 of the 10 crossings a step costs (inst/scripts/measure_ag_step_transfers.R).
# None of those numbers are wanted on the host: they are produced on the device
# and consumed on the device.
#
# Everything below stays in the persistent pool and writes in place, so a step
# neither allocates nor frees: m, v and w are updated through `out =`, which
# copies into the tensor the caller already owns. That matters more here than
# anywhere else -- the persistent pool cannot be reset while live weights are in
# it, so a step that allocated would leak once per iteration, forever.
#
# Bias correction is two scalars (1 - beta^t). They are computed in R and folded
# into ggml_scale, rather than uploaded as tensors: a scalar in a push constant
# costs nothing, a 1x1 tensor costs an allocation and a crossing.
.ag_adam_step_device <- function(env, nm, p, g) {
  # ⚠️ Never deferred. Every graph below is ordered by hand relative to the
  # copies at the end -- read m, v and w first, write them afterwards -- and a
  # queue that postpones the reads until the first write folds the two into one
  # graph. Symptom when this wrapper was missing: the loss sat at 0.212 for four
  # steps and the weight never moved at all. See .ag_defer_suspend.
  .ag_defer_suspend(.ag_adam_step_device_impl(env, nm, p, g))
}

.ag_adam_step_device_impl <- function(env, nm, p, g) {
  m  <- env$m[[nm]]
  v  <- env$v[[nm]]
  wh <- .ag_handle_of(p)
  sh <- .ag_dim(m)

  # The gradient is read into R first, deliberately.
  #
  # It arrives as a handle into the PASS pool -- from the resident backward, or
  # uploaded a few lines up in optimizer_adam(). The graphs below each allocate
  # in that same pool, and a pass-pool operand read across those allocations is
  # not reliable: that is the same exposure the copy ordering below addresses,
  # arriving from the other side.
  #
  # Tried without it, after the ordering fix landed: the closed-form check still
  # passed (0.7499999 for three constant-gradient steps) but two suite tests
  # failed, so the ordering rule alone does not cover every caller.
  #
  # It is copied into the PERSISTENT pool rather than read back to R. Passing a
  # matrix instead made the step upload it once per graph -- three graphs plus
  # the copies, measured as 6 uploads and 2 downloads per step against 2 before,
  # which is worse than the download it was meant to avoid. A persistent tensor
  # is uploaded once and then simply referenced, and nothing this step allocates
  # can disturb it, since every allocation here goes to the pass pool.
  if (.ag_is_handle(g)) {
    gbuf <- env$gbuf[[nm]]
    if (is.null(gbuf) || !.ag_handle_live(gbuf)) {
      gbuf <- .ag_handle(.ag_r_to_gpu(matrix(0, sh[1L], sh[2L]),
                                      scope = "persistent"),
                         sh, scope = "persistent")
      env$gbuf[[nm]] <- gbuf
    }
    .ag_run_op(function(ctx, ptrs) ggml_dup(ctx, ptrs[[1L]]),
               inputs = list(g), out_shape = sh, scope = "pass", out = gbuf)
    g <- gbuf
  }

  b1 <- env$beta1; b2 <- env$beta2
  bc1 <- 1 - b1^env$t                     # bias correction, m
  bc2 <- 1 - b2^env$t                     # bias correction, v

  # scope = "pass" and out = <persistent handle> is the combination that makes
  # this free of leaks, and the two arguments mean different things:
  #   scope  where the op's own intermediate nodes are allocated -- the scaled
  #          moment, the squared gradient, the divisor. These are garbage the
  #          moment the step ends, so they belong in the pool that the next tape
  #          reset frees.
  #   out    where the RESULT lands: a tensor already in the persistent pool.
  # Allocating the intermediates persistently instead would grow that pool by a
  # few nodes every step, and it has no collector -- it cannot be reset while
  # the weights in it are live. Measured before this was fixed: 4 KB to 37 KB
  # over ten steps, growing without bound.

  # ⚠️ A tensor must not be both an operand and the destination of the SAME
  # graph. `out =` appends a ggml_cpy whose source is the expression reading
  # that tensor, so "m <- b1*m + ..." has the kernel writing m while another
  # node still reads it. Nothing orders those, and nothing reports it.
  #
  # It hides at step 1 and only then: the moments start at zero, so writing
  # zeros over zeros lands on the same values whatever the order. From step 2 on
  # the two paths drift -- measured as W1 maxdiff 0 after one step, 0.0465 after
  # two, 0.239 after five, while the GRADIENT stayed bit-identical throughout.
  # That shape of evidence (correct gradient, correct first step, growing
  # divergence) is what points here rather than at the backward pass.
  #
  # Same family as the sched-level trap in the notes: an op that writes into its
  # own src loses the write. Here the fix is to compute into a fresh pass-pool
  # tensor and copy that into the moment afterwards -- two graphs, no aliasing.
  # ORDER MATTERS: the weight is computed FIRST, while m and v still hold the
  # previous step's values, and only afterwards are the moments advanced.
  #
  # Computing the moments first and feeding their results into the weight update
  # is what produced a weight one step ahead: traced, w came out 0.832808 where
  # 0.85 was correct, which is exactly the value obtained from m = 0.076 instead
  # of 0.04 -- the gradient applied twice. The moment graphs run before the
  # weight graph, and by the time the weight graph reads m the tensor no longer
  # holds what it did at entry.
  #
  # The weight graph recomputes b1*m + (1-b1)*g internally, so it needs nothing
  # from those earlier graphs -- every operand it takes is a tensor this call
  # has not written yet.

  # w <- w - lr * (m/bc1) / (sqrt(v/bc2) + eps)
  #
  # The division by bc1 is folded into the lr scale, and the one by bc2 into the
  # scale under the square root, so the correction costs no extra nodes.
  # ggml_scale_bias(a, s, b) computes a*s + b, which is exactly "sqrt(v_hat)
  # plus epsilon" in one op.
  # The update reads the NEW moments -- Adam's m and v are the ones from this
  # step, not the previous. They are still in their own tensors here, so the
  # aliasing rule above is respected: nothing reads m or v while they are being
  # written.
  lr <- env$lr; eps <- env$eps
  # ⚠️ The whole update in ONE graph, from m, v and g -- not from the new_m and
  # new_v computed above.
  #
  # Reading new_m back as an operand of a second graph does not work: it is a
  # handle onto a node, and by the time that node is read the moment tensor it
  # was computed from has been rewritten. Traced, the weight came out as
  # 0.832808 instead of 0.85, which is exactly the value obtained by applying
  # the gradient TWICE (m = 0.076 instead of 0.04) -- the update saw moments a
  # step ahead of where they should have been.
  #
  # Recomputing b1*m + (1-b1)*g inside this graph costs two cheap nodes and
  # removes the dependency on a previous graph's output entirely: every operand
  # here is a tensor whose contents nothing in this call has touched.
  new_w <- .ag_run_op(function(ctx, ptrs) {
               w_ <- ptrs[[1L]]; m_ <- ptrs[[2L]]; v_ <- ptrs[[3L]]; g_ <- ptrs[[4L]]
               nm_ <- ggml_add(ctx, ggml_scale(ctx, m_, b1),
                               ggml_scale(ctx, g_, 1 - b1))
               nv_ <- ggml_add(ctx, ggml_scale(ctx, v_, b2),
                               ggml_scale(ctx, ggml_sqr(ctx, g_), 1 - b2))
               num <- ggml_scale(ctx, nm_, lr / bc1)
               den <- ggml_scale_bias(ctx,
                        ggml_sqrt(ctx, ggml_scale(ctx, nv_, 1 / bc2)), 1, eps)
               ggml_sub(ctx, w_, ggml_div(ctx, num, den))
             },
             inputs = list(wh, m, v, g), out_shape = sh,
             scope = "pass", resident = TRUE)

  # Now the moments, from the values they still hold.
  new_m <- .ag_run_op(function(ctx, ptrs)
                        ggml_add(ctx,
                                 ggml_scale(ctx, ptrs[[1L]], b1),
                                 ggml_scale(ctx, ptrs[[2L]], 1 - b1)),
                      inputs = list(m, g), out_shape = sh,
                      scope = "pass", resident = TRUE)

  new_v <- .ag_run_op(function(ctx, ptrs)
                        ggml_add(ctx,
                                 ggml_scale(ctx, ptrs[[1L]], b2),
                                 ggml_scale(ctx, ggml_sqr(ctx, ptrs[[2L]]), 1 - b2)),
                      inputs = list(v, g), out_shape = sh,
                      scope = "pass", resident = TRUE)

  # Only now, with every read finished, are the persistent tensors overwritten.
  # Each copy is its own graph, so a destination is never also a source.
  #
  # ggml_cpy, not ggml_scale(x, 1): scaling by one is not a no-op on this
  # backend. The value goes through a kernel and comes back rounded to the
  # compute precision, so a "copy" written that way loses a little of m, v and w
  # on EVERY step -- which compounds into a visibly different trajectory while
  # looking like an identity operation in the source.
  # ggml_dup, and the destination through `out =`.
  #
  # Three ways to write this copy are wrong, and each failed differently:
  #   ggml_scale(x, 1)          not an identity on this backend -- the value
  #                             goes through a kernel and comes back rounded to
  #                             the compute precision, losing a little of m, v
  #                             and w on every step.
  #   ggml_cpy with dst in
  #   `inputs`                  dst becomes an ordinary operand, so a fresh
  #                             tensor is built for it and the result is a view
  #                             of that: "leaf_0 (copy of node_2) has no backend
  #                             buffer".
  #   returning ptrs[[1]]       not an operation at all. The graph has no node
  #                             to execute, so nothing is copied and the weight
  #                             silently never changes.
  # ggml_dup is a real op that reproduces its operand exactly, and `out =` lets
  # .ag_run_op append the copy into the tensor the caller already owns.
  cpy <- function(src, dst)
    .ag_run_op(function(ctx, ptrs) ggml_dup(ctx, ptrs[[1L]]),
               inputs = list(src), out_shape = sh, scope = "pass", out = dst)
  # The weight is copied FIRST.
  #
  # new_m, new_v and new_w are all pass-pool nodes, and every cpy() is itself an
  # .ag_run_op that allocates in that same pool. Copying the moments first
  # therefore disturbs the pool while new_w is still only a handle into it, and
  # the value read out afterwards is no longer the one the graph produced:
  # traced, new_w measured 0.85 immediately after its compute and 0.832808 by
  # the time it was copied -- the difference between a correct step and moments
  # advanced twice.
  #
  # Ordering the copies by how soon each result is needed removes the exposure:
  # after this line the weight is in its persistent tensor, and nothing later in
  # the step reads it.
  cpy(new_w, wh)
  cpy(new_m, m)
  cpy(new_v, v)

  # Opt-in trace: print what the step actually fed the update, as opposed to
  # what the formula says it should have. Section 8 of
  # inst/scripts/diag_ag_run_op_out.R showed m and v correct while w came out
  # wrong on the FIRST step, with every node of the expression verified correct
  # in isolation -- so the remaining question is which values the expression
  # received, and only the step itself can answer it.
  if (identical(Sys.getenv("GGMLR_AG_ADAM_TRACE"), "1")) {
    pk <- function(h) tryCatch(.ag_as_matrix(h)[1L], error = function(e) NA_real_)
    message(sprintf(
      "adam[%s] t=%d bc1=%.6f bc2=%.6f | g=%.8f m=%.8f v=%.10f new_m=%.8f new_v=%.10f w=%.8f -> %.8f",
      nm, env$t, bc1, bc2, pk(g), pk(m), pk(v), pk(new_m), pk(new_v),
      pk(wh), pk(new_w)))
    # Read after the copies, so these are the values that were stored, not the
    # ones the graphs produced. The two differ if a pass-pool result is read
    # after something else has allocated in that pool -- the failure this
    # ordering exists to avoid.
  }

  # The weight's buffer changed underneath any cached host copy.
  p$data     <- NULL
  p$data_gen <- NULL
  invisible(NULL)
}

# MSE on the device: the difference stays resident, only the scalar comes back.
#
# The host version computes `pred - target` and `sum(diff^2)/n` in R, which
# forces a download of pred -- and pred is the output of the forward chain, so
# that one call undoes the residency of everything upstream of it. It then hands
# the same `diff` to the tape as the gradient matrix, which the backward graph
# uploads again: one value crossing the bus twice per step.
#
# Here `diff` is computed as a node and kept as a handle, so it goes to the tape
# without ever being materialised. Only the loss scalar is downloaded, because
# something always prints or logs it -- one number, not a matrix.
#
# Returns list(loss = <1x1 matrix>, diff = <handle>).
.ag_gpu_mse_parts <- function(p_data, t_data, n) {
  d <- .ag_run_op(
    op_fn     = function(ctx, ptrs) ggml_sub(ctx, ptrs[[1L]], ptrs[[2L]]),
    inputs    = list(p_data, t_data),
    out_shape = .ag_dim(p_data),
    resident  = TRUE)

  # sum(diff^2)/n as one graph, off the resident diff: no second upload.
  #
  # The division by n is a graph node rather than R arithmetic on the result,
  # and that is the whole point: it is what lets the loss come back as a HANDLE.
  # While the scalar was computed as `as.numeric(s) / n`, reading it was a
  # download -- one of the four crossings left in a step, and the one that
  # forced the forward to be computed before backward() could start, since the
  # value had to exist for the division. A resident loss removes the crossing
  # and, with it, the ordering constraint that made a fused forward+backward
  # graph impossible.
  #
  # Callers that want the number still get it: ag_data() on the loss tensor
  # materialises through the ordinary accessor, draining any deferred queue on
  # the way. What changed is that nothing forces that read to happen early.
  s <- .ag_run_op(
    op_fn     = function(ctx, ptrs)
                  ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, ptrs[[1L]])),
                             1 / n),
    inputs    = list(d),
    out_shape = c(1L, 1L),
    resident  = TRUE)

  list(loss = s, diff = d)
}

# The relu mask, (x > 0) * 1, computed on the device.
#
# The forward used to build this in R as `(x_data > 0) * 1.0`, which forces the
# operand back to the host -- and the operand is an activation, so that download
# breaks the residency chain in the middle of every network with a relu in it.
# ggml_step is the same function: op_step in ggml-cpu/unary-ops.cpp is
# `(x > 0) ? 1 : 0`.
.ag_gpu_step <- function(x_data) {
  .ag_run_op(
    op_fn     = function(ctx, ptrs) ggml_step(ctx, ptrs[[1L]]),
    inputs    = list(x_data),
    out_shape = .ag_dim(x_data),
    resident  = TRUE
  )
}

# Activation derivatives, computed on the device from the activation's OUTPUT.
#
# These differ from the relu mask in where the multiplier comes from: relu's
# depends on the input, sigmoid's and tanh's on the result the forward just
# produced. Either way the host version -- `s * (1 - s)`, `1 - t^2` in R --
# downloads that value, and it is an activation sitting in the middle of the
# network, so the download breaks the residency chain there.
#
# ggml_scale_bias(a, s, b) computes a*s + b, which expresses both:
#   sigmoid   s * (1 - s)   =  s * (s*(-1) + 1)
#   tanh      1 - t^2       =  t^2*(-1) + 1

# s * (1 - s), from the sigmoid output.
.ag_gpu_sigmoid_grad <- function(s_data) {
  .ag_run_op(
    op_fn = function(ctx, ptrs)
              ggml_mul(ctx, ptrs[[1L]],
                       ggml_scale_bias(ctx, ptrs[[1L]], -1, 1)),
    inputs    = list(s_data),
    out_shape = .ag_dim(s_data),
    resident  = TRUE
  )
}

# 1 - t^2, from the tanh output.
.ag_gpu_tanh_grad <- function(t_data) {
  .ag_run_op(
    op_fn = function(ctx, ptrs)
              ggml_scale_bias(ctx, ggml_sqr(ctx, ptrs[[1L]]), -1, 1),
    inputs    = list(t_data),
    out_shape = .ag_dim(t_data),
    resident  = TRUE
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
.ag_r_to_gpu <- function(data, dtype = .ag_device_state$dtype, scope = "pass") {
  scope <- match.arg(scope, .ag_scopes)
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
  ctx <- .ag_ctx_ensure(1L, scope = scope)
  ptr <- ggml_new_tensor_2d(ctx, ggml_type, nr, nc)

  # This tensor has to be readable when the call returns, so its memory must
  # exist now. The flush covers every unallocated tensor of the context at once,
  # so tensors created back-to-back before any upload share a single buffer.
  .ag_ctx_flush(ctx, scope = scope)

  .ag_xfer_up(ptr, as.numeric(data), "r_to_gpu")
  ptr
}

# Create resident tensors for several matrices under ONE allocation.
#
# The single-tensor path has to allocate on every call, because it must return
# something readable. Callers that upload a group of tensors (a layer's
# parameters, a tape's inputs) should use this instead: descriptors are created
# first, allocated in one batch, and only then filled — one buffer for the whole
# group rather than one per tensor.
.ag_r_to_gpu_batch <- function(mats, dtype = .ag_device_state$dtype, scope = "pass") {
  scope <- match.arg(scope, .ag_scopes)
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
  ctx  <- .ag_ctx_ensure(length(mats), scope = scope)
  ptrs <- lapply(mats, function(m)
    ggml_new_tensor_2d(ctx, ggml_type, nrow(m), ncol(m)))

  .ag_ctx_flush(ctx, scope = scope)

  for (i in seq_along(mats)) {
    .ag_xfer_up(ptrs[[i]], as.numeric(mats[[i]]), "r_to_gpu_batch")
  }
  ptrs
}

# Download data from a ggml tensor pointer to an R matrix.
#
# A drain point, and the one that is easy to miss. .ag_data_set_handle installs
# a handle's POINTER on an ag_tensor and drops the handle object, so a tensor
# produced by a deferred op carries no pending flag -- the flag lives on the
# handle, and by here it is gone. Rather than thread it through the tensor, the
# queue is drained on any read from a device pointer: this is the only function
# that turns one into numbers, so covering it covers every ag_tensor read
# (.ag_data, the resident-value rescue, ag_save).
#
# Cheap when nothing is queued, which is always on the non-deferred path.
.ag_gpu_to_r <- function(tensor) {
  .ag_defer_drain()
  ptr   <- tensor$ptr
  shape <- tensor$shape   # [nr, nc] stored at creation time
  raw   <- .ag_xfer_down(ptr, "gpu_to_r")
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
           "pointer, ", .ag_tensor_scope(t), " pool, generation ",
           t$ctx_gen %||% NA, " < ", .ag_scope_gen(.ag_tensor_scope(t)),
           ") and it has no CPU copy to fall back on.",
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
    t$ptr       <- NULL
    t$ctx_gen   <- NULL
    t$ctx_scope <- NULL
    t$shape     <- NULL
  }
  invisible(t)
}

# Pool a resident tensor's pointer came from. Tensors that predate scopes, and
# every tensor allocated the ordinary way, belong to the pass pool.
.ag_tensor_scope <- function(t) t$ctx_scope %||% "pass"

# TRUE when a resident tensor's pointer still belongs to the current generation
# OF ITS OWN POOL. Checking against the pass counter alone would declare every
# persistent weight stale at the first tape reset -- the exact thing the split
# exists to prevent.
# Tensors created before generations were tracked carry no ctx_gen; they count
# as stale, since nothing proves their context survived.
.ag_ptr_is_live <- function(t) {
  identical(t$ctx_gen, .ag_scope_gen(.ag_tensor_scope(t)))
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
    stop("ggmlR: refusing to install a device handle from the ",
         .ag_handle_scope(h), " pool, generation ", h$gen %||% NA,
         " (current is ", .ag_scope_gen(.ag_handle_scope(h)), ").",
         call. = FALSE)
  t$ptr       <- h$ptr
  t$shape     <- h$shape
  t$ctx_gen   <- h$gen
  # The pool travels with the pointer: $ctx_gen is only meaningful next to the
  # counter it was taken from.
  t$ctx_scope <- .ag_handle_scope(h)
  t$data      <- NULL     # not materialised -- see rule 4
  t$data_gen  <- NULL
  invisible(t)
}

# Write a new value into a resident tensor WITHOUT giving up residency.
#
# Why this is not .ag_data_set(). That function drops $ptr by design: it installs
# a host value, and leaving the old pointer in place would leave two versions of
# the value with nothing to say which is current (contract rule 3). That is the
# right rule for a host write -- but it makes a resident weight impossible to
# update, because the optimizer's read-modify-write cycle
# (.ag_data_mut -> .ag_data_set) would strip the residency on the very first
# step and never restore it. The weight would live on the device for exactly one
# step, which is the same as not living there at all.
#
# So this is the device-side counterpart: the new value is uploaded straight
# into the buffer the tensor already owns. There is still only one version of
# the value -- the device one -- so the contract holds; what changes is that the
# pointer survives, because it is the thing being written to.
#
# The cached host copy MUST be dropped: $data would otherwise hold the value
# from before the write, and .ag_data() would serve it whenever $data_gen still
# matched. Clearing $data_gen alone is not enough, since a NULL generation makes
# the cache look authoritative rather than stale.
.ag_data_write_resident <- function(t, value) {
  if (!is_ag_tensor(t))
    stop("ggmlR: .ag_data_write_resident() expects an ag_tensor.", call. = FALSE)
  if (is.null(t$ptr) || !.ag_ptr_is_live(t))
    stop("ggmlR: .ag_data_write_resident() needs a live resident tensor; use ",
         ".ag_data_set() for a host-side value.", call. = FALSE)
  if (is.vector(value) && !is.list(value)) value <- matrix(value, ncol = 1L)
  if (!identical(as.integer(dim(value)), as.integer(t$shape)))
    stop("ggmlR: .ag_data_write_resident() cannot change a tensor's shape (",
         paste(dim(value), collapse = "x"), " into ",
         paste(t$shape, collapse = "x"), ").", call. = FALSE)

  .ag_xfer_up(t$ptr, as.numeric(value), "write_resident")
  t$data     <- NULL
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

# ---------------------------------------------------------------------------
# Rescuing resident VALUES, the same way resident gradients are rescued.
#
# A parameter uploaded by ag_param() has its value only on the device: $data is
# NULL, which the contract calls "not materialised". That is fine while the
# buffer lives -- and a disaster when it does not. ag_device("cpu") frees the
# persistent pool, and without this the weight would simply cease to exist: no
# host copy, no readable pointer, and .ag_data() correctly refusing to guess.
#
# So resident parameters are registered here, and brought back to the host just
# before their buffers go. Unlike a gradient, a weight is often the only copy of
# a trained model, which makes losing it the more expensive failure of the two.
.ag_device_state$resident_values <- list()

.ag_register_resident_value <- function(t) {
  key <- as.character(t$id)
  .ag_device_state$resident_values[[key]] <- t
  invisible(NULL)
}

.ag_forget_resident_values <- function() {
  .ag_device_state$resident_values <- list()
  invisible(NULL)
}

# Pull every registered resident value back to the host, then forget them.
#
# `scope` limits the rescue to tensors of the pool being freed: a pass reset
# must not drag every weight off the device, which would undo the whole point of
# the persistent pool.
.ag_materialise_resident_values <- function(scope) {
  vals <- .ag_device_state$resident_values
  if (length(vals) == 0L) return(invisible(NULL))
  keep <- list()
  for (t in vals) {
    if (!identical(.ag_tensor_scope(t), scope)) { keep[[as.character(t$id)]] <- t; next }
    if (is.null(t$ptr) || !.ag_ptr_is_live(t)) next
    # Read straight from the pointer: .ag_data() would work too, but this keeps
    # the rescue independent of the caching rules it consults.
    val <- tryCatch(.ag_gpu_to_r(t), error = function(e) NULL)
    if (!is.null(val)) {
      t$data     <- val
      t$data_gen <- NULL      # authoritative host value now, not a cache
    }
    t$ptr       <- NULL
    t$ctx_gen   <- NULL
    t$ctx_scope <- NULL
    t$shape     <- NULL
  }
  .ag_device_state$resident_values <- keep
  invisible(NULL)
}

# The operand form of a value: a device handle when one exists, an R matrix
# otherwise.
#
# This is the entry point stage 1 turns on. Every ag_* operation used to start
# with .ag_data(), which materialises -- fine while nothing was resident, but
# once ag_param() keeps weights on the device that call downloads a weight per
# operation per step, which is measurable: 2 gpu_to_r crossings per step in a
# single-layer forward (inst/scripts/measure_ag_step_transfers.R).
#
# .ag_operand() answers the question the operations actually have ("give me
# something .ag_run_op can consume") instead of the question they were asking
# ("give me numbers"). Handles pass through .ag_run_op untouched; matrices are
# uploaded as before. Callers that genuinely need numbers -- the host branch,
# the backward snapshots -- keep calling .ag_data().
.ag_operand <- function(t) {
  h <- .ag_handle_of(t)
  if (!is.null(h)) return(h)

  d <- .ag_data(t)

  # First use of a host-side ag_tensor on the device: upload it ONCE and keep
  # the pointer, instead of letting every .ag_run_op create a fresh tensor and
  # send the same bytes again.
  #
  # Why it matters beyond the forward. An input like x or y is used by the
  # forward AND recorded on the tape as a backward snapshot, so without this it
  # crosses the bus at least twice per step -- measured as `bwd_graph operands`,
  # the last upload left after the gradients went resident.
  #
  # Pass scope, deliberately: an input belongs to the step, not to the run.
  # It is re-uploaded after each tape reset, which is correct -- a training loop
  # feeds a new batch every step, and pinning batches in the persistent pool
  # would grow it without bound. What this removes is the SECOND upload of the
  # same batch within one step, not the per-step upload itself.
  #
  # Only for tape-tracked tensors: a bare matrix has nowhere to keep a pointer.
  if (!is_ag_tensor(t) || is.null(d) || is.null(.ag_device_state$backend)) return(d)
  if (!identical(t$device, "gpu")) return(d)
  # Isolation switch, so this can be varied independently of resident gradients
  # (GGMLR_AG_RESIDENT_GRADS) when attributing a numerical difference.
  if (identical(Sys.getenv("GGMLR_AG_OPERAND_CACHE"), "0")) return(d)

  ptr <- tryCatch(.ag_r_to_gpu(d, dtype = t$dtype %||% .ag_device_state$dtype),
                  error = function(e) NULL)
  if (is.null(ptr)) return(d)

  h <- .ag_handle(ptr, dim(d), scope = "pass")
  # $data is kept: unlike a weight, an input has a host value that stays
  # authoritative, and dropping it would force a download to read the batch
  # back. The pointer is a cache of it for this tape, nothing more.
  t$ptr       <- h$ptr
  t$shape     <- h$shape
  t$ctx_gen   <- h$gen
  t$ctx_scope <- "pass"
  t$data_gen  <- h$gen
  h
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
  # The tensor's own pool, not the default: a persistent weight handed a
  # pass-pool handle would be checked against the wrong generation counter and
  # rejected as stale at the first tape reset, even though its buffer is alive.
  .ag_handle(t$ptr, t$shape, scope = .ag_tensor_scope(t))
}
