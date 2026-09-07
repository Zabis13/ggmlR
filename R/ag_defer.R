# Deferred forward: build the nodes now, compute them once.
#
# WHAT THIS IS FOR. Not, primarily, speed. The measurement that motivated it
# (inst/scripts/proto_ag_forward_graph.R, run 2026-09-05) put a unified forward
# at 1.27x / 1.04x / 1.01x on a full training step -- the per-op path's `compute`
# does collapse (2.61 -> 1.04 ms at d=1024, 81 -> 2.6 at d=4096), but on real
# shapes the forward is dominated by weight `upload`, which one graph does not
# touch (21.68 -> 23.31 ms). A faster forward alone is worth about a percent.
#
# It is built as the FOUNDATION for the fused forward+backward graph. Today the
# backward re-uploads the forward's snapshots as constants (const() in
# R/ag_backward_graph.R): a_snap/b_snap travel as values because the tape has no
# record of the NODE that produced them. Once the forward leaves its nodes in
# the pass context and the tape carries a pointer to each one, the backward can
# attach to them instead of uploading anything, and the snapshots disappear.
# That is the change worth making; deferral is what makes it expressible.
#
# WHY THE NODES SURVIVE. .ag_residency_reset(scope = "pass") runs at the START
# of with_grad_tape(), not at its end, so a forward's nodes are still allocated
# while backward() runs. Nothing here extends any lifetime; the queue only
# postpones the compute.
#
# THE CONTRACT, and the trap it steps around. TODO records that the result of
# .ag_run_op(resident = TRUE) is "a position in the pass pool, not a value",
# valid only until the next allocation in that pool -- five iterations were lost
# to it on the Adam step. Deferral widens the window between building a node and
# filling it, so the rule needs restating rather than repeating:
#
#   A pending handle's POINTER is stable. ggml_backend_alloc_ctx_tensors gives
#   memory to tensors that do not have it yet and does not move tensors that do,
#   so queueing more nodes cannot relocate an earlier one.
#
#   A pending handle's CONTENTS do not exist until the barrier. Reading one
#   without draining returns whatever the buffer happened to hold -- which is
#   why every read goes through .ag_handle_to_r, and why the flag is checked
#   there rather than trusted to callers.
#
# So the danger is not a moved pointer, it is a read that skips the barrier.
# Three places force a drain rather than risk one: a read (the barrier proper),
# an out= write, and any allocation in the persistent pool.

# Queue state. Kept in its own environment rather than in .ag_device_state so a
# residency reset cannot half-clear it: the reset frees buffers, and the queue
# has to be dropped as a unit when that happens.
.ag_defer <- new.env(parent = emptyenv())
.ag_defer$nodes    <- list()   # roots to expand into the graph, in build order
.ag_defer$uploads  <- list()   # list(ptr=, val=) filled after the flush
.ag_defer$enabled  <- NULL     # NULL = consult the environment variable
.ag_defer$depth    <- 0L       # >0 while draining; blocks re-entry

#' Defer forward operations into one graph
#'
#' Component 5 of the resident contract, and the groundwork for a fused
#' forward+backward graph. When on, a forward \code{ag_*} operation builds its
#' ggml node and returns a pending handle instead of computing immediately;
#' everything queued is computed as a single graph the first time any value is
#' read (the loss, on the training path) or when \code{backward()} starts.
#'
#' OFF by default: \code{GGMLR_AG_DEFER=1} or \code{TRUE} here enables it. The
#' measured gain on a training step is 1.0-1.3x (the forward is dominated by
#' weight upload, which this does not change), so it is enabled for the fused
#' graph work rather than for its own sake.
#'
#' @param on `TRUE` to defer, `FALSE` to compute per operation, `NA` to query.
#' @return The previous state, invisibly.
#' @keywords internal
ag_defer_forward <- function(on = TRUE) {
  old <- .ag_defer_enabled()
  if (!is.na(on)) .ag_defer$enabled <- isTRUE(on)
  invisible(old)
}

# OFF by default -- the inverse of the resident-gradients gate, and for a
# different reason: that one was bit-identical and measured a clear win, this
# one is a structural change whose payoff arrives with the fused graph.
.ag_defer_enabled <- function() {
  if (!is.null(.ag_defer$enabled)) return(isTRUE(.ag_defer$enabled))
  v <- Sys.getenv("GGMLR_AG_DEFER")
  identical(v, "1") || identical(toupper(v), "TRUE")
}

# Is deferral available for THIS operation?
#
# Refused for anything the barrier cannot reason about:
#   out=          the caller owns the destination and reads it directly, so the
#                 value has to exist when .ag_run_op returns
#   persistent    weights and optimizer moments outlive the pass pool; mixing
#                 them into a queue drained on a pass-pool read would tie two
#                 lifetimes together for no gain (the Adam step is already one
#                 graph, and is where the five lost iterations came from)
#   cpu / no backend  there is no graph to defer into
.ag_defer_ok <- function(scope, out) {
  .ag_defer_enabled() &&
    identical(scope, "pass") &&
    is.null(out) &&
    .ag_defer$depth == 0L &&
    !is.null(.ag_device_state$backend)
}

# Queue a built node plus the uploads its fresh operands need.
#
# Returns nothing: the caller wraps `node` in a pending handle itself, because
# only it knows the output shape.
.ag_defer_push <- function(node, uploads) {
  .ag_defer$nodes <- c(.ag_defer$nodes, list(node))
  if (length(uploads))
    .ag_defer$uploads <- c(.ag_defer$uploads, uploads)
  invisible(NULL)
}

# Queue several roots at once, with the uploads they collectively need.
#
# Used by the fused backward, which produces one root per leaf gradient rather
# than the single result an .ag_run_op has. Each root is expanded separately at
# the barrier: gradients are not reachable from one another, and an unexpanded
# root is silently skipped rather than reported.
.ag_defer_push_many <- function(nodes, uploads = list()) {
  if (!length(nodes)) return(invisible(NULL))
  .ag_defer$nodes <- c(.ag_defer$nodes, nodes)
  if (length(uploads))
    .ag_defer$uploads <- c(.ag_defer$uploads, uploads)
  invisible(NULL)
}

.ag_defer_len <- function() length(.ag_defer$nodes)

# Run `expr` with deferral suppressed, draining anything already queued first.
#
# ⚠️ WHY THIS EXISTS. Some code orders its graphs by hand, and deferral silently
# reorders them. The Adam device step is the case that proved it: it computes
# the new weight and the new moments as three separate graphs while m, v and w
# still hold the previous step's values, and only then copies the results back
# -- "ORDER MATTERS" in .ag_adam_step_device, arrived at over five iterations.
#
# Deferred, those three graphs are not computed when they appear to be. They are
# computed at the first cpy(), which is itself the write that the ordering was
# protecting them from -- so the reads and the writes end up in one graph and
# the step reads values it was carefully arranged not to see. Measured with the
# gate on: the loss froze at 0.212 across four steps and the weight came back
# bit-identical to its starting value, because every update was computed from
# operands the same graph was overwriting.
#
# The rule this encodes: deferral is safe wherever the ONLY thing that orders
# two operations is data dependency, and unsafe wherever a caller relies on
# "this graph has finished before the next one starts". The second kind has to
# opt out, because nothing in the queue can detect it.
.ag_defer_suspend <- function(expr) {
  drained <- FALSE
  if (.ag_defer_len()) { .ag_defer_drain(); drained <- TRUE }
  old <- .ag_defer$enabled
  .ag_defer$enabled <- FALSE
  on.exit({
    .ag_defer$enabled <- old
    invisible(drained)
  }, add = TRUE)
  force(expr)
}

# Drop the queue without computing it. Called when the pass pool is freed: the
# nodes queued against it are gone, and running them would touch freed memory.
.ag_defer_discard <- function() {
  .ag_defer$nodes   <- list()
  .ag_defer$uploads <- list()
  invisible(NULL)
}

# THE BARRIER. Allocate, upload, build one graph, compute.
#
# Idempotent and cheap when the queue is empty, which is how it can sit in
# .ag_handle_to_r and at the top of backward() without either having to ask
# whether deferral is on.
#
# Every queued root is expanded separately. Most are reachable from the last one
# (a chain), but not all -- two branches of a residual block are both roots --
# and an unreachable node that is not expanded is silently not computed, which
# would surface as a stale intermediate rather than as an error.
.ag_defer_drain <- function() {
  if (!length(.ag_defer$nodes)) return(invisible(FALSE))

  # Re-entry guard. The drain itself calls nothing that reads a handle, but a
  # future edit might, and recursing here would compute the same graph twice
  # and free the context underneath the outer call.
  if (.ag_defer$depth > 0L) return(invisible(FALSE))
  .ag_defer$depth <- .ag_defer$depth + 1L
  on.exit(.ag_defer$depth <- .ag_defer$depth - 1L, add = TRUE)

  nodes   <- .ag_defer$nodes
  uploads <- .ag_defer$uploads
  # Cleared BEFORE the work, not after: if the compute below fails, the queue
  # must not be retried against a context that is now in an unknown state. The
  # pending handles then error on read (their buffer holds nothing), which is
  # the loud failure the contract asks for.
  .ag_defer_discard()

  backend <- .ag_device_state$backend
  if (is.null(backend))
    stop("ggmlR: deferred forward has queued nodes but no backend to run ",
         "them on.", call. = FALSE)

  # One buffer for every node queued since the last drain. Tensors that already
  # have memory are untouched, so operands uploaded earlier keep their contents
  # and their addresses.
  # ⚠️ EVERY pass-pool context, not just the current one.
  #
  # .ag_ctx_ensure() retires a full context and opens a new one when a request
  # does not fit (R/ag_device.R), so a queue long enough -- a deep chain, or a
  # fused forward+backward -- spans several. Flushing only
  # .ag_device_state$ctx would leave the earlier ones' tensors without a
  # backend buffer, and an unallocated node is not an error: the graph computes
  # nothing there and the values read back are whatever the memory held.
  #
  # ggml_backend_alloc_ctx_tensors is a no-op for tensors that already have
  # memory, so re-flushing a context that .ag_ctx_ensure already flushed costs
  # a call and changes nothing.
  for (cx in .ag_device_state[[.ag_pool_slots("pass")$ctxs]])
    .ag_ctx_flush(cx, scope = "pass")

  # Uploads come after the allocation, never before: a tensor has nowhere to
  # write until ggml_backend_alloc_ctx_tensors has run.
  for (u in uploads) .ag_xfer_up(u$ptr, u$val, "defer operands")

  # Throwaway context for the graph itself. Ownership runs one way -- the cgraph
  # holds pointers to tensors that live in the residency context, and ggml_free
  # releases only this context's own mem_buffer -- the same argument .ag_run_op
  # makes for its per-op graph (R/ag_device.R).
  ctx_graph <- ggml_init(.ag_graph_ctx_bytes(), no_alloc = TRUE)
  if (is.null(ctx_graph))
    stop("ggmlR: failed to create a ggml context for the deferred graph.",
         call. = FALSE)
  on.exit(ggml_free(ctx_graph), add = TRUE)

  graph <- ggml_build_forward_expand(ctx_graph, nodes[[1L]])
  for (i in seq_along(nodes)[-1L]) ggml_graph_expand(graph, nodes[[i]])

  ggml_backend_graph_compute(backend, graph)
  invisible(TRUE)
}
