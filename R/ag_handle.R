# Device handles: a value that stays on the GPU between ag_* operations.
#
# Why this exists. Path B materialises every intermediate into an R matrix, so
# each ag_* call uploads its operands and downloads its result. Measured, that
# transfer is 65-78% of a backward pass and the arithmetic is 3-10%
# (inst/scripts/measure_ag_residency_on_backward.R). A resident ceiling on the
# same shapes runs 2.87-4.61x faster than today's path, and beats the CPU where
# today's path loses to it (inst/scripts/measure_ag_residency_ceiling.R).
#
# An ag_handle is the thing that lets a value skip the round trip: it names a
# tensor already living in the residency context, so the next operation can use
# it as an operand without re-uploading, and a chain of operations only pays a
# download when someone actually asks for the numbers.
#
# What it is NOT. It is not an ag_tensor (no tape, no grad, no id) and not a
# value: it carries no data of its own, only a pointer plus what is needed to
# tell whether that pointer is still good. Handles are internal -- nothing
# outside R/ag_device.R and R/autograd.R should see one.
#
# Generation. A pointer is meaningful only while its context lives.
# .ag_residency_reset() frees the contexts and bumps ctx_gen, and an R object
# can easily outlive that, so every handle carries the generation it was made
# in and is checked before use. This mirrors the rule ag_tensor already follows
# (inst/docs/ag_data_contract.md, rule 5): a pointer and its generation travel
# together.
#
# Arithmetic. A handle deliberately has no Ops methods. Rule 3 of the data
# contract spells out why: if `h - lr * g` silently did something instead of
# erroring, a stale-value bug would become a wrong answer rather than a loud
# failure. Bare externalptr already errors in arithmetic; adding methods here
# would take that away.

# Build a handle for a tensor pointer that lives in the current residency
# context. `shape` is c(nrow, ncol) as R sees it -- ggml keeps ne0 as the row
# count because that is how matrices are uploaded.
#
# `scope` names the pool the pointer was allocated from, because the two pools
# have independent lifetimes and independent generation counters. A handle
# checked against the wrong counter would either look stale while its memory is
# fine (a persistent weight after a pass reset) or, worse, look live after its
# buffer was freed. Carrying the pool with the pointer is what keeps the check
# meaningful; the default is "pass", so existing callers are unaffected.
#
# `pending` marks a handle whose tensor has been BUILT but not yet computed --
# the deferred forward (R/ag_defer.R) queues nodes and runs them as one graph.
# Such a pointer is allocated and safe to use as an operand of another node, but
# reading its numbers before the barrier would return whatever the buffer held.
# Every read path goes through .ag_handle_to_r, which drains the queue first, so
# the flag exists to make a bypass loud rather than to be checked by callers.
.ag_handle <- function(ptr, shape, scope = "pass", pending = FALSE) {
  structure(list(ptr     = ptr,
                 shape   = as.integer(shape),
                 scope   = scope,
                 pending = isTRUE(pending),
                 gen     = .ag_scope_gen(scope)),
            class = "ag_handle")
}

.ag_is_handle <- function(x) inherits(x, "ag_handle")

# TRUE while this handle names a node that has been queued but not computed.
# Handles made before deferral existed have no field and are never pending.
.ag_handle_pending <- function(h) isTRUE(h$pending)

# Pool a handle belongs to. Handles made before scopes existed are pass-pool.
.ag_handle_scope <- function(h) h$scope %||% "pass"

# TRUE while the handle's pointer still belongs to the live context of ITS pool.
.ag_handle_live <- function(h) {
  .ag_is_handle(h) && identical(h$gen, .ag_scope_gen(.ag_handle_scope(h)))
}

# Materialise a handle into an R matrix. This is the download; call it only
# when the numbers are actually needed.
.ag_handle_to_r <- function(h) {
  if (!.ag_is_handle(h)) stop("ggmlR: not an ag_handle.", call. = FALSE)
  # The deferred forward's barrier, and the reason deferral is safe: asking for
  # the numbers is exactly the event that makes them have to exist. Everything
  # queued so far is computed as one graph here, before the download below.
  #
  # Placed at the READ rather than at the end of the tape so a chain of ag_*
  # calls stays lazy: on the training path the first read is the loss scalar,
  # by which point the whole forward is one graph. Draining costs nothing when
  # the queue is empty, which is every call on the non-deferred path.
  if (.ag_handle_pending(h)) .ag_defer_drain()
  if (!.ag_handle_live(h))
    stop("ggmlR: this device handle refers to a buffer freed by a tape reset ",
         "(", .ag_handle_scope(h), " pool, generation ", h$gen %||% NA, " < ",
         .ag_scope_gen(.ag_handle_scope(h)), ").",
         call. = FALSE)
  matrix(.ag_xfer_down(h$ptr, "handle_to_r"),
         nrow = h$shape[1L], ncol = h$shape[2L])
}

# Accept either form and return an R matrix. The bridge used wherever code has
# not been converted to handles yet.
.ag_as_matrix <- function(x) if (.ag_is_handle(x)) .ag_handle_to_r(x) else x

# Rows/cols of either form, without materialising a handle.
.ag_nrow <- function(x) if (.ag_is_handle(x)) x$shape[1L] else nrow(x)
.ag_ncol <- function(x) if (.ag_is_handle(x)) x$shape[2L] else ncol(x)
.ag_dim  <- function(x) if (.ag_is_handle(x)) x$shape      else dim(x)

#' @export
print.ag_handle <- function(x, ...) {
  cat(sprintf("<ag_handle %dx%d, %s pool, generation %s%s%s>\n",
              x$shape[1L], x$shape[2L], .ag_handle_scope(x), format(x$gen),
              if (.ag_handle_pending(x)) ", PENDING" else "",
              if (.ag_handle_live(x)) "" else ", STALE"))
  invisible(x)
}
