# Where does the forward pass spend its time?
#
# Why this exists. The backward pass has had per-stage profiling since the
# graph path was built, and it is what made the residency work decidable: it
# said 65-78% of a backward was transfer, and which stages. The forward pass
# has had nothing, and that turned into a measurable blind spot -- on a
# 4-layer 1024-wide model the profiled backward accounts for 62 ms while a full
# training step takes 95, so 36-44 ms are simply unattributed.
#
# That gap is the reason component 4 (making the ag_* helpers pass handles
# rather than matrices, ~21 of them) is not started. Three times this session a
# ceiling estimated from partial measurements came in at roughly a third of the
# prediction -- checkpointing memory, the upload cache, and component 3 itself.
# The common cause each time was a cost that no stage was measuring. So: measure
# the forward first, then recompute the ceiling on real numbers.
#
# What it measures. Every .ag_run_op call is one forward operation, and it has
# the same shape as the backward stages:
#
#   ctx       getting a context with room (may roll over or allocate)
#   create    building tensors for the operands
#   flush     allocating the backend buffer
#   upload    sending operand data -- what component 4 would remove
#   graph     building the one-node graph
#   compute   the arithmetic
#   download  reading the result back -- what component 4 would remove
#
# Off unless switched on, and the check is one `if` on a field that is FALSE,
# so an ordinary session pays nothing.

.ag_fwd <- new.env(parent = emptyenv())
.ag_fwd$prof     <- FALSE
.ag_fwd$totals   <- NULL   # named numeric, milliseconds per stage
.ag_fwd$n        <- 0L     # operations profiled
.ag_fwd$by_op    <- NULL   # named numeric, total ms per ggml op name

#' Profile the forward pass by stage
#'
#' Records where each \code{ag_*} operation spends its time on the GPU path:
#' context handling, tensor creation, buffer allocation, operand upload, graph
#' building, compute, and result download.
#'
#' The upload and download stages are what a resident forward would remove, so
#' their share is the ceiling for that work -- measured rather than estimated.
#'
#' @param on `TRUE` to record, `FALSE` to stop, `NA` to query.
#' @return The previous state, invisibly.
#' @keywords internal
ag_forward_profile <- function(on = TRUE) {
  old <- isTRUE(.ag_fwd$prof)
  if (!is.na(on)) {
    .ag_fwd$prof <- isTRUE(on)
    if (isTRUE(on) && is.null(.ag_fwd$totals)) ag_forward_profile_reset()
  }
  invisible(old)
}

#' @rdname ag_forward_profile
#' @keywords internal
ag_forward_profile_reset <- function() {
  .ag_fwd$totals <- NULL
  .ag_fwd$n      <- 0L
  .ag_fwd$by_op  <- NULL
  invisible(NULL)
}

#' @rdname ag_forward_profile
#' @keywords internal
ag_forward_profile_report <- function() {
  if (is.null(.ag_fwd$totals) || .ag_fwd$n == 0L) {
    cat("no forward operations recorded (is profiling on, and is the device GPU?)\n")
    return(invisible(NULL))
  }
  ms    <- .ag_fwd$totals
  total <- sum(ms)
  cat(sprintf("forward: %d operations, %.2f ms total, %.3f ms per op\n",
              .ag_fwd$n, total, total / .ag_fwd$n))
  ord <- order(ms, decreasing = TRUE)
  for (i in ord)
    cat(sprintf("  %-9s %8.2f ms  %5.1f%%\n",
                names(ms)[i], ms[i], 100 * ms[i] / total))

  # The number component 4 turns on: transfer is what handles remove.
  moved <- sum(ms[intersect(names(ms), c("upload", "download"))])
  cat(sprintf("  -> transfer (upload+download) %.1f%% of the forward\n",
              100 * moved / total))
  cat(sprintf("  -> a resident forward could not beat %.2fx on this workload\n",
              total / max(total - moved, 1e-9)))

  invisible(ms)
}

# Fold one operation's stage timings into the totals.
#
# `acc` is a named numeric built by the caller; op_name is optional and only
# used for the per-op breakdown.
.ag_fwd_prof_record <- function(acc, op_name = NULL) {
  add_named <- function(v, nm, x) {
    if (is.null(v)) return(stats::setNames(x, nm))
    v[nm] <- (if (nm %in% names(v)) v[[nm]] else 0) + x
    v
  }
  t <- .ag_fwd$totals
  for (nm in names(acc)) t <- add_named(t, nm, acc[[nm]])
  .ag_fwd$totals <- t
  .ag_fwd$n      <- .ag_fwd$n + 1L
  if (!is.null(op_name))
    .ag_fwd$by_op <- add_named(.ag_fwd$by_op, op_name, sum(acc))
  invisible(NULL)
}
