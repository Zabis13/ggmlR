# Diagnostics: what does the gradient tape hold alive, and how much does it weigh?
#
# Every ag_* operation that touches a tensor with requires_grad records a node
# on .ag_tape (R/autograd.R, ag_record). A node keeps three kinds of thing
# alive until the tape is cleared:
#
#   inputs   the operand ag_tensors themselves. These are environments, so the
#            node holds a reference, not a copy -- a weight matrix referenced
#            by forty nodes is stored once.
#   grad_fn  an R closure. Its environment captures whatever the backward rule
#            needs, which for most ops is a SNAPSHOT of a forward value:
#            a_snap/b_snap in ag_matmul, p_snap in ag_softmax, s_snap in
#            ag_sigmoid, diff_snap in ag_mse_loss. These are the activations.
#   fields   op plus the named extras passed through ... , which are the same
#            snapshots again for the graph backward path (R/ag_backward_graph.R).
#
# The tape is cleared only by optimizer$zero_grad(), which runs AFTER step().
# So the peak is the whole tape at once, and on a GPU-resident tape the same
# holds in VRAM ("a resident tape has no obvious ceiling", R/ag_device.R:313).
#
# Why this needs measuring rather than reasoning. The three kinds above share
# storage in ways that make a naive sum wrong in both directions:
#
#   * a snapshot captured by the closure and passed again through ... is ONE
#     matrix with two references. object.size() counts it twice -- verified:
#     a 1000x1000 matrix in a list under two names reports 15.3 MB, not 7.6.
#   * an input tensor referenced by many nodes is likewise counted once per
#     node by a naive walk.
#   * an operand's value and the snapshot a closure captured of it are often
#     the SAME object, not a copy -- verified by address: a_data and a_snap in
#     an ag_matmul node share one address. So the interesting question is not
#     what the closures reference but what only they reference, since the rest
#     survives the tape either way.
#
# All three are handled by deduplicating on the object's address (tracemem) and
# charging shared storage to the longest-lived category. The reported
# activation figure is then what clearing the tape would actually free, which
# is the number a lifetime decision needs.
#
# This is a report, not a fix. It exists so that "activation lifetime
# management" can be decided on the shape of a real tape -- which ops dominate,
# whether snapshots or inputs are the bulk -- rather than on the assumption
# that a long tape must be the problem.

# Address of an object, used only to tell shared storage from distinct storage.
#
# tracemem() returns "<0x...>" and is available in base R (lobstr is not a
# dependency). It marks the object for copy tracing, which costs nothing here:
# the objects are snapshots that the tape does not write to. On a build without
# memory profiling tracemem() errors, so the fallback treats every object as
# distinct -- that over-counts shared storage rather than hiding it, which is
# the safe direction for a memory report.
.ag_obj_addr <- function(x) {
  tryCatch(tracemem(x), error = function(e) NA_character_)
}

# Bytes held by a set of objects, counting shared storage once.
#
# `seen` is an environment used as a set of addresses; passing the same one
# across calls is what makes deduplication work across a whole tape rather than
# only within a single node.
.ag_bytes_new <- function(objs, seen) {
  total <- 0
  for (o in objs) {
    if (is.null(o)) next
    if (!is.numeric(o) && !is.logical(o) && !is.complex(o)) next
    addr <- .ag_obj_addr(o)
    if (!is.na(addr)) {
      if (!is.null(get0(addr, envir = seen, inherits = FALSE))) next
      assign(addr, TRUE, envir = seen)
    }
    total <- total + as.numeric(object.size(o))
  }
  total
}

# The snapshots a node's grad_fn closure captured.
#
# The backward rules capture their snapshots as ordinary local variables, so
# the closure's environment lists them. Only atomic values are counted:
# captured ag_tensors are references to the same environments already counted
# as inputs, and counting them here would attribute one object to two
# categories.
.ag_node_snapshots <- function(node) {
  env <- environment(node$grad_fn)
  if (is.null(env) || identical(env, emptyenv())) return(list())
  out <- list()
  for (nm in ls(env, all.names = TRUE)) {
    v <- get0(nm, envir = env, inherits = FALSE)
    if (is.numeric(v) || is.logical(v)) out[[length(out) + 1L]] <- v
  }
  out
}

# The extras recorded alongside the closure for the graph backward path.
.ag_node_fields <- function(node) {
  known <- c("output_id", "grad_fn", "inputs", "op")
  nms   <- setdiff(names(node), known)
  out   <- list()
  for (nm in nms) {
    v <- node[[nm]]
    if (is.numeric(v) || is.logical(v)) out[[length(out) + 1L]] <- v
  }
  out
}

#' Report what the gradient tape is holding
#'
#' Walks the current tape and reports how many nodes it holds and how much
#' memory those nodes keep alive, split into three categories: the operand
#' tensors, which outlive the tape; the activations captured by the backward
#' closures, which clearing the tape would release; and any extras recorded for
#' the graph backward path beyond those.
#'
#' Storage shared between categories is counted once, and charged to the
#' longest-lived one. That matters for reading the result: a weight's matrix
#' and the \code{a_snap} an \code{ag_matmul} closure captured are the same
#' object, so it is reported as an operand. The activation figure is therefore
#' what clearing the tape would actually free, not the sum of everything the
#' closures reference.
#'
#' The tape lives from the start of \code{\link{with_grad_tape}} until
#' \code{zero_grad()} clears it, which happens after the optimizer step. Call
#' this between \code{\link{backward}} and \code{zero_grad()} to see the peak.
#'
#' @param top Number of operations to list in the per-op breakdown, most
#'   expensive first. Set to 0 for totals only.
#' @param quiet Suppress the printed report and return the figures only. For
#'   callers that consume the numbers -- tests, scripts that format their own
#'   table -- rather than read them.
#' @return Invisibly, a list with \code{nodes}, \code{bytes_total},
#'   \code{bytes_snapshots}, \code{bytes_fields}, \code{bytes_inputs} and a
#'   data frame \code{by_op}. Printed as a report as a side effect.
#' @importFrom utils object.size
#' @export
#' @examples
#' \donttest{
#' w <- ag_param(matrix(runif(64), 8, 8))
#' x <- ag_tensor(matrix(runif(64), 8, 8))
#' with_grad_tape({
#'   h    <- ag_relu(ag_matmul(w, x))
#'   loss <- ag_mse_loss(h, matrix(0, 8, 8))
#' })
#' backward(loss)
#' ag_tape_memory()
#' }
ag_tape_memory <- function(top = 10L, quiet = FALSE) {
  nodes <- .ag_tape$nodes
  n     <- length(nodes)

  # One `seen` set for the whole walk: an object counted once is not counted
  # again in another category. The ORDER decides who is charged for storage
  # that several categories reference, and it is not arbitrary.
  #
  # Operands go first. A weight's matrix is literally the same object as the
  # a_snap the closure captured -- verified by address: a_data and a_snap in an
  # ag_matmul node share one address. Charging that to activations would report
  # memory as freeable when clearing the tape cannot free it: the parameter
  # outlives the tape and keeps its value alive regardless. Counting operands
  # first means the activation figure is what a lifetime change would ACTUALLY
  # release, which is the number this report exists to produce.
  seen <- new.env(hash = TRUE, parent = emptyenv())

  b_snap <- 0; b_field <- 0; b_input <- 0
  op_bytes <- new.env(hash = TRUE, parent = emptyenv())
  op_count <- new.env(hash = TRUE, parent = emptyenv())

  for (node in nodes) {
    op <- node$op %||% "<closure only>"

    # Inputs first (see the note on `seen` above). They are ag_tensors: count
    # the value each one holds, once per distinct tensor across the whole tape.
    # A tensor whose value is only on the device has no host matrix to measure,
    # and is skipped rather than downloaded -- a diagnostic must not move data.
    vals <- list()
    for (inp in node$inputs) {
      if (!is_ag_tensor(inp)) next
      d <- inp$data
      if (!is.null(d)) vals[[length(vals) + 1L]] <- d
    }
    i <- .ag_bytes_new(vals, seen)

    s <- .ag_bytes_new(.ag_node_snapshots(node), seen)
    f <- .ag_bytes_new(.ag_node_fields(node),    seen)

    b_snap  <- b_snap  + s
    b_field <- b_field + f
    b_input <- b_input + i

    prev_b <- get0(op, envir = op_bytes, inherits = FALSE) %||% 0
    prev_c <- get0(op, envir = op_count, inherits = FALSE) %||% 0L
    assign(op, prev_b + s + f + i, envir = op_bytes)
    assign(op, prev_c + 1L,        envir = op_count)
  }

  total <- b_snap + b_field + b_input

  ops <- ls(op_bytes)
  by_op <- data.frame(
    op    = ops,
    nodes = vapply(ops, function(o) get(o, envir = op_count), integer(1)),
    mb    = vapply(ops, function(o) get(o, envir = op_bytes), numeric(1)) / 1024^2,
    stringsAsFactors = FALSE
  )
  by_op <- by_op[order(-by_op$mb), , drop = FALSE]
  rownames(by_op) <- NULL

  mb <- function(b) b / 1024^2
  if (!isTRUE(quiet)) {
    cat(sprintf("Gradient tape: %d node%s, %.2f MB held alive\n",
                n, if (n == 1L) "" else "s", mb(total)))
    if (n == 0L) {
      cat("  (empty -- the tape is cleared by zero_grad(), so call this",
          "between backward() and the step)\n")
    } else {
      cat(sprintf("  operands (outlive the tape)     : %8.2f MB\n", mb(b_input)))
      cat(sprintf("  activations (freed by clearing) : %8.2f MB\n", mb(b_snap)))
      cat(sprintf("  graph-path fields (beyond those): %8.2f MB\n", mb(b_field)))
      if (top > 0L && nrow(by_op) > 0L) {
        k <- min(as.integer(top), nrow(by_op))
        cat(sprintf("\n  by operation (top %d):\n", k))
        cat("    op                     nodes        MB\n")
        for (i in seq_len(k)) {
          cat(sprintf("    %-20s %7d  %8.2f\n",
                      by_op$op[i], by_op$nodes[i], by_op$mb[i]))
        }
      }
    }
  }

  invisible(list(nodes           = n,
                 bytes_total     = total,
                 bytes_snapshots = b_snap,
                 bytes_fields    = b_field,
                 bytes_inputs    = b_input,
                 by_op           = by_op))
}
