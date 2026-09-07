# Backward as ONE ggml graph, instead of one R closure per tape node.
#
# STATUS: ON BY DEFAULT. Set GGMLR_AG_BACKWARD_GRAPH=0 for the closure path.
#
# The balance shifted exactly where the earlier note predicted it might -- "a
# resident forward would delete the upload stage and part of the download" --
# and that is what resident weights did. Re-measured on five shapes:
# 0.97x / 2.67x / 1.17x / 1.90x / 2.03x, four of five ahead and the fifth at
# noise. The section below is kept because its reasoning still holds for the
# code it described, and because one of its conclusions must not be relitigated.
#
# WHY IT LOOKED LIKE A WIN, WHY IT THEN WAS NOT, AND WHAT CHANGED
# ---------------------------------------------------------------
# The tape's grad_fn closures capture host matrices (a_snap/b_snap) and compute
# with %*% and t() -- the largest single cost in path B. On a synthetic chain of
# 8 matmuls at 512x512 (inst/scripts/measure_ag_backward.R):
#
#     r_blas       (closures)                 40.2 ms
#     gpu_per_op   (.ag_run_op per gradient)  71.4 ms   <- 1.8x WORSE
#     gpu_one_graph, activations uploaded     19.3 ms   <- 2.1x better
#     gpu_one_graph, activations resident     11.3 ms   <- 3.6x better
#
# That is what this file was built on. It did not survive real models
# (inst/scripts/measure_ag_backward_real.R): eight of nine configurations came
# out BELOW 1x, from a 6-node MLP (0.24x) to a 109-node attention block (0.95x),
# with no monotonic trend in tensor size. The synthetic tape flattered the graph
# because it hoisted the activation upload out of the timed region and paid
# ctx_ensure/flush once, where a real backward pays both on every call.
#
# Stage profiling (GGMLR_AG_BWD_PROF=1) says the cost is SPREAD -- actual GPU
# compute is 12-25% of backward(), the rest divided between uploading snapshots,
# downloading gradients, building nodes in R and writing $grad back. There is no
# single item whose removal turns the ratio around, which is why the direction
# was closed rather than optimised further. Graph reuse across steps does not
# help either: ggml_graph_plan is CPU-only and the Vulkan backend leaves every
# graph_plan_* hook NULL, and even a free cache removes under a fifth of the
# time.
#
# WHAT CHANGED. Two of the four costs named above are gone. ag_param() now keeps
# weights in the persistent pool, so a snapshot that is a weight is already on
# the device; and const() below returns a handle's pointer instead of building a
# tensor and queueing an upload for it. Uploading snapshots and writing $grad
# back were two of the four items the spread was divided between -- removing
# them is what moved the ratio, and it is why the "closed" verdict does not
# reproduce on the current code. The remaining two (building nodes in R,
# downloading gradients) are what a unified forward+backward graph would take.
#
# One conclusion DOES carry over and must not be relitigated: a per-op GPU
# grad_fn (dispatching each closure through .ag_run_op) is a measured 1.8x
# regression. Whatever is done here, it must not be that.
#
# WHAT IS COVERED SO FAR
# ----------------------
# matmul, add (both broadcasts), the three losses, the elementwise activations
# (relu/sigmoid/tanh), transpose, softmax, scale and elementwise mul -- enough
# for ag_linear stacks, classifiers, ag_dropout, and ag_multihead_attention,
# which is itself built from those primitives (its head slicing and
# concatenation go through selector matrices, i.e. ag_matmul).
# Everything else still runs the closure path, and a tape holding any
# unsupported op falls back wholesale rather than mixing the two. The tape and
# the existing grad_fn closures are untouched: this is a second path, not a
# replacement, so a fallback is always available.
#
# Two shapes of rule live here, and they are worth keeping distinct:
#
#   elementwise   dx = g * mult, one node, the multiplier already computed by
#                 the forward pass (relu mask, s*(1-s), 1-t^2)
#   structural    the gradient needs the operands or a reduction: matmul,
#                 add's broadcast reductions, softmax's column coupling
#
#   matmul     out = A %*% B      dA = g %*% t(B)      dB = t(A) %*% g
#   add        out = A + B        dA = g               dB = g, summed over
#                                                      whatever axis B was
#                                                      broadcast along
#   loss_const out = loss(x)      dx = gmat * gscale, both from the forward pass
#   elemwise   out = f(x)         dx = g * f'(x)
#   transpose  out = t(x)         dx = t(g)
#   softmax    out = softmax(x)   dx = p * (g - colSums(p * g))
#
# ggml_mul_mat(src0[K,M], src1[K,N]) -> [M,N] contracts ne[0] of BOTH operands,
# i.e. it computes t(src0) %*% src1. ggml_out_prod(a[m,n], b[p,n]) -> [m,p]
# contracts ne[1] instead, i.e. a %*% t(b). An R matrix [r,c] uploads as ne0=r,
# ne1=c, so which of the two an R-level rule needs depends on WHICH side carries
# the transpose:
#
#   dB = t(A) %*% g   shared axis in ne[0]  ->  mul_mat(A, g)
#   dA = g %*% t(B)   shared axis in ne[1]  ->  out_prod(g, B)
#
# Either way no ggml_transpose node appears: the transposes the closure path
# pays for on the host are already built into how these two ops read their
# operands. That is the structural advantage of emitting backward as a graph.
#
# WARNING: mixing the two up asserts in ggml_can_mul_mat on rectangular shapes
# and passes SILENTLY on square ones. It did slip through here once, because the
# PoC that established the timings used a 512x512 chain throughout. The rules
# are verified against R on distinct m/k/n in
# tests/testthat/test-ag-backward-graph.R.

.ag_bwd <- new.env(parent = emptyenv())

# Graph backward is opt-in while it covers only part of the op set.
#   NULL  = follow the default (currently: off)
#   TRUE  = use the graph path when the tape qualifies
#   FALSE = always use the closure path
.ag_bwd$enabled <- NULL

# Set by the graph path on every backward() so tests and probes can tell which
# route actually ran, without inspecting timings. One of "graph", "closures",
# or "closures (<reason>)" when the graph path was wanted but declined.
.ag_bwd$last_path <- NA_character_

# Component 3 gate: NULL means "not set in this session, consult the env var".
.ag_bwd$resident  <- NULL

#' Compute the backward pass as a single ggml graph
#'
#' Opt-in while the graph path covers only part of the op set: a tape
#' containing any operation it cannot emit runs the ordinary closure path
#' instead. Enable with `GGMLR_AG_BACKWARD_GRAPH=1` or by calling this function.
#'
#' @param on `TRUE` to use the graph path where possible, `FALSE` to always use
#'   closures, `NA` to only query the current state.
#' @return The previous state, invisibly.
#' @keywords internal
ag_backward_graph <- function(on = TRUE) {
  # isTRUE(), not the raw field: $enabled starts NULL when the setting has only
  # ever come from the environment variable, and returning NULL breaks the
  # save-and-restore idiom -- feeding it back lands in is.na(NULL), which errors
  # with "argument is of length zero" rather than restoring anything.
  old <- .ag_bwd_is_enabled()
  if (!is.na(on)) .ag_bwd$enabled <- isTRUE(on)
  invisible(old)
}

#' @rdname ag_backward_graph
#' @keywords internal
ag_backward_path <- function() .ag_bwd$last_path

# ON by default since the residency work; NULL means "nobody has set it".
#
# The measurement that switched this: closures vs graph on five shapes, after
# resident weights (stage 3.1) and after const() stopped re-uploading snapshots
# that are already device handles (stage 1) --
#   d=64  b=32  depth=2   0.97x
#   d=128 b=64  depth=3   2.67x
#   d=256 b=128 depth=2   1.17x
#   d=512 b=128 depth=2   1.90x
#   d=256 b=64  depth=6   2.03x
# Four of five ahead, the fifth at noise on the smallest shape. The earlier
# verdict (eight of nine BELOW 1x) was true of the code it was taken on: two of
# the four costs it named -- uploading snapshots and writing $grad back -- are
# what residency removed. It does not reproduce here.
#
# A tape holding an op the graph cannot emit still falls back to closures
# wholesale, so switching the default changes which path runs, not whether a
# tape can run.
.ag_bwd_is_enabled <- function() {
  if (is.null(.ag_bwd$enabled)) return(TRUE)
  isTRUE(.ag_bwd$enabled)
}

#' Keep backward gradients on the device
#'
#' Component 3 of the resident contract. When on, the graph backward leaves each
#' leaf gradient in its backend buffer and puts a device handle in \code{$grad}
#' instead of an R matrix; the numbers come back when something actually reads
#' them. That removes the two largest stages of a backward pass -- the gradient
#' download and the leaf install, together 43-64% of it on the measured models
#' (\code{inst/scripts/measure_ag_residency_on_backward.R}).
#'
#' On by default; \code{GGMLR_AG_RESIDENT_GRADS=0} or \code{FALSE} here restores
#' host-side gradients. It changes what \code{$grad} contains: code reading it
#' through \code{.ag_data}/\code{.ag_as_matrix} is unaffected, while code doing
#' arithmetic on \code{$grad} directly gets an \code{ag_handle}, which has no
#' arithmetic methods and so fails loudly rather than computing something wrong.
#'
#' @param on `TRUE` to keep gradients resident, `FALSE` to download them as
#'   before, `NA` to only query the current state.
#' @return The previous state, invisibly.
#' @keywords internal
ag_backward_resident <- function(on = TRUE) {
  # Always a length-1 logical, never NULL: the value is meant to be handed
  # straight back to restore the setting, and a NULL round-trip fails inside
  # is.na() with "argument is of length zero".
  old <- .ag_bwd_resident_grads()
  if (!is.na(on)) .ag_bwd$resident <- isTRUE(on)
  invisible(old)
}

# ON by default. GGMLR_AG_RESIDENT_GRADS=0 restores host-side gradients.
#
# What it buys: on a 2-layer MLP a step goes from 9 crossings to 5, removing the
# gradient download and its re-upload into the optimizer -- the same numbers
# crossing the bus twice. Training is bit-identical either way, which is the
# point of a transport change.
#
# ⚠️ The constraint is LIFETIME. A resident $grad lives in the pass pool, so it
# dies at the next with_grad_tape(). That suits the ordinary loop, where the
# optimizer consumes it within the same step, and it is a trap for anything
# holding gradients ACROSS a tape boundary. Two such places existed and both are
# now fixed:
#
#   dp_train()  ran a tape per replica and averaged afterwards, so replica 1's
#               handles were freed by replica 2's tape ("buffer freed by a tape
#               reset, generation 428 < 429"). It now materialises each
#               replica's gradients before returning them, and clears $grad
#               before each replica's pass -- backward() accumulates into an
#               existing $grad, which across iterations meant adding to a
#               pointer whose buffer was already gone.
#
#   ag_checkpoint()  accumulated with `inp$grad + g`, which a handle refuses.
#
# Readers inside one tape were converted earlier: the optimizers pass the handle
# straight into the device Adam step, and print, clipping and the anomaly check
# go through the accessor.
#
# A new caller that holds a gradient past a tape reset will hit a loud error,
# not a wrong number -- the handle checks its generation and has no arithmetic.
# That is the intended failure mode; the fix belongs at the call site, in the
# shape of the two above.
.ag_bwd_resident_grads <- function() {
  if (!is.null(.ag_bwd$resident)) return(isTRUE(.ag_bwd$resident))
  !identical(Sys.getenv("GGMLR_AG_RESIDENT_GRADS"), "0")
}

# Ops this file can emit. A tape is eligible only if EVERY node is in here:
# splitting one backward between a graph and closures would mean moving
# gradients across the bus mid-pass, which is the per-op round trip again.
.AG_BWD_GRAPH_OPS <- c("matmul", "add", "loss_const", "elemwise_mul",
                       "transpose", "softmax", "scale", "mul", "flash_attn")

# Can this tape run as one graph?
#
# Returns NULL when it can, or a short reason when it cannot -- the reason is
# recorded in $last_path so a probe can see WHY it fell back rather than
# guessing from a timing.
.ag_bwd_reject_reason <- function(nodes, loss) {
  if (!length(nodes)) return("empty tape")
  if (!identical(.ag_device_state$device, "gpu")) return("device is not gpu")
  if (is.null(.ag_device_state$backend))          return("no backend")

  for (nd in nodes) {
    if (is.null(nd$op)) return("tape node has no op record")
    if (!nd$op %in% .AG_BWD_GRAPH_OPS) return(paste0("unsupported op: ", nd$op))
  }

  # The loss_const rule folds the incoming gradient, which is only correct while
  # that gradient is the 1x1 seed -- i.e. while the loss node IS the tape's root.
  # A loss consumed by something else (two losses summed, a loss scaled) would
  # get a gradient of its own, and folding it would silently drop that factor.
  # Cheaper and safer to refuse the tape than to emit a wrong number.
  for (nd in nodes) {
    if (identical(nd$op, "loss_const") && !identical(nd$output_id, loss$id))
      return("loss is not the tape root")
  }
  NULL
}


# ---------------------------------------------------------------------------
# Stage profiling (opt-in, GGMLR_AG_BWD_PROF=1 or ag_backward_profile(TRUE)).
#
# The first real-model measurement found the graph path 2-5x SLOWER than the
# closures it replaces, with a large constant term: an MLP whose closure
# backward costs 0.24 ms took 1.31 ms as a graph, and the gap barely grew with
# tensor size. A constant that big is a per-CALL cost, not a per-node one, so
# the question is which of the six stages below owns it. Guessing would be
# cheap to do and expensive to be wrong about -- hence this.
#
# Off by default and checked once per backward, so a normal run pays one
# getenv-backed logical test.
# ---------------------------------------------------------------------------

.ag_bwd$prof         <- FALSE
.ag_bwd$prof_last    <- NULL   # named numeric, milliseconds per stage
.ag_bwd$prof_totals  <- NULL   # accumulated across calls
.ag_bwd$prof_n       <- 0L

#' Profile the graph backward by stage
#'
#' @param on `TRUE` to record per-stage timings on every graph backward,
#'   `FALSE` to stop, `NA` to query.
#' @return The previous state, invisibly.
#' @keywords internal
ag_backward_profile <- function(on = TRUE) {
  old <- .ag_bwd$prof
  if (!is.na(on)) {
    .ag_bwd$prof <- isTRUE(on)
    if (isTRUE(on)) ag_backward_profile_reset()
  }
  invisible(old)
}

#' @rdname ag_backward_profile
#' @keywords internal
ag_backward_profile_reset <- function() {
  .ag_bwd$prof_totals <- NULL
  .ag_bwd$prof_n      <- 0L
  invisible(NULL)
}

#' @rdname ag_backward_profile
#' @keywords internal
ag_backward_profile_report <- function() {
  if (is.null(.ag_bwd$prof_totals) || .ag_bwd$prof_n == 0L) {
    cat("no graph backward recorded (is profiling on, and did the tape qualify?)\n")
    return(invisible(NULL))
  }
  mean_ms <- .ag_bwd$prof_totals / .ag_bwd$prof_n
  total   <- sum(mean_ms)
  ord     <- order(mean_ms, decreasing = TRUE)
  cat(sprintf("graph backward, mean of %d calls: %.3f ms total\n",
              .ag_bwd$prof_n, total))
  for (i in ord)
    cat(sprintf("  %-10s %7.3f ms  %5.1f%%\n",
                names(mean_ms)[i], mean_ms[i], 100 * mean_ms[i] / total))
  invisible(mean_ms)
}

# Record one stage. Called only when profiling is on; the caller keeps a running
# clock rather than wrapping each stage in a closure, so an unprofiled run does
# not pay for the instrumentation at all.
.ag_bwd_prof_add <- function(acc, name, t0) {
  acc[[name]] <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000
  acc
}

# Add a stage recorded OUTSIDE .ag_bwd_run_graph, after that call has already
# folded its own stages into prof_totals.
#
# `subtract = TRUE` means: t0 marks the start of the whole graph branch, so the
# stage to report is the total elapsed minus everything already accounted for.
# That residual is the honest "where else did the time go" figure -- the first
# real-model profile summed to only ~80% of backward() on the large models, and
# guessing at the missing fifth would be exactly the kind of thing that has
# already gone wrong twice in this area.
.ag_bwd_prof_extra <- function(name, t0, subtract = FALSE) {
  ms <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000
  if (subtract) {
    known <- if (is.null(.ag_bwd$prof_last)) 0 else sum(.ag_bwd$prof_last)
    ms <- max(0, ms - known)
  }
  # prof_last and prof_totals are named numeric vectors, not lists: indexing a
  # missing name with [[ ]] errors instead of returning NULL, so grow them by
  # name assignment on the vector.
  add_named <- function(v, nm, x) {
    if (is.null(v)) return(stats::setNames(x, nm))
    v[nm] <- (if (nm %in% names(v)) v[[nm]] else 0) + x
    v
  }
  vl <- .ag_bwd$prof_last
  if (is.null(vl)) vl <- stats::setNames(numeric(0), character(0))
  vl[name] <- ms
  .ag_bwd$prof_last <- vl
  .ag_bwd$prof_totals <- add_named(.ag_bwd$prof_totals, name, ms)
  invisible(NULL)
}

# Build and run the backward graph.
#
# Returns the same thing backward() returns -- an environment mapping tensor id
# to gradient matrix -- or NULL if the tape does not qualify, in which case the
# caller runs the closure path.
#
# Structure mirrors the closure walk in backward(): reverse the tape, take the
# incoming gradient for each output, emit the nodes for that op's rule, and
# accumulate into the inputs. The difference is that a "gradient" here is a
# graph node, not a matrix, and nothing is computed until the single compute at
# the end.
.ag_bwd_run_graph <- function(loss, nodes) {
  reason <- .ag_bwd_reject_reason(nodes, loss)
  if (!is.null(reason)) {
    .ag_bwd$last_path <- paste0("closures (", reason, ")")
    return(NULL)
  }

  backend   <- .ag_device_state$backend
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype())

  prof <- isTRUE(.ag_bwd$prof)
  acc  <- list()
  tk   <- if (prof) Sys.time() else NULL

  # Descriptor budget: every node may need a handful of tensors (its inputs plus
  # the nodes of its rule), and uploads need one each. Sized from the tape with
  # slack, since overflowing a context aborts R inside ggml_new_tensor_impl
  # rather than returning an error we could fall back from.
  #
  # These tensors go into the SHARED persistent context and stay there until the
  # next .ag_residency_reset(). In a training loop that is every step, since
  # with_grad_tape() resets at the start of each tape -- so the backward's
  # scratch is reclaimed one step later and nothing accumulates. Calling
  # backward() repeatedly OUTSIDE a tape does accumulate, which is why the
  # budget check in .ag_ctx_ensure exists; it grows contexts rather than
  # overflowing one.
  ctx <- .ag_ctx_ensure(8L * length(nodes) + 16L)

  # Host matrices that must reach the device, collected while emitting and
  # uploaded in one pass after allocation: ggml_backend_alloc_ctx_tensors has to
  # run before any tensor has memory to write into.
  uploads <- list()

  # Materialise an R matrix as a graph input tensor.
  const <- function(m) {
    # A snapshot that is already on the device needs no tensor and no upload:
    # its pointer IS the constant. This is the other half of stage 1 -- the
    # forward stopped downloading resident operands, so the snapshots it records
    # arrive here as handles, and re-uploading them would put back the round
    # trip from the other end.
    if (.ag_is_handle(m)) {
      if (!.ag_handle_live(m))
        stop("ggmlR: a backward snapshot refers to a buffer freed since the ",
             "forward pass ran.", call. = FALSE)
      return(m$ptr)
    }
    if (is.null(dim(m))) m <- matrix(m, ncol = 1L)
    tt <- ggml_new_tensor_2d(ctx, ggml_type, nrow(m), ncol(m))
    uploads[[length(uploads) + 1L]] <<- list(ptr = tt, val = as.numeric(m))
    tt
  }

  # id -> graph node holding that tensor's accumulated gradient
  gnodes <- new.env(hash = TRUE, parent = emptyenv())

  accumulate <- function(key, node) {
    prev <- get0(key, envir = gnodes)
    assign(key, if (is.null(prev)) node else ggml_add(ctx, prev, node),
           envir = gnodes)
  }

  # Seed: dL/dL = 1. A 1x1 tensor, so the scalar rules below can multiply by it
  # in the graph rather than reading it back.
  assign(as.character(loss$id), const(matrix(1.0)), envir = gnodes)

  for (nd in rev(nodes)) {
    g <- get0(as.character(nd$output_id), envir = gnodes)
    if (is.null(g)) next          # this output did not reach the loss

    inp <- nd$inputs
    if (identical(nd$op, "matmul")) {
      A <- inp$A; B <- inp$B
      # dA = g %*% t(B), shapes [m,n]x[n,k] -> [m,k]. The shared axis is n,
      # which sits in ne[1] of both g[m,n] and B[k,n] -- so this is out_prod,
      # NOT mul_mat: ggml_out_prod(X,Y) shares ne[1] and yields X %*% t(Y),
      # while ggml_mul_mat shares ne[0] and yields t(X) %*% Y.
      #
      # Getting this backwards asserts inside ggml_can_mul_mat on rectangular
      # shapes -- and passes silently on square ones, which is exactly how it
      # first slipped through (the PoC chain was 512x512 throughout). Both
      # rules below are verified against R on m=5, k=4, n=3, all distinct.
      if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id), ggml_out_prod(ctx, g, const(nd$b_snap)))
      # dB = t(A) %*% g, shapes [k,m]x[m,n] -> [k,n]. Shared axis m sits in
      # ne[0] of both A[m,k] and g[m,n], so this one IS mul_mat -- and needs no
      # transpose node, since mul_mat already reads its first argument that way.
      if (is_ag_tensor(B) && isTRUE(B$requires_grad))
        accumulate(as.character(B$id), ggml_mul_mat(ctx, const(nd$a_snap), g))

    } else if (identical(nd$op, "add")) {
      A <- inp$A; B <- inp$B
      if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id), g)
      if (is_ag_tensor(B) && isTRUE(B$requires_grad)) {
        # B may have been broadcast over A in the forward pass -- the bias case
        # of ag_linear -- and then its gradient is g summed over the axis that
        # was broadcast. Both directions are emitted here; neither needs a
        # download.
        #
        # ggml_sum_rows reduces ne[0]: [a,b,c,d] -> [1,b,c,d]. An R matrix [r,c]
        # uploads as ne0=r, ne1=c, so sum_rows collapses R's ROWS, giving
        # colSums. That is exactly the row-broadcast case; the column one needs
        # the other axis and goes through a transpose.
        # Shapes via the accessors: b_orig is a device handle whenever the
        # forward kept it resident, and dim() on one is NULL.
        bo   <- nd$b_orig
        bdim <- .ag_dim(bo)
        gb <-
          if (!is.null(bdim) && bdim[2L] == 1L && nd$out_nc > 1L) {
            # b was [m,1] broadcast across columns -> db = rowSums(g), [m,1].
            # Reducing ne[1] is not something sum_rows does, so transpose to
            # [n,m] and reduce ne[0] instead, then reshape [1,m] back to [m,1].
            # ggml_cont before the reduction: sum_rows over a transposed VIEW is
            # a different (and here wrong) memory walk.
            gt <- ggml_cont(ctx, ggml_transpose(ctx, g))
            ggml_reshape_2d(ctx, ggml_sum_rows(ctx, gt), bdim[1L], 1L)
          } else if (!is.null(bdim) && bdim[1L] == 1L && nd$out_nr > 1L) {
            # b was [1,n] broadcast down rows -> db = colSums(g), [1,n], which
            # is sum_rows' native shape.
            ggml_sum_rows(ctx, g)
          } else {
            g
          }
        accumulate(as.character(B$id), gb)
      }

    } else if (identical(nd$op, "elemwise_mul")) {
      # Every elementwise activation whose derivative depends only on the value
      # already computed in the forward pass: dx = g * mult, one ggml_mul node.
      #
      #   relu     mult = (x > 0)
      #   sigmoid  mult = s * (1 - s)
      #   tanh     mult = 1 - t^2
      #
      # The multiplier is computed on the host, where the forward pass already
      # had it -- the closures capture exactly the same matrix. Recomputing it
      # in the graph would need the activation's input or output as a node,
      # which stage 1 does not keep; uploading it costs one transfer and keeps
      # the rule a single node. Same shape as g, so no broadcast is involved.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id), ggml_mul(ctx, g, const(nd$mult)))

    } else if (identical(nd$op, "scale")) {
      # dx = g * scalar, folded into one ggml_scale node. Distinct from
      # elemwise_mul because the multiplier is a number, not a matrix -- so
      # nothing has to be uploaded at all.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id), ggml_scale(ctx, g, nd$scalar))

    } else if (identical(nd$op, "mul")) {
      # Elementwise product of two tracked tensors: dA = g * B, dB = g * A.
      # Unlike elemwise_mul, BOTH operands can require a gradient, and each
      # one's rule needs the OTHER operand's forward value -- the same
      # a_snap/b_snap pairing matmul uses.
      #
      # Broadcasting is refused rather than emitted. ag_mul's closure reduces a
      # broadcast gradient with colSums/rowSums over the pre-expansion shape,
      # and getting that wrong produces a plausible but incorrect gradient. No
      # caller in the package broadcasts through ag_mul today (dropout, the
      # attention mask and batch_norm's gamma all pass matching shapes), so the
      # case is declined until something needs it.
      A <- inp$A; B <- inp$B
      bc <- !identical(.ag_dim(nd$a_orig), .ag_dim(nd$b_orig))
      if (bc) {
        .ag_bwd$last_path <- "closures (mul: broadcast)"
        return(NULL)
      }
      if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id), ggml_mul(ctx, g, const(nd$b_snap)))
      if (is_ag_tensor(B) && isTRUE(B$requires_grad))
        accumulate(as.character(B$id), ggml_mul(ctx, g, const(nd$a_snap)))

    } else if (identical(nd$op, "transpose")) {
      # dx = t(g). ggml_transpose only relabels ne/nb, so the result is a view;
      # ggml_cont materialises it, which the consumers (and the download at the
      # end) need. Cheap: one copy, no arithmetic.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id), ggml_cont(ctx, ggml_transpose(ctx, g)))

    } else if (identical(nd$op, "softmax")) {
      # dx = p * (g - colSums(p * g)), the softmax Jacobian applied to g.
      #
      # Not an elementwise multiplier like relu/tanh: the column sum couples
      # every entry of a column, so this needs four nodes rather than one. The
      # reduction is sum_rows (which collapses ne[0], i.e. R's rows, giving
      # colSums), and ggml_sub broadcasts the resulting [1,n] back over [m,n] --
      # verified on this backend rather than assumed.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad)) {
        p   <- const(nd$p_snap)
        dot <- ggml_sum_rows(ctx, ggml_mul(ctx, p, g))      # [1, n]
        accumulate(as.character(x$id),
                   ggml_mul(ctx, p, ggml_sub(ctx, g, dot)))
      }
    } else if (identical(nd$op, "flash_attn")) {
      # Attention is one opaque node here, and deliberately so.
      #
      # Every other rule in this file spells its gradient out in ggml
      # primitives. This one delegates to R/ag_flash_attn.R and takes three
      # gradients back, already in [d_model, seq] form. The reason is the
      # boundary the industry draws in the same place: a fused attention kernel
      # is integrated as a single node with a known forward/backward contract,
      # not decomposed into head permutations the surrounding graph can see
      # through. ggml supplies both halves of that kernel already
      # (ggml_flash_attn_ext / ggml_flash_attn_back).
      #
      # Concretely, this branch knows the op's name and nothing else -- not
      # d_head, not the head order, not the 16-byte alignment of the packed
      # gradient buffer. All of that is one file away, where the file header
      # records that the layout was verified against an independent
      # implementation rather than read off ggml.h. Knowledge with that history
      # should break in one place when ggml changes, not two.
      #
      # Nothing crosses the bus: `g` is a node, the operands are the nodes the
      # forward built, and the three results are nodes. So a tape with
      # attention now qualifies for the fused forward+backward graph like any
      # other -- which is the whole reason the missing `op` mattered.
      gr <- .ag_flash_build_bwd(ctx, g, nd)
      for (nm in c("q", "k", "v")) {
        tt <- inp[[nm]]
        if (is_ag_tensor(tt) && isTRUE(tt$requires_grad))
          accumulate(as.character(tt$id), gr[[nm]])
      }

    } else if (identical(nd$op, "loss_const")) {
      # All three losses share one shape of gradient: a matrix the forward pass
      # already built, times a scalar. Only the matrix and the scalar differ, so
      # they share one rule rather than getting three near-identical ones.
      #
      #   mse                    dpred   = (2/n) * diff
      #   cross_entropy          dpred   = (1/n) * (-t / p)
      #   softmax_cross_entropy  dlogits = (1/n) * (p - t)
      #
      # The incoming gradient g is dropped deliberately: a loss sits at the root
      # of the tape, so g is the seed 1x1 identity. ggml_mul does not broadcast a
      # 1x1 node over a matrix, and folding a known 1.0 is free -- but that makes
      # this rule valid ONLY at the root. .ag_bwd_reject_reason enforces that: a
      # loss feeding anything else would need the multiply, and is refused.
      tgt <- inp[[1L]]
      if (is_ag_tensor(tgt) && isTRUE(tgt$requires_grad))
        accumulate(as.character(tgt$id),
                   ggml_scale(ctx, const(nd$gmat), nd$gscale))
    }
  }

  # Nothing to compute: no parameter reached the loss.
  keys <- ls(gnodes)
  outs <- Filter(Negate(is.null), lapply(keys, function(k) get0(k, envir = gnodes)))
  if (!length(outs)) {
    .ag_bwd$last_path <- "closures (no gradients)"
    return(NULL)
  }

  if (prof) { acc <- .ag_bwd_prof_add(acc, "emit", tk); tk <- Sys.time() }

  # FUSED PATH: hand the gradient roots to the forward's queue instead of
  # computing them here, so one graph covers the forward and the backward.
  #
  # This is possible only because nothing above reads a forward VALUE. Every
  # snapshot reaches const() as a handle and contributes its pointer, and the
  # shape decisions (.ag_dim on b_orig, the broadcast branches) read recorded
  # shapes, not contents. So the backward can be expressed entirely as nodes
  # over tensors the forward has not computed yet.
  #
  # What made it impossible until now was the loss, not the backward: MSE
  # computed its scalar with resident = FALSE and divided on the host, so the
  # forward had to run before ag_mse_loss() returned. With the division folded
  # into the graph (.ag_gpu_mse_parts) nothing pulls the forward early, and the
  # single compute happens when the optimizer reads a gradient.
  #
  # Requires resident gradients: the leaf loop below has to hand back handles,
  # since a download here would be exactly the early compute this avoids. When
  # gradients are materialised, the graph must run now.
  fuse <- .ag_defer_enabled() && .ag_defer_len() > 0L &&
          .ag_bwd_resident_grads()

  if (fuse) {
    # Queued, not computed. The uploads travel with them: the barrier flushes
    # the context and fills every operand before it builds the graph, which is
    # the same order this function used, just deferred as a unit.
    .ag_defer_push_many(outs, uploads)
    if (prof) { acc <- .ag_bwd_prof_add(acc, "queue", tk); tk <- Sys.time() }
  } else {

  # One buffer for every tensor built above, then the uploads.
  .ag_ctx_flush(ctx)
  if (prof) { acc <- .ag_bwd_prof_add(acc, "flush", tk); tk <- Sys.time() }
  for (u in uploads) .ag_xfer_up(u$ptr, u$val, "bwd_graph operands")
  if (prof) { acc <- .ag_bwd_prof_add(acc, "upload", tk); tk <- Sys.time() }

  # The graph gets a throwaway context of its own: it holds pointers to tensors
  # that outlive it, and ggml_free releases only the context's own mem_buffer
  # (same ownership rule .ag_run_op relies on, R/ag_device.R:406-421).
  ctx_graph <- ggml_init(.ag_graph_ctx_bytes(), no_alloc = TRUE)
  if (is.null(ctx_graph)) {
    .ag_bwd$last_path <- "closures (no graph context)"
    return(NULL)
  }
  on.exit(ggml_free(ctx_graph), add = TRUE)

  # Each gradient is a separate output, unreachable from the others, so each
  # needs its own expand or it would be pruned.
  graph <- ggml_build_forward_expand(ctx_graph, outs[[1L]])
  for (i in seq_along(outs)[-1L]) ggml_graph_expand(graph, outs[[i]])
  if (prof) { acc <- .ag_bwd_prof_add(acc, "build", tk); tk <- Sys.time() }

  ggml_backend_graph_compute(backend, graph)
  if (prof) { acc <- .ag_bwd_prof_add(acc, "compute", tk); tk <- Sys.time() }

  }

  # Download -- but only the gradients anyone will actually read.
  #
  # This is where the graph path lost its first real-model measurement: it was
  # bringing back a gradient for EVERY tensor on the tape (62 of them for a
  # 4-head attention block) when the caller needs the leaves only (5). Profiling
  # put 51-61% of the whole backward in this loop, against 12-20% for the actual
  # compute -- so the graph was mostly a device-to-host transfer of numbers
  # nobody would look at.
  #
  # Intermediate gradients exist to feed the nodes downstream of them, and that
  # already happened inside the graph. Only leaves reach the caller:
  # .ag_bwd_write_leaf_grads writes $grad for tape inputs with requires_grad,
  # and the optimizers read those. The closure path never paid this because its
  # intermediates stay as R matrices -- no bus in between.
  #
  # A leaf is a tracked tensor that is not itself the output of any tape node.
  produced <- vapply(nodes, function(nd) as.character(nd$output_id), character(1))
  wanted   <- character(0)
  for (nd in nodes)
    for (i in nd$inputs)
      if (is_ag_tensor(i) && isTRUE(i$requires_grad))
        wanted <- c(wanted, as.character(i$id))
  wanted <- setdiff(unique(wanted), produced)

  # Component 3 of the resident contract: leave the gradients on the device.
  #
  # This loop was the largest single stage of a backward pass -- 24-38% as
  # "download", plus another 12-35% that showed up as "leaf_fetch" when the
  # values were installed on the leaves. Both are the same transfer, and both
  # are avoidable: nothing here needs the numbers. The optimizer reads $grad
  # eventually, and .ag_data()/.ag_as_matrix materialise then, once, rather
  # than for every leaf on every pass.
  #
  # Gated because it changes what $grad IS. Any code doing arithmetic on $grad
  # directly gets a handle, which has no Ops methods and therefore errors
  # loudly (rule 3 of the data contract) instead of computing something wrong.
  # The gate stays until every reader goes through .ag_as_matrix.
  # Surviving the next tape reset is handled where the gradient is installed,
  # not here: .ag_bwd_write_leaf_grads registers a resident $grad and
  # .ag_residency_reset materialises the register before freeing the buffers.
  # An earlier attempt refused residency whenever a leaf already had a gradient,
  # which did not work -- by then the PREVIOUS pass's handle was already dead,
  # so the check fired one pass too late and gave up the optimisation as well.
  resident <- .ag_bwd_resident_grads()

  grads <- new.env(hash = TRUE, parent = emptyenv())
  for (k in intersect(keys, wanted)) {
    nd <- get0(k, envir = gnodes)
    if (is.null(nd)) next
    # ne is (ne0, ne1, ne2, ne3); a 2D gradient lives in the first two, and ne0
    # is R's row count because that is how the matrices were uploaded.
    ne <- ggml_tensor_shape(nd)
    # On the fused path the node has not been computed yet, so its handle is
    # pending: reading it drains the queue, which is what makes the single
    # compute happen at the optimizer rather than here.
    val <- if (resident) .ag_handle(nd, c(ne[1L], ne[2L]), pending = fuse)
           else matrix(.ag_xfer_down(nd, "bwd_graph leaf grads"), ne[1L], ne[2L])
    assign(k, val, envir = grads)
  }

  if (prof) {
    acc <- .ag_bwd_prof_add(acc, "download", tk)
    v <- unlist(acc)
    .ag_bwd$prof_last <- v
    .ag_bwd$prof_totals <- if (is.null(.ag_bwd$prof_totals)) v else {
      # Stage names are fixed, so a plain add is safe; guard anyway in case a
      # future early return skips one.
      m <- .ag_bwd$prof_totals
      for (nm in names(v)) m[[nm]] <- (m[[nm]] %||% 0) + v[[nm]]
      m
    }
    .ag_bwd$prof_n <- .ag_bwd$prof_n + 1L
  }

  .ag_bwd$last_path <- "graph"
  grads
}
