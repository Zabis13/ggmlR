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
# (relu/sigmoid/tanh, plus exp and clamp through the same multiplier rule),
# sub, log, sum/mean, transpose, softmax, scale and elementwise mul -- enough
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
#' @noRd
ag_backward_graph <- function(on = TRUE) {
  # isTRUE(), not the raw field: $enabled starts NULL when the setting has only
  # ever come from the environment variable, and returning NULL breaks the
  # save-and-restore idiom -- feeding it back lands in is.na(NULL), which errors
  # with "argument is of length zero" rather than restoring anything.
  old <- .ag_bwd_is_enabled()
  if (!is.na(on)) .ag_bwd$enabled <- isTRUE(on)
  invisible(old)
}

#' @noRd
#' @keywords internal
ag_backward_path <- function() .ag_bwd$last_path

# ---------------------------------------------------------------------------
# Fallbacks to the closure path, counted per session.
#
# A fallback is correct but slow, and silent: the gradients are right, only the
# speed is gone. Counting the reasons makes that visible; strict mode makes the
# worst kind -- an ERROR inside the graph path, which is a bug -- fail loudly.
# ---------------------------------------------------------------------------
.ag_bwd$fallbacks <- list()

.ag_bwd_note_fallback <- function(reason) {
  if (is.null(reason)) return(invisible(NULL))
  n <- .ag_bwd$fallbacks[[reason]]
  .ag_bwd$fallbacks[[reason]] <- if (is.null(n)) 1 else n + 1
  invisible(NULL)
}

# Strict mode: an error in the graph backward is raised instead of falling back.
# GGMLR_AG_BWD_STRICT=1 or options(ggmlR.ag_backward_strict = TRUE). Refusals
# (an unsupported op, f16) still fall back -- they are not bugs.
.ag_bwd_strict <- function() {
  isTRUE(getOption("ggmlR.ag_backward_strict")) ||
    identical(Sys.getenv("GGMLR_AG_BWD_STRICT"), "1")
}

#' Why backward() used the closure path, counted
#'
#' Each time \code{backward()} could have run as one graph but fell back to the
#' R closures, the reason is counted here: an unsupported op on the tape, a
#' compute dtype other than f32, or an error inside the graph path (which strict
#' mode, \code{GGMLR_AG_BWD_STRICT=1}, raises instead). Gradients are correct
#' either way; a fallback costs speed.
#'
#' @param reset \code{TRUE} to clear the counts after reading them.
#' @return A named numeric vector: reason -> number of backward() calls.
#' @noRd
ag_backward_fallbacks <- function(reset = FALSE) {
  v <- unlist(.ag_bwd$fallbacks)
  if (is.null(v)) v <- numeric(0)
  if (isTRUE(reset)) .ag_bwd$fallbacks <- list()
  v
}

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
#' @noRd
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
                       "transpose", "softmax", "scale", "mul", "flash_attn",
                       "sub", "log", "sum", "log_softmax", "slice_rows",
                       "concat_rows", "silu", "rms_norm", "conv2d", "reshape", "permute", "conv_transpose2d",
                       "layer_norm", "pow", "embedding", "batch_norm")

# Does the current device backend run GGML_OP_GET_ROWS_BACK (ag_embedding's
# rule)? On Vulkan it needs float atomics, which not every device has. Asked
# once per backend with a shape-only probe node, cached.
.ag_bwd_get_rows_back_ok <- function() {
  backend <- .ag_device_state$backend
  cache <- .ag_bwd$grb_ok
  if (!is.null(cache) && identical(cache$backend, backend)) return(cache$ok)
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  g   <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4L, 3L)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3L)
  shp <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4L, 5L)
  op  <- ggml_get_rows_back(ctx, g, ids, shp)
  ok  <- isTRUE(ggml_backend_dev_supports_op(ggml_backend_get_device(backend), op))
  .ag_bwd$grb_ok <- list(backend = backend, ok = ok)
  ok
}

# Can this tape run as one graph?
#
# Returns NULL when it can, or a short reason when it cannot -- the reason is
# recorded in $last_path so a probe can see WHY it fell back rather than
# guessing from a timing.
.ag_bwd_reject_reason <- function(nodes, loss) {
  if (!length(nodes)) return("empty tape")
  if (!identical(.ag_device_state$device, "gpu")) return("device is not gpu")
  if (is.null(.ag_device_state$backend))          return("no backend")
  # The graph's constants are built in the compute dtype, and not every rule
  # has an f16/bf16 kernel on Vulkan (out_prod is f32-only), so an f16 tape used
  # to abort inside the graph and fall back through the error handler -- hidden.
  # Refuse it up front, by name: the closures compute the gradient in double.
  cd <- .ag_compute_dtype()
  if (!identical(cd, "f32"))
    return(paste0("compute dtype ", cd, ": the graph backward runs in f32 only"))

  for (nd in nodes) {
    if (is.null(nd$op)) return("tape node has no op record")
    if (!nd$op %in% .AG_BWD_GRAPH_OPS) return(paste0("unsupported op: ", nd$op))
    # The graph runs on the device backend alone (no scheduler), so an op the
    # device refuses has no CPU to go to. GET_ROWS_BACK needs float atomics.
    if (identical(nd$op, "embedding") && !.ag_bwd_get_rows_back_ok())
      return("embedding: device has no GET_ROWS_BACK (float atomics)")
  }

  # The seed below is a 1x1 ones tensor, so the graph is only correct for a
  # scalar loss. A loss_const node that is NOT the root is fine now (its rule
  # multiplies by its own incoming 1x1 gradient), so this is the one shape
  # restriction left -- refused by name rather than by a ggml assert.
  ld <- .ag_dim(.ag_operand(loss))
  if (!is.null(ld) && prod(ld) != 1) return("loss is not a scalar")
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
#' @noRd
ag_backward_profile <- function(on = TRUE) {
  old <- .ag_bwd$prof
  if (!is.na(on)) {
    .ag_bwd$prof <- isTRUE(on)
    if (isTRUE(on)) ag_backward_profile_reset()
  }
  invisible(old)
}

#' @noRd
#' @keywords internal
ag_backward_profile_reset <- function() {
  .ag_bwd$prof_totals <- NULL
  .ag_bwd$prof_n      <- 0L
  invisible(NULL)
}

#' @noRd
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
  # A conv2d rule emits ~20 tensors (reshapes, permute, im2col, im2col_back,
  # two reductions), well above the per-node 8: each gets 24 more.
  n_conv <- sum(vapply(nodes, function(nd) isTRUE(nd$op %in% c("conv2d", "conv_transpose2d")),
                      logical(1L)))
  ctx <- .ag_ctx_ensure(8L * length(nodes) + 24L * n_conv + 16L)

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
      return(.ag_graph_operand(m, ctx))   # leaf alias if computed
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

  # Gradient of a broadcast operand (add's and sub's B). ggml_sum_rows reduces
  # ne[0]: [a,b,c,d] -> [1,b,c,d]. An R matrix [r,c] uploads as ne0=r, ne1=c,
  # so sum_rows collapses R's ROWS, giving colSums. That is exactly the
  # row-broadcast case; the column one needs the other axis and goes through a
  # transpose. Shapes via the accessors: b_orig is a device handle whenever the
  # forward kept it resident, and dim() on one is NULL.
  #
  # Both axes are reduced in turn, so a [1,1] operand broadcast over [m,n]
  # gets the full sum. (One axis only, it reshaped the [1,m] column sums to
  # [1,1] -- an element-count mismatch ggml asserts on.)
  reduce_b <- function(g, bo, out_nr, out_nc) {
    bdim <- .ag_dim(bo)
    if (is.null(bdim)) return(g)
    if (bdim[2L] == 1L && out_nc > 1L) {
      # b was [m,1] broadcast across columns -> db = rowSums(g), [m,1].
      # sum_rows reduces ne[0] only, so through it this took cont(transpose)
      # + sum_rows + reshape -- a full copy of g. One out_prod against a row
      # of ones does it in a single node:
      #   out_prod(a [m,n], b [1,n])[i,0] = sum_k a[i,k] * 1 = rowSums(g).
      # The ones are data-independent, so a recording may freeze them.
      # TEMPORARY (benchmark of TODO item 3A): option ggmlR.bias_grad_outprod
      # = FALSE restores the old path; remove once measured.
      if (isTRUE(getOption("ggmlR.bias_grad_outprod", TRUE))) {
        ones <- const(matrix(1, 1L, out_nc))
        uploads[[length(uploads)]]$const <<- TRUE
        g <- ggml_out_prod(ctx, g, ones)
      } else {
        gt <- ggml_cont(ctx, ggml_transpose(ctx, g))
        g <- ggml_reshape_2d(ctx, ggml_sum_rows(ctx, gt), out_nr, 1L)
      }
    }
    if (bdim[1L] == 1L && out_nr > 1L) {
      # b was [1,n] broadcast down rows -> db = colSums(g), [1,n], which
      # is sum_rows' native shape.
      g <- ggml_sum_rows(ctx, g)
    }
    g
  }

  # Fused bias gradient (GGML_OP_OUT_PROD_ROWSUM). A dense layer is recorded
  # as add(matmul(W, x), b) with b [m,1] broadcast over the columns; its dW is
  # out_prod(g, x) and its db the row sums of the same g. When the matmul's
  # output feeds nothing but that add, g is the same tensor at both nodes, and
  # one ggml_out_prod_rowsum(g, x) gives [dW | db] in one pass over g. The add
  # comes first in this reverse walk, so it leaves db to the matmul
  # (fused_bias[matmul output id] = what the matmul needs for it), and the
  # matmul splits the result into two views. options(ggmlR.bias_grad_fuse =
  # FALSE) turns it off.
  consumers <- new.env(hash = TRUE, parent = emptyenv())
  producer  <- new.env(hash = TRUE, parent = emptyenv())
  for (nd0 in nodes) {
    assign(as.character(nd0$output_id), nd0, envir = producer)
    for (t0 in nd0$inputs) if (is_ag_tensor(t0)) {
      k0 <- as.character(t0$id)
      assign(k0, get0(k0, envir = consumers, inherits = FALSE, ifnotfound = 0L) + 1L,
             envir = consumers)
    }
  }
  fused_bias <- new.env(hash = TRUE, parent = emptyenv())
  # Measured 2026-11-18 (RX 9070, PPO captured update, one process,
  # interleaved): the separate db node was 0.4-5% of the update; fusing takes
  # ~1.5% off pendulum and is within noise on tic-tac-toe -- kept for wider
  # reductions (long K, convolutions) where the separate node costs more.
  fuse_on <- isTRUE(getOption("ggmlR.bias_grad_fuse", TRUE))
  # The add node `nd` with bias B: can its db come from the matmul below it?
  can_fuse_bias <- function(nd, A, B) {
    if (!fuse_on || !is_ag_tensor(A) || !is_ag_tensor(B)) return(FALSE)
    bdim <- .ag_dim(nd$b_orig)
    if (is.null(bdim) || bdim[2L] != 1L || nd$out_nc <= 1L || bdim[1L] != nd$out_nr)
      return(FALSE)
    mm <- get0(as.character(A$id), envir = producer, inherits = FALSE)
    if (is.null(mm) || !identical(mm$op, "matmul")) return(FALSE)
    W <- mm$inputs$A
    is_ag_tensor(W) && isTRUE(W$requires_grad) &&
      identical(get0(as.character(A$id), envir = consumers, inherits = FALSE), 1L)
  }

  # Seed: dL/dL = 1. A 1x1 tensor, so the scalar rules below can multiply by it
  # in the graph rather than reading it back.
  assign(as.character(loss$id), const(matrix(1.0)), envir = gnodes)
  # The seed does not depend on the data, so ag_capture_step() may freeze it
  # into a recording; every other upload here is refused by its guard.
  uploads[[length(uploads)]]$const <- TRUE

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
      fb <- get0(as.character(nd$output_id), envir = fused_bias, inherits = FALSE)
      if (!is.null(fb)) {
        # dW and db of a dense layer in one node: [m, k + 1], dW the first k
        # columns, db the last; both contiguous, so plain views. F32 only --
        # with an f16/bf16 compute dtype the two take the separate path.
        xs <- const(nd$b_snap)
        if (ggml_tensor_type(g) == GGML_TYPE_F32 && ggml_tensor_type(xs) == GGML_TYPE_F32) {
          r   <- ggml_out_prod_rowsum(ctx, g, xs)
          .ag_bwd$fused_bias <- (.ag_bwd$fused_bias %||% 0L) + 1L   # for tests
          sh  <- ggml_tensor_shape(r)
          nb1 <- ggml_tensor_nb(r)[2L]
          accumulate(as.character(A$id), ggml_view_2d(ctx, r, sh[1L], sh[2L] - 1L, nb1, 0))
          accumulate(fb$b_id, ggml_view_2d(ctx, r, sh[1L], 1L, nb1, (sh[2L] - 1L) * nb1))
        } else {
          accumulate(as.character(A$id), ggml_out_prod(ctx, g, xs))
          accumulate(fb$b_id, reduce_b(g, fb$b_orig, fb$out_nr, fb$out_nc))
        }
      } else if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id), ggml_out_prod(ctx, g, const(nd$b_snap)))
      # dB = t(A) %*% g, shapes [k,m]x[m,n] -> [k,n]. Shared axis m sits in
      # ne[0] of both A[m,k] and g[m,n], so this one IS mul_mat -- and needs no
      # transpose node, since mul_mat already reads its first argument that way.
      if (is_ag_tensor(B) && isTRUE(B$requires_grad))
        accumulate(as.character(B$id), .ag_mul_mat(ctx, const(nd$a_snap), g))

    } else if (identical(nd$op, "add")) {
      A <- inp$A; B <- inp$B
      if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id), g)
      if (is_ag_tensor(B) && isTRUE(B$requires_grad)) {
        if (can_fuse_bias(nd, A, B)) {
          # db comes with dW from the matmul (see fused_bias above); what it
          # needs if it has to fall back to reduce_b goes along
          assign(as.character(A$id),
                 list(b_id = as.character(B$id), b_orig = nd$b_orig,
                      out_nr = nd$out_nr, out_nc = nd$out_nc),
                 envir = fused_bias)
        } else {
        # B may have been broadcast over A in the forward pass -- the bias case
        # of ag_linear -- and then its gradient is g summed over the axis that
        # was broadcast. Both directions are emitted here; neither needs a
        # download (reduce_b above).
        accumulate(as.character(B$id),
                   reduce_b(g, nd$b_orig, nd$out_nr, nd$out_nc))
        }
      }

    } else if (identical(nd$op, "sub")) {
      # dA = g, dB = -g, B's broadcast reduced exactly as in add (reduce_b),
      # then negated. A broadcast over B is legal in ag_sub's closure but has no
      # caller; refuse it by name rather than emit an unreduced gradient.
      A <- inp$A; B <- inp$B
      adim <- .ag_dim(nd$a_orig)
      bdim <- .ag_dim(nd$b_orig)
      if (!is.null(adim) && !is.null(bdim) &&
          ((adim[1L] == 1L && bdim[1L] > 1L) || (adim[2L] == 1L && bdim[2L] > 1L))) {
        .ag_bwd$last_path <- "closures (sub: A broadcast)"
        return(NULL)
      }
      if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id), g)
      if (is_ag_tensor(B) && isTRUE(B$requires_grad))
        accumulate(as.character(B$id),
                   ggml_scale(ctx, reduce_b(g, nd$b_orig, adim[1L], adim[2L]), -1))

    } else if (identical(nd$op, "log")) {
      # dx = g / x. Divided in the graph rather than recorded as elemwise_mul
      # with 1/x: x_snap is a handle when the forward kept x resident, and 1/x
      # would have to be computed on the host. x <= 0 gives Inf / sign-flipped
      # values here exactly as g / x does in the closure.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id), ggml_div(ctx, g, const(nd$x_snap)))

    } else if (identical(nd$op, "pow")) {
      # dx = g * p * x^(p - 1), p a fixed scalar. x^(p - 1) is built the way
      # the forward (.ag_gpu_pow) builds x^p, so x <= 0 gives what the
      # closure's R `^` gives: .ag_ggml_int_pow for integer p (finite for
      # negative x), 1 / sqrt(x) for p = 0.5 (Inf at 0, NaN below, as
      # 0.5 * x^-0.5), and exp((p - 1) * log x) otherwise -- NaN for x < 0,
      # where the forward exp(p * log x) is NaN too. p = 0 is exactly zero
      # (the derivative of a constant), as the closure gives.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad)) {
        p  <- as.double(nd$p)
        xs <- const(nd$x_snap)
        dx <- if (p == 0) {
          ggml_scale(ctx, g, 0)
        } else if (p == 1) {
          g
        } else if (.ag_pow_is_int(p)) {
          xp <- .ag_ggml_int_pow(ctx, xs, as.integer(p) - 1L)
          ggml_scale(ctx, ggml_mul(ctx, g, xp), p)
        } else if (p == 0.5) {
          ggml_scale(ctx, ggml_div(ctx, g, ggml_sqrt(ctx, xs)), 0.5)
        } else {
          xp <- ggml_exp(ctx, ggml_scale(ctx, ggml_log(ctx, xs), p - 1))
          ggml_scale(ctx, ggml_mul(ctx, g, xp), p)
        }
        accumulate(as.character(x$id), dx)
      }

    } else if (identical(nd$op, "batch_norm")) {
      # ag_batch_norm's normalisation (R/ag_layers.R). Train: the forward was
      # ggml_norm over the batch axis of the transposed x, so the backward is
      # ggml_norm_back on the transposed g and x, transposed back -- the
      # batch statistics' dependence on x included, as in the closure.
      # Eval: the running statistics are constants, dx = g / std, std an
      # [F, 1] snapshot broadcast over the batch.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad)) {
        gc <- if (ggml_is_contiguous(g)) g else ggml_cont(ctx, g)
        dx <- if (identical(nd$mode, "train")) {
          gt <- ggml_cont(ctx, ggml_transpose(ctx, gc))
          xt <- ggml_cont(ctx, ggml_transpose(ctx, const(nd$x_snap)))
          ggml_cont(ctx, ggml_transpose(ctx, ggml_norm_back(ctx, gt, xt, nd$eps)))
        } else {
          ggml_div(ctx, gc, const(nd$std_snap))
        }
        accumulate(as.character(x$id), dx)
      }

    } else if (identical(nd$op, "embedding")) {
      # ag_embedding: out = W[, idx + 1] ([dim, n] from [dim, vocab]). dW is
      # g scattered back into a zeroed [dim, vocab] table, repeated indices
      # summing: one ggml_get_rows_back -- on Vulkan the ggmlR shader with
      # float atomics, so the order of the sums (and the last bits) may vary
      # between runs. The indices are a constant of the recording, uploaded
      # as f32 (exact below 2^24, checked in the forward) and cast to I32.
      # The shape tensor is never computed: get_rows_back reads only its ne.
      W <- inp$weight
      if (is_ag_tensor(W) && isTRUE(W$requires_grad)) {
        n   <- length(nd$idx)
        gc  <- if (ggml_is_contiguous(g)) g else ggml_cont(ctx, g)
        ids <- ggml_reshape_1d(ctx, ggml_cast(ctx, const(nd$idx), GGML_TYPE_I32), n)
        shp <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ggml_tensor_shape(gc)[1L], nd$vocab)
        accumulate(as.character(W$id), ggml_get_rows_back(ctx, gc, ids, shp))
      }

    } else if (identical(nd$op, "silu")) {
      # dx = g * s * (1 + x * (1 - s)), s = sigmoid(x): one ggml_silu_back
      # node. Argument order is (dy, x), as ggml-graph.c's own SILU rule calls it.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id), ggml_silu_back(ctx, g, const(nd$x_snap)))

    } else if (identical(nd$op, "rms_norm")) {
      # dx = (g - y * mean(g * y)) / rms(x) per column: one ggml_rms_norm_back
      # node, (dy, x, eps) as in ggml-graph.c's RMS_NORM rule. ne0 is the R
      # matrix's row axis, the axis the forward normalised over.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id),
                   ggml_rms_norm_back(ctx, g, const(nd$x_snap), nd$eps))

    } else if (identical(nd$op, "layer_norm")) {
      # dx = (g - mean(g) - y * mean(g * y)) / s per column: one
      # ggml_norm_back node, (dy, x, eps) -- NORM_BACK is ggmlR's own op
      # (CPU ops-elemwise.cpp, Vulkan norm_back), absent upstream.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id),
                   ggml_norm_back(ctx, g, const(nd$x_snap), nd$eps))

    } else if (identical(nd$op, "conv2d")) {
      # ag_conv2d (R/ag_conv.R). g is [OC*OH*OW, B] (W fastest, then H, C).
      # With gt = g as [OC, P*B] (P = OW*OH) and cols = im2col(x) [K, P*B]:
      #   dW = gt %*% t(cols)       out_prod (shared ne1 = P*B)  -> [OC, K]
      #   dx = im2col_back(t(W) %*% gt)   mul_mat (shared ne0 = OC)
      #   db = per-channel sum over P and B
      # cols is recomputed from the x snapshot rather than kept from the
      # forward: it is KH*KW/stride^2 times the input (4x on Nature-CNN conv1,
      # 9x on a 3x3 stride-1 layer) and costs one im2col dispatch. All shapes
      # are fixed by the recording, so ag_capture_step can freeze the graph.
      cf <- nd$conf; B <- nd$batch; P <- cf$OW * cf$OH
      X <- inp$x; Wt <- inp$weight; Bs <- inp$bias
      need <- function(t) is_ag_tensor(t) && isTRUE(t$requires_grad)
      gc <- if (ggml_is_contiguous(g)) g else ggml_cont(ctx, g)
      kshape <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cf$KW, cf$KH, cf$IC, 1L)
      if (need(X) || need(Wt)) {
        g4 <- ggml_reshape_4d(ctx, gc, cf$OW, cf$OH, cf$OC, B)
        gt <- ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, g4, 1L, 2L, 0L, 3L)),
                              cf$OC, P * B)                         # [OC, OW, OH, B]
      }
      if (need(Wt)) {
        x4 <- ggml_reshape_4d(ctx, const(nd$x_snap), cf$IW, cf$IH, cf$IC, B)
        cols <- ggml_im2col(ctx, kshape, x4, cf$sW, cf$sH, cf$pW, cf$pH, cf$dW, cf$dH,
                            TRUE, GGML_TYPE_F32)
        cols <- ggml_reshape_2d(ctx, cols, cf$K, P * B)
        accumulate(as.character(Wt$id), ggml_out_prod(ctx, gt, cols))
      }
      if (need(X)) {
        dcols <- .ag_mul_mat(ctx, const(nd$w_snap), gt)            # [K, P*B]
        dx <- ggml_im2col_back(ctx, ggml_reshape_4d(ctx, dcols, cf$K, cf$OW, cf$OH, B),
                               kshape, c(cf$IW, cf$IH, cf$IC, B),
                               cf$sW, cf$sH, cf$pW, cf$pH, cf$dW, cf$dH)
        accumulate(as.character(X$id),
                   ggml_reshape_2d(ctx, dx, cf$IW * cf$IH * cf$IC, B))
      }
      if (isTRUE(nd$has_bias) && need(Bs)) {
        # sum over P (ne0 of g as [P, OC*B]), then over B on the small [OC, B]
        s <- ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_reshape_2d(ctx, gc, P, cf$OC * B)),
                             cf$OC, B)
        s <- ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, s)))   # [1, OC]
        accumulate(as.character(Bs$id), ggml_reshape_2d(ctx, s, cf$OC, 1L))
      }

    } else if (identical(nd$op, "conv_transpose2d")) {
      # ag_conv_transpose2d (R/ag_conv.R), cf = the DUAL conv2d: its input is
      # the transpose's output (g), its output the transpose's input (x).
      #   cols = im2col(g)                    [K, P*B], P = IW_t*IH_t
      #   dx   = forward of the dual conv2d:  t(W) %*% cols, permuted back
      #   dW   = x_t %*% t(cols)              out_prod (shared ne1 = P*B)
      #   db   = per-channel sum of g
      cf <- nd$conf; B <- nd$batch; P <- cf$OW * cf$OH
      X <- inp$x; Wt <- inp$weight; Bs <- inp$bias
      need <- function(t) is_ag_tensor(t) && isTRUE(t$requires_grad)
      gc <- if (ggml_is_contiguous(g)) g else ggml_cont(ctx, g)
      if (need(X) || need(Wt)) {
        kshape <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cf$KW, cf$KH, cf$IC, 1L)
        g4 <- ggml_reshape_4d(ctx, gc, cf$IW, cf$IH, cf$IC, B)
        cols <- ggml_im2col(ctx, kshape, g4, cf$sW, cf$sH, cf$pW, cf$pH, cf$dW, cf$dH,
                            TRUE, GGML_TYPE_F32)
        cols <- ggml_reshape_2d(ctx, cols, cf$K, P * B)
      }
      if (need(X)) {
        wt <- ggml_cont(ctx, ggml_transpose(ctx, const(nd$w_snap)))   # [K, IC_t]
        y  <- .ag_mul_mat(ctx, wt, cols)                              # [IC_t, P*B]
        y  <- ggml_reshape_4d(ctx, y, cf$OC, cf$OW, cf$OH, B)
        y  <- ggml_cont(ctx, ggml_permute(ctx, y, 2L, 0L, 1L, 3L))     # [OW, OH, IC_t, B]
        accumulate(as.character(X$id), ggml_reshape_2d(ctx, y, cf$OW * cf$OH * cf$OC, B))
      }
      if (need(Wt)) {
        x4 <- ggml_reshape_4d(ctx, const(nd$x_snap), cf$OW, cf$OH, cf$OC, B)
        xt <- ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, x4, 1L, 2L, 0L, 3L)),
                              cf$OC, P * B)                           # [IC_t, P*B]
        accumulate(as.character(Wt$id), ggml_out_prod(ctx, xt, cols))
      }
      if (isTRUE(nd$has_bias) && need(Bs)) {
        Q <- cf$IW * cf$IH
        s <- ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_reshape_2d(ctx, gc, Q, cf$IC * B)),
                             cf$IC, B)
        s <- ggml_sum_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, s)))   # [1, OC_t]
        accumulate(as.character(Bs$id), ggml_reshape_2d(ctx, s, cf$IC, 1L))
      }

    } else if (identical(nd$op, "log_softmax")) {
      # dx = g - p * colSums(g), p = exp(logp). Rebuilt from the logp snapshot
      # (resident when the forward was), so nothing crosses the bus. The
      # forward's stability shift does not appear: the result does not depend
      # on it. p [n,c] * sum_rows(g) [1,c] broadcasts like softmax's rule.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad)) {
        p <- ggml_exp(ctx, const(nd$y_snap))
        accumulate(as.character(x$id),
                   ggml_sub(ctx, g, ggml_mul(ctx, p, ggml_sum_rows(ctx, g))))
      }

    } else if (identical(nd$op, "sum")) {
      # ag_sum and ag_mean, any dim: dx = g spread back over the input shape,
      # times scale (1 for sum, 1/n for mean). g is [1,1], [m,1] or [1,n]; each
      # is a ggml_repeat onto [m,n]. The template tensor only carries the shape.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad)) {
        sh  <- nd$in_shape
        tpl <- ggml_new_tensor_2d(ctx, ggml_type, sh[1L], sh[2L])
        gx  <- ggml_repeat(ctx, g, tpl)
        if (nd$scale != 1) gx <- ggml_scale(ctx, gx, nd$scale)
        accumulate(as.character(x$id), gx)
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
      # Broadcasting (either operand [m,1], [1,n] or [1,1] against the
      # output): g has the OUTPUT shape, ggml_mul repeats the smaller snapshot
      # over it, and the product is reduced back to the operand's own shape by
      # reduce_b -- as the closure does with rowSums/colSums. Needed by a
      # Gaussian policy's state-independent log_std ([act,1] times [act,B]),
      # which before this sent the whole tape to the closures.
      A <- inp$A; B <- inp$B
      adim <- .ag_dim(nd$a_orig); bdim <- .ag_dim(nd$b_orig)
      if (!identical(adim, bdim) && (is.null(adim) || is.null(bdim))) {
        .ag_bwd$last_path <- "closures (mul: broadcast, shape unknown)"
        return(NULL)
      }
      # both NULL (identical, so no broadcast): reduce_b returns g unchanged
      out_nr <- if (is.null(adim)) NA else max(adim[1L], bdim[1L])
      out_nc <- if (is.null(adim)) NA else max(adim[2L], bdim[2L])
      if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id),
                   reduce_b(ggml_mul(ctx, g, const(nd$b_snap)), nd$a_orig, out_nr, out_nc))
      if (is_ag_tensor(B) && isTRUE(B$requires_grad))
        accumulate(as.character(B$id),
                   reduce_b(ggml_mul(ctx, g, const(nd$a_snap)), nd$b_orig, out_nr, out_nc))

    } else if (identical(nd$op, "transpose")) {
      # dx = t(g). ggml_transpose only relabels ne/nb, so the result is a view;
      # ggml_cont materialises it, which the consumers (and the download at the
      # end) need. Cheap: one copy, no arithmetic.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id), ggml_cont(ctx, ggml_transpose(ctx, g)))

    } else if (identical(nd$op, "reshape")) {
      # dx = g reshaped back to x's [nrow_x, ncol_x]: column-major order is
      # kept, so it is the same data under the old shape. reshape needs a
      # contiguous source; the result is made contiguous as in the forward.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad)) {
        src <- if (ggml_is_contiguous(g)) g else ggml_cont(ctx, g)
        accumulate(as.character(x$id),
                   ggml_cont(ctx, ggml_reshape_2d(ctx, src, nd$nrow_x, nd$ncol_x)))
      }

    } else if (identical(nd$op, "permute")) {
      # dx = g permuted back: the output's axes are shape[perm], and order(perm)
      # undoes perm (aperm(aperm(a, p), order(p)) == a). Same kernel as forward.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id),
                   .ag_permute_ggml(ctx, g, nd$shape[nd$perm], order(nd$perm)))

    } else if (identical(nd$op, "slice_rows")) {
      # dx = g in rows from+1..from+n, zero elsewhere: one pad node with zero
      # rows before (from) and after (the rest), along ne0 = R's rows.
      x <- inp$x
      if (is_ag_tensor(x) && isTRUE(x$requires_grad))
        accumulate(as.character(x$id),
                   ggml_pad_ext(ctx, g, lp = c(nd$from, 0L, 0L, 0L),
                                rp = c(nd$nrow_x - nd$from - nd$n, 0L, 0L, 0L)))

    } else if (identical(nd$op, "concat_rows")) {
      # dA = rows 1..nrow_a of g, dB = the rest: a view each (rows are ne0, so
      # the column stride stays and B starts nrow_a elements in), made
      # contiguous. An operand without gradient gets no node.
      A <- inp$a; B <- inp$b
      nb1 <- ggml_tensor_nb(g)[2L]; nc <- ggml_nrows(g)
      es  <- ggml_element_size(g)
      if (is_ag_tensor(A) && isTRUE(A$requires_grad))
        accumulate(as.character(A$id),
                   ggml_cont(ctx, ggml_view_2d(ctx, g, nd$nrow_a, nc, nb1, 0)))
      if (is_ag_tensor(B) && isTRUE(B$requires_grad))
        accumulate(as.character(B$id),
                   ggml_cont(ctx, ggml_view_2d(ctx, g, nd$nrow_b, nc, nb1, nd$nrow_a * es)))

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
      #
      # Below the root (vf_coef * mse + ... in PPO) g is the loss's own 1x1
      # gradient and has to be applied. ggml_repeat spreads it to gmat's shape
      # explicitly rather than relying on ggml_mul's broadcast.
      tgt <- inp[[1L]]
      if (is_ag_tensor(tgt) && isTRUE(tgt$requires_grad)) {
        gm <- ggml_scale(ctx, const(nd$gmat), nd$gscale)
        if (!identical(nd$output_id, loss$id))
          gm <- ggml_mul(ctx, gm, ggml_repeat(ctx, g, gm))
        accumulate(as.character(tgt$id), gm)
      }
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
  # Sized for the tape (.ag_graph_build): an unrolled loop outgrows ggml's
  # default 8192 nodes. Each gradient is a separate output, unreachable from
  # the others, so each is a root of its own or it would be pruned.
  gb <- .ag_graph_build(outs, "backward graph")
  on.exit(ggml_free(gb$ctx), add = TRUE)
  graph <- gb$graph
  if (prof) { acc <- .ag_bwd_prof_add(acc, "build", tk); tk <- Sys.time() }

  .ag_graph_compute(backend, graph, "graph backward")
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
