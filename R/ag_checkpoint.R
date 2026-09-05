# Gradient checkpointing for the ag_* tape: trade compute for memory.
#
# WHY
# ---
# Every ag_* op records the inputs its backward rule will need -- ag_matmul
# keeps a_snap and b_snap, the activations flowing through it. On a 12-layer
# 256x256 stack those snapshots are 9.2 MB of the 10.2 MB the tape holds, and
# 7.5 MB of that is matmul alone. The activations exist only to be read once,
# during backward.
#
# Checkpointing runs a segment of the forward pass WITHOUT recording it, keeps
# just the segment's input, and re-runs the segment during backward to rebuild
# what the rules need. Memory drops by whatever the segment would have stored;
# the cost is running that segment twice.
#
# WHY IT IS A WRAPPER AND NOT A TAPE ANALYSIS
# -------------------------------------------
# The obvious design -- read the tape and replay the recorded ops -- cannot
# work here. A node's `op` field names the BACKWARD rule, not the forward
# operation: ag_relu, ag_sigmoid and ag_tanh all record "elemwise_mul", and the
# three losses all record "loss_const". Nothing on the tape says which function
# produced a node, so nothing can re-run it. The segment therefore has to be
# handed in as R code, which is also how torch.utils.checkpoint works.
#
# THREE CONSTRAINTS, EACH MEASURED (scratchpad/probe_checkpoint.R)
# ---------------------------------------------------------------
#  * Recomputation must NOT go through with_grad_tape(): on the GPU that calls
#    .ag_reset_ggml_ctx(), which frees every context and bumps ctx_gen
#    (measured: 1 -> 2). Doing that mid-backward would turn gradients already
#    accumulated into dangling pointers. Recording by hand leaves the
#    generation alone (measured: 1 -> 1).
#  * with_grad_tape() disables the tape on exit, so a recomputation has to set
#    .ag_tape$enabled itself -- otherwise the segment silently records nothing
#    and its parameters get no gradient at all.
#  * The segment must be reproducible. A pure segment (matmul + relu) replays
#    bit-identically; one containing ag_dropout does NOT, because the mask is
#    drawn from runif() -- the backward would then use a mask the forward never
#    saw, which is wrong gradients with no error anywhere. So the RNG state is
#    saved before the first run and restored before the replay.

#' Run part of the forward pass without storing its activations
#'
#' Executes \code{fn(...)} with the gradient tape switched off, so none of the
#' intermediate activations inside \code{fn} are kept. During
#' \code{\link{backward}()} the segment is run again -- with the same RNG state
#' -- to rebuild what the gradient rules need.
#'
#' This trades compute for memory: the segment's forward pass runs twice, and
#' in exchange the tape never holds its activations. Use it on the deep middle
#' of a network, not on cheap layers.
#'
#' @param fn A function taking the tensors in \code{...} and returning a single
#'   \code{ag_tensor}. It must be reproducible: given the same inputs and RNG
#'   state it has to compute the same thing. Parameters it closes over are
#'   fine, and are given gradients as usual.
#'
#'   One caveat, because it bites in the most natural usage: \code{fn} is
#'   called again during \code{backward()}, so it must not close over a loop
#'   variable. Wrapping layers in a \code{for} loop with
#'   \code{lyr <- layers[[i]]} in the body leaves every segment sharing one
#'   binding, and by replay time they all see the last layer -- the earlier
#'   ones then get no gradient while the loss still looks perfectly normal.
#'   Capture per iteration with \code{local()}, as in the example below.
#' @param ... Input \code{ag_tensor}s, passed to \code{fn}.
#' @return An \code{ag_tensor} holding the segment's output. Its gradient flows
#'   back into both \code{...} and any parameters used inside \code{fn}.
#' @export
#' @examples
#' \donttest{
#' block <- ag_linear(8L, 8L, activation = "relu")
#' x <- ag_tensor(matrix(runif(8 * 4), 8, 4))
#' y <- ag_tensor(matrix(runif(8 * 4), 8, 4))
#' with_grad_tape({
#'   h    <- ag_checkpoint(function(inp) block$forward(inp), x)
#'   loss <- ag_mse_loss(h, y)
#' })
#' backward(loss)
#'
#' # Checkpointing every second block of a deeper stack. local() gives each
#' # segment its own binding -- see the note on `fn` for why that matters.
#' layers <- lapply(1:4, function(i) ag_linear(8L, 8L, activation = "relu"))
#' with_grad_tape({
#'   h <- x
#'   for (i in seq_along(layers)) {
#'     seg <- local({
#'       lyr <- layers[[i]]
#'       function(inp) lyr$forward(inp)
#'     })
#'     h <- if (i %% 2L == 0L) ag_checkpoint(seg, h) else seg(h)
#'   }
#'   loss <- ag_mse_loss(h, y)
#' })
#' backward(loss)
#' }
ag_checkpoint <- function(fn, ...) {
  if (!is.function(fn))
    stop("ag_checkpoint() expects a function as its first argument.", call. = FALSE)
  inputs <- list(...)

  # Outside a tape there is nothing to save: just run the segment.
  if (!.ag_tape$enabled) return(fn(...))

  # The segment has to replay identically, so remember the RNG state it starts
  # from. .Random.seed only exists once the generator has been used, so make
  # sure it does -- otherwise the replay would start from a fresh stream and a
  # dropout mask inside the segment would differ.
  if (!exists(".Random.seed", envir = globalenv(), inherits = FALSE))
    stats::runif(1L)
  rng_state <- get(".Random.seed", envir = globalenv(), inherits = FALSE)

  # Forward with recording OFF: the segment's own ops leave no nodes and, more
  # to the point, no activation snapshots.
  #
  # The restore is a `finally`, not a plain sequence, because a segment that
  # throws must not leave the engine altered: the tape would stay disabled and
  # every op after it would go unrecorded, and the RNG stream would keep
  # whatever draws the half-run segment made -- so one failed checkpoint would
  # silently change the random numbers the rest of the model sees.
  saved_nodes <- .ag_tape$nodes
  .ag_tape$enabled <- FALSE
  out_val <- tryCatch(.ag_data(fn(...)),
                      finally = {
                        .ag_tape$enabled <- TRUE
                        .ag_tape$nodes   <- saved_nodes
                        assign(".Random.seed", rng_state, envir = globalenv())
                      })

  # The value re-enters the tape as a FRESH tensor, deliberately detached from
  # whatever produced it inside the segment.
  #
  # This is the contract of the boundary, and the easiest thing to get wrong:
  #   * detached from the SEGMENT's internals, so the outer backward sees one
  #     opaque "checkpoint" node instead of the segment's expanded subgraph --
  #     without that, a gradient could reach the segment's parameters twice
  #     (once through the outer tape, once through the replay) and be doubled;
  #   * NOT detached from the outer graph -- the node recorded below lists the
  #     caller's inputs, so the gradient still flows on through them to earlier
  #     layers. Dropping that link is the other failure mode: everything before
  #     the checkpoint would silently train on nothing.
  # Both directions are covered by tests in test-ag-checkpoint.R.
  out <- ag_tensor(out_val, device = .ag_device_state$device,
                   dtype = .ag_device_state$dtype)
  out$requires_grad <- TRUE

  fn_ref  <- fn
  in_ref  <- inputs
  out_ref <- out

  # The closure the tape stores. Unlike every other grad_fn it does not compute
  # a gradient directly: it rebuilds the segment and walks the segment's own
  # tape, returning the gradients of the inputs the caller passed in. Gradients
  # for parameters INSIDE the segment are written straight to their $grad, the
  # same way backward() would.
  grad_fn <- function(grad_out) {
    .ag_checkpoint_replay(fn_ref, in_ref, out_ref, grad_out, rng_state)
  }

  out$grad_fn <- grad_fn

  # Recorded directly rather than through ag_record(), which drops any node
  # whose inputs all have requires_grad = FALSE. That test is correct for an
  # ordinary op and wrong for a checkpoint: the trainable parameters live
  # INSIDE the segment and are invisible from out here, so a segment applied to
  # a plain input tensor would be dropped from the tape and its weights would
  # silently receive no gradient at all.
  #
  # Recorded with no `op`, so the graph backward path declines a tape holding a
  # checkpoint rather than trying to emit it -- the segment is R code, not a
  # rule it can turn into graph nodes.
  .ag_tape$nodes <- c(.ag_tape$nodes, list(list(
    output_id = out$id,
    grad_fn   = grad_fn,
    inputs    = .ag_checkpoint_named(inputs),
    op        = NULL
  )))
  out
}

# Give the inputs stable names, so backward()'s "for (nm in names(node$inputs))"
# can match a returned gradient to its tensor. Unnamed arguments become x1, x2…
.ag_checkpoint_named <- function(inputs) {
  nm <- names(inputs)
  if (is.null(nm)) nm <- rep("", length(inputs))
  blank <- !nzchar(nm)
  nm[blank] <- paste0("x", seq_along(nm))[blank]
  stats::setNames(inputs, nm)
}

# Re-run the segment and push `grad_out` back through it.
#
# The segment is recorded on a private tape: the outer nodes are set aside and
# restored afterwards, so the pass currently walking them is unaffected. Note
# what is NOT used -- with_grad_tape(), which would reset the ggml context and
# invalidate the gradients already computed (see the file header).
.ag_checkpoint_replay <- function(fn, inputs, out, grad_out, rng_state) {
  outer_nodes   <- .ag_tape$nodes
  outer_enabled <- .ag_tape$enabled
  outer_rng     <- if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
                     get(".Random.seed", envir = globalenv(), inherits = FALSE) else NULL

  on.exit({
    .ag_tape$nodes   <- outer_nodes
    .ag_tape$enabled <- outer_enabled
    if (!is.null(outer_rng))
      assign(".Random.seed", outer_rng, envir = globalenv())
  }, add = TRUE)

  # Replay from the recorded RNG state so anything stochastic in the segment
  # (dropout masks above all) comes out exactly as it did in the forward pass.
  assign(".Random.seed", rng_state, envir = globalenv())

  # INVARIANT: the replay must not reset the ggml context.
  #
  # with_grad_tape() calls .ag_reset_ggml_ctx(), which frees every context and
  # buffer and bumps ctx_gen. Running that here -- in the middle of a backward
  # pass -- would leave the gradients already accumulated pointing into freed
  # device memory, and .ag_data() would either error on the stale generation or
  # hand back a plausible-looking host copy. Neither is detectable from the
  # result, so the generation is checked rather than trusted: this is the one
  # thing a later refactor could break silently by reaching for the convenient
  # with_grad_tape() wrapper.
  # Both pools are watched. Gradients live in the pass pool, so that counter is
  # the one that matters for them -- but ag_device() frees the persistent pool
  # too, and a segment that dropped resident weights while leaving the pass pool
  # untouched would slip through a pass-only check.
  gen_before   <- .ag_device_state$ctx_gen
  p_gen_before <- .ag_device_state$p_ctx_gen

  .ag_tape$nodes   <- list()
  .ag_tape$enabled <- TRUE
  redone <- do.call(fn, inputs)
  seg_nodes <- .ag_tape$nodes

  if (!identical(gen_before, .ag_device_state$ctx_gen) ||
      !identical(p_gen_before, .ag_device_state$p_ctx_gen))
    stop("ggmlR: ag_checkpoint() replay reset the ggml context (pass ",
         gen_before, " -> ", .ag_device_state$ctx_gen, ", persistent ",
         p_gen_before, " -> ", .ag_device_state$p_ctx_gen, "). Gradients ",
         "already accumulated now point at freed device memory. The segment ",
         "must not call with_grad_tape() or ag_device().", call. = FALSE)

  # Walk the segment's tape exactly as backward() walks the main one, seeded
  # with the gradient arriving from downstream.
  grads <- new.env(hash = TRUE, parent = emptyenv())
  assign(as.character(redone$id), grad_out, envir = grads)

  for (nd in rev(seg_nodes)) {
    go <- get0(as.character(nd$output_id), envir = grads)
    if (is.null(go)) next
    ig <- nd$grad_fn(go)
    for (nm in names(nd$inputs)) {
      inp <- nd$inputs[[nm]]
      if (!is_ag_tensor(inp) || !isTRUE(inp$requires_grad)) next
      g <- ig[[nm]]
      if (is.null(g)) next
      key  <- as.character(inp$id)
      prev <- get0(key, envir = grads)
      assign(key, if (is.null(prev)) g else prev + g, envir = grads)
    }
  }

  # Parameters used inside the segment are leaves of the segment's tape, not of
  # the outer one, so backward() will never see them: write their $grad here.
  # Anything the caller passed in is returned instead, for the outer pass to
  # accumulate in the usual way.
  input_ids <- vapply(inputs, function(i)
    if (is_ag_tensor(i)) as.character(i$id) else NA_character_, character(1))

  for (nd in seg_nodes) {
    for (inp in nd$inputs) {
      if (!is_ag_tensor(inp) || !isTRUE(inp$requires_grad)) next
      key <- as.character(inp$id)
      if (key %in% input_ids) next          # caller's tensor: returned below
      g <- get0(key, envir = grads)
      if (is.null(g)) next
      # Only leaves: a tensor produced inside the segment carries no gradient
      # anyone outside can use.
      if (any(vapply(seg_nodes, function(o) identical(as.character(o$output_id), key),
                     logical(1)))) next
      # Through the accessor: with resident gradients $grad is a device handle,
      # which deliberately has no arithmetic (rule 3 of the data contract), so
      # `inp$grad + g` would error rather than silently compute. Accumulating a
      # gradient needs the numbers, so this is one of the places that pays a
      # materialisation -- once per leaf, not per pass.
      inp$grad <- if (is.null(inp$grad)) g
                  else .ag_as_matrix(inp$grad) + .ag_as_matrix(g)
    }
  }

  named <- .ag_checkpoint_named(inputs)
  res <- lapply(names(named), function(nm) {
    t <- named[[nm]]
    if (!is_ag_tensor(t) || !isTRUE(t$requires_grad)) return(NULL)
    get0(as.character(t$id), envir = grads)
  })
  stats::setNames(res, names(named))
}
