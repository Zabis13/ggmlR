# Target networks: Polyak (soft) and hard updates of one model's weights from
# another's, for DQN / DDPG / TD3 / SAC.
#
# Resident pairs are updated on the device in ONE graph per chunk (no host
# round trip, unlike ag_load_state_dict(target, ag_state_dict(source)), which
# downloads and re-uploads every weight). Pairs that are not both resident go
# through the host.

#' Update a target network from a source network
#'
#' \code{ag_soft_update} is the Polyak update
#' \deqn{target <- tau * source + (1 - tau) * target}
#' applied to every parameter; \code{tau = 1} copies, \code{tau = 0} changes
#' nothing. \code{ag_hard_update} copies the parameters directly (no
#' arithmetic, so the copy is exact).
#'
#' \strong{Argument order}: the TARGET (the model that is written) comes first,
#' as in \code{\link{ag_load_state_dict}}. Stable-Baselines3's
#' \code{polyak_update(params, target_params, tau)} and RLlib put the source
#' first -- mind the order when porting code.
#'
#' \strong{Buffers}: when both arguments are models, persistent buffers
#' (\code{\link{ag_batch_norm}} running statistics) are COPIED from the source
#' whatever \code{tau} is -- they are statistics, not learned weights. This
#' is what SB3 does (it updates them with \code{tau = 1}); the TD3 paper does
#' not mention it. Plain parameter lists have no buffers.
#'
#' Parameters are paired by name when both sides are named, otherwise by
#' position; the counts must be equal and every shape must match. Everything is
#' checked before anything is written. On the device, pairs where both tensors
#' are resident are updated in one graph per 256 parameters; the arithmetic is
#' F32 and the result is converted to the target's type (an F32 master copy of
#' the target, if it has one, gets the unrounded value). Other pairs are
#' updated on the host.
#'
#' @param target Model (anything with \code{parameters()}, as for
#'   \code{\link{ag_state_dict}}) or list of ag_param tensors. Written in place.
#' @param source Model or list of ag_param tensors, same structure. Read only.
#' @param tau Interpolation factor in \code{[0, 1]}; the weight of the source.
#' @return \code{target}, invisibly.
#' @export
#' @examples
#' \donttest{
#' online <- ag_sequential(ag_linear(4L, 8L), ag_linear(8L, 2L))
#' target <- ag_sequential(ag_linear(4L, 8L), ag_linear(8L, 2L))
#' ag_hard_update(target, online)          # start equal
#' ag_soft_update(target, online, 0.005)   # after each training step
#' }
ag_soft_update <- function(target, source, tau) {
  if (!is.numeric(tau) || length(tau) != 1L || is.na(tau) || tau < 0 || tau > 1)
    stop("ag_soft_update: tau must be a number in [0, 1]", call. = FALSE)
  .ag_target_update(target, source, tau, "ag_soft_update")
}

#' @rdname ag_soft_update
#' @export
ag_hard_update <- function(target, source) {
  .ag_target_update(target, source, 1, "ag_hard_update")
}

.ag_target_params <- function(x, caller) {
  if (is.list(x) && !is.environment(x) && !is.function(x[["parameters"]]) &&
      !is.function(x[["params"]])) {
    if (!all(vapply(x, is_ag_tensor, logical(1))))
      stop(caller, ": a parameter list must hold ag_tensors only", call. = FALSE)
    return(x)
  }
  .ag_model_params(x, caller)
}

.ag_target_update <- function(target, source, tau, caller) {
  tp <- .ag_target_params(target, caller)
  sp <- .ag_target_params(source, caller)
  if (length(tp) != length(sp))
    stop(caller, ": target has ", length(tp), " parameters, source has ",
         length(sp), call. = FALSE)
  tn <- names(tp); sn <- names(sp)
  if (!is.null(tn) && !is.null(sn) && all(nzchar(tn)) && all(nzchar(sn))) {
    if (!setequal(tn, sn) || anyDuplicated(tn) || anyDuplicated(sn))
      stop(caller, ": parameter names differ\n  target: ", paste(tn, collapse = ", "),
           "\n  source: ", paste(sn, collapse = ", "), call. = FALSE)
    sp <- sp[tn]
  }
  for (i in seq_along(tp)) {
    dt <- .ag_sel_dim(tp[[i]]); ds <- .ag_sel_dim(sp[[i]])
    if (!identical(as.integer(dt), as.integer(ds)))
      stop(caller, ": shape mismatch for parameter ",
           if (!is.null(tn)) paste0("'", tn[i], "'") else i, ": target ",
           paste(dt, collapse = "x"), ", source ", paste(ds, collapse = "x"),
           call. = FALSE)
  }
  # Buffers (models only) are checked before any write as well.
  is_model <- function(x) is.function(x[["parameters"]]) || is.function(x[["params"]])
  bufs <- if (is_model(target) && is_model(source)) .ag_collect_buffers(source)
          else list()
  if (length(bufs)) {
    tb <- .ag_collect_buffers(target)
    if (!setequal(names(tb), names(bufs)))
      stop(caller, ": buffer names differ between target and source", call. = FALSE)
    for (nm in names(bufs))
      if (!identical(dim(tb[[nm]]), dim(bufs[[nm]])))
        stop(caller, ": shape mismatch for buffer '", nm, "'", call. = FALSE)
  }

  if (tau != 0) {
    dev <- vapply(seq_along(tp), function(i)
      !is.null(.ag_handle_of(tp[[i]])) && !is.null(.ag_handle_of(sp[[i]])), logical(1))
    if (any(dev))
      .ag_defer_suspend(.ag_target_update_device(tp[dev], sp[dev], tau))
    for (i in which(!dev)) {
      s <- .ag_as_matrix(.ag_data(sp[[i]]))
      v <- if (tau == 1) s else tau * s + (1 - tau) * .ag_as_matrix(.ag_data(tp[[i]]))
      .ag_opt_store_weight(tp[[i]], v)
    }
  }
  if (length(bufs)) .ag_restore_buffers(target, bufs)
  invisible(target)
}

# One graph per chunk: per pair, new = tau*src + (1-tau)*dst in F32 (or src
# itself for tau = 1), copied into dst (converting to its type) and into its
# F32 master if it has one. Each copy reads the node that read dst, so the
# write is ordered after the read inside the graph; pairs share no tensor.
.ag_target_update_device <- function(tp, sp, tau) {
  backend <- .ag_device_state$backend
  idx <- seq_along(tp)
  for (chunk in split(idx, ceiling(idx / .ag_adam_fused_chunk))) {
    # up to 2 casts, 2 scales, add, 2 copies per pair
    ctx <- .ag_ctx_ensure(8L * length(chunk) + 4L, scope = "pass")
    nodes <- list()
    for (i in chunk) {
      dh <- .ag_handle_of(tp[[i]]); sh <- .ag_handle_of(sp[[i]])
      if (identical(dh$ptr, sh$ptr)) next          # same tensor: nothing to do
      mh <- .ag_master_separate(tp[[i]])
      # the current target value: its F32 master when it has one (unrounded)
      cur <- if (is.null(mh)) dh$ptr else mh$ptr
      new <- if (tau == 1) sh$ptr else
        ggml_add(ctx,
                 ggml_scale(ctx, .ag_f32_operand(ctx, sh$ptr), tau),
                 ggml_scale(ctx, .ag_f32_operand(ctx, cur), 1 - tau))
      nodes[[length(nodes) + 1L]] <- ggml_cpy(ctx, new, dh$ptr)
      if (!is.null(mh)) nodes[[length(nodes) + 1L]] <- ggml_cpy(ctx, new, mh$ptr)
    }
    if (!length(nodes)) next
    .ag_ctx_flush(ctx, scope = "pass")
    for (nd in nodes) .Call("R_ggml_view_bind", nd, PACKAGE = "ggmlR")
    ctx_graph <- ggml_init(.ag_graph_ctx_bytes(), no_alloc = TRUE)
    if (is.null(ctx_graph))
      stop("ggmlR: failed to create a ggml context for the target update graph.",
           call. = FALSE)
    graph <- ggml_build_forward_expand(ctx_graph, nodes[[1L]])
    for (nd in nodes[-1L]) ggml_graph_expand(graph, nd)
    tryCatch(.ag_graph_compute(backend, graph, "target update"),
             finally = ggml_free(ctx_graph))
  }
  for (p in tp) { p$data <- NULL; p$data_gen <- NULL }
  invisible(NULL)
}
