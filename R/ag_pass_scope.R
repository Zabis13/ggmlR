# ag_pass_scope(): a private pass pool for eager work outside a tape.
#
# The pass pool is freed only at the start of with_grad_tape(). Code that runs
# eager ag_* forwards without a tape -- validation passes between captured
# training steps, typically -- therefore grows it with every call and never
# gives the memory back. Measured on cayleyR's cube_q_train_device(): ~2.7 GB
# of VRAM per score() call, until the driver refused command submission and
# the desktop went down with it.
#
# The scope uses the same swap as .ag_capture_build(): the caller's pass pool
# is set aside, an empty one with a generation of its own takes its place, and
# on exit everything allocated in it is freed and the caller's pool comes back.
#
# Why a swap and not a mark/rollback over the buffer list: rollback cannot
# advance the generation without invalidating the caller's handles too, so a
# handle that escaped the scope would point into freed memory -- a segfault, not
# an error. With its own generation, an escaped handle fails the liveness check
# like any handle from a reset pool.
#
# Contract:
#   * Results must leave as host values (as.matrix(), numbers). Returning an
#     ag_tensor whose value lives only in the scope's pool is an error.
#   * Pass-pool tensors made BEFORE the scope are not live inside it; a value
#     needed there is read from its host copy (or refused, as after a reset).
#     Persistent-pool tensors (ag_param() weights) are unaffected.
#   * Scopes nest: each one saves and restores the pool that was current when
#     it was entered, so an inner exit never frees the outer pool.
#   * Not allowed while ag_capture is recording: the pool being recorded into
#     would be swapped out from under it.
#   * Single-threaded, like the rest of .ag_device_state.

#' Evaluate eager device work in a private, self-freeing pass pool
#'
#' Device memory for activations and intermediates (the "pass" pool) is
#' normally freed only when \code{\link{with_grad_tape}} starts the next step.
#' Eager \code{ag_*} forwards run outside a tape -- for example validation
#' between \code{\link{ag_capture_step}} replays -- never reach that point, so
#' each one keeps its memory until the process ends. \code{ag_pass_scope()}
#' gives \code{expr} a pass pool of its own and frees it when \code{expr}
#' returns or fails.
#'
#' Bring results out as host values (for example with \code{as.matrix()}).
#' Returning an \code{ag_tensor} whose value exists only in the scope's pool is
#' an error, since that memory is freed on exit. Weights in the persistent pool
#' (\code{\link{ag_param}}) are not affected. Scopes may be nested. On the CPU
#' device \code{expr} is evaluated as is.
#'
#' @param expr Expression to evaluate.
#' @return The value of \code{expr}.
#' @export
#' @examples
#' \donttest{
#' ag_device("cpu")
#' W <- ag_param(matrix(rnorm(12), 3, 4))
#' y <- ag_pass_scope(as.matrix(ag_matmul(W, ag_tensor(matrix(1, 4, 2)))))
#' dim(y)
#' }
ag_pass_scope <- function(expr) {
  st <- .ag_device_state
  if (identical(st$device, "cpu") || is.null(st$backend))
    return(expr)
  if (isTRUE(.ag_capture_reg$recording))
    stop("ggmlR: ag_pass_scope() cannot be used while ag_capture is recording.",
         call. = FALSE)

  # Anything queued belongs to the caller's pool and must be computed there.
  if (.ag_defer_len()) .ag_defer_drain()

  slots <- .ag_pool_slots("pass")
  saved <- list(ctxs = st[[slots$ctxs]], bufs = st[[slots$bufs]], cur = st[[slots$cur]],
                gen = st[[slots$gen]], buffer = st$buffer,
                grads = st$pending_grads)
  # Shares the recordings' counter, so the generation is unique among both.
  id  <- .ag_capture_reg$next_id <- .ag_capture_reg$next_id + 1L
  gen <- -id
  st[[slots$ctxs]] <- list(); st[[slots$bufs]] <- list(); st[[slots$cur]] <- NULL
  st[[slots$gen]] <- gen; st$buffer <- NULL
  st$pending_grads <- list()

  on.exit(.ag_pass_scope_exit(saved, slots, gen), add = TRUE)

  res <- expr
  if (.ag_pass_scope_escapes(res, gen))
    stop("ggmlR: ag_pass_scope() returned an ag_tensor whose value lives in ",
         "the scope's pool, which is freed on exit. Return host values ",
         "instead (e.g. as.matrix()).", call. = FALSE)
  res
}

# Free the scope's pool and put the caller's back. Runs on success and error.
.ag_pass_scope_exit <- function(saved, slots, gen) {
  st <- .ag_device_state
  # Settle the queue while its buffers still exist, then wait for the device:
  # Vulkan may still be executing work that writes into the buffers freed next.
  if (.ag_defer_len()) tryCatch(.ag_defer_drain(), error = function(e) NULL)
  .ag_defer_discard()
  if (!is.null(st$backend))
    tryCatch(ggml_backend_synchronize(st$backend), error = function(e) NULL)

  # Gradients and resident values created in the scope go to the host, as at a
  # tape reset. Only the scope's own entries: the caller's are not live here
  # and the shared rescue would drop them.
  .ag_materialise_pending_grads()
  vals <- st$resident_values
  mine <- vapply(vals, function(t) identical(.ag_tensor_scope(t), "pass") &&
                   identical(t$ctx_gen, gen), logical(1))
  if (any(mine)) {
    st$resident_values <- vals[mine]
    .ag_materialise_resident_values("pass")
    st$resident_values <- vals[!mine]
  }

  for (b in st[[slots$bufs]]) tryCatch(ggml_backend_buffer_free(b), error = function(e) NULL)
  for (x in st[[slots$ctxs]]) tryCatch(ggml_free(x), error = function(e) NULL)

  st[[slots$ctxs]] <- saved$ctxs; st[[slots$bufs]] <- saved$bufs
  st[[slots$cur]]  <- saved$cur;  st[[slots$gen]]  <- saved$gen
  st$buffer <- saved$buffer
  st$pending_grads <- saved$grads
  invisible(NULL)
}

# TRUE when `x` is, or holds in a list, an ag_tensor whose only value is in
# the scope's pool (generation `gen`) -- a host copy makes it safe to return.
.ag_pass_scope_escapes <- function(x, gen) {
  if (is_ag_tensor(x))
    return(identical(.ag_tensor_scope(x), "pass") && identical(x$ctx_gen, gen) &&
             is.null(x$data))
  if (is.list(x) && !is.environment(x))
    for (el in x) if (.ag_pass_scope_escapes(el, gen)) return(TRUE)
  FALSE
}
