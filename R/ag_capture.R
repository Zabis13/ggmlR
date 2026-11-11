# Graph capture: record an ag_* computation once, replay it as one launch.
#
# Why. On small networks the GPU path is bound by R, not by the device: in
# graph mode a PPO update records ~470 ag_* ops per iteration at ~0.25 ms each
# with ZERO launches (rltoolsR, inst/examples/profile_ppo_stages.R), so the
# device sits idle while R builds nodes. Graph mode already turns those ops
# into one ggml graph -- and then throws it away after one compute. A capture
# keeps it: the R wrappers run once per input shape, and every later call is
# upload inputs -> one graph_compute -> download outputs.
#
# How. The pass pool's slots are swapped for fresh ones while `fn` runs, so
# every context and buffer the trace allocates belongs to the capture and
# survives the next tape reset. Inputs are created and uploaded BEFORE `fn`
# runs (an op that reads a value back then reads real data, and the guard
# below reports it, instead of ggml aborting on an unallocated tensor).
#
# ⚠️ The guard. Anything R computes from data during the trace is frozen into
# the graph: a mask built with ag_tensor(f(x)), a one-hot from an index, a
# value read back to branch on. Replays would silently reuse the first call's
# values. So the trace must do NO host->device upload (other than the inputs),
# NO download and NO launch; any of them is an error naming the site. Data
# the step depends on has to come in as an input.
#
# ag_capture() records a forward only. ag_capture_step() also records the tape,
# the graph backward and the Adam step (as a second graph). Weights must
# already be resident on the device -- a weight held as an R matrix would be
# uploaded during the trace, which the guard refuses.

.ag_capture_reg <- new.env(parent = emptyenv())
.ag_capture_reg$items   <- new.env(parent = emptyenv())   # id -> capture env
.ag_capture_reg$next_id <- 0L

# Opt-in stage timer for ag_capture_step() replays (GGMLR_AG_CAPTURE_PROF=1;
# off by default):
# seconds per stage accumulated over replays, read with .ag_capture_prof().
# The recording itself is not timed.
.ag_capture_reg$prof   <- numeric(0)
.ag_capture_reg$prof_nodes <- numeric(0)   # graph stage -> nodes in its graph

.ag_capture_prof_on <- function() identical(Sys.getenv("GGMLR_AG_CAPTURE_PROF"), "1")

# Add the time since t0 to `stage`; returns the new t0.
.ag_capture_prof_add <- function(stage, t0) {
  now <- as.numeric(Sys.time())
  pr <- .ag_capture_reg$prof
  pr[stage] <- (if (is.na(pr[stage])) 0 else pr[stage]) + (now - t0)
  .ag_capture_reg$prof <- pr
  now
}

# Milliseconds per replay by stage (and their sum); reset = TRUE clears.
.ag_capture_prof <- function(reset = FALSE) {
  pr <- .ag_capture_reg$prof
  n  <- unname(pr["replays"])
  out <- if (length(pr) && !is.na(n) && n > 0)
    1000 * pr[setdiff(names(pr), "replays")] / n else numeric(0)
  if (length(out)) out <- c(out, total = sum(out), replays = n)
  attr(out, "nodes") <- .ag_capture_reg$prof_nodes
  if (reset) { .ag_capture_reg$prof <- numeric(0); .ag_capture_reg$prof_nodes <- numeric(0) }
  out
}

#' Capture an ag_* computation as a reusable device graph
#'
#' Wraps \code{fn} so that, on the GPU, its ag_* operations are recorded once
#' per input shape and every later call runs the recorded ggml graph: upload the
#' inputs, one graph compute, download the outputs -- no R-level op dispatch.
#' On the CPU the wrapper simply calls \code{fn}, so the same code runs on both.
#'
#' \code{fn} must depend on the data only through its arguments. Anything it
#' computes in R from data, or uploads itself (\code{ag_tensor(f(x))}), would be
#' frozen into the graph at the first call; such uploads, any download and any
#' eager launch during the recording are errors. Weights must be resident on
#' the device (e.g. trained or used on the GPU) before the first call.
#'
#' The recording is redone when the input shapes, compute dtype or matmul
#' precision change, when the persistent device pool is reset (device switch),
#' and when a tensor in \code{params} no longer points at the device memory it
#' had at recording time.
#'
#' @param fn Function of named ag_tensor arguments returning an ag_tensor or a
#'   named list of ag_tensors, or \code{NULL} when its only effect is writes
#'   with \code{\link{ag_set_rows}}. Forward computations only (no
#'   \code{with_grad_tape}).
#' @param params Optional list of ag_tensors (typically the model parameters)
#'   and \code{\link{ag_buffer}}s read with \code{\link{ag_get_rows}}
#'   whose device memory the recording may reference; a change re-records.
#' @return A function taking the same named arguments as numeric matrices (a
#'   vector is one column) and returning a matrix, a named list of matrices,
#'   or \code{NULL} invisibly (writes only: nothing is downloaded).
#' @seealso \code{\link{ag_capture_free}}, \code{\link{ag_set_rows}}
#' @export
#' @examples
#' \donttest{
#' W <- ag_param(matrix(rnorm(6), 2, 3))
#' f <- ag_capture(function(x) ag_relu(ag_matmul(W, x)), params = list(W))
#' f(x = matrix(1, 3, 4))
#' }
ag_capture <- function(fn, params = NULL) {
  stopifnot(is.function(fn))
  if (!is.null(params) && (!is.list(params) || !all(vapply(params, .ag_capture_is_param, logical(1)))))
    stop("ag_capture: params must be a list of ag_tensors or ag_buffers", call. = FALSE)
  cache <- new.env(parent = emptyenv())
  # The recording used by the previous call, and what selected it. A replay is
  # a few hundred microseconds, so building the cache key with paste() on every
  # call was a visible share of it (inst/examples/bench_capture_replay.R in
  # rltoolsR); the common case -- same names, shapes and modes as last time --
  # compares a few short vectors instead.
  last <- new.env(parent = emptyenv())
  step <- function(...) {
    args <- .ag_capture_args(list(...))
    nms <- names(args)
    st <- .ag_device_state
    # set by ag_capture_repeat() for the duration of one call
    times <- last$times %||% 1L
    if (!identical(st$device, "gpu") || is.null(st$backend)) {
      if (times == 1L) return(.ag_capture_plain(fn, args))
      for (i in seq_len(times)) .ag_capture_plain(fn, args)
      return(invisible(NULL))
    }
    dims  <- unlist(lapply(args, dim), use.names = FALSE)
    modes <- c(.ag_compute_dtype(), st$matmul_precision %||% "default")
    cap <- last$cap
    if (is.null(cap) || !identical(nms, last$nms) || !identical(dims, last$dims) ||
        !identical(modes, last$modes)) {
      sig <- paste(paste(nms, vapply(args, function(a) paste(dim(a), collapse = "x"), ""),
                         sep = ":", collapse = ","), modes[1L], modes[2L])
      cap <- cache[[sig]]
      last$sig <- sig
    }
    if (!is.null(cap) && !.ag_capture_valid(cap, params)) {
      .ag_capture_free(cap)
      cap <- NULL
    }
    if (is.null(cap)) {
      cap <- .ag_capture_build(fn, args, params)
      cache[[last$sig]] <- cap
    }
    last$cap <- cap; last$nms <- nms; last$dims <- dims; last$modes <- modes
    .ag_capture_replay(cap, args, times)
  }
  attr(step, "captures") <- cache
  class(step) <- c("ag_captured", "function")
  step
}

#' Run a captured function several times in one go
#'
#' Calls \code{f}, a function from \code{\link{ag_capture}()} that only writes
#' (\code{\link{ag_set_rows}}) and returns \code{NULL}, \code{n} times with
#' the same arguments. On the GPU the inputs are uploaded once and the
#' recorded graph is queued \code{n} times without waiting in between; R waits
#' once, at the end. Each run sees the device state the previous one left (a
#' counter, game boards): the backend orders the runs by the memory they
#' share. On the CPU \code{f} is simply called \code{n} times.
#'
#' On Vulkan the graph's command buffers are recorded once and submitted
#' \code{n} times in one go, so a run costs no command recording on the CPU;
#' the recording is made again after a device buffer it uses was freed (its
#' handle may be reused). Where that is not possible (no push descriptors) each run
#' is recorded and queued as usual. \code{GGMLR_VK_GRAPH_RECORD=0} turns the
#' recorded path off.
#'
#' @param f A function returned by \code{\link{ag_capture}()} whose recording
#'   has no outputs.
#' @param n Number of runs, a positive whole number.
#' @param ... Named arguments of \code{f}, as for a single call.
#' @return \code{NULL}, invisibly.
#' @seealso \code{\link{ag_capture}}, \code{\link{ag_set_rows}}
#' @export
ag_capture_repeat <- function(f, n, ...) {
  if (!inherits(f, "ag_captured") || is.null(environment(f)$last))
    stop("ag_capture_repeat: f must be a function returned by ag_capture()", call. = FALSE)
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 1 || n != round(n))
    stop("ag_capture_repeat: n must be a positive whole number", call. = FALSE)
  last <- environment(f)$last
  last$times <- as.integer(n)
  on.exit(last$times <- NULL, add = TRUE)
  f(...)
  invisible(NULL)
}

#' Release the device memory held by a captured function
#'
#' Frees every recording made by \code{ag_capture()}'s function \code{f}. The
#' function stays usable: the next call records again. Recordings are also
#' released when the persistent device pool is reset (device switch).
#'
#' @param f A function returned by \code{\link{ag_capture}}.
#' @return \code{f}, invisibly.
#' @export
ag_capture_free <- function(f) {
  cache <- attr(f, "captures")
  if (!is.environment(cache)) stop("ag_capture_free: not a captured function", call. = FALSE)
  for (k in ls(cache, all.names = TRUE)) {
    .ag_capture_free(cache[[k]])
    rm(list = k, envir = cache)
  }
  invisible(f)
}

#' Capture a whole training step as reusable device graphs
#'
#' Like \code{\link{ag_capture}}, for a training step: the forward pass in
#' \code{fn}, its backward pass and the Adam update are recorded once, and every
#' later call is upload the inputs, two graph computes, download the outputs --
#' no R-level op dispatch. On the CPU the wrapper runs the same step as ordinary
#' code (\code{zero_grad}, tape, \code{backward}, \code{step}).
#'
#' Two graphs, not one: forward with backward, then the update. Adam writes each
#' weight in place, and a weight is also read inside the first graph (by the
#' forward, and by the backward of the layer below) through tensors that share
#' its memory but are not connected to the update in the graph -- so a single
#' graph would leave the order of that read and that write to the backend.
#'
#' \code{fn} is called WITHOUT \code{with_grad_tape}: the wrapper records the
#' tape itself. Everything \code{ag_capture} requires of \code{fn} applies: data
#' only through the arguments, no reading values back, weights on the device.
#' The backward must be one that runs as a graph (the ops the graph backward
#' supports).
#'
#' Input names and shapes are fixed by the first call; a call with other shapes
#' is an error (pad a short last minibatch, or run it uncaptured). The step
#' counter and the learning rate are read from \code{optimizer} on every call,
#' so learning-rate schedules work, and \code{optimizer$state()} stays valid.
#' Outputs are the values of the forward pass, before the update, as in an
#' ordinary training loop. After a captured step \code{$grad} is not set.
#'
#' Supported: \code{\link{optimizer_adam}} without gradient accumulation, F32
#' compute dtype.
#'
#' Gradient clipping by global norm (\code{max_grad_norm}): the norm of all
#' gradients is computed in the first graph and every gradient is multiplied by
#' \code{min(1, max_grad_norm / (norm + 1e-6))} before the update. Branch-free,
#' unlike \code{\link{clip_grad_norm}}: for a norm within about 1e-6 below
#' \code{max_grad_norm} the factor is slightly under 1 instead of exactly 1.
#' The pre-clipping norm is returned as an extra output \code{grad_norm}. A
#' non-finite norm is an error raised BEFORE the update: weights, moments and
#' the step counter are left as they were. The moments accumulate clipped
#' gradients, so optimizer state saved with and without clipping is not
#' interchangeable on resume. Change the limit between calls with
#' \code{\link{ag_capture_set}} (no re-recording).
#'
#' @param fn Function of named ag_tensor arguments returning the loss (an
#'   ag_tensor), or a named list of ag_tensors with an element \code{loss}.
#' @param params Optional list of further ag_tensors the recording references
#'   (e.g. frozen weights) and \code{\link{ag_buffer}}s read with
#'   \code{\link{ag_get_rows}}; the optimizer's parameters are always included.
#' @param max_grad_norm \code{NULL} (no clipping) or a positive number: clip
#'   gradients to this global norm before the update.
#' @param optimizer An optimizer from \code{\link{optimizer_adam}}, created on
#'   the GPU when the step is to run there.
#' @return A function taking the same named arguments as numeric matrices and
#'   returning the loss as a matrix, or a named list of matrices.
#' @seealso \code{\link{ag_capture}}, \code{\link{ag_capture_set}},
#'   \code{\link{ag_capture_free}}
#' @export
#' @examples
#' \donttest{
#' W <- ag_param(matrix(rnorm(6, sd = 0.1), 2, 3))
#' opt <- optimizer_adam(list(W), lr = 1e-2)
#' train <- ag_capture_step(function(x, y) ag_mse_loss(ag_matmul(W, x), y),
#'                          optimizer = opt)
#' for (i in 1:5) loss <- train(x = matrix(1, 3, 4), y = matrix(0, 2, 4))
#' }
ag_capture_step <- function(fn, params = NULL, optimizer, max_grad_norm = NULL) {
  stopifnot(is.function(fn))
  if (!is.null(max_grad_norm)) .ag_capture_check_clip(max_grad_norm)
  if (!is.null(params) && (!is.list(params) || !all(vapply(params, .ag_capture_is_param, logical(1)))))
    stop("ag_capture_step: params must be a list of ag_tensors or ag_buffers", call. = FALSE)
  if (!inherits(optimizer, "ag_optimizer_adam"))
    stop("ag_capture_step: only optimizer_adam() is supported", call. = FALSE)
  # One recording: shapes are fixed. Kept in an env of its own, so that
  # ag_capture_free() finds only recordings in it.
  cache <- new.env(parent = emptyenv())
  fixed <- new.env(parent = emptyenv())
  # Settings read on every call (ag_capture_set changes them): not recorded.
  settings <- new.env(parent = emptyenv())
  settings$max_grad_norm <- if (!is.null(max_grad_norm)) as.double(max_grad_norm)
  step <- function(...) {
    t0 <- if (.ag_capture_prof_on()) as.numeric(Sys.time())
    args <- .ag_capture_args(list(...))
    nms  <- names(args)
    dims <- lapply(args, dim)
    if (is.null(fixed$dims)) {
      fixed$dims <- dims
    } else if (!identical(dims, fixed$dims)) {
      fmt <- function(d) paste(names(d), vapply(d, paste, "", collapse = "x"),
                               sep = ":", collapse = ", ")
      stop("ag_capture_step: inputs (", fmt(dims), ") differ from the first call (",
           fmt(fixed$dims), "); shapes are fixed by the recording -- pad the ",
           "batch or run this step uncaptured", call. = FALSE)
    }
    st <- .ag_device_state
    if (!identical(st$device, "gpu") || is.null(st$backend))
      return(.ag_capture_step_plain(fn, args, optimizer, settings$max_grad_norm))
    modes <- c(.ag_compute_dtype(), st$matmul_precision %||% "default")
    cap <- cache$step
    if (!is.null(cap) && (!identical(modes, fixed$modes) || !.ag_capture_valid(cap, params))) {
      .ag_capture_free(cap)
      cap <- NULL
    }
    if (is.null(cap)) {
      cap <- .ag_capture_build(fn, args, params, optimizer, settings)
      cache$step <- cap
      fixed$modes <- modes
      if (!is.null(t0)) t0 <- as.numeric(Sys.time())   # recording not timed
    }
    if (!is.null(t0)) .ag_capture_prof_add("call: args+checks", t0)
    .ag_capture_replay(cap, args)
  }
  attr(step, "captures") <- cache
  attr(step, "settings") <- settings
  class(step) <- c("ag_captured", "function")
  step
}

#' Change a setting of a captured training step
#'
#' Settings are inputs of the recording, read on every call, so changing one
#' does not record again.
#'
#' @param f A function returned by \code{\link{ag_capture_step}}.
#' @param max_grad_norm A positive number: the new gradient-clipping limit. The
#'   step must have been created with a \code{max_grad_norm} (clipping cannot be
#'   switched on or off after creation).
#' @return \code{f}, invisibly.
#' @seealso \code{\link{ag_capture_step}}
#' @export
ag_capture_set <- function(f, max_grad_norm) {
  settings <- attr(f, "settings")
  if (!is.environment(settings))
    stop("ag_capture_set: not a function from ag_capture_step()", call. = FALSE)
  if (!missing(max_grad_norm)) {
    if (is.null(settings$max_grad_norm))
      stop("ag_capture_set: this step was created without max_grad_norm; ",
           "create it with one to clip", call. = FALSE)
    .ag_capture_check_clip(max_grad_norm)
    settings$max_grad_norm <- as.double(max_grad_norm)
  }
  invisible(f)
}

# Zero or a negative limit gives a zero or inverted scale -- silently wrong.
.ag_capture_check_clip <- function(x) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x <= 0)
    stop("max_grad_norm must be one positive finite number", call. = FALSE)
  invisible(NULL)
}

# Named numeric matrices from the call's arguments (a vector is one column).
.ag_capture_args <- function(args) {
  nms <- names(args)
  if (!length(args) || is.null(nms) || any(!nzchar(nms)) || anyDuplicated(nms))
    stop("ag_capture: call the captured function with named arguments",
         call. = FALSE)
  for (i in seq_along(args)) {
    a <- args[[i]]
    if (!is.matrix(a) || !is.double(a)) {
      a <- as.matrix(a)
      storage.mode(a) <- "double"
      args[[i]] <- a
    }
  }
  args
}

# CPU path of ag_capture_step: the same step as ordinary code.
.ag_capture_step_plain <- function(fn, args, opt, max_grad_norm = NULL) {
  opt$zero_grad()
  ins <- lapply(args, ag_tensor)
  res <- NULL
  with_grad_tape(res <- do.call(fn, ins))
  loss <- if (is_ag_tensor(res)) res else res[["loss"]]
  if (!is_ag_tensor(loss))
    stop("ag_capture_step: fn must return the loss, or a named list with an ",
         "element 'loss'", call. = FALSE)
  grads <- backward(loss)
  out <- if (is_ag_tensor(res)) list(loss = as.matrix(res)) else lapply(res, as.matrix)
  if (!is.null(max_grad_norm)) {
    ps <- opt$params
    # Checked first: clip_grad_norm() fails on a NaN norm with an R condition
    # error ("missing value where TRUE/FALSE needed") rather than naming it.
    gn <- sqrt(sum(vapply(ps, function(p) {
      g <- get0(as.character(p$id), envir = grads)
      if (is.null(g)) 0 else sum(.ag_as_matrix(g)^2)
    }, numeric(1))))
    .ag_capture_check_norm(gn)
    clip_grad_norm(ps, grads, max_grad_norm)
    out$grad_norm <- matrix(gn, 1L, 1L)
  }
  opt$step(grads)
  if (is_ag_tensor(res) && is.null(max_grad_norm)) out$loss else out
}

# Raised before the update, so a failed step changes nothing.
.ag_capture_check_norm <- function(gn) {
  if (!is.finite(gn))
    stop("ag_capture_step: the gradient norm is ", gn, "; the update was not ",
         "applied (weights, moments and the step counter are unchanged)",
         call. = FALSE)
  invisible(NULL)
}

# CPU (or no backend): run fn as ordinary ag_* code, same return shape.
.ag_capture_plain <- function(fn, args) {
  res <- do.call(fn, lapply(args, ag_tensor))
  if (is.null(res) || (is.list(res) && !is_ag_tensor(res) && !length(res)))
    return(invisible(NULL))                 # writes only (ag_set_rows)
  if (is_ag_tensor(res)) return(as.matrix(res))
  lapply(res, as.matrix)
}

# The device tensor each parameter points at, read straight from the tensor.
# Built via .ag_handle_of() this was ~60% of a replay's R time (a handle object
# and a liveness check per parameter, every call); liveness is covered by the
# persistent-pool generation in .ag_capture_valid().
.ag_capture_param_ptrs <- function(params) {
  lapply(params, function(p) {
    if (!inherits(p, "ag_buffer")) return(p[["ptr"]])
    # the allocation count, not the pointer: a buffer moved to the host and
    # back may get the same address for new memory
    if (is.null(p$core$ptr)) NULL else c(p$core$id, p$core$gen)
  })
}

# A recording may reference weights (ag_tensor) and device buffers (ag_buffer,
# read by ag_get_rows): a buffer freed or moved to the host re-records.
.ag_capture_is_param <- function(p) is_ag_tensor(p) || inherits(p, "ag_buffer")

.ag_capture_valid <- function(cap, params) {
  st <- .ag_device_state
  isTRUE(cap$alive) &&
    identical(cap$backend, st$backend) &&
    identical(cap$p_gen, st$p_ctx_gen) &&
    (is.null(params) || identical(.ag_capture_param_ptrs(params), cap$param_ptrs)) &&
    (is.null(cap$opt) || identical(.ag_capture_opt_ptrs(cap$opt), cap$opt_ptrs))
}

# Everything the recorded Adam graph references besides the gradients: weights
# and moments (the hyperparameters are the recording's own input). A moment
# re-created (load_state after a device switch) or a weight re-uploaded means
# re-recording.
.ag_capture_opt_ptrs <- function(opt) {
  c(.ag_capture_param_ptrs(opt$params),
    lapply(opt$m, function(h) if (.ag_is_handle(h)) h$ptr),
    lapply(opt$v, function(h) if (.ag_is_handle(h)) h$ptr))
}

# What the first version records: Adam's fused device step, nothing else.
.ag_capture_check_opt <- function(opt) {
  if (!inherits(opt, "ag_optimizer_adam"))
    stop("ag_capture_step: only optimizer_adam() is supported", call. = FALSE)
  if (opt$accumulate_steps > 1L)
    stop("ag_capture_step: gradient accumulation (accumulate_steps > 1) is not ",
         "supported", call. = FALSE)
  if (!identical(.ag_compute_dtype(), "f32"))
    stop("ag_capture_step: the fused Adam kernel is F32-only; compute dtype is ",
         .ag_compute_dtype(), call. = FALSE)
  for (i in seq_along(opt$params)) {
    h <- list(.ag_handle_of(opt$params[[i]]), opt$m[[i]], opt$v[[i]])
    if (!isTRUE(opt$resident) ||
        !all(vapply(h, function(x) .ag_is_handle(x) && .ag_handle_live(x), logical(1))))
      stop("ag_capture_step: the optimizer's state is not on the device (create ",
           "the optimizer after ag_device(\"gpu\"), or recreate it after a ",
           "device switch)", call. = FALSE)
    if (!all(vapply(h, function(x) ggml_tensor_type(x$ptr) == GGML_TYPE_F32, logical(1))))
      stop("ag_capture_step: the fused Adam kernel is F32-only; parameter ", i,
           " or its moments are not F32", call. = FALSE)
  }
  invisible(NULL)
}

# Make declared parameters resident in the persistent pool, as ag_param() does
# on creation. A device switch materialises them to the host and frees the
# pool, so back on the GPU they would be uploaded during the recording -- which
# the guard refuses -- although the caller did everything right.
.ag_capture_make_resident <- function(params) {
  for (p in params) {
    if (inherits(p, "ag_buffer")) {
      if (!isTRUE(p$core$freed)) .ag_buffer_on_device(p$core)
      next
    }
    if (!identical(p$device, "gpu") || !is.null(.ag_handle_of(p))) next
    d   <- .ag_data(p)
    dt  <- p$dtype %||% .ag_device_state$dtype
    ptr <- .ag_r_to_gpu(d, dtype = dt, scope = "persistent")
    .ag_data_set_handle(p, .ag_handle(ptr, dim(d), scope = "persistent"))
    .ag_register_resident_value(p)
  }
  invisible(NULL)
}

# `opt` (ag_capture_step only): an Adam optimizer. The tape is recorded too, its
# graph backward appended to the forward, and the optimizer step recorded as a
# SECOND graph -- see ag_capture_step() for why not one.
.ag_capture_build <- function(fn, args, params, opt = NULL, settings = NULL) {
  clip <- !is.null(opt) && !is.null(settings$max_grad_norm)
  st <- .ag_device_state
  .ag_capture_make_resident(params)
  if (!is.null(opt)) {
    .ag_capture_make_resident(opt$params)
    .ag_capture_check_opt(opt)
    # backward() adds to an existing $grad, which means reading it back.
    opt$zero_grad()
  }
  # Anything queued belongs to the caller's pass pool, not to the recording.
  if (.ag_defer_len()) .ag_defer_drain()

  id <- .ag_capture_reg$next_id <- .ag_capture_reg$next_id + 1L
  slots <- .ag_pool_slots("pass")
  saved <- list(ctxs = st[[slots$ctxs]], bufs = st[[slots$bufs]], cur = st[[slots$cur]],
                gen = st[[slots$gen]], buffer = st$buffer)
  # A negative generation cannot collide with the pass pool's own counter, so
  # a handle made during the recording is never mistaken for a live pass one.
  st[[slots$ctxs]] <- list(); st[[slots$bufs]] <- list(); st[[slots$cur]] <- NULL
  st[[slots$gen]] <- -id; st$buffer <- NULL

  prev_graph <- ag_graph_mode(TRUE)
  prev_check <- .ag_defer$check
  .ag_defer$check <- FALSE              # check mode drains per op = a launch
  prev_xfer  <- list(enabled = .ag_xfer$enabled, counts = .ag_xfer$counts)
  prev_launch <- .ag_launch$by_site
  prev_tape  <- list(enabled = .ag_tape$enabled, nodes = .ag_tape$nodes)
  prev_bwd   <- list(enabled = .ag_bwd$enabled, resident = .ag_bwd$resident)
  cap_ctxs <- NULL; cap_bufs <- NULL; ctx_graph <- NULL; ctx_graph_opt <- NULL
  ok <- FALSE
  on.exit({
    .ag_defer_discard()
    if (!is.null(opt)) {
      # The recorded gradients point into the recording, and were never
      # computed: no $grad and no rescue at the next tape reset may keep them.
      opt$zero_grad()
      for (p in params) p$grad <- NULL
      .ag_tape$enabled <- prev_tape$enabled; .ag_tape$nodes <- prev_tape$nodes
      .ag_bwd$enabled <- prev_bwd$enabled; .ag_bwd$resident <- prev_bwd$resident
    }
    cap_ctxs <- st[[slots$ctxs]]; cap_bufs <- st[[slots$bufs]]
    st[[slots$ctxs]] <- saved$ctxs; st[[slots$bufs]] <- saved$bufs
    st[[slots$cur]] <- saved$cur; st[[slots$gen]] <- saved$gen; st$buffer <- saved$buffer
    .ag_xfer$enabled <- prev_xfer$enabled; .ag_xfer$counts <- prev_xfer$counts
    .ag_launch$by_site <- prev_launch
    .ag_defer$check <- prev_check
    ag_graph_mode(prev_graph)
    if (!ok) {
      for (b in cap_bufs) tryCatch(ggml_backend_buffer_free(b), error = function(e) NULL)
      for (x in c(cap_ctxs, list(ctx_graph, ctx_graph_opt)))
        if (!is.null(x)) tryCatch(ggml_free(x), error = function(e) NULL)
    }
  }, add = TRUE)

  # Inputs: allocated and filled before fn runs.
  #
  # All of them -- plus, for a training step, the clipping limit and this
  # recording's own AdamW hyperparameters -- are views into ONE tensor, so a
  # replay sends them in one transfer instead of one each (~0.11 ms apiece,
  # 6-8 per step on the PPO measurement). Every segment starts on a 256-byte
  # boundary: a view's offset must meet the device's storage-buffer offset
  # alignment, and 256 bytes covers every value Vulkan allows.
  ttype <- .ag_dtype_to_ggml(.ag_compute_dtype())
  ctx <- .ag_ctx_ensure(length(args) + 4L)
  esz <- ggml_type_size(ttype)
  seg <- function(n) ceiling(n * esz / 256) * 256 / esz       # padded length
  lens <- vapply(args, function(a) as.double(length(a)), numeric(1))
  offs <- cumsum(c(0, vapply(lens, seg, numeric(1))))
  n_in <- offs[length(offs)]
  clip_off <- n_in
  hp_off   <- n_in + if (clip) seg(1) else 0
  in_len   <- hp_off + if (!is.null(opt)) 7 else 0
  in_buf <- ggml_new_tensor_1d(ctx, ttype, in_len)
  in_ptrs <- lapply(seq_along(args), function(k)
    ggml_view_2d(ctx, in_buf, nrow(args[[k]]), ncol(args[[k]]),
                 nrow(args[[k]]) * esz, offs[k] * esz))
  names(in_ptrs) <- names(args)
  # The clipping limit is an input too, so ag_capture_set() needs no re-record.
  clip_ptr <- if (clip) ggml_view_2d(ctx, in_buf, 1L, 1L, esz, clip_off * esz)
  hp_ptr   <- if (!is.null(opt)) ggml_view_1d(ctx, in_buf, 7L, hp_off * esz)
  .ag_ctx_flush(ctx)
  in_pos <- lapply(seq_along(args), function(k) offs[k] + seq_len(lens[k]))
  names(in_pos) <- names(args)
  in_vals <- numeric(in_len)
  for (nm in names(args)) in_vals[in_pos[[nm]]] <- as.numeric(args[[nm]])
  .ag_xfer_up(in_buf, in_vals, "capture inputs")
  ins <- lapply(names(args), function(nm)
    .ag_tensor_from_handle(.ag_handle(in_ptrs[[nm]], dim(args[[nm]]))))
  names(ins) <- names(args)

  # The recording: everything from here on must stay in the graph.
  .ag_xfer$enabled <- TRUE
  .ag_xfer$counts  <- new.env(parent = emptyenv())
  .ag_launch$by_site <- list()
  if (!is.null(opt)) {
    # The tape without with_grad_tape(): its pass-pool reset would free the
    # recording's own contexts, which are the pass slots right now.
    .ag_tape$enabled <- TRUE; .ag_tape$nodes <- list()
    .ag_bwd$enabled <- TRUE; .ag_bwd$resident <- TRUE
  }

  # ag_set_rows() during fn adds its write nodes to .ag_capture_reg$effects
  .ag_capture_reg$effects <- list()
  .ag_capture_reg$recording <- TRUE
  on.exit({ .ag_capture_reg$recording <- FALSE; .ag_capture_reg$effects <- list() },
          add = TRUE)
  res <- do.call(fn, ins)
  .ag_capture_reg$recording <- FALSE
  effects <- .ag_capture_reg$effects
  if (!is.null(opt)) .ag_tape$enabled <- FALSE
  single <- is_ag_tensor(res)
  no_out <- is.null(res) || (is.list(res) && !single && !length(res))
  if (no_out) {
    if (!is.null(opt))
      stop("ag_capture_step: fn must return the loss, or a named list with an ",
           "element 'loss'", call. = FALSE)
    if (!length(effects))
      stop("ag_capture: fn returned nothing and wrote nothing with ag_set_rows()",
           call. = FALSE)
    outs <- list()
  } else {
    outs <- if (single) list(res) else res
    if (!is.list(outs) || !all(vapply(outs, is_ag_tensor, logical(1))))
      stop("ag_capture: fn must return an ag_tensor, a list of ag_tensors or NULL",
           call. = FALSE)
    if (!single && (is.null(names(outs)) || any(!nzchar(names(outs)))))
      stop("ag_capture: a list returned by fn must be fully named", call. = FALSE)
  }

  grad_items <- list()
  if (!is.null(opt)) {
    loss <- if (single) res else res[["loss"]]
    if (!is_ag_tensor(loss))
      stop("ag_capture_step: fn must return the loss, or a named list with an ",
           "element 'loss'", call. = FALSE)
    backward(loss)
    if (!identical(.ag_bwd$last_path, "graph"))
      stop("ag_capture_step: the backward could not run as one graph (",
           .ag_bwd$last_path, "); only ops the graph backward supports can be ",
           "recorded", call. = FALSE)
    for (i in seq_along(opt$params)) {
      g <- opt$params[[i]]$grad
      if (is.null(g)) next          # not reached by the loss: Adam skips it too
      if (!.ag_is_handle(g))
        stop("ag_capture_step: a gradient came back to R during the recording",
             call. = FALSE)
      grad_items[[length(grad_items) + 1L]] <- list(i = i, p = opt$params[[i]], g = g$ptr)
    }
    if (!length(grad_items))
      stop("ag_capture_step: the loss depends on none of the optimizer's parameters",
           call. = FALSE)
  }

  norm_node <- NULL
  if (clip) {
    if (!single && "grad_norm" %in% names(outs))
      stop("ag_capture_step: 'grad_norm' is the clipping output; fn must not ",
           "return one", call. = FALSE)
    # Global norm over all gradients, then one factor for all of them:
    # min(1, max / (norm + 1e-6)), branch-free (see ag_capture_step). The sum
    # runs in the same order on every replay. Adam, in the second graph, reads
    # the scaled gradients.
    cx <- .ag_ctx_ensure(4L * length(grad_items) + 8L)
    ss <- NULL
    for (it in grad_items) {
      s  <- ggml_sum(cx, ggml_sqr(cx, it$g))
      ss <- if (is.null(ss)) s else ggml_add(cx, ss, s)
    }
    norm_node <- ggml_sqrt(cx, ss)
    scale <- ggml_clamp(cx, ggml_div(cx, clip_ptr, ggml_scale_bias(cx, norm_node, 1, 1e-6)),
                        -3.0e38, 1)
    for (k in seq_along(grad_items))
      grad_items[[k]]$g <- ggml_mul(cx, grad_items[[k]]$g, scale)
  }

  .ag_capture_guard(names(args))

  out_h <- lapply(outs, .ag_handle_of)
  if (any(vapply(out_h, is.null, logical(1))))
    stop("ag_capture: an output of fn is not on the device (computed in R?)",
         call. = FALSE)
  if (clip) {
    if (single) { names(out_h) <- "loss"; single <- FALSE }
    out_h$grad_norm <- list(ptr = norm_node, shape = c(1L, 1L))
  }

  # All outputs flattened and concatenated into one tensor, so a replay reads
  # them back in ONE transfer: each download is a separate device sync, ~64 us
  # apiece on the measurement that motivated this. Outputs of different types
  # (not the case for ag_* results) keep one transfer each.
  out_ptrs <- lapply(out_h, function(h) h$ptr)
  out_len  <- vapply(out_h, function(h) prod(as.double(h$shape)), numeric(1))
  types    <- vapply(out_ptrs, ggml_tensor_type, numeric(1))
  root <- NULL
  if (length(unique(types)) == 1L) {
    cx <- .ag_ctx_ensure(3L * length(out_ptrs) + 2L)
    for (i in seq_along(out_ptrs)) {
      p <- out_ptrs[[i]]
      if (!ggml_is_contiguous(p)) p <- ggml_cont(cx, p)
      flat <- ggml_reshape_1d(cx, p, out_len[i])
      root <- if (is.null(root)) flat else ggml_concat(cx, root, flat, 0L)
    }
    if (length(out_ptrs) == 1L) root <- ggml_cont(cx, root)
  }

  ctx_graph <- ggml_init(.ag_graph_ctx_bytes(), no_alloc = TRUE)
  if (is.null(ctx_graph))
    stop("ag_capture: failed to create a ggml context for the graph", call. = FALSE)
  # The outputs first, then the ag_set_rows() writes in the order they were
  # made: a read of the written buffer reaches the graph before the write.
  roots <- c(if (!is.null(root)) list(root) else out_ptrs,
             lapply(effects, function(h) h$ptr))
  graph <- ggml_build_forward_expand(ctx_graph, roots[[1L]])
  for (p in roots[-1L]) ggml_graph_expand(graph, p)
  # Gradients are roots of their own: not reachable from the outputs.
  for (it in grad_items) ggml_graph_expand(graph, it$g)

  # One buffer per context for every node the recording built.
  for (cx in st[[slots$ctxs]]) .ag_ctx_flush(cx)

  graph_opt <- NULL
  if (!is.null(opt)) {
    # The constants the guard let through (the backward seed), uploaded once.
    for (u in .ag_defer$uploads) .ag_xfer_up(u$ptr, u$val, "capture constants")

    # The Adam step as its own graph over leaf aliases of the gradients, built
    # after the flush (an alias needs its source's memory). Each step node is a
    # view of its weight whose buffer ggml_opt_step_adamw() sets itself.
    cx <- .ag_ctx_ensure(2L * length(grad_items) + 4L)
    nodes <- lapply(grad_items, function(it)
      ggml_opt_step_adamw(cx, .ag_handle_of(it$p)$ptr,
                          .Call("R_ggml_leaf_alias", cx, it$g, PACKAGE = "ggmlR"),
                          opt$m[[it$i]]$ptr, opt$v[[it$i]]$ptr, hp_ptr))
    ctx_graph_opt <- ggml_init(.ag_graph_ctx_bytes(), no_alloc = TRUE)
    if (is.null(ctx_graph_opt))
      stop("ag_capture_step: failed to create a ggml context for the Adam graph",
           call. = FALSE)
    graph_opt <- ggml_build_forward_expand(ctx_graph_opt, nodes[[1L]])
    for (nd in nodes[-1L]) ggml_graph_expand(graph_opt, nd)
  }

  cap <- new.env(parent = emptyenv())
  cap$id         <- as.character(id)
  cap$alive      <- TRUE
  cap$backend    <- st$backend
  cap$p_gen      <- st$p_ctx_gen
  cap$param_ptrs <- if (!is.null(params)) .ag_capture_param_ptrs(params)
  cap$in_buf     <- in_buf
  cap$in_pos     <- in_pos          # input name -> its elements in in_vals
  cap$in_vals    <- in_vals         # what in_buf holds right now
  cap$last_in    <- args           # what the input tensors hold right now
  cap$out_ptrs   <- out_ptrs
  cap$out_root   <- root
  cap$out_len    <- out_len
  cap$out_shapes <- lapply(out_h, function(h) h$shape)
  cap$single     <- single
  cap$graph      <- graph
  cap$ctx_graph  <- ctx_graph
  cap$opt        <- opt
  cap$graph_opt  <- graph_opt
  cap$ctx_graph_opt <- ctx_graph_opt
  cap$clip_pos   <- if (clip) clip_off + 1
  cap$hp_pos     <- if (!is.null(opt)) hp_off + seq_len(7L)
  cap$settings   <- settings
  cap$last_clip  <- NULL           # the limit is written at the first replay
  cap$opt_ptrs   <- if (!is.null(opt)) .ag_capture_opt_ptrs(opt)
  cap$grad_params <- lapply(grad_items, function(it) it$p)
  cap$ctxs       <- st[[slots$ctxs]]
  cap$bufs       <- st[[slots$bufs]]
  assign(cap$id, cap, envir = .ag_capture_reg$items)
  ok <- TRUE
  cap
}

# Refuse a recording that would freeze host data or did work outside the graph.
.ag_capture_guard <- function(input_names) {
  bad <- character(0)
  cnt <- .ag_xfer$counts
  for (k in ls(cnt, all.names = TRUE)) {
    r <- cnt[[k]]
    bad <- c(bad, sprintf("%s '%s' x%d", if (r$dir == "up") "upload" else "download",
                          r$site, r$n))
  }
  # Data-independent constants (the backward seed) are allowed: the recording
  # uploads them once, and every replay rightly reuses them.
  n_up <- sum(!vapply(.ag_defer$uploads, function(u) isTRUE(u$const), logical(1)))
  if (n_up)
    bad <- c(bad, sprintf("upload of %d host value(s) queued by: %s",
                          n_up,
                          paste(unique(.ag_defer$labels), collapse = ", ")))
  l <- unlist(.ag_launch$by_site)
  if (length(l))
    bad <- c(bad, sprintf("eager launch '%s' x%d", names(l), as.integer(l)))
  if (length(bad))
    stop("ag_capture: the recording did work outside the graph, which a replay ",
         "would not repeat (inputs: ", paste(input_names, collapse = ", "), "):\n  ",
         paste(bad, collapse = "\n  "),
         "\nPass data the step depends on as an input; make sure weights are ",
         "resident on the device; do not read values back inside fn.",
         call. = FALSE)
  invisible(NULL)
}

.ag_capture_replay <- function(cap, args, times = 1L) {
  if (times > 1L && (length(cap$out_shapes) || !is.null(cap$opt)))
    stop("ag_capture_repeat: only a captured function that returns NULL ",
         "(writes with ag_set_rows only) can run several times in one go",
         call. = FALSE)
  # Training steps only: forward-only replays (rl_act and the like) would mix in.
  t0 <- if (!is.null(cap$opt) && .ag_capture_prof_on()) as.numeric(Sys.time())
  tick <- function(stage) if (!is.null(t0)) t0 <<- .ag_capture_prof_add(stage, t0)
  # Profiling splits a compute into recording+submit (CPU) and the wait for the
  # device: Vulkan graph_compute returns after submitting when the device
  # supports async, and ggml_backend_synchronize() waits.
  run_graph <- function(graph, site, stage) {
    if (is.null(t0)) return(.ag_graph_compute(cap$backend, graph, site))
    n <- .ag_launch$by_site[[site]]
    .ag_launch$by_site[[site]] <- if (is.null(n)) 1 else n + 1
    .ag_capture_reg$prof_nodes[stage] <- ggml_graph_n_nodes(graph)
    s <- ggml_backend_graph_compute_async(cap$backend, graph)
    tick(paste(stage, "record+submit"))
    ggml_backend_synchronize(cap$backend)
    tick(paste(stage, "wait"))
    s
  }
  # A pending write to a weight (graph mode) must land before the graph reads it.
  if (.ag_defer_len()) .ag_defer_drain()
  tick("replay: drain")
  # One transfer for everything that changed (see .ag_capture_build: all
  # inputs are views into cap$in_buf). An input whose value is the one already
  # on the device is not rewritten -- normalization statistics, for one, are the
  # same for a whole rollout -- and when nothing changed nothing is sent.
  #
  # A training step always sends: AdamW's bias corrections move with t. They
  # are written for t + 1 BEFORE the norm check, into this recording's own
  # input segment -- not optimizer state: a refused step changes no weight, no
  # moment and not t, and the next replay rewrites the segment.
  opt <- cap$opt
  changed <- FALSE
  for (nm in names(cap$in_pos)) {
    v <- args[[nm]]
    if (identical(v, cap$last_in[[nm]])) next
    cap$in_vals[cap$in_pos[[nm]]] <- as.numeric(v)
    cap$last_in[[nm]] <- v
    changed <- TRUE
  }
  if (!is.null(cap$clip_pos)) {
    mx <- cap$settings$max_grad_norm
    if (!identical(mx, cap$last_clip)) {
      cap$in_vals[cap$clip_pos] <- mx
      cap$last_clip <- mx
      changed <- TRUE
    }
  }
  if (!is.null(opt)) {
    cap$in_vals[cap$hp_pos] <- .ag_adam_hp_values(opt, opt$t + 1L)
    changed <- TRUE
  }
  if (changed) .ag_xfer_up(cap$in_buf, cap$in_vals, "capture inputs")
  tick("replay: upload inputs")
  fail <- function(status) {
    cap$last_in <- list()          # unknown state: send everything next time
    cap$last_clip <- NULL
    stop("ag_capture: graph compute returned status ", status, call. = FALSE)
  }
  if (times > 1L) {
    n <- .ag_launch$by_site[["captured graph"]]
    .ag_launch$by_site[["captured graph"]] <- (if (is.null(n)) 0 else n) + times
    .ag_launch$last_nodes <- ggml_graph_n_nodes(cap$graph)
    # Vulkan: the command buffers recorded once and submitted `times` times
    # (no recording on the CPU per run; see ggml_backend_vk_graph_record)
    if (.ag_capture_vk_replay(cap, times)) return(invisible(NULL))
    # otherwise queued without waiting, one wait at the end. Runs that touch
    # the same memory (state written by one, read by the next) are ordered by
    # the backend: Vulkan keeps the unsynchronized writes of earlier graphs
    # until the wait and puts a barrier before an overlapping node.
    for (i in seq_len(times)) {
      status <- ggml_backend_graph_compute_async(cap$backend, cap$graph)
      if (!identical(as.integer(status), 0L)) {
        ggml_backend_synchronize(cap$backend)
        fail(status)
      }
    }
    ggml_backend_synchronize(cap$backend)
    return(invisible(NULL))
  }
  status <- run_graph(cap$graph, "captured graph", "replay: graph 1 (fwd+bwd)")
  if (!identical(as.integer(status), 0L)) fail(status)

  # The outputs are read between the two graphs: they are values of the first,
  # and the clipping norm has to be checked before the update. Still one
  # transfer per step.
  if (!is.null(cap$out_root)) {
    raw <- .ag_xfer_down(cap$out_root, "capture outputs")
    ends <- cumsum(cap$out_len)
    outs <- lapply(seq_along(cap$out_len), function(i) {
      sh <- cap$out_shapes[[i]]
      matrix(raw[(ends[i] - cap$out_len[i] + 1):ends[i]], sh[1L], sh[2L])
    })
    names(outs) <- names(cap$out_shapes)
  } else {
    outs <- Map(function(p, sh) matrix(.ag_xfer_down(p, "capture outputs"), sh[1L], sh[2L]),
                cap$out_ptrs, cap$out_shapes)
  }
  tick("replay: download outputs")

  if (!is.null(opt)) {
    # Nothing of the optimizer's has changed yet: a refused step leaves the
    # weights, the moments and t as they were.
    if (!is.null(cap$clip_pos)) .ag_capture_check_norm(outs$grad_norm[1L])
    tick("replay: norm check")
    # Two graphs, run in order: every read of a weight in the first has
    # finished before the second writes it (see ag_capture_step).
    status <- run_graph(cap$graph_opt, "captured adam step", "replay: graph 2 (adam)")
    if (!identical(as.integer(status), 0L)) fail(status)
    opt$t <- opt$t + 1L
    # Every weight's buffer changed underneath any cached host copy.
    for (p in cap$grad_params) { p$data <- NULL; p$data_gen <- NULL }
  }
  tick("replay: post")
  if (!is.null(t0)) .ag_capture_reg$prof["replays"] <-
    (if (is.na(.ag_capture_reg$prof["replays"])) 0 else .ag_capture_reg$prof["replays"]) + 1
  if (!length(cap$out_shapes)) return(invisible(NULL))   # writes only
  if (cap$single) outs[[1L]] else outs
}

.ag_capture_free <- function(cap) {
  if (!isTRUE(cap$alive)) return(invisible(NULL))
  cap$alive <- FALSE
  # the recorded command buffers point at the buffers freed below
  .ag_capture_vk_rec_free(cap)
  for (b in cap$bufs) tryCatch(ggml_backend_buffer_free(b), error = function(e) NULL)
  for (x in c(cap$ctxs, list(cap$ctx_graph, cap$ctx_graph_opt)))
    if (!is.null(x)) tryCatch(ggml_free(x), error = function(e) NULL)
  cap$bufs <- cap$ctxs <- cap$ctx_graph <- cap$graph <- NULL
  cap$ctx_graph_opt <- cap$graph_opt <- NULL
  if (exists(cap$id, envir = .ag_capture_reg$items, inherits = FALSE))
    rm(list = cap$id, envir = .ag_capture_reg$items)
  invisible(NULL)
}

# ---- recorded command buffers (Vulkan) for ag_capture_repeat() -------------
#
# cap$vk_rec: the step graph's command buffers recorded once by the backend
# (R_ggml_vk_graph_record), replayed n times in one submit. NULL until the
# first repeat; FALSE when the backend refused (not Vulkan, no push
# descriptors, ...), then the async chain is used. A replay that reports
# stale (a buffer the recording uses was destroyed since: its handle may be
# reused) is recorded again once. GGMLR_VK_GRAPH_RECORD=0 switches it off.

.ag_capture_vk_replay <- function(cap, times) {
  if (identical(Sys.getenv("GGMLR_VK_GRAPH_RECORD"), "0") || isFALSE(cap$vk_rec))
    return(FALSE)
  for (attempt in 1:2) {
    if (is.null(cap$vk_rec)) {
      rec <- .Call("R_ggml_vk_graph_record", cap$backend, cap$graph, PACKAGE = "ggmlR")
      if (is.null(rec)) { cap$vk_rec <- FALSE; return(FALSE) }
      cap$vk_rec <- rec
    }
    s <- .Call("R_ggml_vk_graph_replay", cap$backend, cap$vk_rec, as.integer(times),
               PACKAGE = "ggmlR")
    if (identical(s, 0L)) return(TRUE)
    .ag_capture_vk_rec_free(cap)
    if (!identical(s, 1L)) { cap$vk_rec <- FALSE; return(FALSE) }
  }
  FALSE
}

.ag_capture_vk_rec_free <- function(cap) {
  if (inherits(cap$vk_rec, "externalptr"))
    .Call("R_ggml_vk_graph_record_free", cap$vk_rec, PACKAGE = "ggmlR")
  cap$vk_rec <- NULL
  invisible(NULL)
}

# Called by .ag_residency_reset() before the persistent pool goes: recordings
# point into it, and their own buffers must be freed while the backend lives.
.ag_capture_free_all <- function() {
  for (k in ls(.ag_capture_reg$items, all.names = TRUE))
    .ag_capture_free(.ag_capture_reg$items[[k]])
  invisible(NULL)
}

# Diagnostic: the nodes of a ggml graph as a data frame -- op, name, shape and,
# for src0..src2, the 1-based index of the producing node (0 = a leaf, NA = no
# source) and its op. Used to see what a recorded step graph is made of.
.ag_graph_dump <- function(graph) {
  d <- .Call("R_ggml_graph_dump", graph, PACKAGE = "ggmlR")
  data.frame(i = seq_along(d$op), op = d$op, name = d$name,
             shape = apply(d$ne, 1L, function(x) paste(x, collapse = "x")),
             src0 = d$src_index[, 1L], src0_op = d$src_op[, 1L],
             src1 = d$src_index[, 2L], src1_op = d$src_op[, 2L],
             src2 = d$src_index[, 3L], src2_op = d$src_op[, 3L],
             stringsAsFactors = FALSE)
}
