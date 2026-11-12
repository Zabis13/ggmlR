# Device buffers: a large table that lives on the device, is written in place
# by column ranges and read by column indices inside ag_* graphs.
#
# Why. A replay buffer (off-policy RL) holds up to millions of transitions and
# every update reads a random batch of them. As an ag_tensor it would be
# uploaded whole at every write; kept on the host, every batch would be
# uploaded. A buffer keeps the table resident: writes send only the new
# columns, and ag_get_rows() sends only the indices.
#
# Lifetime. The device memory is the buffer's own context and backend buffer,
# outside the residency pools: a pool reset frees everything in it, and the
# buffer must outlive tape resets (pass pool) and be freeable on its own
# (ag_buffer_free). A persistent-pool reset -- a device switch -- still has to
# reach it, because the backend is freed right after: the value is then
# rescued to the host, like a resident weight (.ag_register_resident_value),
# and re-uploaded at the next device use.
#
# The registry holds the CORE of each buffer (pointers and value), not the
# ag_buffer object the caller holds, so the finalizer of that object can still
# run and free the device memory.

.ag_buffer_reg <- new.env(parent = emptyenv())
.ag_buffer_reg$items   <- new.env(parent = emptyenv())   # id -> core
.ag_buffer_reg$next_id <- 0L
.ag_buffer_reg$orphans <- new.env(parent = emptyenv())   # id -> core, collected
                                                        # while a queued node read it

# Columns per transfer chunk when a whole buffer crosses the bus (allocation,
# rescue): bounds the host-side temporary, not the device memory.
.ag_buffer_chunk_elems <- 4194304   # 16 MB of f32

# Share of the free device memory a buffer may take (driver estimates vary).
.ag_buffer_mem_share <- 0.85

#' Device buffer for column-indexed reads
#'
#' A matrix of \code{rows x capacity} kept on the compute device without a
#' gradient: written in place by column ranges with \code{ag_buffer_write()}
#' and read by column indices with \code{\link{ag_get_rows}()} inside ag_*
#' computations, including \code{\link{ag_capture}()} recordings. Typical use
#' is a replay buffer: one column per transition.
#'
#' On the CPU the buffer is an R matrix and the same calls work on it. On the
#' GPU the memory is allocated when the buffer is created (or first used on the
#' device) and zero-filled; creation fails with an error when the buffer needs
#' more than 85\% of the free memory the device reports (driver estimates are
#' approximate). A device switch (\code{ag_device("cpu")}) downloads the value
#' to the host in 16 MB parts before the backend is released (the host copy is
#' an R double matrix, about twice the buffer's device size), and the switch
#' fails with an error if that download does; the next use on the GPU uploads
#' it again, and recordings that read the buffer are then made anew.
#'
#' Writes are synchronous: when \code{ag_buffer_write()} returns, the data is on
#' the device, and any graph computed afterwards, including a captured replay,
#' reads it. Graph nodes queued before the write are computed first, so they
#' read the old value.
#'
#' @param rows Number of rows (values per column).
#' @param capacity Number of columns.
#' @param dtype Storage type. Only \code{"f32"} is supported.
#' @return An object of class \code{ag_buffer} with fields \code{rows},
#'   \code{capacity} and \code{dtype}.
#' @seealso \code{\link{ag_get_rows}}
#' @export
#' @examples
#' buf <- ag_buffer(3, 10)
#' ag_buffer_write(buf, matrix(1:6, 3, 2), col_offset = 4)
#' ag_buffer_read(buf, 4, 2)
#' as.matrix(ag_get_rows(buf, c(5, 4, 5)))
#' ag_buffer_free(buf)
ag_buffer <- function(rows, capacity, dtype = "f32") {
  .ag_buffer_reap()
  rows     <- .ag_buffer_count(rows, "rows")
  capacity <- .ag_buffer_count(capacity, "capacity")
  if (!identical(dtype, "f32"))
    stop("ag_buffer: only dtype = \"f32\" is supported", call. = FALSE)

  .ag_buffer_reg$next_id <- .ag_buffer_reg$next_id + 1L
  core <- new.env(parent = emptyenv())
  core$id       <- as.character(.ag_buffer_reg$next_id)
  core$rows     <- rows
  core$capacity <- capacity
  core$data     <- NULL    # host value, when not on the device
  core$ptr      <- NULL    # device tensor, when on the device
  core$gen      <- 0L      # device allocations so far (recordings compare it)
  core$ctx      <- NULL
  core$bbuf     <- NULL
  core$freed    <- FALSE
  if (.ag_buffer_gpu()) .ag_buffer_alloc(core)
  else core$data <- matrix(0, rows, capacity)

  buf <- new.env(parent = emptyenv())
  buf$core     <- core
  buf$rows     <- rows
  buf$capacity <- capacity
  buf$dtype    <- dtype
  class(buf) <- "ag_buffer"
  reg.finalizer(buf, function(b) .ag_buffer_collect(b$core), onexit = TRUE)
  buf
}

#' @rdname ag_buffer
#' @param buf An \code{ag_buffer}.
#' @param data Numeric matrix with \code{rows} rows (a vector is one column).
#' @param col_offset 0-based index of the first column written or read.
#' @details \code{ag_buffer_write()} writes columns \code{col_offset} to
#'   \code{col_offset + ncol(data) - 1}; the range must lie inside the buffer
#'   (a ring buffer wraps with two writes).
#' @export
ag_buffer_write <- function(buf, data, col_offset) {
  .ag_buffer_reap()
  core <- .ag_buffer_core(buf)
  if (is.vector(data) && !is.list(data)) data <- matrix(data, ncol = 1L)
  if (!is.matrix(data) || !is.numeric(data) && !is.logical(data))
    stop("ag_buffer_write: data must be a numeric matrix", call. = FALSE)
  if (nrow(data) != core$rows)
    stop(sprintf("ag_buffer_write: data has %d rows, the buffer has %d",
                 nrow(data), core$rows), call. = FALSE)
  k <- ncol(data)
  col_offset <- .ag_buffer_offset(col_offset, k, core$capacity, "ag_buffer_write")
  if (k == 0L) return(invisible(buf))

  if (.ag_buffer_gpu()) {
    .ag_buffer_on_device(core)
    # A queued node reading the buffer must see the value it was queued
    # against, so the queue goes first; the write itself is blocking.
    if (.ag_defer_len()) .ag_defer_drain()
    vals <- as.numeric(data)
    if (.ag_xfer$enabled) .ag_xfer_record("up", "ag_buffer_write", length(vals))
    ggml_backend_tensor_set_data(core$ptr, vals,
                                 offset = as.double(col_offset) * core$rows * 4)
  } else {
    .ag_buffer_on_host(core)
    core$data[, col_offset + seq_len(k)] <- data
  }
  invisible(buf)
}

#' Copy a device tensor into buffer columns
#'
#' Writes a ggml tensor that already lives on the GPU -- for example the
#' observations of an environment computed in its own backend -- into columns
#' of an \code{\link{ag_buffer}}, device to device: the data does not pass
#' through the host or R.
#'
#' The contract is strict and nothing is converted:
#' \itemize{
#'   \item \code{tensor} is f32 and contiguous, with one buffer column per
#'     item along its trailing axes: 2-d \code{[rows, k]} in ggml order, or
#'     N-d whose leading axes make up a column -- the buffer's row count is
#'     \code{ne0}, \code{ne0 * ne1} or \code{ne0 * ne1 * ne2}, and the
#'     remaining axes count the columns (a state \code{[S, 6, n]} fills
#'     \code{n} columns of a buffer with \code{6 S} rows). A tensor laid out
#'     \code{[k, rows]} must be transposed by its producer; the buffer does
#'     not.
#'   \item \code{tensor} is on the same device as the buffer, and the ag
#'     device is the GPU.
#' }
#' Otherwise the copy would have to go through the host: with ag on the CPU
#' (the buffer lives on the host), with \code{tensor} in host memory, or with
#' \code{tensor} on another device. By default (\code{via_host = "error"})
#' each of these is an error naming the reason, not a silent slow copy.
#' \code{via_host = "allow"} takes the host path instead -- the tensor is
#' downloaded and written with \code{ag_buffer_write()}: same values, same
#' synchronous semantics. A message names the reason once per session for
#' each reason (silence it with \code{suppressMessages()}).
#'
#' The write is synchronous, like \code{ag_buffer_write()}: queued ag nodes
#' are computed first (they read the old value), and when the call returns
#' the columns hold the new data. The buffer is not reallocated, so
#' recordings that read it stay valid.
#'
#' @param buf An \code{\link{ag_buffer}}.
#' @param tensor A ggml tensor pointer, normally in device memory.
#' @param backend The backend that owns \code{tensor}'s memory (the
#'   environment's backend); its queued work is finished before the copy.
#' @param col_offset 0-based index of the first column written.
#' @param via_host \code{"error"} (default) or \code{"allow"}: whether a copy
#'   that has to go through the host is refused or made.
#' @return \code{buf}, invisibly.
#' @seealso \code{\link{ag_buffer_write}}, \code{\link{ag_get_rows}}
#' @export
ag_buffer_copy_from <- function(buf, tensor, backend, col_offset = 0,
                                via_host = c("error", "allow")) {
  via_host <- match.arg(via_host)
  .ag_buffer_reap()
  core <- .ag_buffer_core(buf)
  if (!is.numeric(col_offset) || length(col_offset) != 1L || is.na(col_offset))
    stop("ag_buffer_copy_from: col_offset must be a single number", call. = FALSE)
  if (!.ag_buffer_gpu())
    return(.ag_buffer_copy_via_host(buf, tensor, backend, col_offset, via_host,
                                    "cpu", "the ag device is the CPU"))
  .ag_buffer_on_device(core)
  # a queued node reading the buffer must see the value it was queued against
  if (.ag_defer_len()) .ag_defer_drain()
  reason <- .Call("R_ag_buffer_copy_cols", backend, .ag_device_state$backend,
                  tensor, core$ptr, as.double(col_offset), PACKAGE = "ggmlR")
  if (!is.null(reason)) {
    kind <- if (startsWith(reason, "the tensor is in host")) "host" else "device"
    return(.ag_buffer_copy_via_host(buf, tensor, backend, col_offset, via_host,
                                    kind, reason))
  }
  invisible(buf)
}

# The host path: refused with the reason unless via_host = "allow". The same
# contract as the device path is checked here, then the backend is
# synchronized so the tensor holds its final value, and the data goes through
# ag_buffer_write().
.ag_buffer_copy_via_host <- function(buf, tensor, backend, col_offset, via_host,
                                     kind, reason) {
  if (identical(via_host, "error"))
    stop(sprintf(paste0("ag_buffer_copy_from: %s, so the copy would go through ",
                        "the host. Pass via_host = \"allow\" to make it anyway."),
                 reason), call. = FALSE)
  if (!identical(ggml_tensor_type(tensor), GGML_TYPE_F32))
    stop("ag_buffer_copy_from: tensor must be f32", call. = FALSE)
  if (!ggml_is_contiguous(tensor))
    stop("ag_buffer_copy_from: tensor must be contiguous", call. = FALSE)
  ne <- as.numeric(ggml_tensor_shape(tensor))
  ne <- c(ne, rep(1, 4L - length(ne)))
  # the leading axes make up one column: rows = ne0, ne0 * ne1, ...
  rows <- buf$rows
  d <- match(TRUE, cumprod(ne) == rows)
  if (is.na(d))
    stop(sprintf(paste0("ag_buffer_copy_from: tensor [%s] does not split into columns of ",
                        "the buffer's %d rows (the rows must be the product of its ",
                        "leading axes)"),
                 paste(ne, collapse = ", "), as.integer(rows)), call. = FALSE)
  k <- prod(ne[-seq_len(d)])
  if (col_offset < 0 || col_offset != round(col_offset) || col_offset + k > buf$capacity)
    stop(sprintf("ag_buffer_copy_from: columns %g..%g are outside the buffer (capacity %d)",
                 col_offset, col_offset + k - 1, as.integer(buf$capacity)), call. = FALSE)
  noted <- .ag_buffer_reg$via_host_noted %||% character()
  if (!kind %in% noted) {
    .ag_buffer_reg$via_host_noted <- c(noted, kind)
    message(sprintf(paste0("ag_buffer_copy_from: %s; copying through the host ",
                           "(via_host = \"allow\"). Shown once per session."), reason))
  }
  if (k == 0) return(invisible(buf))
  ggml_backend_synchronize(backend)
  vals <- ggml_backend_tensor_get_data(tensor, n_elements = rows * k)
  ag_buffer_write(buf, matrix(vals, rows, k), col_offset)
}

#' @rdname ag_buffer
#' @param k Number of columns to read.
#' @details \code{ag_buffer_read()} returns columns \code{col_offset} to
#'   \code{col_offset + k - 1} as a \code{rows x k} matrix; reading a large
#'   buffer in parts bounds the host memory used.
#' @export
ag_buffer_read <- function(buf, col_offset = 0, k = buf$capacity - col_offset) {
  .ag_buffer_reap()
  core <- .ag_buffer_core(buf)
  k <- .ag_buffer_count(k, "k", zero = TRUE)
  col_offset <- .ag_buffer_offset(col_offset, k, core$capacity, "ag_buffer_read")
  if (k == 0L) return(matrix(0, core$rows, 0L))
  if (!is.null(core$ptr) && .ag_buffer_gpu()) {
    if (.ag_defer_len()) .ag_defer_drain()
    .ag_buffer_download(core, col_offset, k)
  } else {
    .ag_buffer_on_host(core)
    core$data[, col_offset + seq_len(k), drop = FALSE]
  }
}

#' @rdname ag_buffer
#' @details \code{ag_buffer_free()} releases the memory at once instead of at
#'   garbage collection; the buffer cannot be used afterwards.
#' @export
ag_buffer_free <- function(buf) {
  stopifnot(inherits(buf, "ag_buffer"))
  # a queued ag_get_rows() node still reads this memory
  if (!is.null(buf$core$ptr) && .ag_defer_len()) .ag_defer_drain()
  .ag_buffer_release(buf$core)
  .ag_buffer_reap()
  invisible(NULL)
}

#' Read buffer columns by index
#'
#' Selects columns of an \code{\link{ag_buffer}} by 0-based index, as a device
#' operation: on the GPU only the indices cross the bus. The name follows
#' \code{ggml_get_rows}, which selects along the second ggml dimension (ne1);
#' in the R \code{rows x capacity} view that is a column.
#'
#' The result is a \code{rows x length(idx)} ag_tensor without gradient, usable
#' as an input to further ag_* operations. Inside \code{\link{ag_capture}()}
#' pass \code{idx} as an argument of the captured function: the recording is
#' then reused for every index vector of the same length (the batch size is
#' part of the recording's shape), and the buffer must be listed in
#' \code{params} so that freeing it invalidates the recording.
#'
#' @param buf An \code{\link{ag_buffer}}.
#' @param idx 0-based column indices: a numeric vector, or an ag_tensor with
#'   one column (f32; indices are exact below \eqn{2^{24}}).
#' @return An ag_tensor of shape \code{rows x length(idx)}.
#' @details Indices given as host values are checked against the buffer and an
#'   index out of range is an error. An index tensor already on the device
#'   (inside a recording) cannot be checked; on the device every index is
#'   clamped to \code{0..capacity-1} before the read, so an index out of range
#'   reads the first or last column instead of memory outside the buffer.
#' @export
ag_get_rows <- function(buf, idx) {
  .ag_buffer_reap()
  core <- .ag_buffer_core(buf)
  host_idx <- if (is_ag_tensor(idx)) {
    if (!is.null(.ag_handle_of(idx))) NULL else .ag_data(idx)
  } else idx
  if (is_ag_tensor(idx) && !identical(.ag_compute_dtype(idx$dtype %||% "f32"), "f32"))
    stop("ag_get_rows: idx must be an f32 tensor", call. = FALSE)
  if (!is.null(host_idx)) {
    host_idx <- as.numeric(host_idx)
    if (anyNA(host_idx) || any(host_idx != round(host_idx)) ||
        any(host_idx < 0) || any(host_idx > core$capacity - 1))
      stop(sprintf("ag_get_rows: idx must be whole numbers in 0..%d",
                   core$capacity - 1L), call. = FALSE)
    if (any(host_idx >= 2^24))
      stop("ag_get_rows: indices from 2^24 on are not exact in f32", call. = FALSE)
  }
  b <- if (is_ag_tensor(idx)) prod(.ag_dim(.ag_handle_of(idx) %||% .ag_data(idx)))
       else length(idx)
  if (b < 1L) stop("ag_get_rows: idx is empty", call. = FALSE)

  if (!.ag_buffer_gpu()) {
    .ag_buffer_on_host(core)
    out <- ag_tensor(core$data[, host_idx + 1, drop = FALSE], device = "cpu")
    out$requires_grad <- FALSE
    return(out)
  }

  .ag_buffer_on_device(core)
  h <- .ag_handle(core$ptr, c(core$rows, core$capacity), scope = "persistent")
  idx_op <- if (is_ag_tensor(idx)) .ag_operand(idx) else matrix(host_idx, ncol = 1L)
  res <- .ag_run_op(
    op_fn = function(ctx, ptrs) {
      # clamped first: an index out of range would read outside the buffer,
      # which on Vulkan can lose the device rather than return garbage
      safe <- ggml_clamp(ctx, ptrs[[2L]], 0, core$capacity - 1)
      i32 <- ggml_reshape_1d(ctx, ggml_cast(ctx, safe, GGML_TYPE_I32), b)
      ggml_get_rows(ctx, ptrs[[1L]], i32)
    },
    inputs    = list(h, idx_op),
    out_shape = c(core$rows, b),
    dtype     = "f32",
    resident  = TRUE
  )
  out <- .ag_wrap_result(res, "gpu", dtype = "f32")
  out$requires_grad <- FALSE
  # the node reads the buffer's memory: while the result lives, so does the
  # buffer, and the collector cannot free it before the node is computed
  out$ag_buffer_ref <- buf
  out
}

#' Write buffer columns by index
#'
#' Writes the columns of \code{x} into columns \code{idx} of an
#' \code{\link{ag_buffer}}. Inside \code{\link{ag_capture}()} the write is a
#' node of the recorded graph (\code{ggml_set_rows}): every replay writes on
#' the device, with nothing crossing the bus, and the state it leaves -- a
#' counter, a board, a rollout column -- is what the next replay reads. A
#' recording whose only effect is such writes may return \code{NULL}: its
#' replays then download nothing.
#'
#' @param buf An \code{\link{ag_buffer}}; inside a recording it must be
#'   listed in \code{params}.
#' @param idx 0-based column indices: a numeric vector, or an ag_tensor with
#'   one column (f32). Inside a recording an ag_tensor (an input or a value
#'   computed on the device).
#' @param x An ag_tensor or numeric matrix of shape \code{rows x length(idx)}.
#' @return \code{buf}, invisibly.
#' @details Outside a recording the write goes through the host
#'   (\code{\link{ag_buffer_write}}), after any queued device work.
#'
#'   Inside a recording the writes run after the outputs, in the order they
#'   were made. A recording may read columns it also writes only when the
#'   written value depends on the read one (\code{t <- t + 1}, a board
#'   updated from itself); otherwise the order of the read and the write is
#'   not defined. Indices are clamped to \code{0..capacity-1} on the device,
#'   as in \code{\link{ag_get_rows}}; with repeated indices which write wins is
#'   not defined (outside a recording repeated indices are an error).
#' @export
ag_set_rows <- function(buf, idx, x) {
  .ag_buffer_reap()
  core <- .ag_buffer_core(buf)
  if (isTRUE(.ag_capture_reg$recording)) return(.ag_set_rows_node(buf, core, idx, x))

  # outside a recording: host values, then ag_buffer_write
  i <- if (is_ag_tensor(idx)) as.numeric(as.matrix(idx)) else as.numeric(idx)
  if (!length(i)) stop("ag_set_rows: idx is empty", call. = FALSE)
  if (anyNA(i) || any(i != round(i)) || any(i < 0) || any(i > core$capacity - 1))
    stop(sprintf("ag_set_rows: idx must be whole numbers in 0..%d", core$capacity - 1L),
         call. = FALSE)
  if (anyDuplicated(i)) stop("ag_set_rows: idx has repeated indices", call. = FALSE)
  v <- if (is_ag_tensor(x)) as.matrix(x) else x
  if (is.vector(v) && !is.list(v)) v <- matrix(v, ncol = 1L)
  if (!is.matrix(v) || !(is.numeric(v) || is.logical(v)))
    stop("ag_set_rows: x must be an ag_tensor or a numeric matrix", call. = FALSE)
  if (nrow(v) != core$rows || ncol(v) != length(i))
    stop(sprintf("ag_set_rows: x is %d x %d, expected %d x %d (rows x length(idx))",
                 nrow(v), ncol(v), core$rows, length(i)), call. = FALSE)
  # one write per run of consecutive indices
  run <- cumsum(c(1L, diff(i) != 1))
  for (r in unique(run)) {
    k <- which(run == r)
    ag_buffer_write(buf, v[, k, drop = FALSE], col_offset = i[k[1L]])
  }
  invisible(buf)
}

# The write as a node of the recording, kept in .ag_capture_reg$effects: the
# build adds it to the graph after the outputs.
.ag_set_rows_node <- function(buf, core, idx, x) {
  if (!is_ag_tensor(idx))
    stop("ag_set_rows: inside ag_capture idx must be an ag_tensor (an input of the ",
         "captured function or a value computed from one)", call. = FALSE)
  if (!identical(.ag_compute_dtype(idx$dtype %||% "f32"), "f32"))
    stop("ag_set_rows: idx must be an f32 tensor", call. = FALSE)
  if (!is_ag_tensor(x))
    stop("ag_set_rows: inside ag_capture x must be an ag_tensor", call. = FALSE)
  b <- prod(.ag_dim(.ag_handle_of(idx) %||% .ag_data(idx)))
  dx <- .ag_dim(.ag_handle_of(x) %||% .ag_data(x))
  if (dx[1L] != core$rows || dx[2L] != b)
    stop(sprintf("ag_set_rows: x is %d x %d, expected %d x %d (rows x length(idx))",
                 dx[1L], dx[2L], core$rows, b), call. = FALSE)
  .ag_buffer_on_device(core)
  h <- .ag_handle(core$ptr, c(core$rows, core$capacity), scope = "persistent")
  res <- .ag_run_op(
    op_fn = function(ctx, ptrs) {
      safe <- ggml_clamp(ctx, ptrs[[2L]], 0, core$capacity - 1)
      i32 <- ggml_reshape_1d(ctx, ggml_cast(ctx, safe, GGML_TYPE_I32), b)
      v <- .ag_f32(ctx, ptrs[[3L]])
      if (!ggml_is_contiguous(v)) v <- ggml_cont(ctx, v)
      ggml_set_rows(ctx, ptrs[[1L]], v, i32)
    },
    inputs    = list(h, .ag_operand(idx), .ag_operand(x)),
    out_shape = c(core$rows, core$capacity),
    dtype     = "f32",
    resident  = TRUE
  )
  .ag_capture_reg$effects[[length(.ag_capture_reg$effects) + 1L]] <- res
  invisible(buf)
}

#' @export
dim.ag_buffer <- function(x) c(x$rows, x$capacity)

#' @export
print.ag_buffer <- function(x, ...) {
  core <- x$core
  where <- if (isTRUE(core$freed)) "freed" else if (!is.null(core$ptr)) "device" else "host"
  cat("<ag_buffer> ", x$rows, " x ", x$capacity, " ", x$dtype, " (", where, ")\n", sep = "")
  invisible(x)
}

# ---------------------------------------------------------------------------

.ag_buffer_count <- function(x, what, zero = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || x != round(x) ||
      x < (if (zero) 0 else 1))
    stop("ag_buffer: `", what, "` must be a ", if (zero) "non-negative" else "positive",
         " whole number", call. = FALSE)
  as.integer(x)
}

.ag_buffer_offset <- function(col_offset, k, capacity, fn) {
  if (!is.numeric(col_offset) || length(col_offset) != 1L || is.na(col_offset) ||
      col_offset != round(col_offset) || col_offset < 0)
    stop(fn, ": col_offset must be a non-negative whole number", call. = FALSE)
  if (col_offset + k > capacity)
    stop(sprintf("%s: columns %d..%d are outside the buffer (capacity %d)",
                 fn, as.integer(col_offset), as.integer(col_offset + k - 1), capacity),
         call. = FALSE)
  as.integer(col_offset)
}

.ag_buffer_core <- function(buf) {
  if (!inherits(buf, "ag_buffer")) stop("ggmlR: not an ag_buffer", call. = FALSE)
  if (isTRUE(buf$core$freed)) stop("ggmlR: the ag_buffer has been freed", call. = FALSE)
  buf$core
}

# TRUE when buffer operations run on the device.
.ag_buffer_gpu <- function() identical(.ag_device_state$device, "gpu")

# Allocate the device tensor and fill it from the host value (zeros if none).
.ag_buffer_alloc <- function(core) {
  if (is.null(.ag_device_state$backend)) .ag_init_gpu_backend()
  backend <- .ag_device_state$backend
  bytes <- as.double(core$rows) * core$capacity * 4
  free <- tryCatch(ggml_backend_dev_memory(ggml_backend_get_device(backend))[["free"]],
                   error = function(e) NA_real_)
  # Some Vulkan drivers report free memory only approximately: keep a margin.
  usable <- .ag_buffer_mem_share * free
  if (is.finite(free) && bytes > usable)
    stop(sprintf(paste0("ag_buffer: %.1f MB requested, the device reports %.1f MB ",
                        "free, of which %.1f MB (%d%%) may be used. Reduce the capacity."),
                 bytes / 1024^2, free / 1024^2, usable / 1024^2,
                 as.integer(100 * .ag_buffer_mem_share)),
         call. = FALSE)
  ctx <- ggml_init(4 * ggml_tensor_overhead() + 1024, no_alloc = TRUE)
  ptr <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, core$rows, core$capacity)
  bbuf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  if (is.null(bbuf)) {
    ggml_free(ctx)
    stop(sprintf("ag_buffer: the device could not allocate %.1f MB", bytes / 1024^2),
         call. = FALSE)
  }
  core$ctx <- ctx; core$bbuf <- bbuf; core$ptr <- ptr
  # a new allocation, even at a reused address, must re-record captures
  core$gen <- core$gen + 1L
  .ag_buffer_reg$items[[core$id]] <- core

  step <- max(1L, .ag_buffer_chunk_elems %/% core$rows)
  for (start in seq(0L, core$capacity - 1L, by = step)) {
    k <- min(step, core$capacity - start)
    vals <- if (is.null(core$data)) numeric(core$rows * k)
            else as.numeric(core$data[, start + seq_len(k)])
    ggml_backend_tensor_set_data(ptr, vals, offset = as.double(start) * core$rows * 4)
  }
  core$data <- NULL
  invisible(core)
}

.ag_buffer_download <- function(core, col_offset, k) {
  n <- as.double(core$rows) * k
  if (.ag_xfer$enabled) .ag_xfer_record("down", "ag_buffer_read", n)
  vals <- ggml_backend_tensor_get_data(core$ptr, offset = as.double(col_offset) * core$rows * 4,
                                       n_elements = n)
  matrix(vals, core$rows, k)
}

.ag_buffer_on_device <- function(core) {
  if (is.null(core$ptr)) .ag_buffer_alloc(core)
  invisible(core)
}

# Bring the value to the host and release the device memory.
.ag_buffer_on_host <- function(core) {
  if (is.null(core$ptr)) return(invisible(core))
  data <- matrix(0, core$rows, core$capacity)
  step <- max(1L, .ag_buffer_chunk_elems %/% core$rows)
  for (start in seq(0L, core$capacity - 1L, by = step)) {
    k <- min(step, core$capacity - start)
    data[, start + seq_len(k)] <- .ag_buffer_download(core, start, k)
  }
  .ag_buffer_free_device(core)
  core$data <- data
  invisible(core)
}

.ag_buffer_free_device <- function(core) {
  if (!is.null(core$bbuf)) tryCatch(ggml_backend_buffer_free(core$bbuf), error = function(e) NULL)
  if (!is.null(core$ctx)) tryCatch(ggml_free(core$ctx), error = function(e) NULL)
  core$bbuf <- core$ctx <- core$ptr <- NULL
  if (exists(core$id, envir = .ag_buffer_reg$items, inherits = FALSE))
    rm(list = core$id, envir = .ag_buffer_reg$items)
  invisible(core)
}

.ag_buffer_release <- function(core) {
  if (isTRUE(core$freed)) return(invisible(NULL))
  .ag_buffer_free_device(core)
  core$data <- NULL
  core$freed <- TRUE
  invisible(NULL)
}

# Called by .ag_residency_reset() before the persistent pool and the backend
# are freed: every device-resident buffer moves its value to the host.
.ag_buffer_rescue_all <- function() {
  # collected buffers are not rescued: the queue is empty by now, free them
  .ag_buffer_reap()
  for (id in ls(.ag_buffer_reg$items, all.names = TRUE)) {
    core <- .ag_buffer_reg$items[[id]]
    # Not swallowed: the backend is freed next, and the value with it.
    tryCatch(.ag_buffer_on_host(core), error = function(e)
      stop("ag_buffer: could not move a buffer to the host before the device ",
           "was released, so the switch is aborted: ", conditionMessage(e), call. = FALSE))
  }
  invisible(NULL)
}

# Finalizer. A node still queued may read the buffer (a result of ag_get_rows()
# keeps the buffer alive, so this is the fallback for a node whose result was
# dropped too): then the memory waits in `orphans` until the queue has run.
.ag_buffer_collect <- function(core) {
  if (!is.null(core$ptr) && .ag_defer_len()) {
    .ag_buffer_reg$orphans[[core$id]] <- core
    return(invisible(NULL))
  }
  .ag_buffer_release(core)
}

# Free the collected buffers once no queued node can read them. Called by every
# ag_buffer_* function, after each queue drain and before a device switch, so
# the list does not grow.
.ag_buffer_reap <- function() {
  ids <- ls(.ag_buffer_reg$orphans, all.names = TRUE)
  if (!length(ids) || .ag_defer_len()) return(invisible(NULL))
  for (id in ids) {
    .ag_buffer_release(.ag_buffer_reg$orphans[[id]])
    rm(list = id, envir = .ag_buffer_reg$orphans)
  }
  invisible(NULL)
}
