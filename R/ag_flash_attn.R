# Flash attention for the ag_* autograd engine.
#
# WHY
# ---
# ag_multihead_attention() loops over heads. ag_* has no slice op, so pulling
# one head out of a [d_model, seq] matrix is a matmul against a selector matrix;
# each head then costs two more matmuls, two transposes and a softmax. A 4-head
# block is ~52 tape nodes for the attention core alone, every one an R-level
# dispatch and -- on the GPU -- its own upload/compute/download.
#
# ggml_flash_attn_ext does all heads in one op, and the ggmlR extension
# ggml_flash_attn_back does their gradients in one more. Measured on the CPU
# backend (inst/scripts/measure_ag_flash_attn.R), attention core only, forward
# and backward:
#
#   d32  h2 seq16    26 nodes   1.47 ms  ->  0.04 ms   36.5x
#   d64  h4 seq32    52 nodes   3.52 ms  ->  0.15 ms   23.2x
#   d128 h8 seq64   104 nodes  10.72 ms  ->  0.70 ms   15.4x
#   d256 h8 seq128  104 nodes  41.69 ms  ->  2.16 ms   19.3x
#
# Those ratios are a ceiling -- the flash side there pays no tape bookkeeping --
# but the margin is large enough that the wrapper below keeps a real win even
# after paying for a node, its snapshots and unpacking the gradients.
#
# LAYOUTS, WHICH ARE THE WHOLE DIFFICULTY
# ---------------------------------------
# ag_* is two-dimensional throughout: .ag_run_op builds only 2D tensors,
# .ag_data() returns a matrix. Flash attention is not -- ggml.h:2424 gives
#
#   q    [n_embd_k, n_batch, n_head]      k,v  [n_embd, n_kv, n_head_kv]
#   res  [n_embd_v, n_head,  n_batch]     <- head and sequence SWAPPED
#
# so this file keeps its own 3D path instead of going through .ag_run_op, and
# converts at the boundary: R sees [d_model, seq] matrices, the op sees 3D
# tensors. The permutation of the result is real and was verified against an
# independent R implementation (maxdiff 4.7e-08), not assumed from the header.
#
# ggml_flash_attn_back returns grad_q, grad_k and grad_v PACKED into one
# contiguous buffer, each slice aligned to 16 bytes. The offsets below match
# tests/testthat/test-flash-attn-back.R and were checked against finite
# differences (dq 2.6e-08, dk 2.1e-08, dv 4.7e-08). Getting the padding wrong
# silently shifts dk and dv, which is why it is computed rather than assumed.

# Byte-aligned element offset of the next packed slice: ggml pads each tensor
# in the packed buffer to a 16-byte boundary, so a slice of n floats occupies
# ceiling(n * 4 / 16) * 4 elements.
.ag_flash_pad <- function(n) ceiling(n * 4 / 16) * 16 / 4

# [d_model, seq] matrix -> [d_head, seq, n_head] array.
#
# Row block h of the matrix is head h, matching how ag_multihead_attention
# slices: rows (h-1)*d_head + 1 ... h*d_head.
.ag_flash_split_heads <- function(m, n_heads) {
  d_model <- nrow(m); seq_len_ <- ncol(m)
  d_head  <- d_model %/% n_heads
  # A [d_model, seq] matrix is column-major: element (r, c) lives at
  # r + (c-1)*d_model, so simply reading the same memory as [d_head, n_head,
  # seq] already groups consecutive rows into heads -- no data moves. Flash
  # wants [d_head, seq, n_head], so only the permute costs anything.
  aperm(array(m, dim = c(d_head, n_heads, seq_len_)), c(1L, 3L, 2L))
}

# [d_head, seq, n_head] array -> [d_model, seq] matrix. Inverse of the above.
.ag_flash_join_heads <- function(a) {
  d_head <- dim(a)[1L]; seq_len_ <- dim(a)[2L]; n_heads <- dim(a)[3L]
  matrix(aperm(a, c(1L, 3L, 2L)), d_head * n_heads, seq_len_)
}


# Build the [n_kv, n_q] mask ggml wants, from either an explicit matrix or the
# `causal` shorthand.
#
# NOTE THE ORIENTATION. ggml indexes the mask [n_kv, n_batch] -- keys down the
# rows, queries across the columns -- which is the transpose of the way an
# attention mask is usually written on paper ("row i = query i"). Supplying a
# [n_q, n_kv] matrix is therefore not a shape error when the two happen to be
# square: it silently masks the wrong entries. So the accepted orientation is
# fixed and checked, and the causal mask below is built directly in ggml's
# orientation rather than transposed into it.
#
# Entries are 0 (attend) and -Inf (do not). F16 represents both exactly, so
# the conversion on upload loses nothing.
.ag_flash_mask <- function(mask, causal, seq_q, seq_kv) {
  if (isTRUE(causal)) {
    if (!is.null(mask))
      stop("ggmlR: ag_flash_attention() takes either `mask` or `causal`, ",
           "not both.", call. = FALSE)
    # Query j (column) may attend to keys 1..j only, so everything below the
    # diagonal of the [n_kv, n_q] matrix is blocked.
    m <- matrix(0, seq_kv, seq_q)
    for (j in seq_len(seq_q)) {
      if (j < seq_kv) m[(j + 1L):seq_kv, j] <- -Inf
    }
    return(m)
  }

  if (is.null(mask)) return(NULL)

  m <- if (is_ag_tensor(mask)) .ag_data(mask) else mask
  if (is.null(dim(m)) || length(dim(m)) != 2L)
    stop("ggmlR: ag_flash_attention() needs `mask` to be a matrix.",
         call. = FALSE)
  if (nrow(m) != seq_kv || ncol(m) != seq_q)
    stop("ggmlR: ag_flash_attention() needs `mask` to be [seq_kv, seq_q] = [",
         seq_kv, ", ", seq_q, "], got [", nrow(m), ", ", ncol(m), "]. Note the ",
         "orientation: keys index the ROWS, queries the columns.",
         call. = FALSE)

  # A logical mask is the friendlier spelling: TRUE where attention is allowed.
  if (is.logical(m)) {
    out <- matrix(0, nrow(m), ncol(m))
    out[!m] <- -Inf
    return(out)
  }
  m
}

# Run one flash attention forward, and optionally its backward, on the current
# ag device. Inputs and outputs are R arrays in flash layout; nothing here
# touches the tape.
#
# Deliberately NOT built on .ag_run_op: that helper is 2D-only, and teaching it
# 3D would change a path every other op depends on. This keeps the dimensional
# special case contained in one file.
.ag_flash_run <- function(q, k, v, scale, grad_out = NULL, mask = NULL) {
  # Respect the selected device. ag_device("cpu") only records the choice and
  # leaves $backend NULL, so falling through to .ag_init_gpu_backend() here
  # would quietly run on Vulkan after the user asked for the CPU -- which
  # showed up as a 4e-04 disagreement with an R reference that turned out to be
  # f16 accumulation on the GPU, not a bug in the layout.
  if (is.null(.ag_device_state$backend)) {
    if (identical(.ag_device_state$device, "cpu")) {
      # The thread count has to be applied to every CPU backend that is
      # created: without it ggml takes omp_get_max_threads(), which breaks the
      # CPU-time ratio R CMD check enforces.
      .ag_device_state$backend <- ggml_backend_cpu_init()
      ggml_backend_cpu_set_n_threads(.ag_device_state$backend,
                                     ggml_get_n_threads())
    } else {
      .ag_init_gpu_backend()
    }
  }
  backend   <- .ag_device_state$backend
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype())

  dk <- dim(q)[1L]; n_q <- dim(q)[2L]; n_head <- dim(q)[3L]
  dv <- dim(v)[1L]; n_kv <- dim(k)[2L]

  # Own context, freed here: these tensors are 3D and short-lived, so they have
  # no business in the shared residency context the 2D ops use.
  ctx <- ggml_init(.ag_flash_ctx_bytes(q, k, v), no_alloc = TRUE)
  if (is.null(ctx)) stop("ggmlR: failed to create a context for flash attention.")
  on.exit(ggml_free(ctx), add = TRUE)

  tq <- ggml_new_tensor_4d(ctx, ggml_type, dk, n_q,  n_head, 1L)
  tk <- ggml_new_tensor_4d(ctx, ggml_type, dk, n_kv, n_head, 1L)
  tv <- ggml_new_tensor_4d(ctx, ggml_type, dv, n_kv, n_head, 1L)

  # The mask is [n_kv, n_batch] -- keys down the rows, queries across the
  # columns -- and ggml requires it in F16 and contiguous
  # (ggml-ops-builders.c:3571). F16 is not a precision compromise here: the
  # entries are 0 and -Inf, both of which survive the conversion exactly
  # (verified, not assumed).
  tm <- NULL
  if (!is.null(mask)) {
    tm <- ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nrow(mask), ncol(mask))
  }

  fwd <- ggml_flash_attn_ext(ctx, tq, tk, tv, tm, scale, 0, 0)

  td <- NULL; bwd <- NULL
  if (!is.null(grad_out)) {
    # d carries the result's permuted layout: [n_embd_v, n_head, n_batch].
    td  <- ggml_new_tensor_4d(ctx, ggml_type, dv, n_head, n_q, 1L)
    bwd <- ggml_flash_attn_back(ctx, tq, tk, tv, tm, td, scale)
  }

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  if (is.null(buf)) stop("ggmlR: failed to allocate a buffer for flash attention.")
  on.exit(tryCatch(ggml_backend_buffer_free(buf), error = function(e) NULL),
          add = TRUE)

  ggml_backend_tensor_set_data(tq, as.numeric(q))
  ggml_backend_tensor_set_data(tk, as.numeric(k))
  ggml_backend_tensor_set_data(tv, as.numeric(v))
  if (!is.null(tm)) ggml_backend_tensor_set_data(tm, as.numeric(mask))
  if (!is.null(td)) ggml_backend_tensor_set_data(td, as.numeric(grad_out))

  graph <- ggml_build_forward_expand(ctx, fwd)
  if (!is.null(bwd)) ggml_graph_expand(graph, bwd)
  ggml_backend_graph_compute(backend, graph)

  # Result comes back permuted: [d_v, n_head, n_q].
  out <- array(ggml_backend_tensor_get_data(fwd), c(dv, n_head, n_q))

  grads <- NULL
  if (!is.null(bwd)) {
    packed <- ggml_backend_tensor_get_data(bwd)
    nq <- length(q); nk <- length(k); nv <- length(v)
    off_k <- .ag_flash_pad(nq)
    off_v <- off_k + .ag_flash_pad(nk)
    if (length(packed) < off_v + nv)
      stop("ggmlR: flash_attn_back returned a shorter buffer than its three ",
           "gradients need (", length(packed), " < ", off_v + nv, ").",
           call. = FALSE)
    grads <- list(
      q = array(packed[seq_len(nq)], dim(q)),
      k = array(packed[off_k + seq_len(nk)], dim(k)),
      v = array(packed[off_v + seq_len(nv)], dim(v)))
  }

  list(out = out, grads = grads)
}

# Context size for one flash call: the three inputs, the packed gradient buffer
# (about the same again), the result, and room for descriptors and the graph.
.ag_flash_ctx_bytes <- function(q, k, v) {
  elems <- (length(q) + length(k) + length(v)) * 3
  as.numeric(elems) * 4 + 64 * 1024 * 1024
}

#' Multi-head attention in a single fused operation
#'
#' Computes scaled dot-product attention over all heads at once with
#' \code{ggml_flash_attn_ext()}, instead of the per-head loop
#' \code{\link{ag_multihead_attention}} uses. Both the forward pass and its
#' gradient are one operation each, so the tape holds a single node rather than
#' the dozens a head loop records.
#'
#' \code{q}, \code{k} and \code{v} are \code{[d_model, seq]} matrices whose rows
#' are split into \code{n_heads} contiguous blocks -- the same head layout
#' \code{ag_multihead_attention()} uses. Projections are not included: apply
#' \code{W_q}, \code{W_k}, \code{W_v} before the call and \code{W_o} after it.
#'
#' @param q,k,v \code{ag_tensor}s of shape \code{[d_model, seq]}. \code{k} and
#'   \code{v} may have a different sequence length from \code{q}
#'   (cross-attention), but must match each other.
#' @param n_heads Number of attention heads. Must divide \code{d_model}.
#' @param scale Softmax scale. Defaults to \code{1/sqrt(d_model / n_heads)}.
#' @param mask Optional attention mask, \code{[seq_kv, seq_q]} -- keys index the
#'   ROWS and queries the columns, which is the transpose of the usual
#'   "row = query" convention and is not caught by shape checking when the two
#'   lengths match. Either logical (\code{TRUE} where attention is allowed) or
#'   numeric (\code{0} to attend, \code{-Inf} to block). The same mask is used
#'   for the gradient.
#' @param causal \code{TRUE} builds the causal mask for you: query \code{j}
#'   attends to keys \code{1..j}. Cannot be combined with \code{mask}.
#' @return An \code{ag_tensor} of shape \code{[d_model, seq_q]}.
#' @export
#' @examples
#' \donttest{
#' d_model <- 16L; seq_len <- 8L
#' q <- ag_param(matrix(runif(d_model * seq_len, -1, 1), d_model, seq_len))
#' k <- ag_param(matrix(runif(d_model * seq_len, -1, 1), d_model, seq_len))
#' v <- ag_param(matrix(runif(d_model * seq_len, -1, 1), d_model, seq_len))
#' with_grad_tape({
#'   out  <- ag_flash_attention(q, k, v, n_heads = 4L)
#'   loss <- ag_mse_loss(out, ag_tensor(matrix(0, d_model, seq_len)))
#' })
#' backward(loss)
#' }
ag_flash_attention <- function(q, k, v, n_heads, scale = NULL,
                               mask = NULL, causal = FALSE) {
  q_data <- .ag_data(q); k_data <- .ag_data(k); v_data <- .ag_data(v)
  n_heads <- as.integer(n_heads)

  d_model <- nrow(q_data)
  if (d_model %% n_heads != 0L)
    stop("ggmlR: ag_flash_attention() needs n_heads to divide d_model (",
         d_model, " %% ", n_heads, " != 0).", call. = FALSE)
  if (nrow(k_data) != d_model || nrow(v_data) != d_model)
    stop("ggmlR: ag_flash_attention() needs q, k and v to share d_model (got ",
         d_model, ", ", nrow(k_data), ", ", nrow(v_data), ").", call. = FALSE)
  if (ncol(k_data) != ncol(v_data))
    stop("ggmlR: ag_flash_attention() needs k and v to share a sequence length ",
         "(got ", ncol(k_data), " and ", ncol(v_data), ").", call. = FALSE)

  d_head <- d_model %/% n_heads
  if (is.null(scale)) scale <- 1 / sqrt(d_head)

  seq_q  <- ncol(q_data)
  seq_kv <- ncol(k_data)
  mask_m <- .ag_flash_mask(mask, causal, seq_q, seq_kv)

  qh <- .ag_flash_split_heads(q_data, n_heads)
  kh <- .ag_flash_split_heads(k_data, n_heads)
  vh <- .ag_flash_split_heads(v_data, n_heads)

  res <- .ag_flash_run(qh, kh, vh, scale, mask = mask_m)

  # Result arrives as [d_head, n_head, seq]; ungroup to [d_head, seq, n_head]
  # before joining the heads back into a [d_model, seq] matrix.
  out_mat <- .ag_flash_join_heads(aperm(res$out, c(1L, 3L, 2L)))

  out <- ag_tensor(out_mat, device = .ag_device_state$device,
                   dtype = .ag_device_state$dtype)
  out$requires_grad <- (is_ag_tensor(q) && q$requires_grad) ||
                       (is_ag_tensor(k) && k$requires_grad) ||
                       (is_ag_tensor(v) && v$requires_grad)

  if (out$requires_grad) {
    q_ref <- q; k_ref <- k; v_ref <- v
    grad_fn <- function(grad_out) {
      # grad_out is [d_model, seq_q] in R terms; flash wants the result's own
      # permuted layout [d_head, n_head, seq_q].
      gh <- aperm(.ag_flash_split_heads(grad_out, n_heads), c(1L, 3L, 2L))
      # Same mask as the forward pass: a gradient computed against a different
      # mask than the values it belongs to is wrong in a way nothing reports.
      g  <- .ag_flash_run(qh, kh, vh, scale, grad_out = gh, mask = mask_m)$grads
      list(
        q = if (is_ag_tensor(q_ref) && q_ref$requires_grad)
              .ag_flash_join_heads(g$q) else NULL,
        k = if (is_ag_tensor(k_ref) && k_ref$requires_grad)
              .ag_flash_join_heads(g$k) else NULL,
        v = if (is_ag_tensor(v_ref) && v_ref$requires_grad)
              .ag_flash_join_heads(g$v) else NULL)
    }
    out$grad_fn <- grad_fn
    # No `op`: the graph backward path cannot emit this, and a tape holding it
    # falls back to closures -- which is where the gradient is computed anyway.
    ag_record(out, grad_fn, list(q = q, k = k, v = v))
  }
  out
}
