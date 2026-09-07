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

  # ⚠️ THE BACKWARD IS F32-ONLY, AND THAT DECIDES THE DTYPE FOR THE WHOLE CALL.
  #
  # ggml_flash_attn_back asserts q, k, v and d are all GGML_TYPE_F32
  # (ggml-ops-builders.c:3701, "the kernel reads q/k/v/d as f32 directly").
  # ggml_flash_attn_ext has no such restriction, so under ag_dtype("f16") this
  # function used to build a perfectly good FORWARD and then abort R the moment
  # a gradient was asked for -- GGML_ABORT, not an R error, so nothing could
  # catch it and no fallback could run. The whole session died.
  #
  # Forcing F32 whenever a gradient is wanted costs precision nothing that
  # matters here (the backward accumulates in f32 anyway) and turns a crash into
  # ordinary arithmetic. The forward-only case keeps the requested dtype, which
  # is where f16 actually buys something.
  ggml_type <- if (is.null(grad_out)) .ag_dtype_to_ggml(.ag_compute_dtype())
               else                   GGML_TYPE_F32

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
  n_heads <- as.integer(n_heads)
  resident <- .ag_flash_resident_ok()

  # Shapes without materialising: on the resident path q/k/v may be handles, and
  # calling .ag_data() on one would download the very values this path exists to
  # keep on the device. .ag_operand() returns a handle when the value is already
  # there and a matrix otherwise, and .ag_nrow/.ag_ncol read either.
  if (resident) {
    qo <- .ag_operand(q); ko <- .ag_operand(k); vo <- .ag_operand(v)
  } else {
    qo <- .ag_data(q); ko <- .ag_data(k); vo <- .ag_data(v)
  }

  d_model <- .ag_nrow(qo)
  if (d_model %% n_heads != 0L)
    stop("ggmlR: ag_flash_attention() needs n_heads to divide d_model (",
         d_model, " %% ", n_heads, " != 0).", call. = FALSE)
  if (.ag_nrow(ko) != d_model || .ag_nrow(vo) != d_model)
    stop("ggmlR: ag_flash_attention() needs q, k and v to share d_model (got ",
         d_model, ", ", .ag_nrow(ko), ", ", .ag_nrow(vo), ").", call. = FALSE)
  if (.ag_ncol(ko) != .ag_ncol(vo))
    stop("ggmlR: ag_flash_attention() needs k and v to share a sequence length ",
         "(got ", .ag_ncol(ko), " and ", .ag_ncol(vo), ").", call. = FALSE)

  d_head <- d_model %/% n_heads
  if (is.null(scale)) scale <- 1 / sqrt(d_head)

  seq_q  <- .ag_ncol(qo)
  seq_kv <- .ag_ncol(ko)
  mask_m <- .ag_flash_mask(mask, causal, seq_q, seq_kv)

  needs_grad <- (is_ag_tensor(q) && q$requires_grad) ||
                (is_ag_tensor(k) && k$requires_grad) ||
                (is_ag_tensor(v) && v$requires_grad)

  if (resident)
    return(.ag_flash_attention_resident(q, k, v, qo, ko, vo, n_heads, scale,
                                        mask_m, d_model, seq_q, seq_kv,
                                        needs_grad))

  # ------------------------------------------------------------------ CPU path
  # Unchanged: its own context, its own compute, values through R. The graph
  # backward refuses a non-gpu tape anyway (.ag_bwd_reject_reason), so there is
  # nothing here for an `op` to buy.
  q_data <- qo; k_data <- ko; v_data <- vo

  qh <- .ag_flash_split_heads(q_data, n_heads)
  kh <- .ag_flash_split_heads(k_data, n_heads)
  vh <- .ag_flash_split_heads(v_data, n_heads)

  res <- .ag_flash_run(qh, kh, vh, scale, mask = mask_m)

  # Result arrives as [d_head, n_head, seq]; ungroup to [d_head, seq, n_head]
  # before joining the heads back into a [d_model, seq] matrix.
  out_mat <- .ag_flash_join_heads(aperm(res$out, c(1L, 3L, 2L)))

  out <- ag_tensor(out_mat, device = .ag_device_state$device,
                   dtype = .ag_device_state$dtype)
  out$requires_grad <- needs_grad

  if (out$requires_grad) {
    q_ref <- q; k_ref <- k; v_ref <- v
    grad_fn <- function(grad_out) {
      # grad_out is [d_model, seq_q] in R terms; flash wants the result's own
      # permuted layout [d_head, n_head, seq_q].
      gh <- aperm(.ag_flash_split_heads(.ag_as_matrix(grad_out), n_heads),
                  c(1L, 3L, 2L))
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
    # No `op` on this path: the tape is not on a device, so the graph backward
    # would refuse it regardless of what is recorded here.
    ag_record(out, grad_fn, list(q = q, k = k, v = v))
  }
  out
}


# ===========================================================================
# Resident path: attention as one opaque node on the device.
#
# WHY A SEPARATE PATH, AND WHY IT STOPS AT THIS FILE'S EDGE
# --------------------------------------------------------
# The tape's graph backward (R/ag_backward_graph.R) emits every op as nodes in
# the shared pass context, and refuses a tape containing anything it cannot
# emit -- all-or-nothing, because splitting one backward between a graph and
# closures would put the per-op round trip back. Attention recorded without an
# `op` was therefore rejecting ENTIRE tapes: a model with one attention block
# lost the graph backward and the fused forward+backward for its matmuls too.
#
# So attention needs an `op`. What it does NOT need is for the rest of the
# engine to understand its insides. The industry shape is a fused kernel that
# the surrounding graph sees as a single node with a known forward/backward
# contract -- PyTorch dispatches scaled_dot_product_attention to one backend
# and the autograd graph records one call, not a head permutation followed by
# a matmul. ggml already supplies both halves of that kernel
# (ggml_flash_attn_ext / ggml_flash_attn_back), so the fused op exists; what
# was missing was a device-resident boundary around it.
#
# Hence the split of responsibility:
#
#   here            everything about layout -- the head split, the permutation
#                   of the result, the packed gradient buffer and its 16-byte
#                   alignment. All of it expressed as ggml nodes so nothing
#                   travels through R, but all of it CONTAINED.
#
#   backward graph  one delegating branch that hands a gradient node in and
#                   takes three out. It knows the op's name and nothing about
#                   d_head, head order or slice offsets.
#
# That boundary is the point. The header of this file records that the layout
# is "the whole difficulty" and that it was verified against an independent
# implementation rather than read off ggml.h; knowledge with that history
# belongs in one file, not spread across two. If ggml changes the result's
# permutation, this file breaks and the backward graph does not.
#
# WHY THE NODES OUTLIVE THE CALL. .ag_flash_run() owns its context and frees it
# on exit, so its tensors are gone by the time backward() runs -- which is why
# it has to download. The functions below allocate from the shared pass pool
# instead: .ag_residency_reset(scope = "pass") runs at the START of
# with_grad_tape(), so a forward's tensors are still live while the backward
# for that same tape is built.
# ===========================================================================

# Is the resident path available? The closure path stays for the CPU backend
# and for a tape that is not running on a device at all.
#
# ⚠️ F32 ONLY, and this is a hard gate rather than a preference.
# ggml_flash_attn_back asserts q, k, v and d are all GGML_TYPE_F32
# (src/ggml-ops-builders.c: "the kernel reads q/k/v/d as f32 directly").
# ggml_flash_attn_ext has no such restriction, so an f16 tape would build a
# perfectly good resident FORWARD and then abort R inside a GGML_ASSERT the
# moment its backward was emitted -- an abort, not an R error, so there would be
# nothing to catch and fall back from. Refusing here sends f16/bf16 down the
# closure path, which computes the same gradients.
.ag_flash_resident_ok <- function() {
  identical(.ag_device_state$device, "gpu") &&
    !is.null(.ag_device_state$backend) &&
    identical(.ag_compute_dtype(), "f32")
}

# [d_model, seq] tensor -> flash layout [d_head, seq, n_head].
#
# The reshape is free: a [d_model, seq] column-major matrix already stores head
# h in rows (h-1)*d_head+1 ... h*d_head, so reading the same bytes as
# [d_head, n_head, seq] groups them without moving anything -- the R path says
# the same thing with array(m, dim = c(d_head, n_heads, seq)).
#
# Only the permute costs, and it costs a copy: ggml_permute relabels ne/nb and
# yields a view, which flash_attn_ext will not take. Per the permute contract in
# CLAUDE.md the arguments are DESTINATION positions, so sending source axis 1
# (n_head) to position 2 and source axis 2 (seq) to position 1 is (0, 2, 1, 3).
.ag_flash_to_heads <- function(ctx, t2d, d_head, n_heads, seq_len_) {
  t3 <- ggml_reshape_3d(ctx, t2d, d_head, n_heads, seq_len_)
  ggml_cont(ctx, ggml_permute(ctx, t3, 0L, 2L, 1L, 3L))
}

# Flash layout [d_head, seq, n_head] -> [d_model, seq] tensor. Inverse of the
# above: send source axis 1 (seq) to position 2 and axis 2 (n_head) to
# position 1, then read the [d_head, n_head, seq] result as [d_model, seq].
.ag_flash_from_heads <- function(ctx, t3d, d_head, n_heads, seq_len_) {
  c3 <- ggml_cont(ctx, ggml_permute(ctx, t3d, 0L, 2L, 1L, 3L))
  ggml_reshape_2d(ctx, c3, d_head * n_heads, seq_len_)
}

# The forward result arrives as [d_v, n_head, n_q] -- head and sequence are
# SWAPPED relative to q/k/v, which the file header flags as real and verified.
# Send axis 1 (n_head) to position 2 and axis 2 (n_q) to position 1 to reach
# [d_v, n_q, n_head], the layout .ag_flash_from_heads expects.
.ag_flash_unswap <- function(ctx, t3d) {
  ggml_cont(ctx, ggml_permute(ctx, t3d, 0L, 2L, 1L, 3L))
}

# Build the forward attention nodes in the shared pass context.
#
# Takes q/k/v as either device handles or R matrices in [d_model, seq] form,
# and returns the pieces the tape node needs: the result node, the flash-layout
# operand nodes the backward will attach to, and the uploads that have to run
# after the context flush.
#
# Nothing is computed here and nothing is flushed. The caller decides whether
# to queue this into the deferred forward or to compute it immediately, which
# is the same choice .ag_run_op makes -- and keeping it there rather than here
# is what lets attention join a fused forward+backward graph instead of forcing
# a compute in the middle of one.
.ag_flash_build_fwd <- function(ctx, ggml_type, q, k, v, n_heads, scale, mask_m,
                                d_model, seq_q, seq_kv) {
  d_head <- d_model %/% n_heads
  uploads <- list()

  # A handle contributes its pointer; a matrix needs a tensor and an upload.
  # Same rule as .ag_run_op, and the same payoff: a projection that is already
  # resident is not sent again.
  operand <- function(x, nc) {
    if (.ag_is_handle(x)) return(x$ptr)
    tt <- ggml_new_tensor_2d(ctx, ggml_type, d_model, nc)
    uploads[[length(uploads) + 1L]] <<- list(ptr = tt, val = as.numeric(x))
    tt
  }

  tq2 <- operand(q, seq_q)
  tk2 <- operand(k, seq_kv)
  tv2 <- operand(v, seq_kv)

  fq <- .ag_flash_to_heads(ctx, tq2, d_head, n_heads, seq_q)
  fk <- .ag_flash_to_heads(ctx, tk2, d_head, n_heads, seq_kv)
  fv <- .ag_flash_to_heads(ctx, tv2, d_head, n_heads, seq_kv)

  # The mask is a constant of the graph, not a tracked value: it is built from
  # `causal` or supplied by the caller and never receives a gradient. ggml wants
  # it F16 and contiguous, and 0/-Inf survive that conversion exactly.
  tm <- NULL
  if (!is.null(mask_m)) {
    tm <- ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nrow(mask_m), ncol(mask_m))
    uploads[[length(uploads) + 1L]] <- list(ptr = tm, val = as.numeric(mask_m))
  }

  res <- ggml_flash_attn_ext(ctx, fq, fk, fv, tm, scale, 0, 0)

  # [d_v, n_head, n_q] -> [d_v, n_q, n_head] -> [d_model, seq_q].
  out <- .ag_flash_from_heads(ctx, .ag_flash_unswap(ctx, res),
                              d_head, n_heads, seq_q)

  list(out = out, fq = fq, fk = fk, fv = fv, tm = tm, uploads = uploads)
}

# Build the backward attention nodes, given the gradient of the output as a
# graph node.
#
# `g` is [d_model, seq_q]; ggml_flash_attn_back wants it in the result's own
# permuted layout [d_v, n_head, n_q], so it goes through the head split and
# then the swap that .ag_flash_unswap undoes on the way out.
#
# The three gradients come back PACKED into one contiguous buffer with each
# slice aligned to 16 bytes. They are cut out with views rather than downloaded
# and re-uploaded: a view costs no memory and no transfer, and the offsets are
# computed from the same .ag_flash_pad() the closure path uses, so the two
# cannot drift apart. Offsets for ggml_view_1d are BYTES, which is why the
# element counts are multiplied by the type size.
#
# Returns the three gradients already back in [d_model, seq] form, so the
# caller accumulates them without knowing anything about heads.
.ag_flash_build_bwd <- function(ctx, g, nd) {
  d_model <- nd$d_model; n_heads <- nd$n_heads
  d_head  <- d_model %/% n_heads
  seq_q   <- nd$seq_q;  seq_kv  <- nd$seq_kv

  # g [d_model, seq_q] -> [d_head, seq_q, n_head] -> [d_head, n_head, seq_q].
  gh <- .ag_flash_to_heads(ctx, g, d_head, n_heads, seq_q)
  td <- .ag_flash_unswap(ctx, gh)

  packed <- ggml_flash_attn_back(ctx, nd$fq, nd$fk, nd$fv, nd$tm, td, nd$scale)

  nq <- d_head * seq_q  * n_heads
  nk <- d_head * seq_kv * n_heads
  # dv == d_head here because ag_flash_attention() requires q, k and v to share
  # d_model and splits all three by the same n_heads. ggml itself allows
  # dv != dk; if this wrapper ever does, this line is the one that has to grow a
  # separate d_v -- getting it wrong shifts nothing in dq or dk and silently
  # truncates dv.
  nv <- d_head * seq_kv * n_heads
  off_k <- .ag_flash_pad(nq)
  off_v <- off_k + .ag_flash_pad(nk)

  # The packed buffer is F32 regardless of the compute dtype: ggml_flash_attn_back
  # allocates it as F32, and .ag_flash_pad() is written in units of 4-byte floats
  # (ceiling(n * 4 / 16) * 4), matching. Sizing the stride from the compute type
  # instead would shift dk and dv silently under f16.
  esz <- ggml_type_size(GGML_TYPE_F32)

  slice <- function(n, off_elems, seq_len_) {
    vw <- ggml_view_1d(ctx, packed, n, off_elems * esz)
    t3 <- ggml_reshape_3d(ctx, vw, d_head, seq_len_, n_heads)
    .ag_flash_from_heads(ctx, t3, d_head, n_heads, seq_len_)
  }

  list(q = slice(nq, 0,     seq_q),
       k = slice(nk, off_k, seq_kv),
       v = slice(nv, off_v, seq_kv))
}

# The resident forward: build, record, hand back a handle.
#
# Mirrors .ag_run_op's structure deliberately -- same context, same deferral
# decision, same choice between queueing and computing -- because attention has
# to be indistinguishable from any other resident op to the machinery around it.
# It cannot literally BE .ag_run_op: that helper builds 2D operand tensors and
# takes one output shape, and attention needs three operands reshaped into 3D
# plus a mask in a different dtype. The 3D special case stays here.
.ag_flash_attention_resident <- function(q, k, v, qo, ko, vo, n_heads, scale,
                                         mask_m, d_model, seq_q, seq_kv,
                                         needs_grad) {
  ggml_type <- .ag_dtype_to_ggml(.ag_compute_dtype())

  # Descriptor budget: three operands, three head conversions (reshape+permute
  # +cont each), the mask, the attention node and the output conversion. Slack
  # on top, since overflowing a context aborts R inside ggml_new_tensor_impl
  # rather than returning something to fall back from.
  ctx <- .ag_ctx_ensure(32L, scope = "pass")

  b <- .ag_flash_build_fwd(ctx, ggml_type, qo, ko, vo, n_heads, scale, mask_m,
                           d_model, seq_q, seq_kv)


  # Deferral, on the same terms as every other op: if the queue is open, this
  # attention joins it and computes with everything else at the barrier. This
  # is the half of the fix that the missing `op` blocked from the other end --
  # a tape whose attention forced a compute could not be one fused graph even
  # if its backward were emittable.
  if (.ag_defer_ok("pass", NULL)) {
    .ag_defer_push(b$out, b$uploads)
    h <- .ag_handle(b$out, c(d_model, seq_q), scope = "pass", pending = TRUE)
  } else {
    if (.ag_defer_len()) .ag_defer_drain()
    .ag_ctx_flush(ctx, scope = "pass")
    for (u in b$uploads) .ag_xfer_up(u$ptr, u$val, "flash_attn operands")

    ctx_graph <- ggml_init(.ag_graph_ctx_bytes(), no_alloc = TRUE)
    if (is.null(ctx_graph))
      stop("ggmlR: failed to create a graph context for flash attention.",
           call. = FALSE)
    on.exit(ggml_free(ctx_graph), add = TRUE)
    graph <- ggml_build_forward_expand(ctx_graph, b$out)
    ggml_backend_graph_compute(.ag_device_state$backend, graph)
    h <- .ag_handle(b$out, c(d_model, seq_q), scope = "pass")
  }
  out <- .ag_tensor_from_handle(h, dtype = .ag_device_state$dtype)
  out$requires_grad <- needs_grad

  if (needs_grad) {
    # The closure stays as the fallback: a tape can still reach the closure path
    # (graph backward off, or another node on the tape unsupported), and then
    # this is what computes the gradient. It reads through .ag_as_matrix so a
    # resident gradient arriving as a handle materialises rather than erroring.
    q_ref <- q; k_ref <- k; v_ref <- v
    fq <- b$fq; fk <- b$fk; fv <- b$fv; tm <- b$tm
    grad_fn <- function(grad_out) {
      qh <- .ag_flash_split_heads(.ag_as_matrix(.ag_data(q_ref)), n_heads)
      kh <- .ag_flash_split_heads(.ag_as_matrix(.ag_data(k_ref)), n_heads)
      vh <- .ag_flash_split_heads(.ag_as_matrix(.ag_data(v_ref)), n_heads)
      gh <- aperm(.ag_flash_split_heads(.ag_as_matrix(grad_out), n_heads),
                  c(1L, 3L, 2L))
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

    # ⚠️ THE POINT OF THE WHOLE CHANGE. With `op` set, a tape carrying attention
    # is no longer rejected wholesale by .ag_bwd_reject_reason -- its matmuls
    # keep the graph backward and the stage-2 fusion they had before attention
    # was added to the model.
    #
    # The recorded fields are the flash-layout NODES, not values: the backward
    # attaches ggml_flash_attn_back to the very tensors the forward built, so
    # nothing crosses the bus in between. They stay valid because the pass pool
    # is reset at the start of the next tape, not at the end of this one.
    ag_record(out, grad_fn, list(q = q, k = k, v = v), op = "flash_attn",
              fq = fq, fk = fk, fv = fv, tm = tm, scale = scale,
              n_heads = n_heads, d_model = d_model,
              seq_q = seq_q, seq_kv = seq_kv)
  }
  out
}
