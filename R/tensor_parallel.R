# ggmlR Tensor Parallelism (P2P) — TPxDP hybrid orchestration (Stage E6.2).
#
# NOT upstream. A thin R layer over the C tensor-parallel primitive
# (ggml_vulkan_split_mul_mat) that runs a hybrid of tensor parallelism (TP, weight
# rows split within a device group) and data parallelism (DP, the batch split
# across independent replicas that each hold the full weights).
#
# The target scenario (4x P100): 4 GPUs = 2 replicas x TP=2. Replica A owns GPUs
# {0,1}, replica B owns {2,3}; the weights are replicated across the two groups
# and each replica computes half the batch. During inference there is NO
# cross-replica communication — the only cross-device traffic is the intra-group
# TP gather (which host-staging handles). DP hides host-staging's cost: throughput
# scales with the number of replicas while each replica's TP gather stays small.

#' Split a batch index range into (near) equal contiguous shards
#'
#' Helper: partition \code{seq_len(M)} into \code{n} contiguous blocks, as even as
#' possible (earlier shards get the +1 when M is not divisible by n).
#' @param M Number of rows (batch size).
#' @param n Number of shards.
#' @return A list of integer vectors of row indices (1-based), length \code{n};
#'   empty vectors are dropped.
#' @keywords internal
.ggmlr_batch_shards <- function(M, n) {
  if (n <= 1L) return(list(seq_len(M)))
  base <- M %/% n
  rem  <- M %%  n
  sizes <- rep(base, n) + c(rep(1L, rem), rep(0L, n - rem))
  ends   <- cumsum(sizes)
  starts <- c(1L, ends[-n] + 1L)
  shards <- Map(function(s, e) if (e >= s) s:e else integer(0), starts, ends)
  Filter(length, shards)
}

#' TPxDP hybrid matrix multiply across replicas of Vulkan device groups
#'
#' Computes \code{Y = X \%*\% t(W)} as a hybrid of tensor parallelism and data
#' parallelism: the weight matrix \code{W} is replicated across \code{replicas}
#' device groups (data parallelism over the batch \code{X}), and within each group
#' the weight rows are split across that group's GPUs (tensor parallelism, via
#' \code{\link{ggml_vulkan_split_mul_mat}}). Each replica computes a contiguous
#' shard of the batch rows; the shards are concatenated back into the full result.
#'
#' This mirrors the inference layout for the 4x P100 target (2 replicas x TP=2):
#' there is no cross-replica communication, so throughput scales with the replica
#' count while each replica's cross-device TP gather stays local to its group.
#'
#' @param W Weight matrix, \code{N x K} (replicated to every group).
#' @param X Activation matrix, \code{M x K}; its rows (the batch) are split across
#'   the replicas.
#' @param replicas A list of integer vectors, each a group of physical GPU indices
#'   (0-based) forming one replica, e.g. \code{list(c(0, 1), c(2, 3))}. Within a
#'   group the weights are tensor-split; across groups the batch is data-split.
#' @param weights Optional numeric vector (length = group size) giving the
#'   per-device TP row weighting inside each group (applied to every group).
#'   \code{NULL} (default) splits evenly.
#' @param transport Cross-device TP gather transport (see
#'   \code{\link{ggml_vulkan_split_mul_mat}}).
#' @return The \code{M x N} result matrix, equal to \code{X \%*\% t(W)} up to the
#'   GPU's floating-point accumulation.
#' @seealso \code{\link{ggml_vulkan_split_mul_mat}} for the single-group TP path.
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 4) {
#'   W <- matrix(rnorm(2048 * 64), nrow = 2048)
#'   X <- matrix(rnorm(8 * 64), nrow = 8)      # batch of 8
#'   # 2 replicas x TP=2: batch split 4/4, weights split within {0,1} and {2,3}
#'   Y <- ggml_tp_dp_forward(W, X, replicas = list(c(0, 1), c(2, 3)))
#'   max(abs(Y - X %*% t(W)))
#' }
#' }
ggml_tp_dp_forward <- function(W, X, replicas,
                               weights = NULL,
                               transport = c("host-staging", "opaque-fd", "device-group")) {
  transport <- match.arg(transport)
  if (!is.list(replicas) || length(replicas) < 1L) {
    stop("replicas must be a non-empty list of integer GPU-index vectors")
  }
  W <- as.matrix(W)
  X <- as.matrix(X)
  N <- nrow(W); K <- ncol(W)
  M <- nrow(X)
  if (ncol(X) != K) {
    stop(sprintf("X has %d columns but W has %d (both are the K/input dimension)",
                 ncol(X), K))
  }

  n_rep  <- length(replicas)
  shards <- .ggmlr_batch_shards(M, n_rep)

  Y <- matrix(0.0, nrow = M, ncol = N)
  for (i in seq_along(shards)) {
    idx   <- shards[[i]]
    group <- as.integer(replicas[[i]])
    # Each replica: tensor-parallel mul_mat of the full weights over its batch
    # shard, split across that group's GPUs. Independent of the other replicas.
    Y[idx, ] <- ggml_vulkan_split_mul_mat(
      W, X[idx, , drop = FALSE],
      device_ids = group, weights = weights, transport = transport)
  }
  Y
}

# --- Pipeline parallelism (Stage E7) -------------------------------------------

#' Hand a pipeline stage's activation tensor to the next stage's input
#'
#' Copies the contents of \code{src} (a Vulkan-backed activation tensor produced
#' by one pipeline stage, living on its device) into \code{dst} (the next stage's
#' input tensor, allocated on another device) via host staging. This is the single
#' cross-device transfer per forward pass that pipeline parallelism needs.
#'
#' @param src Source tensor (stage N output), Vulkan-backed.
#' @param dst Destination tensor (stage N+1 input), Vulkan-backed, same size.
#' @return Invisibly 0 on success; errors on a shape/buffer mismatch.
#' @keywords internal
ggml_vulkan_stage_handoff <- function(src, dst) {
  invisible(.Call("R_ggml_vk_stage_handoff", src, dst, PACKAGE = "ggmlR"))
}

#' Pipeline-parallel forward pass across per-device layer stages
#'
#' Runs a forward pass split BY LAYERS across devices (pipeline parallelism), the
#' complement of \code{\link{ggml_vulkan_split_mul_mat}}'s split-by-matrix tensor
#' parallelism. Each \code{stage} owns a contiguous block of the model's layers on
#' one GPU; the activation tensor is handed from one stage to the next exactly once
#' per pass (a single cross-device copy), versus TP's per-layer gather. This suits
#' models too large for one card's VRAM: at ~1 GB/s host staging the single
#' handoff costs only ~10-20 ms per pass.
#'
#' Each stage is a list with:
#' \describe{
#'   \item{\code{device}}{Physical GPU index (0-based) the stage runs on.}
#'   \item{\code{build}}{A function \code{function(ctx, input)} that builds this
#'     stage's sub-graph from the \code{input} tensor using the \code{ggml_*} ops
#'     and returns either the output tensor, or a list
#'     \code{list(output = <tensor>, set_weights = function() ...)}. The
#'     \code{set_weights} closure (if given) is called AFTER the stage's tensors
#'     are allocated on the device, and is where weight tensors created inside
#'     \code{build} get their values (via \code{ggml_backend_tensor_set_data}).}
#'   \item{\code{in_shape}}{Integer vector: the ggml \code{ne} shape of this
#'     stage's input tensor (fastest dim first). For stage 1 this must match the
#'     supplied \code{x}; for later stages it must match the previous stage's
#'     output shape.}
#' }
#'
#' @param stages A list of stage descriptors (see Details), in pipeline order.
#' @param x A numeric vector/array of input activations for the first stage,
#'   laid out in ggml (column-major) order matching \code{stages[[1]]$in_shape}.
#' @param out_shape Integer vector: the ggml \code{ne} shape of the final stage's
#'   output, used to size the returned vector.
#' @param mem_per_stage Bytes of ggml context metadata to reserve per stage
#'   (default 16 MiB) — raise it for stages with very many ops.
#' @return A numeric vector of the final stage's output activations (ggml order).
#' @seealso \code{\link{ggml_tp_dp_forward}} for the tensor-parallel counterpart.
#' @export
#' @examples
#' \donttest{
#' if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 2) {
#'   K <- 32L; M <- 4L
#'   W1 <- matrix(rnorm(K * K), K); W2 <- matrix(rnorm(K * K), K)
#'   stage <- function(dev, Wt) list(
#'     device = dev, in_shape = c(K, M),
#'     build = function(ctx, input) {
#'       w <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, K)
#'       ggml_mul_mat(ctx, w, input)   # (weights set inside build via attr, see vignette)
#'     })
#'   # See inst/examples/pp_pipeline.R for a complete runnable stage definition.
#' }
#' }
ggml_pp_forward <- function(stages, x, out_shape, mem_per_stage = 16L * 1024L * 1024L) {
  if (!is.list(stages) || length(stages) < 1L) {
    stop("stages must be a non-empty list of stage descriptors")
  }
  x <- as.numeric(x)

  # Resources kept alive across the pipeline: each stage's backend, ctx, input
  # and output tensors must outlive the handoff into the next stage.
  backends <- vector("list", length(stages))
  ctxs     <- vector("list", length(stages))
  inputs   <- vector("list", length(stages))
  outputs  <- vector("list", length(stages))

  on.exit({
    # Free backends (frees their device buffers) once the whole pass is done.
    for (b in backends) if (!is.null(b)) try(ggml_backend_free(b), silent = TRUE)
  }, add = TRUE)

  prev_output <- NULL
  for (i in seq_along(stages)) {
    st  <- stages[[i]]
    dev <- as.integer(st$device)
    ish <- as.integer(st$in_shape)

    backend <- ggml_vulkan_init(dev)
    ctx     <- ggml_init(mem_per_stage, no_alloc = TRUE)
    backends[[i]] <- backend
    ctxs[[i]]     <- ctx

    # Build this stage's input tensor + sub-graph. in_shape is a ggml ne vector;
    # support 1-D and 2-D inputs (the common activation shapes).
    if (length(ish) == 1L) {
      input <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ish[1])
    } else if (length(ish) == 2L) {
      input <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ish[1], ish[2])
    } else {
      stop("stage in_shape must be length 1 or 2 (ggml ne of the input)")
    }
    built <- st$build(ctx, input)
    # build() may return the output tensor directly, or a list with a
    # set_weights() closure to run after allocation.
    if (is.list(built) && !is.null(built$output)) {
      output      <- built$output
      set_weights <- built$set_weights
    } else {
      output      <- built
      set_weights <- NULL
    }
    inputs[[i]]  <- input
    outputs[[i]] <- output

    # Allocate this stage's tensors on its device.
    buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
    if (is.null(buf)) {
      stop(sprintf("pp_forward: allocation failed on device %d (out of VRAM?)", dev))
    }

    # Fill weights now that the stage's tensors have device storage.
    if (is.function(set_weights)) set_weights()

    # Fill the stage input: stage 1 from the host `x`; later stages via a single
    # cross-device handoff from the previous stage's output.
    if (i == 1L) {
      ggml_backend_tensor_set_data(input, x)
    } else {
      ggml_vulkan_stage_handoff(prev_output, input)
    }

    # Run the stage.
    graph <- ggml_build_forward_expand(ctx, output)
    st_status <- ggml_backend_graph_compute(backend, graph)

    prev_output <- output
  }

  # Read the final stage's output back to the host.
  n_out <- prod(as.integer(out_shape))
  ggml_backend_tensor_get_data(prev_output, n_elements = n_out)
}
