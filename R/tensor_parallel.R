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
