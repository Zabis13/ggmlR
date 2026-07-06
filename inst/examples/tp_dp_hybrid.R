#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# ggmlR Tensor Parallelism (P2P) — TPxDP hybrid demo (Stage E6).
#
# Demonstrates the target inference layout for a 4-GPU box: 2 data-parallel
# replicas, each doing TP=2 (tensor parallelism across a pair of GPUs).
#
#   replica A = GPUs {0,1}   |  replica B = GPUs {2,3}
#   weights are replicated to both groups (data parallelism over the batch);
#   within a group the weight rows are split (tensor parallelism).
#
# During inference there is no cross-replica communication, so throughput scales
# with the number of replicas while each replica's cross-device TP gather stays
# local to its pair (host-staging).
#
# Falls back gracefully: with < 4 GPUs it uses whatever device groups fit, and
# with 1 GPU it still runs (TP=1, DP=1) to show the API.
#
# Usage:  Rscript inst/examples/tp_dp_hybrid.R
# ---------------------------------------------------------------------------

suppressMessages(library(ggmlR))

if (!ggml_vulkan_available()) {
  cat("Vulkan not available — nothing to demo.\n")
  quit(save = "no", status = 0)
}

ndev <- ggml_vulkan_device_count()
cat(sprintf("Vulkan devices: %d\n", ndev))

# Choose a replica layout from the available devices: pairs where possible.
replicas <- if (ndev >= 4) {
  list(c(0L, 1L), c(2L, 3L))          # 2 replicas x TP=2 (the target)
} else if (ndev >= 2) {
  list(c(0L, 1L))                     # 1 replica x TP=2
} else {
  list(c(0L))                         # 1 replica x TP=1 (single GPU)
}
cat("Replica layout (each vector = one DP replica of TP GPUs):\n")
for (i in seq_along(replicas)) {
  cat(sprintf("  replica %d -> GPUs {%s}\n", i, paste(replicas[[i]], collapse = ",")))
}

# A small linear layer: Y = X %*% t(W), W = [N x K], batch X = [M x K].
set.seed(1L)
N <- 4096L; K <- 256L; M <- 16L
W <- matrix(rnorm(N * K), nrow = N)
X <- matrix(rnorm(M * K), nrow = M)

cat(sprintf("\nW = %d x %d, batch X = %d x %d\n", N, K, M, K))

# Inspect how much VRAM one replica's split weight occupies across its group.
grp <- replicas[[1]]
bt  <- ggml_vulkan_split_buffer_type(device_ids = grp, probe = c(N, K))
cat(sprintf("Split buffer type: %s\n  total across group = %.2f MB\n",
            bt$name, bt$alloc_size / (1024^2)))

t0 <- Sys.time()
Y  <- ggml_tp_dp_forward(W, X, replicas = replicas)
dt <- as.numeric(Sys.time() - t0, units = "secs")

ref     <- X %*% t(W)
max_err <- max(abs(Y - ref))
cat(sprintf("\nTPxDP forward: Y = %d x %d in %.3fs\n", nrow(Y), ncol(Y), dt))
cat(sprintf("max |Y - X %%*%% t(W)| = %.3e  (%s)\n",
            max_err, if (max_err < 5e-2) "OK" else "MISMATCH"))
