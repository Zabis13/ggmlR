#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# ggmlR Pipeline Parallelism (PP) demo — Stage E7.
#
# Splits a small 2-layer MLP BY LAYERS across devices: stage 1 (layer 1) on GPU 0,
# stage 2 (layer 2) on GPU 1. The activation tensor is handed from stage 1 to
# stage 2 exactly once (a single cross-device copy), the essence of pipeline
# parallelism — versus tensor parallelism's per-layer cross-device gather.
#
# Model:  h  = relu(W1 %*% x)   on stage 1
#         y  =      W2 %*% h    on stage 2
# with x an [K x M] activation batch (ggml layout: ne = c(K, M)).
#
# Falls back to a single device (stages 0 and 0) when < 2 GPUs are present, which
# still exercises the full pipeline + handoff path (a loopback copy).
#
# Usage:  Rscript inst/examples/pp_pipeline.R
# ---------------------------------------------------------------------------

suppressMessages(library(ggmlR))

if (!ggml_vulkan_available()) {
  cat("Vulkan not available — nothing to demo.\n")
  quit(save = "no", status = 0)
}

ndev <- ggml_vulkan_device_count()
cat(sprintf("Vulkan devices: %d\n", ndev))

# Two pipeline stages; use two GPUs if available, else both on device 0.
dev1 <- 0L
dev2 <- if (ndev >= 2) 1L else 0L
cat(sprintf("stage 1 (layer 1) -> GPU %d\nstage 2 (layer 2) -> GPU %d\n", dev1, dev2))

set.seed(1L)
K <- 64L; M <- 8L                       # K features, batch of M
W1 <- matrix(rnorm(K * K), nrow = K)    # layer 1 weights [K x K]
W2 <- matrix(rnorm(K * K), nrow = K)    # layer 2 weights [K x K]
X  <- matrix(rnorm(K * M), nrow = K)    # input, ggml ne = c(K, M): column m is sample m

# ggml_mul_mat(W, input): W is [K cols, K rows] (A), input is [K cols, M rows] (B),
# result is [K, M] = t(W) %*% input in R terms. Build each stage's sub-graph.
make_stage <- function(dev, Wt, relu) {
  list(
    device   = dev,
    in_shape = c(K, M),
    build = function(ctx, input) {
      w <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, K)
      z <- ggml_mul_mat(ctx, w, input)
      out <- if (relu) ggml_relu(ctx, z) else z
      list(
        output = out,
        # Weights are set after allocation. Wt is [K x K]; ggml wants it column-
        # major flattened, and mul_mat's A rows are Wt's rows -> pass as-is via
        # as.numeric(Wt) since R matrices are already column-major (ne = c(K,K)).
        set_weights = function() ggml_backend_tensor_set_data(w, as.numeric(Wt))
      )
    })
}

stages <- list(
  make_stage(dev1, W1, relu = TRUE),
  make_stage(dev2, W2, relu = FALSE)
)

t0 <- Sys.time()
y_ggml <- ggml_pp_forward(stages, x = as.numeric(X), out_shape = c(K, M))
dt <- as.numeric(Sys.time() - t0, units = "secs")

Y <- matrix(y_ggml, nrow = K, ncol = M)

# Reference: ggml_mul_mat(W, v) computes t(W) %*% v, so h = relu(t(W1) %*% X),
# y = t(W2) %*% h.
h_ref <- pmax(t(W1) %*% X, 0)
Y_ref <- t(W2) %*% h_ref

max_err <- max(abs(Y - Y_ref))
cat(sprintf("\nPipeline forward: y = %d x %d in %.3fs\n", nrow(Y), ncol(Y), dt))
cat(sprintf("max |Y - reference| = %.3e  (%s)\n",
            max_err, if (max_err < 5e-2) "OK" else "MISMATCH"))
