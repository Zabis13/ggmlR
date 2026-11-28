#!/usr/bin/env Rscript
#
# Is the stateful DP-over-meta path worth its state?
#
# ggml_tp_dp_forward_graph() (stateless) builds a meta backend per replica and
# uploads the weights on every call. ggml_tp_dp_create() does that once and
# ggml_tp_dp_run() only uploads the input and computes. This script compares:
#
#   cold   -- one ggml_tp_dp_forward_graph() call
#   create -- ggml_tp_dp_create() alone (the one-off cost of the stateful path)
#   first  -- the first ggml_tp_dp_run() on a handle (graph build for the width)
#   warm   -- later ggml_tp_dp_run() calls (cache hit)
#
# on 1 and 2 replicas. If warm is not clearly below cold, the stateful API does
# not pay for itself on this hardware.
#
# Cold and warm are interleaved in ONE process (cross-process A/B on the same
# GPU drifts by tens of percent), and every timing is a median over REPS.
#
# Usage:
#   Rscript inst/scripts/measure_tp_dp_warm.R                 # devices 0,0 (Vulkan)
#   DEVICES=CPU,CPU Rscript inst/scripts/measure_tp_dp_warm.R
#   DEVICES=0,1 N_HID=8192 BATCH=128 REPS=20 Rscript inst/scripts/measure_tp_dp_warm.R
#
# Output: printed and written to /tmp/measure_tp_dp_warm.txt

suppressPackageStartupMessages(library(ggmlR))

env_int <- function(name, default) as.integer(Sys.getenv(name, default))
devs_raw <- strsplit(Sys.getenv("DEVICES", "0,0"), ",")[[1]]
devices  <- if (all(grepl("^[0-9]+$", devs_raw))) as.integer(devs_raw) else devs_raw
N_IN  <- env_int("N_IN",  1024L)
N_HID <- env_int("N_HID", 4096L)
N_OUT <- env_int("N_OUT", 1024L)
BATCH <- env_int("BATCH", 64L)
REPS  <- env_int("REPS",  10L)
OUT_FILE <- "/tmp/measure_tp_dp_warm.txt"

set.seed(1)
W1 <- array(rnorm(N_IN * N_HID) * 0.02, c(N_IN, N_HID))
W2 <- array(rnorm(N_HID * N_OUT) * 0.02, c(N_HID, N_OUT))
X  <- array(rnorm(N_IN * BATCH), c(N_IN, BATCH))
weights <- list(w1 = W1, w2 = W2)
mlp <- function(ctx, input, w)
  ggml_mul_mat(ctx, w$w2, ggml_relu(ctx, ggml_mul_mat(ctx, w$w1, input)))
split <- list(w1 = "rows", w2 = "cols")

ms <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  force(expr)
  (proc.time()[["elapsed"]] - t0) * 1000
}

lines <- c(sprintf("devices per replica: %s | W1 %dx%d, W2 %dx%d, batch %d, reps %d",
                   paste(devs_raw, collapse = ","), N_IN, N_HID, N_HID, N_OUT, BATCH, REPS),
           sprintf("%-9s %10s %10s %10s %10s %8s %12s",
                   "replicas", "cold_ms", "create_ms", "first_ms", "warm_ms",
                   "cold/warm", "max|warm-cold|"))

for (n_rep in 1:2) {
  replicas <- rep(list(devices), n_rep)
  cold <- create <- first <- warm <- numeric(REPS)
  diff <- 0
  for (k in seq_len(REPS)) {
    # cold, stateless
    cold[k] <- ms(y_cold <- ggml_tp_dp_forward_graph(weights, mlp, X, replicas, split = split))
    # stateful: create, first run, then a warm run -- interleaved with cold
    create[k] <- ms(h <- ggml_tp_dp_create(weights, mlp, replicas, split = split))
    first[k]  <- ms(ggml_tp_dp_run(h, X))
    warm[k]   <- ms(y_warm <- ggml_tp_dp_run(h, X))
    ggml_tp_dp_free(h)
    diff <- max(diff, max(abs(y_warm - y_cold)))
  }
  m <- vapply(list(cold, create, first, warm), median, numeric(1))
  lines <- c(lines, sprintf("%-9d %10.2f %10.2f %10.2f %10.2f %8.2f %12.3g",
                            n_rep, m[1], m[2], m[3], m[4], m[1] / m[4], diff))
}

writeLines(lines)
writeLines(lines, OUT_FILE)
cat("written to", OUT_FILE, "\n")
