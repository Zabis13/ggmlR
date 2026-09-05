#!/usr/bin/env Rscript
#
# At what size does the Vulkan path stop losing to the CPU?
#
# measure_ag_upload_cost.R established that one ag_* op on the GPU is dominated
# by re-uploading its operands (~4 ms/MB), not by the arithmetic, and that at
# 1024x1024 the CPU wins outright: 9.9 ms against 19.6 ms. That raises the
# question this script answers -- is there a size where the GPU pulls ahead, and
# is it a size any real model reaches?
#
# The answer decides how much the upload cache in TODO.md section 3 is worth. A
# crossover at, say, d = 2048 makes the GPU path a real option for large layers
# and the cache a straightforward win. A crossover that only appears past the
# memory a card has makes the Vulkan path something to reach for in specific
# situations -- batched flash attention, where a 15-37x win is already measured
# -- rather than a general replacement for the CPU path.
#
# Method: one variable at a time. Sweeping d and batch together would confound
# them -- both change the FLOP count and the bytes moved, in different ratios --
# so section 1 holds the batch fixed and grows d, section 2 holds d fixed and
# grows the batch. Each timing normalises by the work done, so rows within a
# section are comparable rather than merely larger.
#
# Section 3 measures a chain rather than one op: a real forward pass reuses its
# weights, and the upload cost is paid per op, so the crossover for a model is
# not the crossover for a single matmul.
#
# Section 4 is the control. ag_flash_attention already beats the closure path by
# a wide margin, and it does so by uploading Q/K/V once for many operations. If
# the sweeps above show no crossover but this row still wins, the conclusion is
# about the per-op upload model, not about the Vulkan backend.
#
# Run:  Rscript inst/scripts/measure_ag_gpu_threshold.R
# Env:  GGMLR_TH_REPS    iterations per timing (default 10)
#       GGMLR_TH_WARMUP  warm-up calls before each timing (default 3)
#       GGMLR_TH_MAXD    largest d to try (default 2048; raise to probe further)

suppressMessages(library(ggmlR))

reps   <- as.integer(Sys.getenv("GGMLR_TH_REPS",   "10"))
warmup <- as.integer(Sys.getenv("GGMLR_TH_WARMUP", "3"))
max_d  <- as.integer(Sys.getenv("GGMLR_TH_MAXD",   "2048"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure here.\n")
  quit(status = 0L)
}

tm <- function(f) {
  for (i in seq_len(warmup)) f()
  t0 <- Sys.time()
  for (i in seq_len(reps)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps
}

# Build the operands on the device that will be timed, so the figure is the op
# and not a device switch.
on_device <- function(dev, mats) {
  ag_device(dev)
  lapply(mats, ag_tensor)
}

verdict <- function(cpu_ms, gpu_ms) {
  r <- cpu_ms / gpu_ms
  if (r > 1.05)      sprintf("GPU %4.2fx faster", r)
  else if (r < 0.95) sprintf("CPU %4.2fx faster", 1 / r)
  else               "        tie"
}

cat(sprintf("reps = %d, warm-up = %d, max d = %d\n", reps, warmup, max_d))

# ---------------------------------------------------------------------------
# 1. Growing the layer, fixed batch
# ---------------------------------------------------------------------------

cat("\n1. matmul [d,d] x [d,64] -- d grows, batch fixed at 64\n")
cat("       d   CPU ms   GPU ms   verdict\n")
batch <- 64L
ds    <- c(128L, 256L, 512L, 1024L, 1536L, 2048L)
ds    <- ds[ds <= max_d]
for (d in ds) {
  W <- matrix(rnorm(d * d), d, d)
  X <- matrix(rnorm(d * batch), d, batch)

  cw <- on_device("cpu", list(W = W, X = X))
  c_ms <- tm(function() ag_matmul(cw$W, cw$X))
  gw <- on_device("gpu", list(W = W, X = X))
  g_ms <- tm(function() ag_matmul(gw$W, gw$X))

  cat(sprintf("  %6d  %7.2f  %7.2f   %s\n", d, c_ms, g_ms, verdict(c_ms, g_ms)))
}

# ---------------------------------------------------------------------------
# 2. Growing the batch, fixed layer
#
# The weight upload is paid once per call whatever the batch is, so a larger
# batch spreads that fixed cost over more work. This is the direction in which
# the GPU is most likely to catch up.
# ---------------------------------------------------------------------------

cat("\n2. matmul [512,512] x [512,b] -- batch grows, layer fixed at 512\n")
cat("       b   CPU ms   GPU ms   verdict\n")
d <- 512L
W <- matrix(rnorm(d * d), d, d)
for (b in c(16L, 64L, 256L, 1024L, 4096L)) {
  X <- matrix(rnorm(d * b), d, b)

  cw <- on_device("cpu", list(W = W, X = X))
  c_ms <- tm(function() ag_matmul(cw$W, cw$X))
  gw <- on_device("gpu", list(W = W, X = X))
  g_ms <- tm(function() ag_matmul(gw$W, gw$X))

  cat(sprintf("  %6d  %7.2f  %7.2f   %s\n", b, c_ms, g_ms, verdict(c_ms, g_ms)))
}

# ---------------------------------------------------------------------------
# 3. A forward chain, not a single op
#
# Four layers with an activation between them: eight ops, each paying its own
# upload. Whatever the crossover is for one matmul, this is the number that
# decides whether a model runs faster on the GPU.
# ---------------------------------------------------------------------------

cat("\n3. 4-layer MLP forward, batch 64 -- 8 ops, each re-uploading its weight\n")
cat("       d   CPU ms   GPU ms   verdict\n")
mlp4 <- function(Ws, x) {
  h <- x
  for (W in Ws) h <- ag_relu(ag_matmul(W, h))
  h
}
for (d in ds[ds <= 1024L]) {
  raw_W <- replicate(4L, matrix(rnorm(d * d), d, d), simplify = FALSE)
  X     <- matrix(rnorm(d * 64L), d, 64L)

  ag_device("cpu")
  cW <- lapply(raw_W, ag_tensor); cX <- ag_tensor(X)
  c_ms <- tm(function() mlp4(cW, cX))
  ag_device("gpu")
  gW <- lapply(raw_W, ag_tensor); gX <- ag_tensor(X)
  g_ms <- tm(function() mlp4(gW, gX))

  cat(sprintf("  %6d  %7.2f  %7.2f   %s\n", d, c_ms, g_ms, verdict(c_ms, g_ms)))
}

# ---------------------------------------------------------------------------
# 4. Control: the one place the GPU path already wins
#
# ag_flash_attention uploads Q/K/V once and runs the whole attention inside a
# single call. If the sweeps above find no crossover while this still wins, the
# problem being measured is the per-op upload, not the backend.
# ---------------------------------------------------------------------------

cat("\n4. Control -- ag_flash_attention (one upload, many ops inside)\n")
cat("   d_model  seq   CPU ms   GPU ms   verdict\n")
for (cfg in list(c(64L, 128L), c(256L, 256L), c(512L, 512L))) {
  d_model <- cfg[[1L]]; seq_len_ <- cfg[[2L]]
  Q <- matrix(rnorm(d_model * seq_len_), d_model, seq_len_)
  K <- matrix(rnorm(d_model * seq_len_), d_model, seq_len_)
  V <- matrix(rnorm(d_model * seq_len_), d_model, seq_len_)
  nh <- 8L

  ag_device("cpu")
  cq <- ag_tensor(Q); ck <- ag_tensor(K); cv <- ag_tensor(V)
  c_ms <- tm(function() ag_flash_attention(cq, ck, cv, n_heads = nh))
  ag_device("gpu")
  gq <- ag_tensor(Q); gk <- ag_tensor(K); gv <- ag_tensor(V)
  g_ms <- tm(function() ag_flash_attention(gq, gk, gv, n_heads = nh))

  cat(sprintf("   %7d  %4d  %7.2f  %7.2f   %s\n",
              d_model, seq_len_, c_ms, g_ms, verdict(c_ms, g_ms)))
}

ag_device("cpu")

cat("\nReading the numbers:\n")
cat("  A crossover in section 1 or 2 -> the size above which the Vulkan path\n")
cat("    is worth choosing; the upload cache raises the GPU column further.\n")
cat("  No crossover, but section 4 wins -> the per-op upload model is the\n")
cat("    limit, not the backend: prefer whole-operation kernels over ag_* chains.\n")
cat("  Section 3 worse than section 1 -> the per-op cost compounds with depth,\n")
cat("    so a model's crossover sits above a single matmul's.\n")
