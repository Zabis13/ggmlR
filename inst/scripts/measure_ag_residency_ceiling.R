#!/usr/bin/env Rscript
#
# Would a resident path B ever beat the CPU, and at what size?
#
# The question behind the "change the contract" decision. Path B today
# materialises every intermediate into an R matrix, so each ag_* call pays an
# upload and a download; that is measured and closed (52.9% of an ag_matmul is
# the upload alone, inst/scripts/probe_ag_upload_cache.R). The proposal is to
# make tensors live on the device with R holding a handle, which is how PyTorch
# and JAX avoid the same wall.
#
# What must NOT be measured here is today's path B against the CPU -- that
# answer is known and it loses. The useful number is the CEILING: what the same
# arithmetic costs when the data is already resident and only the result comes
# back. If the CPU still wins there, the contract change cannot pay and the
# question is settled without writing it. If there is a crossover, this says at
# which size it happens, which is the target the new contract would aim at.
#
# How the ceiling is built. Not with ag_* -- those are the thing being bypassed
# -- but with the same primitives .ag_run_op uses underneath: create tensors in
# one context, upload once, build a chain of nodes, compute the whole graph in
# one call, download once. That is exactly what a resident path B would do, so
# it bounds it from below. Anything the real implementation adds (bookkeeping,
# version checks, R-side dispatch) only moves it up.
#
# The CPU baseline is R's own %*%, which is threaded BLAS on this machine --
# the honest opponent, not a single-threaded strawman.
#
# Sizes. The sweep deliberately runs past where anyone would train from R:
# industry crossover happens where arithmetic dominates transfer, which needs
# large d AND large batch. Earlier measurement (measure_ag_gpu_threshold.R)
# found the CPU ahead by 12.7x at d=2048 on the per-op path and the gap
# WIDENING -- so if residency does not change that shape, nothing will.
#
# Run:  Rscript inst/scripts/measure_ag_residency_ceiling.R
# Env:  GGMLR_RES_REPS   iterations per timing (default 10)
#       GGMLR_RES_CHAIN  ops per chain (default 8)

suppressMessages(library(ggmlR))

reps  <- as.integer(Sys.getenv("GGMLR_RES_REPS",  "10"))
chain <- as.integer(Sys.getenv("GGMLR_RES_CHAIN", "8"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

tm <- function(f, warm = 2L) {
  for (i in seq_len(warm)) f()
  t0 <- Sys.time()
  for (i in seq_len(reps)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps
}

# A chain of `chain` matmuls sharing one weight, resident: everything uploaded
# once, the whole chain built as ONE graph, one download at the end.
#
# This is the shape a resident path B would produce for a forward pass through
# a repeated block, and the closest honest stand-in for it that can be written
# without changing .ag_run_op.
# The backend is created once for the whole script, not per call. Initialising
# Vulkan is a one-off cost that a resident path B would pay at ag_device("gpu")
# and never again, so charging it to every timed iteration would measure the
# wrong thing.
BACKEND <- ggml_vulkan_init(0L)
if (is.null(BACKEND)) { cat("Vulkan init failed.\n"); quit(status = 0L) }

gpu_resident_chain <- function(W, X, n_ops) {
  # Size the context from the work. A fixed 64 MB ran out at d = 4096, and
  # sizing by descriptor count alone ran out at d = 2048 -- so the context is
  # not holding descriptors only. Budget the full tensor data as well: two
  # operands plus one result per op, at 4 bytes per f32 element, times four for
  # headroom.
  d_ <- as.double(nrow(W)); b_ <- as.double(ncol(X))
  bytes_data <- (d_ * d_ + d_ * b_ * (as.double(n_ops) + 1)) * 4
  bytes_desc <- (as.double(n_ops) + 4) * as.double(ggml_tensor_overhead())
  ctx_bytes  <- max(16 * 1024 * 1024, 4 * (bytes_data + bytes_desc))
  ctx <- ggml_init(ctx_bytes, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  d <- nrow(W); b <- ncol(X)
  # ggml_mul_mat needs the shared dimension in ne[0] of both operands, so the
  # weight goes in transposed -- same convention as .ag_gpu_matmul.
  tW <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d)
  tX <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, b)

  node <- tX
  for (i in seq_len(n_ops)) node <- ggml_mul_mat(ctx, tW, node)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, BACKEND)

  ggml_backend_tensor_set_data(tW, as.numeric(t(W)))
  ggml_backend_tensor_set_data(tX, as.numeric(X))

  ctx_g <- ggml_init(as.double(ggml_graph_overhead()) + 65536, no_alloc = TRUE)
  on.exit(ggml_free(ctx_g), add = TRUE)
  graph <- ggml_build_forward_expand(ctx_g, node)
  ggml_backend_graph_compute(BACKEND, graph)
  invisible(ggml_backend_tensor_get_data(node))
}

cpu_chain <- function(W, X, n_ops) {
  out <- X
  for (i in seq_len(n_ops)) out <- W %*% out
  out
}

cat(sprintf("reps = %d, chain = %d matmuls sharing one weight\n", reps, chain))
cat("Resident GPU = upload once, one graph, download once (the ceiling).\n")
cat("CPU = R's own %*% (threaded BLAS).\n\n")

# ---------------------------------------------------------------------------
# 0. THE REGIME ACTUALLY IN USE -- this section decides the question
#
# Everything below sweeps outward to find where the lines cross, but the
# decision does not depend on where that is in the abstract: it depends on
# whether the crossover falls inside the sizes these models are trained at.
# Past measurement puts that at d = 256-1024 and batch up to 64.
#
# If the ceiling loses here, the contract change is the same verdict as GPU
# optimizers, fused kernels and activation lifetime: physically sound, not
# worth it for this workload. The wider sweeps then only say how far away the
# crossover sits, which is context, not the answer.
# ---------------------------------------------------------------------------

cat("0. The regime actually in use (d 256-1024, batch <= 64)\n")
cat("   d      batch    CPU ms    GPU ms   speedup\n")
for (d in c(256L, 512L, 1024L)) {
  W <- matrix(rnorm(d * d), d, d)
  for (b in c(32L, 64L)) {
    X <- matrix(rnorm(d * b), d, b)
    t_cpu <- tm(function() cpu_chain(W, X, chain))
    t_gpu <- tm(function() gpu_resident_chain(W, X, chain))
    cat(sprintf("   %5d  %5d  %8.1f  %8.1f  %7.2fx%s\n",
                d, b, t_cpu, t_gpu, t_cpu / t_gpu,
                if (t_cpu > t_gpu) "  <-- GPU ahead" else ""))
  }
}

# ---------------------------------------------------------------------------
# 1. Square-ish: batch tracks the layer width
#
# The regime the earlier threshold sweep covered, extended upward. If residency
# changes the picture at all, the change shows up against those known numbers.
# ---------------------------------------------------------------------------

cat("1. Layer width, batch = 256 fixed\n")
cat("   d        CPU ms    GPU ms   speedup   GFLOP\n")
b <- 256L
for (d in c(512L, 1024L, 2048L, 4096L)) {
  W <- matrix(rnorm(d * d), d, d)
  X <- matrix(rnorm(d * b), d, b)
  t_cpu <- tm(function() cpu_chain(W, X, chain))
  t_gpu <- tm(function() gpu_resident_chain(W, X, chain))
  gf    <- 2 * as.double(d) * d * b * chain / 1e9
  cat(sprintf("   %5d  %8.1f  %8.1f  %7.2fx  %6.1f\n",
              d, t_cpu, t_gpu, t_cpu / t_gpu, gf))
}

# ---------------------------------------------------------------------------
# 2. Batch, layer fixed
#
# The one direction that helped on the per-op path (the weight upload is paid
# once however wide the batch). With residency it is paid once per CHAIN, so
# this sweep says whether batch still matters once that is true.
# ---------------------------------------------------------------------------

cat("\n2. Batch size, d = 1024 fixed\n")
cat("   batch    CPU ms    GPU ms   speedup   GFLOP\n")
d <- 1024L
W <- matrix(rnorm(d * d), d, d)
for (b in c(32L, 128L, 512L, 2048L)) {
  X <- matrix(rnorm(d * b), d, b)
  t_cpu <- tm(function() cpu_chain(W, X, chain))
  t_gpu <- tm(function() gpu_resident_chain(W, X, chain))
  gf    <- 2 * as.double(d) * d * b * chain / 1e9
  cat(sprintf("   %5d  %8.1f  %8.1f  %7.2fx  %6.1f\n",
              b, t_cpu, t_gpu, t_cpu / t_gpu, gf))
}

# ---------------------------------------------------------------------------
# 3. Chain length
#
# How much of the win is amortising the one upload. A long chain is the best
# case for residency; if the speedup does not improve with length, the transfer
# was never the binding constraint at this size.
# ---------------------------------------------------------------------------

cat("\n3. Chain length, d = 1024, batch = 256\n")
cat("   n_ops    CPU ms    GPU ms   speedup\n")
d <- 1024L; b <- 256L
W <- matrix(rnorm(d * d), d, d)
X <- matrix(rnorm(d * b), d, b)
for (n in c(1L, 4L, 16L, 64L)) {
  t_cpu <- tm(function() cpu_chain(W, X, n))
  t_gpu <- tm(function() gpu_resident_chain(W, X, n))
  cat(sprintf("   %5d  %8.1f  %8.1f  %7.2fx\n", n, t_cpu, t_gpu, t_cpu / t_gpu))
}

# ---------------------------------------------------------------------------
# 4. What today's path B costs on the same work, for scale
#
# Not the comparison that decides anything -- it is known to lose -- but it
# says how much of the gap the contract change would actually close.
# ---------------------------------------------------------------------------

cat("\n4. Today's per-op path B on the same chain (d = 1024, batch = 256)\n")
ag_device("gpu")
Wt <- ag_tensor(W); Xt <- ag_tensor(X)
t_ag <- tm(function() {
  out <- Xt
  for (i in seq_len(chain)) out <- ag_matmul(Wt, out)
  out
}, warm = 1L)
ag_device("cpu")
t_cpu <- tm(function() cpu_chain(W, X, chain))
t_gpu <- tm(function() gpu_resident_chain(W, X, chain))
cat(sprintf("   path B today : %8.1f ms\n", t_ag))
cat(sprintf("   resident GPU : %8.1f ms  (%.1fx better than today)\n",
            t_gpu, t_ag / t_gpu))
cat(sprintf("   CPU          : %8.1f ms\n", t_cpu))

ggml_vulkan_free(BACKEND)

cat("\nReading the numbers -- section 0 decides, the rest is context:\n")
cat("  sec 0 speedup > 1        -> the crossover is INSIDE the regime these\n")
cat("                              models train at; change the contract.\n")
cat("  sec 0 speedup < 1        -> residency alone does not beat the CPU where\n")
cat("                              it matters. Same verdict as GPU optimizers\n")
cat("                              and fused kernels: sound, not worth it here.\n")
cat("                              Sweeps 1-2 then only say how far off it is.\n")
cat("  sec 1-2 rising with size -> crossover exists but further out; record the\n")
cat("                              size as the condition for reopening.\n")
cat("  sec 3 flat in n_ops      -> transfer was not the binding constraint at\n")
cat("                              this size; the kernel is.\n")
cat("  sec 4 gap today->ceiling -> how much of the loss the contract would\n")
cat("                              close, whether or not it reaches the CPU.\n")
