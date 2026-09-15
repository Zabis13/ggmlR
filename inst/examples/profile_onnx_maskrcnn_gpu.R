#!/usr/bin/env Rscript
# MaskRCNN ONNX GPU profiler — speed of the Vulkan path, nothing else.
#
# This measures how long the GPU path takes. It does NOT check the numbers:
# no CPU run, no ONNX Runtime reference, no comparison of any kind. A model
# that returns zeros will be profiled just as happily as one that works, so
# do not read a good FPS here as evidence that the model is correct.
#
# Unlike profile_onnx_superres_gpu.R there is no sweep over input sizes: a
# detector has one input shape (3x224x224) and resizing it does not mean what
# resizing a super-resolution input means.
#
# Usage:
#   Rscript inst/examples/profile_onnx_maskrcnn_gpu.R
#
# Environment:
#   ONNX_DIR   where the .onnx files live
#   N_RUNS     timed runs (default 1 — see below)
#   N_WARMUP   untimed runs first (default 0 — see below)
#
# ⚠️ This model currently SEGFAULTS on the second onnx_run of the same loaded
#    model, so both defaults are 1 and 0: one run, no warmup. That means the
#    single timing includes one-off setup and there is no spread to report.
#    Raise N_RUNS once repeated runs survive.
#   PERF_LOG=1 turn on GGML_VK_PERF_LOGGER for per-kernel Vulkan timings.
#              ⚠️ OFF by default on purpose: the logger inflates the very
#              timings this script exists to measure. Use it to find WHICH
#              kernel is slow, never to answer HOW slow the model is.

suppressMessages(library(ggmlR))

ONNX_DIR  <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")
ONNX_PATH <- file.path(ONNX_DIR, "MaskRCNN-12-int8.onnx")
N_RUNS    <- as.integer(Sys.getenv("N_RUNS",   "1"))
N_WARMUP  <- as.integer(Sys.getenv("N_WARMUP", "0"))
SHAPE     <- c(3L, 224L, 224L)

if (identical(Sys.getenv("PERF_LOG"), "1")) {
  Sys.setenv(GGML_VK_PERF_LOGGER = "1")
  cat("⚠️ GGML_VK_PERF_LOGGER is on — reported times are inflated.\n\n")
}

cat("=== MaskRCNN ONNX GPU Profiler ===\n\n")

if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)))
  stop("Vulkan not available")
if (!file.exists(ONNX_PATH))
  stop("model not found: ", ONNX_PATH)

gpu_mem <- ggml_vulkan_device_memory(0L)
cat(sprintf("GPU  : %s\n", ggml_vulkan_device_description(0L)))
cat(sprintf("VRAM : %.1f / %.1f GB free\n", gpu_mem$free / 1e9, gpu_mem$total / 1e9))
cat(sprintf("Model: %s  (%.2f MB)\n\n",
            basename(ONNX_PATH), file.size(ONNX_PATH) / 1024 / 1024))

# ---- Load on GPU ----
cat("Loading on Vulkan ... ")
t_load <- system.time({
  model <- onnx_load(ONNX_PATH, device = "vulkan",
                     input_shapes = list(image = SHAPE))
})[3]
cat(sprintf("%.2f s\n\n", t_load))

# ---- Scheduler diagnostics ----
di <- tryCatch(onnx_device_info(model), error = function(e) NULL)
if (!is.null(di)) {
  cat(sprintf("Backends : %s\n", paste(di$backends, collapse = ", ")))
  cat(sprintf("Graph    : %d nodes, %d splits\n", di$n_nodes, di$n_splits))
  cat(sprintf("Ops      : GPU=%d  CPU-only=%d\n", di$gpu_ops, di$cpu_ops))
  if (isTRUE(di$cpu_ops > 0L))
    cat(sprintf("CPU-only : %s\n",
                paste(sprintf("%s(%d)", names(di$cpu_only_ops), di$cpu_only_ops),
                      collapse = ", ")))
  cat("\n")
}

# ---- Input ----
set.seed(42)
inp <- list(image = runif(prod(SHAPE)))

# ---- Warmup ----
# N_WARMUP defaults to 0: this model segfaults on repeated onnx_run, so the
# first run is also the only safe one. Set N_WARMUP>0 once that is fixed —
# without a warmup the first timing carries one-off allocation costs.
if (N_WARMUP > 0L) {
  cat(sprintf("Warmup x%d ... ", N_WARMUP))
  t_warm <- system.time(for (i in seq_len(N_WARMUP)) onnx_run(model, inp))[3]
  cat(sprintf("%.2f s total (%.1f ms/run)\n", t_warm, t_warm / N_WARMUP * 1e3))
} else {
  cat("Warmup skipped (N_WARMUP=0) — first timing includes setup cost.\n")
}

# ---- Timed runs ----
cat(sprintf("Timing x%d ...\n", N_RUNS))
times <- numeric(N_RUNS)
for (i in seq_len(N_RUNS)) {
  t0       <- proc.time()
  out      <- onnx_run(model, inp)
  times[i] <- (proc.time() - t0)[3]
  cat(sprintf("  run %2d: %7.1f ms\n", i, times[i] * 1e3))
}

ms <- times * 1e3
cat("\n=== Summary ===\n")
cat(sprintf("mean   %8.1f ms\n", mean(ms)))
cat(sprintf("median %8.1f ms\n", median(ms)))
cat(sprintf("min    %8.1f ms\n", min(ms)))
cat(sprintf("max    %8.1f ms\n", max(ms)))
cat(sprintf("sd     %8.1f ms\n", sd(ms)))
cat(sprintf("FPS    %8.2f\n", 1 / mean(times)))

# Output shapes only — this script does not judge the values.
cat(sprintf("\nOutputs: %d\n", length(out)))
for (i in seq_along(out)) {
  v  <- out[[i]]
  sh <- dim(v); if (is.null(sh)) sh <- length(v)
  cat(sprintf("  [%d] shape [%s], %d values\n",
              i, paste(sh, collapse = "x"), length(v)))
}

mem_after <- ggml_vulkan_device_memory(0L)
cat(sprintf("\nVRAM after: %.1f GB free (used %.0f MB)\n",
            mem_after$free / 1e9, (gpu_mem$free - mem_after$free) / 1e6))

if (!is.null(di) && isTRUE(di$cpu_ops > 0L))
  cat(sprintf("\n%d CPU-only op(s) fall back mid-graph; each fallback splits the\ngraph and copies tensors across the bus.\n", di$cpu_ops))
