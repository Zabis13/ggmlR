#!/usr/bin/env Rscript
# SuperResolution ONNX GPU profiler
# Loads model once on Vulkan, runs multiple input sizes, reports timing breakdown
# Run: Rscript inst/examples/profile_onnx_superres_gpu.R

Sys.setenv(GGML_VK_PERF_LOGGER = "1")

library(ggmlR)

ONNX_PATH <- "/mnt/Data2/DS_projects/ONNX models-main/super-resolution-10.onnx"
N_WARMUP  <- 3L
N_RUNS    <- 10L

# Input sizes to profile (H x W, grayscale 1-channel)
SIZES <- list(
  "224x224" = c(1L, 1L, 224L, 224L),
  "360x360" = c(1L, 1L, 360L, 360L),
  "480x480" = c(1L, 1L, 480L, 480L),
  "720x720" = c(1L, 1L, 720L, 720L)
)

# ---- System info ----
cat("=== SuperResolution ONNX GPU Profiler ===\n\n")

if (!ggml_vulkan_available()) stop("Vulkan not available")

gpu_name <- ggml_vulkan_device_description(0)
gpu_mem  <- ggml_vulkan_device_memory(0)
cat(sprintf("GPU : %s\n", gpu_name))
cat(sprintf("VRAM: %.1f / %.1f GB\n", gpu_mem$free / 1e9, gpu_mem$total / 1e9))
cat(sprintf("Model: %s  (%.2f MB)\n\n",
            basename(ONNX_PATH), file.size(ONNX_PATH) / 1024 / 1024))

# ---- Load once on GPU ----
cat("Loading model on Vulkan ... ")
t_load <- system.time({
  model <- onnx_load(ONNX_PATH,
                     device       = "vulkan",
                     input_shapes = list(input = c(1L, 1L, 224L, 224L)))
})[3]
cat(sprintf("%.2f s\n\n", t_load))

# ---- Device/scheduler diagnostics ----
di <- onnx_device_info(model)
cat(sprintf("Backends : %s\n", paste(di$backends, collapse = ", ")))
cat(sprintf("Graph    : %d nodes, %d splits\n", di$n_nodes, di$n_splits))
cat(sprintf("Ops      : GPU=%d  CPU-only=%d\n", di$gpu_ops, di$cpu_ops))
if (di$cpu_ops > 0L) {
  ops_str <- paste(sprintf("%s(%d)", names(di$cpu_only_ops), di$cpu_only_ops),
                   collapse = ", ")
  cat(sprintf("CPU-only : %s\n", ops_str))
  cat("*** CPU-only ops cause PCIe round-trips and are likely the bottleneck ***\n")
}
cat("\n")

# ---- Profile each size ----
results <- list()

for (sz_name in names(SIZES)) {
  shape <- SIZES[[sz_name]]
  n_px  <- prod(shape)

  cat(sprintf("--- Input %s ---\n", sz_name))

  set.seed(42)
  inp <- list(input = runif(n_px))

  # Warmup
  cat(sprintf("  Warmup x%d ... ", N_WARMUP))
  t_warmup <- system.time(
    for (i in seq_len(N_WARMUP)) onnx_run(model, inp)
  )[3]
  cat(sprintf("%.3f s total\n", t_warmup))

  # Timed runs
  times <- numeric(N_RUNS)
  for (i in seq_len(N_RUNS)) {
    t0       <- proc.time()
    out      <- onnx_run(model, inp)
    times[i] <- (proc.time() - t0)[3]
  }

  mean_ms <- mean(times) * 1e3
  min_ms  <- min(times)  * 1e3
  max_ms  <- max(times)  * 1e3
  sd_ms   <- sd(times)   * 1e3
  fps     <- 1 / mean(times)

  cat(sprintf("  mean=%.1f ms  min=%.1f ms  max=%.1f ms  sd=%.1f ms  FPS=%.2f\n",
              mean_ms, min_ms, max_ms, sd_ms, fps))

  # Output shape
  out_shape <- dim(out[[1]])
  if (is.null(out_shape)) out_shape <- length(out[[1]])
  cat(sprintf("  Output shape: [%s]\n\n", paste(out_shape, collapse="x")))

  results[[sz_name]] <- list(mean_ms = mean_ms, min_ms = min_ms,
                              max_ms = max_ms, sd_ms = sd_ms, fps = fps)
}

# ---- Summary table ----
cat("=== Summary ===\n\n")
cat(sprintf("%-10s %10s %10s %10s %10s %8s\n",
            "Size", "mean(ms)", "min(ms)", "max(ms)", "sd(ms)", "FPS"))
cat(strrep("-", 62), "\n")
for (sz in names(results)) {
  r <- results[[sz]]
  cat(sprintf("%-10s %10.1f %10.1f %10.1f %10.1f %8.2f\n",
              sz, r$mean_ms, r$min_ms, r$max_ms, r$sd_ms, r$fps))
}

cat("\n")
mem_after <- ggml_vulkan_device_memory(0)
cat(sprintf("VRAM after: %.1f GB free (used: %.1f MB)\n",
            mem_after$free / 1e9,
            (gpu_mem$free - mem_after$free) / 1e6))

# ---- Diagnosis ----
cat("\n=== Diagnosis ===\n")
if (di$n_splits > 1L) {
  cat(sprintf("Graph has %d splits — scheduler is bouncing between GPU and CPU.\n",
              di$n_splits))
  cat("Each split = PCIe transfer. This is the primary GPU slowdown cause.\n")
} else {
  cat("Graph runs in a single split — no PCIe round-trips.\n")
}
if (di$cpu_ops > 0L) {
  cat(sprintf("%d CPU-only op(s) force CPU fallback mid-graph.\n", di$cpu_ops))
}
