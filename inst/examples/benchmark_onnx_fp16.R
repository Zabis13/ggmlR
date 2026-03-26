#!/usr/bin/env Rscript
# ============================================================================
# ONNX Benchmark: GPU F32 vs F16 weight precision comparison
# ============================================================================

library(ggmlR)

cat("==============================================================\n")
cat("  ONNX Benchmark: GPU F32 vs F16 (Vulkan)\n")
cat("==============================================================\n\n")

# --- Каталог с моделями ---
ONNX_DIR <- "/mnt/Data2/DS_projects/ONNX models-main"

# --- Реестр моделей для бенчмарка ---
models <- list(
  list(
    name        = "Inception V3",
    file        = "adv_inception_v3_Opset17.onnx",
    input_name  = "x",
    input_shape = c(1L, 3L, 299L, 299L),
    description = "GoogLeNet v3, 299x299 RGB, 1000 classes"
  ),
  list(
    name        = "MNIST",
    file        = "mnist-8.onnx",
    input_name  = "Input3",
    input_shape = c(1L, 1L, 28L, 28L),
    description = "CNTK CNN, 28x28 grayscale, 10 digits"
  ),
  list(
    name        = "SqueezeNet 1.0",
    file        = "squeezenet1.0-8.onnx",
    input_name  = "data_0",
    input_shape = c(1L, 3L, 224L, 224L),
    description = "Lightweight CNN, 224x224 RGB, 1000 classes"
  ),
  list(
    name        = "SuperResolution",
    file        = "super-resolution-10.onnx",
    input_name  = "input",
    input_shape = c(1L, 1L, 224L, 224L),
    description = "PyTorch SR, 224x224 grayscale, 3x upscale"
  ),
  list(
    name        = "EmotionFerPlus",
    file        = "emotion-ferplus-8.onnx",
    input_name  = "Input3",
    input_shape = c(1L, 1L, 64L, 64L),
    description = "CNTK CNN, 64x64 grayscale, 8 emotions"
  ),
  list(
    name        = "Inception V3 Op18",
    file        = "adv_inception_v3_Opset18.onnx",
    input_name  = "x",
    input_shape = c(1L, 3L, 299L, 299L),
    description = "GoogLeNet v3 Opset18, 299x299 RGB, 1000 classes"
  ),
  list(
    name        = "BAT-ResNeXt26ts",
    file        = "bat_resnext26ts_Opset18.onnx",
    input_name  = "x",
    input_shape = c(1L, 3L, 256L, 256L),
    description = "BAT-ResNeXt26ts, 256x256 RGB, 1000 classes"
  ),
  list(
    name        = "BERT (Opset17)",
    file        = "bert_Opset17.onnx",
    input_name  = "input_ids",
    input_shape = c(1L, 128L),
    extra_inputs = list(attention_mask = c(1L, 128L)),
    description = "BERT base, seq_len=128, token classification"
  ),
  list(
    name        = "GPT-NeoX",
    file        = "gptneox_Opset18.onnx",
    input_name  = "input_ids",
    input_shape = c(1L, 128L),
    extra_inputs = list(attention_mask = c(1L, 128L)),
    description = "GPT-NeoX, seq_len=128, causal LM"
  )
)

# --- Параметры ---
N_WARMUP <- 1L
N_RUNS   <- 3L

# --- Информация о системе ---
n_cores <- parallel::detectCores(logical = FALSE)
if (is.na(n_cores)) n_cores <- 1L
cat(sprintf("CPU cores: %d, threads: %d\n", n_cores, max(n_cores - 1L, 1L)))

vulkan_ok <- ggml_vulkan_available()
if (vulkan_ok) {
  gpu_name <- ggml_vulkan_device_description(0)
  gpu_mem  <- ggml_vulkan_device_memory(0)
  cat(sprintf("GPU: %s (%.1f / %.1f GB)\n", gpu_name,
              gpu_mem$free / 1e9, gpu_mem$total / 1e9))
} else {
  cat("GPU: Vulkan not available — nothing to compare\n")
  quit(status = 0)
}
cat(sprintf("Warmup: %d, Runs: %d\n\n", N_WARMUP, N_RUNS))

# --- Функция бенчмарка ---
bench_one <- function(onnx_path, input_name, input_shape, device,
                      input_data, n_warmup, n_runs,
                      extra_inputs = NULL, extra_data = NULL,
                      dtype = "f32") {
  t0 <- proc.time()
  shapes <- list()
  shapes[[input_name]] <- input_shape
  if (!is.null(extra_inputs)) {
    for (nm in names(extra_inputs))
      shapes[[nm]] <- extra_inputs[[nm]]
  }
  model <- onnx_load(onnx_path, device = device, input_shapes = shapes,
                     dtype = dtype)
  load_time <- (proc.time() - t0)[3]

  inputs <- list()
  inputs[[input_name]] <- input_data
  if (!is.null(extra_data)) {
    for (nm in names(extra_data))
      inputs[[nm]] <- extra_data[[nm]]
  }

  # Прогрев
  for (i in seq_len(n_warmup))
    out <- onnx_run(model, inputs)

  # Замеры
  times <- numeric(n_runs)
  for (i in seq_len(n_runs)) {
    t0 <- proc.time()
    out <- onnx_run(model, inputs)
    times[i] <- (proc.time() - t0)[3]
  }

  probs <- out[[1]]
  top5_idx <- order(probs, decreasing = TRUE)[1:5]

  rm(model, out); gc(verbose = FALSE)

  list(
    load_time = load_time,
    times     = times,
    mean_ms   = mean(times) * 1000,
    min_ms    = min(times) * 1000,
    max_ms    = max(times) * 1000,
    fps       = 1.0 / mean(times),
    top5      = top5_idx
  )
}

# --- Основной цикл ---
all_results <- list()

for (m in models) {
  onnx_path <- file.path(ONNX_DIR, m$file)
  if (!file.exists(onnx_path)) {
    cat(sprintf("SKIP: %s — file not found\n\n", m$name))
    next
  }

  size_mb <- file.size(onnx_path) / 1024 / 1024
  cat("==============================================================\n")
  cat(sprintf("  %s  (%.1f MB)\n", m$name, size_mb))
  cat(sprintf("  %s\n", m$description))
  cat("==============================================================\n")

  set.seed(42)
  input_data <- runif(prod(m$input_shape))
  extra_data <- NULL
  if (!is.null(m$extra_inputs)) {
    extra_data <- list()
    for (nm in names(m$extra_inputs))
      extra_data[[nm]] <- rep(1, prod(m$extra_inputs[[nm]]))
  }

  res <- list(name = m$name)

  # CPU (baseline)
  cat("  CPU     ... ")
  res$cpu <- tryCatch(
    bench_one(onnx_path, m$input_name, m$input_shape, "cpu",
              input_data, N_WARMUP, N_RUNS,
              extra_inputs = m$extra_inputs, extra_data = extra_data),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
  if (!is.null(res$cpu))
    cat(sprintf("%.1f ms\n", res$cpu$mean_ms))

  # GPU F32
  cat("  GPU F32 ... ")
  res$gpu <- tryCatch(
    bench_one(onnx_path, m$input_name, m$input_shape, "vulkan",
              input_data, N_WARMUP, N_RUNS,
              extra_inputs = m$extra_inputs, extra_data = extra_data,
              dtype = "f32"),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
  if (!is.null(res$gpu))
    cat(sprintf("%.1f ms\n", res$gpu$mean_ms))

  # GPU F16
  cat("  GPU F16 ... ")
  res$gpu_f16 <- tryCatch(
    bench_one(onnx_path, m$input_name, m$input_shape, "vulkan",
              input_data, N_WARMUP, N_RUNS,
              extra_inputs = m$extra_inputs, extra_data = extra_data,
              dtype = "f16"),
    error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
  )
  if (!is.null(res$gpu_f16))
    cat(sprintf("%.1f ms\n", res$gpu_f16$mean_ms))

  # Сравнение
  if (!is.null(res$gpu) && !is.null(res$gpu_f16)) {
    f16_gain <- res$gpu$mean_ms / res$gpu_f16$mean_ms
    cat(sprintf("  F16 vs F32: %.2fx\n", f16_gain))
  }
  if (!is.null(res$cpu) && !is.null(res$gpu_f16)) {
    match16 <- identical(res$cpu$top5, res$gpu_f16$top5)
    cat(sprintf("  Top-5 match (F16 vs CPU): %s\n", if (match16) "YES" else "NO"))
  }

  cat("\n")
  all_results[[length(all_results) + 1]] <- res
}

# --- Сводная таблица ---
cat("==============================================================\n")
cat("  Summary\n")
cat("==============================================================\n\n")

cat(sprintf("%-20s %10s %10s %10s %10s %10s %10s\n",
            "Model", "CPU(ms)", "GPU F32", "GPU F16", "F32 spd", "F16 spd", "F16/F32"))
cat(sprintf("%-20s %10s %10s %10s %10s %10s %10s\n",
            "--------------------", "--------", "--------", "--------",
            "--------", "--------", "--------"))

for (r in all_results) {
  cpu_ms    <- if (!is.null(r$cpu))     sprintf("%.1f", r$cpu$mean_ms) else "—"
  gpu_ms    <- if (!is.null(r$gpu))     sprintf("%.1f", r$gpu$mean_ms) else "—"
  gpu16_ms  <- if (!is.null(r$gpu_f16)) sprintf("%.1f", r$gpu_f16$mean_ms) else "—"

  spd32 <- if (!is.null(r$cpu) && !is.null(r$gpu))
    sprintf("%.1fx", r$cpu$mean_ms / r$gpu$mean_ms) else "—"
  spd16 <- if (!is.null(r$cpu) && !is.null(r$gpu_f16))
    sprintf("%.1fx", r$cpu$mean_ms / r$gpu_f16$mean_ms) else "—"
  f16_gain <- if (!is.null(r$gpu) && !is.null(r$gpu_f16))
    sprintf("%.2fx", r$gpu$mean_ms / r$gpu_f16$mean_ms) else "—"

  cat(sprintf("%-20s %10s %10s %10s %10s %10s %10s\n",
              r$name, cpu_ms, gpu_ms, gpu16_ms, spd32, spd16, f16_gain))
}

cat("\n==============================================================\n")
cat("  Benchmark complete\n")
cat("==============================================================\n")
