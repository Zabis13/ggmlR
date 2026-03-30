#!/usr/bin/env Rscript
# ============================================================================
# GPU Stress Test: 100K samples throughput, memory monitoring, stability
# ============================================================================
#
# Light models: 100K samples, heavy models: 10K samples.
# Measures: throughput (samples/sec), per-batch latency, VRAM before/after/peak,
# latency stability (no degradation), memory leaks.
#
# Output: console summary table + CSV file.
# ============================================================================

library(ggmlR)

cat("==============================================================\n")
cat("  GPU Stress Test: 100K samples throughput\n")
cat("==============================================================\n\n")

# --- Parameters ---
TOTAL_LIGHT    <- 1000L  # light models (MNIST, EmotionFerPlus)
TOTAL_HEAVY    <- 1000L  # heavy models (SqueezeNet, BERT, Inception)
MONITOR_EVERY  <- 100L    # VRAM snapshot interval (every N batches)
N_WARMUP       <- 5L       # warmup runs before measurement
CSV_FILE       <- file.path(tempdir(), "stress_test_results.csv")

# API mode: run inference through HTTP (plumber) instead of direct onnx_run.
# Set to TRUE to test production-like latency with HTTP overhead.
# Requires: plumber, httr2, jsonlite (not ggmlR dependencies).
USE_API        <- TRUE
API_PORT       <- 9090L    # first worker port (others: +1, +2, ...)
API_WORKERS    <- 8L       # number of plumber server processes
API_BINARY     <- TRUE     # use raw bytes instead of JSON (much faster for large inputs)

# --- Model directory ---
ONNX_DIR <- "/mnt/Data2/DS_projects/ONNX models-main"

# --- Model registry ---
# batch_size: samples per onnx_run call
# input_shape: shape for a single sample (batch dim = first element)
models <- list(
  list(
    name        = "MNIST",
    file        = "mnist-8.onnx",
    input_name  = "Input3",
    input_shape = c(1L, 1L, 28L, 28L),
    batch_size  = 100L,
    total       = TOTAL_LIGHT,
    description = "CNN 28x28, 10 digits"
  ),
  list(
    name        = "EmotionFerPlus",
    file        = "emotion-ferplus-8.onnx",
    input_name  = "Input3",
    input_shape = c(1L, 1L, 64L, 64L),
    batch_size  = 50L,
    total       = TOTAL_LIGHT,
    description = "CNN 64x64, 8 emotions"
  ),
  list(
    name        = "SqueezeNet",
    file        = "squeezenet1.0-8.onnx",
    input_name  = "data_0",
    input_shape = c(1L, 3L, 224L, 224L),
    batch_size  = 10L,
    total       = TOTAL_HEAVY,
    description = "Lightweight CNN 224x224, 1000 classes"
  ),
  list(
    name        = "BERT",
    file        = "bert_Opset17.onnx",
    input_name  = "input_ids",
    input_shape = c(1L, 128L),
    batch_size  = 1L,
    total       = TOTAL_HEAVY,
    extra_inputs = list(attention_mask = c(1L, 128L)),
    description = "BERT base, seq_len=128"
  ),
  list(
    name        = "Inception V3",
    file        = "adv_inception_v3_Opset17.onnx",
    input_name  = "x",
    input_shape = c(1L, 3L, 299L, 299L),
    batch_size  = 1L,
    total       = TOTAL_HEAVY,
    description = "GoogLeNet v3, 299x299 RGB"
  ),
  list(
    name        = "SuperResolution",
    file        = "super-resolution-10.onnx",
    input_name  = "input",
    input_shape = c(1L, 1L, 224L, 224L),
    batch_size  = 10L,
    total       = TOTAL_HEAVY,
    description = "PyTorch SR, 224x224 grayscale, 3x upscale"
  ),
  list(
    name        = "BAT-ResNeXt26ts",
    file        = "bat_resnext26ts_Opset18.onnx",
    input_name  = "x",
    input_shape = c(1L, 3L, 256L, 256L),
    batch_size  = 1L,
    total       = TOTAL_HEAVY,
    description = "BAT-ResNeXt26ts, 256x256 RGB, 1000 classes"
  ),
  list(
    name        = "GPT-NeoX",
    file        = "gptneox_Opset18.onnx",
    input_name  = "input_ids",
    input_shape = c(1L, 128L),
    batch_size  = 1L,
    total       = TOTAL_HEAVY,
    extra_inputs = list(attention_mask = c(1L, 128L)),
    description = "GPT-NeoX, seq_len=128, causal LM"
  ),
  list(
    name        = "botnet26t",
    file        = "botnet26t_256_Opset16.onnx",
    input_name  = "x",
    input_shape = c(1L, 3L, 256L, 256L),
    batch_size  = 1L,
    total       = TOTAL_HEAVY,
    description = "BoTNet26t, 256x256 RGB, 1000 classes"
  )
)

# --- GPU check ---
if (!ggml_vulkan_available()) {
  stop("Vulkan not available — GPU stress test requires Vulkan backend")
}

gpu_name <- ggml_vulkan_device_description(0)
gpu_mem  <- ggml_vulkan_device_memory(0)
cat(sprintf("GPU: %s (%.1f / %.1f GB)\n", gpu_name,
            gpu_mem$free / 1e9, gpu_mem$total / 1e9))
cat(sprintf("Total samples: %s (light) / %s (heavy)\n",
            format(TOTAL_LIGHT, big.mark = ","),
            format(TOTAL_HEAVY, big.mark = ",")))
cat(sprintf("Output: %s\n\n", CSV_FILE))

# --- Stress test a single model ---
stress_one <- function(m) {
  onnx_path <- file.path(ONNX_DIR, m$file)
  if (!file.exists(onnx_path)) return(NULL)

  batch_size   <- m$batch_size
  n_batches    <- ceiling(m$total / batch_size)
  total_actual <- n_batches * batch_size

  cat(sprintf("  Batches: %s x %d = %s samples\n",
              format(n_batches, big.mark = ","), batch_size,
              format(total_actual, big.mark = ",")))

  # Prepare input shapes
  shapes <- list()
  shapes[[m$input_name]] <- m$input_shape
  if (!is.null(m$extra_inputs)) {
    for (nm in names(m$extra_inputs))
      shapes[[nm]] <- m$extra_inputs[[nm]]
  }

  # VRAM before loading
  mem_before <- ggml_vulkan_device_memory(0)

  # Load model
  t_load <- proc.time()
  model <- onnx_load(onnx_path, device = "vulkan", input_shapes = shapes)
  load_sec <- (proc.time() - t_load)[3]
  cat(sprintf("  Load: %.2f sec\n", load_sec))

  # VRAM after loading
  mem_loaded <- ggml_vulkan_device_memory(0)
  vram_model_mb <- (mem_before$free - mem_loaded$free) / 1e6

  # Input data (single batch, reused)
  set.seed(42)
  input_data <- runif(prod(m$input_shape))
  inputs <- list()
  inputs[[m$input_name]] <- input_data
  if (!is.null(m$extra_inputs)) {
    for (nm in names(m$extra_inputs))
      inputs[[nm]] <- rep(1, prod(m$extra_inputs[[nm]]))
  }

  # Warmup
  for (i in seq_len(N_WARMUP)) onnx_run(model, inputs)

  # VRAM after warmup (peak baseline)
  mem_warmup <- ggml_vulkan_device_memory(0)
  vram_peak_free <- mem_warmup$free  # track min free = max usage

  # Main loop
  batch_times <- numeric(n_batches)
  vram_samples <- numeric(0)  # snapshots for leak analysis

  t_total <- proc.time()
  for (i in seq_len(n_batches)) {
    t0 <- proc.time()
    out <- onnx_run(model, inputs)
    batch_times[i] <- (proc.time() - t0)[3]

    # VRAM monitoring
    if (i %% MONITOR_EVERY == 0 || i == n_batches) {
      mem_now <- ggml_vulkan_device_memory(0)
      if (mem_now$free < vram_peak_free) vram_peak_free <- mem_now$free
      vram_samples <- c(vram_samples, mem_now$free)

      # Progress
      pct <- round(100 * i / n_batches)
      elapsed <- (proc.time() - t_total)[3]
      rate <- (i * batch_size) / elapsed
      cat(sprintf("\r  Progress: %3d%% | %s samples | %.0f samples/sec | VRAM free: %.0f MB",
                  pct, format(i * batch_size, big.mark = ","), rate,
                  mem_now$free / 1e6))
    }
  }
  total_sec <- (proc.time() - t_total)[3]
  cat("\n")

  # VRAM after test
  mem_after <- ggml_vulkan_device_memory(0)

  # Latency stability analysis
  # Compare first 10% vs last 10% of batches
  n10 <- max(1L, as.integer(n_batches * 0.1))
  lat_first <- mean(batch_times[seq_len(n10)]) * 1000
  lat_last  <- mean(batch_times[seq(n_batches - n10 + 1, n_batches)]) * 1000
  lat_drift_pct <- (lat_last - lat_first) / lat_first * 100

  # Memory leak detection:
  #   drift > 0 means free decreased (memory consumed over time)
  #   drift <= 0 means free grew (driver realloc) — never a leak
  #   Leak = drift >= 10 MB AND >50% of consecutive snapshots show declining free
  if (length(vram_samples) >= 2) {
    vram_drift_mb <- (vram_samples[1] - vram_samples[length(vram_samples)]) / 1e6
    diffs <- diff(vram_samples)  # negative diff = free decreased
    n_declining <- sum(diffs < 0)
    monotonic <- n_declining > length(diffs) / 2
  } else {
    vram_drift_mb <- 0
    monotonic <- FALSE
  }

  # Cleanup
  rm(model, out); gc(verbose = FALSE)
  mem_freed <- ggml_vulkan_device_memory(0)

  result <- list(
    name          = m$name,
    total_samples = total_actual,
    batch_size    = batch_size,
    n_batches     = n_batches,
    load_sec      = load_sec,
    total_sec     = total_sec,
    throughput    = total_actual / total_sec,
    lat_mean_ms   = mean(batch_times) * 1000,
    lat_median_ms = median(batch_times) * 1000,
    lat_p99_ms    = quantile(batch_times, 0.99) * 1000,
    lat_min_ms    = min(batch_times) * 1000,
    lat_max_ms    = max(batch_times) * 1000,
    lat_first_ms  = lat_first,
    lat_last_ms   = lat_last,
    lat_drift_pct = lat_drift_pct,
    vram_model_mb = vram_model_mb,
    vram_peak_mb  = (mem_before$free - vram_peak_free) / 1e6,
    vram_after_mb = (mem_before$free - mem_after$free) / 1e6,
    vram_freed_mb = (mem_before$free - mem_freed$free) / 1e6,
    vram_drift_mb = vram_drift_mb,
    vram_leak     = vram_drift_mb >= 10.0 && monotonic
  )

  # Report
  cat(sprintf("  Throughput:  %s samples/sec\n", format(round(result$throughput), big.mark = ",")))
  cat(sprintf("  Latency:     mean=%.2f ms, median=%.2f ms, p99=%.2f ms\n",
              result$lat_mean_ms, result$lat_median_ms, result$lat_p99_ms))
  cat(sprintf("  Lat drift:   first 10%%=%.2f ms, last 10%%=%.2f ms (%+.1f%%)\n",
              lat_first, lat_last, lat_drift_pct))
  cat(sprintf("  VRAM:        model=%.0f MB, peak=%.0f MB, after=%.0f MB, freed=%.0f MB\n",
              result$vram_model_mb, result$vram_peak_mb,
              result$vram_after_mb, result$vram_freed_mb))
  if (result$vram_leak) {
    cat(sprintf("  WARNING:     VRAM drift %.1f MB — possible memory leak!\n", vram_drift_mb))
  } else {
    cat("  VRAM leak:   none detected\n")
  }

  result
}

# --- API mode: stress test a single model via HTTP (multi-worker) ---
stress_one_api <- function(m) {
  onnx_path <- file.path(ONNX_DIR, m$file)
  if (!file.exists(onnx_path)) return(NULL)

  n_requests   <- m$total
  batch_size   <- 1L

  # Auto-limit workers by VRAM: estimate per-model VRAM from file size * 3
  # (weights + intermediate buffers), keep at least 2 GB free
  model_vram_est <- file.size(onnx_path) / 1e6 * 3  # rough estimate in MB
  vram_free <- ggml_vulkan_device_memory(0)$free / 1e6
  vram_reserve <- 2000  # keep 2 GB free
  max_by_vram <- max(1L, as.integer((vram_free - vram_reserve) / max(model_vram_est, 1)))
  n_workers <- min(API_WORKERS, max_by_vram)

  cat(sprintf("  Requests: %s via %d workers (HTTP)\n",
              format(n_requests, big.mark = ","), n_workers))
  if (n_workers < API_WORKERS) {
    cat(sprintf("  (reduced from %d workers — VRAM limit: ~%.0f MB/model, %.0f MB free)\n",
                API_WORKERS, model_vram_est, vram_free))
  }

  # Build server script lines (shared across workers, port is substituted)
  extra_shapes <- ""
  extra_inputs_code <- ""      # for predict handler (uses 'inputs')
  extra_warmup_code <- ""      # for warmup (uses 'wi')
  if (!is.null(m$extra_inputs)) {
    for (nm in names(m$extra_inputs)) {
      sh <- paste(m$extra_inputs[[nm]], collapse = "L, ")
      extra_shapes <- paste0(extra_shapes,
        sprintf('shapes[["%s"]] <- c(%sL)\n', nm, sh))
      extra_inputs_code <- paste0(extra_inputs_code,
        sprintf('inputs[["%s"]] <- rep(1, prod(c(%sL)))\n', nm, sh))
      extra_warmup_code <- paste0(extra_warmup_code,
        sprintf('wi[["%s"]] <- rep(1, prod(c(%sL)))\n', nm, sh))
    }
  }
  sh_main <- paste(m$input_shape, collapse = "L, ")

  make_server_script <- function(port) {
    tf <- tempfile(fileext = ".R")

    if (API_BINARY) {
      # Binary mode: receive raw doubles, return raw "OK"
      predict_handler <- c(
        'pr$handle("POST", "/predict", function(req, res) {',
        '  raw_body <- req$bodyRaw',
        '  input_data <- readBin(raw_body, "double", n = prod(input_shape))',
        '  inputs <- list()',
        '  inputs[[input_name]] <- input_data',
        extra_inputs_code,
        '  out <- onnx_run(model, inputs)',
        '  res$setHeader("Content-Type", "application/octet-stream")',
        '  res$body <- writeBin(1.0, raw())',
        '  res',
        '})'
      )
    } else {
      # JSON mode
      predict_handler <- c(
        'pr$handle("POST", "/predict", function(req, res) {',
        '  body <- jsonlite::fromJSON(req$postBody)',
        '  inputs <- list()',
        '  inputs[[input_name]] <- as.numeric(body$data)',
        extra_inputs_code,
        '  out <- onnx_run(model, inputs)',
        '  list(ok = TRUE)',
        '})'
      )
    }

    writeLines(con = tf, c(
      'library(ggmlR)',
      'library(plumber)',
      if (!API_BINARY) 'library(jsonlite)' else NULL,
      sprintf('onnx_path <- "%s"', onnx_path),
      sprintf('input_name <- "%s"', m$input_name),
      sprintf('input_shape <- c(%sL)', sh_main),
      'shapes <- list()',
      'shapes[[input_name]] <- input_shape',
      extra_shapes,
      'device <- if (ggml_vulkan_available()) "vulkan" else "cpu"',
      'model <- onnx_load(onnx_path, device = device, input_shapes = shapes)',
      'set.seed(42)',
      'wi <- list(); wi[[input_name]] <- runif(prod(input_shape))',
      extra_warmup_code,
      'onnx_run(model, wi)',
      'pr <- Plumber$new()',
      'pr$handle("GET", "/health", function(req, res) list(status = "ok"))',
      predict_handler,
      sprintf('pr$run(host = "127.0.0.1", port = %dL, quiet = TRUE)', port)
    ))
    tf
  }

  # Start N worker servers
  ports <- API_PORT + seq_len(n_workers) - 1L
  server_scripts <- character(n_workers)
  server_procs   <- integer(n_workers)

  cleanup_servers <- function() {
    for (j in seq_len(n_workers)) {
      try(tools::pskill(server_procs[j], signal = 15L), silent = TRUE)
    }
    Sys.sleep(0.5)
    for (j in seq_len(n_workers)) {
      try(tools::pskill(server_procs[j], signal = 9L), silent = TRUE)
      try(unlink(server_scripts[j]), silent = TRUE)
    }
  }
  on.exit(cleanup_servers(), add = TRUE)

  cat(sprintf("  Starting %d API workers (ports %d-%d)...",
              n_workers, min(ports), max(ports)))

  for (j in seq_len(n_workers)) {
    server_scripts[j] <- make_server_script(ports[j])
    server_log <- tempfile(fileext = sprintf("_w%d.log", j))
    server_procs[j]   <- sys::exec_background("Rscript", server_scripts[j],
                                               std_out = server_log, std_err = server_log)
  }

  # Wait for all workers to be ready
  all_ready <- TRUE
  for (j in seq_len(n_workers)) {
    url <- sprintf("http://127.0.0.1:%d/health", ports[j])
    ready <- FALSE
    for (attempt in 1:120) {
      Sys.sleep(0.5)
      ready <- tryCatch({
        httr2::request(url) |> httr2::req_perform()
        TRUE
      }, error = function(e) FALSE)
      if (ready) break
    }
    if (!ready) {
      cat(sprintf(" worker %d FAILED", j))
      all_ready <- FALSE
    }
  }
  if (!all_ready) {
    cat(" (timeout)\n")
    # Show first failed worker's log for diagnostics
    for (j in seq_len(n_workers)) {
      log_file <- tempdir() |> list.files(pattern = sprintf("_w%d\\.log$", j),
                                           full.names = TRUE) |> tail(1)
      if (length(log_file) == 1 && file.exists(log_file)) {
        lines <- readLines(log_file, warn = FALSE)
        if (length(lines) > 0) {
          cat(sprintf("  Worker %d log (last 5 lines):\n", j))
          cat(paste("    ", tail(lines, 5)), sep = "\n")
          cat("\n")
          break
        }
      }
    }
    return(NULL)
  }
  cat(" OK\n")

  # Prepare request body
  set.seed(42)
  input_data <- runif(prod(m$input_shape))
  if (API_BINARY) {
    body_raw <- writeBin(input_data, raw())
    content_type <- "application/octet-stream"
  } else {
    body_raw <- charToRaw(jsonlite::toJSON(list(data = input_data), auto_unbox = TRUE))
    content_type <- "application/json"
  }

  # Build URLs for round-robin
  urls <- sprintf("http://127.0.0.1:%d/predict", ports)

  # VRAM before
  mem_before <- ggml_vulkan_device_memory(0)
  vram_peak_free <- mem_before$free  # track min free = max usage

  cat(sprintf("  Protocol: %s, payload: %.1f KB\n",
              if (API_BINARY) "binary" else "JSON",
              length(body_raw) / 1024))

  # Main loop: round-robin across workers
  batch_times <- numeric(n_requests)
  vram_samples <- numeric(0)

  t_total <- proc.time()
  for (i in seq_len(n_requests)) {
    wk <- ((i - 1L) %% n_workers) + 1L
    t0 <- proc.time()
    httr2::request(urls[wk]) |>
      httr2::req_body_raw(body_raw, type = content_type) |>
      httr2::req_perform()
    batch_times[i] <- (proc.time() - t0)[3]

    # VRAM monitoring
    if (i %% MONITOR_EVERY == 0 || i == n_requests) {
      mem_now <- ggml_vulkan_device_memory(0)
      if (mem_now$free < vram_peak_free) vram_peak_free <- mem_now$free
      vram_samples <- c(vram_samples, mem_now$free)

      pct <- round(100 * i / n_requests)
      elapsed <- (proc.time() - t_total)[3]
      rate <- i / elapsed
      cat(sprintf("\r  Progress: %3d%% | %s req | %.0f req/sec | VRAM free: %.0f MB",
                  pct, format(i, big.mark = ","), rate, mem_now$free / 1e6))
    }
  }
  total_sec <- (proc.time() - t_total)[3]
  cat("\n")

  mem_after <- ggml_vulkan_device_memory(0)

  # Latency analysis
  n10 <- max(1L, as.integer(n_requests * 0.1))
  lat_first <- mean(batch_times[seq_len(n10)]) * 1000
  lat_last  <- mean(batch_times[seq(n_requests - n10 + 1, n_requests)]) * 1000
  lat_drift_pct <- (lat_last - lat_first) / lat_first * 100

  # Leak detection
  if (length(vram_samples) >= 2) {
    vram_drift_mb <- (vram_samples[1] - vram_samples[length(vram_samples)]) / 1e6
    diffs <- diff(vram_samples)
    n_declining <- sum(diffs < 0)
    monotonic <- n_declining > length(diffs) / 2
  } else {
    vram_drift_mb <- 0
    monotonic <- FALSE
  }

  result <- list(
    name          = m$name,
    total_samples = n_requests,
    batch_size    = batch_size,
    n_batches     = n_requests,
    load_sec      = NA,
    total_sec     = total_sec,
    throughput    = n_requests / total_sec,
    lat_mean_ms   = mean(batch_times) * 1000,
    lat_median_ms = median(batch_times) * 1000,
    lat_p99_ms    = quantile(batch_times, 0.99) * 1000,
    lat_min_ms    = min(batch_times) * 1000,
    lat_max_ms    = max(batch_times) * 1000,
    lat_first_ms  = lat_first,
    lat_last_ms   = lat_last,
    lat_drift_pct = lat_drift_pct,
    vram_model_mb = NA,
    vram_peak_mb  = (mem_before$free - vram_peak_free) / 1e6,
    vram_after_mb = (mem_before$free - mem_after$free) / 1e6,
    vram_freed_mb = NA,
    vram_drift_mb = vram_drift_mb,
    vram_leak     = vram_drift_mb >= 10.0 && monotonic
  )

  cat(sprintf("  Workers:     %d\n", n_workers))
  cat(sprintf("  Throughput:  %s req/sec\n", format(round(result$throughput), big.mark = ",")))
  cat(sprintf("  Latency:     mean=%.2f ms, median=%.2f ms, p99=%.2f ms\n",
              result$lat_mean_ms, result$lat_median_ms, result$lat_p99_ms))
  cat(sprintf("  Lat drift:   first 10%%=%.2f ms, last 10%%=%.2f ms (%+.1f%%)\n",
              lat_first, lat_last, lat_drift_pct))
  if (result$vram_leak) {
    cat(sprintf("  WARNING:     VRAM drift %.1f MB — possible memory leak!\n", vram_drift_mb))
  } else {
    cat("  VRAM leak:   none detected\n")
  }

  result
}

# --- Main loop ---
all_results <- list()

if (USE_API) {
  cat(sprintf("Mode: API (HTTP via plumber, %s, %d workers)\n\n",
              if (API_BINARY) "binary" else "JSON", API_WORKERS))
  library(httr2)
  library(jsonlite)
} else {
  cat("Mode: direct (in-process onnx_run)\n\n")
}

for (m in models) {
  onnx_path <- file.path(ONNX_DIR, m$file)
  if (!file.exists(onnx_path)) {
    cat(sprintf("SKIP: %s — file not found\n\n", m$name))
    next
  }

  size_mb <- file.size(onnx_path) / 1024 / 1024
  cat("==============================================================\n")
  cat(sprintf("  %s  (%.1f MB) — %s\n", m$name, size_mb, m$description))
  cat(sprintf("  Input: %s [%s], batch=%d\n", m$input_name,
              paste(m$input_shape, collapse = "x"), m$batch_size))
  cat("==============================================================\n")

  res <- tryCatch(
    if (USE_API) stress_one_api(m) else stress_one(m),
    error = function(e) { cat("  ERROR:", e$message, "\n"); NULL }
  )

  if (!is.null(res)) all_results[[length(all_results) + 1]] <- res
  cat("\n")
}

# --- Summary table ---
cat("==============================================================\n")
cat("  Summary\n")
cat("==============================================================\n\n")

cat(sprintf("%-16s %8s %10s %10s %8s %8s %8s %6s\n",
            "Model", "Batch", "Samples/s", "Mean(ms)", "P99(ms)",
            "VRAM MB", "Drift%", "Leak"))
cat(paste(rep("-", 82), collapse = ""), "\n")

for (r in all_results) {
  cat(sprintf("%-16s %8d %10s %10.2f %8.2f %8.0f %7.1f%% %6s\n",
              r$name, r$batch_size,
              format(round(r$throughput), big.mark = ","),
              r$lat_mean_ms, r$lat_p99_ms,
              r$vram_peak_mb, r$lat_drift_pct,
              if (r$vram_leak) "YES" else "no"))
}

# --- CSV ---
csv_df <- do.call(rbind, lapply(all_results, function(r) {
  data.frame(
    model          = r$name,
    total_samples  = r$total_samples,
    batch_size     = r$batch_size,
    n_batches      = r$n_batches,
    load_sec       = round(r$load_sec, 3),
    total_sec      = round(r$total_sec, 3),
    throughput     = round(r$throughput, 1),
    lat_mean_ms    = round(r$lat_mean_ms, 3),
    lat_median_ms  = round(r$lat_median_ms, 3),
    lat_p99_ms     = round(r$lat_p99_ms, 3),
    lat_min_ms     = round(r$lat_min_ms, 3),
    lat_max_ms     = round(r$lat_max_ms, 3),
    lat_first_ms   = round(r$lat_first_ms, 3),
    lat_last_ms    = round(r$lat_last_ms, 3),
    lat_drift_pct  = round(r$lat_drift_pct, 1),
    vram_model_mb  = round(r$vram_model_mb, 1),
    vram_peak_mb   = round(r$vram_peak_mb, 1),
    vram_after_mb  = round(r$vram_after_mb, 1),
    vram_freed_mb  = round(r$vram_freed_mb, 1),
    vram_drift_mb  = round(r$vram_drift_mb, 1),
    vram_leak      = r$vram_leak,
    gpu            = gpu_name,
    stringsAsFactors = FALSE
  )
}))

write.csv(csv_df, CSV_FILE, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", CSV_FILE))

cat("\n==============================================================\n")
cat("  Stress test complete\n")
cat("==============================================================\n")
