#!/usr/bin/env Rscript
# ============================================================================
# ONNX Inference API Server (for stress testing only)
# ============================================================================
# Usage:
#   Rscript api_server.R [model] [port]
#   Rscript api_server.R mnist 8080
#   Rscript api_server.R bert 8080
#
# Requires: plumber, jsonlite (not ggmlR dependencies, install separately)
# ============================================================================

library(ggmlR)
library(plumber)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
model_name <- if (length(args) >= 1) args[1] else "mnist"
port        <- if (length(args) >= 2) as.integer(args[2]) else 8080L

ONNX_DIR <- "/mnt/Data2/DS_projects/ONNX models-main"

# --- Model registry ---
model_registry <- list(
  mnist = list(
    file        = "mnist-8.onnx",
    input_name  = "Input3",
    input_shape = c(1L, 1L, 28L, 28L)
  ),
  emotion = list(
    file        = "emotion-ferplus-8.onnx",
    input_name  = "Input3",
    input_shape = c(1L, 1L, 64L, 64L)
  ),
  squeezenet = list(
    file        = "squeezenet1.0-8.onnx",
    input_name  = "data_0",
    input_shape = c(1L, 3L, 224L, 224L)
  ),
  bert = list(
    file         = "bert_Opset17.onnx",
    input_name   = "input_ids",
    input_shape  = c(1L, 128L),
    extra_inputs = list(attention_mask = c(1L, 128L))
  ),
  inception = list(
    file        = "adv_inception_v3_Opset17.onnx",
    input_name  = "x",
    input_shape = c(1L, 3L, 299L, 299L)
  )
)

spec <- model_registry[[model_name]]
if (is.null(spec)) {
  stop(sprintf("Unknown model '%s'. Available: %s",
               model_name, paste(names(model_registry), collapse = ", ")))
}

onnx_path <- file.path(ONNX_DIR, spec$file)
if (!file.exists(onnx_path)) stop(sprintf("Model file not found: %s", onnx_path))

# --- Load model ---
cat(sprintf("Loading %s...\n", model_name))
shapes <- list()
shapes[[spec$input_name]] <- spec$input_shape
if (!is.null(spec$extra_inputs)) {
  for (nm in names(spec$extra_inputs))
    shapes[[nm]] <- spec$extra_inputs[[nm]]
}

device <- if (ggml_vulkan_available()) "vulkan" else "cpu"
model <- onnx_load(onnx_path, device = device, input_shapes = shapes)

# Warmup
set.seed(42)
warmup_data <- runif(prod(spec$input_shape))
warmup_inputs <- list()
warmup_inputs[[spec$input_name]] <- warmup_data
if (!is.null(spec$extra_inputs)) {
  for (nm in names(spec$extra_inputs))
    warmup_inputs[[nm]] <- rep(1, prod(spec$extra_inputs[[nm]]))
}
onnx_run(model, warmup_inputs)

cat(sprintf("Model ready on %s, starting server on port %d\n", device, port))
cat(sprintf("Input shape: %s\n", paste(spec$input_shape, collapse = "x")))
cat(sprintf("Endpoints:\n"))
cat(sprintf("  POST /predict  — run inference (JSON body: {\"data\": [...]})\n"))
cat(sprintf("  GET  /health   — server status\n"))
cat(sprintf("  GET  /info     — model info\n\n"))

# --- API definition ---
pr <- Plumber$new()

pr$handle("GET", "/health", function(req, res) {
  list(status = "ok", model = model_name, device = device)
})

pr$handle("GET", "/info", function(req, res) {
  mem <- if (ggml_vulkan_available()) ggml_vulkan_device_memory(0) else NULL
  list(
    model       = model_name,
    device      = device,
    input_name  = spec$input_name,
    input_shape = spec$input_shape,
    input_size  = prod(spec$input_shape),
    vram_free   = if (!is.null(mem)) round(mem$free / 1e6) else NA,
    vram_total  = if (!is.null(mem)) round(mem$total / 1e6) else NA
  )
})

pr$handle("POST", "/predict", function(req, res) {
  body <- tryCatch(jsonlite::fromJSON(req$postBody), error = function(e) NULL)

  if (is.null(body) || is.null(body$data)) {
    res$status <- 400L
    return(list(error = "JSON body must contain 'data' array"))
  }

  input_data <- as.numeric(body$data)
  expected <- prod(spec$input_shape)
  if (length(input_data) != expected) {
    res$status <- 400L
    return(list(error = sprintf("Expected %d values, got %d", expected, length(input_data))))
  }

  inputs <- list()
  inputs[[spec$input_name]] <- input_data
  if (!is.null(spec$extra_inputs)) {
    for (nm in names(spec$extra_inputs))
      inputs[[nm]] <- rep(1, prod(spec$extra_inputs[[nm]]))
  }

  t0 <- proc.time()
  out <- onnx_run(model, inputs)
  elapsed_ms <- (proc.time() - t0)[3] * 1000

  list(
    output     = out[[1]],
    elapsed_ms = round(elapsed_ms, 2)
  )
})

pr$run(host = "0.0.0.0", port = port)
