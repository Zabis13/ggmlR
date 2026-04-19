#!/usr/bin/env Rscript
# ============================================================================
# Test 5D CPY Vulkan path: run ONNX models that may produce 5D graphs,
# compare CPU vs Vulkan output numerically.
#
# Historically FAILing models that may benefit from 5D CPY fix:
#   cait_xs24_384   — 5D QKV split via Gather
#   xcit_tiny       — Expand to 5D
#   MaskRCNN-12     — mixed, includes 5D resize
#
# Baseline models (should not regress):
#   mnist-8, squeezenet1.0-8, inception_v3, bat_resnext26ts
# ============================================================================

suppressPackageStartupMessages(library(ggmlR))

ONNX_DIR <- "/mnt/Data2/DS_projects/ONNX models-main"

tests <- list(
  # name, file, input_name, input_shape, extra_inputs
  list(name = "MNIST",            file = "mnist-8.onnx",
       input_name = "Input3",     input_shape = c(1L, 1L, 28L, 28L),
       category = "baseline"),
  list(name = "SqueezeNet 1.0",   file = "squeezenet1.0-8.onnx",
       input_name = "data_0",     input_shape = c(1L, 3L, 224L, 224L),
       category = "baseline"),
  list(name = "Inception V3",     file = "adv_inception_v3_Opset17.onnx",
       input_name = "x",          input_shape = c(1L, 3L, 299L, 299L),
       category = "baseline"),
  list(name = "SuperResolution",  file = "super-resolution-10.onnx",
       input_name = "input",      input_shape = c(1L, 1L, 224L, 224L),
       category = "baseline"),
  list(name = "EmotionFerPlus",   file = "emotion-ferplus-8.onnx",
       input_name = "Input3",     input_shape = c(1L, 1L, 64L, 64L),
       category = "baseline"),
  list(name = "Inception V3 Op18", file = "adv_inception_v3_Opset18.onnx",
       input_name = "x",          input_shape = c(1L, 3L, 299L, 299L),
       category = "baseline"),
  list(name = "BERT",             file = "bert_Opset17.onnx",
       input_name = "input_ids",  input_shape = c(1L, 128L),
       extra_inputs = list(attention_mask = c(1L, 128L)),
       category = "baseline"),
  list(name = "GPT-NeoX",         file = "gptneox_Opset18.onnx",
       input_name = "input_ids",  input_shape = c(1L, 128L),
       extra_inputs = list(attention_mask = c(1L, 128L)),
       category = "baseline"),
  list(name = "BotNet26t",        file = "botnet26t_256_Opset16.onnx",
       input_name = "x",          input_shape = c(1L, 3L, 256L, 256L),
       category = "baseline"),
  list(name = "BAT-ResNeXt26ts",  file = "bat_resnext26ts_Opset18.onnx",
       input_name = "x",          input_shape = c(1L, 3L, 256L, 256L),
       category = "5d-candidate"),
  list(name = "cait_xs24_384",    file = "cait_xs24_384_Opset16.onnx",
       input_name = "x",          input_shape = c(1L, 3L, 384L, 384L),
       category = "5d-candidate"),
  list(name = "xcit_tiny",        file = "xcit_tiny_12_p8_224_Opset17.onnx",
       input_name = "x",          input_shape = c(1L, 3L, 224L, 224L),
       category = "5d-candidate")
)

THRESHOLD_REL <- 0.01   # 1% relative tolerance for max
THRESHOLD_ABS <- 1e-3

gen_input <- function(shape, name) {
  set.seed(42)
  if (grepl("input_ids", name, fixed = TRUE)) {
    as.numeric(sample.int(1000L, prod(shape), replace = TRUE))
  } else {
    runif(prod(shape))
  }
}

silent <- function(expr) {
  sink_file <- tempfile()
  con <- file(sink_file, open = "wt")
  sink(con, type = "output")
  sink(con, type = "message")
  on.exit({
    sink(type = "message"); sink(type = "output"); close(con); unlink(sink_file)
  }, add = TRUE)
  force(expr)
}

run_model <- function(path, device, input_name, input_shape, extra = NULL) {
  shapes <- list()
  shapes[[input_name]] <- input_shape
  if (!is.null(extra)) for (nm in names(extra)) shapes[[nm]] <- extra[[nm]]

  model <- silent(onnx_load(path, device = device, input_shapes = shapes))

  inputs <- list()
  inputs[[input_name]] <- gen_input(input_shape, input_name)
  if (!is.null(extra)) for (nm in names(extra)) inputs[[nm]] <- rep(1, prod(extra[[nm]]))

  out <- silent(onnx_run(model, inputs))
  rm(model); gc(verbose = FALSE)
  out
}

compare <- function(a, b) {
  if (length(a) != length(b)) return(list(ok = FALSE, reason = "shape mismatch",
                                           max_abs = NA_real_, max_rel = NA_real_,
                                           top5_match = NA))
  a <- as.numeric(a); b <- as.numeric(b)
  if (any(is.na(a)) || any(is.na(b)) || any(!is.finite(a)) || any(!is.finite(b))) {
    return(list(ok = FALSE, reason = "NaN/Inf in output",
                max_abs = NA_real_, max_rel = NA_real_, top5_match = NA))
  }
  diff <- abs(a - b)
  max_abs <- max(diff)
  scale   <- max(abs(a), abs(b), 1e-8)
  max_rel <- max_abs / scale

  top_a <- order(a, decreasing = TRUE)[1:min(5, length(a))]
  top_b <- order(b, decreasing = TRUE)[1:min(5, length(b))]
  top5_match <- identical(top_a, top_b)

  ok <- isTRUE((max_abs < THRESHOLD_ABS) || (max_rel < THRESHOLD_REL))
  list(ok = ok, max_abs = max_abs, max_rel = max_rel,
       top5_match = top5_match, top_a = top_a, top_b = top_b)
}

if (!ggml_vulkan_available()) {
  stop("Vulkan not available — this test requires GPU")
}

results <- list()

# Silently run all tests
for (t in tests) {
  path <- file.path(ONNX_DIR, t$file)
  if (!file.exists(path)) {
    results[[length(results) + 1]] <- list(name = t$name, category = t$category,
                                            status = "SKIP", reason = "file missing")
    next
  }

  cpu_out <- tryCatch(
    run_model(path, "cpu", t$input_name, t$input_shape, t$extra_inputs),
    error = function(e) conditionMessage(e))
  gpu_out <- tryCatch(
    run_model(path, "vulkan", t$input_name, t$input_shape, t$extra_inputs),
    error = function(e) conditionMessage(e))

  cpu_err <- is.character(cpu_out)
  gpu_err <- is.character(gpu_out)

  if (cpu_err || gpu_err) {
    results[[length(results) + 1]] <- list(
      name = t$name, category = t$category, status = "ERROR",
      reason = if (gpu_err) paste0("GPU: ", gpu_out) else paste0("CPU: ", cpu_out))
    next
  }

  cmp <- compare(cpu_out[[1]], gpu_out[[1]])
  status <- if (isTRUE(cmp$ok)) "PASS" else "FAIL"
  results[[length(results) + 1]] <- list(
    name = t$name, category = t$category,
    status = status,
    reason = cmp$reason,
    max_abs = cmp$max_abs, max_rel = cmp$max_rel,
    top5_match = cmp$top5_match
  )
}

# --- Report ---
cat("=======================================================\n")
cat("  5D CPY test — ", ggml_vulkan_device_description(0), "\n", sep = "")
cat("=======================================================\n\n")

cat(sprintf("%-20s %-13s %-7s %10s %10s %-6s\n",
            "Model", "Category", "Status", "max_abs", "max_rel", "top5"))
cat(strrep("-", 72), "\n", sep = "")
for (r in results) {
  cat(sprintf("%-20s %-13s %-7s %10s %10s %-6s %s\n",
              r$name, r$category, r$status,
              if (is.null(r$max_abs)) "—" else sprintf("%.3g", r$max_abs),
              if (is.null(r$max_rel)) "—" else sprintf("%.3g", r$max_rel),
              if (is.null(r$top5_match) || is.na(r$top5_match)) "—"
                else if (r$top5_match) "yes" else "no",
              if (!is.null(r$reason) && !is.na(r$reason)) r$reason else ""))
}

n_pass <- sum(sapply(results, function(r) r$status == "PASS"))
n_fail <- sum(sapply(results, function(r) r$status == "FAIL"))
n_err  <- sum(sapply(results, function(r) r$status == "ERROR"))
n_skip <- sum(sapply(results, function(r) r$status == "SKIP"))

cat(sprintf("\n%d PASS   %d FAIL   %d ERROR   %d SKIP\n", n_pass, n_fail, n_err, n_skip))

# --- Errors detail ---
errs <- Filter(function(r) r$status == "ERROR", results)
if (length(errs) > 0) {
  cat("\nErrors:\n")
  for (r in errs) cat(sprintf("  %s: %s\n", r$name, r$reason))
}

# --- Regression check for baseline ---
baseline_fails <- Filter(function(r) r$category == "baseline" && r$status != "PASS", results)
if (length(baseline_fails) > 0) {
  cat("\nBASELINE REGRESSION:\n")
  for (r in baseline_fails) cat(sprintf("  %s [%s]\n", r$name, r$status))
} else if (n_pass > 0) {
  cat("\nNo baseline regression.\n")
}
