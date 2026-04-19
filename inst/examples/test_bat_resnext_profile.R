#!/usr/bin/env Rscript
# ============================================================================
# Profile BAT-ResNeXt26ts on Vulkan with GGML_VK_PERF_LOGGER=1
# Filter CPY/CONT/DUP nodes and show which path they took (4d/5d/contig).
#
# Usage:
#   Rscript inst/examples/test_bat_resnext_profile.R
# ============================================================================

Sys.setenv(GGML_VK_PERF_LOGGER = "1")

suppressPackageStartupMessages(library(ggmlR))

ONNX_DIR   <- "/mnt/Data2/DS_projects/ONNX models-main"
MODEL_FILE <- "bat_resnext26ts_Opset18.onnx"
path <- file.path(ONNX_DIR, MODEL_FILE)

if (!file.exists(path)) stop("Model not found: ", path)
if (!ggml_vulkan_available()) stop("Vulkan not available")

input_name  <- "x"
input_shape <- c(1L, 3L, 256L, 256L)

set.seed(42)
inp <- runif(prod(input_shape))
inputs <- setNames(list(inp), input_name)
shapes <- setNames(list(input_shape), input_name)

model <- onnx_load(path, device = "vulkan", input_shapes = shapes)

# Capture stderr (PerfLogger prints via r_ggml_printf -> Rprintf -> stderr? -> stdout)
log_file <- tempfile(fileext = ".log")
con <- file(log_file, open = "wt")
sink(con, type = "output")
sink(con, type = "message")
out <- onnx_run(model, inputs)
sink(type = "message")
sink(type = "output")
close(con)

log_lines <- readLines(log_file)

cat("====================================================================\n")
cat("BAT-ResNeXt26ts — CPY/CONT/DUP profile\n")
cat("====================================================================\n\n")

cpy_lines <- grep("^\\s*(CPY|CONT|DUP)", log_lines, value = TRUE)

if (length(cpy_lines) == 0) {
  cat("No CPY/CONT/DUP lines in perf log.\n")
  cat("Full log sample (first 40 lines):\n")
  cat(paste(head(log_lines, 40), collapse = "\n"), "\n")
} else {
  # Group by path: [contig], [5d], [4d]
  contig <- grep("\\[contig\\]", cpy_lines, value = TRUE)
  fived  <- grep("\\[5d\\]",     cpy_lines, value = TRUE)
  fourd  <- grep("\\[4d\\]",     cpy_lines, value = TRUE)

  cat(sprintf("Total CPY/CONT/DUP nodes: %d\n", length(cpy_lines)))
  cat(sprintf("  [contig]: %d\n", length(contig)))
  cat(sprintf("  [5d]    : %d  <-- uses new 5D path\n", length(fived)))
  cat(sprintf("  [4d]    : %d\n\n", length(fourd)))

  summarize_ops <- function(lines, label) {
    if (length(lines) == 0) return(invisible(NULL))
    # strip timing suffix, keep op+shape+path
    stripped <- sub(":\\s+\\d+.*$", "", lines)
    tbl <- sort(table(stripped), decreasing = TRUE)
    cat(sprintf("--- %s (top unique) ---\n", label))
    for (i in seq_len(min(15L, length(tbl)))) {
      cat(sprintf("  %3dx  %s\n", tbl[[i]], names(tbl)[i]))
    }
    cat("\n")
  }

  summarize_ops(fived,  "5D CPY nodes")
  summarize_ops(fourd,  "4D CPY nodes")
  summarize_ops(contig, "CONTIG CPY nodes")
}

cat("\nFull perf log written to: ", log_file, "\n", sep = "")
