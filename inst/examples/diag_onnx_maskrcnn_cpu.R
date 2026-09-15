#!/usr/bin/env Rscript
# MaskRCNN ONNX — CPU diagnostic run. NO GPU IS TOUCHED.
#
# Why this exists: three GPU runs of profile_onnx_maskrcnn_gpu.R hung the
# amdgpu compute ring (comp_1.1.x timeout), and the third hang took the
# graphics ring with it -- MODE1 reset, "VRAM is lost", desktop gone. A static
# audit of the four ONNX shaders found exactly one place where a loop bound
# comes from BUFFER CONTENTS rather than a dimension: RoiAlign's adaptive
# sample count, ceil(roi_h/oh), with nothing upstream bounding roi_h.
#
# ⚠️ That audit proved RoiAlign CAN hang this way. It did NOT prove RoiAlign
#    is what hung. The kernel log names no shader and the devcoredump had
#    already expired. This script settles the question WITHOUT risking another
#    session: if the ROIs are sane, the cap never binds and the hang is
#    somewhere else entirely.
#
# Usage:
#   Rscript inst/examples/diag_onnx_maskrcnn_cpu.R
#
# Read the [roi_align/cpu] line in the output:
#   over-cap=0, non-finite=0   -> ROIs are fine; RoiAlign is NOT the culprit,
#                                 and the hang needs a different explanation.
#   over-cap>0 or non-finite>0 -> ROIs are malformed. The cap is what now
#                                 stands between them and a dead GPU, and the
#                                 real bug is upstream in the box decode.

suppressMessages(library(ggmlR))

ONNX_DIR  <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")
ONNX_PATH <- file.path(ONNX_DIR, "MaskRCNN-12-int8.onnx")
SHAPE     <- c(3L, 224L, 224L)

# The whole point of this script: the trace prints the ROI summary the cap
# keys on. Set before the model runs, not after.
Sys.setenv(ONNX_TRACE_ROI_ALIGN = "1")

cat("=== MaskRCNN CPU diagnostic (no GPU) ===\n\n")

if (!file.exists(ONNX_PATH)) stop("model not found: ", ONNX_PATH)
cat(sprintf("Model: %s  (%.2f MB)\n", basename(ONNX_PATH),
            file.size(ONNX_PATH) / 1024 / 1024))

cat("Loading on CPU ... ")
t_load <- system.time({
  model <- onnx_load(ONNX_PATH, device = "cpu",
                     input_shapes = list(image = SHAPE))
})[3]
cat(sprintf("%.2f s\n\n", t_load))

set.seed(42)                       # same input as the GPU profiler
inp <- list(image = runif(prod(SHAPE)))

cat("Running once on CPU (slow -- this is a diagnostic, not a benchmark).\n")
cat("The [roi_align/cpu] trace lines below are the result:\n\n")

t_run <- system.time(out <- onnx_run(model, inp))[3]

cat(sprintf("\nRun finished in %.1f s\n", t_run))
cat(sprintf("Outputs: %d\n", length(out)))
for (i in seq_along(out)) {
  v  <- out[[i]]
  sh <- dim(v); if (is.null(sh)) sh <- length(v)
  cat(sprintf("  [%d] shape [%s], %d values\n",
              i, paste(sh, collapse = "x"), length(v)))
}

cat("\nNow read the [roi_align/cpu] line(s) above.\n")
cat("over-cap=0 and non-finite=0 means RoiAlign is exonerated.\n")
