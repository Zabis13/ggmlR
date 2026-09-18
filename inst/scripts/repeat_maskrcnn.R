#!/usr/bin/env Rscript
# MaskRCNN repeated-run probe.
#
# The defect is invisible to a single run: run 1 is correct on both backends.
# So this runs the SAME input three times and prints the detection count after
# each one, flushed, so that a crash on run 2 still leaves run 1's number on
# screen.
#
# Usage:
#   Rscript repeat_maskrcnn.R cpu
#   Rscript repeat_maskrcnn.R vulkan
#
# Read the three numbers:
#   51 / 51 / 51        -> the narrow step was enough; src_snap can go.
#   52 / ...            -> a consumer lives further away than the re-mapped
#                          range; measure WHICH segment before widening.
#   51 then a crash     -> consumers are not the issue; the hypothesis changes.
#   "index ... out of range" on CPU
#                       -> this is the already-reverted map_node_range attempt,
#                          not a new measurement. Stop and fix that first.

suppressMessages(library(ggmlR))

args   <- commandArgs(trailingOnly = TRUE)
DEV    <- if (length(args)) args[1] else "cpu"
N      <- as.integer(Sys.getenv("N_RUNS", "3"))
ONNX_DIR  <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")
ONNX_PATH <- file.path(ONNX_DIR, "MaskRCNN-12-int8.onnx")
SHAPE     <- c(3L, 224L, 224L)

if (!file.exists(ONNX_PATH)) stop("model not found: ", ONNX_PATH)

cat(sprintf("=== MaskRCNN repeated-run probe: device=%s, %d runs ===\n\n", DEV, N))

cat("Loading ... "); flush.console()
model <- onnx_load(ONNX_PATH, device = DEV, input_shapes = list(image = SHAPE))
cat("ok\n\n"); flush.console()

# One input, reused. A different input per run would make a difference in the
# detection count ambiguous -- the whole point is that the count must not move.
set.seed(42)
inp <- list(image = runif(prod(SHAPE)))

# Detections = rows of the [N,4] box tensor this model's first output carries.
#
# ⚠️ NOT a stride-3 scan for non-negative batch indices. That rule belongs to a
# [N,3] NMS selected_indices tensor; applied to [N,4] boxes it reads every third
# float of a coordinate list and counts whatever happens to be >= 0, which on
# MaskRCNN-12-int8 reports 68 for the 51 detections ORT also returns. The count
# looked like a real disagreement with the reference until the formula itself
# was checked against ORT's output length.
n_dets <- function(out) {
  v <- out[[1]]
  if (is.null(v)) return(NA_integer_)
  v <- as.numeric(v)
  if (!length(v)) return(0L)
  length(v) %/% 4L
}

for (i in seq_len(N)) {
  cat(sprintf("run %d ... ", i)); flush.console()
  t0  <- proc.time()
  out <- onnx_run(model, inp)
  dt  <- (proc.time() - t0)[3]
  cat(sprintf("%7.1f ms   dets=%s   out[1] len=%d\n",
              dt * 1e3, format(n_dets(out)), length(as.numeric(out[[1]]))))
  flush.console()
}

cat("\nAll runs survived.\n")
