#!/usr/bin/env Rscript
# bench_qconv_i32.R -- is the int32 QLinearConv path a throughput regression?
#
# The int32 path is selected by kernel shape, with no runtime switch, so
# "int32 vs f32" cannot be timed in one binary.  What CAN be timed without a
# rebuild is the suspected cause.
#
# qconv_i32_cpu splits rows across ith/nth and every output element is
# independent, but the call site passes n_tasks = 1
# (onnx_ops_quant.c, ggml_map_custom3(..., qconv_i32_cpu, 1, qp)).  So the
# kernel CAN scale and is CURRENTLY not allowed to.
#
# Prediction if that is the whole story:
#   MaskRCNN-12-int8  -- flat from 1 to N threads (its convs are pinned to one)
#   an int8 model off the int32 path -- scales with threads
# A MaskRCNN that scales anyway would mean the time is not in qconv at all,
# and the "int32 is slower" risk is aimed at the wrong place.
#
# Usage:  Rscript inst/scripts/bench_qconv_i32.R [reps]

suppressMessages(library(ggmlR))

ONNX_DIR <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")
reps     <- as.integer(commandArgs(TRUE)[1]); if (is.na(reps)) reps <- 3L
threads  <- c(1L, 2L, 4L, 8L)

# on the int32 path (1x1 / 3x3, group 1) vs off it -- the contrast that matters.
#
# The shapes are NOT optional.  MaskRCNN declares image = 3x1x1: its spatial
# dims are dynamic and onnx_inputs() reports them collapsed to 1.  Feeding
# that shape convolves a single pixel, which is instant, thread-insensitive,
# and would "confirm" the n_tasks=1 prediction while measuring nothing.
# 3x224x224 is what ref_dump_io.R feeds the same model.
models <- list(
    maskrcnn   = list(file = "MaskRCNN-12-int8.onnx",
                      shapes = list(image = c(3L, 224L, 224L))),
    squeezenet = list(file = "squeezenet1.0-8.onnx",
                      shapes = NULL)          # fully static already
)

find_model <- function(fname) {
    hits <- list.files(ONNX_DIR, pattern = paste0("^", fname, "$"),
                       recursive = TRUE, full.names = TRUE)
    if (length(hits)) hits[1] else NA_character_
}

# Build the inputs once, outside the timed region: allocating them per rep
# would time the allocator, not the convolution.
#
# onnx_inputs() returns a NAMED LIST -- name -> integer vector of dims, with
# -1 for a dynamic dimension.  Not a data frame: nrow() on it is NULL, which
# is what made the first version of this script die in seq_len().
# A FLAT vector per input, not an array with dim().  onnx_run() calls
# as.numeric() on whatever it gets, so dim() carries no information into the
# model -- it only looks like it sets the layout, and ONNX dims (row-major,
# outermost first) are the reverse of ggml ne[] anyway.  The shape comes from
# onnx_load(input_shapes=), which is how ref_dump_io.R feeds the same models.
make_inputs <- function(shapes) {
    lapply(shapes, function(d) runif(prod(d)))
}

cat(sprintf("reps=%d  threads=%s\n\n", reps, paste(threads, collapse = ",")))

for (nm in names(models)) {
    spec <- models[[nm]]
    path <- find_model(spec$file)
    if (is.na(path)) { cat(sprintf("%-12s SKIP (not found)\n", nm)); next }

    # device = "cpu": the int32 kernel is CPU-only and refuses to run on a
    # device buffer, so letting Vulkan claim the graph would time the f32
    # fallback instead of the path under test.
    m <- tryCatch(onnx_load(path, device = "cpu", input_shapes = spec$shapes),
                  error = function(e) NULL)
    if (is.null(m)) { cat(sprintf("%-12s SKIP (load failed)\n", nm)); next }

    shapes <- spec$shapes
    if (is.null(shapes)) shapes <- onnx_inputs(m)   # static model: as declared
    if (any(unlist(shapes) < 1L)) {
        cat(sprintf("%-12s SKIP (dynamic dims, no shape given)\n", nm)); next
    }
    inp <- make_inputs(shapes)

    invisible(tryCatch(onnx_run(m, inp), error = function(e) NULL))  # warm up

    cat(sprintf("%-12s (%s)\n", nm, basename(path)))
    base <- NA_real_
    for (nt in threads) {
        ggml_set_n_threads(nt)
        t <- replicate(reps, system.time(
                 tryCatch(onnx_run(m, inp), error = function(e) NULL)
             )[["elapsed"]])
        best <- min(t)                      # min, not mean: fewer scheduler artefacts
        if (is.na(base)) base <- best
        cat(sprintf("   %d thr  %7.3f s   speedup %4.2fx\n", nt, best, base / best))
    }
    cat("\n")
}

cat("flat MaskRCNN + scaling squeezenet  => n_tasks=1 is the cost\n")
cat("both scaling                        => qconv is not the bottleneck\n")
