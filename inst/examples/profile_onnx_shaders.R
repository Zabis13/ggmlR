#!/usr/bin/env Rscript
# Which Vulkan shaders an ONNX model actually runs, and what each one costs.
#
# GGML_VK_PERF_LOGGER makes the backend print a timing block per computed graph.
# A segmented model produces one block per segment, so the raw log answers two
# different questions at once and this script separates them:
#
#   - the totals: which op dominates the model's GPU time overall
#   - per segment: where in the graph that time sits, which is what tells you
#     whether one stage is an outlier or the cost is spread evenly
#
# ⚠️ The logger inflates what it measures, so read these numbers as proportions,
# not as wall-clock truth. "Which shader is expensive" is answerable here;
# "how fast is the model" is not -- use profile_onnx_maskrcnn_gpu.R for that,
# with the logger off.
#
# ⚠️ Shader time is not run time. On MaskRCNN the blocks below add up to about
# 150 ms against a ~1500 ms inference: the rest goes to CPU-only ops, segment
# boundaries and transfers, none of which appear in a Vulkan timing block. A
# shader saving of 60 ms was invisible in the end-to-end number for exactly
# this reason.
#
# Usage:
#   Rscript inst/examples/profile_onnx_shaders.R                    # MaskRCNN
#   Rscript inst/examples/profile_onnx_shaders.R superres           # by substring
#   Rscript inst/examples/profile_onnx_shaders.R MaskRCNN 12        # top 12 rows
#   PERF_LOG=/tmp/x.log Rscript inst/examples/profile_onnx_shaders.R  # reuse a log
#
# Environment:
#   ONNX_DIR   where the .onnx files live
#   PERF_LOG   parse this existing log instead of running a model
#   KEEP_LOG   write the raw log here as well

suppressMessages(library(ggmlR))

args   <- commandArgs(trailingOnly = TRUE)
FILTER <- if (length(args) >= 1 && nzchar(args[1])) args[1] else "MaskRCNN"
TOP_N  <- if (length(args) >= 2) as.integer(args[2]) else 10L
ONNX_DIR <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")

# Same registry shape as the other example scripts; shapes matter because a
# perf profile of the wrong input size is a profile of a different model.
models <- list(
  list(name = "MaskRCNN int8",   file = "MaskRCNN-12-int8.onnx",
       shapes = list(image = c(3L, 224L, 224L))),
  list(name = "SuperResolution", file = "super-resolution-10.onnx",
       shapes = list(input = c(1L, 1L, 224L, 224L))),
  list(name = "Inception V3",    file = "adv_inception_v3_Opset17.onnx",
       shapes = list(x = c(1L, 3L, 299L, 299L))),
  list(name = "BAT-ResNeXt26ts", file = "bat_resnext26ts_Opset18.onnx",
       shapes = list(x = c(1L, 3L, 256L, 256L))),
  list(name = "CaiT XS24",       file = "cait_xs24_384_Opset16.onnx",
       shapes = list(x = c(1L, 3L, 384L, 384L))),
  list(name = "BoTNet26t",       file = "botnet26t_256_Opset16.onnx",
       shapes = list(x = c(1L, 3L, 256L, 256L)))
)

# ── obtain a log ────────────────────────────────────────────────────
existing <- Sys.getenv("PERF_LOG", "")
if (nzchar(existing)) {
  if (!file.exists(existing)) stop("PERF_LOG does not exist: ", existing)
  log_file <- existing
  cat(sprintf("Parsing existing log: %s\n\n", log_file))
} else {
  m <- Filter(function(x) grepl(FILTER, x$name, ignore.case = TRUE) ||
                          grepl(FILTER, x$file, ignore.case = TRUE), models)
  if (length(m) == 0) stop("no model matching '", FILTER, "'")
  m <- m[[1]]
  path <- file.path(ONNX_DIR, m$file)
  if (!file.exists(path)) stop("model not found: ", path)
  if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)))
    stop("Vulkan not available")

  cat(sprintf("=== Shader profile: %s ===\n\n", m$name))
  Sys.setenv(GGML_VK_PERF_LOGGER = "1")

  set.seed(42)
  inputs <- lapply(m$shapes, function(s) runif(prod(s)))
  model  <- onnx_load(path, device = "vulkan", input_shapes = m$shapes)

  # The logger writes through ggml's own printf, so the output has to be
  # captured rather than returned.
  log_file <- if (nzchar(Sys.getenv("KEEP_LOG"))) Sys.getenv("KEEP_LOG")
              else tempfile(fileext = ".log")
  con <- file(log_file, open = "wt")
  sink(con, type = "output"); sink(con, type = "message")
  invisible(onnx_run(model, inputs))
  sink(type = "message"); sink(type = "output")
  close(con)
}

# ── parse ───────────────────────────────────────────────────────────
# A timing line is "<op and detail>: <n> x <avg> us = <total> us", optionally
# followed by a GFLOPS figure. The op name is the leading token; everything
# after it is shape detail, which is what distinguishes a cheap copy from an
# expensive one of the same op.
lines   <- readLines(log_file, warn = FALSE)
is_time <- grepl("^[A-Za-z_].*: [0-9]+ x [0-9.]+ us = [0-9.]+ us", lines)
is_head <- grepl("^Vulkan Timings:", lines)

seg <- cumsum(is_head)          # which block each line belongs to
rows <- data.frame(
  seg   = seg[is_time],
  full  = sub(":.*$", "", lines[is_time]),
  op    = sub("^([A-Za-z_0-9]+).*", "\\1", lines[is_time]),
  n     = as.numeric(sub("^.*: ([0-9]+) x .*", "\\1", lines[is_time])),
  us    = as.numeric(sub("^.* = ([0-9.]+) us.*", "\\1", lines[is_time])),
  stringsAsFactors = FALSE
)
if (nrow(rows) == 0) stop("no timing lines found -- was GGML_VK_PERF_LOGGER set?")

n_seg <- max(rows$seg)
total <- sum(rows$us)

# ── totals by op ────────────────────────────────────────────────────
by_op <- aggregate(cbind(n, us) ~ op, rows, sum)
by_op <- by_op[order(-by_op$us), ]
by_op$ms  <- round(by_op$us / 1000, 1)
by_op$pct <- round(100 * by_op$us / total, 1)
by_op$us_each <- round(by_op$us / by_op$n, 1)

cat(sprintf("%d segments, %d dispatches, %.1f ms of shader time\n\n",
            n_seg, sum(rows$n), total / 1000))
cat("--- by op -------------------------------------------------\n")
print(head(by_op[, c("op", "n", "ms", "pct", "us_each")], TOP_N), row.names = FALSE)

# ── the expensive individual shapes ─────────────────────────────────
# Two calls to the same op can differ by an order of magnitude depending on the
# tensor it is handed, so the op name alone can hide the actual cost driver.
by_shape <- aggregate(cbind(n, us) ~ full, rows, sum)
by_shape <- by_shape[order(-by_shape$us), ]
by_shape$ms  <- round(by_shape$us / 1000, 2)
by_shape$pct <- round(100 * by_shape$us / total, 1)
cat("\n--- by shape (most expensive) -----------------------------\n")
sh <- head(by_shape, TOP_N)
for (i in seq_len(nrow(sh)))
  cat(sprintf("  %6.2f ms  %4.1f%%  %3dx  %s\n",
              sh$ms[i], sh$pct[i], sh$n[i], substr(sh$full[i], 1, 78)))

# ── per segment ─────────────────────────────────────────────────────
# Where in the graph the time sits. A segment boundary is a point where the
# scheduler had to stop, so an outlier here is a place to look at the graph,
# not only at the shader.
by_seg <- aggregate(cbind(n, us) ~ seg, rows, sum)
by_seg$ms  <- round(by_seg$us / 1000, 2)
by_seg$pct <- round(100 * by_seg$us / total, 1)
top_in <- sapply(by_seg$seg, function(s) {
  r <- rows[rows$seg == s, ]
  a <- aggregate(us ~ op, r, sum)
  a <- a[order(-a$us), ]
  sprintf("%s %.0f%%", a$op[1], 100 * a$us[1] / sum(a$us))
})
by_seg$dominant <- top_in
cat("\n--- by segment --------------------------------------------\n")
print(by_seg[, c("seg", "n", "ms", "pct", "dominant")], row.names = FALSE)

cat(sprintf("\nRaw log: %s\n", log_file))
