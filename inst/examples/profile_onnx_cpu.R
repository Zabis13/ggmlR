#!/usr/bin/env Rscript
# Which CPU kernels an ONNX model actually runs, and what each one costs.
#
# The CPU counterpart of profile_onnx_shaders.R. The Vulkan backend has
# GGML_VK_PERF_LOGGER, which times each dispatch from inside; the CPU backend
# has no such logger, so this script samples the process from OUTSIDE with
# gprofng and attributes time to the kernel functions ggml compiles the ops
# into (ggml_compute_forward_*, plus the ONNX-specific ones like
# qconv_i32_compute).
#
# That difference is worth keeping in mind when reading the two side by side:
#
#   - the shader profile knows the OP and its shapes, because the logger is
#     told what it is timing; this one knows the FUNCTION, and one function
#     can serve several ops (ggml_compute_forward_mul_mat handles every
#     matmul in the graph, whatever its shape).
#   - sampling misses nothing but resolves nothing finer than its interval:
#     a kernel called 5000 times for 3 us each shows up as its total, with no
#     per-call figure. `us_each` therefore has no counterpart here.
#   - the sampler sees EVERYTHING the process does -- graph building, weight
#     loading, R itself -- not just the compute. That is a feature: on a
#     segmented model the non-compute part is the question, and a profile
#     that hid it would answer the wrong one. The summary splits the two.
#
# Usage:
#   Rscript inst/examples/profile_onnx_cpu.R                    # MaskRCNN
#   Rscript inst/examples/profile_onnx_cpu.R superres           # by substring
#   Rscript inst/examples/profile_onnx_cpu.R MaskRCNN 20        # top 20 rows
#   EXPT=/tmp/x.er Rscript inst/examples/profile_onnx_cpu.R     # reuse a run
#
# Environment:
#   ONNX_DIR   where the .onnx files live
#   EXPT       parse this existing gprofng experiment instead of running
#   KEEP_EXPT  write the experiment here as well, to re-read later
#   N_THREADS  ggml CPU threads (default: leave the package default alone)
#
# Requires gprofng (binutils). Unlike perf it samples with the process's own
# timer, so it needs no root and no kernel.perf_event_paranoid change.

suppressMessages(library(ggmlR))

args   <- commandArgs(trailingOnly = TRUE)
FILTER <- if (length(args) >= 1 && nzchar(args[1])) args[1] else "MaskRCNN"
TOP_N  <- if (length(args) >= 2) as.integer(args[2]) else 10L
ONNX_DIR <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")

# Same registry as profile_onnx_shaders.R: the shapes matter, because a
# profile of the wrong input size is a profile of a different model.
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

# ── obtain an experiment ────────────────────────────────────────────
existing <- Sys.getenv("EXPT", "")
if (nzchar(existing)) {
  if (!dir.exists(existing)) stop("EXPT does not exist: ", existing)
  expt  <- existing
  mname <- basename(existing)
  cat(sprintf("Parsing existing experiment: %s\n\n", expt))
} else {
  if (nchar(Sys.which("gprofng")) == 0)
    stop("gprofng not found -- install binutils (it ships gprofng)")

  m <- Filter(function(x) grepl(FILTER, x$name, ignore.case = TRUE) ||
                          grepl(FILTER, x$file, ignore.case = TRUE), models)
  if (length(m) == 0) stop("no model matching '", FILTER, "'")
  m <- m[[1]]
  mname <- m$name
  path  <- file.path(ONNX_DIR, m$file)
  if (!file.exists(path)) stop("model not found: ", path)

  cat(sprintf("=== CPU profile: %s ===\n\n", mname))

  # The model runs in a CHILD process under the sampler: gprofng wraps a
  # command, it does not attach to this one. The child is given the same
  # script's model registry through its arguments rather than a temp copy of
  # the data, so the two stay in step.
  runner <- tempfile(fileext = ".R")
  writeLines(sprintf('
suppressMessages(library(ggmlR))
%s
set.seed(42)
inputs <- lapply(shapes, function(s) runif(prod(s)))
model  <- onnx_load(%s, device = "cpu", input_shapes = shapes)
invisible(onnx_run(model, inputs))
cat("run done\\n")
',
    paste0("shapes <- ", paste(deparse(m$shapes), collapse = "\n"),
           if (nzchar(Sys.getenv("N_THREADS")))
             sprintf("\nggml_set_n_threads(%dL)", as.integer(Sys.getenv("N_THREADS")))
           else ""),
    deparse(path)), runner)

  expt <- if (nzchar(Sys.getenv("KEEP_EXPT"))) Sys.getenv("KEEP_EXPT")
          else file.path(tempdir(), "onnx_cpu.er")
  unlink(expt, recursive = TRUE)

  cat("Sampling the run (this is the slow part) ... ")
  t0 <- proc.time()
  rc <- system2("gprofng",
                c("collect", "app", "-p", "on", "-o", shQuote(expt),
                  "Rscript", shQuote(runner)),
                stdout = TRUE, stderr = TRUE)
  wall <- (proc.time() - t0)[3]
  if (!any(grepl("run done", rc)))
    stop("the profiled run did not finish:\n", paste(tail(rc, 20), collapse = "\n"))
  cat(sprintf("%.1f s\n\n", wall))
}

# ── parse ───────────────────────────────────────────────────────────
# `display text -functions` prints a fixed-width table: two metric pairs
# (exclusive and inclusive, each as seconds and percent) then the name.
# Exclusive time is what attributes cost to a kernel rather than to its
# caller, so that is what is read here.
txt <- system2("gprofng",
               c("display", "text", "-functions", "-limit", "400", shQuote(expt)),
               stdout = TRUE, stderr = TRUE)

body <- txt[grepl("^ *[0-9.]+ +[0-9.]+ +[0-9.]+ +[0-9.]+ +\\S", txt)]
if (length(body) == 0)
  stop("no function rows in the gprofng report -- did the run sample at all?")

f <- sub("^ *([0-9.]+) +([0-9.]+) +([0-9.]+) +([0-9.]+) +(.*)$", "\\1|\\3|\\5", body)
parts <- strsplit(f, "|", fixed = TRUE)
rows <- data.frame(
  excl = as.numeric(vapply(parts, `[`, "", 1)),
  incl = as.numeric(vapply(parts, `[`, "", 2)),
  name = trimws(vapply(parts, `[`, "", 3)),
  stringsAsFactors = FALSE
)
rows <- rows[rows$name != "<Total>", ]
total <- sum(rows$excl)
if (total <= 0) stop("the report totals zero time")

# A C++ kernel's demangled signature runs to well over a terminal width, and
# print.data.frame given one wraps the whole table into unreadable columns.
# The name is only ever used to identify the function, so cut it to fit.
short <- function(s) {
  # Only a C++ signature's argument list goes: gprofng also writes the owning
  # library in parentheses ("<static>@0x2542b (<libgomp.so.1>)"), and that is
  # the only thing identifying an unresolved address -- cutting it turns every
  # such row into an anonymous hex offset.
  s <- ifelse(grepl("\\(<", s), s, sub("\\(.*$", "", s))
  ifelse(nchar(s) > 58, paste0(substr(s, 1, 55), "..."), s)
}

# ── classify ────────────────────────────────────────────────────────
# A ggml op becomes a ggml_compute_forward_<op> function, and the ONNX-only
# ops have their own named kernels. Everything else is the model's overhead:
# building graphs, loading weights, the R interpreter, the allocator.
#
# Anything unrecognised counts as overhead rather than being dropped, so the
# two halves always add up to the whole run -- a kernel this list forgets
# shows up as a suspiciously large overhead row, not as missing time.
# Matched as prefixes, not as whole names: GCC renames a function it has
# cloned or had its constant arguments propagated out of, so the kernel that
# actually runs can be "qmatmul_i32_cpu_impl.isra.0" rather than
# "qmatmul_i32_compute". Anchoring on the full name sent 31% of a MaskRCNN
# profile into the overhead column, where it read as if graph building were
# the expense.
kernel_pat <- paste0(
  "ggml_compute_forward|ggml_vec_|ggml_compute_|",
  "qconv_i32|qmatmul_i32|",
  "roi_align|rel_pos_bias|nms_"
)
rows$kind <- ifelse(grepl(kernel_pat, rows$name), "kernel", "other")

# The op name, for the rows where one can be read off the function. The
# suffixes GCC appends (.isra.0, .constprop.0, .part.0) are cut so that a
# cloned kernel aggregates with its own op rather than beside it.
rows$op <- sub("\\.(isra|constprop|part|cold)\\.?[0-9]*$", "", rows$name)
rows$op <- ifelse(
  grepl("^ggml_compute_forward_", rows$op),
  toupper(sub("^ggml_compute_forward_([a-z_0-9]+).*$", "\\1", rows$op)),
  sub("^(qconv_i32|qmatmul_i32)_(compute|cpu_impl)$", "\\1", rows$op))

kernels <- rows[rows$kind == "kernel", ]
others  <- rows[rows$kind == "other",  ]
k_total <- sum(kernels$excl)

cat(sprintf("%.2f s sampled: %.2f s in kernels (%.1f%%), %.2f s elsewhere (%.1f%%)\n\n",
            total, k_total, 100 * k_total / total,
            total - k_total, 100 * (total - k_total) / total))

# ── kernels, by op ──────────────────────────────────────────────────
if (nrow(kernels) > 0) {
  by_op <- aggregate(excl ~ op, kernels, sum)
  by_op <- by_op[order(-by_op$excl), ]
  # A kernel whose name matched none of the op patterns keeps its full
  # signature here, which is wide enough to wrap the table -- cut it as well.
  by_op$op <- short(by_op$op)
  by_op$sec <- round(by_op$excl, 3)
  by_op$pct_kern <- round(100 * by_op$excl / k_total, 1)
  by_op$pct_run  <- round(100 * by_op$excl / total, 1)
  cat("--- kernels by op -----------------------------------------\n")
  print(head(by_op[, c("op", "sec", "pct_kern", "pct_run")], TOP_N),
        row.names = FALSE)
  cat("\n")
}

# ── everything that is not a kernel ─────────────────────────────────
# On a segmented model this is the interesting half: 39 segments means 39
# graph builds, and none of that time is in a kernel.
if (nrow(others) > 0) {
  o <- others[order(-others$excl), ]
  o$sec <- round(o$excl, 3)
  o$pct <- round(100 * o$excl / total, 1)
  o$name <- short(o$name)
  cat("--- outside the kernels (graph build, load, R, alloc) -----\n")
  print(head(o[, c("name", "sec", "pct")], TOP_N), row.names = FALSE)
  cat("\n")
}

# ── the individual functions, unclassified ──────────────────────────
# The rawest view, for when the classification above hides something: one row
# per function, kernels and overhead together, most expensive first.
a <- rows[order(-rows$excl), ]
a$sec <- round(a$excl, 3)
a$pct <- round(100 * a$excl / total, 1)
a$name <- short(a$name)
cat("--- all functions, by exclusive time ----------------------\n")
print(head(a[, c("name", "kind", "sec", "pct")], TOP_N), row.names = FALSE)

cat(sprintf("\nExperiment: %s\n", expt))
cat("Re-read it without running the model again:\n")
cat(sprintf("  EXPT=%s Rscript inst/examples/profile_onnx_cpu.R\n", expt))
