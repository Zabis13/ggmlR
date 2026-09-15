#!/usr/bin/env Rscript
# Run the ONNX models on the GPU (Vulkan), one model per invocation-safe step.
#
# test_all_onnx.R is hardwired to device = "cpu", so it cannot answer anything
# about the Vulkan path. This is its GPU counterpart: same registry shape, but
# every model is loaded with device = "vulkan" and the scheduler's own view of
# the graph is printed alongside the result -- how many splits it made, how many
# ops it refused, and what it had to copy back to the host.
#
# The scheduler numbers are the point. A model that runs on the GPU can still be
# running most of itself on the CPU: ops the Vulkan backend does not implement
# fall back, and each fallback cuts the graph in two and copies tensors across.
# MaskRCNN is the extreme case -- 63 CPU-only custom ops -- and it aborts in the
# scheduler before computing anything, which no amount of output inspection can
# explain. Hence the split report.
#
# Usage:
#   Rscript inst/examples/test_onnx_gpu.R                 # all models
#   Rscript inst/examples/test_onnx_gpu.R maskrcnn        # one, by substring
#
# Useful environment variables:
#   GGML_SCHED_DEBUG_SPLITS=1  the scheduler reports every oversized split, the
#                              tensors filling it, and the size the input limit
#                              would actually need. Under this flag an overflow
#                              is reported and EXECUTION CONTINUES with a
#                              deliberately wrong graph, so one run lists every
#                              problem split rather than dying at the first.
#                              ⚠️ Results are meaningless with it set.
#   OMP_NUM_THREADS=1          a GGML_ABORT inside a worker thread corrupts the
#                              stack instead of reporting cleanly, so debug
#                              single-threaded.
#   ONNX_TRACE_VALS=1          print each node's output as it is computed --
#                              for when the model RUNS but the numbers are
#                              wrong, which is a different question.
#
# Typical: GGML_SCHED_DEBUG_SPLITS=1 OMP_NUM_THREADS=1 \
#            Rscript inst/examples/test_onnx_gpu.R maskrcnn > /tmp/gpu.log 2>&1

suppressMessages(library(ggmlR))

ONNX_DIR <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")
FILTER   <- commandArgs(trailingOnly = TRUE)[1]

models <- list(
  list(name = "MNIST",            file = "mnist-8.onnx",
       shapes = list(Input3 = c(1L, 1L, 28L, 28L))),
  list(name = "SqueezeNet 1.0",   file = "squeezenet1.0-8.onnx",
       shapes = list(data_0 = c(1L, 3L, 224L, 224L))),
  list(name = "Inception V3",     file = "adv_inception_v3_Opset17.onnx",
       shapes = list(x = c(1L, 3L, 299L, 299L))),
  list(name = "SuperResolution",  file = "super-resolution-10.onnx",
       shapes = list(input = c(1L, 1L, 224L, 224L))),
  list(name = "EmotionFerPlus",   file = "emotion-ferplus-8.onnx",
       shapes = list(Input3 = c(1L, 1L, 64L, 64L))),
  list(name = "BAT-ResNeXt26ts",  file = "bat_resnext26ts_Opset18.onnx",
       shapes = list(x = c(1L, 3L, 256L, 256L))),
  list(name = "BERT Opset17",     file = "bert_Opset17.onnx",
       shapes = list(input_ids = c(1L, 128L)), int_input = TRUE),
  list(name = "RoBERTa SeqClass", file = "roberta-sequence-classification-9.onnx",
       shapes = list(input_ids = c(1L, 128L)), int_input = TRUE),
  list(name = "GPT-NeoX",         file = "gptneox_Opset18.onnx",
       shapes = list(input_ids = c(1L, 8L)), int_input = TRUE),
  list(name = "CaiT XS24",        file = "cait_xs24_384_Opset16.onnx",
       shapes = list(x = c(1L, 3L, 384L, 384L))),
  list(name = "XCiT Tiny12 P8",   file = "xcit_tiny_12_p8_224_Opset17.onnx",
       shapes = list(x = c(1L, 3L, 224L, 224L))),
  list(name = "BoTNet26t",        file = "botnet26t_256_Opset16.onnx",
       shapes = list(x = c(1L, 3L, 256L, 256L))),
  list(name = "SAGEConv",         file = "sageconv_Opset16.onnx",
       shapes = list(x = c(2708L, 1433L), edge_index = c(2L, 10556L))),
  list(name = "MaskRCNN int8",    file = "MaskRCNN-12-int8.onnx",
       shapes = list(image = c(3L, 224L, 224L)))
)

if (!is.na(FILTER) && nzchar(FILTER))
  models <- Filter(function(m) grepl(FILTER, m$name, ignore.case = TRUE) ||
                               grepl(FILTER, m$file, ignore.case = TRUE), models)

if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE))) {
  cat("No Vulkan device -- nothing to do.\n")
  quit(status = 0)
}

cat(sprintf("Running %d model(s) on Vulkan\n\n", length(models)))

n_ok <- 0L; n_fail <- 0L
for (m in models) {
  path <- file.path(ONNX_DIR, m$file)
  cat("==============================================================\n")
  cat(sprintf("  %s  (%s)\n", m$name, m$file))
  cat("==============================================================\n")
  if (!file.exists(path)) { cat("  SKIP: file not found\n\n"); next }

  # Inputs: whole numbers in a plausible id range for token inputs, uniform
  # noise otherwise. The values do not matter for a scheduler question, but a
  # token model fed fractions fails at the embedding lookup rather than where
  # the interesting part is.
  set.seed(42)
  inputs <- list()
  for (nm in names(m$shapes)) {
    n <- prod(m$shapes[[nm]])
    inputs[[nm]] <- if (isTRUE(m$int_input) || grepl("input_ids|edge_index", nm))
                      as.numeric(sample.int(1000L, n, replace = TRUE))
                    else runif(n)
  }

  ok <- tryCatch({
    model <- onnx_load(path, device = "vulkan", input_shapes = m$shapes)

    di <- tryCatch(onnx_device_info(model), error = function(e) NULL)
    if (!is.null(di)) {
      cat(sprintf("  backends: %s\n", paste(di$backends, collapse = ", ")))
      cat(sprintf("  graph: %d nodes, %d splits, GPU ops %d, CPU-only %d\n",
                  di$n_nodes, di$n_splits, di$gpu_ops, di$cpu_ops))
      if (isTRUE(di$cpu_ops > 0))
        cat(sprintf("  CPU-only ops: %s\n",
                    paste(sprintf("%s(%d)", names(di$cpu_only_ops), di$cpu_only_ops),
                          collapse = ", ")))
    }

    t0 <- proc.time()
    out <- onnx_run(model, inputs)
    ms <- (proc.time() - t0)[3] * 1000

    v <- as.numeric(out[[1]])
    cat(sprintf("  OK  %.1f ms  outputs=%d  first length=%d\n",
                ms, length(out), length(v)))
    if (length(v) > 0)
      cat(sprintf("  head: %s\n",
                  paste(format(head(v, 4), digits = 6), collapse = " ")))
    else
      cat("  ⚠️ EMPTY OUTPUT -- the model ran but produced nothing\n")

    rm(model, out); invisible(gc(verbose = FALSE))
    TRUE
  }, error = function(e) {
    cat(sprintf("  FAIL: %s\n", conditionMessage(e)))
    FALSE
  })

  if (ok) n_ok <- n_ok + 1L else n_fail <- n_fail + 1L
  cat("\n")
}

cat("==============================================================\n")
cat(sprintf("  %d ok, %d failed\n", n_ok, n_fail))
cat("==============================================================\n")
