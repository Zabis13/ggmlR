#!/usr/bin/env Rscript
# Debug script for BotNet26t (rel_pos_bias) and cait_xs24_384 (NaN tracing)
#
# Usage: Rscript inst/examples/debug_bat_resnext_diff.R

library(ggmlR)

ONNX_DIR  <- "/mnt/Data2/DS_projects/ONNX models-main"
THRESHOLD <- 0.01

# ============================================================
# 1. BotNet26t — rel_pos_bias_2d_cpu kernel diagnostics
#    [rpb_dbg] lines show x->data ptr, ne, nb, values
#    Compare CPU vs GPU to see if x data differs
# ============================================================
cat("=======================================================\n")
cat("  BotNet26t — rel_pos_bias CPU kernel diagnostics\n")
cat("=======================================================\n\n")

path_bot <- file.path(ONNX_DIR, "botnet26t_256_Opset16.onnx")
shapes_bot <- list(x = c(1L, 3L, 256L, 256L))
set.seed(42)
inp_bot <- list(x = runif(1L * 3L * 256L * 256L))

cat("--- CPU run ---\n")
m_cpu <- onnx_load(path_bot, device = "cpu", input_shapes = shapes_bot)
out_cpu <- onnx_run(m_cpu, inp_bot)[[1]]
cat(sprintf("CPU range: [%.6f, %.6f]\n", min(out_cpu), max(out_cpu)))
rm(m_cpu); gc(verbose = FALSE)

cat("\n--- GPU run ---\n")
m_gpu <- onnx_load(path_bot, device = "vulkan", input_shapes = shapes_bot)
out_gpu <- onnx_run(m_gpu, inp_bot)[[1]]
cat(sprintf("GPU range: [%.6f, %.6f]\n", min(out_gpu), max(out_gpu)))

cat("\n--- GPU stability (3 runs) ---\n")
for (i in 2:4) {
  o <- onnx_run(m_gpu, inp_bot)[[1]]
  cat(sprintf("  run %d vs run 1: max_diff = %.2e\n", i, max(abs(o - out_gpu))))
}
rm(m_gpu); gc(verbose = FALSE)

v_cpu <- as.numeric(out_cpu)
v_gpu <- as.numeric(out_gpu)
d     <- abs(v_cpu - v_gpu)
cat(sprintf("\nmax_abs: %.6f  %s\n", max(d), if (max(d) < THRESHOLD) "[PASS]" else "[FAIL]"))
cat(sprintf("cor:     %.8f\n", cor(v_cpu, v_gpu)))
top5_cpu <- order(v_cpu, decreasing = TRUE)[1:5]
top5_gpu <- order(v_gpu, decreasing = TRUE)[1:5]
cat(sprintf("top5 cpu: %s\n", paste(top5_cpu, collapse = " ")))
cat(sprintf("top5 gpu: %s\n", paste(top5_gpu, collapse = " ")))
cat(sprintf("top5 match: %s\n", if (identical(top5_cpu, top5_gpu)) "[PASS]" else "[FAIL]"))

# ============================================================
# 2. cait_xs24_384 — NaN tracing (requires ONNX_NAN_DEBUG build)
#    [NaN-OUT] lines show first node with NaN output
# ============================================================
cat("\n=======================================================\n")
cat("  cait_xs24_384 — NaN tracing (CPU)\n")
cat("=======================================================\n\n")

path_cait <- file.path(ONNX_DIR, "cait_xs24_384_Opset16.onnx")
shapes_cait <- list(x = c(1L, 3L, 384L, 384L))
set.seed(42)
inp_cait <- list(x = runif(1L * 3L * 384L * 384L))

cat("Loading cait on CPU (NaN trace to stderr)...\n")
m_cait <- onnx_load(path_cait, device = "cpu", input_shapes = shapes_cait)
out_cait <- onnx_run(m_cait, inp_cait)[[1]]
v_cait <- as.numeric(out_cait)
cat(sprintf("n_nan: %d  n_inf: %d  n_ok: %d  (total %d)\n",
            sum(is.nan(v_cait)), sum(is.infinite(v_cait)),
            sum(is.finite(v_cait)), length(v_cait)))
rm(m_cait); gc(verbose = FALSE)

# ============================================================
# 3. xcit_tiny — NaN tracing (CPU then GPU diff)
# ============================================================
cat("\n=======================================================\n")
cat("  xcit_tiny_12_p8_224 — NaN tracing (CPU)\n")
cat("=======================================================\n\n")

path_xcit  <- file.path(ONNX_DIR, "xcit_tiny_12_p8_224_Opset17.onnx")
shapes_xcit <- list(x = c(1L, 3L, 224L, 224L))
set.seed(42)
inp_xcit <- list(x = runif(1L * 3L * 224L * 224L))

cat("Loading xcit on CPU (NaN trace to stderr)...\n")
m_xcit_cpu <- onnx_load(path_xcit, device = "cpu", input_shapes = shapes_xcit)
out_xcit_cpu <- onnx_run(m_xcit_cpu, inp_xcit)[[1]]
v_xcit_cpu <- as.numeric(out_xcit_cpu)
cat(sprintf("n_nan: %d  n_inf: %d  n_ok: %d  (total %d)\n",
            sum(is.nan(v_xcit_cpu)), sum(is.infinite(v_xcit_cpu)),
            sum(is.finite(v_xcit_cpu)), length(v_xcit_cpu)))
rm(m_xcit_cpu); gc(verbose = FALSE)

cat("\nLoading xcit on GPU (NaN trace to stderr)...\n")
m_xcit_gpu <- onnx_load(path_xcit, device = "vulkan", input_shapes = shapes_xcit)
out_xcit_gpu <- onnx_run(m_xcit_gpu, inp_xcit)[[1]]
v_xcit_gpu <- as.numeric(out_xcit_gpu)
cat(sprintf("n_nan: %d  n_inf: %d  n_ok: %d  (total %d)\n",
            sum(is.nan(v_xcit_gpu)), sum(is.infinite(v_xcit_gpu)),
            sum(is.finite(v_xcit_gpu)), length(v_xcit_gpu)))

d_xcit <- abs(v_xcit_cpu - v_xcit_gpu)
cat(sprintf("\nmax_abs: %.6f  %s\n", max(d_xcit), if (max(d_xcit) < THRESHOLD) "[PASS]" else "[FAIL]"))
cat(sprintf("cor:     %.8f\n", cor(v_xcit_cpu, v_xcit_gpu)))
top5_cpu <- order(v_xcit_cpu, decreasing = TRUE)[1:5]
top5_gpu <- order(v_xcit_gpu, decreasing = TRUE)[1:5]
cat(sprintf("top5 cpu: %s\n", paste(top5_cpu, collapse = " ")))
cat(sprintf("top5 gpu: %s\n", paste(top5_gpu, collapse = " ")))
cat(sprintf("top5 match: %s\n", if (identical(top5_cpu, top5_gpu)) "[PASS]" else "[FAIL]"))
rm(m_xcit_gpu); gc(verbose = FALSE)
