#!/usr/bin/env Rscript
# test_split_5d.R — isolate Split aliasing bug on CPU vs GPU
#
# Tests Split op with various tensor shapes and n_splits, comparing
# CPU and GPU output for each individual split output.
#
# Reproduces the bat_resnext26ts divergence:
#   src_ne=[8,8,8,8,16]  axis=1  n_splits=8  → outputs [8,8,8,1,16] each

suppressPackageStartupMessages(library(ggmlR))

`%||%` <- function(a, b) if (!is.null(a) && nzchar(a)) a else b

source("/mnt/Data2/DS_projects/ggmlR/tests/testthat/helper-onnx.R")

# ── ONNX builder for Split model ─────────────────────────────────────────────
# X[input_dims] → Split(axis, n_splits) → Y0, Y1, ..., Y{n-1}
# All outputs collected via Identity → single output tensor Z (Concat along axis)
# Actually: expose all split outputs directly as graph outputs.

make_split_model <- function(input_dims, axis, n_splits) {
  out_names <- paste0("Y", seq_len(n_splits) - 1L)

  inp  <- .onnx_value_info("X", 1L, input_dims)

  out_dim <- input_dims
  out_dim[axis + 1L] <- input_dims[axis + 1L] / n_splits  # ONNX axis (0-based)

  outputs <- lapply(out_names, function(nm) .onnx_value_info(nm, 1L, out_dim))

  node <- .onnx_node("Split", "X", out_names,
                     attrs = list(.onnx_attr_int("axis", axis)))

  graph <- .onnx_graph("split_test", list(node), list(inp), outputs)
  model <- .onnx_model(graph, opset_version = 13L)

  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)
  path
}

run_split <- function(path, input_dims, n_splits, device) {
  shapes <- list(X = as.integer(input_dims))
  set.seed(42)
  data  <- list(X = runif(prod(input_dims)))
  model <- onnx_load(path, device = device, input_shapes = shapes)
  out   <- onnx_run(model, data)
  rm(model); gc(verbose = FALSE)
  out
}

cmp <- function(cpu, gpu, label) {
  ok <- TRUE
  for (i in seq_along(cpu)) {
    a <- as.numeric(cpu[[i]])
    b <- as.numeric(gpu[[i]])
    if (length(a) != length(b)) {
      cat(sprintf("  [FAIL] %s output %d: length mismatch %d vs %d\n",
                  label, i, length(a), length(b)))
      ok <- FALSE
      next
    }
    if (any(!is.finite(a)) || any(!is.finite(b))) {
      cat(sprintf("  [FAIL] %s output %d: NaN/Inf\n", label, i))
      ok <- FALSE
      next
    }
    max_abs <- max(abs(a - b))
    max_rel <- max_abs / max(max(abs(a)), max(abs(b)), 1e-8)
    pass <- max_abs < 1e-3 || max_rel < 0.01
    cat(sprintf("  [%s] %s out%d  max_abs=%.5f  max_rel=%.5f\n",
                if (pass) "PASS" else "FAIL", label, i - 1L, max_abs, max_rel))
    if (!pass) ok <- FALSE
  }
  ok
}

if (!ggml_vulkan_available()) stop("Vulkan not available")

tests <- list(
  # name, input_dims (ONNX order), axis (0-based ONNX), n_splits
  list(name = "4D  [1,8,8,8]    axis=1 n=8",
       dims = c(1L, 8L, 8L, 8L),   axis = 1L, n = 8L),
  list(name = "5D  [16,8,8,8,8]  axis=1 n=8  (bat_resnext pattern)",
       dims = c(16L, 8L, 8L, 8L, 8L), axis = 1L, n = 8L),
  list(name = "5D  [64,2,16,8,2] axis=1 n=2",
       dims = c(64L, 2L, 16L, 8L, 2L), axis = 1L, n = 2L),
  list(name = "5D  [32,4,8,8,4]  axis=1 n=4",
       dims = c(32L, 4L, 8L, 8L, 4L), axis = 1L, n = 4L),
  list(name = "5D  [16,8,8,1,16] axis=3 n=1",
       dims = c(16L, 8L, 8L, 1L, 16L), axis = 3L, n = 1L)
)

n_pass <- 0L; n_fail <- 0L

cat("=== Split 5D aliasing test ===\n\n")
for (t in tests) {
  cat(sprintf("%-50s\n", t$name))
  path <- tryCatch(make_split_model(t$dims, t$axis, t$n), error = function(e) {
    cat(sprintf("  [SKIP] model build failed: %s\n", conditionMessage(e)))
    NULL
  })
  if (is.null(path)) { n_fail <- n_fail + 1L; next }

  cpu_out <- tryCatch(run_split(path, t$dims, t$n, "cpu"),
                      error = function(e) { cat("  [ERR cpu]", conditionMessage(e), "\n"); NULL })
  gpu_out <- tryCatch(run_split(path, t$dims, t$n, "vulkan"),
                      error = function(e) { cat("  [ERR gpu]", conditionMessage(e), "\n"); NULL })

  if (is.null(cpu_out) || is.null(gpu_out)) { n_fail <- n_fail + 1L; next }

  ok <- cmp(cpu_out, gpu_out, t$name)
  if (ok) n_pass <- n_pass + 1L else n_fail <- n_fail + 1L
  cat("\n")
}

cat(sprintf("%d PASS   %d FAIL\n", n_pass, n_fail))
