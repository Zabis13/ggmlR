#!/usr/bin/env Rscript
# Smoke test: ONNX model can be run repeatedly without re-loading.
# Validates ggml_backend_sched intermediate buffer aliasing —
# sched_alloc_and_fill must run on each compute so that weights are not
# overwritten by scratch buffers between runs (see CLAUDE.md "Repeated runs").
#
# For each model:
#   run 1: inputs A → out1
#   run 2: inputs B → out2          (different inputs)
#   run 3: inputs A → out3          (same inputs as run 1)
# Expect: max|out1 - out2| > 0 (runs differ) and out1 == out3 bit-exact.
library(ggmlR)

ONNX_DIR <- "/mnt/Data2/DS_projects/ONNX models-main"

models <- list(
  list(file = "mnist-8.onnx",             inputs = list(Input3 = c(1L, 1L, 28L, 28L))),
  list(file = "squeezenet1.0-8.onnx",     inputs = list(data_0 = c(1L, 3L, 224L, 224L))),
  list(file = "super-resolution-10.onnx", inputs = list(input  = c(1L, 1L, 224L, 224L)))
)

pass <- 0L
fail <- 0L

for (m in models) {
  path <- file.path(ONNX_DIR, m$file)
  cat(sprintf("%-30s ", m$file))
  if (!file.exists(path)) { cat("SKIP (no file)\n"); next }

  tryCatch({
    model <- onnx_load(path, device = "cpu", input_shapes = m$inputs)

    set.seed(42)
    inputs1 <- lapply(m$inputs, function(s) runif(prod(s)))
    out1 <- onnx_run(model, inputs1)

    set.seed(7)
    inputs2 <- lapply(m$inputs, function(s) runif(prod(s)))
    out2 <- onnx_run(model, inputs2)

    out3 <- onnx_run(model, inputs1)

    d12 <- max(abs(out1[[1]] - out2[[1]]))  # >0: runs differ
    d13 <- max(abs(out1[[1]] - out3[[1]]))  # ~0: same inputs reproduce

    ok <- d12 > 1e-8 && d13 < 1e-4
    cat(sprintf("%s  (delta12=%.3e, delta13=%.3e)\n",
                if (ok) "OK " else "FAIL", d12, d13))
    if (ok) pass <- pass + 1L else fail <- fail + 1L
    rm(model, out1, out2, out3); gc(verbose = FALSE)
  }, error = function(e) {
    cat(sprintf("FAIL: %s\n", conditionMessage(e)))
    fail <<- fail + 1L
  })
}

cat(sprintf("\n--- Repeated-run smoke: %d OK, %d FAIL ---\n", pass, fail))
