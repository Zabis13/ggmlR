# ============================================================================
# mlr3 integration example for ggmlR — CPU vs GPU comparison
#
# Run from R: source(system.file("examples/mlr3_integration.R", package = "ggmlR"))
# ============================================================================

library(ggmlR)
library(mlr3)

ggmlR:::.register_mlr3()

# ── Helper ───────────────────────────────────────────────────────────────────
run_classif <- function(backend) {
  task <- tsk("iris")
  learner <- lrn("classif.ggml",
    hidden_layers = c(32, 16),
    activation    = "relu",
    dropout       = 0.1,
    epochs        = 200,
    batch_size    = 16,
    optimizer     = "adam",
    backend       = backend,
    verbose       = 0,
    predict_type  = "prob"
  )
  pipe <- learner
  split <- partition(task, ratio = 0.8)

  elapsed <- system.time({
    pipe$train(task, row_ids = split$train)
    pred <- pipe$predict(task, row_ids = split$test)
  })

  list(
    acc     = pred$score(msr("classif.acc")),
    elapsed = elapsed[["elapsed"]]
  )
}

run_regr <- function(backend) {
  task <- as_task_regr(mtcars, target = "mpg")
  learner <- lrn("regr.ggml",
    hidden_layers = c(32, 16),
    activation    = "relu",
    dropout       = 0.0,
    epochs        = 200,
    batch_size    = 8,
    optimizer     = "adam",
    backend       = backend
  )
  pipe <- learner
  split <- partition(task, ratio = 0.8)

  elapsed <- system.time({
    pipe$train(task, row_ids = split$train)
    pred <- pipe$predict(task, row_ids = split$test)
  })

  list(
    rmse    = pred$score(msr("regr.rmse")),
    elapsed = elapsed[["elapsed"]]
  )
}

# ── Classification ───────────────────────────────────────────────────────────
cat("\n── Classification: iris ────────────────────────────────────────────────\n")

cat("Running on CPU...\n")
cpu_cls <- run_classif("cpu")

cat("Running on GPU...\n")
gpu_cls <- run_classif("gpu")

cat(sprintf("\n  %-8s  acc=%.4f  time=%.2fs\n", "CPU", cpu_cls$acc, cpu_cls$elapsed))
cat(sprintf("  %-8s  acc=%.4f  time=%.2fs\n",   "GPU", gpu_cls$acc, gpu_cls$elapsed))
cat(sprintf("  Speedup GPU/CPU: %.2fx\n", cpu_cls$elapsed / gpu_cls$elapsed))

# ── Regression ───────────────────────────────────────────────────────────────
cat("\n── Regression: mtcars ──────────────────────────────────────────────────\n")

cat("Running on CPU...\n")
cpu_reg <- run_regr("cpu")

cat("Running on GPU...\n")
gpu_reg <- run_regr("gpu")

cat(sprintf("\n  %-8s  RMSE=%.4f  time=%.2fs\n", "CPU", cpu_reg$rmse, cpu_reg$elapsed))
cat(sprintf("  %-8s  RMSE=%.4f  time=%.2fs\n",   "GPU", gpu_reg$rmse, gpu_reg$elapsed))
cat(sprintf("  Speedup GPU/CPU: %.2fx\n", cpu_reg$elapsed / gpu_reg$elapsed))

# ── Resampling 3-fold CV (CPU) ───────────────────────────────────────────────
cat("\n── Resampling: 3-fold CV on iris (CPU) ─────────────────────────────────\n")
task <- tsk("iris")
learner <- lrn("classif.ggml",
  epochs = 200, batch_size = 16, backend = "cpu", verbose = 0
)
pipe <- learner
elapsed_cv <- system.time(
  rr <- resample(task, pipe, rsmp("cv", folds = 3))
)[["elapsed"]]
cat(sprintf("  CV acc=%.4f  time=%.2fs\n", rr$aggregate(msr("classif.acc")), elapsed_cv))
