# ============================================================================
# tidymodels / parsnip integration example for ggmlR — CPU vs GPU comparison
#
# Run from R: source(system.file("examples/tidymodels_integration.R", package = "ggmlR"))
# ============================================================================

library(ggmlR)
library(parsnip)

# ── Helper: standardize numeric features using training stats ────────────────
scale_train_test <- function(train_x, test_x) {
  mu  <- colMeans(train_x)
  sdv <- apply(train_x, 2, stats::sd)
  sdv[sdv == 0] <- 1
  list(
    train = scale(train_x, center = mu, scale = sdv),
    test  = scale(test_x,  center = mu, scale = sdv)
  )
}

# ── Classification: iris ─────────────────────────────────────────────────────
set.seed(42)
idx_cls   <- sample(seq_len(nrow(iris)), size = floor(0.8 * nrow(iris)))
train_cls <- iris[idx_cls, ]
test_cls  <- iris[-idx_cls, ]

sc <- scale_train_test(as.matrix(train_cls[, 1:4]), as.matrix(test_cls[, 1:4]))
train_cls[, 1:4] <- sc$train
test_cls[,  1:4] <- sc$test

run_classif <- function(backend) {
  spec <- mlp(
    hidden_units = 32,
    epochs       = 200,
    dropout      = 0.1,
    activation   = "relu",
    learn_rate   = 0.01
  ) |>
    set_engine("ggml", batch_size = 16, backend = backend, verbose = 0) |>
    set_mode("classification")

  elapsed <- system.time({
    fit  <- fit(spec, Species ~ ., data = train_cls)
    pred <- predict(fit, test_cls)$.pred_class
  })

  list(
    acc     = mean(pred == test_cls$Species),
    prob    = predict(fit, test_cls, type = "prob"),
    elapsed = elapsed[["elapsed"]]
  )
}

cat("\n── Classification: iris ────────────────────────────────────────────────\n")
cat("Running on CPU...\n"); cpu_cls <- run_classif("cpu")
cat("Running on GPU...\n"); gpu_cls <- run_classif("gpu")

cat(sprintf("\n  %-8s  acc=%.4f  time=%.2fs\n", "CPU", cpu_cls$acc, cpu_cls$elapsed))
cat(sprintf("  %-8s  acc=%.4f  time=%.2fs\n",   "GPU", gpu_cls$acc, gpu_cls$elapsed))
cat(sprintf("  Speedup GPU/CPU: %.2fx\n", cpu_cls$elapsed / gpu_cls$elapsed))
cat("\nGPU predicted probabilities (head):\n")
print(head(gpu_cls$prob))

# ── Regression: mtcars ───────────────────────────────────────────────────────
set.seed(42)
idx_reg   <- sample(seq_len(nrow(mtcars)), size = floor(0.8 * nrow(mtcars)))
train_reg <- mtcars[idx_reg, ]
test_reg  <- mtcars[-idx_reg, ]

feat_cols <- setdiff(names(mtcars), "mpg")
sc <- scale_train_test(
  as.matrix(train_reg[, feat_cols]),
  as.matrix(test_reg[,  feat_cols])
)
train_reg[, feat_cols] <- sc$train
test_reg[,  feat_cols] <- sc$test

run_regr <- function(backend) {
  spec <- mlp(
    hidden_units = 32,
    epochs       = 200,
    dropout      = 0.0,
    activation   = "relu",
    learn_rate   = 0.01
  ) |>
    set_engine("ggml", batch_size = 8, backend = backend) |>
    set_mode("regression")

  elapsed <- system.time({
    fit  <- fit(spec, mpg ~ ., data = train_reg)
    pred <- predict(fit, test_reg)$.pred
  })

  list(
    rmse    = sqrt(mean((pred - test_reg$mpg)^2)),
    elapsed = elapsed[["elapsed"]]
  )
}

cat("\n── Regression: mtcars ──────────────────────────────────────────────────\n")
cat("Running on CPU...\n"); cpu_reg <- run_regr("cpu")
cat("Running on GPU...\n"); gpu_reg <- run_regr("gpu")

cat(sprintf("\n  %-8s  RMSE=%.4f  time=%.2fs\n", "CPU", cpu_reg$rmse, cpu_reg$elapsed))
cat(sprintf("  %-8s  RMSE=%.4f  time=%.2fs\n",   "GPU", gpu_reg$rmse, gpu_reg$elapsed))
cat(sprintf("  Speedup GPU/CPU: %.2fx\n", cpu_reg$elapsed / gpu_reg$elapsed))
