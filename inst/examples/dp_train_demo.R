# Data-Parallel Training Demo (dp_train)
#
# Демонстрирует честный data-parallel: каждая реплика получает
# свой уникальный сэмпл на каждой итерации (effective batch = n_replicas).
#
#   - dp_train() с 1, 2 и N репликами (авто-детект GPU)
#   - Синхронизацию начальных весов между репликами
#   - Обучение MLP на синтетической регрессионной задаче
#   - clip_grad_norm внутри dp_train
#   - Встроенные проверки в конце файла
#
# Запуск:
#   Rscript inst/examples/dp_train_demo.R
#
# Тензорный макет: [features, 1] — один сэмпл за шаг на реплику.

library(ggmlR)

cat("ggmlR version:", ggml_version(), "\n")

n_avail <- tryCatch(ggml_vulkan_device_count(), error = function(e) 0L)
cat(sprintf("Vulkan GPUs available: %d\n", n_avail))

# =============================================================================
# 0.  Гиперпараметры
# =============================================================================

set.seed(42L)

D_IN   <- 4L    # входных признаков
D_HID  <- 16L   # скрытый слой
D_OUT  <- 2L    # выходов
N      <- 128L  # сэмплов в датасете
N_ITER <- 100L
LR     <- 5e-3

# =============================================================================
# 1.  Синтетический датасет  y = W_true * x + noise
# =============================================================================

W_true <- matrix(c(0.5, -0.3, 0.8, -0.1,
                   0.2,  0.6, -0.4, 0.9), D_OUT, D_IN)

X <- matrix(rnorm(D_IN * N), D_IN, N)
Y <- W_true %*% X + matrix(rnorm(D_OUT * N, sd = 0.05), D_OUT, N)

# Список сэмплов — каждый: list(x=[D_IN,1], y=[D_OUT,1])
dataset <- lapply(seq_len(N), function(i)
  list(x = X[, i, drop = FALSE],
       y = Y[, i, drop = FALSE]))

cat(sprintf("\nDataset: %d samples, x=[%d,1], y=[%d,1]\n", N, D_IN, D_OUT))
cat("True data-parallel: each replica sees a DIFFERENT sample per iteration.\n")
cat(sprintf("Effective batch size = n_replicas (e.g. 4 replicas => batch=4)\n\n"))

# =============================================================================
# 2.  Фабрика моделей  (Linear -> ReLU -> Linear)
# =============================================================================

make_model <- function() {
  W1 <- ag_param(matrix(rnorm(D_HID * D_IN)  * sqrt(2 / D_IN),  D_HID, D_IN))
  b1 <- ag_param(matrix(0, D_HID, 1L))
  W2 <- ag_param(matrix(rnorm(D_OUT * D_HID) * sqrt(2 / D_HID), D_OUT, D_HID))
  b2 <- ag_param(matrix(0, D_OUT, 1L))

  list(
    forward    = function(x) {
      h <- ag_relu(ag_add(ag_matmul(W1, x), b1))
      ag_add(ag_matmul(W2, h), b2)
    },
    parameters = function() list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)
  )
}

forward_fn <- function(model, s) model$forward(s$x)
target_fn  <- function(s)        s$y
loss_fn    <- function(out, tgt) ag_mse_loss(out, tgt)

# =============================================================================
# 3.  Тренировка: 1 реплика (CPU, batch=1)
# =============================================================================

cat("--- 1 replica  (CPU, effective batch = 1) ---\n")

result_1 <- dp_train(
  make_model = make_model,  data = dataset,
  loss_fn    = loss_fn,     forward_fn = forward_fn,  target_fn = target_fn,
  n_gpu      = 1L,          n_iter = N_ITER,  lr = LR,  max_norm = 5.0,
  verbose    = 25L
)
cat(sprintf("Loss: %.4f → %.4f\n\n",
            result_1$loss_history[1], tail(result_1$loss_history, 1)))

# =============================================================================
# 4.  Тренировка: 2 реплики (GPU если есть, иначе CPU)
# =============================================================================

n_rep2 <- 2L
cat(sprintf("--- 2 replicas (%s, effective batch = 2) ---\n",
            if (n_avail >= 1L) "GPU" else "CPU"))

result_2 <- dp_train(
  make_model = make_model,  data = dataset,
  loss_fn    = loss_fn,     forward_fn = forward_fn,  target_fn = target_fn,
  n_gpu      = n_rep2,      n_iter = N_ITER,  lr = LR,  max_norm = 5.0,
  verbose    = 25L
)
cat(sprintf("Loss: %.4f → %.4f\n\n",
            result_2$loss_history[1], tail(result_2$loss_history, 1)))

# =============================================================================
# 5.  Тренировка: N реплик по числу GPU (или 4 CPU-реплики если GPU нет)
# =============================================================================

n_repN <- if (n_avail >= 2L) n_avail else 4L
cat(sprintf("--- %d replica(s) (%s, effective batch = %d) ---\n",
            n_repN, if (n_avail >= 2L) "multi-GPU" else "CPU", n_repN))

result_N <- dp_train(
  make_model = make_model,  data = dataset,
  loss_fn    = loss_fn,     forward_fn = forward_fn,  target_fn = target_fn,
  n_gpu      = n_repN,      n_iter = N_ITER,  lr = LR,  max_norm = 5.0,
  verbose    = 25L
)
cat(sprintf("Loss: %.4f → %.4f\n\n",
            result_N$loss_history[1], tail(result_N$loss_history, 1)))

# =============================================================================
# 6.  Инференс обученной модели (replica 0 из последнего запуска)
# =============================================================================

cat("--- Inference (replica 0 from last run) ---\n")

x_new <- matrix(rnorm(D_IN * 4L), D_IN, 4L)
with_grad_tape({
  pred <- result_N$model$forward(ag_tensor(x_new))
})
cat("Predictions (4 samples):\n")
print(round(pred$data, 3))

# =============================================================================
# 7.  Встроенные проверки
# =============================================================================

cat("\n--- Checks ---\n")

ok <- TRUE
check <- function(cond, msg) {
  cat(sprintf("  %s  %s\n", if (cond) "PASS" else "FAIL", msg))
  if (!cond) ok <<- FALSE
}

third <- N_ITER %/% 3L

# длина loss_history
check(length(result_1$loss_history) == N_ITER, "loss_history length (1 replica)")
check(length(result_2$loss_history) == N_ITER, "loss_history length (2 replicas)")
check(length(result_N$loss_history) == N_ITER, sprintf("loss_history length (%d replicas)", n_repN))

# нет NaN/Inf
check(all(is.finite(result_1$loss_history)), "no NaN/Inf (1 replica)")
check(all(is.finite(result_2$loss_history)), "no NaN/Inf (2 replicas)")
check(all(is.finite(result_N$loss_history)), sprintf("no NaN/Inf (%d replicas)", n_repN))

# loss убывает (последняя треть < первой трети)
check(mean(tail(result_1$loss_history, third)) < mean(head(result_1$loss_history, third)),
      "loss decreased (1 replica)")
check(mean(tail(result_2$loss_history, third)) < mean(head(result_2$loss_history, third)),
      "loss decreased (2 replicas)")
check(mean(tail(result_N$loss_history, third)) < mean(head(result_N$loss_history, third)),
      sprintf("loss decreased (%d replicas)", n_repN))

# структура результата
check(all(c("params", "loss_history", "model") %in% names(result_N)),
      "result has params / loss_history / model")

# каждая реплика видела разные данные — проверяем косвенно:
# при 2 репликах effective batch вдвое больше => сходимость за те же итерации
# должна быть не хуже чем при 1 реплике
check(mean(tail(result_2$loss_history, third)) <=
      mean(tail(result_1$loss_history, third)) * 1.5,
      "2-replica loss not worse than 1-replica (true DP sanity)")

# device восстановлен
check(ag_default_device() == "cpu", "device restored to CPU after dp_train")

# инференс конечен
check(all(is.finite(pred$data)), "inference output is finite")

cat(if (ok) "\nAll checks passed.\n" else "\nSome checks FAILED.\n")
if (!ok) quit(status = 1L)
