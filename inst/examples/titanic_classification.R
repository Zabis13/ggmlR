# Titanic survival prediction with ggmlR
# Сравнение 10 вариантов классификации:
#
# Sequential API:
#   1. Shallow (1 слой, SGD)
#   2. Deep + dropout (adam)
#   3. Deep + BatchNorm (adam)
#
# Autograd API:
#   4. ag_sequential + ручной early stopping (best-weights restore)
#   5. ag_sequential + adam + cosine LR scheduler
#   8. ag_sequential + SGD + momentum
#   9. ag_sequential + adam + cosine + clip_grad_norm + ag_dataloader + batch_norm
#  10. Голые ag_param (без ag_sequential) + dp_train
#
# Functional API:
#   6. Functional DAG + BatchNorm
#   7. Functional, два входа: числовые + Title one-hot (add merge)
#
# Dataset: https://www.kaggle.com/c/titanic

library(ggmlR)

# ============================================================================
# 1. Загрузка данных
# ============================================================================

train_data <- read.csv("/mnt/Data2/DS_Data/titanic/train.csv",
                       stringsAsFactors = FALSE)
test_data  <- read.csv("/mnt/Data2/DS_Data/titanic/test.csv",
                       stringsAsFactors = FALSE)

# ============================================================================
# 2. Feature engineering
# ============================================================================

prep_features <- function(df, train_df = NULL) {
  ref <- if (is.null(train_df)) df else train_df

  df$Age[is.na(df$Age)]   <- median(ref$Age,  na.rm = TRUE)
  df$Fare[is.na(df$Fare)] <- median(ref$Fare, na.rm = TRUE)
  df$Embarked[df$Embarked == "" | is.na(df$Embarked)] <- "S"

  df$Title <- gsub(".*,\\s*(\\w+)\\..*", "\\1", df$Name)
  df$Title <- ifelse(df$Title == "Mr",                     "Mr",
              ifelse(df$Title %in% c("Mrs","Mme","Ms"),    "Mrs",
              ifelse(df$Title %in% c("Miss","Mlle"),       "Miss",
              ifelse(df$Title == "Master",                 "Master",
                                                           "Rare"))))

  df$FamilySize <- df$SibSp + df$Parch + 1L
  df$IsAlone    <- as.integer(df$FamilySize == 1L)
  df$Sex        <- as.integer(df$Sex == "male")
  df$Embarked   <- as.integer(factor(df$Embarked, levels = c("S","C","Q"))) - 1L
  # TitleIdx: 0-based для embedding в варианте 7
  df$TitleIdx   <- as.integer(factor(df$Title,
                     levels = c("Mr","Mrs","Miss","Master","Rare"))) - 1L

  df[, c("Pclass","Sex","Age","SibSp","Parch","Fare",
         "Embarked","FamilySize","IsAlone","TitleIdx")]
}

x_raw      <- prep_features(train_data)
x_test_raw <- prep_features(test_data, train_df = train_data)

# Числовые фичи (без TitleIdx) — масштабируем
x_scaled     <- scale(x_raw[, -10])
scale_center <- attr(x_scaled, "scaled:center")
scale_scale  <- attr(x_scaled, "scaled:scale")

title_train <- x_raw$TitleIdx
title_test  <- x_test_raw$TitleIdx

x_num      <- as.matrix(x_scaled)
x_test_num <- as.matrix(scale(x_test_raw[, -10],
                               center = scale_center, scale = scale_scale))

# Полная матрица [числовые + TitleIdx/4] для моделей 1-6, 8-10
x      <- cbind(x_num, TitleIdx = title_train / 4.0)
x_test <- cbind(x_test_num, TitleIdx = title_test / 4.0)

survived <- as.integer(train_data$Survived)
y <- cbind(survived, 1L - survived) * 1.0   # one-hot [N x 2]

# ============================================================================
# 3. Стратифицированный split
# ============================================================================

set.seed(42)
idx_surv  <- which(survived == 1L)
idx_dead  <- which(survived == 0L)
val_surv  <- sample(idx_surv, size = floor(0.2 * length(idx_surv)))
val_dead  <- sample(idx_dead, size = floor(0.2 * length(idx_dead)))
idx_val   <- sort(c(val_surv, val_dead))
idx_train <- sort(setdiff(seq_len(nrow(x)), idx_val))

x_train <- x[idx_train, , drop = FALSE]
y_train <- y[idx_train, , drop = FALSE]
x_val   <- x[idx_val,   , drop = FALSE]
y_val   <- y[idx_val,   , drop = FALSE]

cat(sprintf("train: %d  val: %d  features: %d\n",
            nrow(x_train), nrow(x_val), ncol(x)))

# ============================================================================
# Вспомогательные функции
# ============================================================================

eval_metrics <- function(probs_col1, true_col1, label) {
  pred <- ifelse(probs_col1 > 0.5, 1L, 0L)
  true <- as.integer(true_col1)
  acc  <- mean(pred == true)
  eps  <- 1e-7
  p    <- pmin(pmax(probs_col1, eps), 1 - eps)
  loss <- -mean(true * log(p) + (1 - true) * log(1 - p))
  tp   <- sum(pred == 1L & true == 1L)
  tn   <- sum(pred == 0L & true == 0L)
  fp   <- sum(pred == 1L & true == 0L)
  fn   <- sum(pred == 0L & true == 1L)
  prec <- tp / (tp + fp + 1e-9)
  rec  <- tp / (tp + fn + 1e-9)
  f1   <- 2 * prec * rec / (prec + rec + 1e-9)
  cat(sprintf("\n[%s]  acc=%.4f (%.1f%%)  F1=%.4f  loss=%.4f\n",
              label, acc, acc * 100, f1, loss))
  cat(sprintf("  TP=%d TN=%d FP=%d FN=%d\n", tp, tn, fp, fn))
  invisible(list(acc = acc, loss = loss, f1 = f1))
}

# ag inference: forward в чанках, ручной softmax
ag_predict_colmajor <- function(model, x_col_major) {
  n   <- ncol(x_col_major)
  out <- matrix(0.0, 2L, n)
  ch  <- 64L
  for (s in seq(1L, n, by = ch)) {
    e  <- min(s + ch - 1L, n)
    xc <- ag_tensor(x_col_major[, s:e, drop = FALSE])
    lg <- model$forward(xc)$data
    ev <- exp(lg - apply(lg, 2, max))
    out[, s:e] <- ev / colSums(ev)
  }
  out
}

results <- list()
N_FEAT  <- ncol(x)   # 10

# ============================================================================
# 1. Sequential: shallow (1 скрытый слой) + SGD
# ============================================================================

cat("\n=== 1. Sequential: shallow + SGD ===\n")

m1 <- ggml_model_sequential() |>
  ggml_layer_dense(32L, activation = "relu", input_shape = N_FEAT) |>
  ggml_layer_dense(2L,  activation = "softmax") |>
  ggml_compile(optimizer = "sgd", loss = "categorical_crossentropy")

m1 <- ggml_fit(m1, x_train, y_train, epochs = 200L, batch_size = 42L, verbose = 0L)
p1 <- ggml_predict(m1, x_val, batch_size = 32L)
results[["1_seq_shallow_sgd"]] <- eval_metrics(p1[,1], y_val[,1], "1 seq shallow+SGD")

# ============================================================================
# 2. Sequential: deep + dropout + adam
# ============================================================================

cat("\n=== 2. Sequential: deep + dropout + adam ===\n")

m2 <- ggml_model_sequential() |>
  ggml_layer_dense(64L, activation = "relu", input_shape = N_FEAT) |>
  ggml_layer_dropout(0.3, stochastic = TRUE) |>
  ggml_layer_dense(32L, activation = "relu") |>
  ggml_layer_dropout(0.2, stochastic = TRUE) |>
  ggml_layer_dense(2L,  activation = "softmax") |>
  ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

m2 <- ggml_fit(m2, x_train, y_train, epochs = 150L, batch_size = 42L, verbose = 0L)
p2 <- ggml_predict(m2, x_val, batch_size = 32L)
results[["2_seq_deep_dropout"]] <- eval_metrics(p2[,1], y_val[,1], "2 seq deep+dropout+adam")

# ============================================================================
# 3. Sequential: deep + BatchNorm + adam
# ============================================================================

cat("\n=== 3. Sequential: deep + BatchNorm + adam ===\n")

m3 <- ggml_model_sequential() |>
  ggml_layer_dense(64L, activation = "relu", input_shape = N_FEAT) |>
  ggml_layer_batch_norm() |>
  ggml_layer_dense(32L, activation = "relu") |>
  ggml_layer_batch_norm() |>
  ggml_layer_dense(2L,  activation = "softmax") |>
  ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

m3 <- ggml_fit(m3, x_train, y_train, epochs = 150L, batch_size = 42L, verbose = 0L)
p3 <- ggml_predict(m3, x_val, batch_size = 32L)
results[["3_seq_batchnorm"]] <- eval_metrics(p3[,1], y_val[,1], "3 seq deep+BatchNorm+adam")

# ============================================================================
# Общие данные для autograd вариантов: col-major [features x N]
# ============================================================================

x_tr_ag <- t(x_train)   # [10, n_train]
y_tr_ag <- t(y_train)   # [2,  n_train]
x_vl_ag <- t(x_val)
n_tr    <- ncol(x_tr_ag)
BS      <- 32L

# ============================================================================
# 4. Autograd: ag_sequential + ручной early stopping по val_loss
# ============================================================================
# Демонстрирует гибкость autograd API: мониторим val_loss после каждой эпохи,
# восстанавливаем лучшие веса (best-weights restore) при остановке.
# ============================================================================

cat("\n=== 4. Autograd: ag_sequential + manual early stopping ===\n")

m4 <- ag_sequential(
  ag_linear(N_FEAT, 64L, activation = "relu"),
  ag_dropout(0.3),
  ag_linear(64L, 32L, activation = "relu"),
  ag_dropout(0.2),
  ag_linear(32L, 2L)
)
params4 <- m4$parameters()
opt4    <- optimizer_adam(params4, lr = 1e-3)

# val_loss helper: binary cross-entropy на val без gradient tape
val_loss_fn <- function(model, x_cm, y_cm) {
  ag_eval(model)
  p <- ag_predict_colmajor(model, x_cm)   # [2 x n_val]
  eps <- 1e-7
  p   <- pmin(pmax(p, eps), 1 - eps)
  # y_cm[1,] = survived prob target
  -mean(y_cm[1,] * log(p[1,]) + y_cm[2,] * log(p[2,]))
}

# Early stopping state
patience       <- 25L
min_delta      <- 1e-4
best_val_loss  <- Inf
best_weights   <- NULL   # snapshot параметров
wait           <- 0L
stopped_epoch  <- NA_integer_

x_vl_ag4 <- t(x_val)
y_vl_ag4 <- t(y_val)

ag_train(m4)
set.seed(42)
for (ep in seq_len(300L)) {
  perm <- sample(n_tr)
  for (b in seq_len(ceiling(n_tr / BS))) {
    idx <- perm[((b-1L)*BS+1L):min(b*BS, n_tr)]
    xb  <- ag_tensor(x_tr_ag[, idx, drop = FALSE])
    yb  <- y_tr_ag[, idx, drop = FALSE]
    with_grad_tape({ loss4 <- ag_softmax_cross_entropy_loss(m4$forward(xb), yb) })
    grads <- backward(loss4)
    opt4$step(grads)
    opt4$zero_grad()
  }

  vl <- val_loss_fn(m4, x_vl_ag4, y_vl_ag4)
  ag_train(m4)

  if (vl < best_val_loss - min_delta) {
    best_val_loss <- vl
    best_weights  <- lapply(params4, function(p) p$data)
    wait          <- 0L
  } else {
    wait <- wait + 1L
    if (wait >= patience) {
      stopped_epoch <- ep
      break
    }
  }
}

if (!is.na(stopped_epoch)) {
  cat(sprintf("  Early stop at epoch %d  best_val_loss=%.4f\n",
              stopped_epoch, best_val_loss))
} else {
  cat(sprintf("  Completed 300 epochs  best_val_loss=%.4f\n", best_val_loss))
}

# Restore best weights
for (nm in names(params4)) params4[[nm]]$data <- best_weights[[nm]]

ag_eval(m4)
p4 <- ag_predict_colmajor(m4, x_vl_ag4)
results[["4_ag_early_stop"]] <- eval_metrics(p4[1L,], y_val[,1], "4 ag early stopping (manual)")

# ============================================================================
# 5. Autograd: ag_sequential + adam + cosine LR scheduler
# ============================================================================

cat("\n=== 5. Autograd: ag_sequential + adam + cosine LR scheduler ===\n")

m5 <- ag_sequential(
  ag_linear(N_FEAT, 64L, activation = "relu"),
  ag_dropout(0.3),
  ag_linear(64L, 32L, activation = "relu"),
  ag_dropout(0.2),
  ag_linear(32L, 2L)
)
params5 <- m5$parameters()
opt5    <- optimizer_adam(params5, lr = 1e-3)
sch5    <- lr_scheduler_cosine(opt5, T_max = 200L, lr_min = 1e-5)

ag_train(m5)
set.seed(42)
for (ep in seq_len(200L)) {
  perm <- sample(n_tr)
  for (b in seq_len(ceiling(n_tr / BS))) {
    idx <- perm[((b-1L)*BS+1L):min(b*BS, n_tr)]
    xb  <- ag_tensor(x_tr_ag[, idx, drop = FALSE])
    yb  <- y_tr_ag[, idx, drop = FALSE]
    with_grad_tape({ loss5 <- ag_softmax_cross_entropy_loss(m5$forward(xb), yb) })
    grads <- backward(loss5)
    opt5$step(grads)
    opt5$zero_grad()
  }
  sch5$step()
}
ag_eval(m5)
p5 <- ag_predict_colmajor(m5, x_vl_ag)
results[["5_ag_adam_cosine"]] <- eval_metrics(p5[1L,], y_val[,1], "5 ag adam+cosine scheduler")

# ============================================================================
# 6. Functional API: DAG + BatchNorm
# ============================================================================

cat("\n=== 6. Functional: DAG + BatchNorm ===\n")

inp6 <- ggml_input(shape = N_FEAT)
h6   <- inp6 |> ggml_layer_dense(64L, activation = "relu") |> ggml_layer_batch_norm()
h6   <- h6   |> ggml_layer_dense(32L, activation = "relu") |> ggml_layer_batch_norm()
out6 <- h6   |> ggml_layer_dense(2L,  activation = "softmax")

m6 <- ggml_model(inputs = inp6, outputs = out6) |>
  ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

m6 <- ggml_fit(m6, x_train, y_train, epochs = 150L, batch_size = 42L, verbose = 0L)
p6 <- ggml_predict(m6, x_val, batch_size = 32L)
results[["6_functional_batchnorm"]] <- eval_metrics(p6[,1], y_val[,1], "6 functional DAG+BatchNorm")

# ============================================================================
# 7. Functional API: два входа — числовые + Title (shared encoder)
# ============================================================================
# Числовые фичи (9) и Title one-hot (5) — две независимые ветки dense(32),
# результаты складываются (add) и подаются в выходной слой.
# ============================================================================

cat("\n=== 7. Functional: два входа (числовые + Title one-hot, shared encoder) ===\n")

x_num_train <- x_train[, -N_FEAT, drop = FALSE]   # [N x 9]
x_num_val   <- x_val[,   -N_FEAT, drop = FALSE]

n_titles <- 5L
onehot <- function(idx, n) {
  m <- matrix(0.0, nrow = length(idx), ncol = n)
  m[cbind(seq_along(idx), idx + 1L)] <- 1.0
  m
}
title_oh_train <- onehot(title_train[idx_train], n_titles)
title_oh_val   <- onehot(title_train[idx_val],   n_titles)

inp_num   <- ggml_input(shape = 9L)
inp_title <- ggml_input(shape = n_titles)

# Две независимые ветки → одинаковая размерность 32 → сложение → выход
h_num   <- inp_num   |> ggml_layer_dense(32L, activation = "relu")
h_title <- inp_title |> ggml_layer_dense(32L, activation = "relu")

out7 <- ggml_layer_add(list(h_num, h_title)) |> ggml_layer_dense(2L, activation = "softmax")

m7 <- ggml_model(inputs  = list(inp_num, inp_title),
                 outputs = out7) |>
  ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

m7 <- ggml_fit(
  m7,
  x          = list(x_num_train, title_oh_train),
  y          = y_train,
  epochs     = 150L,
  batch_size = 42L,
  verbose    = 0L
)
p7 <- ggml_predict(m7,
                   x          = list(x_num_val, title_oh_val),
                   batch_size = 32L)
results[["7_functional_2inputs"]] <- eval_metrics(p7[,1], y_val[,1], "7 functional 2-inputs+shared")

# ============================================================================
# 8. Autograd: ag_sequential + SGD + momentum
# ============================================================================

cat("\n=== 8. Autograd: ag_sequential + SGD + momentum ===\n")

m8 <- ag_sequential(
  ag_linear(N_FEAT, 64L, activation = "relu"),
  ag_dropout(0.3),
  ag_linear(64L, 32L, activation = "relu"),
  ag_dropout(0.2),
  ag_linear(32L, 2L)
)
opt8 <- optimizer_sgd(m8$parameters(), lr = 0.05, momentum = 0.9)
ag_train(m8)
set.seed(42)
for (ep in seq_len(200L)) {
  perm <- sample(n_tr)
  for (b in seq_len(ceiling(n_tr / BS))) {
    idx <- perm[((b-1L)*BS+1L):min(b*BS, n_tr)]
    xb  <- ag_tensor(x_tr_ag[, idx, drop = FALSE])
    yb  <- y_tr_ag[, idx, drop = FALSE]
    with_grad_tape({ loss8 <- ag_softmax_cross_entropy_loss(m8$forward(xb), yb) })
    grads <- backward(loss8)
    opt8$step(grads)
    opt8$zero_grad()
  }
}
ag_eval(m8)
p8 <- ag_predict_colmajor(m8, x_vl_ag)
results[["8_ag_sgd_momentum"]] <- eval_metrics(p8[1L,], y_val[,1], "8 ag SGD+momentum")

# ============================================================================
# 9. Autograd: ag_sequential + adam + cosine scheduler + clip_grad + dataloader
# ============================================================================

cat("\n=== 9. Autograd: adam + cosine scheduler + clip_grad_norm + dataloader ===\n")

m9 <- ag_sequential(
  ag_linear(N_FEAT, 64L, activation = "relu"),
  ag_batch_norm(64L),
  ag_dropout(0.3),
  ag_linear(64L, 32L, activation = "relu"),
  ag_batch_norm(32L),
  ag_dropout(0.2),
  ag_linear(32L, 2L)
)
params9 <- m9$parameters()
opt9    <- optimizer_adam(params9, lr = 1e-3)
sch9    <- lr_scheduler_cosine(opt9, T_max = 150L, lr_min = 1e-5)
dl9     <- ag_dataloader(x_tr_ag, y_tr_ag, batch_size = BS, shuffle = TRUE)

ag_train(m9)
set.seed(42)
for (ep in seq_len(150L)) {
  for (batch in dl9$epoch()) {
    with_grad_tape({
      loss9 <- ag_softmax_cross_entropy_loss(m9$forward(batch$x), batch$y$data)
    })
    grads <- backward(loss9)
    clip_grad_norm(params9, grads, max_norm = 5.0)
    opt9$step(grads)
    opt9$zero_grad()
  }
  sch9$step()
}
ag_eval(m9)
p9 <- ag_predict_colmajor(m9, x_vl_ag)
results[["9_ag_adam_cosine_clip"]] <- eval_metrics(p9[1L,], y_val[,1], "9 ag adam+cosine+clip")

# ============================================================================
# 10. Autograd: голые ag_param (без ag_sequential) + dp_train
# ============================================================================
# make_model строит сеть вручную через W/b, dp_train управляет циклом.
# ============================================================================

cat("\n=== 10. Autograd: голые ag_param + dp_train ===\n")

dp_data <- lapply(seq_len(n_tr), function(i)
  list(x = x_tr_ag[, i, drop = FALSE],
       y = y_tr_ag[, i, drop = FALSE]))

make_model10 <- function() {
  W1 <- ag_param(matrix(rnorm(64L * N_FEAT) * sqrt(2.0 / N_FEAT), 64L, N_FEAT))
  b1 <- ag_param(matrix(0.0, 64L, 1L))
  W2 <- ag_param(matrix(rnorm(32L * 64L)    * sqrt(2.0 / 64L),    32L, 64L))
  b2 <- ag_param(matrix(0.0, 32L, 1L))
  W3 <- ag_param(matrix(rnorm(2L  * 32L)    * sqrt(2.0 / 32L),    2L,  32L))
  b3 <- ag_param(matrix(0.0, 2L,  1L))

  list(
    forward    = function(x) {
      h <- ag_relu(ag_add(ag_matmul(W1, x), b1))
      h <- ag_relu(ag_add(ag_matmul(W2, h), b2))
      ag_add(ag_matmul(W3, h), b3)
    },
    parameters = function() list(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
  )
}

set.seed(42)
res10 <- dp_train(
  make_model  = make_model10,
  data        = dp_data,
  loss_fn     = function(out, tgt) ag_softmax_cross_entropy_loss(out, tgt),
  forward_fn  = function(model, s) model$forward(ag_tensor(s$x)),
  target_fn   = function(s) s$y,
  n_gpu       = 1L,
  n_iter      = 5000L,   # ~7 эпох по n_tr сэмплам
  lr          = 5e-4,
  max_norm    = 5.0,
  verbose     = FALSE
)

m10 <- res10$model
ag_eval(m10)
p10 <- ag_predict_colmajor(m10, x_vl_ag)
results[["10_ag_raw_dp_train"]] <- eval_metrics(p10[1L,], y_val[,1], "10 ag raw ag_param+dp_train")

# ============================================================================
# Итоговая таблица
# ============================================================================

cat("\n")
cat(strrep("=", 60), "\n", sep = "")
cat(sprintf("%-40s  %s  %s\n", "Вариант", "Accuracy", "F1"))
cat(strrep("-", 60), "\n", sep = "")
for (nm in names(results)) {
  r <- results[[nm]]
  cat(sprintf("%-40s  %.4f   %.4f\n", nm, r$acc, r$f1))
}
cat(strrep("=", 60), "\n", sep = "")

best_nm  <- names(which.max(sapply(results, `[[`, "acc")))
best_acc <- results[[best_nm]]$acc
cat(sprintf("\nЛучшая модель: %s (acc=%.4f)\n", best_nm, best_acc))

# ============================================================================
# Submission: предикт лучшей модели на test_data
# ============================================================================

x_test_col <- t(x_test)   # col-major для ag моделей

pred_labels <- switch(best_nm,
  "1_seq_shallow_sgd" = {
    p <- ggml_predict(m1, x_test, batch_size = 32L)
    ifelse(p[,1] > 0.5, 1L, 0L)
  },
  "2_seq_deep_dropout" = {
    p <- ggml_predict(m2, x_test, batch_size = 32L)
    ifelse(p[,1] > 0.5, 1L, 0L)
  },
  "3_seq_batchnorm" = {
    p <- ggml_predict(m3, x_test, batch_size = 32L)
    ifelse(p[,1] > 0.5, 1L, 0L)
  },
  "4_ag_early_stop" = {
    ag_eval(m4)
    p <- ag_predict_colmajor(m4, x_test_col)
    ifelse(p[1L,] > 0.5, 1L, 0L)
  },
  "5_ag_adam_cosine" = {
    ag_eval(m5)
    p <- ag_predict_colmajor(m5, x_test_col)
    ifelse(p[1L,] > 0.5, 1L, 0L)
  },
  "6_functional_batchnorm" = {
    p <- ggml_predict(m6, x_test, batch_size = 32L)
    ifelse(p[,1] > 0.5, 1L, 0L)
  },
  "7_functional_2inputs" = {
    title_oh_test <- onehot(title_test, n_titles)
    p <- ggml_predict(m7, x = list(x_test_num, title_oh_test), batch_size = 32L)
    ifelse(p[,1] > 0.5, 1L, 0L)
  },
  "8_ag_sgd_momentum" = {
    ag_eval(m8)
    p <- ag_predict_colmajor(m8, x_test_col)
    ifelse(p[1L,] > 0.5, 1L, 0L)
  },
  "9_ag_adam_cosine_clip" = {
    ag_eval(m9)
    p <- ag_predict_colmajor(m9, x_test_col)
    ifelse(p[1L,] > 0.5, 1L, 0L)
  },
  "10_ag_raw_dp_train" = {
    ag_eval(m10)
    p <- ag_predict_colmajor(m10, x_test_col)
    ifelse(p[1L,] > 0.5, 1L, 0L)
  },
  stop("Unknown best model: ", best_nm)
)

submission_csv <- file.path(tempdir(), "submission.csv")
write.csv(
  data.frame(PassengerId = test_data$PassengerId, Survived = pred_labels),
  submission_csv, row.names = FALSE
)
cat(sprintf("Submission (%s): %d rows → %s  (survival rate %.1f%%)\n",
            best_nm, length(pred_labels), submission_csv, 100 * mean(pred_labels)))
