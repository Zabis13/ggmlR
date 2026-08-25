# Прототип перед работой над разделом «Трансформеры» в TODO.md.
#
# Проверяет ОБУЧЕНИЕМ (не сборкой) три вещи, от которых зависит порядок задач:
#   1. учится ли блок attention + нормализация + pos-embedding;
#   2. дифференцируется ли soft_max_ext с ненулевой маской, а не только
#      causal через diag_mask_inf;
#   3. работает ли батч с разной эффективной длиной последовательности.
#
# Forward строится и молчит, даже когда backward отсутствует, поэтому каждый
# пункт проверяется падением лосса, а не тем, что вызов не упал.
#
# Запуск:  Rscript inst/examples/proto_transformer_gaps.R

library(ggmlR)

S <- 6L    # длина последовательности
D <- 16L   # d_model
H <- 4L    # головы
N <- 64L   # примеров

ok   <- function(...) cat("  OK   ", sprintf(...), "\n")
fail <- function(...) cat("  FAIL ", sprintf(...), "\n")

cleanup <- function(model) {
  cp <- model$compilation
  if (is.null(cp)) return(invisible(NULL))
  if (!is.null(cp$buffer))      ggml_backend_buffer_free(cp$buffer)
  if (!is.null(cp$ctx_weights)) ggml_free(cp$ctx_weights)
  if (!is.null(cp$sched))       ggml_backend_sched_free(cp$sched)
  if (!is.null(cp$backend))     ggml_backend_free(cp$backend)
  if (!is.null(cp$cpu_backend)) ggml_backend_free(cp$cpu_backend)
}

# Лосс первой и последней эпохи. Падение лосса — единственное доказательство,
# что градиент дошёл до всех весов графа.
fit_loss <- function(model, x, y, epochs = 30L) {
  h <- ggml_fit(model, x, y, epochs = epochs, batch_size = 16L, verbose = 0L)
  l <- h$history$train_loss
  c(first = l[[1L]], last = l[[length(l)]])
}

report <- function(tag, l) {
  drop <- (l[["first"]] - l[["last"]]) / abs(l[["first"]])
  if (is.finite(drop) && drop > 0.10)
    ok("%-34s loss %.5f -> %.5f  (-%.0f%%)", tag, l[["first"]], l[["last"]], 100 * drop)
  else
    fail("%-34s loss %.5f -> %.5f  НЕ УЧИТСЯ", tag, l[["first"]], l[["last"]])
  invisible(drop)
}

set.seed(42L)

# Copy-task: выход обязан повторить вход. Решается только если attention
# действительно читает нужную позицию, а сеть не выучивает константу.
xa <- array(runif(N * S * D), dim = c(N, S, D))
y  <- matrix(aperm(xa, c(1L, 3L, 2L)), N, S * D)   # тот же тензор, row-major

# --------------------------------------------------------------------------
cat("\n1. Блок: attention + нормализация + pos-embedding\n")
# --------------------------------------------------------------------------

# 1a. Голый attention — базовая линия, тестами уже покрыт.
x  <- ggml_input(shape = c(S, D), name = "x")
o  <- x |> ggml_layer_attention(D, n_heads = H, name = "attn")
m  <- ggml_model(inputs = x, outputs = o)
m  <- ggml_compile(m, loss = "mse", backend = "cpu")
report("attention (базовая линия)", fit_loss(m, xa, y))
cleanup(m)

# 1b. + нормализация + residual + FF -- pre-LN блок целиком.
x  <- ggml_input(shape = c(S, D), name = "x")
h  <- x |> ggml_layer_rms_norm()
h  <- h |> ggml_layer_attention(D, n_heads = H, name = "attn")
h  <- ggml_layer_add(list(x, h))
o  <- h |> ggml_layer_dense(D, time_distributed = TRUE)
m  <- ggml_model(inputs = x, outputs = o)
m  <- ggml_compile(m, loss = "mse", backend = "cpu")
report("+ rms_norm + residual + FF", fit_loss(m, xa, y))
cleanup(m)

# 1c. + обучаемый pos-embedding -- полный pre-LN блок.
x <- ggml_input(shape = c(S, D), name = "x")
h <- x |> ggml_layer_positional_embedding()
n <- h |> ggml_layer_rms_norm()
a <- n |> ggml_layer_attention(D, n_heads = H, name = "attn")
h <- ggml_layer_add(list(h, a))
o <- h |> ggml_layer_dense(D, time_distributed = TRUE)
m <- ggml_model(inputs = x, outputs = o)
m <- ggml_compile(m, loss = "mse", backend = "cpu")
report("+ pos-embedding (блок целиком)", fit_loss(m, xa, y))
cleanup(m)

# --------------------------------------------------------------------------
cat("\n2. Маска в soft_max_ext (causal -- то, что доступно сейчас)\n")
# --------------------------------------------------------------------------

# Causal-задача: выход в позиции i -- среднее входов 1..i. Решается ТОЛЬКО с
# маской: без неё attention видит будущее и минимум лосса другой.
y_causal <- array(0, dim = c(N, S, D))
for (i in seq_len(N)) for (t in seq_len(S))
  y_causal[i, t, ] <- colMeans(matrix(xa[i, seq_len(t), ], t, D))
y_causal <- matrix(aperm(y_causal, c(1L, 3L, 2L)), N, S * D)

x <- ggml_input(shape = c(S, D), name = "x")
o <- x |> ggml_layer_attention(D, n_heads = H, causal = TRUE, name = "attn")
m <- ggml_model(inputs = x, outputs = o)
m <- ggml_compile(m, loss = "mse", backend = "cpu")
report("causal (diag_mask_inf)", fit_loss(m, xa, y_causal))
cleanup(m)

cat("\n2b. Произвольная маска (mask = -1e9 запрещает ключ)\n")

# Маска [n, q, k]: строка -- запрос, столбец -- ключ. Запрещаем каждому запросу
# всё, кроме первых двух ключей: цель считается только по ним, и решается это
# только если маска реально дошла до softmax.
mask_arr <- array(0, dim = c(N, S, S))
mask_arr[, , 3:S] <- -1e9
y_first2 <- array(0, dim = c(N, S, D))
for (i in seq_len(N)) for (t in seq_len(S))
  y_first2[i, t, ] <- colMeans(matrix(xa[i, 1:2, ], 2L, D))
y_first2 <- matrix(aperm(y_first2, c(1L, 3L, 2L)), N, S * D)

xin <- ggml_input(shape = c(S, D), name = "x")
min <- ggml_input(shape = c(S, S), name = "mask")
o   <- ggml_layer_attention(xin, D, n_heads = H, mask = min, name = "attn")
m   <- ggml_model(inputs = list(xin, min), outputs = o)
m   <- ggml_compile(m, loss = "mse", backend = "cpu")
report("маска (только ключи 1-2)", fit_loss(m, list(xa, mask_arr), y_first2))
cleanup(m)

# --------------------------------------------------------------------------
cat("\n3. Батч с разной эффективной длиной\n")
# --------------------------------------------------------------------------

# Без padding-маски короткие примеры добиваются нулями, и attention их видит.
# Цель -- предсказать среднее ТОЛЬКО валидных позиций. Если такая задача
# учится плохо, это количественная цена отсутствия маски.
len <- sample(2:S, N, replace = TRUE)
xv  <- xa
yv  <- matrix(0, N, D)
for (i in seq_len(N)) {
  if (len[i] < S) xv[i, (len[i] + 1L):S, ] <- 0
  yv[i, ] <- colMeans(matrix(xv[i, seq_len(len[i]), ], len[i], D))
}

x <- ggml_input(shape = c(S, D), name = "x")
h <- x |> ggml_layer_attention(D, n_heads = H, name = "attn")
h <- h |> ggml_layer_flatten()
o <- h |> ggml_layer_dense(D)
m <- ggml_model(inputs = x, outputs = o)
m <- ggml_compile(m, loss = "mse", backend = "cpu")
report("переменная длина (padding нулями)", fit_loss(m, xv, yv, epochs = 40L))
cleanup(m)

# То же самое, но padding исключён маской, а не заполнен нулями.
pad_mask <- array(0, dim = c(N, S, S))
for (i in seq_len(N)) if (len[i] < S) pad_mask[i, , (len[i] + 1L):S] <- -1e9

xin <- ggml_input(shape = c(S, D), name = "x")
min <- ggml_input(shape = c(S, S), name = "mask")
h   <- ggml_layer_attention(xin, D, n_heads = H, mask = min, name = "attn")
h   <- h |> ggml_layer_flatten()
o   <- h |> ggml_layer_dense(D)
m   <- ggml_model(inputs = list(xin, min), outputs = o)
m   <- ggml_compile(m, loss = "mse", backend = "cpu")
report("переменная длина (padding-маска)",
       fit_loss(m, list(xv, pad_mask), yv, epochs = 40L))
cleanup(m)

cat("\nГотово.\n")
