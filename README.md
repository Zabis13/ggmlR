# ggmlR — Neural Networks for R

[![R-hub check on the R Consortium cluster](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml/badge.svg)](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml)

A native R package for building, training, and deploying neural networks. Backed by the [ggml](https://github.com/ggml-org/ggml) C library, designed primarily for **Vulkan GPU acceleration** with full CPU fallback — no Python, no TensorFlow, everything runs inside your R session.

> **GPU-first design**: when a Vulkan-capable GPU is available (NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno), all operations run on GPU automatically. On machines without a GPU the package falls back to CPU transparently — no code changes needed.

Two complementary APIs:

| API | Style | When to use |
|---|---|---|
| Sequential / Functional | Keras-like, static graph | Production models, CRAN-standard workflow |
| Dynamic autograd (`ag_*`) | PyTorch-like, eager | Research, custom architectures, Transformers |

Also serves as the backend engine for [llamaR](https://github.com/Zabis13/llamaR) (LLM inference) and [sdR](https://github.com/Zabis13/sdR) (Stable Diffusion).

## Installation

```r
install.packages("ggmlR")
```

GPU (Vulkan) support is auto-detected at build time.

**Ubuntu / Debian** — to enable GPU:
```bash
sudo apt install libvulkan-dev glslc
```

**Windows** — install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and optionally the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) for GPU support.

## Sequential API

The fastest way to get a model running — stack layers with the pipe, compile, train.

```r
library(ggmlR)

model <- ggml_model_sequential() |>
  ggml_layer_dense(128L, activation = "relu",    input_shape = 784L) |>
  ggml_layer_dropout(rate = 0.3) |>
  ggml_layer_dense(10L,  activation = "softmax")

model <- ggml_compile(model,
                      optimizer = "adam",
                      loss      = "categorical_crossentropy",
                      metrics   = "accuracy")

model <- ggml_fit(model, x_train, y_train,
                  epochs           = 10L,
                  batch_size       = 32L,
                  validation_split = 0.1,
                  verbose          = 1L)

plot(model$history)

ggml_evaluate(model, x_test, y_test)
preds <- ggml_predict(model, x_new)
```

### Available layers (Sequential)

| Layer | Function |
|---|---|
| Dense | `ggml_layer_dense(units, activation)` |
| Conv1D | `ggml_layer_conv_1d(filters, kernel_size)` |
| Conv2D | `ggml_layer_conv_2d(filters, kernel_size, padding)` |
| MaxPooling2D | `ggml_layer_max_pooling_2d(pool_size)` |
| GlobalAvgPool2D | `ggml_layer_global_average_pooling_2d()` |
| BatchNorm | `ggml_layer_batch_norm()` |
| Flatten | `ggml_layer_flatten()` |
| Dropout | `ggml_layer_dropout(rate)` |
| Embedding | `ggml_layer_embedding(vocab_size, dim)` |
| LSTM | `ggml_layer_lstm(units, return_sequences)` |
| GRU | `ggml_layer_gru(units, return_sequences)` |

### CNN example (MNIST)

```r
model <- ggml_model_sequential() |>
  ggml_layer_conv_2d(32L, kernel_size = c(3L, 3L), activation = "relu",
                     input_shape = c(28L, 28L, 1L)) |>
  ggml_layer_max_pooling_2d(pool_size = c(2L, 2L)) |>
  ggml_layer_conv_2d(64L, kernel_size = c(3L, 3L), activation = "relu") |>
  ggml_layer_global_average_pooling_2d() |>
  ggml_layer_dense(10L, activation = "softmax")
```

## Functional API

Wire layers into arbitrary graphs — residual connections, multi-input/output, shared weights.

### Residual (skip) connection

```r
inp <- ggml_input(shape = 64L)
x   <- inp |> ggml_layer_dense(64L, activation = "relu")
res <- ggml_layer_add(list(inp, x))        # element-wise add
out <- res |> ggml_layer_dense(10L, activation = "softmax")

m <- ggml_model(inputs = inp, outputs = out)
m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")
m <- ggml_fit(m, x_train, y_train, epochs = 5L, batch_size = 32L)
```

### Embedding + GRU + skip connection (NLP)

```r
inp <- ggml_input(shape = 30L, dtype = "int32", name = "tokens")
emb <- inp |> ggml_layer_embedding(vocab_size = 500L, dim = 32L)

# Branch A: GRU path
proj_a <- emb |>
  ggml_layer_gru(32L, return_sequences = FALSE) |>
  ggml_layer_dense(32L)

# Branch B: flatten + projection
proj_b <- emb |>
  ggml_layer_flatten() |>
  ggml_layer_dense(32L, activation = "relu") |>
  ggml_layer_dense(32L)

# Residual merge
out <- ggml_layer_add(list(proj_a, proj_b)) |>
  ggml_layer_dropout(rate = 0.3) |>
  ggml_layer_dense(2L, activation = "softmax")

m <- ggml_model(inputs = inp, outputs = out)
```

> Token values must be 0-based integers in `[0, vocab_size - 1]`.

### Multi-input model

```r
inp1 <- ggml_input(shape = 20L, name = "timeseries")
inp2 <- ggml_input(shape = 3L,  name = "metadata")

br1 <- inp1 |> ggml_layer_dense(16L, activation = "relu")
br2 <- inp2 |> ggml_layer_dense(8L,  activation = "relu")

out <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
  ggml_layer_dense(2L, activation = "softmax")

m <- ggml_model(inputs = list(inp1, inp2), outputs = out)
m <- ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy")

# Pass x as a list — one matrix per input
m <- ggml_fit(m, x = list(x_ts, x_meta), y = y,
              epochs = 10L, batch_size = 32L)
preds <- ggml_predict(m, list(x_ts, x_meta))
```

### Multi-output model

```r
inp    <- ggml_input(shape = 64L)
hidden <- inp    |> ggml_layer_dense(64L, activation = "relu")
out    <- hidden |> ggml_layer_dense(10L, activation = "softmax")

m     <- ggml_model(inputs = inp, outputs = list(hidden, out))
preds <- ggml_predict(m, x)
# preds[[1]] — hidden activations  [n × 64]
# preds[[2]] — class probabilities [n × 10]
```

### ResNet-like image classifier

```r
residual_block <- function(x, filters, name) {
  main     <- x |> ggml_layer_conv_2d(filters, c(3L, 3L), padding = "same",
                                       name = paste0(name, "_conv"))
  shortcut <- x |> ggml_layer_conv_2d(filters, c(1L, 1L), padding = "same",
                                       name = paste0(name, "_proj"))
  ggml_layer_add(list(main, shortcut), name = paste0(name, "_add"))
}

inp <- ggml_input(shape = c(32L, 32L, 3L))
x   <- inp |> ggml_layer_conv_2d(16L, c(3L, 3L), activation = "relu",
                                  padding = "same")
x   <- residual_block(x, 16L, "res1")
x   <- residual_block(x, 32L, "res2")
out <- x |>
  ggml_layer_global_average_pooling_2d() |>
  ggml_layer_dropout(rate = 0.4) |>
  ggml_layer_dense(3L, activation = "softmax")

m <- ggml_model(inputs = inp, outputs = out)
```

### Shared layers (Siamese / weight sharing)

```r
enc <- ggml_dense(32L, activation = "relu", name = "encoder")

x1 <- ggml_input(shape = 16L, name = "left")
x2 <- ggml_input(shape = 16L, name = "right")

h1 <- ggml_apply(x1, enc)   # identical weights
h2 <- ggml_apply(x2, enc)

out <- ggml_layer_add(list(h1, h2)) |>
  ggml_layer_dense(2L, activation = "softmax")

m <- ggml_model(inputs = list(x1, x2), outputs = out)
```

### Differences from Keras

| Feature | Keras (Python) | ggmlR |
|---|---|---|
| Batch dimension | part of `input_shape` | excluded from `shape` |
| Merge layers | `add([a, b])` | `ggml_layer_add(list(a, b))` |
| Shared layers | reuse layer object | `ggml_dense()` + `ggml_apply()` |
| Multi-input data | list of arrays | `list()` of R matrices |
| Multi-output predict | list of numpy arrays | R list of matrices |
| Backend | TensorFlow / JAX / PyTorch | ggml (Vulkan GPU, CPU fallback) |

## Dynamic Autograd Engine (PyTorch-style)

Build and train arbitrary architectures with eager execution and automatic differentiation.

```r
library(ggmlR)

# Define parameters
W <- ag_param(matrix(rnorm(4 * 8) * 0.1, 8, 4))
b <- ag_param(matrix(0, 8, 1))

# Forward + backward
with_grad_tape({
  h    <- ag_add(ag_matmul(W, x_batch), b)
  h    <- ag_relu(h)
  loss <- ag_mse_loss(h, y_batch)
})
grads <- backward(loss)

opt <- optimizer_adam(list(W = W, b = b), lr = 1e-3)
opt$step(grads)
opt$zero_grad()
```

### Transformer encoder block

```r
model <- ag_sequential(
  ag_linear(64L, 128L, activation = "relu"),
  ag_batch_norm(128L),
  ag_dropout(0.1),
  ag_linear(128L, 10L)
)

params <- model$parameters()
opt    <- optimizer_adam(params, lr = 1e-3)
sch    <- lr_scheduler_cosine(opt, T_max = 50L, lr_min = 1e-5)

dl <- ag_dataloader(x_train, y_train, batch_size = 32L, shuffle = TRUE)

for (epoch in 1:50) {
  for (batch in dl$epoch()) {
    with_grad_tape({
      out  <- model$forward(batch$x)
      loss <- ag_softmax_cross_entropy_loss(out, batch$y)
    })
    grads <- backward(loss)
    clip_grad_norm(params, grads, max_norm = 1.0)
    opt$step(grads)
    opt$zero_grad()
  }
  sch$step()
}
```

### Data-parallel training

`dp_train()` splits data across N replicas, averages gradients, and keeps weights in sync automatically.

```r
make_model <- function() {
  W <- ag_param(matrix(rnorm(4 * 2) * 0.1, 2, 4))
  b <- ag_param(matrix(0, 2, 1))
  list(
    forward    = function(x) ag_add(ag_matmul(W, x), b),
    parameters = function() list(W = W, b = b)
  )
}

result <- dp_train(
  make_model  = make_model,
  data        = my_dataset,           # list of samples
  loss_fn     = function(out, tgt) ag_mse_loss(out, tgt),
  forward_fn  = function(model, s) model$forward(s$x),
  target_fn   = function(s) s$y,
  n_gpu       = 2L,                   # number of replicas
  n_iter      = 100L,
  lr          = 1e-3,
  max_norm    = 5.0
)

result$loss_history   # numeric vector, one value per iteration
result$model          # trained replica 0
```

### Autograd op reference

| Category | Functions |
|---|---|
| Linear | `ag_matmul`, `ag_add`, `ag_sub`, `ag_mul`, `ag_scale` |
| Activations | `ag_relu`, `ag_sigmoid`, `ag_tanh`, `ag_softmax` |
| Reductions | `ag_sum`, `ag_mean` (with `dim`, `keepdim`) |
| Math | `ag_log`, `ag_exp`, `ag_pow`, `ag_clamp` |
| Shape | `ag_reshape`, `ag_transpose` |
| Attention | `ag_multihead_attention` |
| Loss | `ag_mse_loss`, `ag_cross_entropy_loss`, `ag_softmax_cross_entropy_loss` |
| Layers | `ag_linear`, `ag_batch_norm`, `ag_dropout`, `ag_embedding` |
| Containers | `ag_sequential` |
| Optimizers | `optimizer_sgd`, `optimizer_adam` |
| Schedulers | `lr_scheduler_step`, `lr_scheduler_cosine` |
| Utilities | `clip_grad_norm`, `ag_gradcheck`, `dp_train` |

## Save / Load

```r
ggml_save_model(model, "my_model.rds")
model <- ggml_load_model("my_model.rds")
```

## GPU Acceleration

ggmlR is designed GPU-first: Vulkan is auto-detected at build time and, when available, 90%+ of operations run on GPU with 5×–20× speedup over CPU. On machines without a Vulkan-capable GPU the package falls back to CPU transparently — no code changes required.

```r
ggml_vulkan_available()   # TRUE if a Vulkan GPU was detected
ggml_vulkan_status()      # device name, memory, capabilities

# Dynamic autograd: switch device at runtime
ag_device("gpu")   # move subsequent ops to GPU (f16 by default)
ag_device("cpu")   # fall back to CPU
```

Supported GPUs: NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno.

## System Requirements

- R ≥ 4.1.0, C++17 compiler
- **Optional GPU**: `libvulkan-dev` + `glslc` (Linux) or Vulkan SDK (Windows)
- Platforms: Linux, macOS, Windows (x86-64, ARM64)

## See Also

- [llamaR](https://github.com/Zabis13/llamaR) — LLM inference in R
- [sdR](https://github.com/Zabis13/sdR) — Stable Diffusion in R
- [ggml](https://github.com/ggml-org/ggml) — underlying C library

## License

MIT
