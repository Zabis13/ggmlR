# ggmlR — Keras-like Neural Networks for R

[![R-hub check on the R Consortium cluster](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml/badge.svg)](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml)

A native R package for building, training, and deploying neural networks with a Keras-like API. Backed by the [ggml](https://github.com/ggml-org/ggml) C library with CPU and optional Vulkan GPU acceleration. No Python, no TensorFlow — everything runs inside your R session.

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
| Backend | TensorFlow / JAX / PyTorch | ggml (CPU + Vulkan GPU) |

## Save / Load

```r
ggml_save_model(model, "my_model.rds")
model <- ggml_load_model("my_model.rds")
```

## GPU Acceleration

ggmlR auto-detects Vulkan at build time. When available, 90%+ of operations run on GPU with 5×–20× speedup over CPU.

```r
ggml_vulkan_available()   # TRUE if GPU detected
ggml_vulkan_status()      # device name, memory, capabilities
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
