# ggmlR — Neural Networks for R

[![CRAN](https://www.r-pkg.org/badges/version/ggmlR)](https://CRAN.R-project.org/package=ggmlR)
[![R-hub check on the R Consortium cluster](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml/badge.svg)](https://github.com/r-hub2/separate-jaguar-ggmlR/actions/workflows/rhub-rc.yaml)

[![ggmlR](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcranlogs.r-pkg.org%2Fdownloads%2Ftotal%2Flast-month%2FggmlR&query=%24%5B0%5D.downloads&label=ggmlR&color=blue)](https://CRAN.R-project.org/package=ggmlR)
[![sd2R](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcranlogs.r-pkg.org%2Fdownloads%2Ftotal%2Flast-month%2Fsd2R&query=%24%5B0%5D.downloads&label=sd2R&color=blue)](https://CRAN.R-project.org/package=sd2R)
[![llamaR](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcranlogs.r-pkg.org%2Fdownloads%2Ftotal%2Flast-month%2FllamaR&query=%24%5B0%5D.downloads&label=llamaR&color=blue)](https://CRAN.R-project.org/package=llamaR)
[![drogonR](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcranlogs.r-pkg.org%2Fdownloads%2Ftotal%2Flast-month%2FdrogonR&query=%24%5B0%5D.downloads&label=drogonR&color=blue)](https://CRAN.R-project.org/package=drogonR)
[![cayleyR](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcranlogs.r-pkg.org%2Fdownloads%2Ftotal%2Flast-month%2FcayleyR&query=%24%5B0%5D.downloads&label=cayleyR&color=blue)](https://CRAN.R-project.org/package=cayleyR)
[![sgvR](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcranlogs.r-pkg.org%2Fdownloads%2Ftotal%2Flast-month%2FsgvR&query=%24%5B0%5D.downloads&label=sgvR&color=blue)](https://CRAN.R-project.org/package=sgvR)
[![av1R](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcranlogs.r-pkg.org%2Fdownloads%2Ftotal%2Flast-month%2Fav1R&query=%24%5B0%5D.downloads&label=av1R&color=blue)](https://CRAN.R-project.org/package=av1R)

A native R package for building, training, and deploying neural networks. Backed by the [ggml](https://github.com/ggml-org/ggml) C library, designed primarily for **Vulkan GPU acceleration** with full CPU fallback — no Python, no TensorFlow, everything runs inside your R session.

> **GPU-first design**: when a Vulkan-capable GPU is available (NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno), all operations run on GPU automatically. On machines without a GPU the package falls back to CPU transparently — no code changes needed.

Two complementary APIs:

| API | Style | When to use |
|---|---|---|
| Sequential / Functional | Keras-like, static graph | Production models, CRAN-standard workflow |
| Dynamic autograd (`ag_*`) | PyTorch-like, eager | Research, custom architectures, Transformers |

Also serves as the backend engine for [llamaR](https://github.com/Zabis13/llamaR) (LLM inference) and [sd2R](https://github.com/Zabis13/sd2R) (Stable Diffusion).

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

### Build options

Force-enable or disable Vulkan GPU backend:
```r
install.packages("ggmlR", configure.args = "--with-vulkan")
install.packages("ggmlR", configure.args = "--without-vulkan")
```

Enable CPU SIMD acceleration (AVX2, SSE4, etc.) for faster inference on your machine:
```r
install.packages("ggmlR", configure.args = "--with-simd")
```

Enable the hard-exit path used by **multi-GPU** standalone scripts
(`ggml_vulkan_shutdown(hard = TRUE)`; see **Clean shutdown** below and
`vignette("multi-gpu")`). Off by default — the released package must not call
`_exit()`, as CRAN policy forbids a package terminating the R session:
```r
install.packages("ggmlR", configure.args = "--enable-hard-exit")
```

Options can be combined:
```r
install.packages("ggmlR", configure.args = "--with-vulkan --with-simd --enable-hard-exit")
```

### Linux (detailed)

Full step-by-step setup on Ubuntu/Debian, from a clean system to a working
GPU build.

**1. R and the Vulkan loader/tools:**

```bash
sudo apt install -y r-base

sudo apt install vulkan-tools libvulkan-dev
```

**2. The `glslc` shader compiler** (needed to build ggmlR's Vulkan backend):

```bash
# Ubuntu 24.04 (Noble)
sudo add-apt-repository universe
sudo apt update
sudo apt install glslc

# Ubuntu 22.04 (Jammy) — install the LunarG Vulkan SDK instead
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | \
  sudo tee /etc/apt/trusted.gpg.d/lunarg.asc

sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
  https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list

sudo apt update
sudo apt install -y vulkan-sdk
```

**3. Verify the GPU is visible to Vulkan:**

```bash
vulkaninfo --summary
```

**4. Install ggmlR with CPU SIMD acceleration:**

```bash
sudo Rscript -e 'install.packages("ggmlR", configure.args = "--with-simd")'
```

**5. Confirm GPU support from R:**

```bash
Rscript -e 'library(ggmlR)
ggml_vulkan_status()'
```

#### Windows build options

> **Important:** on Windows, R **ignores** `configure.args` / `--configure-args`
> (they are only honoured by the Unix `./configure` path). Use **environment
> variables** instead, set in the same R session *before* installing:

```r
Sys.setenv(GGML_USE_SIMD = "1")     # enable CPU SIMD (AVX2/SSE4/FMA/F16C)
Sys.setenv(GGML_USE_VULKAN = "1")   # force-enable Vulkan  (or "0" to disable)
Sys.setenv(GGML_VK_HARD_EXIT = "1") # enable ggml_vulkan_shutdown(hard = TRUE)
```

The flags are read by `configure.win`, which only runs when the package is
**built from source**. The CRAN binary for Windows is pre-built without them,
so pass `type = "source"` to force a source build:

```r
# From CRAN (force a source build so the flags take effect):
install.packages("ggmlR", type = "source")

# From GitHub (always builds from source):
remotes::install_github("Zabis13/ggmlR", force = TRUE)
```

Vulkan is still auto-detected when the Vulkan SDK is present, so
`GGML_USE_VULKAN` is only needed to force it on/off. Accepted values:
`1` / `yes` / `true` / `on` (and `0` / `no` / `false` / `off` for Vulkan).
OpenMP (multi-threaded CPU executor) is detected automatically on Rtools.

### Diagnostics

`GGMLR_LOG_DEVICE` — when set to a non-empty value other than `0`, enables
Vulkan telemetry logging. Two messages are emitted:

1. The selected device and its capabilities, once a backend context is created:

   ```
   ggml_vulkan: device #0 'Vulkan0' selected | fp16=1 bf16=1 coopmat=1 coopmat2=0 bda=1 max_buffer=4095 MB suballoc_block=1024 MB sysmem_fallback=0
   ```

2. A per-graph summary of any ops the Vulkan backend could not run and fell back
   to CPU (e.g. `OUT_PROD`, `CROSS_ENTROPY_LOSS_BACK` during training):

   ```
   ggml_vulkan: 12 op(s) not supported on GPU during graph compute, ran on CPU (per-type: OUT_PROD=12)
   ```

Both are **off by default** so they do not clutter output (notably during
tests, where many contexts and training graphs are created). Enable them to
diagnose which GPU was picked, which acceleration paths (fp16 / bf16 / coopmat)
are active, and which ops are silently running on CPU:

```r
Sys.setenv(GGMLR_LOG_DEVICE = "1")
```

### CPU performance: multithreaded BLAS/LAPACK

ggmlR runs the heavy linear algebra on the GPU, but some steps stay on the CPU
through R's BLAS/LAPACK — most notably the eigendecomposition in PCA
(`RunGGML(op = "embed")`) and any `eigen()` / `prcomp()` fallback. R ships with
the **reference** BLAS/LAPACK, which is single-threaded: on a multi-core machine
those CPU phases use just one core. Switching to **OpenBLAS** parallelises them
(e.g. PCA on a 3000-gene object dropped from ~24 s to ~1.6 s here).

This is a system-level change (outside the package, needs `sudo`). On
Debian/Ubuntu:

```bash
# 1. Install OpenBLAS. Prefer the OpenMP variant over the default pthreads one
#    (the R-admin manual recommends the openmp build): its threads are governed
#    by OMP_NUM_THREADS and it is better behaved under tools like valgrind.
sudo apt update
sudo apt install libopenblas-openmp-dev   # or libopenblas-dev for the pthreads build

# 2. Point BLAS at OpenBLAS (pick the openblas entry from the list):
sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu

# 3. Point LAPACK at OpenBLAS too — eigen() goes through LAPACK, not BLAS:
sudo update-alternatives --config liblapack.so.3-x86_64-linux-gnu
```

```r
# 4. Confirm R picked it up — both lines should name openblas, not
#    /usr/lib/.../blas/libblas.so or /usr/lib/.../lapack/liblapack.so:
si <- sessionInfo(); cat("BLAS:  ", si$BLAS, "\nLAPACK:", si$LAPACK, "\n")
```

OpenBLAS grabs all cores by default. If that oversubscribes alongside parallel
resampling (tidymodels / mlr3) or trips CRAN's 2-thread limit in checks, cap it:
`Sys.setenv(OPENBLAS_NUM_THREADS = 2)`.

> **Note (valgrind).** A multithreaded OpenBLAS spawns worker threads, and
> valgrind reports each worker's thread-local storage as a one-off
> "possibly lost" block (`definitely lost: 0`). This is a known false positive,
> not a leak. Run checks with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for a
> clean valgrind report; CRAN's own machines use single-threaded reference BLAS,
> so it does not appear there.

## GPU linear algebra (drop-in for `%*%`)

Accelerate an ordinary R matrix multiply on the GPU without rewriting your code —
plain matrices in, plain matrices out, with a transparent CPU fallback.

```r
library(ggmlR)

A <- matrix(rnorm(2000 * 1500), 2000, 1500)
B <- matrix(rnorm(1500 * 1000), 1500, 1000)

C <- ggml_matmul(A, B)        # A %*% B on the GPU, returns a plain matrix
G <- ggml_crossprod(A)        # t(A) %*% A
H <- ggml_tcrossprod(A)       # A %*% t(A)
```

Prefer the operators? Wrap one operand and `%*%` / `crossprod()` / `tcrossprod()`
dispatch to the GPU — nothing else in your code changes:

```r
Ag <- as_gpu_matrix(A)
C  <- Ag %*% B                # GPU
G  <- crossprod(Ag)           # GPU
```

- **Dispatch** — `device = "auto"` (default) uses the GPU when one is present *and*
  the multiply is large enough to amortise the host↔VRAM transfer; small
  multiplies and machines without a GPU stay on the CPU. Force it with
  `device = "gpu"` / `"cpu"`.
- **Precision** — R multiplies in double precision (f64); a GPU offers f32 at
  best, so the GPU path is a fast *approximate* multiply, not a bit-for-bit
  replacement. `prec = "f32"` (default) requests f32 accumulation; how close the
  result lands depends on the Vulkan driver (some, e.g. RADV/Mesa, accumulate in
  f16 regardless, giving ~`1e-3` relative error either way). `prec = "f16"` only
  lowers precision further, for speed.
- **Full double precision** — need bit-accurate results? `ggml_matmul_f64(A, B)`
  runs the multiply in fp64 on the GPU, matching R's `%*%` to ~`1e-15`. It only
  pays off on data-centre cards with fast fp64 (Tesla P100/V100, Instinct);
  consumer GPUs cripple fp64, so it falls back to (and is usually slower than)
  the CPU there.

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
# Important: always assign the return value back to model —
# ggml_fit() returns the model with updated weights.

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

### Multi-GPU: tensor & pipeline parallelism

For machines with several Vulkan GPUs, ggmlR ships a native **tensor-parallel** and **pipeline-parallel** path (not in upstream ggml, which does tensor-split only for CUDA/SYCL). The cross-device transport defaults to portable **host staging** (device→host→device), correct on every driver.

**Tensor parallelism** — split a weight matrix's rows across GPUs, compute each slice on its own device, gather the result:

```r
W <- matrix(rnorm(4096 * 256), nrow = 4096)   # 4096 outputs x 256 inputs
X <- matrix(rnorm(8 * 256),    nrow = 8)       # batch of 8

Y <- ggml_vulkan_split_mul_mat(W, X, n_devices = 2)   # == X %*% t(W)
```

**TP × DP hybrid** — data-parallel over the batch across replicas of tensor-parallel device groups (e.g. 2 replicas × TP=2 on a 4-GPU box):

```r
# replica A = GPUs {0,1}, replica B = GPUs {2,3}; batch split across replicas,
# weights tensor-split within each pair. No cross-replica traffic at inference.
Y <- ggml_tp_dp_forward(W, X, replicas = list(c(0, 1), c(2, 3)))
```

**Pipeline parallelism** — split a model *by layers* across GPUs; the activation tensor is handed between stages just once per pass (a single cross-device copy). Suits models too large for one card:

```r
# stage 1 (layers 1..k) on GPU 0  ->  stage 2 (layers k+1..n) on GPU 1
mk <- function(dev, Wt) list(
  device = dev, in_shape = c(K, M),
  build  = function(ctx, input) {
    w <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, K)
    list(output      = ggml_relu(ctx, ggml_mul_mat(ctx, w, input)),
         set_weights = function() ggml_backend_tensor_set_data(w, as.numeric(Wt)))
  })
y <- ggml_pp_forward(list(mk(0L, W1), mk(1L, W2)), x = as.numeric(X), out_shape = c(K, M))
```

See `inst/examples/tp_dp_hybrid.R` and `inst/examples/pp_pipeline.R` for complete runnable demos.

#### Benchmark: which split strategy to use

Measured with `llamaR` (which links `libggml.a` statically) driving Qwen2.5-1.5B-Instruct Q4_K_M (Vulkan, host-staging cross-device transport) on two 4-GPU hosts, decode throughput, median of 3 runs of 128 tokens:

- **P100** — 4× Tesla P100-SXM2-16GB
- **V100** — 4× Tesla V100-32GB, 2× Xeon E5-2698 v4, 256 GB RAM

| Strategy | GPUs | Split | Decode t/s (P100) | Decode t/s (V100) | Notes |
|---|---|---|---:|---:|---|
| Baseline | 1 | none | **419.7** | **516.9** | model fits in one card — fastest |
| Pipeline (PP) | 2 | layer | 150.4 | 221.5 | layers spread across 2 GPUs |
| Tensor (TP) | 2 | row | 150.4 | 223.2 | rows split, all-reduce per layer |
| Pipeline (PP) | 4 | layer | 133.3 | 176.3 | more hops → slower |
| Tensor (TP) | 4 | row | 130.0 | 176.6 | more hops → slower |
| **TP=2 × DP=2** | 4 | row + replicas | **306** | **446** | 2 replicas × TP=2, run concurrently |
| **DP=4** | 4 | replicas | **975** | **1300** | 4 single-GPU replicas, run concurrently |

Same model on an **8× Tesla V100-32GB** host (2× Xeon E5-2698 v4, 256 GB RAM), showing how the pattern holds as GPU count doubles:

| Strategy | GPUs | Split | Decode t/s | Notes |
|---|---|---|---:|---|
| Baseline | 1 | none | **684.9** | model fits in one card — fastest |
| Pipeline (PP) | 2 | layer | 229.7 | layers spread across 2 GPUs |
| Tensor (TP) | 2 | row | 231.3 | rows split, all-reduce per layer |
| Pipeline (PP) | 4 | layer | 187.5 | more hops → slower |
| Tensor (TP) | 4 | row | 191.0 | more hops → slower |
| Pipeline (PP) | 8 | layer | 142.8 | 8-way split — slowest single-context |
| Tensor (TP) | 8 | row | 137.1 | per-layer all-reduce across 8 cards |
| **TP=2 × DP=4** | 8 | row + replicas | **897** | 4 replicas × TP=2, run concurrently |
| **DP=8** | 8 | replicas | **2290** | 8 single-GPU replicas, run concurrently |

**Takeaway:** when a model **fits in one GPU**, data parallelism (DP — independent replicas) wins by a wide margin: DP=4 delivers ~2.3× the single-card throughput and ~7.5× any split mode. Splitting such a model across cards (PP/TP) only adds cross-device overhead — the ~1 GB/s host-staging transport dominates. **PP and TP earn their keep only when the model does not fit in one card** (e.g. a 30B+ model on 16 GB cards): there a split is the *only* way to run it at all, and PP minimizes cross-device hops (one activation copy per pass) while TP maximizes per-token parallelism at the cost of a per-layer all-reduce. Reproduce with `llamaR`'s `inst/examples/bench_pp_tp_dp.sh`.

> **Clean shutdown**: when a standalone script uses several GPUs, make `ggml_vulkan_shutdown(hard = TRUE)` its **last** statement. This tears down Vulkan and then calls `_exit(0)`, skipping the exit-time loader-static-destruction phase that can otherwise flakily segfault *after* your results are printed (the results are already computed by then, so the crash is harmless-but-noisy). Use plain `ggml_vulkan_shutdown()` (no `hard`) mid-session — it releases the devices and is safe to call repeatedly, but does not guarantee a clean process exit on its own.
>
> The `_exit(0)` path is **compiled out by default**, because CRAN policy forbids a package terminating the R session. In a default build `hard = TRUE` does the normal teardown and **warns** instead of exiting — never silently — so the flaky exit-time segfault can still fire. Compile it in with `--configure-args="--enable-hard-exit"` (Windows: `Sys.setenv(GGML_VK_HARD_EXIT = "1")` before installing), and check the current build with `ggml_vulkan_hard_exit_available()`.
>
> **When to use it — and when not.** `hard = TRUE` only pays off for **multi-GPU** scripts (tensor/pipeline parallelism, `split_mul_mat`/`pp_forward`), where late device finalizers race the loader teardown on exit. A **single-GPU** script never triggers that race, so it does not need `hard = TRUE`. Do **not** use it under a **Jupyter / Kaggle notebook**: there `_exit(0)` kills the R *kernel*, and the notebook runner (papermill/nbclient) reports it as `DeadKernelError: Kernel died` even though all results were already computed and written. In a notebook, either drop the call or use plain `ggml_vulkan_shutdown()` (soft teardown, kernel stays alive). A portable one-liner that keeps the hard exit only outside notebooks:
>
> ```r
> is_notebook <- nzchar(Sys.getenv("KAGGLE_KERNEL_RUN_TYPE")) || nzchar(Sys.getenv("JPY_PARENT_PID"))
> ggml_vulkan_shutdown(hard = !is_notebook)
> ```

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

## mlr3 Integration

ggmlR ships with [mlr3](https://mlr3.mlr-org.com/) learners for tabular classification and regression. After `library(ggmlR)` (with `mlr3` installed), sequential and functional ggmlR networks become available as first-class learners:

```r
library(mlr3)
library(ggmlR)

# Classification on iris (GPU auto-detected via backend = "auto")
task <- tsk("iris")

learner <- lrn("classif.ggml",
               epochs     = 50L,
               batch_size = 16L,
               backend    = "auto")      # "auto" | "cpu" | "gpu"
learner$predict_type <- "prob"

learner$train(task)
pred <- learner$predict(task)
pred$score(msr("classif.logloss"))
```

### Features

- **Both ggmlR APIs** — `model_fn` can return a `ggml_sequential_model` or `ggml_functional_model`. The default builder is `ggml_default_mlp()`, an exported MLP builder you can also use directly.
- **Vulkan GPU** — set `backend = "gpu"` (or leave `"auto"`) and the learner trains and predicts on GPU.
- **Parallel tuning** — the learners declare `properties = "marshal"` and implement in-memory marshalling (SHA-256-checksummed container), so trained models can be shipped to `future` / `callr` workers without file-system round-trips.
- **Weighted training** — `classif.ggml` honours `task$weights_learner`, mapping them to `sample_weight` in `ggml_fit()`.
- **Callbacks for tuning** — pass `ggml_callback_early_stopping()` etc. via the `callbacks` parameter to drive early stopping inside `mlr3` tuning runs.
- **Custom architectures** — set `learner$model_fn <- function(task, n_features, n_out, pars) { ... }` to build any ggmlR network you like; the learner handles task → matrix conversion, compilation, training, and prediction.

```r
# Regression with a custom architecture
library(mlr3)
library(ggmlR)

learner <- lrn("regr.ggml", epochs = 100L)
learner$model_fn <- function(task, n_features, n_out, pars) {
  ggml_model_sequential() |>
    ggml_layer_dense(256L, activation = "gelu", input_shape = n_features) |>
    ggml_layer_dropout(0.2) |>
    ggml_layer_dense( 64L, activation = "gelu") |>
    ggml_layer_dense(n_out, activation = "linear")
}

learner$train(tsk("mtcars"))
```

Only numeric features are supported: wrap the learner in a pipeline
(`po("encode") %>>% po("scale") %>>% lrn("classif.ggml")`) when the task has
factor columns. `mlr3`, `paradox`, `R6`, and `mlr3pipelines` are `Suggests`;
ggmlR only wires up the integration when they are installed.

## tidymodels / parsnip Integration

ggmlR registers a `"ggml"` engine for [`parsnip::mlp()`](https://parsnip.tidymodels.org/reference/mlp.html), letting you use ggmlR networks inside the tidymodels ecosystem.

```r
library(ggmlR)
library(parsnip)

spec <- mlp(
  hidden_units = 64,
  epochs       = 100,
  dropout      = 0.2,
  learn_rate   = 0.001
) |>
  set_engine("ggml",
             batch_size = 32,
             backend    = "auto",   # "auto" | "cpu" | "gpu"
             verbose    = 0) |>
  set_mode("classification")

fit_obj <- fit(spec, Species ~ ., data = iris)
predict(fit_obj, new_data = iris)
predict(fit_obj, new_data = iris, type = "prob")
```

Regression works the same way:

```r
spec_reg <- mlp(hidden_units = 32, epochs = 200) |>
  set_engine("ggml", batch_size = 8, backend = "gpu") |>
  set_mode("regression")

fit_reg <- fit(spec_reg, mpg ~ ., data = mtcars)
predict(fit_reg, new_data = mtcars)
```

`parsnip`, `tibble`, `rlang`, and `dials` are in `Suggests` — ggmlR only wires up the engine when they are installed.

## Single-cell GPU Acceleration (Seurat)

Run GPU-accelerated operations directly on `Seurat` objects — no conversion on your side, and no hard dependency: `Seurat`/`SeuratObject` stay in `Suggests`, so ggmlR installs fine without them and the adapter activates only when they are present.

**Setup.** The adapter needs two packages installed alongside ggmlR — `Seurat` (the object model and its pipeline) and a couple of light helpers it leans on for speed (`Matrix` for the sparse graphs, `FNN` for the kd-tree kNN). They are all `Suggests`, so install them yourself:

```r
install.packages(c("Seurat", "Matrix", "FNN"))   # ggmlR is already installed

library(ggmlR)
library(Seurat)        # ggmlR's S3 methods (RunGGML, ggml_extract, ...) activate
                       # automatically once SeuratObject is on the search path
```

If `FNN` is absent the kNN falls back to the GPU/CPU distance matrix; if `Matrix` is absent the `"neighbors"` op is unavailable (the graphs are sparse). The rest works with `Seurat` alone.

`RunGGML()` is the one-call, Seurat-style entry point (object in, object out — pipe-friendly, mirrors `RunPCA()`). Vulkan is used automatically when a GPU is present, with a transparent CPU fallback. The supported operations are the heavy matrix steps of a standard pipeline:

| `op` | Replaces | What runs on the GPU |
|------|----------|----------------------|
| `"normalize"` | `NormalizeData()` (LogNormalize) | sparse `log1p(x / colSum × sf)` over the stored non-zeros (`sparse_lognorm.comp`) — the counts stay a `dgCMatrix`, never densified |
| `"scale"` | `ScaleData()` | per-gene z-score `(x − mean) / sd` + clamp over the full dense matrix |
| `"embed"` | `RunPCA()` | gene-by-gene covariance multiply (eigendecomposition stays on the CPU — ggml has no eigensolver) |
| `"umap"` | `RunUMAP()` | **two** custom compute shaders: `pairwise_dist.comp` (kNN distances, honest f32) and `umap_sgd.comp` (SGD layout) |
| `"neighbors"` | `FindNeighbors()` | kNN distances feeding the SNN/Jaccard graph; the FNN kd-tree by default, or a fused GPU kNN (`knn_tiled.comp`) with `knn_backend = "vulkan"`. Exact kNN either way |
| `"largest_gene"` | `percent.Largest.Gene` | per-cell highest-expressed gene + its share of the cell total, over the sparse `dgCMatrix` CSC slots (no densify); bit-exact with `qlcMatrix::colMax` |

`"normalize"` and `"scale"` write the transformed matrix back into the assay (the `data` and `scale.data` layers), so the rest of the Seurat pipeline picks them up unchanged. `"embed"` and `"umap"` add a dimensionality reduction; `"neighbors"` writes the `<assay>_nn` and `<assay>_snn` graphs into `@graphs`, exactly where `FindClusters()` looks; `"largest_gene"` adds `largest_gene` and `percent.Largest.Gene` columns to `meta.data` / `colData`.

### What runs where

A standard Seurat workflow maps onto the adapter like this — five of the heavy
steps move to the GPU, and only the final community detection stays on the CPU:

| Standard step | ggmlR | Runs on |
|---------------|-------|---------|
| `NormalizeData()` | `RunGGML(op = "normalize")` | **GPU** |
| `ScaleData()` | `RunGGML(op = "scale")` | **GPU** |
| `RunPCA()` | `RunGGML(op = "embed")` | **GPU** matrix multiply (eigensolve on CPU — ggml has none) |
| `RunUMAP()` | `RunGGML(op = "umap")` | **GPU** (distance + SGD shaders) |
| `FindNeighbors()` | `RunGGML(op = "neighbors")` | kNN on CPU (FNN kd-tree) or **GPU** (`knn_backend = "vulkan"`, `knn_tiled.comp`) → sparse SNN |
| `FindClusters()` | — (use Seurat's) | CPU — iterative graph Louvain/Leiden, a poor GPU fit and already fast on the CPU |

On a Vulkan GPU (AMD RADV) the GPU steps are markedly faster than a naive
reference. Indicative numbers at 2000 cells:

| Step | What was sped up | Speed-up |
|------|------------------|----------|
| `neighbors` distance kernel | tiled f32 vs `stats::dist` | up to ~4× |
| `neighbors` GPU kNN (`knn_backend = "vulkan"`) | fused `knn_tiled.comp` vs FNN kd-tree | ~2–3× at 25–50k cells, and grows — the kNN cost is linear in n rather than the kd-tree's ~O(n²) in ~10-D |
| `largest_gene` | sparse CSC argmax vs `qlcMatrix::colMax` | ~30× (≈20 s → 0.6 s at 53k cells) |
| `umap` pipeline | kd-tree kNN + sparse fuzzy graph + GPU SGD | ~13× (≈1.45 s → 0.11 s) |
| `umap` SGD shader alone | one GPU dispatch per epoch | ~10⁹ edge-updates/s |

The whole pipeline A/B is reproducible with `inst/examples/seurat_op2_gpu.R`,
which runs the classic Seurat route twice — stock CPU vs `RunGGML()` on the GPU —
on the Kaggle *Open Problems – Single-Cell Perturbations* counts (18 211 genes ×
240 090 PBMCs), then checks the two arms agree. A representative run (11 %
subsample = 23 279 cells, 2000 HVGs, 50 PCs, `--gpu-knn`, AMD RADV):

| step | cpu (s) | gpu (s) | speedup |
|------|--------:|--------:|--------:|
| `largest_gene` | 8.92 | 0.28 | 32.2× |
| `normalize` | 2.84 | 1.61 | 1.8× |
| `scale` | 1.84 | 1.94 | 0.9× |
| `embed` (PCA) | 15.92 | 3.03 | 5.3× |
| `neighbors` | 4.43 | 1.45 | 3.1× |
| `umap` | 15.32 | 5.34 | 2.9× |
| **TOTAL (GPU ops)** | **49.27** | **13.66** | **3.6×** |

Every accelerated step matches Seurat to float noise (normalize/scale max abs err
~1e-6, PCA `|cor|` = 1.0000 over PC1–10, clusters ARI 0.94, `largest_gene`
top-gene agreement 1.0000). `scale` comes out ~1× — it is memory-bound with
nothing to accelerate, so it defaults to the CPU even under Vulkan; PCA's
covariance multiply is the biggest matrix-multiply win. See the
`single-cell-seurat` vignette for the full breakdown.

(Numbers are hardware-dependent; reproduce the UMAP shaders separately with
`inst/examples/umap_shaders_bench.R`.)

```r
library(Seurat)

# A whole standard pipeline, GPU-accelerated end to end — every heavy step is a
# RunGGML() call; only FindClusters (graph Louvain) stays on Seurat's side:
pbmc <- RunGGML(pbmc, op = "normalize")                       # -> "data" layer
pbmc <- RunGGML(pbmc, op = "scale")                           # -> "scale.data" layer
pbmc <- RunGGML(pbmc, op = "embed", n_components = 30,
                reduction_name = "pca")                       # -> reduction "pca"

# UMAP on the PC coordinates — both phases on the GPU (distance + SGD shaders):
pbmc <- RunGGML(pbmc, op = "umap", reduction = "pca", dims = 1:30,
                reduction_name = "umap")                      # -> reduction "umap"

# Neighbour graphs on the PC coordinates -> @graphs, then cluster as usual:
pbmc <- RunGGML(pbmc, op = "neighbors", reduction = "pca", dims = 1:30)
pbmc <- FindClusters(pbmc, graph.name = paste0(DefaultAssay(pbmc), "_snn"))

DimPlot(pbmc, reduction = "umap", group.by = "seurat_clusters")
```

The `"umap"` op has two phases. The **graph/distance** phase uses a tiled f32
pairwise-distance kernel that sidesteps `mul_mat` (whose f16 accumulation reorders
nearest neighbours and corrupts the graph), with a kd-tree kNN and a sparse fuzzy
graph. The **SGD layout** phase defaults to the single-threaded reference, which
runs in compiled C (`R_umap_sgd_cpu`) — the earlier interpreted-R loop took
~700 s on ~10k cells; the C version is bit-for-bit identical but finishes in
seconds, so it stays the default for best embedding quality. Passing
`sgd_backend = "vulkan"` opts into the `umap_sgd.comp` shader (one dispatch per
epoch, Hogwild updates — faster still, but the lock-free races smear dense-graph
clusters, a known hard problem for async UMAP-SGD). Layout numerics match the CPU
reference to float32 precision; the SNN graph matches `FindNeighbors()`
bit-for-bit on identical exact kNN.

The adapter is layered, and each layer is a public generic you can call on its own:

| Function | Layer | Responsibility |
|----------|-------|----------------|
| `ggml_extract()` | Extraction | Pull a feature × cell matrix out of a `Seurat`/`dgCMatrix`/`matrix`; handles Seurat v4 (`GetAssayData`) vs v5 (`LayerData`) and sparse → dense |
| `ggml_run()` | Dispatch | Validate against `ggml_ops_registry()`, route to Vulkan GPU or CPU (auto, with fallback) |
| `ggml_inject()` | Injection | Write the result back into the object — a reduction (`CreateDimReducObject()`) for `"embed"`/`"umap"`, an assay layer for the `"normalize"`/`"scale"` transforms, or `<assay>_nn`/`<assay>_snn` `Graph` objects in `@graphs` for `"neighbors"` |

```r
# Compose the layers manually (e.g. on a bare matrix, no Seurat needed):
mat  <- ggml_extract(expr_matrix)               # genes × cells, dense
task <- ggml_task("embed", mat, params = list(n_components = 30))
res  <- ggml_run(task)                           # ggml_result: cells × components
res$embedding

# Check capabilities before dispatch:
ggml_ops_registry()        # all supported operations
ggml_ops_registry("embed") # required params + description
```

A `SingleCellExperiment` (Bioconductor) path with an S4 `runGGML()` generic is planned next.

## ONNX Model Import

Load pre-trained ONNX models from PyTorch, TensorFlow, or other frameworks and run inference on Vulkan GPU or CPU. No Python or external libraries required — ggmlR includes a built-in zero-dependency protobuf parser.

### Quick start

```r
library(ggmlR)

# 1. Load the model
model <- onnx_load("squeezenet.onnx")
model
#> ONNX Model: torch_jit
#>   Producer: pytorch 2.0.1
#>   IR version: 8 / Opset: 18
#>   Nodes: 66 / Weights: 26

# 2. Check expected inputs
onnx_inputs(model)
#> $x.1
#> [1]   1   3 224 224

# 3. Prepare input data (flat numeric vector, row-major NCHW order)
img <- runif(1 * 3 * 224 * 224)

# 4. Run inference — pass a named list matching input names
result <- onnx_run(model, list(x.1 = img))

# 5. Get predictions
scores <- result[[1]]
cat("Predicted class:", which.max(scores) - 1L, "\n")
```

### Loading models

`onnx_load()` parses the .onnx file, builds a ggml computation graph, and allocates tensors on the specified device. Weights are loaded from the file via memory-mapping (zero-copy).

```r
# Auto-detect device (Vulkan GPU if available, else CPU)
model <- onnx_load("model.onnx")

# Force CPU
model <- onnx_load("model.onnx", device = "cpu")

# Force Vulkan GPU
model <- onnx_load("model.onnx", device = "vulkan")
```

### Dynamic shapes

Some models (BERT, RoBERTa, etc.) have dynamic dimensions for batch size or sequence length. Specify fixed shapes at load time:

```r
model <- onnx_load("bert.onnx",
                    input_shapes = list(
                      input_ids      = c(1L, 128L),
                      attention_mask = c(1L, 128L)
                    ))
```

If you forget, `onnx_load()` will tell you which inputs need shapes:

```
Error: Input 'input_ids' has dynamic shape [?x?].
  Specify fixed shape via input_shapes parameter.
```

### Inspecting the model

```r
# Print overview
model
#> ONNX Model: torch_jit
#>   Nodes: 533 / Weights: 199
#>   Ops: MatMul, Add, LayerNormalization, Softmax, ...

# Detailed metadata
onnx_summary(model)

# Input names and shapes (what to pass to onnx_run)
onnx_inputs(model)

# Backend placement: GPU vs CPU split, scheduler info
onnx_device_info(model)
```

### Running inference

```r
# Single input model
result <- onnx_run(model, list(x = my_data))

# Multiple inputs
result <- onnx_run(model, list(
  input_ids      = as.numeric(token_ids),
  attention_mask = rep(1, 128)
))

# Result is a named list of output tensors
str(result)
#> List of 1
#>  $ output: num [1:1000] 0.00123 0.00045 ...
```

### Preparing input data

ONNX models expect inputs in **row-major** order (batch, channels, height, width for images). R matrices are column-major, so you may need to transpose:

```r
# Image classification: model expects [1, 3, 224, 224]
# If you have a 224x224x3 R array:
img_array <- array(runif(224 * 224 * 3), dim = c(224, 224, 3))

# Rearrange to NCHW: [1, 3, 224, 224] — channel first
img_chw <- aperm(img_array, c(3, 1, 2))  # [3, 224, 224]
input <- as.numeric(img_chw)              # flat vector, row-major

result <- onnx_run(model, list(x = input))
```

For NLP models, inputs are typically 1D integer sequences:

```r
# BERT-style: token IDs as numeric vector
tokens <- c(101, 2023, 2003, 1037, 3231, 102, rep(0, 122))  # pad to 128
result <- onnx_run(model, list(input_ids = as.numeric(tokens)))
```

### Interpreting outputs

```r
# Classification: get top-5 predictions
scores <- result[[1]]
top5 <- order(scores, decreasing = TRUE)[1:5]
cat("Top-5 classes:", top5 - 1L, "\n")  # 0-based class indices
cat("Top-5 scores:", scores[top5], "\n")

# Apply softmax if model outputs logits (not probabilities)
probs <- exp(scores) / sum(exp(scores))
```

### Repeated inference

Models can be run multiple times with zero overhead — weights live on GPU permanently and are never re-transferred:

```r
model <- onnx_load("classifier.onnx")

for (batch in data_batches) {
  result <- onnx_run(model, list(x = batch))
  # process result...
}
```

### Tested models

13 out of 15 ONNX Model Zoo models load and run successfully (native 5D tensor support):

| Model | Nodes | Key ops |
|---|---|---|
| mnist-8 | 12 | Conv, Relu, MaxPool, Reshape, MatMul |
| squeezenet1.0-8 | 66 | Conv, Relu, MaxPool, Concat, GlobalAveragePool, Softmax |
| adv_inception_v3 (Opset 17/18) | 215 | Conv, BatchNorm, Relu, Concat, AveragePool |
| emotion-ferplus-8 | 52 | Conv, Relu, MaxPool, Gemm, Constant |
| bat_resnext26ts (Opset 18) | 570 | Conv, BatchNorm, SiLU, Concat, Expand, Split |
| bert (Opset 17) | 533 | MatMul, LayerNorm, GELU/Erf, Softmax, Shape, Gather, Where |
| gptneox (Opset 18) | 482 | MatMul, LayerNorm, GELU, Softmax, Shape, Gather |
| MaskRCNN-12-int8 | 937 | QLinearConv, DequantizeLinear, Resize, Concat, Reshape |
| roberta-9 | 1180 | MatMul, LayerNorm, Erf, Softmax, Shape, Gather, Cast |
| sageconv (Opset 16) | 24 | MatMul, Add, Mul, Sigmoid, ScatterElements |
| super-resolution-10 | 12 | Conv, Reshape, Transpose |
| botnet26t_256 (Opset 16) | 530 | Conv, BatchNorm, RelPosBias2D (fused custom op), Softmax |
| xcit_tiny | 436 | MatMul, LayerNorm, Softmax, Concat, Transpose |

### Supported ONNX ops (50+)

Arithmetic: Add, Sub, Mul, Div, Pow, Sqrt, Exp, Log, Abs, Neg, Floor, Ceil, Clip, Erf, Equal.
Linear: MatMul (batched), Gemm.
Convolution: Conv (1D/2D, grouped, depthwise), ConvTranspose (1D/2D), with `auto_pad` (SAME_UPPER, SAME_LOWER).
Pooling: MaxPool, AveragePool, GlobalAveragePool, Resize/Upsample (nearest, bilinear).
Normalization: BatchNorm, LayerNorm, GroupNorm, RMSNorm.
Activations: Relu, Sigmoid, Tanh, GELU, SiLU, Softmax, LeakyRelu, Elu.
Shape: Reshape, Transpose, Concat, Flatten, Squeeze, Unsqueeze, Expand, Slice, Split, Gather, Pad, Shape, Cast, Identity, EyeLike.
Constants: Constant (TensorProto + scalar), ConstantOfShape (INT64/INT32/DOUBLE/FLOAT value).
Scatter/Gather: ScatterElements (axis=0, reduction=none/add, Vulkan atomicAdd), Gather (axis=0 on rank>2 via reshape).
Logic: Where, Equal.
Reduction: ReduceMean, ReduceSum.
Quantization: DequantizeLinear, QuantizeLinear, QLinearConv, QLinearAdd, QLinearMatMul, QLinearSigmoid, QLinearConcat.
Fused custom ops: RelPosBias2D (BoTNet-style 2D relative position bias).
Pass-through: Dropout.

## GGUF Pre-trained Weights

Load pre-trained weights from GGUF files (llama.cpp, Hugging Face, etc.) with automatic dequantization. Supports all ggml quantization types (F32, F16, Q4_0, Q8_0, K-quants, IQ, MXFP4, Q1_0, NVFP4).

```r
library(ggmlR)

# Load a GGUF file
g <- gguf_load("model.gguf")
g
#> GGUF file: model.gguf
#>   Version:  3
#>   Tensors:  291
#>   Metadata: 24 key-value pairs

# Inspect metadata (architecture, tokenizer, quant info)
meta <- gguf_metadata(g)
meta[["general.architecture"]]

# List all tensor names
gguf_tensor_names(g)

# Get shape and type for a specific tensor
gguf_tensor_info(g, "blk.0.attn_q.weight")
#> $name:  "blk.0.attn_q.weight"
#> $shape: 4096 4096
#> $type:  "Q4_0"

# Extract dequantized weights as R numeric array
w <- gguf_tensor_data(g, "blk.0.attn_q.weight")
dim(w)
#> [1] 4096 4096

# Free when done (also freed by GC)
gguf_free(g)
```

## Examples

Ready-to-run example scripts in `inst/examples/`:

| Script | Description |
|---|---|
| `titanic_classification.R` | Binary classification with feature engineering, dropout, stratified split, manual metrics (~82% val accuracy) |
| `mnist_cnn.R` | CNN image classifier on MNIST |
| `functional_resnet_cifar.R` | ResNet-style model with skip connections (Functional API) |
| `functional_text_gru.R` | Text classification with GRU + embedding (Functional API) |
| `transformer_encoder_demo.R` | Transformer encoder with multi-head attention (autograd) |
| `dp_train_demo.R` | Data-parallel training across multiple replicas |
| `benchmark_onnx.R` | GPU vs CPU inference benchmark for ONNX models |
| `benchmark_ops.R` | Per-op micro-benchmark: every ggml op on CPU and GPU with auto-batching |
| `profile_onnx_superres_gpu.R` | GPU profiler for SuperResolution ONNX model across input sizes |
| `mlr3_integration.R` | mlr3 learners: CPU vs GPU comparison, iris + mtcars, 3-fold CV |
| `tidymodels_integration.R` | parsnip engine: CPU vs GPU comparison, iris + mtcars |

## Save / Load

```r
ggml_save_model(model, "my_model.rds")
model <- ggml_load_model("my_model.rds")
```

## ONNX Benchmark: GPU (Vulkan) vs CPU

Measured on AMD Ryzen 5 5600 + AMD RX 9070, single-image inference:

| Model | CPU (ms) | GPU (ms) | Speedup | CPU FPS | GPU FPS |
|---|---:|---:|---:|---:|---:|
| Inception V3 | 265.3 | 9.7 | 27.5x | 3.8 | 103.4 |
| MNIST | 0.0 | 0.0 | — | Inf | Inf |
| SqueezeNet 1.0 | 22.3 | 1.7 | 13.4x | 44.8 | 600.0 |
| SuperResolution | 100.0 | 3.3 | 30.0x | 10.0 | 300.0 |
| EmotionFerPlus | 31.3 | 2.3 | 13.4x | 31.9 | 428.6 |
| Inception V3 Op18 | 204.7 | 8.3 | 24.6x | 4.9 | 120.0 |
| BAT-ResNeXt26ts | 116.0 | 7.3 | 15.8x | 8.6 | 136.4 |
| BERT (Opset17) | 243.7 | 9.0 | 27.1x | 4.1 | 111.1 |
| GPT-NeoX | 1.3 | 3.3 | 0.4x | 750.0 | 300.0 |

Benchmark scripts: `inst/examples/benchmark_onnx.R`, `inst/examples/profile_onnx_superres_gpu.R`

## LLM Inference Benchmark

Measured on AMD Ryzen 5 5600 + AMD RX 9070, via [llamaR](https://github.com/Zabis13/llamaR). Model: Ministral-3-3B-Instruct-2512-Q8_0, 50 tokens, avg of 3 runs:

| Backend | Speed (tokens/sec) | Speedup |
|---|---:|---:|
| CPU (8 threads) | 8.5 | 1.0x |
| GPU (Vulkan) | 108.0 | 12.7x |

## Flux Image Generation Benchmark (10 steps)

Measured on AMD Ryzen 5 5600 + AMD RX 9070, via [sd2R](https://github.com/Zabis13/sd2R):

| Scenario | Resolution | Time (s) |
|---|---|---:|
| Direct | 768×768 | 13.94 |
| Tiled VAE | 1024×1024 | 25.32 |
| Highres fix | 2048×1024 | 52.53 |
| img2img | 768×768 | 8.73 |
| Direct | 1024×1024 | 25.40 |

## GPU Acceleration

ggmlR is designed GPU-first: Vulkan is auto-detected at build time and, when available, 90%+ of operations run on GPU with up to 78x speedup over CPU. On machines without a Vulkan-capable GPU the package falls back to CPU transparently — no code changes required.

```r
ggml_vulkan_available()   # TRUE if a Vulkan GPU was detected
ggml_vulkan_status()      # device name, memory, capabilities

# Dynamic autograd: switch device at runtime
ag_device("gpu")   # move subsequent ops to GPU (f16 by default)
ag_device("cpu")   # fall back to CPU
```

Supported GPUs: NVIDIA, AMD, Intel, ARM Mali, Qualcomm Adreno.

### Vulkan optimizations

- **Vulkan 1.4** — push constants limit raised to 256 bytes, enabling full 5D tensor parameter blocks in compute shaders without staging buffers.
- **Push Descriptors** (`VK_KHR_push_descriptor`) — when available, descriptors are pushed directly into the command buffer, eliminating descriptor pool allocation overhead. Falls back to descriptor pools on older hardware.
- **Q4_K flash attention** — `GGML_OP_FLASH_ATTN_EXT` with Q4_K key/value tensors now runs fully on GPU (FA_SCALAR and FA_COOPMAT1 paths). Previously Q4_K attention fell back to CPU. Relevant for llamaR with quantized LLMs on AMD/Intel GPU (KHR cooperative matrix).
- **Subgroup-shuffle mmq** (`USE_SUBGROUP_NO_SHMEM`) — on wavefront-64 devices (RDNA4, subgroup_size=64) Q4_K / Q5_K / Q6_K weight tiles are loaded directly into registers via `subgroupShuffle`, eliminating the shared-memory staging round-trip. ~10-15% token-generation throughput gain on LLaMA 3.x models.

## System Requirements

- R ≥ 4.1.0, C++17 compiler
- **Optional GPU**: Vulkan 1.2+, `libvulkan-dev` + `glslc` (Linux) or Vulkan SDK (Windows)
- Platforms: Linux, macOS, Windows (x86-64, ARM64)

## See Also

- [llamaR](https://github.com/Zabis13/llamaR) — LLM inference in R
- [sd2R](https://github.com/Zabis13/sd2R) — Stable Diffusion in R
- [ggml](https://github.com/ggml-org/ggml) — underlying C library

## License

MIT
