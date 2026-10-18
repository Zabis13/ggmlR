#!/usr/bin/env Rscript
# Save / load / trainer / freeze round trip over a minimal set of architectures
# that between them cover every kind of saved state and both graph builders:
#
#   train on the CPU -> predict -> save -> load -> same predictions
#   -> a trainer on the loaded model: $predict() and $snapshot() agree
#   -> freeze one layer (or a block by prefix), train on: it stays, the rest move
#   -> on Vulkan with matmul_precision = "f32": the CPU's predictions
#
# The layers are the ones README lists. The last architecture is a scaled-down
# copy of the cayleyR Q-model (dense per position, positional embedding, two
# RMS transformer blocks, mean pooling, 18 outputs).
#
# Prints one line per architecture and step: PASS, FAIL with the measured
# number, or ERROR with the message. Nothing is left on disk.
#
# Run: Rscript inst/examples/diag_arch_roundtrip.R

suppressPackageStartupMessages(library(ggmlR))

N  <- 32L
BS <- 16L

mat <- function(n, p, mean = 0) matrix(rnorm(n * p, mean = mean), n, p)
arr <- function(...) array(rnorm(prod(c(...))), c(...))

# Weight keys a freeze call reaches: layer indices for a sequential model,
# node ids for a functional one.
fun_keys <- function(model, layer_names = NULL, prefix = NULL) {
  nodes <- ggmlR:::nn_topo_sort(model$outputs)
  nms   <- vapply(nodes, function(nd)
    if (is.null(nd$config$name)) "" else nd$config$name, character(1))
  hit <- nms %in% layer_names
  for (p in prefix) hit <- hit | startsWith(nms, p)
  vapply(nodes[hit], `[[`, character(1), "id")
}

saved_weights <- function(f) {
  d <- readRDS(f)
  if (!is.null(d$layers))
    setNames(lapply(d$layers, `[[`, "weights_data"), seq_along(d$layers))
  else d$node_weights_data
}

seq_compile <- function(m) ggml_compile(m, optimizer = "adam", loss = "mse",
                                        backend = "cpu")
fun_compile <- function(inp, out) ggml_compile(ggml_model(inputs = inp, outputs = out),
                                               optimizer = "adam", loss = "mse",
                                               backend = "cpu")

# Each entry: x, y, and build() returning the compiled model, the freeze
# arguments, and the keys of what that freeze must hold still.
ARCHS <- list(
  mlp = function() {
    m <- ggml_model_sequential() |>
      ggml_layer_dense(16L, activation = "relu", input_shape = 6L) |>
      ggml_layer_dense(3L)
    list(model = seq_compile(m), freeze = list(from = 1L, to = 1L), keys = "1",
         x = mat(N, 6L), y = mat(N, 3L))
  },
  dropout = function() {
    m <- ggml_model_sequential() |>
      ggml_layer_dense(16L, activation = "relu", input_shape = 6L) |>
      ggml_layer_dropout(0.3) |>
      ggml_layer_dense(3L)
    list(model = seq_compile(m), freeze = list(from = 1L, to = 1L), keys = "1",
         x = mat(N, 6L), y = mat(N, 3L))
  },
  conv1d = function() {
    m <- ggml_model_sequential() |>
      ggml_layer_conv_1d(4L, 3L, activation = "relu", input_shape = c(10L, 2L)) |>
      ggml_layer_flatten() |>
      ggml_layer_dense(3L)
    list(model = seq_compile(m), freeze = list(from = 1L, to = 1L), keys = "1",
         x = arr(N, 10L, 2L), y = mat(N, 3L))
  },
  conv2d_pool = function() {
    m <- ggml_model_sequential() |>
      ggml_layer_conv_2d(4L, c(3L, 3L), activation = "relu",
                         input_shape = c(8L, 8L, 1L)) |>
      ggml_layer_max_pooling_2d(c(2L, 2L)) |>
      ggml_layer_flatten() |>
      ggml_layer_dense(3L)
    list(model = seq_compile(m), freeze = list(from = 1L, to = 1L), keys = "1",
         x = arr(N, 8L, 8L, 1L), y = mat(N, 3L))
  },
  batch_norm = function() {
    m <- ggml_model_sequential() |>
      ggml_layer_dense(16L, input_shape = 6L) |>
      ggml_layer_batch_norm() |>
      ggml_layer_dense(3L)
    list(model = seq_compile(m), freeze = list(from = 1L, to = 1L), keys = "1",
         x = mat(N, 6L, mean = 2), y = mat(N, 3L))
  },
  lstm = function() {
    m <- ggml_model_sequential() |>
      ggml_layer_lstm(8L, input_shape = c(5L, 3L)) |>
      ggml_layer_dense(3L)
    list(model = seq_compile(m), freeze = list(from = 1L, to = 1L), keys = "1",
         x = arr(N, 5L, 3L), y = mat(N, 3L))
  },
  gru = function() {
    m <- ggml_model_sequential() |>
      ggml_layer_gru(8L, input_shape = c(5L, 3L)) |>
      ggml_layer_dense(3L)
    list(model = seq_compile(m), freeze = list(from = 1L, to = 1L), keys = "1",
         x = arr(N, 5L, 3L), y = mat(N, 3L))
  },
  layer_norm = function() {
    inp <- ggml_input(shape = 6L)
    out <- inp |> ggml_layer_dense(16L, name = "d1") |>
      ggml_layer_layer_norm(name = "ln") |> ggml_layer_dense(3L)
    m <- fun_compile(inp, out)
    list(model = m, freeze = list(layer_names = "ln"), keys = fun_keys(m, "ln"),
         x = mat(N, 6L), y = mat(N, 3L))
  },
  embedding_pos = function() {
    inp <- ggml_input(shape = 6L, dtype = "int32")
    out <- inp |> ggml_layer_embedding(10L, 8L, name = "emb") |>
      ggml_layer_positional_embedding(name = "pos") |>
      ggml_layer_flatten() |> ggml_layer_dense(3L)
    m <- fun_compile(inp, out)
    list(model = m, freeze = list(layer_names = "emb"), keys = fun_keys(m, "emb"),
         x = matrix(as.numeric(sample(0:9, N * 6L, TRUE)), N, 6L), y = mat(N, 3L))
  },
  attention = function() {
    inp <- ggml_input(shape = c(6L, 8L))
    out <- inp |> ggml_layer_attention(8L, n_heads = 2L, name = "att") |>
      ggml_layer_sequence_pooling(mode = "mean") |> ggml_layer_dense(3L)
    m <- fun_compile(inp, out)
    list(model = m, freeze = list(layer_names = "att"), keys = fun_keys(m, "att"),
         x = arr(N, 6L, 8L), y = mat(N, 3L))
  },
  transformer_block = function() {
    inp <- ggml_input(shape = c(6L, 8L))
    out <- inp |> ggml_layer_transformer_block(8L, n_heads = 2L, ff_dim = 16L,
                                               name = "b1") |>
      ggml_layer_sequence_pooling(mode = "mean") |> ggml_layer_dense(3L)
    m <- fun_compile(inp, out)
    list(model = m, freeze = list(prefix = "b1_"), keys = fun_keys(m, prefix = "b1_"),
         x = arr(N, 6L, 8L), y = mat(N, 3L))
  },
  residual = function() {
    inp <- ggml_input(shape = 6L)
    d1  <- ggml_layer_dense(inp, 6L, activation = "relu", name = "d1")
    out <- ggml_layer_add(list(inp, d1)) |> ggml_layer_dense(3L, name = "head")
    m <- fun_compile(inp, out)
    list(model = m, freeze = list(layer_names = "d1"), keys = fun_keys(m, "d1"),
         x = mat(N, 6L), y = mat(N, 3L))
  },
  cayley_q = function() {
    inp <- ggml_input(shape = c(10L, 12L))
    h   <- ggml_layer_dense(inp, 16L, time_distributed = TRUE, name = "in")
    h   <- ggml_layer_positional_embedding(h, name = "pos")
    for (b in 1:2)
      h <- ggml_layer_transformer_block(h, 16L, n_heads = 2L, ff_dim = 32L,
                                        activation = "silu", norm = "rms",
                                        name = paste0("blk", b))
    out <- h |> ggml_layer_sequence_pooling(mode = "mean") |>
      ggml_layer_dense(18L, name = "head")
    m <- fun_compile(inp, out)
    list(model = m, freeze = list(prefix = "blk1_"),
         keys = fun_keys(m, prefix = "blk1_"),
         x = arr(N, 10L, 12L), y = mat(N, 18L))
  }
)

rel <- function(a, b) max(abs(a - b)) / max(abs(b), 1e-12)

has_vulkan <- ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L
n_fail <- 0L

report <- function(arch, step, ok, detail = "") {
  if (!isTRUE(ok)) n_fail <<- n_fail + 1L
  cat(sprintf("%-6s %-18s %-10s %s\n", if (isTRUE(ok)) "PASS" else "FAIL",
              arch, step, detail))
}

# One step: run it, report PASS/FAIL from its result, or ERROR and carry on.
step <- function(arch, name, expr) {
  r <- tryCatch(expr, error = function(e) {
    n_fail <<- n_fail + 1L
    cat(sprintf("%-6s %-18s %-10s %s\n", "ERROR", arch, name, conditionMessage(e)))
    NULL
  })
  if (!is.null(r)) report(arch, name, r$ok, r$detail)
  invisible(r)
}

for (arch in names(ARCHS)) {
  set.seed(1)
  a <- tryCatch(ARCHS[[arch]](), error = function(e) {
    cat(sprintf("%-6s %-18s %-10s %s\n", "ERROR", arch, "build", conditionMessage(e)))
    NULL
  })
  if (is.null(a)) { n_fail <- n_fail + 1L; next }

  f <- tempfile(fileext = ".rds"); f2 <- tempfile(fileext = ".rds")
  p <- NULL
  step(arch, "fit", {
    m <- ggml_fit(a$model, a$x, a$y, epochs = 1L, batch_size = BS, verbose = 0L)
    p <- ggml_predict(m, a$x, batch_size = BS)
    ggml_save_model(m, f)
    list(ok = all(is.finite(p)), detail = "")
  })
  if (is.null(p)) { unlink(c(f, f2)); next }

  step(arch, "load", {
    d <- rel(ggml_predict(ggml_load_model(f, backend = "cpu"), a$x, batch_size = BS), p)
    list(ok = d < 1e-5, detail = sprintf("rel %.2g", d))
  })

  step(arch, "trainer", {
    tr <- suppressWarnings(ggml_trainer(ggml_load_model(f, backend = "cpu"),
                                        batch_size = BS))   # batch_norm warns
    d1 <- rel(tr$predict(a$x), p)
    snap <- tr$snapshot()
    d2 <- rel(ggml_predict(snap, a$x, batch_size = BS), p)
    ggml_snapshot_free(snap)
    tr$free()
    list(ok = d1 < 1e-5 && d2 < 1e-5,
         detail = sprintf("$predict rel %.2g, snapshot rel %.2g", d1, d2))
  })

  step(arch, "freeze", {
    m3 <- do.call(ggml_freeze_weights,
                  c(list(ggml_load_model(f, backend = "cpu")), a$freeze))
    m3 <- ggml_fit(m3, a$x, a$y, epochs = 2L, batch_size = BS, verbose = 0L)
    ggml_save_model(m3, f2)
    w0 <- saved_weights(f); w1 <- saved_weights(f2)
    keys   <- intersect(a$keys, names(Filter(length, w0)))
    held   <- length(keys) > 0L &&
              all(vapply(keys, function(k) isTRUE(all.equal(w1[[k]], w0[[k]])),
                         logical(1)))
    others <- setdiff(names(Filter(length, w0)), keys)
    moved  <- sum(vapply(others, function(k) !isTRUE(all.equal(w1[[k]], w0[[k]])),
                         logical(1)))
    list(ok = held && moved > 0L,
         detail = sprintf("%d frozen held: %s; %d of %d others moved",
                          length(keys), held, moved, length(others)))
  })

  if (has_vulkan) step(arch, "vulkan", {
    vk <- ggml_load_model(f, backend = "vulkan")
    d  <- rel(ggml_predict(vk, a$x, batch_size = BS, matmul_precision = "f32"), p)
    list(ok = d < 1e-4, detail = sprintf("f32 vs CPU rel %.2g", d))
  })

  unlink(c(f, f2))
}

# A trainer takes one input and one output; a multi-input model is refused.
step("multi_input", "trainer", {
  ia  <- ggml_input(shape = 4L, name = "a")
  ib  <- ggml_input(shape = 4L, name = "b")
  out <- ggml_layer_add(list(ggml_layer_dense(ia, 4L), ggml_layer_dense(ib, 4L))) |>
    ggml_layer_dense(2L)
  m <- ggml_compile(ggml_model(inputs = list(ia, ib), outputs = out),
                    optimizer = "adam", loss = "mse", backend = "cpu")
  msg <- tryCatch({ ggml_trainer(m, batch_size = 8L); "" },
                  error = function(e) conditionMessage(e))
  list(ok = grepl("one input and one output", msg), detail = msg)
})

if (!has_vulkan) cat("\n(no Vulkan device: the vulkan step was skipped)\n")
cat(sprintf("\n%s\n", if (n_fail == 0L) "all passed" else
            sprintf("%d step(s) failed", n_fail)))
