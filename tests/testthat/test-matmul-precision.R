# Matrix-multiply precision at prediction: predict_precision in ggml_compile(),
# matmul_precision per call. On an fp16 Vulkan device the default multiply keeps
# f16 tiles for batches above a few rows (relative error ~1e-3, measured on an
# RX 9070 / RADV with inst/examples/benchmark_coopmat.R); "f32" marks every
# multiply GGML_PREC_F32 and matches the CPU to ~1e-6.

prec_data <- function(n = 64L, nf = 32L) {
  set.seed(1)
  list(x = matrix(rnorm(n * nf), n, nf), y = matrix(rnorm(n * 4L), n, 4L))
}

# Trained on the CPU and saved, so every backend loads the same weights.
saved_prec_model <- function(d, predict_precision = "default") {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(64L, activation = "relu", input_shape = ncol(d$x)) |>
    ggml_layer_dense(64L, activation = "relu") |>
    ggml_layer_dense(4L)
  m <- ggml_compile(m, optimizer = "adam", loss = "mse", backend = "cpu",
                    predict_precision = predict_precision)
  m <- ggml_fit(m, d$x, d$y, epochs = 3L, batch_size = nrow(d$x), verbose = 0L)
  f <- tempfile(fileext = ".rds")
  ggml_save_model(m, f)
  f
}

skip_without_vulkan <- function()
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")

rel <- function(a, b) max(abs(a - b)) / max(abs(b))

test_that("precision values are checked", {
  m <- ggml_model_sequential() |> ggml_layer_dense(2L, input_shape = 4L)
  expect_error(ggml_compile(m, loss = "mse", predict_precision = "f64"),
               "predict_precision")
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  expect_error(ggml_predict(m, matrix(0, 4L, 4L), batch_size = 4L,
                            matmul_precision = "half"),
               "matmul_precision")
})

test_that("predict_precision survives save and load", {
  d <- prec_data()
  f <- saved_prec_model(d, "f32"); on.exit(unlink(f))
  expect_identical(ggml_load_model(f, backend = "cpu")$compilation$predict_precision,
                   "f32")
})

test_that("on the CPU both precisions give the same result", {
  d <- prec_data()
  f <- saved_prec_model(d); on.exit(unlink(f))
  m <- ggml_load_model(f, backend = "cpu")
  expect_identical(ggml_predict(m, d$x, batch_size = 64L, matmul_precision = "f32"),
                   ggml_predict(m, d$x, batch_size = 64L))
})

test_that("on Vulkan, f32 matches the CPU and does not depend on the batch", {
  skip_without_vulkan()
  d <- prec_data()
  f <- saved_prec_model(d); on.exit(unlink(f))
  ref <- ggml_predict(ggml_load_model(f, backend = "cpu"), d$x, batch_size = 64L)
  vk  <- ggml_load_model(f, backend = "vulkan")

  p64 <- ggml_predict(vk, d$x, batch_size = 64L, matmul_precision = "f32")
  p1  <- do.call(rbind, lapply(1:8, function(i)
    ggml_predict(vk, d$x[i, , drop = FALSE], batch_size = 1L,
                 matmul_precision = "f32")))
  expect_lt(rel(p64, ref), 1e-5)
  expect_lt(rel(p1, p64[1:8, ]), 1e-5)
})

test_that("on Vulkan, the default stays within the known f16 error", {
  skip_without_vulkan()
  # Not exact by design: f16 tiles give ~1e-3 relative error for batches above
  # a few rows. This pins that bound, so a regression past it shows up.
  d <- prec_data()
  f <- saved_prec_model(d); on.exit(unlink(f))
  ref <- ggml_predict(ggml_load_model(f, backend = "cpu"), d$x, batch_size = 64L)
  p64 <- ggml_predict(ggml_load_model(f, backend = "vulkan"), d$x, batch_size = 64L)
  expect_lt(rel(p64, ref), 2e-3)
})

test_that("the model's predict_precision applies unless a call overrides it", {
  skip_without_vulkan()
  d <- prec_data()
  f <- saved_prec_model(d, "f32"); on.exit(unlink(f))
  ref <- ggml_predict(ggml_load_model(f, backend = "cpu"), d$x, batch_size = 64L)
  vk  <- ggml_load_model(f, backend = "vulkan")
  expect_lt(rel(ggml_predict(vk, d$x, batch_size = 64L), ref), 1e-5)
  expect_lt(rel(ggml_predict(vk, d$x, batch_size = 64L,
                             matmul_precision = "default"), ref), 2e-3)
})

test_that("a trainer's $predict takes matmul_precision", {
  skip_without_vulkan()
  d <- prec_data()
  f <- saved_prec_model(d); on.exit(unlink(f))
  ref <- ggml_predict(ggml_load_model(f, backend = "cpu"), d$x, batch_size = 64L)
  tr  <- ggml_trainer(ggml_load_model(f, backend = "vulkan"), batch_size = 64L)
  on.exit(tr$free(), add = TRUE)
  expect_lt(rel(tr$predict(d$x, matmul_precision = "f32"), ref), 1e-5)

  target <- tr$snapshot()
  on.exit(ggml_snapshot_free(target), add = TRUE)
  expect_lt(rel(ggml_predict(target, d$x, batch_size = 64L,
                             matmul_precision = "f32"), ref), 1e-5)
})
