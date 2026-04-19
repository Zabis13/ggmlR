library(ggmlR)

skip_if_no_vulkan <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

# ---- caps fields -------------------------------------------------------------

test_that("ggml_vulkan_device_caps returns supports_256_push_constants and max_push_constants_size", {
  skip_if_no_vulkan()
  caps <- ggml_vulkan_device_caps(0L)
  expect_true("supports_256_push_constants" %in% names(caps))
  expect_true("max_push_constants_size"      %in% names(caps))
  expect_type(caps$supports_256_push_constants, "logical")
  expect_type(caps$max_push_constants_size,      "integer")
})

test_that("max_push_constants_size meets Vulkan spec minimum of 128 bytes", {
  skip_if_no_vulkan()
  caps <- ggml_vulkan_device_caps(0L)
  expect_gte(caps$max_push_constants_size, 128L)
})

test_that("supports_256_push_constants is consistent with max_push_constants_size", {
  skip_if_no_vulkan()
  caps <- ggml_vulkan_device_caps(0L)
  if (caps$max_push_constants_size >= 256L) {
    expect_true(caps$supports_256_push_constants)
  } else {
    expect_false(caps$supports_256_push_constants)
  }
})

test_that("supports_256_push_constants is TRUE — ggml_vulkan_init would have aborted otherwise", {
  skip_if_no_vulkan()
  # ggml_vk_init() calls r_ggml_error() if maxPushConstantsSize < 256,
  # so if we reach this point the Vulkan backend is already initialised and
  # the capability must be TRUE.
  caps <- ggml_vulkan_device_caps(0L)
  expect_true(caps$supports_256_push_constants)
})

# ---- 5D ops require 256-byte push constants ---------------------------------

run_5d_add <- function(device, a_vals, b_vals) {
  ne  <- c(4L, 3L, 2L, 5L, 2L)
  n   <- prod(ne)
  env <- new.env(parent = emptyenv())
  env$a <- NULL; env$b <- NULL

  build <- function(ctx) {
    env$a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, ne)
    env$b <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, ne)
    ggml_add(ctx, env$a, env$b)
  }

  ctx     <- ggml_init(mem_size = 16L * 1024L * 1024L, no_alloc = TRUE)
  out     <- build(ctx)
  backend <- if (device == "cpu") {
    b <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(b, 2L); b
  } else {
    ggml_vulkan_init(0L)
  }
  buf   <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(env$a, a_vals)
  ggml_backend_tensor_set_data(env$b, b_vals)
  graph <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_backend_tensor_get_data(out, n_elements = n)
  ggml_backend_buffer_free(buf)
  ggml_backend_free(backend)
  ggml_free(ctx)
  result
}

test_that("5D add CPU matches R reference", {
  ne <- c(4L, 3L, 2L, 5L, 2L); n <- prod(ne)
  set.seed(1L); a <- runif(n); b <- runif(n)
  r <- run_5d_add("cpu", a, b)
  expect_length(r, n)
  expect_true(all(is.finite(r)))
  expect_lt(max(abs(r - (a + b))), 1e-4,
            label = "5D add CPU vs R reference")
})

test_that("5D add Vulkan matches R reference", {
  skip_if_no_vulkan()
  ne <- c(4L, 3L, 2L, 5L, 2L); n <- prod(ne)
  set.seed(1L); a <- runif(n); b <- runif(n)
  gpu <- run_5d_add("vulkan", a, b)
  expect_length(gpu, n)
  expect_true(all(is.finite(gpu)), label = "GPU output contains NaN/Inf")
  expect_lt(max(abs(gpu - (a + b))), 1e-4,
            label = "5D add GPU vs R reference")
})

run_5d_concat <- function(device, a_vals, b_vals) {
  ne    <- c(4L, 3L, 2L, 5L, 2L)
  n_out <- prod(c(4L, 3L, 2L, 5L, 4L))
  env <- new.env(parent = emptyenv())
  ctx <- ggml_init(mem_size = 16L * 1024L * 1024L, no_alloc = TRUE)
  env$a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, ne)
  env$b <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, ne)
  out <- ggml_concat(ctx, env$a, env$b, dim = 4L)
  backend <- if (device == "cpu") {
    b <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(b, 2L); b
  } else {
    ggml_vulkan_init(0L)
  }
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(env$a, a_vals)
  ggml_backend_tensor_set_data(env$b, b_vals)
  graph <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)
  result <- ggml_backend_tensor_get_data(out, n_elements = n_out)
  ggml_backend_buffer_free(buf); ggml_backend_free(backend); ggml_free(ctx)
  result
}

test_that("5D concat axis=4 CPU produces finite output", {
  ne <- c(4L, 3L, 2L, 5L, 2L); n <- prod(ne)
  n_out <- prod(c(4L, 3L, 2L, 5L, 4L))
  set.seed(2L); a <- runif(n); b <- runif(n)
  r <- run_5d_concat("cpu", a, b)
  expect_length(r, n_out)
  expect_true(all(is.finite(r)))
})

test_that("5D concat axis=4 Vulkan matches CPU", {
  skip_if_no_vulkan()
  ne <- c(4L, 3L, 2L, 5L, 2L); n <- prod(ne)
  n_out <- prod(c(4L, 3L, 2L, 5L, 4L))
  set.seed(2L); a <- runif(n); b <- runif(n)
  cpu <- run_5d_concat("cpu", a, b)
  gpu <- run_5d_concat("vulkan", a, b)
  expect_length(gpu, n_out)
  expect_true(all(is.finite(gpu)), label = "GPU concat axis=4 output contains NaN/Inf")
  expect_lt(max(abs(cpu - gpu)), 1e-4,
            label = sprintf("5D concat axis=4 CPU vs GPU max diff = %.2e", max(abs(cpu - gpu))))
})
