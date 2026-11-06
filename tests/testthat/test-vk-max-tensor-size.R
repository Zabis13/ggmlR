# max_tensor_size in ggml_vulkan_device_memory() must be the exact limit the
# Vulkan backend enforces in supports_op (TODO 9.5.5a): sd2R plans VAE tiling
# against it, so a tensor at the limit must run on the GPU and one row more
# must be refused. Tensors live in a no_alloc context -- nothing is allocated.

test_that("max_tensor_size is a positive finite byte count", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  mem <- ggml_vulkan_device_memory(0)
  expect_named(mem, c("free", "total", "max_tensor_size"))
  expect_true(is.numeric(mem$max_tensor_size))
  expect_true(is.finite(mem$max_tensor_size))
  expect_gt(mem$max_tensor_size, 0)
})

test_that("supports_op refuses exactly above max_tensor_size", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  lim <- ggml_vulkan_device_memory(0)$max_tensor_size
  skip_if(lim > 64 * 2^30, "limit above 64 GiB (64-bit indexing): boundary not probed")

  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(gpu), add = TRUE)
  dev <- ggml_backend_get_device(gpu)
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  row_bytes <- 1024 * 4                       # one F32 row of 1024
  rows_fit  <- floor(lim / row_bytes)         # rows_fit * row_bytes <= lim

  fits <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, rows_fit)
  over <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, rows_fit + 1)

  expect_true(ggml_backend_dev_supports_op(dev, ggml_scale(ctx, fits, 2)),
              label = sprintf("SCALE on %.0f bytes (limit %.0f)", rows_fit * row_bytes, lim))
  expect_false(ggml_backend_dev_supports_op(dev, ggml_scale(ctx, over, 2)),
               label = sprintf("SCALE on %.0f bytes (limit %.0f)", (rows_fit + 1) * row_bytes, lim))
})
