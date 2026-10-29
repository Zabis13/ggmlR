# Vulkan async set/get (ggml_backend_tensor_set_async / get_async) round-trip
# their bytes, with and without the separate transfer queue.
#
# After the ggml 0.27 sync ggml_vk_synchronize only submitted the transfer
# context through ggml_vk_submit_transfer_ctx, while set/get_tensor_async were
# still the old ones that recorded into a transfer context on the compute pool:
# without a transfer queue the copies were never submitted, with one the
# staging memcpys were dropped. The whole transport block is now 0.27.
#
# R vectors are never pinned memory, so every case here takes the staging
# path -- the one that lost data. Sizes: tiny, unaligned (odd sizes and byte
# offsets), and large (the staging buffer has to grow).
#
# The transfer queue is chosen once, when the device is created, so each mode
# runs in its own R process: by default AMD RDNA dGPUs use it;
# GGML_VK_ALLOW_GRAPHICS_QUEUE=1 turns it off there. On other devices both runs
# take the compute queue, which still checks that path.

async_script <- '
library(ggmlR)
fail <- function(...) { cat("FAIL", sprintf(...), "\n"); quit(status = 1) }

roundtrip <- function(n_bytes, offset) {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  # byte-addressed: offset/size are bytes, the tensor type only sets capacity
  t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ceiling((n_bytes + offset + 8) / 4))
  be <- ggml_vulkan_init(0)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  set.seed(n_bytes + offset)
  src <- runif(ceiling(n_bytes / 8) + 1)            # any bytes will do
  ggml_backend_tensor_set_async(be, t, src, offset = offset, size = n_bytes)
  dst <- ggml_backend_tensor_get_async(be, t, offset = offset, size = 8 * ceiling(n_bytes / 8))
  ggml_backend_synchronize(be)                      # get_async fills dst here
  want <- writeBin(src, raw())[seq_len(n_bytes)]
  got  <- writeBin(dst, raw())[seq_len(n_bytes)]
  ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx)
  if (!identical(got, want))
    fail("roundtrip n=%d off=%d: %d bytes differ", n_bytes, offset, sum(got != want))
}

for (n in c(1, 3, 8, 1023, 4097, 65536 + 5)) for (off in c(0, 1, 5)) roundtrip(n, off)
roundtrip(48 * 1024 * 1024 + 3, 1)                 # large: staging buffer must grow

# ordering: async upload -> async graph -> async readback, one synchronize
ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4099)
y <- ggml_scale(ctx, x, 3)
gf <- ggml_build_forward_expand(ctx, y)
be <- ggml_vulkan_init(0)
buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
for (rep in 1:5) {
  v <- seq_len(4099) / 7 + rep
  # f32 bytes carried in a double vector (the wrapper copies raw bytes);
  # `packed` stays alive until synchronize, as the deferred memcpy needs it
  bytes <- c(writeBin(v, raw(), size = 4), raw(4))
  packed <- readBin(bytes, "double", n = length(bytes) / 8)
  ggml_backend_tensor_set_async(be, x, packed, size = 4 * 4099)
  ggml_backend_graph_compute_async(be, gf)
  out <- ggml_backend_tensor_get_async(be, y, size = 4 * 4099)
  ggml_backend_synchronize(be)
  got <- readBin(writeBin(out, raw()), "double", n = 4099, size = 4)
  if (max(abs(got - 3 * v)) > 1e-4 * max(abs(v))) fail("graph readback rep %d", rep)
}
cat("OK\n")
'

run_async_mode <- function(env) {
  script <- tempfile(fileext = ".R")
  on.exit(unlink(script))
  writeLines(async_script, script)
  rscript <- file.path(R.home("bin"), "Rscript")
  out <- suppressWarnings(system2(rscript, script, stdout = TRUE, stderr = TRUE, env = env))
  list(status = attr(out, "status") %||% 0L, out = out)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

test_that("Vulkan async set/get round-trip with the default queue setup", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  skip_on_cran()
  r <- run_async_mode(character())
  expect_equal(r$status, 0L, info = paste(r$out, collapse = "\n"))
  expect_true(any(grepl("^OK", r$out)), info = paste(r$out, collapse = "\n"))
})

test_that("Vulkan async set/get round-trip without a transfer queue", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  skip_on_cran()
  r <- run_async_mode("GGML_VK_ALLOW_GRAPHICS_QUEUE=1")
  expect_equal(r$status, 0L, info = paste(r$out, collapse = "\n"))
  expect_true(any(grepl("^OK", r$out)), info = paste(r$out, collapse = "\n"))
})
