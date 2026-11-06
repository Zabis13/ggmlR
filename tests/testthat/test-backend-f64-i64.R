# F64 / I64 through ggml_backend_tensor_set_data / get_data, plus the raw-byte
# length of ggml_backend_tensor_get_async. CPU backend: no Vulkan kernels for
# these types are needed, only the buffer copy.
#
# R has no int64, so I64 travels as double: exact within +-2^53. set_data
# rejects what would not survive the trip; get_data warns when a stored value
# (written by an op or raw bytes) lies outside that range.

with_cpu_tensor <- function(type, n, f) {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend))
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE, after = FALSE)
  t <- ggml_new_tensor_1d(ctx, type, n)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  f(t, backend)
}

test_that("type constants match ggml.h", {
  expect_identical(GGML_TYPE_I64, 27L)
  expect_identical(GGML_TYPE_F64, 28L)
})

test_that("F64 round-trips bit for bit", {
  x <- c(pi, -1 / 3, 1e-300, -1e300, 0, -0, 2^60 + 1, NaN, Inf, -Inf)
  with_cpu_tensor(GGML_TYPE_F64, length(x), function(t, be) {
    ggml_backend_tensor_set_data(t, x)
    expect_identical(ggml_backend_tensor_get_data(t), x)
    # integer and logical input are widened, integer NA -> NaN
    ggml_backend_tensor_set_data(t, c(1:8, NA_integer_, -5L))
    expect_identical(ggml_backend_tensor_get_data(t),
                     c(as.numeric(1:8), NaN, -5))
  })
})

test_that("I64 round-trips whole numbers up to 2^53", {
  x <- c(0, 1, -1, 2^31, -2^31 - 1, 2^53, -2^53, 123456789012345)
  with_cpu_tensor(GGML_TYPE_I64, length(x), function(t, be) {
    ggml_backend_tensor_set_data(t, x)
    expect_identical(ggml_backend_tensor_get_data(t), x)
    ggml_backend_tensor_set_data(t, c(-3:3, 7L))
    expect_identical(ggml_backend_tensor_get_data(t), as.numeric(c(-3:3, 7L)))
    # the stored bytes really are int64, not doubles
    raw8 <- writeBin(ggml_backend_tensor_get_async(be, t, size = 8), raw())
    expect_identical(raw8, as.raw(c(0xfd, rep(0xff, 7))))   # -3 as int64
  })
})

test_that("I64 set_data rejects values doubles cannot carry exactly", {
  with_cpu_tensor(GGML_TYPE_I64, 2L, function(t, be) {
    expect_error(ggml_backend_tensor_set_data(t, c(1, 2.5)), "whole numbers")
    expect_error(ggml_backend_tensor_set_data(t, c(1, NA)), "NA")
    expect_error(ggml_backend_tensor_set_data(t, c(1, Inf)), "whole numbers")
    expect_error(ggml_backend_tensor_set_data(t, c(1, 2^53 + 2)), "2\\^53")
    expect_error(ggml_backend_tensor_set_data(t, c("1", "2")), "numeric")
  })
})

test_that("I64 get_data warns past 2^53", {
  with_cpu_tensor(GGML_TYPE_I64, 1L, function(t, be) {
    # 2^60 as little-endian int64, written as raw bytes
    bits <- readBin(as.raw(c(rep(0, 7), 0x10)), "double")
    ggml_backend_tensor_set_async(be, t, bits, size = 8)
    expect_warning(v <- ggml_backend_tensor_get_data(t), "2\\^53")
    expect_identical(v, 2^60)
  })
})

test_that("get_async returns ceiling(size / 8) doubles, tail zeroed", {
  with_cpu_tensor(GGML_TYPE_F32, 4L, function(t, be) {
    ggml_backend_tensor_set_data(t, c(1, 2, 3, 4))
    for (sz in c(0, 3, 4, 8, 12, 16)) {
      got <- ggml_backend_tensor_get_async(be, t, size = sz)
      ggml_backend_synchronize(be)
      expect_length(got, ceiling(sz / 8))
      bytes <- writeBin(got, raw())
      want <- writeBin(c(1, 2, 3, 4), raw(), size = 4)[seq_len(sz)]
      expect_identical(bytes[seq_len(sz)], want)
      expect_true(all(bytes[-seq_len(sz)] == as.raw(0)))
    }
  })
})
