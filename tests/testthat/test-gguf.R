# Tests for GGUF file reader

# Helper: create a minimal GGUF v3 file for testing
# GGUF format: magic(4) + version(4) + n_tensors(8) + n_kv(8) + kv_pairs + tensor_info + [alignment pad] + tensor_data
create_test_gguf <- function(path) {
  con <- file(path, "wb")
  on.exit(close(con))

  write_u32 <- function(v) writeBin(as.integer(v), con, size = 4, endian = "little")
  write_u64 <- function(v) { write_u32(v); write_u32(0L) }
  write_str <- function(s) writeChar(s, con, nchars = nchar(s), eos = NULL)

  # Header
  write_str("GGUF")       # magic
  write_u32(3L)            # version
  write_u64(1L)            # n_tensors = 1
  write_u64(1L)            # n_kv = 1

  # KV pair: key="test.key", type=STRING(8), value="hello"
  write_u64(8L)            # key length
  write_str("test.key")
  write_u32(8L)            # GGUF_TYPE_STRING
  write_u64(5L)            # string value length
  write_str("hello")

  # Tensor info: name="w", n_dims=1, shape=[4], type=F32(0), offset=0
  write_u64(1L)            # name length
  write_str("w")
  write_u32(1L)            # n_dims
  write_u64(4L)            # shape[0]
  write_u32(0L)            # type = F32
  write_u64(0L)            # offset within data section

  # Pad to alignment boundary (32 bytes)
  alignment <- 32L
  pos <- seek(con)
  remainder <- pos %% alignment
  if (remainder != 0L) {
    pad <- alignment - remainder
    writeBin(raw(pad), con)
  }

  # Tensor data: 4 floats = 16 bytes
  writeBin(c(1.0, 2.0, 3.0, 4.0), con, size = 4, endian = "little")
  # Pad tensor data to alignment (GGML_PAD(16, 32) = 32)
  writeBin(raw(16), con)  # 16 bytes padding to reach 32
}

test_that("gguf_load opens a valid GGUF file", {
  tmp <- tempfile(fileext = ".gguf")
  on.exit(unlink(tmp))
  create_test_gguf(tmp)

  g <- gguf_load(tmp)
  expect_s3_class(g, "gguf")
  expect_equal(g$version, 3L)
  expect_equal(g$n_tensors, 1L)
  expect_true(g$n_kv >= 1L)
  gguf_free(g)
})

test_that("gguf_load errors on non-existent file", {
  expect_error(gguf_load("/nonexistent/file.gguf"))
})

test_that("print.gguf works", {
  tmp <- tempfile(fileext = ".gguf")
  on.exit(unlink(tmp))
  create_test_gguf(tmp)

  g <- gguf_load(tmp)
  out <- capture.output(print(g))
  expect_true(any(grepl("GGUF file", out)))
  gguf_free(g)
})

test_that("gguf_tensor_names returns character vector", {
  tmp <- tempfile(fileext = ".gguf")
  on.exit(unlink(tmp))
  create_test_gguf(tmp)

  g <- gguf_load(tmp)
  nms <- gguf_tensor_names(g)
  expect_type(nms, "character")
  expect_true("w" %in% nms)
  gguf_free(g)
})

test_that("gguf_tensor_info returns shape and type", {
  tmp <- tempfile(fileext = ".gguf")
  on.exit(unlink(tmp))
  create_test_gguf(tmp)

  g <- gguf_load(tmp)
  info <- gguf_tensor_info(g, "w")
  expect_true(is.list(info))
  expect_equal(info$name, "w")
  expect_true("shape" %in% names(info))
  gguf_free(g)
})

test_that("gguf_tensor_data returns numeric data", {
  tmp <- tempfile(fileext = ".gguf")
  on.exit(unlink(tmp))
  create_test_gguf(tmp)

  g <- gguf_load(tmp)
  d <- gguf_tensor_data(g, "w")
  expect_type(d, "double")
  expect_equal(as.numeric(d), c(1, 2, 3, 4), tolerance = 1e-5)
  gguf_free(g)
})

test_that("gguf_metadata returns named list", {
  tmp <- tempfile(fileext = ".gguf")
  on.exit(unlink(tmp))
  create_test_gguf(tmp)

  g <- gguf_load(tmp)
  meta <- gguf_metadata(g)
  expect_type(meta, "list")
  expect_true("test.key" %in% names(meta))
  expect_equal(meta[["test.key"]], "hello")
  gguf_free(g)
})

test_that("gguf_free is safe to call twice", {
  tmp <- tempfile(fileext = ".gguf")
  on.exit(unlink(tmp))
  create_test_gguf(tmp)

  g <- gguf_load(tmp)
  expect_no_error(gguf_free(g))
  expect_no_error(gguf_free(g))
})

test_that("gguf functions error on non-gguf objects", {
  expect_error(gguf_metadata(list()), "Expected a gguf object")
  expect_error(gguf_tensor_names(42), "Expected a gguf object")
  expect_error(gguf_tensor_info("x", "w"), "Expected a gguf object")
  expect_error(gguf_tensor_data(NULL, "w"), "Expected a gguf object")
  expect_error(gguf_free(list()), "Expected a gguf object")
})
