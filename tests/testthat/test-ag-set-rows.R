# ag_set_rows(): writes buffer columns by index -- through the host outside a
# recording, as a graph node inside ag_capture(). A recording that only writes
# returns NULL and downloads nothing; device state (a counter) carries from one
# replay to the next with no transfer at all.

skip_if_no_gpu_sr <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

on_devices_sr <- function(code) {
  for (dev in c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L) "gpu")) {
    ag_device(dev)
    code(dev)
  }
  ag_device("cpu")
}

xfers <- function() {
  cnt <- ggmlR:::.ag_xfer$counts
  sum(vapply(ls(cnt, all.names = TRUE), function(k) as.numeric(cnt[[k]]$n), 0))
}

test_that("ag_set_rows writes columns by index outside a recording", {
  on_devices_sr(function(dev) {
    buf <- ag_buffer(2, 5)
    on.exit(ag_buffer_free(buf), add = TRUE)
    ag_set_rows(buf, c(3, 0, 1), matrix(1:6, 2, 3))
    expect_equal(ag_buffer_read(buf), cbind(c(3, 4), c(5, 6), 0, c(1, 2), 0), info = dev)
    ag_set_rows(buf, ag_tensor(matrix(4, 1, 1)), ag_tensor(matrix(c(7, 8), 2, 1)))
    expect_equal(ag_buffer_read(buf, 4, 1), matrix(c(7, 8), 2, 1), info = dev)
    expect_error(ag_set_rows(buf, c(1, 1), matrix(0, 2, 2)), "repeated")
    expect_error(ag_set_rows(buf, 5, matrix(0, 2, 1)), "0..4")
    expect_error(ag_set_rows(buf, 0:1, matrix(0, 2, 3)), "expected 2 x 2")
  })
})

test_that("a captured function that only writes returns NULL", {
  on_devices_sr(function(dev) {
    buf <- ag_buffer(2, 4)
    on.exit(ag_buffer_free(buf), add = TRUE)
    f <- ag_capture(function(idx, x) {
      ag_set_rows(buf, idx, ag_scale(x, 2))
      NULL
    }, params = list(buf))
    on.exit(ag_capture_free(f), add = TRUE)
    expect_null(f(idx = c(2, 0), x = matrix(1:4, 2, 2)))
    expect_null(f(idx = c(1, 3), x = matrix(5:8, 2, 2)))
    expect_equal(ag_buffer_read(buf), 2 * cbind(c(3, 4), c(5, 6), c(1, 2), c(7, 8)), info = dev)
  })
})

test_that("device state carries across replays with no transfers", {
  skip_if_no_gpu_sr()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cnt <- ag_buffer(1, 1)
  out <- ag_buffer(2, 5)
  on.exit({ ag_buffer_free(cnt); ag_buffer_free(out) }, add = TRUE)
  ones <- ag_param(matrix(1, 2, 1))
  one <- ag_param(matrix(1, 1, 1))
  # column t of `out` gets (t, t); then t <- t + 1, on the device
  f <- ag_capture(function(z) {
    t <- ag_get_rows(cnt, z)
    ag_set_rows(out, t, ag_matmul(ones, t))
    ag_set_rows(cnt, z, ag_add(t, one))
    NULL
  }, params = list(cnt, out, ones, one))
  on.exit(ag_capture_free(f), add = TRUE)
  z <- matrix(0, 1, 1)
  f(z = z)
  ggmlR:::ag_xfer_count(TRUE)
  on.exit(ggmlR:::ag_xfer_count(FALSE), add = TRUE)
  for (i in 1:3) f(z = z)
  expect_equal(xfers(), 0)                     # same input, nothing downloaded
  ggmlR:::ag_xfer_count(FALSE)
  expect_equal(ag_buffer_read(cnt), matrix(4, 1, 1))
  expect_equal(ag_buffer_read(out), cbind(c(0, 0), c(1, 1), c(2, 2), c(3, 3), c(0, 0)))
})

test_that("ag_capture_repeat runs n times with one wait, state carried", {
  on_devices_sr(function(dev) {
    cnt <- ag_buffer(1, 1)
    out <- ag_buffer(2, 40)
    st <- ag_buffer(64, 1)
    on.exit({ ag_buffer_free(cnt); ag_buffer_free(out); ag_buffer_free(st) }, add = TRUE)
    ones <- ag_param(matrix(1, 2, 1))
    one <- ag_param(matrix(1, 1, 1))
    W <- ag_param(diag(64))
    ones64 <- ag_param(matrix(1, 64, 1))
    # a counter, a column per run, and a 64-row state through a matmul (more
    # nodes between the read and the write of the same buffer)
    f <- ag_capture(function(z) {
      t <- ag_get_rows(cnt, z)
      ag_set_rows(out, t, ag_matmul(ones, t))
      s <- ag_get_rows(st, z)
      ag_set_rows(st, z, ag_add(ag_matmul(W, s), ag_matmul(ones64, one)))
      ag_set_rows(cnt, z, ag_add(t, one))
      NULL
    }, params = list(cnt, out, st, ones, one, W, ones64))
    on.exit(ag_capture_free(f), add = TRUE)
    z <- matrix(0, 1, 1)
    f(z = z)                                     # records (GPU), run 1
    expect_null(ag_capture_repeat(f, 39, z = z))
    expect_equal(ag_buffer_read(cnt), matrix(40, 1, 1), info = dev)
    expect_equal(ag_buffer_read(out), rbind(0:39, 0:39) + 0, info = dev)
    expect_equal(ag_buffer_read(st), matrix(40, 64, 1), info = dev)
  })
})

test_that("recorded command buffers (Vulkan) give the async chain's result", {
  skip_if_no_gpu_sr()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  run <- function(record) {
    old <- Sys.getenv("GGMLR_VK_GRAPH_RECORD")
    Sys.setenv(GGMLR_VK_GRAPH_RECORD = if (record) "1" else "0")
    on.exit(Sys.setenv(GGMLR_VK_GRAPH_RECORD = old), add = TRUE)
    cnt <- ag_buffer(1, 1); st <- ag_buffer(64, 4)
    on.exit({ ag_buffer_free(cnt); ag_buffer_free(st) }, add = TRUE)
    set.seed(9)
    ag_buffer_write(st, matrix(stats::runif(256), 64, 4), col_offset = 0)
    W <- ag_param(matrix(stats::rnorm(64 * 64, sd = 0.1), 64, 64))
    one <- ag_param(matrix(1, 1, 1))
    cols <- matrix(0:3, 4, 1)
    # a state through matmul and tanh, and a counter: each run reads what the
    # previous one wrote
    f <- ag_capture(function(z, cols) {
      t <- ag_get_rows(cnt, z)
      s <- ag_get_rows(st, cols)
      ag_set_rows(st, cols, ag_tanh(ag_add(ag_matmul(W, s), ag_matmul(ag_param_ones, t))))
      ag_set_rows(cnt, z, ag_add(t, one))
      NULL
    }, params = list(cnt, st, W, one, ag_param_ones))
    on.exit(ag_capture_free(f), add = TRUE)
    z <- matrix(0, 1, 1)
    ag_capture_repeat(f, 7, z = z, cols = cols)
    # a buffer the recording does not use, destroyed in between: it stays valid
    tmp <- ag_buffer(8, 8)
    ag_buffer_write(tmp, matrix(0, 8, 8), col_offset = 0)   # allocated on the device
    ag_buffer_free(tmp)
    ag_capture_repeat(f, 5, z = z, cols = cols)
    recs <- attr(f, "captures")
    list(st = ag_buffer_read(st), cnt = ag_buffer_read(cnt),
         recorded = inherits(recs[[ls(recs)[1]]]$vk_rec, "externalptr"))
  }
  ag_param_ones <- ag_param(matrix(0.01, 64, 1))
  a <- run(TRUE)
  b <- run(FALSE)
  expect_true(a$recorded)
  expect_false(b$recorded)
  expect_equal(a$cnt, matrix(12, 1, 1))
  expect_identical(a$cnt, b$cnt)
  expect_identical(a$st, b$st)
})

test_that("ag_capture_repeat refuses a recording with outputs", {
  skip_if_no_gpu_sr()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  buf <- ag_buffer(2, 3)
  on.exit(ag_buffer_free(buf), add = TRUE)
  f <- ag_capture(function(idx, x) { ag_set_rows(buf, idx, x); list(y = x) },
                  params = list(buf))
  on.exit(ag_capture_free(f), add = TRUE)
  expect_error(ag_capture_repeat(f, 2, idx = 0, x = matrix(1, 2, 1)), "returns NULL")
  expect_error(ag_capture_repeat(function() NULL, 2), "returned by ag_capture")
})

test_that("a recording can write and return outputs", {
  skip_if_no_gpu_sr()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  buf <- ag_buffer(2, 3)
  on.exit(ag_buffer_free(buf), add = TRUE)
  f <- ag_capture(function(idx, x) {
    ag_set_rows(buf, idx, x)
    list(y = ag_scale(x, 3))
  }, params = list(buf))
  on.exit(ag_capture_free(f), add = TRUE)
  r <- f(idx = 1, x = matrix(c(1, 2), 2, 1))
  expect_equal(r$y, matrix(c(3, 6), 2, 1))
  expect_equal(ag_buffer_read(buf, 1, 1), matrix(c(1, 2), 2, 1))
})

test_that("recording errors name the problem", {
  skip_if_no_gpu_sr()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  buf <- ag_buffer(2, 3)
  on.exit(ag_buffer_free(buf), add = TRUE)
  f <- ag_capture(function(x) { ag_set_rows(buf, 0, x); NULL }, params = list(buf))
  expect_error(f(x = matrix(1, 2, 1)), "idx must be an ag_tensor")
  g <- ag_capture(function(x) NULL, params = list(buf))
  expect_error(g(x = matrix(1, 2, 1)), "wrote nothing")
})
