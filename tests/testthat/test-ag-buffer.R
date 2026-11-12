# ag_buffer(): a table kept on the device, written by column ranges and read
# by column index with ag_get_rows(). Tested on the CPU and the GPU: writes and
# reads by range, bounds checks, gather by index, use inside ag_capture (the
# recording sees later writes, freeing re-records), the rescue to the host on
# a device switch, and explicit free.

skip_if_no_gpu_buf <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

on_devices <- function(code) {
  for (dev in c("cpu", if (ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L) "gpu")) {
    ag_device(dev)
    code(dev)
  }
  ag_device("cpu")
}

test_that("ag_buffer starts at zero and reads back what was written", {
  on_devices(function(dev) {
    buf <- ag_buffer(3, 10)
    expect_equal(dim(buf), c(3L, 10L))
    expect_equal(ag_buffer_read(buf), matrix(0, 3, 10), info = dev)
    x <- matrix(1:6, 3, 2)
    ag_buffer_write(buf, x, col_offset = 4)
    ag_buffer_write(buf, c(7, 8, 9), col_offset = 9)
    want <- matrix(0, 3, 10); want[, 5:6] <- x; want[, 10] <- 7:9
    expect_equal(ag_buffer_read(buf), want, info = dev)
    expect_equal(ag_buffer_read(buf, 4, 2), x, info = dev)
    ag_buffer_free(buf)
  })
})

test_that("ag_buffer_write and ag_buffer_read check their bounds", {
  buf <- ag_buffer(2, 5)
  expect_error(ag_buffer_write(buf, matrix(1, 2, 2), col_offset = 4), "outside the buffer")
  expect_error(ag_buffer_write(buf, matrix(1, 3, 1), col_offset = 0), "has 3 rows")
  expect_error(ag_buffer_write(buf, matrix(1, 2, 1), col_offset = -1), "col_offset")
  expect_error(ag_buffer_read(buf, 3, 3), "outside the buffer")
  expect_error(ag_buffer(0, 5), "rows")
  expect_error(ag_buffer(2, 5, dtype = "f16"), "f32")
  ag_buffer_free(buf)
  expect_error(ag_buffer_write(buf, matrix(1, 2, 1), 0), "freed")
  expect_error(ag_get_rows(buf, 0), "freed")
})

test_that("ag_get_rows gathers columns by 0-based index, repeats allowed", {
  on_devices(function(dev) {
    set.seed(1)
    x <- matrix(rnorm(4 * 7), 4, 7)
    buf <- ag_buffer(4, 7)
    ag_buffer_write(buf, x, 0)
    idx <- c(6, 0, 3, 3)
    g <- ag_get_rows(buf, idx)
    expect_false(g$requires_grad)
    expect_equal(as.matrix(g), x[, idx + 1], tolerance = 1e-6, info = dev)
    # as an operand of further ops
    W <- ag_tensor(matrix(1, 2, 4))
    expect_equal(as.matrix(ag_matmul(W, g)), matrix(1, 2, 4) %*% x[, idx + 1],
                 tolerance = 1e-5, info = dev)
    expect_error(ag_get_rows(buf, 7), "0..6")
    expect_error(ag_get_rows(buf, 1.5), "whole numbers")
    ag_buffer_free(buf)
  })
})

test_that("a ring wraps with two writes across the end", {
  # capacity 7 and batches of 3: the third batch wraps (columns 6, 0, 1)
  on_devices(function(dev) {
    buf <- ag_buffer(1, 7)
    pos <- 0L
    for (b in 1:3) {
      x <- matrix(b * 10 + 1:3, 1)
      first <- min(3L, 7L - pos)
      ag_buffer_write(buf, x[, seq_len(first), drop = FALSE], pos)
      if (first < 3L) ag_buffer_write(buf, x[, -seq_len(first), drop = FALSE], 0)
      pos <- (pos + 3L) %% 7L
    }
    expect_equal(as.numeric(ag_buffer_read(buf)), c(32, 33, 13, 21, 22, 23, 31),
                 info = dev)
    ag_buffer_free(buf)
  })
})

test_that("a captured graph reads the buffer's current value and new indices", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  set.seed(2)
  x <- matrix(rnorm(3 * 8), 3, 8)
  buf <- ag_buffer(3, 8)
  ag_buffer_write(buf, x, 0)
  W <- ag_param(matrix(rnorm(2 * 3), 2, 3))
  f <- ag_capture(function(idx) ag_matmul(W, ag_get_rows(buf, idx)), params = list(W, buf))
  i1 <- matrix(c(0, 5, 7), ncol = 1)
  expect_equal(f(idx = i1), as.matrix(W) %*% x[, i1 + 1], tolerance = 1e-5)
  # new indices, same recording
  i2 <- matrix(c(2, 2, 1), ncol = 1)
  expect_equal(f(idx = i2), as.matrix(W) %*% x[, i2 + 1], tolerance = 1e-5)
  expect_length(ls(attr(f, "captures")), 1L)
  # a write after recording is seen by the replay
  x[, 3] <- c(10, 20, 30)
  ag_buffer_write(buf, x[, 3], 2)
  expect_equal(f(idx = i2), as.matrix(W) %*% x[, i2 + 1], tolerance = 1e-5)
  ag_buffer_free(buf)
  expect_error(f(idx = i2), "freed")
})

test_that("captured ag_get_rows on the GPU equals the CPU on the same indices", {
  skip_if_no_gpu_buf()
  on.exit(ag_device("cpu"), add = TRUE)
  set.seed(3)
  x <- matrix(rnorm(5 * 50), 5, 50)
  w <- matrix(rnorm(4 * 5), 4, 5)
  idx <- matrix(c(49, 0, 17, 17, 3, 25), ncol = 1)
  run <- function(dev) {
    ag_device(dev)
    buf <- ag_buffer(5, 50)
    ag_buffer_write(buf, x, 0)
    W <- ag_param(w)
    f <- ag_capture(function(idx) ag_matmul(W, ag_get_rows(buf, idx)), params = list(W, buf))
    out <- f(idx = idx)
    ag_buffer_free(buf)
    out
  }
  cpu <- run("cpu")
  expect_equal(cpu, w %*% x[, idx + 1])
  expect_equal(run("gpu"), cpu, tolerance = 1e-5)
})

test_that("indices out of range are clamped on the device", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  x <- matrix(1:12, 2, 6)
  buf <- ag_buffer(2, 6)
  ag_buffer_write(buf, x, 0)
  # inside a recording the indices are a device input, not checked on the host
  f <- ag_capture(function(idx) ag_get_rows(buf, idx), params = list(buf))
  expect_equal(f(idx = matrix(c(-3, 2, 6, 1e6), ncol = 1)), x[, c(1, 3, 6, 6)])
  ag_buffer_free(buf)
})

test_that("a recording is made anew after the buffer returns to the device", {
  skip_if_no_gpu_buf()
  on.exit(ag_device("cpu"), add = TRUE)
  ag_device("gpu")
  x <- matrix(1:8, 2, 4)
  buf <- ag_buffer(2, 4)
  ag_buffer_write(buf, x, 0)
  f <- ag_capture(function(idx) ag_get_rows(buf, idx), params = list(buf))
  i <- matrix(c(3, 0), ncol = 1)
  expect_equal(f(idx = i), x[, c(4, 1)])
  gen <- buf$core$gen
  ag_device("cpu")                      # value moves to the host
  x[, 4] <- c(-1, -2)
  ag_buffer_write(buf, x[, 4], 3)       # written on the host
  ag_device("gpu")
  expect_equal(f(idx = i), x[, c(4, 1)])  # uploaded again, new recording
  expect_gt(buf$core$gen, gen)
  ag_buffer_free(buf)
})

test_that("a queued read keeps its buffer alive until it is computed", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  run <- function() {
    ag_local_mode(graph = TRUE)
    x <- matrix(1:6, 2, 3)
    # the result holds the buffer: dropping the buffer object frees nothing
    buf <- ag_buffer(2, 3)
    ag_buffer_write(buf, x, 0)
    g <- ag_get_rows(buf, c(2, 0))
    rm(buf); gc()
    expect_equal(as.matrix(g), x[, c(3, 1)])
    # the fallback: collected while its node is queued, freed after the drain
    buf2 <- ag_buffer(2, 3)
    ag_buffer_write(buf2, x, 0)
    g2 <- ag_get_rows(buf2, c(1, 1))
    core <- buf2$core
    expect_gt(ggmlR:::.ag_defer_len(), 0L)
    ggmlR:::.ag_buffer_collect(core)
    expect_false(is.null(core$ptr))
    expect_equal(as.matrix(g2), x[, c(2, 2)])
    expect_null(core$ptr)
    expect_length(ls(ggmlR:::.ag_buffer_reg$orphans), 0L)
  }
  run()
})

test_that("a device switch keeps the buffer's value", {
  skip_if_no_gpu_buf()
  ag_device("gpu")
  x <- matrix(1:12, 3, 4)
  buf <- ag_buffer(3, 4)
  ag_buffer_write(buf, x, 0)
  ag_device("cpu")
  expect_equal(ag_buffer_read(buf), x)
  expect_equal(as.matrix(ag_get_rows(buf, c(3, 0))), x[, c(4, 1)])
  ag_device("gpu")
  expect_equal(as.matrix(ag_get_rows(buf, c(1, 2))), x[, 2:3])
  ag_device("cpu")
  ag_buffer_free(buf)
})

test_that("ag_buffer refuses more memory than the device has free", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  free <- ggml_backend_dev_memory(ggml_backend_get_device(ggmlR:::.ag_device_state$backend))[["free"]]
  cap <- ceiling(free / 4 / 1024) + 1
  expect_error(ag_buffer(1024, cap), "free")
})

# ag_buffer_copy_from(): a device tensor of another backend (an environment's)
# copied into buffer columns on the device.
env_tensor <- function(backend, vals, rows, k, type = GGML_TYPE_F32) {
  ctx <- ggml_init(4 * ggml_tensor_overhead() + 1024, no_alloc = TRUE)
  t <- ggml_new_tensor_2d(ctx, type, rows, k)
  bb <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  if (type == GGML_TYPE_F32) ggml_backend_tensor_set_data(t, as.numeric(vals))
  list(t = t, ctx = ctx, bb = bb)
}
env_free <- function(e) { ggml_backend_buffer_free(e$bb); ggml_free(e$ctx) }

test_that("ag_buffer_copy_from writes columns from another backend's tensor", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  be <- ggml_vulkan_init(0L); on.exit(ggml_backend_free(be), add = TRUE)
  buf <- ag_buffer(3, 6)
  m <- matrix(1:6 + 0.5, 3, 2)
  e <- env_tensor(be, m, 3, 2); on.exit(env_free(e), add = TRUE)
  ag_buffer_copy_from(buf, e$t, be, col_offset = 3)
  expect_equal(ag_buffer_read(buf), cbind(matrix(0, 3, 3), m, 0))
  # the whole buffer at once
  e2 <- env_tensor(be, 1:18, 3, 6); on.exit(env_free(e2), add = TRUE)
  ag_buffer_copy_from(buf, e2$t, be)
  expect_equal(ag_buffer_read(buf), matrix(1:18 + 0, 3, 6))
  ag_buffer_free(buf)
})

test_that("ag_buffer_copy_from takes an N-d tensor, columns along the trailing axes", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  be <- ggml_vulkan_init(0L); on.exit(ggml_backend_free(be), add = TRUE)
  nd <- function(vals, ne) {
    ctx <- ggml_init(4 * ggml_tensor_overhead() + 1024, no_alloc = TRUE)
    t <- if (length(ne) == 3L) ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne[1], ne[2], ne[3]) else
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[1], ne[2], ne[3], ne[4])
    bb <- ggml_backend_alloc_ctx_tensors(ctx, be)
    ggml_backend_tensor_set_data(t, as.numeric(vals))
    list(t = t, ctx = ctx, bb = bb)
  }
  buf <- ag_buffer(6, 5)
  # [2, 3, 4]: rows = 2 * 3, four columns
  e <- nd(1:24, c(2, 3, 4)); on.exit(env_free(e), add = TRUE)
  ag_buffer_copy_from(buf, e$t, be, col_offset = 1)
  expect_equal(ag_buffer_read(buf), cbind(0, matrix(1:24 + 0, 6, 4)))
  # [2, 3, 1]: one column, the trailing axis of size 1
  e1 <- nd(101:106, c(2, 3, 1)); on.exit(env_free(e1), add = TRUE)
  ag_buffer_copy_from(buf, e1$t, be, col_offset = 0)
  expect_equal(ag_buffer_read(buf, 0, 1), matrix(101:106 + 0, 6, 1))
  # [6, 2, 2]: rows = ne0, 2 * 2 columns
  e2 <- nd(1:24, c(6, 2, 2)); on.exit(env_free(e2), add = TRUE)
  ag_buffer_copy_from(buf, e2$t, be, col_offset = 0)
  expect_equal(ag_buffer_read(buf, 0, 4), matrix(1:24 + 0, 6, 4))
  # [2, 3, 2, 2]: 4-d, rows = 2 * 3
  e3 <- nd(24:1, c(2, 3, 2, 2)); on.exit(env_free(e3), add = TRUE)
  ag_buffer_copy_from(buf, e3$t, be, col_offset = 1)
  expect_equal(ag_buffer_read(buf, 1, 4), matrix(24:1 + 0, 6, 4))
  # refused: no product of leading axes is 6; too many columns
  bad <- nd(1:16, c(4, 2, 2)); on.exit(env_free(bad), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, bad$t, be), "rows")
  expect_error(ag_buffer_copy_from(buf, e$t, be, col_offset = 2), "outside the buffer")
  # the host path follows the same rule
  ag_device("cpu")
  hb <- ag_buffer(6, 5)
  suppressMessages(ag_buffer_copy_from(hb, e$t, be, col_offset = 1, via_host = "allow"))
  expect_equal(ag_buffer_read(hb), cbind(0, matrix(1:24 + 0, 6, 4)))
  expect_error(suppressMessages(ag_buffer_copy_from(hb, bad$t, be, via_host = "allow")), "rows")
  ag_buffer_free(hb)
  ag_device("gpu")
  ag_buffer_free(buf)
})

test_that("ag_buffer_copy_from from the ag backend itself is synchronous", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  buf <- ag_buffer(2, 4)
  be <- ggmlR:::.ag_device_state$backend
  e <- env_tensor(be, c(7, 8), 2, 1); on.exit(env_free(e), add = TRUE)
  ag_buffer_copy_from(buf, e$t, be, col_offset = 1)
  expect_equal(as.matrix(ag_get_rows(buf, 1)), matrix(c(7, 8), 2, 1))
  ag_buffer_free(buf)
})

test_that("a captured graph sees ag_buffer_copy_from writes without re-recording", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  be <- ggml_vulkan_init(0L); on.exit(ggml_backend_free(be), add = TRUE)
  buf <- ag_buffer(2, 3)
  W <- ag_param(diag(2))
  f <- ag_capture(function(idx) ag_matmul(W, ag_get_rows(buf, idx)), params = list(W, buf))
  e1 <- env_tensor(be, c(1, 2), 2, 1); on.exit(env_free(e1), add = TRUE)
  e2 <- env_tensor(be, c(3, 4), 2, 1); on.exit(env_free(e2), add = TRUE)
  i <- matrix(0, ncol = 1)
  ag_buffer_copy_from(buf, e1$t, be, 0)
  expect_equal(f(idx = i), matrix(c(1, 2), 2, 1), tolerance = 1e-6)
  gen <- buf$core$gen
  ag_buffer_copy_from(buf, e2$t, be, 0)
  expect_identical(buf$core$gen, gen)
  expect_equal(f(idx = i), matrix(c(3, 4), 2, 1), tolerance = 1e-6)
  expect_length(ls(attr(f, "captures")), 1L)
  ag_buffer_free(buf)
})

test_that("ag_buffer_copy_from refuses what breaks its contract", {
  skip_if_no_gpu_buf()
  on.exit(ag_device("cpu"), add = TRUE)
  ag_device("gpu")
  be <- ggml_vulkan_init(0L); on.exit(ggml_backend_free(be), add = TRUE)
  buf <- ag_buffer(3, 4)
  ok <- env_tensor(be, 1:6, 3, 2); on.exit(env_free(ok), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, ok$t, be, 3), "outside the buffer")
  expect_error(ag_buffer_copy_from(buf, ok$t, be, -1), "outside the buffer")
  expect_error(ag_buffer_copy_from(buf, ok$t, be, 0.5), "outside the buffer")
  tr <- env_tensor(be, 1:6, 2, 3); on.exit(env_free(tr), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, tr$t, be), "rows")
  i32 <- env_tensor(be, NULL, 3, 2, GGML_TYPE_I32); on.exit(env_free(i32), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, i32$t, be), "f32")
  # nothing was written by the refused calls
  expect_equal(ag_buffer_read(buf), matrix(0, 3, 4))
  ag_buffer_free(buf)
})

# via_host: every copy that would go through the host is an error naming the
# reason by default, and the host copy (same values) with via_host = "allow".
reset_via_host_note <- function() {
  reg <- ggmlR:::.ag_buffer_reg
  reg$via_host_noted <- NULL
}

test_that("via_host: a tensor in host memory, ag on the GPU", {
  skip_if_no_gpu_buf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  cpu <- ggml_backend_cpu_init(); on.exit(ggml_backend_free(cpu), add = TRUE)
  buf <- ag_buffer(3, 4)
  h <- env_tensor(cpu, 1:6, 3, 2); on.exit(env_free(h), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, h$t, cpu), "host memory.*via_host = \"allow\"")
  expect_equal(ag_buffer_read(buf), matrix(0, 3, 4))
  reset_via_host_note()
  expect_message(ag_buffer_copy_from(buf, h$t, cpu, 1, via_host = "allow"), "host memory")
  expect_silent(ag_buffer_copy_from(buf, h$t, cpu, 1, via_host = "allow"))
  expect_equal(ag_buffer_read(buf), cbind(0, matrix(1:6 + 0, 3, 2), 0))
  ag_buffer_free(buf)
})

test_that("via_host: ag on the CPU, the tensor on the GPU", {
  skip_if_no_gpu_buf()
  ag_device("cpu")
  be <- ggml_vulkan_init(0L); on.exit(ggml_backend_free(be), add = TRUE)
  buf <- ag_buffer(3, 4)
  g <- env_tensor(be, 1:6, 3, 2); on.exit(env_free(g), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, g$t, be), "ag device is the CPU.*via_host = \"allow\"")
  expect_equal(ag_buffer_read(buf), matrix(0, 3, 4))
  reset_via_host_note()
  expect_message(ag_buffer_copy_from(buf, g$t, be, 2, via_host = "allow"), "ag device is the CPU")
  expect_equal(ag_buffer_read(buf), cbind(0, 0, matrix(1:6 + 0, 3, 2)))
  ag_buffer_free(buf)
})

test_that("via_host: the tensor on another GPU", {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 2L,
              "fewer than two Vulkan devices")
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  be1 <- ggml_vulkan_init(1L); on.exit(ggml_backend_free(be1), add = TRUE)
  buf <- ag_buffer(3, 4)
  g <- env_tensor(be1, 1:6, 3, 2); on.exit(env_free(g), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, g$t, be1), "on device.*via_host = \"allow\"")
  expect_equal(ag_buffer_read(buf), matrix(0, 3, 4))
  reset_via_host_note()
  expect_message(ag_buffer_copy_from(buf, g$t, be1, 0, via_host = "allow"), "on device")
  expect_equal(ag_buffer_read(buf), cbind(matrix(1:6 + 0, 3, 2), 0, 0))
  ag_buffer_free(buf)
})

test_that("via_host on the CPU only: default error, allow copies, same contract", {
  ag_device("cpu")
  cpu <- ggml_backend_cpu_init(); on.exit(ggml_backend_free(cpu), add = TRUE)
  buf <- ag_buffer(2, 4)
  e <- env_tensor(cpu, 1:4, 2, 2); on.exit(env_free(e), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, e$t, cpu, 1), "ag device is the CPU.*via_host")
  expect_error(ag_buffer_copy_from(buf, e$t, cpu, via_host = "maybe"), "should be one of")
  reset_via_host_note()
  expect_message(ag_buffer_copy_from(buf, e$t, cpu, 1, via_host = "allow"), "once per session")
  expect_silent(ag_buffer_copy_from(buf, e$t, cpu, 1, via_host = "allow"))
  expect_equal(ag_buffer_read(buf), cbind(0, matrix(1:4 + 0, 2, 2), 0))
  # the same refusals as the device path, and nothing written by them
  expect_error(ag_buffer_copy_from(buf, e$t, cpu, 3, via_host = "allow"), "outside the buffer")
  tr <- env_tensor(cpu, 1:4, 4, 1); on.exit(env_free(tr), add = TRUE)
  expect_error(ag_buffer_copy_from(buf, tr$t, cpu, via_host = "allow"), "rows")
  expect_equal(ag_buffer_read(buf), cbind(0, matrix(1:4 + 0, 2, 2), 0))
  ag_buffer_free(buf)
})
