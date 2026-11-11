# Two shape fixes for recorded graphs: the device index from
# ag_sample_categorical(key =) / ag_argmax(as_tensor = TRUE) is [1, batch] in
# ggml too (it concatenates with other rows), and ag_reshape() on the GPU is a
# graph node (no download, so ag_capture() can record it).

skip_if_no_gpu_sf <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

test_that("a device index concatenates with other rows in a recording", {
  skip_if_no_gpu_sf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  f <- ag_capture(function(x, key) {
    lp <- ag_log_softmax(x)
    a <- ag_sample_categorical(lp, key = key)
    g <- ag_argmax(x, as_tensor = TRUE)
    list(r = ag_concat_rows(ag_concat_rows(a, ag_gather(lp, a)), g), lp = lp)
  })
  on.exit(ag_capture_free(f), add = TRUE)
  set.seed(2)
  x <- matrix(rnorm(20), 4, 5)
  out <- f(x = x, key = matrix(c(0.25, 0.5), 2, 1))
  expect_identical(dim(out$r), c(3L, 5L))
  a <- out$r[1, ]
  expect_true(all(a %in% 0:3))
  expect_equal(out$r[2, ], out$lp[cbind(a + 1, 1:5)], tolerance = 1e-6)
  expect_identical(out$r[3, ], apply(x, 2, which.max) - 1)
})

test_that("ag_reshape is a graph node on the GPU", {
  skip_if_no_gpu_sf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  x <- matrix(1:6, 2, 3) + 0
  # inside a recording: an error before (the reshape downloaded its operand)
  f <- ag_capture(function(x) list(y = ag_reshape(ag_transpose(x), 1, 6),
                                   z = ag_reshape(ag_scale(x, 2), 3, -1)))
  on.exit(ag_capture_free(f), add = TRUE)
  out <- f(x = x)
  expect_identical(out$y, matrix(t(x), 1, 6))
  expect_identical(out$z, matrix(2 * x, 3, 2))
  # eager, same values as on the CPU
  g <- as.matrix(ag_reshape(ag_tensor(x), 3, 2))
  ag_device("cpu")
  expect_identical(g, as.matrix(ag_reshape(ag_tensor(x), 3, 2)))
  ag_device("gpu")
  expect_error(ag_reshape(ag_tensor(x), 4, 2), "does not hold")
})
