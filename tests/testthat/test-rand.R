# GGML_OP_RAND (ggml_rand): counter-based random numbers, Philox4x32-10.
# Pinned here: the generator itself (known answer), the CPU kernel and the
# Vulkan shader drawing the same uniform bits, and the three distributions.

skip_if_no_gpu_rand <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

# ggml_rand on a given backend. `key` is two numbers stored as f32; their bit
# patterns are the Philox key.
raw_rand <- function(be, key, dist, dims, stream = 0L) {
  cs <- ggml_init(16e6, no_alloc = TRUE)
  on.exit(ggml_free(cs), add = TRUE, after = FALSE)
  k <- ggml_new_tensor_1d(cs, GGML_TYPE_F32, 2)
  buf <- ggml_backend_alloc_ctx_tensors(cs, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(k, as.numeric(key))
  g <- ggml_init(16e6, no_alloc = TRUE)
  on.exit(ggml_free(g), add = TRUE, after = FALSE)
  r <- ggml_rand(g, k, dist, dims, stream)
  ggml_set_output(r)
  gr <- ggml_build_forward_expand(g, r)
  s <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit(ggml_backend_sched_free(s), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(s, gr)
  ggml_backend_sched_graph_compute(s, gr)
  as.numeric(ggml_backend_tensor_get_data(r))
}

# 23 random bits -> (k + 0.5) / 2^23, as in the kernels.
u01_ref <- function(word) (floor(word / 512) + 0.5) / 2^23

test_that("ggml_rand reproduces the Philox4x32-10 known answer (CPU)", {
  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be), add = TRUE)
  # Random123 kat_vectors, philox4x32 10 rounds, counter 0, key 0:
  #   6627e8d5 e169c58d bc57ac4c 9b00dbd8
  # Element 0, stream 0, key bits 0 (0.0f, 0.0f) is exactly that block.
  u <- raw_rand(be, c(0, 0), "uniform", 4L)
  expect_equal(u[1], u01_ref(0x6627e8d5), tolerance = 0)
  z <- raw_rand(be, c(0, 0), "normal", 4L)
  ref <- sqrt(-2 * log(u01_ref(0x6627e8d5))) * cos(2 * pi * u01_ref(0xe169c58d))
  expect_equal(z[1], ref, tolerance = 1e-5)
})

test_that("ggml_rand: key, stream and index decide the value, shape does not (CPU)", {
  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be), add = TRUE)
  a <- raw_rand(be, c(0.25, 0.75), "uniform", 1000L)
  expect_identical(raw_rand(be, c(0.25, 0.75), "uniform", 1000L), a)
  expect_identical(raw_rand(be, c(0.25, 0.75), "uniform", c(4L, 25L)), a[1:100])
  expect_lt(mean(raw_rand(be, c(0.25, 0.5), "uniform", 1000L) == a), 0.01)
  expect_lt(mean(raw_rand(be, c(0.5, 0.75), "uniform", 1000L) == a), 0.01)
  expect_lt(mean(raw_rand(be, c(0.25, 0.75), "uniform", 1000L, stream = 1L) == a), 0.01)
})

test_that("ggml_rand distributions have the right moments (CPU)", {
  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be), add = TRUE)
  n <- 200000L
  u <- raw_rand(be, c(0.1, 0.9), "uniform", n)
  expect_true(all(u > 0 & u < 1))
  expect_equal(mean(u), 0.5, tolerance = 0.01)
  expect_equal(var(u), 1 / 12, tolerance = 0.02)
  expect_lt(abs(cor(u[-1], u[-n])), 0.01)
  g <- raw_rand(be, c(0.1, 0.9), "gumbel", n)
  expect_true(all(is.finite(g)))
  expect_equal(mean(g), 0.5772157, tolerance = 0.02)
  expect_equal(var(g), pi^2 / 6, tolerance = 0.03)
  z <- raw_rand(be, c(0.1, 0.9), "normal", n)
  expect_true(all(is.finite(z)))
  expect_lt(abs(mean(z)), 0.01)
  expect_equal(var(z), 1, tolerance = 0.02)
  expect_equal(mean(abs(z) < 1.96), 0.95, tolerance = 0.005)
})

test_that("ggml_rand checks its arguments", {
  cs <- ggml_init(1e6)
  on.exit(ggml_free(cs), add = TRUE)
  k <- ggml_new_tensor_1d(cs, GGML_TYPE_F32, 2)
  k1 <- ggml_new_tensor_1d(cs, GGML_TYPE_F32, 1)
  expect_error(ggml_rand(cs, k1, "uniform", 4L), "at least 2")
  expect_error(ggml_rand(cs, k, "uniform", 0L), "positive")
  expect_error(ggml_rand(cs, k, "uniform", 4L, stream = -1L), "stream")
  expect_error(ggml_rand(cs, k, "cauchy", 4L))
})

test_that("ggml_rand: the Vulkan shader draws the CPU's numbers", {
  skip_if_no_gpu_rand()
  cpu <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(cpu), add = TRUE)
  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(gpu), add = TRUE)
  # 100000 elements cross many 512-wide workgroups; 7 is a partial one.
  for (dims in list(7L, c(6L, 16L), 100000L)) {
    for (key in list(c(0, 0), c(0.3, 0.6))) {
      info <- paste("dims", paste(dims, collapse = "x"), "key", paste(key, collapse = ","))
      # uniform: bit-identical
      expect_identical(raw_rand(gpu, key, "uniform", dims, 3L),
                       raw_rand(cpu, key, "uniform", dims, 3L), info = info)
      # log / cos / sqrt: last bits may differ
      expect_equal(raw_rand(gpu, key, "gumbel", dims),
                   raw_rand(cpu, key, "gumbel", dims), tolerance = 1e-4, info = info)
      expect_equal(raw_rand(gpu, key, "normal", dims),
                   raw_rand(cpu, key, "normal", dims), tolerance = 1e-4, info = info)
    }
  }
})
