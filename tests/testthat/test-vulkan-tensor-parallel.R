library(ggmlR)

# ggmlR Tensor Parallelism (P2P).
# Stage E2: pure row-split math (ggml_vulkan_split_row_ranges) — touches no GPU,
#   runs anywhere Vulkan is compiled in.
# Stage E4a: opaque-fd P2P self-test (ggml_vulkan_p2p_selftest) — touches the GPU
#   and skips gracefully when the fd transport is unsupported or when there are
#   fewer than 2 devices for a cross-device transfer.

skip_if_not(ggml_vulkan_available(), "Vulkan not compiled in")

ROUNDING <- 512L  # VK_SPLIT_MATRIX_ROW_PADDING

test_that("even split covers all rows contiguously", {
  nrows <- 4096L
  r <- ggml_vulkan_split_row_ranges(nrows, 2L)

  expect_named(r, c("row_low", "row_high"))
  expect_length(r$row_low, 2L)
  expect_length(r$row_high, 2L)

  # Coverage: first low is 0, last high is nrows.
  expect_equal(r$row_low[1], 0)
  expect_equal(r$row_high[length(r$row_high)], nrows)

  # Contiguous & non-overlapping: high[i] == low[i+1].
  expect_equal(r$row_high[1], r$row_low[2])

  # Even split of 4096 over 2 → 2048 each (2048 is a multiple of 512).
  expect_equal(r$row_high[1], 2048)
})

test_that("single device owns all rows", {
  r <- ggml_vulkan_split_row_ranges(1000L, 1L)
  expect_equal(r$row_low, 0)
  expect_equal(r$row_high, 1000)
})

test_that("boundaries are rounded down to the padding granularity", {
  # 4 devices over 5000 rows: interior boundaries must be multiples of 512.
  r <- ggml_vulkan_split_row_ranges(5000L, 4L)
  interior_highs <- r$row_high[-length(r$row_high)]
  expect_true(all(interior_highs %% ROUNDING == 0),
              info = paste("interior highs:", paste(interior_highs, collapse = ",")))
  # Last device still reaches exactly nrows (not rounded).
  expect_equal(r$row_high[4], 5000)
})

test_that("ranges are total, monotone and non-overlapping for many configs", {
  for (nrows in c(0L, 1L, 511L, 512L, 513L, 4096L, 100000L)) {
    for (nd in 1:8) {
      r <- ggml_vulkan_split_row_ranges(nrows, nd)

      # low[i] <= high[i]
      expect_true(all(r$row_high >= r$row_low),
                  info = sprintf("nrows=%d nd=%d inverted range", nrows, nd))
      # contiguity
      if (nd > 1) {
        expect_equal(r$row_high[-nd], r$row_low[-1],
                     info = sprintf("nrows=%d nd=%d not contiguous", nrows, nd))
      }
      # full coverage
      expect_equal(r$row_low[1], 0)
      expect_equal(r$row_high[nd], nrows)
      # no range exceeds bounds
      expect_true(all(r$row_low >= 0 & r$row_high <= nrows),
                  info = sprintf("nrows=%d nd=%d out of bounds", nrows, nd))
    }
  }
})

test_that("weighted split allocates more rows to heavier devices", {
  nrows <- 8192L
  r <- ggml_vulkan_split_row_ranges(nrows, 2L, weights = c(3, 1))

  rows_dev0 <- r$row_high[1] - r$row_low[1]
  rows_dev1 <- r$row_high[2] - r$row_low[2]

  # Device 0 has ~3x the rows of device 1 (within rounding slack).
  expect_gt(rows_dev0, rows_dev1)
  expect_equal(rows_dev0 + rows_dev1, nrows)
  # roughly 3:1 — allow one rounding block of slack
  expect_lt(abs(rows_dev0 / rows_dev1 - 3), 0.2)
})

test_that("weights length must match n_devices", {
  expect_error(ggml_vulkan_split_row_ranges(4096L, 2L, weights = c(1, 1, 1)))
})

test_that("n_devices must be positive", {
  expect_error(ggml_vulkan_split_row_ranges(4096L, 0L))
})

# ---------------------------------------------------------------------------
# Stage E4a: opaque-fd P2P self-test (correctness + cross-device bandwidth).
# Unlike the pure math above these DO touch the GPU and need external_memory_fd.
# They skip gracefully when the fd transport is unsupported (status == -2) or
# when there are too few real GPUs for a cross-device transfer.
# ---------------------------------------------------------------------------

# How many Vulkan devices does the backend see? Uses the same counter
# (ggml_backend_vk_get_device_count) that the self-test validates indices
# against, so the skip logic and the C side always agree.
vk_n_devices <- function() {
  as.integer(tryCatch(ggml_vulkan_device_count(), error = function(e) 0L))
}

test_that("fd loopback self-test verifies data on a single device", {
  skip_if_not(vk_n_devices() >= 1, "no Vulkan device")
  r <- ggml_vulkan_p2p_selftest(0L, 0L, bytes = 1L * 1024L * 1024L, iters = 1L)
  if (identical(r$status, -2L)) {
    skip("VK_KHR_external_memory_fd not supported on this device")
  }
  expect_identical(r$status, 0L)          # pattern round-tripped through the fd
  expect_true(is.character(r$report) && nzchar(r$report))
})

test_that("cross-device fd P2P transfers data correctly and reports bandwidth", {
  n <- vk_n_devices()
  skip_if_not(n >= 2, "need >= 2 Vulkan devices for cross-device P2P")

  r <- ggml_vulkan_p2p_selftest(0L, 1L, bytes = 64L * 1024L * 1024L, iters = 50L)
  if (identical(r$status, -2L)) {
    skip("VK_KHR_external_memory_fd not supported on these devices")
  }
  expect_identical(r$status, 0L)          # data verified across the two devices
  expect_true(r$gbps > 0)                 # a real bandwidth was measured

  # Informational only: we cannot assert a specific rate (hardware varies) and,
  # per the NVLink discussion, Vulkan exposes no API to confirm the physical
  # route — the rate is an inference, printed for the operator to read.
  cat(sprintf("\n[E4a] dev0->dev1 P2P: %.2f GB/s\n%s\n", r$gbps, r$report))
})

# ---------------------------------------------------------------------------
# Stage E3: tensor-parallel mul_mat (ggml_vulkan_split_mul_mat).
# Y = X %*% t(W) with W's rows split across devices and the result gathered.
# The n_devices == 1 case still exercises the full orchestration (backend init,
# per-device graph, gather) so correctness is testable on a single-GPU machine;
# the multi-device cases skip gracefully when there are < 2 real GPUs.
# Tolerance is loose because the Vulkan matmul accumulates in f16 on some drivers.
# ---------------------------------------------------------------------------

TP_TOL <- 5e-2  # f16 accumulation slack for the GPU matmul

test_that("split mul_mat on a single device equals X %*% t(W)", {
  skip_if_not(vk_n_devices() >= 1, "no Vulkan device")

  set.seed(1L)
  N <- 16L; K <- 8L; M <- 5L
  W <- matrix(rnorm(N * K), nrow = N)
  X <- matrix(rnorm(M * K), nrow = M)

  Y <- ggml_vulkan_split_mul_mat(W, X, n_devices = 1L)
  expect_equal(dim(Y), c(M, N))
  expect_lt(max(abs(Y - X %*% t(W))), TP_TOL)
})

test_that("split mul_mat across 2 devices equals the single-device result", {
  skip_if_not(vk_n_devices() >= 2, "need >= 2 Vulkan devices for a real split")

  set.seed(2L)
  N <- 2048L; K <- 64L; M <- 4L    # N large enough to give each device real rows
  W <- matrix(rnorm(N * K), nrow = N)
  X <- matrix(rnorm(M * K), nrow = M)

  ref <- X %*% t(W)
  Y   <- ggml_vulkan_split_mul_mat(W, X, n_devices = 2L)

  expect_equal(dim(Y), c(M, N))
  expect_lt(max(abs(Y - ref)), TP_TOL)
  cat(sprintf("\n[E3] 2-device split mul_mat: max|Y-ref| = %.2e\n%s\n",
              max(abs(Y - ref)), attr(Y, "report")))
})

test_that("weighted split still computes the correct product", {
  skip_if_not(vk_n_devices() >= 2, "need >= 2 Vulkan devices")

  set.seed(3L)
  N <- 2048L; K <- 32L; M <- 3L
  W <- matrix(rnorm(N * K), nrow = N)
  X <- matrix(rnorm(M * K), nrow = M)

  # 3:1 row split changes which device owns which rows but not the gathered result.
  Y <- ggml_vulkan_split_mul_mat(W, X, n_devices = 2L, weights = c(3, 1))
  expect_lt(max(abs(Y - X %*% t(W))), TP_TOL)
})

test_that("split mul_mat validates shapes", {
  skip_if_not(vk_n_devices() >= 1, "no Vulkan device")
  W <- matrix(rnorm(16 * 8), nrow = 16)   # K = 8
  X <- matrix(rnorm(5 * 7),  nrow = 5)    # K = 7 (mismatch)
  expect_error(ggml_vulkan_split_mul_mat(W, X, n_devices = 1L),
               "input dimension")
})

# ---------------------------------------------------------------------------
# Stage E4: split buffer type factory (ggml_vulkan_split_buffer_type).
# The factory + its get_alloc_size math touch no cross-device path, so they are
# testable on a single GPU. alloc_size must equal the sum of the padded per-device
# slice sizes, which we cross-check against the row-split math (E2) directly.
# ---------------------------------------------------------------------------

test_that("split buffer type factory returns a named config", {
  skip_if_not(vk_n_devices() >= 1, "no Vulkan device")
  bt <- ggml_vulkan_split_buffer_type(n_devices = 1L)
  expect_named(bt, c("ptr", "name", "alloc_size"))
  expect_true(is.character(bt$name) && nzchar(bt$name))
  expect_match(bt$name, "vk_split")
  expect_false(is.null(bt$ptr))
})

# Per-device slice size = nrows_split*K*4 + (K_pad - K)*4, i.e. the slice's own
# bytes plus ONE trailing pad row-remainder (matching the C get_alloc_size, which
# pads the row size once, not per row). Total over the active devices:
#   N*K*4 + n_active*(K_pad - K)*4
split_alloc_bytes <- function(N, K, n_devices, weights = NULL) {
  r <- ggml_vulkan_split_row_ranges(N, n_devices, weights = weights)
  rows <- r$row_high - r$row_low
  active <- sum(rows > 0)
  K_pad <- ceiling(K / ROUNDING) * ROUNDING
  N * K * 4 + active * (K_pad - K) * 4
}

test_that("even-split alloc size matches the padded-tail formula", {
  skip_if_not(vk_n_devices() >= 1, "no Vulkan device")
  N <- 2048L; K <- 64L                       # K not a multiple of 512 -> tail pad
  bt <- ggml_vulkan_split_buffer_type(n_devices = 1L, probe = c(N, K))
  expect_equal(bt$alloc_size, split_alloc_bytes(N, K, 1L))
})

test_that("split alloc size equals the sum of per-device slice bytes", {
  skip_if_not(vk_n_devices() >= 2, "need >= 2 devices to exercise a real split")
  N <- 4096L; K <- 128L
  bt <- ggml_vulkan_split_buffer_type(n_devices = 2L, probe = c(N, K))
  expect_equal(bt$alloc_size, split_alloc_bytes(N, K, 2L))
})

test_that("buffer type is cached: same config returns the same pointer", {
  skip_if_not(vk_n_devices() >= 1, "no Vulkan device")
  a <- ggml_vulkan_split_buffer_type(n_devices = 1L)
  b <- ggml_vulkan_split_buffer_type(n_devices = 1L)
  # identical external pointer address => same cached buffer_type
  expect_identical(
    utils::capture.output(print(a$ptr)),
    utils::capture.output(print(b$ptr))
  )
})

test_that("weighted split total matches the padded-tail formula", {
  skip_if_not(vk_n_devices() >= 2, "need >= 2 devices")
  N <- 4096L; K <- 64L
  bt <- ggml_vulkan_split_buffer_type(n_devices = 2L, weights = c(3, 1), probe = c(N, K))
  # The row weighting shifts which device owns which rows, but as long as both
  # devices own a non-empty slice the total is the same padded-tail sum.
  expect_equal(bt$alloc_size, split_alloc_bytes(N, K, 2L, weights = c(3, 1)))
})

# ---------------------------------------------------------------------------
# Stage E6.1: split across an explicit device subset (device_ids).
# On a single GPU we can still check that device_ids = 0 behaves like the
# default; a real subset like c(2,3) needs >= 4 devices.
# ---------------------------------------------------------------------------

test_that("device_ids = single device matches the default path", {
  skip_if_not(vk_n_devices() >= 1, "no Vulkan device")
  set.seed(4L)
  N <- 32L; K <- 8L; M <- 4L
  W <- matrix(rnorm(N * K), nrow = N)
  X <- matrix(rnorm(M * K), nrow = M)
  Y <- ggml_vulkan_split_mul_mat(W, X, device_ids = 0L)
  expect_lt(max(abs(Y - X %*% t(W))), TP_TOL)
})

test_that("split buffer type distinguishes device subsets in its name/cache", {
  skip_if_not(vk_n_devices() >= 4, "need >= 4 devices for distinct subsets")
  a <- ggml_vulkan_split_buffer_type(device_ids = c(0L, 1L), probe = c(2048, 64))
  b <- ggml_vulkan_split_buffer_type(device_ids = c(2L, 3L), probe = c(2048, 64))
  # Different subsets => different cache entries (different names & pointers).
  expect_false(identical(a$name, b$name))
  # Same total alloc size (same shape, same even split).
  expect_equal(a$alloc_size, b$alloc_size)
})

test_that("split mul_mat on the second GPU group {2,3} equals the reference", {
  skip_if_not(vk_n_devices() >= 4, "need >= 4 devices for a {2,3} group")
  set.seed(5L)
  N <- 2048L; K <- 64L; M <- 4L
  W <- matrix(rnorm(N * K), nrow = N)
  X <- matrix(rnorm(M * K), nrow = M)
  Y <- ggml_vulkan_split_mul_mat(W, X, device_ids = c(2L, 3L))
  expect_lt(max(abs(Y - X %*% t(W))), TP_TOL)
  cat(sprintf("\n[E6.1] TP on GPUs {2,3}: max|Y-ref| = %.2e\n", max(abs(Y - X %*% t(W)))))
})

# ---------------------------------------------------------------------------
# Stage E6.2: TPxDP hybrid orchestration. The batch-shard helper is pure R and
# runs anywhere; the full 2-replica x TP=2 forward needs 4 GPUs.
# ---------------------------------------------------------------------------

test_that("batch shards are contiguous, complete and near-even", {
  # 8 rows over 2 replicas -> 4 + 4
  s <- ggmlR:::.ggmlr_batch_shards(8L, 2L)
  expect_length(s, 2L)
  expect_equal(unlist(s), 1:8)
  expect_equal(lengths(s), c(4L, 4L))

  # 7 rows over 2 -> 4 + 3 (earlier shard gets the remainder)
  s <- ggmlR:::.ggmlr_batch_shards(7L, 2L)
  expect_equal(lengths(s), c(4L, 3L))
  expect_equal(unlist(s), 1:7)

  # more replicas than rows -> empty shards dropped
  s <- ggmlR:::.ggmlr_batch_shards(2L, 4L)
  expect_equal(unlist(s), 1:2)
  expect_true(all(lengths(s) >= 1L))
})

test_that("TPxDP forward equals X %*% t(W) across 2 replicas x TP=2", {
  skip_if_not(vk_n_devices() >= 4, "need >= 4 devices for 2 replicas x TP=2")
  set.seed(6L)
  N <- 2048L; K <- 64L; M <- 8L
  W <- matrix(rnorm(N * K), nrow = N)
  X <- matrix(rnorm(M * K), nrow = M)

  ref <- X %*% t(W)
  Y   <- ggml_tp_dp_forward(W, X, replicas = list(c(0L, 1L), c(2L, 3L)))

  expect_equal(dim(Y), c(M, N))
  expect_lt(max(abs(Y - ref)), TP_TOL)
  cat(sprintf("\n[E6.2] TPxDP 2 replicas x TP=2 (batch %d): max|Y-ref| = %.2e\n",
              M, max(abs(Y - ref))))
})

test_that("TPxDP with a single replica reduces to plain TP", {
  skip_if_not(vk_n_devices() >= 2, "need >= 2 devices")
  set.seed(7L)
  N <- 2048L; K <- 32L; M <- 5L
  W <- matrix(rnorm(N * K), nrow = N)
  X <- matrix(rnorm(M * K), nrow = M)
  Y <- ggml_tp_dp_forward(W, X, replicas = list(c(0L, 1L)))
  expect_lt(max(abs(Y - X %*% t(W))), TP_TOL)
})
