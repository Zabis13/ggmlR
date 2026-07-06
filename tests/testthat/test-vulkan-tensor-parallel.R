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
