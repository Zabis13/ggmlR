library(ggmlR)

# ggmlR Tensor Parallelism (P2P) — Stage E2: row-split math.
# These tests exercise the pure arithmetic of the Vulkan split buffer type
# (ggml_vulkan_split_row_ranges). They touch no GPU and only require Vulkan to
# be compiled in, so they run on a single-GPU machine.

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
