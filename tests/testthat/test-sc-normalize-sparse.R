# Sparse LogNormalize: the sparse path (transforming a dgCMatrix's stored @x in
# place) must reproduce Seurat's LogNormalize = log1p(x / colSum * scale_factor)
# exactly, whether it runs on the CPU (elementwise over @x) or on the GPU shader
# (sparse_lognorm.comp). log1p(0) = 0, so zeros never move: the sparsity pattern
# (@i / @p) is preserved and only @x changes. The sparse path exists to avoid
# densifying the counts matrix (tens of GB on the full dataset); it targets
# numeric parity with the dense reference, not a speed-up (LogNormalize is
# memory-bound O(nnz)).

skip_if_not_installed("Matrix")

# the GPU path flips the global device to "gpu"; restore it when this file ends
# so the state does not leak into later test files (see helper-device.R).
withr::defer(ag_device("cpu"), teardown_env())

# dense reference: exactly what stock Seurat NormalizeData(LogNormalize) computes.
lognorm_dense_ref <- function(mat, scale_factor = 1e4) {
  cs <- colSums(mat); cs[cs == 0] <- 1
  log1p(sweep(mat, 2L, scale_factor / cs, `*`))
}

# a reproducible sparse counts matrix (genes x cells) with ~15% non-zeros, some
# empty cells (all-zero columns) to exercise the colSum == 0 guard.
make_counts <- function(g = 40L, c = 25L, seed = 7L) {
  set.seed(seed)
  vals <- rpois(g * c, lambda = 0.4)             # mostly zeros
  m <- matrix(as.double(vals), g, c)
  m[, 3L] <- 0                                    # force an empty cell
  Matrix::Matrix(m, sparse = TRUE) |> methods::as("dgCMatrix")
}

test_that("sparse CPU path matches the dense reference (no GPU needed)", {
  m   <- make_counts()
  out <- ggmlR:::.ggmlr_normalize_sparse(m, scale_factor = 1e4, backend = "cpu")

  expect_s4_class(out, "dgCMatrix")
  expect_identical(attr(out, "backend"), "cpu")
  # sparsity pattern untouched: log1p(0) = 0
  expect_identical(out@i, m@i)
  expect_identical(out@p, m@p)

  ref <- lognorm_dense_ref(as.matrix(m))
  expect_equal(as.matrix(out), ref, tolerance = 1e-12,
               ignore_attr = TRUE)
})

test_that("sparse engine matches the dense engine end to end (CPU)", {
  m <- make_counts()

  # sparse path through the registered engine
  res_sparse <- ggmlR:::.ggmlr_normalize_gpu(m, backend = "cpu")
  expect_true(isTRUE(res_sparse$metadata$sparse))
  expect_identical(res_sparse$metadata$layer, "data")

  # dense path through the same engine (feed a plain matrix)
  res_dense <- ggmlR:::.ggmlr_normalize_gpu(as.matrix(m), backend = "cpu")

  expect_equal(as.matrix(res_sparse$embedding),
               as.matrix(res_dense$embedding),
               tolerance = 1e-10, ignore_attr = TRUE)
})

test_that("an all-zero cell stays all-zero (colSum guard)", {
  m   <- make_counts()
  out <- ggmlR:::.ggmlr_normalize_sparse(m, scale_factor = 1e4, backend = "cpu")
  # column 3 was forced empty; it must remain all zero (no NaN/Inf from 1/0)
  expect_true(all(as.matrix(out)[, 3L] == 0))
  expect_false(any(is.na(out@x)))
  expect_false(any(is.infinite(out@x)))
})

# ---- GPU shader: sparse_lognorm.comp vs the dense reference -----------------
# The shader runs f32 while the reference runs double, so this is a tolerance
# check, not identity. Only @x is written; the pattern must be preserved.

test_that("sparse GPU shader matches the dense reference", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  m   <- make_counts()
  out <- ggmlR:::.ggmlr_normalize_sparse(m, scale_factor = 1e4, backend = "vulkan")
  skip_if(!identical(attr(out, "backend"), "vulkan"),
          "No live Vulkan backend for sparse LogNormalize dispatch")

  expect_s4_class(out, "dgCMatrix")
  # pattern preserved
  expect_identical(out@i, m@i)
  expect_identical(out@p, m@p)

  ref <- lognorm_dense_ref(as.matrix(m))
  # f32 shader vs double reference
  expect_equal(as.matrix(out), ref, tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("sparse GPU and sparse CPU paths agree", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  m    <- make_counts()
  gpu  <- ggmlR:::.ggmlr_normalize_sparse(m, 1e4, backend = "vulkan")
  skip_if(!identical(attr(gpu, "backend"), "vulkan"),
          "No live Vulkan backend for sparse LogNormalize dispatch")
  cpu  <- ggmlR:::.ggmlr_normalize_sparse(m, 1e4, backend = "cpu")

  expect_equal(gpu@x, cpu@x, tolerance = 1e-5)
})
