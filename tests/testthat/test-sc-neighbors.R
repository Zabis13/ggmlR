# op = "neighbors": Seurat-compatible kNN + SNN (shared nearest neighbour)
# graphs. The engine works on a bare features x cells matrix; the Seurat @graphs
# injection is a separate slice. SNN weights are the Jaccard overlap of the two
# endpoints' neighbourhoods (self included), pruned below prune_snn — this is
# numerically identical to Seurat::FindNeighbors when both use the same exact
# kNN, which test-sc-seurat-compat.R verifies.

# engines flip the global device to "gpu"; restore it when this file ends so the
# state does not leak into later test files (see helper-device.R).
withr::defer(ag_device("cpu"), teardown_env())

test_that("neighbors engine returns well-formed kNN + SNN graphs", {
  skip_if_not_installed("Matrix")
  set.seed(1)
  mat <- matrix(stats::rnorm(10 * 60), 10, 60)        # 10 features x 60 cells
  colnames(mat) <- paste0("cell", seq_len(60))

  res <- ggmlR:::.ggmlr_neighbors_gpu(mat, n_neighbors = 15L, backend = "cpu")

  expect_equal(res$metadata$kind, "graph")
  nn  <- res$metadata$nn
  snn <- res$metadata$snn
  expect_equal(dim(nn),  c(60L, 60L))
  expect_equal(dim(snn), c(60L, 60L))

  # kNN graph is binary, k+1 ones per row (self included), names carried through
  expect_true(all(nn@x == 1))
  expect_equal(unname(Matrix::rowSums(nn)), rep(16, 60))   # k + self
  expect_equal(rownames(snn), colnames(mat))

  # SNN weights are Jaccard in (0, 1]; the diagonal is 1 (a point shares its
  # whole neighbourhood with itself)
  expect_true(all(snn@x > 0 & snn@x <= 1 + 1e-9))
  expect_equal(unname(Matrix::diag(snn)), rep(1, 60))
  # embedding is the SNN graph (what clustering consumes)
  expect_identical(res$embedding, snn)
})

test_that("SNN pruning drops weak edges", {
  skip_if_not_installed("Matrix")
  set.seed(2)
  mat <- matrix(stats::rnorm(8 * 50), 8, 50)
  colnames(mat) <- paste0("c", seq_len(50))

  loose <- ggmlR:::.ggmlr_neighbors_gpu(mat, n_neighbors = 15L,
                                        prune_snn = 0, backend = "cpu")
  tight <- ggmlR:::.ggmlr_neighbors_gpu(mat, n_neighbors = 15L,
                                        prune_snn = 0.5, backend = "cpu")
  # a higher Jaccard threshold can only keep fewer (or equal) edges
  expect_lte(tight$metadata$n_snn_edges, loose$metadata$n_snn_edges)
  expect_true(all(tight$metadata$snn@x >= 0.5))
})
