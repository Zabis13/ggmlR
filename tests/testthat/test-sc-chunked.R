# Chunked (streaming) engines: op = "scale" and op = "embed" (PCA) can process a
# sparse features x cells matrix in cell-blocks so the full dense matrix is never
# held (datasets whose dense form exceeds memory). The chunked result must equal
# the non-chunked result to floating-point noise, at any chunk_size -- including
# a size that does not divide the number of cells, and one larger than it (which
# collapses to a single block). normalize is sparse already, so chunk_size is a
# documented no-op there.

skip_if_not_installed("Matrix")

# restore the device so the (CPU-default) engines here do not leak "gpu" state
withr::defer(ag_device("cpu"), teardown_env())

# a small, reproducible sparse genes x cells matrix with some all-zero structure
set.seed(42)
mk_sparse <- function(n_feat = 30L, n_cell = 47L, density = 0.3) {
  d <- matrix(0, n_feat, n_cell)
  nz <- sample.int(n_feat * n_cell, floor(density * n_feat * n_cell))
  d[nz] <- rpois(length(nz), 5) + 1
  rownames(d) <- paste0("g", seq_len(n_feat))
  colnames(d) <- paste0("c", seq_len(n_cell))
  methods::as(methods::as(d, "CsparseMatrix"), "dgCMatrix")
}

sp  <- mk_sparse()
dn  <- as.matrix(sp)

test_that("chunk_cols slices cover every column exactly once", {
  for (cs in c(1L, 7L, 47L, 100L, NULL)) {
    blocks <- ggmlR:::.ggmlr_chunk_cols(ncol(sp), cs)
    expect_identical(sort(unlist(blocks)), seq_len(ncol(sp)))
  }
  # size >= ncol (or NULL) is a single block; a divisor gives even blocks
  expect_length(ggmlR:::.ggmlr_chunk_cols(47L, NULL), 1L)
  expect_length(ggmlR:::.ggmlr_chunk_cols(47L, 100L), 1L)
  expect_length(ggmlR:::.ggmlr_chunk_cols(20L, 5L), 4L)
})

test_that("chunked scale equals non-chunked scale (CPU)", {
  full <- ggmlR:::.ggmlr_scale_gpu(dn, backend = "cpu")
  for (cs in c(1L, 10L, 13L, 50L)) {          # 13 does not divide 47; 50 > 47
    chk <- ggmlR:::.ggmlr_scale_gpu(sp, backend = "cpu", chunk_size = cs)
    expect_equal(chk$embedding, full$embedding, tolerance = 1e-10,
                 info = paste("chunk_size =", cs))
    expect_true(isTRUE(chk$metadata$chunked))
    expect_identical(dimnames(chk$embedding), dimnames(full$embedding))
  }
})

test_that("chunked PCA equals non-chunked PCA (CPU) up to sign", {
  k <- 5L
  full <- ggmlR:::.ggmlr_pca_gpu(dn, n_components = k, backend = "cpu")
  for (cs in c(1L, 11L, 13L, 60L)) {
    chk <- ggmlR:::.ggmlr_pca_gpu(sp, n_components = k, backend = "cpu",
                                  chunk_size = cs)
    expect_true(isTRUE(chk$metadata$chunked))
    # eigenvector signs are arbitrary; compare |correlation| of each PC's scores
    cors <- vapply(seq_len(k), function(i)
      abs(cor(chk$embedding[, i], full$embedding[, i])), numeric(1))
    expect_gt(min(cors), 1 - 1e-6)
    # standard deviations (eigenvalues) are sign-independent -> match directly
    expect_equal(chk$metadata$stdev, full$metadata$stdev, tolerance = 1e-8,
                 info = paste("chunk_size =", cs))
  }
})

test_that("normalize accepts chunk_size as a no-op", {
  a <- ggmlR:::.ggmlr_normalize_gpu(sp, backend = "cpu")
  b <- ggmlR:::.ggmlr_normalize_gpu(sp, backend = "cpu", chunk_size = 10L)
  expect_equal(a$embedding, b$embedding, tolerance = 1e-12)
})
