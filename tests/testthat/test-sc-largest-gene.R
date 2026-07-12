# op = "largest_gene": per-cell highest-expressed gene + its share of the cell's
# total counts — Seurat's largest_gene / percent.Largest.Gene QC metric
# (qlcMatrix::colMax(counts, which = TRUE)). Runs on the CPU directly over the
# dgCMatrix CSC slots (@x/@i/@p) without densifying, so the sparse and dense
# paths must agree with the plain-R reference exactly, including the all-zero
# (empty) cell edge case.

skip_if_not_installed("Matrix")

# plain-R reference: exactly what Seurat's percent.Largest.Gene computes.
largest_gene_ref <- function(mat) {
  m <- as.matrix(mat)
  cs <- colSums(m)
  idx <- max.col(t(m), ties.method = "first")
  mx  <- m[cbind(idx, seq_len(ncol(m)))]
  empty <- cs == 0
  gene <- rownames(m)[idx]
  gene[empty] <- NA_character_
  denom <- cs; denom[empty] <- 1
  data.frame(largest_gene = gene,
             percent.Largest.Gene = mx / denom * 100,
             row.names = colnames(m), stringsAsFactors = FALSE)
}

# reproducible counts (genes x cells) with an empty cell and a within-column tie.
make_counts <- function(g = 30L, c = 20L, seed = 11L) {
  set.seed(seed)
  m <- matrix(as.double(rpois(g * c, lambda = 0.5)), g, c)
  m[, 4L] <- 0                       # force an empty cell (all-zero column)
  m[1:2, 5L] <- 8; m[3:g, 5L] <- 0   # tie between rows 1 and 2 in cell 5
  rownames(m) <- paste0("Gene", seq_len(g))
  colnames(m) <- paste0("cell", seq_len(c))
  m
}

test_that("sparse CPU path matches the plain-R reference", {
  m   <- make_counts()
  sp  <- methods::as(Matrix::Matrix(m, sparse = TRUE), "dgCMatrix")
  ref <- largest_gene_ref(m)

  res <- ggmlR:::.ggmlr_largest_gene(sp, backend = "cpu")
  expect_equal(res$metadata$kind, "coldata")
  expect_equal(res$metadata$backend, "cpu")
  expect_identical(res$embedding$largest_gene, ref$largest_gene)
  expect_equal(res$embedding$percent.Largest.Gene, ref$percent.Largest.Gene)
  expect_identical(rownames(res$embedding), colnames(m))
})

test_that("dense and sparse paths are bit-identical", {
  m  <- make_counts()
  sp <- methods::as(Matrix::Matrix(m, sparse = TRUE), "dgCMatrix")

  rd <- ggmlR:::.ggmlr_largest_gene(m,  backend = "cpu")$embedding
  rs <- ggmlR:::.ggmlr_largest_gene(sp, backend = "cpu")$embedding
  expect_identical(rd$largest_gene, rs$largest_gene)
  expect_identical(rd$percent.Largest.Gene, rs$percent.Largest.Gene)
})

test_that("empty cell yields NA gene and 0 percent", {
  m  <- make_counts()
  sp <- methods::as(Matrix::Matrix(m, sparse = TRUE), "dgCMatrix")
  res <- ggmlR:::.ggmlr_largest_gene(sp, backend = "cpu")$embedding
  expect_true(is.na(res$largest_gene[4L]))
  expect_equal(res$percent.Largest.Gene[4L], 0)
})

test_that("ties resolve to the first row (matches max.col first)", {
  m  <- make_counts()
  sp <- methods::as(Matrix::Matrix(m, sparse = TRUE), "dgCMatrix")
  res <- ggmlR:::.ggmlr_largest_gene(sp, backend = "cpu")$embedding
  expect_equal(res$largest_gene[5L], "Gene1")   # first of the tied rows 1 and 2
})

test_that("op is registered as sparse_ok and dispatch keeps the matrix sparse", {
  entry <- ggml_ops_registry("largest_gene")
  expect_false(is.null(entry))
  expect_true(isTRUE(entry$sparse_ok))

  m  <- make_counts()
  sp <- methods::as(Matrix::Matrix(m, sparse = TRUE), "dgCMatrix")
  res <- ggml_run(ggml_task("largest_gene", sp, device = "cpu"))
  expect_equal(res$metadata$kind, "coldata")
  expect_s3_class(res$embedding, "data.frame")
})
