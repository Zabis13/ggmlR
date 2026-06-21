# Single-cell adapter on SingleCellExperiment (Bioconductor / S4). The engines
# are shared with the Seurat path; these tests cover the SCE-specific
# extract/inject (assay -> matrix, reducedDim <- embedding, graphs/provenance in
# metadata) and the RunGGML.SingleCellExperiment dispatch. Everything skips
# cleanly when the Bioconductor packages are absent.

# engines flip the global device to "gpu"; restore it when this file ends so the
# state does not leak into later test files (see helper-device.R).
withr::defer(ag_device("cpu"), teardown_env())

make_sce <- function(ng = 40L, nc = 100L, seed = 1) {
  set.seed(seed)
  counts <- matrix(stats::rpois(ng * nc, lambda = 5), ng, nc)
  rownames(counts) <- paste0("gene", seq_len(ng))
  colnames(counts) <- paste0("cell", seq_len(nc))
  SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts, logcounts = log1p(counts)))
}

test_that("ggml_extract reads the logcounts assay by default", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- make_sce()
  mat <- ggml_extract(sce)

  expect_true(is.matrix(mat))
  expect_equal(dim(mat), c(40L, 100L))                    # genes x cells
  expect_equal(mat, as.matrix(SummarizedExperiment::assay(sce, "logcounts")),
               ignore_attr = TRUE)

  # an explicit assay name is honoured; genes/cells subset before materialising
  raw <- ggml_extract(sce, layer = "counts", genes = 1:5, cells = 1:10)
  expect_equal(dim(raw), c(5L, 10L))
})

test_that("RunGGML(embed) writes a reducedDim and provenance", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")

  sce <- make_sce()
  sce <- RunGGML(sce, op = "embed", n_components = 10L, device = "cpu")

  expect_true("ggml" %in% SingleCellExperiment::reducedDimNames(sce))
  rd <- SingleCellExperiment::reducedDim(sce, "ggml")
  expect_equal(dim(rd), c(100L, 10L))                     # cells x components
  expect_equal(rownames(rd), colnames(sce))

  prov <- S4Vectors::metadata(sce)$ggml_ggml
  expect_true(prov$backend %in% c("vulkan", "cpu"))
  expect_true(!is.null(prov$timings))
})

test_that("RunGGML(neighbors) stores nn/snn graphs in metadata", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("FNN")

  sce <- make_sce()
  sce <- RunGGML(sce, op = "embed", n_components = 10L, device = "cpu")
  sce <- RunGGML(sce, op = "neighbors", reduction = "ggml", dims = 1:10,
                 n_neighbors = 20L, device = "cpu")

  md <- S4Vectors::metadata(sce)
  expect_true(!is.null(md$ggml_nn))
  expect_true(!is.null(md$ggml_snn))
  expect_equal(dim(md$ggml_snn), c(100L, 100L))
  # the bulky graphs are NOT duplicated into the provenance entry
  expect_null(md$ggml_ggml$nn)
  expect_null(md$ggml_ggml$snn)
})

test_that("RunGGML(normalize) overwrites the logcounts assay", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- make_sce()
  before <- as.matrix(SummarizedExperiment::assay(sce, "logcounts"))
  sce <- RunGGML(sce, op = "normalize", device = "cpu")
  after <- as.matrix(SummarizedExperiment::assay(sce, "logcounts"))

  # normalize reads counts and rewrites logcounts -> the assay changes
  expect_equal(dim(after), dim(before))
  expect_false(isTRUE(all.equal(before, after)))
  expect_equal(S4Vectors::metadata(sce)$logcounts_ggml$backend %in%
                 c("vulkan", "cpu"), TRUE)
})
