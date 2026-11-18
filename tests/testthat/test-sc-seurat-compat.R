# Seurat compatibility checks for op = "neighbors", split out of
# test-sc-neighbors.R. Excluded from the source tarball via .Rbuildignore while
# Seurat is not in Suggests; runs locally (and skips) through tests/testthat.R.

# engines flip the global device to "gpu"; restore it when this file ends so the
# state does not leak into later test files (see helper-device.R).
withr::defer(ag_device("cpu"), teardown_env())

# The SNN weights must match Seurat::FindNeighbors exactly when Seurat is given
# the same exact kNN (nn.method = "rann"). Seurat's default Annoy is approximate,
# so the agreement is exact only on the exact path; that is the contract we test.
test_that("SNN graph matches Seurat::FindNeighbors (exact kNN)", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("FNN")

  set.seed(1)
  n <- 150L
  X <- matrix(stats::rnorm(n * 10), n, 10)            # cells x features
  rownames(X) <- paste0("cell", seq_len(n))
  mat <- t(X)                                         # engine wants features x cells
  k <- 20L; prune <- 1 / 15

  res <- ggmlR:::.ggmlr_neighbors_gpu(mat, n_neighbors = k, prune_snn = prune,
                                      backend = "cpu")
  ours <- as.matrix(res$metadata$snn)

  sn <- suppressMessages(
    Seurat::FindNeighbors(X, k.param = k + 1L, prune.SNN = prune,
                          nn.method = "rann", verbose = FALSE))
  theirs <- as.matrix(as(sn$snn, "CsparseMatrix"))
  theirs <- theirs[rownames(ours), colnames(ours)]

  expect_equal(ours, theirs, ignore_attr = TRUE, tolerance = 1e-9)
})

# End-to-end: RunGGML(op = "neighbors") on a Seurat object writes the two Graph
# objects into @graphs under the Seurat naming convention, and the resulting SNN
# graph is accepted by Seurat::FindClusters — the real downstream consumer.
test_that("RunGGML neighbors writes @graphs accepted by FindClusters", {
  skip_if_not_installed("Seurat")
  skip_if_not_installed("FNN")

  set.seed(1)
  g <- 30L; n <- 120L
  counts <- matrix(stats::rpois(g * n, lambda = 5), g, n)
  rownames(counts) <- paste0("gene", seq_len(g))
  colnames(counts) <- paste0("cell", seq_len(n))
  obj <- suppressWarnings(Seurat::NormalizeData(
    SeuratObject::CreateSeuratObject(counts = counts), verbose = FALSE))
  assay <- SeuratObject::DefaultAssay(obj)

  # canonical path: PCA first, then neighbours on the PC coordinates (the
  # neighbour engine wants a feature space, not the raw normalised layer).
  obj <- RunGGML(obj, op = "embed", n_components = 10L, reduction_name = "pca")
  obj <- RunGGML(obj, op = "neighbors", reduction = "pca", dims = 1:10,
                 n_neighbors = 20L, device = "cpu")

  # both graphs are present under <assay>_nn / <assay>_snn, as Graph objects
  expect_true(paste0(assay, "_nn")  %in% names(obj@graphs))
  expect_true(paste0(assay, "_snn") %in% names(obj@graphs))
  snn <- obj[[paste0(assay, "_snn")]]
  expect_s4_class(snn, "Graph")
  expect_equal(dim(snn), c(n, n))

  # provenance recorded; the SNN graph clusters without error
  prov <- SeuratObject::Misc(obj, slot = "ggml_ggml")
  expect_equal(prov$backend %in% c("fnn", "vulkan", "cpu"), TRUE)
  clustered <- suppressMessages(
    Seurat::FindClusters(obj, graph.name = paste0(assay, "_snn"),
                         verbose = FALSE))
  expect_true(length(unique(clustered$seurat_clusters)) >= 1L)
})
