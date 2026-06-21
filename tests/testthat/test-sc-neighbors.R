# op = "neighbors": Seurat-compatible kNN + SNN (shared nearest neighbour)
# graphs. The engine works on a bare features x cells matrix; the Seurat @graphs
# injection is a separate slice. SNN weights are the Jaccard overlap of the two
# endpoints' neighbourhoods (self included), pruned below prune_snn — this is
# numerically identical to Seurat::FindNeighbors when both use the same exact
# kNN, which the test below verifies.

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
