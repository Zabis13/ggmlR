#!/usr/bin/env Rscript
# ============================================================================
# Seurat integration — GPU neighbour graphs (op = "neighbors") end-to-end
# ============================================================================
# Runs ggmlR's FindNeighbors-equivalent on a Seurat object: a kNN graph and a
# shared-nearest-neighbour (SNN) graph whose edge weights are the Jaccard overlap
# of the two endpoints' neighbourhoods. The kNN search uses the FNN kd-tree when
# available, otherwise the GPU pairwise_dist.comp shader (honest f32) or the CPU.
# The SNN step is sparse matrix arithmetic. The two graphs land in obj@graphs
# under the Seurat naming convention (<assay>_nn / <assay>_snn), exactly where
# Seurat::FindClusters looks — so clustering runs straight off our graph.
#
# Canonical pipeline: PCA first (op = "embed"), then neighbours on the PC
# coordinates (reduction = "pca"), then FindClusters on <assay>_snn.
#
# Usage:
#   Rscript seurat_neighbors.R               # light default (~300 cells)
#   Rscript seurat_neighbors.R 2000          # 2000 cells
#
# Requires (Suggests, install separately): Seurat, SeuratObject, Matrix, FNN
# ============================================================================

suppressMessages({
  library(ggmlR)
  library(Seurat)
  library(SeuratObject)
})

args    <- commandArgs(trailingOnly = TRUE)
n_cells <- if (length(args) >= 1) as.integer(args[1]) else 300L
n_genes <- if (length(args) >= 2) as.integer(args[2]) else 200L

have_gpu <- isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE))
cat("Vulkan GPU:", if (have_gpu) "yes" else "no", "\n")
cat(sprintf("Config: %d cells x %d genes\n", n_cells, n_genes))

# ---- synthetic dataset (genes x cells), three loose clusters ---------------
set.seed(1)
n_per   <- n_cells %/% 3L
centres <- list(c(2, 0), c(-2, 2), c(0, -2))
counts <- do.call(cbind, lapply(centres, function(mu) {
  base <- matrix(rpois(n_genes * n_per, lambda = 5), nrow = n_genes)
  base[1, ] <- base[1, ] + round(mu[1] * 3)
  base[2, ] <- base[2, ] + round(mu[2] * 3)
  pmax(base, 0L)
}))
rownames(counts) <- paste0("gene", seq_len(n_genes))
colnames(counts) <- paste0("cell", seq_len(ncol(counts)))
obj   <- NormalizeData(CreateSeuratObject(counts = counts), verbose = FALSE)
assay <- DefaultAssay(obj)

# ---- 1. PCA (embed) so the neighbour search has a sensible space -----------
obj <- RunGGML(obj, op = "embed", n_components = 10L, reduction_name = "pca")
cat("\nPCA embeddings (cells x comps):",
    paste(dim(Embeddings(obj, "pca")), collapse = " x "), "\n")

# ---- 2. neighbour graphs on the PC coordinates -----------------------------
t_nn <- system.time(
  obj <- RunGGML(obj, op = "neighbors", reduction = "pca", dims = 1:10,
                 n_neighbors = 20L)
)[["elapsed"]]

prov <- Misc(obj, slot = "ggml_ggml")
cat(sprintf("neighbours: backend %s  k %d  SNN edges %d   (%.3f s)\n",
            prov$backend, prov$n_neighbors, prov$n_snn_edges, t_nn))
cat("graphs written:", paste(names(obj@graphs), collapse = ", "), "\n")

# ---- 3. cluster off our SNN graph (Seurat's own Louvain) -------------------
obj <- FindClusters(obj, graph.name = paste0(assay, "_snn"), verbose = FALSE)
tab <- table(obj$seurat_clusters)
cat("clusters found:", length(tab), " sizes:", paste(tab, collapse = " "), "\n")

cat("\nDone.\n")
