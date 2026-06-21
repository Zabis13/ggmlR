#!/usr/bin/env Rscript
# ============================================================================
# Seurat integration — GPU UMAP (op = "umap") end-to-end
# ============================================================================
# Runs ggmlR's UMAP reduction on a Seurat object. By default both heavy phases
# run on the GPU: the pairwise distance matrix behind the kNN graph goes through
# pairwise_dist.comp (honest f32), and the SGD layout goes through umap_sgd.comp.
# Each phase falls back to its exact CPU reference independently when no GPU is
# live. The top-k selection and fuzzy simplicial set in between stay on the CPU.
# The SGD GPU/CPU paths share the same PCG RNG and a/b schedule, so for a given
# base_seed they agree to within float32 precision.
#
# Canonical pipeline: PCA first (op = "embed"), then UMAP on the PC coordinates
# (reduction = "pca"), exactly as Seurat's own RunUMAP does.
#
# The SGD shader scales with edges (≈ cells × n_neighbors) and epochs (one GPU
# dispatch per epoch), so push those up to put real load on the GPU. Note the
# kNN graph + fuzzy set are built on the CPU (brute-force, O(cells^2)); past a
# few thousand cells that CPU prep, not the GPU SGD, becomes the bottleneck.
#
# Usage:
#   Rscript seurat_umap.R                 # light default (~270 cells)
#   Rscript seurat_umap.R 2000 500        # 2000 cells, 500 epochs (more GPU)
#   Rscript seurat_umap.R 2000 500 400    # ... and 400 genes
#
# Requires (Suggests, install separately): Seurat, SeuratObject, Matrix
# ============================================================================

suppressMessages({
  library(ggmlR)
  library(Seurat)
  library(SeuratObject)
})

# ---- CLI: cells (total, split across 3 clusters), epochs, genes ------------
args    <- commandArgs(trailingOnly = TRUE)
n_cells <- if (length(args) >= 1) as.integer(args[1]) else 270L
n_epochs<- if (length(args) >= 2) as.integer(args[2]) else 200L
n_genes <- if (length(args) >= 3) as.integer(args[3]) else 200L

have_gpu <- isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE))
cat("Vulkan GPU:", if (have_gpu) "yes (SGD on shader)" else
                   "no (SGD on CPU reference)", "\n")
cat(sprintf("Config: %d cells x %d genes, %d SGD epochs\n",
            n_cells, n_genes, n_epochs))

# ---- synthetic dataset (genes x cells), three loose clusters ---------------
make_obj <- function(n_cells, n_genes) {
  set.seed(1)
  n_per   <- n_cells %/% 3L
  centres <- list(c(2, 0), c(-2, 2), c(0, -2))
  counts <- do.call(cbind, lapply(centres, function(mu) {
    base <- matrix(rpois(n_genes * n_per, lambda = 5), nrow = n_genes)
    # nudge the first two genes per cluster so PCA/UMAP have structure to find
    base[1, ] <- base[1, ] + round(mu[1] * 3)
    base[2, ] <- base[2, ] + round(mu[2] * 3)
    pmax(base, 0L)
  }))
  rownames(counts) <- paste0("gene", seq_len(n_genes))
  colnames(counts) <- paste0("cell", seq_len(ncol(counts)))
  NormalizeData(CreateSeuratObject(counts = counts), verbose = FALSE)
}

obj <- make_obj(n_cells, n_genes)

# ---- 1. PCA (embed) so UMAP has a sensible input space ----------------------
obj <- RunGGML(obj, op = "embed", n_components = 10L, reduction_name = "pca")
cat("\nPCA embeddings (cells x comps):",
    paste(dim(Embeddings(obj, "pca")), collapse = " x "), "\n")

# ---- 2. UMAP on the PC coordinates -----------------------------------------
t_umap <- system.time(
  obj <- RunGGML(obj, op = "umap", reduction = "pca", dims = 1:10,
                 reduction_name = "umap", n_neighbors = 15L,
                 n_epochs = n_epochs)
)[["elapsed"]]

emb  <- Embeddings(obj, reduction = "umap")
prov <- Misc(obj, slot = "umap_ggml")
cat("UMAP embeddings (cells x 2):", paste(dim(emb), collapse = " x "), "\n")
cat("backend: dist:", prov$backend_dist, " sgd:", prov$backend_sgd,
    " (overall:", prov$backend, ")",
    " a:", round(prov$a, 4), " b:", round(prov$b, 4),
    " edges:", prov$n_edges, "\n")
# Split the engine time: the graph phase runs the GPU distance shader
# (pairwise_dist.comp) + CPU top-k/fuzzy build; the SGD layout is umap_sgd.comp.
# At scale the CPU top-k selection dominates — the GPU shaders stay cheap.
tm <- prov$timings
cat(sprintf("timing: graph(dist %s) %.3f s  +  SGD(%s) %.3f s  =  engine %.3f s",
            prov$backend_dist, tm[["graph"]], prov$backend_sgd, tm[["sgd"]],
            tm[["total"]]))
cat(sprintf("   (wall incl. Seurat I/O %.3f s)\n", t_umap))

# ---- 3. sanity: the three planted clusters should separate in 2-D ----------
# crude check — k-means on the layout should recover ~3 groups with low spread
km <- kmeans(emb, centers = 3L, nstart = 5L)
cat("within-cluster SS / total SS:",
    round(km$tot.withinss / km$totss, 3), "(lower = cleaner separation)\n")

cat("\nDone.\n")
