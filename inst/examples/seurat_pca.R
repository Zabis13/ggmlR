#!/usr/bin/env Rscript
# ============================================================================
# Seurat integration — GPU PCA (embed) end-to-end
# ============================================================================
# Runs ggmlR's Vulkan-accelerated PCA reduction directly on a Seurat object.
# Covers both the Seurat v5 (Assay5 / LayerData) and the legacy v4 (Assay /
# GetAssayData) extraction paths, and checks the result against base prcomp.
#
# Usage:
#   Rscript seurat_pca.R
#
# Requires (Suggests, install separately): Seurat, SeuratObject, Matrix
# ============================================================================

suppressMessages({
  library(ggmlR)
  library(Seurat)
  library(SeuratObject)
})

# ---- tiny synthetic dataset (genes x cells) --------------------------------
make_obj <- function() {
  set.seed(1)
  counts <- matrix(rpois(200L * 60L, lambda = 5), nrow = 200L, ncol = 60L)
  rownames(counts) <- paste0("gene", seq_len(200L))
  colnames(counts) <- paste0("cell", seq_len(60L))
  NormalizeData(CreateSeuratObject(counts = counts), verbose = FALSE)
}

run_demo <- function(obj, label) {
  cat(sprintf("\n=== %s ===\n", label))
  cat("Assay class:", class(obj[["RNA"]])[1], "\n")

  # one call: extract -> task -> GPU run -> inject reduction
  obj <- RunGGML(obj, op = "embed", n_components = 10L, reduction_name = "ggml")

  emb <- Embeddings(obj, reduction = "ggml")
  cat("Embeddings (cells x comps):", paste(dim(emb), collapse = " x "), "\n")

  # provenance: backend + timings live in Misc(obj, "<reduction>_ggml")
  prov <- Misc(obj, slot = "ggml_ggml")
  cat("backend:", prov$backend, " total(s):", prov$timings[["total"]], "\n")

  # correctness vs base prcomp (PCs are sign-ambiguous -> compare |cor|)
  mat <- ggml_extract(obj, layer = "data")
  ref <- prcomp(t(as.matrix(mat)), center = TRUE, scale. = FALSE)$x[, 1:10]
  cors <- sapply(1:10, function(i) abs(cor(emb[, i], ref[, i])))
  cat("min |cor| vs prcomp:", round(min(cors), 5), "\n")
  invisible(obj)
}

# ---- v5 path (default in Seurat 5: Assay5 / LayerData) ----------------------
run_demo(make_obj(), "Seurat v5 (Assay5)")

# ---- v4 path (legacy Assay / GetAssayData) ----------------------------------
old <- options(Seurat.object.assay.version = "v3")
run_demo(make_obj(), "Seurat v4 (legacy Assay)")
options(old)

cat("\nDone.\n")
