#!/usr/bin/env Rscript
# ============================================================================
# Seurat integration — GPU preprocessing (normalize + scale + PCA)
# ============================================================================
# Runs the heavy matrix steps of a standard Seurat pipeline on the Vulkan GPU
# via RunGGML(), and cross-checks each one against the stock Seurat function:
#   normalize -> NormalizeData (LogNormalize)
#   scale     -> ScaleData (z-score per gene, clamp at +10)
#   embed     -> RunPCA-style covariance PCA
#
# Usage:
#   Rscript seurat_preprocess_gpu.R
#
# Requires (Suggests): Seurat, SeuratObject, Matrix
# ============================================================================

suppressMessages({
  library(ggmlR)
  library(Seurat)
  library(SeuratObject)
})

# ---- tiny synthetic dataset (genes x cells) --------------------------------
set.seed(1)
ng <- 300L; nc <- 120L
counts <- matrix(rpois(ng * nc, lambda = 5), nrow = ng, ncol = nc)
rownames(counts) <- paste0("gene", seq_len(ng))
colnames(counts) <- paste0("cell", seq_len(nc))
counts <- methods::as(counts, "dgCMatrix")
obj <- CreateSeuratObject(counts = counts)

# ---- 1. normalize on the GPU vs Seurat NormalizeData -----------------------
obj <- RunGGML(obj, op = "normalize")                     # writes assay "data"
gpu_data <- as.matrix(SeuratObject::LayerData(obj, layer = "data"))
ref_data <- as.matrix(SeuratObject::LayerData(
  Seurat::NormalizeData(obj, verbose = FALSE), layer = "data"))
cat(sprintf("normalize: max abs err vs Seurat = %.2e\n",
            max(abs(gpu_data - ref_data))))

# ---- 2. scale on the GPU vs Seurat ScaleData -------------------------------
obj <- RunGGML(obj, op = "scale")                         # writes "scale.data"
gpu_scale <- as.matrix(SeuratObject::LayerData(obj, layer = "scale.data"))
ref_scale <- as.matrix(SeuratObject::GetAssayData(
  Seurat::ScaleData(obj, verbose = FALSE), layer = "scale.data"))
cat(sprintf("scale:     max abs err vs Seurat = %.2e   (row means ~0: %.2e)\n",
            max(abs(gpu_scale - ref_scale)), max(abs(rowMeans(gpu_scale)))))

# ---- 3. PCA (embed) on the GPU vs base prcomp ------------------------------
# Seurat's RunPCA() runs on scale.data; point embed at the same layer so the
# reference prcomp below uses exactly the matrix the embedding was built from.
obj <- RunGGML(obj, op = "embed", layer = "scale.data",
               n_components = 20L, reduction_name = "ggml")
emb <- SeuratObject::Embeddings(obj, "ggml")
ref <- prcomp(t(gpu_scale), center = TRUE, scale. = FALSE)$x[, 1:20]
mc  <- min(vapply(1:20, function(i) abs(cor(emb[, i], ref[, i])), numeric(1)))
cat(sprintf("embed:     min |cor| vs prcomp = %.4f\n", mc))

cat("\nBackends used:\n")
for (l in c("data", "scale.data"))
  cat(sprintf("  %-10s %s\n", l, SeuratObject::Misc(obj, slot = paste0(l, "_ggml"))$backend))
cat(sprintf("  %-10s %s\n", "ggml(embed)", SeuratObject::Misc(obj, slot = "ggml_ggml")$backend))
cat("\nDone.\n")
