#!/usr/bin/env Rscript
# ============================================================================
# Seurat integration — GPU PCA on large objects (context-sizing fix check)
# ============================================================================
# Exercises the Vulkan PCA path on matrices large enough that the old fixed
# 32 MB autograd context overflowed ("Not enough memory in context"). With the
# context now sized from the actual tensors, these should run and match prcomp.
#
# Usage:
#   Rscript seurat_pca_large.R
#   Rscript seurat_pca_large.R 6000 8000    # custom max genes x cells
#
# Requires (Suggests): Seurat, SeuratObject, Matrix
# ============================================================================

suppressMessages({
  library(ggmlR)
  library(Seurat)
  library(SeuratObject)
})

if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)))
  stop("No Vulkan GPU available — this check needs a GPU.", call. = FALSE)

# default sweep; the last pair is the heaviest. Override the max via CLI.
args <- commandArgs(trailingOnly = TRUE)
sizes <- list(c(2000L, 3000L), c(3000L, 4000L), c(4000L, 5000L))
if (length(args) >= 2)
  sizes <- c(sizes, list(c(as.integer(args[1]), as.integer(args[2]))))

run_one <- function(ng, nc, k = 50L) {
  set.seed(1)
  counts <- matrix(rpois(ng * nc, lambda = 5), nrow = ng, ncol = nc)
  rownames(counts) <- paste0("gene", seq_len(ng))
  colnames(counts) <- paste0("cell", seq_len(nc))
  counts <- methods::as(counts, "dgCMatrix")
  obj <- suppressWarnings(
    NormalizeData(CreateSeuratObject(counts = counts), verbose = FALSE))

  t <- system.time(
    obj <- RunGGML(obj, op = "embed", n_components = k, device = "vulkan")
  )[["elapsed"]]

  # split GPU vs CPU: only the covariance/projection matmuls run on Vulkan; the
  # eigendecomposition (eigen(), O(genes^3)) and centring stay on the CPU, since
  # ggml has no eigensolver. The engine times the matmul phase separately, so we
  # can show how much of the wall time was actually on the GPU.
  prov   <- SeuratObject::Misc(obj, slot = "ggml_ggml")
  gpu_mm <- tryCatch(prov$timings[["matmul"]], error = function(e) NA_real_)

  emb <- SeuratObject::Embeddings(obj, "ggml")
  mat <- ggml_extract(obj, layer = "data")
  # CPU reference for the |cor| check. A full prcomp computes ALL components via
  # a dense SVD and dominates the runtime on large objects (it dwarfs the GPU
  # PCA it's verifying). We only need the top k, so use a truncated SVD (irlba)
  # when available — same k components, a fraction of the time. Falls back to
  # prcomp if irlba isn't installed (it's a Suggests; Seurat already pulls it).
  ref <- if (requireNamespace("irlba", quietly = TRUE)) {
    irlba::prcomp_irlba(t(mat), n = k, center = TRUE, scale. = FALSE)$x
  } else {
    prcomp(t(mat), center = TRUE, scale. = FALSE)$x[, seq_len(k)]
  }
  mc  <- min(vapply(seq_len(k),
                    function(i) abs(cor(emb[, i], ref[, i])), numeric(1)))

  cat(sprintf(paste0("%5d genes x %5d cells (cov %d x %d): %6.2f s",
                     "  (GPU mm %5.2f s / CPU %5.2f s)   min|cor|=%.4f%s\n"),
              ng, nc, ng, ng, t, gpu_mm, t - gpu_mm, mc,
              if (mc > 0.99) "  OK" else "  *** MISMATCH ***"))
}

cat("Vulkan PCA on increasingly large objects:\n")
for (s in sizes) run_one(s[1], s[2])
cat("\nDone.\n")
