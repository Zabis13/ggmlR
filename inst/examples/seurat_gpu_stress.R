#!/usr/bin/env Rscript
# ============================================================================
# Seurat integration — GPU stress test (single-cell PCA under load)
# ============================================================================
# Drives the Vulkan-accelerated PCA path (RunGGML / embed) on a large synthetic
# Seurat object in a loop, so the GPU stays busy for a couple of minutes. Use it
# to watch GPU utilisation/VRAM (e.g. `radeontop`, `nvtop`) and to confirm the
# reduction stays numerically stable across many repeated runs.
#
# Usage:
#   Rscript seurat_gpu_stress.R [iters] [genes] [cells] [k]
#   Rscript seurat_gpu_stress.R 16            # ~2 min on a mid-range GPU
#   Rscript seurat_gpu_stress.R 30 2000 3000 50
#
# Requires (Suggests): Seurat, SeuratObject, Matrix
# ============================================================================

suppressMessages({
  library(ggmlR)
  library(Seurat)
  library(SeuratObject)
})

args  <- commandArgs(trailingOnly = TRUE)
iters <- if (length(args) >= 1) as.integer(args[1]) else 16L
ng    <- if (length(args) >= 2) as.integer(args[2]) else 2000L
nc    <- if (length(args) >= 3) as.integer(args[3]) else 3000L
k     <- if (length(args) >= 4) as.integer(args[4]) else 50L

if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)))
  stop("No Vulkan GPU available — this stress test needs a GPU.", call. = FALSE)

cat(sprintf("Stress: %d iters of embed(k=%d) on a %d genes x %d cells object\n",
            iters, k, ng, nc))

# ---- build the object once (kept out of the timed loop) --------------------
set.seed(1)
counts <- matrix(rpois(ng * nc, lambda = 5), nrow = ng, ncol = nc)
rownames(counts) <- paste0("gene", seq_len(ng))
colnames(counts) <- paste0("cell", seq_len(nc))
counts <- methods::as(counts, "dgCMatrix")
obj <- NormalizeData(CreateSeuratObject(counts = counts), verbose = FALSE)

# ---- warm-up (shader compile / first alloc) is not counted -----------------
obj <- RunGGML(obj, op = "embed", n_components = k, device = "vulkan")
ref <- SeuratObject::Embeddings(obj, "ggml")
cat("warm-up done; embedding:", paste(dim(ref), collapse = " x "), "\n\n")

# ---- timed loop ------------------------------------------------------------
times <- numeric(iters)
max_drift <- 0
t0 <- Sys.time()
for (i in seq_len(iters)) {
  t <- system.time(
    obj <- RunGGML(obj, op = "embed", n_components = k, device = "vulkan")
  )[["elapsed"]]
  times[i] <- t

  # stability: |cor| of each PC against the warm-up run should stay ~1
  emb  <- SeuratObject::Embeddings(obj, "ggml")
  drift <- 1 - min(vapply(seq_len(k),
                          function(j) abs(cor(emb[, j], ref[, j])), numeric(1)))
  max_drift <- max(max_drift, drift)

  cat(sprintf("  iter %2d/%d  %6.3f s   max(1-|cor|)=%.2e\n",
              i, iters, t, max_drift))
}
wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

cat(sprintf("\nTotal wall: %.1f s over %d iters\n", wall, iters))
cat(sprintf("Per iter: mean %.3f s  min %.3f s  max %.3f s\n",
            mean(times), min(times), max(times)))
cat(sprintf("Backend: %s   max numeric drift over run: %.2e\n",
            Misc(obj, slot = "ggml_ggml")$backend, max_drift))
if (max_drift > 1e-2)
  cat("WARNING: embedding drifted across runs — check scheduler aliasing.\n")
cat("\nDone.\n")
