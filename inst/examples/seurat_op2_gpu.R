#!/usr/bin/env Rscript
# ============================================================================
# Seurat integration — GPU speed-up of the OP2 single-cell pipeline
# ============================================================================
# Benchmarks the standard Seurat preprocessing pipeline step by step, once on
# the CPU (stock Seurat) and once on the Vulkan GPU (ggmlR's RunGGML), and
# checks that the two agree numerically. Every GPU step is one of the five
# operations ggmlR registers: normalize / scale / embed / neighbors / umap.
#
# Dataset: "Open Problems - Single-Cell Perturbations" (Kaggle). 18211 genes x
# 240090 human PBMCs, 24 h after treatment with 144 compounds plus controls.
# Counts live in sparse_count.qs2 (a dgCMatrix), per-cell metadata in
# seurat_meta.qs2. Both are read with qs2; the original .qs files were written
# by the now-archived qs package and must be converted once with qs::qread().
#
# Pipeline (this is the classic Seurat route, NOT SCTransform):
#
#   percent.Largest.Gene -> op = "largest_gene" CPU (sparse argmax; parity)
#   NormalizeData        -> op = "normalize"    GPU
#   FindVariableFeatures -> (no GPU op yet)     CPU in both arms
#   ScaleData            -> op = "scale"        CPU (memory-bound; GPU opt-in)
#   RunPCA               -> op = "embed"        GPU
#   FindNeighbors        -> op = "neighbors"    GPU (kNN CPU kd-tree; --gpu-knn)
#   FindClusters         -> (graph algorithm)   CPU in both arms
#   RunUMAP              -> op = "umap"         GPU graph + CPU layout
#
# The notebook this mirrors normalises with SCTransform, which is a per-gene
# negative-binomial GLM and has no GPU path here; the classic route above is
# the accelerable equivalent. percent.Largest.Gene now has an op
# ("largest_gene"): a memory-bound sparse column argmax kept on the CPU (over the
# dgCMatrix @x, no densify), matching qlcMatrix::colMax without its overhead. One
# heavy notebook step still lacks an op and is skipped: FindAllMarkers (tracked
# in TODO.md).
#
# Memory: the full 240090-cell matrix densifies to ~35 GB, so — exactly as the
# notebook does — the pipeline runs on a random subsample of the cells. Scaling
# and everything after it run on the variable features only, as in Seurat.
#
# Usage:
#   Rscript seurat_op2_gpu.R
#   Rscript seurat_op2_gpu.R --frac 0.05          # smaller subsample
#   Rscript seurat_op2_gpu.R --data /path/to/adata
#   Rscript seurat_op2_gpu.R --no-cpu             # GPU arm only (skip timings)
#   Rscript seurat_op2_gpu.R --chunk 20000        # stream scale/PCA in blocks
#   Rscript seurat_op2_gpu.R --gpu-knn            # op="neighbors" kNN on the GPU
#
# Requires (Suggests, install separately): Seurat, SeuratObject, Matrix, qs2,
#   data.table, FNN, uwot
# ============================================================================

suppressMessages({
  library(ggmlR)
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
})

# ---- configuration ---------------------------------------------------------

DATA_DIR  <- "/mnt/Data2/DS_projects/Seurat data/archive/adata"
FRAC      <- 0.1      # fraction of cells kept, as in the notebook
N_HVG     <- 2000L    # variable features fed to scale/PCA
N_PCS     <- 50L      # principal components
DIMS      <- 1:10     # PCs used for neighbours and UMAP, as in the notebook
K_NEIGH   <- 20L      # FindNeighbors default
SEED      <- 1L

args <- commandArgs(trailingOnly = TRUE)
arg_of <- function(flag, default) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1L]
}
FRAC     <- as.numeric(arg_of("--frac", FRAC))
DATA_DIR <- arg_of("--data", DATA_DIR)
RUN_CPU  <- !("--no-cpu" %in% args)
# --chunk N streams scale/PCA in blocks of N cells so the dense features-by-cells
# matrix is never held whole (for datasets whose dense form exceeds memory).
# NULL keeps the original single-shot path.
CHUNK    <- { v <- arg_of("--chunk", NA); if (is.na(v)) NULL else as.integer(v) }
# --gpu-knn opts op = "neighbors" into the fused GPU kNN (knn_tiled.comp) instead
# of the FNN kd-tree. The kd-tree degrades to ~O(n^2) in ~10-D at large n, which
# the GPU brute-force with per-row top-k avoids; this flag is here to A/B the two.
GPU_KNN  <- "--gpu-knn" %in% args

for (p in c("qs2", "data.table")) {
  if (!requireNamespace(p, quietly = TRUE))
    stop("package '", p, "' is required by this example; install it first.",
         call. = FALSE)
}

have_gpu <- isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE))
cat("Vulkan GPU:", if (have_gpu) "yes" else "no (ops fall back to CPU)", "\n")
cat(sprintf("Subsample: %.0f%% of cells   HVG: %d   PCs: %d\n\n",
            100 * FRAC, N_HVG, N_PCS))

# `timed` runs an expression, prints its elapsed time under a label, and
# records it so the summary table at the end can compare the two arms.
TIMINGS <- new.env(parent = emptyenv())
timed <- function(arm, step, expr) {
  t <- system.time(value <- force(expr))[["elapsed"]]
  TIMINGS[[paste(arm, step)]] <- t
  cat(sprintf("  %-4s %-22s %8.2f s\n", arm, step, t))
  value
}

# ---- 1. load the counts and metadata ---------------------------------------

counts_file <- file.path(DATA_DIR, "sparse_count.qs2")
meta_file   <- file.path(DATA_DIR, "seurat_meta.qs2")
if (!file.exists(counts_file)) {
  stop("counts not found: ", counts_file, "\n",
       "Convert the original qs files once (the qs package is archived on ",
       "CRAN; install stringfish 0.18.0 first, then qs 0.27.3 from the ",
       "CRAN archive):\n",
       '  x <- qs::qread("sparse_count.qs"); qs2::qs_save(x, "sparse_count.qs2")',
       call. = FALSE)
}

cat("Loading data\n")
counts <- timed("io", "read counts", qs2::qs_read(counts_file))
meta   <- as.data.frame(timed("io", "read metadata", qs2::qs_read(meta_file)))
cat(sprintf("  %d genes x %d cells, %.0f%% non-zero\n\n",
            nrow(counts), ncol(counts),
            100 * length(counts@x) / (as.numeric(nrow(counts)) * ncol(counts))))

# The metadata rows are in the same order as the count columns, but carry no
# cell IDs; give them the barcodes so CreateSeuratObject can match them.
stopifnot(nrow(meta) == ncol(counts))
rownames(meta) <- colnames(counts)

# `control` is not stored in seurat_meta.qs2 but is recoverable: the negative
# control is the solvent, DMSO.
meta$control <- ifelse(meta$sm_name == "Dimethyl Sulfoxide", "negative", "other")

# ---- 2. subset: drop negative controls, then subsample ---------------------
#
# The competition's differential-expression targets are fold changes relative
# to the DMSO control, so the notebook drops those cells before looking for
# structure in the treated ones.

keep <- which(meta$control != "negative")
set.seed(SEED)
keep <- sort(sample(keep, floor(FRAC * length(keep))))

counts <- counts[, keep, drop = FALSE]
meta   <- meta[keep, , drop = FALSE]
cat(sprintf("After dropping DMSO and subsampling: %d genes x %d cells\n\n",
            nrow(counts), ncol(counts)))

make_obj <- function() suppressWarnings(
  CreateSeuratObject(counts = counts, project = "OP2",
                     min.cells = 1, min.features = 1, meta.data = meta))

# ============================================================================
# CPU arm — stock Seurat
# ============================================================================

hvg <- NULL   # variable features, chosen once on the CPU and reused by both arms

if (RUN_CPU) {
  cat("CPU arm (stock Seurat)\n")
  cpu <- make_obj()
  # percent.Largest.Gene QC metric: for each cell, the highest-expressed gene and
  # its share of the cell's total counts. Stock Seurat uses qlcMatrix::colMax on
  # the raw counts (its argmax path); fall back to a base-R column argmax when
  # qlcMatrix is absent so the arm still runs.
  cpu <- timed("cpu", "percent.Largest.Gene", {
    cnt <- GetAssayData(cpu, assay = DefaultAssay(cpu), layer = "counts")
    if (requireNamespace("qlcMatrix", quietly = TRUE)) {
      cmax <- qlcMatrix::colMax(cnt, which = TRUE)
      mx   <- as.numeric(cmax$max)
      idx  <- cmax$which@i + 1L               # row of the per-cell max
    } else {
      dm  <- as.matrix(cnt)
      idx <- max.col(t(dm), ties.method = "first")
      mx  <- dm[cbind(idx, seq_len(ncol(dm)))]
    }
    cs <- Matrix::colSums(cnt); cs[cs == 0] <- 1
    cpu$largest_gene         <- rownames(cnt)[idx]
    cpu$percent.Largest.Gene <- mx / cs * 100
    cpu
  })
  cpu <- timed("cpu", "NormalizeData",
               NormalizeData(cpu, verbose = FALSE))
  cpu <- timed("cpu", "FindVariableFeatures",
               FindVariableFeatures(cpu, nfeatures = N_HVG, verbose = FALSE))
  hvg <- VariableFeatures(cpu)
  cpu <- timed("cpu", "ScaleData",
               ScaleData(cpu, features = hvg, verbose = FALSE))
  cpu <- timed("cpu", "RunPCA",
               RunPCA(cpu, features = hvg, npcs = N_PCS, verbose = FALSE,
                      seed.use = SEED))
  cpu <- timed("cpu", "FindNeighbors",
               FindNeighbors(cpu, dims = DIMS, k.param = K_NEIGH,
                             verbose = FALSE))
  cpu <- timed("cpu", "FindClusters",
               FindClusters(cpu, resolution = 0.5, verbose = FALSE,
                            random.seed = SEED))
  cpu <- timed("cpu", "RunUMAP",
               RunUMAP(cpu, dims = DIMS, verbose = FALSE, seed.use = SEED))
  cat(sprintf("  clusters: %d\n\n", nlevels(cpu$seurat_clusters)))
}

# ============================================================================
# GPU arm — ggmlR RunGGML
# ============================================================================

cat("GPU arm (ggmlR RunGGML)\n")
gpu <- make_obj()

# percent.Largest.Gene: per-cell highest-expressed gene + its share of the cell
# total, read from the raw counts. This is a memory-bound sparse column argmax,
# so RunGGML keeps it on the CPU (over the dgCMatrix @x, no densify) — the point
# is parity with qlcMatrix::colMax without its overhead, not a GPU speed-up. The
# two columns land in meta.data, exactly where the QC metric lives.
gpu <- timed("gpu", "largest_gene",
             RunGGML(gpu, op = "largest_gene"))

# normalize: log1p(counts / colSums * 1e4), elementwise on the GPU. Writes the
# "data" layer, exactly where NormalizeData puts it.
gpu <- timed("gpu", "normalize",
             RunGGML(gpu, op = "normalize"))

# No GPU op for variable-feature selection yet (TODO.md, slice 1d), so both
# arms use the same CPU-chosen genes. When the CPU arm is skipped, pick them
# here so the rest of the pipeline still has a feature set.
if (is.null(hvg)) {
  gpu <- timed("gpu", "FindVariableFeatures (cpu)",
               FindVariableFeatures(gpu, nfeatures = N_HVG, verbose = FALSE))
  hvg <- VariableFeatures(gpu)
} else {
  VariableFeatures(gpu) <- hvg
}

# scale: z-score per gene, clamped at +/-10. Restricted to the variable
# features, as ScaleData is in the canonical pipeline. The z-score defaults to
# the CPU even under Vulkan (scale_backend = "cpu"): it is a memory-bound
# elementwise pass with nothing to accelerate, so the GPU's host<->VRAM copy
# makes it slower. Pass scale_backend = "vulkan" to force the GPU path.
gpu <- timed("gpu", "scale",
             RunGGML(gpu, op = "scale", genes = hvg, chunk_size = CHUNK))

# embed: PCA. The gene-by-gene covariance multiply runs on the GPU; the
# eigendecomposition of the small covariance runs on the CPU (ggml has no
# eigensolver). Reads the scaled layer, like RunPCA does.
gpu <- timed("gpu", "embed (PCA)",
             RunGGML(gpu, op = "embed", layer = "scale.data", genes = hvg,
                     n_components = N_PCS, reduction_name = "pca",
                     chunk_size = CHUNK))
# Per-step breakdown of the PCA engine (centre / covariance matmul / eigen /
# projection matmul), to see where the embed time actually goes.
pca_t <- attr(gpu, "ggml_timings")
if (!is.null(pca_t)) {
  parts <- c("centre", "matmul_cov", "eigen", "matmul_proj")
  parts <- parts[parts %in% names(pca_t)]
  cat("       PCA breakdown:",
      paste(sprintf("%s %.2fs", parts, pca_t[parts]), collapse = "  "), "\n")
}

# neighbors: kNN + shared-nearest-neighbour (Jaccard) graphs on the PC
# coordinates. Lands in obj@graphs as <assay>_nn / <assay>_snn, which is where
# FindClusters looks. --gpu-knn opts the kNN search into the fused GPU shader
# (knn_tiled.comp) instead of the FNN kd-tree; the SNN/Jaccard build is the same.
knn_backend <- if (GPU_KNN) "vulkan" else "cpu"
if (GPU_KNN) cat("       (kNN on GPU: knn_tiled.comp)\n")
gpu <- timed("gpu", "neighbors",
             RunGGML(gpu, op = "neighbors", reduction = "pca", dims = DIMS,
                     n_neighbors = K_NEIGH, knn_backend = knn_backend))
# Per-step breakdown of the neighbors engine (kNN search vs SNN/Jaccard build),
# to see whether the time is in the kNN (GPU shader / CPU kd-tree) or the sparse
# SNN maths, plus which backend the kNN actually ran on. The engine records the
# path taken ("vulkan"/"fnn"/"cpu") in the object's Misc provenance.
nn_t <- attr(gpu, "ggml_timings")
kb   <- tryCatch(SeuratObject::Misc(gpu, "ggml_ggml")$backend, error = function(e) NULL)
if (!is.null(nn_t)) {
  nparts <- c("knn", "snn")
  nparts <- nparts[nparts %in% names(nn_t)]
  cat("       neighbors breakdown:",
      paste(sprintf("%s %.2fs", nparts, nn_t[nparts]), collapse = "  "),
      if (!is.null(kb)) sprintf("  (kNN backend: %s)", kb) else "", "\n")
}

# FindClusters is a graph algorithm (Louvain); no GPU op. It runs straight off
# our SNN graph.
assay <- DefaultAssay(gpu)
gpu <- timed("gpu", "FindClusters (cpu)",
             FindClusters(gpu, graph.name = paste0(assay, "_snn"),
                          resolution = 0.5, verbose = FALSE,
                          random.seed = SEED))

# umap: the kNN/distance phase is GPU-eligible; the SGD layout phase defaults to
# the CPU reference. That default is deliberate — the GPU SGD shader is a
# massively parallel Hogwild optimiser, and at that many concurrent edge-threads
# the lock-free races systematically smear the clusters together (a known hard
# problem for async UMAP-SGD; RAPIDS cuML has the same open issue). The SGD over
# the sparse graph is cheap on the CPU at single-cell scale, so little speed is
# lost. Pass sgd_backend = "vulkan" to opt into the GPU shader (faster, lower
# embedding quality).
gpu <- timed("gpu", "umap",
             RunGGML(gpu, op = "umap", reduction = "pca", dims = DIMS,
                     reduction_name = "umap"))

# UMAP reports its two phases separately. The kNN search prefers the FNN
# kd-tree (exact and O(n log n)) over the GPU's brute-force pairwise distance
# shader, so backend_dist is usually "fnn"; backend_sgd is "cpu" by default.
prov <- Misc(gpu, slot = "umap_ggml")
cat(sprintf("  umap: knn %s, sgd %s, %d edges   clusters: %d\n\n",
            prov$backend_dist, prov$backend_sgd, prov$n_edges,
            nlevels(gpu$seurat_clusters)))

# ============================================================================
# Agreement between the two arms
# ============================================================================

if (RUN_CPU) {
  cat("Agreement CPU vs GPU\n")

  # The dense feature x cell layers are ~2.5 GB each at this scale, and the
  # comparisons below densify two copies at once -- enough to OOM the box on a
  # large run. The agreement statistics (max abs error, |cor|) are numerical
  # float32-vs-float64 noise spread evenly across cells, so a fixed random
  # sample of columns estimates them faithfully at a fraction of the memory.
  # Sample unconditionally (not gated on N) so there is a single code path at
  # every dataset size; seed it so the reported numbers are reproducible.
  N_AGREE_CELLS <- 5000L
  set.seed(20260711L)
  agree_cols <- if (ncol(gpu) > N_AGREE_CELLS)
    sort(sample.int(ncol(gpu), N_AGREE_CELLS)) else seq_len(ncol(gpu))

  # The normalised layer is genes x cells and dense on the GPU side; compare it
  # on the variable features only, which is all the rest of the pipeline
  # consumes anyway. Seurat keeps its own copy sparse.
  d_norm <- max(abs(as.matrix(LayerData(gpu, layer = "data")[hvg, agree_cols]) -
                    as.matrix(LayerData(cpu, layer = "data")[hvg, agree_cols])))
  cat(sprintf("  normalize   max abs err  %.2e   (%d features x %d sampled cells)\n",
              d_norm, length(hvg), length(agree_cols)))

  # largest_gene: the per-cell argmax must match exactly (same gene picked) and
  # the percentage to float noise. Report the fraction of cells whose top gene
  # agrees and the max percentage error.
  lg_agree <- mean(gpu$largest_gene == cpu$largest_gene, na.rm = TRUE)
  lg_err   <- max(abs(gpu$percent.Largest.Gene - cpu$percent.Largest.Gene))
  cat(sprintf("  largest_gene top-gene agree  %.4f   percent max abs err  %.2e\n",
              lg_agree, lg_err))

  # Both arms scaled the same feature set, but Seurat may order rows its own
  # way, so index the reference by name.
  s_gpu <- as.matrix(LayerData(gpu, layer = "scale.data")[, agree_cols])
  s_cpu <- as.matrix(LayerData(cpu, layer = "scale.data")[rownames(s_gpu), agree_cols])
  cat(sprintf("  scale       max abs err  %.2e\n", max(abs(s_gpu - s_cpu))))
  rm(s_gpu, s_cpu); invisible(gc())

  # PCs are eigenvectors: their sign is arbitrary. Compare |correlation| of
  # each component's cell loadings.
  pc_gpu <- Embeddings(gpu, "pca")[, DIMS, drop = FALSE]
  pc_cpu <- Embeddings(cpu, "pca")[, DIMS, drop = FALSE]
  pc_cor <- vapply(DIMS, function(i) abs(cor(pc_gpu[, i], pc_cpu[, i])), 0)
  cat(sprintf("  PCA         min |cor| over PC%s  %.4f\n",
              paste0(min(DIMS), "-", max(DIMS)), min(pc_cor)))

  # Clusterings are label-permuted; the adjusted Rand index is invariant.
  ari <- function(a, b) {
    tab <- table(a, b)
    cmb <- function(n) n * (n - 1) / 2
    s_ij <- sum(cmb(tab)); s_i <- sum(cmb(rowSums(tab)))
    s_j  <- sum(cmb(colSums(tab))); s_n <- cmb(length(a))
    exp <- s_i * s_j / s_n
    (s_ij - exp) / ((s_i + s_j) / 2 - exp)
  }
  cat(sprintf("  clusters    ARI  %.4f   (%d GPU vs %d CPU communities)\n",
              ari(gpu$seurat_clusters, cpu$seurat_clusters),
              nlevels(gpu$seurat_clusters), nlevels(cpu$seurat_clusters)))

  # UMAP is stochastic and initialised differently in each arm, so coordinates
  # never match. What must hold is that the embedding preserves the cluster
  # structure: measure how well the CPU clustering separates in GPU UMAP space.
  sep <- function(emb, lab) {
    ctr <- t(vapply(split(seq_len(nrow(emb)), lab),
                    function(i) colMeans(emb[i, , drop = FALSE]), numeric(2)))
    within <- sum(vapply(split(seq_len(nrow(emb)), lab), function(i)
      sum((emb[i, , drop = FALSE] - rep(colMeans(emb[i, , drop = FALSE]),
                                        each = length(i)))^2), 0))
    within / sum((emb - rep(colMeans(emb), each = nrow(emb)))^2)
  }
  cat(sprintf("  UMAP        within/total SS  gpu %.3f  cpu %.3f  (lower = tighter)\n\n",
              sep(Embeddings(gpu, "umap"), gpu$seurat_clusters),
              sep(Embeddings(cpu, "umap"), cpu$seurat_clusters)))
}

# ============================================================================
# Summary
# ============================================================================

steps <- c(largest_gene = "percent.Largest.Gene",
           normalize    = "NormalizeData",
           scale        = "ScaleData",
           `embed (PCA)` = "RunPCA",
           neighbors    = "FindNeighbors",
           umap         = "RunUMAP")

cat("Summary\n")
if (RUN_CPU) {
  cat(sprintf("  %-14s %10s %10s %9s\n", "step", "cpu (s)", "gpu (s)", "speedup"))
  tot_c <- tot_g <- 0
  for (g in names(steps)) {
    tc <- TIMINGS[[paste("cpu", steps[[g]])]]
    tg <- TIMINGS[[paste("gpu", g)]]
    tot_c <- tot_c + tc; tot_g <- tot_g + tg
    cat(sprintf("  %-14s %10.2f %10.2f %8.2fx\n", g, tc, tg, tc / tg))
  }
  cat(sprintf("  %-14s %10.2f %10.2f %8.2fx\n",
              "TOTAL (gpu ops)", tot_c, tot_g, tot_c / tot_g))
  cat("\n  Steps with no GPU op (identical in both arms):",
      "FindVariableFeatures, FindClusters\n")
} else {
  for (g in names(steps))
    cat(sprintf("  %-14s %10.2f s\n", g, TIMINGS[[paste("gpu", g)]]))
}

cat("\nDone.\n")

# Multi-device Vulkan teardown races R's own exit: R can unmap the Vulkan loader
# / Mesa ICD before the device destructors run, so the process hangs or segfaults
# *after* the results above are printed. Shut the GPU down explicitly while the
# loader is still mapped; hard = TRUE then _exit(0)s past all exit handlers for a
# guaranteed-clean finish. No-op when there is no GPU. Must be the last statement.
if (have_gpu) ggml_vulkan_shutdown(hard = TRUE)
