# Single-cell adapter: high-level Seurat entry point -------------------------
#
# RunGGML() is the one call a Seurat user makes. It chains the three layers —
# extract -> run -> inject — in the Seurat house style (object in, object out,
# pipe-friendly), mirroring RunPCA()/RunUMAP(). A .default method on a bare
# matrix returns the raw ggml_result, which is handy for testing the whole
# pipeline without Seurat installed.

#' Run a GGML GPU operation on a Seurat object
#'
#' High-level, Seurat-style entry point: extracts the expression matrix from the
#' object, runs the requested operation on the GGML backend (Vulkan GPU with CPU
#' fallback) and writes the result back into the object. Returns the updated
#' object, so it slots into a \code{\%>\%} / \code{|>} pipeline next to
#' \code{Seurat::RunPCA()}. Methods exist for \code{Seurat} and
#' \code{SingleCellExperiment} objects (and a \code{.default} for bare matrices).
#'
#' Supported operations include \code{"embed"} (PCA), \code{"normalize"},
#' \code{"scale"}, \code{"umap"}, \code{"neighbors"} and \code{"largest_gene"}
#' (per-cell highest-expressed gene QC metric); see
#' \code{\link{ggml_ops_registry}}.
#'
#' @param object A \code{Seurat} or \code{SingleCellExperiment} object, or a bare
#'   feature-by-cell \code{matrix}/\code{dgCMatrix} (the \code{.default} method
#'   returns a \code{\link{ggml_result}} instead of an object).
#' @param op Operation name; see \code{\link{ggml_ops_registry}}. Default
#'   \code{"embed"}.
#' @param assay Assay to read (Seurat); defaults to the object's default assay.
#' @param layer Layer/slot to read; default \code{"data"}.
#' @param n_components Number of components for \code{"embed"} (PCA). Default 50.
#' @param reduction_name Name of the reduction slot to create. Default
#'   \code{"ggml"}.
#' @param device \code{"auto"} (default), \code{"vulkan"} or \code{"cpu"}.
#' @param genes,cells Optional feature/cell subsets passed to extraction.
#' @param reduction Optional name of an existing reduction to use as the input
#'   feature space (e.g. \code{"pca"} for \code{"umap"} / \code{"neighbors"}),
#'   instead of an assay layer. Seurat: \code{Embeddings()}; SCE:
#'   \code{reducedDim()}.
#' @param dims Optional integer vector selecting columns of \code{reduction}.
#' @param ... Additional parameters forwarded to the engine. A notable one is
#'   \code{chunk_size}: for \code{op = "scale"} and \code{op = "embed"} (PCA),
#'   passing an integer streams the sparse input in blocks of that many cells,
#'   densifying one block at a time so the full dense features-by-cells matrix
#'   (tens of GB at scale) is never held. Results are identical to the
#'   non-chunked path. \code{op = "normalize"} is already sparse, so
#'   \code{chunk_size} is a no-op there.
#'
#' @return For a Seurat object, the updated object with a new reduction. For a
#'   bare matrix, a \code{\link{ggml_result}}.
#'
#' @examples
#' \dontrun{
#' library(Seurat)
#' pbmc <- RunGGML(pbmc, op = "embed", n_components = 30)
#' DimPlot(pbmc, reduction = "ggml")
#' }
#' @export
RunGGML <- function(object, op = "embed", assay = NULL, layer = NULL,
                    n_components = 50L, reduction_name = "ggml",
                    device = "auto", genes = NULL, cells = NULL,
                    reduction = NULL, dims = NULL, ...) {
  UseMethod("RunGGML")
}

# default input layer per op: normalize and largest_gene read raw counts,
# everything else reads the (log-)normalised data layer.
.ggmlr_default_layer <- function(op)
  if (op %in% c("normalize", "largest_gene")) "counts" else "data"

# build the params list for an op: n_components only matters to "embed".
.ggmlr_op_params <- function(op, n_components, extra) {
  base <- if (identical(op, "embed"))
    list(n_components = as.integer(n_components)) else list()
  c(base, extra)
}

#' @rdname RunGGML
#' @export
RunGGML.default <- function(object, op = "embed", assay = NULL, layer = NULL,
                            n_components = 50L, reduction_name = "ggml",
                            device = "auto", genes = NULL, cells = NULL, ...) {
  layer <- layer %||% .ggmlr_default_layer(op)
  # keep a sparse input sparse when the op is sparse-aware or a chunk_size was
  # requested, so the engine can stream cell-blocks rather than densify up front.
  entry <- ggml_ops_registry(op)
  keep_sparse <- isTRUE(entry$sparse_ok) || !is.null(list(...)$chunk_size)
  mat  <- ggml_extract(object, assay = assay, layer = layer,
                       genes = genes, cells = cells, keep_sparse = keep_sparse)
  task <- ggml_task(op, mat,
                    params = .ggmlr_op_params(op, n_components, list(...)),
                    device = device)
  ggml_run(task)
}

#' @rdname RunGGML
#' @export
RunGGML.Seurat <- function(object, op = "embed", assay = NULL, layer = NULL,
                           n_components = 50L, reduction_name = "ggml",
                           device = "auto", genes = NULL, cells = NULL,
                           reduction = NULL, dims = NULL, ...) {
  .ggmlr_need_pkg("SeuratObject", "RunGGML on a Seurat object")
  assay  <- assay %||% SeuratObject::DefaultAssay(object)

  if (!is.null(reduction)) {
    # Build from an existing reduction (e.g. UMAP from PCA). Embeddings are
    # cells x dims; the engines expect features x cells, so transpose.
    emb <- SeuratObject::Embeddings(object, reduction = reduction)
    if (!is.null(dims)) emb <- emb[, dims, drop = FALSE]
    mat <- t(emb)
  } else {
    layer <- layer %||% .ggmlr_default_layer(op)
    # normalize keeps the counts sparse all the way to the engine (log1p(0)=0),
    # so the dense matrix — tens of GB on the full dataset — is never formed. A
    # chunk_size request likewise needs the matrix left sparse, so the engine can
    # densify one cell-block at a time instead of the whole thing up front.
    entry <- ggml_ops_registry(op)
    keep_sparse <- isTRUE(entry$sparse_ok) || !is.null(list(...)$chunk_size)
    mat   <- ggml_extract(object, assay = assay, layer = layer,
                          genes = genes, cells = cells,
                          keep_sparse = keep_sparse)
  }
  task   <- ggml_task(op, mat,
                      params = .ggmlr_op_params(op, n_components, list(...)),
                      device = device)
  result <- ggml_run(task)

  # key prefix for the embedding columns: keep "GGML_" for the historical
  # "ggml" reduction, otherwise derive it from the reduction name (e.g.
  # reduction_name = "umap" -> key "umap_"), matching Seurat conventions.
  key <- if (identical(reduction_name, "ggml")) "GGML_"
         else paste0(reduction_name, "_")
  out <- ggml_inject(object, result, reduction_name = reduction_name, key = key,
                     assay = assay)
  # Expose the engine's per-step timings (e.g. PCA centre/matmul/eigen breakdown)
  # for profiling, without disturbing the returned Seurat object's contract.
  attr(out, "ggml_timings") <- result$timings
  out
}
