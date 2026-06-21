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
#' \code{"scale"}, \code{"umap"} and \code{"neighbors"}; see
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
#' @param ... Additional parameters forwarded to the engine.
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

# default input layer per op: normalize reads raw counts, everything else reads
# the (log-)normalised data layer.
.ggmlr_default_layer <- function(op) if (identical(op, "normalize")) "counts" else "data"

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
  mat  <- ggml_extract(object, assay = assay, layer = layer,
                       genes = genes, cells = cells)
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
    mat   <- ggml_extract(object, assay = assay, layer = layer,
                          genes = genes, cells = cells)
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
  ggml_inject(object, result, reduction_name = reduction_name, key = key,
              assay = assay)
}
