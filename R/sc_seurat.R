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
#' fallback) and writes the result back as a dimensionality reduction. Returns
#' the updated Seurat object, so it slots into a \code{\%>\%} / \code{|>}
#' pipeline next to \code{Seurat::RunPCA()}.
#'
#' The first supported operation is \code{"embed"} (PCA): the gene-by-gene
#' covariance multiply runs on the GPU, the eigendecomposition on the CPU.
#'
#' @param object A \code{Seurat} object, or a bare feature-by-cell
#'   \code{matrix}/\code{dgCMatrix} (the \code{.default} method returns a
#'   \code{\link{ggml_result}} instead of an object).
#' @param op Operation name; see \code{\link{ggml_ops_registry}}. Default
#'   \code{"embed"}.
#' @param assay Assay to read (Seurat); defaults to the object's default assay.
#' @param layer Layer/slot to read; default \code{"data"}.
#' @param n_components Number of components for \code{"embed"} (PCA). Default 50.
#' @param reduction_name Name of the reduction slot to create. Default
#'   \code{"ggml"}.
#' @param device \code{"auto"} (default), \code{"vulkan"} or \code{"cpu"}.
#' @param genes,cells Optional feature/cell subsets passed to extraction.
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
RunGGML <- function(object, op = "embed", assay = NULL, layer = "data",
                    n_components = 50L, reduction_name = "ggml",
                    device = "auto", genes = NULL, cells = NULL, ...) {
  UseMethod("RunGGML")
}

#' @rdname RunGGML
#' @export
RunGGML.default <- function(object, op = "embed", assay = NULL, layer = "data",
                            n_components = 50L, reduction_name = "ggml",
                            device = "auto", genes = NULL, cells = NULL, ...) {
  mat  <- ggml_extract(object, assay = assay, layer = layer,
                       genes = genes, cells = cells)
  task <- ggml_task(op, mat,
                    params = c(list(n_components = as.integer(n_components)), list(...)),
                    device = device)
  ggml_run(task)
}

#' @rdname RunGGML
#' @export
RunGGML.Seurat <- function(object, op = "embed", assay = NULL, layer = "data",
                           n_components = 50L, reduction_name = "ggml",
                           device = "auto", genes = NULL, cells = NULL, ...) {
  .ggmlr_need_pkg("SeuratObject", "RunGGML on a Seurat object")
  assay  <- assay %||% SeuratObject::DefaultAssay(object)

  mat    <- ggml_extract(object, assay = assay, layer = layer,
                         genes = genes, cells = cells)
  task   <- ggml_task(op, mat,
                      params = c(list(n_components = as.integer(n_components)), list(...)),
                      device = device)
  result <- ggml_run(task)

  ggml_inject(object, result, reduction_name = reduction_name, assay = assay)
}
