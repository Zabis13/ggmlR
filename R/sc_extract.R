# Single-cell adapter: extraction layer --------------------------------------
#
# One generic, `ggml_extract()`, turns any supported container into a *dense*
# numeric feature-by-cell matrix ready for the engine. The Seurat method hides
# the v4 vs v5 API split. Sparse input is materialised to dense only here, and
# only for the requested genes/cells, so the rest of the pipeline never sees a
# dgCMatrix.

#' Extract a feature-by-cell matrix from a single-cell container
#'
#' Pulls an expression matrix out of a Seurat object, a
#' \code{SingleCellExperiment}, a sparse \code{dgCMatrix} or a plain
#' \code{matrix}, returning a dense numeric matrix with features in rows and
#' cells in columns — the layout the GPU engine expects. Optional
#' \code{genes}/\code{cells} subsetting happens before any sparse-to-dense
#' materialisation.
#'
#' @param x A \code{Seurat}, \code{SingleCellExperiment}, \code{dgCMatrix} or
#'   \code{matrix} object.
#' @param assay Assay to read from. Seurat: defaults to the object's default
#'   assay. Ignored for bare matrices.
#' @param layer Layer / slot to read. Seurat v5: a layer name (default
#'   \code{"data"}); Seurat v4: mapped to the \code{slot} argument of
#'   \code{GetAssayData}. Ignored for bare matrices.
#' @param genes Optional character/integer vector selecting feature rows.
#' @param cells Optional character/integer vector selecting cell columns.
#' @param ... Passed to methods.
#'
#' @return A dense numeric \code{matrix}, features x cells.
#' @export
ggml_extract <- function(x, assay = NULL, layer = "data",
                         genes = NULL, cells = NULL, ...) {
  UseMethod("ggml_extract")
}

#' @rdname ggml_extract
#' @export
ggml_extract.matrix <- function(x, assay = NULL, layer = "data",
                                genes = NULL, cells = NULL, ...) {
  x <- .ggmlr_subset_mat(x, genes, cells)
  storage.mode(x) <- "double"
  x
}

#' @rdname ggml_extract
#' @export
ggml_extract.dgCMatrix <- function(x, assay = NULL, layer = "data",
                                   genes = NULL, cells = NULL, ...) {
  x <- .ggmlr_subset_mat(x, genes, cells)
  # materialise to dense only now, only for the retained submatrix
  as.matrix(x)
}

#' @rdname ggml_extract
#' @export
ggml_extract.Seurat <- function(x, assay = NULL, layer = "data",
                                genes = NULL, cells = NULL, ...) {
  .ggmlr_need_pkg("SeuratObject", "extracting data from a Seurat object")
  assay <- assay %||% SeuratObject::DefaultAssay(x)

  if (.ggmlr_object_is_v5(x, assay)) {
    # Seurat v5: Assay5 layer model
    mat <- SeuratObject::LayerData(x, assay = assay, layer = layer)
  } else {
    # Seurat v4 / legacy Assay: single-slot. SeuratObject >= 5 made the
    # `slot` argument of GetAssayData() defunct, so address it via `layer`.
    mat <- SeuratObject::GetAssayData(x, assay = assay, layer = layer)
  }

  # mat is genes x cells (Seurat convention), possibly sparse -> reuse methods
  ggml_extract(mat, genes = genes, cells = cells)
}

# internal: subset a (dense or sparse) genes x cells matrix by name or index
.ggmlr_subset_mat <- function(mat, genes, cells) {
  if (!is.null(genes)) mat <- mat[genes, , drop = FALSE]
  if (!is.null(cells)) mat <- mat[, cells, drop = FALSE]
  mat
}
