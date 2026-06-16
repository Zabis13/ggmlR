# Single-cell adapter: result injection layer --------------------------------
#
# `ggml_inject()` writes a ggml_result back into the standard reduction slot of
# the container it came from, so downstream tools (UMAP, clustering, plotting)
# see it as an ordinary dimensionality reduction. GPU/run metadata is stashed in
# the object's misc slot for provenance.

#' Inject a single-cell result back into its container
#'
#' Writes the embedding from a \code{\link{ggml_result}} into the standard
#' dimensionality-reduction slot of a Seurat object (a
#' \code{SingleCellExperiment} method is added in a later release), returning the
#' updated object. Component standard deviations and the backend used are
#' recorded alongside so downstream tools and the user can see how the
#' reduction was produced.
#'
#' @param x A \code{Seurat} object (the one the data was extracted from).
#' @param result A \code{\link{ggml_result}}.
#' @param reduction_name Name of the reduction slot to create, e.g.
#'   \code{"ggml"} (default). For Seurat this becomes \code{x[["ggml"]]}.
#' @param key Column-name prefix for the embedding, e.g. \code{"GGML_"}.
#' @param assay Assay to associate the reduction with (Seurat). Defaults to the
#'   object's default assay.
#' @param ... Passed to methods.
#'
#' @return The updated container.
#' @export
ggml_inject <- function(x, result, reduction_name = "ggml", key = "GGML_",
                        assay = NULL, ...) {
  UseMethod("ggml_inject")
}

#' @rdname ggml_inject
#' @importFrom methods new
#' @export
ggml_inject.Seurat <- function(x, result, reduction_name = "ggml", key = "GGML_",
                               assay = NULL, ...) {
  .ggmlr_need_pkg("SeuratObject", "writing a reduction into a Seurat object")
  if (!inherits(result, "ggml_result"))
    stop("`result` must be a ggml_result.", call. = FALSE)

  assay <- assay %||% SeuratObject::DefaultAssay(x)

  emb <- result$embedding
  colnames(emb) <- paste0(key, seq_len(ncol(emb)))

  dr <- SeuratObject::CreateDimReducObject(
    embeddings = emb,
    loadings   = result$metadata$loadings %||% new(Class = "matrix"),
    stdev      = as.numeric(result$metadata$stdev %||% numeric(0)),
    key        = key,
    assay      = assay
  )
  x[[reduction_name]] <- dr

  # provenance: record backend + timings in the object's misc slot
  SeuratObject::Misc(x, slot = paste0(reduction_name, "_ggml")) <- list(
    backend = result$metadata$backend,
    timings = result$timings
  )
  x
}
