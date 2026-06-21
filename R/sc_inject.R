# Single-cell adapter: result injection layer --------------------------------
#
# `ggml_inject()` writes a ggml_result back into the standard reduction slot of
# the container it came from, so downstream tools (UMAP, clustering, plotting)
# see it as an ordinary dimensionality reduction. GPU/run metadata is stashed in
# the object's misc slot for provenance.

#' Inject a single-cell result back into its container
#'
#' Writes a \code{\link{ggml_result}} back into the appropriate slot of a Seurat
#' or \code{SingleCellExperiment} object, returning the updated object. The
#' destination depends on
#' \code{result$metadata$kind}: a dimensionality reduction (the default) goes into
#' a \code{DimReduc}; a \code{"transform"} (normalize / scale) overwrites an assay
#' layer; a \code{"graph"} (neighbors) writes \code{<assay>_nn} and
#' \code{<assay>_snn} \code{Graph} objects into \code{@graphs}, exactly where
#' Seurat's \code{FindNeighbors} puts them so \code{FindClusters} can consume them.
#' Component standard deviations and the backend used are recorded alongside so
#' downstream tools and the user can see how the result was produced.
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
#' @importFrom methods new as
#' @export
ggml_inject.Seurat <- function(x, result, reduction_name = "ggml", key = "GGML_",
                               assay = NULL, ...) {
  .ggmlr_need_pkg("SeuratObject", "writing a reduction into a Seurat object")
  if (!inherits(result, "ggml_result"))
    stop("`result` must be a ggml_result.", call. = FALSE)

  assay <- assay %||% SeuratObject::DefaultAssay(x)

  # provenance kept in Misc: backend + timings, plus any op-specific metadata
  # (e.g. UMAP a/b curve params and n_edges). The bulky DimReduc payload
  # (loadings/stdev) already lives in the reduction object, so drop it here.
  .ggmlr_misc_provenance <- function(meta, timings) {
    bulky <- c("backend", "loadings", "stdev", "nn", "snn")
    extra <- meta[setdiff(names(meta), bulky)]
    c(list(backend = meta$backend, timings = timings), extra)
  }

  # graph ops (neighbors) write two Graph objects into the @graphs slot, exactly
  # as Seurat's FindNeighbors does: <assay>_nn (binary kNN) and <assay>_snn
  # (weighted shared-NN). Downstream FindClusters reads <assay>_snn.
  if (identical(result$metadata$kind, "graph")) {
    nn  <- result$metadata$nn
    snn <- result$metadata$snn

    to_graph <- function(m) {
      g <- SeuratObject::as.Graph(methods::as(m, "CsparseMatrix"))
      SeuratObject::DefaultAssay(g) <- assay
      g
    }
    nn_name  <- paste0(assay, "_nn")
    snn_name <- paste0(assay, "_snn")
    x[[nn_name]]  <- to_graph(nn)
    x[[snn_name]] <- to_graph(snn)

    SeuratObject::Misc(x, slot = paste0(reduction_name, "_ggml")) <-
      .ggmlr_misc_provenance(result$metadata, result$timings)
    return(x)
  }

  # transform ops (normalize / scale) write a feature-by-cell matrix back into
  # an assay layer rather than creating a DimReduc.
  if (identical(result$metadata$kind, "transform")) {
    layer <- result$metadata$layer %||% "data"
    mat   <- result$embedding
    if (.ggmlr_object_is_v5(x, assay)) {
      SeuratObject::LayerData(x, assay = assay, layer = layer) <- mat
    } else {
      x <- SeuratObject::SetAssayData(x, assay = assay, layer = layer,
                                      new.data = mat)
    }
    SeuratObject::Misc(x, slot = paste0(layer, "_ggml")) <-
      .ggmlr_misc_provenance(result$metadata, result$timings)
    return(x)
  }

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

  # provenance: backend + timings + op-specific metadata in the misc slot
  SeuratObject::Misc(x, slot = paste0(reduction_name, "_ggml")) <-
    .ggmlr_misc_provenance(result$metadata, result$timings)
  x
}
