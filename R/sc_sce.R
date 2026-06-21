# Single-cell adapter: SingleCellExperiment (Bioconductor / S4) methods --------
#
# The Bioconductor counterpart to the Seurat methods in sc_extract.R /
# sc_inject.R / sc_seurat.R. SingleCellExperiment is an S4 object, but ggmlR's
# adapter generics are S3; S3 dispatch keys on class(x), which for an SCE is
# "SingleCellExperiment", so RunGGML / ggml_extract / ggml_inject reach these
# methods without an S4 generic. The engines are unchanged — only the
# container-specific extract/inject differs from Seurat.
#
# SingleCellExperiment / SummarizedExperiment / S4Vectors stay in Suggests; these
# methods guard on them and never load at package init, so ggmlR installs and
# runs without Bioconductor present.

#' @rdname ggml_extract
#' @details For a \code{SingleCellExperiment} the \code{layer} argument names an
#'   assay (default \code{"logcounts"}, the log-normalised matrix); it is read
#'   with \code{SummarizedExperiment::assay()}.
#' @export
ggml_extract.SingleCellExperiment <- function(x, assay = NULL, layer = "logcounts",
                                              genes = NULL, cells = NULL, ...) {
  .ggmlr_need_pkg("SummarizedExperiment", "extracting data from a SingleCellExperiment")
  # `assay` is the Seurat term; for SCE the assay is named by `layer`. Accept
  # either, preferring an explicit `assay` if the caller passed one.
  which <- assay %||% layer
  mat <- SummarizedExperiment::assay(x, i = which)
  # SCE assays are genes x cells, possibly sparse -> reuse the matrix methods
  ggml_extract(mat, genes = genes, cells = cells)
}

#' @rdname ggml_inject
#' @export
ggml_inject.SingleCellExperiment <- function(x, result, reduction_name = "ggml",
                                             key = "GGML_", assay = NULL, ...) {
  .ggmlr_need_pkg("SingleCellExperiment",
                  "writing a result into a SingleCellExperiment")
  if (!inherits(result, "ggml_result"))
    stop("`result` must be a ggml_result.", call. = FALSE)

  # provenance kept in metadata(sce), minus the bulky payloads already stored in
  # the reducedDim / assay / colData they came from.
  meta_prov <- function(meta, timings) {
    bulky <- c("backend", "loadings", "stdev", "nn", "snn")
    extra <- meta[setdiff(names(meta), bulky)]
    c(list(backend = meta$backend, timings = timings), extra)
  }
  set_meta <- function(x, slot, value) {
    md <- S4Vectors::metadata(x)
    md[[slot]] <- value
    S4Vectors::metadata(x) <- md
    x
  }

  kind <- result$metadata$kind

  # transform ops (normalize / scale) overwrite a named assay. The engines tag
  # the result with a Seurat layer name ("data" / "scale.data"); map those to the
  # SingleCellExperiment assay convention ("logcounts" / "scaledata").
  if (identical(kind, "transform")) {
    which <- switch(result$metadata$layer %||% "logcounts",
                    "data"       = "logcounts",
                    "scale.data" = "scaledata",
                    result$metadata$layer)
    SummarizedExperiment::assay(x, i = which) <- result$embedding
    x <- set_meta(x, paste0(which, "_ggml"),
                  meta_prov(result$metadata, result$timings))
    return(x)
  }

  # graph ops (neighbors): SCE has no @graphs slot, so the kNN/SNN graphs go into
  # metadata() under <reduction_name>_nn / _snn, alongside the provenance.
  if (identical(kind, "graph")) {
    x <- set_meta(x, paste0(reduction_name, "_nn"),  result$metadata$nn)
    x <- set_meta(x, paste0(reduction_name, "_snn"), result$metadata$snn)
    x <- set_meta(x, paste0(reduction_name, "_ggml"),
                  meta_prov(result$metadata, result$timings))
    return(x)
  }

  # default: a dimensionality reduction -> reducedDim(). SCE stores embeddings
  # cells x components, the same layout ggml_result uses.
  emb <- result$embedding
  colnames(emb) <- paste0(key, seq_len(ncol(emb)))
  SingleCellExperiment::reducedDim(x, type = reduction_name) <- emb
  x <- set_meta(x, paste0(reduction_name, "_ggml"),
                meta_prov(result$metadata, result$timings))
  x
}

#' @rdname RunGGML
#' @export
RunGGML.SingleCellExperiment <- function(object, op = "embed", assay = NULL,
                                         layer = NULL, n_components = 50L,
                                         reduction_name = "ggml", device = "auto",
                                         genes = NULL, cells = NULL,
                                         reduction = NULL, dims = NULL, ...) {
  .ggmlr_need_pkg("SingleCellExperiment", "RunGGML on a SingleCellExperiment")

  if (!is.null(reduction)) {
    # Build from an existing reducedDim (e.g. UMAP / neighbours from PCA).
    # reducedDim is cells x dims; the engines want features x cells -> transpose.
    emb <- SingleCellExperiment::reducedDim(object, type = reduction)
    if (!is.null(dims)) emb <- emb[, dims, drop = FALSE]
    mat <- t(emb)
  } else {
    # default assay per op: SCE keeps raw counts in "counts" and the
    # log-normalised matrix in "logcounts".
    layer <- layer %||% if (identical(op, "normalize")) "counts" else "logcounts"
    mat   <- ggml_extract(object, assay = assay, layer = layer,
                          genes = genes, cells = cells)
  }
  task   <- ggml_task(op, mat,
                      params = .ggmlr_op_params(op, n_components, list(...)),
                      device = device)
  result <- ggml_run(task)

  key <- if (identical(reduction_name, "ggml")) "GGML_"
         else paste0(reduction_name, "_")
  ggml_inject(object, result, reduction_name = reduction_name, key = key,
              assay = assay)
}
