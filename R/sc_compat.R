# Single-cell adapter: package-detection guards ------------------------------
#
# The single-cell integration (Seurat / Bioconductor) lives *inside* ggmlR but
# must never hard-depend on Seurat, SeuratObject, SingleCellExperiment or any
# Bioconductor package. They are all in Suggests. Every entry point that touches
# one of those objects first calls a guard here. This keeps ggmlR installable
# and `R CMD check`-clean with none of them present.

# internal: is a package available without attaching it?
.ggmlr_has_pkg <- function(pkg) {
  isTRUE(requireNamespace(pkg, quietly = TRUE))
}

# internal: stop with a uniform, actionable message if a Suggests pkg is missing
.ggmlr_need_pkg <- function(pkg, what = NULL) {
  if (!.ggmlr_has_pkg(pkg)) {
    msg <- sprintf("Package '%s' is required%s but is not installed.",
                   pkg, if (is.null(what)) "" else paste0(" for ", what))
    stop(msg, call. = FALSE)
  }
  invisible(TRUE)
}

# internal: detect the Seurat object-model generation (v4 vs v5).
#
# Seurat v5 introduced the Assay5 layer model (LayerData / multiple layers per
# assay). v4 and earlier use the single-slot Assay model (GetAssayData with a
# `slot` argument). The extraction strategy differs, so we sniff the installed
# SeuratObject version rather than the object itself (more robust across the
# many Seurat object variants).
#
# Returns: 5L, 4L, or NA_integer_ when SeuratObject is absent.
.ggmlr_seurat_generation <- function() {
  if (!.ggmlr_has_pkg("SeuratObject")) return(NA_integer_)
  v <- tryCatch(utils::packageVersion("SeuratObject"), error = function(e) NULL)
  if (is.null(v)) return(NA_integer_)
  if (v >= "5.0.0") 5L else 4L
}

# internal: does this specific object use the v5 Assay5 layer model?
#
# More precise than the package-version sniff: an object created under v4 but
# loaded in a v5 install may still be a classic Assay. We check the actual
# assay class when we can.
.ggmlr_object_is_v5 <- function(object, assay = NULL) {
  if (!.ggmlr_has_pkg("SeuratObject")) return(FALSE)
  assay <- assay %||% tryCatch(SeuratObject::DefaultAssay(object),
                               error = function(e) NULL)
  if (is.null(assay)) return(.ggmlr_seurat_generation() >= 5L)
  ao <- tryCatch(object[[assay]], error = function(e) NULL)
  if (is.null(ao)) return(.ggmlr_seurat_generation() >= 5L)
  methods::is(ao, "Assay5") || methods::is(ao, "StdAssay")
}
# note: the `%||%` operator used across these files is defined in autograd.R
