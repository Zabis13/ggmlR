# Single-cell adapter: contracts, registry and the PCA GPU engine ------------
#
# This file is the *typed core* of the single-cell integration and is fully
# usable without Seurat or Bioconductor installed: it operates on plain R
# matrices. The Seurat / SCE layers (extraction + injection) only feed matrices
# in and read results out — all compute goes through here.
#
# Three pieces:
#   1. ggml_task / ggml_result  — S3 contract objects passed between layers
#   2. ggml_ops_registry        — declared, introspectable list of operations
#   3. .ggmlr_pca_gpu()         — the actual engine for op = "embed" (PCA)

# ============================================================================
# 1. Contract objects
# ============================================================================

#' Construct a single-cell compute task
#'
#' A \code{ggml_task} is the contract object passed from the extraction layer to
#' the dispatch layer. It bundles the operation name, the dense feature matrix
#' (features in rows, cells in columns — the single-cell convention), the
#' operation parameters and the requested device. It performs no computation.
#'
#' @param op Operation name; must be registered in
#'   \code{\link{ggml_ops_registry}} (e.g. \code{"embed"}).
#' @param matrix A numeric \code{matrix} (dense) or \code{dgCMatrix} (sparse).
#'   Rows are features (genes), columns are cells.
#' @param params Named list of operation parameters (e.g. \code{n_components}).
#' @param device \code{"vulkan"}, \code{"cpu"} or \code{"auto"} (default).
#'
#' @return An object of class \code{ggml_task}.
#' @seealso \code{\link{ggml_run}}, \code{\link{ggml_ops_registry}}
#' @export
ggml_task <- function(op, matrix, params = list(), device = c("auto", "vulkan", "cpu")) {
  device <- match.arg(device)
  if (!is.character(op) || length(op) != 1L)
    stop("`op` must be a single operation name.", call. = FALSE)
  if (!(is.matrix(matrix) || methods::is(matrix, "dgCMatrix")))
    stop("`matrix` must be a dense matrix or a dgCMatrix.", call. = FALSE)
  structure(
    list(op = op, matrix = matrix, params = params, device = device),
    class = "ggml_task"
  )
}

#' @export
print.ggml_task <- function(x, ...) {
  d <- dim(x$matrix)
  cat(sprintf("<ggml_task> op=%s  matrix=%d features x %d cells  device=%s\n",
              x$op, d[1L], d[2L], x$device))
  invisible(x)
}

#' Construct a single-cell result
#'
#' A \code{ggml_result} is the contract object returned by the dispatch layer
#' and consumed by the injection layer. The embedding is stored cell-by-component
#' (cells in rows), ready to drop into \code{reducedDim()} / a Seurat reduction.
#'
#' @param embedding A numeric matrix, cells in rows, components in columns.
#' @param metadata Named list (e.g. \code{stdev}, \code{loadings}, backend used).
#' @param timings Named numeric vector of elapsed seconds per stage.
#'
#' @return An object of class \code{ggml_result}.
#' @export
ggml_result <- function(embedding, metadata = list(), timings = numeric(0)) {
  structure(
    list(embedding = embedding, metadata = metadata, timings = timings),
    class = "ggml_result"
  )
}

#' @export
print.ggml_result <- function(x, ...) {
  d <- dim(x$embedding)
  cat(sprintf("<ggml_result> embedding=%d cells x %d components  backend=%s\n",
              d[1L], d[2L], x$metadata$backend %||% "unknown"))
  invisible(x)
}

# ============================================================================
# 2. Operations registry
# ============================================================================
#
# The registry lets an adapter (or a user) ask "is this op supported, and what
# does it need?" *before* dispatch, so capability checks never become runtime
# surprises. Each entry declares the engine function and required parameters.

.ggmlr_ops_registry <- new.env(parent = emptyenv())

# internal: register one operation
.ggmlr_register_op <- function(op, engine, params = character(0), desc = "") {
  .ggmlr_ops_registry[[op]] <- list(
    op = op, engine = engine, params = params, desc = desc
  )
  invisible(NULL)
}

#' Supported single-cell operations
#'
#' Returns the registry of operations the single-cell adapter can dispatch. Use
#' this to check capabilities (and required parameters) before building a
#' \code{\link{ggml_task}} — capability is declared, never discovered at runtime.
#'
#' @param op Optional operation name. If supplied, returns that single entry (or
#'   \code{NULL} if unknown); otherwise a named list of all entries.
#' @return A list describing the operation(s): \code{op}, \code{params}
#'   (required parameter names) and \code{desc}.
#' @examples
#' ggml_ops_registry()
#' ggml_ops_registry("embed")
#' @export
ggml_ops_registry <- function(op = NULL) {
  if (!is.null(op)) return(.ggmlr_ops_registry[[op]])
  ops <- as.list(.ggmlr_ops_registry)
  ops[order(names(ops))]
}

# ============================================================================
# 3. PCA engine (op = "embed")
# ============================================================================

#' GPU-accelerated PCA on a dense expression matrix
#'
#' Computes principal components of a feature-by-cell matrix. The heavy step —
#' the gene-by-gene covariance (a large matrix multiply) — runs on the Vulkan
#' GPU via the \code{ag_*} backend; the eigendecomposition of the (small,
#' features x features) covariance runs on the CPU, since \code{ggml} has no
#' eigensolver. Cells are projected onto the leading eigenvectors.
#'
#' @param mat Dense numeric matrix, features in rows, cells in columns.
#' @param n_components Number of principal components to return.
#' @param center Logical; subtract the per-feature mean before PCA (default
#'   \code{TRUE}). Single-cell PCA is virtually always centered.
#' @param backend \code{"vulkan"} to use the GPU for the covariance multiply,
#'   \code{"cpu"} to keep it on the CPU. The caller (dispatch layer) resolves
#'   \code{"auto"} to one of these.
#'
#' @return A \code{\link{ggml_result}}: \code{embedding} is cells x
#'   \code{n_components}; \code{metadata} holds \code{stdev} (component standard
#'   deviations), \code{loadings} (features x components) and \code{backend}.
#' @keywords internal
.ggmlr_pca_gpu <- function(mat, n_components = 50L, center = TRUE,
                           backend = c("vulkan", "cpu")) {
  backend <- match.arg(backend)
  storage.mode(mat) <- "double"
  n_feat <- nrow(mat); n_cell <- ncol(mat)
  n_components <- as.integer(min(n_components, n_feat, n_cell))

  t0 <- proc.time()[["elapsed"]]

  # Centre per feature (row means): X_c = X - rowMeans(X)
  if (center) {
    mu  <- rowMeans(mat)
    mat <- mat - mu
  }

  # Covariance over cells: C = (1/(n-1)) X_c %*% t(X_c)  -> features x features.
  # This is the dominant cost; route it to the GPU when asked.
  denom <- max(n_cell - 1L, 1L)
  t_mm0 <- proc.time()[["elapsed"]]
  if (backend == "vulkan") {
    ag_device("gpu")
    cov <- .ag_gpu_matmul(mat, t(mat)) / denom
  } else {
    cov <- tcrossprod(mat) / denom
  }
  t_mm <- proc.time()[["elapsed"]] - t_mm0

  # Eigendecomposition on CPU (features x features is small relative to cells).
  ev   <- eigen(cov, symmetric = TRUE)
  keep <- seq_len(n_components)
  loadings <- ev$vectors[, keep, drop = FALSE]              # features x comps
  vals     <- pmax(ev$values[keep], 0)                      # guard tiny < 0

  # Project cells onto components: scores = t(X_c) %*% loadings  (cells x comps)
  if (backend == "vulkan") {
    scores <- .ag_gpu_matmul(t(mat), loadings)
  } else {
    scores <- crossprod(mat, loadings)
  }

  rownames(scores) <- colnames(mat)
  colnames(scores) <- paste0("PC_", keep)
  rownames(loadings) <- rownames(mat)
  colnames(loadings) <- paste0("PC_", keep)

  ggml_result(
    embedding = scores,
    metadata  = list(stdev = sqrt(vals), loadings = loadings, backend = backend,
                     centered = center),
    timings   = c(total = proc.time()[["elapsed"]] - t0, matmul = t_mm)
  )
}

# register op = "embed" -> PCA engine
.ggmlr_register_op(
  "embed", engine = .ggmlr_pca_gpu,
  params = "n_components",
  desc   = "PCA dimensionality reduction (covariance multiply on GPU, eigen on CPU)"
)
