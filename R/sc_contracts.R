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

  # Eigendecomposition on CPU (ggml has no eigensolver). A full eigen() computes
  # all `nrow(cov)` eigenpairs, but PCA only needs the top n_components. A
  # truncated symmetric solver (RSpectra::eigs_sym, "LA" = largest algebraic)
  # returns just those, which is far cheaper when components << features. Fall
  # back to eigen() when RSpectra is absent or the truncation is not worthwhile
  # (k close to the matrix size, where the Lanczos solver loses its edge and may
  # not converge).
  keep <- seq_len(n_components)
  use_truncated <- requireNamespace("RSpectra", quietly = TRUE) &&
                   n_components <= nrow(cov) %/% 2L
  ev <- if (use_truncated) {
    tryCatch(
      RSpectra::eigs_sym(cov, k = n_components, which = "LA"),
      error = function(e) eigen(cov, symmetric = TRUE))
  } else {
    eigen(cov, symmetric = TRUE)
  }
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

# ============================================================================
# 3b. Transform engines (op = "normalize", op = "scale")
# ============================================================================
# Unlike "embed" (which returns a reduction), these return a *transformed*
# feature-by-cell matrix that is written back into an assay layer. They carry
# metadata$kind = "transform" so the injection layer knows to put the matrix in
# a layer (data / scale.data) rather than a DimReduc slot.

#' GPU-accelerated LogNormalize (op = "normalize")
#'
#' Library-size normalisation followed by log1p, matching Seurat's
#' \code{NormalizeData(method = "LogNormalize")}:
#' \code{log1p(x / colSums(x) * scale_factor)}. The per-cell scaling and the
#' \code{log1p} run elementwise on the GPU (broadcast a per-cell factor across
#' genes); the column sums are a cheap reduction.
#'
#' @param mat Dense numeric matrix, features x cells (raw/counts).
#' @param scale_factor Library size to scale each cell to (default 1e4).
#' @param backend \code{"vulkan"} or \code{"cpu"} (dispatch resolves "auto").
#' @return A \code{\link{ggml_result}} whose \code{embedding} is the normalised
#'   features x cells matrix; \code{metadata$kind = "transform"},
#'   \code{metadata$layer = "data"}.
#' @keywords internal
.ggmlr_normalize_gpu <- function(mat, scale_factor = 1e4,
                                 backend = c("vulkan", "cpu")) {
  backend <- match.arg(backend)
  storage.mode(mat) <- "double"
  t0 <- proc.time()[["elapsed"]]

  cs  <- colSums(mat)
  cs[cs == 0] <- 1                              # guard empty cells
  fac <- matrix(scale_factor / cs, nrow = 1L)   # [1, cells] per-cell factor

  if (backend == "vulkan") {
    ag_device("gpu")
    scaled <- .ag_gpu_mul(mat, fac)             # broadcast across genes
    out    <- .ag_gpu_log(.ag_gpu_add(scaled, matrix(1, 1L, 1L)))  # log1p
  } else {
    out <- log1p(sweep(mat, 2L, as.vector(fac), `*`))
  }

  dimnames(out) <- dimnames(mat)
  ggml_result(
    embedding = out,
    metadata  = list(kind = "transform", layer = "data", backend = backend,
                     scale_factor = scale_factor),
    timings   = c(total = proc.time()[["elapsed"]] - t0)
  )
}

#' GPU-accelerated ScaleData / z-score (op = "scale")
#'
#' Per-gene centering and scaling to unit variance, matching Seurat's
#' \code{ScaleData}: \code{(x - rowMeans) / rowSds}, then clamp to
#' \code{[-Inf, max_value]} (Seurat clips at +10 by default). The dominant cost
#' — elementwise subtract/divide/clamp over the full dense matrix — runs on the
#' GPU; the per-gene mean and sd are cheap row reductions.
#'
#' @param mat Dense numeric matrix, features x cells (log-normalised data).
#' @param max_value Upper clip after scaling (default 10; Seurat's default).
#' @param backend \code{"vulkan"} or \code{"cpu"} (dispatch resolves "auto").
#' @return A \code{\link{ggml_result}} whose \code{embedding} is the scaled
#'   features x cells matrix; \code{metadata$kind = "transform"},
#'   \code{metadata$layer = "scale.data"}.
#' @keywords internal
.ggmlr_scale_gpu <- function(mat, max_value = 10, backend = c("vulkan", "cpu")) {
  backend <- match.arg(backend)
  storage.mode(mat) <- "double"
  n_cell <- ncol(mat)
  t0 <- proc.time()[["elapsed"]]

  mu <- matrix(rowMeans(mat), ncol = 1L)        # [features, 1] per-gene mean

  if (backend == "vulkan") {
    ag_device("gpu")
    xc  <- .ag_gpu_sub(mat, mu)                 # centre (broadcast across cells)
    # population-style sd over cells: Seurat uses sd() (n-1 divisor)
    ss  <- rowSums(xc * xc)
    sd  <- sqrt(ss / max(n_cell - 1L, 1L))
    sd[sd == 0] <- 1
    inv <- matrix(1 / sd, ncol = 1L)            # [features, 1]
    xs  <- .ag_gpu_mul(xc, inv)                 # divide (broadcast across cells)
    out <- .ag_gpu_clamp(xs, -Inf, max_value)
  } else {
    xc  <- mat - as.vector(mu)
    sd  <- sqrt(rowSums(xc * xc) / max(n_cell - 1L, 1L))
    sd[sd == 0] <- 1
    out <- pmin((xc / sd), max_value)
  }

  dimnames(out) <- dimnames(mat)
  ggml_result(
    embedding = out,
    metadata  = list(kind = "transform", layer = "scale.data", backend = backend,
                     max_value = max_value),
    timings   = c(total = proc.time()[["elapsed"]] - t0)
  )
}

# register op = "embed" -> PCA engine
.ggmlr_register_op(
  "embed", engine = .ggmlr_pca_gpu,
  params = "n_components",
  desc   = "PCA dimensionality reduction (covariance multiply on GPU, eigen on CPU)"
)

# register op = "normalize" -> LogNormalize engine
.ggmlr_register_op(
  "normalize", engine = .ggmlr_normalize_gpu,
  params = character(0),
  desc   = "LogNormalize: per-cell library-size scaling + log1p (elementwise on GPU)"
)

# register op = "scale" -> z-score engine
.ggmlr_register_op(
  "scale", engine = .ggmlr_scale_gpu,
  params = character(0),
  desc   = "ScaleData z-score per gene + clamp (elementwise on GPU)"
)
