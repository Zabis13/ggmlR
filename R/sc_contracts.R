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
.ggmlr_register_op <- function(op, engine, params = character(0), desc = "",
                               sparse_ok = FALSE) {
  # sparse_ok = TRUE means the engine accepts a dgCMatrix directly (no densify);
  # the dispatch layer then skips its as.matrix() coercion for this op.
  .ggmlr_ops_registry[[op]] <- list(
    op = op, engine = engine, params = params, desc = desc,
    sparse_ok = sparse_ok
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
# 2b. Chunked-column iteration (datasets whose dense form exceeds memory)
# ============================================================================
# Several engines densify a features x cells matrix whose dense form is tens of
# GB at single-cell scale. When the caller passes chunk_size, the sparse
# dgCMatrix is kept intact and only a block of `chunk_size` cell-columns is
# densified at a time, so peak memory is one block, not the whole matrix. Each
# engine defines *how* it consumes the blocks (single pass, two pass, or
# accumulate); this helper only slices the column range.

# Column-block boundaries as a list of integer index vectors. chunk_size = NULL
# (or >= ncol) yields a single block spanning every column, i.e. the original
# non-chunked behaviour.
.ggmlr_chunk_cols <- function(ncol, chunk_size = NULL) {
  if (is.null(chunk_size) || !is.finite(chunk_size) || chunk_size >= ncol)
    return(list(seq_len(ncol)))
  chunk_size <- max(1L, as.integer(chunk_size))
  starts <- seq.int(1L, ncol, by = chunk_size)
  lapply(starts, function(s) s:min(s + chunk_size - 1L, ncol))
}

# Densify one column block of a (sparse or dense) genes x cells matrix to a plain
# double matrix. Keeps row names; the block is the only dense allocation.
.ggmlr_densify_block <- function(mat, cols) {
  blk <- mat[, cols, drop = FALSE]
  if (!is.matrix(blk)) blk <- as.matrix(blk)
  storage.mode(blk) <- "double"
  blk
}

# ============================================================================
# 3. PCA engine (op = "embed")
# ============================================================================

# Streaming PCA over cell-blocks: same result as .ggmlr_pca_gpu but never holds
# the full dense matrix. Pass 1: per-gene mean. Pass 2: accumulate the centred
# covariance C = sum_b (X_b - mu)(X_b - mu)^T. Pass 3: project each block onto
# the eigenvectors. The covariance multiply per block goes to the GPU when
# backend = "vulkan"; the (small) eigendecomposition stays on the CPU.
.ggmlr_pca_chunked <- function(mat, n_components, center, backend, chunk_size) {
  n_feat <- nrow(mat); n_cell <- ncol(mat)
  denom  <- max(n_cell - 1L, 1L)
  blocks <- .ggmlr_chunk_cols(n_cell, chunk_size)
  t0 <- proc.time()[["elapsed"]]
  if (backend == "vulkan") ag_device("gpu")

  # Pass 1: per-feature mean over all cells (skip when not centering).
  mu <- numeric(n_feat)
  if (center) {
    s1 <- numeric(n_feat)
    for (cols in blocks) s1 <- s1 + rowSums(.ggmlr_densify_block(mat, cols))
    mu <- s1 / n_cell
  }

  # Pass 2: accumulate the covariance from centred blocks (features x features).
  cov <- matrix(0, n_feat, n_feat)
  for (cols in blocks) {
    blk <- .ggmlr_densify_block(mat, cols)
    if (center) blk <- blk - mu
    cov <- cov + if (backend == "vulkan") .ag_gpu_matmul(blk, t(blk))
                 else tcrossprod(blk)
  }
  cov <- cov / denom

  keep <- seq_len(n_components)
  use_truncated <- requireNamespace("RSpectra", quietly = TRUE) &&
                   n_components <= nrow(cov) %/% 2L
  ev <- if (use_truncated) {
    tryCatch(RSpectra::eigs_sym(cov, k = n_components, which = "LA"),
             error = function(e) eigen(cov, symmetric = TRUE))
  } else {
    eigen(cov, symmetric = TRUE)
  }
  loadings <- ev$vectors[, keep, drop = FALSE]
  vals     <- pmax(ev$values[keep], 0)

  # Pass 3: project each centred block onto the loadings -> scores (cells x comps)
  scores <- matrix(0, n_cell, n_components)
  for (cols in blocks) {
    blk <- .ggmlr_densify_block(mat, cols)
    if (center) blk <- blk - mu
    scores[cols, ] <- if (backend == "vulkan") .ag_gpu_matmul(t(blk), loadings)
                      else crossprod(blk, loadings)
  }

  rownames(scores) <- colnames(mat)
  colnames(scores) <- paste0("PC_", keep)
  rownames(loadings) <- rownames(mat)
  colnames(loadings) <- paste0("PC_", keep)

  ggml_result(
    embedding = scores,
    metadata  = list(stdev = sqrt(vals), loadings = loadings, backend = backend,
                     centered = center, chunked = TRUE),
    timings   = c(total = proc.time()[["elapsed"]] - t0)
  )
}

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
                           backend = c("vulkan", "cpu"), chunk_size = NULL) {
  backend <- match.arg(backend)
  n_feat <- nrow(mat); n_cell <- ncol(mat)
  n_components <- as.integer(min(n_components, n_feat, n_cell))

  # Chunked path: stream the matrix in cell-blocks so the full dense features x
  # cells matrix is never held. The covariance is additive over cells --
  # C = sum_b (X_b - mu)(X_b - mu)^T -- so it accumulates block by block into a
  # small features x features matrix; the projection is likewise per-block. The
  # per-feature mean must be known first, so pass 1 accumulates it, pass 2 the
  # covariance, pass 3 the scores. feat x feat covariance and feat/cell x comps
  # outputs are small; only one densified block is ever resident.
  if (!is.null(chunk_size) && !is.matrix(mat)) {
    return(.ggmlr_pca_chunked(mat, n_components, center, backend, chunk_size))
  }

  storage.mode(mat) <- "double"
  t0 <- proc.time()[["elapsed"]]

  # Centre per feature (row means): X_c = X - rowMeans(X)
  t_ctr0 <- proc.time()[["elapsed"]]
  if (center) {
    mu  <- rowMeans(mat)
    mat <- mat - mu
  }
  t_ctr <- proc.time()[["elapsed"]] - t_ctr0

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
  t_eig0 <- proc.time()[["elapsed"]]
  ev <- if (use_truncated) {
    tryCatch(
      RSpectra::eigs_sym(cov, k = n_components, which = "LA"),
      error = function(e) eigen(cov, symmetric = TRUE))
  } else {
    eigen(cov, symmetric = TRUE)
  }
  t_eig <- proc.time()[["elapsed"]] - t_eig0
  loadings <- ev$vectors[, keep, drop = FALSE]              # features x comps
  vals     <- pmax(ev$values[keep], 0)                      # guard tiny < 0

  # Project cells onto components: scores = t(X_c) %*% loadings  (cells x comps)
  t_prj0 <- proc.time()[["elapsed"]]
  if (backend == "vulkan") {
    scores <- .ag_gpu_matmul(t(mat), loadings)
  } else {
    scores <- crossprod(mat, loadings)
  }
  t_prj <- proc.time()[["elapsed"]] - t_prj0

  rownames(scores) <- colnames(mat)
  colnames(scores) <- paste0("PC_", keep)
  rownames(loadings) <- rownames(mat)
  colnames(loadings) <- paste0("PC_", keep)

  ggml_result(
    embedding = scores,
    metadata  = list(stdev = sqrt(vals), loadings = loadings, backend = backend,
                     centered = center),
    timings   = c(total = proc.time()[["elapsed"]] - t0, centre = t_ctr,
                  matmul_cov = t_mm, eigen = t_eig, matmul_proj = t_prj)
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
                                 backend = c("vulkan", "cpu"),
                                 chunk_size = NULL) {
  backend <- match.arg(backend)
  # chunk_size is accepted for a uniform RunGGML interface but is a no-op here:
  # the sparse LogNormalize path already transforms @x in place without ever
  # densifying, so there is no full dense matrix to stream in blocks.
  t0 <- proc.time()[["elapsed"]]

  # Sparse path: mat is a dgCMatrix (dispatch left it sparse; sparse_ok). Because
  # log1p(0) = 0, LogNormalize only touches the stored non-zeros @x, so we never
  # densify. Column sums come from Matrix::colSums (cheap, O(nnz)); the per-column
  # factor scale_factor/colSum and the per-nnz column index are uploaded, and the
  # shader maps @x[k] -> log1p(@x[k] * factor[col]) in place. The transformed @x
  # drops straight back into a dgCMatrix with the same sparsity pattern. NOTE:
  # LogNormalize is memory-bound O(nnz), so the GPU path targets parity with (and
  # the removal of the densify/OOM ceiling versus) Seurat's sparse CPU path, not
  # a speed-up over it.
  if (methods::is(mat, "dgCMatrix")) {
    out  <- .ggmlr_normalize_sparse(mat, scale_factor, backend)
    used <- attr(out, "backend")
    attr(out, "backend") <- NULL                # keep the dgCMatrix contract clean
    return(ggml_result(
      embedding = out,
      metadata  = list(kind = "transform", layer = "data", backend = used,
                       scale_factor = scale_factor, sparse = TRUE),
      timings   = c(total = proc.time()[["elapsed"]] - t0)
    ))
  }

  storage.mode(mat) <- "double"
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

# Sparse LogNormalize on a dgCMatrix, transforming @x in place (no densify).
# Returns a dgCMatrix with the same pattern; attr "backend" records the path
# actually taken ("vulkan" when the GPU shader ran, else "cpu"). Under
# backend = "vulkan" it dispatches sparse_lognorm.comp; if the GPU is
# unavailable it falls back to the elementwise CPU form on @x.
.ggmlr_normalize_sparse <- function(mat, scale_factor, backend) {
  out <- mat
  nnz <- length(mat@x)
  # per-column sums over the stored values; empty cells guarded to 1
  cs  <- Matrix::colSums(mat)
  cs[cs == 0] <- 1
  factor <- scale_factor / cs                   # length ncol(mat)

  used <- "cpu"
  if (backend == "vulkan" && nnz > 0L) {
    ok <- tryCatch({ ag_device("gpu"); TRUE }, error = function(e) FALSE)
    vk <- if (ok) .ag_device_state$backend else NULL
    if (!is.null(vk) && ggml_vulkan_is_backend(vk)) {
      # @p is the CSC column pointer (length ncol+1); expand to a 0-based column
      # index per stored value so the shader needs no binary search.
      col_of_nnz <- rep.int(seq_len(ncol(mat)) - 1L, diff(mat@p))
      newx <- tryCatch(
        .Call("R_ggml_sparse_lognorm", vk, as.double(mat@x),
              as.double(factor), as.integer(col_of_nnz),
              as.integer(nnz), as.integer(ncol(mat)), PACKAGE = "ggmlR"),
        error = function(e) NULL)
      if (!is.null(newx)) { out@x <- newx; used <- "vulkan" }
    }
  }

  if (used == "cpu") {
    # elementwise on the stored values only: log1p(x * factor[col])
    col_of_nnz <- rep.int(seq_len(ncol(mat)), diff(mat@p))   # 1-based for R
    out@x <- log1p(mat@x * factor[col_of_nnz])
  }

  attr(out, "backend") <- used
  out
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
#' @param scale_backend Which backend actually runs the z-score: \code{"cpu"}
#'   (default) or \code{"vulkan"}. Defaults to CPU \emph{even under}
#'   \code{backend = "vulkan"}, because ScaleData is a memory-bound elementwise
#'   O(nnz) pass (centre / divide / clamp) with almost no arithmetic per element:
#'   the GPU pays for the host<->VRAM copy but has nothing to accelerate, so it is
#'   slower than the CPU here (measured ~0.4x). Same rationale and pattern as
#'   UMAP's \code{sgd_backend}. Pass \code{"vulkan"} to force the GPU path.
#' @return A \code{\link{ggml_result}} whose \code{embedding} is the scaled
#'   features x cells matrix; \code{metadata$kind = "transform"},
#'   \code{metadata$layer = "scale.data"}.
#' @keywords internal
.ggmlr_scale_gpu <- function(mat, max_value = 10, backend = c("vulkan", "cpu"),
                             scale_backend = c("cpu", "vulkan"),
                             chunk_size = NULL) {
  backend       <- match.arg(backend)
  scale_backend <- match.arg(scale_backend)
  # scale is memory-bound; run it on the CPU by default even when the GPU is
  # live. Only go to Vulkan when both the device is Vulkan and the user opted in.
  backend <- if (backend == "vulkan" && scale_backend == "vulkan") "vulkan" else "cpu"
  n_cell <- ncol(mat)
  t0 <- proc.time()[["elapsed"]]

  # Chunked path: when chunk_size is set the matrix is streamed in cell-blocks
  # (kept sparse until each block is densified), so the full dense features x
  # cells matrix is never held. Two passes: (1) accumulate per-gene mean and
  # sum-of-squares over all cells, (2) re-densify each block and write the
  # z-scored, clamped values into the (dense) output. Runs on the CPU only:
  # z-score is memory-bound (see scale_backend), and streaming to VRAM per block
  # would only add host<->device copies. The single unavoidable dense allocation
  # is the output, which Seurat's scale.data layer must hold in full anyway.
  if (!is.null(chunk_size) && !is.matrix(mat)) {
    blocks <- .ggmlr_chunk_cols(n_cell, chunk_size)
    n_feat <- nrow(mat)
    s1 <- numeric(n_feat); s2 <- numeric(n_feat)      # sum(x), sum(x^2) per gene
    for (cols in blocks) {
      blk <- .ggmlr_densify_block(mat, cols)
      s1  <- s1 + rowSums(blk)
      s2  <- s2 + rowSums(blk * blk)
    }
    mu <- s1 / n_cell
    # population sd with n-1 divisor (Seurat uses sd()): var = (sum(x^2) - n*mu^2)/(n-1)
    var <- (s2 - n_cell * mu * mu) / max(n_cell - 1L, 1L)
    sd  <- sqrt(pmax(var, 0)); sd[sd == 0] <- 1
    out <- matrix(0, n_feat, n_cell, dimnames = dimnames(mat))
    for (cols in blocks) {
      blk <- .ggmlr_densify_block(mat, cols)
      out[, cols] <- pmin((blk - mu) / sd, max_value)
    }
    return(ggml_result(
      embedding = out,
      metadata  = list(kind = "transform", layer = "scale.data", backend = backend,
                       max_value = max_value, chunked = TRUE),
      timings   = c(total = proc.time()[["elapsed"]] - t0)
    ))
  }

  storage.mode(mat) <- "double"
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

# ============================================================================
# 3c. Per-cell reduction engine (op = "largest_gene")
# ============================================================================
# Unlike "embed" (a reduction) and the transforms, this returns *per-cell
# columns* — the highest-expressed gene per cell and its share of the cell's
# total counts, matching Seurat's `percent.Largest.Gene` QC metric
# (qlcMatrix::colMax(counts, which = TRUE)). Two vectors of length ncell carry
# metadata$kind = "coldata" so the injection layer writes them into meta.data /
# colData rather than a reduction or an assay layer.

#' Highest-expressed gene per cell (op = "largest_gene")
#'
#' For every cell, finds the gene with the largest value and that value's share
#' of the cell's column sum — Seurat's \code{largest_gene} /
#' \code{percent.Largest.Gene} QC metric (\code{qlcMatrix::colMax(counts,
#' which = TRUE)}). Runs on the CPU directly over the sparse \code{dgCMatrix}
#' CSC slots (\code{@x}, \code{@i}, \code{@p}) without ever densifying, so it
#' scales to the full counts matrix. This is a memory-bound O(nnz) column
#' argmax/max with no arithmetic to accelerate, so — like ScaleData and the
#' UMAP layout — there is nothing for the GPU to speed up; \code{backend} is
#' accepted for a uniform interface but the compute always stays on the CPU.
#'
#' @param mat A \code{dgCMatrix} (preferred; kept sparse) or dense numeric
#'   matrix, features x cells (raw counts).
#' @param backend Accepted for interface uniformity; ignored (always CPU).
#' @return A \code{\link{ggml_result}} with \code{metadata$kind = "coldata"} and
#'   \code{embedding} a data.frame of two columns: \code{largest_gene} (chr, the
#'   feature name, \code{NA} for empty cells) and \code{percent.Largest.Gene}
#'   (dbl, \code{max / colSum * 100}, 0 for empty cells), one row per cell.
#' @keywords internal
.ggmlr_largest_gene <- function(mat, backend = c("vulkan", "cpu"),
                                chunk_size = NULL) {
  match.arg(backend)                        # accepted but unused (CPU-only op)
  t0 <- proc.time()[["elapsed"]]

  n_cell <- ncol(mat)
  genes  <- rownames(mat)
  cells  <- colnames(mat)

  gene_idx <- integer(n_cell)               # 1-based row of the per-cell max (0 = none)
  max_val  <- numeric(n_cell)               # the max value per cell
  col_sum  <- numeric(n_cell)               # per-cell total (for the percentage)

  if (methods::is(mat, "dgCMatrix")) {
    # Sparse CSC: column j holds stored values @x[(p[j]+1):p[j+1]] at rows
    # @i[...] (0-based). counts are non-negative, so the column max is the max of
    # its stored values (an all-zero column stays 0 -> empty). No densify.
    p <- mat@p; i <- mat@i; x <- mat@x
    for (j in seq_len(n_cell)) {
      lo <- p[j] + 1L; hi <- p[j + 1L]
      if (hi >= lo) {
        seg <- x[lo:hi]
        col_sum[j] <- sum(seg)
        k <- which.max(seg)                 # first max within the column
        max_val[j]  <- seg[k]
        gene_idx[j] <- i[lo + k - 1L] + 1L  # 0-based row -> 1-based
      }
    }
  } else {
    storage.mode(mat) <- "double"
    col_sum  <- colSums(mat)
    gene_idx <- max.col(t(mat), ties.method = "first")
    max_val  <- mat[cbind(gene_idx, seq_len(n_cell))]
    gene_idx[col_sum == 0] <- 0L            # treat all-zero cells as empty
  }

  largest_gene <- rep(NA_character_, n_cell)
  have <- gene_idx > 0L
  largest_gene[have] <- if (!is.null(genes)) genes[gene_idx[have]] else gene_idx[have]

  denom <- col_sum; denom[denom == 0] <- 1  # guard empty cells (percent stays 0)
  percent <- max_val / denom * 100

  df <- data.frame(largest_gene = largest_gene,
                   percent.Largest.Gene = percent,
                   row.names = cells, stringsAsFactors = FALSE)

  ggml_result(
    embedding = df,
    metadata  = list(kind = "coldata", backend = "cpu"),
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
  desc   = "LogNormalize: per-cell library-size scaling + log1p (elementwise on GPU)",
  sparse_ok = TRUE   # engine handles a dgCMatrix without densifying (log1p(0)=0)
)

# register op = "scale" -> z-score engine
.ggmlr_register_op(
  "scale", engine = .ggmlr_scale_gpu,
  params = character(0),
  desc   = "ScaleData z-score per gene + clamp (elementwise on GPU)"
)

# register op = "largest_gene" -> per-cell argmax/max engine
.ggmlr_register_op(
  "largest_gene", engine = .ggmlr_largest_gene,
  params = character(0),
  desc   = "Highest-expressed gene per cell + its percent of the cell total (CPU, sparse)",
  sparse_ok = TRUE   # engine reads the dgCMatrix CSC slots directly (no densify)
)
