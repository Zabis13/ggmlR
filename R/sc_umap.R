# Single-cell adapter: UMAP embedding (op = "umap") -------------------------
#
# UMAP has three phases (cf. the reference implementation and RAPIDS cuML):
#   1. kNN graph              — nearest neighbours in the input space
#   2. fuzzy simplicial set   — symmetrised edge weights in [0, 1]
#   3. SGD optimisation       — lay the points out in 2-D by attraction along
#                               graph edges and repulsion from random negatives
#
# Phase 3 is the part destined for the GPU (a per-edge compute shader; see TODO
# slice 1c). This file is the CPU reference: it defines the exact numerics the
# shader must reproduce, and is the engine behind op = "umap" until the shader
# lands. The CPU path is intentionally simple and deterministic given a seed, so
# the GPU output can be cross-checked against it.

# ----------------------------------------------------------------------------
# a, b curve parameters: fit  1 / (1 + a * d^(2b))  to the UMAP membership
# function defined by min_dist / spread. Matches uwot::find_ab_params.
# ----------------------------------------------------------------------------
.ggmlr_umap_find_ab <- function(spread = 1, min_dist = 0.1) {
  xv <- seq(0, spread * 3, length.out = 300)
  yv <- ifelse(xv < min_dist, 1, exp(-(xv - min_dist) / spread))
  # non-linear least squares for the 1/(1 + a d^(2b)) curve
  fit <- tryCatch(
    stats::nls(yv ~ 1 / (1 + a * xv^(2 * b)),
               start = list(a = 1, b = 1),
               control = stats::nls.control(maxiter = 200, warnOnly = TRUE)),
    error = function(e) NULL)
  if (is.null(fit)) return(c(a = 1.577, b = 0.895))  # uwot defaults
  co <- stats::coef(fit)
  c(a = unname(co[["a"]]), b = unname(co[["b"]]))
}

# ----------------------------------------------------------------------------
# kNN + fuzzy simplicial set -> a symmetric edge list with weights.
# Exact (brute-force) kNN; fine for the moderate cell counts this adapter
# targets, and it keeps the reference unambiguous. Returns 0-based vertex
# indices (ggml convention) and per-edge weights.
# ----------------------------------------------------------------------------
# Pairwise euclidean distance matrix (cells x cells): GPU shader with CPU
# fallback.
#
# The GPU path is pairwise_dist.comp (a tiled 32x32 compute shader that
# accumulates each D^2 in f32). An earlier attempt via the Gram matrix
# (D^2 = ||x_i||^2 + ||x_j||^2 - 2 X X^T on mul_mat) was rejected: the
# Vulkan/RADV mul_mat accumulates in f16 even with GGML_PREC_F32 forced on the
# node (rel error ~2.7e-4), which the Gram identity amplifies into ~0.1 absolute
# distance error — enough to reorder the top-k nearest neighbours. f16 distances
# are a kNN bug, so the dedicated f32 shader is the supported GPU path and the
# exact CPU dist() is the fallback. gpu_neighbor_max_cells caps the GPU path
# (the n x n output is n^2 * 4 bytes of VRAM); above it we fall back to the CPU.
# ----------------------------------------------------------------------------
.ggmlr_dist_matrix <- function(X, gpu_neighbor_max_cells = NULL) {
  D <- .ggmlr_dist_gpu(X, gpu_neighbor_max_cells)   # f32 Vulkan, NULL -> fallback
  if (!is.null(D)) {
    attr(D, "backend") <- "vulkan"                  # pairwise_dist.comp path
    return(D)
  }
  D <- as.matrix(stats::dist(X))                     # exact CPU euclidean
  attr(D, "backend") <- "cpu"
  D
}

# Largest cell count whose pairwise distance fits on the GPU. The output is the
# dominant cost: n*n f32 for D2 plus n*dims f32 for X. We budget a fraction of
# the device's *free* VRAM (not total — other allocations are live) and solve for
# n. Falls back to a conservative fixed cap if the memory query is unavailable.
.ggmlr_dist_gpu_max_cells <- function(dims, vram_fraction = 0.5,
                                      fallback = 10000L) {
  mem <- tryCatch(ggml_vulkan_device_memory(0L), error = function(e) NULL)
  free <- mem$free %||% mem$total %||% NA_real_
  if (!is.finite(free) || free <= 0) return(fallback)

  budget <- free * vram_fraction
  # budget >= n^2 * 4 + n * dims * 4  ->  solve the quadratic 4 n^2 + 4 dims n - B
  a <- 4; b <- 4 * dims; cc <- -budget
  n_max <- (-b + sqrt(b * b - 4 * a * cc)) / (2 * a)
  as.integer(max(2L, floor(n_max)))
}

# GPU pairwise Euclidean distance via the pairwise_dist.comp shader (honest f32
# accumulation, sidestepping mul_mat's f16 path). Returns an n x n matrix, or
# NULL to fall back to the CPU dist() above. Mirrors .ggmlr_umap_sgd_gpu's guards:
# no GPU, a non-Vulkan backend, or n above the VRAM ceiling -> NULL.
# gpu_neighbor_max_cells = NULL derives the ceiling from free VRAM (the default);
# 0 disables the GPU path; a positive integer is an explicit override.
.ggmlr_dist_gpu <- function(X, gpu_neighbor_max_cells = NULL) {
  n <- nrow(X)
  if (n < 2L) return(NULL)

  ok <- tryCatch({ ag_device("gpu"); TRUE }, error = function(e) FALSE)
  if (!ok) return(NULL)
  backend <- .ag_device_state$backend
  if (is.null(backend) || !ggml_vulkan_is_backend(backend)) return(NULL)

  dims <- ncol(X)
  ceiling <- gpu_neighbor_max_cells %||% .ggmlr_dist_gpu_max_cells(dims)
  if (n > ceiling) return(NULL)
  # shader reads X row-major; R matrices are column-major, so t() gives the
  # [row0_d0,row0_d1,...] stream the kernel expects.
  xrow <- as.double(t(X))

  # R_ggml_dist_f32 already returns Euclidean distances (sqrt + clamp done in C),
  # and the matrix is symmetric, so the flat row-major D[i*n+j] reads back
  # correctly as a plain column-major matrix() with no transpose pass.
  dvec <- tryCatch(
    .Call("R_ggml_dist_f32", backend, xrow,
          as.integer(n), as.integer(dims), PACKAGE = "ggmlR"),
    error = function(e) NULL)
  if (is.null(dvec)) return(NULL)

  matrix(dvec, n, n)
}

# k nearest neighbours of each row of X (excluding self), returned sorted by
# distance: a list of knn_idx (n x k, 1-based), knn_dist (n x k), and the method.
# FNN's kd-tree gives the exact k-NN in O(n log n) without materialising the full
# n x n distance matrix, so it is both faster and far lighter on memory than
# sorting every row of a dense D — that is the hot path here. Without FNN we fall
# back to the pairwise distance matrix (GPU pairwise_dist.comp or CPU dist) and a
# per-row order(). The GPU distance shader stays the path for op = "neighbors",
# where the full matrix is wanted; here we only need the k smallest per row.
.ggmlr_umap_knn <- function(X, n_neighbors, gpu_neighbor_max_cells = NULL) {
  n <- nrow(X)
  if (.ggmlr_has_pkg("FNN")) {
    nn <- FNN::get.knn(X, k = n_neighbors, algorithm = "kd_tree")
    return(list(idx = nn$nn.index, dist = nn$nn.dist, backend = "fnn"))
  }

  # fallback: full distance matrix + per-row partial selection
  D <- .ggmlr_dist_matrix(X, gpu_neighbor_max_cells)   # n x n euclidean
  dist_backend <- attr(D, "backend") %||% "cpu"
  knn_idx  <- matrix(0L, n, n_neighbors)
  knn_dist <- matrix(0,  n, n_neighbors)
  for (i in seq_len(n)) {
    ord <- order(D[i, ])[-1][seq_len(n_neighbors)]     # drop self (distance 0)
    knn_idx[i, ]  <- ord
    knn_dist[i, ] <- D[i, ord]
  }
  list(idx = knn_idx, dist = knn_dist, backend = dist_backend)
}

.ggmlr_umap_fuzzy_graph <- function(X, n_neighbors = 15L,
                                    gpu_neighbor_max_cells = NULL) {
  n <- nrow(X)
  n_neighbors <- as.integer(min(n_neighbors, n - 1L))

  knn <- .ggmlr_umap_knn(X, n_neighbors, gpu_neighbor_max_cells)
  knn_idx      <- knn$idx
  knn_dist     <- knn$dist
  dist_backend <- knn$backend

  # smooth knn distances: per-point rho (nearest) and sigma (bandwidth) so the
  # membership strengths sum to log2(k). Bisection on sigma, as in UMAP.
  rho   <- knn_dist[, 1]
  sigma <- numeric(n)
  target <- log2(n_neighbors)
  for (i in seq_len(n)) {
    lo <- 0; hi <- Inf; mid <- 1
    d  <- pmax(knn_dist[i, ] - rho[i], 0)
    for (it in seq_len(64)) {
      psum <- sum(exp(-d / mid))
      if (abs(psum - target) < 1e-5) break
      if (psum > target) { hi <- mid; mid <- (lo + hi) / 2 }
      else { lo <- mid; mid <- if (is.infinite(hi)) mid * 2 else (lo + hi) / 2 }
    }
    sigma[i] <- mid
  }

  # Directed membership strengths W[i, knn]. Only n * n_neighbors entries are
  # non-zero, so building a dense n x n matrix (and the three n^2 passes the
  # symmetrisation below would take on it) is wasteful. With Matrix available we
  # carry W as a sparse matrix and symmetrise over its non-zeros only; without
  # it we fall back to the dense path.
  ii <- rep.int(seq_len(n), n_neighbors)       # row index, column-major over j
  jj <- as.integer(knn_idx)                    # neighbour index (knn_idx is n x k)
  ww <- as.numeric(exp(-pmax(knn_dist - rho, 0) / sigma))  # recycles by column

  if (.ggmlr_has_pkg("Matrix")) {
    W    <- Matrix::sparseMatrix(i = ii, j = jj, x = ww, dims = c(n, n))
    Wt   <- Matrix::t(W)
    Wsym <- W + Wt - W * Wt                     # probabilistic t-conorm (sparse)
    tri  <- Matrix::summary(Matrix::triu(Wsym, k = 1L))  # upper triangle only
    keep <- tri$x > 1e-3
    list(
      from = tri$i[keep] - 1L,                  # 0-based
      to   = tri$j[keep] - 1L,
      weight = tri$x[keep],
      n = n,
      dist_backend = dist_backend
    )
  } else {
    W <- matrix(0, n, n)
    W[cbind(ii, jj)] <- ww
    Wsym <- W + t(W) - W * t(W)                 # probabilistic t-conorm (dense)
    idx  <- which(Wsym > 1e-3 & upper.tri(Wsym), arr.ind = TRUE)
    list(
      from = idx[, 1] - 1L,                     # 0-based
      to   = idx[, 2] - 1L,
      weight = Wsym[idx],
      n = n,
      dist_backend = dist_backend
    )
  }
}

# ----------------------------------------------------------------------------
# PCG hash RNG, an exact mirror of pcg_hash() in vulkan-shaders/umap_sgd.comp.
# All arithmetic is done modulo 2^32 in double precision: R's bitwXor/bitwShiftR
# operate on 32-bit *signed* ints and return NA for our uint32 values (> 2^31).
# Shifts become integer division by 2^k; XOR is done on 16-bit halves (each
# < 2^16, safely within bitwXor's range) and recombined. This lets the CPU
# reference reproduce the GPU shader's negative samples bit-for-bit.
.ggmlr_u32 <- function(x) x %% 4294967296            # wrap to uint32
.ggmlr_shr <- function(x, k) floor(x / 2^k)          # logical right shift
.ggmlr_xor32 <- function(a, b) {                     # 32-bit XOR via 16-bit halves
  ah <- floor(a / 65536); al <- a %% 65536
  bh <- floor(b / 65536); bl <- b %% 65536
  bitwXor(ah, bh) * 65536 + bitwXor(al, bl)
}
# uint32 multiply mod 2^32. a*b can reach 2^64, past double's 53-bit mantissa,
# so split b into 16-bit halves: each partial product stays < 2^48 (exact).
.ggmlr_mul32 <- function(a, b) {
  bh <- floor(b / 65536); bl <- b %% 65536
  ((a * bh) %% 65536 * 65536 + a * bl) %% 4294967296
}
.ggmlr_pcg_hash <- function(x) {
  state <- (.ggmlr_mul32(x, 747796405) + 2891336453) %% 4294967296
  shift <- .ggmlr_shr(state, 28) + 4
  word  <- .ggmlr_mul32(.ggmlr_xor32(.ggmlr_shr(state, shift), state), 277803737)
  .ggmlr_u32(.ggmlr_xor32(.ggmlr_shr(word, 22), word))
}

# ----------------------------------------------------------------------------
# SGD layout optimisation (the GPU-bound phase). One epoch = one pass over the
# positive edges; each edge attracts its endpoints and repels a few random
# negatives. Hogwild-style: updates are applied in place without locking, which
# is the standard UMAP-SGD behaviour the shader mirrors. The negative-sample RNG
# matches the shader exactly (PCG hash seeded per edge/epoch), so the GPU output
# can be validated bit-for-bit against this reference.
# ----------------------------------------------------------------------------
.ggmlr_umap_sgd <- function(graph, embedding, a, b, n_epochs = 200L,
                            n_neg = 5L, alpha0 = 1.0, gamma = 1.0,
                            base_seed = 42L) {
  Y <- embedding                               # n x 2, modified in place
  n <- graph$n
  from <- graph$from + 1L; to <- graph$to + 1L # back to 1-based for R indexing
  ne   <- length(from)
  eps  <- 1e-3
  golden <- 2654435769                          # 0x9e3779b9

  clampg <- function(g) pmax(pmin(g, 4), -4)   # UMAP clamps gradients to [-4,4]

  for (epoch in seq_len(n_epochs)) {
    alpha <- alpha0 * (1 - (epoch - 1) / n_epochs)
    epoch_seed <- .ggmlr_u32(base_seed + (epoch - 1))   # host varies seed/epoch
    for (e in seq_len(ne)) {
      i <- from[e]; j <- to[e]
      dy  <- Y[i, ] - Y[j, ]
      d2  <- sum(dy * dy)

      # attraction
      if (d2 > 0) {
        coef <- (-2 * a * b * d2^(b - 1)) / (1 + a * d2^b)
        grad <- clampg(coef * dy) * alpha
        Y[i, ] <- Y[i, ] + grad
        Y[j, ] <- Y[j, ] - grad
      }

      # repulsion against n_neg random negatives (PCG RNG, mirrors the shader:
      # rng = epoch_seed ^ ((e-1) * golden), then pcg_hash each draw)
      rng <- .ggmlr_xor32(epoch_seed, .ggmlr_mul32(e - 1, golden))
      for (k in seq_len(n_neg)) {
        rng <- .ggmlr_pcg_hash(rng)
        c <- (rng %% n) + 1L                    # 0-based draw -> 1-based index
        if (c == i) next
        dy <- Y[i, ] - Y[c, ]
        d2 <- sum(dy * dy)
        if (d2 > 0) {
          coef <- (2 * gamma * b) / ((eps + d2) * (1 + a * d2^b))
          grad <- clampg(coef * dy) * alpha
        } else {
          grad <- c(4, 4) * alpha
        }
        Y[i, ] <- Y[i, ] + grad
      }
    }
  }
  Y
}

# ----------------------------------------------------------------------------
# GPU SGD: dispatch the per-edge layout shader (umap_sgd.comp) via Vulkan.
# Mirrors .ggmlr_umap_sgd exactly (same PCG RNG, same a/b/alpha schedule), so
# the output matches the CPU reference bit-for-bit for a given base_seed.
# Returns the optimised n x 2 embedding, or NULL if no Vulkan backend is live
# (caller falls back to the CPU loop).
# ----------------------------------------------------------------------------
.ggmlr_umap_sgd_gpu <- function(graph, embedding, a, b, n_epochs = 200L,
                                n_neg = 5L, alpha0 = 1.0, gamma = 1.0,
                                base_seed = 42L) {
  # ag_device("gpu") initialises the backend and throws if no GPU is present;
  # swallow that so we fall back to the CPU loop instead of erroring out.
  ok <- tryCatch({ ag_device("gpu"); TRUE }, error = function(e) FALSE)
  if (!ok) return(NULL)
  backend <- .ag_device_state$backend
  if (is.null(backend) || !ggml_vulkan_is_backend(backend)) return(NULL)

  n  <- graph$n
  ne <- length(graph$from)
  coords  <- as.double(t(embedding))                 # [x0,y0,x1,y1,...]
  edges   <- as.integer(rbind(graph$from, graph$to)) # [from0,to0,...] (0-based)
  weights <- as.double(graph$weight)

  out <- .Call("R_ggml_umap_sgd", backend, coords, edges, weights,
               as.integer(n), as.integer(ne),
               as.integer(n_epochs), as.integer(n_neg),
               as.double(a), as.double(b),
               as.double(alpha0), as.double(gamma), as.integer(base_seed),
               PACKAGE = "ggmlR")

  matrix(out, n, 2L, byrow = TRUE)                   # undo [x,y,...] flattening
}

#' GPU-bound UMAP embedding (op = "umap")
#'
#' Lays a feature-by-cell matrix out in 2-D with UMAP. The kNN graph uses the
#' FNN kd-tree when available (exact, O(n log n), light on memory); without FNN
#' it falls back to a full distance matrix, computed on the GPU via the
#' \code{pairwise_dist.comp} shader (honest f32, sidestepping mul_mat's f16 path)
#' or on the CPU. The SGD layout optimisation runs on the GPU via
#' \code{umap_sgd.comp} under \code{backend = "vulkan"} (the default), falling
#' back to its exact CPU reference when the GPU is unavailable. The fuzzy
#' simplicial set in between stays on the CPU (sparse). \code{backend_dist} in
#' the metadata reports the kNN path actually taken (\code{"fnn"},
#' \code{"vulkan"}, or \code{"cpu"}).
#'
#' @param mat Dense numeric matrix, features x cells.
#' @param n_components Output dimensionality (UMAP is virtually always 2).
#' @param n_neighbors kNN graph size (default 15).
#' @param min_dist Minimum spacing of points in the embedding (default 0.1).
#' @param spread Scale of the embedding (default 1).
#' @param n_epochs SGD epochs (default 200).
#' @param gpu_neighbor_max_cells Cell ceiling for the GPU distance shader.
#'   \code{NULL} (default) derives it from free VRAM; a positive integer forces
#'   an explicit cap; above the ceiling the distance phase falls back to the CPU.
#' @param backend \code{"vulkan"} (default; both GPU shaders, with per-phase CPU
#'   fallback) or \code{"cpu"} (force the CPU reference for both phases).
#' @return A \code{\link{ggml_result}} whose \code{embedding} is cells x
#'   \code{n_components}. \code{metadata} records the a/b curve parameters and the
#'   per-phase backend (\code{backend_dist}, \code{backend_sgd}); the summary
#'   \code{backend} is \code{"vulkan"} only when both phases ran on the GPU.
#' @keywords internal
.ggmlr_umap_gpu <- function(mat, n_components = 2L, n_neighbors = 15L,
                            min_dist = 0.1, spread = 1, n_epochs = 200L,
                            gpu_neighbor_max_cells = NULL,
                            backend = c("vulkan", "cpu")) {
  backend <- match.arg(backend)
  storage.mode(mat) <- "double"
  t0 <- proc.time()[["elapsed"]]

  X <- t(mat)                                  # cells x features
  n <- nrow(X)

  # Phase 1+2: kNN graph + fuzzy simplicial set. kNN prefers the FNN kd-tree;
  # without FNN it builds the full distance matrix on the GPU (pairwise_dist.comp,
  # up to the VRAM-derived ceiling) or CPU. graph$dist_backend reports the path
  # taken ("fnn"/"vulkan"/"cpu"). Forcing backend = "cpu" sets the ceiling to 0 so
  # the distance fallback never touches the GPU shader (FNN, a CPU method, may
  # still run); NULL (the default under vulkan) sizes the ceiling from free VRAM.
  # The fuzzy set is sparse. Timed separately from the GPU SGD below.
  dist_ceiling <- if (backend == "cpu") 0L else gpu_neighbor_max_cells
  t_g0  <- proc.time()[["elapsed"]]
  graph <- .ggmlr_umap_fuzzy_graph(X, n_neighbors = n_neighbors,
                                   gpu_neighbor_max_cells = dist_ceiling)
  ab    <- .ggmlr_umap_find_ab(spread = spread, min_dist = min_dist)
  t_graph <- proc.time()[["elapsed"]] - t_g0
  backend_dist <- graph$dist_backend %||% "cpu"

  # init from a small random spectral-like layout (random is enough here)
  Y0 <- matrix(stats::rnorm(n * n_components, sd = 1e-4), n, n_components)

  # Phase 3 (GPU): SGD layout via umap_sgd.comp when a backend is live, else the
  # CPU reference. The GPU path mirrors the CPU numerics (same PCG RNG /
  # schedule); if no Vulkan backend is available it returns NULL and we fall back.
  t_s0 <- proc.time()[["elapsed"]]
  Y <- NULL
  backend_sgd <- "cpu"
  if (backend == "vulkan") {
    Y <- .ggmlr_umap_sgd_gpu(graph, Y0, a = ab[["a"]], b = ab[["b"]],
                             n_epochs = n_epochs)
    if (!is.null(Y)) backend_sgd <- "vulkan"
  }
  if (is.null(Y)) {
    Y <- .ggmlr_umap_sgd(graph, Y0, a = ab[["a"]], b = ab[["b"]],
                         n_epochs = n_epochs)
  }
  t_sgd <- proc.time()[["elapsed"]] - t_s0

  rownames(Y) <- colnames(mat)
  colnames(Y) <- paste0("UMAP_", seq_len(n_components))

  # summary backend = "vulkan" only when BOTH shaders ran on the GPU
  backend_used <- if (backend_dist == "vulkan" && backend_sgd == "vulkan")
    "vulkan" else "cpu"

  ggml_result(
    embedding = Y,
    metadata  = list(backend = backend_used,
                     backend_dist = backend_dist, backend_sgd = backend_sgd,
                     a = ab[["a"]], b = ab[["b"]],
                     n_neighbors = n_neighbors, n_edges = length(graph$from)),
    # graph = CPU kNN/fuzzy build (GPU distance) ; sgd = layout (GPU or CPU)
    timings   = c(total = proc.time()[["elapsed"]] - t0,
                  graph = t_graph, sgd = t_sgd)
  )
}

# register op = "umap" -> UMAP engine (both phases GPU by default, CPU fallback)
.ggmlr_register_op(
  "umap", engine = .ggmlr_umap_gpu,
  params = character(0),
  desc   = "UMAP 2-D embedding (pairwise_dist.comp + umap_sgd.comp on GPU; CPU fallback)"
)

#' kNN + shared-nearest-neighbour graphs (op = "neighbors")
#'
#' Builds the two neighbour graphs that Seurat's \code{FindNeighbors} produces:
#' a binary kNN graph and a shared-nearest-neighbour (SNN) graph whose edge
#' weights are the Jaccard overlap of the two endpoints' neighbourhoods. The kNN
#' search uses the FNN kd-tree when available, otherwise the pairwise distance
#' matrix (GPU \code{pairwise_dist.comp} or CPU). The SNN step is sparse matrix
#' arithmetic. Matches Seurat numerically: SNN weight = |N(i) cap N(j)| /
#' |N(i) cup N(j)| with the neighbour set including the point itself, pruned
#' below \code{prune_snn}.
#'
#' @param mat Dense numeric matrix, features x cells (e.g. PC coordinates).
#' @param n_neighbors kNN graph size (Seurat default 20).
#' @param prune_snn Drop SNN edges with Jaccard below this (Seurat default 1/15).
#' @param gpu_neighbor_max_cells Cell ceiling for the GPU distance fallback;
#'   \code{NULL} derives it from free VRAM.
#' @param backend \code{"vulkan"} or \code{"cpu"} (dispatch resolves "auto").
#' @return A \code{\link{ggml_result}} with \code{metadata$kind = "graph"} and
#'   the two graphs (\code{nn}, \code{snn}) as sparse matrices in
#'   \code{metadata}; \code{embedding} is the SNN graph (the one clustering uses).
#' @keywords internal
.ggmlr_neighbors_gpu <- function(mat, n_neighbors = 20L, prune_snn = 1 / 15,
                                 gpu_neighbor_max_cells = NULL,
                                 backend = c("vulkan", "cpu")) {
  backend <- match.arg(backend)
  .ggmlr_need_pkg("Matrix", "building neighbour graphs (op = \"neighbors\")")
  storage.mode(mat) <- "double"
  t0 <- proc.time()[["elapsed"]]

  X <- t(mat)                                  # cells x features
  n <- nrow(X)
  if (n < 2L || ncol(X) < 1L) {
    stop("op = \"neighbors\" needs at least 2 cells and 1 feature; got a ",
         n, " x ", ncol(X), " matrix. Build neighbours from a feature space ",
         "such as PCA (reduction = \"pca\"), not an empty layer.", call. = FALSE)
  }
  k <- as.integer(min(n_neighbors, n - 1L))
  dist_ceiling <- if (backend == "cpu") 0L else gpu_neighbor_max_cells

  knn <- .ggmlr_umap_knn(X, k, dist_ceiling)   # idx (n x k), dist, backend
  idx <- knn$idx

  # binary kNN adjacency, with self included (Seurat counts the point itself in
  # its own neighbourhood, so |N| = k + 1 and the Jaccard denominators match).
  ii  <- c(seq_len(n), rep.int(seq_len(n), k))
  jj  <- c(seq_len(n), as.integer(idx))
  # sparseMatrix sums duplicated (i,j); cap entries at 1 so the graph is binary.
  KNN <- Matrix::sparseMatrix(i = ii, j = jj, x = 1, dims = c(n, n))
  KNN@x[] <- 1                                 # collapse any accidental dups

  # SNN: overlap = |N(i) cap N(j)| via KNN %*% t(KNN); Jaccard normalises by the
  # union |N(i)| + |N(j)| - overlap. Both neighbourhoods have size k + 1.
  overlap <- Matrix::tcrossprod(KNN)           # n x n counts of shared neighbours
  nsize   <- k + 1L
  snn     <- overlap / (2 * nsize - overlap)   # Jaccard (elementwise on non-zeros)
  snn     <- Matrix::drop0(snn * (snn >= prune_snn))  # prune weak edges

  cell_names <- colnames(mat)
  if (!is.null(cell_names)) {
    dimnames(KNN) <- list(cell_names, cell_names)
    dimnames(snn) <- list(cell_names, cell_names)
  }

  ggml_result(
    embedding = snn,                           # the graph clustering operates on
    metadata  = list(kind = "graph", backend = knn$backend,
                     nn = KNN, snn = snn,
                     n_neighbors = k, prune_snn = prune_snn,
                     n_snn_edges = length(snn@x)),
    timings   = c(total = proc.time()[["elapsed"]] - t0)
  )
}

# register op = "neighbors" -> Seurat-compatible kNN + SNN graphs
.ggmlr_register_op(
  "neighbors", engine = .ggmlr_neighbors_gpu,
  params = character(0),
  desc   = "kNN + shared-NN (SNN/Jaccard) graphs, Seurat FindNeighbors-compatible"
)
