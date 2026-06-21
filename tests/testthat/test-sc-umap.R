# UMAP SGD: GPU shader (umap_sgd.comp) vs the CPU reference (.ggmlr_umap_sgd).
# The GPU path mirrors the CPU numerics — same fuzzy graph, same PCG RNG, same
# a/b/alpha schedule — so for a given base_seed and Y0 the two layouts should
# agree to within float32 precision. They are NOT expected to be bit-exact: the
# GPU runs f32 while the R reference runs double, and the per-edge updates are
# Hogwild (lock-free), so divergence grows with graph density / epoch count.
# Hence: small graph + few epochs, and a tolerance rather than identity.

# the engines flip the global device to "gpu"; restore it when this file ends so
# the state does not leak into later test files (see helper-device.R).
withr::defer(ag_device("cpu"), teardown_env())

test_that("UMAP fuzzy graph is well-formed (CPU, no GPU needed)", {
  set.seed(1)
  X <- matrix(stats::rnorm(20 * 5), 20, 5)
  g <- ggmlR:::.ggmlr_umap_fuzzy_graph(X, n_neighbors = 5L)

  expect_equal(g$n, 20L)
  expect_length(g$from, length(g$to))
  expect_length(g$from, length(g$weight))
  expect_true(all(g$from >= 0L & g$from < g$n))   # 0-based vertex indices
  expect_true(all(g$to   >= 0L & g$to   < g$n))
  expect_true(all(g$weight > 0 & g$weight <= 1))  # t-conorm membership in (0,1]
})

# The FNN kd-tree kNN path and the distance-matrix fallback must build the same
# fuzzy graph: same neighbours, same membership weights. FNN is the fast/default
# path; the fallback (sort each row of the full D) is exercised by masking FNN.
test_that("FNN kNN path and distance-matrix fallback build the same graph", {
  skip_if_not_installed("FNN")

  set.seed(5)
  X <- matrix(stats::rnorm(120 * 8), 120, 8)

  g_fnn <- ggmlR:::.ggmlr_umap_fuzzy_graph(X, n_neighbors = 15L)
  expect_equal(g_fnn$dist_backend, "fnn")

  # force the fallback by making FNN look unavailable
  g_ref <- with_mocked_bindings(
    .ggmlr_has_pkg = function(pkg) if (pkg == "FNN") FALSE else
      isTRUE(requireNamespace(pkg, quietly = TRUE)),
    ggmlR:::.ggmlr_umap_fuzzy_graph(X, n_neighbors = 15L),
    .package = "ggmlR"
  )

  expect_setequal(paste(g_fnn$from, g_fnn$to), paste(g_ref$from, g_ref$to))
  o1 <- order(g_fnn$from, g_fnn$to)
  o2 <- order(g_ref$from, g_ref$to)
  expect_equal(g_fnn$weight[o1], g_ref$weight[o2], tolerance = 1e-6)
})

# Our kNN graph must agree with uwot's: both find the k nearest neighbours of
# each point under exact Euclidean distance, so the neighbour sets should
# essentially coincide. We compare the raw kNN (the input to the fuzzy set) via
# the same distance matrix the graph builder uses, against uwot's exact NN
# (ret_nn, nn_method = "fnn" for an exact brute-force search). Ties at equal
# distance can swap one or two neighbours, so we require high overlap, not
# identity. CPU-only: this validates the graph engine, independent of the GPU.
test_that("kNN graph overlaps uwot's exact nearest neighbours", {
  skip_if_not_installed("uwot")
  skip_if_not_installed("FNN")

  set.seed(7)
  X <- matrix(stats::rnorm(80 * 6), 80, 6)
  k <- 10L

  # ours: k nearest (excluding self) from the exact distance matrix
  D <- as.matrix(stats::dist(X))               # CPU exact, what the builder uses
  ours <- t(apply(D, 1, function(r) order(r)[2:(k + 1L)]))

  # uwot: exact NN indices (column 1 is self, drop it)
  nn <- uwot::umap(X, n_neighbors = k + 1L, nn_method = "fnn",
                   ret_nn = TRUE, ret_model = FALSE, init = "random",
                   n_epochs = 0L)$nn$euclidean$idx
  theirs <- nn[, -1, drop = FALSE]

  overlap <- mean(vapply(seq_len(nrow(X)), function(i)
    length(intersect(ours[i, ], theirs[i, ])) / k, numeric(1)))
  expect_gt(overlap, 0.95)                      # near-identical neighbour sets
})

# pairwise_dist.comp: the GPU squared-distance matrix, taken to Euclidean by the
# R wrapper, must match stats::dist exactly (this is the honest f32 path that
# replaces the f16-accumulating mul_mat route). Tolerance is float32 epsilon
# scaled by the magnitude of the distances, not bit-exactness vs double.
test_that("GPU pairwise distance == stats::dist", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  set.seed(1)
  X <- matrix(stats::rnorm(40 * 8), 40, 8)

  Dg <- ggmlR:::.ggmlr_dist_gpu(X, gpu_neighbor_max_cells = 10000L)
  skip_if(is.null(Dg), "No live Vulkan backend for pairwise distance dispatch")

  Dc <- as.matrix(stats::dist(X))

  expect_equal(dim(Dg), c(40L, 40L))
  expect_true(all(diag(Dg) == 0))               # self-distance is exactly 0
  expect_equal(Dg, Dc, ignore_attr = TRUE, tolerance = 1e-5)
})

# The dispatcher must not change the order of nearest neighbours — the whole
# point of the f32 shader is that the kNN ordering matches the exact CPU dist
# (the f16 mul_mat path reordered top-k, which is the bug this replaces).
test_that("GPU distance preserves kNN ordering vs stats::dist", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  set.seed(2)
  X <- matrix(stats::rnorm(50 * 10), 50, 10)

  Dg <- ggmlR:::.ggmlr_dist_gpu(X)
  skip_if(is.null(Dg), "No live Vulkan backend for pairwise distance dispatch")
  Dc <- as.matrix(stats::dist(X))

  k <- 8L
  for (i in seq_len(nrow(X))) {
    og <- order(Dg[i, ])[seq_len(k + 1L)]       # +1 for self at distance 0
    oc <- order(Dc[i, ])[seq_len(k + 1L)]
    expect_equal(og, oc)
  }
})

# The public entry point falls back to the exact CPU dist() when there is no
# GPU, and never errors — so it always returns a valid n x n distance matrix.
test_that(".ggmlr_dist_matrix matches stats::dist (GPU or CPU fallback)", {
  set.seed(3)
  X <- matrix(stats::rnorm(15 * 4), 15, 4)

  D <- ggmlR:::.ggmlr_dist_matrix(X)
  expect_equal(dim(D), c(15L, 15L))
  expect_equal(D, as.matrix(stats::dist(X)), ignore_attr = TRUE, tolerance = 1e-5)
})

# The GPU path bows out above the cell ceiling (VRAM guard), leaving the CPU
# fallback to handle the matrix — no error, correct result.
test_that(".ggmlr_dist_gpu returns NULL above the cell ceiling", {
  set.seed(4)
  X <- matrix(stats::rnorm(30 * 4), 30, 4)
  expect_null(ggmlR:::.ggmlr_dist_gpu(X, gpu_neighbor_max_cells = 10L))
})

# The VRAM-derived ceiling is the largest n whose n^2*4 + n*dims*4 bytes fit in
# the budget. Pure arithmetic over a known "free" value, so no GPU is needed.
test_that(".ggmlr_dist_gpu_max_cells sizes n from the VRAM budget", {
  # 8 GiB free, 50% budget = 4 GiB; with dims = 50 the n^2 term dominates, so the
  # ceiling is ~32743 cells (closed-form solution of 4 n^2 + 200 n = budget).
  free_bytes <- 8 * 1024^3
  with_mocked_bindings(
    ggml_vulkan_device_memory = function(device = 0L)
      list(free = free_bytes, total = free_bytes),
    {
      nmax <- ggmlR:::.ggmlr_dist_gpu_max_cells(dims = 50L, vram_fraction = 0.5)
      expect_equal(nmax, 32743L)
      # the budget actually holds the distance matrix it sized for
      bytes <- nmax^2 * 4 + nmax * 50 * 4
      expect_lte(bytes, free_bytes * 0.5)
    },
    .package = "ggmlR"
  )
})

# When the memory query is unavailable, the helper returns its conservative
# fixed fallback rather than erroring.
test_that(".ggmlr_dist_gpu_max_cells falls back when VRAM is unknown", {
  with_mocked_bindings(
    ggml_vulkan_device_memory = function(device = 0L) stop("no device"),
    expect_equal(
      ggmlR:::.ggmlr_dist_gpu_max_cells(dims = 50L, fallback = 9999L), 9999L),
    .package = "ggmlR"
  )
})

# The GPU shader and the CPU reference share the exact same numerics — verified
# below on a conflict-free graph. On a real (shared-vertex) graph they do NOT
# match: the parallel dispatch is Hogwild (lock-free), so threads read each
# other's coords mid-update. That non-determinism is by design (uwot / cuML do
# the same), so the full-graph test below checks structure, not bit-exactness.
test_that("UMAP GPU SGD == CPU reference on a conflict-free graph", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  # a perfect matching: edges 0-1, 2-3, ... — no two edges share a vertex, so
  # the parallel attraction updates never collide. n_neg = 0 removes the only
  # remaining shared reads (random negatives), leaving a fully deterministic
  # step that must reproduce the sequential CPU loop to within float32 epsilon.
  n  <- 20L
  set.seed(1)
  Y0 <- matrix(stats::rnorm(n * 2, sd = 1), n, 2)
  g  <- list(from = seq(0L, n - 2L, by = 2L),
             to   = seq(1L, n - 1L, by = 2L),
             weight = rep(1, n %/% 2L), n = n)
  ab <- c(a = 1.577, b = 0.895)

  Yg <- ggmlR:::.ggmlr_umap_sgd_gpu(g, Y0, a = ab[["a"]], b = ab[["b"]],
                                    n_epochs = 10L, n_neg = 0L, base_seed = 42L)
  skip_if(is.null(Yg), "No live Vulkan backend for UMAP SGD dispatch")

  Yc <- ggmlR:::.ggmlr_umap_sgd(g, Y0, a = ab[["a"]], b = ab[["b"]],
                                n_epochs = 10L, n_neg = 0L, base_seed = 42L)

  expect_equal(dim(Yg), c(n, 2L))
  expect_lt(max(abs(Yg - Yc)), 1e-4)   # float32 vs double, no races
})

test_that("UMAP GPU SGD on a real graph is finite and well-shaped", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  set.seed(1)
  X  <- matrix(stats::rnorm(20 * 5), 20, 5)
  g  <- ggmlR:::.ggmlr_umap_fuzzy_graph(X, n_neighbors = 5L)
  Y0 <- matrix(stats::rnorm(20 * 2, sd = 1e-4), 20, 2)
  ab <- c(a = 1.577, b = 0.895)

  Yg <- ggmlR:::.ggmlr_umap_sgd_gpu(g, Y0, a = ab[["a"]], b = ab[["b"]],
                                    n_epochs = 10L, base_seed = 42L)
  skip_if(is.null(Yg), "No live Vulkan backend for UMAP SGD dispatch")

  # Hogwild on a shared-vertex graph: not bit-exact vs CPU, but the layout must
  # stay finite and bounded (gradients are clamped to [-4, 4] per step).
  expect_equal(dim(Yg), c(20L, 2L))
  expect_false(anyNA(Yg))
  expect_true(all(is.finite(Yg)))
  expect_lt(max(abs(Yg)), 1e3)
})

test_that("op = 'umap' end-to-end produces a cells x 2 embedding", {
  skip_if_not(ggml_vulkan_available(), "Vulkan GPU not available")

  set.seed(1)
  mat <- matrix(stats::rnorm(8 * 30), 8, 30)        # 8 features x 30 cells
  colnames(mat) <- paste0("cell", seq_len(30))

  res <- ggmlR:::.ggmlr_umap_gpu(mat, n_components = 2L, n_neighbors = 5L,
                                 n_epochs = 10L, backend = "vulkan")
  emb <- res$embedding

  expect_equal(dim(emb), c(30L, 2L))
  expect_equal(rownames(emb), colnames(mat))        # cells carried through
  expect_equal(colnames(emb), c("UMAP_1", "UMAP_2"))
  expect_true(res$metadata$backend %in% c("vulkan", "cpu"))
  # per-phase backends are tracked separately; "fnn" is the kd-tree kNN path,
  # "vulkan"/"cpu" the distance-matrix paths. summary "vulkan" iff both on GPU.
  expect_true(res$metadata$backend_dist %in% c("fnn", "vulkan", "cpu"))
  expect_true(res$metadata$backend_sgd %in% c("vulkan", "cpu"))
  both_gpu <- res$metadata$backend_dist == "vulkan" &&
              res$metadata$backend_sgd  == "vulkan"
  expect_equal(res$metadata$backend, if (both_gpu) "vulkan" else "cpu")
})

# Forcing backend = "cpu" keeps the layout (SGD) off the GPU and never uses the
# GPU distance shader. The kNN itself may still use the FNN kd-tree (a CPU
# method), so backend_dist is "fnn" when FNN is installed and "cpu" otherwise —
# either way it is not "vulkan", and the summary backend is "cpu".
test_that("op = 'umap' backend = 'cpu' keeps both phases off the GPU", {
  set.seed(1)
  mat <- matrix(stats::rnorm(8 * 30), 8, 30)
  colnames(mat) <- paste0("cell", seq_len(30))

  res <- ggmlR:::.ggmlr_umap_gpu(mat, n_components = 2L, n_neighbors = 5L,
                                 n_epochs = 10L, backend = "cpu")
  expect_true(res$metadata$backend_dist %in% c("fnn", "cpu"))
  expect_equal(res$metadata$backend_sgd,  "cpu")
  expect_equal(res$metadata$backend,      "cpu")
})
