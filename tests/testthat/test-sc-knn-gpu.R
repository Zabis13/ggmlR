# Fused GPU k-NN (knn_tiled.comp) via knn_backend = "vulkan". The shader computes
# each row's k nearest in honest f32 without materialising the n x n distance
# matrix. It is an EXACT k-NN, so the neighbour SETS must match a CPU kd-tree
# (FNN) or brute-force order(); the per-row order can differ only on exact-distance
# ties. All GPU assertions skip gracefully without a Vulkan device.

skip_if_not_installed("Matrix")

has_gpu <- isTRUE(tryCatch(
  ggml_vulkan_available() && ggml_vulkan_device_count() > 0L,
  error = function(e) FALSE))

# the GPU path flips the global device to "gpu"; restore it when this file ends
# so the state does not leak into later test files (see helper-device.R).
withr::defer(ag_device("cpu"), teardown_env())

# a reproducible clustered feature space (cells x dims), the regime op="neighbors"
# runs in: a handful of PCA-like dimensions with clear structure.
make_X <- function(n = 300L, dims = 10L, centres = 6L, seed = 13L) {
  set.seed(seed)
  mu <- matrix(rnorm(centres * dims, sd = 5), centres, dims)
  lab <- sample.int(centres, n, replace = TRUE)
  mu[lab, ] + matrix(rnorm(n * dims), n, dims)
}

# brute-force exact kNN reference (idx 1-based, self excluded, sorted ascending)
ref_knn <- function(X, k) {
  D <- as.matrix(dist(X))
  t(apply(D, 1L, function(row) order(row)[-1L][seq_len(k)]))
}

test_that(".ggmlr_knn_gpu returns NULL when no GPU (pure fallback contract)", {
  skip_if(has_gpu, "GPU present; this checks the no-GPU contract")
  X <- make_X(50L, 5L)
  expect_null(ggmlR:::.ggmlr_knn_gpu(X, k = 10L))
})

test_that(".ggmlr_umap_knn falls back to FNN/CPU when GPU declines", {
  # knn_backend = "vulkan" but (here) no usable GPU -> must still return a valid
  # kNN via FNN or the CPU distance path, never error.
  skip_if_not_installed("FNN")
  skip_if(has_gpu, "GPU present; this checks the decline-to-CPU fallback")
  X   <- make_X(120L, 8L)
  out <- ggmlR:::.ggmlr_umap_knn(X, n_neighbors = 15L, knn_backend = "vulkan")
  expect_true(out$backend %in% c("fnn", "cpu", "vulkan"))
  expect_equal(dim(out$idx), c(nrow(X), 15L))
})

test_that("GPU k-NN matches the exact reference (neighbour sets)", {
  skip_if_not(has_gpu, "no Vulkan GPU")
  X <- make_X(300L, 10L)
  k <- 20L

  g <- ggmlR:::.ggmlr_knn_gpu(X, k = k)
  skip_if(is.null(g), "GPU declined the kNN (ceiling/capacity)")
  expect_equal(g$backend, "vulkan")
  expect_equal(dim(g$idx),  c(nrow(X), k))
  expect_equal(dim(g$dist), c(nrow(X), k))

  ref <- ref_knn(X, k)
  # neighbour SETS must match exactly (order may differ only on ties)
  agree <- vapply(seq_len(nrow(X)),
                  function(i) setequal(g$idx[i, ], ref[i, ]), logical(1))
  expect_true(all(agree))

  # distances must be ascending per row and match the true distances
  expect_true(all(apply(g$dist, 1L, function(r) !is.unsorted(r))))
  Dtrue <- as.matrix(dist(X))
  err <- max(vapply(seq_len(nrow(X)),
                    function(i) max(abs(g$dist[i, ] - Dtrue[i, g$idx[i, ]])),
                    numeric(1)))
  expect_lt(err, 1e-3)                     # honest f32 vs f64
})

test_that("GPU k-NN handles k below and at the pipeline capacity", {
  skip_if_not(has_gpu, "no Vulkan GPU")
  X <- make_X(200L, 6L)
  for (k in c(5L, 32L)) {
    g <- ggmlR:::.ggmlr_knn_gpu(X, k = k)
    skip_if(is.null(g), "GPU declined the kNN")
    ref   <- ref_knn(X, k)
    agree <- vapply(seq_len(nrow(X)),
                    function(i) setequal(g$idx[i, ], ref[i, ]), logical(1))
    expect_true(all(agree))
  }
})

test_that("k above the pipeline capacity (32) declines to fallback", {
  skip_if_not(has_gpu, "no Vulkan GPU")
  X <- make_X(100L, 6L)
  expect_null(ggmlR:::.ggmlr_knn_gpu(X, k = 33L))  # > capacity -> NULL (fall back)
})

test_that("op = 'neighbors' runs end-to-end with knn_backend = 'vulkan'", {
  skip_if_not(has_gpu, "no Vulkan GPU")
  set.seed(1)
  mat <- t(make_X(250L, 10L))                       # features x cells
  res <- ggmlR:::.ggmlr_neighbors_gpu(mat, n_neighbors = 20L,
                                      backend = "vulkan", knn_backend = "vulkan")
  expect_equal(res$metadata$kind, "graph")
  expect_false(is.null(res$metadata$nn))
  expect_false(is.null(res$metadata$snn))
})
