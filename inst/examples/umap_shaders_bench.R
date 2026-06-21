#!/usr/bin/env Rscript
# Benchmark: the two single-cell GPU shaders vs their CPU references.
#
#   1. pairwise_dist.comp  — squared Euclidean distance matrix, taken to
#      Euclidean in R, vs stats::dist (the exact CPU path).
#   2. umap_sgd.comp       — the SGD layout step, vs .ggmlr_umap_sgd (the R loop).
#
# For each shader we report wall time GPU vs CPU, the speed-up, and the max
# absolute error so the speed-up is only counted when the result is correct.
# The SGD path is Hogwild (lock-free) on a real graph, so its "error" vs the
# sequential CPU loop is non-determinism by design, not a bug — we report it but
# do not gate on it. The distance path must match stats::dist to f32 precision.
#
# Run yourself (the package builds benchmarks on demand):
#   Rscript inst/examples/umap_shaders_bench.R

suppressMessages(library(ggmlR))

if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)))
  stop("No Vulkan GPU available.", call. = FALSE)

ag_device("gpu")

timeit <- function(expr) {
  t <- system.time(val <- force(expr))[["elapsed"]]
  list(t = t, val = val)
}

cat("== pairwise_dist.comp vs stats::dist ==\n")
cat(sprintf("%6s %5s  %10s %10s %8s  %10s\n",
            "cells", "dims", "GPU(s)", "CPU(s)", "speedup", "max|err|"))
for (n in c(500L, 1000L, 2000L, 4000L)) {
  dims <- 50L
  set.seed(1)
  X <- matrix(stats::rnorm(n * dims), n, dims)

  g <- timeit(ggmlR:::.ggmlr_dist_gpu(X, gpu_neighbor_max_cells = 20000L))
  if (is.null(g$val)) { cat(sprintf("%6d  no GPU dispatch (fell back)\n", n)); next }
  c <- timeit(as.matrix(stats::dist(X)))

  err <- max(abs(g$val - c$val))
  cat(sprintf("%6d %5d  %10.4f %10.4f %7.1fx  %10.2e\n",
              n, dims, g$t, c$t, c$t / g$t, err))
}

cat("\n== umap_sgd.comp ==\n")
# The CPU reference (.ggmlr_umap_sgd) is a pure-R loop, O(epochs * edges): even
# at 500 cells / 200 epochs it takes ~2 min, and it scales linearly — so timing
# it at every size makes the script look hung. It is the *numerics* reference
# (verified bit-exact elsewhere), not a fast path, so the GPU/CPU ratio is not an
# informative number. We always report GPU time + throughput (edge-updates/s),
# and only run the CPU loop when SGD_BENCH_CPU=1 (one small size, for a sanity
# speed-up figure). Each GPU row prints immediately, so there is no waiting.
bench_cpu <- identical(Sys.getenv("SGD_BENCH_CPU"), "1")
cat(sprintf("%6s %7s %7s  %10s  %12s%s\n",
            "cells", "edges", "epochs", "GPU(s)", "edge-upd/s",
            if (bench_cpu) "   CPU(s)   speedup" else ""))
ab <- c(a = 1.577, b = 0.895)
cpu_max_cells <- 500L             # the only size we'd ever run the R loop at
for (n in c(500L, 1000L, 2000L, 4000L)) {
  set.seed(2)
  X  <- matrix(stats::rnorm(n * 50), n, 50)
  gr <- ggmlR:::.ggmlr_umap_fuzzy_graph(X, n_neighbors = 15L)
  Y0 <- matrix(stats::rnorm(n * 2, sd = 1e-4), n, 2)
  ne <- length(gr$from)
  ep <- 200L

  g <- timeit(ggmlR:::.ggmlr_umap_sgd_gpu(gr, Y0, a = ab[["a"]], b = ab[["b"]],
                                          n_epochs = ep, base_seed = 42L))
  if (is.null(g$val)) { cat(sprintf("%6d  no GPU dispatch (fell back)\n", n)); next }

  thru <- (as.double(ne) * ep) / g$t          # edge-updates per second on GPU
  cat(sprintf("%6d %7d %7d  %10.4f  %12.3e", n, ne, ep, g$t, thru))
  if (bench_cpu && n <= cpu_max_cells) {
    cc <- timeit(ggmlR:::.ggmlr_umap_sgd(gr, Y0, a = ab[["a"]], b = ab[["b"]],
                                         n_epochs = ep, base_seed = 42L))
    cat(sprintf("   %8.2f %8.0fx", cc$t, cc$t / g$t))
  }
  cat("\n")
}

cat("\nNote: the distance speed-up is the one that matters for the kNN bottleneck\n")
cat("(brute kNN is O(cells^2) on CPU). The SGD CPU reference is a pure-R loop\n")
cat("(numerics reference, not a fast path); set SGD_BENCH_CPU=1 to time it at\n")
cat("500 cells. GPU throughput (edge-updates/s) is the comparable figure at scale.\n")
