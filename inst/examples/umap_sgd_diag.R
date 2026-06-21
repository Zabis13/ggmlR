#!/usr/bin/env Rscript
# Diagnostic: isolate the UMAP GPU shader from Hogwild races.
#
# With one epoch, n_neg = 0 (attraction only), and a conflict-free edge set
# (a perfect matching — no two edges share a vertex), the parallel GPU dispatch
# has NO write collisions, so it must reproduce the sequential CPU reference to
# within float32 precision. A large diff here means a real numerics/layout bug;
# a small diff here (but large with many epochs / a dense graph) means the
# earlier mismatch was just Hogwild non-determinism, as expected.

suppressMessages(library(ggmlR))

if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE)))
  stop("No Vulkan GPU available.", call. = FALSE)

set.seed(1)
n  <- 20L
Y0 <- matrix(stats::rnorm(n * 2, sd = 1), n, 2)   # sd=1 so steps are visible

# conflict-free graph: matching (0-1, 2-3, 4-5, ...), every vertex used once
from <- seq(0L, n - 2L, by = 2L)
to   <- seq(1L, n - 1L, by = 2L)
g <- list(from = from, to = to, weight = rep(1, length(from)), n = n)

ab <- c(a = 1.577, b = 0.895)

run <- function(label, n_epochs, n_neg) {
  Yg <- ggmlR:::.ggmlr_umap_sgd_gpu(g, Y0, a = ab[["a"]], b = ab[["b"]],
                                    n_epochs = n_epochs, n_neg = n_neg,
                                    base_seed = 42L)
  Yc <- ggmlR:::.ggmlr_umap_sgd(g, Y0, a = ab[["a"]], b = ab[["b"]],
                                n_epochs = n_epochs, n_neg = n_neg,
                                base_seed = 42L)
  cat(sprintf("%-28s max|GPU-CPU| = %.3e\n", label, max(abs(Yg - Yc))))
  invisible(list(g = Yg, c = Yc))
}

cat("Conflict-free matching graph:\n")
r <- run("1 epoch, n_neg=0 (attract)", 1L, 0L)   # must be ~float eps
run("1 epoch, n_neg=5 (full)",   1L, 5L)          # RNG path, still no i-collisions
run("10 epochs, n_neg=0",        10L, 0L)

# show the first few rows so we can eyeball where it diverges, if it does
cat("\nGPU[1:4,]:\n"); print(round(r$g[1:4, ], 5))
cat("CPU[1:4,]:\n"); print(round(r$c[1:4, ], 5))
