#!/usr/bin/env Rscript
#
# How many scheduler splits does a transformer attention block cost when its
# backward runs on the CPU?
#
# FLASH_ATTN_BACK has no Vulkan shader (the op appears nowhere under
# src/ggml-vulkan/, so supports_op falls through to false), and the scheduler
# moves it to the CPU. This answers whether that fallback stays local to the one
# op or drags neighbouring nodes with it -- the way mamba once produced two
# splits where one was enough. Every split is a host<->device round trip on
# every training step.
#
# Run:  Rscript inst/scripts/measure_attn_splits.R
# Env:  GGMLR_LOG_DEVICE=1 also prints which ops the Vulkan backend rejected.

suppressMessages(library(ggmlR))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() == 0L) {
  cat("No Vulkan device -- nothing to measure (splits are a mixed-backend thing).\n")
  quit(save = "no")
}

# A single attention block's shapes, small enough to build fast but with the
# head structure that matters (GQA, batch > 1).
DK <- 64L; DV <- 64L; N <- 32L; M <- 32L; H <- 8L; Hkv <- 4L; B <- 2L

# Two things this stand must get right, both learned the hard way:
#
#  1. ggml_backend_sched_new() appends its own CPU backend (ggml requires CPU
#     last), so pass ONLY the GPU. Passing list(gpu, cpu) yields [Vulkan,CPU,CPU].
#  2. The weights must live in a GPU buffer before the graph is built. The
#     scheduler places an op by where its sources already are, so a graph whose
#     inputs sit in host memory stays on the CPU no matter what supports_op says
#     -- an earlier version of this script measured exactly that artefact.
build <- function(ctx, gpu, nblocks, train) {
  qs <- vector("list", nblocks)
  ks <- vector("list", nblocks)
  vs <- vector("list", nblocks)

  for (i in seq_len(nblocks)) {
    qs[[i]] <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, H, B)
    ks[[i]] <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, Hkv, B)
    vs[[i]] <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, Hkv, B)

    ggml_set_input(qs[[i]]); ggml_set_input(ks[[i]]); ggml_set_input(vs[[i]])
    if (train) {
      ggml_set_param(qs[[i]]); ggml_set_param(ks[[i]]); ggml_set_param(vs[[i]])
    }
  }

  # Weights on the GPU, the way a compiled model holds them (see nn_model.R).
  # Returned to the caller so it can be freed: a backend buffer is not owned by
  # the context, so ggml_free() does not release it, and leaking one per call
  # skews whichever measurement runs last.
  buf <- ggml_backend_alloc_ctx_tensors(ctx, gpu)
  for (i in seq_len(nblocks)) {
    ggml_backend_tensor_set_data(qs[[i]], runif(DK * N * H * B, -1, 1))
    ggml_backend_tensor_set_data(ks[[i]], runif(DK * M * Hkv * B, -1, 1))
    ggml_backend_tensor_set_data(vs[[i]], runif(DV * M * Hkv * B, -1, 1))
  }

  # Independent blocks, summed at the end: enough to see whether the fallback
  # cost is paid once or once per block.
  outs <- lapply(seq_len(nblocks), function(i)
    ggml_sum(ctx, ggml_flash_attn_ext(ctx, qs[[i]], ks[[i]], vs[[i]], NULL, 1 / sqrt(DK))))

  total <- outs[[1]]
  if (nblocks > 1) {
    for (i in 2:nblocks) total <- ggml_add(ctx, total, outs[[i]])
  }

  if (!train) {
    return(list(graph = ggml_build_forward_expand(ctx, total), buf = buf))
  }

  ggml_set_loss(total)
  graph <- ggml_build_forward_expand_grads(ctx, total)
  ggml_build_backward_expand(ctx, graph)
  list(graph = graph, buf = buf)
}

measure <- function(label, nblocks, train) {
  gpu <- ggml_vulkan_init(0L)
  on.exit(ggml_vulkan_free(gpu), add = TRUE)

  # Only the GPU: the wrapper adds the CPU fallback itself.
  sched <- ggml_backend_sched_new(list(gpu), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ctx <- ggml_init(1024 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  built <- build(ctx, gpu, nblocks, train)
  graph <- built$graph
  on.exit(ggml_backend_buffer_free(built$buf), add = TRUE, after = FALSE)

  ggml_backend_sched_alloc_graph(sched, graph)
  if (train) ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  n_nodes  <- ggml_graph_n_nodes(graph)
  n_splits <- ggml_backend_sched_get_n_splits(sched)

  # Count nodes that actually ran off the GPU.
  n_cpu <- 0L
  for (i in seq_len(n_nodes)) {
    be <- ggml_backend_sched_get_tensor_backend(sched, ggml_graph_node(graph, i - 1L))
    if (!is.null(be) && ggml_backend_name(be) == "CPU") n_cpu <- n_cpu + 1L
  }

  cat(sprintf("%-26s nodes=%3d  splits=%d  on_cpu=%d\n", label, n_nodes, n_splits, n_cpu))
  invisible(c(nodes = n_nodes, splits = n_splits, on_cpu = n_cpu))
}

cat("Attention block, DK=", DK, " N=", N, " M=", M, " H=", H, " Hkv=", Hkv,
    " B=", B, "\n\n", sep = "")

# Measure at several depths: one block alone cannot tell a fixed cost apart
# from a per-block one, and that difference is the whole argument for a shader.
depths <- c(1L, 2L, 4L)
res <- lapply(depths, function(nb) {
  f <- measure(sprintf("%d block(s), forward:", nb), nb, FALSE)
  b <- measure(sprintf("%d block(s), fwd+bwd:", nb), nb, TRUE)
  c(blocks = nb, fwd = f[["splits"]], bwd = b[["splits"]], on_cpu = b[["on_cpu"]])
})

cat("\n")
blocks <- vapply(res, function(r) r[["blocks"]], numeric(1))
bwd    <- vapply(res, function(r) r[["bwd"]],    numeric(1))
fwd    <- vapply(res, function(r) r[["fwd"]],    numeric(1))
on_cpu <- vapply(res, function(r) r[["on_cpu"]], numeric(1))

# Fit splits = a * blocks + b over the measured depths rather than reading one
# point: the interesting question is the SLOPE (does the cost grow with depth?),
# and a single measurement cannot tell a per-block cost from a fixed one.
slope     <- if (length(unique(blocks)) > 1L) coef(lm(bwd ~ blocks))[[2]] else NA_real_
cpu_slope <- if (length(unique(blocks)) > 1L) coef(lm(on_cpu ~ blocks))[[2]] else NA_real_

cat(sprintf("forward:  %s splits for %s blocks -- %s\n",
            paste(fwd, collapse = "/"), paste(blocks, collapse = "/"),
            if (length(unique(fwd)) == 1L) "flat, the GPU takes the whole pass"
            else "grows with depth"))
cat(sprintf("fwd+bwd:  %s splits, %s nodes left on the CPU\n",
            paste(bwd, collapse = "/"), paste(on_cpu, collapse = "/")))

if (!is.na(slope) && slope >= 0.5) {
  cat(sprintf("\nThe backward fallback costs ~%.0f split(s) and ~%.0f CPU node(s) PER BLOCK.\n",
              slope, cpu_slope))
  cat(sprintf("A %d-layer model would pay ~%.0f splits, against %d for the forward pass.\n",
              12L, slope * 12, fwd[1]))
  cat("Each split is a host<->device round trip on every training step, so the\n")
  cat("cost grows with depth: a Vulkan shader for FLASH_ATTN_BACK pays for\n")
  cat("itself well beyond this one op's own runtime.\n")
} else if (!is.na(slope)) {
  cat("\nThe backward fallback does not grow with depth -- a fixed cost.\n")
}
cat("\nRe-run with GGMLR_LOG_DEVICE=1 to see which ops the GPU rejected.\n")
