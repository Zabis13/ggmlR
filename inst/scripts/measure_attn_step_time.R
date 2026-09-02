#!/usr/bin/env Rscript
#
# What do the scheduler splits of an attention backward actually cost in seconds?
#
# measure_attn_splits.R establishes the shape of the problem: the forward pass
# is one split on the GPU at any depth, while the backward pays 2 splits and 4
# CPU nodes per block because FLASH_ATTN_BACK has no Vulkan shader. That is a
# count, not a cost. This turns it into wall-clock time.
#
# The fallback cannot simply be switched off -- there is no shader to switch to
# -- so instead of a with/without comparison this measures three reference
# points and lets them bracket the answer:
#
#   forward, GPU        the ideal: one split, everything resident
#   fwd+bwd, GPU sched  what training actually does today (2n splits)
#   fwd+bwd, CPU only   one split, but no GPU at all
#
# The telling comparison is the last two. If CPU-only training is no slower --
# or faster -- than the "GPU" path, then the round trips are eating the whole
# benefit of running the forward on the GPU, and a shader for FLASH_ATTN_BACK
# is not an optimisation but the thing that makes GPU attention training worth
# doing at all.
#
# Run:  Rscript inst/scripts/measure_attn_step_time.R
# Env:  GGMLR_ATTN_BENCH_REPS  iterations per timing (default 20)
#       GGMLR_ATTN_BENCH_BLOCKS  comma-separated depths (default 1,2,4)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_ATTN_BENCH_REPS", "50"))
n_batches <- as.integer(Sys.getenv("GGMLR_ATTN_BENCH_BATCHES", "5"))
# Depths up to 16: real models are far deeper than a toy benchmark (GPT-2 small
# and BERT-base are 12 layers, LLaMA-7B is 32, larger ones 40-96), so measuring
# only 1-4 blocks says little about where the GPU path actually stands. 16 is
# past the smallest real models without making a run take minutes; whether the
# advantage keeps growing or flattens between 4 and 16 is the thing to read.
depths <- as.integer(strsplit(Sys.getenv("GGMLR_ATTN_BENCH_BLOCKS", "1,2,4,8,12,16"), ",")[[1]])

# A realistic-ish attention block. Big enough that the GPU has real work to do,
# small enough to iterate quickly.
DK <- 64L; DV <- 64L; N <- 128L; M <- 128L; H <- 8L; Hkv <- 4L; B <- 4L

have_gpu <- ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

# Build nblocks independent attention blocks summed into one scalar. When
# `backend` is given the weights are allocated there, which is what decides
# where the scheduler puts the graph (see measure_attn_splits.R for that trap).
build <- function(ctx, backend, nblocks, train, bwd_only = FALSE) {
  qs <- vector("list", nblocks); ks <- qs; vs <- qs

  for (i in seq_len(nblocks)) {
    qs[[i]] <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, H, B)
    ks[[i]] <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, Hkv, B)
    vs[[i]] <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, Hkv, B)
    ggml_set_input(qs[[i]]); ggml_set_input(ks[[i]]); ggml_set_input(vs[[i]])
    if (train) {
      ggml_set_param(qs[[i]]); ggml_set_param(ks[[i]]); ggml_set_param(vs[[i]])
    }
  }

  # Returned to the caller, which frees it. Dropping the handle here leaks one
  # allocation per call -- about 1.1 GB of device memory over a full sweep --
  # and the deepest case, measured last, pays for every earlier one in
  # allocator pressure and fragmentation.
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  for (i in seq_len(nblocks)) {
    ggml_backend_tensor_set_data(qs[[i]], runif(DK * N * H * B, -1, 1))
    ggml_backend_tensor_set_data(ks[[i]], runif(DK * M * Hkv * B, -1, 1))
    ggml_backend_tensor_set_data(vs[[i]], runif(DV * M * Hkv * B, -1, 1))
  }

  # Backward node ALONE: ggml_flash_attn_back called directly, with no forward
  # in the graph. 'backward adds' elsewhere is a subtraction (train minus
  # forward), which folds in the loss node, the graph reset and any scheduling
  # difference; this isolates the shader itself.
  if (bwd_only) {
    ds <- lapply(seq_len(nblocks), function(i) {
      d <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, H, N, B)
      ggml_set_input(d)
      d
    })
    bks <- lapply(seq_len(nblocks), function(i)
      ggml_flash_attn_back(ctx, qs[[i]], ks[[i]], vs[[i]], NULL, ds[[i]], 1 / sqrt(DK)))
    # ggml_build_forward_expand takes one tensor, so the blocks have to be
    # chained into a single root. Chain them through ONE-ELEMENT VIEWS, not
    # ggml_sum: summing the packed gradient reduces 0.52M elements per block, so
    # the closing node grew with depth and eventually dwarfed what it was meant
    # to measure -- at 16 blocks this column read 170 ms against 117 ms for the
    # full forward+backward, i.e. the part came out larger than the whole. A
    # view costs nothing and still keeps every backward node in the graph.
    root <- ggml_view_1d(ctx, bks[[1]], 1, 0)
    if (nblocks > 1) {
      for (i in 2:nblocks) root <- ggml_add(ctx, root, ggml_view_1d(ctx, bks[[i]], 1, 0))
    }
    return(list(graph = ggml_build_forward_expand(ctx, root), buf = buf))
  }

  outs <- lapply(seq_len(nblocks), function(i)
    ggml_sum(ctx, ggml_flash_attn_ext(ctx, qs[[i]], ks[[i]], vs[[i]], NULL, 1 / sqrt(DK))))

  total <- outs[[1]]
  if (nblocks > 1) for (i in 2:nblocks) total <- ggml_add(ctx, total, outs[[i]])

  if (!train) return(list(graph = ggml_build_forward_expand(ctx, total), buf = buf))

  ggml_set_loss(total)
  graph <- ggml_build_forward_expand_grads(ctx, total)
  ggml_build_backward_expand(ctx, graph)
  list(graph = graph, buf = buf)
}

# One timing run. `mode` is "gpu" (Vulkan first, CPU fallback added by the
# wrapper) or "cpu" (CPU only).
time_it <- function(mode, nblocks, train, bwd_only = FALSE) {
  if (mode == "gpu") {
    backend <- ggml_vulkan_init(0L)
    on.exit(ggml_vulkan_free(backend), add = TRUE)
  } else {
    backend <- ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(backend, 2L)
    on.exit(ggml_backend_free(backend), add = TRUE)
  }

  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ctx <- ggml_init(2048 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  built <- build(ctx, backend, nblocks, train, bwd_only)
  graph <- built$graph
  # Free the device buffer with the rest; leaking it across a sweep skews the
  # deepest measurements, which run last.
  on.exit(ggml_backend_buffer_free(built$buf), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(sched, graph)

  step <- function() {
    # Only a graph built with _grads carries gradient storage to reset; the
    # bwd_only graph is a plain forward graph over the backward node.
    if (train) ggml_graph_reset(graph)
    ggml_backend_sched_graph_compute(sched, graph)
  }

  # Warm up properly: shader compilation, first allocation, clocks spinning up.
  for (i in seq_len(3L)) step()
  ggml_backend_sched_synchronize(sched)

  # Several timed batches, reported by MEDIAN. A single batch of a few-ms
  # workload is dominated by noise -- an early version of this script produced
  # "2 blocks faster than 1 block", which is not a property of the system.
  batches <- vapply(seq_len(n_batches), function(b) {
    t0 <- Sys.time()
    for (i in seq_len(reps)) step()
    # graph_compute is the synchronous entry point (there is a separate _async
    # one), but synchronise anyway so no GPU work escapes the clock.
    ggml_backend_sched_synchronize(sched)
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
  }, numeric(1))

  list(ms     = stats::median(batches),
       lo     = min(batches),
       hi     = max(batches),
       splits = ggml_backend_sched_get_n_splits(sched))
}

cat(sprintf("Attention: DK=%d N=%d M=%d H=%d Hkv=%d B=%d, %d reps\n\n",
            DK, N, M, H, Hkv, B, reps))

if (!have_gpu) {
  cat("No Vulkan device -- only the CPU column will be measured.\n\n")
}

cat(sprintf("%-8s %-22s %10s %16s %8s\n",
            "blocks", "configuration", "ms/step", "(min-max)", "splits"))
cat(strrep("-", 70), "\n")

show <- function(nb, label, r) {
  cat(sprintf("%-8d %-22s %10.2f %16s %8d\n", nb, label, r$ms,
              sprintf("%.2f-%.2f", r$lo, r$hi), r$splits))
}

rows <- list()
for (nb in depths) {
  r <- list(blocks = nb)

  if (have_gpu) {
    f <- time_it("gpu", nb, FALSE)
    show(nb, "forward, GPU", f)
    r$fwd_gpu <- f$ms

    g <- time_it("gpu", nb, TRUE)
    show(nb, "fwd+bwd, GPU sched", g)
    r$train_gpu <- g$ms
    r$spread_gpu <- (g$hi - g$lo) / g$ms
    r$splits <- g$splits

    # The backward node on its own -- the shader's own cost, with no forward,
    # no loss node and no autodiff bookkeeping mixed in.
    bo <- time_it("gpu", nb, FALSE, bwd_only = TRUE)
    show(nb, "bwd only, GPU", bo)
    r$bwd_only_gpu <- bo$ms

    bc <- time_it("cpu", nb, FALSE, bwd_only = TRUE)
    show(nb, "bwd only, CPU", bc)
    r$bwd_only_cpu <- bc$ms
  }

  c_ <- time_it("cpu", nb, TRUE)
  show(nb, "fwd+bwd, CPU only", c_)
  r$train_cpu <- c_$ms

  rows[[length(rows) + 1L]] <- r
  cat("\n")
}

if (have_gpu) {
  cat(strrep("=", 70), "\n")

  for (r in rows) {
    ratio <- r$train_cpu / r$train_gpu
    cat(sprintf("%d block(s): train GPU %6.1f ms | CPU-only %6.1f ms | ratio %.2fx | %d splits\n",
                r$blocks, r$train_gpu, r$train_cpu, ratio, r$splits))
    cat(sprintf("            backward adds %5.1f ms on top of the forward's %.1f ms\n",
                r$train_gpu - r$fwd_gpu, r$fwd_gpu))
    if (!is.null(r$bwd_only_gpu)) {
      cat(sprintf("            bwd node alone: GPU %5.1f ms | CPU %5.1f ms  (%.1fx)\n",
                  r$bwd_only_gpu, r$bwd_only_cpu, r$bwd_only_cpu / r$bwd_only_gpu))
    }
  }

  ratios <- vapply(rows, function(r) r$train_cpu / r$train_gpu, numeric(1))
  spread <- max(vapply(rows, function(r) r$spread_gpu, numeric(1)))
  train  <- vapply(rows, function(r) r$train_gpu, numeric(1))
  blocks <- vapply(rows, function(r) r$blocks, numeric(1))

  cat("\n")
  # Sanity check before drawing any conclusion: more blocks must not be faster.
  if (is.unsorted(train)) {
    cat("⚠ Timings are NOT monotonic in depth (more blocks came out faster).\n")
    cat("  The measurement is dominated by noise -- raise GGMLR_ATTN_BENCH_REPS\n")
    cat("  or _BATCHES before believing any of the numbers below.\n\n")
  }
  if (spread > 0.25) {
    cat(sprintf("⚠ Run-to-run spread reaches %.0f%% of the median: treat ratios as rough.\n\n",
                100 * spread))
  }

  # Where the GPU path overtakes the CPU, and whether the advantage keeps
  # growing. This matters because real models are much deeper than a benchmark:
  # GPT-2 small and BERT-base are 12 layers, LLaMA-7B is 32, larger ones 40-96.
  # A ratio measured at 1-4 blocks says little about any of them.
  cat("depth:  ")
  cat(paste(sprintf("%d:%.2fx", blocks, ratios), collapse = "  "), "\n")

  crossing <- blocks[which(ratios >= 1.0)]
  if (length(crossing) == 0) {
    cat("\nThe GPU path is slower at every depth measured.\n")
  } else {
    cat(sprintf("\nGPU overtakes the CPU from %d block(s) on.\n", min(crossing)))
  }

  # Is the advantage still climbing at the deepest point, or has it flattened?
  if (length(ratios) >= 3) {
    tail_trend <- ratios[length(ratios)] / ratios[max(1, length(ratios) - 2)]
    if (tail_trend > 1.15) {
      cat(sprintf("Still climbing at %d blocks (%.2fx -> %.2fx over the last stretch),\n",
                  blocks[length(blocks)], ratios[max(1, length(ratios) - 2)],
                  ratios[length(ratios)]))
      cat("so a real model's depth should do better than these numbers, not worse.\n")
    } else if (tail_trend < 0.87) {
      cat("The advantage is shrinking with depth -- worth checking why before\n")
      cat("extrapolating to real model sizes.\n")
    } else {
      cat(sprintf("The advantage has flattened near %.2fx; deeper models should land\n",
                  ratios[length(ratios)]))
      cat("around there rather than climbing further.\n")
    }
  }

  cat("\nNote: 'backward adds' includes the CPU kernel's own runtime, not just the\n")
  cat("transfers, so it is an UPPER bound on what a shader would recover.\n")
}
