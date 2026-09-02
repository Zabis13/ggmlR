#!/usr/bin/env Rscript
#
# Which of the remaining backward gaps actually costs anything?
#
# Three ops still have no Vulkan backward, so a training graph that uses them
# splits and falls to the CPU, exactly as flash attention did before its shader:
#
#   IM2COL_BACK             backward of convolution
#   POOL_2D_BACK            backward of pooling
#   CROSS_ENTROPY_LOSS      the FORWARD; only _BACK has a shader
#
# Writing all three is days of work, and this session has shown three times over
# that a confident guess about which one matters is worth less than one
# measurement. So: measure first, then write the shader for whichever actually
# hurts.
#
# For each op this reports how the training graph is placed (splits, nodes left
# on the CPU) and what a step costs against a CPU-only baseline. The comparison
# that decides priority is the last column: an op whose GPU path already beats
# the CPU is not urgent, however many splits it causes.
#
# Run:  Rscript inst/scripts/measure_missing_backward_ops.R

suppressMessages(library(ggmlR))

reps    <- as.integer(Sys.getenv("GGMLR_OPS_BENCH_REPS", "20"))
batches <- 5L

have_gpu <- ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
if (!have_gpu) {
  cat("No Vulkan device -- nothing to compare against.\n"); quit(save = "no")
}

# ---- the three graphs under test -------------------------------------------
# Each builds a trainable graph ending in a scalar, with the weights allocated
# on `backend` BEFORE the graph is built: the scheduler places an op by where
# its sources already are, so host-resident weights would keep everything on the
# CPU regardless of what supports_op says.

# Convolution: im2col is how ggml_conv_2d lowers, so its backward is IM2COL_BACK.
build_conv <- function(ctx, backend, n) {
  # Cout must equal Cin: the kernel is reused for every layer of the stack, and
  # ggml_conv_2d asserts a->ne[2] == b->ne[2], so a widening kernel would only
  # work for the first one. (An earlier version used 16->32 and the whole case
  # died at the second layer, which the outer tryCatch reported as an unhelpful
  # "graph failed to build or run".)
  W <- 64L; H <- 64L; Cin <- 16L; Cout <- 16L; K <- 3L; B <- 8L
  # F32 kernel, not F16: the CPU refuses IM2COL_BACK unless both sources are F32
  # (ggml-cpu.cpp), and Vulkan has no shader at all, so an F16 kernel leaves the
  # backward node with NO backend and the scheduler aborts.
  ker <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, K, Cin, Cout)
  img <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, B)
  ggml_set_input(ker); ggml_set_input(img)
  ggml_set_param(img)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)   # freed by measure()
  ggml_backend_tensor_set_data(ker, runif(K * K * Cin * Cout, -1, 1))
  ggml_backend_tensor_set_data(img, runif(W * H * Cin * B, -1, 1))

  out <- img
  for (i in seq_len(n)) out <- ggml_conv_2d(ctx, ker, out, 1L, 1L, 1L, 1L, 1L, 1L)
  list(loss = ggml_sum(ctx, out), buf = buf)
}

# Pooling: POOL_2D over a feature map, repeated to give the graph some depth.
build_pool <- function(ctx, backend, n) {
  W <- 128L; H <- 128L; C <- 32L
  x <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, W, H, C)
  ggml_set_input(x); ggml_set_param(x)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)   # freed by measure()
  ggml_backend_tensor_set_data(x, runif(W * H * C, -1, 1))

  # Stride 1 so repeated pooling does not shrink the map away.
  out <- x
  for (i in seq_len(n)) {
    out <- ggml_pool_2d(ctx, out, GGML_OP_POOL_AVG, 2L, 2L, 1L, 1L, 0L, 0L)
  }
  list(loss = ggml_sum(ctx, out), buf = buf)
}

# Cross-entropy: the forward has no shader, the backward does -- the reverse of
# the other two, so the split lands in a different place.
build_ce <- function(ctx, backend, n) {
  nc <- 1024L; nr <- 64L
  logits <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr)
  labels <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr)
  ggml_set_input(logits); ggml_set_input(labels)
  ggml_set_param(logits)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)   # freed by measure()
  ggml_backend_tensor_set_data(logits, runif(nc * nr, -1, 1))
  lab <- matrix(0, nc, nr); lab[cbind(sample.int(nc, nr, TRUE), seq_len(nr))] <- 1
  ggml_backend_tensor_set_data(labels, as.numeric(lab))

  out <- ggml_cross_entropy_loss(ctx, logits, labels)
  if (n > 1) for (i in 2:n) out <- ggml_add(ctx, out, ggml_cross_entropy_loss(ctx, logits, labels))
  list(loss = out, buf = buf)
}

# Embedding: GET_ROWS gathers rows of a vocabulary table, and its backward
# scatters the gradients back. The table is the point -- a toy 6x32 costs
# nothing either way, while a real 30k x 512 vocabulary is 15M weights touched
# per step, which is where the TODO expects this to start hurting.
build_embed <- function(ctx, backend, n) {
  VOCAB <- 30000L; DIM <- 512L; TOKENS <- 256L
  tbl <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DIM, VOCAB)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, TOKENS)
  ggml_set_input(tbl); ggml_set_input(ids)
  ggml_set_param(tbl)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)   # freed by measure()
  ggml_backend_tensor_set_data(tbl, runif(DIM * VOCAB, -0.1, 0.1))
  # 0-based token ids, as ggml expects.
  ggml_backend_tensor_set_data(ids, sample.int(VOCAB, TOKENS, replace = TRUE) - 1L)

  out <- ggml_get_rows(ctx, tbl, ids)
  if (n > 1) for (i in 2:n) out <- ggml_add(ctx, out, ggml_get_rows(ctx, tbl, ids))
  list(loss = ggml_sum(ctx, out), buf = buf)
}

measure <- function(builder, mode, n) {
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
  ctx <- ggml_init(3072 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  built <- builder(ctx, backend, n)
  loss  <- built$loss
  # A backend buffer is not owned by the context: free it explicitly, and
  # before the context, or each case leaks its weights.
  on.exit(ggml_backend_buffer_free(built$buf), add = TRUE, after = FALSE)
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_backend_sched_alloc_graph(sched, graph)

  step <- function() {
    ggml_graph_reset(graph)
    ggml_backend_sched_graph_compute(sched, graph)
  }

  for (i in seq_len(2L)) step()
  ggml_backend_sched_synchronize(sched)

  ts <- vapply(seq_len(batches), function(...) {
    t0 <- Sys.time()
    for (i in seq_len(reps)) step()
    ggml_backend_sched_synchronize(sched)
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
  }, numeric(1))

  n_cpu <- 0L
  for (i in seq_len(ggml_graph_n_nodes(graph))) {
    be <- ggml_backend_sched_get_tensor_backend(sched, ggml_graph_node(graph, i - 1L))
    if (!is.null(be) && ggml_backend_name(be) == "CPU") n_cpu <- n_cpu + 1L
  }

  list(ms = stats::median(ts), splits = ggml_backend_sched_get_n_splits(sched),
       nodes = ggml_graph_n_nodes(graph), cpu = n_cpu)
}

cases <- list(
  list(name = "IM2COL_BACK (conv 2d)",   build = build_conv, n = 4L),
  list(name = "POOL_2D_BACK (pooling)",  build = build_pool, n = 4L),
  list(name = "CROSS_ENTROPY_LOSS fwd",  build = build_ce,   n = 4L),
  list(name = "GET_ROWS_BACK (embedding)", build = build_embed, n = 4L)
)

cat(sprintf("%d reps x %d batches\n\n", reps, batches))
cat(sprintf("%-26s %7s %7s %7s %10s %10s %8s\n",
            "op", "nodes", "splits", "on_cpu", "GPU ms", "CPU ms", "ratio"))
cat(strrep("-", 84), "\n")

results <- list()
for (cs in cases) {
  g <- tryCatch(measure(cs$build, "gpu", cs$n), error = function(e) e)
  if (inherits(g, "error")) {
    # Print the reason: a swallowed error hid a channel mismatch for a while.
    cat(sprintf("%-26s  FAILED: %s\n", cs$name, conditionMessage(g)))
    next
  }
  c_ <- measure(cs$build, "cpu", cs$n)
  ratio <- c_$ms / g$ms
  cat(sprintf("%-26s %7d %7d %7d %10.2f %10.2f %7.2fx\n",
              cs$name, g$nodes, g$splits, g$cpu, g$ms, c_$ms, ratio))
  results[[cs$name]] <- list(splits = g$splits, cpu = g$cpu, ratio = ratio)
}

cat("\n")
for (nm in names(results)) {
  r <- results[[nm]]
  verdict <- if (r$cpu == 0) {
    "runs entirely on the GPU -- nothing to fix"
  } else if (r$ratio >= 1.3) {
    sprintf("%d node(s) on the CPU, but the GPU path still wins %.1fx -- low priority",
            r$cpu, r$ratio)
  } else {
    sprintf("%d node(s) on the CPU and only %.2fx over CPU-only -- worth a shader",
            r$cpu, r$ratio)
  }
  cat(sprintf("%-26s %s\n", nm, verdict))
}
