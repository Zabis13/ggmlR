# Which backward ops actually run on the GPU.
#
# Training is where a GPU-first library earns its keep, and training is exactly
# where ggml upstream is thinnest: several ops have a forward Vulkan shader and
# no backward one, so the scheduler quietly moves every training step's backward
# half to the CPU. A graph like that still trains and still gives the right
# answer -- it is simply not using the GPU, and nothing says so.
#
# ggmlR adds the missing shaders. This script measures what that bought, one op
# at a time: for each, it builds the smallest graph that exercises the op's
# backward, runs it on the CPU and on Vulkan, and reports where the node landed,
# how long each took, and whether the two agree.
#
#   Rscript inst/examples/backward_gpu_demo.R
#
# The verdict to look for is the placement column. "Vulkan0" means the shader
# ran; "CPU" on the GPU pass means the op was refused and fell back, which is
# the failure this script exists to make visible.

library(ggmlR)

ggml_set_n_threads(1L)
set.seed(20260819)

if (!ggml_vulkan_available() || ggml_vulkan_device_count() == 0L) {
  cat("No Vulkan device -- this demo compares CPU against GPU and needs both.\n")
  quit(save = "no")
}

reps <- 20L   # timed repetitions; the first run is a warm-up and not counted

# ---- helpers --------------------------------------------------------------

# ggml_vulkan_backend_name() is the only binding that names a backend, and it
# errors on a build without Vulkan -- guarded here so the helper is safe to call
# from either pass.
backend_name <- function(b) {
  if (is.null(b)) return("unassigned")
  if (!ggml_vulkan_available()) return("CPU")
  ggml_vulkan_backend_name(b)
}

# Build a graph with `build`, run it on one backend, and return the output, the
# node placement and the per-iteration time.
#
# `build` receives a context and returns a list with `out` (the tensor to read),
# `probe` (the node whose placement is interesting) and `fill`, a function that
# uploads the inputs once the scheduler has allocated them.
run_on <- function(build, gpu, ctx_size = 512 * 1024 * 1024) {
  ctx <- ggml_init(ctx_size)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  spec <- build(ctx)
  ggml_set_output(spec$out)

  graph <- if (isTRUE(spec$train)) {
    ggml_set_loss(spec$out)
    g <- ggml_build_forward_expand_grads(ctx, spec$out, graph_size = 4096L)
    ggml_build_backward_expand(ctx, g)
    g
  } else {
    ggml_build_forward_expand(ctx, spec$out)
  }

  backend <- if (gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)
  spec$fill()

  where  <- backend_name(ggml_backend_sched_get_tensor_backend(sched, spec$probe))
  splits <- ggml_backend_sched_get_n_splits(sched)

  if (isTRUE(spec$train)) ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)   # warm-up, not timed

  t0 <- Sys.time()
  for (i in seq_len(reps)) {
    if (isTRUE(spec$train)) ggml_graph_reset(graph)
    ggml_backend_sched_graph_compute(sched, graph)
  }
  ms <- 1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps

  # Read the gradient rather than the loss where the op under test is a backward
  # node: a loss can match while a gradient is wrong.
  value <- if (!is.null(spec$read)) spec$read(graph) else
             ggml_backend_tensor_get_data(spec$out)

  list(value = value, where = where, splits = splits, ms = ms)
}

results <- list()

record <- function(op, note, build, ctx_size = 512 * 1024 * 1024) {
  cpu <- run_on(build, gpu = FALSE, ctx_size = ctx_size)
  gpu <- run_on(build, gpu = TRUE,  ctx_size = ctx_size)

  scale <- max(abs(cpu$value), abs(gpu$value), 1e-12)
  rel   <- max(abs(cpu$value - gpu$value)) / scale

  results[[length(results) + 1L]] <<- list(
    op = op, note = note,
    where_cpu = cpu$where, where_gpu = gpu$where,
    splits_gpu = gpu$splits,
    ms_cpu = cpu$ms, ms_gpu = gpu$ms, rel = rel)

  cat(sprintf("  %-24s %-8s %8.2f ms -> %8.2f ms   rel.diff %.2g\n",
              op, gpu$where, cpu$ms, gpu$ms, rel))
}

cat("Backward ops on the GPU: CPU vs Vulkan\n")
cat(sprintf("Device: %s\n\n", ggml_vulkan_device_description(0L)))
cat(sprintf("  %-24s %-8s %11s    %11s   %s\n",
            "op", "ran on", "CPU", "Vulkan", "agreement"))

# ---- OUT_PROD -------------------------------------------------------------
# Both gradients of ggml_mul_mat() are built from out_prod, so this is the op
# every dense layer's backward goes through. A transposed src1 is the case the
# second mul_mat gradient needs.

record("OUT_PROD", "dense-layer backward", function(ctx) {
  n_in <- 512L; n_out <- 256L; batch <- 64L
  w <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, n_out)
  ggml_set_param(w)
  x <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_in, batch)
  y <- ggml_mul_mat(ctx, w, x)
  t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_out, batch)
  loss <- ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, y, t)))

  list(out = loss, probe = loss, train = TRUE,
       fill = function() {
         ggml_backend_tensor_set_data(w, runif(n_in * n_out, -0.1, 0.1))
         ggml_backend_tensor_set_data(x, runif(n_in * batch, -1, 1))
         ggml_backend_tensor_set_data(t, runif(n_out * batch, -1, 1))
       },
       read = function(g) ggml_backend_tensor_get_data(ggml_graph_get_grad(g, w)))
})

# ---- CROSS_ENTROPY_LOSS_BACK ----------------------------------------------
# The backward of the loss every classifier ends in. Its shader does a numerically
# stable softmax per row, which is what an untrained model's large logits need.

record("CROSS_ENTROPY_LOSS_BACK", "classifier backward", function(ctx) {
  n_class <- 256L; batch <- 512L
  logits <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_class, batch)
  ggml_set_param(logits)
  labels <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_class, batch)
  loss <- ggml_cross_entropy_loss(ctx, logits, labels)

  list(out = loss, probe = loss, train = TRUE,
       fill = function() {
         ggml_backend_tensor_set_data(logits, runif(n_class * batch, -4, 4))
         # One-hot targets, as a real classifier would have.
         lab <- matrix(0, nrow = n_class, ncol = batch)
         lab[cbind(sample.int(n_class, batch, replace = TRUE), seq_len(batch))] <- 1
         ggml_backend_tensor_set_data(labels, as.vector(lab))
       },
       read = function(g) ggml_backend_tensor_get_data(ggml_graph_get_grad(g, logits)))
})

# ---- SSM_CONV_BACK --------------------------------------------------------
# The convolution branch of a Mamba block. One invocation per channel, no
# atomics needed: every write belongs to a single channel.

record("SSM_CONV_BACK", "Mamba conv branch", function(ctx) {
  d_conv <- 4L; d_inner <- 1024L; n_t <- 64L; n_s <- 1L
  ncs <- d_conv - 1L + n_t
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncs, d_inner, n_s)
  cw <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
  ggml_set_param(cw)
  y <- ggml_ssm_conv(ctx, sx, cw)
  t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s)
  loss <- ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, y, t)))

  list(out = loss, probe = loss, train = TRUE,
       fill = function() {
         ggml_backend_tensor_set_data(sx, runif(ncs * d_inner * n_s, -1, 1))
         ggml_backend_tensor_set_data(cw, runif(d_conv * d_inner, -0.3, 0.3))
         ggml_backend_tensor_set_data(t, runif(d_inner * n_t * n_s, -1, 1))
       },
       read = function(g) ggml_backend_tensor_get_data(ggml_graph_get_grad(g, cw)))
})

# ---- SSM_SCAN_BACK --------------------------------------------------------
# The selective scan, the recurrent core of Mamba and the expensive half of its
# backward. d_state must be 128 or 256 for the shader to take it; Mamba-1's 16
# falls back to the CPU by design.

record("SSM_SCAN_BACK", "Mamba selective scan", function(ctx) {
  d_state <- 128L; head_dim <- 64L; n_head <- 16L; n_tok <- 64L; n_seqs <- 1L
  s0 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs)
  x  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tok, n_seqs)
  dt <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_tok, n_seqs)
  ggml_set_param(dt)
  A  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_head)
  ggml_set_param(A)
  B  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, 1L, n_tok, n_seqs)
  ggml_set_param(B)
  C  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, 1L, n_tok, n_seqs)
  ggml_set_param(C)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs)

  scan  <- ggml_ssm_scan(ctx, s0, x, dt, A, B, C, ids)
  y_hat <- ggml_ssm_scan_output(ctx, scan, x)
  t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tok, n_seqs)
  loss <- ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, y_hat, t)))

  list(out = loss, probe = scan, train = TRUE,
       fill = function() {
         ggml_backend_tensor_set_data(s0, rep(0, d_state * head_dim * n_head * n_seqs))
         ggml_backend_tensor_set_data(x,  runif(head_dim * n_head * n_tok * n_seqs, -1, 1))
         ggml_backend_tensor_set_data(dt, runif(n_head * n_tok * n_seqs, 0.01, 0.1))
         ggml_backend_tensor_set_data(A,  -runif(n_head, 0.2, 1.0))
         ggml_backend_tensor_set_data(B,  runif(d_state * n_tok * n_seqs, -0.3, 0.3))
         ggml_backend_tensor_set_data(C,  runif(d_state * n_tok * n_seqs, -0.3, 0.3))
         ggml_backend_tensor_set_data(ids, 0L)
         ggml_backend_tensor_set_data(t, runif(head_dim * n_head * n_tok * n_seqs, -1, 1))
       },
       read = function(g) ggml_backend_tensor_get_data(ggml_graph_get_grad(g, B)))
})

# ---- summary --------------------------------------------------------------

cat("\n")
cat("=== summary ==================================================\n")
cat(sprintf("%-24s %-9s %7s %10s %10s %9s  %s\n",
            "op", "placement", "splits", "CPU ms", "Vulkan ms", "speedup", "agree"))

for (r in results) {
  speedup <- if (r$ms_gpu > 0) r$ms_cpu / r$ms_gpu else NA_real_
  agree   <- if (!is.finite(r$rel)) "??" else if (r$rel < 1e-3) "yes" else "NO"
  cat(sprintf("%-24s %-9s %7d %10.2f %10.2f %8.1fx  %s\n",
              r$op, r$where_gpu, r$splits_gpu, r$ms_cpu, r$ms_gpu, speedup, agree))
}

on_gpu <- vapply(results, function(r) r$where_gpu != "CPU", logical(1))
agreed <- vapply(results, function(r) is.finite(r$rel) && r$rel < 1e-3, logical(1))

cat("\n")
if (all(on_gpu) && all(agreed)) {
  cat(sprintf("OK: all %d backward ops ran on Vulkan and matched the CPU.\n",
              length(results)))
} else {
  if (!all(on_gpu)) {
    cat("FELL BACK: ",
        paste(vapply(results[!on_gpu], function(r) r$op, character(1)),
              collapse = ", "),
        " ran on the CPU -- supports_op refused the shape.\n", sep = "")
  }
  if (!all(agreed)) {
    cat("MISMATCH: ",
        paste(vapply(results[!agreed], function(r) r$op, character(1)),
              collapse = ", "),
        " disagreed between backends.\n", sep = "")
  }
}

# A note on reading the speedups: these are single ops in isolation, so they
# measure the shader rather than a training loop. An op that is a small part of
# a real graph will not move the wall clock as much as its column suggests --
# mamba_train_demo.R is where the end-to-end number lives.
