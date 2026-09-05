#!/usr/bin/env Rscript
#
# The blind spot: what is the forward pass made of, and what could a resident
# forward (component 4) actually remove?
#
# The gap this closes. On a 4-layer 1024-wide model the profiled backward
# accounts for ~62 ms while a full training step takes ~95 -- so 33 ms belong to
# the forward and nothing measures them. Component 4 is ~21 helpers of work, and
# committing to it on an unmeasured third of the step is exactly how the last
# three ceilings came in at a third of their prediction (checkpointing memory,
# the upload cache, component 3).
#
# So this script answers two questions with numbers, before any helper is
# touched:
#
#   1. how the forward divides between transfer and everything else
#   2. what fraction of a WHOLE training step the forward is -- because a
#      component that halves a third of the step buys a sixth, however good it
#      looks in isolation
#
# The ceiling printed at the end is upload+download only. Everything else in the
# forward (context handling, tensor creation, buffer allocation, graph building,
# compute) stays whatever component 4 does, exactly as `emit` and `compute`
# stayed in the backward.
#
# Run:  Rscript inst/scripts/measure_ag_forward_profile.R
# Env:  GGMLR_FP_REPS  passes per model (default 20)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_FP_REPS", "20"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns         <- asNamespace("ggmlR")
fwd_prof   <- get("ag_forward_profile",       envir = ns)
fwd_reset  <- get("ag_forward_profile_reset", envir = ns)
fwd_env    <- get(".ag_fwd",                  envir = ns)
bwd_graph  <- get("ag_backward_graph",        envir = ns)
bwd_res    <- get("ag_backward_resident",     envir = ns)
bwd_env    <- get(".ag_bwd",                  envir = ns)
bwd_prof   <- get("ag_backward_profile",      envir = ns)

ag_device("gpu")
on.exit({ fwd_prof(FALSE); bwd_prof(FALSE); bwd_res(FALSE); bwd_graph(FALSE)
          ag_device("cpu") }, add = TRUE)

mk <- function(d, b, depth) {
  ws <- lapply(seq_len(depth),
               function(i) ag_param(matrix(rnorm(d * d) * 0.05, d, d)))
  x  <- ag_tensor(matrix(rnorm(d * b), d, b))
  y  <- matrix(0.0, d, b)
  list(
    fwd = function() {
      with_grad_tape({
        h <- x
        for (w in ws) h <- ag_relu(ag_matmul(w, h))
        loss <- ag_mse_loss(h, y)
      })
      loss
    },
    step = function() {
      with_grad_tape({
        h <- x
        for (w in ws) h <- ag_relu(ag_matmul(w, h))
        loss <- ag_mse_loss(h, y)
      })
      backward(loss)
      invisible(NULL)
    })
}

tm <- function(f, warm = 3L) {
  for (i in seq_len(warm)) f()
  t0 <- Sys.time()
  for (i in seq_len(reps)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps
}

cfgs <- list(list("d=256  b=32  depth=4",  256L,  32L,  4L),
             list("d=512  b=64  depth=4",  512L,  64L,  4L),
             list("d=512  b=256 depth=4",  512L, 256L,  4L),
             list("d=1024 b=256 depth=4", 1024L, 256L,  4L),
             list("d=512  b=64  depth=12", 512L,  64L, 12L))

# ---------------------------------------------------------------------------
# 1. How much of a training step is the forward at all?
#
# The framing question. A perfect component 4 removes transfer from the forward
# and nothing else, so the forward's share of the step bounds it before any
# stage breakdown matters.
# ---------------------------------------------------------------------------

cat(sprintf("reps = %d\n\n1. Forward as a share of a full training step\n", reps))
cat("   model                   step ms   fwd ms   fwd share\n")
bwd_graph(TRUE); bwd_res(TRUE); fwd_prof(FALSE)
shares <- list()
for (cfg in cfgs) {
  m <- mk(cfg[[2L]], cfg[[3L]], cfg[[4L]])
  t_step <- tm(m$step)
  t_fwd  <- tm(m$fwd)
  shares[[cfg[[1L]]]] <- c(step = t_step, fwd = t_fwd)
  cat(sprintf("   %-22s %8.2f %8.2f  %8.1f%%\n",
              cfg[[1L]], t_step, t_fwd, 100 * t_fwd / t_step))
}

# ---------------------------------------------------------------------------
# 2. What the forward is made of
#
# Per-stage, the same shape as the backward profile. upload+download is what a
# resident forward removes; the rest is what it cannot.
# ---------------------------------------------------------------------------

cat("\n2. Forward stages\n")
for (cfg in cfgs) {
  m <- mk(cfg[[2L]], cfg[[3L]], cfg[[4L]])
  fwd_reset(); fwd_prof(TRUE)
  for (i in seq_len(reps)) m$fwd()
  fwd_prof(FALSE)

  ms <- fwd_env$totals
  if (is.null(ms) || fwd_env$n == 0L) {
    cat(sprintf("   %-22s (nothing recorded)\n", cfg[[1L]]))
    next
  }
  ms    <- ms / reps                      # per pass
  total <- sum(ms)
  moved <- sum(ms[intersect(names(ms), c("upload", "download"))])
  cat(sprintf("\n   %s -- %.2f ms per forward, %d ops\n",
              cfg[[1L]], total, as.integer(fwd_env$n / reps)))
  for (nm in names(sort(ms, decreasing = TRUE)))
    cat(sprintf("     %-9s %7.2f ms  %5.1f%%\n", nm, ms[[nm]],
                100 * ms[[nm]] / total))
  cat(sprintf("     -> transfer %.1f%%, forward ceiling %.2fx\n",
              100 * moved / total, total / max(total - moved, 1e-9)))

  # And the number that actually decides component 4: what it does to a STEP.
  sh <- shares[[cfg[[1L]]]]
  if (!is.null(sh)) {
    saved     <- moved                     # ms removed from the step
    step_ceil <- sh[["step"]] / max(sh[["step"]] - saved, 1e-9)
    cat(sprintf("     -> on a full step: %.2fx  (saves %.1f of %.1f ms)\n",
                step_ceil, saved, sh[["step"]]))
  }
}

# ---------------------------------------------------------------------------
# 3. Which upload is it? Weights or activations?
#
# Section 2 says upload dominates -- up to 72% of the forward -- but not what is
# being uploaded, and the two operands have opposite properties:
#
#   the weight       d x d, IDENTICAL on every op that uses it and on every
#                    step until the optimizer writes to it
#   the activation   d x batch, different every single time
#
# That distinction decides which fix applies. Handles (component 4) stop the
# activation from making a round trip between consecutive ops. Only a version
# cache stops the weight being re-sent, and only within a pass -- the optimizer
# rewrites it every step (probe_ag_upload_cache.R).
#
# Computed from shapes rather than timed: upload cost is linear in bytes at
# ~4 ms/MB (measure_ag_upload_cost.R), so the byte split IS the time split.
# ---------------------------------------------------------------------------

cat("\n3. What the upload consists of (by bytes, 4 bytes per f32 element)\n")
cat("   model                  weights MB  activ. MB  weight share\n")
for (cfg in cfgs) {
  d <- as.double(cfg[[2L]]); b <- as.double(cfg[[3L]]); depth <- as.double(cfg[[4L]])
  # per forward: each layer uploads its weight once and its input activation once
  w_mb <- depth * d * d * 4 / 1024^2
  a_mb <- depth * d * b * 4 / 1024^2
  cat(sprintf("   %-22s %10.2f %10.2f %12.0f%%\n",
              cfg[[1L]], w_mb, a_mb, 100 * w_mb / (w_mb + a_mb)))
}

# ---------------------------------------------------------------------------
# 4. The whole step as ONE graph -- the thing components 1-4 approach piecewise
#
# The industry model is not "resident tensors between ops", it is a single graph
# built once and executed on the device, with the host seeing nothing in
# between: CUDA Graphs, torch.compile, and how llama.cpp uses ggml itself
# (ggml_cgraph assembled in full, then one ggml_backend_graph_compute).
#
# Components 1-4 approach that piecewise -- backward as a graph, then gradients
# resident, then the forward -- and each piece is bounded by the piece it does
# not cover. This measures the destination directly: forward built as one graph
# on resident tensors, one upload of the weights, one download of the loss.
#
# It is not an implementation, just the same primitives .ag_run_op uses. So it
# bounds a hypothetical unified path from below, the way
# measure_ag_residency_ceiling.R bounded component 3 -- and unlike the per-stage
# ceilings above, it cannot miss a cost, because it times the whole thing.
# ---------------------------------------------------------------------------

cat("\n4. Forward as a single graph (the unified-graph destination)\n")
cat("   model                   per-op ms  1 graph ms  speedup\n")

BACKEND <- ggml_vulkan_init(0L)
if (!is.null(BACKEND)) {
  one_graph_fwd <- function(d, b, depth) {
    ws <- lapply(seq_len(depth), function(i) matrix(rnorm(d * d) * 0.05, d, d))
    x  <- matrix(rnorm(d * b), d, b)
    function() {
      # Context sized from the work: descriptors plus the data ggml keeps here.
      bytes <- (depth * d * d + (depth + 1) * d * b) * 4
      ctx <- ggml_init(max(32 * 1024 * 1024, 4 * bytes), no_alloc = TRUE)
      on.exit(ggml_free(ctx), add = TRUE)

      tw <- lapply(seq_len(depth),
                   function(i) ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, d))
      tx <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, b)

      node <- tx
      for (i in seq_len(depth)) node <- ggml_relu(ctx, ggml_mul_mat(ctx, tw[[i]], node))

      ggml_backend_alloc_ctx_tensors(ctx, BACKEND)
      for (i in seq_len(depth))
        ggml_backend_tensor_set_data(tw[[i]], as.numeric(t(ws[[i]])))
      ggml_backend_tensor_set_data(tx, as.numeric(x))

      cg <- ggml_init(as.double(ggml_graph_overhead()) + 65536, no_alloc = TRUE)
      on.exit(ggml_free(cg), add = TRUE)
      ggml_backend_graph_compute(BACKEND, ggml_build_forward_expand(cg, node))
      invisible(ggml_backend_tensor_get_data(node))
    }
  }

  for (cfg in cfgs) {
    m  <- mk(cfg[[2L]], cfg[[3L]], cfg[[4L]])
    t_perop <- tm(m$fwd)
    t_one   <- tm(one_graph_fwd(cfg[[2L]], cfg[[3L]], cfg[[4L]]))
    cat(sprintf("   %-22s %10.2f %11.2f  %6.2fx\n",
                cfg[[1L]], t_perop, t_one, t_perop / t_one))
  }
  ggml_vulkan_free(BACKEND)
} else {
  cat("   (Vulkan init failed)\n")
}

cat("\nReading the numbers:\n")
cat("  forward share small        -> component 4 cannot move the step much,\n")
cat("                                whatever it does to the forward itself.\n")
cat("  transfer share high        -> the forward IS transfer-bound, so handles\n")
cat("                                are the right fix for it.\n")
cat("  step ceiling near 1.0x     -> stop; 21 helpers cannot pay for that.\n")
cat("  step ceiling well above    -> component 4 is worth costing out, and this\n")
cat("                                is the number to compare the result to.\n")
cat("  sec 3 weight share high    -> handles alone leave most of the upload in\n")
cat("                                place; the weight needs a version cache.\n")
cat("  sec 4 >> sec 2 ceilings    -> the win is in the unified graph, not in\n")
cat("                                per-op residency. That is a different and\n")
cat("                                larger piece of work than component 4.\n")
cat("  sec 4 ~ sec 2 ceilings     -> per-op residency captures most of it, and\n")
cat("                                the unified graph adds little.\n")
