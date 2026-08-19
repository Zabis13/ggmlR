# End-to-end training of a Mamba-style block.
#
# ggml has no backward pass for any of the state-space or RWKV recurrences, so
# until ggmlR added them these ops were inference-only. This script is the
# end-to-end check that the gradients actually train something: it fits the
# convolution kernel and the selective-scan projections of a small Mamba block
# by gradient descent and reports whether the loss falls.
#
# The individual gradients are verified element by element against a numeric
# gradient in tests/testthat/test-ssm-rwkv.R. What this adds is the question
# those tests cannot answer: does a whole block, wired together and stepped by
# an optimizer, converge?
#
#   Rscript inst/examples/mamba_train_demo.R          # GPU when available
#   GGMLR_DEMO_BACKEND=cpu Rscript inst/examples/mamba_train_demo.R
#
# On the GPU the graph is mixed: the scheduler keeps what Vulkan supports there
# and moves the rest to the CPU. The SSM backward kernels have no Vulkan shader
# (like OUT_PROD and CROSS_ENTROPY_LOSS_BACK before ggmlR added theirs), so the
# forward runs on the GPU and the backward nodes fall back. Running both ways
# and comparing is the point: the loss curve should not depend on where the
# nodes ran.

library(ggmlR)

# One thread: the SSM backward kernels are single-threaded by design (B and C
# are shared across the heads of a group, so accumulating from several threads
# would race), and this keeps the run deterministic.
ggml_set_n_threads(1L)

set.seed(20260819)

# ---- backend selection ----------------------------------------------------
# ggml_graph_compute() always builds a CPU backend internally, so a graph that
# should touch the GPU has to go through a scheduler instead. The scheduler is
# also what real GPU-first training uses: it assigns each node to a backend that
# supports it and copies tensors across as needed.

# The demo runs the same training twice -- once entirely on the CPU, once with
# Vulkan -- and compares the two at the end. Running one and eyeballing the
# curve cannot tell a backend bug from a bad hyperparameter; running both from
# identical weights on identical batches can, because then the only difference
# left between the two columns is where the nodes executed.
#
# `use_gpu` is set per run by the loop at the bottom; step_grads() reads it.
use_gpu <- FALSE
# Set back to FALSE at the start of each run so both runs announce their backend.
.demo_backend_reported <- FALSE
# Trace the first step of each run: inputs, per-stage outputs, gradients.
# GGMLR_DEMO_TRACE=0 turns it off.
.demo_trace <- !identical(Sys.getenv("GGMLR_DEMO_TRACE"), "0")

have_gpu <- ggml_vulkan_available() && ggml_vulkan_device_count() > 0L
runs <- if (identical(tolower(Sys.getenv("GGMLR_DEMO_BACKEND")), "cpu")) {
  list(list(name = "CPU", gpu = FALSE))
} else if (have_gpu) {
  list(list(name = "CPU",    gpu = FALSE),
       list(name = "Vulkan", gpu = TRUE))
} else {
  cat("No Vulkan device found -- running the CPU pass only.\n")
  list(list(name = "CPU", gpu = FALSE))
}

# ---- problem -------------------------------------------------------------
# A task the recurrence can solve and a memoryless map cannot: every output
# position must reflect the running sum of the inputs before it. Only a layer
# that carries state across time can represent that.

# Sizes close to a real Mamba block rather than a toy. The defaults below put
# roughly 100k elements through the scan per step, which is enough for the GPU
# forward to be doing real work; GGMLR_DEMO_SMALL=1 shrinks it back to a size
# that runs in a second, for a quick check.
small <- identical(Sys.getenv("GGMLR_DEMO_SMALL"), "1")

# SSM state width. The Vulkan ssm_scan shader accepts d_state 128 or 256 only
# (see GGML_OP_SSM_SCAN in ggml_backend_vk_device_supports_op) -- with Mamba-1's
# d_state=16 the op is refused, the scheduler puts the whole scan on the CPU,
# and the "GPU" run below is silently identical to the CPU one. 128 is the
# Mamba-2 width and is what actually exercises the shader.
d_state  <- if (small)  4L else 128L
head_dim <- if (small)  2L else  64L   # channels per head
n_head   <- if (small)  2L else  16L   # heads -> d_inner = 1024
n_tok    <- if (small)  6L else  64L   # tokens per sequence
n_seqs   <- 1L
d_conv   <- if (small)  3L else   4L   # Mamba uses 4
d_inner  <- head_dim * n_head

n_samples <- if (small) 24L else 16L

make_batch <- function() {
  x <- array(runif(head_dim * n_head * n_tok * n_seqs, -1, 1),
             dim = c(head_dim, n_head, n_tok, n_seqs))
  # target[.., t, .] = cumulative mean of x up to t
  y <- x
  acc <- array(0, dim = c(head_dim, n_head))
  for (t in seq_len(n_tok)) {
    acc <- acc + x[, , t, 1]
    y[, , t, 1] <- acc / t
  }
  list(x = as.vector(x), y = as.vector(y))
}

batches <- lapply(seq_len(n_samples), function(i) make_batch())

# What a block that learned nothing would score. The loss is a sum of squares
# against the target, so predicting a flat zero costs sum(y^2) -- and a
# recurrence whose B and C have been driven to zero predicts exactly that.
# Comparing the final loss against the first epoch cannot tell the two apart
# (a run that starts by exploding beats its own first epoch while collapsing),
# so the verdict below is against this number instead.
baseline <- mean(vapply(batches, function(b) mean(b$y^2), numeric(1)))

# ---- trainable parameters -------------------------------------------------
# Kept as plain R vectors between steps; each step rebuilds the graph, uploads
# these, computes the gradient, and applies it. That is slower than a persistent
# optimizer context, but it keeps the script readable and exercises exactly the
# path a training loop would.

# Drawn once, before either run. Both backends start from this same copy --
# re-drawing them per run would put an RNG difference into the comparison and
# make any backend discrepancy unreadable.
params0 <- list(
  # ssm_conv kernel, [d_conv, d_inner]
  conv = runif(d_conv * d_inner, -0.3, 0.3),
  # ssm_scan projections
  B    = runif(d_state * 1L * n_tok * n_seqs, -0.3, 0.3),
  C    = runif(d_state * 1L * n_tok * n_seqs, -0.3, 0.3),
  # per-head decay; negative keeps the recurrence stable
  A    = -runif(n_head, 0.2, 1.0),
  # Step size. The recurrence contracts only when dt * A < 0, so with A < 0 the
  # step has to stay positive -- Mamba itself never uses a raw dt, it passes it
  # through softplus for exactly this reason. Seeding dt symmetric around zero
  # instead makes exp(dt * A) > 1 for half the heads and the state blows up
  # exponentially over the sequence (a starting loss around 1e16 at these sizes).
  dt   = runif(n_head * n_tok * n_seqs, 0.01, 0.1)
)

# ---- one forward+backward pass -------------------------------------------
# Returns the loss and the gradient of every parameter.
step_grads <- function(p, batch) {
  ctx <- ggml_init(128 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  # Tensors are allocated by the backend, not the context, so the same code
  # serves both paths.
  ggml_set_no_alloc(ctx, TRUE)

  # Inputs. The conv branch needs d_conv-1 positions of left context, so the
  # sequence fed to ssm_conv is longer than the token count.
  conv_len <- d_conv - 1L + n_tok
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, conv_len, d_inner, n_seqs)
  # None of these are marked with ggml_set_input(). That flag tells the
  # scheduler the tensor is fed from outside, which pins it to the CPU -- and
  # since a node inherits the backend of its sources, one flagged input drags
  # the whole graph off the GPU. Uploading with ggml_backend_tensor_set_data()
  # after sched_alloc_graph() works either way, so the flag buys nothing here
  # and costs the entire Vulkan path.

  cw <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)

  ggml_set_param(cw)

  # conv branch: [d_inner, n_tok, n_seqs]
  conv_out <- ggml_ssm_conv(ctx, sx, cw)

  # The scan takes x as [head_dim, n_head, n_tok, n_seqs]; the conv output has
  # the same element count, so a reshape re-labels the axes.
  x_scan <- ggml_reshape_4d(ctx, conv_out, head_dim, n_head, n_tok, n_seqs)

  s0 <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs)

  dt <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_tok, n_seqs)
  ggml_set_param(dt)

  A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_head)
  ggml_set_param(A)

  B <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, 1L, n_tok, n_seqs)
  ggml_set_param(B)

  C <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, 1L, n_tok, n_seqs)
  ggml_set_param(C)

  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs)

  scan <- ggml_ssm_scan(ctx, s0, x_scan, dt, A, B, C, ids)
  # The scan packs outputs then final state; only the outputs are scored.
  y_hat <- ggml_ssm_scan_output(ctx, scan, x_scan)

  target <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tok, n_seqs)

  diff <- ggml_sub(ctx, y_hat, target)
  # Mean squared error, as sum/n rather than ggml_mean(): GGML_OP_MEAN reduces
  # along ne[0] only, so on a 4D tensor it returns [1, n_head, n_tok, n_seqs]
  # rather than a scalar, and the loss has to be a scalar. Scaling the sum is
  # what makes the gradient independent of the element count -- with a raw sum
  # a step size tuned at one size diverges at another, which is what the "* 24"
  # fudge on lr and the sqrt(length)-scaled gradient clip were compensating for.
  n_out_elem <- head_dim * n_head * n_tok * n_seqs
  loss <- ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, diff)), 1 / n_out_elem)
  ggml_set_output(loss)

  # Keep the intermediates readable. Without ggml_set_output() the scheduler is
  # free to reuse their buffers for later nodes, and the trace below would read
  # whatever happened to land there instead of the layer's own output.
  if (isTRUE(.demo_trace)) {
    ggml_set_output(conv_out)
    # `scan`, not the y_hat view into it: flagging a view leaves the underlying
    # buffer unflagged, and on Vulkan the trace then reads back zeros for a
    # tensor the shader computed correctly.
    ggml_set_output(scan)
  }
  ggml_set_loss(loss)

  # The full low-level autodiff sequence. ggml_graph_reset() is not optional:
  # it seeds d(loss)/d(loss) = 1, and without it every gradient computes as
  # zero -- silently, since nothing errors.
  graph <- ggml_build_forward_expand_grads(ctx, loss, graph_size = 4096L)
  ggml_build_backward_expand(ctx, graph)

  # Allocate on the chosen backend, then upload. The order matters: with
  # no_alloc the tensors have no storage until the backend provides it.
  backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  # Report where the work actually landed, once per run. Creating a Vulkan
  # backend proves nothing: the scheduler assigns each node to a backend that
  # accepts it, and an op the shader refuses (ssm_scan outside d_state 128/256,
  # or any of the backward kernels, which have no shader at all) falls back to
  # the CPU without a word. Asking the scheduler which backend holds the scan
  # output is the difference between "a GPU was present" and "the GPU ran it".
  report_placement <- function(sched) {
    if (isTRUE(.demo_backend_reported)) return(invisible(NULL))
    .demo_backend_reported <<- TRUE
    # ggml_vulkan_backend_name() is the only way to ask a backend its name, and
    # on a build without Vulkan it errors rather than answering -- even about a
    # CPU backend. Guard it, so the CPU pass still runs where there is no GPU.
    name_of <- function(b) {
      if (is.null(b)) return("unassigned")
      if (!ggml_vulkan_available()) return("CPU")
      ggml_vulkan_backend_name(b)
    }
    where <- function(t) {
      name_of(tryCatch(ggml_backend_sched_get_tensor_backend(sched, t),
                       error = function(e) NULL))
    }
    cat(sprintf("  backend created: %s, splits=%d\n",
                name_of(backend),
                ggml_backend_sched_get_n_splits(sched)))
    cat(sprintf("  ssm_conv out -> %s | ssm_scan out -> %s | loss -> %s\n",
                where(conv_out), where(scan), where(loss)))
  }
  on.exit(ggml_backend_free(backend), add = TRUE)
  # The scheduler appends a CPU backend of its own, which is what lets the
  # backward nodes Vulkan cannot run fall back rather than fail.
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)
  # After allocation, not before: the assignment does not exist until the
  # scheduler has walked the graph.
  report_placement(sched)

  ggml_backend_tensor_set_data(sx, c(rep(0, (d_conv - 1L) * d_inner * n_seqs), batch$x))
  ggml_backend_tensor_set_data(cw, p$conv)
  ggml_backend_tensor_set_data(s0, rep(0, d_state * head_dim * n_head * n_seqs))
  ggml_backend_tensor_set_data(dt, p$dt)
  ggml_backend_tensor_set_data(A,  p$A)
  ggml_backend_tensor_set_data(B,  p$B)
  ggml_backend_tensor_set_data(C,  p$C)
  ggml_backend_tensor_set_data(ids, as.integer(seq_len(n_seqs) - 1L))
  ggml_backend_tensor_set_data(target, batch$y)

  ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  # ---- trace ---------------------------------------------------------------
  # One pass over the graph, printed once per run: what went in, what each stage
  # produced, and what came back as gradients. Summary statistics rather than
  # the tensors themselves -- with d_inner=1024 the interesting question is
  # whether a stage is saturating or exploding, and min/max/rms answers it.
  if (isTRUE(.demo_trace)) {
    .demo_trace <<- FALSE
    stat <- function(label, v) {
      finite <- v[is.finite(v)]
      cat(sprintf("  %-16s n=%-8d min %11.4g  max %11.4g  rms %11.4g%s\n",
                  label, length(v),
                  if (length(finite)) min(finite) else NA_real_,
                  if (length(finite)) max(finite) else NA_real_,
                  if (length(finite)) sqrt(mean(finite^2)) else NA_real_,
                  if (length(finite) < length(v))
                    sprintf("  [%d non-finite]", length(v) - length(finite)) else ""))
    }
    cat("\n  -- forward ------------------------------------------------\n")
    stat("x (input)",   batch$x)
    stat("target",      batch$y)
    stat("conv w",      p$conv)
    stat("ssm_conv out", ggml_backend_tensor_get_data(conv_out))
    # The scan packs outputs then final states; slice the output half off the
    # parent tensor rather than reading the view.
    scan_all <- ggml_backend_tensor_get_data(scan)
    n_out    <- head_dim * n_head * n_tok * n_seqs
    stat("ssm_scan out",   scan_all[seq_len(n_out)])
    stat("ssm_scan state", scan_all[-seq_len(n_out)])
    cat("  -- parameters ---------------------------------------------\n")
    stat("A",  p$A);  stat("dt", p$dt)
    stat("B",  p$B);  stat("C",  p$C)
    cat(sprintf("  %-16s %.6g\n", "loss",
                ggml_backend_tensor_get_data(loss)[1]))
    cat("  -- gradients ----------------------------------------------\n")
    for (nm in c("conv", "dt", "A", "B", "C")) {
      tn <- switch(nm, conv = cw, dt = dt, A = A, B = B, C = C)
      gg <- ggml_graph_get_grad(graph, tn)
      if (is.null(gg)) cat(sprintf("  %-16s (no gradient node)\n", nm))
      else stat(paste0("d/d", nm), ggml_backend_tensor_get_data(gg))
    }
    cat("\n")
  }

  grab <- function(t) {
    g <- ggml_graph_get_grad(graph, t)
    if (is.null(g)) rep(0, ggml_nelements(t)) else ggml_backend_tensor_get_data(g)
  }

  list(loss  = ggml_backend_tensor_get_data(loss)[1],
       grads = list(conv = grab(cw), dt = grab(dt), A = grab(A),
                    B = grab(B), C = grab(C)))
}

# ---- training loop --------------------------------------------------------

# One learning rate for both sizes: the loss is a mean, so the gradient no
# longer grows with d_inner and the size-dependent fudge factor is gone.
#
# The value looks large only because the loss is a mean over 65536 elements:
# that divides every gradient by the same factor, leaving them around 1e-5 at
# these sizes (the trace below prints them). At lr = 0.05 a step would move each
# weight by roughly five millionths of its own size, which is why an earlier
# version of this demo fell by 4% in 60 epochs and called it training. lr = 40
# puts the step near 1% of the parameter, which is an ordinary SGD step.
lr      <- 40
n_epoch <- if (small) 40L else 60L

cat("Training a Mamba-style block (ssm_conv -> ssm_scan)\n")
cat(sprintf("  d_state=%d head_dim=%d n_head=%d d_inner=%d tokens=%d samples=%d\n",
            d_state, head_dim, n_head, d_inner, n_tok, n_samples))
cat(sprintf("  scan work: %.1fM multiply-adds per step\n",
            n_tok * n_head * head_dim * d_state / 1e6))
cat(sprintf("  predicting zero scores %.4f -- the number to beat\n", baseline))

# One full training run on whichever backend `use_gpu` selects. Everything it
# needs is passed in or drawn before it is called, so calling it twice runs the
# same arithmetic on two backends rather than two different problems.
train_once <- function(label) {
  params  <- params0          # both runs start from the same weights
  .demo_backend_reported <<- FALSE
  .demo_trace            <<- !identical(Sys.getenv("GGMLR_DEMO_TRACE"), "0")
  history <- numeric(n_epoch)

  cat(sprintf("\n=== %s ===\n", label))
  # Say out loud which backend the graph actually got. A run that silently fell
  # back to the CPU would otherwise show up as a suspiciously perfect agreement
  # in the comparison table rather than as an error.
  cat(sprintf("  use_gpu=%s  vulkan devices=%d\n",
              use_gpu, ggml_vulkan_device_count()))
  t_start <- Sys.time()

  for (epoch in seq_len(n_epoch)) {
    epoch_loss <- 0

    for (b in batches) {
      r <- step_grads(params, b)
      epoch_loss <- epoch_loss + r$loss

      # Plain SGD. Gradients are clipped because the recurrence can amplify them
      # early on, before A has settled into a decaying range.
      for (nm in names(r$grads)) {
        g <- r$grads[[nm]]
        g[!is.finite(g)] <- 0
        # Clip each parameter's gradient to unit norm. With a mean loss the
        # gradients no longer scale with tensor size, so one flat cap suits all
        # of them -- and A needs it: its gradient comes in two orders of
        # magnitude above the rest (rms ~43 against ~1-3 for the others), enough
        # to throw a 16-element vector of ~0.5 values clean out of range in a
        # single step.
        gn  <- sqrt(sum(g^2))
        # Sized against the gradients this problem actually produces, whose
        # norms sit near 1e-3 at the start (the trace prints them). A cap of 1
        # would be a comment rather than a safeguard -- it could never trigger.
        # This one is loose enough not to touch an ordinary step and tight
        # enough to catch the recurrence amplifying one, which is what the cap
        # is for.
        cap <- 0.05
        if (gn > cap) g <- g * (cap / gn)
        params[[nm]] <- params[[nm]] - lr * g
      }
      # Keep the decay stable: A must stay negative and dt positive for
      # exp(dt*A) to contract. This is the projected-gradient stand-in for the
      # softplus a real Mamba block puts on dt.
      params$A  <- pmin(params$A, -0.05)
      params$dt <- pmax(params$dt, 1e-3)
    }

    history[epoch] <- epoch_loss / length(batches)
    if (epoch %% max(1L, n_epoch %/% 8L) == 0L || epoch == 1L) {
      cat(sprintf("  epoch %2d   loss %.5f\n", epoch, history[epoch]))
    }
  }

  elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  cat(sprintf("  %.1f s total, %.2f s per epoch\n", elapsed, elapsed / n_epoch))

  list(label = label, history = history, elapsed = elapsed, params = params)
}

results <- lapply(runs, function(run) {
  use_gpu <<- run$gpu       # step_grads() picks its backend from this
  train_once(run$name)
})
names(results) <- vapply(runs, function(r) r$name, character(1))

# ---- comparison -----------------------------------------------------------

verdict <- function(last) {
  if (!is.finite(last)) "FAIL (not finite)"
  else if (last >= baseline) "FAIL (no better than zero)"
  else if (last > 0.5 * baseline) "WEAK"
  else "OK"
}

fmt <- function(v) if (!is.finite(v)) "       n/a" else sprintf("%10.4f", v)

cat("\n")
cat("=== comparison ===============================================\n")
cat(sprintf("%-10s %12s %12s %10s %8s  %s\n",
            "backend", "first loss", "final loss", "% of zero", "sec", "verdict"))
for (r in results) {
  last <- r$history[n_epoch]
  cat(sprintf("%-10s %12s %12s %9s%% %8.1f  %s\n",
              r$label, fmt(r$history[1]), fmt(last),
              if (is.finite(last)) sprintf("%9.1f", 100 * last / baseline) else "      n/a",
              r$elapsed, verdict(last)))
}

# With both backends present the interesting number is not either curve but the
# gap between them: same weights, same batches, same order of updates, so the
# losses should track each other to within float noise. They will not match bit
# for bit -- Vulkan reduces in a different order, and the backward nodes it
# cannot run come back from the CPU -- but a divergence past a few percent is a
# backend bug, not rounding.
if (length(results) == 2L) {
  hc <- results[["CPU"]]$history
  hv <- results[["Vulkan"]]$history
  ok <- is.finite(hc) & is.finite(hv)
  scale <- pmax(abs(hc), abs(hv), 1e-12)
  rel <- ifelse(ok, abs(hc - hv) / scale, NA_real_)

  cat("\n")
  cat(sprintf("per-epoch relative gap CPU vs Vulkan: max %.3g, median %.3g\n",
              max(rel, na.rm = TRUE), median(rel, na.rm = TRUE)))
  cat(sprintf("final loss   CPU %.5f   Vulkan %.5f   rel.diff %.3g\n",
              hc[n_epoch], hv[n_epoch], rel[n_epoch]))

  if (anyNA(rel)) {
    cat("MISMATCH: one backend produced a non-finite loss where the other did not.\n")
  } else if (max(rel) > 0.05) {
    cat("MISMATCH: the two backends disagree by more than 5% -- suspect the\n")
    cat("          Vulkan path, not the hyperparameters.\n")
  } else {
    cat("AGREE: both backends follow the same loss curve.\n")
  }
}
