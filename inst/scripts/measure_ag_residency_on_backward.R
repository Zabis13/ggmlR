#!/usr/bin/env Rscript
#
# Step 1 of the residency decision: how much of a REAL backward would residency
# actually remove?
#
# measure_ag_residency_ceiling.R answered the question on an isolated matmul
# chain and found a 6x gap (path B 59.2 ms, resident ceiling 9.9, CPU 21.8).
# That number is an upper bound on a forward chain, and it would be a mistake to
# assume a training step improves by the same factor: a backward pass spends its
# time on more than transfer.
#
# The graph backward already profiles itself by stage (R/ag_backward_graph.R),
# and those stages split exactly along the line that matters:
#
#   REMOVED by residency   upload   forward snapshots sent to the device
#                          download gradients read back
#                          flush    buffer allocation per pass
#
#   UNTOUCHED by residency emit     building the nodes in R
#                          build    graph assembly
#                          write_leaf  writing $grad on the leaves
#                          bwd_other   the rest of backward() around the call
#
#   UNCHANGED either way   compute  the arithmetic itself
#
# So the honest estimate is: speedup_ceiling = total / (total - removable).
# A 6x gap on a matmul chain becomes something smaller here, and this script
# says how much smaller BEFORE the contract is changed -- which is the whole
# point of running it first.
#
# Note what this still does not capture. Residency also changes the closure
# path, not just the graph path, and it would let intermediates stay on the
# device between the forward and the backward -- neither is visible in these
# stages. So treat the number as a floor for the graph path, not as the final
# answer for the redesign.
#
# Run:  Rscript inst/scripts/measure_ag_residency_on_backward.R
# Env:  GGMLR_RESB_REPS  backward passes per model (default 20)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_RESB_REPS", "20"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns  <- asNamespace("ggmlR")
gf  <- function(nm) get(nm, envir = ns)

bwd_graph  <- gf("ag_backward_graph")
bwd_path   <- gf("ag_backward_path")
bwd_prof   <- gf("ag_backward_profile")
bwd_report <- gf("ag_backward_profile_report")
bwd_env    <- gf(".ag_bwd")

# Stages residency removes, and stages it leaves alone. Keep these two vectors
# as the single statement of that split: everything below reads from them.
#
# leaf_fetch belongs with the removed group, and the first run of this script
# got that wrong by trusting its old name ("write_leaf"). Measured directly:
# .ag_bwd_write_leaf_grads costs 0.05-0.11 ms on the CPU path but 4-5 ms on the
# GPU path at the same shapes, because installing $grad materialises values that
# are still on the device. It is transfer wearing a bookkeeping name -- so
# residency removes most of it, and classifying it as "stays" understated the
# case for the contract change.
REMOVED   <- c("upload", "download", "flush", "leaf_fetch")
UNTOUCHED <- c("emit", "build", "bwd_other", "compute")

reset_prof <- function() {
  bwd_env$prof_totals <- NULL
  bwd_env$prof_n      <- 0L
}

# Run one model's forward+backward `reps` times with profiling on, and return
# the mean milliseconds per stage.
profile_model <- function(build_fn) {
  reset_prof()
  for (i in seq_len(reps)) build_fn()
  if (is.null(bwd_env$prof_totals) || bwd_env$prof_n == 0L) return(NULL)
  bwd_env$prof_totals / bwd_env$prof_n
}

report <- function(label, ms) {
  if (is.null(ms)) {
    cat(sprintf("\n%s: graph path did not run (%s)\n", label, bwd_path()))
    return(invisible(NULL))
  }
  total <- sum(ms)
  rem   <- sum(ms[intersect(names(ms), REMOVED)],   na.rm = TRUE)
  keep  <- sum(ms[intersect(names(ms), UNTOUCHED)], na.rm = TRUE)
  other <- total - rem - keep      # any stage not classified above

  cat(sprintf("\n%s -- %.2f ms per backward\n", label, total))
  ord <- order(ms, decreasing = TRUE)
  for (i in ord) {
    nm  <- names(ms)[i]
    tag <- if (nm %in% REMOVED) "removed by residency"
           else if (nm %in% UNTOUCHED) "stays"
           else "unclassified"
    cat(sprintf("   %-11s %7.2f ms  %5.1f%%   %s\n",
                nm, ms[i], 100 * ms[i] / total, tag))
  }
  if (abs(other) > 1e-9)
    cat(sprintf("   %-11s %7.2f ms  %5.1f%%   (not in either list)\n",
                "unlisted", other, 100 * other / total))

  cat(sprintf("   -> removable %.1f%%, so the graph-path ceiling is %.2fx\n",
              100 * rem / total, total / max(total - rem, 1e-9)))
  invisible(c(total = total, removable = rem, kept = keep))
}

ag_device("gpu")
bwd_graph(TRUE)
bwd_prof(TRUE)
on.exit({ bwd_prof(FALSE); bwd_graph(FALSE); ag_device("cpu") }, add = TRUE)

cat(sprintf("reps = %d per model, graph backward + stage profiling on\n", reps))
cat("Stages counted as removed by residency: ",
    paste(REMOVED, collapse = ", "), "\n", sep = "")

# ---------------------------------------------------------------------------
# Models. Only the graph-path-eligible ops (matmul, add, relu/sigmoid/tanh,
# softmax, scale, mul, transpose, the three losses) -- a tape with anything else
# falls back to closures and profiles nothing, which ag_backward_path() reports.
#
# Sizes stay inside the regime the ceiling sweep found favourable: d <= 1024,
# batch 32-256. Extrapolating past it is explicitly not the point: sweep 1 of
# the ceiling script peaked at d=1024 and fell off at 2048 and 4096.
# ---------------------------------------------------------------------------

mlp <- function(d, b, depth) {
  ws <- lapply(seq_len(depth), function(i) ag_param(matrix(rnorm(d * d) * 0.05, d, d)))
  x  <- ag_tensor(matrix(rnorm(d * b), d, b))
  y  <- matrix(0.0, d, b)
  function() {
    with_grad_tape({
      h <- x
      for (w in ws) h <- ag_relu(ag_matmul(w, h))
      loss <- ag_mse_loss(h, y)
    })
    backward(loss)
    invisible(NULL)
  }
}

r1 <- report("MLP  d=256  batch=32  depth=4",  profile_model(mlp(256L,  32L, 4L)))
r2 <- report("MLP  d=512  batch=64  depth=4",  profile_model(mlp(512L,  64L, 4L)))
r3 <- report("MLP  d=512  batch=256 depth=4",  profile_model(mlp(512L, 256L, 4L)))
r4 <- report("MLP  d=1024 batch=256 depth=4",  profile_model(mlp(1024L, 256L, 4L)))
r5 <- report("MLP  d=512  batch=64  depth=12", profile_model(mlp(512L,  64L, 12L)))

cat("\nReading the numbers:\n")
cat("  removable > 60%  -> residency carries most of a backward too; the 6x\n")
cat("                      from the matmul chain roughly survives.\n")
cat("  removable 30-60% -> real but smaller than the chain suggested; the\n")
cat("                      contract change is worth it, expect ~2x not ~6x.\n")
cat("  removable < 30%  -> a backward is dominated by R-side node building,\n")
cat("                      which residency does NOT fix. Reconsider: the win\n")
cat("                      would be on forward/inference, not on training.\n")
cat("  emit or build large -> that is the second cost the ceiling script could\n")
cat("                      not see; it needs graph reuse, and Vulkan implements\n")
cat("                      no graph_plan_* hooks (README).\n")
