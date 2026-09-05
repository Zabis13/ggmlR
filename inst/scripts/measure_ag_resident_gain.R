#!/usr/bin/env Rscript
#
# How much of the predicted residency gain actually arrived?
#
# The chain of estimates that led here, each measured rather than assumed:
#
#   measure_ag_residency_ceiling.R      an isolated matmul chain: resident GPU
#                                       9.9 ms vs 21.8 CPU vs 59.2 for today's
#                                       path B -- a 6x gap
#   measure_ag_residency_on_backward.R  on a real backward, 65-78% of the time
#                                       is stages residency removes, so the
#                                       ceiling there is 2.87-4.61x
#
# Both are upper bounds. This measures the implementation: the same models, the
# same profiler, with resident gradients on and off. The gap between what
# arrived and the 2.87-4.61x ceiling is what the remaining components would
# have to close.
#
# Read the result against the ceiling, not against zero. A change that delivers
# 1.5x of a predicted 3x is not a failure -- it says half the transfer is gone
# and where the other half is. A change that delivers 1.0x means the flag did
# nothing, which is a different and more urgent finding.
#
# Run:  Rscript inst/scripts/measure_ag_resident_gain.R
# Env:  GGMLR_RG_REPS  backward passes per configuration (default 20)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_RG_REPS", "20"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns        <- asNamespace("ggmlR")
bwd_graph <- get("ag_backward_graph",    envir = ns)
bwd_res   <- get("ag_backward_resident", envir = ns)
bwd_path  <- get("ag_backward_path",     envir = ns)
bwd_prof  <- get("ag_backward_profile",  envir = ns)
bwd_env   <- get(".ag_bwd",              envir = ns)

ag_device("gpu")
on.exit({ bwd_res(FALSE); bwd_graph(FALSE); bwd_prof(FALSE); ag_device("cpu") },
        add = TRUE)

# One forward+backward on a dense stack, built fresh so nothing carries over.
mlp_step <- function(d, b, depth) {
  ws <- lapply(seq_len(depth),
               function(i) ag_param(matrix(rnorm(d * d) * 0.05, d, d)))
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

tm <- function(f, warm = 3L) {
  for (i in seq_len(warm)) f()
  t0 <- Sys.time()
  for (i in seq_len(reps)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps
}

# Per-stage means for one configuration, so the win can be attributed rather
# than just observed.
stages <- function(step) {
  bwd_env$prof_totals <- NULL; bwd_env$prof_n <- 0L
  bwd_prof(TRUE)
  for (i in seq_len(reps)) step()
  bwd_prof(FALSE)
  if (is.null(bwd_env$prof_totals) || bwd_env$prof_n == 0L) return(NULL)
  bwd_env$prof_totals / bwd_env$prof_n
}

cat(sprintf("reps = %d per configuration\n\n", reps))
cat("Ceiling from measure_ag_residency_on_backward.R: 2.87-4.61x\n\n")
cat("   model                    off ms    on ms   speedup   of ceiling\n")

ceilings <- c("d=256  b=32  depth=4"  = 2.87,
              "d=512  b=64  depth=4"  = 3.41,
              "d=512  b=256 depth=4"  = 4.19,
              "d=1024 b=256 depth=4"  = 4.61,
              "d=512  b=64  depth=12" = 3.79)

cfgs <- list(list("d=256  b=32  depth=4",  256L,  32L,  4L),
             list("d=512  b=64  depth=4",  512L,  64L,  4L),
             list("d=512  b=256 depth=4",  512L, 256L,  4L),
             list("d=1024 b=256 depth=4", 1024L, 256L,  4L),
             list("d=512  b=64  depth=12", 512L,  64L, 12L))

results <- list()
for (cfg in cfgs) {
  label <- cfg[[1L]]
  step  <- mlp_step(cfg[[2L]], cfg[[3L]], cfg[[4L]])

  bwd_graph(TRUE)
  bwd_res(FALSE); t_off <- tm(step)
  path <- bwd_path()
  bwd_res(TRUE);  t_on  <- tm(step)

  if (!identical(path, "graph")) {
    cat(sprintf("   %-22s  (graph path declined: %s)\n", label, path))
    next
  }
  sp   <- t_off / t_on
  ceil <- ceilings[[label]]
  cat(sprintf("   %-22s %8.2f %8.2f  %7.2fx  %8.0f%%\n",
              label, t_off, t_on, sp, 100 * (sp - 1) / (ceil - 1)))
  results[[label]] <- c(off = t_off, on = t_on)
}

# ---------------------------------------------------------------------------
# Where the time went, on the largest case: which stages actually shrank.
# ---------------------------------------------------------------------------

cat("\nStage breakdown, d=1024 b=256 depth=4\n")
step <- mlp_step(1024L, 256L, 4L)
bwd_graph(TRUE)
bwd_res(FALSE); s_off <- stages(step)
bwd_res(TRUE);  s_on  <- stages(step)

if (!is.null(s_off) && !is.null(s_on)) {
  nms <- union(names(s_off), names(s_on))
  cat("   stage           off ms    on ms    change\n")
  pick <- function(v, nm) if (nm %in% names(v)) v[[nm]] else 0
  for (nm in nms) {
    a <- pick(s_off, nm)
    b <- pick(s_on,  nm)
    cat(sprintf("   %-11s %8.2f %8.2f  %+8.2f\n", nm, a, b, b - a))
  }
  cat(sprintf("   %-11s %8.2f %8.2f  %+8.2f\n",
              "TOTAL", sum(s_off), sum(s_on), sum(s_on) - sum(s_off)))
}

# ---------------------------------------------------------------------------
# The comparison that decides whether path B is now competitive.
# ---------------------------------------------------------------------------

cat("\nAgainst the CPU, d=512 b=64 depth=4 (full forward+backward)\n")
d <- 512L; b <- 64L; depth <- 4L
gpu_step <- mlp_step(d, b, depth)
bwd_graph(TRUE); bwd_res(TRUE)
t_gpu <- tm(gpu_step)

ag_device("cpu")
cpu_step <- mlp_step(d, b, depth)
t_cpu <- tm(cpu_step)
ag_device("gpu")

cat(sprintf("   CPU (closures, threaded BLAS) : %8.2f ms\n", t_cpu))
cat(sprintf("   GPU (graph + resident grads)  : %8.2f ms  %.2fx\n",
            t_gpu, t_cpu / t_gpu))

# Sanity check on the CPU baseline, because this line has already produced two
# opposite conclusions from the same implementation in one session: 0.54x while
# something else was using the cores, then 4.29x while something else was using
# them differently. The CPU side is far more sensitive to load than the GPU
# side -- it loses cores, the GPU does not -- so an unattended run can invert
# the comparison without touching any code.
#
# ~9 ms is what this configuration costs on an idle machine here. A large
# departure means the number above is about the machine, not about the code.
CPU_IDLE_MS <- 9.2
if (t_cpu > 3 * CPU_IDLE_MS) {
  cat(sprintf("   !! CPU baseline is %.1fx its idle figure (%.1f ms): the machine\n",
              t_cpu / CPU_IDLE_MS, CPU_IDLE_MS))
  cat("      is loaded and this comparison is meaningless. The GPU-to-GPU\n")
  cat("      numbers above are still valid; this line is not.\n")
}

cat("\nReading the numbers:\n")
cat("  speedup near the ceiling  -> components 1-3 captured what was there;\n")
cat("                               the rest needs resident FORWARD too.\n")
cat("  speedup well under it     -> a stage that should have shrunk did not;\n")
cat("                               the breakdown above says which.\n")
cat("  speedup ~1.0x             -> the flag did nothing. Check that the graph\n")
cat("                               path ran at all (it prints when it declines).\n")
