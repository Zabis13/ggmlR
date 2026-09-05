#!/usr/bin/env Rscript
#
# Would a GPU AdamW/SGD kernel make optimizer$step() faster on the ag_* path?
#
# TODO.md section 3 carries the item "GPU-оптимизаторы для ag_*" with the note
# "⚠️сначала замер". This script is that measurement. The claim to test is not
# "is the AdamW kernel fast" -- it is, ggml_opt_step_adamw runs on Vulkan
# already (ggml-vulkan-graph.cpp:577) and path A uses it. The claim is that
# calling it from path B would be a net win, and that depends on transfer, not
# on the kernel.
#
# What a step() would have to move. Adam updates a parameter from four
# matrices of identical size:
#
#   a     the weight       -- an R matrix after every step: .ag_data_set()
#                             drops $ptr (R/ag_device.R:832), so a parameter is
#                             never resident across an update
#   grad  the gradient     -- an R matrix always: backward() accumulates
#                             inp$grad in double (R/autograd.R:848), the device
#                             never sees it
#   m, v  the moments      -- R matrices in the optimizer env (R/autograd.R)
#
# and writes the new weight back through .ag_data_set(). So today's step() is
# pure host arithmetic, and a GPU step() would be 4 uploads + kernel + 1
# download. The upload cost is already known to be roughly linear in bytes at
# ~4 ms/MB (measure_ag_upload_cost.R), which for a 1024 square in f32 is 4 MB
# per matrix. The question this script answers with numbers is whether that
# transfer is smaller or larger than the host arithmetic it would replace.
#
# There is no R binding for ggml_opt_step_adamw -- only the backend comparison
# harness in R/backend_ops_test.R uses it -- so the kernel itself cannot be
# timed from here. That does not block the decision, and section 4 explains
# why: the same visit to the device is priced with ag_* ops that DO have
# bindings, which bounds a fused kernel from below. If a single trivial GPU op
# on these operands already costs more than the whole host step, no kernel can
# win, because the kernel cannot be cheaper than one visit.
#
# Sections:
#   1. host cost of a real Adam step, by parameter size
#   2. host cost split into arithmetic vs the .ag_data_mut/.ag_data_set path
#   3. bytes a GPU step would have to move, and the implied transfer time
#   4. measured floor for a GPU step: one trivial ag_* op on the same operands
#   5. the same for SGD, whose arithmetic is much lighter than Adam's
#   6. residency: what stays if m/v are kept on the device and only w/grad move
#
# Run:  Rscript inst/scripts/measure_ag_opt_step.R
# Env:  GGMLR_OPT_REPS   iterations per timing (default 20)
#       GGMLR_OPT_WARMUP warm-up calls before each timing (default 4)
#       GGMLR_OPT_MBPS   ms per MB used for the transfer estimate in section 3
#                        (default 4.0, from measure_ag_upload_cost.R on this
#                        machine -- re-measure and override if the hardware
#                        differs)

suppressMessages(library(ggmlR))

reps   <- as.integer(Sys.getenv("GGMLR_OPT_REPS",   "20"))
warmup <- as.integer(Sys.getenv("GGMLR_OPT_WARMUP", "4"))
ms_mb  <- as.numeric(Sys.getenv("GGMLR_OPT_MBPS",   "4.0"))

tm <- function(f) {
  for (i in seq_len(warmup)) f()
  t0 <- Sys.time()
  for (i in seq_len(reps)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps
}

mb <- function(nr, nc) nr * nc * 4 / 1024^2

# The shapes to sweep. Squares grow the operand; the tall and wide pair keeps
# the element count equal so any shape sensitivity shows up as a difference
# between them rather than as noise on the squares.
shapes <- list(c(128L, 128L), c(512L, 512L), c(1024L, 1024L), c(2048L, 2048L),
               c(4096L, 256L), c(256L, 4096L))

cat(sprintf("reps = %d, warm-up = %d, transfer estimate = %.2f ms/MB\n\n",
            reps, warmup, ms_mb))

has_gpu <- ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L
if (!has_gpu) {
  cat("No Vulkan device: sections 4 and 6 will be skipped, the host-side\n",
      "sections still answer what a step costs today.\n\n", sep = "")
}

# ---------------------------------------------------------------------------
# 1. What does a step() cost today?
#
# A real optimizer_adam over one parameter, gradient pre-set so that only the
# update is timed and no forward/backward noise enters. This is the number a
# GPU kernel would have to beat, transfer included.
# ---------------------------------------------------------------------------

cat("1. optimizer_adam$step() on the host, one parameter\n")
cat("   shape           MB      ms    ms per MB\n")
ag_device("cpu")
host_step <- numeric(0)
for (shape in shapes) {
  nr <- shape[[1L]]; nc <- shape[[2L]]
  p  <- ag_param(matrix(rnorm(nr * nc), nr, nc))
  g  <- matrix(rnorm(nr * nc), nr, nc)
  opt <- optimizer_adam(list(w = p), lr = 1e-3)
  ms <- tm(function() { p$grad <- g; opt$step() })
  host_step <- c(host_step, ms)
  cat(sprintf("   %5dx%-5d  %6.2f  %6.2f  %9.2f\n",
              nr, nc, mb(nr, nc), ms, ms / mb(nr, nc)))
}

# ---------------------------------------------------------------------------
# 2. Where does that time go?
#
# A GPU kernel replaces the arithmetic only. The value-access path
# (.ag_data_mut + .ag_data_set) stays: the result still has to become the
# parameter's value. If the access path is a large share, the reachable saving
# is smaller than section 1 suggests even before any transfer is counted.
# ---------------------------------------------------------------------------

cat("\n2. Adam step split: arithmetic vs the value-access path\n")
cat("   shape          total    arith   access\n")
for (shape in shapes) {
  nr <- shape[[1L]]; nc <- shape[[2L]]
  p  <- ag_param(matrix(rnorm(nr * nc), nr, nc))
  g  <- matrix(rnorm(nr * nc), nr, nc)
  m  <- matrix(0.0, nr, nc); v <- matrix(0.0, nr, nc)
  w  <- matrix(rnorm(nr * nc), nr, nc)
  b1 <- 0.9; b2 <- 0.999; eps <- 1e-8; lr <- 1e-3

  # the arithmetic of one Adam update, exactly as optimizer_adam does it
  arith <- tm(function() {
    m2 <- b1 * m + (1 - b1) * g
    v2 <- b2 * v + (1 - b2) * g^2
    mh <- m2 / (1 - b1)
    vh <- v2 / (1 - b2)
    w - lr * mh / (sqrt(vh) + eps)
  })
  # the access path alone: read for modification, write straight back
  data_set <- get(".ag_data_set", envir = asNamespace("ggmlR"))
  data_mut <- get(".ag_data_mut", envir = asNamespace("ggmlR"))
  access <- tm(function() data_set(p, data_mut(p)))

  opt <- optimizer_adam(list(w = p), lr = lr)
  total <- tm(function() { p$grad <- g; opt$step() })
  cat(sprintf("   %5dx%-5d  %6.2f   %6.2f   %6.2f\n",
              nr, nc, total, arith, access))
}

# ---------------------------------------------------------------------------
# 3. What a GPU step would have to move
#
# Adam touches four same-sized matrices and writes one back. None of them is
# resident: the weight loses $ptr on every .ag_data_set, and grad/m/v are host
# matrices that the device has never seen. The estimate uses the ms/MB from
# measure_ag_upload_cost.R; section 4 then measures a real device visit so the
# estimate does not have to be trusted on its own.
# ---------------------------------------------------------------------------

cat("\n3. Transfer a GPU Adam step would pay (4 uploads + 1 download)\n")
cat("   shape          MB each   MB moved   est ms   host ms   verdict\n")
for (i in seq_along(shapes)) {
  nr <- shapes[[i]][[1L]]; nc <- shapes[[i]][[2L]]
  one     <- mb(nr, nc)
  moved   <- 5 * one              # w, grad, m, v up; w down
  est     <- moved * ms_mb
  host    <- host_step[[i]]
  cat(sprintf("   %5dx%-5d  %7.2f  %9.2f  %7.2f  %7.2f   %s\n",
              nr, nc, one, moved, est, host,
              if (est < host) "transfer alone fits" else "transfer alone loses"))
}

# ---------------------------------------------------------------------------
# 4. The floor for any GPU step: one visit to the device
#
# ag_scale does no meaningful arithmetic, so its time is the cost of getting an
# operand there and the result back -- a fused AdamW kernel cannot be cheaper
# than that on the same operand, since it must also move w, grad, m and v. This
# turns the section 3 estimate into a measured lower bound.
# ---------------------------------------------------------------------------

if (has_gpu) {
  cat("\n4. Measured floor: one trivial GPU op (ag_scale) on the same operand\n")
  cat("   shape          scale ms   x4 operands   host Adam ms\n")
  ag_device("gpu")
  for (i in seq_along(shapes)) {
    nr <- shapes[[i]][[1L]]; nc <- shapes[[i]][[2L]]
    t  <- ag_tensor(matrix(rnorm(nr * nc), nr, nc))
    ms <- tm(function() ag_scale(t, 1.0))
    cat(sprintf("   %5dx%-5d  %9.2f  %12.2f  %12.2f\n",
                nr, nc, ms, 4 * ms, host_step[[i]]))
  }
  ag_device("cpu")
}

# ---------------------------------------------------------------------------
# 5. SGD, where the host arithmetic is far lighter
#
# Adam is the favourable case for a GPU kernel: it does the most host work per
# byte moved. Plain SGD is one multiply-subtract, so if Adam does not pay off,
# SGD cannot either -- this section makes that concrete rather than assumed.
# ---------------------------------------------------------------------------

cat("\n5. optimizer_sgd$step() on the host, for comparison\n")
cat("   shape             ms   Adam ms   SGD share of Adam\n")
ag_device("cpu")
for (i in seq_along(shapes)) {
  nr <- shapes[[i]][[1L]]; nc <- shapes[[i]][[2L]]
  p  <- ag_param(matrix(rnorm(nr * nc), nr, nc))
  g  <- matrix(rnorm(nr * nc), nr, nc)
  opt <- optimizer_sgd(list(w = p), lr = 1e-3)
  ms <- tm(function() { p$grad <- g; opt$step() })
  cat(sprintf("   %5dx%-5d  %6.2f  %8.2f  %16.2f\n",
              nr, nc, ms, host_step[[i]], ms / host_step[[i]]))
}

# ---------------------------------------------------------------------------
# 6. The best case a redesign could reach
#
# Suppose m and v were made permanently resident, so they never move. The
# weight still cannot be: .ag_data_set drops $ptr by contract, and the gradient
# is built by R closures on the host. So the floor becomes 2 uploads + 1
# download instead of 4 + 1. This section prices that optimistic variant, to
# check whether the item is merely blocked on residency work or loses outright.
# ---------------------------------------------------------------------------

cat("\n6. Optimistic variant: m/v resident, only w and grad move\n")
cat("   shape          MB moved   est ms   host ms   verdict\n")
for (i in seq_along(shapes)) {
  nr <- shapes[[i]][[1L]]; nc <- shapes[[i]][[2L]]
  moved <- 3 * mb(nr, nc)          # w, grad up; w down
  est   <- moved * ms_mb
  host  <- host_step[[i]]
  cat(sprintf("   %5dx%-5d  %9.2f  %7.2f  %7.2f   %s\n",
              nr, nc, moved, est, host,
              if (est < host) "worth pursuing" else "still loses"))
}

cat("\nReading the numbers:\n")
cat("  Section 3 est ms > host ms          -> transfer alone exceeds the whole\n")
cat("                                         host step; no kernel can win.\n")
cat("  Section 4 scale ms > host Adam ms   -> the same conclusion, measured\n")
cat("                                         rather than estimated.\n")
cat("  Section 2 access comparable to arith-> even a free kernel saves only the\n")
cat("                                         arithmetic share.\n")
cat("  Section 6 still loses               -> not blocked on residency work; the\n")
cat("                                         item closes like fused kernels did.\n")
cat("  Section 6 worth pursuing            -> revisit AFTER the upload cache,\n")
cat("                                         with m/v residency as the first step.\n")
