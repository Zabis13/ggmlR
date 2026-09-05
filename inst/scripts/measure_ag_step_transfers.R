#!/usr/bin/env Rscript
#
# CENSUS. How many host<->device crossings does one training step cost, and
# which stage pays for each of them?
#
# Why it exists. Full residency is a sequence of changes, each of which is
# supposed to remove traffic from one part of the step. "Removed" is only
# meaningful against a number, and the number has to be per stage -- see the
# measured points below for the run where the total stayed put while two
# different things changed underneath it.
#
# What is being counted, and why counts rather than milliseconds. A forgotten
# .ag_data() does not show up as an error or a wrong answer -- it shows up as a
# speedup that never arrives, and timings alone cannot separate "the transfer is
# gone" from "the transfer is there but noise hid it". Crossings are discrete
# and attributable to a call site, so they answer a sharper question: WHICH code
# still moves bytes, and how many times per step.
#
# Measured points so far, per step, on the shape below:
#
#   before 3.1   10 crossings, 0.188 MB   weights uploaded per op
#   after  3.3   10 crossings, 0.188 MB   optimizer stopped touching the host,
#                                         but the forward began materialising
#                                         the now-resident weights instead
#   after  1     9 crossings, 0.157 MB   forward 6 / backward 1 / step 2
#
# The middle row is why the breakdown below exists. The total did not move,
# and reading only the total would have said "3.3 changed nothing" -- while in
# fact the optimizer's four crossings were gone and two new ones had appeared
# in the forward, from a different cause. A stage-by-stage count separates
# those; a total cannot.
#
# What remains after stage 1, and which stage owns it:
#   forward   ag_add materialises its broadcast operand and downloads its
#             result (class 3: broadcast indexes an R matrix)
#   backward  the closure path computes in R, so it pulls one snapshot back
#   step      the gradient arrives as a host matrix and is uploaded once
#
# Run:  Rscript inst/scripts/measure_ag_step_transfers.R
# Env:  GGMLR_XFER_STEPS   optimizer steps to measure (default 5)
#       GGMLR_XFER_DIN     input width  (default 128)
#       GGMLR_XFER_DOUT    output width (default 64)
#       GGMLR_XFER_BATCH   batch size   (default 64)

suppressMessages(library(ggmlR))

ns          <- asNamespace("ggmlR")
xfer_count  <- get("ag_xfer_count",  envir = ns)
xfer_report <- get("ag_xfer_report", envir = ns)
xfer_reset  <- get("ag_xfer_reset",  envir = ns)
tape_mem    <- get(".ag_tape_mem",   envir = ns)

steps <- as.integer(Sys.getenv("GGMLR_XFER_STEPS", "5"))
d_in  <- as.integer(Sys.getenv("GGMLR_XFER_DIN",   "128"))
d_out <- as.integer(Sys.getenv("GGMLR_XFER_DOUT",  "64"))
batch <- as.integer(Sys.getenv("GGMLR_XFER_BATCH", "64"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: this measures GPU traffic, so there is nothing to do.\n")
  quit(save = "no", status = 0)
}

cat(sprintf("Transfer census: resident weights, device Adam, stage-1 operands\n"))
cat(sprintf("shape: %d x %d, batch %d, %d steps\n\n", d_in, d_out, batch, steps))

ag_device("gpu")
on.exit(ag_device("cpu"), add = TRUE)
set.seed(1L)

# A single dense layer trained with Adam: the smallest thing that exercises the
# whole cycle (forward, loss, backward, optimizer step) without the shape of a
# bigger model obscuring where the traffic comes from.
W <- ag_param(matrix(rnorm(d_in * d_out) * 0.05, d_out, d_in))
b <- ag_param(matrix(0, d_out, 1L))
x <- ag_tensor(matrix(rnorm(d_in * batch), d_in, batch))
y <- ag_tensor(matrix(rnorm(d_out * batch), d_out, batch))

opt <- optimizer_adam(list(W = W, b = b), lr = 0.01)

one_step <- function() {
  with_grad_tape({
    out  <- ag_add(ag_matmul(W, x), b)
    loss <- ag_mse_loss(out, y)
  })
  backward(loss)
  opt$step()
  opt$zero_grad()
  invisible(NULL)
}

# One untimed step first. The first step through any path pays for things that
# happen once -- context creation, buffer allocation, shader warm-up -- and
# counting those as part of "a step" would overstate the steady state that the
# residency work is actually trying to improve.
one_step()

xfer_count(TRUE)
t0 <- Sys.time()
for (i in seq_len(steps)) one_step()
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs")) * 1000
xfer_count(FALSE)

df <- xfer_report()

cat(sprintf("\n%.1f ms for %d steps (%.2f ms/step)\n", elapsed, steps, elapsed / steps))

if (!is.null(df)) {
  per_step <- sum(df$n) / steps
  cat(sprintf("%.1f crossings per step, %.3f MB per step\n",
              per_step, sum(df$mb) / steps))

}

# Per-stage breakdown, measured rather than attributed.
#
# The total alone is not enough to tell one stage's improvement from another's,
# and naming a site after the stage that used to own it goes stale: gpu_to_r was
# the optimizer reading weights before stage 3.3, and is the forward
# materialising a broadcast operand after it. Same site, different cause. So the
# stages are timed separately and each reports its own crossings; whatever moves
# next, the number is attributed to the stage that actually produced it.
cat("\nper-stage crossings (the breakdown, not the total, is what to compare)\n")
cat(strrep("-", 72), "\n")

stage_count <- function(label, f) {
  xfer_count(TRUE)
  f()
  xfer_count(FALSE)
  d <- suppressMessages(utils::capture.output(sdf <- xfer_report(0L)))
  n  <- if (is.null(sdf)) 0L else sum(sdf$n)
  mb <- if (is.null(sdf)) 0   else sum(sdf$mb)
  sites <- if (is.null(sdf) || !nrow(sdf)) "-" else
             paste(sprintf("%s %s x%d", sdf$dir, sdf$site, sdf$n), collapse = ", ")
  cat(sprintf("%-9s %5.1f crossings %8.3f MB   %s\n", label, n, mb, sites))
  invisible(NULL)
}

local({
  loss <- NULL
  fwd  <- function() with_grad_tape({
    out   <- ag_add(ag_matmul(W, x), b)
    loss <<- ag_mse_loss(out, y)
  })
  fwd(); backward(loss); opt$step(); opt$zero_grad()   # warm, then measure

  stage_count("forward",  function() fwd())
  stage_count("backward", function() backward(loss))
  stage_count("step",     function() opt$step())
  opt$zero_grad()
})

m <- tape_mem()
cat(sprintf("\npersistent pool: %.2f MB in %d buffers (weights + moments)\n",
            m$p_buffer_bytes / 1024^2, m$p_n_buffers))
cat(sprintf("pass pool:       %.2f MB in %d buffers\n",
            m$buffer_bytes / 1024^2, m$n_buffers))
