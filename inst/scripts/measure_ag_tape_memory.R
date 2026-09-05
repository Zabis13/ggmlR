#!/usr/bin/env Rscript
#
# Is activation lifetime management worth doing on the ag_* path?
#
# TODO.md section 3 carries "Activation lifetime management -- явный учёт, что
# живо до backward". The tape (.ag_tape, R/autograd.R) is cleared only by
# zero_grad(), which runs AFTER the optimizer step, so the peak is the whole
# tape at once. The proposed change is to release each node's captured
# activations as backward() walks past it, which would lower that peak.
#
# What decides whether that pays is not the size of the tape but the FREEABLE
# share of it. Most of what a backward closure references is not freeable:
#
#   a weight's matrix and the a_snap an ag_matmul closure captured are the SAME
#   object, not a copy (verified by address). Clearing the tape drops the
#   reference, but the parameter still holds the value, so nothing is released.
#
# ag_tape_memory() separates the two: operands, which outlive the tape, from
# activations, which are the intermediates only the tape holds. On the unit
# tests the freeable share came out around a third -- of fractions of a
# megabyte, on tapes of three to nine nodes. That is too small and too shallow
# to decide on, which is what this script is for: real depths, real widths, and
# the ratio as a function of both.
#
# The shape of the answer to look for:
#
#   * activations rise with DEPTH while operands stay flat -- deep models are
#     dominated by intermediates, and freeing them as backward passes is worth
#     doing. The deeper the model the better the case.
#   * the ratio stays flat as depth grows -- every layer brings its own weights
#     too, the freeable share does not concentrate, and the change buys a fixed
#     fraction rather than a growing one. Judge it against the cost of the
#     bookkeeping, which is not free: backward() would have to know which nodes
#     are finished, and .ag_bwd_write_leaf_grads walks the tape AFTER the loop
#     (R/autograd.R), so inputs cannot be dropped even when snapshots can.
#
# Batch size is swept separately from depth for the same reason it matters in
# training: activations scale with the batch, weights do not. A model that
# looks weight-dominated at batch 1 can be activation-dominated at batch 128,
# and that is the regime where training actually runs.
#
# Run:  Rscript inst/scripts/measure_ag_tape_memory.R
# Env:  GGMLR_TAPE_D      hidden width for the depth sweep (default 128)
#       GGMLR_TAPE_BATCH  batch for the depth sweep (default 32)

suppressMessages(library(ggmlR))

d_hidden <- as.integer(Sys.getenv("GGMLR_TAPE_D",     "128"))
batch    <- as.integer(Sys.getenv("GGMLR_TAPE_BATCH", "32"))

ag_device("cpu")

mb <- function(b) b / 1024^2

# One MLP forward+backward, then report the tape while it is still alive.
#
# The report has to happen BEFORE zero_grad(): that is what clears the tape, so
# calling it afterwards measures an empty one. Nothing here calls step().
mlp_tape <- function(depth, d = d_hidden, n = batch) {
  ws <- lapply(seq_len(depth), function(i) ag_param(matrix(rnorm(d * d) * 0.05, d, d)))
  x  <- ag_tensor(matrix(rnorm(d * n), d, n))
  y  <- matrix(0.0, d, n)
  with_grad_tape({
    h <- x
    for (w in ws) h <- ag_relu(ag_matmul(w, h))
    loss <- ag_mse_loss(h, y)
  })
  backward(loss)
  ag_tape_memory(quiet = TRUE)
}

row <- function(label, r) {
  frac <- if (r$bytes_total > 0) 100 * r$bytes_snapshots / r$bytes_total else 0
  cat(sprintf("   %-12s %6d  %8.2f  %9.2f  %10.2f  %7.1f%%\n",
              label, r$nodes, mb(r$bytes_total), mb(r$bytes_inputs),
              mb(r$bytes_snapshots), frac))
}

hdr <- function() {
  cat("   case          nodes  total MB  operands MB  activ. MB  freeable\n")
}

cat(sprintf("MLP sweep: hidden = %d, batch = %d\n\n", d_hidden, batch))

# ---------------------------------------------------------------------------
# 1. Depth
#
# Each layer adds one weight (not freeable) and one activation per op (freeable).
# If the freeable share climbs with depth, the change is worth more on the
# models people actually train; if it is flat, depth is not the argument.
# ---------------------------------------------------------------------------

cat("1. Depth sweep (batch fixed)\n")
hdr()
for (depth in c(1L, 2L, 4L, 8L, 16L)) {
  row(sprintf("depth %d", depth), mlp_tape(depth))
}

# ---------------------------------------------------------------------------
# 2. Batch
#
# Activations carry the batch dimension, weights do not. This is the sweep that
# says whether the freeable share grows in the regime training runs in.
# ---------------------------------------------------------------------------

cat("\n2. Batch sweep (depth 4)\n")
hdr()
for (n in c(1L, 8L, 32L, 128L)) {
  row(sprintf("batch %d", n), mlp_tape(4L, n = n))
}

# ---------------------------------------------------------------------------
# 3. Width
#
# Weights grow as d^2, activations as d*batch. Widening should therefore move
# the ratio TOWARDS the weights -- the opposite of the batch sweep. If it does
# not, the model of what the tape holds is wrong and the other two sweeps need
# re-reading before anything is built on them.
# ---------------------------------------------------------------------------

cat("\n3. Width sweep (depth 4, batch fixed)\n")
hdr()
for (d in c(64L, 128L, 256L, 512L)) {
  row(sprintf("width %d", d), mlp_tape(4L, d = d))
}

# ---------------------------------------------------------------------------
# 4. Where does it sit, by operation?
#
# A per-op breakdown of the deepest case. If the freeable memory concentrates
# in a few ops, a targeted fix beats a general lifetime mechanism -- the same
# reasoning that sent the fused-kernel item to arithmetic intensity rather than
# to pattern frequency.
# ---------------------------------------------------------------------------

cat("\n4. Per-op breakdown (depth 16)\n")
r <- mlp_tape(16L)
print(r$by_op, row.names = FALSE)

# ---------------------------------------------------------------------------
# 5. An attention block, which is not shaped like an MLP
#
# ag_multihead_attention holds more per node than a dense layer does: softmax
# keeps p_snap, the whole attention matrix, whose size goes as seq^2 rather
# than as d*batch. If any shape makes the freeable share dominate, it is this
# one -- and it is the shape flash attention was added for.
# ---------------------------------------------------------------------------

cat("\n5. Attention block\n")
hdr()
for (seq_len_ in c(32L, 64L, 128L)) {
  d_model <- 64L
  mha <- ag_multihead_attention(d_model = d_model, n_heads = 4L)
  x   <- ag_tensor(matrix(rnorm(d_model * seq_len_), d_model, seq_len_))
  y   <- matrix(0.0, d_model, seq_len_)
  with_grad_tape({
    out  <- mha$forward(x)
    loss <- ag_mse_loss(out, y)
  })
  backward(loss)
  row(sprintf("seq %d", seq_len_), ag_tape_memory(quiet = TRUE))
}

cat("\nReading the numbers:\n")
cat("  freeable share rising with depth  -> lifetime management pays on deep\n")
cat("                                       models; build it.\n")
cat("  freeable share flat across depth  -> a fixed fraction, weigh it against\n")
cat("                                       the bookkeeping backward() needs.\n")
cat("  freeable share rising with batch  -> it pays in the training regime even\n")
cat("                                       if depth alone does not show it.\n")
cat("  share falling with width          -> expected (weights d^2 vs activ.\n")
cat("                                       d*batch); if not, re-read sweeps 1-2.\n")
cat("  attention >> MLP                  -> target attention specifically\n")
cat("                                       rather than the tape in general.\n")
