#!/usr/bin/env Rscript
#
# DIAGNOSTIC. Which residency change moved the numbers?
#
# The problem this exists for. GPU training diverged from CPU over ten steps
# (loss 0.4199 vs 0.4003, W maxdiff 0.434) while step 1 matched exactly. Two
# changes landed together and either could explain it:
#
#   * resident gradients   -- backward leaves $grad on the device
#   * operand caching      -- .ag_operand() keeps an uploaded input's pointer
#                             on the ag_tensor so the second use skips the
#                             upload
#
# A cumulative divergence with an exact first step is the signature of a stale
# pointer: something cached on step N is still trusted on step N+1, after the
# tape reset freed the buffer under it.
#
# HOW THIS ISOLATES, AND WHY NOT git stash. Both changes sit behind environment
# switches, so all four combinations run against ONE working tree:
#
#   GGMLR_AG_OPERAND_CACHE=0    disables operand caching
#   GGMLR_AG_RESIDENT_GRADS=0   disables resident gradients
#
# Toggling the tree with git stash to compare instead is how a whole session's
# work ended up stashed and the tree rolled back when the command was
# interrupted between stash and pop. A flag cannot do that.
#
# READING THE RESULT
#   diverges only with cache=1        -> operand caching
#   diverges only with grads_res=TRUE -> resident gradients
#   diverges in all four              -> predates both; not from this work
#   W maxdiff <= ~1e-5                -> ordinary f32 disagreement, not a bug
#
# Section 1 tests the stale-pointer hypothesis directly and costs a second: it
# prints, at every step boundary, whether the cached input pointer still claims
# to be live and how its generation compares to the pool's. After a tape reset
# the generation must have moved on and live must be FALSE. A TRUE there, with
# the generations apart, IS the defect.
#
# Run:  Rscript inst/scripts/diag_ag_residency_isolate.R
#       GGMLR_AG_OPERAND_CACHE=0 Rscript inst/scripts/diag_ag_residency_isolate.R
# Env:  GGMLR_ISO_STEPS   training steps to compare (default 10)

suppressMessages(library(ggmlR))

ns    <- asNamespace("ggmlR")
agd   <- get(".ag_data",         envir = ns)
live  <- get(".ag_ptr_is_live",  envir = ns)
state <- get(".ag_device_state", envir = ns)
bres  <- get("ag_backward_resident", envir = ns)

steps <- as.integer(Sys.getenv("GGMLR_ISO_STEPS", "10"))
cache <- Sys.getenv("GGMLR_AG_OPERAND_CACHE"); if (!nzchar(cache)) cache <- "1"

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to compare against the CPU.\n")
  quit(save = "no", status = 0)
}

cat(sprintf("operand cache = %s,  %d steps\n\n", cache, steps))

# ---------------------------------------------------------------------------
# 1. Does a cached input pointer survive the tape reset it should not survive?
# ---------------------------------------------------------------------------
cat("step boundaries -- a cached pointer must be dead after the reset\n")
cat(strrep("-", 72), "\n")

ag_device("gpu")
set.seed(1L)
local({
  W <- ag_param(matrix(rnorm(64) * 0.1, 8L, 8L))
  x <- ag_tensor(matrix(rnorm(64), 8L, 8L))
  y <- ag_tensor(matrix(rnorm(64), 8L, 8L))
  for (i in seq_len(3L)) {
    with_grad_tape({ l <- ag_mse_loss(ag_matmul(W, x), y) })
    gen  <- if (is.null(x$ctx_gen)) NA_integer_ else x$ctx_gen
    ok   <- live(x)
    # After a pass reset the pointer's generation is behind the pool's, so the
    # tensor is expected to be NOT live. Live with generations apart means the
    # check is being bypassed -- the stale-pointer case.
    flag <- if (!is.na(gen) && !identical(gen, state$ctx_gen) && ok)
              "  <== STALE, generations differ but reported live" else ""
    cat(sprintf("  step %d after tape:  x gen=%-4s pool gen=%-4s live=%-5s%s\n",
                i, gen, state$ctx_gen, ok, flag))
    backward(l)
  }
})
ag_device("cpu")

# ---------------------------------------------------------------------------
# 2. Training trajectory, CPU against GPU with and without resident gradients.
# ---------------------------------------------------------------------------
set.seed(3L)
A1 <- matrix(rnorm(64) * 0.1, 8L, 8L)
A2 <- matrix(rnorm(64) * 0.1, 8L, 8L)
X  <- matrix(rnorm(64), 8L, 8L)
Y  <- matrix(rnorm(64), 8L, 8L)

# Same weights, same data, same number of steps on both devices: the only
# variable is where the arithmetic happens.
run <- function(dev, resident) {
  ag_device(dev)
  if (dev == "gpu") bres(resident)
  W1  <- ag_param(A1); W2 <- ag_param(A2)
  x   <- ag_tensor(X); y <- ag_tensor(Y)
  opt <- optimizer_adam(list(W1 = W1, W2 = W2), lr = 0.05)
  ls  <- numeric(steps)
  for (i in seq_len(steps)) {
    with_grad_tape({
      l <- ag_mse_loss(ag_matmul(W2, ag_relu(ag_matmul(W1, x))), y)
    })
    backward(l); opt$step(); opt$zero_grad()
    ls[i] <- as.numeric(agd(l))
  }
  out <- list(ls = ls, w = agd(W1))
  if (dev == "gpu") bres(FALSE)
  ag_device("cpu")
  out
}

cpu <- run("cpu", FALSE)

cat("\ntrajectories\n")
cat(strrep("-", 72), "\n")
cat(sprintf("%-28s loss[1] %.6f  loss[%d] %.6f\n",
            "cpu", cpu$ls[1L], steps, cpu$ls[steps]))

for (res in c(FALSE, TRUE)) {
  g <- run("gpu", res)
  # loss[1] is printed too: an exact first step with a divergent last one is
  # the accumulation signature, and tells this apart from a plain shape or
  # formula error, which would already be wrong at step 1.
  cat(sprintf("cache=%s grads_res=%-5s      loss[1] %.6f  loss[%d] %.6f  W maxdiff %.4g\n",
              cache, res, g$ls[1L], steps, g$ls[steps],
              max(abs(cpu$w - g$w))))
}

cat("\nrun again with GGMLR_AG_OPERAND_CACHE=0 for the other half of the matrix\n")

# ---------------------------------------------------------------------------
# 3. Where does the resident path stop agreeing -- the gradient, or the step?
#
# Section 2 compares the END of training, which cannot say whether the gradient
# was already wrong or the optimizer consumed a correct one badly. This asks the
# narrower question: for ONE backward pass, from identical weights, does the
# resident gradient equal the downloaded one?
#
#   grads differ           -> the backward graph or its leaf handles
#   grads equal, W differs -> the optimizer step consuming the handle
# ---------------------------------------------------------------------------
cat("\none backward pass, resident gradient against downloaded\n")
cat(strrep("-", 72), "\n")

asm <- get(".ag_as_matrix", envir = ns)

one_pass <- function(resident) {
  ag_device("gpu"); bres(resident)
  set.seed(3L)
  W1 <- ag_param(A1); W2 <- ag_param(A2)
  x  <- ag_tensor(X); y <- ag_tensor(Y)
  with_grad_tape({
    l <- ag_mse_loss(ag_matmul(W2, ag_relu(ag_matmul(W1, x))), y)
  })
  backward(l)
  # Materialise INSIDE the tape's lifetime: a resident gradient lives in the
  # pass pool, which the next with_grad_tape() frees.
  out <- list(g1 = asm(W1$grad), g2 = asm(W2$grad))
  bres(FALSE); ag_device("cpu")
  out
}

a <- one_pass(FALSE)
b <- one_pass(TRUE)
cat(sprintf("  W1 grad maxdiff %.4g\n", max(abs(a$g1 - b$g1))))
cat(sprintf("  W2 grad maxdiff %.4g\n", max(abs(a$g2 - b$g2))))

# And the same after several steps: a difference that only appears later is the
# accumulation signature, not a wrong rule.
multi <- function(resident, k) {
  ag_device("gpu"); bres(resident)
  set.seed(3L)
  W1 <- ag_param(A1); W2 <- ag_param(A2)
  x  <- ag_tensor(X); y <- ag_tensor(Y)
  opt <- optimizer_adam(list(W1 = W1, W2 = W2), lr = 0.05)
  for (i in seq_len(k)) {
    with_grad_tape({
      l <- ag_mse_loss(ag_matmul(W2, ag_relu(ag_matmul(W1, x))), y)
    })
    backward(l); opt$step(); opt$zero_grad()
  }
  out <- agd(W1)
  bres(FALSE); ag_device("cpu")
  out
}
for (k in c(1L, 2L, 3L, 5L)) {
  cat(sprintf("  after %d step(s): W1 maxdiff %.4g\n", k,
              max(abs(multi(FALSE, k) - multi(TRUE, k)))))
}
