#!/usr/bin/env Rscript
#
# Does the ag_batch_norm train-mode gradient really differ from the naive
# grad_out/std, and which of the two matches finite differences?
#
# The tests in test-ag-layers.R pass, but a passing test proves nothing until
# the wrong code is shown to fail it. Rebuilding the package twice just to see
# that is expensive, so this script compares the numbers directly instead:
# finite differences are the arbiter, and both candidate formulas are scored
# against them on the same data.

suppressMessages(library(ggmlR))

cat("ggmlR from:", find.package("ggmlR"), "\n\n")

set.seed(26)
F_ <- 4L; N <- 5L
x0  <- matrix(rnorm(F_ * N), F_, N)
eps <- 1e-5

# Loss = sum(w * bn_train(x)) with gamma=1, beta=0.
# w = 1 is the constant-grad_out case the "not naive" test uses; a non-constant
# w is added because that is where the two formulas differ in a general way.
run_case <- function(w, label) {

  # ---- forward in plain R, so finite differences never touch the tape ----
  loss_of <- function(x_val) {
    mu  <- rowMeans(x_val)
    var <- rowMeans((x_val - mu)^2)
    std <- sqrt(var + eps)
    sum(w * ((x_val - mu) / std))
  }

  # ---- 1. finite differences: the arbiter ----
  h  <- 1e-6
  fd <- matrix(0, F_, N)
  for (i in seq_along(x0)) {
    xp <- x0; xp[i] <- xp[i] + h
    xm <- x0; xm[i] <- xm[i] - h
    fd[i] <- (loss_of(xp) - loss_of(xm)) / (2 * h)
  }

  # ---- 2. what the naive formula would give ----
  mu    <- rowMeans(x0)
  var   <- rowMeans((x0 - mu)^2)
  std   <- sqrt(var + eps)
  naive <- w / std

  # ---- 3. what the package actually computes ----
  bn <- ag_batch_norm(F_)
  ag_train(bn)
  x  <- ag_param(x0)
  with_grad_tape({
    loss <- ag_sum(ag_mul(bn$forward(x), ag_tensor(w)))
  })
  pkg <- get0(as.character(x$id), envir = backward(loss))

  err_pkg   <- max(abs(pkg   - fd))
  err_naive <- max(abs(naive - fd))

  cat(sprintf("--- %s ---\n", label))
  cat(sprintf("  package vs finite differences : %.3e  %s\n",
              err_pkg, if (err_pkg < 1e-6) "MATCH" else "MISMATCH"))
  cat(sprintf("  naive   vs finite differences : %.3e  %s\n",
              err_naive, if (err_naive < 1e-6) "MATCH" else "MISMATCH"))
  cat(sprintf("  gap between the two formulas  : %.3e\n\n",
              max(abs(pkg - naive))))

  invisible(list(pkg = err_pkg, naive = err_naive,
                 gap = max(abs(pkg - naive))))
}

r1 <- run_case(matrix(1.0, F_, N), "constant grad_out (the 'not naive' test)")
r2 <- run_case(matrix(seq(0.3, by = 0.17, length.out = F_ * N), F_, N),
               "varying grad_out (the gradcheck test)")

ok <- r1$pkg < 1e-6 && r2$pkg < 1e-6 &&
      r1$naive > 1e-6 && r2$naive > 1e-6 &&
      r1$gap > 1e-6 && r2$gap > 1e-6

cat(if (ok) {
  "VERDICT: the exact formula matches finite differences, the naive one does\nnot, and the two differ on this data -- the tests are testing something.\n"
} else {
  "VERDICT: inconclusive. Either the two formulas agree on this data (the\ntests would pass on broken code) or the package disagrees with finite\ndifferences (the fix is wrong). Read the numbers above.\n"
})
