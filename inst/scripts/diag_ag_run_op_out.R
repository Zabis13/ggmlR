#!/usr/bin/env Rscript
#
# DIAGNOSTIC. What exactly does .ag_run_op(out =) do?
#
# Four attempts at "copy this tensor into that one" inside the device Adam step
# each failed differently -- rounding through a scale kernel, a destination
# passed as an operand, a root with no operation, and now a numeric difference
# that survives all of it. Rather than guess a fifth time, this exercises the
# mechanism on its own, away from the optimizer, where every value is known in
# advance and any discrepancy is attributable.
#
# What is being established, in order:
#   1. does out = write the destination at all, and keep its pointer;
#   2. is the write EXACT, or does the value change on the way (the failure that
#      ggml_scale(x, 1) had -- an "identity" that quietly rounds);
#   3. does it still hold when source and destination differ in dtype;
#   4. what happens when a tensor is both an operand and the destination -- the
#      aliasing case the Adam step originally had;
#   5. does a chain of copies accumulate error across repetitions, which is what
#      a training loop actually does.
#
# Run:  Rscript inst/scripts/diag_ag_run_op_out.R

suppressMessages(library(ggmlR))

ns     <- asNamespace("ggmlR")
run_op <- get(".ag_run_op",   envir = ns)
mkh    <- get(".ag_handle",   envir = ns)
r2g    <- get(".ag_r_to_gpu", envir = ns)
asm    <- get(".ag_as_matrix", envir = ns)

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to exercise.\n")
  quit(save = "no", status = 0)
}

ag_device("gpu")
# Not on.exit(): at the top level of a script that is source()d, the handler
# runs as soon as source() returns from the current expression -- releasing the
# backend before the first section gets to use it. The device is put back at the
# end of the file instead, so this works under Rscript and under
# pkgload::load_all() + source() alike.

sh  <- c(3L, 4L)
mk  <- function(m, scope = "persistent")
  mkh(r2g(m, scope = scope), dim(m), scope = scope)

report <- function(label, got, want) {
  d <- max(abs(got - want))
  cat(sprintf("  %-46s maxdiff %.3g%s\n", label, d,
              if (d > 1e-6) "   <== DIFFERS" else ""))
}

cat("1. does out= write, and keep the destination pointer\n")
local({
  src <- matrix(seq_len(12) / 12, 3L, 4L)
  dst <- mk(matrix(0, 3L, 4L))
  ptr <- dst$ptr
  got <- run_op(function(ctx, p) ggml_dup(ctx, p[[1L]]),
                inputs = list(src), out_shape = sh, scope = "pass", out = dst)
  cat(sprintf("  %-46s %s\n", "returned handle is the destination",
              identical(got$ptr, ptr)))
  report("copied value", asm(dst), src)
})

cat("\n2. is the copy exact, or does it round on the way\n")
local({
  # Values chosen to be representable in f32 but NOT in f16: if the copy passes
  # through a reduced-precision kernel, these lose their tail digits.
  src <- matrix(c(0.100000001, 0.200000003, 1/3, 2/3,
                  1e-7, 1e7, 123456.78, 0.987654321,
                  -0.1, -1/7, 3.14159265, -2.71828182), 3L, 4L)
  dst <- mk(matrix(0, 3L, 4L))
  run_op(function(ctx, p) ggml_dup(ctx, p[[1L]]),
         inputs = list(src), out_shape = sh, scope = "pass", out = dst)
  report("dup", asm(dst), src)

  # The comparison that matters: scale-by-one was the first attempt, and it is
  # here to show what "not an identity" looks like on this backend.
  dst2 <- mk(matrix(0, 3L, 4L))
  run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 1),
         inputs = list(src), out_shape = sh, scope = "pass", out = dst2)
  report("scale(x, 1)  [for contrast]", asm(dst2), src)
})

cat("\n3. dtype of the destination\n")
local({
  cat(sprintf("  session dtype: %s\n", ag_default_dtype()))
  src <- matrix(seq(0.001, 0.012, by = 0.001), 3L, 4L)
  dst <- mk(matrix(0, 3L, 4L))
  run_op(function(ctx, p) ggml_dup(ctx, p[[1L]]),
         inputs = list(src), out_shape = sh, scope = "pass", out = dst)
  report("small magnitudes", asm(dst), src)
})

cat("\n4. aliasing: destination also an operand (the original Adam bug)\n")
local({
  a   <- matrix(seq_len(12) / 12, 3L, 4L)
  dst <- mk(a)
  # b*dst + 0.5 computed INTO dst -- reads and writes the same tensor in one
  # graph. If this comes out right it is luck, not a guarantee.
  run_op(function(ctx, p) ggml_scale_bias(ctx, p[[1L]], 2, 0.5),
         inputs = list(dst), out_shape = sh, scope = "pass", out = dst)
  report("dst <- 2*dst + 0.5, aliased", asm(dst), a * 2 + 0.5)
})

cat("\n5. repeated copies, as a training loop does\n")
local({
  # m <- 0.9*m + 0.1*g, twenty times, against the same recurrence in R. This is
  # the Adam moment update in miniature: if a single copy is slightly lossy, the
  # error compounds here and nowhere else.
  g    <- matrix(seq_len(12) / 12, 3L, 4L)
  m    <- mk(matrix(0, 3L, 4L))
  gh   <- mk(g, scope = "pass")
  ref  <- matrix(0, 3L, 4L)
  for (i in seq_len(20L)) {
    new_m <- run_op(function(ctx, p)
                      ggml_add(ctx, ggml_scale(ctx, p[[1L]], 0.9),
                               ggml_scale(ctx, p[[2L]], 0.1)),
                    inputs = list(m, gh), out_shape = sh,
                    scope = "pass", resident = TRUE)
    run_op(function(ctx, p) ggml_dup(ctx, p[[1L]]),
           inputs = list(new_m), out_shape = sh, scope = "pass", out = m)
    ref <- 0.9 * ref + 0.1 * g
  }
  report("m after 20 updates", asm(m), ref)
})

cat("\n6. the Adam step itself, three steps of a CONSTANT gradient\n")
local({
  # A constant gradient makes the answer known in closed form: Adam's m/sqrt(v)
  # normalises to 1, so each step moves the weight by exactly lr. Three steps
  # from 0.9 at lr = 0.05 must land on 0.75, and any other number says the
  # device is not computing the update the formula describes -- the arithmetic
  # is identical on paper (verified in R: both spellings give 0.75), so a
  # difference here is in the kernels or the operand wiring, not the algebra.
  step_dev <- get(".ag_adam_step_device", envir = ns)
  w  <- ag_param(matrix(0.9, 3L, 3L))
  gm <- matrix(0.4, 3L, 3L)
  opt <- optimizer_adam(list(w = w), lr = 0.05)
  cat(sprintf("  resident optimizer: %s\n", isTRUE(opt$resident)))
  for (i in seq_len(3L)) { w$grad <- gm; opt$step() }
  got <- get(".ag_data", envir = ns)(w)
  report("w after 3 constant-gradient steps", got, matrix(0.75, 3L, 3L))

  # And the moments, which say WHICH half is wrong: after three steps of a
  # constant g, m = g*(1-b1^3) and v = g^2*(1-b2^3) exactly.
  b1 <- 0.9; b2 <- 0.999
  report("  m after 3 steps", asm(opt$m$w), matrix(0.4 * (1 - b1^3), 3L, 3L))
  report("  v after 3 steps", asm(opt$v$w), matrix(0.16 * (1 - b2^3), 3L, 3L))
})

cat("\n7. the weight update, node by node\n")
local({
  # Section 6 narrowed it to the weight update: m and v come out right, w does
  # not. This evaluates the same expression one node at a time against R, so the
  # step that loses the value is named rather than inferred.
  #
  # The numbers are the ones a constant gradient produces at t = 3, where the
  # true update is exactly lr:
  #   m = 0.1084,  v = 0.00047952,  bc1 = 0.271, bc2 = 0.002997
  #   num = m*(lr/bc1) = 0.02,  den = sqrt(v/bc2) = 0.4,  num/den = 0.05
  # Note how far apart the magnitudes are: v/bc2 multiplies by ~334, and eps is
  # 1e-8 against a divisor of 0.4. Either is a candidate for being lost.
  lr <- 0.05; eps <- 1e-8
  b1 <- 0.9; b2 <- 0.999; t <- 3L
  bc1 <- 1 - b1^t; bc2 <- 1 - b2^t
  mM <- matrix(0.1084, 3L, 4L); vM <- matrix(0.00047952, 3L, 4L)
  mh <- mk(mM, scope = "pass"); vh <- mk(vM, scope = "pass")

  num <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], lr / bc1),
                inputs = list(mh), out_shape = sh, scope = "pass")
  report("num = m * (lr/bc1)", num, mM * (lr / bc1))

  vs <- run_op(function(ctx, p) ggml_scale(ctx, p[[1L]], 1 / bc2),
               inputs = list(vh), out_shape = sh, scope = "pass")
  report("v / bc2", vs, vM / bc2)

  sq <- run_op(function(ctx, p) ggml_sqrt(ctx, ggml_scale(ctx, p[[1L]], 1 / bc2)),
               inputs = list(vh), out_shape = sh, scope = "pass")
  report("sqrt(v / bc2)", sq, sqrt(vM / bc2))

  den <- run_op(function(ctx, p)
                  ggml_scale_bias(ctx,
                    ggml_sqrt(ctx, ggml_scale(ctx, p[[1L]], 1 / bc2)), 1, eps),
                inputs = list(vh), out_shape = sh, scope = "pass")
  report("den = sqrt(v/bc2) + eps", den, sqrt(vM / bc2) + eps)

  # The division and the subtraction, as the step builds them.
  wM <- matrix(0.9, 3L, 4L); wh2 <- mk(wM, scope = "pass")
  upd <- run_op(function(ctx, p) {
                  n_ <- ggml_scale(ctx, p[[2L]], lr / bc1)
                  d_ <- ggml_scale_bias(ctx,
                          ggml_sqrt(ctx, ggml_scale(ctx, p[[3L]], 1 / bc2)), 1, eps)
                  ggml_sub(ctx, p[[1L]], ggml_div(ctx, n_, d_))
                },
                inputs = list(wh2, mh, vh), out_shape = sh, scope = "pass")
  report("w - num/den  (whole expression)", upd, wM - (mM * (lr / bc1)) / (sqrt(vM / bc2) + eps))

  # Does the error scale with the bias-correction constants, or survive without
  # them? With bc1 = bc2 = 1 the small-divisor and large-multiplier both go
  # away: an error that persists here is in the scalar constants or the ops
  # themselves, one that disappears is about the magnitudes.
  #
  # Worth stating what the arithmetic already rules out. Reproducing the
  # observed weight needs the denominator too big by a factor of 1.254 -- 25%.
  # No relaxed-precision sqrt is off by that much (those land at 1e-3..1e-2),
  # and no plausible substitution of t, of an unupdated v, or of a shifted bc2
  # produces it either (all checked in R: 0.0762, 0.1500, 0.1896, none 0.1196).
  # So the expected finding here is a NODE that disagrees, not a small drift.
  upd1 <- run_op(function(ctx, p) {
                   n_ <- ggml_scale(ctx, p[[2L]], lr)
                   d_ <- ggml_scale_bias(ctx, ggml_sqrt(ctx, p[[3L]]), 1, eps)
                   ggml_sub(ctx, p[[1L]], ggml_div(ctx, n_, d_))
                 },
                 inputs = list(wh2, mh, vh), out_shape = sh, scope = "pass")
  report("same with bc1 = bc2 = 1", upd1, wM - (mM * lr) / (sqrt(vM) + eps))

  # And the two binary ops on their own, away from any scaling.
  d0 <- mk(matrix(0.4, 3L, 4L), scope = "pass")
  n0 <- mk(matrix(0.02, 3L, 4L), scope = "pass")
  dv <- run_op(function(ctx, p) ggml_div(ctx, p[[1L]], p[[2L]]),
               inputs = list(n0, d0), out_shape = sh, scope = "pass")
  report("div: 0.02 / 0.4", dv, matrix(0.05, 3L, 4L))
  sb <- run_op(function(ctx, p) ggml_sub(ctx, p[[1L]], p[[2L]]),
               inputs = list(wh2, n0), out_shape = sh, scope = "pass")
  report("sub: 0.9 - 0.02", sb, matrix(0.88, 3L, 4L))
})

cat("\n8. ONE Adam step, every intermediate named\n")
local({
  # Section 7 showed the expression is right when its operands are handed in
  # directly, and section 6 showed the step gets a different answer. So the
  # difference is in what the step FEEDS the expression, not in the expression.
  #
  # At t = 1 with m = v = 0 and a constant g, every value is known:
  #   m1 = 0.1*g,  v1 = 0.001*g^2,  bc1 = 0.1,  bc2 = 0.001
  #   num = m1*(lr/bc1) = lr*g,  den = sqrt(v1/bc2) = g,  update = lr exactly.
  # Printing each one says which of them the step actually produced.
  lr <- 0.05; eps <- 1e-8; b1 <- 0.9; b2 <- 0.999
  gv <- 0.4
  w  <- ag_param(matrix(0.9, 3L, 3L))
  opt <- optimizer_adam(list(w = w), lr = lr)
  w$grad <- matrix(gv, 3L, 3L)
  opt$step()

  agd <- get(".ag_data", envir = ns)
  cat(sprintf("  m1  got %.8f  want %.8f\n", asm(opt$m$w)[1L], (1 - b1) * gv))
  cat(sprintf("  v1  got %.10f want %.10f\n", asm(opt$v$w)[1L], (1 - b2) * gv^2))
  cat(sprintf("  w1  got %.8f  want %.8f   (a step of exactly lr)\n",
              agd(w)[1L], 0.9 - lr))

  # Second step, same gradient: the update is lr again, and this is where a
  # stale operand would first show, since step 1 reads zero moments.
  w$grad <- matrix(gv, 3L, 3L)
  opt$step()
  cat(sprintf("  w2  got %.8f  want %.8f\n", agd(w)[1L], 0.9 - 2 * lr))
})

cat("\nA DIFFERS on 1 or 2 means the copy itself is wrong.\n")
cat("Only on 5 means it is precision compounding, not a logic error.\n")
cat("Only on 4 means aliasing, which the Adam step already avoids.\n")
cat("On 6 with m and v correct, the weight update is at fault; with m or v\n")
cat("wrong, the moment update is -- and the copy back into them is the suspect.\n")

ag_device("cpu")
