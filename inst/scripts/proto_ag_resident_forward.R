#!/usr/bin/env Rscript
#
# PROTOTYPE, not an implementation. Throw away after reading the number.
#
# The question. measure_ag_forward_profile.R says the forward is 40-55% of a
# training step and 46-85% of it is transfer, so the transfer-only ceiling on a
# whole step is 1.22-1.35x. Component 4 is ~21 helpers of work. Before writing
# them, measure what a resident forward ACTUALLY costs on a short chain -- the
# same method that settled checkpointing and flash attention.
#
# Why the weight cache is NOT a row here. Section 3 of the profile says 67-89%
# of the upload is weights, so a row with weights resident looks like the
# obvious third comparison. It cannot be measured in this geometry: each layer
# has its OWN weight and uses it once per pass, so there is nothing to reuse
# WITHIN a pass, and the reset between passes (see below) invalidates any handle
# held ACROSS passes. A weight cache pays off between optimizer steps, which is
# a different experiment with its own invalidation policy. Measuring it here
# would fold two variables into one number -- the exact mistake this script
# exists to avoid.
#
#   today          .ag_run_op(resident = FALSE), matrices in, matrix out.
#                  Every operand uploaded, every result downloaded.
#   resident_act   the activation travels as an ag_handle; each weight is still
#                  uploaded from an R matrix on every op. Pure residency, one
#                  variable changed.
#   cpu            R's own %*% and pmax, threaded BLAS -- the honest opponent.
#
# Geometry is the model's, not the ceiling script's: ONE WEIGHT PER LAYER, as in
# measure_ag_forward_profile.R's mk(). measure_ag_residency_ceiling.R reuses a
# single weight down the chain, which hands the weight cache a win it does not
# get in a real forward pass, where each weight is touched once.
#
# What the number means. The forward speedup is NOT the answer. It is divided
# by Amdahl against the measured forward share of a full step -- component 3
# looked like 8.20x on the forward and came in at 1.35x on the step for exactly
# this reason. The step column is the one that decides.
#
# Correctness: maxdiff of resident_act against today. This changes no gradient
# logic, so no FD check -- the same call made for the flash attention layout.
#
# Run:  Rscript inst/scripts/proto_ag_resident_forward.R
# Env:  GGMLR_PROTO_REPS   timed passes per row (default 20)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_PROTO_REPS", "20"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns        <- asNamespace("ggmlR")
run_op    <- get(".ag_run_op",        envir = ns)
h_to_r    <- get(".ag_handle_to_r",   envir = ns)
fwd_prof  <- get("ag_forward_profile",       envir = ns)
fwd_reset <- get("ag_forward_profile_reset", envir = ns)
fwd_env   <- get(".ag_fwd",           envir = ns)

ag_device("gpu")
on.exit({ fwd_prof(FALSE); ag_device("cpu") }, add = TRUE)

# The tape budget is unlimited by default, which on the biggest point means the
# refusal would come from the Vulkan driver rather than from R -- and a driver
# refusal inside ggml is an abort, not a condition tryCatch can catch. Cap it
# below the card so an oversized point fails as an R error and the points after
# it still run. 12 GB of 16.
get(".ag_tape_mem_limit", envir = ns)(12 * 1024^3)

RP <- reps          # passes for the point being measured; set per model below

reset <- get(".ag_residency_reset", envir = ns)

# Every timed pass starts from a fresh tape.
#
# This is not tidiness, it is the difference between a measurement and an
# artefact. A resident op leaves its result in the shared residency context for
# good, so without a reset the tape grows across all four rows and every repeat:
# an earlier run of this script accumulated 12 GB by its fourth point and drove
# the `ctx` stage from 0.3 ms to 8 ms, which then dominated the small models and
# contaminated the `today` row that shares the same context.
#
# The reset is inside the timed region and identical for all four rows, so it
# neither favours nor penalises any of them -- and it is what a real training
# loop does anyway, once per step via with_grad_tape().
# For rows that touch no ggml context (the CPU baseline).
tm_plain <- function(f, warm = 2L) {
  for (i in seq_len(warm)) f()
  t0 <- Sys.time()
  for (i in seq_len(RP)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / RP
}

tm <- function(f, warm = 2L) {
  for (i in seq_len(warm)) { reset(); f() }
  reset()
  t0 <- Sys.time()
  for (i in seq_len(RP)) { f(); reset() }
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / RP
}

# --- the three forward paths -------------------------------------------------
#
# All three build the SAME chain: h <- relu(W_i %*% h), i = 1..depth. They differ only
# in what crosses the bus. Written against .ag_run_op itself -- the real one,
# with its generation check, memory budget, .ag_ctx_ensure and flush -- so this
# measures what component 4 would actually cost, not an idealised ceiling.

# ggml_mul_mat(W, h) contracts ne[0] of both operands, so the weight goes in
# transposed. Same convention .ag_gpu_matmul uses.
mm <- function(a, b) function(ctx, p) ggml_mul_mat(ctx, p[[1L]], p[[2L]])
rl <- function() function(ctx, p) ggml_relu(ctx, p[[1L]])

fwd_today <- function(Wt, X, d, b) {
  h <- X
  for (W in Wt) {
    h <- run_op(mm(), list(W, h), c(d, b))
    h <- run_op(rl(), list(h),    c(d, b))
  }
  h
}

fwd_resident_act <- function(Wt, X, d, b) {
  h <- X                                   # first op uploads it, as today
  for (W in Wt) {
    # W stays an R matrix: uploaded on every op, exactly as today.
    h <- run_op(mm(), list(W, h), c(d, b), resident = TRUE)
    h <- run_op(rl(), list(h),    c(d, b), resident = TRUE)
  }
  h_to_r(h)                                # one download for the whole chain
}

fwd_cpu <- function(Wt, X, d, b) {
  h <- X
  for (W in Wt) h <- pmax(W %*% h, 0)
  h
}

# --- per-stage profile of one row -------------------------------------------
#
# Summed over every .ag_run_op in one pass. Component 3's leaf_fetch taught the
# lesson: a total that improves can hide cost moving between stages rather than
# disappearing, so each row is broken out, not just its total.
stages_of <- function(f, warm = 2L) {
  for (i in seq_len(warm)) { reset(); f() }
  reset()
  fwd_reset(); fwd_prof(TRUE)
  for (i in seq_len(RP)) { f(); reset() }
  fwd_prof(FALSE)
  tot <- fwd_env$totals
  list(stages = if (is.null(tot)) numeric(0) else tot / RP,
       n_ops  = fwd_env$n / RP)
}

# --- the models ---------------------------------------------------------------
#
# The three already profiled, so the forward share used for Amdahl below is a
# measured number from measure_ag_forward_profile.R rather than a fresh guess.
models <- list(
  # Measured in measure_ag_forward_profile.R -- these three carry a step time
  # and a forward share, so Amdahl is a measurement for them.
  list(tag = "d=256  b=32   depth=4",  d = 256L,  b = 32L,   depth = 4L,
       step_ms = 9.95,  fwd_share = 0.401),
  list(tag = "d=512  b=64   depth=4",  d = 512L,  b = 64L,   depth = 4L,
       step_ms = 21.89, fwd_share = 0.492),
  list(tag = "d=1024 b=256  depth=4",  d = 1024L, b = 256L,  depth = 4L,
       step_ms = 91.61, fwd_share = 0.429),

  # Scaling probes: 4x on ONE axis at a time from the d=1024 b=256 depth=4
  # base, so a reversal can be attributed to the axis that caused it. No step
  # time was measured for these, so they print forward numbers only -- an
  # Amdahl column built on a borrowed share would be a prediction wearing a
  # measurement's clothes, which is the error this whole exercise avoids.
  list(tag = "d=4096 b=256  depth=4",  d = 4096L, b = 256L,  depth = 4L),
  list(tag = "d=1024 b=1024 depth=4",  d = 1024L, b = 1024L, depth = 4L),
  list(tag = "d=1024 b=256  depth=16", d = 1024L, b = 256L,  depth = 16L),

  # All three axes at once. Weights alone are 4.3 GB here and the resident
  # chain adds ~1 GB more, against 16 GB of VRAM, with R holding the same
  # weights as matrices for the `today` row. Expected to fail; wrapped so that
  # it cannot take the measured points down with it.
  list(tag = "d=4096 b=1024 depth=16", d = 4096L, b = 1024L, depth = 16L)
)

# Big points do not need 20 passes to separate rows that differ by tens of
# percent, and at d=4096 twenty passes of four rows plus three profiling runs
# is minutes of wall clock.
reps_for <- function(m) {
  w <- as.double(m$d) * m$d * m$depth + as.double(m$d) * m$b
  if (w > 2e8) 3L else if (w > 3e7) 6L else reps
}

cat(sprintf("reps = %d for small points, fewer for large ones;\n", reps))
cat("chain = relu(W_i %*% h), one weight per layer\n\n")
cat("Rows differ in ONE thing at a time:\n")
cat("  today         every operand up, every result down (current .ag_run_op)\n")
cat("  resident_act  activations stay on device; weights uploaded per op\n")
cat("  cpu           R %*% + pmax, threaded BLAS\n\n")

res <- list()

for (m in models) {
  RP <- reps_for(m)
  cat(sprintf("  measuring %-24s (%d passes) ... ", m$tag, RP))
  flush.console()
  ok <- tryCatch({
  set.seed(1L)
  d <- m$d; b <- m$b
  Wt <- lapply(seq_len(m$depth),
               function(i) matrix(rnorm(d * d) * 0.05, d, d))
  X  <- matrix(rnorm(d * b), d, b)

  # Correctness before timing: if the resident path disagrees, the timings are
  # meaningless and should not be printed as if they were comparable.
  reset()
  ref  <- fwd_today(Wt, X, d, b)
  reset()
  got  <- fwd_resident_act(Wt, X, d, b)
  md   <- max(abs(ref - got))
  mdc  <- max(abs(ref - fwd_cpu(Wt, X, d, b)))

  t_today <- tm(function() fwd_today(Wt, X, d, b))
  t_ract  <- tm(function() fwd_resident_act(Wt, X, d, b))
  # The CPU row uses a timer WITHOUT the reset: it touches no ggml context, so
  # charging it a tape reset would slow the opponent down for nothing and
  # flatter the GPU rows in the `vs cpu` column.
  t_cpu   <- tm_plain(function() fwd_cpu(Wt, X, d, b))

  s_today <- stages_of(function() fwd_today(Wt, X, d, b))
  s_ract  <- stages_of(function() fwd_resident_act(Wt, X, d, b))

  # The final download, which no stage records on the resident path. Timed as
  # the difference between the full chain and the chain without its last
  # h_to_r(), so it is measured the same way the rest of the row is.
  chain_ract_nodl <- function() {
    h <- X
    for (W in Wt) {
      h <- run_op(mm(), list(W, h), c(d, b), resident = TRUE)
      h <- run_op(rl(), list(h),    c(d, b), resident = TRUE)
    }
    invisible(h)
  }
  dl_ract <- max(0, t_ract - tm(chain_ract_nodl))

  res[[m$tag]] <- list(m = m, reps = RP, t_today = t_today, t_ract = t_ract,
                       t_cpu = t_cpu, md = md, mdc = mdc,
                       s_today = s_today, s_ract = s_ract,
                       dl_ract = dl_ract)
  TRUE
  }, error = function(e) { cat("FAILED: ", conditionMessage(e), "\n", sep = ""); FALSE })
  if (isTRUE(ok)) cat("ok\n")
  # A point that ran out of memory leaves tensors in the shared residency
  # context; clearing it keeps the failure from poisoning the points after it.
  try(get(".ag_residency_reset", envir = ns)(), silent = TRUE)
  invisible(gc(FALSE))
}
cat("\n")

# --- 1. correctness ----------------------------------------------------------
cat("1. Correctness (maxdiff against the current path)\n")
cat("   model                  resident_act      cpu\n")
for (r in res)
  cat(sprintf("   %-22s %12.3g %8.3g\n", r$m$tag, r$md, r$mdc))
cat("\n")

# --- 2. the forward itself ---------------------------------------------------
cat("2. Forward chain, milliseconds per pass\n")
cat("   model                    today  res_act      cpu     act x   vs cpu\n")
for (r in res)
  cat(sprintf("   %-22s %7.2f %8.2f %8.2f %8.2fx %8.2fx\n",
              r$m$tag, r$t_today, r$t_ract, r$t_cpu,
              r$t_today / r$t_ract, r$t_cpu / r$t_ract))
cat("\n   act x  = what residency alone buys on the forward chain.\n")
cat("   vs cpu = resident GPU against threaded BLAS; below 1.0 the CPU wins.\n\n")

# --- 3. where the time went --------------------------------------------------
cat("3. Per-stage, milliseconds per pass (summed over the chain's ops)\n")
keys <- c("ctx", "create", "flush", "upload", "graph", "compute", "download")
for (r in res) {
  cat(sprintf("\n   %s  (%s ops per pass, %d passes)\n", r$m$tag,
              format(r$s_today$n_ops), r$reps))
  cat("     stage        today   res_act\n")
  g <- function(s, k) { v <- s$stages[k]; if (is.na(v) || is.null(v)) 0 else as.numeric(v) }
  for (k in keys)
    cat(sprintf("     %-10s %8.2f %9.2f\n", k,
                g(r$s_today, k), g(r$s_ract, k)))
  # The resident path returns BEFORE .ag_run_op records a download stage, so
  # the one download at the end of the chain appears in no stage at all. Timed
  # separately and printed here: an unmeasured cost is how the last three
  # ceilings came in high.
  cat(sprintf("     %-10s %8.2f %9.2f\n", "final_dl", 0, r$dl_ract))
  tot <- function(s) sum(s$stages)
  cat(sprintf("     %-10s %8.2f %9.2f\n", "TOTAL",
              tot(r$s_today), tot(r$s_ract) + r$dl_ract))
}
cat("\n   A stage that shrinks while another grows is cost moving, not cost\n")
cat("   removed -- the leaf_fetch lesson from component 3.\n\n")

# --- 4. Amdahl: what this is worth on a whole step ---------------------------
#
# The decision column. A forward speedup s on a step whose forward share is p
# leaves (1 - p) + p/s of the step, so the step speedup is 1 / ((1-p) + p/s).
# step_ms and fwd_share are the MEASURED values from
# measure_ag_forward_profile.R, not re-timed here.
cat("4. On a full training step (Amdahl, using the measured forward share)\n")
cat("   Only the three points with a MEASURED step time and forward share\n")
cat("   appear here. The scaling probes have neither, and borrowing a share\n")
cat("   from a neighbouring point would turn this column into a prediction.\n\n")
cat("   model                  fwd share   step ms   act step x   saves ms\n")
for (r in res) {
  p <- r$m$fwd_share
  if (is.null(p)) next
  amd <- function(s) 1 / ((1 - p) + p / s)
  sa <- amd(r$t_today / r$t_ract)
  cat(sprintf("   %-22s %8.1f%% %9.2f %11.2fx %10.2f\n",
              r$m$tag, 100 * p, r$m$step_ms, sa,
              r$m$step_ms - r$m$step_ms / sa))
}
np <- Filter(function(r) is.null(r$m$fwd_share), res)
if (length(np))
  cat(sprintf("\n   (%d scaling probe%s above have forward numbers only)\n",
              length(np), if (length(np) == 1L) "" else "s"))

cat("\nReading the result:\n")
cat("  act step x near 1.0      -> residency alone cannot pay for 21 helpers.\n")
cat("                              Component 4 is closed, as a measurement.\n")
cat("  act step x >= ~1.2       -> residency earns it; component 4 is worth\n")
cat("                              costing out against section 4 of the profile\n")
cat("                              (the unified graph, 1.91-3.50x on forward).\n")
cat("  res_act slower than today-> the handle bookkeeping costs more than the\n")
cat("                              transfer it removes at this size.\n")
