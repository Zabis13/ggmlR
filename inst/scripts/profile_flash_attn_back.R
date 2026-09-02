#!/usr/bin/env Rscript
#
# Where does the flash-attention backward shader spend its time?
#
# The shader replaced a CPU fallback and came out THREE TIMES SLOWER than the
# CPU it replaced, and two confident guesses at the reason were both wrong:
#
#   - "barrier overhead in the dq reduction" -- rewriting dq to need no barriers
#     at all changed the time by 0.5% (79.3 -> 78.8 ms).
#   - "atomic contention on dk/dv" -- appeared to be refuted by a by-N sweep,
#     except that sweep varied the useful work along with the contention and so
#     answered neither question.
#
# So this stops guessing. Copies of the shader truncated after each phase are
# built alongside the real one; running them in turn gives the cost of each
# phase as a difference. GGMLR_FAB_PROFILE=1..4 selects one.
#
#   stage 1   logits + dP into shared memory, row max      (reads q, k, v, d)
#   stage 2   + exponentiate and sum                       (shared only)
#   stage 3   + normalise and fold dot(P, dP)              (shared only)
#   stage 4   + dS, and the dk/dv atomicAdds               (67M atomics at the
#                                                           default sizes)
#   full      + dq                                         (O(M*DK) per row)
#
# The truncated stages compute WRONG gradients on purpose -- they exist to be
# timed, never to be used. Run:
#
#   Rscript inst/scripts/profile_flash_attn_back.R

suppressMessages(library(ggmlR))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() == 0L) {
  cat("No Vulkan device.\n"); quit(save = "no")
}

reps    <- as.integer(Sys.getenv("GGMLR_FAB_PROF_REPS", "30"))
batches <- 3L

DK <- 64L; DV <- 64L; N <- 128L; M <- 128L; H <- 8L; Hkv <- 4L; B <- 4L

# One backward node alone: no forward, no loss, no autodiff bookkeeping.
time_backward <- function() {
  gpu   <- ggml_vulkan_init(0L)
  sched <- ggml_backend_sched_new(list(gpu), parallel = FALSE)
  ctx   <- ggml_init(2048 * 1024 * 1024, no_alloc = TRUE)
  on.exit({
    ggml_free(ctx); ggml_backend_sched_free(sched); ggml_vulkan_free(gpu)
  }, add = TRUE)

  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, H, B)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, Hkv, B)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, Hkv, B)
  d <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, H, N, B)
  for (t in list(q, k, v, d)) ggml_set_input(t)

  # A backend buffer is not owned by the context, so ggml_free() does not
  # release it: free it explicitly or every stage leaks one allocation. It must
  # go BEFORE the context, and on.exit(add = TRUE) runs handlers in the order
  # they were added, so this one is prepended instead.
  buf <- ggml_backend_alloc_ctx_tensors(ctx, gpu)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)

  ggml_backend_tensor_set_data(q, runif(DK * N * H * B, -1, 1))
  ggml_backend_tensor_set_data(k, runif(DK * M * Hkv * B, -1, 1))
  ggml_backend_tensor_set_data(v, runif(DV * M * Hkv * B, -1, 1))
  ggml_backend_tensor_set_data(d, runif(DV * H * N * B, -1, 1))

  bk    <- ggml_flash_attn_back(ctx, q, k, v, NULL, d, 1 / sqrt(DK))
  graph <- ggml_build_forward_expand(ctx, ggml_sum(ctx, bk))
  ggml_backend_sched_alloc_graph(sched, graph)

  step <- function() ggml_backend_sched_graph_compute(sched, graph)

  for (i in seq_len(3L)) step()
  ggml_backend_sched_synchronize(sched)

  b <- vapply(seq_len(batches), function(...) {
    t0 <- Sys.time()
    for (i in seq_len(reps)) step()
    ggml_backend_sched_synchronize(sched)
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
  }, numeric(1))

  stats::median(b)
}

cat(sprintf("Backward node alone: DK=%d DV=%d N=%d M=%d H=%d Hkv=%d B=%d\n",
            DK, DV, N, M, H, Hkv, B))

rows      <- N * H * B
atomics   <- rows * M * (DK + DV)
dq_work   <- rows * M * DK
pass1_rd  <- rows * M * (2 * DK + DV)
cat(sprintf("%d query rows | %.1fM atomicAdd | %.1fM dq mul-adds | %.1fM pass-1 reads\n\n",
            rows, atomics / 1e6, dq_work / 1e6, pass1_rd / 1e6))

stages <- c("1: logits + dP + row max",
            "2: + exp and sum",
            "3: + normalise, dot(P,dP)",
            "4: + dS, dk/dv atomics",
            "full: + dq")

times <- numeric(length(stages))
for (s in seq_along(stages)) {
  if (s < length(stages)) {
    Sys.setenv(GGMLR_FAB_PROFILE = as.character(s))
  } else {
    Sys.unsetenv("GGMLR_FAB_PROFILE")
  }
  times[s] <- time_backward()
}
Sys.unsetenv("GGMLR_FAB_PROFILE")

cat(sprintf("%-28s %10s %12s\n", "stage", "ms", "phase cost"))
cat(strrep("-", 54), "\n")
for (s in seq_along(stages)) {
  delta <- if (s == 1) times[s] else times[s] - times[s - 1]
  cat(sprintf("%-28s %10.2f %12s\n", stages[s], times[s],
              sprintf("%+.2f", delta)))
}

cat("\n")
deltas <- c(times[1], diff(times))
worst  <- which.max(deltas)
cat(sprintf("Dominant phase: %s -- %.2f ms of %.2f (%.0f%%).\n",
            stages[worst], deltas[worst], times[length(times)],
            100 * deltas[worst] / times[length(times)]))

if (worst == 4) {
  cat("That is the atomics. Private accumulators merged once per workgroup, or a\n")
  cat("split-K pass, would remove the contention.\n")
} else if (worst == 1) {
  cat("That is pass 1's global-memory traffic: every invocation re-reads the whole\n")
  cat("q row per kv position. Caching q in shared memory would cut it.\n")
} else if (worst == length(stages)) {
  cat("That is dq. It is O(M*DK) per row and reads k again -- caching or a\n")
  cat("different partition would help.\n")
}
