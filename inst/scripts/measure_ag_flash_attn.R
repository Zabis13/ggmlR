#!/usr/bin/env Rscript
#
# Would flash attention actually be faster than the ag_* attention we have?
#
# ag_multihead_attention builds a python-style loop over heads: each head slices
# Q/K/V with selector matrices (so a slice costs a matmul), transposes, softmaxes
# and matmuls again. On a 4-head block that is ~57 tape nodes, and every one of
# them is an R-level dispatch plus, on the GPU, its own round trip.
#
# ggml_flash_attn_ext does the whole thing -- all heads, all batch -- in ONE op,
# and the ggmlR extension ggml_flash_attn_back does its gradient in one more.
# The obvious conclusion is that replacing the loop must be faster.
#
# That conclusion is exactly the kind this project has been wrong about before:
# per-op GPU backward looked obviously faster and was a 1.8x regression; a graph
# backward measured 2.1x on synthetic shapes and lost on every real model. So
# before building the wrapper (which needs a 3D execution path next to
# .ag_run_op, and unpacking of the three packed gradients), measure the ceiling.
#
# Three timings on identical shapes:
#
#   ag_attention     what exists: forward + backward through the tape
#   flash_ceiling    one ggml_flash_attn_ext + one ggml_flash_attn_back, graph
#                    built and computed per call, inputs uploaded each time --
#                    i.e. what a wrapper could achieve WITHOUT residency
#   flash_compute    the same two ops with the graph built and the inputs
#                    uploaded once, timing only the computes -- the floor, which
#                    a wrapper cannot beat
#
# The gap between ag_attention and flash_ceiling is the honest prospective win.
# If it is small, the wrapper is not worth building; if flash_ceiling is far
# above flash_compute, the cost is transport rather than attention, and that is
# a different (already-explored) problem.
#
# Run:  Rscript inst/scripts/measure_ag_flash_attn.R
# Env:  GGMLR_AG_BENCH_REPS   iterations per timing (default 20)

suppressMessages(library(ggmlR))

reps    <- as.integer(Sys.getenv("GGMLR_AG_BENCH_REPS", "20"))
batches <- 5L

timed <- function(f) {
  f()
  b <- vapply(seq_len(batches), function(...) {
    t0 <- Sys.time()
    for (i in seq_len(reps)) f()
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
  }, numeric(1))
  stats::median(b)
}

# --- 1. the attention we have ------------------------------------------------
#
# ONLY THE PART FLASH WOULD REPLACE.
#
# The first draft of this script timed the whole ag_multihead_attention block
# against the flash core and reported 46x. That number was meaningless: the ag
# side also carried the four projections (W_q/W_k/W_v/W_o) and the loss, while
# the flash side carried none of them -- more work on one side of the ratio.
#
# What follows is the attention CORE alone, built from the same ag_* ops
# ag_multihead_attention uses per head: slice each head out (a matmul against a
# selector matrix, since ag_* has no slice op), scores, scale, softmax,
# weighted sum, concatenate. Q, K and V arrive already projected -- as they
# would in a wrapper that leaves the projections outside the flash call.
ag_attention_core <- function(d_model, n_heads, seq_len_) {
  dk <- d_model %/% n_heads
  set.seed(5L)
  Q <- ag_param(matrix(runif(d_model * seq_len_, -1, 1), d_model, seq_len_))
  K <- ag_param(matrix(runif(d_model * seq_len_, -1, 1), d_model, seq_len_))
  V <- ag_param(matrix(runif(d_model * seq_len_, -1, 1), d_model, seq_len_))
  scale <- 1 / sqrt(dk)

  core <- function() {
    heads <- vector("list", n_heads)
    for (h in seq_len(n_heads)) {
      lo <- (h - 1L) * dk + 1L; hi <- h * dk
      q_h <- ggmlR:::.ag_row_slice(Q, lo, hi)
      k_h <- ggmlR:::.ag_row_slice(K, lo, hi)
      v_h <- ggmlR:::.ag_row_slice(V, lo, hi)
      scores <- ag_scale(ag_matmul(ag_transpose(q_h), k_h), scale)
      attn   <- ag_transpose(ag_softmax(ag_transpose(scores)))
      heads[[h]] <- ag_matmul(v_h, ag_transpose(attn))
    }
    ggmlR:::.ag_row_concat(heads)
  }

  list(
    step = function() {
      out <- NULL
      with_grad_tape({
        out <- core()
        # A scalar to differentiate, as cheap as possible so the loss does not
        # show up in the timing: sum of the output via a fixed target.
        loss <- ag_mse_loss(out, ag_tensor(matrix(0, nrow(ggmlR:::.ag_data(out)),
                                                  ncol(ggmlR:::.ag_data(out)))))
      })
      backward(loss)
      invisible(NULL)
    },
    nodes = function() {
      with_grad_tape({
        out <- core()
        invisible(ag_mse_loss(out, ag_tensor(matrix(0, nrow(ggmlR:::.ag_data(out)),
                                                    ncol(ggmlR:::.ag_data(out))))))
      })
      length(ggmlR:::.ag_tape$nodes)
    })
}

# --- 2/3. the flash ceiling --------------------------------------------------
#
# Shapes follow ggml.h:
#   q [n_embd_k, n_batch, n_head, 1]   k,v [n_embd, n_kv, n_head_kv, 1]
#   grad-in d has the PERMUTED result layout [n_embd_v, n_head, n_batch, 1]
# One sequence per call (ne3 = 1), n_head_kv = n_head: the plain MHA case that
# ag_multihead_attention implements.
flash_ceiling <- function(d_model, n_heads, seq_len_, resident) {
  dk <- d_model %/% n_heads
  backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(backend, 2L)

  ctx <- ggml_init(512 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)

  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dk, seq_len_, n_heads, 1L)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dk, seq_len_, n_heads, 1L)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dk, seq_len_, n_heads, 1L)
  d <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dk, n_heads, seq_len_, 1L)

  scale <- 1 / sqrt(dk)
  fwd <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0, 0)
  bwd <- ggml_flash_attn_back(ctx, q, k, v, NULL, d, scale)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  set.seed(6L)
  qd <- runif(dk * seq_len_ * n_heads, -1, 1)
  kd <- runif(dk * seq_len_ * n_heads, -1, 1)
  vd <- runif(dk * seq_len_ * n_heads, -1, 1)
  dd <- runif(dk * seq_len_ * n_heads, -1, 1)

  upload <- function() {
    ggml_backend_tensor_set_data(q, qd)
    ggml_backend_tensor_set_data(k, kd)
    ggml_backend_tensor_set_data(v, vd)
    ggml_backend_tensor_set_data(d, dd)
  }
  if (resident) upload()

  gf <- ggml_build_forward_expand(ctx, fwd)
  gb <- ggml_build_forward_expand(ctx, bwd)

  run <- function() {
    if (!resident) upload()
    ggml_backend_graph_compute(backend, gf)
    ggml_backend_graph_compute(backend, gb)
    if (!resident) {
      # A wrapper has to bring the results back to R, so charge that too.
      invisible(ggml_backend_tensor_get_data(fwd))
      invisible(ggml_backend_tensor_get_data(bwd))
    }
    invisible(NULL)
  }

  list(run = run, free = function() {
    tryCatch(ggml_backend_buffer_free(buf), error = function(e) NULL)
    tryCatch(ggml_free(ctx), error = function(e) NULL)
    tryCatch(ggml_backend_free(backend), error = function(e) NULL)
  })
}

row <- function(d_model, n_heads, seq_len_) {
  core  <- ag_attention_core(d_model, n_heads, seq_len_)
  nodes <- core$nodes()
  t_ag  <- timed(core$step)

  fc <- flash_ceiling(d_model, n_heads, seq_len_, resident = FALSE)
  t_ceil <- timed(fc$run); fc$free()

  fr <- flash_ceiling(d_model, n_heads, seq_len_, resident = TRUE)
  t_flr <- timed(fr$run); fr$free()

  cat(sprintf("  d%-4d h%-2d seq%-4d %3d nodes | ag %8.2f ms | flash %6.2f ms (%5.1fx) | compute only %6.2f ms\n",
              d_model, n_heads, seq_len_, nodes, t_ag, t_ceil, t_ag / t_ceil, t_flr))
  invisible(NULL)
}

cat("ggmlR: ag attention vs a flash-attention ceiling (CPU backend)\n")
cat("  forward + backward, identical shapes; reps ", reps, " x ", batches, "\n\n", sep = "")

ag_device("cpu")

row(32L,  2L, 16L)
row(64L,  4L, 32L)
row(128L, 8L, 64L)
row(256L, 8L, 128L)

cat("\nRead it this way:\n")
cat("  * Both sides cover the SAME work: the attention core only. Q/K/V arrive\n")
cat("    projected and the output projection is excluded, because a wrapper\n")
cat("    would leave those outside the flash call. Timing the full ag block\n")
cat("    against the flash core inflates the ratio -- an earlier draft of this\n")
cat("    script did exactly that and reported 46x.\n")
cat("  * The ratio is the prospective win, and it is an UPPER bound: the flash\n")
cat("    side pays no R-level tape bookkeeping at all, whereas a real\n")
cat("    ag_flash_attention() would still record a node, hold snapshots and\n")
cat("    unpack three packed gradients. Treat it as 'is this worth building',\n")
cat("    not as the speedup a user would see.\n")
cat("  * `compute only` strips the transfers. If flash sits far above it, the\n")
cat("    remaining cost is moving data, not attention -- the same wall the\n")
cat("    graph-backward work already hit.\n")
cat("  * Node counts are what flash collapses: every one is an R dispatch and,\n")
cat("    on the GPU, its own round trip.\n")
