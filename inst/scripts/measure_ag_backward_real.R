#!/usr/bin/env Rscript
#
# Does the graph backward pay off on REAL networks, not just the PoC chain?
#
# inst/scripts/measure_ag_backward.R established the idea on a synthetic tape:
# 8 matmuls of 512x512, one op type, all square. It measured 40.2 ms of closure
# backward against 19.3 ms for one graph with the activations uploaded, and that
# is what justified building R/ag_backward_graph.R.
#
# That tape is not what real models look like, and the difference cuts in the
# expensive direction:
#
#   PoC chain              8 nodes,  all 512x512 matmuls
#   MLP 128->64->10, b=64  6 nodes,  small matrices, mixed ops
#   attention d=64 h=4     57 nodes, many tiny ops (head slicing is matmul by
#                          selector matrices, so heads multiply the node count)
#
# Per-node costs the PoC barely felt are charged 57 times here: building the
# node, reserving a descriptor, and -- for every rule that folds a forward
# snapshot -- one upload. A graph of 57 tiny nodes can lose to R BLAS even
# though a graph of 8 big ones wins by 2x. That is the question this script
# answers, and it is the reason not to declare the feature done on PoC numbers.
#
# Both paths compute the same gradients from the same tape; the only variable is
# which backward runs. Correctness is checked first (max-abs against the closure
# path) because a fast wrong backward is not a result.
#
# Run:  Rscript inst/scripts/measure_ag_backward_real.R
# Env:  GGMLR_AG_BENCH_REPS   iterations per timing (default 20)
#       GGMLR_AG_BENCH_PROF=1 print a per-stage breakdown under each row

suppressMessages(library(ggmlR))

reps    <- as.integer(Sys.getenv("GGMLR_AG_BENCH_REPS", "20"))
# GGMLR_AG_BENCH_PROF=1 adds a per-stage breakdown under every row.
profile_stages <- identical(Sys.getenv("GGMLR_AG_BENCH_PROF"), "1")
batches <- 5L

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: the graph path only runs on the GPU, nothing to measure.\n")
  quit(status = 0L)
}

ag_backward_graph <- ggmlR:::ag_backward_graph
ag_backward_path  <- ggmlR:::ag_backward_path
ag_tape           <- ggmlR:::.ag_tape
ag_bwd_profile        <- ggmlR:::ag_backward_profile
ag_bwd_profile_reset  <- ggmlR:::ag_backward_profile_reset
ag_bwd_profile_report <- ggmlR:::ag_backward_profile_report

ag_device("gpu")

timed <- function(f) {
  f()
  b <- vapply(seq_len(batches), function(...) {
    t0 <- Sys.time()
    for (i in seq_len(reps)) f()
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
  }, numeric(1))
  list(ms = stats::median(b), lo = min(b), hi = max(b))
}

# ---------------------------------------------------------------------------
# The models. Each returns a builder that rebuilds the tape from a fixed seed,
# so both paths see identical inputs and identical random dropout masks.
# ---------------------------------------------------------------------------

mlp <- function(n_in = 128L, n_hidden = 64L, n_out = 10L, batch = 64L) {
  function() {
    set.seed(1L)
    l1 <- ag_linear(n_in, n_hidden, activation = "relu")
    l2 <- ag_linear(n_hidden, n_out)
    x  <- ag_tensor(matrix(runif(n_in * batch, -1, 1), n_in, batch))
    y  <- rep(seq_len(n_out) - 1L, length.out = batch)
    loss <- NULL
    with_grad_tape({
      loss <- ag_softmax_cross_entropy_loss(l2$forward(l1$forward(x)), y)
    })
    list(loss = loss, params = list(l1$W, l1$b, l2$W, l2$b))
  }
}

attn <- function(d_model = 64L, heads = 4L, seq_len_ = 32L) {
  function() {
    set.seed(2L)
    at <- ag_multihead_attention(d_model, heads)
    x  <- ag_tensor(matrix(runif(d_model * seq_len_, -1, 1), d_model, seq_len_))
    y  <- ag_tensor(matrix(runif(d_model * seq_len_, -1, 1), d_model, seq_len_))
    loss <- NULL
    with_grad_tape({
      loss <- ag_mse_loss(at$forward(x), y)
    })
    list(loss = loss, params = at$parameters())
  }
}

# ---------------------------------------------------------------------------

measure <- function(build, label) {
  # Correctness first, and on the gradients themselves rather than the loss.
  ag_backward_graph(FALSE)
  ref <- build(); backward(ref$loss)
  ref_g <- lapply(ref$params, function(p) p$grad)
  n_nodes <- length(ag_tape$nodes)

  ag_backward_graph(TRUE)
  got <- build(); backward(got$loss)
  got_g <- lapply(got$params, function(p) p$grad)
  path  <- ag_backward_path()

  d <- max(mapply(function(a, b) {
    if (is.null(a) || is.null(b)) Inf else max(abs(a - b))
  }, ref_g, got_g))

  if (!identical(path, "graph")) {
    cat(sprintf("  %-26s %s\n", label,
                paste0("NOT ON THE GRAPH PATH -- ", path)))
    return(invisible(NULL))
  }

  # Time backward() alone. The tape is rebuilt outside the timed region, so the
  # forward pass is not charged to either path -- this compares backward with
  # backward, as the PoC did.
  bench <- function(flag) {
    ag_backward_graph(flag)
    v <- build()
    timed(function() backward(v$loss))
  }
  t_clo <- bench(FALSE)
  t_gph <- bench(TRUE)

  cat(sprintf("  %-26s %3d nodes | closures %6.2f ms | graph %6.2f ms | %4.2fx | maxdiff %.1e\n",
              label, n_nodes, t_clo$ms, t_gph$ms, t_clo$ms / t_gph$ms, d))

  # Stage breakdown for the graph path. Where the ratio is bad, this says which
  # stage owns it -- the first run of this script put 51-61% of the time in the
  # download loop, which is how the leaf-only download fix was found.
  if (profile_stages) {
    ag_bwd_profile(TRUE); ag_bwd_profile_reset()
    ag_backward_graph(TRUE)
    v <- build(); for (i in seq_len(reps)) backward(v$loss)
    ag_bwd_profile_report()
    ag_bwd_profile(FALSE)
    cat("\n")
  }
  invisible(list(label = label, nodes = n_nodes,
                 closures = t_clo$ms, graph = t_gph$ms))
}

cat("ggmlR graph backward on real models\n")
cat("  reps: ", reps, " x ", batches, " batches\n\n", sep = "")
cat("backward() only; the tape is built outside the timed region:\n")

res <- list()
res$mlp_small  <- measure(mlp(64L,   32L,  10L, 32L),  "MLP 64-32-10, b32")
res$mlp        <- measure(mlp(128L,  64L,  10L, 64L),  "MLP 128-64-10, b64")
res$mlp_wide   <- measure(mlp(512L,  256L, 10L, 128L), "MLP 512-256-10, b128")
res$mlp_big    <- measure(mlp(1024L, 512L, 10L, 256L), "MLP 1024-512-10, b256")
res$mlp_huge   <- measure(mlp(2048L, 1024L, 10L, 512L),"MLP 2048-1024-10, b512")
res$attn_small <- measure(attn(32L,  2L, 16L),  "attention d32 h2 seq16")
res$attn       <- measure(attn(64L,  4L, 32L),  "attention d64 h4 seq32")
res$attn_big   <- measure(attn(128L, 8L, 64L),  "attention d128 h8 seq64")
res$attn_huge  <- measure(attn(256L, 8L, 128L), "attention d256 h8 seq128")

cat("\nRead it this way:\n")
cat("  * The PoC measured 2.1x on 8 square 512x512 matmuls. Real tapes are\n")
cat("    smaller per node and (for attention) far more numerous, so a lower\n")
cat("    ratio is expected -- what matters is whether it stays ABOVE 1.\n")
cat("  * The first run of this script found EVERY row below 1 (0.18-0.42x).\n")
cat("    Stage profiling put 51-61% of the time in the download loop: the\n")
cat("    path was fetching a gradient for every tensor on the tape (62 for a\n")
cat("    4-head attention block) when the caller reads only the leaves (5).\n")
cat("    Downloading leaves only moved attention 0.22 -> 0.74x and the wide\n")
cat("    MLP 0.42 -> 1.08x. Run with GGMLR_AG_BENCH_PROF=1 to see the current\n")
cat("    breakdown rather than assuming this one still holds.\n")
cat("  * With that fixed the largest stage is emit -- walking the tape and\n")
cat("    building nodes in R. It is per-node and pure R, so it should scale\n")
cat("    with tape length, not tensor size: the big MLPs (6 nodes, large\n")
cat("    matrices) are where the graph path can win, attention (109+ nodes,\n")
cat("    small matrices) is where it struggles.\n")
cat("  * maxdiff is f32 noise unless a rule is wrong, in which case the\n")
cat("    timings above mean nothing.\n")
