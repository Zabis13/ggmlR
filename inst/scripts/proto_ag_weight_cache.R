#!/usr/bin/env Rscript
#
# PROTOTYPE. What does a weight cache buy on top of the unified graph?
#
# Why this now. proto_ag_forward_graph.R settled that the unified graph's own
# overhead is nil -- `emit` 0.04-0.08 ms, `create` 0.07-0.19, `graph` 0.03-0.17,
# `ctx` 0.03-0.05 -- and that `upload` is 74-98% of what a graph forward costs.
# The graph removes almost none of it, because both paths send the same weights;
# it only sends them once per pass instead of once per operation. So the number
# that decides whether rebuilding with_grad_tape is worth doing is not the
# graph's -- it is the weight cache's, and that has never been measured.
#
# The experiment. Weights change once per optimizer step, not once per forward,
# so a cache pays BETWEEN steps. Three rows, one variable at a time:
#
#   graph_cold   the unified graph as measured before: weights created and
#                uploaded on every pass. The baseline.
#   graph_warm   weights created and uploaded ONCE, then N passes reuse the same
#                device tensors; only the input and the chain's nodes are new
#                each pass. Time is per pass, so the one upload is amortised the
#                way a real training loop amortises it.
#   graph_warm1  the same, but the weights are re-uploaded once per pass into
#                the SAME tensors (no new allocation). This is the honest cost
#                of a cache that has to invalidate every step -- which is what
#                an optimizer that writes new weights would force.
#
# The third row is the one that keeps this honest. A cache that never
# invalidates measures a model with frozen weights, i.e. inference. Training
# changes the weights every step, so the question is whether re-uploading into
# an existing tensor is cheaper than the create+alloc+upload cycle -- and
# graph_warm1 is what answers it. graph_warm is the ceiling, graph_warm1 the
# realistic figure, graph_cold today.
#
# ⚠️ The reset rule (feedback_reset_tape_between_reps) is what makes the warm
# rows delicate: .ag_residency_reset() frees the buffers the cached weights live
# in, so it CANNOT run between passes of a warm row -- only around the row as a
# whole. That is the real constraint a weight cache imposes too, and the reason
# tape memory has to be watched here: the warm rows accumulate the chain's nodes
# for every pass in one context. Printed below as `ctx MB`, so growth is visible
# rather than inferred.
#
# Run:  Rscript inst/scripts/proto_ag_weight_cache.R
# Env:  GGMLR_PROTO_REPS   passes per row (default 20)

suppressMessages(library(ggmlR))

reps <- as.integer(Sys.getenv("GGMLR_PROTO_REPS", "20"))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to measure.\n"); quit(status = 0L)
}

ns        <- asNamespace("ggmlR")
ctx_ens   <- get(".ag_ctx_ensure",      envir = ns)
ctx_flush <- get(".ag_ctx_flush",       envir = ns)
reset     <- get(".ag_residency_reset", envir = ns)
gctx_b    <- get(".ag_graph_ctx_bytes", envir = ns)
dt_ggml   <- get(".ag_dtype_to_ggml",   envir = ns)
cdtype    <- get(".ag_compute_dtype",   envir = ns)
state     <- get(".ag_device_state",    envir = ns)
tape_mem  <- get(".ag_tape_mem",        envir = ns)

ag_device("gpu")
on.exit(ag_device("cpu"), add = TRUE)
get(".ag_tape_mem_limit", envir = ns)(12 * 1024^3)

RP <- reps

# --- the chain, built three ways ---------------------------------------------

# Create the weight tensors and upload them. Returns the device tensors, which
# stay valid until the next .ag_residency_reset().
make_weights <- function(Wt, d) {
  gt  <- dt_ggml(cdtype())
  ctx <- ctx_ens(length(Wt) + 8L)
  tW  <- lapply(Wt, function(m) ggml_new_tensor_2d(ctx, gt, nrow(m), ncol(m)))
  ctx_flush(ctx)
  for (i in seq_along(tW)) ggml_backend_tensor_set_data(tW[[i]], as.numeric(Wt[[i]]))
  tW
}

# One forward pass over weight tensors that already exist on the device.
# `reupload` re-sends the weight data into those same tensors first, which is
# what an optimizer step would force a cache to do.
pass_warm <- function(tW, X, d, b, Wt = NULL, reupload = FALSE) {
  gt  <- dt_ggml(cdtype())
  ctx <- ctx_ens(2L * length(tW) + 8L)

  if (reupload)
    for (i in seq_along(tW)) ggml_backend_tensor_set_data(tW[[i]], as.numeric(Wt[[i]]))

  tX <- ggml_new_tensor_2d(ctx, gt, nrow(X), ncol(X))
  node <- tX
  for (w in tW) node <- ggml_relu(ctx, ggml_mul_mat(ctx, w, node))

  # A no-op for the weights, which already have memory; it allocates the input
  # and this pass's nodes.
  ctx_flush(ctx)
  ggml_backend_tensor_set_data(tX, as.numeric(X))

  ctx_g <- ggml_init(gctx_b(), no_alloc = TRUE)
  on.exit(ggml_free(ctx_g), add = TRUE)
  graph <- ggml_build_forward_expand(ctx_g, node)
  ggml_backend_graph_compute(state$backend, graph)
  matrix(ggml_backend_tensor_get_data(node), d, b)
}

# The cold path: everything created and uploaded inside the pass.
pass_cold <- function(Wt, X, d, b) {
  gt  <- dt_ggml(cdtype())
  ctx <- ctx_ens(3L * length(Wt) + 16L)
  uploads <- list()
  const <- function(m) {
    tt <- ggml_new_tensor_2d(ctx, gt, nrow(m), ncol(m))
    uploads[[length(uploads) + 1L]] <<- list(ptr = tt, val = m)
    tt
  }
  tX <- const(X)
  tW <- lapply(Wt, const)
  node <- tX
  for (w in tW) node <- ggml_relu(ctx, ggml_mul_mat(ctx, w, node))
  ctx_flush(ctx)
  for (u in uploads) ggml_backend_tensor_set_data(u$ptr, as.numeric(u$val))
  ctx_g <- ggml_init(gctx_b(), no_alloc = TRUE)
  on.exit(ggml_free(ctx_g), add = TRUE)
  graph <- ggml_build_forward_expand(ctx_g, node)
  ggml_backend_graph_compute(state$backend, graph)
  matrix(ggml_backend_tensor_get_data(node), d, b)
}

fwd_cpu <- function(Wt, X, d, b) {
  h <- X
  for (W in Wt) h <- pmax(W %*% h, 0)
  h
}

# --- timing -------------------------------------------------------------------
#
# The cold row resets between passes, as every measurement in this series does.
# The warm rows CANNOT: the reset would free the weights they exist to keep. So
# they reset once before the row and once after, and the tape growth that
# results is reported rather than hidden.

tm_cold <- function(f, warm = 2L) {
  for (i in seq_len(warm)) { reset(); f() }
  reset()
  t0 <- Sys.time()
  for (i in seq_len(RP)) { f(); reset() }
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / RP
}

# Returns time per pass and the tape size the row ended with.
tm_warm <- function(setup, f, warm = 2L) {
  reset()
  h <- setup()
  for (i in seq_len(warm)) f(h)
  t0 <- Sys.time()
  for (i in seq_len(RP)) f(h)
  ms <- as.numeric(Sys.time() - t0, units = "secs") * 1000 / RP
  mb <- tape_mem()$buffer_bytes / 1024^2
  reset()
  list(ms = ms, mb = mb)
}

tm_plain <- function(f, warm = 2L) {
  for (i in seq_len(warm)) f()
  t0 <- Sys.time()
  for (i in seq_len(RP)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / RP
}

models <- list(
  list(tag = "d=256  b=32   depth=4",  d = 256L,  b = 32L,   depth = 4L,
       step_ms = 9.95,  fwd_share = 0.401),
  list(tag = "d=512  b=64   depth=4",  d = 512L,  b = 64L,   depth = 4L,
       step_ms = 21.89, fwd_share = 0.492),
  list(tag = "d=1024 b=256  depth=4",  d = 1024L, b = 256L,  depth = 4L,
       step_ms = 91.61, fwd_share = 0.429),
  list(tag = "d=4096 b=256  depth=4",  d = 4096L, b = 256L,  depth = 4L),
  list(tag = "d=1024 b=1024 depth=4",  d = 1024L, b = 1024L, depth = 4L),
  list(tag = "d=1024 b=256  depth=16", d = 1024L, b = 256L,  depth = 16L),
  list(tag = "d=4096 b=1024 depth=16", d = 4096L, b = 1024L, depth = 16L)
)

reps_for <- function(m) {
  w <- as.double(m$d) * m$d * m$depth + as.double(m$d) * m$b
  if (w > 2e8) 3L else if (w > 3e7) 6L else reps
}

cat(sprintf("reps = %d for small points, fewer for large ones;\n", reps))
cat("chain = relu(W_i %*% h), one weight per layer, always ONE graph\n\n")
cat("  graph_cold   weights created + uploaded every pass (today)\n")
cat("  graph_warm   weights uploaded ONCE, reused (the ceiling: frozen weights)\n")
cat("  graph_warm1  weights re-uploaded into the SAME tensors each pass\n")
cat("               (the realistic cache: the optimizer changes them every step)\n\n")

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

    reset(); ref <- pass_cold(Wt, X, d, b)
    reset()
    tW0 <- make_weights(Wt, d)
    got <- pass_warm(tW0, X, d, b)
    md  <- max(abs(ref - got))
    reset()

    t_cold <- tm_cold(function() pass_cold(Wt, X, d, b))
    w_warm <- tm_warm(function() make_weights(Wt, d),
                      function(h) pass_warm(h, X, d, b))
    w_warm1 <- tm_warm(function() make_weights(Wt, d),
                       function(h) pass_warm(h, X, d, b, Wt = Wt, reupload = TRUE))
    t_cpu  <- tm_plain(function() fwd_cpu(Wt, X, d, b))

    res[[m$tag]] <- list(m = m, reps = RP, md = md,
                         t_cold = t_cold, t_warm = w_warm$ms, mb_warm = w_warm$mb,
                         t_warm1 = w_warm1$ms, mb_warm1 = w_warm1$mb,
                         t_cpu = t_cpu)
    TRUE
  }, error = function(e) {
    cat("FAILED: ", conditionMessage(e), "\n", sep = ""); FALSE })
  if (isTRUE(ok)) cat("ok\n")
  try(reset(), silent = TRUE)
  invisible(gc(FALSE))
}
cat("\n")

cat("1. Correctness (maxdiff of the warm path against the cold one)\n")
for (r in res) cat(sprintf("   %-22s %12.3g\n", r$m$tag, r$md))
cat("\n")

cat("2. Milliseconds per pass\n")
cat("   model                    cold     warm    warm1      cpu   warm x  warm1 x\n")
for (r in res)
  cat(sprintf("   %-22s %8.2f %8.2f %8.2f %8.2f %7.2fx %7.2fx\n",
              r$m$tag, r$t_cold, r$t_warm, r$t_warm1, r$t_cpu,
              r$t_cold / r$t_warm, r$t_cold / r$t_warm1))
cat("\n   warm x  = ceiling, weights never re-sent (inference-shaped).\n")
cat("   warm1 x = realistic, weights re-sent into existing tensors each step.\n\n")

cat("3. Tape growth in the warm rows (buffers held at the end of the row)\n")
cat("   The warm rows cannot reset between passes -- that is what keeps the\n")
cat("   weights alive -- so every pass's nodes accumulate. A real cache has to\n")
cat("   deal with this; the numbers say how urgently.\n")
cat("   model                  warm MB  warm1 MB   passes\n")
for (r in res)
  cat(sprintf("   %-22s %8.1f %9.1f %8d\n",
              r$m$tag, r$mb_warm, r$mb_warm1, r$reps))
cat("\n")

cat("4. On a full training step (Amdahl, measured forward share only)\n")
cat("   model                  fwd share   step ms   warm1 step x   saves ms\n")
for (r in res) {
  p <- r$m$fwd_share
  if (is.null(p)) next
  s  <- r$t_cold / r$t_warm1
  sa <- 1 / ((1 - p) + p / s)
  cat(sprintf("   %-22s %8.1f%% %9.2f %12.2fx %10.2f\n",
              r$m$tag, 100 * p, r$m$step_ms, sa,
              r$m$step_ms - r$m$step_ms / sa))
}
np <- Filter(function(r) is.null(r$m$fwd_share), res)
if (length(np))
  cat(sprintf("\n   (%d scaling probes above have forward numbers only)\n",
              length(np)))

cat("\nReading the result:\n")
cat("  warm1 ~ cold        -> re-uploading costs what creating did; the cache\n")
cat("                         buys nothing for TRAINING, and the 74-98% upload\n")
cat("                         share is simply the price of moving weights.\n")
cat("                         Level 3 (weights that never leave) becomes the\n")
cat("                         only way to touch it.\n")
cat("  warm1 ~ warm        -> re-upload is cheap; a cache pays and should be\n")
cat("                         built before with_grad_tape is rebuilt.\n")
cat("  warm >> warm1       -> the win needs weights that do not change, i.e.\n")
cat("                         inference. Note it and stop calling it a training\n")
cat("                         optimisation.\n")
cat("  tape MB growing fast-> a cache cannot simply hold the context open;\n")
cat("                         it needs the chain's nodes freed per pass, which\n")
cat("                         is a design constraint on level 2.\n")
