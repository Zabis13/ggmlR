#!/usr/bin/env Rscript
#
# Probe: is an upload cache in .ag_run_op feasible, and does it pay?
#
# Three questions, any of which can close the item before code is written:
#
#   Q1 separability -- .ag_run_op does new_tensor -> ctx_flush -> set_data.
#      A cache skips only set_data. If the allocation, not the transfer, is
#      what costs, the ceiling is far below the 35-40% the TODO assumes.
#   Q2 survival    -- a cached tensor must stay valid across the NEXT call's
#      .ag_ctx_flush. If flush disturbs earlier residents, there is nothing to
#      cache.
#   Q3 identity    -- .ag_run_op receives R matrices, not ag_tensors, so it
#      cannot tell "same weight again" from "new matrix". Worse, ag_matmul
#      calls t(a_data) first, which allocates a fresh object every call. Can
#      the caller side supply identity at all?

suppressMessages(library(ggmlR))

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("No Vulkan device: nothing to probe.\n"); quit(status = 0L)
}

ns   <- asNamespace("ggmlR")
g    <- function(nm) get(nm, envir = ns)
st   <- g(".ag_device_state")

reps <- 20L
tm <- function(f) {
  for (i in 1:4) f()
  t0 <- Sys.time(); for (i in seq_len(reps)) f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000 / reps
}

ag_device("gpu")
d <- 512L
W <- matrix(rnorm(d * d), d, d)
X <- matrix(rnorm(d * d), d, d)

cat("=== Q1: what does the per-call cost consist of? ===\n")
# Time the pieces of .ag_run_op separately: context+tensor creation, flush
# (buffer allocation), set_data (the upload a cache would skip), and compute.
ctx_ensure <- g(".ag_ctx_ensure")
ctx_flush  <- g(".ag_ctx_flush")
dtype2type <- g(".ag_dtype_to_ggml")
cdtype     <- g(".ag_compute_dtype")
ty <- dtype2type(cdtype(st$dtype))

t_create <- tm(function() {
  ctx <- ctx_ensure(2L)
  ggml_new_tensor_2d(ctx, ty, d, d)
})
t_create_flush <- tm(function() {
  ctx <- ctx_ensure(2L)
  p <- ggml_new_tensor_2d(ctx, ty, d, d)
  ctx_flush(ctx)
})
t_full <- tm(function() {
  ctx <- ctx_ensure(2L)
  p <- ggml_new_tensor_2d(ctx, ty, d, d)
  ctx_flush(ctx)
  ggml_backend_tensor_set_data(p, as.numeric(W))
})
cat(sprintf("  create tensor            : %7.2f ms\n", t_create))
cat(sprintf("  + flush (alloc buffer)   : %7.2f ms  (delta %.2f)\n",
            t_create_flush, t_create_flush - t_create))
cat(sprintf("  + set_data (the upload)  : %7.2f ms  (delta %.2f)\n",
            t_full, t_full - t_create_flush))
cat(sprintf("  -> upload share of these : %5.1f%%\n",
            100 * (t_full - t_create_flush) / t_full))

# For reference: a whole matmul through the normal path.
A <- ag_tensor(W); B <- ag_tensor(X)
t_op <- tm(function() ag_matmul(A, B))
cat(sprintf("  whole ag_matmul          : %7.2f ms\n", t_op))
cat(sprintf("  -> upload share of an op : %5.1f%%  <-- the real ceiling\n",
            100 * (t_full - t_create_flush) / t_op))

cat("\n=== Q2: does a uploaded tensor survive later calls? ===\n")
# Upload once, run several unrelated ops (each of which calls ctx_flush), then
# read the tensor back and compare. If it still matches, caching is possible.
ctx <- ctx_ensure(2L)
p   <- ggml_new_tensor_2d(ctx, ty, d, d)
ctx_flush(ctx)
ggml_backend_tensor_set_data(p, as.numeric(W))
before <- ggml_backend_tensor_get_data(p)

gen_before <- st$ctx_gen
for (i in 1:5) invisible(ag_matmul(A, B))
after <- ggml_backend_tensor_get_data(p)
gen_after <- st$ctx_gen

cat(sprintf("  ctx_gen %s -> %s\n", gen_before, gen_after))
cat(sprintf("  data intact after 5 ops  : %s (maxdiff %.3g)\n",
            identical(all.equal(before, after), TRUE),
            max(abs(before - after))))

cat("\n=== Q3: can the caller supply identity? ===\n")
# .ag_run_op sees matrices. Do repeated calls even present the SAME object?
addr <- function(x) tryCatch(tracemem(x), error = function(e) NA_character_)
a1 <- .ag_data_probe <- g(".ag_data")(A)
a2 <- g(".ag_data")(A)
cat(sprintf("  .ag_data(A) twice, same object : %s\n", identical(addr(a1), addr(a2))))
t1 <- t(a1); t2 <- t(a2)
cat(sprintf("  t(.ag_data(A)) twice, same     : %s  <-- ag_matmul does this\n",
            identical(addr(t1), addr(t2))))
cat(sprintf("  A$id stable across calls       : %s (id=%s)\n",
            !is.null(A$id), A$id))

cat("\n=== Q4: the prize -- 8 ops sharing one W ===\n")
# What the cache would buy on the pattern the TODO names.
t_chain <- tm(function() {
  out <- B
  for (i in 1:8) out <- ag_matmul(A, out)
  out
})
one_upload <- t_full - t_create_flush
cat(sprintf("  8 matmul with one W      : %7.2f ms\n", t_chain))
cat(sprintf("  uploads of W in there    : 8 x %.2f = %.2f ms\n",
            one_upload, 8 * one_upload))
cat(sprintf("  cache would save at most : %5.1f%% of the chain\n",
            100 * 7 * one_upload / t_chain))
cat("  (7, not 8: the first upload still has to happen)\n")

ag_device("cpu")
