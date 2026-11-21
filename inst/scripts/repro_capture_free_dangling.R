# Repro: a freed capture leaves dangling tensor pointers in the Vulkan
# context's unsynced-node lists (ggml_backend_vk_graph_record/_replay only
# called the internal ggml_vk_synchronize, never ggml_vk_graph_cleanup).
# The next graph compute then dereferences freed memory.
#
# Each iteration:
#   1. capture A, replayed through a recorded command buffer (record ON)
#   2. A freed (tensors and buffers released)
#   3. the freed heap is overwritten: many malloc'd raw vectors filled with
#      0x41, so a stale pointer read becomes 0x4141414141414141 (crash) instead
#      of a still-valid old value (silent pass)
#   4. capture B, replayed through the async chain (record OFF)
#   5. results checked against the expected counter / state
#
# Before the fix this should segfault within a few iterations; after the fix it
# must run all iterations and print "OK".
#
# Usage: Rscript repro_capture_free_dangling.R [iterations=50]

suppressMessages(library(ggmlR))

args <- commandArgs(trailingOnly = TRUE)
n_iter <- if (length(args)) as.integer(args[1]) else 50L

if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
  cat("SKIP: no Vulkan device\n")
  quit(status = 0)
}

ag_device("gpu")
ag_dtype("f32")

ones64 <- ag_param(matrix(0.01, 64, 1))

# Scribble over freed heap memory: sizes spread over the range a ggml context
# and its tensor metadata are likely to occupy.
scribble <- function() {
  junk <- lapply(rep(c(256L, 1024L, 4096L, 16384L, 65536L, 262144L), each = 8L),
                 function(n) rep(as.raw(0x41), n))
  invisible(length(junk))
}

one_capture <- function(record, times) {
  old <- Sys.getenv("GGMLR_VK_GRAPH_RECORD")
  Sys.setenv(GGMLR_VK_GRAPH_RECORD = if (record) "1" else "0")
  on.exit(Sys.setenv(GGMLR_VK_GRAPH_RECORD = old), add = TRUE)

  cnt <- ag_buffer(1, 1)
  st <- ag_buffer(64, 4)
  ag_buffer_write(st, matrix(0, 64, 4), col_offset = 0)
  W <- ag_param(diag(64))
  one <- ag_param(matrix(1, 1, 1))
  cols <- matrix(0:3, 4, 1)
  f <- ag_capture(function(z, cols) {
    t <- ag_get_rows(cnt, z)
    s <- ag_get_rows(st, cols)
    ag_set_rows(st, cols, ag_add(ag_matmul(W, s), ag_matmul(ones64, t)))
    ag_set_rows(cnt, z, ag_add(t, one))
    NULL
  }, params = list(cnt, st, W, one, ones64))
  z <- matrix(0, 1, 1)
  ag_capture_repeat(f, times, z = z, cols = cols)
  res <- list(cnt = ag_buffer_read(cnt), st = ag_buffer_read(st))
  # the order of the test that crashed: buffers first, then the capture
  ag_buffer_free(cnt)
  ag_buffer_free(st)
  ag_capture_free(f)
  res
}

# st after k runs: s_k = s_{k-1} + 0.01 * (k - 1), s_0 = 0
expected_st <- function(k) 0.01 * sum(seq_len(k) - 1)

for (i in seq_len(n_iter)) {
  a <- one_capture(record = TRUE, times = 7L)
  scribble()
  gc(verbose = FALSE)
  scribble()
  b <- one_capture(record = FALSE, times = 7L)
  scribble()

  ok <- identical(as.numeric(a$cnt), 7) && identical(as.numeric(b$cnt), 7) &&
    isTRUE(all.equal(a$st, b$st)) &&
    isTRUE(all.equal(as.numeric(a$st), rep(expected_st(7L), 256), tolerance = 1e-5))
  if (!ok) {
    cat(sprintf("FAIL at iteration %d: cnt a=%g b=%g, st a[1]=%g b[1]=%g (expected %g)\n",
                i, a$cnt[1], b$cnt[1], a$st[1], b$st[1], expected_st(7L)))
    quit(status = 1)
  }
  if (i %% 10L == 0L) cat(sprintf("iteration %d/%d\n", i, n_iter))
}

ag_device("cpu")
cat(sprintf("OK: %d iterations\n", n_iter))
