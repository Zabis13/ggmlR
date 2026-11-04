# SET_ROWS over ne[4] > 1, against a reference written in R.
#
# The CPU kernel used to walk only i02/i03, so every ne[4] slice past the first
# was silently left untouched; copy_to_quant.comp already traversed it. Both
# backends are checked against the same reference, not against each other.
# The F16 source into an F32 destination takes the per-thread wdata scratch
# row; with SET_ROWS now n_tasks = n_threads the plan runs it on both threads
# and sizes the work buffer for them (the old n_tasks = 1 undersized it
# whenever another node raised the thread count -- an overflow only ASan sees).

# dst a: [nc, nr_dst, ne2, ne3, ne4], src b: [nc, nr, ne2, ne3, ne4],
# idx c: [nr, ne2, ne3, 1, ne4] (no broadcast, so c matches b on dims 1..2)
NC <- 4L; NR_DST <- 5L; NR <- 3L; NE2 <- 2L; NE3 <- 2L; NE4 <- 3L

set.seed(7)
A_DAT <- runif(NC * NR_DST * NE2 * NE3 * NE4, -1, 1)
# multiples of 1/256 in [-1, 1] need at most 9 significant bits: exact in f16
B_DAT <- round(runif(NC * NR * NE2 * NE3 * NE4, -1, 1) * 256) / 256
# distinct destination rows per (i02, i03, i04) so no write overwrites another
C_DAT <- as.integer(unlist(lapply(seq_len(NE2 * NE3 * NE4),
                                  function(k) sample.int(NR_DST, NR) - 1L)))

set_rows_ref <- function() {
  a <- array(A_DAT, c(NC, NR_DST, NE2, NE3, NE4))
  b <- array(B_DAT, c(NC, NR,     NE2, NE3, NE4))
  c <- array(C_DAT, c(NR, NE2, NE3, 1L, NE4))
  for (i4 in seq_len(NE4)) for (i3 in seq_len(NE3)) for (i2 in seq_len(NE2))
    for (i in seq_len(NR))
      a[, c[i, i2, i3, 1L, i4] + 1L, i2, i3, i4] <- b[, i, i2, i3, i4]
  as.numeric(a)
}

run_set_rows_5d <- function(backend, src_type) {
  ctx <- ggml_init(32 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, c(NC, NR_DST, NE2, NE3, NE4))
  b <- ggml_new_tensor(ctx, src_type,      5L, c(NC, NR,     NE2, NE3, NE4))
  c <- ggml_new_tensor(ctx, GGML_TYPE_I32, 5L, c(NR, NE2, NE3, 1L, NE4))
  out <- ggml_set_rows(ctx, a, b, c)
  gf  <- ggml_build_forward_expand(ctx, out)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(a, A_DAT)
  ggml_backend_tensor_set_data(b, B_DAT)
  ggml_backend_tensor_set_data(c, C_DAT)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

for (src in c("F32", "F16")) {
  local({
    src_name <- src
    src_type <- if (src_name == "F32") GGML_TYPE_F32 else GGML_TYPE_F16

    test_that(paste("CPU set_rows covers ne[4], src", src_name), {
      backend <- ggml_backend_cpu_init()
      on.exit(ggml_backend_free(backend))
      ggml_backend_cpu_set_n_threads(backend, 2L)
      expect_equal(run_set_rows_5d(backend, src_type), set_rows_ref(),
                   tolerance = 1e-6)
    })

    test_that(paste("Vulkan set_rows covers ne[4], src", src_name), {
      skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      backend <- ggml_vulkan_init(0)
      on.exit(ggml_backend_free(backend))
      expect_equal(run_set_rows_5d(backend, src_type), set_rows_ref(),
                   tolerance = 1e-6)
    })
  })
}

test_that("set_rows rejects mismatched ne[4]", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, c(NC, NR_DST, 1L, 1L, 2L))
  b <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, c(NC, NR,     1L, 1L, 2L))
  c <- ggml_new_tensor(ctx, GGML_TYPE_I32, 5L, c(NR, 1L, 1L, 1L, 1L))
  expect_error(ggml_set_rows(ctx, a, b, c), "agree in ne\\[4\\]")
})
