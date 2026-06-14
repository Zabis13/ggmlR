#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# vk_probe_split_errorfree.R — split experiment #2
#
# Does a cycle of  ggml_init -> ggml_new_tensor_1d that THROWS an R error
# ("Not enough memory in context") -> ggml_free(ctx) in the error handler
# corrupt a later Phase B?  NO Vulkan device-buffer allocation happens in the
# cycle (the tensor creation errors out before alloc_ctx_tensors).
#
#   - Phase B reaches DONE  -> the error()/longjmp + ggml_free path is innocent.
#   - Phase B crashes silently -> R's error() longjmp out of R_ggml_new_tensor_1d
#     (skipping C cleanup) on MinGW corrupts the heap.
#
# Usage (Windows):
#     set GGMLR_DBG_LOG=C:\models\split_errorfree.log
#     Rscript inst/examples/vk_probe_split_errorfree.R
# ---------------------------------------------------------------------------

suppressMessages(library(ggmlR))
say <- function(...) { cat(sprintf(...)); cat("\n"); flush.console(); flush(stdout()) }

say("== split_errorfree START ==")
if (!isTRUE(ggml_vulkan_available())) { say("No Vulkan. Exit."); quit(status = 0) }
backend <- ggml_vulkan_init(0L)
say("Backend init OK: %s", ggml_vulkan_backend_name(backend))

# N cycles: create a tiny context, then try to create a tensor that does NOT
# fit -> R_ggml_new_tensor_1d calls error() -> longjmp. We catch it and free
# the context in the handler. No Vulkan alloc happens.
N <- 7L
say("")
say("===== init+error+free cycle x%d =====", N)
for (i in seq_len(N)) {
  say("[E%02d] init + oversized tensor (expect error)", i)
  ctx <- ggml_init(mem_size = 1L * 1024L * 1024L, no_alloc = TRUE)
  tryCatch(
    {
      # 64 M elements * 4 bytes = 256 MB >> 1 MB context guard -> error()
      ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64L * 1024L * 1024L)
      say("[E%02d] UNEXPECTED: tensor created without error", i)
    },
    error = function(e) say("[E%02d] caught: %s", i, conditionMessage(e))
  )
  ggml_free(ctx)
  say("[E%02d] ctx freed", i)
}
gc()
say("cycle complete")

# ---- Phase B (identical to the isolated probe that passed) ----
say("")
say("===== PHASE B =====")
n_elems <- 4096L
ctx <- ggml_init(mem_size = 1L * 1024L * 1024L, no_alloc = TRUE)
t   <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elems)
ggml_set_name(t, "probeB")
buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
say("Phase B buffer allocated OK")

offs <- unique(c(0:16, seq(32L, n_elems - 1L, by = 257L), n_elems - 2L, n_elems - 1L))
offs <- offs[offs < n_elems]
bad <- 0L
for (k in seq_along(offs)) {
  eoff <- offs[k]; byteoff <- eoff * 4L; val <- as.numeric(eoff + 0.5)
  say("[B%04d] write at elem=%d", k, eoff)
  tryCatch(ggml_backend_tensor_set_data(t, val, offset = byteoff),
           error = function(e) say("[B%04d] SET error: %s", k, conditionMessage(e)))
  got <- tryCatch(ggml_backend_tensor_get_data(t, offset = byteoff, n_elements = 1L),
                  error = function(e) { say("[B%04d] GET error: %s", k, conditionMessage(e)); NA })
  if (!(length(got) == 1L && !is.na(got) && isTRUE(all.equal(got, val)))) {
    bad <- bad + 1L; say("[B%04d] MISMATCH wrote=%.1f read=%s", k, val, paste(format(got), collapse=","))
  }
}
say("Phase B complete: %d mismatches", bad)
say("== split_errorfree DONE (clean) ==")
