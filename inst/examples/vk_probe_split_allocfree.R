#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# vk_probe_split_allocfree.R — split experiment #1
#
# Does a cycle of Vulkan device-buffer alloc+free (NO error iterations) corrupt
# a later Phase B?  Run N successful small Vulkan allocs, freeing each
# immediately, then run Phase B (single buffer + write/readback).
#
#   - Phase B reaches DONE  -> Vulkan alloc+free cycle is innocent.
#   - Phase B crashes silently -> ggml_backend_buffer_free (VkDeviceMemory
#     release) is the corrupter.
#
# Usage (Windows):
#     set GGMLR_DBG_LOG=C:\models\split_allocfree.log
#     Rscript inst/examples/vk_probe_split_allocfree.R
# ---------------------------------------------------------------------------

suppressMessages(library(ggmlR))
say <- function(...) { cat(sprintf(...)); cat("\n"); flush.console(); flush(stdout()) }

say("== split_allocfree START ==")
if (!isTRUE(ggml_vulkan_available())) { say("No Vulkan. Exit."); quit(status = 0) }
backend <- ggml_vulkan_init(0L)
say("Backend init OK: %s", ggml_vulkan_backend_name(backend))

# N successful small Vulkan allocs, each freed immediately. NO error path.
N <- 7L
say("")
say("===== alloc+free cycle x%d (no errors) =====", N)
for (i in seq_len(N)) {
  say("[C%02d] alloc", i)
  ctx <- ggml_init(mem_size = 64L * 1024L * 1024L, no_alloc = TRUE)  # 64MB like the isolated probe that PASSED
  t   <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256L)   # 1 KB, always fits
  ggml_set_name(t, sprintf("c_%02d", i))
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  say("[C%02d] free", i)
  ggml_backend_buffer_free(buf)
  ggml_free(ctx)
}
gc()
say("cycle complete")

# ---- Phase B (identical to the isolated probe that passed) ----
say("")
say("===== PHASE B =====")
n_elems <- 4096L
ctx <- ggml_init(mem_size = 64L * 1024L * 1024L, no_alloc = TRUE)  # 64MB like the isolated probe that PASSED
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
say("== split_allocfree DONE (clean) ==")
