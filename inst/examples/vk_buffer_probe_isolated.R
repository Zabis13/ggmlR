#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# vk_buffer_probe_isolated.R — PHASE B ONLY, no Phase A.
#
# Purpose: test whether the silent crash in ggml_backend_buffer_init depends on
# accumulated state (9 leaked ggml_init contexts + 4 live buffers from Phase A)
# or reproduces on the VERY FIRST device buffer in a clean process.
#
#   - If probeB allocates + writes + reads cleanly here, the crash is caused by
#     accumulated heap corruption / context overflow in Phase A, NOT by the
#     buffer_init path itself (the "5th buffer" was an artifact of the script:
#     4 allocs in A + 1 in B).
#   - If it still crashes on this single buffer, the bug is in
#     vk_alloc_buffer -> ggml_backend_buffer_init on the first buffer.
#
# Usage (Windows):
#     set GGMLR_DBG_LOG=C:\tmp\vk_probe_iso.log
#     Rscript inst/examples/vk_buffer_probe_isolated.R
# ---------------------------------------------------------------------------

suppressMessages(library(ggmlR))

say <- function(...) { cat(sprintf(...)); cat("\n"); flush.console(); flush(stdout()) }

say("== vk_buffer_probe_isolated START ==")
say("ggmlR loaded. Vulkan available: %s", as.character(ggml_vulkan_available()))

if (!isTRUE(ggml_vulkan_available())) {
  say("No Vulkan device. Nothing to probe. Exiting.")
  quit(status = 0)
}

say("Initializing Vulkan backend (device 0)...")
backend <- ggml_vulkan_init(0L)
say("Backend init OK: %s (is_vk=%s)",
    ggml_vulkan_backend_name(backend),
    as.character(ggml_vulkan_is_backend(backend)))

# Single device buffer, allocated in a FRESH context. No prior allocations.
n_elems <- 4096L                                   # 16 KB buffer
ctx <- ggml_init(mem_size = 64L * 1024L * 1024L, no_alloc = TRUE)
t   <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elems)
ggml_set_name(t, "probeB")
say("Tensor created. Allocating device buffer (first and only)...")
buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
say("Device buffer allocated OK. (If you see this line, buffer_init survived.)")

bsz <- tryCatch(ggml_backend_buffer_size(buf), error = function(e) NA)
say("buffer_size=%s", format(bsz, scientific = FALSE))

# ---------------------------------------------------------------------------
# Byte-level write / readback (same as Phase B in the original probe)
# ---------------------------------------------------------------------------
say("")
say("===== write/readback =====")

elem_offsets <- unique(c(
  0:16,
  seq(32L, n_elems - 1L, by = 257L),
  n_elems - 2L, n_elems - 1L
))
elem_offsets <- elem_offsets[elem_offsets < n_elems]

bad <- 0L
for (k in seq_along(elem_offsets)) {
  eoff    <- elem_offsets[k]
  byteoff <- eoff * 4L
  val     <- as.numeric(eoff + 0.5)
  say("[B%04d] write F32=%.1f at elem=%d (byte_off=%d)", k, val, eoff, byteoff)
  tryCatch(
    ggml_backend_tensor_set_data(t, val, offset = byteoff),
    error = function(e) say("[B%04d] SET error: %s", k, conditionMessage(e))
  )
  got <- tryCatch(
    ggml_backend_tensor_get_data(t, offset = byteoff, n_elements = 1L),
    error = function(e) { say("[B%04d] GET error: %s", k, conditionMessage(e)); NA }
  )
  if (length(got) == 1L && !is.na(got) && isTRUE(all.equal(got, val))) {
    # ok, silent
  } else {
    bad <- bad + 1L
    say("[B%04d] MISMATCH at elem=%d: wrote=%.4f read=%s", k, eoff, val,
        paste(format(got), collapse = ","))
  }
}
say("write/readback complete: %d mismatches over %d writes", bad, length(elem_offsets))

say("")
say("== vk_buffer_probe_isolated DONE (reached the end cleanly) ==")
