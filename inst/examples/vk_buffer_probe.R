#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# vk_buffer_probe.R — Vulkan device-buffer allocation / write-readback probe
#
# Purpose: isolate the silent crash in ggml_backend_buffer_init (and the
# device write path) on Windows/MinGW WITHOUT sd2R / flux. It exercises the
# exact code path that crashed:
#     ggml_backend_alloc_ctx_tensors -> vk_alloc_buffer -> ggml_backend_buffer_init
# and the device write/readback path:
#     ggml_backend_tensor_set_data / ggml_backend_tensor_get_data
#
# Two phases:
#   PHASE A — many device-buffer allocations in a row, growing sizes, including
#             the exact sizes seen in the crashing flux run (weights 1038094336,
#             compute 195952688). Prints AFTER each alloc with flush() so a
#             silent crash tells you exactly which alloc index / size died.
#   PHASE B — byte-level write/readback: write small chunks at growing offsets
#             into one device buffer, read each back and compare. A mismatch or
#             crash pinpoints corruption on the write path and the offset.
#
# Usage (Windows, in repo root, with the package installed):
#     Rscript inst/examples/vk_buffer_probe.R
#   optional, capture a survivable log even on silent abort:
#     set GGMLR_DBG_LOG=C:\tmp\vk_probe.log   (then run; tail the file)
#
# Every line is flush()ed immediately so the LAST printed line localizes a
# silent crash. Do not buffer.
# ---------------------------------------------------------------------------

suppressMessages(library(ggmlR))

say <- function(...) { cat(sprintf(...)); cat("\n"); flush.console(); flush(stdout()) }

say("== vk_buffer_probe START ==")
say("ggmlR loaded. Vulkan available: %s", as.character(ggml_vulkan_available()))

if (!isTRUE(ggml_vulkan_available())) {
  say("No Vulkan device. Nothing to probe. Exiting.")
  quit(status = 0)
}

ndev <- ggml_vulkan_device_count()
say("Vulkan device count: %d", ndev)
for (d in seq_len(ndev) - 1L) {
  say("  device %d: %s", d, ggml_vulkan_device_description(d))
  mem <- tryCatch(ggml_vulkan_device_memory(d), error = function(e) NA)
  say("    memory info: %s", paste(utils::capture.output(print(mem)), collapse = " | "))
}

# Vulkan backend handle (device 0). This forces instance + device init, which
# is where shader compile threads spin up — the same state as during flux.
say("Initializing Vulkan backend (device 0)...")
backend <- ggml_vulkan_init(0L)
say("Backend init OK: %s (is_vk=%s)",
    ggml_vulkan_backend_name(backend),
    as.character(ggml_vulkan_is_backend(backend)))

# Helper: allocate a single F32 tensor of `nbytes` bytes onto the Vulkan
# backend via ggml_backend_alloc_ctx_tensors. Returns list(buffer, ctx, tensor)
# or stops. This is the precise path that crashed (vk_alloc_buffer ->
# ggml_backend_buffer_init).
alloc_device_buffer <- function(nbytes, label) {
  n_f32 <- max(1L, as.integer(ceiling(nbytes / 4)))
  # no_alloc = TRUE: tensors get metadata only, real memory comes from the
  # backend buffer (mirrors how weights / compute buffers are allocated).
  # The metadata context only holds tensor *headers* (no_alloc), so a small
  # fixed size is enough regardless of nbytes — using 64 MB here made R's
  # tensor-size guard reject large probes AFTER the context was created, and
  # the resulting error() longjmp leaked the context (host-RAM leak that, on
  # MinGW, corrupted the heap and crashed a later operator new). Keep it tiny
  # and ALWAYS free the context on any failure path.
  ctx <- ggml_init(mem_size = 1L * 1024L * 1024L, no_alloc = TRUE)
  res <- tryCatch({
    t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_f32)
    ggml_set_name(t, label)
    buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
    list(buffer = buf, ctx = ctx, tensor = t, n_f32 = n_f32)
  }, error = function(e) {
    ggml_free(ctx)          # never leak a half-initialized context
    stop(e)
  })
  res
}

# ---------------------------------------------------------------------------
# PHASE A — repeated device allocations, growing + the exact flux sizes
# ---------------------------------------------------------------------------
say("")
say("===== PHASE A: repeated device-buffer allocations =====")

# Growing sizes (bytes) + the two sizes from the crashing flux run.
sizes <- c(
  1024,                      # 1 KB
  64 * 1024,                 # 64 KB
  1024 * 1024,               # 1 MB
  16 * 1024 * 1024,          # 16 MB
  64 * 1024 * 1024,          # 64 MB
  195952688,                 # flux COMPUTE buffer (deterministic crash #2)
  256 * 1024 * 1024,         # 256 MB
  512 * 1024 * 1024,         # 512 MB
  1038094336                 # flux WEIGHTS buffer (50/50 crash #1)
)

# Free each probe's buffer + metadata context right after measuring it. The
# original version kept every successful alloc alive in `allocs` AND leaked a
# context on every failed alloc; the accumulated unfreed contexts (not the
# Vulkan buffers) corrupted the MinGW heap and crashed Phase B's operator new.
# Releasing per-iteration keeps host memory flat and isolates each alloc.
free_probe <- function(res) {
  if (is.null(res)) return(invisible())
  tryCatch(ggml_backend_buffer_free(res$buffer), error = function(e) NULL)
  tryCatch(ggml_free(res$ctx),                   error = function(e) NULL)
}

n_ok <- 0L
for (i in seq_along(sizes)) {
  sz <- sizes[i]
  say("[A%02d] BEFORE alloc size=%.0f bytes (%.2f MB)", i, sz, sz / 1024 / 1024)
  res <- tryCatch(
    alloc_device_buffer(sz, sprintf("probeA_%02d", i)),
    error = function(e) { say("[A%02d] R ERROR: %s", i, conditionMessage(e)); NULL }
  )
  if (is.null(res)) {
    say("[A%02d] alloc returned R error (graceful) — continuing", i)
    next
  }
  bsz <- tryCatch(ggml_backend_buffer_size(res$buffer), error = function(e) NA)
  say("[A%02d] AFTER alloc OK  buffer_size=%s", i, format(bsz, scientific = FALSE))
  free_probe(res)                # release immediately — no accumulation
  say("[A%02d] freed", i)
  n_ok <- n_ok + 1L
}
gc()
say("PHASE A complete: %d successful allocs (all freed)", n_ok)

# ---------------------------------------------------------------------------
# PHASE B — byte-level write / readback into one device buffer
# ---------------------------------------------------------------------------
say("")
say("===== PHASE B: byte-level write/readback =====")

# One modest device buffer; we write one F32 (4 bytes = smallest writable unit
# via the data API) at a time at growing element offsets, then read it back and
# compare. Mismatch => corruption on the device write path; crash => the offset
# that triggered it is the LAST printed [Bxx].
n_elems  <- 4096L                  # 16 KB buffer
probeB   <- alloc_device_buffer(n_elems * 4L, "probeB")
say("PHASE B buffer allocated: %d F32 elements (%d bytes)", probeB$n_f32, probeB$n_f32 * 4L)

# Write offsets to probe: dense at the start, then sparse toward the end, plus
# the very last element (boundary — most likely to reveal an off-by-one write).
elem_offsets <- unique(c(
  0:16,
  seq(32L, n_elems - 1L, by = 257L),
  n_elems - 2L, n_elems - 1L
))
elem_offsets <- elem_offsets[elem_offsets < n_elems]

bad <- 0L
for (k in seq_along(elem_offsets)) {
  eoff   <- elem_offsets[k]
  byteoff <- eoff * 4L
  val    <- as.numeric(eoff + 0.5)         # distinct per offset
  say("[B%04d] write F32=%.1f at elem=%d (byte_off=%d)", k, val, eoff, byteoff)
  tryCatch(
    ggml_backend_tensor_set_data(probeB$tensor, val, offset = byteoff),
    error = function(e) say("[B%04d] SET error: %s", k, conditionMessage(e))
  )
  got <- tryCatch(
    ggml_backend_tensor_get_data(probeB$tensor, offset = byteoff, n_elements = 1L),
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
say("PHASE B complete: %d mismatches over %d writes", bad, length(elem_offsets))

# ---------------------------------------------------------------------------
say("")
say("== vk_buffer_probe DONE (reached the end cleanly) ==")
