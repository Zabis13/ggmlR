#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# ggmlR Tensor Parallelism — P2P / cross-device transport diagnostic driver.
#
# Runs the Vulkan P2P self-test across transports and device pairs, with the
# crash-survivable teardown trace enabled, so a segfault during buffer/device
# teardown is localized to an exact step (the trace file's last line names it).
#
# The trace is written by r_tp_tracef() at key points (P2P copy, ~vk_buffer_struct,
# ~vk_device_struct). It is SILENT in normal use; this script turns it on via the
# GGMLR_TP_TRACE env var. Set it to "-" for stderr, or to a file path.
#
# Usage:
#   Rscript inst/examples/tp_p2p_diagnose.R                 # all pairs, host-staging
#   GGMLR_TP_TRACE=/tmp/tp.log Rscript ... tp_p2p_diagnose.R # trace to a file
#   Rscript inst/examples/tp_p2p_diagnose.R 0 3 opaque-fd   # one pair+transport
#
# Args (all optional): <src_dev> <dst_dev> <transport>
#   transport in {host-staging, opaque-fd, device-group}. Default: host-staging.
# ---------------------------------------------------------------------------

suppressMessages(library(ggmlR))

# Trace destination. IMPORTANT for teardown crashes: device destructors
# (~vk_device_struct) run at PROCESS EXIT, during static-object destruction,
# where /dev/stderr may already be closed — a "-" (stderr) trace loses exactly
# those lines. To capture teardown, trace to a FILE (fopen works at exit; the
# filesystem outlives stderr). Default here is a temp file; set GGMLR_TP_TRACE
# yourself to override (use "-" only when you don't care about the exit phase).
if (Sys.getenv("GGMLR_TP_TRACE") == "") {
  Sys.setenv(GGMLR_TP_TRACE = file.path(tempdir(), "ggmlr_tp_trace.log"))
}
trace_dst <- Sys.getenv("GGMLR_TP_TRACE")
# Start each run with a fresh trace file so the last line is unambiguous.
if (trace_dst != "-" && trace_dst != "stderr" && file.exists(trace_dst)) {
  file.remove(trace_dst)
}

if (!ggml_vulkan_available()) {
  cat("Vulkan not available — nothing to diagnose.\n")
  quit(save = "no", status = 0)
}

ndev <- ggml_vulkan_device_count()
cat(sprintf("Vulkan devices: %d   trace -> %s\n", ndev, trace_dst))

args <- commandArgs(trailingOnly = TRUE)

run_one <- function(src, dst, transport, bytes = 64L * 1024L * 1024L, iters = 20L) {
  cat(sprintf("\n=== P2P self-test  src=dev%d  dst=dev%d  transport=%s ===\n",
              src, dst, transport))
  r <- ggml_vulkan_p2p_selftest(as.integer(src), as.integer(dst),
                                bytes = bytes, iters = iters, transport = transport)
  cat(r$report)
  cat(sprintf("status=%d  gbps=%.3f\n", r$status, r$gbps))
  invisible(r)
}

if (length(args) >= 2) {
  # Explicit single run.
  src <- as.integer(args[[1]])
  dst <- as.integer(args[[2]])
  transport <- if (length(args) >= 3) args[[3]] else "host-staging"
  run_one(src, dst, transport)
} else {
  # Sweep: loopback on dev0, then dev0 -> each other device, host-staging.
  transport <- if (length(args) >= 1) args[[1]] else "host-staging"
  run_one(0L, 0L, transport)                 # loopback sanity
  if (ndev >= 2) {
    for (dst in seq_len(ndev - 1L)) {
      run_one(0L, dst, transport)
    }
  } else {
    cat("\nOnly 1 device — cross-device path not exercised.\n")
  }
}

cat("\nAll self-tests returned. If the process now segfaults, the crash is in\n")
cat("teardown at process exit — the trace file's LAST line names the failing step.\n")
if (trace_dst != "-" && trace_dst != "stderr") {
  cat(sprintf("Trace file: %s\n", trace_dst))
  cat("  (device teardown lines are appended AFTER this point, at process exit;\n")
  cat("   re-read the file after the process ends: `tail -n 40 <file>`)\n")
}
# Note: we deliberately do NOT read the trace file here — the interesting
# teardown lines are written after this script's last statement, during exit.

# Explicit Vulkan teardown while the loader is still mapped: this is the clean way
# to exit without the flaky exit-time segfault: hard = TRUE calls _exit(0) after
# teardown, skipping the loader-static-destruction phase entirely. Must be the LAST
# statement (does not return). Drop `hard = TRUE` to observe the crash / trace.
ggml_vulkan_shutdown(hard = TRUE)

