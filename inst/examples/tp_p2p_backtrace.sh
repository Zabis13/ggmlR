#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ggmlR TP — capture a backtrace of the cross-device P2P exit-teardown segfault.
#
# The crash is a heisenbug: enabling the file trace (GGMLR_TP_TRACE) perturbs
# timing enough to hide it. So this script runs the FAILING path — trace OFF —
# under gdb in batch mode, lets it segfault, and prints the stack. No core dump
# and no interactive session needed.
#
# What to look for in the output:
#   * frames in libnvidia*/radeonsi/libvulkan, or __run_exit_handlers /
#     _dl_fini / __cxa_finalize  -> crash is in the driver/loader at process
#     exit, NOT in ggmlR code (confirms the diagnosis).
#   * frames in ggml_vk_* / ~vk_buffer_struct / ~vk_device_struct  -> it IS our
#     code and we fix that function.
#
# Usage:
#   bash tp_p2p_backtrace.sh            # default pair dev0 -> dev3
#   bash tp_p2p_backtrace.sh 0 1        # pick the pair
# ---------------------------------------------------------------------------
set -u

SRC="${1:-0}"
DST="${2:-3}"

# The real R binary (R on the CLI is a shell wrapper gdb cannot run directly).
R_BIN="$(R RHOME)/bin/exec/R"
if [ ! -x "$R_BIN" ]; then
  # Fallback to common location.
  R_BIN="/usr/lib/R/bin/exec/R"
fi

if ! command -v gdb >/dev/null 2>&1; then
  echo "gdb not found. Install it:  sudo apt-get install -y gdb" >&2
  exit 127
fi

echo "R binary : $R_BIN"
echo "gdb      : $(command -v gdb)"
echo "pair     : dev${SRC} -> dev${DST}   (trace OFF, so the crash reproduces)"
echo "-----------------------------------------------------------------------"

# Trace OFF (empty) so the heisenbug is NOT masked. --vanilla to avoid site/user
# profiles. gdb: run to completion; on the signal, dump the crashing thread's
# backtrace, then all threads, then quit.
GGMLR_TP_TRACE= gdb -batch \
  -ex 'set pagination off' \
  -ex 'set confirm off' \
  -ex "run --vanilla -e 'library(ggmlR); ggml_vulkan_p2p_selftest(${SRC}L, ${DST}L)'" \
  -ex 'echo \n===== BACKTRACE (crashing thread) =====\n' \
  -ex 'bt' \
  -ex 'echo \n===== BACKTRACE (all threads) =====\n' \
  -ex 'thread apply all bt' \
  -ex 'quit' \
  "$R_BIN" 2>&1

echo "-----------------------------------------------------------------------"
echo "If no crash appeared above, the process exited cleanly under gdb"
echo "(gdb can itself perturb timing — like the trace does). In that case the"
echo "bug is confirmed timing-sensitive; re-run a few times or accept the"
echo "heisenbug diagnosis."
