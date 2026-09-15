#!/usr/bin/env bash
# Check ggmlR's numbers against ONNX Runtime, end to end.
#
# Runs the models through this package, runs them again through ONNX Runtime,
# and diffs the two.  The reference exists because output of the right length
# says nothing about the values in it: roberta returned NaN for a long time and
# test_all_onnx.R, which checks only the length, called it OK.
#
# Usage:
#   inst/scripts/ref_check_vs_onnxruntime.sh                 # all models
#   inst/scripts/ref_check_vs_onnxruntime.sh roberta         # one, by substring
#
# Override with environment variables if the paths differ:
#   ORT_DIR    unpacked onnxruntime-linux-x64 release
#   ONNX_DIR   directory holding the .onnx files
#   DATA_DIR   where the dumps go

set -u

ORT_DIR="${ORT_DIR:-/mnt/Data2/DS_projects/onnxruntime-linux-x64-1.29.0}"
ONNX_DIR="${ONNX_DIR:-/mnt/Data2/DS_projects/ONNX models-main}"
DATA_DIR="${DATA_DIR:-/tmp/ggmlR-ref/data}"
FILTER="${1:-}"
# Comma-separated ggmlR backends to check, each against the same ORT reference.
# Default is cpu alone, which is what this script has always done; DEVICES=cpu,vulkan
# adds a second dump per model, reported as <model>@vulkan.
DEVICES="${DEVICES:-cpu}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$DATA_DIR/ort_reference"

for p in "$ORT_DIR/include/onnxruntime_cxx_api.h" "$ORT_DIR/lib/libonnxruntime.so"; do
    [ -e "$p" ] || { echo "missing $p"; echo "set ORT_DIR to an unpacked onnxruntime-linux-x64 release"; exit 1; }
done
[ -d "$ONNX_DIR" ] || { echo "missing models dir: $ONNX_DIR"; exit 1; }
mkdir -p "$DATA_DIR"

echo "=== 1/3  ggmlR ==============================================="
Rscript "$here/ref_dump_io.R" "$DATA_DIR" "$FILTER" "$DEVICES" || exit 1

echo
echo "=== 2/3  ONNX Runtime ========================================"
# Rebuilt only when the source is newer, so repeat runs skip straight to the
# comparison.
if [ ! -x "$runner" ] || [ "$here/ref_ort_reference.cpp" -nt "$runner" ]; then
    echo "building reference runner..."
    g++ -O2 -std=c++17 "$here/ref_ort_reference.cpp" -o "$runner" \
        -I"$ORT_DIR/include" -L"$ORT_DIR/lib" -lonnxruntime \
        -Wl,-rpath,"$ORT_DIR/lib" || exit 1
fi
"$runner" "$ONNX_DIR" "$DATA_DIR" || exit 1

echo
echo "=== 3/3  comparison =========================================="
Rscript "$here/ref_compare.R" "$DATA_DIR"
