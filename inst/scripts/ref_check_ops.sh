#!/usr/bin/env bash
# Check single operations against ONNX Runtime, on CPU and on Vulkan.
#
# ref_check_vs_onnxruntime.sh compares whole models and answers "do the numbers
# agree"; this answers "which op disagrees", by building a one-node .onnx per
# case.  It also covers ops the 15 models do not exercise, and checks the two
# ggmlR backends separately against the same reference -- a shader can diverge
# from the CPU kernel implementing the same op, and vice versa.
#
# Usage:
#   inst/scripts/ref_check_ops.sh              # all cases
#   inst/scripts/ref_check_ops.sh topk         # one group, by substring
#
#   ORT_DIR   unpacked onnxruntime-linux-x64 release
#   DATA_DIR  where the generated models and dumps go

set -u

ORT_DIR="${ORT_DIR:-/mnt/Data2/DS_projects/onnxruntime-linux-x64-1.29.0}"
DATA_DIR="${DATA_DIR:-/tmp/ggmlR-ref/data/ops}"
FILTER="${1:-}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$DATA_DIR/ort_ops"

for p in "$ORT_DIR/include/onnxruntime_cxx_api.h" "$ORT_DIR/lib/libonnxruntime.so"; do
    [ -e "$p" ] || { echo "missing $p"; echo "set ORT_DIR to an unpacked onnxruntime-linux-x64 release"; exit 1; }
done
mkdir -p "$DATA_DIR"

echo "=== 1/3  ggmlR (cpu + vulkan) ================================"
Rscript "$here/ref_ops_vs_onnxruntime.R" "$FILTER" || exit 1

echo
echo "=== 2/3  ONNX Runtime ========================================"
if [ ! -x "$runner" ] || [ "$here/ref_ops_reference.cpp" -nt "$runner" ]; then
    echo "building reference runner..."
    g++ -O2 -std=c++17 "$here/ref_ops_reference.cpp" -o "$runner" \
        -I"$ORT_DIR/include" -L"$ORT_DIR/lib" -lonnxruntime \
        -Wl,-rpath,"$ORT_DIR/lib" || exit 1
fi
"$runner" "$DATA_DIR" 2>/dev/null || exit 1

echo
echo "=== 3/3  comparison =========================================="
Rscript "$here/ref_compare_ops.R" "$DATA_DIR"
