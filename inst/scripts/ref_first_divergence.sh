#!/usr/bin/env bash
# Find the FIRST node where the Vulkan and CPU backends disagree.
#
# ref_check_vs_onnxruntime.sh compares whole-model outputs and says only that
# they differ -- on MaskRCNN it does not even get that far, failing on the
# LENGTH (100 detections against ONNX Runtime's 51) before any value is
# compared. That is the wrong end of the graph to start from: by the output,
# a single wrong node has already cascaded through a detector's box decode,
# NMS and mask head.
#
# So run the same model twice, once per backend, with ONNX_TRACE_VALS=1 -- which
# prints every node's output as it is computed -- and diff the two traces. The
# first differing line is the node to look at; everything after it is
# downstream noise.
#
# ⚠️ This compares the two ggmlR backends against EACH OTHER, not against ONNX
#    Runtime. A defect both backends share is invisible here (the CPU path is
#    the one that holds max|d|=0 against ORT, so in practice CPU is the
#    reference -- but that is an assumption this script cannot check).
#
# Usage:
#   inst/scripts/ref_first_divergence.sh [model.onnx]
#
#   ONNX_DIR  directory holding the .onnx files
#   OUT_DIR   where the two traces go (default /tmp/ggmlR-div)

set -u

ONNX_DIR="${ONNX_DIR:-/mnt/Data2/DS_projects/ONNX models-main}"
MODEL="${1:-MaskRCNN-12-int8.onnx}"
OUT_DIR="${OUT_DIR:-/tmp/ggmlR-div}"
MODEL_PATH="$ONNX_DIR/$MODEL"

[ -f "$MODEL_PATH" ] || { echo "missing model: $MODEL_PATH"; exit 1; }
mkdir -p "$OUT_DIR"

run_one() {
    local dev="$1" out="$2"
    echo "=== $dev ==="
    ONNX_TRACE_VALS=1 stdbuf -oL -eL Rscript -e "
        suppressMessages(library(ggmlR))
        m <- onnx_load('$MODEL_PATH', device = '$dev',
                       input_shapes = list(image = c(3L, 224L, 224L)))
        set.seed(42)
        invisible(onnx_run(m, list(image = runif(3*224*224))))
    " 2>&1 | grep '^\[val\]' > "$out"
    echo "  $(wc -l < "$out") nodes traced -> $out"
}

run_one cpu    "$OUT_DIR/cpu.txt"
run_one vulkan "$OUT_DIR/vulkan.txt"

echo
echo "=== first divergence ========================================="

# Compare line by line. The node index and name must line up; if the two runs
# traced a different NUMBER of nodes, that is itself the finding -- the graphs
# were cut into segments differently and no value comparison is meaningful.
n_cpu=$(wc -l < "$OUT_DIR/cpu.txt")
n_vk=$(wc -l < "$OUT_DIR/vulkan.txt")
if [ "$n_cpu" -ne "$n_vk" ]; then
    echo "⚠️  different node counts: cpu=$n_cpu vulkan=$n_vk"
    echo "    The graphs do not correspond; compare the first lines that differ:"
fi

diff_line=$(diff <(cat "$OUT_DIR/cpu.txt") <(cat "$OUT_DIR/vulkan.txt") \
            | grep -m1 -n '^<' || true)
if [ -z "$diff_line" ]; then
    echo "✅ identical traces -- the backends agree on every node."
    exit 0
fi

# Show the first differing pair with a little context on each side.
first=$(diff --unchanged-line-format= --old-line-format='%dn:%L' \
             --new-line-format= "$OUT_DIR/cpu.txt" "$OUT_DIR/vulkan.txt" \
        | head -1 | cut -d: -f1)
echo "first differing trace line: $first"
echo
echo "--- cpu ---"
sed -n "$((first > 2 ? first - 2 : 1)),$((first + 2))p" "$OUT_DIR/cpu.txt"
echo "--- vulkan ---"
sed -n "$((first > 2 ? first - 2 : 1)),$((first + 2))p" "$OUT_DIR/vulkan.txt"
echo
echo "The node on the 'first differing' line is where to look; everything"
echo "below it is downstream of that one difference."
