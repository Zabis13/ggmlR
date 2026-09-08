#!/usr/bin/env bash
# Compare ggmlR against ONNX Runtime at several depths of one model.
#
# Walking every node is slower than asking a few well-placed questions.  A
# disagreement that grows steadily with depth is a cause repeating in each
# block; one that appears at a single cut and stays flat is one node, and a
# bisection between the last agreeing cut and the first disagreeing one finds
# it in a handful of runs.
#
# Usage:
#   inst/scripts/ref_probe_depth.sh <model.onnx> <shape-expr> <tensor> [tensor...]
# e.g.
#   inst/scripts/ref_probe_depth.sh ".../botnet26t_256_Opset16.onnx" \
#       "list(x=c(1L,3L,256L,256L))" /stages/... /stages/...

set -u
MODEL="$1"; SHAPE="$2"; shift 2
ORT_DIR="${ORT_DIR:-/mnt/Data2/DS_projects/onnxruntime-linux-x64-1.29.0}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${DEPTH_DIR:-inst/scripts/ref_depth}"
mkdir -p "$work"

runner="$work/ort_reference"
if [ ! -x "$runner" ] || [ "$here/ref_ort_reference.cpp" -nt "$runner" ]; then
    g++ -O2 -std=c++17 "$here/ref_ort_reference.cpp" -o "$runner" \
        -I"$ORT_DIR/include" -L"$ORT_DIR/lib" -lonnxruntime \
        -Wl,-rpath,"$ORT_DIR/lib" || exit 1
fi

i=0
: > "$work/manifest.tsv"
for cut in "$@"; do
    i=$((i+1))
    tag="cut$i"
    printf '%-3s %s\n' "$i" "$cut"
    Rscript "$here/ref_truncate_onnx.R" "$MODEL" "$work/$tag.onnx" "$cut" >/dev/null || continue
    Rscript -e "
      suppressMessages(library(ggmlR))
      sh <- $SHAPE
      set.seed(42)
      inp <- lapply(sh, function(s) runif(prod(s)))
      m <- onnx_load('$work/$tag.onnx', device='cpu', input_shapes=sh)
      v <- onnx_run(m, inp)[[1]]
      for (nm in names(inp))
        writeBin(as.numeric(inp[[nm]]),
                 sprintf('$work/%s.in.%s.bin', '$tag', nm), size=4, endian='little')
      writeBin(as.numeric(v), '$work/$tag.ggmlr.bin', size=4, endian='little')
      rows <- character(0)
      for (nm in names(inp))
        rows <- c(rows, sprintf('in\t%s\t%s.onnx\t%s\t%s\t0', '$tag', '$tag', nm,
                                paste(sh[[nm]], collapse=',')))
      rows <- c(rows, sprintf('out\t%s\t%s.onnx\t%d', '$tag', '$tag', length(v)))
      cat(rows, sep='\n', file='$work/manifest.tsv', append=TRUE)
      cat('\n', file='$work/manifest.tsv', append=TRUE)
    " 2>&1 | grep -v '^$' | sed 's/^/    /'
done

"$runner" "$work" "$work" 2>&1 | sed 's/^/    /'

echo
echo "depth   max|d|        rel"
Rscript -e "
  r <- function(p){n<-file.info(p)\$size/4; readBin(p,'numeric',n=n,size=4,endian='little')}
  for (i in 1:$i) {
    fo <- sprintf('$work/cut%d.ggmlr.bin', i); ft <- sprintf('$work/cut%d.ort.bin', i)
    if (!file.exists(fo) || !file.exists(ft)) { cat(sprintf('cut%-3d  (missing)\n', i)); next }
    o <- r(fo); t <- r(ft)
    if (length(o) != length(t)) { cat(sprintf('cut%-3d  length %d vs %d\n', i, length(o), length(t))); next }
    d <- max(abs(o-t))
    cat(sprintf('cut%-3d  %-12.4g  %.3g\n', i, d, d/max(1e-12, max(abs(t)))))
  }
"
