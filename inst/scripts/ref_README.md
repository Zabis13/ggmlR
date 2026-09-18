# Reference outputs

Numbers this package produces are checked here against ONNX Runtime, a separate
implementation of the same models. The check exists because output of the right
length and shape says nothing about the values in it: roberta returned
`NaN NaN` for a long time while `test_all_onnx.R`, which looks only at the
length, reported it as OK.

## Running

    inst/scripts/ref_check_vs_onnxruntime.sh            # all 15 models
    inst/scripts/ref_check_vs_onnxruntime.sh roberta    # one, by substring

Paths are overridable: `ORT_DIR` (an unpacked onnxruntime-linux-x64 release,
from the project's GitHub releases — the C++ artifact, not the Python package),
`ONNX_DIR` (the .onnx files), `DATA_DIR` (where dumps land).

## Reading the result

Tolerance is 1e-3 absolute, with the relative figure printed alongside since
1e-3 means something different on a logit near 1 than on an image scaled to
255. F32 accumulated in a different order drifts by about 1e-5..1e-4 across a
deep network, so the two implementations will not match exactly; a difference
past the tolerance is a disagreement about the arithmetic, not rounding.

## The pieces

- `dump_io.R` — runs the models through ggmlR, writes inputs and first outputs
  as flat float32 plus `manifest.tsv`
- `ort_reference.cpp` — reads that manifest, runs the same models through ONNX
  Runtime, writes its outputs beside them
- `compare.R` — diffs the two, per model

Inputs travel through files rather than being regenerated on each side: R's RNG
would otherwise have to be reimplemented in C++, putting a third implementation
between the two results being compared. Inputs are matched to the session by
NAME, so a difference in ordering cannot silently swap two tensors of the same
shape.

## Finding WHICH node disagrees

The whole-model check above says only that the numbers differ. To find the node
responsible:

- `ref_edge_types.R` — the element type of named internal edges, read from the
  model's `value_info`. `ref_patch_outputs.R` needs this and getting it wrong
  makes ORT reject the model outright; the ggmlR trace cannot supply it (it
  casts indices to f32, so an i64 edge looks like f32).
- `ref_patch_outputs.R` — add those edges to `graph.output` on a copy, so a
  reference run can be asked to return them.
- `ort_nodes` — run the patched model through ONNX Runtime, one `.ort.bin` per
  edge.
- `ref_compare_traces.R` — compare two `ONNX_TRACE_VALS=1 ONNX_TRACE_SUM=1`
  traces (e.g. cpu vs vulkan) node by node. Matches on
  `name|op|shape|n` + occurrence, NOT on trace index: the backends compute in
  different orders and node names repeat, so both of the obvious keys pair up
  unrelated nodes and report hundreds of phantom differences.
- `repeat_maskrcnn.R` — run one model N times. A single run does not test graph
  reuse, and reuse is where a whole class of defects lives.

A crashing run is usable input: its trace stops, and the node after its last
line is where the backend died. If every matched node agrees and one run still
dies, the defect is memory or scheduling, not arithmetic — look at the
CPU-only/GPU boundary rather than at a kernel.

## ⚠️ Which numbers are comparable

Two traps, both of which sent a diagnosis the wrong way on MaskRCNN:

1. **Never compare figures across runs whose SHAPES differ.** A detector's
   geometry is data-dependent: fix one defect and the detection count changes,
   so `ne[3]` changes, so every sum downstream changes. A trace kept from an
   earlier session describes the model as it was then. Re-take both sides after
   any fix — the cost is one run, and the alternative is chasing a difference
   that is only the batch size moving.

2. **`GGMLR_QCONV_DEBUG_ACC` (and the shader's `debug_acc`) substitutes the
   op's output.** Everything after that node is a different computation, and on
   a shape-dependent model the shapes move too. It answers "what did this ONE
   element sum to" (with `GGMLR_QCONV_ELEM`); it cannot measure a tensor and its
   numbers do not belong beside ORT's.

`ONNX_DUMP_NODES` / `ONNX_DUMP_DIR` is the non-invasive reader: it copies a
named tensor's contents without touching the computation, and it reports the
full `ne[]`, so a per-batch breakdown from it is trustworthy. Prefer it
whenever the question is about values rather than about one element's
arithmetic.
