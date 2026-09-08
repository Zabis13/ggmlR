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
