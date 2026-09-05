# Two APIs: `nn_*` (sequential/functional) and `ag_*` (autograd)

ggmlR has two ways to build and train a model, and they are separate systems
rather than two spellings of one. This document says where the boundary runs,
what each side is good at, and which limitations are structural rather than
missing features.

Referred to below as **path A** (the compiled `nn_*` API: `ggml_model_sequential()`,
`ggml_model()`, `ggml_compile()`, `ggml_fit()`) and **path B** (the eager `ag_*`
API: `ag_param`, `with_grad_tape()`, `backward()`, `optimizer_adam()`).

## The one-line version

Path A compiles a static graph and hands the whole training loop to ggml, which
runs it on the GPU. Path B executes one operation at a time from R and
differentiates with R closures.

Use path A to train a model whose shape is fixed. Use path B when the shape
depends on the data, when you need a gradient you can inspect, or when you are
prototyping. Path A is the faster of the two by a wide margin whenever it
applies.

## They do not share objects

This is the boundary that catches people first, so it is worth stating plainly:

- `ag_linear()`, `ag_multihead_attention()`, `ag_batch_norm()` and the rest of
  the `ag_*` layers **cannot** be passed to `ggml_model_sequential()` or
  `ggml_model()`.
- `ggml_layer_dense()`, `ggml_layer_conv_2d()` and the rest of the `nn_*` layers
  **cannot** be used inside `with_grad_tape()`.
- An `ag_tensor` is not a ggml tensor handle, and a compiled model's weights are
  not `ag_param`s.

The names rhyme because the concepts do; the objects are unrelated. There is no
converter, and adding one would mean reconciling two different notions of what a
graph is — a static one built once by `ggml_compile()`, and a tape rebuilt on
every forward pass.

What *is* shared: the tensor ops underneath (`ggml_add`, `ggml_mul_mat`, ...),
the backends, and the device selection machinery.

## What each path can do that the other cannot

| | path A (`nn_*`) | path B (`ag_*`) |
|---|---|---|
| graph | static, compiled once | rebuilt each forward |
| training loop | inside ggml (C) | in R |
| control flow on data | no | yes |
| shapes known at | compile time | run time |
| gradients | inside the graph, not exposed per-tensor | `$grad` on every tensor, inspectable |
| GPU | the whole graph runs there | per-op upload/download, usually a loss |
| dtype | f32/f16/quantised, real | f32/f16 for forward only (see below) |
| speed | much faster | prototyping speed |

## Structural limitations of path B

These are consequences of the design, not gaps waiting to be filled. Each was
established by measurement; the memory notes name the scripts.

**Gradients and optimizer state are R doubles.** `ag_dtype()` sets the type used
when a forward tensor is uploaded into a ggml buffer. It does not reach
`backward()`, which accumulates `$grad` in R closures, nor the optimizer's `m`
and `v`, which are R matrices. So a parameter costs 8 bytes per scalar per copy,
and a training budget is twice a naive f32 estimate. Two upsides fall out of
this: gradient underflow does not occur (no loss scaling needed), and optimizer
states are already in higher precision than the forward pass. Use
`ag_estimate_training_memory()` for the arithmetic.

**The tape holds activations until `zero_grad()`.** Every recorded operation
keeps what its backward rule needs, and the tape is cleared only after the
optimizer step, so the peak is the whole tape at once. `ag_tape_memory()` reports
what a live tape holds, split into what the parameters keep alive anyway and what
clearing would actually release. On a dense stack the freeable share runs 15–30%,
rising with batch size and falling with both depth and width.

**The GPU is usually slower per op.** Each `ag_*` call uploads its operands,
computes a one-node graph and reads the result back. The transfer dominates: a
trivial `ag_scale` on a 4 MB operand costs more than a 1024×1024 matmul does on
the CPU. This does not improve with size — growing the layer makes it worse,
since the upload grows as `d²` while the CPU has threaded BLAS to fall back on.
The exception is a fused op like `ag_flash_attention()`, which uploads Q, K and V
once and does the whole attention on the device. See the README section "When the
GPU is worth it on the `ag_*` path".

**Some ops are inference-only.** The SSM/RWKV bindings have no backward and
cannot appear in a trained graph.

## Marshalling: what survives being sent to another process

`LearnerClassifGGML` and `LearnerRegrGGML` implement mlr3's `marshal` property,
which mlr3 needs whenever a fitted learner crosses a process boundary — parallel
resampling with `future`, `benchmark()`, saving a learner to disk.

Live `ag_*` modules are trees of environments and closures. Serializing those
directly is fragile across package versions, so ggmlR does not: it saves a
**state dict** of plain numeric matrices (`ag_save_model()`, design note "M2" at
the top of `R/ag_save.R`) — trainable parameters plus non-trainable buffers such
as batch-norm running statistics — together with a zero-argument closure that
rebuilds the architecture. On the far side the architecture is rebuilt and the
values copied back by name.

Two limitations follow directly:

1. **A `model_fn` that reads `task` is not marshalable on the autograd
   tradepath.** The rebuild closure deliberately captures only dimensions and
   hyperparameters, so that it serializes cheaply
   (`R/LearnerClassifGGML.R`). A custom `model_fn` that closes over the task
   would drag the data along with it, so it is not captured — and the rebuild
   will fail on the other side.
2. **Values are matched by name.** A rebuild that produces differently named
   parameters than the ones saved will not error at load time; the mismatch
   surfaces later as a model that has architecture but not training.

Observation weights are a related asymmetry worth knowing: the autograd
tradepath ignores `task` weights and warns, while the sequential/functional
tradepath applies them.

## Where to read more

- `inst/docs/ag_data_contract.md` — how an `ag_tensor`'s value must be read and
  written once the tape can be device-resident.
- `vignettes/autograd-engine.Rmd` — path B as a tutorial.
- `vignettes/keras-like-api.Rmd` — path A as a tutorial.
- `vignettes/mlr3-integration.Rmd` — the learners, including marshalling.
- README, "When the GPU is worth it on the `ag_*` path" — the measurements
  behind the GPU note above.
