#!/usr/bin/env Rscript
# Check single operations against ONNX Runtime, on CPU and on Vulkan.
#
# ref_check_vs_onnxruntime.sh compares whole models, which says whether the
# arithmetic agrees end to end but not which op disagrees when it does not --
# and it only exercises what those 15 models happen to use.  This builds a
# one-node .onnx per case instead, so a disagreement names its own op.
#
# Comparing the two ggmlR backends against each other is not enough, and that
# is the reason this exists: ggml_top_k returned its top two entries swapped
# on BOTH backends, deliberately, and no CPU-vs-Vulkan check could see it.
# ONNX Runtime is a separate implementation and does not share the mistake.
#
# Usage:
#   Rscript inst/scripts/ref_ops_vs_onnxruntime.R            # all cases
#   Rscript inst/scripts/ref_ops_vs_onnxruntime.R sort       # by substring
#
# ORT_DIR overrides where the onnxruntime release is unpacked.

suppressMessages(library(ggmlR))

ORT_DIR  <- Sys.getenv("ORT_DIR", "/mnt/Data2/DS_projects/onnxruntime-linux-x64-1.29.0")
DATA_DIR <- Sys.getenv("DATA_DIR", "/tmp/ggmlR-ref/data/ops")
FILTER   <- commandArgs(trailingOnly = TRUE)[1]
TOL      <- 1e-3

dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)

# ── protobuf writing, same encoding as tests/testthat/helper-onnx.R ──
# Duplicated rather than sourced: that file is a testthat helper, loaded into
# the test environment, and this script runs outside it.
.pb_varint <- function(value) {
  value <- as.numeric(value)
  if (value < 0) value <- value + 2^64
  out <- raw(0)
  repeat {
    b <- value %% 128; value <- value %/% 128
    if (value > 0) b <- b + 128
    out <- c(out, as.raw(b))
    if (value == 0) break
  }
  out
}
.pb_tag          <- function(f, w) .pb_varint(bitwShiftL(f, 3) + w)
.pb_bytes        <- function(f, d) c(.pb_tag(f, 2L), .pb_varint(length(d)), d)
.pb_varint_field <- function(f, v) c(.pb_tag(f, 0L), .pb_varint(v))
.pb_string       <- function(f, s) .pb_bytes(f, charToRaw(s))
.float_bytes     <- function(x) writeBin(as.numeric(x), raw(), size = 4, endian = "little")
.int64_bytes     <- function(x) {
  # Little-endian two's complement, built a byte at a time: writeBin with
  # size = 8 on a numeric writes an IEEE double, whose bit pattern an int64
  # reader takes for a huge or nonsensical integer -- K came out as 0.
  out <- raw(0)
  for (v in x) {
    n <- as.numeric(v)
    if (n < 0) n <- n + 2^64
    b <- raw(8)
    for (i in 1:8) { b[i] <- as.raw(n %% 256); n <- n %/% 256 }
    out <- c(out, b)
  }
  out
}

.dim       <- function(v) .pb_varint_field(1L, v)
.shape     <- function(dims) { o <- raw(0); for (d in dims) o <- c(o, .pb_bytes(1L, .dim(d))); o }
.ttype     <- function(et, dims) c(.pb_varint_field(1L, et), .pb_bytes(2L, .shape(dims)))
.tproto    <- function(et, dims) .pb_bytes(1L, .ttype(et, dims))
.vinfo     <- function(name, et = 1L, dims = integer(0))
  c(.pb_string(1L, name), .pb_bytes(2L, .tproto(et, dims)))
.tensor    <- function(name, dims, dt = 1L, raw_data = raw(0)) {
  o <- raw(0)
  for (d in dims) o <- c(o, .pb_varint_field(1L, d))
  c(o, .pb_varint_field(2L, dt), .pb_string(8L, name), .pb_bytes(9L, raw_data))
}
.attr_int  <- function(name, v)
  c(.pb_string(1L, name), .pb_varint_field(3L, v), .pb_varint_field(20L, 2L))
.attr_ints <- function(name, vs) {
  o <- c(.pb_string(1L, name), .pb_varint_field(20L, 7L))
  for (v in vs) o <- c(o, .pb_varint_field(8L, v))
  o
}
# AttributeProto: field 2 is a float (wire type 5, four bytes), field 4 a
# string, and field 20 the type tag -- 1 = FLOAT, 3 = STRING.
.pb_fixed32  <- function(f, v) c(.pb_tag(f, 5L), .float_bytes(v))
.attr_float  <- function(name, v)
  c(.pb_string(1L, name), .pb_fixed32(2L, v), .pb_varint_field(20L, 1L))
.attr_string <- function(name, s)
  c(.pb_string(1L, name), .pb_bytes(4L, charToRaw(s)), .pb_varint_field(20L, 3L))

# Quantised initializers. int8 and uint8 tensors go in raw_data one byte per
# element, which is also how the ONNX exporters write them; a value is stored
# as its two's-complement byte so a reader that takes it unsigned still sees
# the right bits.
.int8_bytes  <- function(x)
  as.raw(vapply(x, function(v) { n <- as.integer(v); if (n < 0) n + 256L else n }, integer(1)))
.uint8_bytes <- function(x) as.raw(as.integer(x))
.int32_bytes <- function(x) writeBin(as.integer(x), raw(), size = 4, endian = "little")
.node <- function(op, inputs, outputs, attrs = list()) {
  o <- raw(0)
  for (i in inputs)  o <- c(o, .pb_string(1L, i))
  for (i in outputs) o <- c(o, .pb_string(2L, i))
  o <- c(o, .pb_string(4L, op))
  for (a in attrs) o <- c(o, .pb_bytes(5L, a))
  o
}
.graph <- function(nodes, inputs, outputs, inits = list()) {
  o <- raw(0)
  for (n in nodes)  o <- c(o, .pb_bytes(1L, n))
  o <- c(o, .pb_string(2L, "g"))
  for (i in inits)  o <- c(o, .pb_bytes(5L, i))
  for (i in inputs) o <- c(o, .pb_bytes(11L, i))
  for (i in outputs) o <- c(o, .pb_bytes(12L, i))
  o
}
.model <- function(graph, opset = 13L)
  c(.pb_varint_field(1L, 7L),
    .pb_bytes(8L, .pb_varint_field(2L, opset)),
    .pb_bytes(7L, graph))

# ── the cases ────────────────────────────────────────────────────
#
# Values are drawn from a small set on purpose.  A tie is where an op's
# contract stops being arithmetic and starts being a convention, and the
# conventions are what differ between implementations: which of two equal
# elements a sort keeps first, which index an argmax reports.  Distinct
# random values would pass regardless.

set.seed(11L)
ties16  <- as.numeric(rep(c(3, 1, 4, 1), each = 4))
ties64  <- as.numeric(sample.int(6, 64, replace = TRUE))
plain32 <- as.numeric(round(rnorm(32), 3))

# Each case: name, the .onnx bytes, the input, and its ONNX dims.
make_case <- function(name, op, dims, input, attrs = list(),
                      out_dims = dims, out_type = 1L, extra_out = NULL) {
  nodes <- list(.node(op, "X", "Y", attrs))
  outs  <- list(.vinfo("Y", out_type, out_dims))
  if (!is.null(extra_out)) {
    nodes <- list(.node(op, "X", c("Y", "Z"), attrs))
    outs  <- list(.vinfo("Y", out_type, out_dims),
                  .vinfo("Z", extra_out$type, extra_out$dims))
  }
  g <- .graph(nodes, list(.vinfo("X", 1L, dims)), outs)
  list(name = name, model = .model(g), input = input, dims = dims)
}

# TopK takes K as a second input; it is an initializer here so the model is
# self-contained and ggmlR can resolve it at build time.
# axis is given positively, never as -1: the varint writer above encodes a
# negative int64 as ten bytes, which ORT reads back as 0 -- the model then asks
# for k elements along an axis of length 1 and fails at run time.  The same
# encoding trap cost a session once already (see the ONNX ops notes).
make_topk <- function(name, dims, input, k, axis = length(dims) - 1L) {
  kinit <- .tensor("K", 1L, 7L, .int64_bytes(k))
  n <- .node("TopK", c("X", "K"), c("V", "I"),
             attrs = list(.attr_int("axis", axis), .attr_int("largest", 1L),
                          .attr_int("sorted", 1L)))
  odims <- dims; odims[length(odims)] <- k
  # K is declared as a graph input as well as an initializer.  An initializer
  # that no input declares is legal in the spec but leaves K's shape unstated,
  # and ORT's shape inference then reads it as empty and rejects the model
  # with "Axis has less than the requested k elements".
  g <- .graph(list(n), list(.vinfo("X", 1L, dims), .vinfo("K", 7L, 1L)),
              list(.vinfo("V", 1L, odims), .vinfo("I", 7L, odims)),
              inits = list(kinit))
  list(name = name, model = .model(g), input = input, dims = dims)
}

# ── the four ops that moved onto Vulkan shaders ──────────────────
#
# These exist because each has a CPU kernel and a shader implementing the SAME
# arithmetic separately, and the shaders are ports whose whole value is being
# bit-identical: MaskRCNN-12-int8 agrees with ONNX Runtime exactly, and the
# quantised ops reproduce ORT's own precision losses (VPMADDUBSW pair
# saturation) on purpose, because being more accurate than the reference is
# what made them disagree.  Whole-model checks say the total still matches;
# these say WHICH side of WHICH op moved when it stops matching.
#
# Every input but the first is an initializer: the reference runner feeds
# exactly one f32 tensor (input 0) and reads the rest from the model.

# RoiAlign: X is the fed input; rois and batch_indices are baked in.
make_roialign <- function(name, dims, input, rois, batch_idx,
                          oh = 2L, ow = 2L, sampling_ratio = 2L,
                          spatial_scale = 1.0, mode = "avg") {
  num_rois <- length(rois) %/% 4L
  rinit <- .tensor("rois", c(num_rois, 4L), 1L,
                   unlist(lapply(rois, .float_bytes)))
  binit <- .tensor("bi", num_rois, 7L, .int64_bytes(batch_idx))
  n <- .node("RoiAlign", c("X", "rois", "bi"), "Y",
             attrs = list(.attr_int("output_height", oh),
                          .attr_int("output_width", ow),
                          .attr_int("sampling_ratio", sampling_ratio),
                          .attr_float("spatial_scale", spatial_scale),
                          .attr_string("mode", mode)))
  g <- .graph(list(n),
              list(.vinfo("X", 1L, dims), .vinfo("rois", 1L, c(num_rois, 4L)),
                   .vinfo("bi", 7L, num_rois)),
              list(.vinfo("Y", 1L, c(num_rois, dims[2], oh, ow))),
              inits = list(rinit, binit))
  list(name = name, model = .model(g), input = input, dims = dims)
}

# NonMaxSuppression: boxes is the fed input, everything else baked in.
#
# The output is int64 [n_selected, 3] and its LENGTH is data dependent, so the
# comparison covers both which boxes survived and how many -- exactly the part
# a tie-break or a threshold comparison gets wrong.
make_nms <- function(name, boxes, scores, num_classes = 1L,
                     max_out = 0L, iou = 0.5, score_thresh = NULL,
                     center_point_box = 0L) {
  num_boxes <- length(boxes) %/% 4L
  sinit <- .tensor("scores", c(1L, num_classes, num_boxes), 1L,
                   unlist(lapply(scores, .float_bytes)))
  minit <- .tensor("maxout", 1L, 7L, .int64_bytes(max_out))
  iinit <- .tensor("iou", 1L, 1L, .float_bytes(iou))
  ins  <- c("boxes", "scores", "maxout", "iou")
  vis  <- list(.vinfo("boxes", 1L, c(1L, num_boxes, 4L)),
               .vinfo("scores", 1L, c(1L, num_classes, num_boxes)),
               .vinfo("maxout", 7L, 1L), .vinfo("iou", 1L, 1L))
  inits <- list(sinit, minit, iinit)
  if (!is.null(score_thresh)) {
    inits <- c(inits, list(.tensor("sth", 1L, 1L, .float_bytes(score_thresh))))
    ins   <- c(ins, "sth")
    vis   <- c(vis, list(.vinfo("sth", 1L, 1L)))
  }
  attrs <- if (center_point_box != 0L)
             list(.attr_int("center_point_box", center_point_box)) else list()
  n <- .node("NonMaxSuppression", ins, "Y", attrs = attrs)
  # Selected indices: [n_selected, 3]. n_selected is only known at run time, so
  # the declared first dimension is a placeholder ORT does not enforce.
  g <- .graph(list(n), vis, list(.vinfo("Y", 7L, c(num_boxes, 3L))), inits = inits)
  list(name = name, model = .model(g), input = boxes,
       dims = c(1L, num_boxes, 4L), in_name = "boxes")
}

# QLinearMatMul: A is the fed input (as f32; the loader quantises), B and every
# scale / zero point are baked in.
make_qmatmul <- function(name, M, K, N, a_vals, b_vals,
                         a_scale = 0.02, b_scale = 0.03, y_scale = 0.05,
                         a_zp = 128L, b_zp = 0L, y_zp = 128L) {
  sc <- function(nm, v) .tensor(nm, integer(0), 1L, .float_bytes(v))
  zu <- function(nm, v) .tensor(nm, integer(0), 2L, .uint8_bytes(v))
  zi <- function(nm, v) .tensor(nm, integer(0), 3L, .int8_bytes(v))
  inits <- list(
    sc("a_sc", a_scale), zu("a_zp", a_zp),
    .tensor("B", c(K, N), 3L, .int8_bytes(b_vals)),
    sc("b_sc", b_scale), zi("b_zp", b_zp),
    sc("y_sc", y_scale), zu("y_zp", y_zp))
  n <- .node("QLinearMatMul",
             c("A", "a_sc", "a_zp", "B", "b_sc", "b_zp", "y_sc", "y_zp"), "Y")
  g <- .graph(list(n),
              list(.vinfo("A", 2L, c(M, K)),
                   .vinfo("a_sc", 1L, integer(0)), .vinfo("a_zp", 2L, integer(0)),
                   .vinfo("B", 3L, c(K, N)),
                   .vinfo("b_sc", 1L, integer(0)), .vinfo("b_zp", 3L, integer(0)),
                   .vinfo("y_sc", 1L, integer(0)), .vinfo("y_zp", 2L, integer(0))),
              list(.vinfo("Y", 2L, c(M, N))), inits = inits)
  list(name = name, model = .model(g), input = a_vals, dims = c(M, K))
}

# QLinearConv: x is the fed input, the weight and all quantisation parameters
# are baked in.
#
# Kept to 1x1 and 3x3 with group = 1 on purpose -- that is the scope of the i32
# kernel, and anything else silently takes the f32 fallback, which would make
# the case check a path it was not written for.
make_qconv <- function(name, C_in, H, W, C_out, KH, KW, x_vals, w_vals,
                       pad = 0L, stride = 1L,
                       x_scale = 0.02, w_scale = 0.01, y_scale = 0.05,
                       x_zp = 128L, w_zp = 0L, y_zp = 128L,
                       bias = NULL) {
  sc <- function(nm, v) .tensor(nm, integer(0), 1L, .float_bytes(v))
  zu <- function(nm, v) .tensor(nm, integer(0), 2L, .uint8_bytes(v))
  zi <- function(nm, v) .tensor(nm, integer(0), 3L, .int8_bytes(v))
  inits <- list(
    sc("x_sc", x_scale), zu("x_zp", x_zp),
    .tensor("W", c(C_out, C_in, KH, KW), 3L, .int8_bytes(w_vals)),
    sc("w_sc", w_scale), zi("w_zp", w_zp),
    sc("y_sc", y_scale), zu("y_zp", y_zp))
  ins <- c("x", "x_sc", "x_zp", "W", "w_sc", "w_zp", "y_sc", "y_zp")
  vis <- list(.vinfo("x", 2L, c(1L, C_in, H, W)),
              .vinfo("x_sc", 1L, integer(0)), .vinfo("x_zp", 2L, integer(0)),
              .vinfo("W", 3L, c(C_out, C_in, KH, KW)),
              .vinfo("w_sc", 1L, integer(0)), .vinfo("w_zp", 3L, integer(0)),
              .vinfo("y_sc", 1L, integer(0)), .vinfo("y_zp", 2L, integer(0)))
  if (!is.null(bias)) {
    inits <- c(inits, list(.tensor("Bi", C_out, 6L, .int32_bytes(bias))))
    ins   <- c(ins, "Bi")
    vis   <- c(vis, list(.vinfo("Bi", 6L, C_out)))
  }
  OH <- (H + 2L * pad - KH) %/% stride + 1L
  OW <- (W + 2L * pad - KW) %/% stride + 1L
  n <- .node("QLinearConv", ins, "Y",
             attrs = list(.attr_ints("kernel_shape", c(KH, KW)),
                          .attr_ints("pads", c(pad, pad, pad, pad)),
                          .attr_ints("strides", c(stride, stride)),
                          .attr_ints("dilations", c(1L, 1L)),
                          .attr_int("group", 1L)))
  g <- .graph(list(n), vis, list(.vinfo("Y", 2L, c(1L, C_out, OH, OW))),
              inits = inits)
  list(name = name, model = .model(g), input = x_vals,
       dims = c(1L, C_in, H, W))
}

make_reducesum <- function(name, dims, input, axis, out_dims) {
  ainit <- .tensor("axes", 1L, 7L, .int64_bytes(axis))
  n <- .node("ReduceSum", c("X", "axes"), "Y",
             attrs = list(.attr_int("keepdims", 1L)))
  g <- .graph(list(n),
              list(.vinfo("X", 1L, dims), .vinfo("axes", 7L, 1L)),
              list(.vinfo("Y", 1L, out_dims)),
              inits = list(ainit))
  list(name = name, model = .model(g), input = input, dims = dims)
}

# ScatterElements: data is fed, indices and updates are baked in.
#
# Shaped after the case that exposed the defect rather than a toy: MaskRCNN
# scatters [7,7,256,895] into [7,7,256,1000] along the OUTERMOST ONNX axis, so
# the interesting part is a scatter whose rows are large and whose target rows
# are scattered across a bigger destination. A 2x3 example writes one
# workgroup and passes with the wrong dst strides, the wrong axis, and a
# shader that reads only gl_GlobalInvocationID.x -- all three of which were
# live defects here.
#
# `n_rows` rows of `row` elements each land on rows given by `targets`; every
# element of a row shares that row's target, which is how ONNX expresses
# "move this row there" and what the real model does.
make_scatter <- function(name, n_rows, row, n_dst, targets, axis = 0L,
                         reduction = NULL) {
  stopifnot(length(targets) == n_rows)
  set.seed(7L)
  upd  <- as.numeric(round(rnorm(n_rows * row), 3))
  idx  <- rep(as.integer(targets), each = row)   # ONNX row-major: row varies fastest
  data <- as.numeric(rep(0, n_dst * row))

  uinit <- .tensor("updates", c(n_rows, row), 1L,
                   unlist(lapply(upd, .float_bytes)))
  iinit <- .tensor("indices", c(n_rows, row), 7L, .int64_bytes(idx))
  attrs <- list(.attr_int("axis", axis))
  if (!is.null(reduction)) attrs <- c(attrs, list(.attr_string("reduction", reduction)))
  n <- .node("ScatterElements", c("X", "indices", "updates"), "Y", attrs = attrs)
  g <- .graph(list(n),
              list(.vinfo("X", 1L, c(n_dst, row)),
                   .vinfo("indices", 7L, c(n_rows, row)),
                   .vinfo("updates", 1L, c(n_rows, row))),
              list(.vinfo("Y", 1L, c(n_dst, row))),
              inits = list(iinit, uinit))
  list(name = name, model = .model(g), input = data, dims = c(n_dst, row))
}

cases <- list(
  # ── sorting: ties decide the answer ──
  make_topk("topk_k4_ties",   c(1L, 16L), ties16,  4L),
  make_topk("topk_k8_ties",   c(1L, 16L), ties16,  8L),
  make_topk("topk_k16_all",   c(1L, 16L), ties16,  16L),
  make_topk("topk_k5_wide",   c(1L, 64L), ties64,  5L),
  make_topk("topk_k32_wide",  c(1L, 64L), ties64,  32L),
  make_topk("topk_distinct",  c(1L, 32L), plain32, 6L),

  make_case("argmax_ties",    "ArgMax", c(1L, 16L), ties16,
            attrs = list(.attr_int("axis", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(1L, 1L), out_type = 7L),
  make_case("argmin_ties",    "ArgMin", c(1L, 16L), ties16,
            attrs = list(.attr_int("axis", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(1L, 1L), out_type = 7L),
  make_case("argmax_wide",    "ArgMax", c(1L, 64L), ties64,
            attrs = list(.attr_int("axis", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(1L, 1L), out_type = 7L),

  # ── reductions: accumulation order and empty/degenerate axes ──
  # ReduceSum takes axes as an INPUT from opset 13 on, not an attribute --
  # ORT rejects the attribute form outright.  The other Reduce* ops kept the
  # attribute until opset 18, which is why only this one is built differently.
  make_reducesum("reducesum_axis1", c(4L, 8L), as.numeric(1:32), 1L, c(4L, 1L)),
  make_reducesum("reducesum_axis0", c(4L, 8L), as.numeric(1:32), 0L, c(1L, 8L)),
  make_case("reducemean_axis1", "ReduceMean", c(4L, 8L), plain32,
            attrs = list(.attr_ints("axes", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(4L, 1L)),
  make_case("reducemax_ties",  "ReduceMax", c(1L, 16L), ties16,
            attrs = list(.attr_ints("axes", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(1L, 1L)),
  make_case("reducemin_ties",  "ReduceMin", c(1L, 16L), ties16,
            attrs = list(.attr_ints("axes", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(1L, 1L)),
  make_case("reduceprod_small", "ReduceProd", c(2L, 4L), as.numeric(c(1,2,3,4,1,1,2,2)),
            attrs = list(.attr_ints("axes", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(2L, 1L)),
  make_case("reducel2_axis1",  "ReduceL2", c(4L, 8L), plain32,
            attrs = list(.attr_ints("axes", 1L), .attr_int("keepdims", 1L)),
            out_dims = c(4L, 1L)),
  # Summing many equal magnitudes is where a different accumulation order
  # shows up first, so this one is deliberately long and flat.
  make_reducesum("reducesum_long", c(1L, 1024L),
                 rep(c(1, -1, 1e-3, -1e-3), 256), 1L, c(1L, 1L)),

  make_case("cumsum_placeholder", "Softmax", c(4L, 8L), plain32,
            attrs = list(.attr_int("axis", 1L))),

  # ── RoiAlign: CPU kernel vs Vulkan shader ──
  #
  # The edge cases are the point. Ordinary interior sampling agrees almost by
  # construction; what separates two implementations is a degenerate ROI (ORT
  # forces it to 1x1 rather than an epsilon, and MaskRCNN's RPN emits 95 of
  # those out of 895), taps that fall outside the map, and the adaptive grid,
  # where the two must agree on the tap COUNT before they can agree on a value.
  make_roialign("roialign_full", c(1L, 1L, 4L, 4L), as.numeric(1:16),
                c(0, 0, 4, 4), 0L),
  make_roialign("roialign_frac", c(1L, 1L, 8L, 8L), as.numeric(1:64) / 7,
                c(0.3, 0.7, 5.9, 6.1), 0L, oh = 3L, ow = 3L),
  make_roialign("roialign_degenerate", c(1L, 1L, 4L, 4L), as.numeric(1:16),
                c(2, 2, 2, 2), 0L),
  make_roialign("roialign_outside", c(1L, 1L, 4L, 4L), as.numeric(1:16),
                c(2.5, 2.5, 7.5, 7.5), 0L),
  make_roialign("roialign_adaptive", c(1L, 1L, 8L, 8L), as.numeric(1:64) / 3,
                c(0, 0, 7.5, 7.5), 0L, sampling_ratio = 0L),
  make_roialign("roialign_max", c(1L, 1L, 8L, 8L), as.numeric(1:64) / 5,
                c(0.5, 0.5, 6.5, 6.5), 0L, mode = "max"),
  make_roialign("roialign_multi", c(1L, 2L, 4L, 4L), as.numeric(1:32),
                c(0, 0, 8, 8,  0, 0, 4, 4), c(0L, 0L),
                sampling_ratio = 1L, spatial_scale = 0.5),
  make_roialign("roialign_batch", c(2L, 1L, 4L, 4L),
                c(as.numeric(1:16), as.numeric(1:16) * -3),
                c(0, 0, 4, 4), 1L),

  # ── NonMaxSuppression: CPU kernel vs Vulkan shader ──
  #
  # Ties first: NMS is order dependent, so equal scores must break the same way
  # (lowest index first) or a different box does the suppressing. Then the
  # per-class cap, which is where "mark suppressed ahead" and "test against
  # those already kept" stop being equivalent -- 49 disagreements versus 3 when
  # this was measured on MaskRCNN node 1170.
  make_nms("nms_basic",
           c(0, 0, 1, 1,   0, 0.1, 1, 1.1,   0, -0.1, 1, 0.9,   0, 10, 1, 11),
           c(0.9, 0.75, 0.6, 0.95), iou = 0.5),
  make_nms("nms_ties",
           c(0, 0, 1, 1,   0, 0.1, 1, 1.1,   0, 0.2, 1, 1.2,   0, 10, 1, 11),
           c(0.8, 0.8, 0.8, 0.8), iou = 0.5),
  make_nms("nms_cap",
           c(0, 0, 1, 1,   0, 0.1, 1, 1.1,   0, 5, 1, 6,   0, 10, 1, 11),
           c(0.9, 0.85, 0.8, 0.7), max_out = 2L, iou = 0.5),
  make_nms("nms_score_thresh",
           c(0, 0, 1, 1,   0, 5, 1, 6,   0, 10, 1, 11,   0, 15, 1, 16),
           c(0.9, 0.4, 0.8, 0.2), iou = 0.5, score_thresh = 0.5),
  make_nms("nms_two_classes",
           c(0, 0, 1, 1,   0, 0.1, 1, 1.1,   0, 10, 1, 11,   0, 10.1, 1, 11.1),
           c(0.9, 0.8, 0.7, 0.6,   0.5, 0.95, 0.4, 0.85),
           num_classes = 2L, iou = 0.5),
  # Threshold 1.0 keeps everything, 0.0 suppresses any overlap at all: the two
  # ends of the comparison, where a `>` written as `>=` changes the answer.
  make_nms("nms_iou_all",
           c(0, 0, 1, 1,   0, 0.1, 1, 1.1,   0, 0.2, 1, 1.2),
           c(0.9, 0.8, 0.7), iou = 1.0),
  make_nms("nms_iou_none",
           c(0, 0, 1, 1,   0, 0.1, 1, 1.1,   0, 0.2, 1, 1.2),
           c(0.9, 0.8, 0.7), iou = 0.0),
  make_nms("nms_center_box",
           c(0.5, 0.5, 1, 1,   0.55, 0.55, 1, 1,   10, 10, 1, 1),
           c(0.9, 0.8, 0.7), iou = 0.5, center_point_box = 1L),

  # ── QLinearMatMul and QLinearConv are NOT here, deliberately ──
  #
  # Their first input is uint8, and this harness can only feed f32: the
  # reference runner builds one Ort::Value with CreateTensor<float> from the
  # case's .in.bin, and ORT rejects the model outright -- "Unexpected input
  # data type. Actual: (tensor(float)), expected: (tensor(uint8))". Writing the
  # cases anyway produced seven FAILs that said nothing about the shaders.
  #
  # Covering them here means teaching the runner and the R side a per-case
  # input dtype. Until then those two ops are checked end to end instead, by
  # ref_check_vs_onnxruntime.sh on the int8 models -- MaskRCNN-12-int8 exercises
  # both heavily and holds max|d| = 0, which is a strict check, just not a
  # per-op one.

  # ── ScatterElements ──
  #
  # Three sizes, because the defects this op had were all size-dependent and
  # the small case passes through every one of them:
  #   tiny  -- the contract: does it write the right rows at all
  #   wide  -- rows long enough that one row spans several workgroups
  #   big   -- > 65536 elements, so the dispatch splits over y and a shader
  #            reading only gl_GlobalInvocationID.x stops covering the input
  make_scatter("scatter_tiny",  4L,   3L,  8L, c(1, 5, 2, 7)),
  make_scatter("scatter_wide",  6L, 512L, 16L, c(0, 15, 3, 9, 1, 12)),
  make_scatter("scatter_big",  40L, 4096L, 64L,
               c(seq(0, 39) * 3 %% 64)),
  NULL
)
cases <- Filter(Negate(is.null), cases)

if (!is.na(FILTER) && nzchar(FILTER))
  cases <- Filter(function(c) grepl(FILTER, c$name, fixed = TRUE), cases)

# ── run each case through ggmlR, on both backends ────────────────
manifest <- character(0)
cat(sprintf("Running %d op cases through ggmlR\n\n", length(cases)))

for (cs in cases) {
  path <- file.path(DATA_DIR, paste0(cs$name, ".onnx"))
  writeBin(cs$model, path)
  writeBin(as.numeric(cs$input), file.path(DATA_DIR, paste0(cs$name, ".in.bin")),
           size = 4, endian = "little")
  manifest <- c(manifest, sprintf("%s\t%s", cs$name, paste(cs$dims, collapse = ",")))

  # Most cases feed a tensor called "X"; ops whose first input has a name of
  # its own (NonMaxSuppression takes "boxes") say so in the case.
  in_name <- if (is.null(cs$in_name)) "X" else cs$in_name

  for (dev in c("cpu", "gpu")) {
    if (dev == "gpu" && !ggml_vulkan_available()) next
    tag <- sprintf("%-22s %s", cs$name, dev)
    res <- tryCatch({
      m <- onnx_load(path, device = dev,
                     input_shapes = setNames(list(as.integer(cs$dims)), in_name))
      out <- onnx_run(m, setNames(list(array(cs$input, dim = rev(cs$dims))), in_name))
      # Every output, matching the reference runner.  For the sorting ops the
      # second one carries the indices, which is the half a tie convention
      # shows up in -- the values can agree while the ranking does not.
      for (k in seq_along(out)) {
        suffix <- if (k == 1) "" else as.character(k - 1L)
        writeBin(as.numeric(out[[k]]),
                 file.path(DATA_DIR, sprintf("%s.%s%s.bin", cs$name, dev, suffix)),
                 size = 4, endian = "little")
      }
      v <- as.numeric(out[[1]])
      cat(sprintf("%s  OK  n=%d  head=%s\n", tag, length(v),
                  paste(format(head(v, 4), digits = 6), collapse = " ")))
      TRUE
    }, error = function(e) {
      cat(sprintf("%s  FAIL %s\n", tag, conditionMessage(e)))
      FALSE
    })
  }
}

writeLines(manifest, file.path(DATA_DIR, "manifest.tsv"))
cat(sprintf("\nwritten to %s\n", DATA_DIR))
