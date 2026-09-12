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
            attrs = list(.attr_int("axis", 1L)))
)

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

  for (dev in c("cpu", "gpu")) {
    if (dev == "gpu" && !ggml_vulkan_available()) next
    tag <- sprintf("%-22s %s", cs$name, dev)
    res <- tryCatch({
      m <- onnx_load(path, device = dev,
                     input_shapes = setNames(list(as.integer(cs$dims)), "X"))
      out <- onnx_run(m, setNames(list(array(cs$input, dim = rev(cs$dims))), "X"))
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
