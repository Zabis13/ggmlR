#!/usr/bin/env Rscript
# Rewrite a model's graph outputs to name an intermediate tensor, producing a
# truncated copy that computes only the prefix up to that point.
#
# Both implementations then run the same prefix and can be compared with the
# existing machinery.  This is how a disagreement gets located by depth: if it
# grows steadily the cause repeats in every block, and if it appears at one cut
# and stays flat afterwards a bisection finds the node in a few runs.
#
# Only the outputs are touched.  The nodes past the cut stay in the file and
# are simply never reached, which keeps the edit small and reversible -- and
# avoids having to work out which initializers would become unused.
#
# The output's type and shape are declared as an undefined-rank tensor of the
# element type given, so no shape has to be predicted here; ONNX Runtime infers
# what it needs, and ggmlR reads the shape off its own graph.
#
# Usage: Rscript inst/scripts/ref_truncate_onnx.R <in.onnx> <out.onnx> <tensor-name>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: truncate_onnx.R <in.onnx> <out.onnx> <tensor>")
IN <- args[1]; OUT <- args[2]; CUT <- args[3]

source("tests/testthat/helper-onnx.R")

buf <- readBin(IN, "raw", n = file.info(IN)$size)

# ── Minimal protobuf reader, enough to walk top-level fields ─────────
rd_varint <- function(pos) {
  v <- 0; shift <- 1
  repeat {
    b <- as.integer(buf[pos]); pos <- pos + 1
    v <- v + bitwAnd(b, 0x7F) * shift
    if (b < 128) break
    shift <- shift * 128
  }
  list(value = v, pos = pos)
}

# Walk one message, recording for each field both its payload range and the
# full span including its tag.  Keeping `span` means a field can be copied
# verbatim without reconstructing how many bytes its tag occupied -- deriving
# that afterwards is easy to get wrong and fails silently.
scan_fields <- function(from, to) {
  out <- list(); pos <- from
  while (pos <= to) {
    field_start <- pos
    t <- rd_varint(pos); tag <- t$value; pos <- t$pos
    field <- floor(tag / 8); wire <- tag %% 8
    start <- pos
    if (wire == 2) {
      l <- rd_varint(pos); pos <- l$pos
      body <- c(pos, pos + l$value - 1)
      pos <- pos + l$value
    } else if (wire == 0) {
      l <- rd_varint(pos); pos <- l$pos; body <- c(start, pos - 1)
    } else if (wire == 5) { body <- c(pos, pos + 3); pos <- pos + 4
    } else if (wire == 1) { body <- c(pos, pos + 7); pos <- pos + 8
    } else stop("unsupported wire type ", wire)
    out[[length(out) + 1]] <- list(field = field, wire = wire,
                                   body = body,
                                   span = c(field_start, pos - 1))
  }
  out
}

MP_GRAPH <- 7; GP_OUTPUT <- 12

top <- scan_fields(1, length(buf))
gi  <- Filter(function(f) f$field == MP_GRAPH, top)
if (length(gi) != 1) stop("expected exactly one graph field, found ", length(gi))
graph <- gi[[1]]

gfields <- scan_fields(graph$body[1], graph$body[2])

# A ValueInfoProto for the cut tensor: name (1) plus a type (2) holding a
# tensor_type (1) with only elem_type (1) set -- rank left undefined, so
# nothing has to be predicted about the shape here.
# Element type of the cut tensor: float unless told otherwise.  Shape-carrying
# tensors inside a graph are int64, and declaring one as float makes ONNX
# Runtime reject the whole model -- usefully, since it names the mismatch.
elem_float <- as.integer(Sys.getenv("CUT_ELEM_TYPE", "1"))
vi <- .pb_bytes(1L, charToRaw(CUT))
tt <- .pb_bytes(1L, .pb_varint_field(1L, elem_float))   # tensor_type{elem_type}
vi <- c(vi, .pb_bytes(2L, tt))                          # type{...}
new_output <- .pb_bytes(GP_OUTPUT, vi)

# Rebuild the graph, dropping every existing output and appending the new one.
keep <- raw(0)
for (f in gfields) {
  if (f$field == GP_OUTPUT) next
  keep <- c(keep, buf[f$span[1]:f$span[2]])
}
new_graph_body <- c(keep, new_output)

# Rebuild the model: everything outside the graph field, with the graph replaced.
before <- if (graph$span[1] > 1) buf[1:(graph$span[1] - 1)] else raw(0)
after  <- if (graph$span[2] < length(buf)) buf[(graph$span[2] + 1):length(buf)] else raw(0)

out_buf <- c(before, .pb_bytes(MP_GRAPH, new_graph_body), after)
writeBin(out_buf, OUT)
cat(sprintf("%s -> %s\n  cut at: %s\n  %d -> %d bytes\n",
            basename(IN), basename(OUT), CUT, length(buf), length(out_buf)))
