#!/usr/bin/env Rscript
# Add internal edges to a model's graph.output, so a reference run can be
# asked to return them.
#
# ONNX Runtime returns what the graph declares as an output and nothing else.
# Comparing an intermediate value against it therefore needs the model itself
# changed -- which is what this does, on a copy: append one ValueInfoProto per
# requested edge to the graph's output list, and fix the two lengths above it.
#
# The whole file is NOT reparsed.  A detector is tens of megabytes, almost all
# of it initialiser bytes, and rebuilding that to add a few hundred bytes at a
# known offset is work with nothing to show for it.  The graph is one
# length-delimited field of the model; its outputs are repeated field 12 inside
# it.  A repeated field may appear anywhere among its siblings and readers must
# accept it, so the new entries go at the END of the graph payload, and only
# the graph's own length varint has to change.
#
# The element type must be stated and must be a real one: ORT rejects the
# model outright with "Invalid tensor data type 0" if the ValueInfoProto says
# UNDEFINED, so leaving it for inference is not an option.  The SHAPE is left
# out, which ORT does accept -- and it has to be, because the interesting edges
# are exactly the ones whose extent is data-dependent.
#
# Type therefore comes from the caller, as a suffix on the name:
#   2516        float   (the default)
#   2525:i64    int64   -- indices
#   1170:i32    int32
# Getting it wrong is loud, not silent: ORT fails the run rather than
# reinterpreting the bytes.
#
# Usage:
#   Rscript inst/scripts/ref_patch_outputs.R <in.onnx> <out.onnx> <name>[:<type>][,...]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("usage: ref_patch_outputs.R <in.onnx> <out.onnx> <name>[,<name>...]\n")
  quit(status = 1)
}
IN <- args[1]; OUT <- args[2]
SPEC  <- trimws(strsplit(args[3], ",", fixed = TRUE)[[1]])
SPEC  <- SPEC[nzchar(SPEC)]

if (!file.exists(IN)) { cat("missing:", IN, "\n"); quit(status = 1) }

# ── protobuf primitives ──────────────────────────────────────────
# Varints are built and read a byte at a time on purpose: R has no 64-bit
# integer, and bitwAnd/bitwShiftR are 32-bit, so anything above 2^31 (a 45 MB
# length is not, but an offset arithmetic slip is easy) must go through
# doubles with %% and %/%.
.varint <- function(value) {
  value <- as.numeric(value)
  out <- raw(0)
  repeat {
    b <- value %% 128; value <- value %/% 128
    if (value > 0) b <- b + 128
    out <- c(out, as.raw(b))
    if (value == 0) break
  }
  out
}
.read_varint <- function(buf, i) {
  v <- 0; s <- 0
  repeat {
    x <- as.integer(buf[i]); i <- i + 1L
    v <- v + bitwAnd(x, 127L) * 2^s
    s <- s + 7
    if (x < 128L) break
  }
  list(v = v, i = i)
}
.tag        <- function(f, w) .varint(bitwShiftL(f, 3) + w)
.bytes      <- function(f, d) c(.tag(f, 2L), .varint(length(d)), d)
.string     <- function(f, s) .bytes(f, charToRaw(s))
.varint_fld <- function(f, v) c(.tag(f, 0L), .varint(v))

# TensorProto.DataType, the subset these dumps need.
.ELEM <- c(f32 = 1L, i32 = 6L, i64 = 7L, u8 = 2L, i8 = 3L,
           f16 = 10L, bool = 9L, u16 = 4L, i16 = 5L, f64 = 11L)

# ValueInfoProto: name (1), type (2).  The type is a TypeProto holding a
# Tensor (1) with elem_type (1) set and shape (2) omitted -- an absent shape
# is legal and means "unknown", which is the honest answer for an edge whose
# length depends on the data.
.value_info <- function(name, elem) {
  ttensor <- .varint_fld(1L, elem)
  ttype   <- .bytes(1L, ttensor)
  c(.string(1L, name), .bytes(2L, ttype))
}

# ── locate the graph field (7) at the top level ──────────────────
size <- file.info(IN)$size
con <- file(IN, "rb")
head_buf <- readBin(con, "raw", n = 4096)

i <- 1L
g_len_at <- NA_integer_   # offset of the graph's length varint
g_payload_at <- NA_integer_
g_len <- NA_real_
repeat {
  if (i > length(head_buf) - 16L) break
  t <- .read_varint(head_buf, i); tag <- t$v; i <- t$i
  f <- bitwShiftR(tag, 3L); w <- bitwAnd(tag, 7L)
  if (w == 2L) {
    len_at <- i
    L <- .read_varint(head_buf, i); i <- L$i
    if (f == 7L) { g_len_at <- len_at; g_len <- L$v; g_payload_at <- i; break }
    i <- i + L$v
  } else if (w == 0L) {
    V <- .read_varint(head_buf, i); i <- V$i
  } else if (w == 5L) { i <- i + 4L
  } else if (w == 1L) { i <- i + 8L
  } else break
}
if (is.na(g_len_at)) { close(con); cat("no graph field (7) found\n"); quit(status = 1) }

g_end <- g_payload_at + g_len            # first byte after the graph payload
cat(sprintf("graph: payload %.0f..%.0f (len %.0f), %.0f trailing byte(s)\n",
            g_payload_at, g_end - 1, g_len, size - (g_end - 1)))

# ── read the pieces around the splice point ──────────────────────
# Read back from the start: the prefix before the graph's length varint, the
# old length, then payload and tail.  Only the length varint is rewritten.
invisible(seek(con, 0))
prefix  <- readBin(con, "raw", n = g_len_at - 1L)
old_len <- readBin(con, "raw", n = g_payload_at - g_len_at)
close(con)

add <- raw(0)
for (s in SPEC) {
  parts <- strsplit(s, ":", fixed = TRUE)[[1]]
  nm    <- parts[1]
  tnm   <- if (length(parts) > 1) parts[2] else "f32"
  if (!tnm %in% names(.ELEM))
    stop("unknown type '", tnm, "' for '", nm, "'; one of: ",
         paste(names(.ELEM), collapse = ", "))
  add <- c(add, .bytes(12L, .value_info(nm, .ELEM[[tnm]])))
  cat(sprintf("  + %-12s %s\n", nm, tnm))
}
cat(sprintf("appending %d output(s), %d byte(s)\n", length(SPEC), length(add)))

new_len <- .varint(g_len + length(add))

# ── write: prefix, new length, payload, new outputs, tail ────────
# Streamed in chunks rather than held in memory: the payload is the whole
# model, and reading 45 MB into an R vector of raw doubles the footprint for
# no reason.
inc <- file(IN, "rb"); outc <- file(OUT, "wb")
invisible(readBin(inc, "raw", n = g_payload_at - 1L))   # skip prefix+old length
writeBin(prefix, outc)
writeBin(new_len, outc)

remaining <- g_len
CHUNK <- 8L * 1024L * 1024L
while (remaining > 0) {
  n <- min(CHUNK, remaining)
  buf <- readBin(inc, "raw", n = n)
  if (length(buf) == 0) break
  writeBin(buf, outc)
  remaining <- remaining - length(buf)
}
writeBin(add, outc)                                      # the new outputs

repeat {                                                 # the model-level tail
  buf <- readBin(inc, "raw", n = CHUNK)
  if (length(buf) == 0) break
  writeBin(buf, outc)
}
close(inc); close(outc)

new_size <- file.info(OUT)$size
cat(sprintf("wrote %s: %.0f -> %.0f bytes (+%.0f, expected +%d)\n",
            OUT, size, new_size, new_size - size, length(add)))
if (new_size - size != length(add) + (length(new_len) - length(old_len)))
  cat("WARNING: size delta does not match; the length varint may have grown\n")
