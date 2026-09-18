#!/usr/bin/env Rscript
# Element type of named edges, read from the model's own value_info.
#
# ref_patch_outputs.R needs the type of every edge it exposes, and getting it
# wrong is not a small error: ORT rejects the patched model outright ("Invalid
# tensor data type 0"), so the whole reference run is lost. Guessing from the
# ggmlR trace does not work either -- ggmlR casts indices to f32, so an i64
# edge looks like f32 there. The model is the only authority.
#
# Read the bytes rather than a protobuf library, like the rest of the ONNX
# tooling here:
#   GraphProto:       input = 11, output = 12, value_info = 13
#   ValueInfoProto:   name = 1, type = 2
#   TypeProto:        tensor_type = 1
#   TypeProto.Tensor: elem_type = 1 (varint)
#
# ⚠️ An edge missing from value_info prints "NOT in value_info" rather than a
#    type: quantized edges (*_quantized) usually are absent and their type has
#    to be supplied by hand (u8 in practice). A folded edge is absent too.
#
# Usage:
#   Rscript inst/scripts/ref_edge_types.R <model.onnx> <name>[,<name>...]
#
# Example (MaskRCNN-12-int8, the NMS output region):
#   Rscript inst/scripts/ref_edge_types.R "$ONNX_DIR/MaskRCNN-12-int8.onnx" \
#           2163,2169,2171,2172,2177
#   -> 2163 f32 / 2169 i64 / 2171 i64 / 2172 i64 / 2177 i64
args <- commandArgs(trailingOnly = TRUE)
MODEL <- args[1]; WANT <- strsplit(args[2], ",")[[1]]
raw <- readBin(MODEL, "raw", file.info(MODEL)$size)

gv <- function(p) { s <- 0L; sh <- 0L; repeat { b <- as.integer(raw[p]); p <- p + 1L
  s <- s + bitwAnd(b, 127L) * 2^sh; sh <- sh + 7L; if (b < 128L) break }; list(v = s, p = p) }

# ModelProto.graph = field 7
p <- 1L; gstart <- NA; gend <- NA
while (p <= length(raw)) {
  k <- gv(p); key <- k$v; p <- k$p
  fld <- bitwShiftR(key, 3L); wt <- bitwAnd(key, 7L)
  if (wt == 2L) { L <- gv(p); p <- L$p
    if (fld == 7L) { gstart <- p; gend <- p + L$v - 1L; break }
    p <- p + L$v
  } else if (wt == 0L) { p <- gv(p)$p } else if (wt == 5L) { p <- p + 4L
  } else if (wt == 1L) { p <- p + 8L } else break
}
stopifnot(!is.na(gstart))

TYPES <- c("1"="f32","2"="u8","3"="i8","4"="u16","5"="i16","6"="i32","7"="i64",
           "9"="bool","10"="f16","11"="f64","12"="u32","13"="u64")
found <- list()
p <- gstart
while (p <= gend) {
  k <- gv(p); key <- k$v; p <- k$p
  fld <- bitwShiftR(key, 3L); wt <- bitwAnd(key, 7L)
  if (wt == 2L) { L <- gv(p); p <- L$p; vs <- p; ve <- p + L$v - 1L
    if (fld %in% c(11L, 12L, 13L)) {
      q <- vs; nm <- NA; et <- NA
      while (q <= ve) { k2 <- gv(q); ky <- k2$v; q <- k2$p
        f2 <- bitwShiftR(ky, 3L); w2 <- bitwAnd(ky, 7L)
        if (w2 == 2L) { L2 <- gv(q); q <- L2$p
          if (f2 == 1L) nm <- rawToChar(raw[q:(q + L2$v - 1L)])
          if (f2 == 2L) { # TypeProto
            r <- q; re <- q + L2$v - 1L
            while (r <= re) { k3 <- gv(r); k3v <- k3$v; r <- k3$p
              f3 <- bitwShiftR(k3v,3L); w3 <- bitwAnd(k3v,7L)
              if (w3 == 2L) { L3 <- gv(r); r <- L3$p
                if (f3 == 1L) { s <- r; se <- r + L3$v - 1L
                  while (s <= se) { k4 <- gv(s); k4v <- k4$v; s <- k4$p
                    f4 <- bitwShiftR(k4v,3L); w4 <- bitwAnd(k4v,7L)
                    if (f4 == 1L && w4 == 0L) { e <- gv(s); et <- e$v; s <- e$p }
                    else if (w4 == 2L) { L4 <- gv(s); s <- L4$p + L4$v }
                    else if (w4 == 0L) { s <- gv(s)$p } else break } }
                r <- r + L3$v
              } else if (w3 == 0L) { r <- gv(r)$p } else break } }
          q <- q + L2$v
        } else if (w2 == 0L) { q <- gv(q)$p } else break }
      if (!is.na(nm) && nm %in% WANT) found[[nm]] <- if (!is.na(et)) TYPES[[as.character(et)]] else "?"
    }
    p <- ve + 1L
  } else if (wt == 0L) { p <- gv(p)$p } else if (wt == 5L) { p <- p + 4L
  } else if (wt == 1L) { p <- p + 8L } else break
}
for (w in WANT) cat(sprintf("%-12s %s\n", w, if (!is.null(found[[w]])) found[[w]] else "NOT in value_info"))
