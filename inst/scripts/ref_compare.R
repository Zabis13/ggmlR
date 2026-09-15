#!/usr/bin/env Rscript
# Compare ggmlR's outputs against the ONNX Runtime reference, model by model.
#
# Tolerance is absolute.  F32 arithmetic accumulated in a different order
# drifts by roughly 1e-5..1e-4 over a deep network, so equality is the wrong
# test; a difference past the tolerance is a disagreement about the arithmetic
# rather than rounding.
#
# Usage: Rscript inst/scripts/ref_compare.R [datadir] [tolerance]

args <- commandArgs(trailingOnly = TRUE)
DIR  <- if (length(args) >= 1 && nzchar(args[1])) args[1] else "inst/scripts/ref_data"
TOL  <- if (length(args) >= 2) as.numeric(args[2]) else 1e-3

read_f32 <- function(path) {
  n <- file.info(path)$size / 4
  readBin(path, "numeric", n = n, size = 4, endian = "little")
}

mf <- file.path(DIR, "manifest.tsv")
if (!file.exists(mf)) stop("no manifest in ", DIR, " — run dump_io.R first")
tags <- unique(sub("^(in|out)\t([^\t]+)\t.*$", "\\2", readLines(mf)))

cat(sprintf("Comparing against ONNX Runtime, tolerance %g\n\n", TOL))
n_ok <- 0L; n_bad <- 0L; n_skip <- 0L

for (tag in tags) {
  # A tag of the form <model>@<device> is one backend's dump of <model>. The
  # reference runner only ever produces <model>.ort.bin -- it knows nothing
  # about our backends -- so the comparison strips the suffix to find it, and
  # every backend is then read against the same reference.
  base   <- sub("@.*$", "", tag)
  f_ours <- file.path(DIR, paste0(tag,  ".ggmlr.bin"))
  f_ref  <- file.path(DIR, paste0(base, ".ort.bin"))
  cat(sprintf("%-45s ", tag))

  if (!file.exists(f_ours) || !file.exists(f_ref)) {
    cat(sprintf("SKIP (no %s output)\n",
                if (!file.exists(f_ours)) "ggmlR" else "reference"))
    n_skip <- n_skip + 1L
    next
  }

  ours <- read_f32(f_ours); ref <- read_f32(f_ref)
  if (length(ours) != length(ref)) {
    cat(sprintf("FAIL length %d vs %d\n", length(ours), length(ref)))
    n_bad <- n_bad + 1L
    next
  }

  d <- max(abs(ours - ref))
  # A relative figure as well: 1e-3 absolute means something different on
  # logits near 1 than on a super-resolution image scaled to 255.
  rel <- d / max(1e-12, max(abs(ref)))
  if (is.finite(d) && d < TOL) {
    cat(sprintf("OK    max|d|=%.3g (rel %.2g)\n", d, rel))
    n_ok <- n_ok + 1L
  } else {
    cat(sprintf("FAIL  max|d|=%.3g (rel %.2g)\n", d, rel))
    cat(sprintf("      ggmlR head: %s\n",
                paste(format(head(ours, 4), digits = 6), collapse = " ")))
    cat(sprintf("      ort   head: %s\n",
                paste(format(head(ref,  4), digits = 6), collapse = " ")))
    n_bad <- n_bad + 1L
  }
}

cat(sprintf("\n--- %d agree, %d disagree, %d skipped ---\n", n_ok, n_bad, n_skip))
quit(status = if (n_bad == 0L) 0L else 1L)
