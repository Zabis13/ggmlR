#!/usr/bin/env Rscript
# Compare per-op dumps: ggmlR (cpu and vulkan) against ONNX Runtime.
#
# Both backends are compared to the reference rather than to each other,
# because agreeing with each other is not the property under test: ggml_top_k
# swapped its first two results on both, by design, and only an outside
# implementation showed it.

args <- commandArgs(trailingOnly = TRUE)
DATA_DIR <- if (length(args) > 0) args[1] else "/tmp/ggmlR-ref/data/ops"
TOL <- 1e-3

rd <- function(p) if (file.exists(p)) readBin(p, "double", n = 1e7, size = 4) else NULL

man <- file.path(DATA_DIR, "manifest.tsv")
if (!file.exists(man)) stop("no manifest in ", DATA_DIR, " -- run the generator first")
rows <- strsplit(readLines(man), "\t")

cat(sprintf("Comparing against ONNX Runtime, tolerance %g\n\n", TOL))

agree <- 0L; disagree <- 0L; skipped <- 0L

for (r in rows) {
  name <- r[1]
  # Output 0 is the values; output 1, where present, the indices.  They are
  # reported separately: for a sort the values can agree while the ranking
  # does not, and it is the ranking that decides what downstream keeps.
  for (oi in c(0L, 1L)) {
    sfx_ort <- if (oi == 0L) ".ort.bin" else sprintf(".ort%d.bin", oi)
    ref <- rd(file.path(DATA_DIR, paste0(name, sfx_ort)))
    if (is.null(ref)) next
    label <- if (oi == 0L) name else paste0(name, " [idx]")

    for (dev in c("cpu", "gpu")) {
      sfx <- if (oi == 0L) sprintf(".%s.bin", dev) else sprintf(".%s%d.bin", dev, oi)
      got <- rd(file.path(DATA_DIR, paste0(name, sfx)))
      tag <- sprintf("%-26s %-3s", label, dev)
      if (is.null(got)) { cat(sprintf("%s SKIP (no dump)\n", tag)); skipped <- skipped + 1L; next }
      if (length(got) == 0L) {
        # An empty dump means the graph never built -- an op ggmlR does not
        # implement.  That is a gap, not a wrong answer, and counting it as a
        # disagreement would bury the cases where the arithmetic itself is off.
        cat(sprintf("%s MISSING (op not implemented)\n", tag))
        skipped <- skipped + 1L
        next
      }
      if (length(got) != length(ref)) {
        cat(sprintf("%s FAIL length %d vs %d\n", tag, length(got), length(ref)))
        disagree <- disagree + 1L
        next
      }
      d <- max(abs(got - ref))
      if (is.finite(d) && d <= TOL) {
        cat(sprintf("%s OK    max|d|=%.3g\n", tag, d))
        agree <- agree + 1L
      } else {
        cat(sprintf("%s FAIL  max|d|=%.3g\n", tag, d))
        n <- min(8L, length(ref))
        cat(sprintf("      ggmlR: %s\n", paste(format(got[1:n], digits = 6), collapse = " ")))
        cat(sprintf("      ort  : %s\n", paste(format(ref[1:n], digits = 6), collapse = " ")))
        disagree <- disagree + 1L
      }
    }
  }
}

cat(sprintf("\n--- %d agree, %d disagree, %d skipped ---\n", agree, disagree, skipped))
if (disagree > 0) quit(status = 1)
