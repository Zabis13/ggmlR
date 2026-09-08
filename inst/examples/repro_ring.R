#!/usr/bin/env Rscript
# Repro for the roberta failure, with the two input patterns side by side.
#
# test_all_onnx.R feeds rep(1, 128) -- every token the same id -- and reports
# only the output length, so it passes whatever the values are.  This script
# prints the values, and also feeds ids with a real spread, which is what the
# model sees in use.
library(ggmlR)

path <- "/mnt/Data2/DS_projects/ONNX models-main/roberta-sequence-classification-9.onnx"
stopifnot(file.exists(path))

model <- onnx_load(path, device = "cpu", input_shapes = list(input = c(1L, 128L)))

show <- function(label, inp) {
  out <- onnx_run(model, list(input = inp))
  v <- out[[1]]
  cat(sprintf("%-22s len=%d  vals=%s  finite=%d/%d\n",
              label, length(v), paste(format(head(v, 4)), collapse=" "),
              sum(is.finite(v)), length(v)))
  invisible(v)
}

# The exact input test_all_onnx.R uses.
a1 <- show("rep(1) run 1", rep(1, 128))
a2 <- show("rep(1) run 2", rep(1, 128))

# Token ids with a spread, as a real sequence would have.
set.seed(42)
b1 <- show("sample run 1", sample.int(1000, 128, replace = TRUE))
set.seed(7)
b2 <- show("sample run 2", sample.int(1000, 128, replace = TRUE))

# Same input twice must give the same answer.
cat("rep(1) runs identical:", identical(a1, a2), "\n")

cat("=== gc ===\n"); gc(); cat("gc survived\n")
