#!/usr/bin/env Rscript
# Isolate one operator and check it against ONNX Runtime.
#
# botnet is the only model still disagreeing whose sole unique operator is
# BatchNormalization -- present in it and xcit, in none of the eleven models
# that agree.  A single-node graph settles whether that operator is the defect,
# without the noise of the 200 others botnet also runs.
#
# Writes the same flat-float32 files the main comparison uses, so
# ort_reference and compare.R handle it with no changes.

suppressMessages(library(ggmlR))
source("tests/testthat/helper-onnx.R")

OUT <- "inst/scripts/ref_data"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# [N,C,H,W] with C the channel axis BN normalises over.
N <- 1L; C <- 4L; H <- 3L; W <- 5L
f32 <- function(v) writeBin(as.numeric(v), raw(), size = 4, endian = "little")

set.seed(42)
x     <- runif(N*C*H*W, -2, 2)
scale <- runif(C, 0.5, 1.5)
bias  <- runif(C, -0.5, 0.5)
mean_ <- runif(C, -1, 1)
# Deliberately spread over orders of magnitude: a var handled at the wrong
# axis, or an epsilon added to the wrong operand, shows up far more clearly
# when the channels do not all divide by roughly the same number.
var_  <- c(0.01, 0.5, 2.0, 10.0)

nodes <- list(.onnx_node("BatchNormalization",
                         c("X","scale","bias","mean","var"), "Y",
                         attrs = list(.onnx_attr_float("epsilon", 1e-5))))
graph <- .onnx_graph("bn_probe", nodes,
  list(.onnx_value_info("X", 1L, c(N,C,H,W))),
  list(.onnx_value_info("Y", 1L, c(N,C,H,W))),
  list(.onnx_tensor("scale", C, 1L, f32(scale)),
       .onnx_tensor("bias",  C, 1L, f32(bias)),
       .onnx_tensor("mean",  C, 1L, f32(mean_)),
       .onnx_tensor("var",   C, 1L, f32(var_))))

path <- file.path(OUT, "bn_probe.onnx")
writeBin(.onnx_model(graph), path)

m   <- onnx_load(path, device = "cpu", input_shapes = list(X = c(N,C,H,W)))
out <- onnx_run(m, list(X = x))[[1]]

# What BN is defined to compute, done here in R as a third opinion: if ggmlR
# and ONNX Runtime disagree, this says which of them is right.
expected <- as.numeric(vapply(seq_len(C), function(ch) {
  xs <- x[((ch-1)*H*W + 1):(ch*H*W)]
  (xs - mean_[ch]) / sqrt(var_[ch] + 1e-5) * scale[ch] + bias[ch]
}, numeric(H*W)))

cat(sprintf("ggmlR   head: %s\n", paste(format(head(out, 5), digits = 6), collapse = " ")))
cat(sprintf("formula head: %s\n", paste(format(head(expected, 5), digits = 6), collapse = " ")))
cat(sprintf("max|ggmlR - formula| = %.3g\n", max(abs(out - expected))))

writeBin(as.numeric(x),   file.path(OUT, "bn_probe.in.X.bin"), size = 4, endian = "little")
writeBin(as.numeric(out), file.path(OUT, "bn_probe.ggmlr.bin"), size = 4, endian = "little")
writeLines(c(sprintf("in\tbn_probe\tbn_probe.onnx\tX\t%d,%d,%d,%d\t0", N, C, H, W),
             sprintf("out\tbn_probe\tbn_probe.onnx\t%d", length(out))),
           file.path(OUT, "manifest_bn.tsv"))
cat("wrote", OUT, "\n")

# ── Split, the other operator botnet has and the agreeing models mostly do not.
#
# BoTNet splits a [1,768,16,16] qkv tensor into three [1,256,16,16] along the
# channel axis.  Split appears in exactly one agreeing model, so it is only
# half-exonerated by that: the axis and the sizes matter, and this reproduces
# the ones botnet actually uses.
cat("\n--- Split probe ---\n")
Cq <- 12L; Hs <- 2L; Ws <- 2L      # 12 channels -> 3 x 4, same shape of split
set.seed(7)
xs <- runif(Cq*Hs*Ws, -2, 2)

sp_nodes <- list(
  .onnx_node("Split", c("X","split"), c("A","B","Cc"),
             attrs = list(.onnx_attr_int("axis", 1L))),
  # Concat them back in a different order, so a wrong split is visible in the
  # output rather than cancelling out.
  .onnx_node("Concat", c("Cc","A","B"), "Y",
             attrs = list(.onnx_attr_int("axis", 1L))))
sp_graph <- .onnx_graph("split_probe", sp_nodes,
  list(.onnx_value_info("X", 1L, c(1L,Cq,Hs,Ws))),
  list(.onnx_value_info("Y", 1L, c(1L,Cq,Hs,Ws))),
  list(.onnx_tensor("split", 3L, 7L,
                    writeBin(as.integer(c(4,4,4)), raw(), size = 8, endian = "little"))))
sp_path <- file.path(OUT, "split_probe.onnx")
writeBin(.onnx_model(sp_graph), sp_path)

sp_out <- tryCatch({
  ms <- onnx_load(sp_path, device = "cpu", input_shapes = list(X = c(1L,Cq,Hs,Ws)))
  onnx_run(ms, list(X = xs))[[1]]
}, error = function(e) { cat("FAIL:", conditionMessage(e), "\n"); NULL })

if (!is.null(sp_out)) {
  per <- Hs*Ws*4L   # elements per split piece
  sp_expected <- c(xs[(2*per+1):(3*per)], xs[1:per], xs[(per+1):(2*per)])
  cat(sprintf("max|ggmlR - expected| = %.3g\n", max(abs(sp_out - sp_expected))))
  cat(sprintf("  ggmlR    head: %s\n", paste(format(head(sp_out, 6), digits = 5), collapse = " ")))
  cat(sprintf("  expected head: %s\n", paste(format(head(sp_expected, 6), digits = 5), collapse = " ")))
}
