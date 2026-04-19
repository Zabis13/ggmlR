#!/usr/bin/env Rscript
# Debug: Cast F32->I32->F32 truncation failure
# Expected: truncation toward zero (1.9->1, -2.7->-2, 0.5->0, 3.1->3)
# Actual:   values pass through unchanged

library(ggmlR)
source("tests/testthat/helper-onnx.R")

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

x <- c(1.9, -2.7, 0.5, 3.1)

cat("=== Test 1: F32 → I32 only (expect int bytes 1,-2,0,3) ===\n")
{
  inp  <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 6L, c(4L))   # ONNX type 6 = INT32
  node <- .onnx_node("Cast", "X", "Y", attrs = list(.onnx_attr_int("to", 6L)))
  g    <- .onnx_graph("t1", list(node), list(inp), list(outp))
  p    <- tempfile(fileext = ".onnx"); writeBin(.onnx_model(g), p)
  r    <- run_onnx(p, list(X = x))
  # r comes back as raw bytes; reinterpret as int32
  bytes <- writeBin(as.numeric(r), raw(), size = 4L)
  ints  <- readBin(bytes, integer(), n = length(x), size = 4L)
  cat("  raw float-repr:", as.numeric(r), "\n")
  cat("  as int32:      ", ints, "\n")
  cat("  expected int32:", as.integer(x), "\n")
  cat("  PASS:", identical(ints, as.integer(x)), "\n\n")
}

cat("=== Test 2: I32 → F32 only (feed int32 bytes, expect float 1,-2,0,3) ===\n")
{
  # Build an I32 initializer holding truncated values, then cast to F32
  # We simulate this via a Reshape(Cast) trick: use a Constant node of type INT32
  # Simpler: feed the model x already truncated via MAP_CUSTOM — not available here.
  # Instead just test the CPY path directly: is ggml_cast(I32->F32) numeric or reinterpret?
  # We build: Cast(X:F32 -> tmp:I32) -> Cast(tmp:I32 -> Y:F32) and check intermediate.
  inp  <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  n1   <- .onnx_node("Cast", "X", "i32", attrs = list(.onnx_attr_int("to", 6L)))
  n2   <- .onnx_node("Cast", "i32", "Y", attrs = list(.onnx_attr_int("to", 1L)))
  g    <- .onnx_graph("t2", list(n1, n2), list(inp), list(outp))
  p    <- tempfile(fileext = ".onnx"); writeBin(.onnx_model(g), p)
  r    <- as.numeric(run_onnx(p, list(X = x)))
  cat("  result:  ", r, "\n")
  cat("  expected:", as.numeric(as.integer(x)), "\n")
  cat("  PASS:", isTRUE(all.equal(r, as.numeric(as.integer(x)))), "\n\n")
}

cat("=== Test 3: F32 → I32 → F32 → Floor (full chain from failing test) ===\n")
{
  inp   <- .onnx_value_info("X", 1L, c(4L))
  outp  <- .onnx_value_info("Y", 1L, c(4L))
  n1    <- .onnx_node("Cast",  "X",   "i32", attrs = list(.onnx_attr_int("to", 6L)))
  n2    <- .onnx_node("Cast",  "i32", "f32", attrs = list(.onnx_attr_int("to", 1L)))
  n3    <- .onnx_node("Floor", "f32", "Y")
  g     <- .onnx_graph("t3", list(n1, n2, n3), list(inp), list(outp))
  p     <- tempfile(fileext = ".onnx"); writeBin(.onnx_model(g), p)
  r     <- as.numeric(run_onnx(p, list(X = x)))
  exp   <- as.numeric(as.integer(x))
  cat("  result:  ", r, "\n")
  cat("  expected:", exp, "\n")
  cat("  PASS:", isTRUE(all.equal(r, exp)), "\n\n")
}

cat("=== Diagnosis: is ggml_cast I32->F32 numeric or reinterpret? ===\n")
{
  # Feed known int32 bit pattern: int32(1) = 0x00000001; as float = 1.4e-45 (reinterpret)
  # Correct numeric cast of int32(1) -> float = 1.0
  # We create a model that casts F32->I32 (known to work), then I32->F32
  # and compare result for x=1.0 (int32(1.0)=1, float(1)=1.0 either way)
  # Use x=1.5: int32(1.5)=1; numeric cast -> 1.0; reinterpret -> 1.4e-45
  xd  <- 1.5
  inp  <- .onnx_value_info("X", 1L, c(1L))
  outp <- .onnx_value_info("Y", 1L, c(1L))
  n1   <- .onnx_node("Cast", "X", "i32", attrs = list(.onnx_attr_int("to", 6L)))
  n2   <- .onnx_node("Cast", "i32", "Y", attrs = list(.onnx_attr_int("to", 1L)))
  g    <- .onnx_graph("diag", list(n1, n2), list(inp), list(outp))
  p    <- tempfile(fileext = ".onnx"); writeBin(.onnx_model(g), p)
  r    <- as.numeric(run_onnx(p, list(X = xd)))
  cat("  input=1.5, after F32->I32->F32:\n")
  cat("  result:", r, "\n")
  cat("  1.0 = numeric cast (correct)\n")
  cat("  1.4e-45 = reinterpret (bug)\n")
  cat("  Verdict:", if (abs(r - 1.0) < 0.01) "NUMERIC (correct)" else
                    if (abs(r - 1.5) < 0.01) "PASS-THROUGH (cast not applied)" else
                    "REINTERPRET (bug)", "\n\n")
}
