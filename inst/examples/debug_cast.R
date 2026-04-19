library(ggmlR)
source("tests/testthat/helper-onnx.R")

run_onnx <- function(path, inputs) {
  m <- onnx_load(path, device = "cpu")
  onnx_run(m, inputs)[[1]]
}

x <- c(1.9, -2.7, 0.5, 3.1)

# Test 1: F32 -> I32 только
cat("=== Test 1: Cast F32->I32 only ===\n")
inp  <- .onnx_value_info("X",   1L, c(4L))
outp <- .onnx_value_info("Y",   6L, c(4L))
graph <- .onnx_graph("test",
  list(.onnx_node("Cast", "X", "Y", attrs=list(.onnx_attr_int("to", 6L)))),
  list(inp), list(outp))
path <- tempfile(fileext=".onnx"); writeBin(.onnx_model(graph), path)
r <- as.numeric(run_onnx(path, list(X=x)))
cat("result:  ", r, "\n")
cat("expected:", as.numeric(as.integer(x)), "\n\n")

# Test 2: I32 input -> Cast I32->F32 только (input объявлен как INT32)
cat("=== Test 2: Cast I32->F32 only (INT32 input) ===\n")
inp2  <- .onnx_value_info("X", 6L, c(4L))   # elem_type=6 = INT32
outp2 <- .onnx_value_info("Y", 1L, c(4L))
graph2 <- .onnx_graph("test",
  list(.onnx_node("Cast", "X", "Y", attrs=list(.onnx_attr_int("to", 1L)))),
  list(inp2), list(outp2))
path2 <- tempfile(fileext=".onnx"); writeBin(.onnx_model(graph2), path2)
r2 <- as.numeric(run_onnx(path2, list(X=c(1, -2, 0, 3))))
cat("result:  ", r2, "\n")
cat("expected: 1 -2 0 3\n\n")

# Test 3: полная цепочка F32->I32->F32->Floor
cat("=== Test 3: F32->I32->F32->Floor chain ===\n")
inp3  <- .onnx_value_info("X", 1L, c(4L))
outp3 <- .onnx_value_info("Y", 1L, c(4L))
graph3 <- .onnx_graph("test", list(
  .onnx_node("Cast",  "X",   "i32", attrs=list(.onnx_attr_int("to", 6L))),
  .onnx_node("Cast",  "i32", "f32", attrs=list(.onnx_attr_int("to", 1L))),
  .onnx_node("Floor", "f32", "Y")
), list(inp3), list(outp3))
path3 <- tempfile(fileext=".onnx"); writeBin(.onnx_model(graph3), path3)
r3 <- as.numeric(run_onnx(path3, list(X=x)))
cat("result:  ", r3, "\n")
cat("expected:", as.numeric(as.integer(x)), "\n")
