library(ggmlR)
source("tests/testthat/helper-onnx.R")

run_onnx <- function(path, inputs) {
  m <- onnx_load(path, device = "cpu")
  res <- onnx_run(m, inputs)
  res[[1]]
}

cat("=== Test 1: ConvTranspose 2D -> Relu ===\n")
inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 2L, 2L))
outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 4L, 4L))
w_data <- rep(0.5, 16)
w_raw <- unlist(lapply(w_data, .float_bytes))
w_t  <- .onnx_tensor("W", c(2L, 2L, 2L, 2L), 1L, w_raw)
w_vi <- .onnx_value_info("W", 1L, c(2L, 2L, 2L, 2L))
ct_node   <- .onnx_node("ConvTranspose", c("X", "W"), "ct",
                         attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                      .onnx_attr_ints("strides", c(2L, 2L))))
relu_node <- .onnx_node("Relu", "ct", "Y")
graph <- .onnx_graph("test", list(ct_node, relu_node),
                      list(inp, w_vi), list(outp), list(w_t))
path <- tempfile(fileext = ".onnx")
writeBin(.onnx_model(graph), path)
x <- c(1, -1, 2, -2, 0.5, 1, -0.5, -1)
tryCatch({
  r <- as.numeric(run_onnx(path, list(X = x)))
  cat("length:", length(r), "(expected 32)\n")
  cat("all >= 0 (Relu):", all(r >= 0), "\n")
  cat("first 4:", r[1:4], "\n")
}, error = function(e) cat("ERROR:", conditionMessage(e), "\n"))

cat("\n=== Test 2: ConvTranspose 1D ===\n")
inp2  <- .onnx_value_info("X", 1L, c(1L, 1L, 3L))
outp2 <- .onnx_value_info("Y", 1L, c(1L, 1L, 6L))
w2_raw <- unlist(lapply(rep(1.0, 2), .float_bytes))
w2_t  <- .onnx_tensor("W", c(1L, 1L, 2L), 1L, w2_raw)
w2_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L))
node2 <- .onnx_node("ConvTranspose", c("X", "W"), "Y",
                     attrs = list(.onnx_attr_ints("strides", c(2L))))
graph2 <- .onnx_graph("test", list(node2), list(inp2, w2_vi), list(outp2), list(w2_t))
path2 <- tempfile(fileext = ".onnx")
writeBin(.onnx_model(graph2), path2)
tryCatch({
  r2 <- as.numeric(run_onnx(path2, list(X = c(1, 2, 3))))
  cat("length:", length(r2), "(expected 6)\n")
  cat("values:", r2, "\n")
  cat("expected: 1 1 2 2 3 3\n")
}, error = function(e) cat("ERROR:", conditionMessage(e), "\n"))

cat("\n=== Test 3: ConvTranspose 2D 1x1 boundary ===\n")
inp3  <- .onnx_value_info("X", 1L, c(1L, 1L, 1L, 1L))
outp3 <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))
w3_raw <- unlist(lapply(c(1, 2, 3, 4), .float_bytes))
w3_t  <- .onnx_tensor("W", c(1L, 1L, 2L, 2L), 1L, w3_raw)
w3_vi <- .onnx_value_info("W", 1L, c(1L, 1L, 2L, 2L))
node3 <- .onnx_node("ConvTranspose", c("X", "W"), "Y",
                     attrs = list(.onnx_attr_ints("kernel_shape", c(2L, 2L)),
                                  .onnx_attr_ints("strides", c(2L, 2L))))
graph3 <- .onnx_graph("test", list(node3), list(inp3, w3_vi), list(outp3), list(w3_t))
path3 <- tempfile(fileext = ".onnx")
writeBin(.onnx_model(graph3), path3)
tryCatch({
  r3 <- as.numeric(run_onnx(path3, list(X = c(2.0))))
  cat("length:", length(r3), "(expected 4)\n")
  cat("values:", sort(r3), "\n")
  cat("expected: 2 4 6 8\n")
}, error = function(e) cat("ERROR:", conditionMessage(e), "\n"))
