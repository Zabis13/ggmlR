# Tests for ONNX model metadata and Constant op

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Model metadata ───────────────────────────────────────────────

test_that("onnx_load returns correct metadata", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  expect_s3_class(m, "onnx_model")
  expect_equal(m$n_nodes, 1L)
  expect_true("Relu" %in% m$ops)
})

test_that("onnx_summary works", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  s <- onnx_summary(m)
  expect_true(is.list(s))
  expect_equal(s$n_nodes, 1L)
})

test_that("onnx_inputs returns correct info", {
  path <- .onnx_make_binary("Add", c(4L))
  m <- onnx_load(path, device = "cpu")
  inp <- onnx_inputs(m)
  expect_true(is.list(inp))
  expect_true(length(inp) >= 2)
})

test_that("print.onnx_model works", {
  path <- .onnx_make_unary("Relu", c(4L))
  m <- onnx_load(path, device = "cpu")
  expect_output(print(m), "ONNX Model")
})

# ── Constant op ──────────────────────────────────────────────────

test_that("ONNX Constant tensor + Add works", {
  const_data <- unlist(lapply(c(10, 20, 30, 40), .float_bytes))
  const_attr <- .onnx_attr_tensor("value", c(4L), 1L, const_data)
  const_node <- .onnx_node("Constant", character(0), "C", attrs = list(const_attr))

  inp  <- .onnx_value_info("X", 1L, c(4L))
  outp <- .onnx_value_info("Y", 1L, c(4L))
  add_node <- .onnx_node("Add", c("C", "X"), "Y")

  graph <- .onnx_graph("test", list(const_node, add_node),
                        list(inp), list(outp))
  model <- .onnx_model(graph)
  path <- tempfile(fileext = ".onnx")
  writeBin(model, path)

  x <- c(1, 2, 3, 4)
  result <- run_onnx(path, list(X = x))
  expect_equal(as.numeric(result), c(11, 22, 33, 44), tolerance = 1e-5)
})
