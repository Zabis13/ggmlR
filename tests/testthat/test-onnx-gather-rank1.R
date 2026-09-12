# Gather over a 1-D data tensor, and the rank bookkeeping that feeds it.
#
# Why these were missing.  Every Gather test that existed used either a rank-2
# data tensor -- where the gathered axis already lands on ggml ne[1] and the
# op works -- or a Shape->Gather pair, where both operands are known at build
# time and the whole thing is folded to a constant without ggml_get_rows ever
# being called.  Neither shape reaches the branch that handles a rank-1 data
# tensor, so a defect there was invisible: on MaskRCNN it silently returned
# the WHOLE tensor instead of the selected element on 36 nodes, and only blew
# up on the five where the index happened not to be 0.
#
# ggml_get_rows always selects along ne[1].  A rank-1 ONNX tensor puts its
# data on ne[0], so the axis has to be rotated first -- the assertion these
# tests really make.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Build a model computing Gather(data, indices, axis=0) for a 1-D `data`
# supplied as an initializer, with `indices` as a runtime input.
.gather1d_model <- function(values, idx_dims) {
  data_t  <- .onnx_tensor("D", c(length(values)), 1L,
                          unlist(lapply(values, .float_bytes)))
  data_vi <- .onnx_value_info("D", 1L, c(length(values)))
  idx_vi  <- .onnx_value_info("I", 7L, idx_dims)
  out_vi  <- .onnx_value_info("Y", 1L, idx_dims)
  node    <- .onnx_node("Gather", c("D", "I"), "Y",
                        attrs = list(.onnx_attr_int("axis", 0L)))
  graph   <- .onnx_graph("test", list(node), list(data_vi, idx_vi),
                          list(out_vi), list(data_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

test_that("ONNX Gather selects from a 1-D tensor with a vector index", {
  # The plain case the old code got wrong without saying so: data on ne[0],
  # so get_rows had to be handed a rotated view.  Returning all of `values`
  # would pass a length check against a longer index, hence both assertions.
  vals <- c(10, 20, 30, 40, 50)
  path <- .gather1d_model(vals, c(3L))

  res <- as.numeric(run_onnx(path, list(I = c(4, 0, 2))))
  expect_equal(length(res), 3L)
  expect_equal(res, c(50, 10, 30), tolerance = 1e-5)
})

test_that("ONNX Gather on a 1-D tensor picks one element, not the whole row", {
  # A length-1 index is exactly the shape that used to pass unnoticed: the
  # index 0 falls inside [0,1), so the range assert stayed quiet while the op
  # copied every element.  The length check is the whole point here.
  vals <- c(1.5, 2.5, 3.5, 4.5)
  path <- .gather1d_model(vals, c(1L))

  res <- as.numeric(run_onnx(path, list(I = 2)))
  expect_equal(length(res), 1L)
  expect_equal(res, 3.5, tolerance = 1e-5)
})

test_that("ONNX Gather on a 1-D tensor reads an index other than zero", {
  # Index 0 is the value that hid the defect, because it is in range for a
  # ne[1] of 1.  Any other index aborted the run outright.
  vals <- c(7, 8, 9)
  path <- .gather1d_model(vals, c(1L))

  res <- as.numeric(run_onnx(path, list(I = 2)))
  expect_equal(res, 9, tolerance = 1e-5)
})

test_that("ONNX Gather on a 1-D tensor repeats and reorders indices", {
  vals <- c(100, 200, 300)
  path <- .gather1d_model(vals, c(4L))

  res <- as.numeric(run_onnx(path, list(I = c(2, 2, 0, 1))))
  expect_equal(res, c(300, 300, 100, 200), tolerance = 1e-5)
})

# ── The rank that feeds it ─────────────────────────────────────

test_that("ONNX Squeeze reports rank 1 when every remaining dim is 1", {
  # Squeeze([1,1]) over axis 1 is rank 1, and saying so matters even though
  # nothing about the tensor shows it: ggml_n_dims collapses [1] and [1,1]
  # alike, so a consumer cannot recover the rank by looking.  Reporting the
  # input's rank instead made MaskRCNN's Gather see a rank-2 index, which
  # pushed its output rank to 3 and left the Concat downstream refusing to
  # join shapes that no longer matched.
  #
  # Asserted through Gather, since the rank is not observable on its own: a
  # rank-1 index keeps the gathered axis, so the result has one element per
  # index -- a rank-2 index would splice a dimension in instead.
  sq_in   <- .onnx_value_info("S", 7L, c(1L, 1L))
  data_t  <- .onnx_tensor("D", c(3L), 1L,
                          unlist(lapply(c(11, 22, 33), .float_bytes)))
  data_vi <- .onnx_value_info("D", 1L, c(3L))
  out_vi  <- .onnx_value_info("Y", 1L, c(1L))

  nodes <- list(
    .onnx_node("Squeeze", "S", "idx",
               attrs = list(.onnx_attr_ints("axes", 1L))),
    .onnx_node("Gather", c("D", "idx"), "Y",
               attrs = list(.onnx_attr_int("axis", 0L))))

  graph <- .onnx_graph("test", nodes, list(data_vi, sq_in), list(out_vi),
                        list(data_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  res <- as.numeric(run_onnx(path, list(S = 1)))
  expect_equal(length(res), 1L)
  expect_equal(res, 22, tolerance = 1e-5)
})
