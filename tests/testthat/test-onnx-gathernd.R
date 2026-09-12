# GatherND: reading whole slices of a tensor at index tuples.
#
# The mirror of ScatterND, and built the same way: folding the q addressed
# axes of `data` into one turns an ND slice into a row, which is what
# ggml_get_rows takes.  The flat row index is computed in the graph, so
# indices produced at runtime work as well as constant ones.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Reference in ONNX terms: for each index tuple, take the named slice.
# Arrays here are column-major, so ONNX axis k is R axis (r + 1 - k).
gathernd_ref <- function(data_arr, idx_mat, dims) {
  r <- length(dims); q <- ncol(idx_mat)
  rows <- lapply(seq_len(nrow(idx_mat)), function(m) {
    args <- vector("list", r)
    for (k in seq_len(q)) args[[r + 1 - k]] <- idx_mat[m, k] + 1
    if (r > q) for (k in seq_len(r - q)) args[[k]] <- seq_len(dims[r + 1 - k])
    as.numeric(do.call(`[`, c(list(data_arr), args)))
  })
  # ONNX output is [n_idx, ...slice]; flattened row-major that is slice-fastest,
  # which in column-major terms is exactly the rows laid side by side.
  unlist(rows)
}

make_gathernd <- function(dims, idx_mat) {
  inp  <- .onnx_value_info("X", 1L, as.integer(dims))
  idx_flat <- as.integer(t(idx_mat))            # row-major
  i_t  <- .onnx_tensor("idx", as.integer(dim(idx_mat)), 7L,
                       do.call(c, lapply(idx_flat, .int64_bytes)))
  i_vi <- .onnx_value_info("idx", 7L, as.integer(dim(idx_mat)))
  # output shape is left undeclared (rank unknown to the test harness)
  outp <- .onnx_value_info("Y", 1L, integer(0))
  node  <- .onnx_node("GatherND", c("X", "idx"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp, i_vi), list(outp), list(i_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

test_that("GatherND reads whole rows when q < rank", {
  # data [2,3,4], indices [2,2] -> two [4] rows.
  dims <- c(2L, 3L, 4L)
  idx  <- rbind(c(0L, 1L), c(1L, 2L))
  x <- as.numeric(seq_len(prod(dims)) - 1)
  result <- as.numeric(run_onnx(make_gathernd(dims, idx), list(X = x)))
  ref <- gathernd_ref(array(x, dim = rev(dims)), idx, dims)
  expect_equal(length(result), 8L)
  expect_equal(result, ref, tolerance = 1e-5)
})

test_that("GatherND reads single elements when q equals rank", {
  # Each tuple names one element; the row degenerates to length 1.
  dims <- c(2L, 3L)
  idx  <- rbind(c(0L, 1L), c(1L, 2L), c(1L, 0L))
  x <- as.numeric(seq_len(prod(dims)) - 1)
  result <- as.numeric(run_onnx(make_gathernd(dims, idx), list(X = x)))
  ref <- gathernd_ref(array(x, dim = rev(dims)), idx, dims)
  expect_equal(length(result), 3L)
  expect_equal(result, ref, tolerance = 1e-5)
})

test_that("GatherND reads whole planes when q is 1", {
  # indices [2,1]: each tuple names a [3,4] plane, so the output is [2,3,4].
  dims <- c(2L, 3L, 4L)
  idx  <- matrix(c(1L, 0L), ncol = 1)
  x <- as.numeric(seq_len(prod(dims)) - 1)
  result <- as.numeric(run_onnx(make_gathernd(dims, idx), list(X = x)))
  ref <- gathernd_ref(array(x, dim = rev(dims)), idx, dims)
  expect_equal(length(result), 24L)
  expect_equal(result, ref, tolerance = 1e-5)
})

test_that("GatherND repeats an index without complaint", {
  # Nothing in the spec says the tuples are distinct, and a scatter/gather
  # implemented by row copying must not care.
  dims <- c(2L, 3L)
  idx  <- rbind(c(1L, 1L), c(1L, 1L), c(0L, 0L))
  x <- as.numeric(seq_len(prod(dims)) - 1)
  result <- as.numeric(run_onnx(make_gathernd(dims, idx), list(X = x)))
  ref <- gathernd_ref(array(x, dim = rev(dims)), idx, dims)
  expect_equal(result, ref, tolerance = 1e-5)
  expect_equal(result[1], result[2])
})

test_that("GatherND round-trips with ScatterND", {
  # Writing values at index tuples and reading them back at the same tuples
  # must return what was written -- a check on the flat-index arithmetic that
  # does not depend on either op's reference being right, only on the two
  # agreeing.
  dims <- c(3L, 4L)
  idx  <- rbind(c(0L, 1L), c(2L, 3L), c(1L, 0L))
  vals <- c(7, 8, 9)

  i_t <- .onnx_tensor("idx", as.integer(dim(idx)), 7L,
                      do.call(c, lapply(as.integer(t(idx)), .int64_bytes)))
  u_t <- .onnx_tensor("upd", c(3L), 1L, .float_bytes(vals))
  vis <- list(.onnx_value_info("X", 1L, dims),
              .onnx_value_info("idx", 7L, as.integer(dim(idx))),
              .onnx_value_info("upd", 1L, c(3L)))
  nodes <- list(.onnx_node("ScatterND", c("X", "idx", "upd"), "s"),
                .onnx_node("GatherND",  c("s", "idx"), "Y"))
  graph <- .onnx_graph("test", nodes, vis,
                        list(.onnx_value_info("Y", 1L, integer(0))),
                        list(i_t, u_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- as.numeric(run_onnx(path, list(X = rep(0, prod(dims)))))
  expect_equal(result, vals, tolerance = 1e-5)
})

test_that("GatherND refuses batch_dims", {
  # batch_dims changes which axes are addressed and which are carried; no
  # model here uses it, so it is declined rather than approximated.
  dims <- c(2L, 3L)
  idx  <- rbind(c(0L, 1L))
  i_t  <- .onnx_tensor("idx", c(1L, 2L), 7L,
                       do.call(c, lapply(c(0L, 1L), .int64_bytes)))
  node <- .onnx_node("GatherND", c("X", "idx"), "Y",
                     attrs = list(.onnx_attr_int("batch_dims", 1L)))
  graph <- .onnx_graph("test", list(node),
                        list(.onnx_value_info("X", 1L, dims),
                             .onnx_value_info("idx", 7L, c(1L, 2L))),
                        list(.onnx_value_info("Y", 1L, integer(0))),
                        list(i_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  # A declined op leaves its output unregistered rather than raising.
  res <- suppressWarnings(try(run_onnx(path, list(X = rep(0, 6))), silent = TRUE))
  if (!inherits(res, "try-error")) {
    expect_equal(length(as.numeric(res)), 0L)
  } else {
    succeed()
  }
})
