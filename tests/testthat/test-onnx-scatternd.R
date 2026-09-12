# ScatterND: writing whole slices of a tensor at index tuples.
#
# Mask2Former's Swin blocks build their shifted-window attention mask with
# nine of these in a row, each feeding the next, and their indices come from
# Mod on runtime shapes -- so the index arithmetic has to happen in the graph.
# A build-time reading of the indices is the defect that made strided Slice
# publish uninitialised memory; these tests use constant indices only because
# a test has to be reproducible, and the code path is the same either way.
#
# Implemented on ggml_scatter_elements over a 2-D view rather than a new
# kernel: folding the q addressed axes into one turns an ND slice into a row.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# Reference, in ONNX terms: for each index tuple, write the matching slice of
# updates into data.  Arrays are column-major here, so ONNX axis k is R axis
# (r + 1 - k).
scatternd_ref <- function(data_arr, idx_mat, upd, dims) {
  out <- data_arr
  r <- length(dims); q <- ncol(idx_mat)
  for (m in seq_len(nrow(idx_mat))) {
    args <- vector("list", r)
    for (k in seq_len(q)) args[[r + 1 - k]] <- idx_mat[m, k] + 1
    if (r > q) for (k in seq_len(r - q)) args[[k]] <- seq_len(dims[r + 1 - k])
    sl <- if (is.matrix(upd)) upd[m, ] else upd[m]
    out <- do.call(`[<-`, c(list(out), args, list(sl)))
  }
  out
}

# X (data) is a model input so nothing is folded away at build time;
# indices and updates are initializers.
make_scatternd <- function(dims, idx_mat, upd_dims, upd_vals) {
  n <- prod(dims)
  inp  <- .onnx_value_info("X", 1L, as.integer(dims))
  outp <- .onnx_value_info("Y", 1L, as.integer(dims))

  idx_flat <- as.integer(t(idx_mat))           # row-major
  i_t  <- .onnx_tensor("idx", as.integer(dim(idx_mat)), 7L,
                       do.call(c, lapply(idx_flat, .int64_bytes)))
  i_vi <- .onnx_value_info("idx", 7L, as.integer(dim(idx_mat)))
  u_t  <- .onnx_tensor("upd", as.integer(upd_dims), 1L, .float_bytes(upd_vals))
  u_vi <- .onnx_value_info("upd", 1L, as.integer(upd_dims))

  node  <- .onnx_node("ScatterND", c("X", "idx", "upd"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp, i_vi, u_vi), list(outp),
                        list(i_t, u_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

test_that("ScatterND writes whole rows when q < rank", {
  # data [2,3,4], indices [2,2] -> each tuple names a [4] row.
  dims <- c(2L, 3L, 4L)
  idx  <- rbind(c(0L, 1L), c(1L, 2L))
  updm <- rbind(c(10, 11, 12, 13), c(20, 21, 22, 23))

  x <- rep(0, prod(dims))
  path <- make_scatternd(dims, idx, c(2L, 4L), as.numeric(t(updm)))
  result <- as.numeric(run_onnx(path, list(X = x)))

  ref <- scatternd_ref(array(x, dim = rev(dims)), idx, updm, dims)
  expect_equal(result, as.numeric(ref), tolerance = 1e-5)
})

test_that("ScatterND writes single elements when q equals rank", {
  # data [2,3], indices [2,2] -> each tuple names one element, and the row
  # degenerates to length 1.  This is the boundary the row-folding has to
  # get right: row_len becomes 1 and the index needs no repeating.
  dims <- c(2L, 3L)
  idx  <- rbind(c(0L, 1L), c(1L, 2L))
  upd  <- c(7, 9)

  x <- rep(0, prod(dims))
  path <- make_scatternd(dims, idx, c(2L), upd)
  result <- as.numeric(run_onnx(path, list(X = x)))

  ref <- scatternd_ref(array(x, dim = rev(dims)), idx, upd, dims)
  expect_equal(result, as.numeric(ref), tolerance = 1e-5)
})

test_that("ScatterND leaves untouched positions as they were", {
  # data starts non-zero, so an implementation that builds the output from
  # scratch instead of copying `data` first shows up here.
  dims <- c(2L, 3L, 2L)
  idx  <- rbind(c(0L, 0L))
  updm <- rbind(c(-1, -2))

  set.seed(8)
  x <- round(rnorm(prod(dims)), 2)
  path <- make_scatternd(dims, idx, c(1L, 2L), as.numeric(updm))
  result <- as.numeric(run_onnx(path, list(X = x)))

  ref <- scatternd_ref(array(x, dim = rev(dims)), idx, updm, dims)
  expect_equal(result, as.numeric(ref), tolerance = 1e-5)
  # exactly two positions changed
  expect_equal(sum(abs(result - x) > 1e-6), 2L)
})

test_that("ScatterND writes the first axis when q is 1", {
  # indices [2,1]: each tuple names a whole [3,4] plane.
  dims <- c(2L, 3L, 4L)
  idx  <- matrix(c(1L), nrow = 1, ncol = 1)
  updm <- matrix(seq_len(12) * 1.0, nrow = 1)

  x <- rep(0, prod(dims))
  path <- make_scatternd(dims, idx, c(1L, 3L, 4L), as.numeric(updm))
  result <- as.numeric(run_onnx(path, list(X = x)))

  ref <- scatternd_ref(array(x, dim = rev(dims)), idx, updm, dims)
  expect_equal(result, as.numeric(ref), tolerance = 1e-5)
})

test_that("ScatterND applied twice in sequence, as Swin masks do", {
  # The model chains nine of these, each taking the previous output as data.
  # Chaining is where an implementation that writes into its input rather
  # than into a copy goes wrong.
  dims <- c(4L, 4L)
  inp  <- .onnx_value_info("X", 1L, dims)
  outp <- .onnx_value_info("Y", 1L, dims)

  i1 <- rbind(c(0L, 0L), c(1L, 1L))
  i2 <- rbind(c(2L, 2L), c(3L, 3L))
  mk_idx <- function(nm, m) .onnx_tensor(nm, as.integer(dim(m)), 7L,
                                         do.call(c, lapply(as.integer(t(m)), .int64_bytes)))
  t1 <- mk_idx("i1", i1); t2 <- mk_idx("i2", i2)
  u1 <- .onnx_tensor("u1", c(2L), 1L, .float_bytes(c(1, 2)))
  u2 <- .onnx_tensor("u2", c(2L), 1L, .float_bytes(c(3, 4)))
  vis <- list(.onnx_value_info("i1", 7L, as.integer(dim(i1))),
              .onnx_value_info("i2", 7L, as.integer(dim(i2))),
              .onnx_value_info("u1", 1L, c(2L)),
              .onnx_value_info("u2", 1L, c(2L)))

  n1 <- .onnx_node("ScatterND", c("X", "i1", "u1"), "s1")
  n2 <- .onnx_node("ScatterND", c("s1", "i2", "u2"), "Y")
  graph <- .onnx_graph("test", list(n1, n2), c(list(inp), vis), list(outp),
                        list(t1, u1, t2, u2))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rep(0, 16)
  result <- as.numeric(run_onnx(path, list(X = x)))

  ref <- scatternd_ref(array(x, dim = rev(dims)), i1, c(1, 2), dims)
  ref <- scatternd_ref(ref, i2, c(3, 4), dims)
  expect_equal(result, as.numeric(ref), tolerance = 1e-5)
  # the diagonal 1,2,3,4 -- both writes survived
  expect_equal(sum(result != 0), 4L)
})

test_that("ScatterND refuses a reduction it does not implement", {
  dims <- c(2L, 3L)
  idx  <- rbind(c(0L, 1L))
  i_t  <- .onnx_tensor("idx", c(1L, 2L), 7L,
                       do.call(c, lapply(c(0L, 1L), .int64_bytes)))
  u_t  <- .onnx_tensor("upd", c(1L), 1L, .float_bytes(5))
  node <- .onnx_node("ScatterND", c("X", "idx", "upd"), "Y",
                     attrs = list(.onnx_attr_string("reduction", "mul")))
  graph <- .onnx_graph("test", list(node),
                        list(.onnx_value_info("X", 1L, dims),
                             .onnx_value_info("idx", 7L, c(1L, 2L)),
                             .onnx_value_info("upd", 1L, c(1L))),
                        list(.onnx_value_info("Y", 1L, dims)),
                        list(i_t, u_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  # As above: the refusal shows up as a missing output, not an exception.
  res <- suppressWarnings(try(
    onnx_run(onnx_load(path, device = "cpu"), list(X = rep(0, 6))),
    silent = TRUE))
  if (!inherits(res, "try-error")) {
    expect_equal(length(as.numeric(res[[1]])), 0L)
  } else {
    succeed()
  }
})
