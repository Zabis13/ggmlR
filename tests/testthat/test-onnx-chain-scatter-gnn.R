# Chain tests: ScatterElements for GNN patterns (SageConv)
#
# ScatterElements with non-matching shapes: updates shorter than data,
# flexible indexing with 1D/2D indices. Tests the pattern that causes
# sageconv_Opset16 to fail.

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal: ScatterElements axis=0, same shapes ──────────────────

test_that("chain scatter-gnn: ScatterElements axis=0 same shapes (overwrite)", {
  # data[3,4], indices[3,4], updates[3,4] → scatter → [3,4]
  inp_d <- .onnx_value_info("D", 1L, c(3L, 4L))
  inp_u <- .onnx_value_info("U", 1L, c(3L, 4L))
  outp  <- .onnx_value_info("Y", 1L, c(3L, 4L))

  # indices: all zeros (write all rows to row 0)
  idx_raw <- rep(as.raw(0), 3L * 4L * 4L)  # 12 int32 zeros
  idx_t  <- .onnx_tensor("I", c(3L, 4L), 6L, idx_raw)  # INT32
  idx_vi <- .onnx_value_info("I", 6L, c(3L, 4L))

  sc_node <- .onnx_node("ScatterElements", c("D", "I", "U"), "Y",
                          attrs = list(.onnx_attr_int("axis", 0L)))

  graph <- .onnx_graph("test", list(sc_node),
                        list(inp_d, idx_vi, inp_u), list(outp),
                        list(idx_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  d <- rep(0, 12)
  u <- 1:12
  result <- run_onnx(path, list(D = d, U = as.numeric(u)))
  r <- as.numeric(result)
  expect_equal(length(r), 12)
  expect_true(all(is.finite(r)))
  # Row 0 should have been overwritten 3 times
})

# ── ScatterElements axis=0 with reduction=add ─────────────────────

test_that("chain scatter-gnn: ScatterElements axis=0 reduction=add", {
  # data[4,3], indices[4,3]=all-zeros, updates[4,3]
  # Result: row 0 = sum of all update rows, other rows = original data
  inp_d <- .onnx_value_info("D", 1L, c(4L, 3L))
  inp_u <- .onnx_value_info("U", 1L, c(4L, 3L))
  outp  <- .onnx_value_info("Y", 1L, c(4L, 3L))

  idx_raw <- rep(as.raw(0), 4L * 3L * 4L)  # 12 int32 zeros
  idx_t  <- .onnx_tensor("I", c(4L, 3L), 6L, idx_raw)
  idx_vi <- .onnx_value_info("I", 6L, c(4L, 3L))

  sc_node <- .onnx_node("ScatterElements", c("D", "I", "U"), "Y",
                          attrs = list(.onnx_attr_int("axis", 0L),
                                       .onnx_attr_string("reduction", "add")))

  graph <- .onnx_graph("test", list(sc_node),
                        list(inp_d, idx_vi, inp_u), list(outp),
                        list(idx_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  d <- rep(0, 12)
  u <- rep(1, 12)  # all ones
  result <- run_onnx(path, list(D = d, U = u))
  r <- as.numeric(result)
  expect_equal(length(r), 12)
  expect_true(all(is.finite(r)))
  # Row 0 of D (3 values) should each be 4.0 (sum of 4 ones)
  expect_equal(r[1], 4.0, tolerance = 1e-5)
})

# ── ScatterElements with different update/data sizes (GNN edge scatter) ──

test_that("chain scatter-gnn: ScatterElements non-matching shapes (2 updates into 4 rows)", {
  # data[4,3] (4 nodes, 3 features)
  # indices[2,3] (2 edges pointing to nodes)
  # updates[2,3] (2 edge features to scatter)
  # This is the SageConv pattern: fewer updates than data rows

  inp_d <- .onnx_value_info("D", 1L, c(4L, 3L))
  inp_u <- .onnx_value_info("U", 1L, c(2L, 3L))
  outp  <- .onnx_value_info("Y", 1L, c(4L, 3L))

  # indices: [[0,0,0],[2,2,2]] — scatter to row 0 and row 2
  idx_vals <- c(0L, 0L, 0L, 2L, 2L, 2L)
  idx_raw <- unlist(lapply(idx_vals, function(v) writeBin(v, raw(), size = 4)))
  idx_t  <- .onnx_tensor("I", c(2L, 3L), 6L, idx_raw)
  idx_vi <- .onnx_value_info("I", 6L, c(2L, 3L))

  sc_node <- .onnx_node("ScatterElements", c("D", "I", "U"), "Y",
                          attrs = list(.onnx_attr_int("axis", 0L)))

  graph <- .onnx_graph("test", list(sc_node),
                        list(inp_d, idx_vi, inp_u), list(outp),
                        list(idx_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  d <- rep(0, 12)         # 4x3 zeros
  u <- c(1, 2, 3, 4, 5, 6)  # 2x3 updates
  result <- run_onnx(path, list(D = d, U = u))
  r <- as.numeric(result)
  expect_equal(length(r), 12)
  expect_true(all(is.finite(r)))
})

# ── ScatterElements with reduction=add + non-matching (GNN message passing) ──

test_that("chain scatter-gnn: ScatterElements add reduction + non-matching (message passing)", {
  # Simulate GNN aggregation: 5 edges scatter-add into 4 nodes
  # data[4,2] (4 nodes), indices[5,2], updates[5,2]

  inp_d <- .onnx_value_info("D", 1L, c(4L, 2L))
  inp_u <- .onnx_value_info("U", 1L, c(5L, 2L))
  outp  <- .onnx_value_info("Y", 1L, c(4L, 2L))

  # indices: edges point to nodes 0,1,1,2,3
  idx_vals <- c(0L, 0L, 1L, 1L, 1L, 1L, 2L, 2L, 3L, 3L)
  idx_raw <- unlist(lapply(idx_vals, function(v) writeBin(v, raw(), size = 4)))
  idx_t  <- .onnx_tensor("I", c(5L, 2L), 6L, idx_raw)
  idx_vi <- .onnx_value_info("I", 6L, c(5L, 2L))

  sc_node <- .onnx_node("ScatterElements", c("D", "I", "U"), "Y",
                          attrs = list(.onnx_attr_int("axis", 0L),
                                       .onnx_attr_string("reduction", "add")))

  graph <- .onnx_graph("test", list(sc_node),
                        list(inp_d, idx_vi, inp_u), list(outp),
                        list(idx_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  d <- rep(0, 8)          # 4x2 zeros
  u <- rep(1, 10)         # 5x2 ones
  result <- run_onnx(path, list(D = d, U = u))
  r <- as.numeric(result)
  expect_equal(length(r), 8)
  expect_true(all(is.finite(r)))
  # node 1 receives 2 messages → should be 2.0
})
