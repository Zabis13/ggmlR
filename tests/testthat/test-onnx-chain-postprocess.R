# Chain tests: Postprocessing / tensor manipulation patterns
# Slice → Unsqueeze → Expand → Tile
#
# Covers: Slice, Unsqueeze, Expand, Tile

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── Minimal (2 ops): Slice → Unsqueeze ──────────────────────

test_that("chain postprocess: Slice→Unsqueeze (minimal)", {
  # Input: [6], Slice [1:4] → [3], Unsqueeze(axis=0) → [1, 3]
  inp <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 3L))

  # Slice params: starts=[1], ends=[4], axes=[0], steps=[1]
  starts_raw <- writeBin(1L, raw(), size = 8, endian = "little")
  ends_raw   <- writeBin(4L, raw(), size = 8, endian = "little")
  axes_raw   <- writeBin(0L, raw(), size = 8, endian = "little")
  steps_raw  <- writeBin(1L, raw(), size = 8, endian = "little")

  starts_t  <- .onnx_tensor("starts", c(1L), 7L, starts_raw)
  ends_t    <- .onnx_tensor("ends",   c(1L), 7L, ends_raw)
  axes_t    <- .onnx_tensor("axes",   c(1L), 7L, axes_raw)
  steps_t   <- .onnx_tensor("steps",  c(1L), 7L, steps_raw)
  starts_vi <- .onnx_value_info("starts", 7L, c(1L))
  ends_vi   <- .onnx_value_info("ends",   7L, c(1L))
  axes_vi   <- .onnx_value_info("axes",   7L, c(1L))
  steps_vi  <- .onnx_value_info("steps",  7L, c(1L))

  # Unsqueeze axes
  usq_axes_raw <- writeBin(0L, raw(), size = 8, endian = "little")
  usq_t  <- .onnx_tensor("usq_axes", c(1L), 7L, usq_axes_raw)
  usq_vi <- .onnx_value_info("usq_axes", 7L, c(1L))

  slice_node <- .onnx_node("Slice", c("X", "starts", "ends", "axes", "steps"), "sliced")
  usq_node   <- .onnx_node("Unsqueeze", c("sliced", "usq_axes"), "Y")

  graph <- .onnx_graph("test",
                        list(slice_node, usq_node),
                        list(inp, starts_vi, ends_vi, axes_vi, steps_vi, usq_vi),
                        list(outp),
                        list(starts_t, ends_t, axes_t, steps_t, usq_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(10, 20, 30, 40, 50, 60)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 3)
  expect_equal(r, c(20, 30, 40), tolerance = 1e-4)
})

# ── Real (4 ops): Slice → Unsqueeze → Expand → Tile ─────────

test_that("chain postprocess: Slice→Unsqueeze→Expand→Tile (broadcast pattern)", {
  # Input: [6], Slice [0:3] → [3]
  # Unsqueeze(axis=0) → [1, 3]
  # Expand to [2, 3]
  # Tile [1, 2] → [2, 6]

  inp <- .onnx_value_info("X", 1L, c(6L))
  outp <- .onnx_value_info("Y", 1L, c(2L, 6L))

  # Slice
  starts_raw <- writeBin(0L, raw(), size = 8, endian = "little")
  ends_raw   <- writeBin(3L, raw(), size = 8, endian = "little")
  axes_raw   <- writeBin(0L, raw(), size = 8, endian = "little")
  steps_raw  <- writeBin(1L, raw(), size = 8, endian = "little")
  starts_t  <- .onnx_tensor("starts", c(1L), 7L, starts_raw)
  ends_t    <- .onnx_tensor("ends",   c(1L), 7L, ends_raw)
  axes_t    <- .onnx_tensor("axes",   c(1L), 7L, axes_raw)
  steps_t   <- .onnx_tensor("steps",  c(1L), 7L, steps_raw)
  starts_vi <- .onnx_value_info("starts", 7L, c(1L))
  ends_vi   <- .onnx_value_info("ends",   7L, c(1L))
  axes_vi   <- .onnx_value_info("axes",   7L, c(1L))
  steps_vi  <- .onnx_value_info("steps",  7L, c(1L))

  # Unsqueeze axis=0
  usq_raw <- writeBin(0L, raw(), size = 8, endian = "little")
  usq_t  <- .onnx_tensor("usq_axes", c(1L), 7L, usq_raw)
  usq_vi <- .onnx_value_info("usq_axes", 7L, c(1L))

  # Expand shape: [2, 3]
  expand_raw <- c(writeBin(2L, raw(), size = 8, endian = "little"),
                  writeBin(3L, raw(), size = 8, endian = "little"))
  expand_t  <- .onnx_tensor("exp_shape", c(2L), 7L, expand_raw)
  expand_vi <- .onnx_value_info("exp_shape", 7L, c(2L))

  # Tile reps: [1, 2]
  tile_raw <- c(.int64_bytes(1L), .int64_bytes(2L))
  tile_t  <- .onnx_tensor("reps", c(2L), 7L, tile_raw)
  tile_vi <- .onnx_value_info("reps", 7L, c(2L))

  slice_node  <- .onnx_node("Slice", c("X", "starts", "ends", "axes", "steps"), "sl")
  usq_node    <- .onnx_node("Unsqueeze", c("sl", "usq_axes"), "usq")
  expand_node <- .onnx_node("Expand", c("usq", "exp_shape"), "exp")
  tile_node   <- .onnx_node("Tile", c("exp", "reps"), "Y")

  graph <- .onnx_graph("test",
                        list(slice_node, usq_node, expand_node, tile_node),
                        list(inp, starts_vi, ends_vi, axes_vi, steps_vi,
                             usq_vi, expand_vi, tile_vi),
                        list(outp),
                        list(starts_t, ends_t, axes_t, steps_t,
                             usq_t, expand_t, tile_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(1, 2, 3, 4, 5, 6)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 12)
  # Slice [1,2,3] → Unsqueeze → [[1,2,3]] → Expand → [[1,2,3],[1,2,3]]
  # Tile [1,2] → [[1,2,3,1,2,3],[1,2,3,1,2,3]]
  expect_equal(r, rep(c(1, 2, 3, 1, 2, 3), 2), tolerance = 1e-4)
})

# ── Boundary: empty slice ────────────────────────────────────

test_that("chain postprocess: Slice full range (boundary)", {
  # Slice with start=0, end=MAX → identity
  inp <- .onnx_value_info("X", 1L, c(3L))
  outp <- .onnx_value_info("Y", 1L, c(3L))

  starts_raw <- writeBin(0L, raw(), size = 8, endian = "little")
  ends_raw   <- writeBin(999L, raw(), size = 8, endian = "little")
  axes_raw   <- writeBin(0L, raw(), size = 8, endian = "little")
  starts_t  <- .onnx_tensor("starts", c(1L), 7L, starts_raw)
  ends_t    <- .onnx_tensor("ends",   c(1L), 7L, ends_raw)
  axes_t    <- .onnx_tensor("axes",   c(1L), 7L, axes_raw)
  starts_vi <- .onnx_value_info("starts", 7L, c(1L))
  ends_vi   <- .onnx_value_info("ends",   7L, c(1L))
  axes_vi   <- .onnx_value_info("axes",   7L, c(1L))

  slice_node <- .onnx_node("Slice", c("X", "starts", "ends", "axes"), "Y")

  graph <- .onnx_graph("test", list(slice_node),
                        list(inp, starts_vi, ends_vi, axes_vi),
                        list(outp),
                        list(starts_t, ends_t, axes_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  result <- run_onnx(path, list(X = c(7, 8, 9)))
  r <- as.numeric(result)
  expect_equal(r, c(7, 8, 9), tolerance = 1e-4)
})
