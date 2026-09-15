# RoiAlign: the Vulkan shader must agree with the CPU kernel BIT FOR BIT.
#
# This is the acceptance gate for moving RoiAlign onto the GPU, and the reason
# it is a test rather than a one-off measurement: a manual run proves the
# shader was right once, a test proves it stays right -- and the thing being
# protected here is fragile in a specific way.
#
# MaskRCNN-12-int8 matches ONNX Runtime exactly (max|d| = 0), an agreement that
# took five debugging sessions and, crucially, was reached by copying ORT's
# arithmetic rather than writing a cleaner equivalent. Two of ORT's own choices
# are load-bearing and both live in this op: a degenerate ROI is forced to 1x1
# rather than clamped to an epsilon (95 of MaskRCNN's 895 RPN ROIs are
# degenerate), and the bilinear tap rejects samples outside the padded box
# before clamping. A shader that is merely close reintroduces the class of
# defect the whole exercise removed: quantised detection scores sit on
# thresholds, so one differing bit drops a detection rather than perturbing a
# number.
#
# Hence expect_identical on the raw doubles, not expect_equal with a tolerance.
# If this test ever fails, the correct response is to leave RoiAlign on the CPU
# kernel -- not to widen the comparison.

skip_if_not(exists("skip_if_no_gpu"), "helper-device.R not loaded")

# Build a RoiAlign model and run it on one device.
#
# spatial_scale, sampling_ratio and mode are parameters because they select
# genuinely different paths through the kernel: sampling_ratio = 0 switches to
# the adaptive grid (ceil(roi_size / out_size)), and mode = "max" replaces the
# averaged accumulation with a running maximum.
roialign_run <- function(x, roi_data, bi_data, x_dims,
                         output_height = 2L, output_width = 2L,
                         sampling_ratio = 2L, spatial_scale = 1.0,
                         mode = "avg", device = "cpu") {
  num_rois <- length(roi_data) %/% 4L

  inp <- .onnx_value_info("X", 1L, x_dims)

  roi_raw <- unlist(lapply(roi_data, .float_bytes))
  roi_t   <- .onnx_tensor("rois", c(num_rois, 4L), 1L, roi_raw)
  roi_vi  <- .onnx_value_info("rois", 1L, c(num_rois, 4L))

  bi_raw <- unlist(lapply(bi_data, .int64_bytes))
  bi_t   <- .onnx_tensor("bi", num_rois, 7L, bi_raw)
  bi_vi  <- .onnx_value_info("bi", 7L, num_rois)

  outp <- .onnx_value_info("Y", 1L,
                           c(num_rois, x_dims[2], output_height, output_width))

  roi_node <- .onnx_node("RoiAlign", c("X", "rois", "bi"), "Y",
    attrs = list(
      .onnx_attr_int("output_height", output_height),
      .onnx_attr_int("output_width", output_width),
      .onnx_attr_int("sampling_ratio", sampling_ratio),
      .onnx_attr_float("spatial_scale", spatial_scale),
      .onnx_attr_string("mode", mode)))

  graph <- .onnx_graph("roialign", list(roi_node),
                       list(inp, roi_vi, bi_vi), list(outp),
                       list(roi_t, bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  m <- onnx_load(path, device = device)
  as.numeric(onnx_run(m, list(X = x))[[1]])
}

# Run the same model on both devices and demand identical numbers.
expect_roialign_identical <- function(..., info = NULL) {
  cpu <- roialign_run(..., device = "cpu")
  gpu <- roialign_run(..., device = "vulkan")

  # Length first: a shorter GPU result would otherwise fail as a value
  # mismatch and send the reader looking at arithmetic instead of at shape.
  expect_equal(length(gpu), length(cpu), info = info)
  expect_true(all(is.finite(gpu)), info = info)
  expect_identical(gpu, cpu, info = info)
}

test_that("roialign vulkan matches cpu bit for bit: single ROI, full coverage", {
  skip_if_no_gpu()
  expect_roialign_identical(
    x = as.numeric(1:16), roi_data = c(0, 0, 4, 4), bi_data = 0L,
    x_dims = c(1L, 1L, 4L, 4L))
})

test_that("roialign vulkan matches cpu bit for bit: multiple ROIs and channels", {
  skip_if_no_gpu()
  # Two ROIs over a 2-channel map, and a scale that puts the sample points
  # between pixels so the bilinear blend actually does something.
  expect_roialign_identical(
    x = as.numeric(seq_len(32)),
    roi_data = c(0, 0, 8, 8,
                 0, 0, 4, 4),
    bi_data = c(0L, 0L),
    x_dims = c(1L, 2L, 4L, 4L),
    sampling_ratio = 1L, spatial_scale = 0.5)
})

test_that("roialign vulkan matches cpu bit for bit: fractional ROI bounds", {
  skip_if_no_gpu()
  # Deliberately off-grid: every tap lands strictly between four pixels, which
  # is the case where a reassociated blend (or an FMA contraction) diverges in
  # the last bit while an aligned one would not.
  expect_roialign_identical(
    x = as.numeric(seq_len(64)) / 7,
    roi_data = c(0.3, 0.7, 5.9, 6.1),
    bi_data = 0L,
    x_dims = c(1L, 1L, 8L, 8L),
    output_height = 3L, output_width = 3L, sampling_ratio = 2L)
})

test_that("roialign vulkan matches cpu bit for bit: degenerate ROI forced to 1x1", {
  skip_if_no_gpu()
  # Zero width and zero height. ORT forces such a box to 1x1 rather than
  # clamping it to an epsilon, and MaskRCNN's RPN emits 95 of these out of 895;
  # getting it wrong makes every channel of every one of them wrong, so the two
  # implementations have to agree on the SAME wrong-looking answer.
  expect_roialign_identical(
    x = as.numeric(seq_len(16)),
    roi_data = c(2, 2, 2, 2),
    bi_data = 0L,
    x_dims = c(1L, 1L, 4L, 4L))
})

test_that("roialign vulkan matches cpu bit for bit: samples outside the map", {
  skip_if_no_gpu()
  # A ROI hanging off the bottom-right corner. Taps beyond [-1, H] x [-1, W]
  # contribute nothing, and the low/high index pair collapses on the last row
  # and column instead of reading past the end -- both are edge behaviours the
  # shader reproduces by hand.
  expect_roialign_identical(
    x = as.numeric(seq_len(16)),
    roi_data = c(2.5, 2.5, 7.5, 7.5),
    bi_data = 0L,
    x_dims = c(1L, 1L, 4L, 4L))
})

test_that("roialign vulkan matches cpu bit for bit: adaptive sampling ratio", {
  skip_if_no_gpu()
  # sampling_ratio = 0 makes the grid depend on the ROI size via ceilf(), so
  # the two implementations must agree on the tap COUNT before they can agree
  # on the value.
  expect_roialign_identical(
    x = as.numeric(seq_len(64)) / 3,
    roi_data = c(0, 0, 7.5, 7.5),
    bi_data = 0L,
    x_dims = c(1L, 1L, 8L, 8L),
    sampling_ratio = 0L)
})

test_that("roialign vulkan matches cpu bit for bit: max mode", {
  skip_if_no_gpu()
  # max mode takes a running maximum instead of an average, so it has no
  # division at the end and cannot hide a differing tap behind the averaging.
  expect_roialign_identical(
    x = as.numeric(seq_len(64)) / 5,
    roi_data = c(0.5, 0.5, 6.5, 6.5),
    bi_data = 0L,
    x_dims = c(1L, 1L, 8L, 8L),
    mode = "max")
})

test_that("roialign vulkan matches cpu bit for bit: batch index selects the map", {
  skip_if_no_gpu()
  # Two images in the batch with different contents, one ROI reading from the
  # second. A shader that ignored batch_indices would still return finite
  # numbers of the right shape -- just from the wrong image.
  expect_roialign_identical(
    x = c(as.numeric(seq_len(16)), as.numeric(seq_len(16)) * -3),
    roi_data = c(0, 0, 4, 4),
    bi_data = 1L,
    x_dims = c(2L, 1L, 4L, 4L))
})
