# Chain tests: RoiAlign and NonMaxSuppression ops
# These are key ops for detection models (MaskRCNN, Faster-RCNN)

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  res <- onnx_run(m, inputs)
  res[[1]]
}

# ── RoiAlign basic: single ROI covering full feature map ────

test_that("chain roialign: single ROI full coverage", {
  # Feature map X: [1, 1, 4, 4] (N=1, C=1, H=4, W=4)
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 4L, 4L))

  # ROIs: [1, 4] — single ROI covering full spatial extent
  # roi = [x1=0, y1=0, x2=4, y2=4] (at scale=1.0)
  roi_data <- c(0.0, 0.0, 4.0, 4.0)
  roi_raw <- unlist(lapply(roi_data, .float_bytes))
  roi_t  <- .onnx_tensor("rois", c(1L, 4L), 1L, roi_raw)
  roi_vi <- .onnx_value_info("rois", 1L, c(1L, 4L))

  # batch_indices: [1] = 0
  bi_data <- c(0.0)
  bi_raw <- .float_bytes(bi_data)
  bi_t  <- .onnx_tensor("bi", c(1L), 7L, .int64_bytes(0L))
  bi_vi <- .onnx_value_info("bi", 7L, c(1L))

  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 2L, 2L))

  roi_node <- .onnx_node("RoiAlign", c("X", "rois", "bi"), "Y",
    attrs = list(
      .onnx_attr_int("output_height", 2L),
      .onnx_attr_int("output_width", 2L),
      .onnx_attr_int("sampling_ratio", 2L),
      .onnx_attr_float("spatial_scale", 1.0),
      .onnx_attr_string("mode", "avg")))

  graph <- .onnx_graph("test", list(roi_node),
    list(inp, roi_vi, bi_vi), list(outp),
    list(roi_t, bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # 4x4 feature map, values 1..16
  x <- as.numeric(1:16)
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 4)  # 1*1*2*2
  expect_true(all(is.finite(r)))
  # Average pooling: each 2x2 output bin averages a quadrant
  expect_true(all(r > 0))
})

# ── RoiAlign: multiple ROIs, spatial_scale < 1 ──────────────

test_that("chain roialign: 2 ROIs with spatial_scale=0.5", {
  # Feature map X: [1, 2, 4, 4]
  inp <- .onnx_value_info("X", 1L, c(1L, 2L, 4L, 4L))

  # 2 ROIs in original image coords (spatial_scale=0.5 maps to feature map)
  roi_data <- c(0, 0, 8, 8,   # ROI 0: full
                0, 0, 4, 4)   # ROI 1: top-left quarter
  roi_raw <- unlist(lapply(roi_data, .float_bytes))
  roi_t  <- .onnx_tensor("rois", c(2L, 4L), 1L, roi_raw)
  roi_vi <- .onnx_value_info("rois", 1L, c(2L, 4L))

  bi_raw <- c(.int64_bytes(0L), .int64_bytes(0L))
  bi_t  <- .onnx_tensor("bi", c(2L), 7L, bi_raw)
  bi_vi <- .onnx_value_info("bi", 7L, c(2L))

  outp <- .onnx_value_info("Y", 1L, c(2L, 2L, 2L, 2L))

  roi_node <- .onnx_node("RoiAlign", c("X", "rois", "bi"), "Y",
    attrs = list(
      .onnx_attr_int("output_height", 2L),
      .onnx_attr_int("output_width", 2L),
      .onnx_attr_int("sampling_ratio", 1L),
      .onnx_attr_float("spatial_scale", 0.5),
      .onnx_attr_string("mode", "avg")))

  graph <- .onnx_graph("test", list(roi_node),
    list(inp, roi_vi, bi_vi), list(outp),
    list(roi_t, bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- as.numeric(seq_len(32))  # 1*2*4*4
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 16)  # 2*2*2*2
  expect_true(all(is.finite(r)))
})

# ── NonMaxSuppression: basic filtering ───────────────────────

test_that("chain nms: basic box filtering", {
  # boxes: [1, 4, 4] — 4 boxes in corner format [y1,x1,y2,x2]
  inp_boxes <- .onnx_value_info("boxes", 1L, c(1L, 4L, 4L))
  # scores: [1, 1, 4] — 1 class, 4 scores
  inp_scores <- .onnx_value_info("scores", 1L, c(1L, 1L, 4L))

  # max_output_boxes = 2
  mob_raw <- .int64_bytes(2L)
  mob_t  <- .onnx_tensor("mob", c(1L), 7L, mob_raw)
  mob_vi <- .onnx_value_info("mob", 7L, c(1L))

  # iou_threshold = 0.5
  iou_raw <- .float_bytes(0.5)
  iou_t  <- .onnx_tensor("iou", c(1L), 1L, iou_raw)
  iou_vi <- .onnx_value_info("iou", 1L, c(1L))

  outp <- .onnx_value_info("Y", 7L, c(-1L, 3L))

  nms_node <- .onnx_node("NonMaxSuppression",
    c("boxes", "scores", "mob", "iou"), "Y",
    attrs = list(.onnx_attr_int("center_point_box", 0L)))

  graph <- .onnx_graph("test", list(nms_node),
    list(inp_boxes, inp_scores, mob_vi, iou_vi),
    list(outp), list(mob_t, iou_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  # 4 boxes, two pairs overlap heavily
  # Box 0: [0,0,10,10], Box 1: [0,0,9,9] (high overlap with 0)
  # Box 2: [20,20,30,30], Box 3: [20,20,29,29] (high overlap with 2)
  boxes <- c(0,0,10,10,  0,0,9,9,  20,20,30,30,  20,20,29,29)
  scores <- c(0.9, 0.8, 0.7, 0.6)  # descending

  result <- run_onnx(path, list(boxes = boxes, scores = scores))
  r <- as.numeric(result)
  # Should select box 0 (highest score) and box 2 (no overlap with 0)
  expect_true(length(r) >= 3)  # at least 1 selection * 3 columns
})

# ── RoiAlign → Sigmoid chain ────────────────────────────────

test_that("chain roialign-sigmoid: RoiAlign → Sigmoid (mask head)", {
  inp <- .onnx_value_info("X", 1L, c(1L, 1L, 8L, 8L))

  roi_data <- c(0, 0, 8, 8)
  roi_raw <- unlist(lapply(roi_data, .float_bytes))
  roi_t  <- .onnx_tensor("rois", c(1L, 4L), 1L, roi_raw)
  roi_vi <- .onnx_value_info("rois", 1L, c(1L, 4L))

  bi_raw <- .int64_bytes(0L)
  bi_t  <- .onnx_tensor("bi", c(1L), 7L, bi_raw)
  bi_vi <- .onnx_value_info("bi", 7L, c(1L))

  outp <- .onnx_value_info("Y", 1L, c(1L, 1L, 4L, 4L))

  roi_node <- .onnx_node("RoiAlign", c("X", "rois", "bi"), "roi_out",
    attrs = list(
      .onnx_attr_int("output_height", 4L),
      .onnx_attr_int("output_width", 4L),
      .onnx_attr_int("sampling_ratio", 2L),
      .onnx_attr_float("spatial_scale", 1.0),
      .onnx_attr_string("mode", "avg")))
  sig_node <- .onnx_node("Sigmoid", "roi_out", "Y")

  graph <- .onnx_graph("test", list(roi_node, sig_node),
    list(inp, roi_vi, bi_vi), list(outp),
    list(roi_t, bi_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- rnorm(64)  # 1*1*8*8
  result <- run_onnx(path, list(X = x))
  r <- as.numeric(result)
  expect_equal(length(r), 16)  # 1*1*4*4
  expect_true(all(r >= 0 & r <= 1))  # sigmoid output
})
