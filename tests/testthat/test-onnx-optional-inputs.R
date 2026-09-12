# An absent optional input is not the same as its type's zero.
#
# ONNX marks many inputs optional, and each has a defined meaning when it is
# missing -- usually "do not apply this at all", which is NOT what the value 0
# would mean if it were supplied.  Collapsing the two is a silent wrong answer:
# nothing errors, the graph runs, and some elements quietly go missing.
#
# This cost MaskRCNN 217 of 511 RPN proposals on one pyramid level.
# NonMaxSuppression's score_threshold is optional; absent became 0.0f and was
# then applied with a strict ">", so every candidate scoring exactly 0 was
# dropped -- and a quantised score range bottoms out at exactly 0, so there
# were hundreds of them.  Per the spec an absent threshold filters nothing.
#
# Each test is a PAIR: the same graph with the input absent, and with it
# present and set to the value the buggy default would have been.  One model
# alone cannot catch this class -- whichever case is written, the wrong
# implementation passes it and the OTHER one is what breaks.  The pair must
# disagree, and a later "fix" that flattens them is the bug returning.
#
# An omitted optional input is written as an empty input NAME, which is how
# ONNX spells "not supplied" for an input that later positions follow.

# .int64_bytes takes one value at a time; pads needs four.
.i64v <- function(xs) {
  out <- raw(0)
  for (x in xs) out <- c(out, .int64_bytes(x))
  out
}

run1 <- function(model_bytes, input, dims, name = "X", device = "cpu") {
  path <- tempfile(fileext = ".onnx")
  writeBin(model_bytes, path)
  on.exit(unlink(path), add = TRUE)
  m <- onnx_load(path, device = device,
                 input_shapes = setNames(list(as.integer(dims)), name))
  out <- onnx_run(m, setNames(list(array(as.numeric(input),
                                         dim = rev(as.integer(dims)))), name))
  as.numeric(out[[1]])
}

test_that("NMS: absent score_threshold keeps a zero-scoring box, present 0 drops it", {
  # Two boxes far apart, so IoU suppression cannot be what removes either.
  # The second scores exactly 0 -- the value the old default made unreachable.
  boxes  <- c(0, 0, 10, 10,  100, 100, 110, 110)
  scores <- c(0.9, 0.0)

  build_nms <- function(with_thresh) {
    inits <- list(
      .onnx_tensor("boxes",  c(1L, 2L, 4L), 1L, .float_bytes(boxes)),
      .onnx_tensor("scores", c(1L, 1L, 2L), 1L, .float_bytes(scores)),
      .onnx_tensor("maxout", 1L, 7L, .int64_bytes(10L)),
      .onnx_tensor("iou",    1L, 1L, .float_bytes(0.5))
    )
    ins <- c("boxes", "scores", "maxout", "iou")
    if (with_thresh) {
      inits <- c(inits, list(.onnx_tensor("sthr", 1L, 1L, .float_bytes(0))))
      ins <- c(ins, "sthr")
    }
    # X is a declared input the graph ignores: onnx_run needs something to
    # feed, and every real value here is an initializer.
    nodes <- list(.onnx_node("NonMaxSuppression", ins, "sel"))
    g <- .onnx_graph("g", nodes,
                     list(.onnx_value_info("X", 1L, c(1L, 1L))),
                     list(.onnx_value_info("sel", 7L, c(-1L, 3L))),
                     initializers = inits)
    .onnx_model(g, opset_version = 11L)
  }

  # Each selection is a [batch, class, index] triple, and the output is sized
  # to the worst case: unused slots are left at -1, which is the kernel's
  # "empty", not an index.  Counting the raw length would count those too --
  # in a whole model resolve_segment_sizes shrinks the capacity to the
  # measured count, but a one-node graph has no such measurement.
  count_sel <- function(v) sum(v[seq(3L, length(v), by = 3L)] >= 0)

  n_absent  <- count_sel(run1(build_nms(FALSE), 0, c(1L, 1L)))
  n_present <- count_sel(run1(build_nms(TRUE),  0, c(1L, 1L)))

  # Absent: no score filter at all, both boxes are candidates and both survive.
  expect_equal(n_absent, 2)
  # Present and 0: the comparison is strict, so the 0-scoring box is filtered.
  expect_equal(n_present, 1)
  # The pair is the test.  Stated separately so that a change making both
  # agree fails here with a readable message, rather than in a detector
  # three months later.
  expect_false(n_absent == n_present)
})

test_that("Clip: absent min/max do not clamp to zero", {
  # The input straddles zero on purpose: a zero default for either bound turns
  # Clip into a one-sided ReLU, which looks perfectly reasonable on
  # nonnegative test data and wrong only on negatives.
  x <- c(-2, -0.5, 0, 0.5, 2)
  dims <- c(1L, 5L)

  build_clip <- function(lo, hi) {
    inits <- list(); ins <- "X"
    if (!is.null(lo)) {
      inits <- c(inits, list(.onnx_tensor("lo", 1L, 1L, .float_bytes(lo))))
      ins <- c(ins, "lo")
    } else if (!is.null(hi)) {
      ins <- c(ins, "")      # omitted min, but max follows: empty name
    }
    if (!is.null(hi)) {
      inits <- c(inits, list(.onnx_tensor("hi", 1L, 1L, .float_bytes(hi))))
      ins <- c(ins, "hi")
    }
    g <- .onnx_graph("g", list(.onnx_node("Clip", ins, "Y")),
                     list(.onnx_value_info("X", 1L, dims)),
                     list(.onnx_value_info("Y", 1L, dims)),
                     initializers = inits)
    .onnx_model(g, opset_version = 13L)
  }

  expect_equal(run1(build_clip(NULL, NULL), x, dims), x, tolerance = 1e-6)
  expect_equal(run1(build_clip(0, NULL), x, dims), pmax(x, 0), tolerance = 1e-6)
  expect_equal(run1(build_clip(NULL, 0), x, dims), pmin(x, 0), tolerance = 1e-6)
})

test_that("Slice: absent steps mean 1, not 0", {
  x <- as.numeric(1:10)
  dims <- c(1L, 10L)

  build_slice <- function(step) {
    inits <- list(
      .onnx_tensor("starts", 1L, 7L, .int64_bytes(2L)),
      .onnx_tensor("ends",   1L, 7L, .int64_bytes(8L)),
      .onnx_tensor("axes",   1L, 7L, .int64_bytes(1L))
    )
    ins <- c("X", "starts", "ends", "axes")
    if (!is.null(step)) {
      inits <- c(inits, list(.onnx_tensor("steps", 1L, 7L, .int64_bytes(step))))
      ins <- c(ins, "steps")
    }
    nout <- if (is.null(step)) 6L else length(seq(3L, 8L, by = step))
    g <- .onnx_graph("g", list(.onnx_node("Slice", ins, "Y")),
                     list(.onnx_value_info("X", 1L, dims)),
                     list(.onnx_value_info("Y", 1L, c(1L, nout))),
                     initializers = inits)
    .onnx_model(g, opset_version = 13L)
  }

  # A zero step is not a smaller step -- it is a hang or a divide by zero, so
  # this one fails loudly rather than quietly.  Same class all the same.
  expect_equal(run1(build_slice(NULL), x, dims), x[3:8], tolerance = 1e-6)
  expect_equal(run1(build_slice(2L), x, dims), x[c(3, 5, 7)], tolerance = 1e-6)
})

test_that("Pad: absent constant_value really does default to zero", {
  # The exception that proves the rule.  Here the spec's default IS 0, so
  # absent and present-0 must AGREE -- the question is never what the zero
  # looks like, it is what the spec says the absence means.  A blanket
  # "absent is never 0" rule would break this case.
  #
  x <- c(1, 2, 3)
  dims <- c(1L, 3L)

  build_pad <- function(cval, pads = c(0L, 1L, 0L, 1L)) {
    inits <- list(.onnx_tensor("pads", 4L, 7L, .i64v(pads)))
    ins <- c("X", "pads")
    if (!is.null(cval)) {
      inits <- c(inits, list(.onnx_tensor("cv", 1L, 1L, .float_bytes(cval))))
      ins <- c(ins, "cv")
    }
    g <- .onnx_graph("g", list(.onnx_node("Pad", ins, "Y")),
                     list(.onnx_value_info("X", 1L, dims)),
                     list(.onnx_value_info("Y", 1L, c(1L, 5L))),
                     initializers = inits)
    .onnx_model(g, opset_version = 13L)
  }

  expect_equal(run1(build_pad(NULL), x, dims), c(0, 1, 2, 3, 0), tolerance = 1e-6)
  expect_equal(run1(build_pad(0),    x, dims), c(0, 1, 2, 3, 0), tolerance = 1e-6)
})

test_that("Pad: begin and end are distinct positions, not a total", {
  # begin and end used to be summed and handed to ggml_pad, which appends
  # only.  All three of these then returned the same thing -- the total width
  # was right and the data was in the wrong place, which is the failure a
  # shape-only assertion cannot see.
  x <- c(1, 2, 3)
  dims <- c(1L, 3L)

  build_pad <- function(pads) {
    inits <- list(.onnx_tensor("pads", 4L, 7L, .i64v(pads)))
    g <- .onnx_graph("g", list(.onnx_node("Pad", c("X", "pads"), "Y")),
                     list(.onnx_value_info("X", 1L, dims)),
                     list(.onnx_value_info("Y", 1L, c(1L, 5L))),
                     initializers = inits)
    .onnx_model(g, opset_version = 13L)
  }

  cases <- list(
    list(pads = c(0L, 1L, 0L, 1L), want = c(0, 1, 2, 3, 0)),  # one each side
    list(pads = c(0L, 0L, 0L, 2L), want = c(1, 2, 3, 0, 0)),  # both at the end
    list(pads = c(0L, 2L, 0L, 0L), want = c(0, 0, 1, 2, 3))   # both at the front
  )

  for (cs in cases) {
    expect_equal(run1(build_pad(cs$pads), x, dims), cs$want, tolerance = 1e-6,
                 info = sprintf("cpu pads=[%s]", paste(cs$pads, collapse = ",")))
  }
})

test_that("Pad: the shader agrees with the reference, not just with the CPU op", {
  # Checked against the same expected values as the CPU test above, never
  # against the CPU result: the two are separate implementations of one
  # contract, and a mistake present in both is invisible to a CPU-vs-GPU
  # comparison.  That is exactly how the deliberate top_k swap survived --
  # it was in the kernel AND all three shaders, and only an outside reference
  # showed it.
  skip_if_no_gpu()

  x <- c(1, 2, 3)
  dims <- c(1L, 3L)
  cases <- list(
    list(pads = c(0L, 1L, 0L, 1L), want = c(0, 1, 2, 3, 0)),
    list(pads = c(0L, 0L, 0L, 2L), want = c(1, 2, 3, 0, 0)),
    list(pads = c(0L, 2L, 0L, 0L), want = c(0, 0, 1, 2, 3))
  )

  build_pad <- function(pads) {
    inits <- list(.onnx_tensor("pads", 4L, 7L, .i64v(pads)))
    g <- .onnx_graph("g", list(.onnx_node("Pad", c("X", "pads"), "Y")),
                     list(.onnx_value_info("X", 1L, dims)),
                     list(.onnx_value_info("Y", 1L, c(1L, 5L))),
                     initializers = inits)
    .onnx_model(g, opset_version = 13L)
  }

  for (cs in cases) {
    expect_equal(run1(build_pad(cs$pads), x, dims, device = "gpu"), cs$want,
                 tolerance = 1e-6,
                 info = sprintf("gpu pads=[%s]", paste(cs$pads, collapse = ",")))
  }
})
