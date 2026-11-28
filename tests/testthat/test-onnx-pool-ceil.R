# MaxPool / AveragePool: ceil_mode, asymmetric pads, count_include_pad,
# checked against a plain-R reference of the ONNX semantics.

# ONNX pooling on one H x W plane. pads = c(h_begin, w_begin, h_end, w_end).
pool_ref <- function(x, k, s, pads = c(0, 0, 0, 0), ceil = FALSE,
                     op = "max", include_pad = FALSE) {
  out_len <- function(n, kk, ss, pb, pe) {
    span <- n + pb + pe - kk
    o <- if (ceil) ceiling(span / ss) + 1 else floor(span / ss) + 1
    if (ceil && (o - 1) * ss >= n + pb) o <- o - 1   # window starting in the tail
    o
  }
  H <- nrow(x); W <- ncol(x)
  Ho <- out_len(H, k[1], s[1], pads[1], pads[3])
  Wo <- out_len(W, k[2], s[2], pads[2], pads[4])
  y <- matrix(0, Ho, Wo)
  for (i in seq_len(Ho)) for (j in seq_len(Wo)) {
    r0 <- (i - 1) * s[1] - pads[1]; c0 <- (j - 1) * s[2] - pads[2]
    rows <- r0 + seq_len(k[1]) - 1; cols <- c0 + seq_len(k[2]) - 1
    vr <- rows[rows >= 0 & rows < H]; vc <- cols[cols >= 0 & cols < W]
    vals <- x[vr + 1, vc + 1]
    if (op == "max") {
      y[i, j] <- max(vals)
    } else {
      n <- if (include_pad) {
        sum(rows >= -pads[1] & rows < H + pads[3]) *
          sum(cols >= -pads[2] & cols < W + pads[4])
      } else length(vals)
      y[i, j] <- sum(vals) / n
    }
  }
  y
}

pool_model <- function(op, H, W, Ho, Wo, k, s, pads, ceil, include_pad) {
  attrs <- list(.onnx_attr_ints("kernel_shape", as.integer(k)),
                .onnx_attr_ints("strides", as.integer(s)),
                .onnx_attr_ints("pads", as.integer(pads)),
                .onnx_attr_int("ceil_mode", as.integer(ceil)))
  if (op == "AveragePool") {
    attrs <- c(attrs, list(.onnx_attr_int("count_include_pad",
                                          as.integer(include_pad))))
  }
  node <- .onnx_node(op, "X", "Y", attrs = attrs)
  graph <- .onnx_graph("pool", list(node),
                       list(.onnx_value_info("X", 1L, c(1L, 1L, H, W))),
                       list(.onnx_value_info("Y", 1L, c(1L, 1L, Ho, Wo))))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

run_pool <- function(x, k, s, pads = c(0, 0, 0, 0), ceil = FALSE, op = "max",
                     include_pad = FALSE, device = "cpu") {
  ref <- pool_ref(x, k, s, pads, ceil, op, include_pad)
  path <- pool_model(if (op == "max") "MaxPool" else "AveragePool",
                     nrow(x), ncol(x), nrow(ref), ncol(ref), k, s, pads, ceil,
                     include_pad)
  on.exit(unlink(path))
  m <- onnx_load(path, device = device)
  r <- onnx_run(m, list(X = as.numeric(t(x))))[[1]]   # row-major in
  list(got = t(matrix(as.numeric(r), ncol(ref), nrow(ref))), ref = ref)
}

cases <- list(
  list(name = "ceil, exact division (8x8 k2 s2)",   H = 8, W = 8, k = c(2, 2), s = c(2, 2), pads = c(0, 0, 0, 0), ceil = TRUE),
  list(name = "ceil, odd input (7x7 k2 s2)",        H = 7, W = 7, k = c(2, 2), s = c(2, 2), pads = c(0, 0, 0, 0), ceil = TRUE),
  list(name = "ceil, k3 s2, 8x10",                  H = 8, W = 10, k = c(3, 3), s = c(2, 2), pads = c(0, 0, 0, 0), ceil = TRUE),
  list(name = "ceil + pads 1 (6x6 k3 s2)",          H = 6, W = 6, k = c(3, 3), s = c(2, 2), pads = c(1, 1, 1, 1), ceil = TRUE),
  list(name = "ceil, tail window dropped (5x5 k2 s2 pads 1)", H = 5, W = 5, k = c(2, 2), s = c(2, 2), pads = c(1, 1, 1, 1), ceil = TRUE),
  list(name = "asymmetric pads, no ceil (5x7 k3 s2)", H = 5, W = 7, k = c(3, 3), s = c(2, 2), pads = c(1, 0, 0, 1), ceil = FALSE),
  list(name = "symmetric pads, no ceil (6x6 k3 s1)", H = 6, W = 6, k = c(3, 3), s = c(1, 1), pads = c(1, 1, 1, 1), ceil = FALSE),
  list(name = "ceil, stride 1 (no tail) (5x5 k3 s1)", H = 5, W = 5, k = c(3, 3), s = c(1, 1), pads = c(0, 0, 0, 0), ceil = TRUE)
)

check_cases <- function(device) {
  set.seed(1)
  for (cs in cases) {
    x <- matrix(stats::rnorm(cs$H * cs$W), cs$H, cs$W)   # negatives on purpose
    r <- run_pool(x, cs$k, cs$s, cs$pads, cs$ceil, "max", device = device)
    expect_equal(dim(r$got), dim(r$ref), info = paste("max", cs$name))
    expect_equal(r$got, r$ref, tolerance = 1e-5, info = paste("max", cs$name))
    for (incl in c(FALSE, TRUE)) {
      r <- run_pool(x, cs$k, cs$s, cs$pads, cs$ceil, "avg", incl, device = device)
      lbl <- paste0("avg (count_include_pad=", incl, ") ", cs$name)
      expect_equal(dim(r$got), dim(r$ref), info = lbl)
      expect_equal(r$got, r$ref, tolerance = 1e-5, info = lbl)
    }
  }
}

test_that("pooling matches ONNX semantics on CPU", {
  check_cases("cpu")
})

test_that("pooling matches ONNX semantics on Vulkan", {
  skip_if_no_gpu()
  check_cases("vulkan")
})

test_that("onnx_load fails when an output cannot be built", {
  node <- .onnx_node("NoSuchOp", "X", "Y")
  graph <- .onnx_graph("broken", list(node),
                       list(.onnx_value_info("X", 1L, c(1L, 4L))),
                       list(.onnx_value_info("Y", 1L, c(1L, 4L))))
  path <- tempfile(fileext = ".onnx")
  on.exit(unlink(path))
  writeBin(.onnx_model(graph), path)
  expect_error(onnx_load(path, device = "cpu"),
               "1 of 1 model outputs cannot be built \\(first: 'Y'\\).*NoSuchOp")
})
